import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

data_dir = 'dataset_raw'
split_path = 'saved_models/split_indices.npz'
batch_size = 128
num_classes = 45
num_epochs = 500
temperature = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs('saved_models', exist_ok=True)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Helper function for this experiment module
def create_datasets_from_split(data_dir, split_path):
    full_dataset = datasets.ImageFolder(data_dir)
    split_data = np.load(split_path)
    train_indices = split_data['train_indices']
    val_indices   = split_data['val_indices']
    test_indices  = split_data['test_indices']

    train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    val_dataset   = datasets.ImageFolder(data_dir, transform=data_transforms['val'])
    test_dataset  = datasets.ImageFolder(data_dir, transform=data_transforms['test'])

    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    val_dataset.samples   = [full_dataset.samples[i] for i in val_indices]
    test_dataset.samples  = [full_dataset.samples[i] for i in test_indices]
    return train_dataset, val_dataset, test_dataset, full_dataset.classes

train_dataset, val_dataset, test_dataset, class_names = create_datasets_from_split(data_dir, split_path)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
print(f"Train set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")
print(f"Test set size: {dataset_sizes['test']}")

class PenultimateWrapper(nn.Module):
    """
    Output:
      feats: [B, 2048]  (flattened after avgpool, before fc)
      logits: [B, num_classes]  (fc output)
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.avgpool(x)         # [B, 2048, 1, 1]
        feats = torch.flatten(x, 1) # [B, 2048]
        logits = self.fc(feats)     # [B, C]
        return feats, logits

#Teacher
teacher_backbone = models.resnet152(pretrained=False)
nf_t = teacher_backbone.fc.in_features
teacher_backbone.fc = nn.Linear(nf_t, num_classes)
teacher_backbone.load_state_dict(torch.load('saved_models/resnet152_final_model.pth', map_location='cpu'))
teacher_backbone = teacher_backbone.to(device).eval()
teacher = PenultimateWrapper(teacher_backbone).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False  # teacher is not updated

#Student
student_backbone = models.resnet50(pretrained=False)
nf_s = student_backbone.fc.in_features  # 2048
student_backbone.fc = nn.Linear(nf_s, num_classes)
student_backbone = student_backbone.to(device)
student = PenultimateWrapper(student_backbone).to(device)

#Feature MSE + KD(KL) + hard-label CE
def feature_mse_loss(student_feats_2048, teacher_feats_2048):
    # L2 normalization before MSE
    p = torch.nn.functional.normalize(student_feats_2048, p=2, dim=1)
    q = torch.nn.functional.normalize(teacher_feats_2048, p=2, dim=1)
    return torch.nn.functional.mse_loss(p, q)

def kd_kl_loss(student_logits, teacher_logits, T: float):
    with torch.no_grad():
        soft_teacher = torch.nn.functional.softmax(teacher_logits / T, dim=1)
    soft_student_log = torch.nn.functional.log_softmax(student_logits / T, dim=1)
    loss = -(soft_teacher * soft_student_log).sum(dim=1).mean() * (T ** 2)
    return loss

ce_criterion = nn.CrossEntropyLoss()

#Uncertainty weighting term
class UncertaintyWeightedFeatKDCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_feat = nn.Parameter(torch.zeros(1))
        self.log_sigma_kd   = nn.Parameter(torch.zeros(1))
        self.log_sigma_ce   = nn.Parameter(torch.zeros(1))

    def forward(self, L_feat, L_kd, L_ce):
        sigma_feat = torch.exp(self.log_sigma_feat)
        sigma_kd   = torch.exp(self.log_sigma_kd)
        sigma_ce   = torch.exp(self.log_sigma_ce)
        loss = (L_feat / (2 * sigma_feat**2)) \
             + (L_kd   / (2 * sigma_kd**2)) \
             + (L_ce   / (2 * sigma_ce**2)) \
             + torch.log(sigma_feat) + torch.log(sigma_kd) + torch.log(sigma_ce)
        # Return intermediate values (for logging only)
        return loss, sigma_feat.detach(), sigma_kd.detach(), sigma_ce.detach()

criterion = UncertaintyWeightedFeatKDCE().to(device)

#Two parameter groups + cosine warm restarts
optimizer = optim.AdamW([
    {"params": student.parameters()},
    {"params": criterion.parameters(), "lr": 0.001}
], lr=0.001, weight_decay=1e-4)

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

def train_model(student, optimizer, scheduler, num_epochs=num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(student.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        #Train
        student.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Teacher forward pass
            with torch.no_grad():
                t_feats, t_logits = teacher(inputs)

            # Student forward pass
            s_feats, s_logits = student(inputs)

            L_feat = feature_mse_loss(s_feats, t_feats)
            L_kd   = kd_kl_loss(s_logits, t_logits, temperature)
            L_ce   = ce_criterion(s_logits, labels)

            # Dynamically weighted total loss
            loss, sigma_feat_det, sigma_kd_det, sigma_ce_det = criterion(L_feat, L_kd, L_ce)

            # Training accuracy
            _, preds = torch.max(s_logits, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc  = running_corrects.double() / dataset_sizes['train']
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        #Val
        student.eval()
        running_loss_val = 0.0
        running_corrects_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                t_feats, t_logits = teacher(inputs)
                s_feats, s_logits = student(inputs)

                L_feat = feature_mse_loss(s_feats, t_feats)
                L_kd   = kd_kl_loss(s_logits, t_logits, temperature)
                L_ce   = ce_criterion(s_logits, labels)

                # Validation total loss (no backprop), for monitoring
                val_loss, _, _, _ = criterion(L_feat, L_kd, L_ce)

                _, preds = torch.max(s_logits, 1)

                running_loss_val += val_loss.item() * inputs.size(0)
                running_corrects_val += torch.sum(preds == labels.data)

        epoch_loss_val = running_loss_val / dataset_sizes['val']
        epoch_acc_val  = running_corrects_val.double() / dataset_sizes['val']
        history['val_loss'].append(epoch_loss_val)
        history['val_acc'].append(epoch_acc_val.item())

        print(f'val  Loss: {epoch_loss_val:.4f} Acc: {epoch_acc_val:.4f}')

        #Save best checkpoint
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(student.state_dict())
            torch.save(student.state_dict(), 'saved_models/best_model.pth')
            print(f"Saved best model: epoch {epoch}, val_acc={epoch_acc_val:.4f}")

        scheduler.step()
        print(f"Current learning rates: {[pg['lr'] for pg in optimizer.param_groups]}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    student.load_state_dict(best_model_wts)
    return student, history

def evaluate_model(student, dataloader):
    student.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, logits = student(inputs)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)
    return acc, cm, all_preds, all_labels

print("Starting model training...")
student, history = train_model(student, optimizer, scheduler, num_epochs=num_epochs)

#Plot accuracy curves
epochs_axis = np.arange(len(history['train_acc']))
plt.figure()
plt.plot(epochs_axis, history['train_acc'], label='Train Acc')
plt.plot(epochs_axis, history['val_acc'],   label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
try:
    plt.show()
except Exception:
    pass

#Keep original checkpoint filename unchanged
torch.save(student.state_dict(), 'saved_models/hardlabel_kl_feat-mse_uncertainty-weighted_resnet50_student_final.pth')

print("Evaluating on the test set...")
test_acc, cm, all_preds, all_labels = evaluate_model(student, test_loader)
print(f"Test set accuracy: {test_acc:.4f}")
