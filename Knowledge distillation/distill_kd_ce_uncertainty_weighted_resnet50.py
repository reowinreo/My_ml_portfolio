import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
import os
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

data_dir = 'dataset_raw'
split_path = 'saved_models/split_indices.npz'
batch_size = 128
num_classes = 45
num_epochs = 500
temperature = 4

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
    val_indices = split_data['val_indices']
    test_indices = split_data['test_indices']

    train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val'])
    test_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['test'])

    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]
    test_dataset.samples = [full_dataset.samples[i] for i in test_indices]

    return train_dataset, val_dataset, test_dataset, full_dataset.classes

train_dataset, val_dataset, test_dataset, class_names = create_datasets_from_split(data_dir, split_path)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}
print(f"Train set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")
print(f"Test set size: {dataset_sizes['test']}")

# Load teacher model
teacher_model = models.resnet152(pretrained=False)
num_features_teacher = teacher_model.fc.in_features
teacher_model.fc = nn.Linear(num_features_teacher, num_classes)
teacher_model.load_state_dict(torch.load('saved_models/resnet152_final_model.pth', map_location='cpu'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
teacher_model = teacher_model.to(device)
teacher_model.eval()

non_shuffle_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

teacher_logits = []
with torch.no_grad():
    for inputs, _ in non_shuffle_train_loader:
        inputs = inputs.to(device)
        logits = teacher_model(inputs)
        teacher_logits.append(logits.cpu())
teacher_logits = torch.cat(teacher_logits, dim=0)

# Student model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

#Uncertainty weighting term distillation loss
def distillation_loss(student_logits, teacher_logits, T):
    # Keep original KD formulation: Cross-Entropy(q_teacher, p_student) * T^2
    soft_teacher = torch.nn.functional.softmax(teacher_logits / T, dim=1)
    soft_student = torch.nn.functional.log_softmax(student_logits / T, dim=1)
    return -torch.mean(torch.sum(soft_teacher * soft_student, dim=1)) * (T ** 2)

class UncertaintyWeightedKDLoss(nn.Module):
    """
    Paper-style total objective:
    L = L_hard/(2*sigma_hard^2) + L_kd/(2*sigma_soft^2) + log(sigma_hard) + log(sigma_soft)
    """
    def __init__(self, temperature=4):
        super().__init__()
        self.temperature = temperature
        self.log_sigma_hard = nn.Parameter(torch.zeros(1))
        self.log_sigma_soft = nn.Parameter(torch.zeros(1))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, labels, teacher_logits):
        L_hard = self.ce(student_logits, labels)
        L_kd = distillation_loss(student_logits, teacher_logits, self.temperature)

        sigma_hard = torch.exp(self.log_sigma_hard)  # sigma > 0
        sigma_soft = torch.exp(self.log_sigma_soft)

        loss = (L_hard / (2 * sigma_hard**2)) + (L_kd / (2 * sigma_soft**2)) \
               + torch.log(sigma_hard) + torch.log(sigma_soft)
        return loss, L_hard.detach(), L_kd.detach(), sigma_hard.detach(), sigma_soft.detach()

criterion = UncertaintyWeightedKDLoss(temperature=temperature).to(device)

#Include learnable uncertainty parameters
optimizer = optim.AdamW([
    {"params": model.parameters()},
    {"params": criterion.parameters(), "lr": 0.001}
], lr=0.001, weight_decay=1e-4)

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

def train_model(model, optimizer, scheduler, num_epochs=num_epochs, criterion=criterion):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Create a dataset with teacher logits for distillation
    class DistillationDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, teacher_logits):
            self.dataset = dataset
            self.teacher_logits = teacher_logits

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img, hard_label = self.dataset[idx]
            tlogit = self.teacher_logits[idx]
            return img, hard_label, tlogit

    distillation_dataset = DistillationDataset(train_dataset, teacher_logits)
    distillation_loader = torch.utils.data.DataLoader(distillation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Train phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels, t_logits in distillation_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            t_logits = t_logits.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            #Auto-weighted KD + CE
            loss, L_hard_det, L_kd_det, sigma_hard_det, sigma_soft_det = criterion(
                student_logits=outputs, labels=labels, teacher_logits=t_logits
            )
            _, preds = torch.max(outputs, 1)
            _, t_preds = torch.max(t_logits, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == t_preds.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        try:
            eff_w_hard = float((1.0 / (2.0 * (sigma_hard_det**2))).cpu().numpy())
            eff_w_soft = float((1.0 / (2.0 * (sigma_soft_det**2))).cpu().numpy())
            print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        except Exception:
            print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Use hard labels for validation metrics
        model.eval()
        running_loss_val = 0.0
        running_corrects_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss_val = nn.CrossEntropyLoss()(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss_val += loss_val.item() * inputs.size(0)
                running_corrects_val += torch.sum(preds == labels.data)

        epoch_loss_val = running_loss_val / dataset_sizes['val']
        epoch_acc_val = running_corrects_val.double() / dataset_sizes['val']

        history['val_loss'].append(epoch_loss_val)
        history['val_acc'].append(epoch_acc_val.item())

        print(f'val Loss: {epoch_loss_val:.4f} Acc: {epoch_acc_val:.4f}')

        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/best_model.pth')
            print(f"Saved best model: epoch {epoch}, val_acc={epoch_acc_val:.4f}")

        scheduler.step()
        print(f"Current learning rates: {[group['lr'] for group in optimizer.param_groups]}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)

    return model, history

def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, cm, all_preds, all_labels

os.makedirs('saved_models', exist_ok=True)

print("Starting model training...")
model, history = train_model(model, optimizer, scheduler, num_epochs=num_epochs)

try:
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history['train_acc'], label='train')
    plt.plot(epochs, history['val_acc'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
except Exception as e:
    print(f"Issue while plotting curves: {e}")

torch.save(model.state_dict(), 'saved_models/auto_weight_update_teacher+hardlabel_temp4_weight_distilled_final_model.pth')

print("Evaluating model on the test set...")
test_acc, cm, all_preds, all_labels = evaluate_model(model, test_loader)
print(f"Test set accuracy: {test_acc:.4f}")
