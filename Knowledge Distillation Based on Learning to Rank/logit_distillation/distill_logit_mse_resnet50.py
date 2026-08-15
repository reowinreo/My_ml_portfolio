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
import math

data_dir = 'dataset_raw'
split_path = 'saved_models/split_indices.npz'
batch_size = 128
num_classes = 45
num_epochs = 500
temperature = 1  # Not used, but kept for compatibility

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

# Teacher
teacher_model = models.resnet152(pretrained=False)
num_features_teacher = teacher_model.fc.in_features
teacher_model.fc = nn.Linear(num_features_teacher, num_classes)
teacher_model.load_state_dict(torch.load('saved_models/resnet152_final_model.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
teacher_model = teacher_model.to(device)
teacher_model.eval()

non_shuffle_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

teacher_logits = []
with torch.no_grad():
    for inputs, _ in non_shuffle_train_loader:
        inputs = inputs.to(device)
        logits = teacher_model(inputs)          # raw logits
        teacher_logits.append(logits.cpu())
teacher_logits = torch.cat(teacher_logits, dim=0).float()
teacher_logits = teacher_logits - teacher_logits.mean(dim=1, keepdim=True)  # per-sample zero-mean
teacher_logits = teacher_logits.to(device)

# Student
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

# ----------------- KD (MSE) component -----------------
def kd_mse_loss(student_logits, teacher_logits):
    return torch.mean((student_logits - teacher_logits) ** 2)

# ----------------- Paper-style uncertainty weighting (MSE + CE) -----------------
class UncertaintyWeightedMSECE(nn.Module):
    """
    L = L_ce/(2*sigma_ce^2) + L_mse/(2*sigma_mse^2) + log(sigma_ce) + log(sigma_mse)
    log_sigma_* are learnable parameters (sigma>0 via exp)
    """
    def __init__(self):
        super().__init__()
        self.log_sigma_ce = nn.Parameter(torch.zeros(1))   # initialized to 0 -> sigma=1
        self.log_sigma_mse = nn.Parameter(torch.zeros(1))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, hard_labels, teacher_logits):
        L_ce = self.ce(student_logits, hard_labels)
        L_mse = kd_mse_loss(student_logits, teacher_logits)

        sigma_ce  = torch.exp(self.log_sigma_ce)
        sigma_mse = torch.exp(self.log_sigma_mse)

        loss = (L_ce  / (2 * sigma_ce**2)) \
             + (L_mse / (2 * sigma_mse**2)) \
             + torch.log(sigma_ce) + torch.log(sigma_mse)

        return loss, L_ce.detach(), L_mse.detach(), sigma_ce.detach(), sigma_mse.detach()

criterion = UncertaintyWeightedMSECE().to(device)

# ================= Learning rate and scheduler: as in your code =================
optimizer = optim.AdamW([
    {"params": model.parameters()},
    {"params": criterion.parameters(), "lr": 0.001}  # jointly optimize uncertainty parameters
], lr=0.001, weight_decay=1e-4)

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
# ============================================================

def train_model(model, optimizer, scheduler, num_epochs=num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # contains labels (for CE)
    class DistillationDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, teacher_logits):
            self.dataset = dataset
            self.teacher_logits = teacher_logits

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img, hard_label = self.dataset[idx]
            return img, hard_label, self.teacher_logits[idx]

    distillation_dataset = DistillationDataset(train_dataset, teacher_logits)
    distillation_loader = torch.utils.data.DataLoader(distillation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, hard_labels, t_logits in distillation_loader:
            inputs = inputs.to(device)
            hard_labels = hard_labels.to(device)
            t_logits = t_logits.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)  # student logits

            # ---- Paper-style dynamic weighting (MSE+CE) total loss ----
            loss, L_ce_det, L_mse_det, sigma_ce_det, sigma_mse_det = criterion(
                student_logits=outputs, hard_labels=hard_labels, teacher_logits=t_logits
            )

            # Training accuracy still computed as "student vs teacher hard predictions" (consistent with your original logic)
            s_preds = torch.argmax(outputs, dim=1)
            t_preds = torch.argmax(t_logits, dim=1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(s_preds == t_preds.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation (hard labels used as metrics)
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

        # Best model (by val_acc)
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/best_model.pth')
            print(f"Saved best model: epoch {epoch}, val_acc={epoch_acc_val:.4f}")

        # cosine annealing (warm restarts) step
        scheduler.step()
        print(f"Current learning rates: {[group['lr'] for group in optimizer.param_groups]}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)

    os.makedirs('saved_models', exist_ok=True)
    epochs = np.arange(len(history['train_acc']))
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    try:
        plt.show()
    except Exception:
        pass

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

torch.save(model.state_dict(), 'saved_models/dynamic_weight_hard_label_mse_final_model.pth')

print("Evaluating model on test set...")
test_acc, cm, all_preds, all_labels = evaluate_model(model, test_loader)
print(f"Test set accuracy: {test_acc:.4f}")
