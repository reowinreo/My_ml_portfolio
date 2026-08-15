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

data_dir = 'dataset_raw'
split_path = 'saved_models/split_indices.npz'
batch_size = 128
num_classes = 45

# Train Stage2 only
num_epochs_stage2 = 200

# Stage1 best weights (backbone aligned)
stage1_best_ckpt = 'saved_models/student_stage1_best_by_valMSE.pth'

# Stage2 model save path
stage2_best_ckpt = 'saved_models/student_stage2_best_by_valAcc.pth'
final_ckpt       = 'saved_models/pca_stage2_only_final_model.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs('saved_models', exist_ok=True)

# Data augmentation / preprocessing
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

# ---- Model wrapper (same as Stage1) ----
class PenultimateWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

    def forward_features(self, x):
        x = self.stem(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        feats = self.forward_features(x)  # 2048-D
        logits = self.fc(feats)           # num_classes
        return feats, logits

# Student model (loaded from Stage1 best)
student_model = models.resnet50(pretrained=False)
num_features_student = student_model.fc.in_features
student_model.fc = nn.Linear(num_features_student, num_classes)
student_model = student_model.to(device)
student_wrap = PenultimateWrapper(student_model).to(device)

assert os.path.exists(stage1_best_ckpt), f"Stage1 best weights not found: {stage1_best_ckpt}"
state = torch.load(stage1_best_ckpt, map_location='cpu')
student_wrap.load_state_dict(state)
print(f"Loaded Stage1 best weights: {stage1_best_ckpt}")

# Freeze backbone, train fc only (Stage2)
for p in student_wrap.stem.parameters():
    p.requires_grad = False
for p in student_wrap.avgpool.parameters():
    p.requires_grad = False
for p in student_wrap.fc.parameters():
    p.requires_grad = True

# ---- Optimizer: LR=0.01, StepLR ×0.5 every 10 epochs ----
init_lr = 1e-2
optim_stage2 = optim.AdamW(student_wrap.fc.parameters(), lr=init_lr, weight_decay=1e-4)
sched_stage2 = lr_scheduler.StepLR(optim_stage2, step_size=10, gamma=0.5)
criterion_ce = nn.CrossEntropyLoss()

def evaluate_val_accuracy(model_wrap, loader):
    model_wrap.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device); labels = labels.to(device)
            feats, logits = model_wrap(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc

def get_current_lrs(optimizer):
    return [pg['lr'] for pg in optimizer.param_groups]

best_val_acc = 0.0
best_stage2_state = copy.deepcopy(student_wrap.state_dict())

print("\n===== Stage 2 =====")
for epoch in range(num_epochs_stage2):
    # Training
    student_wrap.train()
    running_loss = 0.0; ns = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device); labels = labels.to(device)
        optim_stage2.zero_grad()
        feats, logits = student_wrap(inputs)
        loss = criterion_ce(logits, labels)
        loss.backward()
        optim_stage2.step()
        running_loss += loss.item() * inputs.size(0)
        ns += inputs.size(0)
    avg_train_loss = running_loss / max(1, ns)

    # Validation
    val_acc = evaluate_val_accuracy(student_wrap, val_loader)

    # Save Stage2 best (max val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_stage2_state = copy.deepcopy(student_wrap.state_dict())
        torch.save(best_stage2_state, stage2_best_ckpt)
        print(f"Saved Stage2 best (val_acc={val_acc:.4f})")

    # Print & update learning rate (StepLR)
    cur_lrs = get_current_lrs(optim_stage2)
    print(f"Epoch {epoch+1}/{num_epochs_stage2}  val_acc={val_acc:.4f}  lr={cur_lrs}")
    print("==============")
    sched_stage2.step()  # lr * 0.5 automatically every 10 epochs

# Evaluate with best weights and save final model
student_wrap.load_state_dict(best_stage2_state)

def evaluate_model_logits(model_wrap, dataloader):
    model_wrap.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device); labels = labels.to(device)
            feats, logits = model_wrap(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, cm, all_preds, all_labels

print("\nEvaluating final model on test set...")
test_acc, cm, all_preds, all_labels = evaluate_model_logits(student_wrap, test_loader)
print(f"Test set accuracy: {test_acc:.4f}")

torch.save(student_wrap.state_dict(), final_ckpt)

