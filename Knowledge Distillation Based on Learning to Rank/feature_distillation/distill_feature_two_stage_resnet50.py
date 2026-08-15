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

# Number of training epochs
num_epochs_stage1 = 350
num_epochs_stage2 = 200

stage1_min_delta = 3e-7
stage1_patience_trigger = 10
lr_scale_factor = 1.0 / 5.0

# Stage1 Dropout probability (new)
stage1_dropout_p = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs('saved_models', exist_ok=True)

# Transforms & dataset construction
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

train_loader_base = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader_base   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader_base  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
print(f"Train set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")
print(f"Test set size: {dataset_sizes['test']}")

teacher_model = models.resnet152(pretrained=False)
num_features_teacher = teacher_model.fc.in_features  # 2048
teacher_model.fc = nn.Linear(num_features_teacher, num_classes)
teacher_model.load_state_dict(torch.load('saved_models/resnet152_final_model.pth', map_location='cpu'))
teacher_model = teacher_model.to(device)
teacher_model.eval()

# Wrapper to extract penultimate-layer features; add switchable dropout (student only, Stage1)
class PenultimateWrapper(nn.Module):
    def __init__(self, backbone, dropout_p: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        # New: optional dropout
        self.drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else None
        self.enable_dropout = (self.drop is not None)

    def set_dropout(self, enabled: bool):
        self.enable_dropout = (self.drop is not None) and enabled

    def forward_features(self, x):
        x = self.stem(x)
        x = self.avgpool(x)         # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)     # [B, 2048]
        # Use dropout only when enabled and in training mode (disabled for val/test)
        if self.enable_dropout and self.training and (self.drop is not None):
            x = self.drop(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)  # 2048-D
        logits = self.fc(feats)           # num_classes
        return feats, logits

teacher_wrap = PenultimateWrapper(teacher_model, dropout_p=0.0).to(device).eval()  # Teacher does not need dropout

def extract_teacher_feats(dataset, batch_size=128):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_feats = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            feats, _ = teacher_wrap(inputs)      # [B, 2048]
            feats = F.normalize(feats, p=2, dim=1)  # L2 normalization
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype(np.float32)

print("Extracting teacher training features...")
q_train = extract_teacher_feats(train_dataset, batch_size=batch_size)  # [Ntrain, 2048]
print("Extracting teacher validation features...")
q_val   = extract_teacher_feats(val_dataset,   batch_size=batch_size)  # [Nval, 2048]

class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, q_array):
        assert len(base_dataset) == len(q_array)
        self.base = base_dataset
        self.q = q_array

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        q = torch.from_numpy(self.q[idx])  # [2048]
        return img, q

distill_train = DistillDataset(train_dataset, q_train)
distill_val   = DistillDataset(val_dataset,   q_val)

distill_train_loader = torch.utils.data.DataLoader(distill_train, batch_size=batch_size, shuffle=True,  num_workers=0)
distill_val_loader   = torch.utils.data.DataLoader(distill_val,   batch_size=batch_size, shuffle=False, num_workers=0)

# Student model
student_model = models.resnet50(pretrained=False)
num_features_student = student_model.fc.in_features  # 2048
student_model.fc = nn.Linear(num_features_student, num_classes)
student_model = student_model.to(device)
# Note: student wrapper has dropout
student_wrap = PenultimateWrapper(student_model, dropout_p=stage1_dropout_p).to(device)

# Stage 1: freeze fc, train backbone only
for p in student_wrap.fc.parameters():
    p.requires_grad = False

# Optimizer + cosine annealing with warm restarts — keep original logic
optim_stage1 = optim.AdamW(filter(lambda p: p.requires_grad, student_wrap.parameters()),
                           lr=1e-3, weight_decay=1e-4)
sched_stage1 = lr_scheduler.CosineAnnealingWarmRestarts(optim_stage1, T_0=50, T_mult=2)

def feature_mse_loss(student_feats_2048, q_batch):
    # Student and teacher features are L2-normalized before MSE
    p = F.normalize(student_feats_2048, p=2, dim=1)
    q = F.normalize(q_batch,            p=2, dim=1)
    return F.mse_loss(p, q)

def scale_lr_with_scheduler(optimizer, scheduler, factor):
    for pg in optimizer.param_groups:
        pg['lr'] = pg['lr'] * factor
    if hasattr(scheduler, 'base_lrs'):
        scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]

def get_current_lrs(optimizer):
    return [pg['lr'] for pg in optimizer.param_groups]

print("\n===== Stage 1 =====")
best_val_mse = float('inf')
best_stage1_state = copy.deepcopy(student_wrap.state_dict())

ema = None           # EMA based on validation MSE (only triggers LR reduction)
prev_ema = None
no_improve_epochs = 0

for epoch in range(num_epochs_stage1):
    # Training (Stage1 dropout enabled)
    student_wrap.set_dropout(True)
    student_wrap.train()
    running_loss = 0.0; ntrain = 0
    for imgs, q in distill_train_loader:
        imgs = imgs.to(device); q = q.to(device)
        optim_stage1.zero_grad()
        feats, _ = student_wrap(imgs)   # [B, 2048] (with dropout)
        loss = feature_mse_loss(feats, q)
        loss.backward()
        optim_stage1.step()
        running_loss += loss.item() * imgs.size(0)
        ntrain += imgs.size(0)
    train_mse = running_loss / max(1, ntrain)

    # Validation (Stage1 dropout disabled)
    student_wrap.set_dropout(False)
    student_wrap.eval()
    val_running = 0.0; nval = 0
    with torch.no_grad():
        for imgs, q in distill_val_loader:
            imgs = imgs.to(device); q = q.to(device)
            feats, _ = student_wrap(imgs)   # eval mode, no dropout
            vloss = feature_mse_loss(feats, q)
            val_running += vloss.item() * imgs.size(0)
            nval += imgs.size(0)
    val_mse = val_running / max(1, nval)

    # Validation MSE EMA (used only to trigger LR reduction)
    if ema is None:
        ema = val_mse
        prev_ema = ema
    else:
        ema = 0.9 * ema + 0.1 * val_mse

    improve = prev_ema - ema
    if improve < stage1_min_delta:
        no_improve_epochs += 1
    else:
        no_improve_epochs = 0
    prev_ema = ema

    # Print
    cur_lrs = get_current_lrs(optim_stage1)
    print(f"Epoch {epoch+1}/{num_epochs_stage1}  "
          f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  "
          f"EMA(val_mse)={ema:.6f}  patience={no_improve_epochs}  lr={cur_lrs}")

    # Save best by val_mse
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_stage1_state = copy.deepcopy(student_wrap.state_dict())
        torch.save(best_stage1_state, 'saved_models/student_stage1_best_by_valMSE.pth')
        print(f"Saved Stage1 best (val_mse={val_mse:.6f})")

    # If patience exceeded, LR /= 5 and continue cosine annealing
    if no_improve_epochs > stage1_patience_trigger:
        print(f"patience({no_improve_epochs}) > {stage1_patience_trigger}, learning rate divided by 5")
        scale_lr_with_scheduler(optim_stage1, sched_stage1, lr_scale_factor)
        no_improve_epochs = 0

    sched_stage1.step()

# Load Stage1 best (as Stage2 starting point)
student_wrap.load_state_dict(best_stage1_state)

print("\n===== Stage 2 =====")

# Freeze backbone; disable dropout
for p in student_wrap.stem.parameters():
    p.requires_grad = False
for p in student_wrap.avgpool.parameters():
    p.requires_grad = False
student_wrap.set_dropout(False)  # Explicitly disable

# Unfreeze fc
for p in student_wrap.fc.parameters():
    p.requires_grad = True

# Optimizer + StepLR (no cosine annealing & no EMA)
init_lr_stage2 = 1e-2
optim_stage2 = optim.AdamW(student_wrap.fc.parameters(), lr=init_lr_stage2, weight_decay=1e-4)
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

best_val_acc = 0.0
best_stage2_state = copy.deepcopy(student_wrap.state_dict())

for epoch in range(num_epochs_stage2):
    student_wrap.train()
    running_loss = 0.0; ns = 0
    for inputs, labels in train_loader_base:
        inputs = inputs.to(device); labels = labels.to(device)
        optim_stage2.zero_grad()
        feats, logits = student_wrap(inputs)
        loss = criterion_ce(logits, labels)
        loss.backward()
        optim_stage2.step()
        running_loss += loss.item() * inputs.size(0)
        ns += inputs.size(0)
    avg_train_loss = running_loss / max(1, ns)

    val_acc = evaluate_val_accuracy(student_wrap, val_loader_base)

    # Save Stage2 best (max val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_stage2_state = copy.deepcopy(student_wrap.state_dict())
        torch.save(best_stage2_state, 'saved_models/student_stage2_best_by_valAcc.pth')
        print(f"Saved Stage2 best (val_acc={val_acc:.4f})")

    # Print & StepLR
    cur_lr = [pg['lr'] for pg in optim_stage2.param_groups]
    print(f"Epoch {epoch+1}/{num_epochs_stage2}  train_ce={avg_train_loss:.4f}  val_acc={val_acc:.4f}  lr={cur_lr}")
    print("================")
    sched_stage2.step()

# Load Stage2 best for testing
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

print("\nEvaluating final student model on test set...")
test_acc, cm, all_preds, all_labels = evaluate_model_logits(student_wrap, test_loader_base)
print(f"Test set accuracy: {test_acc:.4f}")

# Save complete model
torch.save(student_wrap.state_dict(),
           'saved_models/stage1_dropout0.2_stage2_lr_halved_distilled_feature_student_final_model.pth')
print("All done.")
