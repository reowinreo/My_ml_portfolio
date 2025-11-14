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
from sklearn.decomposition import PCA

data_dir = 'dataset_raw'
split_path = 'saved_models/split_indices.npz'
batch_size = 128
num_classes = 45

# 训练轮次
num_epochs_stage1 = 400
num_epochs_stage2 = 200

# Stage1 的 EMA/LR 缩放规则（保持原逻辑）
stage1_min_delta = 3e-7
stage1_patience_trigger = 10
lr_scale_factor = 1.0 / 5.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs('saved_models', exist_ok=True)

# 变换 & 数据集构造
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
print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")
print(f"测试集大小: {dataset_sizes['test']}")

# --------------------
# 老师模型（ResNet152）
# --------------------
teacher_model = models.resnet152(pretrained=False)
num_features_teacher = teacher_model.fc.in_features  # 2048
teacher_model.fc = nn.Linear(num_features_teacher, num_classes)
teacher_model.load_state_dict(torch.load('saved_models/resnet152_final_model.pth', map_location='cpu'))
teacher_model = teacher_model.to(device)
teacher_model.eval()

# 提取倒数第二层（无 dropout 版本）
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
        x = self.avgpool(x)         # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)     # [B, 2048]
        return x

    def forward(self, x):
        feats = self.forward_features(x)  # 2048-D
        logits = self.fc(feats)           # num_classes
        return feats, logits

teacher_wrap = PenultimateWrapper(teacher_model).to(device).eval()

# ---------- 提取老师 2048D 归一化特征 ----------
def extract_teacher_feats(dataset, batch_size=128):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_feats = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            feats, _ = teacher_wrap(inputs)      # [B, 2048]
            feats = F.normalize(feats, p=2, dim=1)  # L2 归一化
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype(np.float32)

print("提取老师训练集特征")
q_train_2048 = extract_teacher_feats(train_dataset, batch_size=batch_size)  # [Ntrain, 2048]
print("提取老师验证集特征")
q_val_2048   = extract_teacher_feats(val_dataset,   batch_size=batch_size)  # [Nval, 2048]

pca = PCA(n_components=0.90, svd_solver='full', whiten=False)
pca.fit(q_train_2048)   # 在训练集归一化特征上拟合

k = pca.components_.shape[0]     # 自适应得到 k
pca_mean = pca.mean_.astype(np.float32)          # [2048]
pca_P = pca.components_.astype(np.float32)       # [k, 2048]
print(f"PCA 完成：k={k}")

# 用同一 (mean, P) 投影老师的训练/验证特征，得到 k 维 q
def np_project_k(q_2048: np.ndarray, mean_vec: np.ndarray, P_k_2048: np.ndarray):
    # (N,2048) -> (N,k)
    return (q_2048 - mean_vec[None, :]) @ P_k_2048.T

q_train_k = np_project_k(q_train_2048, pca_mean, pca_P).astype(np.float32)  # [Ntrain, k]
q_val_k   = np_project_k(q_val_2048,   pca_mean, pca_P).astype(np.float32)  # [Nval,   k]

# 将 mean 与 P 放到 GPU，以便训练时对学生特征做相同投影
pca_mean_t = torch.from_numpy(pca_mean).to(device)          # [2048]
pca_P_t    = torch.from_numpy(pca_P).to(device)             # [k, 2048]

# ---------- 蒸馏数据集：返回 (image, q_k) ----------
class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, q_k_array):
        assert len(base_dataset) == len(q_k_array)
        self.base = base_dataset
        self.qk = q_k_array

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        qk = torch.from_numpy(self.qk[idx])  # [k]
        return img, qk

distill_train = DistillDataset(train_dataset, q_train_k)
distill_val   = DistillDataset(val_dataset,   q_val_k)

distill_train_loader = torch.utils.data.DataLoader(distill_train, batch_size=batch_size, shuffle=True,  num_workers=0)
distill_val_loader   = torch.utils.data.DataLoader(distill_val,   batch_size=batch_size, shuffle=False, num_workers=0)

# --------------------
# 学生模型（ResNet50）
# --------------------
student_model = models.resnet50(pretrained=False)
num_features_student = student_model.fc.in_features  # 2048
student_model.fc = nn.Linear(num_features_student, num_classes)
student_model = student_model.to(device)
student_wrap = PenultimateWrapper(student_model).to(device)

# 阶段1：冻结 fc，只训练骨干
for p in student_wrap.fc.parameters():
    p.requires_grad = False

# 阶段1优化器 + 余弦退火（热重启）——保持原逻辑
optim_stage1 = optim.AdamW(filter(lambda p: p.requires_grad, student_wrap.parameters()),
                           lr=1e-3, weight_decay=1e-4)
sched_stage1 = lr_scheduler.CosineAnnealingWarmRestarts(optim_stage1, T_0=50, T_mult=2)

def project_student_to_k(student_feats_2048: torch.Tensor) -> torch.Tensor:
    """
    学生 2048D -> 归一化 -> 用老师的 mean/P 投影到 k 维
    inputs: [B, 2048]
    returns: [B, k]
    """
    p_norm = F.normalize(student_feats_2048, p=2, dim=1)     # [B, 2048]
    centered = p_norm - pca_mean_t.unsqueeze(0)              # [B, 2048]
    kvec = torch.matmul(centered, pca_P_t.t())               # [B, k]
    return kvec

def feature_k_mse_loss(student_feats_2048, qk_batch):
    pk = project_student_to_k(student_feats_2048)   # [B, k]
    return F.mse_loss(pk, qk_batch)

def scale_lr_with_scheduler(optimizer, scheduler, factor):
    for pg in optimizer.param_groups:
        pg['lr'] = pg['lr'] * factor
    if hasattr(scheduler, 'base_lrs'):
        scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]

def get_current_lrs(optimizer):
    return [pg['lr'] for pg in optimizer.param_groups]

# --------------------
# 阶段1：PCA特征对齐（用 q_k 做监督），以 val_MSE 最小保存；EMA(val_MSE) 仅触发降LR
# --------------------
print("\n===== 阶段 1 =====")
best_val_mse = float('inf')
best_stage1_state = copy.deepcopy(student_wrap.state_dict())

ema = None
prev_ema = None
no_improve_epochs = 0

for epoch in range(num_epochs_stage1):
    # Train
    student_wrap.train()
    running_loss = 0.0; ntrain = 0
    for imgs, qk in distill_train_loader:
        imgs = imgs.to(device); qk = qk.to(device)
        optim_stage1.zero_grad()
        feats, _ = student_wrap(imgs)              # [B, 2048]
        loss = feature_k_mse_loss(feats, qk)       # MSE(p_k, q_k)
        loss.backward()
        optim_stage1.step()
        running_loss += loss.item() * imgs.size(0)
        ntrain += imgs.size(0)
    train_mse = running_loss / max(1, ntrain)

    # Val
    student_wrap.eval()
    val_running = 0.0; nval = 0
    with torch.no_grad():
        for imgs, qk in distill_val_loader:
            imgs = imgs.to(device); qk = qk.to(device)
            feats, _ = student_wrap(imgs)
            vloss = feature_k_mse_loss(feats, qk)
            val_running += vloss.item() * imgs.size(0)
            nval += imgs.size(0)
    val_mse = val_running / max(1, nval)

    # EMA(val_mse) -> 只用于触发 LR÷5
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

    # log
    cur_lrs = get_current_lrs(optim_stage1)
    print(f"Epoch {epoch+1}/{num_epochs_stage1}  "
          f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  "
          f"EMA(val_mse)={ema:.6f}  patience={no_improve_epochs}  lr={cur_lrs}")

    # save best by val_mse
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_stage1_state = copy.deepcopy(student_wrap.state_dict())
        torch.save(best_stage1_state, 'saved_models/student_stage1_best_by_valMSE.pth')
        print(f"保存阶段1最优（val_mse={val_mse:.6f}）")

    # LR decay trigger
    if no_improve_epochs > stage1_patience_trigger:
        print(f"patience({no_improve_epochs}) > {stage1_patience_trigger}，学习率除以5")
        scale_lr_with_scheduler(optim_stage1, sched_stage1, lr_scale_factor)
        no_improve_epochs = 0

    sched_stage1.step()

# 载入阶段1最优（用于阶段2起点）
student_wrap.load_state_dict(best_stage1_state)

# --------------------
# 阶段 2（仅训练 fc）：LR=0.01，StepLR 每 10 epoch ×0.5；无 EMA
# --------------------
print("\n===== 阶段 2 =====")

# 冻结骨干
for p in student_wrap.stem.parameters():
    p.requires_grad = False
for p in student_wrap.avgpool.parameters():
    p.requires_grad = False

# 解冻 fc
for p in student_wrap.fc.parameters():
    p.requires_grad = True

# 优化器 + StepLR（不使用余弦退火 & 无 EMA）
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

    # 保存阶段2最优（val_acc 最大）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_stage2_state = copy.deepcopy(student_wrap.state_dict())
        torch.save(best_stage2_state, 'saved_models/student_stage2_best_by_valAcc.pth')
        print(f"保存阶段2最优（val_acc={val_acc:.4f}）")

    # 打印 & StepLR
    cur_lr = [pg['lr'] for pg in optim_stage2.param_groups]
    print(f"Epoch {epoch+1}/{num_epochs_stage2}  train_ce={avg_train_loss:.4f}  val_acc={val_acc:.4f}  lr={cur_lr}")
    print("================")
    sched_stage2.step()

# 载入阶段2最优进行测试
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

print("\n在测试集上评估最终学生模型...")
test_acc, cm, all_preds, all_labels = evaluate_model_logits(student_wrap, test_loader_base)
print(f"测试集准确率: {test_acc:.4f}")

# 完整保存
torch.save(student_wrap.state_dict(), 'saved_models/一二阶段distilled_feature_student_final_model.pth')
print("全部完成。")
