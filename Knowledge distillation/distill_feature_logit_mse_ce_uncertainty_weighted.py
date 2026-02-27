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

#基本设置
data_dir = 'dataset_raw'
split_path = 'saved_models/split_indices.npz'
batch_size = 128
num_classes = 45
num_epochs = 500
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
print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")
print(f"测试集大小: {dataset_sizes['test']}")

class PenultimateWrapper(nn.Module):
    """
    输出:
      feats: [B, 2048]  (avgpool后展平，位于fc之前)
      logits: [B, num_classes]  (fc输出)
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
    p.requires_grad = False 

#Student: ResNet50
student_backbone = models.resnet50(pretrained=False)
nf_s = student_backbone.fc.in_features  # 2048
student_backbone.fc = nn.Linear(nf_s, num_classes)
student_backbone = student_backbone.to(device)
student = PenultimateWrapper(student_backbone).to(device)

#特征MSE 对率MSE 硬标签CE
def feature_mse_loss(student_feats_2048, teacher_feats_2048):
    # L2 归一化后做 MSE
    p = torch.nn.functional.normalize(student_feats_2048, p=2, dim=1)
    q = torch.nn.functional.normalize(teacher_feats_2048, p=2, dim=1)
    return torch.nn.functional.mse_loss(p, q)

def logit_mse_loss(student_logits, teacher_logits):
    t = teacher_logits - teacher_logits.mean(dim=1, keepdim=True)
    s = student_logits - student_logits.mean(dim=1, keepdim=True)
    return torch.nn.functional.mse_loss(s, t)

ce_criterion = nn.CrossEntropyLoss()

#不确定性加权
class UncertaintyWeightedFeatLogitCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_feat  = nn.Parameter(torch.zeros(1))
        self.log_sigma_logit = nn.Parameter(torch.zeros(1))
        self.log_sigma_ce    = nn.Parameter(torch.zeros(1))

    def forward(self, L_feat, L_logit, L_ce):
        sigma_feat  = torch.exp(self.log_sigma_feat)
        sigma_logit = torch.exp(self.log_sigma_logit)
        sigma_ce    = torch.exp(self.log_sigma_ce)
        loss = (L_feat  / (2 * sigma_feat**2)) \
             + (L_logit / (2 * sigma_logit**2)) \
             + (L_ce    / (2 * sigma_ce**2)) \
             + torch.log(sigma_feat) + torch.log(sigma_logit) + torch.log(sigma_ce)
        # 返回中间量（仅打印监控用）
        return loss, sigma_feat.detach(), sigma_logit.detach(), sigma_ce.detach()

criterion = UncertaintyWeightedFeatLogitCE().to(device)

#两组参数 + 余弦热重启
optimizer = optim.AdamW([
    {"params": student.parameters()},
    {"params": criterion.parameters(), "lr": 0.001}  # 学习不确定性参数
], lr=0.001, weight_decay=1e-4)

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

#训练与评估
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

            # teacher 前向
            with torch.no_grad():
                t_feats, t_logits = teacher(inputs)

            # student 前向
            s_feats, s_logits = student(inputs)

            L_feat  = feature_mse_loss(s_feats,  t_feats)
            L_logit = logit_mse_loss( s_logits, t_logits)
            L_ce    = ce_criterion(s_logits, labels)

            # 动态加权总损失
            loss, sigma_feat_det, sigma_logit_det, sigma_ce_det = criterion(L_feat, L_logit, L_ce)

            # 训练准确率
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

                L_feat  = feature_mse_loss(s_feats, t_feats)
                L_logit = logit_mse_loss(s_logits, t_logits)
                L_ce    = ce_criterion(s_logits, labels)

                # 验证总损失（不回传），用于监控
                val_loss, _, _, _ = criterion(L_feat, L_logit, L_ce)

                _, preds = torch.max(s_logits, 1)

                running_loss_val += val_loss.item() * inputs.size(0)
                running_corrects_val += torch.sum(preds == labels.data)

        epoch_loss_val = running_loss_val / dataset_sizes['val']
        epoch_acc_val  = running_corrects_val.double() / dataset_sizes['val']
        history['val_loss'].append(epoch_loss_val)
        history['val_acc'].append(epoch_acc_val.item())

        print(f'val  Loss: {epoch_loss_val:.4f} Acc: {epoch_acc_val:.4f}')

        # 保存最佳
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(student.state_dict())
            torch.save(student.state_dict(), 'saved_models/best_model.pth')
            print(f"保存最优模型: epoch {epoch}, val_acc={epoch_acc_val:.4f}")

        scheduler.step()
        print(f"当前学习率: {[pg['lr'] for pg in optimizer.param_groups]}")

    time_elapsed = time.time() - since
    print(f'训练完成于 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')

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

print("开始训练模型（特征MSE + 对率MSE + 硬标签CE，动态权重）...")
student, history = train_model(student, optimizer, scheduler, num_epochs=num_epochs)

# 画准确率曲线
epochs_axis = np.arange(len(history['train_acc']))
plt.figure()
plt.plot(epochs_axis, history['train_acc'], label='Train Acc')
plt.plot(epochs_axis, history['val_acc'],   label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
try:
    plt.show()
except Exception:
    pass

torch.save(student.state_dict(), 'saved_models/硬标签_feat-mse+logit-mse_uncertainty-weighted_resnet50_student_final.pth')

print("在测试集上评估...")
test_acc, cm, all_preds, all_labels = evaluate_model(student, test_loader)
print(f"测试集准确率: {test_acc:.4f}")
