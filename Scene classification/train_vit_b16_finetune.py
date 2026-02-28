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
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dataset root directory
data_dir = 'dataset_raw'  # NWPU-RESISC45数据集根目录

# Training hyperparameters
batch_size = 50  # Original script batch size
num_epochs = 119 
num_classes = 45  # NWPU-RESISC45有45个类别

# Data preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


# Create datasets
def create_datasets(data_dir, train_ratio=0.2):
    """创建训练和验证数据集，按照论文使用20%训练，80%验证的比例"""
    full_dataset = datasets.ImageFolder(data_dir)

    class_indices = {}
    for idx, (path, label) in enumerate(full_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        split_point = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_dataset = TransformedDataset(train_subset, data_transforms['train'])
    val_dataset = TransformedDataset(val_subset, data_transforms['val'])

    return train_dataset, val_dataset, full_dataset.classes


print("Create datasets...")
train_dataset, val_dataset, class_names = create_datasets(data_dir, train_ratio=0.1)

# Use num_workers=0 to avoid multiprocessing issues
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
print(f"Training集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")

print("加载预Training ViT-B/16 模型...")

try:
    model = models.vit_b_16(weights=None)
except TypeError:
    model = models.vit_b_16(pretrained=False)

# Pretrained weights: load local first, otherwise download into pretrained_models/
vit_url = "https://download.pytorch.org/models/vit_b_16-c867db91.pth"
pretrained_path = "pretrained_models/vit_b_16-c867db91.pth"
os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)

if os.path.exists(pretrained_path):
    state_dict = torch.load(pretrained_path, map_location="cpu")
    print(f"已加载本地预Training权重: {pretrained_path}")
else:
    print("未找到本地预Training权重，尝试在线下载（需要联网）...")
    state_dict = torch.hub.load_state_dict_from_url(
        vit_url,
        model_dir=os.path.dirname(pretrained_path),  # download into pretrained_models/
        map_location="cpu",
        check_hash=True
    )
    print(f"已下载并Load pretrained weights到: {pretrained_path}")

# Load ImageNet pretrained weights first
model.load_state_dict(state_dict)

# Replace classifier head with num_classes (45)
def replace_vit_head(vit_model: nn.Module, out_dim: int) -> None:
    if not hasattr(vit_model, "heads"):
        raise AttributeError("当前 ViT 模型没有 heads 属性，无法替换分类头。请检查 torchvision 版本/模型定义。")

    heads = vit_model.heads

    if hasattr(heads, "head") and isinstance(heads.head, nn.Linear):
        in_features = heads.head.in_features
        heads.head = nn.Linear(in_features, out_dim)
        return

    if isinstance(heads, nn.Linear):
        in_features = heads.in_features
        vit_model.heads = nn.Linear(in_features, out_dim)
        return

    children = list(heads.named_children())
    if len(children) == 0:
        raise ValueError("heads 没有子模块，无法替换分类头。")
    last_name, last_module = children[-1]
    if isinstance(last_module, nn.Linear):
        in_features = last_module.in_features
        setattr(heads, last_name, nn.Linear(in_features, out_dim))
        return

    raise ValueError("未能识别 ViT 分类头结构（heads.*），无法自动替换，请把模型结构打印出来再改。")

replace_vit_head(model, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()

backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if name.startswith("heads"):
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = optim.SGD([
    {'params': backbone_params, 'lr': 0.001},
    {'params': head_params, 'lr': 0.01}
], momentum=0.9, weight_decay=0.0005)

# Learning-rate scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
                dataloader = train_loader
            else:
                model.eval() 
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'saved_models/best_model.pth')

        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f'Training完成于 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm, all_preds, all_labels


# Create directory for model checkpoints
os.makedirs('saved_models', exist_ok=True)

# Train model
print("开始Train model...")
model, history = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

# Save final model
torch.save(model.state_dict(), 'saved_models/final_model.pth')

# Evaluate model
print("Evaluate model...")
val_accuracy, cm, all_preds, all_labels = evaluate_model(model, val_loader)
print(f"验证集准确率: {val_accuracy:.4f}")
