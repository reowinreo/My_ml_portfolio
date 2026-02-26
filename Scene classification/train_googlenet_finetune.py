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

torch.manual_seed(42)
np.random.seed(42)

# 数据目录
data_dir = 'dataset_raw'

# 训练参数
batch_size = 128
num_classes = 45

# 数据预处理
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


def create_datasets(data_dir, train_ratio=0.1):
    full_dataset = datasets.ImageFolder(data_dir)
    class_indices = {}
    for idx, (path, label) in enumerate(full_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_indices, val_indices = [], []
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


print("创建数据集...")
train_dataset, val_dataset, class_names = create_datasets(data_dir, train_ratio=0.1)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")

# 计算 epoch 数
train_samples = dataset_sizes['train']
target_iterations = 15000
num_epochs = 119
print(f"使用 {num_epochs} 个 epoch ≈ 15000 iterations")

# 加载模型
print("加载预训练GoogLeNet模型...")
model = models.googlenet(pretrained=False, aux_logits=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

pretrained_path = 'pretrained_models/googlenet.pth'
if os.path.exists(pretrained_path):
    state_dict = torch.load(pretrained_path, map_location="cpu")
    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    model.load_state_dict(state_dict, strict=False)
    print(f"已加载预训练权重: {pretrained_path}")
else:
    print("警告: 未找到预训练权重文件，将使用随机初始化的权重")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# 参数分组
fc_params = list(model.fc.parameters())
fc_param_ids = {id(p) for p in fc_params}
base_params = [p for p in model.parameters() if id(p) not in fc_param_ids]

optimizer = optim.SGD([
    {'params': base_params, 'lr': 1e-5},   # 初始 lr=1e-5
    {'params': fc_params, 'lr': 1e-2}      # 初始 lr=1e-2
], momentum=0.9, weight_decay=5e-4)

def lr_lambda_base(epoch):
    # 前10个epoch线性从1e-5到1e-3
    if epoch < 10:
        return (1e-5 + (1e-3 - 1e-5) * (epoch / 9)) / 1e-5
    else:
        return 1e-3 / 1e-5  # 保持在1e-3

def lr_lambda_fc(epoch):
    # 最后10个epoch线性从1e-2降到1e-4
    if epoch < num_epochs - 10:
        return 1.0  # 保持在1e-2
    else:
        progress = (epoch - (num_epochs - 10)) / 9
        target = 1e-2 + (1e-4 - 1e-2) * progress
        return target / 1e-2


scheduler = lr_scheduler.LambdaLR(optimizer,
                                  lr_lambda=[lr_lambda_base, lr_lambda_fc])


def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

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
                inputs, labels = inputs.to(device), labels.to(device)
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
            history[phase + '_loss'].append(epoch_loss)
            history[phase + '_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'saved_models/best_model.pth')

        # 更新学习率
        scheduler.step()
        print(f"当前学习率: {[group['lr'] for group in optimizer.param_groups]}")

    time_elapsed = time.time() - since
    print(f'训练完成于 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')

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



def freeze_layers(model, freeze_until="inception3b"):
    """
    冻结 GoogLeNet 的部分层参数。
    """
    freeze_list = ["conv1", "conv2", "inception3a", "inception3b"]

    if freeze_until not in freeze_list:
        raise ValueError(f"freeze_until 必须是 {freeze_list} 之一")
    stop_idx = freeze_list.index(freeze_until)

    for name in freeze_list[:stop_idx+1]:
        layer = getattr(model, name)
        for param in layer.parameters():
            param.requires_grad = False

    print(f"已冻结层: {freeze_list[:stop_idx+1]}")


os.makedirs('saved_models', exist_ok=True)
freeze_layers(model, freeze_until="inception3b")

print("开始训练模型...")
model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

torch.save(model.state_dict(), 'saved_models/final_model.pth')


print("评估模型...")
val_acc, cm, all_preds, all_labels = evaluate_model(model, val_loader)
print(f"验证集准确率: {val_acc:.4f}")

