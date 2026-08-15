import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = 'dataset_raw'  # NWPU-RESISC45 root directory
batch_size = 128
num_classes = 45

# ==================== Data preprocessing ====================
# Align 'train' transforms with 'val' and remove random ops for consistency
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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

# ==================== Dataset splitting ====================
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


print("Creating datasets...")
train_dataset, val_dataset, class_names = create_datasets(data_dir, train_ratio=0.1)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
print(f"Train set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")

# ==================== Load GoogLeNet ====================
print("Loading pretrained GoogLeNet model...")
model = models.googlenet(pretrained=False, aux_logits=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Load pretrained weights (excluding the final fc)
pretrained_path = 'pretrained_models/googlenet.pth'
if os.path.exists(pretrained_path):
    state_dict = torch.load(pretrained_path, map_location="cpu")
    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    model.load_state_dict(state_dict, strict=False)
    print(f"Pretrained weights loaded: {pretrained_path}")
else:
    print("Warning: pretrained weights not found, using randomly initialized weights")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)


def extract_features(model, dataloader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)

            # Forward propagate to the penultimate layer
            x = model.conv1(inputs)
            x = model.maxpool1(x)
            x = model.conv2(x)
            x = model.conv3(x)
            x = model.maxpool2(x)
            x = model.inception3a(x)
            x = model.inception3b(x)
            x = model.maxpool3(x)
            x = model.inception4a(x)
            x = model.inception4b(x)
            x = model.inception4c(x)
            x = model.inception4d(x)
            x = model.inception4e(x)
            x = model.maxpool4(x)
            x = model.inception5a(x)
            x = model.inception5b(x)
            x = model.avgpool(x)            # (batch, 1024, 1, 1)
            x = torch.flatten(x, 1)         # (batch, 1024)

            features.append(x.cpu().numpy())
            labels.append(targets.numpy())

    return np.concatenate(features), np.concatenate(labels)


# ==================== Extract features ====================
train_features, train_labels = extract_features(model, train_loader, device)
val_features, val_labels = extract_features(model, val_loader, device)

print(f"Training feature dimension: {train_features.shape}")
print(f"Validation feature dimension: {val_features.shape}")

# ==================== Feature normalization ====================
train_features = normalize(train_features, norm='l2')
val_features = normalize(val_features, norm='l2')

# ==================== Classify with LibSVM (Linear) ====================
print("Training Linear SVM (LibSVM) classifier...")
param_grid = {'C': [0.1, 1, 10]}
clf = GridSearchCV(SVC(kernel='linear'), param_grid, cv=3)
clf.fit(train_features, train_labels)

val_preds = clf.predict(val_features)
acc = accuracy_score(val_labels, val_preds)
cm = confusion_matrix(val_labels, val_preds)
print(f"Validation accuracy: {acc:.4f}")