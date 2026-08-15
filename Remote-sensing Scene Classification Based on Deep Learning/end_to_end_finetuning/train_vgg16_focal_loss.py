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
data_dir = 'dataset_raw'  # NWPU-RESISC45 dataset root directory

# Training hyperparameters (as in the reference paper)
batch_size = 50  # VGGNet-16 batch size
num_epochs = 119  # 15000 iterations total, ~126 iterations per epoch (6300/50=126, 15000/126~119)
num_classes = 45  # NWPU-RESISC45 has 45 classes

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


# Serializable dataset transform wrapper
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
    """Create train/validation datasets using the 20% train / 80% validation split from the paper"""
    full_dataset = datasets.ImageFolder(data_dir)

    # Collect sample indices per class
    class_indices = {}
    for idx, (path, label) in enumerate(full_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Split train/validation indices for each class
    train_indices = []
    val_indices = []

    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        split_point = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    # Create subsets
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    # Apply transforms
    train_dataset = TransformedDataset(train_subset, data_transforms['train'])
    val_dataset = TransformedDataset(val_subset, data_transforms['val'])

    return train_dataset, val_dataset, full_dataset.classes


# Create data loaders
print("Creating datasets...")
train_dataset, val_dataset, class_names = create_datasets(data_dir, train_ratio=0.1)

# Set num_workers=0 to avoid multiprocessing issues
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
print(f"Train set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")

# Load pretrained VGG16 model
print("Loading pretrained VGG16 model...")
model = models.vgg16(pretrained=False)  # Use local pretrained weights instead of downloading

# Load pretrained weights
pretrained_path = 'pretrained_models/vgg16-397923af.pth'
if os.path.exists(pretrained_path):
    model.load_state_dict(torch.load(pretrained_path))
    print(f"Pretrained weights loaded: {pretrained_path}")
else:
    print("Warning: pretrained weights not found, using randomly initialized weights")

# Replace the final layer to match the number of classes
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ====================== Only change here: switch to Focal Loss ======================
class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification (consistent with the original Focal Loss paper)"""
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [N, C]   targets: [N]
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)                    # probability of the correct class
        focal_weight = (1 - pt) ** self.gamma       # modulation factor
        loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Define loss function and optimizer
criterion = FocalLoss(gamma=2)   # <<< switch to Focal Loss (gamma=2 as in the paper)

# Set optimizer parameters as in the paper
# Higher learning rate (0.01) for the final layer, lower (0.001) for the rest
optimizer = optim.SGD([
    {'params': model.features.parameters(), 'lr': 0.001},
    {'params': model.classifier[:-1].parameters(), 'lr': 0.001},
    {'params': model.classifier[-1].parameters(), 'lr': 0.01}
], momentum=0.9, weight_decay=0.0005)

# Learning-rate scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# =====================================================================

# Training function (unchanged)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Track training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # training mode
                dataloader = train_loader
            else:
                model.eval()  # evaluation mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backpropagation and optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep-copy best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save best model
                torch.save(model.state_dict(), 'saved_models/best_model.pth')

        # Update learning rate after each epoch
        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


# Evaluation function (unchanged)
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

    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm, all_preds, all_labels


# Plot training history (not implemented in the original script, can be added later)




# Create directory for saving models
os.makedirs('saved_models', exist_ok=True)

# Train model
print("Starting model training...")
model, history = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

# Save final model
torch.save(model.state_dict(), 'saved_models/focal_vgg_final_model.pth')


# Evaluate model
print("Evaluating model...")
val_accuracy, cm, all_preds, all_labels = evaluate_model(model, val_loader)
print(f"Validation accuracy: {val_accuracy:.4f}")