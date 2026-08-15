import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import svm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


# Helper function for this experiment module
def load_data(X_path, y_path, num_channels=4):
    X = np.loadtxt(X_path, delimiter=",")
    y = np.loadtxt(y_path, delimiter=",")
    X = X.reshape(-1, num_channels, 28, 28)  # SAT-6 per image 28x28
    y = np.argmax(y, axis=1)  # one-hot to class indices
    return X, y


class PairDataset(Dataset):
    def __init__(self, X, y, num_pairs=60000):
        self.X = X
        self.y = y
        self.pairs = []
        self.labels = []

        n_classes = len(np.unique(y))
        class_dict = {i: np.where(y == i)[0] for i in range(n_classes)}

        num_pos = num_pairs // 2
        per_class = num_pos // n_classes
        for c in range(n_classes):
            for _ in range(per_class):
                i1, i2 = np.random.choice(class_dict[c], 2, replace=False)
                self.pairs.append([X[i1], X[i2]])
                self.labels.append(1)

        num_neg = num_pairs // 2
        class_combinations = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
        per_combo = num_neg // len(class_combinations)

        for (c1, c2) in class_combinations:
            for _ in range(per_combo):
                i1 = np.random.choice(class_dict[c1])
                i2 = np.random.choice(class_dict[c2])
                self.pairs.append([X[i1], X[i2]])
                self.labels.append(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        y = self.labels[idx]
        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0): 
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# CNN backbone (deeper variant)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        # -------- Feature normalization --------
        x = F.normalize(x, p=2, dim=1)
        return x


class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.branch = SimpleCNN()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        f1 = self.branch(x1)
        f2 = self.branch(x2)
        diff = torch.abs(f1 - f2)
        pred = self.classifier(diff)
        return f1, f2, pred


# Training function (contrastive loss + BCE)
def train_siamese(model, dataloader, device):
    contrastive_loss = ContrastiveLoss(margin=2.0)
    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for x1, x2, labels in dataloader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            f1, f2, pred = model(x1, x2)
            loss1 = contrastive_loss(f1, f2, labels)
            loss2 = bce_loss(pred.squeeze(), labels)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# Extract features, normalized before SVM classifier
def extract_features(model, X, device):
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            batch = torch.tensor(X[i:i + 128], dtype=torch.float32).to(device)
            f = model.branch(batch)  # already normalized in forward
            features.append(f.cpu().numpy())
    return np.vstack(features)


# Main pipeline
if __name__ == "__main__":
    #SAT-6
    X_train, y_train = load_data("X_train_sat6.csv", "y_train_sat6.csv", num_channels=4)
    X_test, y_test = load_data("X_test_sat6.csv", "y_test_sat6.csv", num_channels=4)

    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    idx = np.arange(len(y_all))
    np.random.shuffle(idx)
    split = int(0.8 * len(y_all))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    #Build training pairs
    train_dataset = PairDataset(X_train, y_train, num_pairs=200000)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    #Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseCNN().to(device)
    train_siamese(model, train_loader, device)

    #SVM classifier
    svm_train_sample = np.random.choice(len(train_idx), 20000, replace=False)
    X_svm_train_raw = X_train[svm_train_sample]
    y_svm_train = y_train[svm_train_sample]
    X_svm_train = extract_features(model, X_svm_train_raw, device)

    svm_test_sample = np.random.choice(len(test_idx), 20000, replace=False)
    X_svm_test_raw = X_test[svm_test_sample]
    y_svm_test = y_test[svm_test_sample]
    X_svm_test = extract_features(model, X_svm_test_raw, device)
    clf = svm.SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X_svm_train, y_svm_train)

    y_pred = clf.predict(X_svm_test)
    acc = accuracy_score(y_svm_test, y_pred)
    print(f"SVM Accuracy: {acc:.4f}")
