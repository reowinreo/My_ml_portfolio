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
    y = np.argmax(y, axis=1)                 # one-hot to class indices
    return X, y


# Triplet dataset
class TripletDataset(Dataset):
    def __init__(self, X, y, num_triplets=60000):
        self.X = X
        self.y = y
        self.triplets = []

        n_classes = len(np.unique(y))
        class_dict = {i: np.where(y == i)[0] for i in range(n_classes)}

        for _ in range(num_triplets):
            c = np.random.choice(n_classes)
            anchor, positive = np.random.choice(class_dict[c], 2, replace=False)
            neg_class = np.random.choice([i for i in range(n_classes) if i != c])
            negative = np.random.choice(class_dict[neg_class])
            self.triplets.append([X[anchor], X[positive], X[negative]])

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        return (
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(n, dtype=torch.float32),
        )


# Triplet loss
class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = F.pairwise_distance(anchor, positive, keepdim=True)
        d_neg = F.pairwise_distance(anchor, negative, keepdim=True)
        losses = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        return torch.mean(losses)


# CNN
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
        x = F.normalize(x, p=2, dim=1)  # normalize
        return x


class TripletCNN(nn.Module):
    def __init__(self):
        super(TripletCNN, self).__init__()
        self.branch = SimpleCNN()

    def forward(self, a, p, n):
        fa = self.branch(a)
        fp = self.branch(p)
        fn = self.branch(n)
        return fa, fp, fn


# Training function
def train_triplet(model, dataloader, device):
    criterion = TripletLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(10):
        total_loss = 0
        for a, p, n in dataloader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            fa, fp, fn = model(a, p, n)
            loss = criterion(fa, fp, fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


# Extract features
def extract_features(model, X, device):
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            batch = torch.tensor(X[i:i+128], dtype=torch.float32).to(device)
            f = model.branch(batch)
            features.append(f.cpu().numpy())
    return np.vstack(features)


# Main pipeline
if __name__ == "__main__":
    #SAT-6
    X_train, y_train = load_data("X_train_sat6.csv", "y_train_sat6.csv", num_channels=4)
    X_test, y_test   = load_data("X_test_sat6.csv",  "y_test_sat6.csv",  num_channels=4)

    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    idx = np.arange(len(y_all))
    np.random.shuffle(idx)
    split = int(0.8 * len(y_all))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test   = X_all[test_idx], y_all[test_idx]

    #Build triplet training set
    train_dataset = TripletDataset(X_train, y_train, num_triplets=200000)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    #Train the CNN encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletCNN().to(device)
    train_triplet(model, train_loader, device)

    #Extract features and train SVM classifier
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
