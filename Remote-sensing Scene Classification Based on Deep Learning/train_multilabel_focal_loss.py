# -*- coding: utf-8 -*-
"""
Multi-label scene classification with a pretrained ResNet50 + Focal Loss.

Same structure as the BCE baseline (train_multilabel_bce.py); only the loss is replaced
with Focal Loss. Data, model, input size, batch size, optimizer, epochs, learning-rate
schedule, augmentation, threshold, and metrics are all identical.

Focal Loss (Lin et al., ICCV 2017):
    p_t  = p (y=1) or 1-p (y=0), with p = sigmoid(logit)
    FL   = -alpha_t * (1 - p_t)^gamma * log(p_t)
    alpha_t = alpha (y=1) or (1-alpha) (y=0)
Paper default hyperparameters: gamma = 2, alpha = 0.25.

Dataset: AID-ML (3000 images / 30 classes / 17 labels), images_tr:images_test = 8:2.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ============================ Hyperparameters ============================
CONFIG = {
    "data_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"),
    "csv_path": None,          # resolved below
    "image_size": 512,         # inputs are resized to 512x512
    "batch_size": 16,
    "epochs": 400,
    # Piecewise LR schedule: (last epoch, learning rate).
    "lr_schedule": [(200, 1e-5), (300, 1e-6), (400, 5e-7)],
    "weight_decay": 1e-4,
    "val_ratio": 0.10,         # fraction of the training set held out for validation
    "num_workers": 4,
    "threshold": 0.5,          # sigmoid threshold for predicting a positive class
    "seed": 42,

    # ---- Focal Loss hyperparameters ----
    "focal_gamma": 2.0,        # focusing parameter gamma
    "focal_alpha": 0.62,       # positive-class weight alpha
}
CONFIG["csv_path"] = os.path.join(CONFIG["data_dir"], "multilabel .csv")

LABELS = ["airplane", "bare-soil", "buildings", "cars", "chaparral", "court",
          "dock", "field", "grass", "mobile-home", "pavement", "sand", "sea",
          "ship", "tanks", "trees", "water"]
NUM_CLASSES = len(LABELS)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(epoch, schedule):
    """Return the learning rate for the given (1-based) epoch from a piecewise table."""
    for end, lr in schedule:
        if epoch <= end:
            return lr
    return schedule[-1][1]


# ============================ Loss: Focal Loss ============================
class FocalLoss(nn.Module):
    """
    Per-label binary Focal Loss for multi-label classification, computed in a numerically
    stable way (sigmoid and BCE combined).

    logits:  (B, C) raw model logits (no sigmoid applied).
    targets: (B, C) binary labels (0/1).
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * modulating * ce_loss
        return loss.mean()


# ============================ Model: pretrained ResNet50 ============================
def build_model(num_classes=NUM_CLASSES):
    """Load an ImageNet-pretrained ResNet50 and replace its fc layer with `num_classes` logits."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================ Dataset ============================
# Fixed 5x geometric augmentation variants (identity, flips, and 90-degree rotations).
VARIANTS = [
    lambda im: im,                                          # 0: identity
    lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),   # 1: horizontal flip
    lambda im: im.transpose(Image.Transpose.FLIP_TOP_BOTTOM),   # 2: vertical flip
    lambda im: im.transpose(Image.Transpose.ROTATE_270),        # 3: 90 degrees clockwise
    lambda im: im.transpose(Image.Transpose.ROTATE_90),         # 4: 90 degrees counter-clockwise
]


class MultiLabelDataset(Dataset):
    """When expand > 1, each source image is replicated into `expand` fixed variants."""

    def __init__(self, samples, transform=None, expand=1):
        self.samples = samples
        self.transform = transform
        self.expand = expand

    def __len__(self):
        return len(self.samples) * self.expand

    def __getitem__(self, idx):
        base_idx = idx // self.expand
        variant = idx % self.expand
        path, label = self.samples[base_idx]
        img = Image.open(path).convert("RGB")
        img = VARIANTS[variant](img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_samples(data_dir):
    """Walk images_tr / images_test and look up each image's 17-dim label from the CSV."""
    import csv
    label_map = {}
    with open(CONFIG["csv_path"], "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            name = row[0].strip()
            label = torch.tensor([int(v) for v in row[1:]], dtype=torch.float32)
            label_map[name] = label

    samples = []
    for split in ["images_tr", "images_test"]:
        split_dir = os.path.join(data_dir, split)
        for class_folder in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_folder)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                stem = os.path.splitext(fname)[0]
                if stem not in label_map:
                    raise ValueError(f"Image {fname} has no matching label in the CSV")
                samples.append((os.path.join(class_dir, fname), label_map[stem]))

    return samples


# ============================ Metrics ============================
def example_based_metrics(preds, targets):
    """Example-based P/R/F1/F2, averaged over samples (zero denominator -> 0)."""
    preds = preds.float()
    targets = targets.float()
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    p_denom = tp + fp
    r_denom = tp + fn
    precision = torch.where(p_denom > 0, tp / p_denom, torch.zeros_like(tp))
    recall = torch.where(r_denom > 0, tp / r_denom, torch.zeros_like(tp))

    denom_f1 = precision + recall
    f1 = torch.where(denom_f1 > 0, 2 * precision * recall / denom_f1, torch.zeros_like(precision))
    denom_f2 = 4 * precision + recall
    f2 = torch.where(denom_f2 > 0, 5 * precision * recall / denom_f2, torch.zeros_like(precision))

    return (precision.mean().item(), recall.mean().item(), f1.mean().item(), f2.mean().item())


def label_based_macro_metrics(preds, targets):
    """Label-based macro-averaged precision/recall (zero denominator -> 0)."""
    preds = preds.float()
    targets = targets.float()
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    p_denom = tp + fp
    r_denom = tp + fn
    precision = torch.where(p_denom > 0, tp / p_denom, torch.zeros_like(tp))
    recall = torch.where(r_denom > 0, tp / r_denom, torch.zeros_like(tp))
    return precision.mean().item(), recall.mean().item()


def evaluate(model, loader, criterion, device):
    """Return (mean loss, 6-metric dict)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > CONFIG["threshold"]).float()
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    ex_p, ex_r, ex_f1, ex_f2 = example_based_metrics(preds, targets)
    label_p, label_r = label_based_macro_metrics(preds, targets)

    metrics = {
        "ex_P": ex_p, "ex_R": ex_r,
        "label_P": label_p, "label_R": label_r,
        "ex_F1": ex_f1, "ex_F2": ex_f2,
    }
    return total_loss / len(loader.dataset), metrics


# ============================ Main ============================
def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Hyperparameters:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print(f"  device: {device}")
    print("=" * 60)

    # ---------- Data ----------
    all_samples = build_samples(cfg["data_dir"])
    print(f"Total images: {len(all_samples)}")

    train_samples = [s for s in all_samples if os.sep + "images_tr" + os.sep in s[0]]
    test_samples = [s for s in all_samples if os.sep + "images_test" + os.sep in s[0]]
    print(f"Train (images_tr): {len(train_samples)}, test (images_test): {len(test_samples)}")

    idx = list(range(len(train_samples)))
    random.shuffle(idx)
    val_size = int(len(train_samples) * cfg["val_ratio"])
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_base = [train_samples[i] for i in train_idx]
    val_base = [train_samples[i] for i in val_idx]

    # Training set uses the fixed 5x expansion; validation/test sets do not.
    train_ds = MultiLabelDataset(train_base, transform=transform, expand=5)
    val_ds = MultiLabelDataset(val_base, transform=transform, expand=1)
    test_ds = MultiLabelDataset(test_samples, transform=transform, expand=1)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)
    print(f"Train {len(train_base)} x5 = {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")

    # ---------- Model / loss / optimizer ----------
    model = build_model(NUM_CLASSES).to(device)
    criterion = FocalLoss(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr_schedule"][0][1],
                           weight_decay=cfg["weight_decay"])

    # ---------- Training ----------
    print("\nStarting training...")
    # Track two best models: best F1 over all epochs + best F1 over epochs 301-400.
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_val_f1_after300 = -1.0
    best_epoch_after300 = 0
    best_state_after300 = None

    for epoch in range(1, cfg["epochs"] + 1):
        lr_now = get_lr(epoch, cfg["lr_schedule"])
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_m = evaluate(model, val_loader, criterion, device)

        # Track the global best (all epochs).
        if val_m["ex_F1"] > best_val_f1:
            best_val_f1 = val_m["ex_F1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        # Track the best within epochs 301-400.
        if epoch > 300 and val_m["ex_F1"] > best_val_f1_after300:
            best_val_f1_after300 = val_m["ex_F1"]
            best_epoch_after300 = epoch
            best_state_after300 = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"[Epoch {epoch:3d}/{cfg['epochs']}] "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"ex_P={val_m['ex_P']:.4f} ex_R={val_m['ex_R']:.4f} "
              f"label_P={val_m['label_P']:.4f} label_R={val_m['label_R']:.4f} "
              f"F1={val_m['ex_F1']:.4f} F2={val_m['ex_F2']:.4f} | lr={lr_now:.1e}")

    # ---------- Evaluate and save both best models ----------
    def report_and_save(state, tag, val_f1, epoch):
        if state is None:
            print(f"\n[{tag}] no model available (epoch never exceeded 300), skipping")
            return
        model.load_state_dict(state)
        print("\n" + "=" * 60)
        print(f"[{tag}] best validation F1: {val_f1:.4f} (epoch {epoch})")
        test_loss, tm = evaluate(model, test_loader, criterion, device)
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test example-based  precision P={tm['ex_P']:.4f} | recall R={tm['ex_R']:.4f} "
              f"| F1={tm['ex_F1']:.4f} | F2={tm['ex_F2']:.4f}")
        print(f"Test label-based  precision P={tm['label_P']:.4f} | recall R={tm['label_R']:.4f}")
        print("=" * 60)
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), tag + ".pth")
        torch.save(state, save_path)
        print(f"[{tag}] model saved to: {save_path}")

    report_and_save(best_state, "best_model_focal", best_val_f1, best_epoch)
    report_and_save(best_state_after300, "best_model_focal_after300", best_val_f1_after300, best_epoch_after300)


if __name__ == "__main__":
    main()
