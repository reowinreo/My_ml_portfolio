# -*- coding: utf-8 -*-
"""
Multi-label classification via sample ranking: pretrained ResNet50 + ApproxNDCG loss.

This implements the "sample ranking" task of the R4C framework, complementary to label
ranking: where label ranking fixes an image and ranks its 17 labels, sample ranking fixes
a label and ranks a batch of images.

  - The model outputs an M x 17 score matrix Z (M = batch size): rows are images,
    columns are labels.
  - Label ranking sorts the 17 scores of each row (image); sample ranking sorts the M
    scores of each column (label) — equivalently, apply the same ranking to the
    transposed matrix Z^T.
  - Hence the loss applies approx_ndcg to the transposed matrix:
      L_sample = -approx_ndcg(Z^T, Y^T).
  - Relevances: positive sample (image has this label) = 2, negative = 0,
    virtual neutral = 1.
  - Virtual neutral-sample scores are fixed constants (default 0.6 / 0.5 / 0.4, i.e.
    M_tilde = 3) that act as anchors: they take part in ranking and loss but receive
    no gradient.
  - At test time a label is predicted positive when its score exceeds `threshold` (0.5).

Dataset: AID-ML (3000 images / 30 classes / 17 labels), images_tr:images_test = 8:2.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ============================ Hyperparameters ============================
CONFIG = {
    "data_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"),
    "csv_path": None,          # resolved below

    # ---- Data ----
    "image_size": 512,         # inputs are resized to 512x512
    "val_ratio": 0.10,         # fraction of the training set held out for validation
    "expand": 5,               # fixed 5x training-set expansion (flips and rotations)

    # ---- Sample-ranking loss (ApproxNDCG) ----
    "virtual_scores": [0.6, 0.5, 0.4],  # fixed scores of the virtual neutral samples (M_tilde = 3)
    "virtual_relevance": 1.0,  # relevance of the virtual neutral samples
    "pos_relevance": 2.0,      # relevance of positive samples
    "neg_relevance": 0.0,      # relevance of negative samples
    "alpha": 10.0,             # sigmoid sharpness (10 as recommended by the ApproxNDCG paper)
    "threshold": 0.5,          # test threshold: score > 0.5 predicts positive

    # ---- Training ----
    "batch_size": 16,
    "epochs": 400,
    "lr_schedule": [(200, 1e-5), (300, 1e-6), (400, 5e-7)],  # piecewise LR
    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 42,
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


# ============================ Model: pretrained ResNet50 ============================
def build_model(num_classes=NUM_CLASSES):
    """Load an ImageNet-pretrained ResNet50 and replace its fc layer with `num_classes` scores."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================ Dataset ============================
# Fixed geometric augmentation variants (identity, flips, and 90-degree rotations).
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


# ============================ Ranking loss: ApproxNDCG ============================
def approx_ndcg(scores, labels, cfg=None):
    """
    Compute the mean ApproxNDCG over a batch of "queries"; differentiable.

    For label ranking: scores=(B,17) with one row per image, labels=(B,17).
    For sample ranking: scores=(N,B) with one row per label (the M x 17 matrix passed
    transposed), labels=(N,B) with each row the binary annotation of that label over images.

    scores: (Q, L) score matrix; each row is the list of scores for one query.
    labels: (Q, L) binary labels (1 = positive, 0 = negative).
    Returns a scalar NDCG in [0, 1]; train with loss = -approx_ndcg(...).
    """
    if cfg is None:
        cfg = CONFIG
    Q = scores.size(0)
    dev, dt = scores.device, scores.dtype
    K = len(cfg["virtual_scores"])

    # 1) Append the virtual neutral items: fixed constant scores with fixed relevance.
    virtual = torch.tensor(cfg["virtual_scores"], dtype=dt, device=dev).unsqueeze(0).expand(Q, -1)
    scores_full = torch.cat([scores, virtual], dim=1)                    # (Q, L+K)

    real_rel = torch.where(labels > 0.5,
                           torch.tensor(cfg["pos_relevance"], dtype=dt, device=dev),
                           torch.tensor(cfg["neg_relevance"], dtype=dt, device=dev))
    virt_rel = torch.full((Q, K), cfg["virtual_relevance"], dtype=dt, device=dev)
    rel_full = torch.cat([real_rel, virt_rel], dim=1)                    # (Q, L+K)

    N = scores_full.size(1)
    gain = 2.0 ** rel_full - 1.0                                         # positive=3, virtual=1, negative=0

    # 2) Approximate ranks: rank(i) ~ 1 + sum_{j != i} sigma(alpha * (s_j - s_i)).
    diff = scores_full.unsqueeze(2) - scores_full.unsqueeze(1)           # diff[q,j,i] = s_j - s_i
    sigma = torch.sigmoid(cfg["alpha"] * diff)
    mask = 1.0 - torch.eye(N, device=dev, dtype=dt)
    rank = 1.0 + (sigma * mask).sum(dim=1)                               # (Q,N), summed over j

    discount = 1.0 / torch.log2(1.0 + rank)
    approx_dcg = (gain * discount).sum(dim=1)                            # (Q,)

    # 3) Ideal DCG: gains sorted in descending order.
    gain_sorted, _ = torch.sort(gain, dim=1, descending=True)
    ideal_rank = torch.arange(1, N + 1, device=dev, dtype=dt)
    ideal_dcg = (gain_sorted / torch.log2(1.0 + ideal_rank)).sum(dim=1)
    ideal_dcg = ideal_dcg.clamp(min=1e-8)

    return (approx_dcg / ideal_dcg).mean()


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


def evaluate(model, loader, device, cfg=None):
    """Return (mean sample-ranking ApproxNDCG, 6-metric dict)."""
    model.eval()
    total_ndcg = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                       # (M,17)
            # Sample ranking: apply ApproxNDCG to the transposed matrix (rank M images per label).
            ndcg = approx_ndcg(outputs.t(), labels.t(), cfg)
            total_ndcg += ndcg.item() * images.size(0)
            preds = (outputs > cfg["threshold"]).float()
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
    return total_ndcg / len(loader.dataset), metrics


# ============================ Main ============================
def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Hyperparameters:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print(f"  device: {device}")
    print("=" * 70)

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
    train_ds = MultiLabelDataset(train_base, transform=transform, expand=cfg["expand"])
    val_ds = MultiLabelDataset(val_base, transform=transform, expand=1)
    test_ds = MultiLabelDataset(test_samples, transform=transform, expand=1)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)
    print(f"Train {len(train_base)} x{cfg['expand']} = {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")

    # ---------- Model / optimizer ----------
    model = build_model(NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr_schedule"][0][1],
                           weight_decay=cfg["weight_decay"])

    # ---------- Training ----------
    print("\nStarting training (sample-ranking loss = -ApproxNDCG(Z^T, Y^T))...")
    for epoch in range(1, cfg["epochs"] + 1):
        lr_now = get_lr(epoch, cfg["lr_schedule"])
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        model.train()
        running_ndcg = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)                       # (M,17)
            # Sample ranking: transposed matrix ranks M images per label.
            ndcg = approx_ndcg(outputs.t(), labels.t(), cfg)
            loss = -ndcg
            loss.backward()
            optimizer.step()
            running_ndcg += ndcg.item() * images.size(0)

        train_ndcg = running_ndcg / len(train_ds)
        val_ndcg, val_m = evaluate(model, val_loader, device, cfg)

        print(f"[Epoch {epoch:3d}/{cfg['epochs']}] "
              f"train_NDCG={train_ndcg:.4f} | val_NDCG={val_ndcg:.4f} | "
              f"ex_P={val_m['ex_P']:.4f} ex_R={val_m['ex_R']:.4f} "
              f"label_P={val_m['label_P']:.4f} label_R={val_m['label_R']:.4f} "
              f"F1={val_m['ex_F1']:.4f} F2={val_m['ex_F2']:.4f} | lr={lr_now:.1e}")

    # ---------- Test-set evaluation ----------
    print("\n" + "=" * 70)
    test_ndcg, tm = evaluate(model, test_loader, device, cfg)
    print(f"Test ApproxNDCG (sample ranking): {test_ndcg:.4f}")
    print(f"Test example-based  precision P={tm['ex_P']:.4f} | recall R={tm['ex_R']:.4f} "
          f"| F1={tm['ex_F1']:.4f} | F2={tm['ex_F2']:.4f}")
    print(f"Test label-based  precision P={tm['label_P']:.4f} | recall R={tm['label_R']:.4f}")
    print("=" * 70)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_sample_ranking.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
