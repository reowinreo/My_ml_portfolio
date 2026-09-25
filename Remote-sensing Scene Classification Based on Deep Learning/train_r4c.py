# -*- coding: utf-8 -*-
"""
Full R4C multi-label classification: pretrained ResNet50 + three listwise ranking tasks.

Paper: R4C (Ranking for Classification). Three listwise tasks share a single model:
  - L2R-label   (label ranking): fix an image, rank its 17 labels (+ N_hat virtual neutral labels).
  - L2R-sample  (sample ranking): fix a label, rank the M images of a batch (+ M_tilde virtual neutral samples).
  - L2R-feature (feature ranking): fix an image, rank the other M-1 images by feature
    Euclidean distance (relevance = label Jaccard similarity).

Total loss: L_R4C = L_label + L_sample + lambda * L_feature, with lambda = 1/8.

Model (one shared backbone with two readout points):
      ResNet50 backbone
          |-- 2048-dim global feature f (penultimate layer) --> feature ranking (Euclidean distance)
          `-- fc(2048 -> 17) -> Sigmoid -> z in (0,1) (17 probabilities)
                                           |--> label ranking (per row)
                                           `--> sample ranking (per column)

Task switches: use_label / use_sample / use_feature toggle each task for ablations.
(Note: with only use_feature enabled, the fc classification head receives no gradient,
so feature ranking should be combined with label/sample ranking.)

All hyperparameters follow the paper's AID-ML setting.
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

    # ---- Task switches (for ablations) ----
    "use_label": True,         # label ranking
    "use_sample": True,        # sample ranking
    "use_feature": True,       # feature ranking

    # ---- Data ----
    "image_size": 512,         # inputs are resized to 512x512
    "val_ratio": 0.10,         # fraction of the training set held out for validation
    "expand": 5,               # fixed 5x training-set expansion (flips and rotations)
    "use_augmentation": True,  # enable random online augmentation (h/v flips + random +/-90 deg rotation)

    # ---- Ranking hyperparameters (aligned with the paper) ----
    "num_neutral_labels": 4,   # N_hat: virtual neutral labels for label ranking (paper AID-ML uses 4)
    "num_neutral_samples": 16, # M_tilde: virtual neutral samples for sample ranking (set to batch size = 16)
    "theta": 0.1,              # half-width of the buffer interval: virtuals lie in [0.5-theta, 0.5+theta]
    "alpha": 100.0,            # sigmoid sharpness for label/sample ranking
    "alpha_feature": 10.0,     # separate sigmoid sharpness for feature ranking (different distance scale)

    # ---- Dynamic alpha (only for label/sample ranking; feature ranking is unaffected) ----
    "use_alpha_schedule": False,
    "alpha_schedule": [(10, 15), (70, 10), (400, 5)],  # (last epoch, alpha)

    # ---- Per-task loss weights ----
    "weight_label": 1.0,       # label-ranking weight
    "weight_sample": 1.0,      # sample-ranking weight
    "weight_feature": 1.0 / 8, # feature-ranking weight (paper lambda = 1/8)

    # ---- Loss aggregation mode ----
    # True : official aggregation (label ranking divided by per-image positive count and
    #        summed; sample/feature ranking summed after dropping queries with idcg == 0).
    # False: take the mean NDCG per task, weight, and negate.
    "use_official_loss": True,

    # ---- Virtual-label mode ----
    # False: paper setting (label ranking N_hat=4, sample ranking M_tilde=16, virtuals
    #        spread uniformly by Eq. (6)).
    # True : fixed three virtual items with values 0.6 / 0.5 / 0.4.
    "use_three_virtuals": False,
    "three_virtual_scores": [0.4, 0.5, 0.6],  # used only when use_three_virtuals=True

    # ---- Output sigmoid switch ----
    # True : paper setting, output z = sigmoid(fc(f)) in (0,1); softer ranking signal and
    #        slower convergence, requires the full 400 epochs.
    # False: raw logits; stronger ranking signal and faster convergence, but not the paper.
    "use_sigmoid": True,

    # ---- Feature normalization (L2-normalize the 2048-dim feature before fc) ----
    "use_feature_norm": False,  # True = normalized feature (distances fall in [0,2]), False = raw feature

    "pos_relevance": 2.0,      # relevance of positive labels/samples
    "neg_relevance": 0.0,      # relevance of negative labels/samples
    "virtual_relevance": 1.0,  # relevance of virtual neutral items
    "threshold": 0.5,          # test threshold: z > 0.5 predicts positive

    # ---- Rare-label relevance weighting (only for label/sample ranking) ----
    "use_rare_label_weighting": True,   # give rare labels a higher positive relevance
    "rare_labels": ["mobile-home", "airplane", "tanks", "chaparral"],  # rare label names
    "rare_relevance": 2.5,              # rare-label positive relevance (regular positive = 2.0)

    # ---- Training ----
    "batch_size": 16,
    "epochs": 400,
    "lr_schedule": [(200, 1e-5), (300, 1e-6), (400, 5e-7)],  # piecewise LR
    "weight_decay": 0,      # L2 regularization (helps against overfitting over 400 epochs)

    # ---- Optimizer switch (adam / sgd) ----
    "optimizer": "adam",       # "adam" or "sgd"
    "sgd_momentum": 0.7,       # SGD momentum (0.9 recommended)
    "sgd_lr_schedule": [(200, 1e-3), (300, 1e-4), (400, 5e-5)],  # SGD LR schedule (~100x Adam)

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


def get_alpha(epoch, schedule):
    """Return the alpha for the given (1-based) epoch from a piecewise table (label/sample ranking only)."""
    for end, a in schedule:
        if epoch <= end:
            return a
    return schedule[-1][1]


# ============================ Model ============================
class R4CNet(nn.Module):
    """Shared backbone that outputs both the 17-dim probabilities z and the 2048-dim feature f."""

    def __init__(self, num_classes=NUM_CLASSES, use_sigmoid=True, use_feature_norm=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = backbone.fc.in_features          # ResNet50 = 2048
        backbone.fc = nn.Identity()                          # drop the classifier head, keep the 2048-dim feature
        self.backbone = backbone
        self.fc = nn.Linear(self.feature_dim, num_classes)   # 2048 -> 17
        self.use_sigmoid = use_sigmoid
        self.use_feature_norm = use_feature_norm

    def forward(self, x):
        f = self.backbone(x)                # (B, 2048) global feature
        if self.use_feature_norm:
            f = F.normalize(f, dim=1)       # L2-normalize before fc (unit norm, distances in [0,2])
        z = self.fc(f)                      # (B, 17) logits
        if self.use_sigmoid:
            z = torch.sigmoid(z)            # paper: probabilities in (0,1)
        return z, f


# ============================ Dataset ============================
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


# ============================ Ranking losses ============================
def make_virtual_scores(K, theta, device, dtype):
    """Generate K virtual neutral scores by paper Eq. (6): 0.5 + (2i/(K-1) - 1) * theta, i=0..K-1."""
    if K == 1:
        return torch.tensor([0.5], device=device, dtype=dtype)
    i = torch.arange(K, device=device, dtype=dtype)
    return 0.5 + (2.0 * i / (K - 1) - 1.0) * theta


def virtual_scores_for(kind, device, dtype, cfg):
    """
    Return the virtual-score tensor for the current mode.
    kind: "label" or "sample".
    use_three_virtuals=True  -> fixed three values [0.6, 0.5, 0.4].
    Otherwise                 -> paper Eq. (6), count = num_neutral_labels / num_neutral_samples.
    """
    if cfg["use_three_virtuals"]:
        return torch.tensor(cfg["three_virtual_scores"], device=device, dtype=dtype)
    K = cfg["num_neutral_labels"] if kind == "label" else cfg["num_neutral_samples"]
    return make_virtual_scores(K, cfg["theta"], device, dtype)


def ndcg_per_query(scores, relevance, alpha=10.0):
    """
    Compute ApproxNDCG for each query (row); returns (n_dcg_q, ideal_dcg_q), both shape (Q,).
    ideal_dcg_q may be 0 (no positive-relevance item in that query), left for the caller
    to drop or normalize.
    """
    Q, K = scores.shape
    dev, dt = scores.device, scores.dtype
    gain = 2.0 ** relevance - 1.0                                   # (Q,K)

    diff = scores.unsqueeze(2) - scores.unsqueeze(1)                # diff[q,j,i] = s_j - s_i
    sigma = torch.sigmoid(alpha * diff)
    mask = 1.0 - torch.eye(K, device=dev, dtype=dt)
    rank = 1.0 + (sigma * mask).sum(dim=1)                          # (Q,K)

    discount = 1.0 / torch.log2(1.0 + rank)
    approx_dcg = (gain * discount).sum(dim=1)                       # (Q,)

    gain_sorted, _ = torch.sort(gain, dim=1, descending=True)
    ideal_rank = torch.arange(1, K + 1, device=dev, dtype=dt)
    ideal_dcg = (gain_sorted / torch.log2(1.0 + ideal_rank)).sum(dim=1)  # (Q,)

    n_dcg = approx_dcg / ideal_dcg.clamp(min=1e-8)
    return n_dcg, ideal_dcg


def ndcg(scores, relevance, alpha=10.0):
    """Return the mean ApproxNDCG (scalar, differentiable; for monitoring)."""
    n_dcg, _ = ndcg_per_query(scores, relevance, alpha)
    return n_dcg.mean()


# ---- Build each task's (scores, relevance) inputs ----
def per_label_relevance(dev, dt, cfg):
    """Return the per-label positive relevance tensor of shape (17,). Rare labels use rare_relevance, others pos_relevance."""
    rel = torch.full((len(LABELS),), cfg["pos_relevance"], dtype=dt, device=dev)
    if cfg["use_rare_label_weighting"]:
        for name in cfg["rare_labels"]:
            rel[LABELS.index(name)] = cfg["rare_relevance"]
    return rel


def build_label_inputs(z, labels, cfg):
    Q = z.size(0)
    dev, dt = z.device, z.dtype
    virt_scores = virtual_scores_for("label", dev, dt, cfg)
    K = virt_scores.numel()
    virt_scores = virt_scores.unsqueeze(0).expand(Q, -1)
    scores = torch.cat([z, virt_scores], dim=1)
    # Positive-label relevance is per-label (rare labels can be higher); negatives use neg_relevance (0).
    real_rel = labels * per_label_relevance(dev, dt, cfg)          # (Q, 17)
    virt_rel = torch.full((Q, K), cfg["virtual_relevance"], dtype=dt, device=dev)
    relevance = torch.cat([real_rel, virt_rel], dim=1)
    return scores, relevance


def build_sample_inputs(z, labels, cfg):
    zT = z.t()
    labelsT = labels.t()
    Q = zT.size(0)
    dev, dt = z.device, z.dtype
    virt_scores = virtual_scores_for("sample", dev, dt, cfg)
    K = virt_scores.numel()
    virt_scores = virt_scores.unsqueeze(0).expand(Q, -1)
    scores = torch.cat([zT, virt_scores], dim=1)
    real_rel = labelsT * per_label_relevance(dev, dt, cfg).unsqueeze(1)  # (17, M)
    virt_rel = torch.full((Q, K), cfg["virtual_relevance"], dtype=dt, device=dev)
    relevance = torch.cat([real_rel, virt_rel], dim=1)
    return scores, relevance


def build_feature_inputs(f, labels, cfg):
    M = f.size(0)
    dev, dt = f.device, f.dtype
    dist = torch.cdist(f, f)
    inter = labels @ labels.t()
    sums = labels.sum(dim=1, keepdim=True)
    union = sums + sums.t() - inter
    jaccard = inter / union.clamp(min=1e-8)
    eye = torch.eye(M, device=dev, dtype=dt)
    relevance = jaccard * (1.0 - eye)
    scores = -dist - 1e6 * eye
    return scores, relevance


# ---- Per-task mean NDCG (for monitoring, always returns mean) ----
def label_ranking_ndcg(z, labels, cfg):
    scores, relevance = build_label_inputs(z, labels, cfg)
    return ndcg(scores, relevance, cfg["alpha"])


def sample_ranking_ndcg(z, labels, cfg):
    scores, relevance = build_sample_inputs(z, labels, cfg)
    return ndcg(scores, relevance, cfg["alpha"])


def feature_ranking_ndcg(f, labels, cfg):
    scores, relevance = build_feature_inputs(f, labels, cfg)
    return ndcg(scores, relevance, cfg["alpha_feature"])


# ---- Official aggregation losses (used when use_official_loss=True) ----
def official_label_loss(z, labels, cfg):
    """Official label-ranking loss: sum_q (1 - nDCG_q) / (per-image positive count)."""
    scores, relevance = build_label_inputs(z, labels, cfg)
    n_dcg, _ = ndcg_per_query(scores, relevance, cfg["alpha"])
    num_pos = (labels > 0.5).sum(dim=1).float().clamp(min=1)
    return ((1.0 - n_dcg) / num_pos).sum()


def official_sample_loss(z, labels, cfg):
    """Official sample-ranking loss: sum_q (1 - nDCG_q), dropping labels whose ideal DCG is 0."""
    scores, relevance = build_sample_inputs(z, labels, cfg)
    n_dcg, idcg = ndcg_per_query(scores, relevance, cfg["alpha"])
    valid = idcg > 1e-8
    return (1.0 - n_dcg[valid]).sum()


def official_feature_loss(f, labels, cfg):
    """Official feature-ranking loss: sum_q (1 - nDCG_q), dropping queries whose ideal DCG is 0."""
    scores, relevance = build_feature_inputs(f, labels, cfg)
    n_dcg, idcg = ndcg_per_query(scores, relevance, cfg["alpha_feature"])
    valid = idcg > 1e-8
    return (1.0 - n_dcg[valid]).sum()


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


def task_ndcgs(z, f, labels, cfg):
    """Compute the NDCG of each enabled task; returns a dict keyed by label/sample/feature."""
    out = {}
    if cfg["use_label"]:
        out["label"] = label_ranking_ndcg(z, labels, cfg)
    if cfg["use_sample"]:
        out["sample"] = sample_ranking_ndcg(z, labels, cfg)
    if cfg["use_feature"]:
        out["feature"] = feature_ranking_ndcg(f, labels, cfg)
    return out


def combined_loss(z, f, labels, cfg):
    """Total loss. use_official_loss=True uses the official aggregation, otherwise -sum(w * mean(nDCG))."""
    if cfg["use_official_loss"]:
        total = 0.0
        if cfg["use_label"]:
            total += cfg["weight_label"] * official_label_loss(z, labels, cfg)
        if cfg["use_sample"]:
            total += cfg["weight_sample"] * official_sample_loss(z, labels, cfg)
        if cfg["use_feature"]:
            total += cfg["weight_feature"] * official_feature_loss(f, labels, cfg)
        return total
    else:
        ndcgs = task_ndcgs(z, f, labels, cfg)
        total = 0.0
        if "label" in ndcgs:
            total += cfg["weight_label"] * ndcgs["label"]
        if "sample" in ndcgs:
            total += cfg["weight_sample"] * ndcgs["sample"]
        if "feature" in ndcgs:
            total += cfg["weight_feature"] * ndcgs["feature"]
        return -total


def evaluate(model, loader, device, cfg):
    """Return (6-metric dict, per-task mean NDCG dict).

    When use_sigmoid=False, the metrics dict additionally holds the mean/max/min gap
    between adjacent-rank logits.
    """
    model.eval()
    n_samples = 0
    ndcg_sums = {}
    all_z, all_labels = [], []
    gap_label_list, gap_sample_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            z, f = model(images)
            ndcgs = task_ndcgs(z, f, labels, cfg)
            n = images.size(0)
            for k, v in ndcgs.items():
                ndcg_sums[k] = ndcg_sums.get(k, 0.0) + v.item() * n
            n_samples += n
            all_z.append(z.cpu())
            all_labels.append(labels.cpu())
            # Gap between adjacent-rank (rank diff 1) logits, monitored only when
            # use_sigmoid=False (to observe saturation / alpha behavior).
            if not cfg["use_sigmoid"]:
                if cfg["use_label"]:
                    virt = virtual_scores_for("label", device, z.dtype, cfg)
                    for row in z:
                        s = torch.cat([row, virt])
                        s_sorted, _ = torch.sort(s, descending=True)
                        gap_label_list.append(s_sorted[:-1] - s_sorted[1:])
                if cfg["use_sample"]:
                    virt = virtual_scores_for("sample", device, z.dtype, cfg)
                    for col in z.t():
                        s = torch.cat([col, virt])
                        s_sorted, _ = torch.sort(s, descending=True)
                        gap_sample_list.append(s_sorted[:-1] - s_sorted[1:])

    z_all = torch.cat(all_z)
    labels_all = torch.cat(all_labels)
    preds = (z_all > cfg["threshold"]).float()

    ex_p, ex_r, ex_f1, ex_f2 = example_based_metrics(preds, labels_all)
    label_p, label_r = label_based_macro_metrics(preds, labels_all)
    metrics = {
        "ex_P": ex_p, "ex_R": ex_r,
        "label_P": label_p, "label_R": label_r,
        "ex_F1": ex_f1, "ex_F2": ex_f2,
    }
    if not cfg["use_sigmoid"]:
        if gap_label_list:
            g = torch.cat(gap_label_list)
            metrics["gap_label"] = (g.mean().item(), g.max().item(), g.min().item())
        if gap_sample_list:
            g = torch.cat(gap_sample_list)
            metrics["gap_sample"] = (g.mean().item(), g.max().item(), g.min().item())
    ndcg_avg = {k: v / n_samples for k, v in ndcg_sums.items()}
    return metrics, ndcg_avg


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

    # Base transform (validation/test, no augmentation).
    base_transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # Training transform: random online augmentation (h/v flips + random +/-90 deg rotation).
    train_transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_base = [train_samples[i] for i in train_idx]
    val_base = [train_samples[i] for i in val_idx]
    # Augmentation is random online (not the fixed 5x expansion), so expand is always 1.
    used_transform = train_transform if cfg["use_augmentation"] else base_transform
    train_ds = MultiLabelDataset(train_base, transform=used_transform, expand=1)
    val_ds = MultiLabelDataset(val_base, transform=base_transform, expand=1)
    test_ds = MultiLabelDataset(test_samples, transform=base_transform, expand=1)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)
    print(f"Train {len(train_base)} (aug={'on' if cfg['use_augmentation'] else 'off'}) / val {len(val_ds)} / test {len(test_ds)}")

    # ---------- Model / optimizer ----------
    model = R4CNet(NUM_CLASSES, use_sigmoid=cfg["use_sigmoid"],
                   use_feature_norm=cfg["use_feature_norm"]).to(device)
    # Optimizer switch: adam / sgd.
    if cfg["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg["sgd_lr_schedule"][0][1],
                              momentum=cfg["sgd_momentum"], weight_decay=cfg["weight_decay"])
        lr_schedule = cfg["sgd_lr_schedule"]
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr_schedule"][0][1],
                               weight_decay=cfg["weight_decay"])
        lr_schedule = cfg["lr_schedule"]

    # ---------- Training ----------
    task_names = []
    if cfg["use_label"]:
        task_names.append("label")
    if cfg["use_sample"]:
        task_names.append("sample")
    if cfg["use_feature"]:
        task_names.append("feature")
    print(f"\nEnabled tasks: {task_names}, starting training...")

    # Track two best models: best F1 over all epochs + best F1 after epoch 300
    # (no early stopping; run all epochs).
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_val_f1_after300 = -1.0
    best_epoch_after300 = 0
    best_state_after300 = None

    for epoch in range(1, cfg["epochs"] + 1):
        lr_now = get_lr(epoch, lr_schedule)
        for g in optimizer.param_groups:
            g["lr"] = lr_now
        # Dynamically adjust alpha (label/sample ranking only).
        if cfg["use_alpha_schedule"]:
            cfg["alpha"] = get_alpha(epoch, cfg["alpha_schedule"])

        model.train()
        running = {k: 0.0 for k in task_names}
        n_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            z, f = model(images)
            ndcgs = task_ndcgs(z, f, labels, cfg)
            loss = combined_loss(z, f, labels, cfg)
            loss.backward()
            optimizer.step()
            n = images.size(0)
            for k in task_names:
                running[k] += ndcgs[k].item() * n
            n_train += n

        val_m, val_ndcg = evaluate(model, val_loader, device, cfg)

        # Global best (all epochs).
        if val_m["ex_F1"] > best_val_f1:
            best_val_f1 = val_m["ex_F1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        # Best after epoch 300 (only considers epoch > 300).
        if epoch > 300 and val_m["ex_F1"] > best_val_f1_after300:
            best_val_f1_after300 = val_m["ex_F1"]
            best_epoch_after300 = epoch
            best_state_after300 = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Print per-task NDCG + 6 metrics.
        train_str = " ".join(f"train_{k}={running[k]/n_train:.4f}" for k in task_names)
        val_str = " ".join(f"val_{k}={val_ndcg.get(k, 0.0):.4f}" for k in task_names)
        gap_str = ""
        if "gap_label" in val_m:
            m, mx, mn = val_m["gap_label"]
            gap_str += f" gapL(mean={m:.3f} max={mx:.3f} min={mn:.3f})"
        if "gap_sample" in val_m:
            m, mx, mn = val_m["gap_sample"]
            gap_str += f" gapS(mean={m:.3f} max={mx:.3f} min={mn:.3f})"
        print(f"[Epoch {epoch:3d}/{cfg['epochs']}] {train_str} | {val_str} | "
              f"ex_P={val_m['ex_P']:.4f} ex_R={val_m['ex_R']:.4f} "
              f"label_P={val_m['label_P']:.4f} label_R={val_m['label_R']:.4f} "
              f"F1={val_m['ex_F1']:.4f} F2={val_m['ex_F2']:.4f} | lr={lr_now:.1e} a={cfg['alpha']:.2g}{gap_str}")

    # ---------- Evaluate and save both best models ----------
    def report_and_save(state, tag, val_f1, epoch):
        if state is None:
            print(f"\n[{tag}] no model available (epoch never exceeded 300), skipping")
            return
        model.load_state_dict(state)
        print("\n" + "=" * 70)
        print(f"[{tag}] best validation F1: {val_f1:.4f} (epoch {epoch})")
        tm, tndcg = evaluate(model, test_loader, device, cfg)
        for k in task_names:
            print(f"Test {k} NDCG: {tndcg.get(k, 0.0):.4f}")
        print(f"Test example-based  precision P={tm['ex_P']:.4f} | recall R={tm['ex_R']:.4f} "
              f"| F1={tm['ex_F1']:.4f} | F2={tm['ex_F2']:.4f}")
        print(f"Test label-based  precision P={tm['label_P']:.4f} | recall R={tm['label_R']:.4f}")
        print("=" * 70)
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), tag + ".pth")
        torch.save(state, save_path)
        print(f"[{tag}] model saved to: {save_path}")

    report_and_save(best_state, "best_model_r4c", best_val_f1, best_epoch)
    report_and_save(best_state_after300, "best_model_r4c_after300", best_val_f1_after300, best_epoch_after300)


if __name__ == "__main__":
    main()
