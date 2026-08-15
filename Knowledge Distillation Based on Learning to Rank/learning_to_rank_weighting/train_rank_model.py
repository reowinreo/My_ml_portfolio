# -*- coding: utf-8 -*-
"""
Script 2: train a ranking-counting model with pseudo labels (single dataset)
Usage:
    python train_rank_model.py --dataset RSOC-Building
"""

import os
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============== Model definitions ==============
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.out_channels = 2048

    def forward(self, x):
        return self.features(x)


class RankCountingNet(nn.Module):
    def __init__(self, num_patches=16, feat_dim=2048):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=True)
        self.num_patches = num_patches
        self.feat_dim = feat_dim

        self.global_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 1),
        )

        self.local_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x):
        feat_map = self.backbone(x)           # (B,C,Hf,Wf)
        B, C, Hf, Wf = feat_map.shape

        global_feat = F.adaptive_avg_pool2d(feat_map, (1, 1)).view(B, C)
        global_scores = self.global_head(global_feat).view(B)

        local_feat = F.adaptive_avg_pool2d(feat_map, (4, 4))  # (B,C,4,4)
        local_feat = local_feat.view(B, C, -1).permute(0, 2, 1)  # (B,16,C)
        local_scores = self.local_head(local_feat).view(B, -1)   # (B,16)

        return global_scores, local_scores


# ============== Ranking loss (RankNet from the paper) ==============
def ranknet_loss(scores_i, scores_j, y_ij):
    diff = (scores_i - scores_j) * y_ij
    return torch.mean(torch.log1p(torch.exp(-diff)))


def build_global_pairs(global_counts_tensor):
    counts = global_counts_tensor.cpu().numpy()
    B = len(counts)
    i_idx, j_idx, y_list = [], [], []
    for i in range(B):
        for j in range(i + 1, B):
            if counts[i] == counts[j]:
                continue
            i_idx.append(i)
            j_idx.append(j)
            y_list.append(1.0 if counts[i] > counts[j] else -1.0)
    if len(i_idx) == 0:
        return None, None, None
    device = global_counts_tensor.device
    i_idx = torch.tensor(i_idx, dtype=torch.long, device=device)
    j_idx = torch.tensor(j_idx, dtype=torch.long, device=device)
    y_ij = torch.tensor(y_list, dtype=torch.float32, device=device)
    return i_idx, j_idx, y_ij


def build_local_pairs(patch_counts_tensor):
    counts = patch_counts_tensor.cpu().numpy()
    B, P = counts.shape
    b_list, p_list, q_list, y_list = [], [], [], []
    for b in range(B):
        for p in range(P):
            for q in range(p + 1, P):
                if counts[b, p] == counts[b, q]:
                    continue
                b_list.append(b)
                p_list.append(p)
                q_list.append(q)
                y_list.append(1.0 if counts[b, p] > counts[b, q] else -1.0)
    if len(b_list) == 0:
        return None, None, None, None
    device = patch_counts_tensor.device
    b_idx = torch.tensor(b_list, dtype=torch.long, device=device)
    p_idx = torch.tensor(p_list, dtype=torch.long, device=device)
    q_idx = torch.tensor(q_list, dtype=torch.long, device=device)
    y_pq = torch.tensor(y_list, dtype=torch.float32, device=device)
    return b_idx, p_idx, q_idx, y_pq


# ============== Dataset ==============
class PseudoCountingDataset(Dataset):
    def __init__(self, pseudo_path, split="train", transform=None):
        super().__init__()
        arr = np.load(pseudo_path, allow_pickle=True)
        entries = arr.tolist()
        self.samples = [e for e in entries if e["split"] == split]
        self.transform = transform
        print(f"[Dataset] {os.path.basename(pseudo_path)}, split={split}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        e = self.samples[idx]
        img_path = e["img_path"]
        g_count = e["global_count"]
        p_counts = np.array(e["patch_counts"], dtype=np.float32)

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, g_count, p_counts


def create_dataloaders(pseudo_path, batch_size=4, img_size=512):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # train uses split=train
    train_set = PseudoCountingDataset(pseudo_path, split="train", transform=transform_train)

    # val: prefer split=val, otherwise fall back to test
    arr_all = np.load(pseudo_path, allow_pickle=True).tolist()
    has_val = any(e["split"] == "val" for e in arr_all)
    val_split = "val" if has_val else "test"
    val_set = PseudoCountingDataset(pseudo_path, split=val_split, transform=transform_val)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ============== Training ==============
def train_model(pseudo_path,
                dataset_label,
                num_epochs=100,
                lambda_global=1.0,
                lambda_local=1.0,
                lambda_reg=0.0,
                lr=1e-4,
                img_size=512,
                batch_size=4):

    train_loader, val_loader = create_dataloaders(
        pseudo_path, batch_size=batch_size, img_size=img_size
    )

    model = RankCountingNet(num_patches=16, feat_dim=2048).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\n[Train] Epoch {epoch+1}/{num_epochs}")

        # ---------- Train ----------
        model.train()
        total_loss = 0.0
        for step, (imgs, g_counts, p_counts) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            g_counts = g_counts.to(DEVICE).float()
            p_counts = p_counts.to(DEVICE).float()

            optimizer.zero_grad()
            g_scores, l_scores = model(imgs)

            gi, gj, gy = build_global_pairs(g_counts)
            if gi is not None:
                g_loss = ranknet_loss(g_scores[gi], g_scores[gj], gy)
            else:
                g_loss = torch.tensor(0.0, device=DEVICE)

            bi, pi, qi, py = build_local_pairs(p_counts)
            if bi is not None:
                s_p = l_scores[bi, pi]
                s_q = l_scores[bi, qi]
                l_loss = ranknet_loss(s_p, s_q, py)
            else:
                l_loss = torch.tensor(0.0, device=DEVICE)

            if lambda_reg > 0:
                target = torch.log1p(g_counts)
                reg_loss = F.mse_loss(g_scores, target)
            else:
                reg_loss = torch.tensor(0.0, device=DEVICE)

            loss = lambda_global * g_loss + lambda_local * l_loss + lambda_reg * reg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                print(f"  step {step+1}/{len(train_loader)}, "
                      f"loss={loss.item():.4f}, "
                      f"g={g_loss.item():.4f}, "
                      f"l={l_loss.item():.4f}, "
                      f"reg={reg_loss.item():.4f}")

        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f"[Train] Average loss: {avg_train_loss:.4f}")

        # ---------- Val ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, g_counts, p_counts in val_loader:
                imgs = imgs.to(DEVICE)
                g_counts = g_counts.to(DEVICE).float()
                p_counts = p_counts.to(DEVICE).float()

                g_scores, l_scores = model(imgs)

                gi, gj, gy = build_global_pairs(g_counts)
                if gi is not None:
                    g_loss = ranknet_loss(g_scores[gi], g_scores[gj], gy)
                else:
                    g_loss = torch.tensor(0.0, device=DEVICE)

                bi, pi, qi, py = build_local_pairs(p_counts)
                if bi is not None:
                    s_p = l_scores[bi, pi]
                    s_q = l_scores[bi, qi]
                    l_loss = ranknet_loss(s_p, s_q, py)
                else:
                    l_loss = torch.tensor(0.0, device=DEVICE)

                if lambda_reg > 0:
                    target = torch.log1p(g_counts)
                    reg_loss = F.mse_loss(g_scores, target)
                else:
                    reg_loss = torch.tensor(0.0, device=DEVICE)

                loss = lambda_global * g_loss + lambda_local * l_loss + lambda_reg * reg_loss
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"[Val] Average loss: {avg_val_loss:.4f}")
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"rank_model_best_{dataset_label.replace('-', '_')}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"[Save] Best model saved to {save_path}")

    print("[Train] Training finished")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="e.g. RSOC-Building / VD-People / VD-Vehicle ...")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_label = args.dataset
    pseudo_path = os.path.join(
        "pseudo_labels", f"pseudo_{dataset_label.replace('-', '_')}.npy"
    )
    if not os.path.exists(pseudo_path):
        print(f"[Error] Pseudo-label file not found: {pseudo_path}")
        print("Please run build_pseudo_labels.py first")
        return

    train_model(
        pseudo_path=pseudo_path,
        dataset_label=dataset_label,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        lambda_global=1.0,
        lambda_local=1.0,
        lambda_reg=0.0,  # Use ranking loss only for now
    )


if __name__ == "__main__":
    main()
