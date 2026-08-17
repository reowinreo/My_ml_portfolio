# -*- coding: utf-8 -*-
"""
script3: evaluationrankingcounting model(based on SAM3 pseudo label)
- computeglobalscoreandpseudocount  Spearman correlation coefficient
- computeglobal ranking  pairwise accuracy(hasmanyfewsampletorankingsidetoconsistent)
"""

import os
import argparse

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from train_rank_model import RankCountingNet  # directreusemodel definition

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PseudoCountingDataset(Dataset):
    def __init__(self, pseudo_path, split="test", transform=None):
        super().__init__()
        arr = np.load(pseudo_path, allow_pickle=True)
        entries = arr.tolist()
        self.samples = [e for e in entries if e["split"] == split]
        if len(self.samples) == 0 and split != "train":
            # ifthis split no, thenfall backuse train doevaluation
            self.samples = [e for e in entries if e["split"] == "train"]
            print(f"[Dataset] split={split} nosample, changeuse train, total {len(self.samples)} ")
        else:
            print(f"[Dataset] split={split}, number of samples={len(self.samples)}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        e = self.samples[idx]
        img_path = e["img_path"]
        g_count = e["global_count"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, g_count


def create_eval_loader(pseudo_path, img_size=512, split="test"):
    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    dataset = PseudoCountingDataset(pseudo_path, split=split, transform=transform_eval)
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=4, pin_memory=True)
    return loader


def spearman_correlation(x, y):
    """
    simpleimplement Spearman rho: 
    - to x, y partdo notrankinggetrank, thencompute Pearson correlation coefficient
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape

    def rankdata(a):
        temp = np.argsort(a)
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(a))
        # nothandle ties(onegeneralaffectnotlarge), ifneedstrictcellcanaddaveragerank
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)

    rx_mean = rx.mean()
    ry_mean = ry.mean()
    num = ((rx - rx_mean) * (ry - ry_mean)).sum()
    den = np.sqrt(((rx - rx_mean) ** 2).sum() * ((ry - ry_mean) ** 2).sum())
    if den == 0:
        return 0.0
    return float(num / den)


def pairwise_ranking_accuracy(counts, scores, max_pairs=100000):
    """
    computeglobal ranking  pairwise accuracy: 
    - selecttakesohas (i,j) orrandomsubset
    - if sign(count_i - count_j) == sign(score_i - score_j), thencomputejustconfirm
    """
    counts = np.asarray(counts)
    scores = np.asarray(scores)
    N = len(counts)
    if N < 2:
        return 1.0

    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if counts[i] == counts[j]:
                continue
            pairs.append((i, j))
    if len(pairs) == 0:
        return 1.0

    if len(pairs) > max_pairs:
        idx = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    correct = 0
    total = 0
    for i, j in pairs:
        ci, cj = counts[i], counts[j]
        si, sj = scores[i], scores[j]
        sign_c = 1 if ci > cj else -1
        if si == sj:
            continue
        sign_s = 1 if si > sj else -1
        if sign_c == sign_s:
            correct += 1
        total += 1
    if total == 0:
        return 1.0
    return correct / total


def eval_model(pseudo_path, model_path, img_size=512, split="test"):
    loader = create_eval_loader(pseudo_path, img_size=img_size, split=split)

    model = RankCountingNet(num_patches=16, feat_dim=2048).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_counts = []
    all_scores = []

    with torch.no_grad():
        for imgs, g_counts in loader:
            imgs = imgs.to(DEVICE)
            g_counts = g_counts.float()
            g_scores, _ = model(imgs)
            all_counts.extend(g_counts.numpy().tolist())
            all_scores.extend(g_scores.cpu().numpy().tolist())

    all_counts = np.array(all_counts, dtype=float)
    all_scores = np.array(all_scores, dtype=float)

    rho = spearman_correlation(all_counts, all_scores)
    acc = pairwise_ranking_accuracy(all_counts, all_scores)

    print(f"[Eval] Spearman correlation coefficient: {rho:.4f}")
    print(f"[Eval] Pairwise rankingaccuracy: {acc:.4f}")
    print(f"[Eval] evaluationnumber of samples: {len(all_counts)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="for example RSOC-Building / VD-People / VD-Vehicle ...")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_label = args.dataset

    pseudo_path = os.path.join(
        "pseudo_labels", f"pseudo_{dataset_label.replace('-', '_')}.npy"
    )
    model_path = f"rank_model_best_{dataset_label.replace('-', '_')}.pth"

    if not os.path.exists(pseudo_path):
        print(f"[Error] findnottopseudolabel file: {pseudo_path}")
        return
    if not os.path.exists(model_path):
        print(f"[Error] findnottomodelfile: {model_path}")
        return

    eval_model(pseudo_path, model_path, img_size=args.img_size, split=args.split)


if __name__ == "__main__":
    main()
