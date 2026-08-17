# finetune_cbam_calib_with_gt.py
# based onyou"CBAMversionmainprogram" model architecture: 
# 1) load my_model belowtraininggood weight
# 2) freeze CBAMBackbone + Regression
# 3) onlytrainingonesmallcalibrationlayer: count = k * score + b
# 4) usereal label(GT)training/evaluation

import os
import glob
import math
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# =========================================================
# only modify here: changedataset / trainingparameter
# =========================================================
ROOT_DATA_DIR = "dataset"
MY_MODEL_DIR = "my_model"

# optionaldataset: 
# "RSOC-Building", "RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle", "VD-People", "VD-Vehicle"
DATASET_NAME = "RSOC-Building"

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"  # building not val willautomatic fallback to train

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 4

EPOCHS = 50
LR = 1e-2
WEIGHT_DECAY = 0.0

IMG_SIZE = 512
CLIPNUM = 16


# =========================================================
# 1) select my_model belowlatest weight
# =========================================================
def pick_latest_checkpoint(model_dir: str) -> str:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"findnottomodeldirectory: {model_dir}")
    cks = glob.glob(os.path.join(model_dir, "*.pth"))
    if not cks:
        raise FileNotFoundError(f"{model_dir} belowno .pth file")
    cks.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cks[0]


# =========================================================
# 2) -- you  CBAMBackbone + Regression(minimummustneedpart)
#     Description:here"referenceyouemit mainprogram", notdependency ALTGVT.py
# =========================================================
# ifyoucurrentscriptsamedirectorybelowhas CBAM.py(define  CBAM class), herewillimportuse
from CBAM import CBAM


class Regression(nn.Module):
    def __init__(self, clipnum=16, chan=256):
        super(Regression, self).__init__()
        self.v1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.v2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.v3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )
        self.output = nn.Sequential(  # globalregression(youoriginalprogramthenis Sigmoid 0~1)
            nn.Linear(4096 + clipnum, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.c16 = nn.Sequential(  # local16gridregression
            nn.Linear(chan, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.clipnum = clipnum
        self.init_param()

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def block(self, y):
        n = int(math.sqrt(self.clipnum))
        b, h, w = y.size(0), y.size(-2) // n, y.size(-1) // n
        y_c16 = torch.zeros(y.size(0), self.clipnum).to(y.device)
        num = 0
        for i in range(0, y.size(-2) - h + 1, h):
            for j in range(0, y.size(-1) - w + 1, w):
                sub_y = y[:, :, i:i + h, j:j + w]
                sub_y_output = self.c16(sub_y.contiguous().view(sub_y.size(0), -1))
                y_c16[:, num:num + 1] = sub_y_output
                num += 1
        return y_c16

    def forward(self, x1, x2, x3):
        x1 = self.v1(x1)
        x2 = self.v2(x2)
        x3 = self.v3(x3)
        x = x1 + x2 + x3
        y = self.res(x)
        y_c16 = self.block(y)
        y = y.view(y.size(0), -1)
        y_concat = torch.cat([y, y_c16], dim=1)
        y_concat = self.output(y_concat)  # [B,1] 0~1
        return y_c16, y_concat


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CBAMBackbone(nn.Module):
    def __init__(self, clipnum=16, in_chans=3):
        super().__init__()
        self.stage0 = nn.Sequential(
            _ConvBNReLU(in_chans, 64, stride=4, k=7, p=3),
            _ConvBNReLU(64, 64, stride=1, k=3, p=1),
            CBAM(64),
        )
        self.stage1 = nn.Sequential(
            _ConvBNReLU(64, 128, stride=2, k=3, p=1),
            _ConvBNReLU(128, 128, stride=1, k=3, p=1),
            CBAM(128),
        )
        self.stage2 = nn.Sequential(
            _ConvBNReLU(128, 256, stride=2, k=3, p=1),
            _ConvBNReLU(256, 256, stride=1, k=3, p=1),
            CBAM(256),
        )
        self.stage3 = nn.Sequential(
            _ConvBNReLU(256, 512, stride=2, k=3, p=1),
            _ConvBNReLU(512, 512, stride=1, k=3, p=1),
            CBAM(512),
        )
        self.regression = Regression(clipnum=clipnum)

    def forward_features(self, x):
        outs = []
        x = self.stage0(x); outs.append(x)
        x = self.stage1(x); outs.append(x)
        x = self.stage2(x); outs.append(x)
        x = self.stage3(x); outs.append(x)
        return outs

    def forward(self, x):
        feats = self.forward_features(x)
        y_c16, y_global = self.regression(feats[1], feats[2], feats[3])
        return y_c16, y_global


def cbam_small(pretrained=False, **kwargs):
    return CBAMBackbone(**kwargs)


# =========================================================
# 3) real labelread(andononeitemyouto structureconsistent)
# =========================================================
def _build_allow_set_for_rsoc_mixed(dataset_label: str, split: str) -> set:
    cat_token = {
        "RSOC-Ship": "ship",
        "RSOC-S-Vehicle": "small-vehicle",
        "RSOC-L-Vehicle": "large-vehicle",
    }.get(dataset_label, None)
    if cat_token is None:
        return set()

    dota_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "DOTA_data")
    txt_path = os.path.join(dota_dir, f"{split}_{cat_token}.txt")
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"findnottoclassfilterfile: {txt_path}")

    allow = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                allow.add(name)  # notcontainextension
    return allow


def _list_images_for_dataset_split(dataset_label: str, split: str) -> List[str]:
    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
        if split == "train":
            img_dir = os.path.join(rsoc_root, "train_data", "images")
        elif split == "test":
            img_dir = os.path.join(rsoc_root, "test_data", "images")
        elif split == "val":
            return []
        else:
            return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if dataset_label in ["RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"]:
        img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split, "images")
        if not os.path.isdir(img_dir):
            return []
        allow = _build_allow_set_for_rsoc_mixed(dataset_label, split)
        out = [p for p in glob.glob(os.path.join(img_dir, "*.jpg"))
               if os.path.splitext(os.path.basename(p))[0] in allow]
        return sorted(out)

    if dataset_label == "VD-People":
        img_dir = os.path.join(ROOT_DATA_DIR, "VisDrone-People", split, "images")
        if not os.path.isdir(img_dir):
            return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if dataset_label == "VD-Vehicle":
        img_dir1 = os.path.join(ROOT_DATA_DIR, "VisDrone-Vehicle", split, "images")
        img_dir2 = os.path.join(ROOT_DATA_DIR, "VisDrone-Vehicle", split, "Images")
        img_dir = img_dir1 if os.path.isdir(img_dir1) else img_dir2
        if not os.path.isdir(img_dir):
            return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    return []


def load_building_count_from_mat(mat_path: str) -> float:
    import scipy.io
    m = scipy.io.loadmat(mat_path)
    if "center" not in m:
        raise KeyError(f"{mat_path} find innotto 'center'")
    center = m["center"]
    # center[1,0] -> [[count]]
    try:
        return float(center[1, 0][0, 0])
    except Exception:
        pts = center[0, 0]
        return float(pts.shape[0])


def load_dota_count_from_label(label_path: str, target_cls: str) -> float:
    if not os.path.isfile(label_path):
        return 0.0
    cnt = 0
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("imagesource:") or line.startswith("gsd:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            cls = parts[-2]  # reversenumbersecondusuallyisclass
            if cls == target_cls:
                cnt += 1
    return float(cnt)


def load_visdrone_count(gt_dir: str, img_basename: str) -> float:
    base = os.path.splitext(img_basename)[0]
    npy_path = os.path.join(gt_dir, base + ".npy")
    txt_path = os.path.join(gt_dir, base + ".txt")

    if os.path.isfile(npy_path):
        arr = np.load(npy_path)
        return float(arr.sum())

    if os.path.isfile(txt_path):
        cnt = 0
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    cnt += 1
        return float(cnt)

    return 0.0


class GroundTruthCountDataset(Dataset):
    """
    Returns:(img_tensor, true_count_tensor[B,1], img_name)
    """
    def __init__(self, dataset_label: str, split: str, transform=None):
        self.dataset_label = dataset_label
        self.split = split
        self.img_paths = _list_images_for_dataset_split(dataset_label, split)

        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.dota_cls = {
            "RSOC-Ship": "ship",
            "RSOC-S-Vehicle": "small-vehicle",
            "RSOC-L-Vehicle": "large-vehicle",
        }.get(dataset_label, None)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = os.path.basename(img_path)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # ===== realcountread =====
        if self.dataset_label == "RSOC-Building":
            rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
            if self.split == "train":
                gt_dir = os.path.join(rsoc_root, "train_data", "ground_truth")
            else:
                gt_dir = os.path.join(rsoc_root, "test_data", "ground_truth")
            base = os.path.splitext(img_name)[0]
            mat_path = os.path.join(gt_dir, "GT_" + base + ".mat")
            true_count = load_building_count_from_mat(mat_path)

        elif self.dataset_label in ["RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"]:
            if self.split not in ["train", "val"]:
                raise RuntimeError("RSOC mixclassno test label, notcanuse test dotraining. pleaseuse train/val. ")
            label_dir = os.path.join(
                ROOT_DATA_DIR, "ASPDNet_dataset", self.split,
                "labelTxt-v1.0", f"{self.split}set_reclabelTxt"
            )
            base = os.path.splitext(img_name)[0]
            label_path = os.path.join(label_dir, base + ".txt")
            true_count = load_dota_count_from_label(label_path, self.dota_cls)

        elif self.dataset_label == "VD-People":
            gt_dir = os.path.join(ROOT_DATA_DIR, "VisDrone-People", self.split, "Ground Truth")
            true_count = load_visdrone_count(gt_dir, img_name)

        elif self.dataset_label == "VD-Vehicle":
            gt_dir = os.path.join(ROOT_DATA_DIR, "VisDrone-Vehicle", self.split, "Ground Truth")
            true_count = load_visdrone_count(gt_dir, img_name)

        else:
            true_count = 0.0

        return img, torch.tensor([true_count], dtype=torch.float32), img_name


# =========================================================
# 4) onlytrainingfinally "0.x -> count"Arguments:k,b
# =========================================================
class Calibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, score):
        return self.k * score + self.b


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


@torch.no_grad()
def evaluate(model, calib, loader) -> Tuple[float, float]:
    model.eval()
    calib.eval()
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    for img, gt_count, _ in loader:
        img = img.to(DEVICE)
        gt_count = gt_count.to(DEVICE)
        _, score = model(img)         # [B,1] 0~1
        pred = calib(score)           # [B,1] -> count
        mae_sum += (pred - gt_count).abs().sum().item()
        mse_sum += ((pred - gt_count) ** 2).sum().item()
        n += img.size(0)
    mae = mae_sum / max(n, 1)
    rmse = math.sqrt(mse_sum / max(n, 1))
    return mae, rmse


# =========================================================
# 5) mainfunction: loadweight -> freeze -> useGTtrainingcalibrationlayer
# =========================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1) build CBAM modelandload my_model weight
    model = cbam_small(pretrained=False, clipnum=CLIPNUM)
    model.to(DEVICE)

    ckpt = pick_latest_checkpoint(MY_MODEL_DIR)
    print(f"[INFO] use my_model latestweight: {ckpt}")
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state, strict=True)

    # 2) freezemodel
    freeze_all(model)
    model.eval()

    # 3) calibrationlayer(onlyonecantrainingmodule)
    calib = Calibrator().to(DEVICE)
    optimizer = optim.Adam(calib.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.L1Loss()

    # 4) dataset: real label
    train_set = GroundTruthCountDataset(DATASET_NAME, TRAIN_SPLIT)
    val_set = GroundTruthCountDataset(DATASET_NAME, VAL_SPLIT)

    if len(val_set) == 0:
        print("[WARN] current dataset has no val split, automatically use train as val evaluation. ")
        val_set = train_set

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=NUM_WORKERS)

    print(f"[INFO] Dataset={DATASET_NAME} train={len(train_set)} val={len(val_set)} device={DEVICE}")

    os.makedirs("calib_ckpt", exist_ok=True)
    best_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        calib.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=120)

        running = 0.0
        steps = 0

        for img, gt_count, _ in pbar:
            img = img.to(DEVICE)
            gt_count = gt_count.to(DEVICE)

            with torch.no_grad():
                _, score = model(img)   # [B,1] 0~1

            pred = calib(score)         # [B,1] -> count
            loss = loss_fn(pred, gt_count)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            steps += 1
            pbar.set_postfix(
                loss=running / max(steps, 1),
                k=float(calib.k.detach().cpu().item()),
                b=float(calib.b.detach().cpu().item())
            )

        mae, rmse = evaluate(model, calib, val_loader)
        print(f"[VAL] epoch={epoch} MAE={mae:.3f} RMSE={rmse:.3f}   (k={calib.k.item():.4f}, b={calib.b.item():.4f})")

        if mae < best_mae:
            best_mae = mae
            save_path = os.path.join("calib_ckpt", f"{DATASET_NAME}_best_calib.pth")
            torch.save({
                "dataset": DATASET_NAME,
                "checkpoint_used": ckpt,
                "k": calib.k.detach().cpu(),
                "b": calib.b.detach().cpu(),
            }, save_path)
            print(f"[SAVE] best calib saved to: {save_path}")

    print("[DONE] GT calibration finished.")


if __name__ == "__main__":
    main()
