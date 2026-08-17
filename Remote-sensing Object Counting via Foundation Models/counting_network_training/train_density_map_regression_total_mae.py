from __future__ import annotations
import os
import glob
import math
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# =========================
# Configuration:according toneedmodify
# =========================
CONFIG = {
    "dataset_root": "dataset",          # datasetroot directory(andtraining scriptconsistent)
    # optional: "RSOC-Ship" | "RSOC-S-Vehicle" | "RSOC-L-Vehicle" | "RSOC-Building"
    #      "VD-People" | "VD-Vehicle"
    "dataset": "RSOC-Building",

    "img_size": 512,
    "sigma": 2,
    "truncate": 3.0,

    "batch_size": 1,
    "num_workers": 4,
    "seed": 42,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",

    # herefillyoutraininggoodmodel  .pth path
    "model_path": "my_model_density/best_countmse518.695451_epoch22_RSOC-Building.pth",

    # andtraining scriptconsistent: iftrainingwhenwill GT density mapmultiplywiththismultiple(density_scale), 
    # thenpredictcountwhenneedexceptbackgo: pred_count = sum(pred_density) / density_scale
    "density_scale": 100.0,

    # CSRNet mutualclose(usuallyevaluationwhennotneedagainbelowload ImageNet pretrained; willdirectloadyoutraininggood  state_dict)
    "csrnet_imagenet_pretrained": False,
    "vgg16_weights_path": "",  # optional: manualspecifylocal vgg16 weight .pth
}

# =========================
# Utils
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _build_allow_set_for_rsoc_mixed(root: str, dataset_label: str, split: str) -> set[str]:
    """use DOTA_data inside split_xxx.txt filter ship / small-vehicle / large-vehicle. """
    cat_token = {
        "RSOC-Ship": "ship",
        "RSOC-S-Vehicle": "small-vehicle",
        "RSOC-L-Vehicle": "large-vehicle",
    }.get(dataset_label)
    if cat_token is None:
        return set()

    dota_dir = os.path.join(root, "ASPDNet_dataset", "DOTA_data")
    txt_path = os.path.join(dota_dir, f"{split}_{cat_token}.txt")
    if os.path.isfile(txt_path):
        allow = set()
        for line in _safe_read_lines(txt_path):
            allow.add(os.path.splitext(line)[0])
        return allow

    label_dir = os.path.join(
        root,
        "ASPDNet_dataset",
        split,
        "labelTxt-v1.0",
        f"{split}set_reclabelTxt",
    )
    allow = set()
    for lbl in glob.glob(os.path.join(label_dir, "*.txt")):
        base = os.path.splitext(os.path.basename(lbl))[0]
        try:
            hit = False
            for line in _safe_read_lines(lbl):
                parts = line.split()
                if len(parts) < 9:
                    continue
                cls = parts[8] if len(parts) >= 10 else parts[-1]
                if cls == cat_token:
                    hit = True
                    break
            if hit:
                allow.add(base)
        except Exception:
            continue
    return allow


def _list_images(root: str, dataset_label: str, split: str) -> List[str]:
    """according todatasetclasstypelistimagepath(andtraining scriptkeepconsistent)"""
    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(root, "ASPDNet_dataset", "RSOC_building", "building")
        if split == "train":
            sub = "train_data"
        elif split in ("val", "test"):
            # building commononlyhas train/test, here val/test allpointto test_data
            sub = "test_data"
        else:
            return []
        img_dir = os.path.join(rsoc_root, sub, "images")
        if not os.path.isdir(img_dir):
            return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if dataset_label in ("RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"):
        img_dir = os.path.join(root, "ASPDNet_dataset", split, "images")
        if not os.path.isdir(img_dir):
            return []
        allow = _build_allow_set_for_rsoc_mixed(root, dataset_label, split)
        if not allow:
            return []
        out = [
            p
            for p in glob.glob(os.path.join(img_dir, "*.jpg"))
            if os.path.splitext(os.path.basename(p))[0] in allow
        ]
        return sorted(out)

    if dataset_label in ("VD-People", "VD-Vehicle"):
        vd_root = os.path.join(
            root,
            "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle",
        )
        img_dir1 = os.path.join(vd_root, split, "images")
        img_dir2 = os.path.join(vd_root, split, "Images")
        img_dir = img_dir1 if os.path.isdir(img_dir1) else img_dir2
        if not os.path.isdir(img_dir):
            return []
        exts = ("*.jpg", "*.png", "*.jpeg")
        out: List[str] = []
        for e in exts:
            out.extend(glob.glob(os.path.join(img_dir, e)))
        return sorted(out)

    return []


def _read_dota_points(
    label_path: str,
    width: int,
    height: int,
    class_token: str,
    out_size: int = 512,
) -> np.ndarray:
    """DOTA labelTxt: per line 8 coordinate + class + difficult. takecenter pointandmap to 0..out_size"""
    pts = []
    for line in _safe_read_lines(label_path):
        parts = line.split()
        if len(parts) < 9:
            continue
        cls = parts[8] if len(parts) >= 10 else parts[-1]
        if cls != class_token:
            continue
        try:
            coords = list(map(float, parts[:8]))
        except Exception:
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        cx = sum(xs) / 4.0
        cy = sum(ys) / 4.0
        cx = cx / max(width, 1) * out_size
        cy = cy / max(height, 1) * out_size
        pts.append((cx, cy))

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _read_building_mat_points(
    mat_path: str,
    width: int,
    height: int,
    out_size: int = 512,
) -> np.ndarray:
    """RSOC-Building: .mat insidehas center, parsefor Nx2 againmap to 0..out_size"""
    import scipy.io as sio

    mat = sio.loadmat(mat_path)
    if "center" not in mat:
        return np.zeros((0, 2), dtype=np.float32)

    center = mat["center"]
    pts: Optional[np.ndarray] = None

    try:
        # commonsituation: object array, first itemis Nx2
        cand = center[0, 0]
        cand = np.asarray(cand, dtype=np.float32)
        if cand.ndim == 2 and cand.shape[1] == 2:
            pts = cand
    except Exception:
        pts = None

    if pts is None:
        # fallback: hassome mat maybedirectis Nx2
        try:
            cand = np.asarray(center, dtype=np.float32)
            if cand.ndim == 2 and cand.shape[1] == 2:
                pts = cand
        except Exception:
            pts = None

    if pts is None:
        return np.zeros((0, 2), dtype=np.float32)

    pts2 = np.zeros_like(pts, dtype=np.float32)
    pts2[:, 0] = pts[:, 0] / max(width, 1) * out_size
    pts2[:, 1] = pts[:, 1] / max(height, 1) * out_size
    return pts2


def _read_visdrone_density_map(npy_path: str) -> np.ndarray:
    dm = np.load(npy_path)
    dm = np.asarray(dm, dtype=np.float32)
    if dm.ndim == 3:
        dm = dm[:, :, 0]
    return dm


def _read_visdrone_txt_points(
    txt_path: str,
    width: int,
    height: int,
    out_size: int = 512,
) -> np.ndarray:
    """standard VisDrone annotation: x,y,w,h,score,category,truncation,occlusion"""
    pts = []
    for line in _safe_read_lines(txt_path):
        parts = line.replace(",", " ").split()
        if len(parts) < 6:
            continue
        try:
            x, y, bw, bh = map(float, parts[:4])
            cat = int(float(parts[5]))
        except Exception:
            continue
        if cat <= 0:
            continue
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        cx = cx / max(width, 1) * out_size
        cy = cy / max(height, 1) * out_size
        pts.append((cx, cy))

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _gaussian_kernel2d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """generate 2D highkernel, andnormalizationfor sum=1"""
    radius = int(math.ceil(truncate * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    ys = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma))
    s = float(kernel.sum())
    if s > 0:
        kernel /= s
    return kernel.astype(np.float32)


def points_to_density_map(
    points_xy: np.ndarray,
    out_size: int = 512,
    sigma: float = 4.0,
    truncate: float = 3.0,
) -> np.ndarray:
    """point -> density map, density mapsum = pointnumber"""
    H = W = int(out_size)
    dm = np.zeros((H, W), dtype=np.float32)
    if points_xy.size == 0:
        return dm

    kernel = _gaussian_kernel2d(float(sigma), float(truncate))
    radius = kernel.shape[0] // 2

    for (x, y) in points_xy:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue

        x0 = max(0, xi - radius)
        x1 = min(W, xi + radius + 1)
        y0 = max(0, yi - radius)
        y1 = min(H, yi + radius + 1)

        kx0 = x0 - (xi - radius)
        kx1 = kx0 + (x1 - x0)
        ky0 = y0 - (yi - radius)
        ky1 = ky0 + (y1 - y0)

        dm[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1]

    return dm


# =========================
# Dataset: readimageandreal label, generate GT density map + realcount
# =========================
class DensityMapDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset_label: str,
        split: str,
        img_size: int = 512,
        sigma: float = 4.0,
        truncate: float = 3.0,
    ):
        self.root = root
        self.dataset_label = dataset_label
        self.split = split
        self.img_size = int(img_size)
        self.sigma = float(sigma)
        self.truncate = float(truncate)

        self.img_paths = _list_images(root, dataset_label, split)

        # manual normalize(andtraining scriptconsistent)
        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def _transform(self, img_pil: Image.Image) -> torch.Tensor:
        img = img_pil.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        arr = (arr - self._mean) / self._std
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr)

    def __len__(self) -> int:
        return len(self.img_paths)

    def _get_points_or_density(
        self,
        img_path: str,
        img_pil: Image.Image,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """return (points, density_map_if_already_given)"""
        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]
        w0, h0 = img_pil.size

        # RSOC Ship / Vehicle (DOTA labelTxt)
        if self.dataset_label in ("RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"):
            class_token = {
                "RSOC-Ship": "ship",
                "RSOC-S-Vehicle": "small-vehicle",
                "RSOC-L-Vehicle": "large-vehicle",
            }[self.dataset_label]
            label_dir = os.path.join(
                self.root,
                "ASPDNet_dataset",
                self.split,
                "labelTxt-v1.0",
                f"{self.split}set_reclabelTxt",
            )
            label_path = os.path.join(label_dir, base + ".txt")
            if not os.path.isfile(label_path):
                raise FileNotFoundError(f"Label not found: {label_path}")
            pts = _read_dota_points(label_path, w0, h0, class_token, out_size=self.img_size)
            return pts, None

        # RSOC Building (.mat)
        if self.dataset_label == "RSOC-Building":
            rsoc_root = os.path.join(self.root, "ASPDNet_dataset", "RSOC_building", "building")
            sub = "train_data" if self.split == "train" else "test_data"
            gt_dir = os.path.join(rsoc_root, sub, "ground_truth")

            # keymodify: preferuse"imagesamenamebutno GT_ prefix" label file
            # For example:IMG_000153.jpg -> ground_truth/IMG_000153.mat
            mat_path1 = os.path.join(gt_dir, base + ".mat")
            # meanwhilecompatibleoriginal RSOC-Building: GT_IMG_000153.mat
            mat_path2 = os.path.join(gt_dir, f"GT_{base}.mat")
            mat_path3 = os.path.join(gt_dir, f"GT_{img_name}".replace(".jpg", ".mat"))

            if os.path.isfile(mat_path1):
                mat_path = mat_path1
            elif os.path.isfile(mat_path2):
                mat_path = mat_path2
            elif os.path.isfile(mat_path3):
                mat_path = mat_path3
            else:
                raise FileNotFoundError(
                    f"Building GT not found: {mat_path1} / {mat_path2} / {mat_path3}"
                )

            pts = _read_building_mat_points(mat_path, w0, h0, out_size=self.img_size)
            return pts, None

        # VisDrone
        if self.dataset_label in ("VD-People", "VD-Vehicle"):
            vd_root = os.path.join(
                self.root,
                "VisDrone-People" if self.dataset_label == "VD-People" else "VisDrone-Vehicle",
            )
            gt_dir1 = os.path.join(vd_root, self.split, "Ground Truth")
            gt_dir2 = os.path.join(vd_root, self.split, "ground_truth")
            gt_dir = gt_dir1 if os.path.isdir(gt_dir1) else gt_dir2
            if not os.path.isdir(gt_dir):
                raise FileNotFoundError(f"VisDrone GT dir not found: {gt_dir1} / {gt_dir2}")

            npy_path = os.path.join(gt_dir, base + ".npy")
            if os.path.isfile(npy_path):
                dm = _read_visdrone_density_map(npy_path)
                return np.zeros((0, 2), dtype=np.float32), dm

            txt_path = os.path.join(gt_dir, base + ".txt")
            if not os.path.isfile(txt_path):
                raise FileNotFoundError(f"VisDrone GT not found: {npy_path} or {txt_path}")
            pts = _read_visdrone_txt_points(txt_path, w0, h0, out_size=self.img_size)
            return pts, None

        raise ValueError(f"Unknown dataset label: {self.dataset_label}")

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)

        img_pil = Image.open(img_path).convert("RGB")
        img = self._transform(img_pil)

        pts, dm_given = self._get_points_or_density(img_path, img_pil)

        if dm_given is not None:
            # iforiginalannotationdirecttodensity map(VisDrone .npy), resize to img_size, andkeep sum notchange
            dm = torch.from_numpy(dm_given).unsqueeze(0).unsqueeze(0)  # 1x1xH xW
            dm = dm.to(torch.float32)
            old_sum = float(dm.sum().item())
            dm = F.interpolate(
                dm,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            new_sum = float(dm.sum().item())
            if new_sum > 0 and old_sum > 0:
                dm = dm * (old_sum / new_sum)
            gt_density = dm.squeeze(0)  # 1xH xW
        else:
            dm_np = points_to_density_map(
                pts,
                out_size=self.img_size,
                sigma=self.sigma,
                truncate=self.truncate,
            )
            gt_density = torch.from_numpy(dm_np).unsqueeze(0)  # 1xH xW

        gt_count = gt_density.sum().to(torch.float32)
        return img, gt_density.to(torch.float32), gt_count, img_name


# =========================
# model architecture: andtraining scriptcorrespond(CSRNet)
# =========================
class CSRNet(nn.Module):
    """CSRNet: VGG16 frontend (up to conv4_3) + dilated convolution backend.

    in order tonotchangeyouoriginalcome dataAPI, herewilltake 1/8 resolutionoutputupsampleback 512x512, 
    andaccording topixelarearatiodo oncescalewithas much as possiblekeep sum(count)consistent. 
    """

    def __init__(self, imagenet_pretrained: bool = True, vgg16_weights_path: str = ""):
        super().__init__()

        # compatibledifferent torchvision version: weights=... or pretrained=True
        try:
            from torchvision import models  # type: ignore
            if imagenet_pretrained:
                # torchvision>=0.13
                try:
                    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                except Exception:
                    vgg = models.vgg16(pretrained=True)  # older API
            else:
                try:
                    vgg = models.vgg16(weights=None)
                except Exception:
                    vgg = models.vgg16(pretrained=False)
        except Exception as e:
            raise RuntimeError(
                "need torchvision only thencanuse CSRNet   VGG16 beforeend. pleasefirstinstall torchvision: \n"
                "  pip install torchvision\n"
                f"detail: {e}"
            )

        # ifyoumanualprovide localweight, thenforcefromlocalload(notwillinternetbelowload)
        if vgg16_weights_path:
            sd = torch.load(vgg16_weights_path, map_location="cpu")
            vgg.load_state_dict(sd)

        # VGG16 conv1_1 ~ conv4_3(contain 3 times maxpool) -> output 1/8
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        # CSRNet backend(dilation=2)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        # ✅Note:backend output channelis 64, soheremustis Conv2d(64, 1, 1)
        # use Softplus guaranteenon-negativeandnoteasy"deadgradient"
        self.out = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),
        )

        self._init_backend_weights()

    def _init_backend_weights(self) -> None:
        # onlyinitialize backend/out(VGG beforeendifuse pretrained thennotmove)
        for m in list(self.backend.modules()) + list(self.out.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        y = self.frontend(x)   # Bx512xH/8xW/8
        y = self.backend(y)
        y = self.out(y)        # Bx1xH/8xW/8 (non-negative)

        # upsamplebackinputresolution, andaccording toarearatioscalewithas much as possiblekeep sum(count)consistent
        if y.shape[-2:] != (H, W):
            old_h, old_w = y.shape[-2], y.shape[-1]
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            y = y * (float(old_h * old_w) / float(H * W))

        return y


# =========================
# evaluation: predictdensity map -> count -> MAE / RMSE
# =========================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n_samples = 0

    per_image_results = []  # saveeachimage (name, pred_count, gt_count)

    for img, gt_density, gt_count, img_name in tqdm(loader, desc="eval", ncols=110):
        img = img.to(device)
        gt_count = gt_count.to(device)

        # predictdensity map
        pred_density = model(img)
        # todensity mapsumgetcount(B dimension)
        pred_count = pred_density.sum(dim=(1, 2, 3)) / float(CONFIG.get("density_scale", 1.0))

        diff = pred_count - gt_count
        mae_sum += float(torch.abs(diff).sum().item())
        mse_sum += float((diff ** 2).sum().item())
        n_samples += img.size(0)

        per_image_results.append(
            (img_name[0], float(pred_count.item()), float(gt_count.item()))
        )

    if n_samples == 0:
        raise RuntimeError("No samples in loader for evaluation.")

    mae = mae_sum / n_samples
    rmse = math.sqrt(mse_sum / n_samples)

    return mae, rmse, per_image_results


def main():
    # fixedrandom seed
    set_seed(int(CONFIG["seed"]))

    root = CONFIG["dataset_root"]
    dataset_label = CONFIG["dataset"]
    img_size = int(CONFIG["img_size"])
    sigma = float(CONFIG["sigma"])
    truncate = float(CONFIG["truncate"])
    device = CONFIG["device"]
    model_path = CONFIG["model_path"]

    # construct test Dataset
    test_ds = DensityMapDataset(
        root,
        dataset_label,
        "test",
        img_size=img_size,
        sigma=sigma,
        truncate=truncate,
    )
    if len(test_ds) == 0:
        raise RuntimeError(
            f"No test images found for dataset={dataset_label} under root={root}"
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=False,
        num_workers=int(CONFIG["num_workers"]),
        pin_memory=True,
    )

    # buildmodelandloadweight
    model = CSRNet(
        imagenet_pretrained=bool(CONFIG.get("csrnet_imagenet_pretrained", False)),
        vgg16_weights_path=str(CONFIG.get("vgg16_weights_path", "")),
    ).to(device)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    print(f"Loaded model from: {model_path}")

    # evaluation: get MAE / RMSE
    mae, rmse, per_image_results = evaluate(model, test_loader, device)

    print("======= test setcountmetric =======")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")

    # sidethenpersonworkcheck: printbeforeifdoimageimage predict/realcount
    print("======= eachimagecount(before 20 item)=======")
    for i, (name, pred_c, gt_c) in enumerate(per_image_results[:20]):
        print(f"[{i:03d}] {name} | pred={pred_c:.3f} | gt={gt_c:.3f}")


if __name__ == "__main__":
    main()