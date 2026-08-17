"""density_map_regression_train.py

inyounowhas"density mapregression"scriptbasison, onlyaccording to your  3 pointneedrequestdochange: 
1) takecurrent simple CNN replacefor CSRNet(density mapregression/crowdcountoftenusestructure, VGG16 beforeend + emptyholeconvolutionafterend). 
2) training setaddfitwhendata augmentation(geometryaugmentationwillsynchronizeroletodensity map; coloraugmentationonlyasused forimage). 
3) will GT density mapmultiplywithonemultiple(density_scale, for example 100/1000), modelregression"putlargeversion"density map; 
   computecount/evaluationwhenwilltakepredictexceptbackgo(sum(pred)/density_scale). 

restdataread/trainingloop/saveweight/input 512x512 etc.keepnotchange. 
"""

from __future__ import annotations

import os
import glob
import math
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from tqdm import tqdm


# =========================
# only modify herethencanchangedataset
# =========================
CONFIG = {
    "dataset_root": "dataset",

    # optional: 
    #   "RSOC-Ship" | "RSOC-S-Vehicle" | "RSOC-L-Vehicle" | "RSOC-Building"
    #   "VD-People" | "VD-Vehicle"
    "dataset": "RSOC-Building",

    # ========= youneedrequestnewly added: local pseudo label(unify txt) =========
    # labeldirectory: {dataset_root}/local pseudo
    # label filenameformat: rsoc-<datasetname>_<grid>_<split>.txt
    # For example:rsoc-building_grid16_train.txt
    "use_local_pseudo_labels": True,
    "local_pseudo_label_subdir": "local pseudo",
    "local_pseudo_label_grid": "grid16",
    # ==========================================

    # inputsetting
    "img_size": 512,

    # density maphighkernelparameter
    "sigma": 20,            # fixed sigma; youalsocanaccording todataadjust
    "truncate": 3.0,       # radius = truncate*sigma

    # trainingsetting
    "batch_size": 8,
    "epochs": 200,

    "lr": 1e-4,            # initiallearning rate
    "min_lr": 1e-6,        # cosine decaysubtractto minimumlearning rate(exceedsmalldecaysubtractexceed"thoroughbottom")
    # ==========================================================

    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 42,

    # ========= youoriginalscriptinsidealreadyhas: additionaladd L1 + putlarge loss =========
    "l1_weight": 0,      # totalloss = MSE + l1_weight * L1(canaccording toeffecttune 0.1~2.0)
    "loss_scale": 100.0,   # totallossagainmultiplyoftennumber: loss = (MSE + w*L1) * loss_scale
    # ======================================================

    # ========= youneedrequestnewly added: GT density mapscalemultiple(regression"putlargeversion"density map) =========
    "density_scale": 100.0,   # for example 100 or 1000; trainingwhen gt*=density_scale, count/evaluationwhen pred/=density_scale
    # ===================================================================

    # ========= youneedrequestnewly added: CSRNet setting =========
    "csrnet_imagenet_pretrained": True,   # isor notuse ImageNet   VGG16 pretraineddobeforeendinitialize(recommend True)
    "vgg16_weights_path": "",             # optional: manualspecifylocal vgg16 weight .pth path(notfillthenautomaticbelowloadto torch cache)
    # ==========================================

    # ========= youneedrequestnewly added: data augmentation(onlytraining setenable) =========
    "use_augmentation": True,
    "hflip_prob": 0.5,
    "vflip_prob": 0.0,      # remote sensingimageifup and downnottocall, canset 0; ifnosocallcanset 0.5
    "rot90_prob": 0.5,      # random 0/90/180/270 rotate(onlyincommandinprobabilitywhen)
    "jitter_prob": 0.8,     # colorperturbprobability
    "jitter_brightness": 0.2,
    "jitter_contrast": 0.2,
    "jitter_saturation": 0.2,
    # ==========================================

    # deviceandsave
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "save_dir": "my_model_density",
    "resume": "",  # continuetraining: fill .pth(willoverwrite CSRNet   ImageNet initialize)
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


def _dataset_label_to_local_pseudo_name(dataset_label: str) -> str:
    """take CONFIG['dataset']  namecharactermap toyou label filenameinsideuse namecharacter. 
    For example:RSOC-Building -> building, correspond rsoc-building_grid16_train.txt
    """
    mapping = {
        "RSOC-Building": "building",
        "RSOC-Ship": "ship",
        "RSOC-S-Vehicle": "small-vehicle",
        "RSOC-L-Vehicle": "large-vehicle",
        "VD-People": "people",
        "VD-Vehicle": "vehicle",
    }
    if dataset_label in mapping:
        return mapping[dataset_label]
    # fallback: removeemptycell, unifysmallwrite
    return dataset_label.strip().lower()


def _local_pseudo_label_path(root: str, dataset_label: str, split: str) -> str:
    label_subdir = str(CONFIG.get("local_pseudo_label_subdir", "local pseudo"))
    grid = str(CONFIG.get("local_pseudo_label_grid", "grid16"))
    name = _dataset_label_to_local_pseudo_name(dataset_label)
    # you name: rsoc-building_grid16_train.txt
    fname = f"rsoc-{name}_{grid}_{split}.txt"
    return os.path.join(root, label_subdir, fname)


def _load_local_pseudo_boxes(label_path: str) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """parseyou label txt: per line
        imagename x1 y1 x2 y2
    sameoneimagecanoutnowmanyrow(manybox). 
    """
    box_map: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for line in _safe_read_lines(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        img_name = parts[0]
        try:
            x1, y1, x2, y2 = map(float, parts[1:5])
        except Exception:
            continue
        box_map.setdefault(img_name, []).append((x1, y1, x2, y2))
    return box_map


def _sigma_from_area(area: float) -> float:
    """
    partsegmentlinearitymap(according to yourlatestrule): 
    - area < 1024            -> sigma = 1
    - 1024 ~ 5354            -> sigma = 1  to 15  linearity
    - 5355 ~ 16129           -> sigma = 15 to 32  linearity
    - area > 16129           -> sigma = 32

    note: strictcellaccording to yourto intervalendpoint 5354 / 5355 partsegment. 
    """
    a1 = 1024.0
    a2 = 5354.0
    a3 = 5355.0
    a4 = 16129.0

    s1 = 1.0
    s2 = 15.0
    s3 = 32.0

    area = float(area)

    if area < a1:
        return s1

    # [1024, 5354] -> [1, 15]
    if area <= a2:
        t = (area - a1) / (a2 - a1)
        return s1 + t * (s2 - s1)

    # (5354, 5355) extremenarrowinterval: according to 15 handle(basethisnotwilloutnow, butguaranteelogicstableset)
    if area < a3:
        return s2

    # [5355, 16129] -> [15, 32]
    if area <= a4:
        t = (area - a3) / (a4 - a3)
        return s2 + t * (s3 - s2)


    return s3



def boxes_to_density_map_dynamic_sigma(
    boxes_xyxy: List[Tuple[float, float, float, float]],
    orig_w: int,
    orig_h: int,
    out_size: int = 512,
    truncate: float = 3.0,
) -> np.ndarray:
    """takeoneimageon  N  box(originalimagecoordinate system)turnbecome out_size x out_size density map. 
    - every box takecenter pointdo"point"
    - everypointuseself  sigma(by box areadecide, andaccording tooutputcoordinate systemareacompute)
    - everypointoverlayone"sum=1" highkernel -> density mapsum=pointnumber(boxnumber)
    """
    H = W = int(out_size)
    dm = np.zeros((H, W), dtype=np.float32)
    if len(boxes_xyxy) == 0:
        return dm

    sx = float(out_size) / float(max(orig_w, 1))
    sy = float(out_size) / float(max(orig_h, 1))

    kernel_cache: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}
    # cache key: (round(sigma*1000), round(truncate*1000)) -> (kernel, radius)

    for (x1, y1, x2, y2) in boxes_xyxy:
        # center(map to out_size coordinate system)
        cx = (float(x1) + float(x2)) * 0.5 * sx
        cy = (float(y1) + float(y2)) * 0.5 * sy

        # area(according to out_size coordinate systemcompute)
        bw = max(0.0, float(x2) - float(x1)) * sx
        bh = max(0.0, float(y2) - float(y1)) * sy
        area = bw * bh

        sigma = float(_sigma_from_area(area))
        key = (int(round(sigma * 1000)), int(round(float(truncate) * 1000)))
        if key in kernel_cache:
            kernel, radius = kernel_cache[key]
        else:
            kernel = _gaussian_kernel2d(float(sigma), float(truncate))
            radius = kernel.shape[0] // 2
            kernel_cache[key] = (kernel, radius)

        xi = int(round(cx))
        yi = int(round(cy))
        # exceedboundarythenignore(andoriginal points_to_density_map  rowforconsistent)
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue

        x0 = max(0, xi - radius)
        x1i = min(W, xi + radius + 1)
        y0 = max(0, yi - radius)
        y1i = min(H, yi + radius + 1)

        kx0 = x0 - (xi - radius)
        kx1 = kx0 + (x1i - x0)
        ky0 = y0 - (yi - radius)
        ky1 = ky0 + (y1i - y0)

        dm[y0:y1i, x0:x1i] += kernel[ky0:ky1, kx0:kx1]

    return dm


def _build_allow_set_for_rsoc_mixed(root: str, dataset_label: str, split: str) -> set[str]:
    """use DOTA_data inside split_xxx.txt filter ship / small-vehicle / large-vehicle. 
    if split_xxx.txt notsavein, thenfall backscan labelTxt insideisor notincludethis class. 
    """
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
    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(root, "ASPDNet_dataset", "RSOC_building", "building")
        if split == "train":
            sub = "train_data"
        elif split in ("val", "test"):
            # building commononlyhas train/test
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
        vd_root = os.path.join(root, "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle")
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


def _read_dota_points(label_path: str, width: int, height: int, class_token: str, out_size: int = 512) -> np.ndarray:
    """DOTA labelTxt: per line 8coordinate + class + difficult. takecenter point, map to 512 coordinate system. """
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


def _read_building_mat_points(mat_path: str, width: int, height: int, out_size: int = 512) -> np.ndarray:
    """RSOC-Building: GT_*.mat insideusuallyhas center. 
    youto exampleinside center is object array: center[0,0] = Nx2  point; center[1,0] insidemaybecount. 
    """
    import scipy.io as sio

    mat = sio.loadmat(mat_path)
    if "center" not in mat:
        return np.zeros((0, 2), dtype=np.float32)
    center = mat["center"]

    pts: Optional[np.ndarray] = None
    try:
        # Common:object array, first itemis Nx2
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

    # map to 512 coordinate system
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


def _read_visdrone_txt_points(txt_path: str, width: int, height: int, out_size: int = 512) -> np.ndarray:
    """standard VisDrone GT: x,y,w,h,score,category,truncation,occlusion. heredefaultcount category>0. """
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
    """take Nx2 pointcoordinate(alreadythroughin 0..out_size coordinate system)turnbecome out_size x out_size  density map. 
    everypointoverlayone"sum=1" highkernel -> density mapsum=pointnumber. 
    """
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
# Dataset
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
        augment: bool = False,
        density_scale: float = 1.0,
    ):
        self.root = root
        self.dataset_label = dataset_label
        self.split = split
        self.img_size = int(img_size)
        self.sigma = float(sigma)
        self.truncate = float(truncate)
        self.augment = bool(augment)
        self.density_scale = float(density_scale)

        self.img_paths = _list_images(root, dataset_label, split)

        # =========================
        # local pseudo label: one split one txt, insidesurfaceincludethis split  sohasobjectbox. 
        # Note:label fileinsidenotoutnow image, viewforthisimageobjectnumber=0(density mapall 0, butnotwillskipsample). 
        # =========================
        self.use_local_pseudo_labels = bool(CONFIG.get("use_local_pseudo_labels", False))
        self._box_map: Dict[str, List[Tuple[float, float, float, float]]] = {}
        if self.use_local_pseudo_labels:
            label_path = _local_pseudo_label_path(self.root, self.dataset_label, self.split)
            if os.path.isfile(label_path):
                self._box_map = _load_local_pseudo_boxes(label_path)
            else:
                # training setlacklabeldirectreportwrong; val/test iflacklabelthenaccording to"all 0"handle(sidethenyoufirstrunthroughtraining)
                if self.split == "train":
                    raise FileNotFoundError(f"Local pseudo label file not found: {label_path}")
                print(f"[WARN] Local pseudo label file not found for split={self.split}: {label_path}. Treat all images as 0 targets.")
                self._box_map = {}

        # notuse torchvision.transforms, keepyouoriginalcomescript  PIL + numpy + torch pipeline
        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

        # augmentationparameter(onlyin self.augment=True whenuse)
        self.hflip_prob = float(CONFIG.get("hflip_prob", 0.5))
        self.vflip_prob = float(CONFIG.get("vflip_prob", 0.0))
        self.rot90_prob = float(CONFIG.get("rot90_prob", 0.5))
        self.jitter_prob = float(CONFIG.get("jitter_prob", 0.8))
        self.jitter_brightness = float(CONFIG.get("jitter_brightness", 0.2))
        self.jitter_contrast = float(CONFIG.get("jitter_contrast", 0.2))
        self.jitter_saturation = float(CONFIG.get("jitter_saturation", 0.2))

    def _maybe_color_jitter(self, img: Image.Image) -> Image.Image:
        """coloraugmentation(onlyasused forimage, notroledensity map). """
        if not self.augment:
            return img
        if random.random() > self.jitter_prob:
            return img

        # brightness
        if self.jitter_brightness > 0:
            fac = random.uniform(max(0.0, 1.0 - self.jitter_brightness), 1.0 + self.jitter_brightness)
            img = ImageEnhance.Brightness(img).enhance(fac)
        # contrast
        if self.jitter_contrast > 0:
            fac = random.uniform(max(0.0, 1.0 - self.jitter_contrast), 1.0 + self.jitter_contrast)
            img = ImageEnhance.Contrast(img).enhance(fac)
        # saturation
        if self.jitter_saturation > 0:
            fac = random.uniform(max(0.0, 1.0 - self.jitter_saturation), 1.0 + self.jitter_saturation)
            img = ImageEnhance.Color(img).enhance(fac)

        return img

    def _transform(self, img_pil: Image.Image) -> torch.Tensor:
        img = img_pil.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        img = self._maybe_color_jitter(img)

        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        arr = (arr - self._mean) / self._std
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr)

    def __len__(self) -> int:
        return len(self.img_paths)

    def _get_points_or_density(self, img_path: str, img_pil: Image.Image) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """return (points, density_map_if_already_given)
        - points: Nx2(alreadythroughmap to 0..img_size)
        - density_map_if_already_given: iforiginalannotationthenprovidedensity map(VisDrone .npy), thenreturnthisdensity map(originalresolution). 
        """
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
            mat_path = os.path.join(gt_dir, f"GT_{base}.mat")
            if not os.path.isfile(mat_path):
                # compatible: GT_IMG_000053.mat / IMG_000053.jpg
                mat_path2 = os.path.join(gt_dir, f"GT_{img_name}".replace(".jpg", ".mat"))
                mat_path = mat_path2
            if not os.path.isfile(mat_path):
                raise FileNotFoundError(f"Building GT not found: {mat_path}")
            pts = _read_building_mat_points(mat_path, w0, h0, out_size=self.img_size)
            return pts, None

        # VisDrone
        if self.dataset_label in ("VD-People", "VD-Vehicle"):
            vd_root = os.path.join(self.root, "VisDrone-People" if self.dataset_label == "VD-People" else "VisDrone-Vehicle")
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

    def _maybe_geo_aug(self, img: torch.Tensor, gt_density: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """geometryaugmentation: willsynchronizeroletodensity map(guaranteelabelconsistent). """
        if not self.augment:
            return img, gt_density

        # Hflip
        if random.random() < self.hflip_prob:
            img = torch.flip(img, dims=[2])          # CHW: flip W
            gt_density = torch.flip(gt_density, dims=[2])

        # Vflip
        if self.vflip_prob > 0 and random.random() < self.vflip_prob:
            img = torch.flip(img, dims=[1])          # flip H
            gt_density = torch.flip(gt_density, dims=[1])

        # Rot90 (0/90/180/270)
        if self.rot90_prob > 0 and random.random() < self.rot90_prob:
            k = random.randint(0, 3)
            if k != 0:
                img = torch.rot90(img, k=k, dims=[1, 2])
                gt_density = torch.rot90(gt_density, k=k, dims=[1, 2])

        return img, gt_density


    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)

        img_pil = Image.open(img_path).convert("RGB")
        img = self._transform(img_pil)

        if self.use_local_pseudo_labels:
            boxes = self._box_map.get(img_name, [])
            dm_np = boxes_to_density_map_dynamic_sigma(
                boxes_xyxy=boxes,
                orig_w=img_pil.size[0],
                orig_h=img_pil.size[1],
                out_size=self.img_size,
                truncate=self.truncate,
            )
            gt_density = torch.from_numpy(dm_np).unsqueeze(0)  # 1x512x512
        else:
            pts, dm_given = self._get_points_or_density(img_path, img_pil)

            if dm_given is not None:
                # ifannotationdirectto isdensity map: resize to 512, andas much as possiblekeep sum notchange
                dm = torch.from_numpy(dm_given).unsqueeze(0).unsqueeze(0)  # 1x1xH xW
                dm = dm.to(torch.float32)
                old_sum = float(dm.sum().item())
                dm = F.interpolate(dm, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
                new_sum = float(dm.sum().item())
                if new_sum > 0 and old_sum > 0:
                    dm = dm * (old_sum / new_sum)
                gt_density = dm.squeeze(0)  # 1x512x512
            else:
                dm_np = points_to_density_map(pts, out_size=self.img_size, sigma=self.sigma, truncate=self.truncate)
                gt_density = torch.from_numpy(dm_np).unsqueeze(0)  # 1x512x512

        # geometryaugmentation(synchronizedensity map)
        img, gt_density = self._maybe_geo_aug(img, gt_density)

        # here  gt_count is"originaldensity map" count(notfollow density_scale change)
        gt_count = gt_density.sum().to(torch.float32)

        # youneedrequest: trainingwhenregression"putlargeversion" density map
        if self.density_scale != 1.0:
            gt_density = gt_density * self.density_scale

        return img.to(torch.float32), gt_density.to(torch.float32), gt_count, img_name


# =========================
# Model: CSRNet (VGG16 beforeend + dilated conv afterend)
# =========================
class CSRNet(nn.Module):
    """CSRNet: VGG16 frontend (up to conv4_3) + dilated convolution backend.

    in order tonotchangeyouoriginalcome training/dataAPI, herewilltake 1/8 resolutionoutputupsampleback 512x512, 
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
# Train / Eval
# =========================
def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,   # stillthenis MSE, usecomestatistics density_mse
    device: str,
):
    model.train()
    loop = tqdm(loader, desc=f"train {epoch}", ncols=110)
    mse_avg = 0.0
    l1_avg = 0.0
    count_mse_avg = 0.0
    n = 0

    # ========= youoriginalscriptinsidealreadyhas: L1 + loss_scale =========
    l1_weight = float(CONFIG.get("l1_weight", 1.0))
    loss_scale = float(CONFIG.get("loss_scale", 1.0))
    # ====================================================

    # ========= youneedrequestnewly added: density mapscalemultiple(regressionputlargeversion) =========
    density_scale = float(CONFIG.get("density_scale", 1.0))
    if density_scale <= 0:
        density_scale = 1.0
    # ======================================================

    for img, gt_density, gt_count, _ in loop:
        img = img.to(device)
        gt_density = gt_density.to(device)  # Note:here  gt_density alreadythroughis"putlargeversion"(dataset insidemultiply  density_scale)
        gt_count = gt_count.to(device)      # here  gt_count is"originalcount"(notputlarge)

        pred_density = model(img)           # predict alsois"putlargeversion"

        # ========= totalloss = (MSE + w*L1) * loss_scale =========
        mse = criterion(pred_density, gt_density)
        l1 = F.l1_loss(pred_density, gt_density)
        loss = (mse + l1_weight * l1) * loss_scale
        # =====================================================

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # countevaluation: take pred_density exceptbackgoagainsum
            pred_count = pred_density.sum(dim=(1, 2, 3)) / density_scale
            batch_count_mse = torch.mean((pred_count - gt_count) ** 2)

        n += 1
        mse_avg += float(mse.item())
        l1_avg += float(l1.item())
        count_mse_avg += float(batch_count_mse.item())
        loop.set_postfix(density_mse=mse_avg / n, density_l1=l1_avg / n, count_mse=count_mse_avg / n)

    return mse_avg / max(n, 1), l1_avg / max(n, 1), count_mse_avg / max(n, 1)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str, desc: str = "eval") -> Tuple[float, float, float]:
    model.eval()
    mse_avg = 0.0
    l1_avg = 0.0
    count_mse_avg = 0.0
    n = 0

    density_scale = float(CONFIG.get("density_scale", 1.0))
    if density_scale <= 0:
        density_scale = 1.0

    for img, gt_density, gt_count, _ in tqdm(loader, desc=desc, ncols=110):
        img = img.to(device)
        gt_density = gt_density.to(device)
        gt_count = gt_count.to(device)

        pred_density = model(img)

        # evaluationstillthenprint MSE/L1(aligntrainingObjective:hereis"putlargeversion"density map error)
        mse = criterion(pred_density, gt_density)
        l1 = F.l1_loss(pred_density, gt_density)

        pred_count = pred_density.sum(dim=(1, 2, 3)) / density_scale
        count_mse = torch.mean((pred_count - gt_count) ** 2)

        n += 1
        mse_avg += float(mse.item())
        l1_avg += float(l1.item())
        count_mse_avg += float(count_mse.item())

    return mse_avg / max(n, 1), l1_avg / max(n, 1), count_mse_avg / max(n, 1)


def main():
    set_seed(int(CONFIG["seed"]))
    root = CONFIG["dataset_root"]
    dataset_label = CONFIG["dataset"]
    device = CONFIG["device"]
    img_size = int(CONFIG["img_size"])
    sigma = float(CONFIG["sigma"])
    truncate = float(CONFIG["truncate"])

    # CSRNet
    model = CSRNet(
        imagenet_pretrained=bool(CONFIG.get("csrnet_imagenet_pretrained", True)),
        vgg16_weights_path=str(CONFIG.get("vgg16_weights_path", "")),
    ).to(device)
    print(model.__class__.__name__)

    if CONFIG.get("resume"):
        model.load_state_dict(torch.load(CONFIG["resume"], map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(CONFIG["lr"]), weight_decay=float(CONFIG["weight_decay"]))

    # ========= youoriginalscriptinsidealreadyhas: learning rateone by onegradualdecaysubtract(Cosine) =========
    epochs = int(CONFIG["epochs"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(CONFIG.get("min_lr", 1e-6)),
    )
    # ===========================================================

    density_scale = float(CONFIG.get("density_scale", 1.0))
    if density_scale <= 0:
        density_scale = 1.0

    # onlytraining setenableaugmentation
    use_aug = bool(CONFIG.get("use_augmentation", True))

    train_ds = DensityMapDataset(root, dataset_label, "train", img_size=img_size, sigma=sigma, truncate=truncate, augment=use_aug, density_scale=density_scale)
    val_ds = DensityMapDataset(root, dataset_label, "val", img_size=img_size, sigma=sigma, truncate=truncate, augment=False, density_scale=density_scale)
    test_ds = DensityMapDataset(root, dataset_label, "test", img_size=img_size, sigma=sigma, truncate=truncate, augment=False, density_scale=density_scale)

    # somedatasetno val(such as building), alonguseyouoriginalcomescript strategy: use train fillwhen val
    if len(val_ds) == 0:
        print("[WARN] current dataset has no val split, automatically use train as val. ")
        val_ds = train_ds

    if len(train_ds) == 0:
        raise RuntimeError(f"No training images found for {dataset_label} under {root}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=True,
        num_workers=int(CONFIG["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(CONFIG["num_workers"]) // 2),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(CONFIG["num_workers"]) // 2),
        pin_memory=True,
    )

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    best_val_count_mse = float("inf")

    for epoch in range(1, epochs + 1):
        train_mse, train_l1, train_count_mse = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
        val_mse, val_l1, val_count_mse = eval_epoch(model, val_loader, criterion, device, desc="val")

        # current lr
        cur_lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:03d} | lr={cur_lr:.8f} | "
            f"train density_mse={train_mse:.6f} density_l1={train_l1:.6f} count_mse={train_count_mse:.6f} "
            f"| val density_mse={val_mse:.6f} density_l1={val_l1:.6f} count_mse={val_count_mse:.6f}"
        )

        # save: according to val   count_mse best
        if val_count_mse < best_val_count_mse:
            best_val_count_mse = val_count_mse
            out_path = os.path.join(CONFIG["save_dir"], f"best_countmse{best_val_count_mse:.6f}_epoch{epoch}_{dataset_label}.pth")
            torch.save(model.state_dict(), out_path)
            print(f"[SAVE] {out_path}")

        # ========= youoriginalscriptinsidealreadyhas: every epoch afterupdate scheduler =========
        scheduler.step()
        # =======================================================

    # test setevaluation
    if len(test_ds) == 0:
        print("[WARN] notfindto test split image; skip test evaluation. ")
        return

    try:
        test_mse, test_l1, test_count_mse = eval_epoch(model, test_loader, criterion, device, desc="test")
        print(f"Test density_mse={test_mse:.6f} | Test density_l1={test_l1:.6f} | Test count_mse={test_count_mse:.6f}")
    except FileNotFoundError as e:
        # ifsomedataset test no GT, willin __getitem__ throw FileNotFoundError
        print(f"[WARN] test split nomethodread GT(maybenoannotation). skip test MSE. \n  detail: {e}")


if __name__ == "__main__":
    main()
