# finetune_calibration_with_gt.py
# usereal label(non-pseudo label)to"finallymaplayer(k,b)"performfine-tunecalibration
# Usage:directinIDErunthisfile, notneedcommand line

import os
import glob
import math
import time
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# =========================
# 1) only modify herethencanchangedataset
# =========================
ROOT_DATA_DIR = "dataset"          # datasetroot directory(andyounowhasprogramconsistent)
MODEL_DEF_FILE = "mainprogram.py"       # you "trainingpseudo label mainprogram"filepath(insidesurfacehas alt_gvt_small define)
MY_MODEL_DIR = "my_model"          # yousavepseudo labeltrainingmodel directory

# optionaldataset: 
# "RSOC-Building", "RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle", "VD-People", "VD-Vehicle"
DATASET_NAME = "RSOC-Building"

# traininguse  split(building usuallyonlyhas train/test; ifno val, willautomaticuse train do val)
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

# trainingparameter
BATCH_SIZE = 8
NUM_WORKERS = 4
LR = 1e-2                  # onlytraining(k,b)socanthanmaindolarge
WEIGHT_DECAY = 0.0
EPOCHS = 50
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# =========================
# 2) dynamicloadyou model definition(avoid import intextmodulenameproblem)
#    notuse ALTGVT.py, onlyuseyou "mainprogram.py"
# =========================
def load_model_factory_from_file(py_file: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_model_def", py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # onlyexecutedefine, notwillrun __main__
    if not hasattr(module, "alt_gvt_small"):
        raise RuntimeError(f"in {py_file} find innotto alt_gvt_small, please confirmyoutoI mainprograminhasthisfunction. ")
    return module.alt_gvt_small


def pick_latest_checkpoint(model_dir: str) -> str:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"findnottomodeldirectory: {model_dir}")
    cks = glob.glob(os.path.join(model_dir, "*.pth"))
    if not cks:
        raise FileNotFoundError(f"{model_dir} belowno .pth file")
    cks.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cks[0]


# =========================
# 3) readreal label: differentdatasetdifferent GT format
# =========================
def _build_allow_set_for_rsoc_mixed(dataset_label: str, split: str) -> set:
    """
    ship / small-vehicle / large-vehicle mixed intogether, 
    use DOTA_data inside split_xxx.txt comefilteroutbelonginthis class filename(notcontainextension)
    """
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
    # RSOC-Building
    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
        # youto structure: train_data / test_data
        if split == "train":
            img_dir = os.path.join(rsoc_root, "train_data", "images")
        elif split == "test":
            img_dir = os.path.join(rsoc_root, "test_data", "images")
        elif split == "val":
            # building notto val, wereturnempty, outsidesurfacewillautomatic fallback
            return []
        else:
            return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    # RSOC-Ship / RSOC-S-Vehicle / RSOC-L-Vehicle
    if dataset_label in ["RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"]:
        img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split, "images")
        if not os.path.isdir(img_dir):
            return []
        allow = _build_allow_set_for_rsoc_mixed(dataset_label, split)
        out = [p for p in glob.glob(os.path.join(img_dir, "*.jpg"))
               if os.path.splitext(os.path.basename(p))[0] in allow]
        return sorted(out)

    # VisDrone
    if dataset_label == "VD-People":
        img_dir = os.path.join(ROOT_DATA_DIR, "VisDrone-People", split, "images")
        if not os.path.isdir(img_dir):
            return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if dataset_label == "VD-Vehicle":
        # youto structure: val insidemaybe Images(largewrite)
        img_dir1 = os.path.join(ROOT_DATA_DIR, "VisDrone-Vehicle", split, "images")
        img_dir2 = os.path.join(ROOT_DATA_DIR, "VisDrone-Vehicle", split, "Images")
        img_dir = img_dir1 if os.path.isdir(img_dir1) else img_dir2
        if not os.path.isdir(img_dir):
            return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    return []


def load_building_count_from_mat(mat_path: str) -> float:
    """
    building   GT: GT_IMG_xxx.mat
    youto exampleinside mat onlyhasonevariable 'center', itsin center[1] save isnumber
    """
    try:
        import scipy.io
        m = scipy.io.loadmat(mat_path)
    except Exception as e:
        raise RuntimeError(f"read mat fail: {mat_path}\n{e}")

    if "center" not in m:
        raise KeyError(f"{mat_path} find innotto 'center' field")

    center = m["center"]  # shape (2,1)
    # center[0,0] = Nx2 pointcoordinate; center[1,0] = [[count]]
    try:
        cnt = float(center[1, 0][0, 0])
        return cnt
    except Exception:
        # fallback: usecoordinatepointnumber
        pts = center[0, 0]
        return float(pts.shape[0])


def load_dota_count_from_label(label_path: str, target_cls: str) -> float:
    """
    DOTA/RSOC mixclasslabel: per linefinallyis "class difficulty"
    onlystatisticsspecify class  number
    """
    if not os.path.isfile(label_path):
        return 0.0

    cnt = 0
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # skip header row
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
    """
    VisDrone GT: maybe .txt or .npy
    - .npy: usuallyisdensity map, sum i.e. count
    - .txt: one per lineobject, rownumberi.e. count(morethroughuse)
    """
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
                line = line.strip()
                if not line:
                    continue
                # VisDrone GT onegeneralis 8 column, non-emptyrowi.e.object
                cnt += 1
        return float(cnt)

    # findnotto GT thenwhen 0
    return 0.0


class GroundTruthCountDataset(Dataset):
    """
    Output:(image_tensor, true_count, img_name)
    """
    def __init__(self, dataset_label: str, split: str, transform=None):
        self.dataset_label = dataset_label
        self.split = split
        self.img_paths = _list_images_for_dataset_split(dataset_label, split)

        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # class token(used formixclasscount)
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
            # train/val only thenhas labelTxt
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


# =========================
# 4) onlytrainingfinally  k,b: score -> count
# =========================
class Calibrator(nn.Module):
    """
    onlyhastwolearnableArguments:k and b
    count = k * score + b
    """
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, score):
        # score: [B,1]
        return self.k * score + self.b


def freeze_model_all_params(model: nn.Module):
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

        _, score = model(img)             # score: [B,1], inyou networkinsideis sigmoid output 0~1
        pred = calib(score)               # mapbecometoolvolumenumber

        diff = (pred - gt_count).abs().sum().item()
        mae_sum += diff
        mse_sum += ((pred - gt_count) ** 2).sum().item()
        n += img.size(0)

    mae = mae_sum / max(n, 1)
    rmse = math.sqrt(mse_sum / max(n, 1))
    return mae, rmse


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1) loadmodel definition
    alt_gvt_small = load_model_factory_from_file(MODEL_DEF_FILE)

    # 2) buildmodelandloadpseudo labeltraining weight
    model = alt_gvt_small(pretrained=False)
    model.to(DEVICE)

    ckpt_path = pick_latest_checkpoint(MY_MODEL_DIR)
    print(f"[INFO] use my_model inlatestweight: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)

    # 3) freezemodel
    freeze_model_all_params(model)
    model.eval()  # maindoforever eval(onlytraining calibrator)

    # 4) onlytraining calibrator
    calib = Calibrator().to(DEVICE)
    optimizer = optim.Adam(calib.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.L1Loss()

    # 5) dataset(real label)
    train_set = GroundTruthCountDataset(DATASET_NAME, TRAIN_SPLIT)
    val_set = GroundTruthCountDataset(DATASET_NAME, VAL_SPLIT)

    # ifno val(such as building), use train fillwhen val(youalsocanaftersurfaceselfdisassemblepart)
    if len(val_set) == 0:
        print("[WARN] current dataset has no val split, automatically use train as val evaluation. ")
        val_set = train_set

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"[INFO] Dataset={DATASET_NAME} train={len(train_set)} val={len(val_set)} device={DEVICE}")

    # 6) trainingloop: maindo no_grad, onlyhas (k,b) update
    best_mae = float("inf")
    os.makedirs("calib_ckpt", exist_ok=True)

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

        # 7) evaluation
        mae, rmse = evaluate(model, calib, val_loader)
        print(f"[VAL] epoch={epoch} MAE={mae:.3f} RMSE={rmse:.3f}   (k={calib.k.item():.4f}, b={calib.b.item():.4f})")

        # 8) saveoptimalcalibrationlayer
        if mae < best_mae:
            best_mae = mae
            save_path = os.path.join("calib_ckpt", f"{DATASET_NAME}_best_calib.pth")
            torch.save({
                "dataset": DATASET_NAME,
                "checkpoint_used": ckpt_path,
                "k": calib.k.detach().cpu(),
                "b": calib.b.detach().cpu(),
            }, save_path)
            print(f"[SAVE] best calib saved to: {save_path}")

    print("[DONE] Calibration training finished.")


if __name__ == "__main__":
    main()
