# -*- coding: utf-8 -*-
"""
simpleversion: use SAM3 dofeatureraisetake + linearityclassification head
- retainpathconfiguration(LOCAL_SAM3_PATH, SAM3_CHECKPOINT_PATH, DATA_DIR, SAVE_DIR, SPLIT_PATH)
- Windows onmask triton / flash_attn
- keychange: inputresolutionchangefor 1008x1008(compatible SAM3 ViTDet   RoPE)
"""

import sys
import os
import types
from unittest.mock import MagicMock
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ========================================================
# 0. Windows compatibilitypatchsmall (Triton/FlashAttn simulate)
# ========================================================
if sys.platform.startswith("win"):
    print("[Windows Fix] mount dummy triton / flash_attn ...")

    def _dummy_decorator(*args, **kwargs):
        if kwargs or (args and not callable(args[0])):
            return lambda f: f
        return args[0]

    mock_triton = types.ModuleType("triton")
    mock_triton.__spec__ = MagicMock()
    mock_triton.__spec__.name = "triton"
    mock_triton.__file__ = "dummy_triton_path"
    mock_triton.__path__ = []
    mock_triton.__version__ = "2.0.0"

    mock_triton.jit = _dummy_decorator
    mock_triton.autotune = _dummy_decorator
    mock_triton.heuristics = _dummy_decorator
    mock_triton.cdiv = lambda x, y: (x + y - 1) // y
    mock_triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()

    class DummyConfig:
        def __init__(self, *args, **kwargs):
            pass

    mock_triton.Config = DummyConfig
    mock_triton.language = MagicMock()
    mock_triton.impl = MagicMock()

    sys.modules["triton"] = mock_triton
    sys.modules["triton.language"] = mock_triton.language
    sys.modules["triton.impl"] = mock_triton.impl

    # flash_attn directmask
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        mock_flash = types.ModuleType("flash_attn")
        mock_flash.__spec__ = MagicMock()
        sys.modules["flash_attn"] = mock_flash
        sys.modules["flash_attn.flash_attn_interface"] = MagicMock()

    print("[Windows Fix] dummy triton / flash_attn mountfinish")

# ========================================================
# 1. pathconfiguration(retainyou setting)
# ========================================================
LOCAL_SAM3_PATH = r"D:\sam3_source"
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"  # ifnotuselocal ckpt, can be changedfor None
DATA_DIR = "dataset_raw"
SAVE_DIR = "saved_models"
SAVE_PATH = os.path.join(SAVE_DIR, "sam3_finetuned.pth")
SPLIT_PATH = os.path.join(SAVE_DIR, "split_indices.npz")

# *** key: resolutionchange to 1008, compatible ViTDet   RoPE ***
INPUT_SIZE = 1008  # 1008/14 = 72, 72*72=5184  patch, andpretrainedconsistent
BATCH_SIZE = 2
LR = 1e-3
NUM_EPOCHS = 15
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if os.path.exists(LOCAL_SAM3_PATH) and (LOCAL_SAM3_PATH not in sys.path):
    sys.path.insert(0, LOCAL_SAM3_PATH)

# ========================================================
# 2. import SAM3 buildfunction
# ========================================================
try:
    # according toofficialtextfileuse sam3.model_builder insidefunctionname
    from sam3.model_builder import build_sam3_image_model

    print("[Info] successimport build_sam3_image_model (sam3.model_builder)")
except ImportError as e:
    print(f"[Error] cannot import build_sam3_image_model: {e}")
    sys.exit(1)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)


# ========================================================
# 3. classificationmodelencapsulate: freeze SAM3, onlytrainingonelinearityhead
# ========================================================
class SAM3Classifier(nn.Module):
    def __init__(self, num_classes: int, checkpoint_path: str | None = None):
        super().__init__()
        print("[Model] build SAM3 classificationmodel...")

        # === 3.1 buildoriginal SAM3 model: preferuselocal checkpoint ===
        use_local = checkpoint_path is not None and os.path.exists(checkpoint_path)
        try:
            if use_local:
                print(f"[Model] uselocalweight: {checkpoint_path}")
                self.sam_model = build_sam3_image_model(
                    checkpoint_path=checkpoint_path,
                    load_from_HF=False,
                    device=str(DEVICE).split(":")[0]
                )
            else:
                print("[Model] notfindtolocalweight, tryfrom HF belowload(needcanusenetwork)")
                self.sam_model = build_sam3_image_model(
                    checkpoint_path=None,
                    load_from_HF=True,
                    device=str(DEVICE).split(":")[0]
                )
        except Exception as e:
            print(f"[Fatal] build SAM3 modelfail: {e}")
            sys.exit(1)


        # === 3.2 findtovision backbone ===
        # officialimplementinside Image modelusuallyhas backbone(VL groupcombinedevice)and trunk(purevisiondopath)
        self.image_encoder = None
        self.needs_captions = False

        if hasattr(self.sam_model, "trunk"):
            # this kindsituationusuallyispurevision backbone(classsimilar SAM2)
            self.image_encoder = self.sam_model.trunk
            print("[Model] use sam_model.trunk asimageencoder")
        elif hasattr(self.sam_model, "backbone"):
            # VL groupcombine backbone, needpropagate captions
            self.image_encoder = self.sam_model.backbone
            self.needs_captions = True
            print("[Model] use sam_model.backbone asimageencoder(need captions)")
        elif hasattr(self.sam_model, "image_encoder"):
            self.image_encoder = self.sam_model.image_encoder
            print("[Model] use sam_model.image_encoder asimageencoder")
        else:
            print("[Error] in SAM3 modelinfindnotto trunk/backbone/image_encoder, nomethodcontinue")
            sys.exit(1)

        # freeze SAM3 sohasparameter
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        # === 3.3 detectoutputfeaturenumber of channels ===
        self.feature_dim = 1024  # todefaultvalue, preventdetectfail
        try:
            self.image_encoder.to(DEVICE)
            dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)

            if self.needs_captions:
                raw_out = self.image_encoder(dummy, [""])
            else:
                raw_out = self.image_encoder(dummy)

            feat = self._pick_feature_map(raw_out)
            if feat is None:
                raise RuntimeError("notcanin image_encoder outputinfindto 4D feature map")

            self.feature_dim = int(feat.shape[1])
            print(f"[Model] featuredimensiondetectsuccess: {self.feature_dim}")
        except Exception as e:
            print(f"[Warning] featuredimensiondetectfail: {e}, usedefault feature_dim = {self.feature_dim}")

        # === 3.4 classification head ===
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_dim, num_classes)

    @staticmethod
    def _pick_feature_map(raw_out):
        """
        from SAM3 backbone  outputinside, selectone [B, C, H, W]  feature mapout. 
        compatible dict / list / tensor etc.manykindstyle. 
        """
        if isinstance(raw_out, torch.Tensor) and raw_out.dim() == 4:
            return raw_out

        if isinstance(raw_out, dict):
            # prefertrysomecommon key
            for k in ["vision_features", "image_features", "trunk", "features"]:
                if k in raw_out and isinstance(raw_out[k], torch.Tensor) and raw_out[k].dim() == 4:
                    return raw_out[k]
            # elseinsohas value find inNo.one 4D Tensor
            for v in raw_out.values():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    return v

        if isinstance(raw_out, (list, tuple)):
            for v in raw_out:
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    return v

        return None

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        """onlydo once forward, takeimageencodebecomefeature map(notrequestgradient)"""
        with torch.no_grad():
            if self.needs_captions:
                bsz = x.size(0)
                captions = [""] * bsz
                raw_out = self.image_encoder(x, captions)
            else:
                raw_out = self.image_encoder(x)

            feat = self._pick_feature_map(raw_out)
            if feat is None:
                raise RuntimeError("in forward innotcangethaseffect  4D feature map")
            return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_feat(x)                # [B, C, H, W]
        pooled = self.avg_pool(feat)              # [B, C, 1, 1]
        flat = torch.flatten(pooled, 1)           # [B, C]
        logits = self.fc(flat)                    # [B, num_classes]
        return logits


# ========================================================
# 4. datasetand DataLoader
# ========================================================
if not os.path.exists(DATA_DIR):
    print(f"[Error] data directorynotsavein: {DATA_DIR}")
    sys.exit(1)

full_ds = datasets.ImageFolder(DATA_DIR)
class_names = full_ds.classes
num_classes = len(class_names)
print(f"[Data] class: {class_names} (num_classes={num_classes})")

# imageaugment: force resize to 1008x1008, guarantee patch numberandpretrainedconsistent
print(f"[Data] useresolution: {INPUT_SIZE}x{INPUT_SIZE}")
train_tf = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, transform):
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        img = self.transform(img)
        return img, label


# read / generate train/val split
if os.path.exists(SPLIT_PATH):
    split = np.load(SPLIT_PATH)
    train_idx = split["train_indices"]
    val_idx = split["val_indices"]
    print("[Data] usealreadyhassplit split_indices.npz")
else:
    indices = np.arange(len(full_ds))
    np.random.shuffle(indices)
    sp = int(0.8 * len(indices))
    train_idx, val_idx = indices[:sp], indices[sp:]
    np.savez(SPLIT_PATH, train_indices=train_idx, val_indices=val_idx, test_indices=np.array([], dtype=int))
    print("[Data] alreadycreatenew  train/val split, andsaveto split_indices.npz")

train_ds = TransformedDataset(Subset(full_ds, train_idx), train_tf)
val_ds = TransformedDataset(Subset(full_ds, val_idx), train_tf)

# Windows below num_workers=0, avoidmultiprocessing dataloader eachkindedge cases
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ========================================================
# 5. trainingloop
# ========================================================
print(f"[Train] device: {DEVICE}")

model = SAM3Classifier(num_classes=num_classes, checkpoint_path=SAM3_CHECKPOINT_PATH).to(DEVICE)

# onlytrainingclassification head
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # ---------- Train ----------
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(logits, dim=1)
        running_total += labels.size(0)
        running_correct += (preds == labels).sum().item()

    train_loss = running_loss / max(1, running_total)
    train_acc = running_correct / max(1, running_total)

    # ---------- Val ----------
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(inputs)
            _, preds = torch.max(logits, dim=1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / max(1, val_total)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
        f"Val Acc: {val_acc:.2%}"
    )

    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  -> new best modelalreadysaveto {SAVE_PATH} (Val Acc: {val_acc:.2%})")

print("\n[Done] training finished. ")
print(f"[Best] bestvalidationaccuracy: {best_acc:.2%}")
