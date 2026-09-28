"""
train.py
========
Training pipeline for JCNN Mudra Recognition.

Steps:
  1. Discover classes from dataset/ subfolders
  2. Extract MediaPipe joints from all images (cached in .npz)
  3. Phase 1: Pretrain JGCN + VIE  (λ4 = 0,  Eq.12 without Lcls)
  4. Phase 2: Finetune with JWC     (λ4 = 1.0, full Eq.12)
  5. Save best weights + build prototypes
"""
# handles version 3.0 API endpoint /train and training logic; polled by /train/status for progress updates
from __future__ import annotations 
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import mediapipe.python.solutions.hands as _mp_hands

from model.jcnn_model import JCNN, compute_loss, KEY_JOINT_IDX, NUM_KEY_JOINTS

log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT  = Path("/app/dataset")
WEIGHTS_DIR   = Path("/app/weights")
WEIGHTS_PATH  = WEIGHTS_DIR / "mudra_model.pth"
CLASSES_PATH  = WEIGHTS_DIR / "classes.json"
CACHE_PATH    = WEIGHTS_DIR / "joint_cache.npz"

# ── hyper-parameters (paper Section IV-B) ─────────────────────────────────────
EPOCHS_PRE  = 20
EPOCHS_FT   = 25
BATCH_SIZE  = 64
LR          = 3e-4    # AdamW learning rate 0.0003 (paper)
VAL_RATIO   = 0.15
EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── global status (polled by API) ─────────────────────────────────────────────
STATUS: dict = {"running": False, "progress": 0,
                "message": "Idle", "accuracy": 0.0, "phase": ""}

# ── progress update functions ─────────────────────────────────────────────────────────
def _upd(pct, msg, acc=0.0, accuracy=None, phase=""):
    """acc and accuracy are both accepted for backward compatibility."""
    final_acc = accuracy if accuracy is not None else acc
    STATUS.update(running=True, progress=pct,
                  message=msg, accuracy=final_acc, phase=phase)
    log.info("[%3d%%] %s", pct, msg)

#── class discovery functions ─────────────────────────────────────────────────────────
def discover_classes(root: Path) -> list[str]:
    classes = sorted(d.name for d in root.iterdir()
                     if d.is_dir() and not d.name.startswith("."))
    if not classes:
        raise FileNotFoundError(
            f"No class folders found in {root}.\n"
            "Create: dataset/<mudra_name>/img001.jpg ...")
    return classes

# ── MediaPipe joint extraction and caching ─────────────────────────────────
def extract_joints(bgr, hands_obj):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = hands_obj.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    pad = np.zeros((21, 3), dtype=np.float32)
    per = []
    for lm in res.multi_hand_landmarks[:2]:
        per.append(np.array([[p.x, p.y, p.z] for p in lm.landmark],
                             dtype=np.float32))
    while len(per) < 2:
        per.append(pad.copy())
    return np.concatenate(per[:2], axis=0).astype(np.float32)


def build_cache(root, classes, path):
    _upd(5, "Extracting MediaPipe joints (one-time cache)…")
    hands = _mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                            min_detection_confidence=0.4)
    all_j, all_l = [], []
    skip = 0
    for ci, cls in enumerate(classes):
        imgs = [p for p in sorted((root / cls).iterdir())
                if p.suffix.lower() in EXTS]
        got = 0
        for img_path in imgs:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                skip += 1; continue
            j = extract_joints(bgr, hands)
            if j is None:
                skip += 1; continue
            all_j.append(j); all_l.append(ci); got += 1
        pct = 5 + int(40 * (ci + 1) / len(classes))
        _upd(pct, f"  {cls}: {got}/{len(imgs)} extracted")
    hands.close()

    J = np.array(all_j, dtype=np.float32)
    L = np.array(all_l, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), joints=J, labels=L, classes=np.array(classes))
    log.info("Cache: %d samples (%d skipped)", len(J), skip)
    return J, L


def load_cache(root, classes, path):
    if path.exists():
        d = np.load(str(path), allow_pickle=True)
        if list(d["classes"]) == classes:
            log.info("Loaded cache: %d samples", len(d["joints"]))
            return d["joints"], d["labels"]
        log.info("Class list changed — rebuilding cache")
    return build_cache(root, classes, path)


# ── Dataset ───────────────────────────────────────────────────────────────────
class JointDataset(Dataset):
    def __init__(self, J, L, augment=False):
        self.J = torch.tensor(J, dtype=torch.float32)
        self.L = torch.tensor(L, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.J)

    def __getitem__(self, idx):
        j = self.J[idx].clone()
        if self.augment:
            j += 0.007 * torch.randn_like(j)   # coordinate jitter
            if torch.rand(1) > 0.5:             # random horizontal flip
                j[:, 0] = 1.0 - j[:, 0]
        return j, self.L[idx]


def evaluate(model, loader, device):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for J, Y in loader:
            J, Y   = J.to(device), Y.to(device)
            logits, *_ = model(J, use_jwc=True)
            c += (logits.argmax(-1) == Y).sum().item()
            t += Y.size(0)
    return 100.0 * c / t if t else 0.0


def build_prototypes(model, loader, device, nc):
    """Average fvi and p_can per class → w_y and Q̃c (paper Section III-B-3)."""
    from model.jcnn_model import FEAT_DIM
    fs  = torch.zeros(nc, NUM_KEY_JOINTS, FEAT_DIM).to(device)
    cs  = torch.zeros(nc, NUM_KEY_JOINTS, 3).to(device)
    cnt = torch.zeros(nc).to(device)
    model.eval()
    with torch.no_grad():
        for J, Y in loader:
            J, Y = J.to(device), Y.to(device)
            _, fvi, p_can, _, _ = model(J)
            for i in range(Y.size(0)):
                c = Y[i].item()
                fs[c] += fvi[i]; cs[c] += p_can[i]; cnt[c] += 1
    cnt = cnt.clamp(min=1).view(-1, 1, 1)
    return fs / cnt, cs / cnt


# ── Main training entry point ─────────────────────────────────────────────────
def run_training() -> bool:
    global STATUS
    STATUS = {"running": True, "progress": 2,
              "message": "Starting…", "accuracy": 0.0, "phase": "init"}
    try:
        if not DATASET_ROOT.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {DATASET_ROOT}\n"
                "Place images at: backend/dataset/<ClassName>/*.jpg")

        classes = discover_classes(DATASET_ROOT)
        nc      = len(classes)
        _upd(3, f"Found {nc} classes: {classes[:6]}…")

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CLASSES_PATH, "w") as f:
            json.dump(classes, f, indent=2)

        J, L = load_cache(DATASET_ROOT, classes, CACHE_PATH)
        if len(J) == 0:
            raise ValueError("No valid hand images found — check dataset.")

        # Train/val split
        idx = np.random.permutation(len(J))
        nv  = max(1, int(VAL_RATIO * len(idx)))
        vi, ti = idx[:nv], idx[nv:]

        tr_ds = JointDataset(J[ti], L[ti], augment=True)
        va_ds = JointDataset(J[vi], L[vi], augment=False)
        tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,
                           num_workers=0, drop_last=False)
        va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0)
        log.info("Train: %d  Val: %d", len(tr_ds), len(va_ds))

        device = torch.device("cpu")
        model  = JCNN(nc).to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                     opt, T_max=EPOCHS_PRE + EPOCHS_FT)
        best   = 0.0

        # ── Phase 1: Pretrain (λ4 = 0, paper Section IV-B-1) ─────────────────
        for ep in range(EPOCHS_PRE):
            model.train()
            tl = 0.0
            for J_b, Y_b in tr_ld:
                J_b, Y_b = J_b.to(device), Y_b.to(device)
                opt.zero_grad()
                logits, fvi, pcan, p3d, _ = model(J_b)
                gt_can = J_b[:, KEY_JOINT_IDX, :]
                loss, _ = compute_loss(logits, Y_b, p3d, J_b,
                                       pcan, gt_can, phase="pretrain")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl += loss.item()
            sched.step()
            pct = 45 + int(20 * (ep + 1) / EPOCHS_PRE)
            _upd(pct, f"Phase 1 [{ep+1}/{EPOCHS_PRE}] loss={tl/len(tr_ld):.4f}",
                 phase="pretrain")

        # ── Phase 2: Finetune (λ4 = 1.0) ─────────────────────────────────────
        for ep in range(EPOCHS_FT):
            model.train()
            tl = correct = total = 0
            for J_b, Y_b in tr_ld:
                J_b, Y_b = J_b.to(device), Y_b.to(device)
                opt.zero_grad()
                logits, fvi, pcan, p3d, _ = model(J_b, use_jwc=True)
                gt_can = J_b[:, KEY_JOINT_IDX, :]
                loss, _ = compute_loss(logits, Y_b, p3d, J_b,
                                       pcan, gt_can, phase="finetune")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl      += loss.item()
                correct += (logits.argmax(-1) == Y_b).sum().item()
                total   += Y_b.size(0)
            sched.step()
            tr_acc = 100.0 * correct / total if total else 0
            va_acc = evaluate(model, va_ld, device)
            pct    = 65 + int(30 * (ep + 1) / EPOCHS_FT)
            _upd(pct,
                 f"Phase 2 [{ep+1}/{EPOCHS_FT}] train={tr_acc:.1f}% val={va_acc:.1f}%",
                 accuracy=round(va_acc, 1), phase="finetune")

            if va_acc > best:
                best = va_acc
                torch.save({"model_state": model.state_dict(),
                            "val_acc": best, "num_classes": nc},
                           str(WEIGHTS_PATH))
                log.info("  Saved best model val_acc=%.1f%%", best)

        # ── Build prototypes w_y and Q̃c ────────────────────────────────────
        _upd(97, "Building support prototypes…")
        proto, can = build_prototypes(model, tr_ld, device, nc)
        ck         = torch.load(str(WEIGHTS_PATH), map_location="cpu")
        ck["prototypes"]  = proto.cpu()
        ck["support_can"] = can.cpu()
        torch.save(ck, str(WEIGHTS_PATH))

        STATUS = {"running": False, "progress": 100,
                  "message": f"Training complete! Best val acc: {best:.1f}%",
                  "accuracy": round(best, 1), "phase": "done"}
        return True

    except Exception as e:
        log.exception("Training failed")
        STATUS = {"running": False, "progress": 0,
                  "message": f"Error: {e}", "accuracy": 0.0, "phase": "error"}
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")
    run_training()