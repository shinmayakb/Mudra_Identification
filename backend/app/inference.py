"""
inference.py
============
JCNN inference engine.
Pipeline: base64 image → MediaPipe → (42,3) joints → JCNN → prediction dict
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.hand_detector import HandDetector, decode_b64_to_bgr
from model.jcnn_model  import JCNN, KEY_JOINT_IDX

log = logging.getLogger(__name__)

WEIGHTS_PATH = Path("/app/weights/mudra_model.pth")
CLASSES_PATH = Path("/app/weights/classes.json")


class MudraPredictor:
    def __init__(self):
        self.model:  JCNN | None = None
        self.names:  list        = []
        self.device              = torch.device("cpu")
        self.detector            = HandDetector(static_mode=True, max_hands=2)
        self._load()

    def _load(self):
        if not WEIGHTS_PATH.exists() or not CLASSES_PATH.exists():
            log.warning("Model weights not found — run training first.")
            return
        with open(CLASSES_PATH) as f:
            self.names = json.load(f)
        ck  = torch.load(str(WEIGHTS_PATH), map_location="cpu")
        m   = JCNN(num_classes=len(self.names))
        m.load_state_dict(ck["model_state"])
        if "prototypes" in ck:
            m.jwc.prototypes.data   = ck["prototypes"]
        if "support_can" in ck:
            m.jwc.support_can.data  = ck["support_can"]
        m.eval()
        self.model = m
        log.info("Model loaded — %d classes, val_acc=%.1f%%",
                 len(self.names), ck.get("val_acc", 0))

    def reload(self):
        self._load()

    @property
    def ready(self) -> bool:
        return self.model is not None and len(self.names) > 0

    def predict(self, b64: str) -> dict:
        if not self.ready:
            return {"error": "Model not trained. Click Train Model."}

        bgr = decode_b64_to_bgr(b64)
        det = self.detector.detect(bgr)

        if det.hand_count == 0:
            return {
                "hand_detected": False,
                "error":         "No hand detected. Ensure hand is visible.",
                "drawn_image":   det.drawn_image,
            }

        with torch.no_grad():
            J      = torch.tensor(det.landmarks).unsqueeze(0)
            logits, fvi, p_can, p3d, alpha = self.model(J, use_jwc=True)
            probs  = F.softmax(logits, dim=-1).squeeze(0).numpy()

        pred_idx  = int(np.argmax(probs))
        conf      = float(probs[pred_idx])
        top5_idx  = np.argsort(probs)[::-1][:5]
        top5      = [{"rank": i + 1,
                      "class_id": int(top5_idx[i]),
                      "name":     self.names[top5_idx[i]],
                      "confidence": float(probs[top5_idx[i]]),
                      "confidence_pct": round(float(probs[top5_idx[i]]) * 100, 2)}
                     for i in range(len(top5_idx))]

        return {
            "hand_detected":       True,
            "mudra":               self.names[pred_idx],
            "confidence":          round(conf, 4),
            "confidence_pct":      round(conf * 100, 2),
            "class_id":            pred_idx,
            "top5":                top5,
            "hand_count":          det.hand_count,
            "hand_labels":         det.hand_labels,
            "bbox":                det.bbox,
            "joints":              det.landmarks.tolist(),
            "canonical_joints":    p_can.squeeze(0).tolist(),
            "joint_angles":        alpha.squeeze(0).tolist(),
            "drawn_image":         det.drawn_image,
        }
