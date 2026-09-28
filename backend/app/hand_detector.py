"""
hand_detector.py
================
Dual-hand detector — MediaPipe version-safe import.
Detects up to 2 hands, returns (42,3) joint array,
draws skeleton + bounding boxes, returns base64 JPEG.

VERSION FIX: Uses direct sub-module import to work on
mediapipe 0.9.x, 0.10.x, and 0.10.14 (pinned in requirements).
"""

from __future__ import annotations
import base64
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# Version-safe MediaPipe import
import mediapipe.python.solutions.hands          as _mp_hands
import mediapipe.python.solutions.drawing_utils  as _mp_draw
import mediapipe.python.solutions.drawing_styles as _mp_styles

NUM_JOINTS_PER_HAND = 21
NUM_HANDS           = 2
TOTAL_JOINTS        = NUM_JOINTS_PER_HAND * NUM_HANDS  # 42

# Drawing constants
_COLOUR_LEFT   = (34, 197, 94)    # green
_COLOUR_RIGHT  = (59, 130, 246)   # blue
_COLOUR_CONN   = (156, 163, 175)  # grey
_COLOUR_BBOX   = (251, 146, 60)   # orange
_COLOUR_TEXT   = (255, 255, 255)

_SPEC_CONN = _mp_draw.DrawingSpec(color=_COLOUR_CONN, thickness=2)


@dataclass
class DetectionResult:
    landmarks:       np.ndarray   # (42, 3) float32 — both hands combined
    bbox:            dict         # {"x","y","w","h"}
    hand_count:      int          # 0, 1, 2
    hand_labels:     list         # ["Left","Right"]
    drawn_image:     str          # base64 JPEG with skeleton drawn
    joints_per_hand: list         # list of (21,3) arrays


class HandDetector:
    """
    Detects up to 2 hands.
    Returns (42,3) combined landmark array (zero-padded if < 2 hands).
    Draws MediaPipe skeleton with colour-coded Left/Right hands.
    """
    def __init__(self, static_mode=True, max_hands=2,
                 min_detect=0.5, min_track=0.5):
        self._hands = _mp_hands.Hands(
            static_image_mode        = static_mode,
            max_num_hands            = max_hands,
            min_detection_confidence = min_detect,
            min_tracking_confidence  = min_track,
        )

    def detect(self, bgr: np.ndarray) -> DetectionResult:
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res    = self._hands.process(rgb)
        canvas = bgr.copy()

        joints_per_hand = []
        hand_labels     = []
        bboxes          = []

        if res.multi_hand_landmarks:
            for lm, hi in zip(res.multi_hand_landmarks,
                               res.multi_handedness):
                label  = hi.classification[0].label   # "Left" / "Right"
                hand_labels.append(label)

                arr = np.array([[p.x, p.y, p.z] for p in lm.landmark],
                               dtype=np.float32)      # (21,3)
                joints_per_hand.append(arr)

                colour  = _COLOUR_LEFT if label == "Left" else _COLOUR_RIGHT
                spec_lm = _mp_draw.DrawingSpec(
                    color=colour, thickness=2, circle_radius=4)
                _mp_draw.draw_landmarks(
                    canvas, lm, _mp_hands.HAND_CONNECTIONS,
                    spec_lm, _SPEC_CONN)

                h, w = bgr.shape[:2]
                xs   = [p.x * w for p in lm.landmark]
                ys   = [p.y * h for p in lm.landmark]
                x1   = max(0, int(min(xs)) - 18)
                y1   = max(0, int(min(ys)) - 18)
                x2   = min(w, int(max(xs)) + 18)
                y2   = min(h, int(max(ys)) + 18)
                bboxes.append({"x": x1, "y": y1,
                               "w": x2 - x1, "h": y2 - y1})
                cv2.rectangle(canvas, (x1, y1), (x2, y2), _COLOUR_BBOX, 2)
                cv2.putText(canvas, label, (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            _COLOUR_TEXT, 2, cv2.LINE_AA)

        combined = self._combine(joints_per_hand)
        merged   = self._merge_bboxes(bboxes)
        img_b64  = _encode_b64(canvas)

        return DetectionResult(
            landmarks       = combined,
            bbox            = merged,
            hand_count      = len(joints_per_hand),
            hand_labels     = hand_labels,
            drawn_image     = img_b64,
            joints_per_hand = joints_per_hand,
        )

    def close(self):
        self._hands.close()

    @staticmethod
    def _combine(per_hand):
        pad = np.zeros((NUM_JOINTS_PER_HAND, 3), dtype=np.float32)
        while len(per_hand) < NUM_HANDS:
            per_hand.append(pad.copy())
        return np.concatenate(per_hand[:NUM_HANDS], axis=0)

    @staticmethod
    def _merge_bboxes(boxes):
        if not boxes:
            return {"x": 0, "y": 0, "w": 0, "h": 0}
        x1 = min(b["x"] for b in boxes)
        y1 = min(b["y"] for b in boxes)
        x2 = max(b["x"] + b["w"] for b in boxes)
        y2 = max(b["y"] + b["h"] for b in boxes)
        return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


# ─── helpers ─────────────────────────────────────────────────────────────────

def _encode_b64(bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def decode_b64_to_bgr(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    data = base64.b64decode(b64)
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
