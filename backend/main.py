"""
main.py — FastAPI backend for Mudra Recognition
Endpoints:
  GET  /health
  POST /predict
  POST /train
  GET  /train/status
  GET  /classes
"""
from __future__ import annotations
import json
import logging
import sys
import threading
from pathlib import Path
# Add /app to sys.path for sub-module imports
sys.path.insert(0, "/app")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.inference import MudraPredictor
from train        import run_training, STATUS

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Mudra Recognition API", version="3.0")
# Allows frontend apps to call backend
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize predictor (loads model weights if available)
predictor = MudraPredictor()

# ── request/response models ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    image:  str
    source: str = "upload"

# ── health check endpoint ─────────────────────────────────────────────────
@app.get("/health")
def health():
    classes_path = Path("/app/weights/classes.json")
    dataset_path = Path("/app/dataset")
    names: list  = []
    if classes_path.exists():
        with open(classes_path) as f:
            names = json.load(f)
    n_ds = sum(1 for d in dataset_path.iterdir()
               if d.is_dir() and not d.name.startswith(".")) \
           if dataset_path.exists() else 0
    return {
        "status":          "ok",
        "model_ready":     predictor.ready,
        "num_classes":     len(names),
        "class_names":     names,
        "dataset_exists":  dataset_path.exists(),
        "dataset_classes": n_ds,
        "training_status": STATUS,
    }

# ── prediction endpoint ─────────────────────────────────────────────────
@app.post("/predict")
def predict(req: PredictRequest):
    return predictor.predict(req.image)

# ── training endpoint ───────────────────────────────────────────────────────
@app.post("/train")
def train():
    if STATUS.get("running"):
        return {"message": "Already running", "status": STATUS}

    def worker():
        run_training()
        predictor.reload()

    threading.Thread(target=worker, daemon=True).start()
    return {"message": "Training started"}

# ── training status endpoint ───────────────────────────────────────────────
@app.get("/train/status")
def train_status():
    return STATUS

# ── utility endpoints ────────────────────────────────────────────────────────
@app.get("/classes")
def classes():
    p = Path("/app/weights/classes.json")
    if not p.exists():
        return {"classes": [], "total": 0}
    with open(p) as f:
        names = json.load(f)
    return {"classes": names, "total": len(names)}
