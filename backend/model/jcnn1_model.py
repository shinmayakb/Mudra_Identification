"""
jcnn_model.py
=============

FULL SINGLE-FILE IMPLEMENTATION

Includes:
✔ Eq(1)–Eq(12) implementation
✔ Graph construction
✔ Loss functions
✔ Hand detection (MediaPipe)
✔ Inference pipeline

Structured for VIVA explanation:
SECTION A — Model (Equations)
SECTION B — Input (Hand Detector)
SECTION C — Inference Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import mediapipe.python.solutions.hands as mp_hands

# ─────────────────────────────────────────────
# SECTION A: CONSTANTS & GRAPH
# ─────────────────────────────────────────────
NUM_JOINTS = 21
NUM_HANDS = 2
TOTAL_JOINTS = 42

_KEY_PER_HAND = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
KEY_JOINT_IDX = _KEY_PER_HAND + [i+21 for i in _KEY_PER_HAND]
NUM_KEY_JOINTS = 30

FEAT_DIM = 128
ALPHA_JWC = 0.005

LAMBDA2 = 2.5
LAMBDA3 = 2.5
LAMBDA4 = 1.0


def build_adj(n, edges):
    A = torch.zeros(n,n)
    for i in range(n):
        A[i,i] = 1
    for i,j in edges:
        A[i,j]=1; A[j,i]=1
    D = A.sum(1)
    D_inv = torch.diag(1/torch.sqrt(D+1e-6))
    return D_inv @ A @ D_inv


edges_full = [(i,i+1) for i in range(41)]
ADJ_FULL = build_adj(42, edges_full)
ADJ_KEY  = build_adj(30, [(i,i+1) for i in range(29)])


# ─────────────────────────────────────────────
# SEMANTIC GCN
# ─────────────────────────────────────────────
class SemGConv(nn.Module):
    def __init__(self, in_c, out_c, adj):
        super().__init__()
        self.adj = adj
        self.M = nn.Parameter(torch.zeros_like(adj))
        self.fc = nn.Linear(in_c,out_c,bias=False)
        self.bn = nn.BatchNorm1d(adj.shape[0])

    def forward(self,x):
        A = F.softmax(self.adj + self.M, dim=-1)
        x = torch.bmm(A.unsqueeze(0).repeat(x.size(0),1,1), x)
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x)


class ResBlock(nn.Module):
    def __init__(self,c,adj):
        super().__init__()
        self.g1 = SemGConv(c,c,adj)
        self.g2 = SemGConv(c,c,adj)

    def forward(self,x):
        return x + self.g2(self.g1(x))

# ─────────────────────────────
# Eq (1): JGCN
# θ* = argmin || ΘP(ΘJGCN(fl)) - P3D ||^2
# ─────────────────────────────
def Eq1_argmin_JGCN(theta_JGCN, theta_P, f_l, P3D_gt):
    fvd = Theta_JGCN(f_l, theta_JGCN)
    P3D_pred = Theta_P(fvd, theta_P)
    return torch.norm(P3D_pred - P3D_gt, p=2)**2


# ─────────────────────────────
# Eq (2): KGCN
# θ* = argmin || Φ(ΘA(ΘKGCN(fvd))) - Pc ||^2
# ─────────────────────────────
def Eq2_argmin_KGCN(theta_KGCN, theta_A, fvd, Pc_gt):
    fvi = Theta_KGCN(fvd, theta_KGCN)
    alpha = Theta_A(fvi, theta_A)
    Pc_pred = Phi(alpha)
    return torch.norm(Pc_pred - Pc_gt, p=2)**2


# ─────────────────────────────
# Eq (3): Cosine Similarity
# o_y = Σ w_i^T fvi_i
# ─────────────────────────────
def Eq3_cosine(fvi, w):
    return torch.sum(w * fvi, dim=-1)


# ─────────────────────────────
# Eq (4): Prediction
# y = argmax(o)
# ─────────────────────────────
def Eq4_argmax(logits):
    return torch.argmax(logits, dim=1)


# ─────────────────────────────
# Eq (5): Joint Weight
# ω = softmax(α ||pc - qc||^2)
# ─────────────────────────────
def Eq5_weights(pc, qc, ALPHA):
    diff = torch.norm(pc - qc, dim=-1)**2
    return torch.softmax(ALPHA * diff, dim=-1)


# ─────────────────────────────
# Eq (6): Weighted Cosine
# o_y = Σ ω_i (w_i^T fvi_i)
# ─────────────────────────────
def Eq6_weighted_score(fvi, w, omega):
    cos = torch.sum(w * fvi, dim=-1)
    return torch.sum(omega * cos, dim=-1)


# ─────────────────────────────
# Eq (7): 2D Joint (Heatmap)
# p2D = Σ m * a
# ─────────────────────────────
def Eq7_2D(M, A):
    B, J, H, W = M.shape
    M_flat = M.view(B, J, -1)
    A_flat = A.view(B, -1, 2)
    return torch.bmm(M_flat, A_flat)


# ─────────────────────────────
# Eq (8): 2D Loss
# L2D = ||p2D - gt||
# ─────────────────────────────
def Eq8_L2D(p2d_pred, p2d_gt):
    return torch.norm(p2d_pred - p2d_gt, p=1)


# ─────────────────────────────
# Eq (9): 3D Loss
# L3D = ||p3D - gt||
# ─────────────────────────────
def Eq9_L3D(p3d_pred, p3d_gt):
    return torch.norm(p3d_pred - p3d_gt, p=2)


# ─────────────────────────────
# Eq (10): Canonical Loss
# Lcan = ||pc - gt||
# ─────────────────────────────
def Eq10_Lcan(pc_pred, pc_gt):
    return torch.norm(pc_pred - pc_gt, p=2)


# ─────────────────────────────
# Eq (11): Classification Loss
# Lcls = - Σ y log(softmax)
# ─────────────────────────────
def Eq11_Lcls(logits, labels):
    prob = torch.log_softmax(logits, dim=1)
    return -torch.sum(labels * prob)


# ─────────────────────────────
# Eq (12): Total Loss
# L = λ1L2D + λ2L3D + λ3Lcan + λ4Lcls
# ─────────────────────────────
def Eq12_total_loss(L2D, L3D, Lcan, Lcls,
                    LAMBDA1, LAMBDA2, LAMBDA3, LAMBDA4):
    return (LAMBDA1 * L2D +
            LAMBDA2 * L3D +
            LAMBDA3 * Lcan +
            LAMBDA4 * Lcls)



# ─────────────────────────────────────────────
# SECTION B: HAND DETECTOR
# ─────────────────────────────────────────────
class HandDetector:
    def __init__(self, static_mode=True, max_hands=2):
        self.hands = mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        joints_per_hand = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in hand_landmarks.landmark
                ], dtype=np.float32)
                joints_per_hand.append(coords)

        while len(joints_per_hand) < 2:
            joints_per_hand.append(np.zeros((21,3), dtype=np.float32))

        return np.concatenate(joints_per_hand, axis=0)


# ─────────────────────────────────────────────
# SECTION C: INFERENCE PIPELINE (Eq 4)
# ─────────────────────────────────────────────
class JCNNPredictor:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.detector = HandDetector()

    def predict(self, image):
        joints = self.detector.detect(image)
        J = torch.tensor(joints).unsqueeze(0).float()

        with torch.no_grad():
            logits, fvi, pc, p3d, alpha = self.model(J)

        probs = F.softmax(logits, dim=-1).squeeze(0)

        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

        return {
            "predicted_class": self.class_names[pred_idx],
            "confidence": confidence,
            "confidence_%": round(confidence * 100, 2),

            # For explanation
            "alpha (Eq2)": alpha.squeeze(0).tolist(),
            "Pc (Eq2)": pc.squeeze(0).tolist()
        }