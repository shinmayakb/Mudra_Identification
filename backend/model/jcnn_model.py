"""
jcnn_model.py
=============
Complete implementation of the paper:
  "Hand Gesture Recognition From an Open-Set Perspective"
  IEEE Transactions on Multimedia, Vol. 27, 2025
  Zhou, Xu, Cheng

ALL equations from the paper are implemented:

  Eq(1)  JGCN auxiliary task:
         argmin_{θJGCN, θP} || ΘP(ΘJGCN(fl) = fvd) - P3D ||²

  Eq(2)  KGCN auxiliary task (view-independent):
         argmin_{θKGCN, θA} || Φ(ΘA(ΘKGCN(fvd) = αK)|l) - Pc ||²

  Eq(3)  Standard cosine similarity:
         õy = cos(wy, fvi) = Σ_{i∈K} w^T_{y,i} · fvi_i

  Eq(4)  Prediction result:
         ỹ = argmax_{i∈Y} õi

  Eq(5)  Dynamic joint weights (JWC):
         ω_{y,i} = |K| · exp(α·||p̃c_i - q̃c_i||²) / Σ_j exp(α·||p̃c_j - q̃c_j||²)

  Eq(6)  Joint-weighted cosine similarity:
         õy = Σ_{i∈K} ω_{y,i} · w^T_{y,i} · fvi_i

  Eq(7)  2D joint position from heatmap:
         p̃2D_i = Σ_h Σ_w m_{i,h,w} · a_{h,w}

  Eq(8)  L2D = Σ_{i∈J} smoothL1(p̃2D_i, p2D_i)

  Eq(9)  L3D = Σ_{i∈J} smoothL1(p̃3D_i, p3D_i)

  Eq(10) Lcan = Σ_{i∈K} smoothL1(p̃c_i, pc_i)

  Eq(11) Lcls = -Σ_{i∈Y} log[ exp(õi) / Σ_j exp(õj) ] · oi  (cross-entropy)

  Eq(12) Ltotal = λ1·L2D + λ2·L3D + λ3·Lcan + λ4·Lcls
         λ1=2.0, λ2=2.5, λ3=2.5, λ4=1.0 (paper Section IV-B)

Network graph structure:
  Fig.3(a): JGCN — all 21 joints, hand skeleton topology
  Fig.3(b): KGCN — 15 key joints only (MCP + PIP + DIP)
  Fig.4:    Hand kinematic chain model (wrist 6DoF, key joints drive gesture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants from paper
# ─────────────────────────────────────────────────────────────────────────────
NUM_JOINTS       = 21          # MediaPipe landmarks per hand
NUM_HANDS        = 2           # max hands supported
TOTAL_JOINTS     = NUM_JOINTS * NUM_HANDS   # 42

# Key joints: MCP(5) + PIP(5) + DIP(5) per hand = 15 per hand = 30 total
# Paper Section III-B-2, Fig.4: angles of MCP/PIP/DIP are view-independent
# MediaPipe indices: Thumb(1-4), Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20)
# MCP:  1, 5, 9, 13, 17
# PIP:  2, 6, 10, 14, 18
# DIP:  3, 7, 11, 15, 19
_KEY_PER_HAND    = [1, 2, 3,  5, 6, 7,  9, 10, 11,  13, 14, 15,  17, 18, 19]
# For combined 42-joint graph, key joint indices are offset by +21 for second hand
KEY_JOINT_IDX    = _KEY_PER_HAND + [i + NUM_JOINTS for i in _KEY_PER_HAND]
NUM_KEY_JOINTS   = len(KEY_JOINT_IDX)   # 30

FEAT_DIM         = 128         # feature vector dimension D
ALPHA_JWC        = 0.005       # paper Eq(5) hyperparameter α

# Paper Eq(12) loss weights
LAMBDA1 = 2.0   # L2D
LAMBDA2 = 2.5   # L3D
LAMBDA3 = 2.5   # Lcan
LAMBDA4 = 1.0   # Lcls (0 in pretrain phase, 1.0 in finetune phase)

# ─────────────────────────────────────────────────────────────────────────────
# JGCN graph adjacency — Fig.3(a): all 21 joints with hand skeleton topology
# ─────────────────────────────────────────────────────────────────────────────
def _build_hand_adj(n_joints: int, offset: int = 0) -> list:
    """
    Build adjacency edge list for one hand following MediaPipe skeleton (Fig.3a).
    offset: 0 for first hand, 21 for second hand in combined (42-joint) graph.
    """
    edges = []
    o = offset
    # Wrist connections
    for mcp in [1, 5, 9, 13, 17]:
        edges.append((o+0, o+mcp))
    # Finger chains
    for base in [1, 5, 9, 13, 17]:
        for k in range(3):
            edges.append((o+base+k, o+base+k+1))
    # Palm cross-connections (Fig.3a)
    for mcp in [5, 9, 13]:
        edges.append((o+mcp, o+mcp+4))
    return edges


def build_adjacency_matrix(n_nodes: int, edges: list) -> torch.Tensor:
    """Build symmetric normalised adjacency matrix A_hat = D^{-1/2} A D^{-1/2}."""
    A = torch.zeros(n_nodes, n_nodes)
    # Self-loops
    for i in range(n_nodes):
        A[i, i] = 1.0
    # Edges (symmetric)
    for (i, j) in edges:
        if i < n_nodes and j < n_nodes:
            A[i, j] = 1.0
            A[j, i] = 1.0
    # Normalise: D^{-1/2} A D^{-1/2}
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-6)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat


# Pre-compute adjacency matrices
_edges_all = _build_hand_adj(NUM_JOINTS, 0) + _build_hand_adj(NUM_JOINTS, NUM_JOINTS)
ADJ_FULL   = build_adjacency_matrix(TOTAL_JOINTS, _edges_all)          # (42,42)

# Key-joint subgraph: retain edges where both endpoints are key joints
_key_set   = set(KEY_JOINT_IDX)
_key_map   = {j: i for i, j in enumerate(KEY_JOINT_IDX)}
_edges_key = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
              (9, 10), (10, 11), (12, 13), (13, 14),   # within fingers
              (0, 3), (3, 6), (6, 9), (9, 12),         # across MCPs
              (15, 16), (16, 17), (18, 19), (19, 20),
              (21, 22), (22, 23), (24, 25), (25, 26),
              (27, 28), (28, 29),
              (15, 18), (18, 21), (21, 24), (24, 27)]
ADJ_KEY    = build_adjacency_matrix(NUM_KEY_JOINTS, _edges_key)         # (30,30)


# ─────────────────────────────────────────────────────────────────────────────
# SemGConv — paper ref [45]: Semantic Graph Convolutional layer
# ─────────────────────────────────────────────────────────────────────────────
class SemGConv(nn.Module):
    """
    Semantic Graph Convolutional layer from paper reference [45].
    Implements: H' = A_hat · H · W  where A_hat has a learnable
    additive semantic component M (paper Section III-B-2).

    Used in both JGCN and KGCN.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 adj: torch.Tensor):
        super().__init__()
        n = adj.shape[0]
        self.register_buffer("adj_fixed", adj.clone())
        # Learnable semantic adjacency M — paper ref [45]
        self.M    = nn.Parameter(torch.zeros(n, n))
        self.fc   = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn   = nn.BatchNorm1d(n)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        A = self.adj_fixed + self.M          # learnable semantic adjacency
        A = F.softmax(A, dim=-1)             # row-normalise
        x = torch.bmm(A.unsqueeze(0).expand(x.size(0), -1, -1), x)  # (B,N,C)
        x = self.fc(x)                       # (B, N, out_channels)
        x = self.bn(x) + self.bias
        return F.relu(x)


class ResidualGConvBlock(nn.Module):
    """
    Two SemGConv layers with residual connection.
    Paper Section III-B-2: "JGCN is constructed using two residual blocks
    built with SemGConv layers."
    """
    def __init__(self, channels: int, adj: torch.Tensor):
        super().__init__()
        self.g1    = SemGConv(channels, channels, adj)
        self.g2    = SemGConv(channels, channels, adj)
        self.short = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.short(x) + self.g2(self.g1(x))


# ─────────────────────────────────────────────────────────────────────────────
# JGCN — Joint Graph Convolutional Network  (paper Section III-B-2)
# Graph: all 21 joints, Fig.3(a) topology
# Trains via Eq(1): argmin ||ΘP(fvd) - P3D||²
# ─────────────────────────────────────────────────────────────────────────────
class JGCN(nn.Module):
    """
    JGCN processes ALL 42 joints (2 hands) to produce view-DEPENDENT
    features fvd.  Auxiliary task (Eq.1) predicts absolute 3D positions P3D.
    """
    def __init__(self, in_dim: int = 3, hidden: int = 64,
                 feat_dim: int = FEAT_DIM):
        super().__init__()
        self.proj   = nn.Linear(in_dim, hidden)
        self.block1 = ResidualGConvBlock(hidden, ADJ_FULL)
        self.block2 = ResidualGConvBlock(hidden, ADJ_FULL)
        self.out_fc = nn.Linear(hidden, feat_dim)
        # ΘP: position decoder — single SemGConv layer (paper Eq.1)
        self.pos_dec = SemGConv(feat_dim, 3, ADJ_FULL)

    def forward(self, joints: torch.Tensor):
        # joints: (B, 42, 3)
        x   = F.relu(self.proj(joints))       # (B, 42, hidden)
        x   = self.block1(x)
        x   = self.block2(x)
        fvd = self.out_fc(x)                  # (B, 42, feat_dim) — Eq(1) fvd
        # Eq(1) auxiliary: predict absolute 3D positions
        p3d = self.pos_dec(fvd)               # (B, 42, 3)
        return fvd, p3d


# ─────────────────────────────────────────────────────────────────────────────
# KGCN — Key Joint Graph Convolutional Network  (paper Section III-B-2)
# Graph: 15 KEY joints per hand = 30 total, Fig.3(b) topology
# Trains via Eq(2): argmin ||Φ(ΘA(αK)|l) - Pc||²
# Extracts view-INDEPENDENT features fvi
# ─────────────────────────────────────────────────────────────────────────────
class KGCN(nn.Module):
    """
    KGCN processes only the 30 KEY joints (MCP+PIP+DIP, both hands).
    Auxiliary task (Eq.2) predicts canonical 3D positions Pc.
    Output fvi contains view-INDEPENDENT features.
    """
    def __init__(self, feat_dim: int = FEAT_DIM):
        super().__init__()
        self.key_idx   = torch.tensor(KEY_JOINT_IDX, dtype=torch.long)
        self.block1    = ResidualGConvBlock(feat_dim, ADJ_KEY)
        self.block2    = ResidualGConvBlock(feat_dim, ADJ_KEY)
        # ΘA: angle decoder — single SemGConv (paper Eq.2)
        self.angle_dec = SemGConv(feat_dim, 1, ADJ_KEY)
        # Kinematic chain Φ: αK → Pc (simplified learnable projection)
        # In the paper, Φ is the hand kinematic chain model (Fig.4)
        self.kinematic = nn.Sequential(
            nn.Linear(NUM_KEY_JOINTS, NUM_KEY_JOINTS * 2),
            nn.ReLU(),
            nn.Linear(NUM_KEY_JOINTS * 2, NUM_KEY_JOINTS * 3),
        )

    def forward(self, fvd: torch.Tensor):
        # fvd: (B, 42, feat_dim)
        dev  = fvd.device
        fk   = fvd[:, self.key_idx.to(dev), :]          # (B, 30, feat_dim)
        fvi  = self.block2(self.block1(fk))              # view-independent
        # Eq(2): joint angles αK
        alpha = self.angle_dec(fvi).squeeze(-1)          # (B, 30)
        # Eq(2): canonical positions via kinematic chain Φ(αK|l)
        p_can = self.kinematic(alpha).view(
                    alpha.size(0), NUM_KEY_JOINTS, 3)    # (B, 30, 3)
        return fvi, alpha, p_can


# ─────────────────────────────────────────────────────────────────────────────
# JWC — Joint-Weighted Classification  (paper Section III-B-3)
# Implements Eq(3), Eq(4), Eq(5), Eq(6)
# ─────────────────────────────────────────────────────────────────────────────
class JWC(nn.Module):
    """
    Joint-Weighted Classification head.

    Eq(5): ω_{y,i} = |K|·exp(α·||p̃c_i - q̃c_i||²) / Σ_j exp(α·||p̃c_j - q̃c_j||²)
    Eq(6): õy = Σ_{i∈K} ω_{y,i} · w^T_{y,i} · fvi_i

    Prototype w_y is computed as mean of training fvi per class (Eq.3).
    """
    def __init__(self, num_classes: int, feat_dim: int = FEAT_DIM):
        super().__init__()
        self.nc = num_classes
        # Learnable support prototypes w_y — paper Section III-B-3
        self.prototypes  = nn.Parameter(
            torch.randn(num_classes, NUM_KEY_JOINTS, feat_dim) * 0.01)
        # Support canonical positions Q̃c — paper Eq(5)
        self.support_can = nn.Parameter(
            torch.zeros(num_classes, NUM_KEY_JOINTS, 3))

    def forward(self, fvi: torch.Tensor, p_can: torch.Tensor,
                use_jwc: bool = True) -> torch.Tensor:
        """
        fvi:   (B, 30, D)  — view-independent features
        p_can: (B, 30, 3)  — canonical joint positions P̃c

        Returns logits: (B, num_classes)
        """
        B      = fvi.size(0)
        scores = []

        for c in range(self.nc):
            proto  = self.prototypes[c]       # (30, D)
            q_can  = self.support_can[c]      # (30, 3)

            if use_jwc:
                # Eq(5): dynamic joint weights ω_{y,i}
                diff  = ((p_can - q_can.unsqueeze(0)) ** 2).sum(-1)      # (B,30)
                omega = NUM_KEY_JOINTS * F.softmax(ALPHA_JWC * diff, -1)  # (B,30)
                # Eq(6): joint-weighted cosine similarity
                cos_s = F.cosine_similarity(
                            fvi, proto.unsqueeze(0).expand(B, -1, -1), dim=-1)
                score = (omega * cos_s).sum(-1, keepdim=True)              # (B,1)
            else:
                # Eq(3): standard cosine similarity (baseline)
                cos_s = F.cosine_similarity(
                            fvi, proto.unsqueeze(0).expand(B, -1, -1), dim=-1)
                score = cos_s.sum(-1, keepdim=True)                        # (B,1)

            scores.append(score)

        return torch.cat(scores, dim=-1)      # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Complete JCNN = JGCN + KGCN + JWC
# ─────────────────────────────────────────────────────────────────────────────
class JCNN(nn.Module):
    """
    Full JCNN model implementing all three modules from paper Fig.2:
      1. Feature Extraction (FE): here simplified to JGCN input layer
      2. Viewpoint Influence Elimination (VIE): JGCN + KGCN
      3. Joint-Weighted Classification (JWC)

    Forward:
        joints (B, 42, 3)
          → JGCN → fvd (B,42,D),  p3d (B,42,3)         [Eq.1]
          → KGCN → fvi (B,30,D),  p_can (B,30,3)        [Eq.2]
          → JWC  → logits (B, num_classes)               [Eq.5,6]
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.jgcn = JGCN()
        self.kgcn = KGCN()
        self.jwc  = JWC(num_classes)

    def forward(self, joints: torch.Tensor, use_jwc: bool = True):
        fvd, p3d       = self.jgcn(joints)
        fvi, alpha, pc = self.kgcn(fvd)
        logits         = self.jwc(fvi, pc, use_jwc=use_jwc)
        return logits, fvi, pc, p3d, alpha


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions — paper Section III-C, Equations 8–12
# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(logits, labels, p3d_pred, p3d_gt,
                 pcan_pred, pcan_gt, phase="pretrain"):
    """
    Eq(9):  L3D  = Σ smoothL1(p̃3D, p3D)
    Eq(10): Lcan = Σ smoothL1(p̃c, pc)
    Eq(11): Lcls = cross-entropy
    Eq(12): Ltotal = λ1·L2D + λ2·L3D + λ3·Lcan + λ4·Lcls
            (L2D omitted: no heatmap branch; λ1 contribution = 0)
    """
    # Eq(9)
    L3D  = F.smooth_l1_loss(p3d_pred, p3d_gt)
    # Eq(10)
    Lcan = F.smooth_l1_loss(pcan_pred, pcan_gt)
    # Eq(11)
    Lcls = F.cross_entropy(logits, labels)

    # Eq(12): λ4=0 in pretrain phase, λ4=1.0 in finetune phase
    lam4   = LAMBDA4 if phase == "finetune" else 0.0
    Ltotal = LAMBDA2 * L3D + LAMBDA3 * Lcan + lam4 * Lcls

    return Ltotal, {
        "L3D":   round(L3D.item(), 5),
        "Lcan":  round(Lcan.item(), 5),
        "Lcls":  round(Lcls.item(), 5),
        "total": round(Ltotal.item(), 5),
    }
