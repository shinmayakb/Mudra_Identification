# Paper Equations — Where Each Is Implemented
## "Hand Gesture Recognition From an Open-Set Perspective" IEEE TMM 2025

---

## Eq(1) — JGCN Auxiliary Task
**Paper:** argmin_{θJGCN, θP} || ΘP(ΘJGCN(fl) = fvd) - P3D ||²

**What it means:** The Joint Graph Conv Net (JGCN) produces view-DEPENDENT
features fvd. To train JGCN, it must predict absolute 3D joint positions P3D.

**Where implemented:**
- `backend/model/jcnn_model.py` → class `JGCN` → method `forward()`
  - `self.block1`, `self.block2` = ΘJGCN (two residual SemGConv blocks)
  - `self.pos_dec` = ΘP (single SemGConv decoder)
  - Returns `fvd` and `p3d`
- `backend/train.py` → `compute_loss()` → `L3D = smooth_l1_loss(p3d_pred, joints)`
**VIVA Explanation :** Eq(1) trains the JGCN to extract meaningful features by forcing it to reconstruct the original 3D joint positions.
The features produced are called view-dependent because they still vary with camera angle.

❓ Why do we allow view-dependent features?
Answer:
“Because we first learn full spatial structure accurately, then KGCN removes viewpoint effects in the next stage.”

---

## Eq(2) — KGCN Auxiliary Task (View-Independent)
**Paper:** argmin_{θKGCN, θA} || Φ(ΘA(ΘKGCN(fvd) = αK)|l) - Pc ||²

**What it means:** The Key Joint Graph Conv Net (KGCN) operates on only 15
key joints (MCP + PIP + DIP) per hand = 30 total. It produces view-INDEPENDENT
features fvi. Trained by predicting canonical 3D positions Pc via the hand
kinematic chain Φ.

**Where implemented:**
- `backend/model/jcnn_model.py` → class `KGCN` → method `forward()`
  - `self.key_idx` = selects 30 key joints from 42
  - `self.block1`, `self.block2` = ΘKGCN (two residual SemGConv blocks on KEY joints)
  - `self.angle_dec` = ΘA (single SemGConv, outputs joint angles αK)
  - `self.kinematic` = Φ (learnable kinematic chain, αK → Pc)
  - Returns `fvi`, `alpha` (αK), `p_can` (Pc)
- `backend/train.py` → `compute_loss()` → `Lcan = smooth_l1_loss(pcan_pred, gt_can)`

**VIVA Explanation :** KGCN converts view-dependent features into view-independent features by learning joint angles, which remain constant across different viewpoints.

❓ Why only MCP, PIP, DIP joints?
Answer:
“Because these joints define finger articulation, which is essential for gesture recognition and invariant to viewpoint.”

❓ Why not use all 42 joints?
Answer:
“Including wrist and tip positions introduces view dependency, so only key joints are used for invariance.”

---

## Eq(3) — Standard Cosine Similarity
**Paper:** õy = cos(wy, fvi) = Σ_{i∈K} w^T_{y,i} · fvi_i

**What it means:** Baseline: compare query features fvi with prototype w_y
using cosine similarity equally over all key joints.

**Where implemented:**
- `backend/model/jcnn_model.py` → class `JWC` → `forward()` → `use_jwc=False` branch
  - `F.cosine_similarity(fvi, proto.expand(B,-1,-1), dim=-1)` per class c

  **VIVA Explanation :** Eq(3) measures how similar the input gesture is to each class by comparing their feature vectors using cosine similarity across all joints equally

---

## Eq(4) — Prediction
**Paper:** ỹ = argmax_{i∈Y} õi

**What it means:** Select the class with the highest score.

**Where implemented:**
- `backend/app/inference.py` → `predict()` → `pred_idx = int(np.argmax(probs))`
- `backend/train.py` → `evaluate()` → `logits.argmax(-1)`

**VIVA Explanation :** Eq(4) selects the final predicted class by choosing the class with the highest similarity score computed from the previous stage.

❓ Where are features used here?
Answer:
“Features are used in Eq(3) and Eq(6) to compute scores. Eq(4) only selects the best class based on those scores.”
---

## Eq(5) — Dynamic Joint Weights (JWC core)
**Paper:** ω_{y,i} = |K| · exp(α·||p̃c_i - q̃c_i||²) / Σ_j exp(α·||p̃c_j - q̃c_j||²)

**What it means:** For each class y, compute how much each key joint i differs
from the support prototype in canonical space. Joints with larger spatial
differences get higher weights — emphasising discriminative joints.
α = 0.005 makes weights stay in range (0.9, 1.1).

**Where implemented:**
- `backend/model/jcnn_model.py` → class `JWC` → `forward()` → `use_jwc=True` branch:
  ```python
  diff  = ((p_can - q_can.unsqueeze(0)) ** 2).sum(-1)      # ||p̃c - q̃c||²
  omega = NUM_KEY_JOINTS * F.softmax(ALPHA_JWC * diff, -1)  # Eq(5)
  ```
  `ALPHA_JWC = 0.005` is defined at top of file.

  **VIVA Explanation :** Eq(5) assigns higher weights to joints that are more different from the prototype, highlighting important joints for classification.

❓ Why larger difference = higher weight?
Answer:
“Because joints that differ more carry more discriminative information for distinguishing between gestures.”
---

## Eq(6) — Joint-Weighted Cosine Similarity (JWC output)
**Paper:** õy = Σ_{i∈K} ω_{y,i} · w^T_{y,i} · fvi_i

**What it means:** Weighted sum of cosine similarities, emphasising joints
where query and support differ most.

**Where implemented:**
- `backend/model/jcnn_model.py` → class `JWC` → `forward()`:
  ```python
  cos_s = F.cosine_similarity(fvi, proto.expand(B,-1,-1), dim=-1)
  score = (omega * cos_s).sum(-1, keepdim=True)             # Eq(6)
  ```

  **VIVA Explanation :** Eq(6) computes the final classification score by combining cosine similarity with dynamic joint weights. Important joints contribute more, improving recognition accuracy

---

## Eq(7) — 2D Joint Position from Heatmap
**Paper:** p̃2D_i = Σ_h Σ_w m_{i,h,w} · a_{h,w}

**What it means:** In the paper's full FE module, a heatmap M weights image
patches to locate each joint in 2D.

**Where implemented:** Simplified in this implementation — MediaPipe provides
2D/3D joint coordinates directly (x, y, z), replacing the full ViT+DeConv
heatmap branch. The 3D coordinates from MediaPipe serve as fl directly.
The HeatmapBranch is omitted because MediaPipe already solves this task.

**VIVA Explanation :**

---

## Eq(8) — L2D Loss
**Paper:** L2D = Σ_{i∈J} smoothL1(p̃2D_i, p2D_i)

**What it means:** Supervises 2D joint heatmap generation.

**Where implemented:** λ1·L2D = 0 in this implementation because MediaPipe
replaces the heatmap branch. The loss weight LAMBDA1=2.0 is defined in
`jcnn_model.py` for completeness.

**VIVA Explanation :** 

---

## Eq(9) — L3D Loss
**Paper:** L3D = Σ_{i∈J} smoothL1(p̃3D_i, p3D_i)

**What it means:** Supervises JGCN to predict absolute 3D positions.

**Where implemented:**
- `backend/model/jcnn_model.py` → `compute_loss()`:
  ```python
  L3D = F.smooth_l1_loss(p3d_pred, p3d_gt)      # Eq(9)
  ```
- `backend/train.py` → training loop passes `p3d` (JGCN output) vs `joints` (MediaPipe ground truth)

---

## Eq(10) — Lcan Loss
**Paper:** Lcan = Σ_{i∈K} smoothL1(p̃c_i, pc_i)

**What it means:** Supervises KGCN to predict canonical 3D positions.
Uses only 30 KEY joints (K = MCP+PIP+DIP, both hands).

**Where implemented:**
- `backend/model/jcnn_model.py` → `compute_loss()`:
  ```python
  Lcan = F.smooth_l1_loss(pcan_pred, pcan_gt)    # Eq(10)
  ```
- `backend/train.py`:
  ```python
  gt_can = J_b[:, KEY_JOINT_IDX, :]              # ground truth for key joints
  loss, _ = compute_loss(..., pcan_pred=pcan, pcan_gt=gt_can, ...)
  ```

---

## Eq(11) — Lcls Loss
**Paper:** Lcls = -Σ_{i∈Y} log[exp(õi)/Σ_j exp(õj)] · oi

**What it means:** Standard cross-entropy classification loss over all classes.

**Where implemented:**
- `backend/model/jcnn_model.py` → `compute_loss()`:
  ```python
  Lcls = F.cross_entropy(logits, labels)          # Eq(11)
  ```

---

## Eq(12) — Total Loss
**Paper:** Ltotal = λ1·L2D + λ2·L3D + λ3·Lcan + λ4·Lcls
          λ1=2.0, λ2=2.5, λ3=2.5, λ4=1.0

**What it means:** Weighted sum of all losses. λ4=0 during Phase 1 (pretrain)
so only geometric losses train JGCN+KGCN. λ4=1.0 during Phase 2 adds
classification supervision.

**Where implemented:**
- `backend/model/jcnn_model.py` → constants at top:
  ```python
  LAMBDA1, LAMBDA2, LAMBDA3, LAMBDA4 = 2.0, 2.5, 2.5, 1.0
  ```
- `backend/model/jcnn_model.py` → `compute_loss()`:
  ```python
  lam4   = LAMBDA4 if phase == "finetune" else 0.0
  Ltotal = LAMBDA2 * L3D + LAMBDA3 * Lcan + lam4 * Lcls   # Eq(12)
  ```
- `backend/train.py`:
  - Phase 1 loop: `compute_loss(..., phase="pretrain")` → λ4=0
  - Phase 2 loop: `compute_loss(..., phase="finetune")` → λ4=1.0

---

## Graph Structure — Fig.3(a) and Fig.3(b)

**Fig.3(a) — JGCN graph: all 21 joints, hand skeleton topology**
**Where implemented:**
- `backend/model/jcnn_model.py` → `_build_hand_adj()`:
  - Wrist → MCP connections for all 5 fingers
  - Finger chain: MCP → PIP → DIP → TIP
  - Palm cross-connections between adjacent MCPs
- `ADJ_FULL = build_adjacency_matrix(42, edges_all)` — (42×42) for both hands

**Fig.3(b) — KGCN graph: 15 key joints per hand = 30 total**
**Where implemented:**
- `backend/model/jcnn_model.py` → `_edges_key` list:
  - Within-finger connections between MCP-PIP-DIP
  - Cross-finger MCP connections
- `ADJ_KEY = build_adjacency_matrix(30, _edges_key)` — (30×30) for both hands

**SemGConv (paper ref [45]):**
- `backend/model/jcnn_model.py` → class `SemGConv`:
  - `self.adj_fixed` = normalised adjacency D^{-1/2} A D^{-1/2}
  - `self.M` = learnable semantic adjacency (additive, paper ref [45])
  - Combined: `A = adj_fixed + M`, row-softmax normalised

---

## Hand Kinematic Chain — Fig.4
**Paper:** Φ(αK|l) — kinematic chain from joint angles αK and bone lengths l

**What it means:** Given joint angles of MCP/PIP/DIP, compute canonical 3D
positions independent of wrist rotation/translation.

**Where implemented:**
- `backend/model/jcnn_model.py` → class `KGCN` → `self.kinematic`:
  ```python
  self.kinematic = nn.Sequential(
      nn.Linear(NUM_KEY_JOINTS, NUM_KEY_JOINTS * 2),
      nn.ReLU(),
      nn.Linear(NUM_KEY_JOINTS * 2, NUM_KEY_JOINTS * 3),
  )
  ```
  Learnable approximation of the kinematic chain Φ.
  αK (B,30) → p_can (B,30,3)

---

## Two-Phase Training (paper Section IV-B-1)

| Phase | λ4 | What trains | Code |
|-------|----|-------------|------|
| Phase 1 (pretrain) | 0 | JGCN + KGCN geometry only | `train.py` epochs loop with `phase="pretrain"` |
| Phase 2 (finetune) | 1.0 | Full JCNN with JWC classification | `train.py` epochs loop with `phase="finetune"` |

AdamW optimizer, lr=0.0003, CosineAnnealingLR scheduler — all from paper Section IV-B.
