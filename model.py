"""
model.py

HMSG-Net: Heterogeneous Multi-modal Symptom-Guided Graph Network
================================================================

Two-modal variant: Text (BERT, 768-dim) + Audio (wav2vec, 768-dim).

Key components
--------------
FocalLoss          – handles severe class imbalance
SymptomRoutedRGAT  – the novel SR-RGAT layer
HMSGNet            – full model (encoder → SR-RGAT × 3 → readout → fusion → heads)
compute_loss       – combines focal + BCE + SmoothL1
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add, scatter_softmax

# ─────────────────────────────────────────────────────────────────────────────
# Architecture hyper-parameters (can be overridden when constructing HMSGNet)
# ─────────────────────────────────────────────────────────────────────────────

TEXT_DIM   = 768   # BERT
AUDIO_DIM  = 768   # wav2vec2
UNIFIED_DIM = TEXT_DIM + AUDIO_DIM  # 1536

NUM_EDGE_TYPES = 4   # T→T, A→A, T→A, A→T
NUM_NODE_TYPES = 2   # 0=text, 1=audio
NUM_SYMPTOMS   = 8   # PHQ-8 items

HIDDEN_DIM     = 256
NUM_GNN_LAYERS = 3
DROPOUT        = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary Focal Loss.

    Parameters
    ----------
    alpha : float
        Balancing factor for the positive class (set < 0.5 to down-weight
        the majority negative class).  Default 0.75 (down-weight negatives).
    gamma : float
        Focusing exponent. Default 2.0.
    reduction : str
        'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.reduction       = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Parameters
        ----------
        logits  : (B,) raw logits
        targets : (B,) binary float {0, 1}
        """
        targets = targets.float()
        probs   = torch.sigmoid(logits)

        if self.label_smoothing > 0.0:
            smooth_t = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smooth_t = targets
        bce     = F.binary_cross_entropy_with_logits(
            logits, smooth_t, reduction="none"
        )
        p_t     = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss    = alpha_t * (1.0 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xavier_linear(in_f: int, out_f: int, bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_f, out_f, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


class _MLP(nn.Module):
    """Two-layer MLP with ReLU + optional dropout."""

    def __init__(self, in_f: int, hidden_f: int, out_f: int,
                 dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            _xavier_linear(in_f, hidden_f),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            _xavier_linear(hidden_f, out_f),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# SR-RGAT: Symptom-Routed Relational Graph Attention
# ─────────────────────────────────────────────────────────────────────────────

class SymptomRoutedRGAT(nn.Module):
    """
    One SR-RGAT layer.

    Forward pass
    ─────────────
    Given:
      h          : (N, H)           node features from previous layer
      edge_index : (2, E)           COO edge list
      edge_type  : (E,)             integer in 0..R-1 (R = num_edge_types)

    Returns:
      h_out      : (N, H)           updated node features
    """

    def __init__(
        self,
        hidden_dim:     int = HIDDEN_DIM,
        num_edge_types: int = NUM_EDGE_TYPES,
        num_symptoms:   int = NUM_SYMPTOMS,
        n_heads:        int = 4,
        dropout:        float = DROPOUT,
        drop_edge:      float = 0.3,
    ):
        super().__init__()
        self.H  = hidden_dim
        self.R  = num_edge_types
        self.K  = num_symptoms
        self.nh = n_heads
        self.drop_edge_p = drop_edge
        assert hidden_dim % n_heads == 0, \
            "hidden_dim must be divisible by n_heads"
        self.head_dim = hidden_dim // n_heads

        self.W_q = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        self.W_k = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        self.W_v = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        for w in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(w.view(num_edge_types * hidden_dim, hidden_dim))

        self.routing_mlp = _MLP(hidden_dim, hidden_dim, num_symptoms, dropout=dropout)

        self.sym_edge_logits = nn.Parameter(
            torch.empty(num_symptoms, num_edge_types)
        )
        nn.init.xavier_uniform_(self.sym_edge_logits)

        self.W_symptom = nn.ModuleList([
            _xavier_linear(hidden_dim, hidden_dim, bias=True)
            for _ in range(num_symptoms)
        ])

        self.cross_symptom_linear = _xavier_linear(
            num_symptoms * hidden_dim, hidden_dim
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(p=dropout)

        self._scale = math.sqrt(self.head_dim)

    def _relational_agg(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_type:  Tensor,
    ) -> Tensor:
        N, H = h.shape
        R    = self.R
        device = h.device

        agg_stack = torch.zeros(N, R, H, device=device, dtype=h.dtype)

        if edge_index.shape[1] == 0:
            return agg_stack

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        for t in range(R):
            mask = (edge_type == t)
            if not mask.any():
                continue

            e_src = src_idx[mask]
            e_dst = dst_idx[mask]

            h_src = h[e_src]
            h_dst = h[e_dst]

            Wq = self.W_q[t]
            Wk = self.W_k[t]
            Wv = self.W_v[t]

            q = h_dst @ Wq.t()
            k = h_src @ Wk.t()
            v = h_src @ Wv.t()

            q = q.view(-1, self.nh, self.head_dim)
            k = k.view(-1, self.nh, self.head_dim)
            v = v.view(-1, self.nh, self.head_dim)

            score = (q * k).sum(-1) / self._scale

            E_t     = e_dst.shape[0]
            dst_exp = e_dst.unsqueeze(1).expand(E_t, self.nh).reshape(-1)
            score_flat  = score.reshape(-1)
            alpha_flat  = scatter_softmax(score_flat, dst_exp, dim=0, dim_size=N)
            alpha = alpha_flat.view(E_t, self.nh, 1)

            weighted_v = (alpha * v).view(E_t, H)
            agg_t = scatter_add(weighted_v, e_dst, dim=0, dim_size=N)
            agg_stack[:, t, :] = agg_t

        return agg_stack

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_type:  Tensor,
    ) -> Tensor:
        N = h.shape[0]

        if self.training and self.drop_edge_p > 0.0 and edge_index.shape[1] > 0:
            keep_mask = torch.rand(edge_index.shape[1], device=h.device) > self.drop_edge_p
            edge_index = edge_index[:, keep_mask]
            edge_type  = edge_type[keep_mask]

        agg_stack = self._relational_agg(h, edge_index, edge_type)

        gate = F.softmax(self.routing_mlp(h), dim=-1)
        sym_edge_w = F.softmax(self.sym_edge_logits, dim=1)

        symptom_channels = []
        for k in range(self.K):
            edge_weights = sym_edge_w[k]
            s_k = (agg_stack * edge_weights.view(1, self.R, 1)).sum(dim=1)
            s_k = self.W_symptom[k](s_k)
            s_k = F.relu(s_k)
            s_k = s_k * gate[:, k].unsqueeze(-1)
            symptom_channels.append(s_k)

        cat_symptoms = torch.cat(symptom_channels, dim=-1)
        h_new = self.cross_symptom_linear(cat_symptoms)
        h_new = F.relu(h_new)
        h_new = self.dropout(h_new)

        h_out = self.layer_norm(h + h_new)
        return h_out


# ─────────────────────────────────────────────────────────────────────────────
# Modal-specific encoder block
# ─────────────────────────────────────────────────────────────────────────────

class ModalEncoder(nn.Module):
    """Linear(in_f, H) + LayerNorm + ReLU + Dropout."""

    def __init__(self, in_features: int, hidden_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            _xavier_linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# HMSGNet: full model (2-modal: Text + Audio)
# ─────────────────────────────────────────────────────────────────────────────

class HMSGNet(nn.Module):
    """
    Heterogeneous Multi-modal Symptom-Guided Network (2-modal).

    Inputs : Text nodes (BERT, 768-dim) + Audio nodes (wav2vec, 768-dim).

    Graph layout per participant:
      Nodes 0..N-1      → text  (node_type == 0)
      Nodes N..2N-1     → audio (node_type == 1)

    Edge types (4):
      0  T→T  temporal
      1  A→A  temporal
      2  T→A  same-utterance
      3  A→T  same-utterance

    Input  : PyG Batch with fields: x, edge_index, edge_type, node_type,
             pos, batch (standard PyG batch vector).
    Output : (dep_logit, symptom_logits, phq_pred)
               dep_logit      : (B,)     depression logit
               symptom_logits : (B, 8)   PHQ-8 item logits
               phq_pred       : (B,)     continuous PHQ score prediction
    """

    def __init__(
        self,
        hidden_dim:     int   = HIDDEN_DIM,
        num_gnn_layers: int   = NUM_GNN_LAYERS,
        num_edge_types: int   = NUM_EDGE_TYPES,
        num_symptoms:   int   = NUM_SYMPTOMS,
        n_heads:        int   = 4,
        dropout:        float = DROPOUT,
        drop_edge:      float = 0.3,
        feat_noise:     float = 0.05,
        text_dim:       int   = TEXT_DIM,
        audio_dim:      int   = AUDIO_DIM,
    ):
        super().__init__()
        self.H          = hidden_dim
        self.K          = num_symptoms
        self.dropout_p  = dropout
        self.feat_noise = feat_noise
        self._text_dim  = text_dim
        self._audio_dim = audio_dim

        # ── Modal-specific encoders ───────────────────────────────────────────
        self.text_encoder  = ModalEncoder(text_dim,  hidden_dim, dropout)
        self.audio_encoder = ModalEncoder(audio_dim, hidden_dim, dropout)

        # ── Node-type embedding (2 types: text=0, audio=1) ────────────────────
        self.node_type_emb = nn.Embedding(2, hidden_dim)
        nn.init.normal_(self.node_type_emb.weight, std=0.02)

        # ── Positional embedding (scalar → H via small MLP) ───────────────────
        self.pos_encoder = nn.Sequential(
            _xavier_linear(1, hidden_dim // 2),
            nn.ReLU(inplace=True),
            _xavier_linear(hidden_dim // 2, hidden_dim),
        )

        # ── SR-RGAT layers ────────────────────────────────────────────────────
        self.gnn_layers = nn.ModuleList([
            SymptomRoutedRGAT(
                hidden_dim=hidden_dim,
                num_edge_types=num_edge_types,
                num_symptoms=num_symptoms,
                n_heads=n_heads,
                dropout=dropout,
                drop_edge=drop_edge,
            )
            for _ in range(num_gnn_layers)
        ])

        # ── Attention readout (per modality) ──────────────────────────────────
        self.text_att  = _xavier_linear(hidden_dim, 1)
        self.audio_att = _xavier_linear(hidden_dim, 1)

        # ── Cross-modal gated fusion (2 modalities) ───────────────────────────
        self.modal_fusion = nn.Sequential(
            _xavier_linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ── Task heads ────────────────────────────────────────────────────────
        self.dep_head = nn.Sequential(
            _xavier_linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            _xavier_linear(hidden_dim // 2, 1),
        )

        self.symptom_heads = nn.ModuleList([
            _xavier_linear(hidden_dim, 1) for _ in range(num_symptoms)
        ])

        self.phq_head = _xavier_linear(hidden_dim, 1)

    # ── Per-modality feature extraction from the unified x ───────────────────

    def _encode_by_node_type(
        self,
        x:         Tensor,   # (N_total, unified_dim)
        node_type: Tensor,   # (N_total,)
    ) -> Tensor:
        """
        Route each row of x to its modal encoder and return h: (N_total, H).
        Unified x layout: [text_feat(text_dim) | audio_feat(audio_dim)]
        """
        device = x.device
        H = self.H
        N = x.shape[0]
        h = torch.zeros(N, H, device=device, dtype=x.dtype)

        if self.training and self.feat_noise > 0.0:
            x = x + torch.randn_like(x) * self.feat_noise

        t_dim = self._text_dim
        a_dim = self._audio_dim

        # Text nodes (node_type == 0): first t_dim dims
        mask0 = (node_type == 0)
        if mask0.any():
            h[mask0] = self.text_encoder(x[mask0, :t_dim])

        # Audio nodes (node_type == 1): dims t_dim .. t_dim+a_dim
        mask1 = (node_type == 1)
        if mask1.any():
            h[mask1] = self.audio_encoder(x[mask1, t_dim: t_dim + a_dim])

        return h

    # ── Attention pooling within each graph-modality group ───────────────────

    @staticmethod
    def _att_pool(
        h:      Tensor,
        idx:    Tensor,
        att_w:  Tensor,
        B:      int,
    ) -> Tensor:
        alpha = scatter_softmax(att_w.squeeze(-1), idx, dim=0, dim_size=B)
        out   = scatter_add(h * alpha.unsqueeze(-1), idx, dim=0, dim_size=B)
        return out

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, data) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        data : torch_geometric.data.Batch (or Data for single graph)
            Required fields: x, edge_index, edge_type, node_type, pos, batch

        Returns
        -------
        dep_logit      : (B,)
        symptom_logits : (B, K)
        phq_pred       : (B,)
        """
        x          = data.x
        edge_index = data.edge_index
        edge_type  = data.edge_type
        node_type  = data.node_type
        pos        = data.pos
        batch      = data.batch

        B = int(batch.max().item()) + 1

        # 1. Modal-specific encoding
        h = self._encode_by_node_type(x, node_type)

        # 2. Add positional and node_type embeddings
        pos_emb  = self.pos_encoder(pos.unsqueeze(-1))
        type_emb = self.node_type_emb(node_type)
        h = h + pos_emb + type_emb

        # 3. SR-RGAT layers
        for gnn in self.gnn_layers:
            h = gnn(h, edge_index, edge_type)

        # 4. Attention readout per modality
        mask_t = (node_type == 0)
        mask_a = (node_type == 1)

        h_t = h[mask_t]
        h_a = h[mask_a]

        batch_t = batch[mask_t]
        batch_a = batch[mask_a]

        att_t = torch.sigmoid(self.text_att(h_t))
        att_a = torch.sigmoid(self.audio_att(h_a))

        text_emb  = self._att_pool(h_t, batch_t, att_t, B)
        audio_emb = self._att_pool(h_a, batch_a, att_a, B)

        # 5. Cross-modal gated fusion (text + audio)
        fused = self.modal_fusion(
            torch.cat([text_emb, audio_emb], dim=-1)
        )

        # 6. Task heads
        dep_logit = self.dep_head(fused).squeeze(-1)

        symptom_logits = torch.cat(
            [head(fused) for head in self.symptom_heads], dim=-1
        )

        phq_pred = self.phq_head(fused).squeeze(-1)

        return dep_logit, symptom_logits, phq_pred


# ─────────────────────────────────────────────────────────────────────────────
# compute_loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    dep_logit:       Tensor,
    symptom_logits:  Tensor,
    phq_pred:        Tensor,
    dep_labels:      Tensor,
    phq8_labels:     Tensor,
    phq_scores:      Tensor,
    w_symptom:       float = 0.3,
    w_phq:           float = 0.1,
    focal_alpha:     float = 0.80,
    label_smoothing: float = 0.0,
    device:          Optional[torch.device] = None,
) -> Tuple[Tensor, dict]:
    """
    Combined loss:
        L = FocalLoss(dep) + w_symptom * BCE(symptoms) + w_phq * SmoothL1(phq)
    """
    dep_labels  = dep_labels.float()
    phq8_labels = phq8_labels.float()
    phq_scores  = phq_scores.float()

    B = dep_logit.shape[0]
    if phq8_labels.dim() == 1 and phq8_labels.shape[0] == B * 8:
        phq8_labels = phq8_labels.view(B, 8)

    focal_fn = FocalLoss(alpha=focal_alpha, gamma=2.0, reduction="mean",
                         label_smoothing=label_smoothing)
    loss_dep = focal_fn(dep_logit, dep_labels)

    phq8_binary = (phq8_labels > 0).float()
    loss_symp   = F.binary_cross_entropy_with_logits(
        symptom_logits, phq8_binary, reduction="mean"
    )

    loss_phq = F.smooth_l1_loss(phq_pred, phq_scores / 24.0, reduction="mean")

    total = loss_dep + w_symptom * loss_symp + w_phq * loss_phq

    loss_dict = {
        "loss_total":   total.item(),
        "loss_dep":     loss_dep.item(),
        "loss_symptom": loss_symp.item(),
        "loss_phq":     loss_phq.item(),
    }
    return total, loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    def _make_dummy(n_utt: int = 20):
        N = 2 * n_utt   # text + audio nodes
        ei, et = [], []
        T_off, A_off = 0, n_utt

        # T→T, A→A temporal
        for i in range(n_utt):
            for j in range(max(0, i - 3), min(n_utt, i + 4)):
                if i != j:
                    ei.append([T_off + i, T_off + j]); et.append(0)
                    ei.append([A_off + i, A_off + j]); et.append(1)
        # T→A, A→T cross-modal
        for i in range(n_utt):
            ei.append([T_off + i, A_off + i]); et.append(2)
            ei.append([A_off + i, T_off + i]); et.append(3)

        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_type  = torch.tensor(et, dtype=torch.long)
        node_type  = torch.tensor([0]*n_utt + [1]*n_utt, dtype=torch.long)
        pos        = torch.linspace(0, 1, n_utt).repeat(2)
        x          = torch.randn(N, UNIFIED_DIM)
        return Data(
            x=x, edge_index=edge_index, edge_type=edge_type,
            node_type=node_type, pos=pos,
            y=torch.tensor([1], dtype=torch.long),
            phq_score=torch.tensor([10.0]),
            phq8=torch.rand(8),
        )

    graphs = [_make_dummy(20), _make_dummy(15)]
    batch  = Batch.from_data_list(graphs)

    model = HMSGNet()
    model.eval()
    with torch.no_grad():
        dep_logit, sym_logits, phq_pred = model(batch)
    print(f"dep_logit:      {dep_logit.shape}")
    print(f"symptom_logits: {sym_logits.shape}")
    print(f"phq_pred:       {phq_pred.shape}")

    dep_labels = batch.y.squeeze()
    phq8_lbl   = torch.stack([d.phq8 for d in graphs])
    phq_scores = batch.phq_score.squeeze()
    loss, ld   = compute_loss(dep_logit, sym_logits, phq_pred,
                              dep_labels, phq8_lbl, phq_scores)
    print(f"loss: {loss.item():.4f}  {ld}")
    print("Smoke-test PASSED.")
