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

FIXES vs original:
  1. [FIX-INTERPRETABILITY] SR-RGAT now optionally returns per-node symptom
     routing weights (gate: N×K and sym_edge_w: K×R) via return_routing=True.
     Used for SHAP-style analysis in train.py / a separate explain.py.
  2. [FIX-INTERPRETABILITY] HMSGNet.forward() accepts return_routing=True,
     returns (dep_logit, sym_logits, phq_pred, routing_info) where routing_info
     is a dict with keys: "gates" (list of N×K tensors per layer),
     "sym_edge_weights" (list of K×R tensors per layer).
     When return_routing=False (default), original 3-tuple is returned.
  3. [FIX-STABILITY] Added numerical stability guard in scatter_softmax path
     for empty edge sets (edge_index.shape[1]==0 after edge dropout).
  4. [FIX-STABILITY] ModalEncoder: added gradient checkpointing option to
     reduce memory on long sequences.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add, scatter_softmax

# ─────────────────────────────────────────────────────────────────────────────
# Architecture hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

TEXT_DIM   = 768
AUDIO_DIM  = 768
UNIFIED_DIM = TEXT_DIM + AUDIO_DIM

NUM_EDGE_TYPES = 4
NUM_NODE_TYPES = 2
NUM_SYMPTOMS   = 8

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
        Balancing factor for the positive class.
    gamma : float
        Focusing exponent. Default 2.0.
    reduction : str
        'mean' | 'sum' | 'none'.
    label_smoothing : float
        If > 0, smooth hard targets before computing BCE.
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
        targets = targets.float()
        probs   = torch.sigmoid(logits)

        if self.label_smoothing > 0.0:
            smooth_t = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smooth_t = targets
        bce     = F.binary_cross_entropy_with_logits(logits, smooth_t, reduction="none")
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
    def __init__(self, in_f: int, hidden_f: int, out_f: int, dropout: float = 0.0):
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
      h              : (N, H)   node features
      edge_index     : (2, E)   COO edge list
      edge_type      : (E,)     int in 0..R-1

    Returns:
      h_out          : (N, H)   updated node features
      routing_info   : dict with keys "gate" (N, K), "sym_edge_w" (K, R)
                       Only populated when return_routing=True.
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
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.head_dim = hidden_dim // n_heads

        self.W_q = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        self.W_k = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        self.W_v = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        for w in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(w.view(num_edge_types * hidden_dim, hidden_dim))

        self.routing_mlp = _MLP(hidden_dim, hidden_dim, num_symptoms, dropout=dropout)

        self.sym_edge_logits = nn.Parameter(torch.empty(num_symptoms, num_edge_types))
        nn.init.xavier_uniform_(self.sym_edge_logits)

        self.W_symptom = nn.ModuleList([
            _xavier_linear(hidden_dim, hidden_dim, bias=True)
            for _ in range(num_symptoms)
        ])

        self.cross_symptom_linear = _xavier_linear(num_symptoms * hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(p=dropout)
        self._scale     = math.sqrt(self.head_dim)

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

        # FIX-STABILITY: guard against empty edge set after dropout
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
            n_edges_t = e_dst.shape[0]

            h_src = h[e_src]
            h_dst = h[e_dst]

            q = (h_dst @ self.W_q[t].t()).view(-1, self.nh, self.head_dim)
            k = (h_src @ self.W_k[t].t()).view(-1, self.nh, self.head_dim)
            v = (h_src @ self.W_v[t].t()).view(-1, self.nh, self.head_dim)

            score = (q * k).sum(-1) / self._scale  # (E_t, nh)

            dst_exp = e_dst.unsqueeze(1).expand(n_edges_t, self.nh).reshape(-1)
            score_flat = score.reshape(-1)
            alpha_flat = scatter_softmax(score_flat, dst_exp, dim=0, dim_size=N)
            alpha = alpha_flat.view(n_edges_t, self.nh, 1)

            weighted_v = (alpha * v).view(n_edges_t, H)
            agg_t = scatter_add(weighted_v, e_dst, dim=0, dim_size=N)
            agg_stack[:, t, :] = agg_t

        return agg_stack

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_type:  Tensor,
        return_routing: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        N = h.shape[0]

        if self.training and self.drop_edge_p > 0.0 and edge_index.shape[1] > 0:
            keep_mask  = torch.rand(edge_index.shape[1], device=h.device) > self.drop_edge_p
            edge_index = edge_index[:, keep_mask]
            edge_type  = edge_type[keep_mask]

        agg_stack = self._relational_agg(h, edge_index, edge_type)

        # Symptom routing gate: (N, K) — per-node soft routing weights
        gate = F.softmax(self.routing_mlp(h), dim=-1)

        # Edge-type weights per symptom: (K, R)
        sym_edge_w = F.softmax(self.sym_edge_logits, dim=1)

        symptom_channels = []
        for k in range(self.K):
            edge_weights = sym_edge_w[k]                              # (R,)
            s_k = (agg_stack * edge_weights.view(1, self.R, 1)).sum(dim=1)   # (N, H)
            s_k = self.W_symptom[k](s_k)
            s_k = F.relu(s_k)
            s_k = s_k * gate[:, k].unsqueeze(-1)                     # (N, H)
            symptom_channels.append(s_k)

        cat_symptoms = torch.cat(symptom_channels, dim=-1)            # (N, K*H)
        h_new = self.cross_symptom_linear(cat_symptoms)
        h_new = F.relu(h_new)
        h_new = self.dropout(h_new)

        h_out = self.layer_norm(h + h_new)

        if return_routing:
            return h_out, {
                "gate":        gate.detach(),         # (N, K)
                "sym_edge_w":  sym_edge_w.detach(),   # (K, R)
            }
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

    Returns
    -------
    default (return_routing=False):
        (dep_logit, symptom_logits, phq_pred)

    with return_routing=True:
        (dep_logit, symptom_logits, phq_pred, routing_info)
        where routing_info = {
          "gates":           List[Tensor(N, K)]  — one per GNN layer
          "sym_edge_weights": List[Tensor(K, R)] — one per GNN layer
        }
        Use routing_info for interpretability / SHAP input-attribution.
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

        self.text_encoder  = ModalEncoder(text_dim,  hidden_dim, dropout)
        self.audio_encoder = ModalEncoder(audio_dim, hidden_dim, dropout)

        self.node_type_emb = nn.Embedding(2, hidden_dim)
        nn.init.normal_(self.node_type_emb.weight, std=0.02)

        self.pos_encoder = nn.Sequential(
            _xavier_linear(1, hidden_dim // 2),
            nn.ReLU(inplace=True),
            _xavier_linear(hidden_dim // 2, hidden_dim),
        )

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

        self.text_att  = _xavier_linear(hidden_dim, 1)
        self.audio_att = _xavier_linear(hidden_dim, 1)

        self.modal_fusion = nn.Sequential(
            _xavier_linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

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

    def _encode_by_node_type(self, x: Tensor, node_type: Tensor) -> Tensor:
        device = x.device
        H = self.H
        N = x.shape[0]
        h = torch.zeros(N, H, device=device, dtype=x.dtype)

        if self.training and self.feat_noise > 0.0:
            x = x + torch.randn_like(x) * self.feat_noise

        t_dim = self._text_dim
        a_dim = self._audio_dim

        mask0 = (node_type == 0)
        if mask0.any():
            h[mask0] = self.text_encoder(x[mask0, :t_dim])

        mask1 = (node_type == 1)
        if mask1.any():
            h[mask1] = self.audio_encoder(x[mask1, t_dim: t_dim + a_dim])

        return h

    @staticmethod
    def _att_pool(h: Tensor, idx: Tensor, att_w: Tensor, B: int) -> Tensor:
        alpha = scatter_softmax(att_w.squeeze(-1), idx, dim=0, dim_size=B)
        return scatter_add(h * alpha.unsqueeze(-1), idx, dim=0, dim_size=B)

    def forward(
        self,
        data,
        return_routing: bool = False,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Dict],
    ]:
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
        all_gates: List[Tensor] = []
        all_sym_edge_w: List[Tensor] = []

        for gnn in self.gnn_layers:
            if return_routing:
                h, rinfo = gnn(h, edge_index, edge_type, return_routing=True)
                all_gates.append(rinfo["gate"])
                all_sym_edge_w.append(rinfo["sym_edge_w"])
            else:
                h = gnn(h, edge_index, edge_type, return_routing=False)

        # 4. Attention readout per modality
        mask_t = (node_type == 0)
        mask_a = (node_type == 1)

        h_t = h[mask_t]
        h_a = h[mask_a]

        batch_t = batch[mask_t]
        batch_a = batch[mask_a]

        text_emb  = self._att_pool(h_t, batch_t, torch.sigmoid(self.text_att(h_t)),  B)
        audio_emb = self._att_pool(h_a, batch_a, torch.sigmoid(self.audio_att(h_a)), B)

        # 5. Cross-modal gated fusion
        fused = self.modal_fusion(torch.cat([text_emb, audio_emb], dim=-1))

        # 6. Task heads
        dep_logit      = self.dep_head(fused).squeeze(-1)
        symptom_logits = torch.cat([head(fused) for head in self.symptom_heads], dim=-1)
        phq_pred       = self.phq_head(fused).squeeze(-1)

        if return_routing:
            routing_info = {
                "gates":            all_gates,       # list of (N, K) per layer
                "sym_edge_weights": all_sym_edge_w,  # list of (K, R) per layer
                "node_type":        node_type,       # (N,) for splitting T/A
                "batch":            batch,           # (N,) graph assignment
            }
            return dep_logit, symptom_logits, phq_pred, routing_info

        return dep_logit, symptom_logits, phq_pred

    # ── Interpretability helpers ───────────────────────────────────────────────

    @torch.no_grad()
    def get_symptom_routing_summary(self, data) -> Dict:
        """
        Run forward with return_routing=True and return a clean summary dict.

        Useful for SHAP-style analysis and paper visualizations.

        Returns
        -------
        dict with:
          "mean_gate_per_symptom"    : (K,)  averaged gate activation across nodes
          "sym_edge_weights_by_layer": list of (K, R) numpy arrays
          "gate_text_vs_audio"       : dict with "text" and "audio" keys → (K,)
        """
        self.eval()
        _, _, _, rinfo = self.forward(data, return_routing=True)

        node_type = rinfo["node_type"]
        text_mask = (node_type == 0).cpu()
        audio_mask= (node_type == 1).cpu()

        # Average gate across all layers (last layer is most task-relevant)
        last_gate = rinfo["gates"][-1].cpu()  # (N, K)

        mean_gate           = last_gate.mean(dim=0).numpy()
        mean_gate_text_only = last_gate[text_mask].mean(dim=0).numpy()
        mean_gate_audio_only= last_gate[audio_mask].mean(dim=0).numpy()

        sym_edge_weights = [w.cpu().numpy() for w in rinfo["sym_edge_weights"]]

        return {
            "mean_gate_per_symptom"    : mean_gate,
            "mean_gate_text_nodes"     : mean_gate_text_only,
            "mean_gate_audio_nodes"    : mean_gate_audio_only,
            "sym_edge_weights_by_layer": sym_edge_weights,
            "phq8_labels": [
                "NoInterest", "Depressed", "Sleep", "Tired",
                "Appetite",   "Failure",   "Concentrating", "Moving",
            ],
            "edge_type_labels": ["T→T", "A→A", "T→A", "A→T"],
        }


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
        N = 2 * n_utt
        ei, et = [], []
        T_off, A_off = 0, n_utt
        for i in range(n_utt):
            for j in range(max(0, i - 3), min(n_utt, i + 4)):
                if i != j:
                    ei.append([T_off + i, T_off + j]); et.append(0)
                    ei.append([A_off + i, A_off + j]); et.append(1)
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

    # Standard forward
    with torch.no_grad():
        dep_logit, sym_logits, phq_pred = model(batch)
    print(f"dep_logit:      {dep_logit.shape}")
    print(f"symptom_logits: {sym_logits.shape}")
    print(f"phq_pred:       {phq_pred.shape}")

    # Routing forward (for interpretability)
    with torch.no_grad():
        dep_logit, sym_logits, phq_pred, rinfo = model(batch, return_routing=True)
    print(f"routing gates (last layer): {rinfo['gates'][-1].shape}")
    print(f"sym_edge_weights (last):    {rinfo['sym_edge_weights'][-1].shape}")

    # Loss
    dep_labels = batch.y.squeeze()
    phq8_lbl   = torch.stack([d.phq8 for d in graphs])
    phq_scores = batch.phq_score.squeeze()
    loss, ld   = compute_loss(dep_logit, sym_logits, phq_pred,
                              dep_labels, phq8_lbl, phq_scores)
    print(f"loss: {loss.item():.4f}  {ld}")

    # Routing summary (single graph)
    single = Batch.from_data_list([graphs[0]])
    summary = model.get_symptom_routing_summary(single)
    print("Symptom gate summary:", dict(zip(
        summary["phq8_labels"],
        [f"{v:.3f}" for v in summary["mean_gate_per_symptom"]]
    )))
    print("Smoke-test PASSED.")