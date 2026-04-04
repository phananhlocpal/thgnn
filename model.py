"""
model.py — HMSGNet v2

HMSG-Net: Heterogeneous Multi-modal Symptom-Guided Graph Network
================================================================

Thay đổi vs v1
──────────────
[FIX] NUM_EDGE_TYPES = 6 (thêm T->T_same_question, A->A_same_question).
[FIX] TEXT_DIM=777, AUDIO_DIM=777 (768 + 9 acoustic side-channels).
      UNIFIED_DIM = 1554.
[FIX] drop_edge = 0.2 (giảm từ 0.3) để không drop toàn bộ same-question edges
      trong batches nhỏ.
[IMPROVE] SR-RGAT: thêm residual gate (sigmoid-gated) thay vì additive residual.
[IMPROVE] GatedModalFusion: text và audio attend to each other qua gated attention,
          thay vì naive concatenation.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add, scatter_softmax

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

TEXT_DIM       = 777     # 768 (Mental-BERT) + 9 (text acoustic side-channels)
AUDIO_DIM      = 777     # 768 (WavLM-base-plus) + 9 (audio acoustic side-channels)
UNIFIED_DIM    = TEXT_DIM + AUDIO_DIM   # 1554

NUM_EDGE_TYPES = 6       # T->T_temp, A->A_temp, T->A_utt, A->T_utt, T->T_q, A->A_q
NUM_NODE_TYPES = 2
NUM_SYMPTOMS   = 8

HIDDEN_DIM     = 256
NUM_GNN_LAYERS = 3
DROPOUT        = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha:           float = 0.75,
        gamma:           float = 2.0,
        reduction:       str   = "mean",
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
        smooth_t = (targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
                    if self.label_smoothing > 0 else targets)
        bce     = F.binary_cross_entropy_with_logits(logits, smooth_t, reduction="none")
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = alpha_t * (1 - p_t) ** self.gamma * bce
        if self.reduction == "mean":  return loss.mean()
        if self.reduction == "sum":   return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xavier_linear(in_f: int, out_f: int, bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_f, out_f, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if bias: nn.init.zeros_(layer.bias)
    return layer


class _MLP(nn.Module):
    def __init__(self, in_f: int, h_f: int, out_f: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            _xavier_linear(in_f, h_f), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            _xavier_linear(h_f, out_f),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# SR-RGAT v2: Symptom-Routed Relational Graph Attention
# ─────────────────────────────────────────────────────────────────────────────

class SymptomRoutedRGAT(nn.Module):
    """
    SR-RGAT layer v2.

    Forward: h, edge_index (2,E), edge_type (E,) → h_out (N, H).

    Changes from v1:
    - Residual gate: h_out = LayerNorm(gate*h_new + (1-gate)*h)
      where gate = sigmoid(W_gate * concat(h, h_new))
    - Supports 6 edge types.
    """

    def __init__(
        self,
        hidden_dim:     int   = HIDDEN_DIM,
        num_edge_types: int   = NUM_EDGE_TYPES,
        num_symptoms:   int   = NUM_SYMPTOMS,
        n_heads:        int   = 4,
        dropout:        float = DROPOUT,
        drop_edge:      float = 0.2,
    ):
        super().__init__()
        self.H          = hidden_dim
        self.R          = num_edge_types
        self.K          = num_symptoms
        self.nh         = n_heads
        self.drop_edge_p = drop_edge
        assert hidden_dim % n_heads == 0
        self.head_dim   = hidden_dim // n_heads
        self._scale     = math.sqrt(self.head_dim)

        self.W_q = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        self.W_k = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        self.W_v = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        for w in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(w.view(num_edge_types * hidden_dim, hidden_dim))

        self.routing_mlp     = _MLP(hidden_dim, hidden_dim, num_symptoms, dropout=dropout)
        self.sym_edge_logits = nn.Parameter(torch.empty(num_symptoms, num_edge_types))
        nn.init.xavier_uniform_(self.sym_edge_logits)

        self.W_symptom = nn.ModuleList([
            _xavier_linear(hidden_dim, hidden_dim) for _ in range(num_symptoms)
        ])
        self.cross_symptom = _xavier_linear(num_symptoms * hidden_dim, hidden_dim)

        # Residual gate
        self.W_gate    = _xavier_linear(2 * hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(p=dropout)

    def _relational_agg(self, h: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        N = h.shape[0]
        agg = torch.zeros(N, self.R, self.H, device=h.device, dtype=h.dtype)
        if edge_index.shape[1] == 0:
            return agg

        src = edge_index[0]
        dst = edge_index[1]

        for t in range(self.R):
            mask = (edge_type == t)
            if not mask.any():
                continue
            e_src = src[mask]
            e_dst = dst[mask]
            E_t   = e_src.shape[0]

            q = (h[e_dst] @ self.W_q[t].t()).view(E_t, self.nh, self.head_dim)
            k = (h[e_src] @ self.W_k[t].t()).view(E_t, self.nh, self.head_dim)
            v = (h[e_src] @ self.W_v[t].t()).view(E_t, self.nh, self.head_dim)

            score   = (q * k).sum(-1) / self._scale
            dst_exp = e_dst.unsqueeze(1).expand(E_t, self.nh).reshape(-1)
            alpha   = scatter_softmax(score.reshape(-1), dst_exp, dim=0, dim_size=N)
            alpha   = alpha.view(E_t, self.nh, 1)

            agg[:, t, :] = scatter_add((alpha * v).view(E_t, self.H), e_dst, dim=0, dim_size=N)
        return agg

    def forward(self, h: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        if self.training and self.drop_edge_p > 0 and edge_index.shape[1] > 0:
            keep       = torch.rand(edge_index.shape[1], device=h.device) > self.drop_edge_p
            edge_index = edge_index[:, keep]
            edge_type  = edge_type[keep]

        agg_stack  = self._relational_agg(h, edge_index, edge_type)
        gate_vals  = F.softmax(self.routing_mlp(h), dim=-1)    # (N, K)
        sym_edge_w = F.softmax(self.sym_edge_logits, dim=1)    # (K, R)

        channels = []
        for k in range(self.K):
            s_k = (agg_stack * sym_edge_w[k].view(1, self.R, 1)).sum(dim=1)
            s_k = F.relu(self.W_symptom[k](s_k)) * gate_vals[:, k].unsqueeze(-1)
            channels.append(s_k)

        h_new = F.relu(self.cross_symptom(torch.cat(channels, dim=-1)))
        h_new = self.dropout(h_new)

        # Residual gate
        gate  = torch.sigmoid(self.W_gate(torch.cat([h, h_new], dim=-1)))
        return self.layer_norm(gate * h_new + (1.0 - gate) * h)


# ─────────────────────────────────────────────────────────────────────────────
# Modal encoder
# ─────────────────────────────────────────────────────────────────────────────

class ModalEncoder(nn.Module):
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
# Gated Cross-modal Fusion
# ─────────────────────────────────────────────────────────────────────────────

class GatedModalFusion(nn.Module):
    """
    Text and audio attend to each other via learned gates.
    text_gate  = sigmoid(W_t * [text; audio])
    audio_gate = sigmoid(W_a * [audio; text])
    fused = MLP([text_gate*text ; audio_gate*audio])
    """
    def __init__(self, hidden_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.W_tg  = _xavier_linear(2 * hidden_dim, hidden_dim)
        self.W_ag  = _xavier_linear(2 * hidden_dim, hidden_dim)
        self.out   = nn.Sequential(
            _xavier_linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, t: Tensor, a: Tensor) -> Tensor:
        tg = torch.sigmoid(self.W_tg(torch.cat([t, a], dim=-1)))
        ag = torch.sigmoid(self.W_ag(torch.cat([a, t], dim=-1)))
        return self.out(torch.cat([tg * t, ag * a], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# HMSGNet v2
# ─────────────────────────────────────────────────────────────────────────────

class HMSGNet(nn.Module):
    def __init__(
        self,
        hidden_dim:     int   = HIDDEN_DIM,
        num_gnn_layers: int   = NUM_GNN_LAYERS,
        num_edge_types: int   = NUM_EDGE_TYPES,
        num_symptoms:   int   = NUM_SYMPTOMS,
        n_heads:        int   = 4,
        dropout:        float = DROPOUT,
        drop_edge:      float = 0.2,
        feat_noise:     float = 0.05,
        text_dim:       int   = TEXT_DIM,
        audio_dim:      int   = AUDIO_DIM,
    ):
        super().__init__()
        self.H          = hidden_dim
        self.K          = num_symptoms
        self.feat_noise = feat_noise
        self._text_dim  = text_dim
        self._audio_dim = audio_dim

        self.text_encoder  = ModalEncoder(text_dim,  hidden_dim, dropout)
        self.audio_encoder = ModalEncoder(audio_dim, hidden_dim, dropout)
        self.node_type_emb = nn.Embedding(2, hidden_dim)
        nn.init.normal_(self.node_type_emb.weight, std=0.02)
        self.pos_encoder   = nn.Sequential(
            _xavier_linear(1, hidden_dim // 2), nn.ReLU(inplace=True),
            _xavier_linear(hidden_dim // 2, hidden_dim),
        )

        self.gnn_layers = nn.ModuleList([
            SymptomRoutedRGAT(
                hidden_dim=hidden_dim, num_edge_types=num_edge_types,
                num_symptoms=num_symptoms, n_heads=n_heads,
                dropout=dropout, drop_edge=drop_edge,
            ) for _ in range(num_gnn_layers)
        ])

        self.text_att    = _xavier_linear(hidden_dim, 1)
        self.audio_att   = _xavier_linear(hidden_dim, 1)
        self.modal_fusion = GatedModalFusion(hidden_dim, dropout)

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
        N = x.shape[0]
        h = torch.zeros(N, self.H, device=x.device, dtype=x.dtype)
        if self.training and self.feat_noise > 0:
            x = x + torch.randn_like(x) * self.feat_noise
        m0 = (node_type == 0)
        if m0.any(): h[m0] = self.text_encoder(x[m0, :self._text_dim])
        m1 = (node_type == 1)
        if m1.any(): h[m1] = self.audio_encoder(x[m1, self._text_dim: self._text_dim + self._audio_dim])
        return h

    @staticmethod
    def _att_pool(h: Tensor, idx: Tensor, w: Tensor, B: int) -> Tensor:
        alpha = scatter_softmax(w.squeeze(-1), idx, dim=0, dim_size=B)
        return scatter_add(h * alpha.unsqueeze(-1), idx, dim=0, dim_size=B)

    def forward(self, data) -> Tuple[Tensor, Tensor, Tensor]:
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        node_type, pos, batch    = data.node_type, data.pos, data.batch
        B = int(batch.max().item()) + 1

        h = self._encode_by_node_type(x, node_type)
        h = h + self.pos_encoder(pos.unsqueeze(-1)) + self.node_type_emb(node_type)

        for gnn in self.gnn_layers:
            h = gnn(h, edge_index, edge_type)

        mt = (node_type == 0)
        ma = (node_type == 1)
        text_emb  = self._att_pool(h[mt], batch[mt], torch.sigmoid(self.text_att(h[mt])),  B)
        audio_emb = self._att_pool(h[ma], batch[ma], torch.sigmoid(self.audio_att(h[ma])), B)
        fused     = self.modal_fusion(text_emb, audio_emb)

        dep_logit      = self.dep_head(fused).squeeze(-1)
        symptom_logits = torch.cat([head(fused) for head in self.symptom_heads], dim=-1)
        phq_pred       = self.phq_head(fused).squeeze(-1)
        return dep_logit, symptom_logits, phq_pred


# ─────────────────────────────────────────────────────────────────────────────
# Loss
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
    B = dep_logit.shape[0]
    phq8_labels = phq8_labels.float()
    if phq8_labels.dim() == 1 and phq8_labels.shape[0] == B * 8:
        phq8_labels = phq8_labels.view(B, 8)

    focal_fn  = FocalLoss(alpha=focal_alpha, gamma=2.0, reduction="mean",
                          label_smoothing=label_smoothing)
    loss_dep  = focal_fn(dep_logit, dep_labels.float())
    loss_symp = F.binary_cross_entropy_with_logits(
        symptom_logits, (phq8_labels > 0).float(), reduction="mean"
    )
    loss_phq  = F.smooth_l1_loss(phq_pred, phq_scores.float() / 24.0, reduction="mean")
    total     = loss_dep + w_symptom * loss_symp + w_phq * loss_phq

    return total, {
        "loss_total":   total.item(),
        "loss_dep":     loss_dep.item(),
        "loss_symptom": loss_symp.item(),
        "loss_phq":     loss_phq.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from collections import defaultdict
    from torch_geometric.data import Data, Batch

    def _make_dummy(n_utt: int = 20):
        N     = 2 * n_utt
        T_off = 0
        A_off = n_utt
        q_ids = [str(i // 3) for i in range(n_utt)]

        ei, et = [], []
        for i in range(n_utt):
            for j in range(max(0, i-3), min(n_utt, i+4)):
                if i != j:
                    ei.append([T_off+i, T_off+j]); et.append(0)
                    ei.append([A_off+i, A_off+j]); et.append(1)
        for i in range(n_utt):
            ei.append([T_off+i, A_off+i]); et.append(2)
            ei.append([A_off+i, T_off+i]); et.append(3)

        q_map = defaultdict(list)
        for i, q in enumerate(q_ids):
            q_map[q].append(i)
        for idxs in q_map.values():
            for a in range(len(idxs)):
                for b in range(len(idxs)):
                    if a != b:
                        ei.append([T_off+idxs[a], T_off+idxs[b]]); et.append(4)
                        ei.append([A_off+idxs[a], A_off+idxs[b]]); et.append(5)

        return Data(
            x          = torch.randn(N, UNIFIED_DIM),
            edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous(),
            edge_type  = torch.tensor(et, dtype=torch.long),
            node_type  = torch.tensor([0]*n_utt + [1]*n_utt, dtype=torch.long),
            pos        = torch.linspace(0, 1, n_utt).repeat(2),
            y          = torch.tensor([1], dtype=torch.long),
            phq_score  = torch.tensor([10.0]),
            phq8       = torch.rand(8),
        )

    graphs = [_make_dummy(20), _make_dummy(15)]
    batch  = Batch.from_data_list(graphs)
    model  = HMSGNet()
    model.eval()

    with torch.no_grad():
        dep_logit, sym_logits, phq_pred = model(batch)

    print(f"dep_logit:      {dep_logit.shape}")
    print(f"symptom_logits: {sym_logits.shape}")
    print(f"phq_pred:       {phq_pred.shape}")

    loss, ld = compute_loss(
        dep_logit, sym_logits, phq_pred,
        batch.y.reshape(-1).float(),
        torch.stack([d.phq8 for d in graphs]),
        batch.phq_score.reshape(-1),
    )
    print(f"loss: {loss.item():.4f}  {ld}")
    print("Smoke-test PASSED.")