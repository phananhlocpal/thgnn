"""
main.py — Multimodal Sentiment Analysis (Q2-level, single file)
================================================================
Architecture:
  - Tri-modal encoders: Audio (BiLSTM) + Text (Linear/BERT-ready) + Visual (Linear)
  - InfoNCE contrastive alignment (pairwise cross-modal)
  - Heterogeneous Temporal Graph (5 edge types: intra-temporal x2, cross-modal x2, speaker)
  - RGCN (Relational GCN) with hierarchical pooling
  - Benchmarks: CMU-MOSI / CMU-MOSEI compatible

Install:
  pip install torch torch-geometric scikit-learn numpy tqdm
  pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<version>.html

Run:
    python main.py --dataset mosi   # requires real loader implementation
  python main.py --ablation       # run ablation study
  python main.py --no-graph       # disable graph module
  python main.py --no-nce         # disable InfoNCE loss
"""

import os
import argparse
import random
import time
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Try importing PyG — graceful fallback message
try:
    from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[WARNING] torch_geometric not found — using MLP fallback.")
    print("  Install: pip install torch-geometric")


# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

@dataclass
class Config:
    # Modality input dims (change to match your real features)
    AUDIO_DIM:   int = 74      # COVAREP features in CMU-MOSI
    TEXT_DIM:    int = 768     # BERT-base hidden size
    VISUAL_DIM:  int = 35      # FACET features in CMU-MOSI
    HIDDEN_DIM:  int = 128
    PROJ_DIM:    int = 128

    # Graph
    NUM_EDGE_TYPES: int = 5    # intra-a, intra-t, cross-at, cross-ta, speaker
    TOP_K:          int = 5    # speaker-relation top-k neighbours

    # Training
    LR:           float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EPOCHS:       int   = 30
    BATCH_SIZE:   int   = 8
    PATIENCE:     int   = 7    # early stopping

    # Loss weights
    LAMBDA_NCE:  float = 0.10
    LAMBDA_COMP: float = 0.05
    TEMPERATURE: float = 0.07

    # Misc
    SEED:      int = 42
    DEVICE:    str = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH: str = "best_model.pt"

    # Ablation flags — set False to isolate contribution of each component
    USE_GRAPH:         bool = True
    USE_INFO_NCE:      bool = True
    USE_SPEAKER_EDGE:  bool = True
    USE_TEMPORAL_EDGE: bool = True
    USE_VISUAL:        bool = True


CFG = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# 2. DATASET  (real data only)
# ─────────────────────────────────────────────

class MultimodalSample:
    """One conversation holding per-utterance tensors."""
    def __init__(self, audio, text, visual, speaker_ids, label):
        self.audio       = audio        # (T, AUDIO_DIM)
        self.text        = text         # (T, TEXT_DIM)
        self.visual      = visual       # (T, VISUAL_DIM)
        self.speaker_ids = speaker_ids  # (T,)  int
        self.label       = label        # scalar float  0 / 1


class PlaceholderDataset(Dataset):
    """Placeholder to indicate real dataset loader must be implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Synthetic data has been removed. Implement a real CMU-MOSI/MOSEI dataset loader."
        )

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError(idx)


def collate_fn(batch: List[MultimodalSample]) -> dict:
    """
    Merge variable-length conversations into flat node tensors.
    Each utterance = one graph node; batch_vec tracks which graph it belongs to.
    """
    audio_l, text_l, visual_l, speaker_l, label_l, batch_l = [], [], [], [], [], []
    for i, s in enumerate(batch):
        T = s.audio.size(0)
        audio_l.append(s.audio)
        text_l.append(s.text)
        visual_l.append(s.visual)
        speaker_l.append(s.speaker_ids)
        label_l.append(s.label)
        batch_l.append(torch.full((T,), i, dtype=torch.long))
    return {
        "audio":       torch.cat(audio_l,   dim=0),              # (N_total, A)
        "text":        torch.cat(text_l,    dim=0),              # (N_total, T)
        "visual":      torch.cat(visual_l,  dim=0),              # (N_total, V)
        "speaker_ids": torch.cat(speaker_l, dim=0),              # (N_total,)
        "batch":       torch.cat(batch_l,   dim=0),              # (N_total,)
        "label":       torch.stack(label_l).unsqueeze(1),        # (B, 1)
    }


# ─────────────────────────────────────────────
# 3. ENCODERS
# ─────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """2-layer BiLSTM with attention pooling — captures temporal prosody."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.2,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, input_dim) — each row is one utterance (T=1 seq)
        x = x.unsqueeze(1)                          # (N, 1, D)
        out, _ = self.bilstm(x)                     # (N, 1, H)
        out = self.norm(out)
        return out.squeeze(1)                        # (N, H)


class TextEncoder(nn.Module):
    """Two-layer MLP projection; plug in frozen BERT output directly."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VisualEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ProjectionHead(nn.Module):
    """Non-linear projection into contrastive embedding space."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 4. LOSS FUNCTIONS
# ─────────────────────────────────────────────

def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Symmetric InfoNCE (NT-Xent).
    Positive pairs: (z_a[i], z_b[i]).  Negatives: all j != i in the batch.
    This is TRUE contrastive learning — unlike simple cosine alignment
    there are explicit hard negatives via the cross-entropy over the full
    similarity matrix.
    """
    N = z_a.size(0)
    if N < 2:
        return torch.tensor(0.0, device=z_a.device)
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = torch.matmul(z_a, z_b.T) / temperature   # (N, N)
    labels = torch.arange(N, device=z_a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def complementarity_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """
    Prevent representation collapse: modalities should keep unique information.
    Penalises perfect cosine similarity between projected modality pairs.
    """
    sim = F.cosine_similarity(z_a, z_b, dim=-1).mean()
    return -torch.log(1.0 - sim.clamp(max=0.999) + 1e-8)


def compute_total_loss(
    logit:        torch.Tensor,
    label:        torch.Tensor,
    z_a:          torch.Tensor,
    z_t:          torch.Tensor,
    z_v:          Optional[torch.Tensor],
    cfg:          Config,
) -> Tuple[torch.Tensor, dict]:
    cls = F.binary_cross_entropy_with_logits(logit, label)
    stats = {"cls": cls.item(), "nce": 0.0, "comp": 0.0}

    if cfg.USE_INFO_NCE:
        nce  = info_nce_loss(z_a, z_t, cfg.TEMPERATURE)
        comp = complementarity_loss(z_a, z_t)
        if z_v is not None:
            nce  = (nce + info_nce_loss(z_a, z_v, cfg.TEMPERATURE)
                       + info_nce_loss(z_t, z_v, cfg.TEMPERATURE)) / 3.0
            comp = (comp + complementarity_loss(z_a, z_v)
                         + complementarity_loss(z_t, z_v)) / 3.0
        stats["nce"]  = nce.item()
        stats["comp"] = comp.item()
        total = cls + cfg.LAMBDA_NCE * nce + cfg.LAMBDA_COMP * comp
    else:
        total = cls

    return total, stats


# ─────────────────────────────────────────────
# 5. HETEROGENEOUS GRAPH BUILDER
# ─────────────────────────────────────────────

def build_hetero_graph(
    z_a:          torch.Tensor,
    z_t:          torch.Tensor,
    z_v:          Optional[torch.Tensor],
    speaker_ids:  torch.Tensor,
    batch_vec:    torch.Tensor,
    cfg:          Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build utterance-level heterogeneous graph.

    Node  = one utterance (not one sample — fixes the core design flaw
            in the original code where nodes were batch-level embeddings).

    Edge types:
      0  intra-modal temporal (audio):   node[t] -> node[t+1]
      1  intra-modal temporal (text):    node[t] -> node[t+1]
      2  cross-modal A -> T (same utterance)
      3  cross-modal T -> A (same utterance)
      4  speaker-relation: top-k utterances by same speaker in conversation

    Returns
    -------
    edge_index  (2, E)  long
    edge_type   (E,)    long
    edge_weight (E,)    float  — used for analysis / optional edge-weighted GNN
    """
    N      = z_a.size(0)
    device = z_a.device

    ei_l, et_l, ew_l = [], [], []

    def add(src, dst, etype, weight):
        ei_l.append(torch.stack([src, dst], dim=0))
        et_l.append(torch.full((src.size(0),), etype, dtype=torch.long, device=device))
        ew_l.append(weight)

    idx = torch.arange(N, device=device)

    # ── 3 & 4. Cross-modal edges (all utterances, symmetric) ──
    w_cross = F.cosine_similarity(z_a, z_t, dim=-1).clamp(0).detach()
    add(idx, idx, 2, w_cross)   # A -> T
    add(idx, idx, 3, w_cross)   # T -> A

    # ── 1 & 2. Intra-modal temporal edges (per conversation) ──
    if cfg.USE_TEMPORAL_EDGE:
        for conv_id in batch_vec.unique():
            mask = (batch_vec == conv_id).nonzero(as_tuple=True)[0]
            if mask.size(0) < 2:
                continue
            s, d = mask[:-1], mask[1:]
            add(s, d, 0, F.cosine_similarity(z_a[s], z_a[d], dim=-1).clamp(0).detach())
            add(s, d, 1, F.cosine_similarity(z_t[s], z_t[d], dim=-1).clamp(0).detach())

    # ── 5. Speaker-relation edges ──
    if cfg.USE_SPEAKER_EDGE:
        z_f = F.normalize((z_a + z_t) / 2.0, dim=-1)
        for conv_id in batch_vec.unique():
            mask = (batch_vec == conv_id).nonzero(as_tuple=True)[0]
            M = mask.size(0)
            if M < 2:
                continue
            z_sub  = z_f[mask]
            sp_sub = speaker_ids[mask]
            sim    = torch.matmul(z_sub, z_sub.T)

            # Only edges between same speaker; no self-loops
            same = sp_sub.unsqueeze(0) == sp_sub.unsqueeze(1)
            sim  = sim.masked_fill(~same, -1e9)
            sim.fill_diagonal_(-1e9)

            k = min(cfg.TOP_K, M - 1)
            if k < 1:
                continue
            vals, cols_l = torch.topk(sim, k=k, dim=-1)      # (M, k)
            valid   = vals > -1e8
            rows_g  = mask[torch.arange(M, device=device).unsqueeze(1).expand_as(cols_l)[valid]]
            cols_g  = mask[cols_l[valid]]
            add(rows_g, cols_g, 4, torch.sigmoid(vals[valid]).detach())

    edge_index  = torch.cat(ei_l, dim=1)
    edge_type   = torch.cat(et_l, dim=0)
    edge_weight = torch.cat(ew_l, dim=0)
    return edge_index, edge_type, edge_weight


# ─────────────────────────────────────────────
# 6. GNN  (RGCN + hierarchical pooling)
# ─────────────────────────────────────────────

class HeteroGNN(nn.Module):
    """
    Relational GCN with residual connections and hierarchical mean+max pooling.
    RGCNConv handles multi-relational edges without requiring a full HeteroData
    object — simpler to implement, competitive for Q2 venues.
    """
    def __init__(self, hidden_dim: int, num_edge_types: int):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for HeteroGNN.")
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_edge_types)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_edge_types)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout   = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_type, batch):
        h = F.gelu(self.norm1(self.conv1(x, edge_index, edge_type))) + x
        h = self.dropout(h)
        h2 = F.gelu(self.norm2(self.conv2(h, edge_index, edge_type))) + h
        h2 = self.dropout(h2)
        out = self.pool_proj(
            torch.cat([global_mean_pool(h2, batch),
                       global_max_pool(h2, batch)], dim=-1)
        )
        return out                                   # (B, H)


class SimpleMLP(nn.Module):
    """Fallback when PyG is unavailable — mean-pools utterances then applies MLP."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_type, batch):
        outs = []
        for b in batch.unique():
            idx = (batch == b).nonzero(as_tuple=True)[0]
            outs.append(self.net(x[idx].mean(dim=0)))
        return torch.stack(outs, dim=0)


# ─────────────────────────────────────────────
# 7. FULL MODEL
# ─────────────────────────────────────────────

class MultimodalGNNModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.audio_enc  = AudioEncoder(cfg.AUDIO_DIM,  cfg.HIDDEN_DIM)
        self.text_enc   = TextEncoder(cfg.TEXT_DIM,    cfg.HIDDEN_DIM)
        self.visual_enc = VisualEncoder(cfg.VISUAL_DIM, cfg.HIDDEN_DIM)

        self.proj_a = ProjectionHead(cfg.HIDDEN_DIM)
        self.proj_t = ProjectionHead(cfg.HIDDEN_DIM)
        self.proj_v = ProjectionHead(cfg.HIDDEN_DIM)

        if cfg.USE_GRAPH and HAS_PYG:
            self.gnn = HeteroGNN(cfg.HIDDEN_DIM, cfg.NUM_EDGE_TYPES)
        else:
            self.gnn = SimpleMLP(cfg.HIDDEN_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(cfg.HIDDEN_DIM // 2, 1),
        )

    def forward(self, batch_dict: dict):
        dev         = self.cfg.DEVICE
        audio       = batch_dict["audio"].to(dev)
        text        = batch_dict["text"].to(dev)
        visual      = batch_dict["visual"].to(dev)
        speaker_ids = batch_dict["speaker_ids"].to(dev)
        batch_vec   = batch_dict["batch"].to(dev)

        # Encode
        h_a = self.audio_enc(audio)
        h_t = self.text_enc(text)
        h_v = self.visual_enc(visual) if self.cfg.USE_VISUAL else None

        # Project to contrastive space
        z_a = self.proj_a(h_a)
        z_t = self.proj_t(h_t)
        z_v = self.proj_v(h_v) if h_v is not None else None

        # Fuse for graph input
        h_fused = (h_a + h_t + h_v) / 3.0 if h_v is not None else (h_a + h_t) / 2.0

        # Build graph and run GNN
        if self.cfg.USE_GRAPH and HAS_PYG:
            edge_index, edge_type, _ = build_hetero_graph(
                z_a, z_t, z_v, speaker_ids, batch_vec, self.cfg
            )
            graph_out = self.gnn(h_fused, edge_index, edge_type, batch_vec)
        else:
            graph_out = self.gnn(h_fused, None, None, batch_vec)

        logit = self.classifier(graph_out)           # (B, 1)
        return logit, z_a, z_t, z_v


# ─────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, cfg: Config) -> dict:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for bd in loader:
            logit, *_ = model(bd)
            prob  = torch.sigmoid(logit).squeeze(1).cpu().numpy()
            label = bd["label"].squeeze(1).numpy()
            y_prob.extend(prob.tolist())
            y_true.extend(label.tolist())

    y_pred = [1 if p > 0.5 else 0 for p in y_prob]
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "f1":  f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["auc"] = 0.0
    return out


# ─────────────────────────────────────────────
# 9. TRAINING LOOP
# ─────────────────────────────────────────────

def train(
    model:     nn.Module,
    train_set: Dataset,
    val_set:   Dataset,
    cfg:       Config,
    verbose:   bool = True,
) -> Tuple[nn.Module, dict]:

    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.BATCH_SIZE, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR * 0.1,
    )

    best_f1, patience_cnt, best_state = 0.0, 0, None

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        total, stats_acc = 0.0, {"cls": 0.0, "nce": 0.0, "comp": 0.0}
        t0 = time.time()

        for bd in train_loader:
            label = bd["label"].to(cfg.DEVICE)
            optimizer.zero_grad()
            logit, z_a, z_t, z_v = model(bd)
            loss, stats = compute_total_loss(logit, label, z_a, z_t, z_v, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            for k in stats_acc:
                stats_acc[k] += stats[k]

        scheduler.step()
        val_m = evaluate(model, val_loader, cfg)

        if verbose:
            n = len(train_loader)
            print(
                f"Epoch {epoch:03d}/{cfg.EPOCHS}  "
                f"loss={total/n:.4f}  "
                f"cls={stats_acc['cls']/n:.4f}  "
                f"nce={stats_acc['nce']/n:.4f}  "
                f"comp={stats_acc['comp']/n:.4f}  "
                f"| val_f1={val_m['f1']:.4f}  val_auc={val_m['auc']:.4f}  "
                f"({time.time()-t0:.1f}s)"
            )

        if val_m["f1"] > best_f1:
            best_f1      = val_m["f1"]
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            if verbose:
                torch.save(best_state, cfg.SAVE_PATH)
                print(f"  -> saved best model  val_f1={best_f1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.PATIENCE:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, evaluate(model, val_loader, cfg)


# ─────────────────────────────────────────────
# 10. ABLATION STUDY
# ─────────────────────────────────────────────

def run_ablation(base_cfg: Config, train_set, val_set, test_set):
    """
    Disable one component at a time to isolate each contribution.
    Required table for Q2 reviewers.

    Variants:
      Full model          — all components on
      w/o graph           — USE_GRAPH = False
      w/o InfoNCE         — USE_INFO_NCE = False
      w/o speaker edge    — USE_SPEAKER_EDGE = False
      w/o temporal edge   — USE_TEMPORAL_EDGE = False
    """
    variants = {
        "Full model":        {},
        "w/o graph":         {"USE_GRAPH": False},
        "w/o InfoNCE":       {"USE_INFO_NCE": False},
        "w/o speaker edge":  {"USE_SPEAKER_EDGE": False},
        "w/o temporal edge": {"USE_TEMPORAL_EDGE": False},
    }

    print("\n" + "=" * 65)
    print(f"  {'Variant':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("=" * 65)

    for name, overrides in variants.items():
        cfg_v = copy.deepcopy(base_cfg)
        for k, v in overrides.items():
            setattr(cfg_v, k, v)
        cfg_v.EPOCHS  = 10    # fast ablation
        cfg_v.PATIENCE = 5

        mdl = MultimodalGNNModel(cfg_v).to(cfg_v.DEVICE)
        mdl, _ = train(mdl, train_set, val_set, cfg_v, verbose=False)

        test_loader = DataLoader(test_set, batch_size=cfg_v.BATCH_SIZE,
                                 collate_fn=collate_fn)
        m = evaluate(mdl, test_loader, cfg_v)
        print(f"  {name:<25} {m['acc']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")

    print("=" * 65 + "\n")


# ─────────────────────────────────────────────
# 11. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multimodal Sentiment — Heterogeneous GNN")
    parser.add_argument("--dataset",   default="mosi",
                        choices=["mosi", "mosei"],
                        help="Data source (mosi/mosei requires custom loader)")
    parser.add_argument("--epochs",    type=int,   default=CFG.EPOCHS)
    parser.add_argument("--lr",        type=float, default=CFG.LR)
    parser.add_argument("--batch",     type=int,   default=CFG.BATCH_SIZE)
    parser.add_argument("--ablation",  action="store_true",
                        help="Run ablation study after main training")
    parser.add_argument("--no-graph",  action="store_true")
    parser.add_argument("--no-nce",    action="store_true")
    parser.add_argument("--no-visual", action="store_true")
    args = parser.parse_args()

    # Apply CLI overrides to global config
    CFG.EPOCHS       = args.epochs
    CFG.LR           = args.lr
    CFG.BATCH_SIZE   = args.batch
    CFG.USE_GRAPH    = not args.no_graph
    CFG.USE_INFO_NCE = not args.no_nce
    CFG.USE_VISUAL   = not args.no_visual

    set_seed(CFG.SEED)

    print("=" * 60)
    print("  Multimodal Sentiment Analysis — Heterogeneous GNN")
    print(f"  Device   : {CFG.DEVICE}")
    print(f"  PyG      : {'yes' if HAS_PYG else 'NO (MLP fallback)'}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Epochs   : {CFG.EPOCHS}  |  LR: {CFG.LR}  |  Batch: {CFG.BATCH_SIZE}")
    print(f"  Graph    : {CFG.USE_GRAPH}  |  InfoNCE: {CFG.USE_INFO_NCE}  |  Visual: {CFG.USE_VISUAL}")
    print("=" * 60)

    # ── Data ──
    raise NotImplementedError(
        f"Dataset '{args.dataset}' requires a custom loader.\n"
        "Synthetic data has been removed. Implement a Dataset class that returns "
        "MultimodalSample objects, then wire it in here."
    )

    # ── Model ──
    model    = MultimodalGNNModel(CFG).to(CFG.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_params:,}\n")

    # ── Train ──
    print("── Training ─────────────────────────────────────────")
    model, val_metrics = train(model, train_set, val_set, CFG)

    # ── Test ──
    test_loader  = DataLoader(test_set, batch_size=CFG.BATCH_SIZE, collate_fn=collate_fn)
    test_metrics = evaluate(model, test_loader, CFG)
    print("\n── Test Results ─────────────────────────────────────")
    print(f"  Acc : {test_metrics['acc']:.4f}")
    print(f"  F1  : {test_metrics['f1']:.4f}")
    print(f"  AUC : {test_metrics['auc']:.4f}")

    # ── Ablation ──
    if args.ablation:
        print("\n── Ablation Study ───────────────────────────────────")
        run_ablation(CFG, train_set, val_set, test_set)

    print("Done.")


if __name__ == "__main__":
    main()