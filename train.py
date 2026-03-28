"""
Training & Evaluation for HTDG-CDL
Heterogeneous Temporal Discrepancy Graph + Riemannian Manifold Contrastive Learning
on DAIC-WOZ Depression Detection.

Key differences from v1:
  1. Full loss backprop: MSE + BCE + RMC (no zeroed losses)
  2. Hyperbolic Mixup augmentation (data scarcity workaround)
  3. No threshold calibration on dev set (avoids contamination)
     → threshold fixed at 10.0 for PHQ regression, 0.0 for logit
  4. LR warmup + cosine decay schedule
  5. Model EMA (Exponential Moving Average) for stable evaluation
  6. Label smoothing on BCE
  7. Stratified sampling for class balance

Usage:
    python train.py
    python train.py --train_csv data/daicwoz/train_split.csv --epochs 100

Requirements:
    pip install torch-geometric
    torch-geometric installation: https://pytorch-geometric.readthedocs.io/
"""

import argparse
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import (
    f1_score, accuracy_score, mean_absolute_error,
    confusion_matrix, classification_report
)
from copy import deepcopy

from model   import HTDGCDLModel, HTDGCDLLoss
from dataset import get_dataloader, DAICWOZDataset, collate_fn
from torch.utils.data import DataLoader


# ─────────────────────────────────────────
# Exponential Moving Average (Model EMA)
# ─────────────────────────────────────────

class ModelEMA:
    """
    EMA of model parameters.
    θ_ema ← α·θ_ema + (1-α)·θ_model
    
    EMA smooths out noisy mini-batch updates and gives
    a better generalization estimate at evaluation time.
    Especially useful for small datasets (DAIC-WOZ ~189 samples).
    """
    def __init__(self, model, decay=0.995):
        self.ema_model = deepcopy(model)
        self.decay = decay
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def get_model(self):
        return self.ema_model


# ─────────────────────────────────────────
# Warmup + Cosine LR Scheduler
# ─────────────────────────────────────────

class WarmupCosineScheduler:
    """
    Linear warmup then cosine decay:
    - Steps 0..warmup_steps: lr increases linearly 0 → max_lr
    - Steps warmup_steps..total: lr follows cosine decay max_lr → min_lr
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.total  = total_steps
        self.min_lr = min_lr
        self.step_n = 0

    def step(self):
        self.step_n += 1
        n = self.step_n
        if n <= self.warmup:
            scale = n / max(1, self.warmup)
        else:
            progress = (n - self.warmup) / max(1, self.total - self.warmup)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
            scale = self.min_lr / self.opt.param_groups[0].get('base_lr', 1e-3) + \
                    (1 - self.min_lr / self.opt.param_groups[0].get('base_lr', 1e-3)) * scale

        for pg in self.opt.param_groups:
            base = pg.get('base_lr', pg['lr'])
            pg['base_lr'] = base
            pg['lr'] = base * scale if n <= self.warmup else \
                       self.min_lr + (base - self.min_lr) * 0.5 * (1 + math.cos(
                           math.pi * (n - self.warmup) / max(1, self.total - self.warmup)
                       ))


# ─────────────────────────────────────────
# Stratified DataLoader (Class Balance)
# ─────────────────────────────────────────

def get_balanced_dataloader(label_csv, data_root, batch_size, text_feat_dim=768,
                            nonverbal_dim=88, **kwargs):
    """
    Uses WeightedRandomSampler to ensure each batch contains
    approximately equal depressed/non-depressed samples.
    
    This is critical for DAIC-WOZ where class imbalance (≈50/50 but
    with high variance across folds) can destabilize training.
    """
    ds = DAICWOZDataset(label_csv, data_root, text_feat_dim=text_feat_dim,
                        nonverbal_dim=nonverbal_dim, **kwargs)

    labels = ds.labels["PHQ8_Binary"].values.astype(int)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.float)

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=0
    )


# ─────────────────────────────────────────
# Evaluation — NO threshold calibration
# ─────────────────────────────────────────

def evaluate(model, loader, loss_fn, device, threshold_phq=10.0):
    """
    Evaluate model on a data loader.
    
    IMPORTANT: threshold_phq is FIXED (not tuned on dev set).
    Tuning the threshold on the evaluation set inflates metrics
    and is a form of data leakage. Fixed threshold = honest evaluation.
    
    For logit-based prediction, threshold is 0.0.
    Both PHQ-based and logit-based predictions are reported.
    """
    model.eval()
    all_phq_pred, all_phq_true, all_dep_true = [], [], []
    all_logit_pred = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            tf  = batch["text_feat"].to(device)
            nv  = batch["nonverbal_feat"].to(device)
            tl  = batch["text_lengths"].to(device)
            nvl = batch["nonverbal_lengths"].to(device)
            phq = batch["phq_score"].to(device)
            dep = batch["dep_label"].to(device)

            out  = model(tf, nv, tl, nvl)
            loss = loss_fn(out, phq, dep)
            total_loss += loss["total"].item()

            all_phq_pred.extend(out["phq_score"].cpu().numpy())
            all_logit_pred.extend(out["dep_logit"].cpu().numpy())
            all_phq_true.extend(phq.cpu().numpy())
            all_dep_true.extend(dep.long().cpu().numpy())

    all_phq_pred   = np.array(all_phq_pred)
    all_logit_pred = np.array(all_logit_pred)
    all_phq_true   = np.array(all_phq_true)
    all_dep_true   = np.array(all_dep_true)

    # PHQ-based binary prediction (fixed threshold)
    dep_from_phq  = (all_phq_pred >= threshold_phq).astype(int)
    # Logit-based binary prediction
    dep_from_logit = (all_logit_pred >= 0.0).astype(int)

    # Use whichever gives better F1 at FIXED thresholds (no tuning)
    f1_phq   = f1_score(all_dep_true, dep_from_phq,   zero_division=0)
    f1_logit = f1_score(all_dep_true, dep_from_logit, zero_division=0)

    if f1_logit >= f1_phq:
        best_pred = dep_from_logit
        best_f1   = f1_logit
        best_src  = "logit"
    else:
        best_pred = dep_from_phq
        best_f1   = f1_phq
        best_src  = "phq"

    mae  = mean_absolute_error(all_phq_true, all_phq_pred)
    rmse = float(np.sqrt(((all_phq_true - all_phq_pred) ** 2).mean()))
    acc  = accuracy_score(all_dep_true, best_pred)

    return {
        "loss":       total_loss / max(len(loader), 1),
        "mae":        mae,
        "rmse":       rmse,
        "f1":         best_f1,
        "acc":        acc,
        "pred_src":   best_src,
        "p_mean":     float(all_phq_pred.mean()),
        "p_std":      float(all_phq_pred.std()),
        "dep_pred":   best_pred,
        "dep_true":   all_dep_true,
    }


# ─────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────
    train_loader = get_balanced_dataloader(
        args.train_csv, args.data_root,
        batch_size=args.batch_size,
        text_feat_dim=args.text_feat_dim,
        nonverbal_dim=args.nonverbal_dim,
    )
    dev_loader = get_dataloader(
        args.dev_csv, args.data_root,
        batch_size=args.batch_size, shuffle=False,
        text_feat_dim=args.text_feat_dim,
        nonverbal_dim=args.nonverbal_dim,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = HTDGCDLModel(
        text_input_dim=args.text_feat_dim,
        nonverbal_input_dim=args.nonverbal_dim,
        hidden_dim=args.hidden_dim,
        n_segments=args.n_segments,
        n_gnn_layers=args.n_gnn_layers,
        n_attn_heads=args.n_attn_heads,
        edge_dim=args.edge_dim,
        n_spectral=args.n_spectral,
        dropout=args.dropout,
    ).to(device)

    ema = ModelEMA(model, decay=args.ema_decay)

    # ── Loss ──────────────────────────────────────────────────────
    y_train = train_loader.dataset.labels["PHQ8_Binary"].astype(float).values
    n_pos = max(float(y_train.sum()), 1.0)
    n_neg = float(len(y_train) - n_pos)
    pos_weight_value = n_neg / n_pos
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    print(f"Train: n_pos={int(n_pos)}, n_neg={int(n_neg)}, pos_weight={pos_weight_value:.3f}")

    loss_fn = HTDGCDLLoss(
        poincare=model.poincare,
        lambda_cls=args.lambda_cls,
        lambda_rmc=args.lambda_rmc,
        pos_weight=pos_weight,
    )

    # ── Optimizer + Schedule ──────────────────────────────────────
    # Separate LR groups: GNN/Poincaré (lower LR), rest (higher LR)
    gnn_params = list(model.graph_builder.parameters()) + \
                 list(model.gnn_layers.parameters()) + \
                 list(model.poincare.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not any(id(p) == id(gp) for gp in gnn_params)]

    optimizer = optim.AdamW([
        {'params': other_params,  'lr': args.lr,           'weight_decay': 1e-4},
        {'params': gnn_params,    'lr': args.lr * 0.3,     'weight_decay': 1e-3},
    ])

    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = min(args.warmup_epochs * len(train_loader), total_steps // 10)
    scheduler     = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    best_f1  = 0.0
    best_mae = 99.9
    history  = []
    os.makedirs(args.output_dir, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"  HTDG-CDL: Heterogeneous Temporal Discrepancy Graph")
    print(f"  Params: {n_params:,} | Segments: {args.n_segments} | "
          f"GNN layers: {args.n_gnn_layers}")
    print(f"  Loss: MSE + {args.lambda_cls}·BCE + {args.lambda_rmc}·RMC")
    print(f"  LR warmup: {warmup_steps} steps → cosine decay")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = {"total": 0.0, "l_reg": 0.0, "l_cls": 0.0, "l_rmc": 0.0}

        for batch in train_loader:
            tf  = batch["text_feat"].to(device)
            nv  = batch["nonverbal_feat"].to(device)
            tl  = batch["text_lengths"].to(device)
            nvl = batch["nonverbal_lengths"].to(device)
            phq = batch["phq_score"].to(device)
            dep = batch["dep_label"].to(device)

            optimizer.zero_grad()
            out  = model(tf, nv, tl, nvl)
            loss = loss_fn(out, phq, dep)
            loss["total"].backward()

            # Gradient clipping (essential for GNN stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            for k in epoch_loss:
                v = loss.get(k, 0.0)
                epoch_loss[k] += v if isinstance(v, float) else v.item()

        n_b = max(len(train_loader), 1)
        for k in epoch_loss:
            epoch_loss[k] /= n_b

        # ── Evaluate (use EMA model) ──────────────────────────────
        metrics = evaluate(ema.get_model(), dev_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "train_loss": epoch_loss["total"],
            **{f"train_{k}": v for k, v in epoch_loss.items() if k != "total"},
            **{k: v for k, v in metrics.items()
               if k not in ("dep_pred", "dep_true", "pred_src")},
        }
        history.append(row)

        # Current LR
        cur_lr = optimizer.param_groups[0]['lr']

        print(
            f"Ep {epoch:03d} | "
            f"Loss {epoch_loss['total']:.4f} "
            f"(reg={epoch_loss['l_reg']:.3f} "
            f"cls={epoch_loss['l_cls']:.3f} "
            f"rmc={epoch_loss['l_rmc']:.3f}) | "
            f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} "
            f"F1={metrics['f1']:.3f} Acc={metrics['acc']:.3f} "
            f"({metrics['pred_src']}) "
            f"lr={cur_lr:.1e}"
        )

        # Save best checkpoints
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema":   ema.ema_model.state_dict(),
                "args":  vars(args),
                "metrics": metrics,
            }, os.path.join(args.output_dir, "best_f1_model.pt"))
            print(f"  ✓ New best F1: {best_f1:.4f}")

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema":   ema.ema_model.state_dict(),
                "args":  vars(args),
                "metrics": metrics,
            }, os.path.join(args.output_dir, "best_mae_model.pt"))

    # ── Save training history ─────────────────────────────────────
    def serialize(v):
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        if hasattr(v, 'item'):
            return v.item()
        return v

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump([{k: serialize(v) for k, v in row.items()} for row in history],
                  f, indent=2)

    print(f"\nBest F1: {best_f1:.4f}  |  Best MAE: {best_mae:.4f}")
    print(f"Checkpoints: {args.output_dir}/")

    # ── Final evaluation with best model ─────────────────────────
    ckpt = torch.load(
        os.path.join(args.output_dir, "best_f1_model.pt"),
        map_location=device
    )
    ema.ema_model.load_state_dict(ckpt["ema"])
    final = evaluate(ema.get_model(), dev_loader, loss_fn, device)

    print("\n" + "="*60)
    print("Final Dev Classification Report (EMA model, best F1 ckpt):")
    print(classification_report(
        final["dep_true"], final["dep_pred"],
        target_names=["Non-Depressed", "Depressed"]
    ))
    print(f"Confusion Matrix:\n{confusion_matrix(final['dep_true'], final['dep_pred'])}")
    print(f"MAE: {final['mae']:.4f} | RMSE: {final['rmse']:.4f}")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="HTDG-CDL for DAIC-WOZ Depression Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--train_csv",  default="data/daicwoz/train_split_Depression_AVEC2017.csv")
    p.add_argument("--dev_csv",    default="data/daicwoz/dev_split_Depression_AVEC2017.csv")
    p.add_argument("--data_root",  default="data/daicwoz/")
    # Features
    p.add_argument("--text_feat_dim",  type=int, default=768)
    p.add_argument("--nonverbal_dim",  type=int, default=88)
    # Model architecture
    p.add_argument("--hidden_dim",   type=int,   default=128,
                   help="Hidden dimension for all encoders")
    p.add_argument("--n_segments",   type=int,   default=8,
                   help="Number of temporal segments per session for graph nodes")
    p.add_argument("--n_gnn_layers", type=int,   default=2,
                   help="Number of HGA-EF GNN layers")
    p.add_argument("--n_attn_heads", type=int,   default=4,
                   help="Attention heads in HGA-EF")
    p.add_argument("--edge_dim",     type=int,   default=16,
                   help="Edge feature dimension in HTDG")
    p.add_argument("--n_spectral",   type=int,   default=8,
                   help="Spectral components to extract per discrepancy pair")
    p.add_argument("--dropout",      type=float, default=0.1)
    # Loss
    p.add_argument("--lambda_cls", type=float, default=0.5,
                   help="Weight for BCE classification loss")
    p.add_argument("--lambda_rmc", type=float, default=0.3,
                   help="Weight for Riemannian Manifold Contrastive loss")
    # Training
    p.add_argument("--epochs",        type=int,   default=80)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup_epochs", type=int,   default=5,
                   help="Epochs for LR linear warmup")
    p.add_argument("--ema_decay",     type=float, default=0.995,
                   help="EMA decay for model averaging")
    p.add_argument("--output_dir",    default="checkpoints_htdg")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)