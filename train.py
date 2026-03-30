"""
train.py

Training script for HMSG-Net.

Configuration constants are defined at the top of this file.
Run with:
    python train.py                    # default: E-DAIC
    python train.py --dataset daicwoz  # DAIC-WOZ corpus
    python train.py --dataset edaic    # E-DAIC corpus (explicit)
"""

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch_geometric.loader import DataLoader

from edaic_dataset import DepressionDataset
from daicwoz_dataset import DaicWozDataset
from model import HMSGNet, compute_loss

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Reproducibility
SEED = 42

# Dataset-specific settings (keyed by --dataset value)
# Each entry fully specifies model arch, training HP, and feature dims.
DATASET_CFG = {
    "edaic": {
        # Paths
        "cache_dir":      "C:/Users/ezycloudx-admin/Desktop/thgnn/cache",
        "checkpoint_dir": Path("C:/Users/ezycloudx-admin/Desktop/thgnn/checkpoints"),
        # Feature dims (2-modal: BERT text + wav2vec audio)
        "text_dim":  768,
        "audio_dim": 768,   # wav2vec2
        # Model
        "hidden_dim":      256,
        "num_gnn_layers":  3,
        "n_heads":         4,
        "dropout":         0.35,
        "drop_edge":       0.10,
        "feat_noise":      0.02,
        # Loss
        "focal_alpha":     0.80,   # E-DAIC ~21% positive — upweight heavily
        "w_symptom":       0.3,
        "w_phq":           0.1,
        # Optimiser
        "lr":              3e-4,
        "weight_decay":    2e-4,
        # Scheduler
        "warmup_epochs":   5,
        "cosine_epochs":   250,
        "eta_min":         1e-6,
        # Training loop
        "batch_size":      8,
        "max_epochs":      250,
        "early_stop_pat":  30,
    },
    "daicwoz": {
        # Paths
        "cache_dir":      "C:/Users/ezycloudx-admin/Desktop/thgnn/cache_daicwoz",
        "checkpoint_dir": Path("C:/Users/ezycloudx-admin/Desktop/thgnn/checkpoints_daicwoz"),
        # Feature dims (2-modal: BERT text + wav2vec audio)
        "text_dim":  768,
        "audio_dim": 768,   # wav2vec2
        # Model
        "hidden_dim":      256,
        "num_gnn_layers":  3,
        "n_heads":         4,
        "dropout":         0.45,
        "drop_edge":       0.20,
        "feat_noise":      0.03,
        # Loss
        "focal_alpha":     0.72,
        "label_smoothing": 0.0,
        "w_symptom":       0.05,
        "w_phq":           0.05,
        # Optimiser
        "lr":              2e-4,
        "weight_decay":    5e-4,
        # Scheduler
        "warmup_epochs":   8,
        "cosine_epochs":   300,
        "eta_min":         1e-6,
        # Training loop
        "batch_size":      8,
        "max_epochs":      300,
        "early_stop_pat":  50,
        "use_weighted_sampler": False,
    },
}

# Kept for backward-compat reference; actual values come from DATASET_CFG
NUM_SYMPTOMS   = 8
NUM_EDGE_TYPES = 9
MAX_GRAD_NORM  = 1.0

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Sweep thresholds [0.30, 0.70] and return the one maximising F1-macro.
    Range is intentionally constrained: on small dev sets (≤50 samples) a wider
    sweep will overfit to noise and return clinically meaningless thresholds
    (e.g. 0.10 that predicts nearly all positive).
    """
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.02):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def compute_metrics(
    labels: np.ndarray,
    probs:  np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics from predicted probabilities."""
    preds = (probs >= threshold).astype(int)
    metrics: Dict[str, float] = {}
    metrics["accuracy"]    = accuracy_score(labels, preds)
    metrics["f1_macro"]    = f1_score(labels, preds, average="macro",     zero_division=0)
    metrics["f1_weighted"] = f1_score(labels, preds, average="weighted",  zero_division=0)
    metrics["precision"]   = precision_score(labels, preds, zero_division=0)
    metrics["recall"]      = recall_score(labels, preds, zero_division=0)
    metrics["threshold"]   = threshold
    try:
        metrics["auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# One epoch (train or eval)
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:           HMSGNet,
    loader:          DataLoader,
    optimizer:       Optional[torch.optim.Optimizer],
    device:          torch.device,
    is_train:        bool,
    w_symptom:       float = 0.3,
    w_phq:           float = 0.1,
    focal_alpha:     float = 0.80,
    label_smoothing: float = 0.0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Run one full pass over `loader`.

    Returns
    -------
    loss_dict  : dict with averaged loss components
    all_labels : (N,) integer array
    all_probs  : (N,) float array of sigmoid probabilities
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_losses: Dict[str, float] = {
        "loss_total": 0.0, "loss_dep": 0.0, "loss_symptom": 0.0, "loss_phq": 0.0
    }
    n_batches = 0
    all_labels: list = []
    all_probs:  list = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in loader:
            batch = batch.to(device)

            dep_logit, sym_logits, phq_pred = model(batch)

            dep_labels  = batch.y.squeeze().long()
            phq8_labels = batch.phq8.view(-1, NUM_SYMPTOMS).float()
            phq_scores  = batch.phq_score.squeeze().float()

            loss, ld = compute_loss(
                dep_logit, sym_logits, phq_pred,
                dep_labels.float(), phq8_labels, phq_scores,
                w_symptom=w_symptom, w_phq=w_phq,
                focal_alpha=focal_alpha,
                label_smoothing=label_smoothing,
                device=device,
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            for k in total_losses:
                total_losses[k] += ld.get(k, 0.0)
            n_batches += 1

            probs = torch.sigmoid(dep_logit).detach().cpu().numpy()
            lbls  = dep_labels.detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbls.tolist())

    if n_batches > 0:
        for k in total_losses:
            total_losses[k] /= n_batches

    return (
        total_losses,
        np.array(all_labels, dtype=int),
        np.array(all_probs,  dtype=float),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler factory: linear warmup → cosine annealing
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer:         torch.optim.Optimizer,
    num_warmup_epochs: int,
    num_cosine_epochs: int,
    eta_min:           float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Builds a SequentialLR composed of:
      1. Linear warmup from 0 → 1 over `num_warmup_epochs` epochs.
      2. CosineAnnealingLR for `num_cosine_epochs` epochs.
    """
    def warmup_lambda(epoch: int) -> float:
        if epoch < num_warmup_epochs:
            return float(epoch + 1) / float(max(1, num_warmup_epochs))
        return 1.0

    warmup_sched  = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_sched  = CosineAnnealingLR(
        optimizer, T_max=num_cosine_epochs, eta_min=eta_min
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[num_warmup_epochs],
    )
    return scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:     HMSGNet,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    best_f1:   float,
    path:      Path,
) -> None:
    torch.save(
        {
            "epoch":      epoch,
            "best_f1":    best_f1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        str(path),
    )


def load_checkpoint(
    model:     HMSGNet,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    path:      Path,
    device:    torch.device,
) -> Tuple[int, float]:
    """Load checkpoint and return (start_epoch, best_f1)."""
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_f1", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────

def log_epoch(
    epoch:       int,
    phase:       str,
    losses:      Dict[str, float],
    metrics:     Dict[str, float],
    lr:          float,
    elapsed:     float,
) -> None:
    print(
        f"[Epoch {epoch:03d}] {phase:5s} | "
        f"loss={losses['loss_total']:.4f} "
        f"(dep={losses['loss_dep']:.4f} "
        f"sym={losses['loss_symptom']:.4f} "
        f"phq={losses['loss_phq']:.4f}) | "
        f"acc={metrics.get('accuracy', 0):.4f} "
        f"f1_macro={metrics.get('f1_macro', 0):.4f} "
        f"f1_w={metrics.get('f1_weighted', 0):.4f} "
        f"prec={metrics.get('precision', 0):.4f} "
        f"rec={metrics.get('recall', 0):.4f} "
        f"auc={metrics.get('auc', float('nan')):.4f} "
        f"thr={metrics.get('threshold', 0.5):.2f} | "
        f"lr={lr:.2e} | {elapsed:.1f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train HMSG-Net")
    parser.add_argument(
        "--dataset",
        choices=["edaic", "daicwoz"],
        default="edaic",
        help="Dataset to train on: 'edaic' (default) or 'daicwoz'",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Re-process dataset from scratch (clears cache)",
    )
    args = parser.parse_args()

    cfg = DATASET_CFG[args.dataset]

    # ── Unpack all config values ──────────────────────────────────────────────
    cache_dir      = cfg["cache_dir"]
    checkpoint_dir = cfg["checkpoint_dir"]
    text_dim       = cfg["text_dim"]
    audio_dim      = cfg["audio_dim"]
    hidden_dim     = cfg["hidden_dim"]
    num_gnn_layers = cfg["num_gnn_layers"]
    n_heads        = cfg["n_heads"]
    dropout        = cfg["dropout"]
    drop_edge      = cfg["drop_edge"]
    feat_noise     = cfg["feat_noise"]
    focal_alpha          = cfg["focal_alpha"]
    label_smoothing      = cfg.get("label_smoothing", 0.0)
    w_symptom            = cfg["w_symptom"]
    w_phq                = cfg["w_phq"]
    use_weighted_sampler = cfg.get("use_weighted_sampler", False)
    lr             = cfg["lr"]
    weight_decay   = cfg["weight_decay"]
    warmup_epochs  = cfg["warmup_epochs"]
    cosine_epochs  = cfg["cosine_epochs"]
    eta_min        = cfg["eta_min"]
    batch_size     = cfg["batch_size"]
    max_epochs     = cfg["max_epochs"]
    early_stop_pat = cfg["early_stop_pat"]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pt"

    set_seed(SEED)
    print(f"Dataset: {args.dataset}  |  Device: {DEVICE}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print("Loading datasets …")
    if args.dataset == "edaic":
        train_ds = DepressionDataset(split="train", root=cache_dir,
                                     force_reload=args.force_reload)
        dev_ds   = DepressionDataset(split="dev",   root=cache_dir,
                                     force_reload=args.force_reload)
    else:
        train_ds = DaicWozDataset(split="train", root=cache_dir,
                                   force_reload=args.force_reload)
        dev_ds   = DaicWozDataset(split="dev",   root=cache_dir,
                                   force_reload=args.force_reload)

    if use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        labels_arr  = train_ds.labels().numpy()
        class_count = np.bincount(labels_arr)
        sample_w    = (1.0 / class_count)[labels_arr]   # per-sample weight
        sampler     = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).float(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=0, pin_memory=(DEVICE.type == "cuda"), drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(DEVICE.type == "cuda"), drop_last=False,
        )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

    print(f"  Train: {len(train_ds)} graphs,  Dev: {len(dev_ds)} graphs")
    label_counts = torch.bincount(train_ds.labels())
    print(f"  Train class distribution: neg={label_counts[0]}, pos={label_counts[1]}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HMSGNet(
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        n_heads=n_heads,
        dropout=dropout,
        num_symptoms=NUM_SYMPTOMS,
        num_edge_types=NUM_EDGE_TYPES,
        drop_edge=drop_edge,
        feat_noise=feat_noise,
        text_dim=text_dim,
        audio_dim=audio_dim,
    ).to(DEVICE)

    unified_dim = text_dim + audio_dim
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}  (unified_dim={unified_dim})")
    print(f"  focal_alpha={focal_alpha}  label_smooth={label_smoothing}  "
          f"dropout={dropout}  drop_edge={drop_edge}  lr={lr:.1e}  "
          f"batch={batch_size}  patience={early_stop_pat}  "
          f"weighted_sampler={use_weighted_sampler}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        num_warmup_epochs=warmup_epochs,
        num_cosine_epochs=cosine_epochs,
        eta_min=eta_min,
    )

    # Resume if checkpoint exists (skip when force-reload clears data)
    start_epoch = 1
    best_dev_f1 = 0.0
    if best_model_path.exists() and not args.force_reload:
        print(f"  Resuming from checkpoint: {best_model_path}")
        start_epoch, best_dev_f1 = load_checkpoint(
            model, optimizer, scheduler, best_model_path, DEVICE
        )
        start_epoch += 1
        print(f"  Resuming at epoch {start_epoch}, best_dev_f1={best_dev_f1:.4f}")

    no_improve_count = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        # --- Train ---
        train_losses, train_labels, train_probs = run_epoch(
            model, train_loader, optimizer, DEVICE, is_train=True,
            w_symptom=w_symptom, w_phq=w_phq, focal_alpha=focal_alpha,
            label_smoothing=label_smoothing,
        )
        train_metrics = compute_metrics(train_labels, train_probs, threshold=0.5)
        lr_now = optimizer.param_groups[0]["lr"]
        log_epoch(epoch, "TRAIN", train_losses, train_metrics, lr_now,
                  time.time() - t0)

        # --- Dev (with threshold optimisation) ---
        t1 = time.time()
        dev_losses, dev_labels, dev_probs = run_epoch(
            model, dev_loader, None, DEVICE, is_train=False,
            w_symptom=w_symptom, w_phq=w_phq, focal_alpha=focal_alpha,
            label_smoothing=0.0,   # no smoothing at eval
        )
        best_thr = find_best_threshold(dev_labels, dev_probs)
        dev_metrics = compute_metrics(dev_labels, dev_probs, threshold=best_thr)
        log_epoch(epoch, "DEV", dev_losses, dev_metrics, lr_now,
                  time.time() - t1)

        scheduler.step()

        # --- Early stopping & checkpointing ---
        # Use AUC as primary criterion: more robust than threshold-optimised F1
        # on small dev sets (35 samples), where threshold search can overfit.
        dev_f1  = dev_metrics.get("f1_macro", 0.0)
        dev_auc = dev_metrics.get("auc", 0.0)
        if dev_auc > best_dev_f1:   # best_dev_f1 now tracks best AUC
            best_dev_f1 = dev_auc
            no_improve_count = 0
            ckpt_data = {
                "epoch": epoch, "best_auc": best_dev_f1, "best_f1": dev_f1,
                "best_threshold": best_thr,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(ckpt_data, str(best_model_path))
            print(f"  * New best dev AUC={best_dev_f1:.4f} "
                  f"(F1={dev_f1:.4f} thr={best_thr:.2f}) -> saved")
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{early_stop_pat}). "
                  f"Best dev AUC = {best_dev_f1:.4f}")
            if no_improve_count >= early_stop_pat:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best dev AUC = {best_dev_f1:.4f}")
    print(f"Best model saved at: {best_model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
