"""
train.py

Training script for HMSG-Net on DAIC-WOZ.

Configuration constants are defined at the top of this file.

Run:
    python train.py --dataset daicwoz
    python train.py --dataset daicwoz --context_mode none       # anti-shortcut
    python train.py --dataset daicwoz --context_mode full       # shortcut baseline
    python train.py --dataset daicwoz --multi_seed              # variance report

FIXES vs original:
  1. [FIX-SHORTCUT]     --context_mode flag: none / truncated / full
                        Passes to DaicWozDataset, ablates interviewer influence.
  2. [FIX-VARIANCE]     --multi_seed flag: train with N seeds (default 5),
                        report mean ± std for F1/AUC. Required for small datasets.
  3. [FIX-VARIANCE]     Per-run JSON log with full metrics per epoch → easy to
                        aggregate results across seeds in post-hoc analysis.
  4. [FIX-THRESHOLD]    find_best_threshold now constrained to [0.30, 0.70] with
                        step 0.02. On dev sets of ~35 samples a wider sweep
                        overfits to noise → meaningless thresholds.
  5. [FIX-REPORTING]    print_final_summary() prints a paper-ready table with
                        mean±std across seeds for all key metrics.
  6. [FIX-EDAIC-STUB]   E-DAIC config kept but clearly marked TODO for future.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

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

from daicwoz_dataset import DaicWozDataset
from model import HMSGNet, compute_loss

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_SEED  = 42
MULTI_SEEDS = [42, 123, 456, 789, 2024]   # 5 seeds for variance reporting

DATASET_CFG = {
    # ── E-DAIC (TODO: implement after DAIC-WOZ is solid) ──────────────────────
    "edaic": {
        "cache_dir":      "C:/Users/Administrator/Desktop/thgnn/cache",
        "checkpoint_dir": Path("C:/Users/Administrator/Desktop/thgnn/checkpoints"),
        "text_dim":  768,
        "audio_dim": 768,
        "hidden_dim":      256,
        "num_gnn_layers":  3,
        "n_heads":         4,
        "dropout":         0.35,
        "drop_edge":       0.10,
        "feat_noise":      0.02,
        "focal_alpha":     0.80,
        "w_symptom":       0.3,
        "w_phq":           0.1,
        "lr":              3e-4,
        "weight_decay":    2e-4,
        "warmup_epochs":   5,
        "cosine_epochs":   250,
        "eta_min":         1e-6,
        "batch_size":      8,
        "max_epochs":      250,
        "early_stop_pat":  30,
    },
    # ── DAIC-WOZ ────────────────────────────────────────────────────────────
    "daicwoz": {
        "cache_dir":      "C:/Users/Administrator/Desktop/thgnn/cache_daicwoz",
        "checkpoint_dir": Path("C:/Users/Administrator/Desktop/thgnn/checkpoints_daicwoz"),
        "text_dim":  776,    # BERT(768) + prosodic(8)
        "audio_dim": 776,    # wav2vec(768) + acoustic(8)
        # Tiny model justified by ~107 training samples
        "hidden_dim":      64,
        "num_gnn_layers":  1,
        "n_heads":         2,   # head_dim = 32
        "dropout":         0.70,
        "drop_edge":       0.40,
        "feat_noise":      0.05,
        # Loss — pure dep focal; aux losses add noise with few samples
        "focal_alpha":     0.75,
        "label_smoothing": 0.05,
        "w_symptom":       0.0,
        "w_phq":           0.0,
        # Optimiser
        "lr":              5e-5,
        "weight_decay":    5e-3,
        # Scheduler
        "warmup_epochs":   1,
        "cosine_epochs":   300,
        "eta_min":         1e-7,
        # Training loop
        "batch_size":      8,
        "max_epochs":      300,
        "early_stop_pat":  60,
        "use_weighted_sampler": True,
    },
}

NUM_SYMPTOMS   = 8
NUM_EDGE_TYPES = 4
MAX_GRAD_NORM  = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Output teeing
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)
        sys.stderr = _Tee(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


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
    """
    FIX-THRESHOLD: Sweep [0.30, 0.70] with step 0.02 and return the threshold
    that maximises F1-macro.

    The range is INTENTIONALLY constrained:
    - On small dev sets (~35 samples) a wider sweep [0.10, 0.90] overfits to
      noise, returning thresholds like 0.12 that predict nearly all positive.
    - The constrained range forces the model to learn a well-calibrated score
      rather than relying on threshold tricks.
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
    preds = (probs >= threshold).astype(int)
    metrics: Dict[str, float] = {}
    metrics["accuracy"]    = accuracy_score(labels, preds)
    metrics["f1_macro"]    = f1_score(labels, preds, average="macro",    zero_division=0)
    metrics["f1_weighted"] = f1_score(labels, preds, average="weighted", zero_division=0)
    metrics["precision"]   = precision_score(labels, preds, zero_division=0)
    metrics["recall"]      = recall_score(labels, preds, zero_division=0)
    metrics["threshold"]   = threshold
    try:
        metrics["auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc"] = float("nan")

    # Confusion matrix — useful for clinical analysis (FP/FN tradeoff)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["tp"] = int(tp)
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["sensitivity"] = float(tp / max(tp + fn, 1))  # same as recall
        metrics["specificity"] = float(tn / max(tn + fp, 1))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
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

            all_probs.extend(torch.sigmoid(dep_logit).detach().cpu().numpy().tolist())
            all_labels.extend(dep_labels.detach().cpu().numpy().tolist())

    if n_batches > 0:
        for k in total_losses:
            total_losses[k] /= n_batches

    return (
        total_losses,
        np.array(all_labels, dtype=int),
        np.array(all_probs,  dtype=float),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer:         torch.optim.Optimizer,
    num_warmup_epochs: int,
    num_cosine_epochs: int,
    eta_min:           float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    def warmup_lambda(epoch: int) -> float:
        if epoch < num_warmup_epochs:
            return float(epoch + 1) / float(max(1, num_warmup_epochs))
        return 1.0

    warmup_sched = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=num_cosine_epochs, eta_min=eta_min)
    return SequentialLR(optimizer,
                        schedulers=[warmup_sched, cosine_sched],
                        milestones=[num_warmup_epochs])


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, path):
    torch.save({
        "epoch": epoch, "best_metric": best_metric,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, str(path))


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_epoch(epoch, phase, losses, metrics, lr, elapsed):
    print(
        f"[Epoch {epoch:03d}] {phase:5s} | "
        f"loss={losses['loss_total']:.4f} | "
        f"acc={metrics.get('accuracy', 0):.4f} "
        f"f1_macro={metrics.get('f1_macro', 0):.4f} "
        f"auc={metrics.get('auc', float('nan')):.4f} "
        f"sens={metrics.get('sensitivity', 0):.4f} "
        f"spec={metrics.get('specificity', 0):.4f} "
        f"thr={metrics.get('threshold', 0.5):.2f} | "
        f"lr={lr:.2e} | {elapsed:.1f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-seed training run
# ─────────────────────────────────────────────────────────────────────────────

def train_one_seed(
    cfg: dict,
    seed: int,
    dataset_name: str,
    context_mode: str,
    run_log_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Train one full run with the given seed.
    Returns best dev metrics dict.
    """
    set_seed(seed)

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

    # Seed-specific checkpoint to avoid collisions in multi-seed run
    ckpt_dir = Path(checkpoint_dir) / f"ctx_{context_mode}" / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / "best_model.pt"

    print(f"\n{'='*60}")
    print(f"  SEED={seed}  context_mode={context_mode!r}  dataset={dataset_name}")
    print(f"{'='*60}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = DaicWozDataset(
        split="train", root=cache_dir, context_mode=context_mode
    )
    dev_ds = DaicWozDataset(
        split="dev", root=cache_dir, context_mode=context_mode
    )

    if use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        labels_arr  = train_ds.labels().numpy()
        class_count = np.bincount(labels_arr)
        sample_w    = (1.0 / class_count)[labels_arr]
        sampler     = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).float(),
            num_samples=len(train_ds), replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=0, pin_memory=(DEVICE.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(DEVICE.type == "cuda"),
        )

    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )

    label_counts = torch.bincount(train_ds.labels())
    print(f"  Train: {len(train_ds)} graphs | "
          f"neg={label_counts[0]}, pos={label_counts[1]}")
    print(f"  Dev:   {len(dev_ds)} graphs")

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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}  dropout={dropout}  drop_edge={drop_edge}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, warmup_epochs, cosine_epochs, eta_min)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_dev_auc     = 0.0
    best_dev_metrics = {}
    no_improve_count = 0
    epoch_log        = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_losses, train_labels, train_probs = run_epoch(
            model, train_loader, optimizer, DEVICE, is_train=True,
            w_symptom=w_symptom, w_phq=w_phq, focal_alpha=focal_alpha,
            label_smoothing=label_smoothing,
        )
        train_metrics = compute_metrics(train_labels, train_probs, threshold=0.5)
        lr_now = optimizer.param_groups[0]["lr"]
        log_epoch(epoch, "TRAIN", train_losses, train_metrics, lr_now,
                  time.time() - t0)

        t1 = time.time()
        dev_losses, dev_labels, dev_probs = run_epoch(
            model, dev_loader, None, DEVICE, is_train=False,
            w_symptom=w_symptom, w_phq=w_phq, focal_alpha=focal_alpha,
            label_smoothing=0.0,
        )
        # FIX-THRESHOLD: constrained sweep
        best_thr    = find_best_threshold(dev_labels, dev_probs)
        dev_metrics = compute_metrics(dev_labels, dev_probs, threshold=best_thr)
        log_epoch(epoch, "DEV", dev_losses, dev_metrics, lr_now,
                  time.time() - t1)

        scheduler.step()

        # Primary criterion: AUC (more robust than threshold-optimised F1
        # on ~35-sample dev sets)
        dev_auc = dev_metrics.get("auc", 0.0)
        if not math.isnan(dev_auc) and dev_auc > best_dev_auc:
            best_dev_auc     = dev_auc
            best_dev_metrics = {**dev_metrics, "epoch": epoch}
            no_improve_count = 0
            save_checkpoint(model, optimizer, scheduler, epoch,
                            best_dev_auc, best_model_path)
            print(f"  * New best AUC={best_dev_auc:.4f} "
                  f"(F1={dev_metrics.get('f1_macro', 0):.4f} "
                  f"thr={best_thr:.2f}) → saved")
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{early_stop_pat}). "
                  f"Best AUC={best_dev_auc:.4f}")
            if no_improve_count >= early_stop_pat:
                print(f"  Early stopping at epoch {epoch}.")
                break

        # Per-epoch log entry
        epoch_log.append({
            "epoch": epoch, "seed": seed, "context_mode": context_mode,
            "train_loss": train_losses["loss_total"],
            "dev_loss":   dev_losses["loss_total"],
            "dev_auc":    dev_auc,
            "dev_f1_macro": dev_metrics.get("f1_macro", 0.0),
        })

    # Write per-run epoch log
    if run_log_path is not None:
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(run_log_path, "w") as f:
            json.dump({"seed": seed, "context_mode": context_mode,
                       "best": best_dev_metrics, "epochs": epoch_log}, f, indent=2)
        print(f"  Run log saved: {run_log_path}")

    print(f"  [Seed {seed}] Done. Best dev AUC={best_dev_auc:.4f}  "
          f"F1={best_dev_metrics.get('f1_macro', 0):.4f}")
    return best_dev_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed variance reporting
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_seed(
    cfg: dict,
    seeds: List[int],
    dataset_name: str,
    context_mode: str,
    log_dir: Path,
) -> None:
    """
    FIX-VARIANCE: Train with multiple seeds and print mean ± std.
    Required for credible reporting on small datasets like DAIC-WOZ (107 samples).
    """
    all_metrics: List[Dict] = []

    for seed in seeds:
        log_path = log_dir / f"run_ctx_{context_mode}_seed_{seed}.json"
        metrics  = train_one_seed(cfg, seed, dataset_name, context_mode, log_path)
        all_metrics.append(metrics)

    print_multi_seed_summary(all_metrics, seeds, context_mode)

    # Save aggregate report
    agg_path = log_dir / f"aggregate_ctx_{context_mode}.json"
    with open(agg_path, "w") as f:
        json.dump({"context_mode": context_mode, "seeds": seeds,
                   "runs": all_metrics}, f, indent=2)
    print(f"\nAggregate report: {agg_path}")


def print_multi_seed_summary(
    all_metrics: List[Dict],
    seeds: List[int],
    context_mode: str,
) -> None:
    """Print a paper-ready table of mean ± std across seeds."""
    keys = ["auc", "f1_macro", "f1_weighted", "precision", "recall",
            "sensitivity", "specificity", "accuracy"]

    print(f"\n{'='*72}")
    print(f"  MULTI-SEED SUMMARY  |  context_mode={context_mode!r}  "
          f"|  seeds={seeds}")
    print(f"{'='*72}")
    print(f"  {'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*52}")
    for key in keys:
        vals = [m.get(key, float("nan")) for m in all_metrics]
        vals_clean = [v for v in vals if not math.isnan(v)]
        if not vals_clean:
            print(f"  {key:<20} {'N/A':>8}")
            continue
        mn = np.mean(vals_clean)
        sd = np.std(vals_clean)
        lo = np.min(vals_clean)
        hi = np.max(vals_clean)
        print(f"  {key:<20} {mn:>8.4f} {sd:>8.4f} {lo:>8.4f} {hi:>8.4f}")
    print(f"{'='*72}\n")


def print_shortcut_ablation_reminder() -> None:
    """Print guide for interpreting ablation results."""
    print(
        "\n"
        "=" * 72 + "\n"
        "  HOW TO INTERPRET SHORTCUT LEARNING ABLATION\n"
        "=" * 72 + "\n"
        "  Run all 3 modes and compare mean AUC / F1-macro:\n"
        "\n"
        "  --context_mode none      → BERT sees only participant text\n"
        "  --context_mode truncated → + last 20 words of Ellie question\n"
        "  --context_mode full      → + full Ellie turn\n"
        "\n"
        "  GOOD (no shortcut): F1(none) ≈ F1(truncated) > baseline\n"
        "  BAD  (shortcut):    F1(none) << F1(truncated)\n"
        "\n"
        "  If shortcut detected:\n"
        "    → Report both numbers in paper (honest evaluation)\n"
        "    → Use context_mode=none as your primary result\n"
        "    → Discuss as limitation / future work\n"
        "=" * 72 + "\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train HMSG-Net")
    parser.add_argument(
        "--dataset", choices=["daicwoz"], default="daicwoz",
        help="Dataset to train on. (E-DAIC: TODO for future work)"
    )
    parser.add_argument(
        "--context_mode",
        default="truncated",
        choices=["none", "truncated", "full"],
        help=(
            "BERT context mode for shortcut learning ablation.\n"
            "  none      : participant text only (anti-shortcut, recommended)\n"
            "  truncated : + last 20 words of Ellie question (default)\n"
            "  full      : + full Ellie turn (shortcut upper-bound)\n"
            "Run all 3 and compare F1 / AUC for ablation table in paper."
        ),
    )
    parser.add_argument(
        "--multi_seed",
        action="store_true",
        help=(
            "Train with multiple seeds and report mean ± std. "
            "REQUIRED for credible reporting on DAIC-WOZ (~107 train samples). "
            f"Seeds used: {MULTI_SEEDS}"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seeds for multi-seed run (e.g. --seeds 42 123 456)",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Directory to save per-run JSON logs (default: checkpoint_dir/logs)",
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Re-process dataset cache from scratch.",
    )
    args = parser.parse_args()

    cfg = DATASET_CFG[args.dataset]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    seed_label = "multi" if args.multi_seed else str(args.seeds[0] if args.seeds else BASE_SEED)

    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else Path(cfg["checkpoint_dir"]) / "logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    txt_log_path = log_dir / f"train_{args.dataset}_ctx_{args.context_mode}_seed_{seed_label}_{timestamp}.txt"

    with tee_output(txt_log_path):
        print(f"Dataset      : {args.dataset}")
        print(f"Context mode : {args.context_mode}")
        print(f"Multi-seed   : {args.multi_seed}")
        print(f"Device       : {DEVICE}")
        print(f"Log dir      : {log_dir}")
        print(f"TXT log      : {txt_log_path}")

        if args.multi_seed:
            seeds = args.seeds if args.seeds else MULTI_SEEDS
            run_multi_seed(cfg, seeds, args.dataset, args.context_mode, log_dir)
            print_shortcut_ablation_reminder()
        else:
            seed     = args.seeds[0] if args.seeds else BASE_SEED
            log_path = log_dir / f"run_ctx_{args.context_mode}_seed_{seed}.json"
            metrics  = train_one_seed(
                cfg, seed, args.dataset, args.context_mode, log_path
            )
            print(f"\nFinal best dev metrics:")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"  {k:<20}: {v:.4f}")
                else:
                    print(f"  {k:<20}: {v}")
            print_shortcut_ablation_reminder()


if __name__ == "__main__":
    main()