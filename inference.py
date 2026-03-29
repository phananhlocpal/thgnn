"""
inference.py

Evaluation / inference script for HMSG-Net.

Usage:
    python inference.py [--checkpoint PATH] [--threshold auto|float]

Steps:
  1. Load best checkpoint.
  2. Find optimal classification threshold on the dev set.
  3. Evaluate on the test set using the optimal threshold.
  4. Print full classification report:
       accuracy, F1 (macro + weighted), precision, recall, AUC,
       confusion matrix, per-class metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.loader import DataLoader

from edaic_dataset import DepressionDataset
from model import HMSGNet

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (mirrors train.py; keep in sync)
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR  = Path("C:/Users/ezycloudx-admin/Desktop/thgnn/checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
CACHE_DIR       = "C:/Users/ezycloudx-admin/Desktop/thgnn/cache"
BATCH_SIZE      = 8

HIDDEN_DIM      = 256
NUM_GNN_LAYERS  = 3
N_HEADS         = 4
DROPOUT         = 0.3
NUM_SYMPTOMS    = 8
NUM_EDGE_TYPES  = 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path) -> Tuple[HMSGNet, dict]:
    """
    Instantiate HMSGNet and load weights from checkpoint.

    Returns
    -------
    model  : HMSGNet on DEVICE, in eval mode
    ckpt   : raw checkpoint dict (for metadata)
    """
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE)
    model = HMSGNet(
        hidden_dim=HIDDEN_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        num_symptoms=NUM_SYMPTOMS,
        num_edge_types=NUM_EDGE_TYPES,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    model:  HMSGNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model over `loader`.

    Returns
    -------
    labels     : (N,) int array of ground-truth binary labels
    probs      : (N,) float array of sigmoid probabilities
    phq_preds  : (N,) float array of predicted PHQ scores (× 24 back-scaled)
    """
    all_labels:    list = []
    all_probs:     list = []
    all_phq_preds: list = []

    for batch in loader:
        batch = batch.to(device)
        dep_logit, _sym_logits, phq_pred = model(batch)

        probs    = torch.sigmoid(dep_logit).cpu().numpy()
        labels   = batch.y.squeeze().cpu().numpy()
        phq_vals = (phq_pred * 24.0).cpu().numpy()   # back-scale from normalised

        # Handle batch size == 1 (scalar → 1-d)
        probs    = np.atleast_1d(probs)
        labels   = np.atleast_1d(labels)
        phq_vals = np.atleast_1d(phq_vals)

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
        all_phq_preds.extend(phq_vals.tolist())

    return (
        np.array(all_labels,    dtype=int),
        np.array(all_probs,     dtype=float),
        np.array(all_phq_preds, dtype=float),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Threshold selection
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(
    labels: np.ndarray,
    probs:  np.ndarray,
    metric: str = "f1_macro",
) -> float:
    """
    Sweep thresholds ∈ [0.1, 0.9] with step 0.01 and return the one that
    maximises `metric` on the provided split.

    Supported metrics: 'f1_macro', 'f1_weighted', 'f1_binary'.
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_thr   = 0.5
    best_score = -1.0

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        if metric == "f1_macro":
            score = f1_score(labels, preds, average="macro", zero_division=0)
        elif metric == "f1_weighted":
            score = f1_score(labels, preds, average="weighted", zero_division=0)
        elif metric == "f1_binary":
            score = f1_score(labels, preds, average="binary", zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_thr   = thr

    print(f"  Optimal threshold ({metric}): {best_thr:.2f}  -> score={best_score:.4f}")
    return float(best_thr)


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    labels:    np.ndarray,
    probs:     np.ndarray,
    phq_preds: np.ndarray,
    threshold: float,
    split:     str = "TEST",
) -> Dict[str, float]:
    """
    Print and return a comprehensive evaluation report.
    """
    preds = (probs >= threshold).astype(int)

    acc     = accuracy_score(labels, preds)
    f1_mac  = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_wei  = f1_score(labels, preds, average="weighted", zero_division=0)
    prec    = precision_score(labels, preds, zero_division=0)
    rec     = recall_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    # PHQ regression MAE / correlation
    phq_true = np.array(labels, dtype=float)   # placeholder when true scores unavailable
    # (ground-truth PHQ scores are not stored in this function's inputs,
    #  so we just report mean predicted score per class)
    mean_phq_neg = phq_preds[labels == 0].mean() if (labels == 0).any() else float("nan")
    mean_phq_pos = phq_preds[labels == 1].mean() if (labels == 1).any() else float("nan")

    cm = confusion_matrix(labels, preds)

    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  Evaluation Report  [{split}]  (threshold = {threshold:.2f})")
    print(sep)
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  F1 (macro)        : {f1_mac:.4f}")
    print(f"  F1 (weighted)     : {f1_wei:.4f}")
    print(f"  Precision (pos)   : {prec:.4f}")
    print(f"  Recall    (pos)   : {rec:.4f}")
    print(f"  AUC-ROC           : {auc:.4f}")
    print(f"  Mean PHQ pred (neg class): {mean_phq_neg:.2f}")
    print(f"  Mean PHQ pred (pos class): {mean_phq_pos:.2f}")
    print()
    print("  Confusion Matrix:")
    print(f"    Predicted  ->  Neg   Pos")
    for i, row in enumerate(cm):
        label_str = "Neg" if i == 0 else "Pos"
        print(f"    True {label_str}    : {row[0]:5d}  {row[1]:5d}")
    print()
    print("  Per-class Classification Report:")
    print(classification_report(
        labels, preds,
        target_names=["Not Depressed", "Depressed"],
        digits=4,
        zero_division=0,
    ))
    print(sep)

    result = {
        "split":        split,
        "threshold":    threshold,
        "accuracy":     acc,
        "f1_macro":     f1_mac,
        "f1_weighted":  f1_wei,
        "precision":    prec,
        "recall":       rec,
        "auc":          auc,
        "mean_phq_neg": mean_phq_neg,
        "mean_phq_pos": mean_phq_pos,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    checkpoint_path: Path = BEST_MODEL_PATH,
    threshold_mode:  str  = "auto",   # 'auto' or a float string
) -> None:
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run train.py first."
        )

    model, ckpt_meta = load_model(checkpoint_path)
    print(f"  Checkpoint epoch:   {ckpt_meta.get('epoch', '?')}")
    print(f"  Best dev F1 (ckpt): {ckpt_meta.get('best_f1', float('nan')):.4f}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets …")
    dev_ds   = DepressionDataset(split="dev",  root=CACHE_DIR)
    test_ds  = DepressionDataset(split="test", root=CACHE_DIR)

    dev_loader  = DataLoader(dev_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Dev: {len(dev_ds)} graphs,  Test: {len(test_ds)} graphs")

    # ── Dev predictions ───────────────────────────────────────────────────────
    print("\nRunning inference on DEV split …")
    dev_labels, dev_probs, dev_phq = predict(model, dev_loader, DEVICE)

    # ── Threshold selection ───────────────────────────────────────────────────
    if threshold_mode == "auto":
        print("\nFinding optimal threshold on DEV set (maximising F1-macro) …")
        threshold = find_best_threshold(dev_labels, dev_probs, metric="f1_macro")
    else:
        threshold = float(threshold_mode)
        print(f"\nUsing user-specified threshold: {threshold:.2f}")

    # ── Dev report ────────────────────────────────────────────────────────────
    dev_results = print_report(dev_labels, dev_probs, dev_phq, threshold, split="DEV")

    # ── Test predictions ──────────────────────────────────────────────────────
    print("\nRunning inference on TEST split …")
    test_labels, test_probs, test_phq = predict(model, test_loader, DEVICE)

    # ── Test report ───────────────────────────────────────────────────────────
    test_results = print_report(test_labels, test_probs, test_phq,
                                threshold, split="TEST")

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {"dev": dev_results, "test": test_results}
    # Convert nan to None for JSON serialisation
    def _sanitize(d):
        return {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in d.items()
        }

    out_path = CHECKPOINT_DIR / "eval_results.json"
    with open(str(out_path), "w") as fh:
        json.dump(
            {"dev": _sanitize(dev_results), "test": _sanitize(test_results)},
            fh,
            indent=2,
        )
    print(f"\nResults saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate HMSG-Net on DAIC-WOZ test split."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(BEST_MODEL_PATH),
        help="Path to the model checkpoint (.pt file). "
             f"Default: {BEST_MODEL_PATH}",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="auto",
        help="Classification threshold. Pass 'auto' to select on dev set, "
             "or a float (e.g., 0.5). Default: auto.",
    )
    args = parser.parse_args()
    main(
        checkpoint_path=Path(args.checkpoint),
        threshold_mode=args.threshold,
    )
