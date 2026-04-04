"""
train_cv.py — 5-Fold Stratified Cross-Validation cho HMSGNet trên DAIC-WOZ.

Mục đích
────────
Thay thế fixed dev split (35 samples, variance cao) bằng 5-fold CV
trên train+dev combined (142 participants). Đây là standard practice
cho clinical NLP với small dataset.

Flow
────
1. Load toàn bộ train+dev participants (142 PIDs)
2. Stratified 5-fold split theo participant label (28% positive rate)
3. Mỗi fold: train trên 113-114 participants (với SDS), val trên 28-29
4. Report mean ± std AUC, F1 across 5 folds
5. Save best model per fold + summary JSON

Paper reporting
───────────────
CV AUC:  mean ± std  (reliable estimation, lower variance vs fixed split)
Test AUC: evaluate best_fold_model OR final_model trên test=47 (held-out hoàn toàn)

Usage
─────
    python train_cv.py --aug-sds --n-folds 5
    python train_cv.py --aug-sds --n-folds 5 --force-reload
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from daicwoz_dataset import DaicWozDataset, build_graph, apply_sds, DAICWOZ_DATA_DIR
from model import HMSGNet, compute_loss

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path("C:/Users/Administrator/Desktop/thgnn")
CACHE_DIR  = BASE_DIR / "cache_daicwoz"
CKPT_DIR   = BASE_DIR / "checkpoints_cv"

SPLIT_FILES = {
    "train": DAICWOZ_DATA_DIR / "train_split_Depression_AVEC2017.csv",
    "dev":   DAICWOZ_DATA_DIR / "dev_split_Depression_AVEC2017.csv",
    "test":  DAICWOZ_DATA_DIR / "full_test_split.csv",
}

PHQ8_COLS = [
    "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
    "PHQ8_Appetite",   "PHQ8_Failure",   "PHQ8_Concentrating", "PHQ8_Moving",
]

# Best hyperparams từ run 1 (AUC=0.7428)
CV_CFG = {
    "text_dim": 777, "audio_dim": 777,
    "hidden_dim": 128, "num_gnn_layers": 2, "num_edge_types": 6,
    "n_heads": 4, "dropout": 0.60, "drop_edge": 0.25, "feat_noise": 0.05,
    "focal_alpha": 0.75, "label_smoothing": 0.08,
    "w_symptom": 0.0, "w_phq": 0.0,
    "aug_copies": 2, "aug_prob": 0.90,
    "lr": 5e-5, "weight_decay": 1e-2,
    "warmup_epochs": 3, "cosine_epochs": 150, "eta_min": 1e-7,
    "batch_size": 8, "max_epochs": 150, "early_stop_pat": 40,
}

NUM_SYMPTOMS  = 8
MAX_GRAD_NORM = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.allow_tf32       = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark        = True

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.20, 0.81, 0.02):
        preds = (probs >= thr).astype(int)
        f1    = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    m = {
        "accuracy":  accuracy_score(labels, preds),
        "f1_macro":  f1_score(labels, preds, average="macro",    zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "threshold": threshold,
    }
    try:    m["auc"] = roc_auc_score(labels, probs)
    except: m["auc"] = float("nan")
    return m


def build_scheduler(optimizer, warmup_epochs, cosine_epochs, eta_min):
    def wfn(e): return float(e+1)/max(1,warmup_epochs) if e < warmup_epochs else 1.0
    return SequentialLR(optimizer,
        schedulers=[LambdaLR(optimizer, wfn),
                    CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=eta_min)],
        milestones=[warmup_epochs])


# ─────────────────────────────────────────────────────────────────────────────
# In-memory fold dataset (không cache, build từ graphs list)
# ─────────────────────────────────────────────────────────────────────────────

class FoldDataset(torch.utils.data.Dataset):
    """Simple wrapper around a list of Data objects."""
    def __init__(self, graphs: List[Data]):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def labels(self) -> torch.Tensor:
        return torch.tensor([g.y.item() for g in self.graphs], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Load all participant graphs
# ─────────────────────────────────────────────────────────────────────────────

def load_all_participants(splits: List[str]) -> Tuple[List[Data], List[int], List[int]]:
    """
    Load graphs for all participants in given splits.
    Returns (graphs, pids, labels).
    """
    all_rows = []
    for split in splits:
        df = pd.read_csv(SPLIT_FILES[split])
        df.columns = df.columns.str.strip()
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)

    label_col = "PHQ8_Binary" if "PHQ8_Binary" in combined.columns else "PHQ_Binary"
    score_col = "PHQ8_Score"  if "PHQ8_Score"  in combined.columns else "PHQ_Score"

    phq8_dict: Dict[int, np.ndarray] = {}
    if all(c in combined.columns for c in PHQ8_COLS):
        for _, row in combined.iterrows():
            vals = row[PHQ8_COLS].values.astype(np.float32)
            vals = np.where(np.isnan(vals), 0.0, vals)
            phq8_dict[int(row["Participant_ID"])] = vals

    graphs, pids, labels = [], [], []
    skipped = 0

    for _, row in combined.iterrows():
        pid       = int(row["Participant_ID"])
        label     = int(row[label_col])
        phq_score = float(row.get(score_col, 0.0))
        phq8      = phq8_dict.get(pid, np.zeros(8, dtype=np.float32))

        # Check features exist
        missing = any(
            not (DAICWOZ_DATA_DIR / f"{pid}_{m}_feats.npy").exists()
            for m in ("text", "audio")
        )
        if missing:
            print(f"  [SKIP] PID {pid}: missing feature files")
            skipped += 1
            continue

        try:
            g = build_graph(pid, label, phq_score, phq8)
        except Exception as e:
            print(f"  [SKIP] PID {pid}: {e}")
            skipped += 1
            continue

        if g is None:
            skipped += 1
            continue

        graphs.append(g)
        pids.append(pid)
        labels.append(label)

    print(f"Loaded {len(graphs)} participants ({skipped} skipped)")
    print(f"  Positive: {sum(labels)}  Negative: {len(labels)-sum(labels)}")
    return graphs, pids, labels


# ─────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, is_train, cfg):
    model.train() if is_train else model.eval()
    totals = {k: 0.0 for k in ("loss_total","loss_dep","loss_symptom","loss_phq")}
    n, all_labels, all_probs = 0, [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in loader:
            batch = batch.to(device)
            dep_logit, sym_logits, phq_pred = model(batch)
            dep_labels  = batch.y.reshape(-1).long()
            phq8_labels = batch.phq8.view(-1, NUM_SYMPTOMS).float()
            phq_scores  = batch.phq_score.reshape(-1).float()

            loss, ld = compute_loss(
                dep_logit, sym_logits, phq_pred,
                dep_labels.float(), phq8_labels, phq_scores,
                w_symptom=cfg["w_symptom"], w_phq=cfg["w_phq"],
                focal_alpha=cfg["focal_alpha"],
                label_smoothing=cfg.get("label_smoothing", 0.0),
                device=device,
            )
            if is_train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            for k in totals: totals[k] += ld.get(k, 0.0)
            n += 1
            all_probs.extend(torch.sigmoid(dep_logit).detach().cpu().numpy().tolist())
            all_labels.extend(dep_labels.detach().cpu().numpy().tolist())

    if n > 0:
        for k in totals: totals[k] /= n
    return totals, np.array(all_labels, int), np.array(all_probs, float)


# ─────────────────────────────────────────────────────────────────────────────
# Train one fold
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    fold_idx:    int,
    train_graphs: List[Data],
    val_graphs:   List[Data],
    cfg:          dict,
    aug_sds:      bool,
    ckpt_path:    Path,
    seed:         int,
) -> dict:
    set_seed(seed + fold_idx)
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx+1}  |  train={len(train_graphs)}  val={len(val_graphs)}")
    print(f"{'='*60}")

    # ── SDS augmentation on training set ──────────────────────────────────
    if aug_sds:
        augmented = []
        for g in train_graphs:
            augmented.append(g)
            for _ in range(cfg["aug_copies"]):
                aug = apply_sds(g, aug_prob=cfg["aug_prob"])
                augmented.append(aug)
        train_graphs_final = augmented
    else:
        train_graphs_final = train_graphs

    train_ds = FoldDataset(train_graphs_final)
    val_ds   = FoldDataset(val_graphs)

    print(f"  After SDS: train={len(train_ds)}  val={len(val_ds)}")

    # ── Weighted sampler ──────────────────────────────────────────────────
    la = train_ds.labels().numpy()
    lc = np.bincount(la)
    sw = (1.0 / lc)[la]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sw).float(), len(train_ds), replacement=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        sampler=sampler, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = HMSGNet(
        hidden_dim=cfg["hidden_dim"], num_gnn_layers=cfg["num_gnn_layers"],
        num_edge_types=cfg["num_edge_types"], num_symptoms=NUM_SYMPTOMS,
        n_heads=cfg["n_heads"], dropout=cfg["dropout"],
        drop_edge=cfg["drop_edge"], feat_noise=cfg["feat_noise"],
        text_dim=cfg["text_dim"], audio_dim=cfg["audio_dim"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = build_scheduler(
        optimizer, cfg["warmup_epochs"], cfg["cosine_epochs"], cfg["eta_min"]
    )

    best_auc    = 0.0
    best_f1     = 0.0
    best_thr    = 0.5
    best_epoch  = 0
    no_improve  = 0
    history     = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        tl, tlab, tprob = run_epoch(model, train_loader, optimizer, DEVICE, True, cfg)
        t1 = time.time()
        vl, vlab, vprob = run_epoch(model, val_loader,   None,      DEVICE, False, cfg)

        scheduler.step()

        bthr  = find_best_threshold(vlab, vprob)
        vmets = compute_metrics(vlab, vprob, threshold=bthr)
        tmets = compute_metrics(tlab, tprob)

        val_auc = vmets.get("auc", 0.0)
        val_f1  = vmets.get("f1_macro", 0.0)

        print(
            f"  [Fold {fold_idx+1} Ep {epoch:03d}] "
            f"tr_loss={tl['loss_dep']:.4f} tr_auc={tmets['auc']:.4f} | "
            f"val_auc={val_auc:.4f} val_f1={val_f1:.4f} thr={bthr:.2f} | "
            f"{time.time()-t0:.1f}s"
        )

        history.append({
            "epoch": epoch, "train_auc": tmets["auc"],
            "val_auc": val_auc, "val_f1": val_f1,
        })

        if val_auc > best_auc:
            best_auc    = val_auc
            best_f1     = val_f1
            best_thr    = bthr
            best_epoch  = epoch
            no_improve  = 0
            torch.save({
                "epoch": epoch, "fold": fold_idx,
                "best_auc": best_auc, "best_f1": best_f1, "best_thr": best_thr,
                "model_state_dict": model.state_dict(),
            }, str(ckpt_path))
            print(f"  *** New best AUC={best_auc:.4f} F1={best_f1:.4f} → saved")
        else:
            no_improve += 1
            if no_improve >= cfg["early_stop_pat"]:
                print(f"  Early stopping at epoch {epoch} (no_improve={no_improve})")
                break

    print(f"\n  Fold {fold_idx+1} done: best_AUC={best_auc:.4f}  best_F1={best_f1:.4f}  epoch={best_epoch}")

    return {
        "fold":        fold_idx + 1,
        "best_auc":    best_auc,
        "best_f1":     best_f1,
        "best_thr":    best_thr,
        "best_epoch":  best_epoch,
        "n_train":     len(train_graphs),
        "n_val":       len(val_graphs),
        "n_train_aug": len(train_graphs_final),
        "history":     history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="5-Fold Stratified CV for HMSGNet on DAIC-WOZ"
    )
    parser.add_argument("--aug-sds",     action="store_true", help="Use SDS augmentation")
    parser.add_argument("--n-folds",     type=int, default=5)
    parser.add_argument("--force-reload",action="store_true", help="Ignore cached graphs")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Folds  : {args.n_folds}")
    print(f"SDS    : {args.aug_sds}")
    print(f"Seed   : {args.seed}")
    print()

    # ── Load train + dev participants ──────────────────────────────────────
    print("Loading train + dev participants (combined for CV) ...")
    all_graphs, all_pids, all_labels = load_all_participants(["train", "dev"])

    all_labels_arr = np.array(all_labels)
    print(f"\nClass distribution: {np.bincount(all_labels_arr).tolist()}")
    print(f"  ({all_labels_arr.mean()*100:.1f}% positive)")

    # ── Stratified K-Fold ─────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_graphs, all_labels_arr)):
        train_graphs = [all_graphs[i] for i in train_idx]
        val_graphs   = [all_graphs[i] for i in val_idx]
        train_labels = all_labels_arr[train_idx]
        val_labels   = all_labels_arr[val_idx]

        print(f"\nFold {fold_idx+1}: train={len(train_graphs)} "
              f"(pos={train_labels.sum()}, neg={(train_labels==0).sum()}) | "
              f"val={len(val_graphs)} "
              f"(pos={val_labels.sum()}, neg={(val_labels==0).sum()})")

        ckpt_path = CKPT_DIR / f"fold{fold_idx+1}_best.pt"
        result = train_fold(
            fold_idx=fold_idx,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            cfg=CV_CFG,
            aug_sds=args.aug_sds,
            ckpt_path=ckpt_path,
            seed=args.seed,
        )
        fold_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    aucs    = [r["best_auc"] for r in fold_results]
    f1s     = [r["best_f1"]  for r in fold_results]
    epochs  = [r["best_epoch"] for r in fold_results]

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION SUMMARY ({args.n_folds}-Fold Stratified)")
    print(f"{'='*60}")
    print(f"\n  {'Fold':>6} | {'AUC':>8} | {'F1 Macro':>10} | {'Best Ep':>8}")
    print(f"  {'-'*42}")
    for r in fold_results:
        print(f"  {r['fold']:>6} | {r['best_auc']:>8.4f} | {r['best_f1']:>10.4f} | {r['best_epoch']:>8}")
    print(f"  {'-'*42}")
    print(f"  {'Mean':>6} | {np.mean(aucs):>8.4f} | {np.mean(f1s):>10.4f} | {np.mean(epochs):>8.1f}")
    print(f"  {'Std':>6} | {np.std(aucs):>8.4f} | {np.std(f1s):>10.4f} | {np.std(epochs):>8.1f}")
    print(f"  {'Min':>6} | {np.min(aucs):>8.4f} | {np.min(f1s):>10.4f} |")
    print(f"  {'Max':>6} | {np.max(aucs):>8.4f} | {np.max(f1s):>10.4f} |")

    # Paper-ready string
    print(f"\n  >>> PAPER: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  >>> PAPER: F1  = {np.mean(f1s):.4f}  ± {np.std(f1s):.4f}")
    print(f"\n  Best fold for final evaluation: Fold {np.argmax(aucs)+1} (AUC={max(aucs):.4f})")
    print(f"  Average best epoch: {np.mean(epochs):.0f} → use as target for final model")

    # ── Save summary ──────────────────────────────────────────────────────
    summary = {
        "n_folds":     args.n_folds,
        "aug_sds":     args.aug_sds,
        "seed":        args.seed,
        "config":      {k: str(v) if isinstance(v, Path) else v for k, v in CV_CFG.items()},
        "folds":       fold_results,
        "mean_auc":    float(np.mean(aucs)),
        "std_auc":     float(np.std(aucs)),
        "mean_f1":     float(np.mean(f1s)),
        "std_f1":      float(np.std(f1s)),
        "mean_epoch":  float(np.mean(epochs)),
        "best_fold":   int(np.argmax(aucs) + 1),
        "paper_auc":   f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
        "paper_f1":    f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
    }

    summary_path = CKPT_DIR / "cv_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")
    print(f"\n  Next step: run train_final.py to retrain on all 142 participants")
    print(f"  and evaluate ONCE on test set (47 participants).")


if __name__ == "__main__":
    main()