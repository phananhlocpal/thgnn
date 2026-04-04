"""
train_final.py — Final model training sau khi có CV results.

Mục đích
────────
Sau khi train_cv.py cho ra mean_epoch (số epoch tốt nhất trung bình qua 5 folds),
script này:
  1. Retrain HMSGNet trên TOÀN BỘ train+dev (142 participants) với SDS
  2. Train đúng mean_epoch epochs (không có early stopping — đã biết target)
  3. Evaluate MỘT LẦN DUY NHẤT trên test set (47 participants)
  4. Report final paper metrics

Usage
─────
    python train_final.py --target-epoch 60 --aug-sds
    python train_final.py --cv-summary checkpoints_cv/cv_summary.json --aug-sds

Lưu ý quan trọng
─────────────────
Test set (47 participants) KHÔNG ĐƯỢC NHÌN trong suốt quá trình CV.
Đây là single final evaluation — đúng scientific practice.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from daicwoz_dataset import (DaicWozDataset, build_graph, apply_sds,
                               DAICWOZ_DATA_DIR)
from model import HMSGNet, compute_loss
from train_cv import (
    load_all_participants, FoldDataset, run_epoch,
    find_best_threshold, compute_metrics, build_scheduler,
    set_seed, CV_CFG, SPLIT_FILES, NUM_SYMPTOMS, MAX_GRAD_NORM, DEVICE,
    PHQ8_COLS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR  = Path("C:/Users/Administrator/Desktop/thgnn")
CKPT_DIR  = BASE_DIR / "checkpoints_final"

torch.backends.cudnn.allow_tf32       = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark        = True


# ─────────────────────────────────────────────────────────────────────────────
# Load test participants
# ─────────────────────────────────────────────────────────────────────────────

def load_test_participants():
    df = pd.read_csv(SPLIT_FILES["test"])
    df.columns = df.columns.str.strip()
    label_col = "PHQ8_Binary" if "PHQ8_Binary" in df.columns else "PHQ_Binary"
    score_col = "PHQ8_Score"  if "PHQ8_Score"  in df.columns else "PHQ_Score"

    phq8_dict = {}
    if all(c in df.columns for c in PHQ8_COLS):
        for _, row in df.iterrows():
            vals = row[PHQ8_COLS].values.astype(np.float32)
            vals = np.where(np.isnan(vals), 0.0, vals)
            phq8_dict[int(row["Participant_ID"])] = vals

    graphs, pids, labels = [], [], []
    for _, row in df.iterrows():
        pid       = int(row["Participant_ID"])
        label     = int(row[label_col])
        phq_score = float(row.get(score_col, 0.0))
        phq8      = phq8_dict.get(pid, np.zeros(8, dtype=np.float32))

        missing = any(
            not (DAICWOZ_DATA_DIR / f"{pid}_{m}_feats.npy").exists()
            for m in ("text", "audio")
        )
        if missing:
            print(f"  [SKIP] PID {pid}: missing features")
            continue
        try:
            g = build_graph(pid, label, phq_score, phq8)
            if g is not None:
                graphs.append(g); pids.append(pid); labels.append(label)
        except Exception as e:
            print(f"  [SKIP] PID {pid}: {e}")

    print(f"Test: {len(graphs)} participants "
          f"({sum(labels)} pos, {len(labels)-sum(labels)} neg)")
    return graphs, pids, labels


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate on test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_test(model, test_loader, threshold):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            dep_logit, _, _ = model(batch)
            all_probs.extend(torch.sigmoid(dep_logit).cpu().numpy().tolist())
            all_labels.extend(batch.y.reshape(-1).cpu().numpy().tolist())

    labels = np.array(all_labels, int)
    probs  = np.array(all_probs, float)
    preds  = (probs >= threshold).astype(int)

    try:    auc = roc_auc_score(labels, probs)
    except: auc = float("nan")

    f1_mac = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_wei = f1_score(labels, preds, average="weighted", zero_division=0)
    prec   = precision_score(labels, preds, zero_division=0)
    rec    = recall_score(labels, preds, zero_division=0)
    acc    = accuracy_score(labels, preds)
    cm     = confusion_matrix(labels, preds)

    sep = "="*65
    print(f"\n{sep}")
    print(f"  FINAL TEST RESULTS  (threshold={threshold:.2f})")
    print(sep)
    print(f"  AUC-ROC    : {auc:.4f}")
    print(f"  F1 (macro) : {f1_mac:.4f}")
    print(f"  F1 (weighted): {f1_wei:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  Accuracy   : {acc:.4f}")
    print()
    print("  Confusion Matrix:")
    print(f"    {'':12} Pred Neg  Pred Pos")
    print(f"    True Neg : {cm[0][0]:8d}  {cm[0][1]:8d}")
    print(f"    True Pos : {cm[1][0]:8d}  {cm[1][1]:8d}")
    print()
    print(classification_report(
        labels, preds,
        target_names=["Not Depressed", "Depressed"],
        digits=4, zero_division=0,
    ))
    print(sep)

    return {
        "auc": auc, "f1_macro": f1_mac, "f1_weighted": f1_wei,
        "precision": prec, "recall": rec, "accuracy": acc,
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Final model training + test evaluation for HMSGNet"
    )
    parser.add_argument("--target-epoch", type=int, default=None,
                        help="Number of epochs to train (from CV mean_epoch)")
    parser.add_argument("--cv-summary",   type=str, default=None,
                        help="Path to cv_summary.json from train_cv.py")
    parser.add_argument("--aug-sds",      action="store_true")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    # Get target epoch from CV summary or CLI
    target_epoch = args.target_epoch
    cv_results   = None

    if args.cv_summary and Path(args.cv_summary).exists():
        with open(args.cv_summary) as f:
            cv_results = json.load(f)
        if target_epoch is None:
            target_epoch = int(round(cv_results["mean_epoch"]))
        print(f"CV Summary loaded: AUC={cv_results['paper_auc']}  F1={cv_results['paper_f1']}")
        print(f"Mean best epoch from CV: {cv_results['mean_epoch']:.1f}")

    if target_epoch is None:
        target_epoch = 80  # safe default
        print(f"No CV summary provided. Using default target_epoch={target_epoch}")

    print(f"\nTarget epochs: {target_epoch}")

    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ALL train+dev ─────────────────────────────────────────────────
    print("\nLoading train+dev participants (all 142 for final training) ...")
    all_graphs, all_pids, all_labels = load_all_participants(["train", "dev"])

    # ── Load test (held-out, never seen during training) ───────────────────
    print("\nLoading test participants (held-out, evaluated ONCE) ...")
    test_graphs, test_pids, test_labels = load_test_participants()

    # ── SDS augmentation ──────────────────────────────────────────────────
    cfg = dict(CV_CFG)
    if args.aug_sds:
        augmented = []
        for g in all_graphs:
            augmented.append(g)
            for _ in range(cfg["aug_copies"]):
                aug = apply_sds(g, aug_prob=cfg["aug_prob"])
                augmented.append(aug)
        train_graphs_final = augmented
        print(f"\nSDS: {len(all_graphs)} → {len(train_graphs_final)} samples")
    else:
        train_graphs_final = all_graphs
        print(f"\nNo SDS: {len(train_graphs_final)} samples")

    train_ds = FoldDataset(train_graphs_final)
    test_ds  = FoldDataset(test_graphs)

    # Weighted sampler for imbalance
    la = train_ds.labels().numpy()
    lc = np.bincount(la)
    sw = (1.0 / lc)[la]
    sampler = WeightedRandomSampler(torch.from_numpy(sw).float(), len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=0,
                              pin_memory=(DEVICE.type=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0,
                              pin_memory=(DEVICE.type=="cuda"))

    # ── Model ────────────────────────────────────────────────────────────
    model = HMSGNet(
        hidden_dim=cfg["hidden_dim"], num_gnn_layers=cfg["num_gnn_layers"],
        num_edge_types=cfg["num_edge_types"], num_symptoms=NUM_SYMPTOMS,
        n_heads=cfg["n_heads"], dropout=cfg["dropout"],
        drop_edge=cfg["drop_edge"], feat_noise=cfg["feat_noise"],
        text_dim=cfg["text_dim"], audio_dim=cfg["audio_dim"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {n_params:,}")
    print(f"Training for {target_epoch} epochs on {len(train_ds)} samples ...")

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # Use target_epoch as cosine period
    scheduler = build_scheduler(optimizer, cfg["warmup_epochs"], target_epoch, cfg["eta_min"])

    ckpt_path = CKPT_DIR / "final_model.pt"

    for epoch in range(1, target_epoch + 1):
        t0 = time.time()
        tl, tlab, tprob = run_epoch(model, train_loader, optimizer, DEVICE, True, cfg)
        scheduler.step()
        tmets = compute_metrics(tlab, tprob)
        print(f"[Ep {epoch:03d}/{target_epoch}] "
              f"loss={tl['loss_dep']:.4f} train_auc={tmets['auc']:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} "
              f"| {time.time()-t0:.1f}s")

    # Save final model
    torch.save({
        "epoch": target_epoch,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "cv_results": cv_results,
    }, str(ckpt_path))
    print(f"\nFinal model saved: {ckpt_path}")

    # ── Find threshold on training set (no dev leakage) ──────────────────
    # Use training predictions to find threshold (common practice)
    # Alternatively: use mean threshold from CV folds
    if cv_results and "folds" in cv_results:
        thrs = [f["best_thr"] for f in cv_results["folds"]]
        threshold = float(np.mean(thrs))
        print(f"\nThreshold from CV fold mean: {threshold:.3f}")
    else:
        # Fallback: find on train set
        _, tlab_final, tprob_final = run_epoch(
            model, train_loader, None, DEVICE, False, cfg
        )
        threshold = find_best_threshold(tlab_final, tprob_final)
        print(f"\nThreshold from train set: {threshold:.3f}")

    # ── FINAL TEST EVALUATION (một lần duy nhất) ─────────────────────────
    print("\n" + "!"*65)
    print("  FINAL TEST EVALUATION — This number goes in the paper")
    print("!"*65)

    test_results = evaluate_test(model, test_loader, threshold)

    # ── Save results ─────────────────────────────────────────────────────
    final_report = {
        "target_epoch":  target_epoch,
        "aug_sds":       args.aug_sds,
        "seed":          args.seed,
        "n_train":       len(all_graphs),
        "n_train_aug":   len(train_graphs_final),
        "n_test":        len(test_graphs),
        "threshold":     threshold,
        "cv_summary":    {
            "mean_auc": cv_results.get("mean_auc") if cv_results else None,
            "std_auc":  cv_results.get("std_auc")  if cv_results else None,
            "mean_f1":  cv_results.get("mean_f1")  if cv_results else None,
            "std_f1":   cv_results.get("std_f1")   if cv_results else None,
        },
        "test_results":  test_results,
    }

    report_path = CKPT_DIR / "final_report.json"
    with open(str(report_path), "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    print(f"\nFinal report saved: {report_path}")
    print()
    print("=" * 65)
    print("  PAPER NUMBERS")
    print("=" * 65)
    if cv_results:
        print(f"  CV AUC (5-fold): {cv_results.get('paper_auc', 'N/A')}")
        print(f"  CV F1  (5-fold): {cv_results.get('paper_f1',  'N/A')}")
    print(f"  Test AUC : {test_results['auc']:.4f}")
    print(f"  Test F1  : {test_results['f1_macro']:.4f}")
    print(f"  Test Prec: {test_results['precision']:.4f}")
    print(f"  Test Rec : {test_results['recall']:.4f}")
    print(f"  Test Acc : {test_results['accuracy']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()