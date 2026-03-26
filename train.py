"""
Training & Evaluation for CDL (Contrastive Discrepancy Learning)
on DAIC-WOZ depression detection.

Data directory structure:
  data/daicwoz/
    {ID}_TRANSCRIPT.csv         ← Transcript (TSV format)
    {ID}_text_feats.npy         ← BERT embeddings (from extract_bert.py)
    {ID}_CLNF_AUs.txt           ← OpenFace Action Units
    {ID}_COVAREP.csv            ← Acoustic features
    train_split_Depression_AVEC2017.csv
    dev_split_Depression_AVEC2017.csv

Usage:
    python train.py
    python train.py --train_csv custom_train.csv --dev_csv custom_dev.csv
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (
    f1_score, accuracy_score, mean_absolute_error,
    confusion_matrix, classification_report
)

from model   import CDLModel, CDLLoss
from dataset import get_dataloader


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────

def evaluate(model, loader, loss_fn, device, threshold=0.5):
    model.eval()
    all_phq_pred, all_phq_true = [], []
    all_dep_pred, all_dep_true = [], []
    all_cosine_dist = []
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
            all_phq_true.extend(phq.cpu().numpy())
            all_dep_pred.extend((torch.sigmoid(out["dep_logit"]) > threshold).long().cpu().numpy())
            all_dep_true.extend(dep.long().cpu().numpy())
            all_cosine_dist.extend(out["cosine_dist"].cpu().numpy())

    all_phq_pred = np.array(all_phq_pred)
    all_phq_true = np.array(all_phq_true)
    all_dep_pred = np.array(all_dep_pred)
    all_dep_true = np.array(all_dep_true)

    mae  = mean_absolute_error(all_phq_true, all_phq_pred)
    rmse = np.sqrt(((all_phq_true - all_phq_pred) ** 2).mean())
    f1   = f1_score(all_dep_true, all_dep_pred, zero_division=0)
    acc  = accuracy_score(all_dep_true, all_dep_pred)
    avg_disc_dep   = np.mean([all_cosine_dist[i] for i in range(len(all_dep_true)) if all_dep_true[i] == 1]) if any(all_dep_true) else 0
    avg_disc_nodep = np.mean([all_cosine_dist[i] for i in range(len(all_dep_true)) if all_dep_true[i] == 0]) if any(1 - d for d in all_dep_true) else 0

    return {
        "loss":        total_loss / len(loader),
        "mae":         mae,
        "rmse":        rmse,
        "f1":          f1,
        "acc":         acc,
        "disc_dep":    avg_disc_dep,     # ← key CDL diagnostic
        "disc_nodep":  avg_disc_nodep,
        "dep_pred":    all_dep_pred,
        "dep_true":    all_dep_true,
    }


# ─────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────
    train_loader = get_dataloader(
        args.train_csv, args.data_root,
        batch_size=args.batch_size, shuffle=True,
        text_feat_dim=args.text_feat_dim,
        nonverbal_dim=args.nonverbal_dim,
    )
    dev_loader = get_dataloader(
        args.dev_csv, args.data_root,
        batch_size=args.batch_size, shuffle=False,
        text_feat_dim=args.text_feat_dim,
        nonverbal_dim=args.nonverbal_dim,
    )

    # ── Model + Optimizer ─────────────────
    model = CDLModel(
        text_input_dim=args.text_feat_dim,
        nonverbal_input_dim=args.nonverbal_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    loss_fn   = CDLLoss(
        margin=args.contrastive_margin,
        lambda_cls=args.lambda_cls,
        lambda_contrast=args.lambda_contrast,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_f1    = 0.0
    best_mae   = 99.9
    history    = []
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CDL Model — DAIC-WOZ Depression Detection")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = {"total": 0, "l_reg": 0, "l_cls": 0, "l_contrast": 0}

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in epoch_loss:
                epoch_loss[k] += loss[k] if isinstance(loss[k], float) else loss[k].item()

        n_batch = len(train_loader)
        for k in epoch_loss:
            epoch_loss[k] /= n_batch

        # ── Evaluate ──
        metrics = evaluate(model, dev_loader, loss_fn, device)
        scheduler.step(metrics["loss"])

        row = {
            "epoch":       epoch,
            "train_loss":  epoch_loss["total"],
            "l_reg":       epoch_loss["l_reg"],
            "l_cls":       epoch_loss["l_cls"],
            "l_contrast":  epoch_loss["l_contrast"],
            **{k: v for k, v in metrics.items()
               if k not in ("dep_pred", "dep_true")},
        }
        history.append(row)

        print(
            f"Ep {epoch:03d} | "
            f"Loss {epoch_loss['total']:.4f} "
            f"(reg={epoch_loss['l_reg']:.3f} cls={epoch_loss['l_cls']:.3f} "
            f"ctr={epoch_loss['l_contrast']:.3f}) | "
            f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} "
            f"F1={metrics['f1']:.3f} Acc={metrics['acc']:.3f} | "
            f"Disc dep={metrics['disc_dep']:.3f} nodep={metrics['disc_nodep']:.3f}"
        )

        # Save best
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_f1_model.pt"))

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_mae_model.pt"))

    # ── Save history ──
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump([{k: float(v) if hasattr(v, 'item') else v
                    for k, v in row.items()} for row in history], f, indent=2)

    print(f"\nBest F1: {best_f1:.4f}  |  Best MAE: {best_mae:.4f}")
    print(f"Models saved to: {args.output_dir}/")

    # Final classification report
    print("\nFinal Dev Classification Report:")
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_f1_model.pt"),
                   map_location=device)
    )
    final = evaluate(model, dev_loader, loss_fn, device)
    print(classification_report(
        final["dep_true"], final["dep_pred"],
        target_names=["Non-Depressed", "Depressed"]
    ))
    print(f"Confusion Matrix:\n{confusion_matrix(final['dep_true'], final['dep_pred'])}")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CDL for DAIC-WOZ Depression Detection")
    # Data
    p.add_argument("--train_csv",  default="data/daicwoz/train_split_Depression_AVEC2017.csv")
    p.add_argument("--dev_csv",    default="data/daicwoz/dev_split_Depression_AVEC2017.csv")
    p.add_argument("--data_root",  default="data/daicwoz/")
    # Features
    p.add_argument("--text_feat_dim",  type=int, default=768,
                   help="Text embedding dim (768 for BERT)")
    p.add_argument("--nonverbal_dim",  type=int, default=88,
                   help="Non-verbal feature dim after preprocessing (default 88 for current DAIC-WOZ files)")
    # Model
    p.add_argument("--hidden_dim", type=int,   default=128)
    p.add_argument("--dropout",    type=float, default=0.3)
    # Loss
    p.add_argument("--contrastive_margin", type=float, default=1.0)
    p.add_argument("--lambda_cls",         type=float, default=1.0)
    p.add_argument("--lambda_contrast",    type=float, default=0.5)
    # Training
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--output_dir", default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)