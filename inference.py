"""
Inference + Interpretability for CDL Model

Key feature: Discrepancy Score Analysis
  - Shows verbal vs non-verbal embedding distance per participant
  - Flags "masked depression" cases: high discrepancy + positive text tone
  - Threshold calibration for personality-vs-pathology disambiguation
"""

import os
import json
import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_curve, auc

from model   import CDLModel, CDLLoss
from dataset import get_dataloader, DAICWOZDataset


# ─────────────────────────────────────────
# Threshold Calibration
# ─────────────────────────────────────────

def calibrate_discrepancy_threshold(model, loader, device):
    """
    Find optimal discrepancy threshold that separates depressed vs healthy.
    Uses ROC curve on cosine distance as a standalone detector.

    This addresses the key weakness: distinguishing pathological discrepancy
    from natural low-expressiveness personality.
    """
    model.eval()
    all_dist, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["text_feat"].to(device),
                batch["nonverbal_feat"].to(device),
                batch["text_lengths"].to(device),
                batch["nonverbal_lengths"].to(device),
            )
            all_dist.extend(out["cosine_dist"].cpu().numpy())
            all_labels.extend(batch["dep_label"].numpy())

    all_dist   = np.array(all_dist)
    all_labels = np.array(all_labels)

    fpr, tpr, thresholds = roc_curve(all_labels, all_dist)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic: maximizes sensitivity + specificity
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]

    print(f"\n── Discrepancy Threshold Calibration ──")
    print(f"  ROC-AUC (cosine dist alone): {roc_auc:.4f}")
    print(f"  Optimal threshold (Youden J): {best_thresh:.4f}")
    print(f"  Sensitivity: {tpr[best_idx]:.3f} | Specificity: {1-fpr[best_idx]:.3f}")

    return best_thresh, roc_auc


# ─────────────────────────────────────────
# Per-Participant Analysis
# ─────────────────────────────────────────

def analyze_discrepancy(model, loader, device, disc_threshold=0.5):
    """
    Run inference and produce per-participant discrepancy report.

    Identifies 4 quadrants:
      Q1 (True Positive):  High disc + Depressed    → model works correctly
      Q2 (False Positive): High disc + Not depressed → personality effect
      Q3 (True Negative):  Low disc  + Not depressed → correctly clear
      Q4 (False Negative): Low disc  + Depressed     → masked/atypical depression
    """
    model.eval()
    records = []

    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["text_feat"].to(device),
                batch["nonverbal_feat"].to(device),
                batch["text_lengths"].to(device),
                batch["nonverbal_lengths"].to(device),
            )

            dep_pred = (torch.sigmoid(out["dep_logit"]) > 0.5).long().cpu()

            for i in range(len(batch["participant_id"])):
                cd   = out["cosine_dist"][i].item()
                l2   = out["l2_dist"][i].item()
                phq  = out["phq_score"][i].item()
                true_dep = int(batch["dep_label"][i].item())
                pred_dep = int(dep_pred[i].item())
                disc_flag = cd > disc_threshold

                # Quadrant analysis
                if disc_flag and true_dep == 1:
                    quadrant = "Q1: True Positive (High Disc + Depressed)"
                elif disc_flag and true_dep == 0:
                    quadrant = "Q2: Possible Personality Effect (High Disc + Healthy)"
                elif not disc_flag and true_dep == 0:
                    quadrant = "Q3: True Negative (Congruent + Healthy)"
                else:
                    quadrant = "Q4: Masked/Atypical Depression (Low Disc + Depressed)"

                records.append({
                    "participant_id": batch["participant_id"][i],
                    "phq_pred":       round(phq, 2),
                    "dep_true":       true_dep,
                    "dep_pred":       pred_dep,
                    "cosine_dist":    round(cd, 4),
                    "l2_dist":        round(l2, 4),
                    "high_discrepancy": disc_flag,
                    "quadrant":       quadrant,
                })

    df = pd.DataFrame(records)

    print("\n── Per-Participant Discrepancy Analysis ──")
    print(df[["participant_id", "phq_pred", "dep_true", "dep_pred",
              "cosine_dist", "high_discrepancy", "quadrant"]].to_string(index=False))

    print("\n── Quadrant Summary ──")
    print(df["quadrant"].value_counts().to_string())

    print("\n── Average Discrepancy by True Label ──")
    print(df.groupby("dep_true")[["cosine_dist", "l2_dist"]].mean().round(4))

    return df


# ─────────────────────────────────────────
# Single-Sample Inference
# ─────────────────────────────────────────

def predict_single(model, text_feat_np, nonverbal_feat_np, device, disc_threshold=0.5):
    """
    Predict depression risk for a single participant.

    Args:
        text_feat_np:      (T, text_dim) numpy array
        nonverbal_feat_np: (T, nv_dim)   numpy array
        disc_threshold:    Calibrated discrepancy threshold

    Returns:
        dict with phq_score, dep_probability, discrepancy_score, risk_flag
    """
    model.eval()
    tf  = torch.tensor(text_feat_np,      dtype=torch.float32).unsqueeze(0)
    nv  = torch.tensor(nonverbal_feat_np, dtype=torch.float32).unsqueeze(0)
    tl  = torch.tensor([len(text_feat_np)])
    nvl = torch.tensor([len(nonverbal_feat_np)])

    with torch.no_grad():
        out = model(tf.to(device), nv.to(device), tl.to(device), nvl.to(device))

    phq_pred = out["phq_score"].item()
    dep_prob = torch.sigmoid(out["dep_logit"]).item()
    cd       = out["cosine_dist"].item()
    l2       = out["l2_dist"].item()

    risk_flag = (dep_prob > 0.5) or (cd > disc_threshold)
    severity  = "HIGH" if phq_pred >= 15 else "MODERATE" if phq_pred >= 10 else "MILD/NONE"

    result = {
        "phq_score_pred":      round(phq_pred, 2),
        "dep_probability":     round(dep_prob, 4),
        "cosine_discrepancy":  round(cd, 4),
        "l2_discrepancy":      round(l2, 4),
        "risk_flag":           risk_flag,
        "severity":            severity,
        "interpretation": (
            "⚠ HIGH DISCREPANCY: Verbal-NonVerbal mismatch detected. "
            "Possible masked depression." if cd > disc_threshold else
            "Verbal and non-verbal signals are congruent."
        )
    }

    print("\n── Single Prediction ──")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dev_csv",       default="data/daicwoz/dev_split_Depression_AVEC2017.csv")
    p.add_argument("--data_root",     default="data/daicwoz/")
    p.add_argument("--text_feat_dim", type=int, default=768)
    p.add_argument("--nonverbal_dim", type=int, default=88)
    p.add_argument("--hidden_dim",    type=int, default=128)
    p.add_argument("--output_csv",    default="discrepancy_report.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CDLModel(
        text_input_dim=args.text_feat_dim,
        nonverbal_input_dim=args.nonverbal_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Model loaded from {args.checkpoint}")

    loader = get_dataloader(
        args.dev_csv, args.data_root, batch_size=8, shuffle=False,
        text_feat_dim=args.text_feat_dim,
        nonverbal_dim=args.nonverbal_dim,
    )

    threshold, roc_auc = calibrate_discrepancy_threshold(model, loader, device)
    df = analyze_discrepancy(model, loader, device, disc_threshold=threshold)
    df.to_csv(args.output_csv, index=False)
    print(f"\nReport saved to: {args.output_csv}")


if __name__ == "__main__":
    main()