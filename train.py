"""
train.py v2 — Training script cho HMSGNet v2.
Xem docstring đầy đủ trong file gốc.
"""

import argparse, math, os, random, time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch_geometric.loader import DataLoader

from daicwoz_dataset import DaicWozDataset
from model import HMSGNet, compute_loss

try:
    from edaic_dataset import DepressionDataset
    HAS_EDAIC = True
except ImportError:
    HAS_EDAIC = False

SEED = 42

DATASET_CFG: Dict[str, dict] = {
    "edaic": {
        "cache_dir":      "C:/Users/Administrator/Desktop/thgnn/cache",
        "checkpoint_dir": Path("C:/Users/Administrator/Desktop/thgnn/checkpoints"),
        "text_dim": 777, "audio_dim": 777,
        "hidden_dim": 256, "num_gnn_layers": 3, "num_edge_types": 6,
        "n_heads": 4, "dropout": 0.35, "drop_edge": 0.20, "feat_noise": 0.02,
        "focal_alpha": 0.80, "label_smoothing": 0.0,
        "w_symptom": 0.3, "w_phq": 0.1,
        "aug_sds": False, "aug_copies": 0, "aug_prob": 0.8,
        "lr": 3e-4, "weight_decay": 2e-4,
        "warmup_epochs": 5, "cosine_epochs": 250, "eta_min": 1e-6,
        "batch_size": 8, "max_epochs": 250, "early_stop_pat": 30,
        "use_weighted_sampler": False,
    },
    "daicwoz": {
        "cache_dir":      "C:/Users/Administrator/Desktop/thgnn/cache_daicwoz",
        "checkpoint_dir": Path("C:/Users/Administrator/Desktop/thgnn/checkpoints_daicwoz"),
        # 768 BERT + 9 acoustic | 768 WavLM + 9 acoustic
        "text_dim": 777, "audio_dim": 777,
        # Upscaled nhờ SDS: 107 → ~321 training samples
        "hidden_dim": 128, "num_gnn_layers": 2, "num_edge_types": 6,
        "n_heads": 4, "dropout": 0.60, "drop_edge": 0.20, "feat_noise": 0.05,
        "focal_alpha": 0.75, "label_smoothing": 0.05,
        "w_symptom": 0.20, "w_phq": 0.05,
        # SDS: 107 * (1 + 2) = 321 effective training samples
        "aug_sds": True, "aug_copies": 2, "aug_prob": 0.80,
        "lr": 1e-4, "weight_decay": 5e-3,
        "warmup_epochs": 2, "cosine_epochs": 400, "eta_min": 1e-7,
        "batch_size": 8, "max_epochs": 400, "early_stop_pat": 80,
        "use_weighted_sampler": True,
    },
}

NUM_SYMPTOMS  = 8
MAX_GRAD_NORM = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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
        "accuracy":    accuracy_score(labels, preds),
        "f1_macro":    f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision":   precision_score(labels, preds, zero_division=0),
        "recall":      recall_score(labels, preds, zero_division=0),
        "threshold":   threshold,
    }
    try:    m["auc"] = roc_auc_score(labels, probs)
    except: m["auc"] = float("nan")
    return m


def run_epoch(model, loader, optimizer, device, is_train,
              w_symptom=0.3, w_phq=0.1, focal_alpha=0.80, label_smoothing=0.0):
    model.train() if is_train else model.eval()
    totals = {k: 0.0 for k in ("loss_total","loss_dep","loss_symptom","loss_phq")}
    n, all_labels, all_probs = 0, [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            dep_logit, sym_logits, phq_pred = model(batch)
            dep_labels  = batch.y.squeeze().long()
            phq8_labels = batch.phq8.view(-1, NUM_SYMPTOMS).float()
            phq_scores  = batch.phq_score.squeeze().float()
            loss, ld = compute_loss(dep_logit, sym_logits, phq_pred,
                                    dep_labels.float(), phq8_labels, phq_scores,
                                    w_symptom=w_symptom, w_phq=w_phq,
                                    focal_alpha=focal_alpha,
                                    label_smoothing=label_smoothing, device=device)
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


def build_scheduler(optimizer, warmup_epochs, cosine_epochs, eta_min=1e-6):
    def wfn(e): return float(e+1)/max(1,warmup_epochs) if e < warmup_epochs else 1.0
    return SequentialLR(optimizer,
        schedulers=[LambdaLR(optimizer, wfn),
                    CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=eta_min)],
        milestones=[warmup_epochs])


def log_epoch(epoch, phase, losses, metrics, lr, elapsed):
    print(f"[Epoch {epoch:03d}] {phase:5s} | "
          f"loss={losses['loss_total']:.4f} (dep={losses['loss_dep']:.4f} "
          f"sym={losses['loss_symptom']:.4f} phq={losses['loss_phq']:.4f}) | "
          f"acc={metrics.get('accuracy',0):.4f} "
          f"f1_macro={metrics.get('f1_macro',0):.4f} "
          f"prec={metrics.get('precision',0):.4f} "
          f"rec={metrics.get('recall',0):.4f} "
          f"auc={metrics.get('auc',float('nan')):.4f} "
          f"thr={metrics.get('threshold',0.5):.2f} | "
          f"lr={lr:.2e} | {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      choices=["edaic","daicwoz"], default="edaic")
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--aug-sds",      action="store_true")
    parser.add_argument("--no-aug-sds",   action="store_true")
    args = parser.parse_args()

    cfg = dict(DATASET_CFG[args.dataset])
    if args.aug_sds:    cfg["aug_sds"] = True
    if args.no_aug_sds: cfg["aug_sds"] = False

    ckpt_dir = cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"

    set_seed(SEED)
    print(f"Dataset: {args.dataset}  Device: {DEVICE}  SDS: {cfg['aug_sds']}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    if args.dataset == "edaic":
        if not HAS_EDAIC: raise ImportError("edaic_dataset.py not found")
        train_ds = DepressionDataset(split="train", root=cfg["cache_dir"],
                                     force_reload=args.force_reload)
        dev_ds   = DepressionDataset(split="dev",   root=cfg["cache_dir"],
                                     force_reload=args.force_reload)
    else:
        train_ds = DaicWozDataset(split="train", root=cfg["cache_dir"],
                                  force_reload=args.force_reload,
                                  aug_sds=cfg["aug_sds"],
                                  aug_copies=cfg.get("aug_copies", 2),
                                  aug_prob=cfg.get("aug_prob", 0.80))
        dev_ds   = DaicWozDataset(split="dev",   root=cfg["cache_dir"],
                                  force_reload=args.force_reload, aug_sds=False)

    # ── Loaders ───────────────────────────────────────────────────────────────
    if cfg.get("use_weighted_sampler", False):
        from torch.utils.data import WeightedRandomSampler
        la = train_ds.labels().numpy()
        cc = np.bincount(la)
        sw = (1.0 / cc)[la]
        sampler     = WeightedRandomSampler(torch.from_numpy(sw).float(),
                                            len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  sampler=sampler, num_workers=0,
                                  pin_memory=(DEVICE.type=="cuda"))
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True, num_workers=0,
                                  pin_memory=(DEVICE.type=="cuda"))
    dev_loader = DataLoader(dev_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=0,
                            pin_memory=(DEVICE.type=="cuda"))

    lc = torch.bincount(train_ds.labels())
    print(f"  Train: {len(train_ds)} (neg={lc[0]}, pos={lc[1]})  Dev: {len(dev_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HMSGNet(
        hidden_dim=cfg["hidden_dim"], num_gnn_layers=cfg["num_gnn_layers"],
        num_edge_types=cfg["num_edge_types"], num_symptoms=NUM_SYMPTOMS,
        n_heads=cfg["n_heads"], dropout=cfg["dropout"],
        drop_edge=cfg["drop_edge"], feat_noise=cfg["feat_noise"],
        text_dim=cfg["text_dim"], audio_dim=cfg["audio_dim"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}  hidden={cfg['hidden_dim']} "
          f"layers={cfg['num_gnn_layers']} edges={cfg['num_edge_types']} "
          f"dropout={cfg['dropout']} lr={cfg['lr']:.1e}")

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg["warmup_epochs"],
                                 cfg["cosine_epochs"], cfg["eta_min"])

    start_epoch, best_metric = 1, 0.0
    if best_path.exists() and not args.force_reload:
        ckpt = torch.load(str(best_path), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_metric  = ckpt.get("best_metric", ckpt.get("best_auc", 0.0))
        print(f"  Resumed from epoch {start_epoch-1}, best_metric={best_metric:.4f}")

    no_improve = 0

    for epoch in range(start_epoch, cfg["max_epochs"] + 1):
        t0 = time.time()
        tl, tlab, tprob = run_epoch(model, train_loader, optimizer, DEVICE, True,
                                     cfg["w_symptom"], cfg["w_phq"],
                                     cfg["focal_alpha"], cfg.get("label_smoothing",0.0))
        log_epoch(epoch, "TRAIN", tl, compute_metrics(tlab, tprob),
                  optimizer.param_groups[0]["lr"], time.time()-t0)

        t1 = time.time()
        dl, dlab, dprob = run_epoch(model, dev_loader, None, DEVICE, False,
                                     cfg["w_symptom"], cfg["w_phq"],
                                     cfg["focal_alpha"], 0.0)
        bthr  = find_best_threshold(dlab, dprob)
        dmets = compute_metrics(dlab, dprob, threshold=bthr)
        log_epoch(epoch, "DEV", dl, dmets, optimizer.param_groups[0]["lr"],
                  time.time()-t1)

        scheduler.step()

        dev_auc = dmets.get("auc", 0.0)
        dev_f1  = dmets.get("f1_macro", 0.0)

        if dev_auc > best_metric:
            best_metric = dev_auc
            no_improve  = 0
            torch.save({
                "epoch": epoch, "best_metric": best_metric, "best_f1": dev_f1,
                "best_threshold": bthr, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, str(best_path))
            print(f"  * New best AUC={best_metric:.4f} (F1={dev_f1:.4f} thr={bthr:.2f}) → saved")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{cfg['early_stop_pat']}). Best={best_metric:.4f}")
            if no_improve >= cfg["early_stop_pat"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"\nDone. Best dev AUC = {best_metric:.4f}")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()