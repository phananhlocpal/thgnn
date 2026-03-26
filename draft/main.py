"""
THGNN – PHQ-8 Depression Detection  (v3)
=========================================
Fixes vs v2:
  1. score_head dùng Sigmoid → output trong [0,1], nhân 24 → gradient luôn chảy
  2. binary_head: BCEWithLogits + focal, threshold được tune trên dev
  3. Focal loss γ=2 cho edge + binary
  4. Oversampling: lặp lại positive transcript trong train graph
  5. Warm-up LR 5 epoch + CosineAnnealing thủ công
  6. In chi tiết loss từng thành phần để debug
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, mean_absolute_error
)
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if (path.startswith("/") or path.startswith("\\")) and not os.path.splitdrive(path)[0]:
        return os.path.join(PROJECT_ROOT, path.lstrip("/\\"))
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


# =========================================================
# 0. Config
# =========================================================
@dataclass
class Config:
    train_csv: str = "data/daicwoz/train_split_Depression_AVEC2017.csv"
    dev_csv:   str = "data/daicwoz/dev_split_Depression_AVEC2017.csv"
    test_csv:  str = "data/daicwoz/full_test_split.csv"

    transcript_dir: str = "data/daicwoz"
    bert_name:      str = "bert-base-uncased"

    max_len:    int   = 256
    hidden_dim: int   = 320
    gnn_layers: int   = 3
    dropout:    float = 0.25

    lr:             float = 2e-4
    weight_decay:   float = 1e-4
    epochs:         int   = 80
    warmup_epochs:  int   = 5
    edge_mask_ratio: float = 0.15

    batch_size_encode: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed:   int = 42

    depression_threshold: int = 10

    lambda_edge:   float = 1.15
    lambda_binary: float = 0.5
    lambda_score:  float = 0.8   # cao để score head thực sự học

    focal_gamma: float = 2.0

    oversample_pos: int = 1   # 1 = no oversampling (safer for precision)
    max_pos_weight: float = 1.0

    # set ở runtime
    binary_pos_weight: float = 2.0
    _binary_threshold: float = 0.5
    _binary_alpha: float = 1.0


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


# =========================================================
# 1. PHQ-8 symptom definitions
# =========================================================
SYMPTOMS = [
    "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep",    "PHQ8_Tired",
    "PHQ8_Appetite",   "PHQ8_Failure",   "PHQ8_Concentrating", "PHQ8_Moving",
]

SYMPTOM_TEXT = {
    "PHQ8_NoInterest":    "Little interest or pleasure in doing things",
    "PHQ8_Depressed":     "Feeling down depressed or hopeless",
    "PHQ8_Sleep":         "Trouble falling or staying asleep or sleeping too much",
    "PHQ8_Tired":         "Feeling tired or having little energy",
    "PHQ8_Appetite":      "Poor appetite or overeating",
    "PHQ8_Failure":       "Feeling bad about yourself or that you are a failure",
    "PHQ8_Concentrating": "Trouble concentrating on things",
    "PHQ8_Moving":        "Moving or speaking slowly or being fidgety or restless",
}


# =========================================================
# 2. Read transcript (giữ cả 2 speaker để có ngữ cảnh)
# =========================================================
def read_transcript_file(participant_id: int, transcript_dir: str) -> str:
    path = os.path.join(resolve_path(transcript_dir), f"{participant_id}_TRANSCRIPT.csv")
    if not os.path.exists(path):
        return ""
    try:
        df = pd.read_csv(path, sep="\t")
        if "speaker" in df.columns and "value" in df.columns:
            lines = [f"{r.speaker}: {r.value}" for _, r in df[["speaker", "value"]].fillna("").iterrows()]
            return " ".join(lines).strip()
        return " ".join(df.astype(str).fillna("").values.flatten()).strip()
    except Exception:
        return ""


# =========================================================
# 3. BERT encoder (frozen)
# =========================================================
class BertTextEncoder:
    def __init__(self, model_name: str, max_len: int, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.max_len = max_len
        self.device  = device

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        all_vecs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding BERT"):
            batch = texts[i:i + batch_size]
            toks  = self.tokenizer(batch, padding=True, truncation=True,
                                   max_length=self.max_len, return_tensors="pt").to(self.device)
            out = self.model(**toks)
            all_vecs.append(out.last_hidden_state[:, 0, :].cpu())
        return torch.cat(all_vecs, dim=0)


# =========================================================
# 4. Data helpers
# =========================================================
def load_split(train_csv, dev_csv, test_csv):
    return (pd.read_csv(resolve_path(train_csv)),
            pd.read_csv(resolve_path(dev_csv)),
            pd.read_csv(resolve_path(test_csv)))


def get_pid_col(df: pd.DataFrame) -> str:
    return "Participant_ID" if "Participant_ID" in df.columns else "participant_ID"


def build_all_transcripts(df_list: List[pd.DataFrame], transcript_dir: str) -> Dict[int, str]:
    ids = []
    for df in df_list:
        ids.extend(df[get_pid_col(df)].tolist())
    return {pid: read_transcript_file(pid, transcript_dir) for pid in sorted(set(ids))}


def prepare_labels(df: pd.DataFrame, has_symptom_label: bool = True):
    pids      = df[get_pid_col(df)].tolist()
    y_symptom = df[SYMPTOMS].fillna(0).astype(int).values if has_symptom_label else None

    total_score = None
    for c in ("PHQ8_Score", "PHQ_Score"):
        if c in df.columns:
            total_score = df[c].fillna(0).astype(int).values; break

    y_binary = None
    for c in ("PHQ8_Binary", "PHQ_Binary"):
        if c in df.columns:
            y_binary = df[c].fillna(0).astype(int).values; break
    if y_binary is None and total_score is not None:
        y_binary = (total_score >= cfg.depression_threshold).astype(int)

    return pids, y_symptom, total_score, y_binary


# =========================================================
# 5. Oversampling positives
# =========================================================
def oversample_positives(emb, y_sym, y_score, y_bin, n: int = 2):
    pos_idx = np.where(y_bin == 1)[0]
    if len(pos_idx) == 0 or n <= 1:
        return emb, y_sym, y_score, y_bin

    rep_emb   = torch.cat([emb] + [emb[pos_idx]] * (n - 1), dim=0)
    rep_sym   = np.concatenate([y_sym]   + [y_sym[pos_idx]]   * (n - 1), axis=0)
    rep_score = np.concatenate([y_score] + [y_score[pos_idx]] * (n - 1), axis=0)
    rep_bin   = np.concatenate([y_bin]   + [y_bin[pos_idx]]   * (n - 1), axis=0)

    idx = np.random.permutation(len(rep_bin))
    return rep_emb[idx], rep_sym[idx], rep_score[idx], rep_bin[idx]


# =========================================================
# 6. Build hetero graph
# =========================================================
def make_edges(num_t: int, sym_labels: np.ndarray,
               mask_ratio: float, train_mode: bool):
    q_src, q_dst, q_y = [], [], []
    p_src, p_dst       = [], []

    for i in range(num_t):
        for j in range(len(SYMPTOMS)):
            lab = int(sym_labels[i, j])
            q_src.append(i); q_dst.append(j); q_y.append(lab)
            if lab > 0:
                p_src.append(i); p_dst.append(j)

    q_ei = torch.tensor([q_src, q_dst], dtype=torch.long)
    q_el = torch.tensor(q_y,            dtype=torch.long)
    p_ei = torch.tensor([p_src, p_dst], dtype=torch.long)

    if train_mode and p_ei.size(1) > 0:
        n_mask = max(1, int(mask_ratio * p_ei.size(1)))
        perm   = torch.randperm(p_ei.size(1))
        mp_ei  = p_ei[:, perm[n_mask:]]
    else:
        mp_ei = p_ei

    return mp_ei, q_ei, q_el


def build_hetero_data(t_emb, s_emb, sym_labels,
                      binary_labels=None, score_labels=None,
                      edge_mask_ratio=0.25, train_mode=True) -> HeteroData:
    data = HeteroData()
    N    = t_emb.size(0)

    mp_ei, q_ei, q_el = make_edges(N, sym_labels, edge_mask_ratio, train_mode)

    data["transcript"].x = t_emb.float()
    data["symptom"].x    = s_emb.float()

    data["transcript", "has_symptom",    "symptom"].edge_index   = mp_ei
    rev = torch.stack([mp_ei[1], mp_ei[0]], 0) if mp_ei.size(1) > 0 \
          else torch.empty((2, 0), dtype=torch.long)
    data["symptom", "rev_has_symptom", "transcript"].edge_index  = rev
    data["transcript", "predicts",     "symptom"].edge_label_index = q_ei
    data["transcript", "predicts",     "symptom"].edge_label       = q_el

    if binary_labels is not None:
        data["transcript"].binary_label = torch.tensor(binary_labels, dtype=torch.long)
    if score_labels is not None:
        data["transcript"].score_label  = torch.tensor(score_labels,  dtype=torch.float)

    return data


# =========================================================
# 7. Model
# =========================================================
class HeteroPHQGNN(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.transcript_proj = nn.Linear(in_dim, hidden_dim)
        self.symptom_proj    = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList([
            HeteroConv({
                ("transcript", "has_symptom",    "symptom"):    SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("symptom",    "rev_has_symptom", "transcript"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            }, aggr="sum")
            for _ in range(num_layers)
        ])

        # Edge decoder: 4 lớp {0,1,2,3}
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

        # Binary head → raw logit → sigmoid ở ngoài
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Score head: Sigmoid bên trong → output ∈ (0,1), nhân 24 ở predict
        # Đảm bảo gradient luôn chảy, không bị clamp triệt tiêu như ReLU
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x_dict, ei_dict):
        x = {
            "transcript": F.dropout(F.gelu(self.transcript_proj(x_dict["transcript"])), p=self.dropout, training=self.training),
            "symptom":    F.dropout(F.gelu(self.symptom_proj(x_dict["symptom"])),       p=self.dropout, training=self.training),
        }
        for conv in self.convs:
            x = conv(x, ei_dict)
            for k in x:
                x[k] = F.dropout(F.gelu(x[k]), p=self.dropout, training=self.training)
        return x

    def forward(self, data: HeteroData):
        z        = self.encode(data.x_dict, data.edge_index_dict)
        z_t, z_s = z["transcript"], z["symptom"]
        ei       = data["transcript", "predicts", "symptom"].edge_label_index

        zt = z_t[ei[0]]; zs = z_s[ei[1]]
        edge_logits  = self.edge_mlp(torch.cat([zt, zs, (zt - zs).abs()], -1))
        binary_logit = self.binary_head(z_t).squeeze(-1)      # [N]
        score_frac   = self.score_head(z_t).squeeze(-1)       # [N] ∈ (0,1)

        return edge_logits, binary_logit, score_frac, z


# =========================================================
# 8. Loss functions
# =========================================================
def focal_multiclass(logits, labels, gamma=2.0):
    counts  = torch.bincount(labels, minlength=logits.size(-1)).float().clamp(min=1)
    w       = (1.0 / counts) / (1.0 / counts).sum() * logits.size(-1)
    log_p   = F.log_softmax(logits, -1)
    pt      = log_p.exp().gather(1, labels.unsqueeze(1)).squeeze(1)
    log_pt  = log_p.gather(1, labels.unsqueeze(1)).squeeze(1)
    return (-w.to(logits.device)[labels] * (1 - pt) ** gamma * log_pt).mean()


def focal_binary(logit, labels, pos_weight=2.0, gamma=2.0):
    prob  = torch.sigmoid(logit)
    pt    = torch.where(labels == 1, prob, 1 - prob)
    alpha = torch.where(labels == 1,
                        torch.full_like(prob, pos_weight),
                        torch.ones_like(prob))
    return (-alpha * (1 - pt) ** gamma * pt.clamp(min=1e-8).log()).mean()


def compute_loss(edge_logits, binary_logit, score_frac, data, device):
    el     = data["transcript", "predicts", "symptom"].edge_label.to(device)
    l_edge = focal_multiclass(edge_logits, el, cfg.focal_gamma)
    total  = cfg.lambda_edge * l_edge
    log    = {"l_edge": l_edge.item()}

    if hasattr(data["transcript"], "binary_label"):
        bl    = data["transcript"].binary_label.to(device)
        l_bin = focal_binary(binary_logit, bl, cfg.binary_pos_weight, cfg.focal_gamma)
        total += cfg.lambda_binary * l_bin
        log["l_bin"] = l_bin.item()

    if hasattr(data["transcript"], "score_label"):
        sl      = data["transcript"].score_label.to(device) / 24.0  # normalize
        l_score = F.mse_loss(score_frac, sl)                        # MSE trên [0,1]
        total  += cfg.lambda_score * l_score
        log["l_score"] = l_score.item()

    return total, log


# =========================================================
# 9. Predict & Evaluate
# =========================================================
@torch.no_grad()
def predict(model, data, device, binary_threshold=0.5, binary_alpha=1.0):
    model.eval()
    data = data.to(device)
    edge_logits, binary_logit, score_frac, _ = model(data)

    N            = data["transcript"].x.size(0)
    pred_symptom = edge_logits.argmax(-1).cpu().numpy().reshape(N, len(SYMPTOMS))
    bin_prob     = torch.sigmoid(binary_logit).cpu().numpy()
    score_prob   = score_frac.cpu().numpy()
    pred_prob    = binary_alpha * bin_prob + (1.0 - binary_alpha) * score_prob
    pred_binary  = (pred_prob >= binary_threshold).astype(int)
    pred_score   = (score_frac.cpu().numpy() * 24).clip(0, 24).round().astype(int)

    return pred_symptom, pred_score, pred_binary


@torch.no_grad()
def evaluate(model, data, gold_sym, gold_score, gold_bin, device,
             binary_threshold=0.5, binary_alpha=1.0):
    pred_sym, pred_score, pred_bin = predict(
        model, data, device, binary_threshold=binary_threshold, binary_alpha=binary_alpha
    )
    return {
        "symptom_acc":      accuracy_score(gold_sym.reshape(-1), pred_sym.reshape(-1)),
        "symptom_macro_f1": f1_score(gold_sym.reshape(-1), pred_sym.reshape(-1),
                                     average="macro", zero_division=0),
        "score_mae":        mean_absolute_error(gold_score, pred_score),
        "binary_acc":       accuracy_score(gold_bin, pred_bin),
        "binary_f1":        f1_score(gold_bin, pred_bin, average="binary", zero_division=0),
        "binary_macro_f1":  f1_score(gold_bin, pred_bin, average="macro", zero_division=0),
    }, pred_sym, pred_score, pred_bin


# =========================================================
# 10. LR with warmup
# =========================================================
def get_lr(epoch, base_lr, warmup, total):
    if epoch <= warmup:
        return base_lr * epoch / warmup
    p = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * p))


# =========================================================
# 11. Main
# =========================================================
def main():
    print("Device:", cfg.device)

    train_df, dev_df, test_df = load_split(cfg.train_csv, cfg.dev_csv, cfg.test_csv)
    transcript_dict = build_all_transcripts([train_df, dev_df, test_df], cfg.transcript_dir)

    encoder     = BertTextEncoder(cfg.bert_name, cfg.max_len, cfg.device)
    symptom_emb = encoder.encode_texts([SYMPTOM_TEXT[s] for s in SYMPTOMS], batch_size=8)

    # ---- Train
    train_ids, train_y_sym, train_score, train_binary = prepare_labels(train_df)
    train_emb = encoder.encode_texts(
        [transcript_dict.get(pid, "") for pid in train_ids],
        batch_size=cfg.batch_size_encode,
    )

    n_neg = int((train_binary == 0).sum())
    n_pos = int((train_binary == 1).sum())
    print(f"Train class balance  — neg: {n_neg}, pos: {n_pos}")

    t_emb_os, t_sym_os, t_sc_os, t_bin_os = oversample_positives(
        train_emb, train_y_sym, train_score, train_binary, n=cfg.oversample_pos
    )
    os_neg = int((t_bin_os == 0).sum())
    os_pos = int((t_bin_os == 1).sum())
    raw_pos_weight = os_neg / max(1, os_pos)
    cfg.binary_pos_weight = round(min(cfg.max_pos_weight, raw_pos_weight), 2)
    print(f"After oversampling   — neg: {os_neg}, pos: {os_pos}, total: {len(t_bin_os)}")
    print(f"Raw pos_weight       — {raw_pos_weight:.2f}")
    print(f"Effective pos_weight — {cfg.binary_pos_weight}")

    train_data = build_hetero_data(
        t_emb_os, symptom_emb, t_sym_os,
        binary_labels=t_bin_os, score_labels=t_sc_os,
        edge_mask_ratio=cfg.edge_mask_ratio, train_mode=True,
    )

    # ---- Dev
    dev_ids, dev_y_sym, dev_score, dev_binary = prepare_labels(dev_df)
    dev_emb = encoder.encode_texts(
        [transcript_dict.get(pid, "") for pid in dev_ids],
        batch_size=cfg.batch_size_encode,
    )
    dev_data = build_hetero_data(
        dev_emb, symptom_emb, dev_y_sym,
        binary_labels=dev_binary, score_labels=dev_score,
        edge_mask_ratio=0.0, train_mode=False,
    )

    # ---- Model
    model = HeteroPHQGNN(
        in_dim=train_emb.size(1),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.gnn_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_combined = -1.0
    best_state    = None

    hdr = (f"{'Ep':>4} | {'Loss':>7} | {'l_edge':>7} | {'l_bin':>7} | "
           f"{'l_sc':>7} | {'Sym_F1':>7} | {'MAE':>7} | {'Bin_F1':>7} | {'Bin_mF1':>7} | {'Bin_Acc':>7}")
    print(f"\n{hdr}\n{'-' * len(hdr)}")

    for epoch in range(1, cfg.epochs + 1):
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(epoch, cfg.lr, cfg.warmup_epochs, cfg.epochs)

        model.train()
        optimizer.zero_grad()

        batch = train_data.to(cfg.device)
        edge_logits, binary_logit, score_frac, _ = model(batch)
        loss, log = compute_loss(edge_logits, binary_logit, score_frac, batch, cfg.device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        dev_m, _, _, _ = evaluate(
            model, dev_data, dev_y_sym, dev_score, dev_binary, cfg.device
        )

        print(
            f"{epoch:>4} | {loss.item():>7.4f} | "
            f"{log.get('l_edge', 0):>7.4f} | {log.get('l_bin', 0):>7.4f} | "
            f"{log.get('l_score', 0):>7.4f} | "
            f"{dev_m['symptom_macro_f1']:>7.4f} | {dev_m['score_mae']:>7.4f} | "
            f"{dev_m['binary_f1']:>7.4f} | {dev_m['binary_macro_f1']:>7.4f} | {dev_m['binary_acc']:>7.4f}"
        )

        combined = 0.5 * dev_m["binary_macro_f1"] + 0.5 * dev_m["symptom_macro_f1"]
        if combined > best_combined:
            best_combined = combined
            best_state    = {k: v.cpu() for k, v in model.state_dict().items()}

    print(f"\nBest combined (dev): {best_combined:.4f}")
    if best_state:
        model.load_state_dict(best_state)

    # ---- Tune binary fusion + threshold on dev
    model.eval()
    with torch.no_grad():
        _, bl, sf, _ = model(dev_data.to(cfg.device))
        bin_prob = torch.sigmoid(bl).cpu().numpy()
        score_prob = sf.cpu().numpy()

    best_alpha, best_thresh = 1.0, 0.5
    best_bmf1, best_bf1, best_bacc = -1.0, -1.0, -1.0
    for alpha in np.arange(0.0, 1.01, 0.1):
        mix_prob = alpha * bin_prob + (1.0 - alpha) * score_prob
        for thr in np.arange(0.05, 0.96, 0.01):
            preds = (mix_prob >= thr).astype(int)
            bmf1  = f1_score(dev_binary, preds, average="macro", zero_division=0)
            bf1   = f1_score(dev_binary, preds, average="binary", zero_division=0)
            bacc  = accuracy_score(dev_binary, preds)
            if (
                (bmf1 > best_bmf1)
                or (abs(bmf1 - best_bmf1) < 1e-12 and bf1 > best_bf1)
                or (abs(bmf1 - best_bmf1) < 1e-12 and abs(bf1 - best_bf1) < 1e-12 and bacc > best_bacc)
            ):
                best_bmf1 = bmf1
                best_bf1 = bf1
                best_bacc = bacc
                best_alpha = float(alpha)
                best_thresh = float(thr)

    cfg._binary_alpha = best_alpha
    cfg._binary_threshold = best_thresh
    print(
        f"Best binary fusion: alpha={best_alpha:.2f}, threshold={best_thresh:.2f} "
        f"(macroF1={best_bmf1:.4f}, F1={best_bf1:.4f}, Acc={best_bacc:.4f})"
    )

    # ---- Final dev metrics
    dev_m, dev_pred_sym, dev_pred_score, dev_pred_bin = evaluate(
        model, dev_data, dev_y_sym, dev_score, dev_binary, cfg.device,
        binary_threshold=cfg._binary_threshold,
        binary_alpha=cfg._binary_alpha,
    )

    print("\n[DEV METRICS]")
    for k, v in dev_m.items():
        print(f"  {k}: {v:.4f}")

    print("\n[DEV BINARY REPORT]")
    print(classification_report(dev_binary, dev_pred_bin, digits=4, zero_division=0))

    # ---- Test inference
    test_ids = test_df[get_pid_col(test_df)].tolist()
    test_emb = encoder.encode_texts(
        [transcript_dict.get(pid, "") for pid in test_ids],
        batch_size=cfg.batch_size_encode,
    )
    dummy_sym = np.zeros((len(test_ids), len(SYMPTOMS)), dtype=int)
    test_data = build_hetero_data(test_emb, symptom_emb, dummy_sym,
                                  edge_mask_ratio=0.0, train_mode=False)

    test_pred_sym, test_pred_score, test_pred_bin = predict(
        model, test_data, cfg.device,
        binary_threshold=cfg._binary_threshold,
        binary_alpha=cfg._binary_alpha,
    )

    result_df = pd.DataFrame({
        "Participant_ID":   test_ids,
        "Pred_PHQ8_Score":  test_pred_score,
        "Pred_PHQ8_Binary": test_pred_bin,
    })
    for i, s in enumerate(SYMPTOMS):
        result_df[s + "_pred"] = test_pred_sym[:, i]

    result_df.to_csv("test_predictions.csv", index=False)
    print("\nSaved test predictions to test_predictions.csv")

    if test_ids:
        print("\nExample prediction:")
        print(f"  Participant_ID : {test_ids[0]}")
        print(f"  Symptom vector : {test_pred_sym[0].tolist()}")
        print(f"  PHQ-8 score    : {int(test_pred_score[0])}")
        print(f"  Depressed      : {int(test_pred_bin[0])}")


if __name__ == "__main__":
    main()