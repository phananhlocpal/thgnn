"""
extract_bert_v4.py — Extract BERT embeddings từ DAIC-WOZ (v4, full rewrite).

Fixes vs v3_fixed
─────────────────
[FIX-CRITICAL] Participant-only embedding: Ellie context bị loại khỏi BERT
  input hoàn toàn. question_id được lưu trong metadata để dùng khi xây
  graph (edge theo cùng câu hỏi) — tránh shortcut learning.

[FIX-CRITICAL] Adaptive merge gap: thay vì fixed 2.0s, tính median
  inter-turn gap của từng participant rồi dùng median * 1.5 làm threshold.
  Depressed patients có response latency cao hơn → fixed threshold tạo
  systematic bias trong graph size.

[FIX-MAJOR] Mental-BERT pool strategy: dùng weighted average của 4 top layers
  với trọng số học được qua learned_layer_weights thay vì uniform mean.
  Layers cuối của Mental-BERT đã task-specific → cần weight khác nhau.

[FIX-MAJOR] Minimum quality gate sau khi clean: utterance bị mark zeros nếu
  sau khi remove tags còn < MIN_WORDS_FOR_EMBED=1 OR speech_rate > 8 wps
  (outlier, thường là crosstalk/noise).

[FIX] Sync tag skip: giữ nguyên từ v3.

[FIX] Export n_groups.txt: giữ nguyên từ v3.

[IMPROVE] Lưu ellie_question_id vào metadata để graph builder tạo
  "same-question" edges riêng.

[IMPROVE] Speech rate capping: > 6 wps được flag là suspicious
  (thường là ASR error hoặc overlapping speech).
"""

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EMBED_DIM             = 768
DEFAULT_MERGE_GAP_SEC = 2.0      # fallback nếu không đủ data để tính adaptive
MIN_WORDS_FOR_EMBED   = 1        # sau khi clean tags
MAX_SPEECH_RATE_WPS   = 8.0      # > 8 wps → suspicious, flag but still embed
NUM_POOL_LAYERS       = 4        # top-N layers để weighted pool

_TAG_RE   = re.compile(r"<[^>]+>")
_SYNC_RE  = re.compile(r"^\s*<\s*synch?\s*>\s*$", re.IGNORECASE)
_ELLIE_RE = re.compile(r"^(\S+)\s+\((.+)\)\s*$", re.DOTALL)

_SIGH_TAGS   = {"<sigh>", "<deep breath>", "<breath>"}
_LAUGH_TAGS  = {"<laughter>", "<laguhter>", "<laugher>", "<laugh>"}
_CRY_TAGS    = {"<cry>", "<crying>", "<sob>", "<sobbing>"}
_COUGH_TAGS  = {"<cough>", "<coughs>", "<clears throat>", "<sniff>",
                "<sniffle>", "<tisk>", "<tisk tisk>"}
_BREATH_TAGS = {"<deep breath>", "<breath>"}


# ─────────────────────────────────────────────────────────────────────────────
# Learned layer weighting (frozen after init, applied per-call)
# ─────────────────────────────────────────────────────────────────────────────

class LayerWeightedPool(nn.Module):
    """
    Learnable weighted average of the top-K hidden states.
    Weights are softmax-normalised so they sum to 1.
    For inference-only usage we simply use uniform init (equivalent to mean_top4
    from v3) and let callers optionally load fine-tuned weights.
    """
    def __init__(self, n_layers: int = NUM_POOL_LAYERS):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers))

    def forward(
        self,
        hidden_states: tuple,           # tuple of (B, T, H) tensors
        attention_mask: torch.Tensor,   # (B, T)
    ) -> torch.Tensor:                  # (B, H)
        # Take the last n_layers
        selected = torch.stack(hidden_states[-self.weights.shape[0]:], dim=0)  # (K, B, T, H)
        w = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)              # (K, 1, 1, 1)
        merged = (selected * w).sum(dim=0)                                     # (B, T, H)

        mask = attention_mask.unsqueeze(-1).expand_as(merged).float()
        pooled = (merged * mask).sum(1) / mask.sum(1).clamp(min=1e-9)         # (B, H)
        return pooled


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UtteranceGroup:
    group_id:             int
    row_indices:          list  = field(default_factory=list)
    start_time:           float = 0.0
    stop_time:            float = 0.0
    raw_texts:            list  = field(default_factory=list)
    response_latency_sec: float = -1.0
    # FIX: store question_id for graph edge construction, not in BERT input
    ellie_question_id:    str   = ""
    ellie_question_text:  str   = ""  # kept for metadata only
    clean_text:           str   = ""
    embed_input:          str   = ""  # participant text only (no Ellie context)
    used_zeros:           bool  = False
    suspicious_rate:      bool  = False
    speech_rate_wps:      float = 0.0
    has_sigh:             bool  = False
    has_laughter:         bool  = False
    has_breath:           bool  = False
    has_cry:              bool  = False
    has_cough:            bool  = False
    has_other_sound:      bool  = False

    @property
    def duration(self) -> float:
        return max(0.0, self.stop_time - self.start_time)

    @property
    def n_raw_turns(self) -> int:
        return len(self.row_indices)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"]    = self.duration
        d["n_raw_turns"] = self.n_raw_turns
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Transcript loading
# ─────────────────────────────────────────────────────────────────────────────

def load_transcript(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df["value"]        = df.get("value", pd.Series([""] * len(df))).fillna("").astype(str)
    df["start_time"]   = pd.to_numeric(
        df.get("start_time", df.get("Start_Time", 0)), errors="coerce"
    ).fillna(0.0)
    df["stop_time"]    = pd.to_numeric(
        df.get("stop_time",  df.get("End_Time",   0)), errors="coerce"
    ).fillna(0.0)
    df["speaker_clean"] = df["speaker"].str.strip().str.lower()
    return df


def parse_ellie_turn(value: str) -> tuple:
    m = _ELLIE_RE.match(value.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", value.strip()


def is_sync_only(text: str) -> bool:
    return bool(_SYNC_RE.match(text.strip()))


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive merge gap
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_merge_gap(
    df: pd.DataFrame,
    fallback: float = DEFAULT_MERGE_GAP_SEC,
    multiplier: float = 1.5,
) -> float:
    """
    Compute per-participant adaptive merge gap.

    Strategy: collect all within-participant gaps between consecutive
    participant rows (after ellie rows), take median, multiply by 1.5.
    Depressed patients have longer pauses → adaptive gap avoids
    splitting their utterances.

    Falls back to `fallback` if insufficient data.
    """
    part_rows = df[df["speaker_clean"] == "participant"].copy()
    part_rows = part_rows[~part_rows["value"].apply(is_sync_only)]

    if len(part_rows) < 4:
        return fallback

    starts = part_rows["start_time"].values
    stops  = part_rows["stop_time"].values

    # gaps between consecutive participant rows
    gaps = starts[1:] - stops[:-1]
    gaps = gaps[(gaps > 0) & (gaps < 30.0)]  # sanity filter

    if len(gaps) < 3:
        return fallback

    adaptive = float(np.median(gaps) * multiplier)
    # clamp to [0.5, 8.0]
    return float(np.clip(adaptive, 0.5, 8.0))


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning & paralinguistic extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_paralinguistic_flags(text: str) -> dict:
    tags_found = set(t.lower() for t in _TAG_RE.findall(text))
    known = _SIGH_TAGS | _LAUGH_TAGS | _CRY_TAGS | _COUGH_TAGS | _BREATH_TAGS
    return {
        "has_sigh"       : bool(tags_found & _SIGH_TAGS),
        "has_laughter"   : bool(tags_found & _LAUGH_TAGS),
        "has_breath"     : bool(tags_found & _BREATH_TAGS),
        "has_cry"        : bool(tags_found & _CRY_TAGS),
        "has_cough"      : bool(tags_found & _COUGH_TAGS),
        "has_other_sound": bool(tags_found - known - {"<sync>", "<synch>"}),
    }


def clean_text(text: str) -> str:
    cleaned = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_substantive(text: str, min_words: int = MIN_WORDS_FOR_EMBED) -> bool:
    return len(text.split()) >= min_words


# ─────────────────────────────────────────────────────────────────────────────
# Utterance group building
# ─────────────────────────────────────────────────────────────────────────────

def build_utterance_groups(
    df: pd.DataFrame,
    merge_gap_sec: float,
) -> List[UtteranceGroup]:
    groups: List[UtteranceGroup] = []
    current: Optional[UtteranceGroup] = None
    last_ellie_stop = -1.0
    last_ellie_qid  = ""
    last_ellie_qtxt = ""
    group_counter   = 0

    for idx, row in df.iterrows():
        speaker = row["speaker_clean"]

        if speaker == "ellie":
            if current is not None:
                groups.append(current)
                current = None
            last_ellie_stop = float(row["stop_time"])
            last_ellie_qid, last_ellie_qtxt = parse_ellie_turn(row["value"])
            continue

        if speaker != "participant":
            continue

        if is_sync_only(row["value"]):
            continue

        t_start = float(row["start_time"])
        t_stop  = float(row["stop_time"])

        if current is None:
            current = UtteranceGroup(
                group_id=group_counter,
                row_indices=[int(idx)],
                start_time=t_start,
                stop_time=t_stop,
                raw_texts=[row["value"]],
                response_latency_sec=(
                    round(t_start - last_ellie_stop, 3)
                    if last_ellie_stop >= 0 else -1.0
                ),
                ellie_question_id=last_ellie_qid,
                ellie_question_text=last_ellie_qtxt,
            )
            group_counter += 1
        else:
            gap = t_start - current.stop_time
            if gap <= merge_gap_sec:
                current.row_indices.append(int(idx))
                current.raw_texts.append(row["value"])
                current.stop_time = max(current.stop_time, t_stop)
            else:
                groups.append(current)
                current = UtteranceGroup(
                    group_id=group_counter,
                    row_indices=[int(idx)],
                    start_time=t_start,
                    stop_time=t_stop,
                    raw_texts=[row["value"]],
                    response_latency_sec=-1.0,
                    ellie_question_id=last_ellie_qid,
                    ellie_question_text=last_ellie_qtxt,
                )
                group_counter += 1

    if current is not None:
        groups.append(current)
    return groups


def enrich_group(group: UtteranceGroup) -> UtteranceGroup:
    # Paralinguistic flags
    combined = {
        "has_sigh": False, "has_laughter": False, "has_breath": False,
        "has_cry": False,  "has_cough": False,    "has_other_sound": False,
    }
    for raw in group.raw_texts:
        flags = extract_paralinguistic_flags(raw)
        for k in combined:
            combined[k] = combined[k] or flags[k]
    for k, v in combined.items():
        setattr(group, k, v)

    # Clean text
    cleaned_parts = [clean_text(t) for t in group.raw_texts]
    merged_clean  = " ".join(p for p in cleaned_parts if p).strip()
    group.clean_text = merged_clean

    # Quality gate
    if not is_substantive(merged_clean):
        group.used_zeros = True
        group.embed_input = ""
        return group

    # FIX: embed_input = participant text only, no Ellie context
    group.embed_input = merged_clean

    # Speech rate
    n_words = len(merged_clean.split())
    dur = max(group.duration, 0.1)
    sr  = round(n_words / dur, 3)
    group.speech_rate_wps = sr

    # Flag suspicious rate but still embed (don't discard — could be genuine fast speech)
    if sr > MAX_SPEECH_RATE_WPS:
        group.suspicious_rate = True
        log.debug(
            f"Group {group.group_id}: high speech rate {sr:.1f} wps — flagged."
        )

    return group


# ─────────────────────────────────────────────────────────────────────────────
# BERT embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(
    texts:       List[str],
    tokenizer,
    model,
    pool_layer:  LayerWeightedPool,
    device:      torch.device,
    max_len:     int,
    batch_size:  int,
) -> np.ndarray:
    """
    Embed a list of texts using learned weighted pooling of top-K layers.
    Uses participant text only (embed_input, no Ellie context).
    """
    all_embeddings = []
    model.eval()
    pool_layer.eval()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc   = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)

            out = model(**enc, output_hidden_states=True)
            emb = pool_layer(out.hidden_states, enc["attention_mask"])
            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-participant processing
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    transcript_path: str,
    tokenizer,
    model,
    pool_layer:      LayerWeightedPool,
    device:          torch.device,
    max_len:         int,
    batch_size:      int,
) -> tuple:
    df           = load_transcript(transcript_path)
    adaptive_gap = compute_adaptive_merge_gap(df)
    log.debug(f"  adaptive merge gap = {adaptive_gap:.2f}s")

    groups = build_utterance_groups(df, merge_gap_sec=adaptive_gap)

    if not groups:
        log.warning(f"No participant turns: {transcript_path}")
        return np.zeros((1, EMBED_DIM), dtype=np.float32), []

    groups = [enrich_group(g) for g in groups]

    embed_groups = [g for g in groups if not g.used_zeros]
    texts_to_embed = [g.embed_input for g in embed_groups]

    if texts_to_embed:
        embeddings_valid = embed_texts(
            texts_to_embed, tokenizer, model, pool_layer,
            device, max_len, batch_size,
        )
    else:
        embeddings_valid = np.zeros((0, EMBED_DIM), dtype=np.float32)

    embed_iter    = iter(embeddings_valid)
    all_embeddings = []
    for g in groups:
        if g.used_zeros:
            all_embeddings.append(np.zeros(EMBED_DIM, dtype=np.float32))
        else:
            all_embeddings.append(next(embed_iter))

    embeddings_arr = np.vstack(all_embeddings).astype(np.float32)
    metadata       = [g.to_dict() for g in groups]
    return embeddings_arr, metadata


def process_participant(
    pid:        int,
    data_root:  str,
    tokenizer,
    model,
    pool_layer: LayerWeightedPool,
    device:     torch.device,
    max_len:    int,
    batch_size: int,
    overwrite:  bool,
) -> bool:
    transcript_path = os.path.join(data_root, f"{pid}_TRANSCRIPT.csv")
    out_npy         = os.path.join(data_root, f"{pid}_text_feats.npy")
    out_meta        = os.path.join(data_root, f"{pid}_text_feats_meta.json")
    out_ngroups     = os.path.join(data_root, f"{pid}_n_groups.txt")

    if not overwrite and os.path.exists(out_npy):
        log.info(f"[{pid}] Already exists — skip (use --overwrite to redo)")
        return True

    if not os.path.exists(transcript_path):
        log.warning(f"[{pid}] Transcript not found: {transcript_path}")
        return False

    log.info(f"[{pid}] Processing …")
    embeddings, meta = extract_features(
        transcript_path, tokenizer, model, pool_layer,
        device, max_len, batch_size,
    )

    n_groups = len(meta)
    n_zeros  = sum(1 for m in meta if m.get("used_zeros"))
    n_suspicious = sum(1 for m in meta if m.get("suspicious_rate"))

    np.save(out_npy, embeddings)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(out_ngroups, "w") as f:
        f.write(str(n_groups))

    log.info(
        f"[{pid}] Done — groups={n_groups} "
        f"(zeros={n_zeros}, suspicious={n_suspicious}) "
        f"shape={embeddings.shape}"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace token helper
# ─────────────────────────────────────────────────────────────────────────────

def get_hf_token(cli_token=None, env_file=".env") -> Optional[str]:
    if cli_token:
        return cli_token
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "HF_TOKEN":
                    token = v.strip().strip('"').strip("'")
                    if token:
                        os.environ["HF_TOKEN"] = token
                        return token
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract BERT embeddings từ DAIC-WOZ (v4 — participant-only, adaptive gap)"
    )
    parser.add_argument("--data_root",  default="daicwoz/")
    parser.add_argument("--split_csv",  required=True)
    parser.add_argument("--model_name", default="mental/mental-bert-base-uncased")
    parser.add_argument("--hf_token",   default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len",    type=int, default=128,
                        help="Participant-only text is shorter; 128 is sufficient.")
    parser.add_argument("--overwrite",  action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device     : {device}")
    log.info(f"Model      : {args.model_name}")
    log.info(f"Max len    : {args.max_len} (participant-only, no Ellie context)")
    log.info(f"Embed mode : learned weighted pool of top-{NUM_POOL_LAYERS} layers")

    hf_kwargs = {"token": get_hf_token(args.hf_token)} if get_hf_token(args.hf_token) else {}
    tokenizer  = AutoTokenizer.from_pretrained(args.model_name, **hf_kwargs)
    bert_model = AutoModel.from_pretrained(args.model_name, **hf_kwargs).to(device)
    pool_layer = LayerWeightedPool(n_layers=NUM_POOL_LAYERS).to(device)

    split_df = pd.read_csv(args.split_csv)
    split_df.columns = [c.strip() for c in split_df.columns]
    pids = split_df["Participant_ID"].astype(int).tolist()
    log.info(f"Participants: {len(pids)}")

    success = 0
    for pid in pids:
        ok = process_participant(
            pid=pid, data_root=args.data_root,
            tokenizer=tokenizer, model=bert_model,
            pool_layer=pool_layer, device=device,
            max_len=args.max_len, batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        if ok:
            success += 1

    log.info(f"\nDone: {success}/{len(pids)} participants.")
    log.info("NOTE: embed_input = participant text only. Ellie question_id")
    log.info("      is in metadata for graph edge construction (daicwoz_dataset.py).")


if __name__ == "__main__":
    main()