"""
extract_bert_v3_fixed.py — Extract BERT embeddings từ DAIC-WOZ.

Fixes vs original v3:
  1. [FIX-CRITICAL] MIN_WORDS_FOR_EMBED 2→1: single-word responses ("yes", "no",
     "mhm") are clinically meaningful in depression interviews. They were being
     replaced by zero vectors, creating ~20-30% noise nodes. Now only truly
     empty/tag-only utterances get zeros.
  2. [FIX-CRITICAL] Export N_GROUPS to a side file {ID}_n_groups.txt so that
     daicwoz_dataset.py can read the CORRECT node count instead of counting
     raw transcript rows (which is 2x the actual group count).
  3. [FIX-MAJOR] Ellie context truncation: long Ellie texts (e.g. 66-word intro)
     dominated BERT attention, drowning out short participant responses.
     Now cap Ellie context at MAX_ELLIE_WORDS=20 (keep last 20 words, which
     carry the actual question).
  4. [FIX-MAJOR] Skip <sync> rows entirely: they are alignment markers, not speech.
     Previously they created groups with empty text → zeros vectors at position 0.
  5. [IMPROVE] Use [SEP] only when tokenizer actually has it; for models without
     NSP training (like mental-bert), use a simple " | " separator instead.

  ── SHORTCUT LEARNING FIXES (v3_fixed → anti-shortcut) ──────────────────────
  6. [FIX-SHORTCUT] --context_mode flag với 3 giá trị:
       "none"       : chỉ participant text (no Ellie context) — gold standard
       "truncated"  : Ellie context cắt ≤ MAX_ELLIE_WORDS (default, original behavior)
       "full"       : full Ellie text (upper-bound shortcut baseline)
     Ablation quan trọng: train với cả 3 modes, nếu "none" ≈ "truncated" → model
     học genuine signals. Nếu "none" << "truncated" → shortcut learning detected.
  7. [FIX-SHORTCUT] participant_only mode: khi context_mode="none", BERT không
     thấy bất kỳ text nào của interviewer. Đây là setting để report trong paper.
  8. [FIX-SHORTCUT] Export context_mode vào metadata JSON để audit sau này.
  9. [FIX-SHORTCUT] Lưu embedding riêng theo context_mode:
       {pid}_text_feats_ctx_none.npy
       {pid}_text_feats_ctx_truncated.npy  (default)
       {pid}_text_feats_ctx_full.npy
     → Cho phép load ablation versions mà không cần re-extract tất cả.

All other features (paralinguistic flags, speech rate, merge logic, pool modes)
are preserved from v3.
"""

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
EMBED_DIM = 768
DEFAULT_MERGE_GAP_SEC = 2.0

# FIX: Lowered from 2 to 1. Single-word answers like "yes", "no", "mhm"
# are clinically significant in depression screening interviews.
# Only truly empty strings (after tag removal) get zeros.
MIN_WORDS_FOR_EMBED = 1

# FIX: Cap Ellie context to avoid drowning participant text in BERT attention.
# Keep the LAST N words (the actual question), drop preamble.
MAX_ELLIE_WORDS = 20

# Valid context modes for ablation study
CONTEXT_MODES = ("none", "truncated", "full")

# Regex
_TAG_RE  = re.compile(r"<[^>]+>")
_ELLIE_RE = re.compile(r"^(\S+)\s+\((.+)\)\s*$", re.DOTALL)
_SYNC_RE  = re.compile(r"^\s*<\s*synch?\s*>\s*$", re.IGNORECASE)

# Tag classification
_SIGH_TAGS   = {"<sigh>", "<deep breath>", "<breath>"}
_LAUGH_TAGS  = {"<laughter>", "<laguhter>", "<laugher>", "<laugh>"}
_CRY_TAGS    = {"<cry>", "<crying>", "<sob>", "<sobbing>"}
_COUGH_TAGS  = {"<cough>", "<coughs>", "<clears throat>", "<sniff>",
                "<sniffle>", "<tisk>", "<tisk tisk>"}
_BREATH_TAGS = {"<deep breath>", "<breath>"}


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────
@dataclass
class UtteranceGroup:
    group_id: int
    row_indices: list = field(default_factory=list)
    start_time: float = 0.0
    stop_time: float = 0.0
    raw_texts: list = field(default_factory=list)
    response_latency_sec: float = -1.0
    ellie_question_id: str = ""
    ellie_question_text: str = ""
    clean_text: str = ""
    context_input: str = ""
    context_mode_used: str = "truncated"   # NEW: track which mode was used
    used_zeros: bool = False
    speech_rate_wps: float = 0.0
    has_sigh: bool = False
    has_laughter: bool = False
    has_breath: bool = False
    has_cry: bool = False
    has_cough: bool = False
    has_other_sound: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.stop_time - self.start_time)

    @property
    def n_raw_turns(self) -> int:
        return len(self.row_indices)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        d["n_raw_turns"] = self.n_raw_turns
        return d


# ──────────────────────────────────────────────────────────────
# Transcript helpers
# ──────────────────────────────────────────────────────────────
def load_transcript(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df["value"] = df.get("value", pd.Series([""] * len(df))).fillna("").astype(str)
    df["start_time"] = pd.to_numeric(
        df.get("start_time", df.get("Start_Time", 0)), errors="coerce"
    ).fillna(0.0)
    df["stop_time"] = pd.to_numeric(
        df.get("stop_time", df.get("End_Time", 0)), errors="coerce"
    ).fillna(0.0)
    df["speaker_clean"] = df["speaker"].str.strip().str.lower()
    return df


def parse_ellie_turn(value: str) -> tuple[str, str]:
    m = _ELLIE_RE.match(value.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", value.strip()


def is_sync_only(text: str) -> bool:
    """Check if text is just a <sync>/<synch> marker."""
    return bool(_SYNC_RE.match(text.strip()))


# ──────────────────────────────────────────────────────────────
# Text cleaning & paralinguistic extraction
# ──────────────────────────────────────────────────────────────
def extract_paralinguistic_flags(text: str) -> dict:
    tags_found = set(t.lower() for t in _TAG_RE.findall(text))
    known_tags = _SIGH_TAGS | _LAUGH_TAGS | _CRY_TAGS | _COUGH_TAGS | _BREATH_TAGS
    return {
        "has_sigh"       : bool(tags_found & _SIGH_TAGS),
        "has_laughter"   : bool(tags_found & _LAUGH_TAGS),
        "has_breath"     : bool(tags_found & _BREATH_TAGS),
        "has_cry"        : bool(tags_found & _CRY_TAGS),
        "has_cough"      : bool(tags_found & _COUGH_TAGS),
        "has_other_sound": bool(tags_found - known_tags - {"<sync>", "<synch>"}),
    }


def clean_text(text: str) -> str:
    cleaned = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_substantive(text: str, min_words: int = MIN_WORDS_FOR_EMBED) -> bool:
    return len(text.split()) >= min_words


def truncate_ellie_text(text: str, max_words: int = MAX_ELLIE_WORDS) -> str:
    """Keep only the last max_words of Ellie text (the actual question part)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[-max_words:])


def build_context_input(
    ellie_text: str,
    participant_text: str,
    context_mode: str,
    separator: str = " [SEP] ",
) -> str:
    """
    Build the input string for BERT based on context_mode.

    SHORTCUT ABLATION:
      "none"      → participant text only (no interviewer influence)
      "truncated" → last MAX_ELLIE_WORDS of Ellie + participant
      "full"      → full Ellie text + participant (maximum shortcut risk)

    For paper: report all 3 modes. If F1(none) ≈ F1(truncated), the model
    learns genuine signals. If F1(none) << F1(truncated), shortcut learning.
    """
    if context_mode == "none" or not ellie_text:
        return participant_text

    if context_mode == "truncated":
        ellie_part = truncate_ellie_text(ellie_text, MAX_ELLIE_WORDS)
    elif context_mode == "full":
        ellie_part = ellie_text.strip()
    else:
        raise ValueError(f"Unknown context_mode: {context_mode!r}. "
                         f"Must be one of {CONTEXT_MODES}")

    return f"{ellie_part.strip()}{separator}{participant_text}"


# ──────────────────────────────────────────────────────────────
# Utterance group building
# ──────────────────────────────────────────────────────────────
def build_utterance_groups(
    df: pd.DataFrame,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
) -> list[UtteranceGroup]:
    groups: list[UtteranceGroup] = []
    current_group: Optional[UtteranceGroup] = None
    last_ellie_stop: float = -1.0
    last_ellie_qid: str = ""
    last_ellie_qtext: str = ""
    group_counter = 0

    for idx, row in df.iterrows():
        speaker = row["speaker_clean"]

        if speaker == "ellie":
            if current_group is not None:
                groups.append(current_group)
                current_group = None
            last_ellie_stop = float(row["stop_time"])
            last_ellie_qid, last_ellie_qtext = parse_ellie_turn(row["value"])
            continue

        if speaker != "participant":
            continue

        # FIX: Skip <sync>/<synch> rows — they are alignment markers, not speech.
        if is_sync_only(row["value"]):
            continue

        t_start = float(row["start_time"])
        t_stop  = float(row["stop_time"])

        if current_group is None:
            current_group = UtteranceGroup(
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
                ellie_question_text=last_ellie_qtext,
            )
            group_counter += 1
        else:
            gap = t_start - current_group.stop_time
            if gap <= merge_gap_sec:
                current_group.row_indices.append(int(idx))
                current_group.raw_texts.append(row["value"])
                current_group.stop_time = max(current_group.stop_time, t_stop)
            else:
                groups.append(current_group)
                current_group = UtteranceGroup(
                    group_id=group_counter,
                    row_indices=[int(idx)],
                    start_time=t_start,
                    stop_time=t_stop,
                    raw_texts=[row["value"]],
                    response_latency_sec=-1.0,
                    ellie_question_id=last_ellie_qid,
                    ellie_question_text=last_ellie_qtext,
                )
                group_counter += 1

    if current_group is not None:
        groups.append(current_group)

    return groups


def enrich_group(
    group: UtteranceGroup,
    context_mode: str,
    separator: str = " [SEP] ",
) -> UtteranceGroup:
    """
    Enrich group with cleaned text, paralinguistic flags, speech rate,
    and context_input based on the requested context_mode.
    """
    # Paralinguistic flags
    combined_flags = {
        "has_sigh": False, "has_laughter": False, "has_breath": False,
        "has_cry": False,  "has_cough": False,    "has_other_sound": False,
    }
    for raw in group.raw_texts:
        flags = extract_paralinguistic_flags(raw)
        for k in combined_flags:
            combined_flags[k] = combined_flags[k] or flags[k]
    for k, v in combined_flags.items():
        setattr(group, k, v)

    # Clean & merge
    cleaned_parts = [clean_text(t) for t in group.raw_texts]
    merged_clean = " ".join(p for p in cleaned_parts if p).strip()
    group.clean_text = merged_clean
    group.context_mode_used = context_mode

    if not is_substantive(merged_clean):
        group.used_zeros = True
        group.context_input = ""
        return group

    # Build context input according to mode
    group.context_input = build_context_input(
        group.ellie_question_text,
        merged_clean,
        context_mode=context_mode,
        separator=separator,
    )

    # Speech rate
    n_words = len(merged_clean.split())
    dur = max(group.duration, 0.1)
    group.speech_rate_wps = round(n_words / dur, 3)

    return group


# ──────────────────────────────────────────────────────────────
# BERT embedding helpers
# ──────────────────────────────────────────────────────────────
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_len: int,
    batch_size: int,
    pool_mode: str,
) -> np.ndarray:
    all_embeddings = []
    model.eval()
    output_hidden = (pool_mode == "mean_top4")

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)

            out = model(**enc, output_hidden_states=output_hidden)

            if pool_mode == "cls":
                emb = out.last_hidden_state[:, 0, :]
            elif pool_mode == "mean_top4":
                hidden_states = out.hidden_states
                top4 = torch.stack(hidden_states[-4:], dim=0).mean(0)
                emb = mean_pool(top4, enc["attention_mask"])
            else:
                emb = mean_pool(out.last_hidden_state, enc["attention_mask"])

            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Per-participant processing
# ──────────────────────────────────────────────────────────────
def extract_features(
    transcript_path: str,
    tokenizer,
    model,
    device: torch.device,
    max_len: int,
    batch_size: int,
    pool_mode: str,
    merge_gap_sec: float,
    context_mode: str,
    separator: str,
) -> tuple[np.ndarray, list[dict]]:
    df     = load_transcript(transcript_path)
    groups = build_utterance_groups(df, merge_gap_sec=merge_gap_sec)

    if not groups:
        log.warning(f"No participant turns in {transcript_path}")
        return np.zeros((1, EMBED_DIM), dtype=np.float32), []

    groups = [enrich_group(g, context_mode=context_mode, separator=separator)
              for g in groups]

    embed_groups = [g for g in groups if not g.used_zeros]
    embed_texts_list = [g.context_input for g in embed_groups]

    if embed_texts_list:
        embeddings_valid = embed_texts(
            embed_texts_list, tokenizer, model, device,
            max_len=max_len, batch_size=batch_size, pool_mode=pool_mode,
        )
    else:
        embeddings_valid = np.zeros((0, EMBED_DIM), dtype=np.float32)

    embed_iter = iter(embeddings_valid)
    all_embeddings = []
    for g in groups:
        if g.used_zeros:
            all_embeddings.append(np.zeros(EMBED_DIM, dtype=np.float32))
        else:
            all_embeddings.append(next(embed_iter))

    embeddings_arr = np.vstack(all_embeddings).astype(np.float32)
    metadata = [g.to_dict() for g in groups]
    return embeddings_arr, metadata


def process_participant(
    pid: int,
    data_root: str,
    tokenizer,
    model,
    device: torch.device,
    max_len: int,
    batch_size: int,
    pool_mode: str,
    merge_gap_sec: float,
    context_mode: str,
    overwrite: bool,
    separator: str,
) -> bool:
    transcript_path = os.path.join(data_root, f"{pid}_TRANSCRIPT.csv")

    # SHORTCUT ABLATION: embeddings saved with context_mode suffix
    # → allows loading different ablation versions without re-extracting
    out_npy     = os.path.join(data_root, f"{pid}_text_feats_ctx_{context_mode}.npy")
    out_meta    = os.path.join(data_root, f"{pid}_text_feats_meta.json")   # shared
    out_ngroups = os.path.join(data_root, f"{pid}_n_groups.txt")           # shared

    if not os.path.exists(transcript_path):
        log.warning(f"[{pid}] Transcript not found: {transcript_path}")
        return False

    # Skip if already done (unless overwrite)
    if not overwrite and os.path.exists(out_npy):
        log.info(f"[{pid}] Already exists (ctx={context_mode}) — skip (use --overwrite to redo)")
        return True

    log.info(f"[{pid}] Processing (context_mode={context_mode}) …")

    embeddings, meta = extract_features(
        transcript_path=transcript_path,
        tokenizer=tokenizer, model=model, device=device,
        max_len=max_len, batch_size=batch_size,
        pool_mode=pool_mode, merge_gap_sec=merge_gap_sec,
        context_mode=context_mode, separator=separator,
    )

    n_groups = len(meta)
    n_zeros  = sum(1 for m in meta if m.get("used_zeros"))
    n_embed  = n_groups - n_zeros

    np.save(out_npy, embeddings)

    # Meta JSON is shared across modes (paralinguistic flags are mode-independent)
    # Only write if it doesn't exist yet (first mode wins; context_mode_used is
    # stored per utterance inside the JSON for traceability)
    if not os.path.exists(out_meta) or overwrite:
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # FIX: Write n_groups so dataset.py knows the correct node count
    if not os.path.exists(out_ngroups) or overwrite:
        with open(out_ngroups, "w") as f:
            f.write(str(n_groups))

    log.info(
        f"[{pid}] Done (ctx={context_mode}) — "
        f"groups={n_groups} (embed={n_embed}, zeros={n_zeros}) "
        f"| shape={embeddings.shape} → {out_npy}"
    )
    return True


# ──────────────────────────────────────────────────────────────
# Shortcut diagnosis helper
# ──────────────────────────────────────────────────────────────
def print_shortcut_reminder() -> None:
    """Print a reminder about what to do with the 3 modes."""
    print(
        "\n"
        "=" * 72 + "\n"
        "  SHORTCUT LEARNING ABLATION GUIDE\n"
        "=" * 72 + "\n"
        "  Run this script 3 times with --context_mode:\n"
        "    1) none      → {pid}_text_feats_ctx_none.npy\n"
        "    2) truncated → {pid}_text_feats_ctx_truncated.npy  (default)\n"
        "    3) full      → {pid}_text_feats_ctx_full.npy\n"
        "\n"
        "  In daicwoz_dataset.py, set TEXT_CONTEXT_MODE to load the right one.\n"
        "\n"
        "  Expected outcome if model is NOT shortcutting:\n"
        "    F1(none) ≈ F1(truncated) > F1_baseline\n"
        "\n"
        "  Red flag (shortcut detected):\n"
        "    F1(none) << F1(truncated)  →  model learns interviewer phrasing\n"
        "=" * 72 + "\n"
    )


# ──────────────────────────────────────────────────────────────
# HuggingFace token helper
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract BERT embeddings từ DAIC-WOZ transcripts (v3-fixed, anti-shortcut)"
    )
    parser.add_argument("--data_root",    default="daicwoz/")
    parser.add_argument("--split_csv",    required=True)
    parser.add_argument("--model_name",   default="mental/mental-bert-base-uncased")
    parser.add_argument("--hf_token",     default=None)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--max_len",      type=int, default=256)
    parser.add_argument("--pool_mode",    default="mean_top4",
                        choices=["mean_last", "cls", "mean_top4"])
    parser.add_argument("--merge_gap",    type=float, default=DEFAULT_MERGE_GAP_SEC)

    # ── SHORTCUT ABLATION FLAG ────────────────────────────────────────────────
    parser.add_argument(
        "--context_mode",
        default="truncated",
        choices=list(CONTEXT_MODES),
        help=(
            "How to handle Ellie (interviewer) context for BERT input.\n"
            "  none      : participant text only — RECOMMENDED for anti-shortcut\n"
            "  truncated : last 20 words of Ellie + participant (default)\n"
            "  full      : entire Ellie turn + participant (max shortcut risk)\n"
            "Run all 3 and compare F1 in ablation table."
        ),
    )
    # ── END SHORTCUT ABLATION FLAG ───────────────────────────────────────────

    parser.add_argument("--overwrite",    action="store_true")
    parser.add_argument(
        "--all_modes",
        action="store_true",
        help="Extract all 3 context modes in one run (for ablation convenience).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Decide which modes to run
    modes_to_run = list(CONTEXT_MODES) if args.all_modes else [args.context_mode]

    log.info(f"Device      : {device}")
    log.info(f"Model       : {args.model_name}")
    log.info(f"Pool mode   : {args.pool_mode}")
    log.info(f"Merge gap   : {args.merge_gap}s")
    log.info(f"Context mode: {modes_to_run}")
    log.info(f"Min words   : {MIN_WORDS_FOR_EMBED}")
    log.info(f"Max ellie   : {MAX_ELLIE_WORDS} words (truncated mode)")

    hf_token  = get_hf_token(args.hf_token)
    hf_kwargs = {"token": hf_token} if hf_token else {}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **hf_kwargs)
    model     = AutoModel.from_pretrained(args.model_name, **hf_kwargs).to(device)

    # Detect separator: use [SEP] only if tokenizer knows it
    if tokenizer.sep_token:
        separator = f" {tokenizer.sep_token} "
        log.info(f"Separator   : {tokenizer.sep_token}")
    else:
        separator = " | "
        log.info("Separator   : ' | ' (model has no [SEP] token)")

    split_df = pd.read_csv(args.split_csv)
    split_df.columns = [c.strip() for c in split_df.columns]
    pids = split_df["Participant_ID"].astype(int).tolist()
    log.info(f"Participants: {len(pids)}")

    for mode in modes_to_run:
        log.info(f"\n{'='*60}")
        log.info(f"  Extracting context_mode = {mode!r}")
        log.info(f"{'='*60}")
        success = 0
        for pid in pids:
            ok = process_participant(
                pid=pid, data_root=args.data_root,
                tokenizer=tokenizer, model=model, device=device,
                max_len=args.max_len, batch_size=args.batch_size,
                pool_mode=args.pool_mode, merge_gap_sec=args.merge_gap,
                context_mode=mode, overwrite=args.overwrite,
                separator=separator,
            )
            if ok:
                success += 1
        log.info(f"Mode={mode}: {success}/{len(pids)} participants done.")

    print_shortcut_reminder()


if __name__ == "__main__":
    main()