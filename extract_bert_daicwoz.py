"""
extract_bert_v2.py — Cải tiến pipeline extract BERT embeddings từ DAIC-WOZ transcripts.

Cải tiến so với v1:
  1. Pre-cleaning: strip disfluency tags, normalize whitespace
  2. Context window: ghép Ellie turn trước vào input để giải quyết ambiguity
  3. Lọc turns quá ngắn (< MIN_TOKENS token sau khi clean)
  4. Logging chi tiết + thống kê per-participant
  5. Lưu metadata JSON kèm theo file .npy

Output:
  {ID}_text_feats.npy       — shape (N_valid_turns, 768), float32
  {ID}_text_feats_meta.json — thông tin turn: index gốc, text đã clean, start/stop time

Usage:
  python extract_bert_v2.py --split_csv data/daicwoz/train_split_Depression_AVEC2017.csv
  python extract_bert_v2.py --split_csv data/daicwoz/dev_split_Depression_AVEC2017.csv

Requires: pip install transformers torch pandas numpy
"""

import os
import re
import json
import argparse
import logging
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
# Disfluency / noise tags có trong DAIC-WOZ
# ──────────────────────────────────────────────────────────────
# Tags này được transcriber gán — không phải lời nói thực, cần loại bỏ
# trước khi đưa vào BERT.
DISFLUENCY_TAGS = {
    # Âm thanh phi ngôn ngữ
    "<laughter>", "<laguhter>",   # typo trong data gốc
    "<sigh>", "<cough>", "<sniffle>", "<clears throat>",
    # Annotation tags của DAIC-WOZ (single-letter)
    "<p>", "<s>", "<l>", "<f>", "<o>", "<t>", "<be>", "<bu>",
    "<ha>", "<li>", "<si>", "<ss>", "<ta>", "<wa>",
    # Annotation tags đa ký tự
    "<ano>", "<con>", "<enj>", "<insen>", "<inv>", "<motiv>",
    "<ok>", "<pe>", "<rea>", "<see>", "<so>", "<strug>", "<think>",
    "<to>", "<wou>", "<disisinterested>",
}

# Regex match bất kỳ <tag> nào (kể cả tags chưa biết)
_TAG_RE = re.compile(r"<[^>]+>")

# Filler words rất ngắn — embed riêng lẻ không mang thông tin
_FILLER_ONLY_RE = re.compile(
    r"^(yeah|yes|no|okay|ok|um|uh|mm|hmm|mhm|hm|oh|ah|eh|right|sure"
    r"|alright|yep|nope|nah|so|like|and|but|whatever|i|"
    r"goodbye|bye|music|quiet|patience|spoiled|sometimes)[\s\.]*$",
    re.IGNORECASE,
)


def clean_turn(text: str) -> str:
    """
    Làm sạch một participant turn:
      1. Strip tất cả <disfluency tags>
      2. Normalize whitespace
    """
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_valid_turn(text: str, min_tokens: int = 3) -> bool:
    """
    Kiểm tra turn có đủ nội dung để embed không.
    Loại bỏ:
      - Turns rỗng sau khi clean
      - Turns chỉ có filler words (yeah, uh, mm, ...)
      - Turns quá ngắn (< min_tokens words)
    """
    if not text:
        return False
    if _FILLER_ONLY_RE.match(text):
        return False
    words = text.split()
    if len(words) < min_tokens:
        return False
    return True


def build_context_input(ellie_text: str, participant_text: str) -> str:
    """
    Ghép Ellie question + Participant answer theo format BERT NSP:
      [CLS] ellie_question [SEP] participant_answer [SEP]

    Lợi ích: BERT hiểu "yes" trong context "Are you doing okay? → yes"
    thay vì embed "yes" độc lập không có nghĩa.

    Nếu không có Ellie turn trước (đầu session), chỉ dùng participant text.
    """
    if ellie_text:
        return f"{ellie_text.strip()} [SEP] {participant_text}"
    return participant_text


def get_hf_token(cli_token=None, env_file=".env"):
    """Resolve HF token: CLI > env var > .env file."""
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


def mean_pool(token_embeddings, attention_mask):
    """Mean pooling trên các non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def load_transcript(path: str) -> pd.DataFrame:
    """Load DAIC-WOZ transcript CSV (tab-separated), strip column names."""
    try:
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    return df


def extract_features(
    transcript_path: str,
    tokenizer,
    model,
    device,
    max_len: int = 128,
    batch_size: int = 32,
    min_tokens: int = 2,
    use_context: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """
    Extract BERT embeddings từ một transcript file.

    Returns:
      embeddings : np.ndarray shape (N_valid, 768)
      metadata   : list of dict, mỗi dict chứa thông tin của turn tương ứng
                   {original_row_idx, start_time, stop_time, raw_text,
                    clean_text, context_input, ellie_context}
    """
    df = load_transcript(transcript_path)

    if "value" not in df.columns:
        df["value"] = ""
    df["value"] = df["value"].fillna("").astype(str)
    df["speaker_clean"] = df["speaker"].str.strip().str.lower()

    valid_inputs = []   # (context_text, metadata_dict)
    last_ellie = ""     # Ellie turn gần nhất

    for idx, row in df.iterrows():
        speaker = row["speaker_clean"]

        if speaker == "ellie":
            last_ellie = clean_turn(row["value"])
            continue

        if speaker != "participant":
            continue

        raw_text   = row["value"]
        clean_text = clean_turn(raw_text)

        if not is_valid_turn(clean_text, min_tokens=min_tokens):
            continue

        # Xây dựng context input
        context_input = (
            build_context_input(last_ellie, clean_text)
            if use_context
            else clean_text
        )

        meta = {
            "original_row_idx": int(idx),
            "start_time"      : float(row.get("start_time", -1)),
            "stop_time"       : float(row.get("stop_time", -1)),
            "raw_text"        : raw_text,
            "clean_text"      : clean_text,
            "ellie_context"   : last_ellie,
            "context_input"   : context_input,
        }
        valid_inputs.append((context_input, meta))

    if not valid_inputs:
        log.warning(f"No valid participant turns in {transcript_path}")
        return np.zeros((1, 768), dtype=np.float32), []

    texts    = [x[0] for x in valid_inputs]
    metadata = [x[1] for x in valid_inputs]

    all_embeddings = []
    model.eval()

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
            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    return embeddings, metadata


def process_participant(
    pid: int,
    data_root: str,
    tokenizer,
    model,
    device,
    max_len: int,
    batch_size: int,
    min_tokens: int,
    use_context: bool,
    overwrite: bool,
) -> bool:
    """Xử lý một participant. Trả về True nếu thành công."""
    transcript_path = os.path.join(data_root, f"{pid}_TRANSCRIPT.csv")
    out_npy         = os.path.join(data_root, f"{pid}_text_feats.npy")
    out_meta        = os.path.join(data_root, f"{pid}_text_feats_meta.json")

    if not os.path.exists(transcript_path):
        log.warning(f"[{pid}] Transcript not found: {transcript_path}")
        return False

    if not overwrite and os.path.exists(out_npy):
        log.info(f"[{pid}] Already extracted — skip (use --overwrite to force)")
        return True

    log.info(f"[{pid}] Processing ...")
    df = load_transcript(transcript_path)
    df["speaker_clean"] = df["speaker"].str.strip().str.lower()

    total_turns       = (df["speaker_clean"] == "participant").sum()
    embeddings, meta  = extract_features(
        transcript_path, tokenizer, model, device,
        max_len=max_len, batch_size=batch_size,
        min_tokens=min_tokens, use_context=use_context,
    )
    valid_turns = len(meta)
    dropped     = total_turns - valid_turns

    np.save(out_npy, embeddings)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info(
        f"[{pid}] Done — total_participant_turns={total_turns}, "
        f"valid={valid_turns}, dropped={dropped} "
        f"| shape={embeddings.shape} → {out_npy}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract BERT embeddings from DAIC-WOZ transcripts (v2)"
    )
    parser.add_argument("--data_root",   default="data/daicwoz/")
    parser.add_argument("--split_csv",   required=True,
                        help="Train/dev CSV có cột Participant_ID")
    parser.add_argument("--model_name",  default="mental/mental-bert-base-uncased",
                        help="HuggingFace model ID")
    parser.add_argument("--hf_token",    default=None)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--max_len",     type=int, default=192,
                        help="Max tokens cho tokenizer (default 192). "
                             "Tăng lên 192-256 nếu dùng context window")
    parser.add_argument("--min_tokens",  type=int, default=2,
                        help="Số từ tối thiểu sau khi clean để giữ turn (default 2)")
    parser.add_argument("--no_context",  action="store_true",
                        help="Tắt context window (không ghép Ellie turn)")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Ghi đè file đã có")
    args = parser.parse_args()

    use_context = not args.no_context
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Device      : {device}")
    log.info(f"Model       : {args.model_name}")
    log.info(f"Max len     : {args.max_len}")
    log.info(f"Min tokens  : {args.min_tokens}")
    log.info(f"Context win : {use_context}")

    hf_token = get_hf_token(args.hf_token)
    if hf_token:
        log.info("HuggingFace: authenticated access")
    else:
        log.info("HuggingFace: anonymous access")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    model     = AutoModel.from_pretrained(args.model_name, token=hf_token).to(device)

    split_df = pd.read_csv(args.split_csv)
    split_df.columns = [c.strip() for c in split_df.columns]
    pids     = split_df["Participant_ID"].astype(int).tolist()

    log.info(f"Participants: {len(pids)}")

    success = 0
    for pid in pids:
        ok = process_participant(
            pid          = pid,
            data_root    = args.data_root,
            tokenizer    = tokenizer,
            model        = model,
            device       = device,
            max_len      = args.max_len,
            batch_size   = args.batch_size,
            min_tokens   = args.min_tokens,
            use_context  = use_context,
            overwrite    = args.overwrite,
        )
        if ok:
            success += 1

    log.info(f"\nDone: {success}/{len(pids)} participants extracted successfully.")


if __name__ == "__main__":
    main()