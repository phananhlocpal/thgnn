"""
extract_bert_v3.py — Extract BERT embeddings từ DAIC-WOZ, phiên bản cải tiến toàn diện.

Cải tiến so với v2:
  1. [FIX-CRITICAL] Đồng bộ N_turns với audio pipeline:
       - Luôn giữ ALL participant utterance groups (không drop),
         turns ngắn/invalid → zeros vector thay vì bị loại bỏ.
       - Metadata ghi rõ flag `used_zeros` để audio script biết align đúng.
  2. [FIX-MAJOR] Merge consecutive participant turns thành 1 utterance:
       - Gom các dòng participant liên tiếp (gap ≤ MERGE_GAP_SEC) vào cùng
         một utterance group trước khi embed.
       - Giảm số "utterances" từ ví dụ 295→43 group thực chất.
  3. [NEW] Ellie question type làm node feature:
       - Parse question_id từ format "question_id (text)" của Ellie.
       - Lưu `ellie_question_id` và `ellie_question_text` vào metadata.
  4. [NEW] Response latency feature:
       - Tính khoảng cách `ellie_stop_time → participant_start_time`.
       - Lưu `response_latency_sec` vào metadata.
  5. [NEW] Paralinguistic event flags:
       - Thay vì xóa <sigh>, <laughter>, ... hoàn toàn, extract thành binary
         flags: `has_sigh`, `has_laughter`, `has_breath`, `has_cry`,
         `has_cough`, `has_sound` (bất kỳ tag nào còn lại).
       - Lưu vào metadata, có thể dùng làm extra node features trong graph.
  6. [NEW] Speech rate proxy:
       - Tính n_words / duration để ước lượng tốc độ nói.
  7. [IMPROVE] Multi-layer BERT embedding:
       - Hỗ trợ ba chế độ pool_mode: mean_last (default), cls, mean_top4.
       - mean_top4 = weighted average của 4 layer cuối (tốt hơn cho
         sentiment/clinical tasks theo nhiều nghiên cứu).

Output cho mỗi participant (lưu vào --data_root):
  {ID}_text_feats.npy       — shape (N_groups, 768), float32
  {ID}_text_feats_meta.json — metadata đầy đủ mỗi utterance group

JSON metadata schema (mỗi phần tử):
  {
    "group_id"             : int,   # 0-indexed, dùng để align với audio
    "row_indices"          : [int], # index gốc trong transcript CSV
    "start_time"           : float, # start_time của turn đầu tiên trong group
    "stop_time"            : float, # stop_time của turn cuối cùng trong group
    "duration"             : float, # stop - start (giây)
    "response_latency_sec" : float, # ellie_stop → participant_start (-1 nếu đầu session)
    "ellie_question_id"    : str,   # VD: "easy_sleep", "depression_diagnosed"
    "ellie_question_text"  : str,   # text đầy đủ câu hỏi của Ellie
    "n_raw_turns"          : int,   # số dòng participant được merge
    "raw_texts"            : [str], # danh sách text gốc từng dòng
    "clean_text"           : str,   # text đã clean & merge để embed
    "context_input"        : str,   # input thực sự gửi vào BERT
    "used_zeros"           : bool,  # True nếu text quá ngắn → dùng zeros vector
    "speech_rate_wps"      : float, # words-per-second (proxy)
    "has_sigh"             : bool,
    "has_laughter"         : bool,
    "has_breath"           : bool,
    "has_cry"              : bool,
    "has_cough"            : bool,
    "has_other_sound"      : bool   # bất kỳ <tag> nào không thuộc các loại trên
  }

Usage:
  python extract_bert_v3.py \\
      --split_csv daicwoz/train_split_Depression_AVEC2017.csv

  python extract_bert_v3.py \\
      --split_csv daicwoz/dev_split_Depression_AVEC2017.csv \\
      --model_name mental/mental-bert-base-uncased \\
      --pool_mode mean_top4 \\
      --merge_gap 2.0 \\
      --data_root daicwoz/ \\
      --overwrite

Requires: pip install transformers torch pandas numpy
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

# Khoảng cách tối đa (giây) giữa 2 dòng participant liên tiếp
# để coi là cùng một utterance group
DEFAULT_MERGE_GAP_SEC = 2.0

# Minimum words sau khi clean để không dùng zeros
MIN_WORDS_FOR_EMBED = 2

# Regex: bất kỳ <tag>
_TAG_RE = re.compile(r"<[^>]+>")

# Regex parse format Ellie: "question_id (text of question)"
_ELLIE_RE = re.compile(r"^(\S+)\s+\((.+)\)\s*$", re.DOTALL)

# Tag classification cho paralinguistic flags
_SIGH_TAGS    = {"<sigh>", "<deep breath>", "<breath>"}
_LAUGH_TAGS   = {"<laughter>", "<laguhter>", "<laugher>", "<laugh>"}
_CRY_TAGS     = {"<cry>", "<crying>", "<sob>", "<sobbing>"}
_COUGH_TAGS   = {"<cough>", "<coughs>", "<clears throat>", "<sniff>",
                 "<sniffle>", "<tisk>", "<tisk tisk>"}
_BREATH_TAGS  = {"<deep breath>", "<breath>"}  # subset of sigh for specificity


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────
@dataclass
class UtteranceGroup:
    """Một utterance group = một hoặc nhiều dòng participant liên tiếp."""
    group_id: int
    row_indices: list = field(default_factory=list)
    start_time: float = 0.0
    stop_time: float = 0.0
    raw_texts: list = field(default_factory=list)
    # Điền sau khi biết Ellie context
    response_latency_sec: float = -1.0
    ellie_question_id: str = ""
    ellie_question_text: str = ""
    # Điền sau khi clean
    clean_text: str = ""
    context_input: str = ""
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
    """Load DAIC-WOZ transcript CSV (tab-separated)."""
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
    """
    Parse Ellie turn value: "question_id (text)" → (question_id, text).
    Trả về ("", value) nếu không match format.
    """
    m = _ELLIE_RE.match(value.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Fallback: toàn bộ value là text, không có ID
    return "", value.strip()


# ──────────────────────────────────────────────────────────────
# Text cleaning & paralinguistic extraction
# ──────────────────────────────────────────────────────────────
def extract_paralinguistic_flags(text: str) -> dict:
    """
    Extract binary flags cho các sự kiện phi ngôn ngữ từ raw text.
    Trả về dict với các key has_sigh, has_laughter, has_breath, has_cry,
    has_cough, has_other_sound.
    """
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
    """Strip tất cả <tags>, normalize whitespace."""
    cleaned = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_substantive(text: str, min_words: int = MIN_WORDS_FOR_EMBED) -> bool:
    """True nếu text có đủ từ để embed có nghĩa."""
    return len(text.split()) >= min_words


def build_context_input(ellie_text: str, participant_text: str) -> str:
    """Ghép Ellie question + Participant answer theo format BERT NSP."""
    if ellie_text:
        return f"{ellie_text.strip()} [SEP] {participant_text}"
    return participant_text


# ──────────────────────────────────────────────────────────────
# Utterance group building
# ──────────────────────────────────────────────────────────────
def build_utterance_groups(
    df: pd.DataFrame,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
) -> list[UtteranceGroup]:
    """
    Duyệt transcript, gom các participant turns liên tiếp thành groups.

    Hai dòng participant coi là cùng group nếu:
      - Không có Ellie turn ở giữa
      - Khoảng cách thời gian ≤ merge_gap_sec

    Đồng thời gắn thông tin Ellie context và response_latency cho mỗi group.
    """
    groups: list[UtteranceGroup] = []
    current_group: Optional[UtteranceGroup] = None
    last_ellie_stop: float = -1.0
    last_ellie_qid: str = ""
    last_ellie_qtext: str = ""

    group_counter = 0

    for idx, row in df.iterrows():
        speaker = row["speaker_clean"]

        if speaker == "ellie":
            # Khi gặp Ellie: flush current_group nếu đang mở
            if current_group is not None:
                groups.append(current_group)
                current_group = None
            # Cập nhật Ellie context cho group tiếp theo
            last_ellie_stop = float(row["stop_time"])
            last_ellie_qid, last_ellie_qtext = parse_ellie_turn(row["value"])
            continue

        if speaker != "participant":
            continue

        t_start = float(row["start_time"])
        t_stop  = float(row["stop_time"])

        if current_group is None:
            # Bắt đầu group mới
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
            # Kiểm tra có nên merge với group hiện tại không
            gap = t_start - current_group.stop_time
            if gap <= merge_gap_sec:
                # Merge vào group hiện tại
                current_group.row_indices.append(int(idx))
                current_group.raw_texts.append(row["value"])
                current_group.stop_time = max(current_group.stop_time, t_stop)
            else:
                # Gap quá lớn → flush và bắt đầu group mới
                groups.append(current_group)
                current_group = UtteranceGroup(
                    group_id=group_counter,
                    row_indices=[int(idx)],
                    start_time=t_start,
                    stop_time=t_stop,
                    raw_texts=[row["value"]],
                    # Latency kể từ group trước (không phải Ellie)
                    response_latency_sec=-1.0,
                    ellie_question_id=last_ellie_qid,
                    ellie_question_text=last_ellie_qtext,
                )
                group_counter += 1

    # Flush group cuối
    if current_group is not None:
        groups.append(current_group)

    return groups


def enrich_group(group: UtteranceGroup, use_context: bool) -> UtteranceGroup:
    """
    Điền các trường derived: clean_text, paralinguistic flags,
    context_input, speech_rate, used_zeros.
    """
    # Paralinguistic flags — union tất cả turns trong group
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

    # Clean & merge text
    cleaned_parts = [clean_text(t) for t in group.raw_texts]
    merged_clean = " ".join(p for p in cleaned_parts if p).strip()
    group.clean_text = merged_clean

    # used_zeros nếu quá ngắn
    if not is_substantive(merged_clean):
        group.used_zeros = True
        group.context_input = ""
        return group

    # Context input
    group.context_input = (
        build_context_input(group.ellie_question_text, merged_clean)
        if use_context else merged_clean
    )

    # Speech rate (words per second)
    n_words = len(merged_clean.split())
    dur = max(group.duration, 0.1)
    group.speech_rate_wps = round(n_words / dur, 3)

    return group


# ──────────────────────────────────────────────────────────────
# BERT embedding helpers
# ──────────────────────────────────────────────────────────────
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling trên non-padding tokens."""
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
    """
    Embed danh sách text strings → numpy array (N, 768).

    pool_mode:
      "mean_last" — mean pool của last_hidden_state (default)
      "cls"       — [CLS] token của last_hidden_state
      "mean_top4" — mean pool của trung bình 4 hidden layers cuối
    """
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
                emb = out.last_hidden_state[:, 0, :]  # (B, 768)
            elif pool_mode == "mean_top4":
                # Stack 4 layer cuối, average, rồi mean pool
                hidden_states = out.hidden_states  # tuple of (B, T, 768)
                top4 = torch.stack(hidden_states[-4:], dim=0).mean(0)  # (B, T, 768)
                emb = mean_pool(top4, enc["attention_mask"])
            else:  # mean_last
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
    use_context: bool,
) -> tuple[np.ndarray, list[dict]]:
    """
    Extract BERT embeddings cho tất cả utterance groups của 1 participant.

    Returns:
      embeddings : np.ndarray (N_groups, 768) — N_groups = số utterance groups
                   Groups với used_zeros=True → zero vector (KHÔNG bị drop).
      metadata   : list[dict], mỗi phần tử là to_dict() của UtteranceGroup
    """
    df     = load_transcript(transcript_path)
    groups = build_utterance_groups(df, merge_gap_sec=merge_gap_sec)

    if not groups:
        log.warning(f"No participant turns in {transcript_path}")
        return np.zeros((1, EMBED_DIM), dtype=np.float32), []

    # Enrich tất cả groups (flags, clean text, context_input)
    groups = [enrich_group(g, use_context=use_context) for g in groups]

    # Tách groups cần embed vs. groups dùng zeros
    embed_groups = [g for g in groups if not g.used_zeros]
    embed_texts_list = [g.context_input for g in embed_groups]

    # Embed batch (chỉ những group có nội dung)
    if embed_texts_list:
        embeddings_valid = embed_texts(
            embed_texts_list, tokenizer, model, device,
            max_len=max_len, batch_size=batch_size, pool_mode=pool_mode,
        )
    else:
        embeddings_valid = np.zeros((0, EMBED_DIM), dtype=np.float32)

    # Ghép lại đúng thứ tự, zeros cho groups không embed
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

    # if not overwrite and os.path.exists(out_npy):
    #     log.info(f"[{pid}] Already exists — skip (use --overwrite to force)")
    #     return True

    log.info(f"[{pid}] Processing …")

    embeddings, meta = extract_features(
        transcript_path=transcript_path,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
        pool_mode=pool_mode,
        merge_gap_sec=merge_gap_sec,
        use_context=use_context,
    )

    n_groups    = len(meta)
    n_zeros     = sum(1 for m in meta if m.get("used_zeros"))
    n_embed     = n_groups - n_zeros
    n_with_sigh = sum(1 for m in meta if m.get("has_sigh"))
    n_with_laug = sum(1 for m in meta if m.get("has_laughter"))

    np.save(out_npy, embeddings)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info(
        f"[{pid}] Done — groups={n_groups} (embed={n_embed}, zeros={n_zeros}) "
        f"| sigh={n_with_sigh}, laughter={n_with_laug} "
        f"| shape={embeddings.shape} → {out_npy}"
    )
    return True


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
        description="Extract BERT embeddings từ DAIC-WOZ transcripts (v3)"
    )
    parser.add_argument("--data_root",    default="daicwoz/")
    parser.add_argument("--split_csv",    required=True,
                        help="Train/dev CSV có cột Participant_ID")
    parser.add_argument("--model_name",   default="mental/mental-bert-base-uncased",
                        help="HuggingFace model ID")
    parser.add_argument("--hf_token",     default=None)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--max_len",      type=int, default=256)
    parser.add_argument("--pool_mode",    default="mean_top4",
                        choices=["mean_last", "cls", "mean_top4"],
                        help="Cách pool BERT output: mean_last | cls | mean_top4")
    parser.add_argument("--merge_gap",    type=float, default=DEFAULT_MERGE_GAP_SEC,
                        help=f"Khoảng cách tối đa (giây) để merge consecutive turns "
                             f"(default: {DEFAULT_MERGE_GAP_SEC})")
    parser.add_argument("--no_context",   action="store_true",
                        help="Tắt context window (không ghép Ellie turn)")
    parser.add_argument("--overwrite",    action="store_true")
    args = parser.parse_args()

    use_context = not args.no_context
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Device      : {device}")
    log.info(f"Model       : {args.model_name}")
    log.info(f"Pool mode   : {args.pool_mode}")
    log.info(f"Max len     : {args.max_len}")
    log.info(f"Merge gap   : {args.merge_gap}s")
    log.info(f"Context win : {use_context}")

    hf_token  = get_hf_token(args.hf_token)
    hf_kwargs = {"token": hf_token} if hf_token else {}
    if hf_token:
        log.info("HuggingFace: authenticated")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **hf_kwargs)
    model     = AutoModel.from_pretrained(args.model_name, **hf_kwargs).to(device)

    split_df = pd.read_csv(args.split_csv)
    split_df.columns = [c.strip() for c in split_df.columns]
    pids = split_df["Participant_ID"].astype(int).tolist()
    log.info(f"Participants: {len(pids)}")

    success = 0
    for pid in pids:
        ok = process_participant(
            pid=pid, data_root=args.data_root,
            tokenizer=tokenizer, model=model, device=device,
            max_len=args.max_len, batch_size=args.batch_size,
            pool_mode=args.pool_mode, merge_gap_sec=args.merge_gap,
            use_context=use_context, overwrite=args.overwrite,
        )
        if ok:
            success += 1

    log.info(f"\nDone: {success}/{len(pids)} participants.")


if __name__ == "__main__":
    main()