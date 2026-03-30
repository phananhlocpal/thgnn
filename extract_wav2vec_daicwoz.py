"""
extract_wav2vec_v2.py — Extract wav2vec2 audio embeddings từ DAIC-WOZ, phiên bản cải tiến.

Cải tiến so với v1:
  1. [FIX-CRITICAL] Đồng bộ N_groups với text pipeline (extract_bert_v3.py):
       - Đọc {ID}_text_feats_meta.json để biết chính xác từng utterance group
         (start_time, stop_time) đã được text pipeline tạo ra.
       - Embed audio theo ĐÚNG group_id đó thay vì tự parse transcript.
       - Đảm bảo hàng N của text_feats.npy ↔ hàng N của audio_feats.npy.
  2. [FIX-MAJOR] Merge consecutive turns thành 1 segment âm thanh:
       - Cắt audio từ group.start_time → group.stop_time (toàn bộ group).
       - Capture cả khoảng silence ngắn bên trong câu trả lời.
  3. [NEW] Silence / pause features:
       - Tính toán từ bên trong audio segment của mỗi group:
         * mean_pause_sec    — trung bình độ dài khoảng lặng
         * max_pause_sec     — khoảng lặng dài nhất
         * n_pauses          — số khoảng lặng (VAD-based, energy threshold)
         * speech_ratio      — tỉ lệ thời gian có giọng nói
       - Các feature này bổ sung quan trọng vào metadata (không ghép vào vector).
  4. [NEW] Prosody features (pitch + energy):
       - Tính per-group: mean_pitch_hz, std_pitch_hz, mean_energy, std_energy.
       - Lưu vào metadata; có thể concat với wav2vec vector cho graph node.
       - Nếu librosa không cài → bỏ qua prosody, log warning.
  5. [IMPROVE] Fallback mode khi không có text meta:
       - Nếu chưa chạy text pipeline, tự parse transcript theo cùng logic
         merge_gap như extract_bert_v3 (MERGE_GAP_SEC default giống nhau).
       - Log rõ ràng để người dùng biết đang dùng fallback.

Output cho mỗi participant (lưu vào --data_root):
  {ID}_audio_feats.npy       — shape (N_groups, 768), float32
                               N_groups khớp với {ID}_text_feats.npy
  {ID}_audio_feats_meta.json — metadata đầy đủ

JSON metadata schema (mỗi phần tử):
  {
    "group_id"         : int,
    "start_time"       : float,
    "stop_time"        : float,
    "duration"         : float,
    "used_zeros"       : bool,   # True nếu segment quá ngắn
    "mean_pause_sec"   : float,
    "max_pause_sec"    : float,
    "n_pauses"         : int,
    "speech_ratio"     : float,  # fraction of segment that is speech
    "mean_pitch_hz"    : float,  # -1 nếu librosa không có
    "std_pitch_hz"     : float,
    "mean_energy"      : float,
    "std_energy"       : float
  }

Usage:
  # Chạy sau extract_bert_v3.py (recommended — dùng text meta để align)
  python extract_wav2vec_v2.py \\
      --split_csv daicwoz/train_split_Depression_AVEC2017.csv

  # Standalone (fallback tự parse transcript)
  python extract_wav2vec_v2.py \\
      --split_csv daicwoz/dev_split_Depression_AVEC2017.csv \\
      --no_text_meta \\
      --merge_gap 2.0 \\
      --overwrite

Requires:
  pip install transformers torch torchaudio pandas numpy
  pip install librosa  # optional, cho prosody features
"""

import argparse
import json
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

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
TARGET_SR = 16_000
EMBED_DIM  = 768

# Minimum duration (giây) để thực sự embed (tránh segment quá ngắn)
MIN_DURATION = 0.1

# Khoảng cách tối đa (giây) để merge consecutive turns
# Phải khớp với giá trị dùng trong extract_bert_v3.py
DEFAULT_MERGE_GAP_SEC = 2.0

# VAD: Energy threshold để phân biệt speech vs silence
# (tính trên normalized waveform, 0-1 scale)
VAD_ENERGY_THRESHOLD = 0.01
VAD_FRAME_SEC        = 0.02   # 20ms frame

# Regex (dùng cho fallback transcript parsing)
_TAG_RE = re.compile(r"<[^>]+>")


# ──────────────────────────────────────────────────────────────
# Audio I/O
# ──────────────────────────────────────────────────────────────
def load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    """Load audio, trả về (waveform_1d, sample_rate)."""
    waveform, sr = torchaudio.load(audio_path)
    return waveform[0], sr   # mono, shape (samples,)


def resample_if_needed(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == TARGET_SR:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
    return resampler(waveform)


def slice_segment(
    waveform: torch.Tensor,
    t_start: float,
    t_stop: float,
) -> torch.Tensor:
    """Cắt waveform [t_start, t_stop] giây (đã ở TARGET_SR)."""
    i_start = max(0, int(t_start * TARGET_SR))
    i_stop  = min(len(waveform), int(t_stop  * TARGET_SR))
    return waveform[i_start:i_stop]


# ──────────────────────────────────────────────────────────────
# VAD-based silence features
# ──────────────────────────────────────────────────────────────
def compute_silence_features(segment: torch.Tensor) -> dict:
    """
    Phân tích khoảng silence trong segment audio.

    Dùng energy-based VAD đơn giản:
      - Chia segment thành frames 20ms
      - Frame có RMS energy < threshold → silence
      - Gom các silence frames liên tiếp thành pause events

    Returns dict với: mean_pause_sec, max_pause_sec, n_pauses, speech_ratio.
    """
    wav = segment.numpy()
    frame_len = max(1, int(VAD_FRAME_SEC * TARGET_SR))
    n_frames  = len(wav) // frame_len

    if n_frames == 0:
        return {"mean_pause_sec": 0.0, "max_pause_sec": 0.0,
                "n_pauses": 0, "speech_ratio": 0.0}

    frames = wav[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms    = np.sqrt((frames ** 2).mean(axis=1))

    # Normalize RMS to [0, 1] để threshold không phụ thuộc volume
    rms_max = rms.max()
    if rms_max > 0:
        rms_norm = rms / rms_max
    else:
        rms_norm = rms

    is_silence = rms_norm < VAD_ENERGY_THRESHOLD

    # Đếm speech ratio
    speech_ratio = float((~is_silence).mean())

    # Tìm pause events (chuỗi silence frames liên tiếp)
    pauses = []
    in_pause = False
    pause_start = 0
    for i, sil in enumerate(is_silence):
        if sil and not in_pause:
            in_pause = True
            pause_start = i
        elif not sil and in_pause:
            in_pause = False
            pauses.append((i - pause_start) * VAD_FRAME_SEC)
    if in_pause:
        pauses.append((n_frames - pause_start) * VAD_FRAME_SEC)

    # Chỉ tính pause > 0.15s (bỏ qua micro-pauses)
    pauses = [p for p in pauses if p >= 0.15]

    return {
        "mean_pause_sec" : round(float(np.mean(pauses)), 3) if pauses else 0.0,
        "max_pause_sec"  : round(float(np.max(pauses)), 3)  if pauses else 0.0,
        "n_pauses"       : len(pauses),
        "speech_ratio"   : round(speech_ratio, 3),
    }


# ──────────────────────────────────────────────────────────────
# Prosody features (librosa optional)
# ──────────────────────────────────────────────────────────────
def compute_prosody_features(segment: torch.Tensor) -> dict:
    """
    Tính pitch và energy features nếu librosa có sẵn.

    Returns dict với: mean_pitch_hz, std_pitch_hz, mean_energy, std_energy.
    Trả về -1.0 cho tất cả nếu librosa không có hoặc segment quá ngắn.
    """
    empty = {"mean_pitch_hz": -1.0, "std_pitch_hz": -1.0,
             "mean_energy": -1.0, "std_energy": -1.0}

    if not HAS_LIBROSA or len(segment) < TARGET_SR * 0.1:
        return empty

    wav = segment.numpy().astype(np.float32)

    try:
        # Pitch (F0) via pyin
        f0, voiced_flag, _ = librosa.pyin(
            wav, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=TARGET_SR,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        if len(voiced_f0) > 0:
            mean_pitch = float(np.nanmean(voiced_f0))
            std_pitch  = float(np.nanstd(voiced_f0))
        else:
            mean_pitch = std_pitch = 0.0

        # Energy (RMS per frame)
        rms = librosa.feature.rms(y=wav, frame_length=512, hop_length=256)[0]
        mean_energy = float(np.mean(rms))
        std_energy  = float(np.std(rms))

        return {
            "mean_pitch_hz": round(mean_pitch, 2),
            "std_pitch_hz" : round(std_pitch, 2),
            "mean_energy"  : round(mean_energy, 5),
            "std_energy"   : round(std_energy, 5),
        }
    except Exception as e:
        log.debug(f"Prosody error: {e}")
        return empty


# ──────────────────────────────────────────────────────────────
# wav2vec2 embedding
# ──────────────────────────────────────────────────────────────
def extract_utterance_embedding(
    segment: torch.Tensor,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> np.ndarray:
    """
    Embed audio segment 1-D (float32, 16 kHz) qua wav2vec2.
    Trả về mean-pooled embedding shape (768,).
    """
    inputs = processor(
        segment.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = model(input_values)

    hidden = outputs.last_hidden_state.squeeze(0)   # (T_frames, 768)
    return hidden.mean(dim=0).cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Group definition helpers
# ──────────────────────────────────────────────────────────────
def load_text_meta_groups(text_meta_path: str) -> list[dict]:
    """
    Load utterance group definitions từ text pipeline metadata.
    Mỗi phần tử có ít nhất: group_id, start_time, stop_time.
    """
    with open(text_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def build_groups_from_transcript(
    transcript_path: str,
    merge_gap_sec: float,
) -> list[dict]:
    """
    Fallback: tự parse transcript và build groups theo cùng logic
    với extract_bert_v3.py (merge consecutive turns theo merge_gap_sec).

    Trả về list[dict] với các key: group_id, start_time, stop_time.
    """
    try:
        df = pd.read_csv(transcript_path, sep="\t", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(transcript_path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df["start_time"] = pd.to_numeric(
        df.get("start_time", 0), errors="coerce"
    ).fillna(0.0)
    df["stop_time"] = pd.to_numeric(
        df.get("stop_time", 0), errors="coerce"
    ).fillna(0.0)
    df["speaker_clean"] = df["speaker"].str.strip().str.lower()

    groups = []
    group_id = 0
    current = None

    for _, row in df.iterrows():
        if row["speaker_clean"] == "ellie":
            if current is not None:
                groups.append(current)
                current = None
            continue
        if row["speaker_clean"] != "participant":
            continue

        t_start = float(row["start_time"])
        t_stop  = float(row["stop_time"])

        if current is None:
            current = {"group_id": group_id, "start_time": t_start, "stop_time": t_stop}
            group_id += 1
        else:
            gap = t_start - current["stop_time"]
            if gap <= merge_gap_sec:
                current["stop_time"] = max(current["stop_time"], t_stop)
            else:
                groups.append(current)
                current = {"group_id": group_id, "start_time": t_start, "stop_time": t_stop}
                group_id += 1

    if current is not None:
        groups.append(current)

    return groups


# ──────────────────────────────────────────────────────────────
# Per-participant processing
# ──────────────────────────────────────────────────────────────
def find_audio_file(data_root: str, pid: int) -> Optional[str]:
    """Tìm file audio của participant (thử các tên phổ biến trong DAIC-WOZ)."""
    candidates = [
        f"{pid}_AUDIO.wav", f"{pid}_audio.wav",
        f"{pid}_P.wav",     f"{pid}.wav",
    ]
    for name in candidates:
        path = os.path.join(data_root, name)
        if os.path.exists(path):
            return path
    return None


def extract_features(
    audio_path: str,
    groups: list[dict],
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> tuple[np.ndarray, list[dict]]:
    """
    Extract wav2vec2 embeddings cho tất cả utterance groups của 1 participant.

    Args:
      audio_path : path đến file WAV của participant
      groups     : list các group dicts với start_time và stop_time

    Returns:
      embeddings : (N_groups, 768) — zeros nếu segment không hợp lệ
      metadata   : list[dict] với audio features cho mỗi group
    """
    if not groups:
        return np.zeros((1, EMBED_DIM), dtype=np.float32), []

    waveform, sr = load_audio(audio_path)
    waveform     = resample_if_needed(waveform, sr)

    embeddings = []
    metadata   = []

    for group in groups:
        t_start  = float(group["start_time"])
        t_stop   = float(group["stop_time"])
        duration = t_stop - t_start
        group_id = group.get("group_id", len(metadata))

        meta = {
            "group_id"       : group_id,
            "start_time"     : round(t_start, 3),
            "stop_time"      : round(t_stop, 3),
            "duration"       : round(duration, 3),
            "used_zeros"     : False,
            "mean_pause_sec" : 0.0,
            "max_pause_sec"  : 0.0,
            "n_pauses"       : 0,
            "speech_ratio"   : 0.0,
            "mean_pitch_hz"  : -1.0,
            "std_pitch_hz"   : -1.0,
            "mean_energy"    : -1.0,
            "std_energy"     : -1.0,
        }

        # Segment quá ngắn → zeros
        if duration < MIN_DURATION:
            meta["used_zeros"] = True
            embeddings.append(np.zeros(EMBED_DIM, dtype=np.float32))
            metadata.append(meta)
            continue

        segment = slice_segment(waveform, t_start, t_stop)

        if segment.numel() < int(TARGET_SR * MIN_DURATION):
            meta["used_zeros"] = True
            embeddings.append(np.zeros(EMBED_DIM, dtype=np.float32))
            metadata.append(meta)
            continue

        # Silence / pause features
        silence_feats = compute_silence_features(segment)
        meta.update(silence_feats)

        # Prosody features
        prosody_feats = compute_prosody_features(segment)
        meta.update(prosody_feats)

        # wav2vec2 embedding
        try:
            emb = extract_utterance_embedding(segment, processor, model, device)
        except Exception as e:
            log.warning(f"  Embed error group {group_id} [{t_start:.2f}-{t_stop:.2f}s]: {e}")
            emb = np.zeros(EMBED_DIM, dtype=np.float32)
            meta["used_zeros"] = True

        embeddings.append(emb)
        metadata.append(meta)

    embeddings_arr = np.vstack(embeddings).astype(np.float32)
    return embeddings_arr, metadata


def process_participant(
    pid: int,
    data_root: str,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
    merge_gap_sec: float,
    use_text_meta: bool,
    overwrite: bool,
) -> bool:
    """Xử lý 1 participant. Trả về True nếu thành công."""
    transcript_path = os.path.join(data_root, f"{pid}_TRANSCRIPT.csv")
    text_meta_path  = os.path.join(data_root, f"{pid}_text_feats_meta.json")
    out_npy         = os.path.join(data_root, f"{pid}_audio_feats.npy")
    out_meta        = os.path.join(data_root, f"{pid}_audio_feats_meta.json")

    # if not overwrite and os.path.exists(out_npy):
    #     log.info(f"[{pid}] Already exists — skip (use --overwrite to force)")
    #     return True

    audio_path = find_audio_file(data_root, pid)
    if audio_path is None:
        log.warning(f"[{pid}] Audio file not found in {data_root}")
        return False

    # Quyết định nguồn group definitions
    if use_text_meta and os.path.exists(text_meta_path):
        groups = load_text_meta_groups(text_meta_path)
        log.info(f"[{pid}] Using text meta ({len(groups)} groups): {text_meta_path}")
    else:
        if use_text_meta:
            log.warning(
                f"[{pid}] text_feats_meta.json not found — "
                f"falling back to transcript parsing (run extract_bert_v3.py first "
                f"to ensure perfect alignment)."
            )
        if not os.path.exists(transcript_path):
            log.warning(f"[{pid}] Transcript not found either: {transcript_path}")
            return False
        groups = build_groups_from_transcript(transcript_path, merge_gap_sec)
        log.info(f"[{pid}] Fallback: parsed transcript → {len(groups)} groups")

    log.info(f"[{pid}] Embedding audio: {audio_path}")

    try:
        embeddings, meta = extract_features(
            audio_path=audio_path,
            groups=groups,
            processor=processor,
            model=model,
            device=device,
        )
    except Exception as e:
        log.error(f"[{pid}] Feature extraction failed: {e}")
        return False

    n_groups = len(meta)
    n_zeros  = sum(1 for m in meta if m.get("used_zeros"))

    avg_speech = (
        round(np.mean([m["speech_ratio"] for m in meta if not m["used_zeros"]]), 3)
        if any(not m["used_zeros"] for m in meta) else 0.0
    )
    avg_pauses = (
        round(np.mean([m["n_pauses"] for m in meta if not m["used_zeros"]]), 1)
        if any(not m["used_zeros"] for m in meta) else 0.0
    )

    np.save(out_npy, embeddings)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info(
        f"[{pid}] Done — groups={n_groups} (zeros={n_zeros}) "
        f"| avg_speech_ratio={avg_speech}, avg_pauses/group={avg_pauses} "
        f"| shape={embeddings.shape} → {out_npy}"
    )
    return True


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract wav2vec2 audio embeddings từ DAIC-WOZ (v2)"
    )
    parser.add_argument("--data_root",    default="daicwoz/")
    parser.add_argument("--split_csv",    required=True,
                        help="Train/dev CSV có cột Participant_ID")
    parser.add_argument("--model_name",   default="facebook/wav2vec2-base-960h")
    parser.add_argument("--hf_token",     default=None)
    parser.add_argument("--merge_gap",    type=float, default=DEFAULT_MERGE_GAP_SEC,
                        help=f"Khoảng cách tối đa (giây) để merge turns (fallback mode). "
                             f"Phải khớp với giá trị dùng trong extract_bert_v3.py. "
                             f"(default: {DEFAULT_MERGE_GAP_SEC})")
    parser.add_argument("--no_text_meta", action="store_true",
                        help="Không dùng text_feats_meta.json, tự parse transcript. "
                             "Dùng khi chạy audio pipeline độc lập (alignment kém hơn).")
    parser.add_argument("--overwrite",    action="store_true")
    args = parser.parse_args()

    use_text_meta = not args.no_text_meta
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Device       : {device}")
    log.info(f"Model        : {args.model_name}")
    log.info(f"Use text meta: {use_text_meta}")
    log.info(f"Merge gap    : {args.merge_gap}s (fallback only)")
    log.info(f"Librosa      : {'available' if HAS_LIBROSA else 'NOT installed — prosody features disabled'}")

    if not HAS_LIBROSA:
        log.warning("Install librosa for pitch/energy features: pip install librosa")

    hf_kwargs = {"token": args.hf_token} if args.hf_token else {}
    log.info("Loading wav2vec2 processor & model …")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name, **hf_kwargs)
    model     = Wav2Vec2Model.from_pretrained(args.model_name, **hf_kwargs).to(device)
    model.eval()
    log.info("Model loaded.")

    split_df = pd.read_csv(args.split_csv)
    split_df.columns = [c.strip() for c in split_df.columns]
    pids = split_df["Participant_ID"].astype(int).tolist()
    log.info(f"Participants : {len(pids)}")

    success = 0
    for pid in pids:
        ok = process_participant(
            pid=pid, data_root=args.data_root,
            processor=processor, model=model, device=device,
            merge_gap_sec=args.merge_gap,
            use_text_meta=use_text_meta,
            overwrite=args.overwrite,
        )
        if ok:
            success += 1

    log.info(f"\nDone: {success}/{len(pids)} participants.")


if __name__ == "__main__":
    main()