"""
extract_wav2vec_v3.py — Extract audio embeddings từ DAIC-WOZ (v3, full rewrite).

Fixes vs v2_optimized
─────────────────────
[FIX-CRITICAL] Switch từ wav2vec2-base-960h (ASR fine-tuned) sang WavLM
  (microsoft/wavlm-base-plus, self-supervised). ASR fine-tuning đẩy model
  focus vào phoneme identity, mất prosodic/para-linguistic features — chính
  xác những gì depression detection cần. WavLM được trained với masked
  speech prediction + denoising → robust hơn với clinical speech quality.

[FIX-CRITICAL] Adaptive VAD threshold: thay vì fixed 0.01, tính per-segment
  noise floor từ silent prefix/suffix của recording rồi set threshold =
  noise_floor * 3.0. Giảm confounding giữa recording quality và speech ratio.

[FIX-MAJOR] Minimum speech quality gate: skip embedding nếu speech_ratio < 0.1
  sau VAD (segment chủ yếu là silence). Trước đây những segments này tạo
  ra garbage embeddings (mean-pool of silence frames).

[FIX-MAJOR] Layer-wise feature extraction: WavLM có 12 transformer layers,
  mỗi layer encode thông tin khác nhau. Dùng weighted average của layers 6-11
  (upper-mid layers capture prosody tốt hơn last layer cho depression tasks).

[KEEP] Batched inference, FP16, duration-sorted batching, prefetch.
[KEEP] Prosody features (pitch, energy) từ librosa.
[KEEP] Silence/pause features từ energy VAD.
[KEEP] Sync với text meta groups (groups từ BERT extractor).
"""

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import AutoProcessor, WavLMModel

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logging.warning("librosa not installed — prosody features will be zeros.")

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
TARGET_SR            = 16_000
EMBED_DIM            = 768
MIN_DURATION_SEC     = 0.1
MAX_SEGMENT_SEC      = 30.0
DEFAULT_MERGE_GAP    = 2.0

# WavLM layer selection: layers 6-11 (0-indexed) — upper-mid layers
# proven better for paralinguistic tasks vs last layer
WAVLM_LAYER_START    = 6
WAVLM_LAYER_END      = 12   # exclusive → layers [6,7,8,9,10,11]

# Adaptive VAD
VAD_FRAME_SEC        = 0.02
VAD_NOISE_MULTIPLIER = 3.0   # threshold = noise_floor * 3.0
MIN_SPEECH_RATIO     = 0.10  # below this → skip embedding (silence segment)
MIN_PAUSE_DUR_SEC    = 0.15  # pauses shorter than this ignored

_SYNC_RE = re.compile(r"^\s*<\s*synch?\s*>\s*$", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(audio_path: str) -> tuple:
    waveform, sr = torchaudio.load(audio_path)
    return waveform[0], sr  # mono


def resample_if_needed(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == TARGET_SR:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
    return resampler(waveform)


def slice_segment(
    waveform: torch.Tensor,
    t_start:  float,
    t_stop:   float,
    max_sec:  float = MAX_SEGMENT_SEC,
) -> torch.Tensor:
    if t_stop - t_start > max_sec:
        t_stop = t_start + max_sec
    i_start = max(0, int(t_start * TARGET_SR))
    i_stop  = min(len(waveform), int(t_stop * TARGET_SR))
    return waveform[i_start:i_stop]


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive VAD
# ─────────────────────────────────────────────────────────────────────────────

def estimate_noise_floor(waveform: torch.Tensor) -> float:
    """
    Estimate noise floor using the p5 of 1-second window RMS distribution.

    BUG FIX (critical): the previous implementation used the first 0.5s of
    the recording as the noise floor estimate. In DAIC-WOZ, Ellie (the
    interviewer) speaks FIRST — so the prefix is Ellie's voice, not silence.
    For PID 310, this caused a 40x overestimate of the noise floor
    (0.0196 vs true ~0.0005), setting VAD threshold so high that 100% of
    participant speech frames were classified as silence → all zeros.

    Fix: use the 5th percentile of 1s-window RMS values across the full
    recording. The quietest 5% of 1s windows reliably capture true silence
    regardless of who speaks at the beginning.
    """
    wav     = waveform.numpy()
    win_len = TARGET_SR  # 1 second windows
    n_wins  = max(1, len(wav) // win_len)

    win_rms = np.array([
        float(np.sqrt(np.mean(wav[i * win_len:(i + 1) * win_len] ** 2)))
        for i in range(n_wins)
    ])

    # p5 of window RMS = a robust estimate of the recording noise floor
    noise_floor = float(np.percentile(win_rms, 5))
    return float(np.clip(noise_floor, 1e-6, 0.05))


def compute_silence_features(
    segment:         torch.Tensor,
    noise_floor:     float,
) -> dict:
    """
    Per-segment pause/silence features using adaptive VAD threshold.
    threshold = noise_floor * VAD_NOISE_MULTIPLIER
    """
    wav        = segment.numpy()
    frame_len  = max(1, int(VAD_FRAME_SEC * TARGET_SR))
    n_frames   = len(wav) // frame_len

    if n_frames == 0:
        return {
            "mean_pause_sec": 0.0, "max_pause_sec": 0.0,
            "n_pauses": 0, "speech_ratio": 0.0,
        }

    frames    = wav[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms       = np.sqrt((frames ** 2).mean(axis=1))
    threshold = noise_floor * VAD_NOISE_MULTIPLIER

    is_silence   = rms < threshold
    speech_ratio = float((~is_silence).mean())

    pauses = []
    in_pause, pause_start = False, 0
    for i, sil in enumerate(is_silence):
        if sil and not in_pause:
            in_pause, pause_start = True, i
        elif not sil and in_pause:
            in_pause = False
            pauses.append((i - pause_start) * VAD_FRAME_SEC)
    if in_pause:
        pauses.append((n_frames - pause_start) * VAD_FRAME_SEC)

    pauses = [p for p in pauses if p >= MIN_PAUSE_DUR_SEC]

    return {
        "mean_pause_sec" : round(float(np.mean(pauses)), 3) if pauses else 0.0,
        "max_pause_sec"  : round(float(np.max(pauses)),  3) if pauses else 0.0,
        "n_pauses"       : len(pauses),
        "speech_ratio"   : round(speech_ratio, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prosody features
# ─────────────────────────────────────────────────────────────────────────────

def compute_prosody_features(segment: torch.Tensor) -> dict:
    empty = {
        "mean_pitch_hz": -1.0, "std_pitch_hz": -1.0,
        "mean_energy": -1.0,   "std_energy": -1.0,
    }
    if not HAS_LIBROSA or len(segment) < TARGET_SR * 0.1:
        return empty
    wav = segment.numpy().astype(np.float32)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            wav,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=TARGET_SR,
        )
        voiced_f0  = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        mean_pitch = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        std_pitch  = float(np.nanstd(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        rms        = librosa.feature.rms(y=wav, frame_length=512, hop_length=256)[0]
        return {
            "mean_pitch_hz": round(mean_pitch, 2),
            "std_pitch_hz" : round(std_pitch,  2),
            "mean_energy"  : round(float(np.mean(rms)), 5),
            "std_energy"   : round(float(np.std(rms)),  5),
        }
    except Exception as e:
        log.debug(f"Prosody error: {e}")
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# WavLM layer-weighted embedding (KEY FIX)
# ─────────────────────────────────────────────────────────────────────────────

def embed_segments_batched(
    segments:   List[torch.Tensor],
    processor,
    model:      WavLMModel,
    device:     torch.device,
    batch_size: int = 16,
    use_fp16:   bool = False,
) -> List[np.ndarray]:
    """
    Batch-embed segments using WavLM with weighted average of layers 6-11.

    Key difference from v2: instead of taking last_hidden_state, we
    take hidden states from WAVLM_LAYER_START to WAVLM_LAYER_END,
    compute uniform mean (learnable weighting is in the dataset/model).
    Upper-mid layers of WavLM capture prosodic and speaker-level features
    better than the final layer.
    """
    if not segments:
        return []

    n = len(segments)
    sorted_indices = sorted(range(n), key=lambda i: len(segments[i]))
    all_embeddings: List[Optional[np.ndarray]] = [None] * n
    model.eval()
    use_autocast = use_fp16 and device.type == "cuda"

    with torch.no_grad():
        for batch_start in range(0, n, batch_size):
            batch_idx = sorted_indices[batch_start: batch_start + batch_size]
            batch_np  = [segments[i].numpy() for i in batch_idx]

            inputs = processor(
                batch_np,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
            input_values  = inputs["input_values"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(
                        input_values,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                hidden_states = [h.float() for h in outputs.hidden_states]
            else:
                outputs = model(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = list(outputs.hidden_states)

            # FIX: use layers 6-11 (upper-mid), uniform mean
            selected_layers = hidden_states[WAVLM_LAYER_START:WAVLM_LAYER_END]
            stacked = torch.stack(selected_layers, dim=0)   # (K, B, T, H)
            merged  = stacked.mean(dim=0)                   # (B, T, H)

            # Mean-pool over time with attention mask
            if attention_mask is not None:
                mask   = attention_mask.unsqueeze(-1).float()
                pooled = (merged * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = merged.mean(dim=1)

            batch_embs = pooled.cpu().numpy().astype(np.float32)
            for j, orig_idx in enumerate(batch_idx):
                all_embeddings[orig_idx] = batch_embs[j]

    return all_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Group loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_text_meta_groups(text_meta_path: str) -> List[dict]:
    with open(text_meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_groups_from_transcript(
    transcript_path: str,
    merge_gap_sec:   float,
) -> List[dict]:
    try:
        df = pd.read_csv(transcript_path, sep="\t", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(transcript_path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df["start_time"]    = pd.to_numeric(df.get("start_time", 0), errors="coerce").fillna(0.0)
    df["stop_time"]     = pd.to_numeric(df.get("stop_time",  0), errors="coerce").fillna(0.0)
    df["speaker_clean"] = df["speaker"].str.strip().str.lower()
    df["value"]         = df["value"].fillna("").astype(str)

    groups = []
    group_id, current = 0, None

    for _, row in df.iterrows():
        if row["speaker_clean"] == "ellie":
            if current is not None:
                groups.append(current)
                current = None
            continue
        if row["speaker_clean"] != "participant":
            continue
        if _SYNC_RE.match(row["value"].strip()):
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


# ─────────────────────────────────────────────────────────────────────────────
# Audio file finder
# ─────────────────────────────────────────────────────────────────────────────

def find_audio_file(data_root: str, pid: int) -> Optional[str]:
    for name in [f"{pid}_AUDIO.wav", f"{pid}_audio.wav", f"{pid}_P.wav", f"{pid}.wav"]:
        path = os.path.join(data_root, name)
        if os.path.exists(path):
            return path
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-participant feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    waveform:        torch.Tensor,
    groups:          List[dict],
    processor,
    model:           WavLMModel,
    device:          torch.device,
    batch_size:      int,
    use_fp16:        bool,
    max_segment_sec: float,
) -> tuple:
    if not groups:
        return np.zeros((1, EMBED_DIM), dtype=np.float32), []

    # Estimate recording-level noise floor (first 0.5s of full waveform)
    recording_noise_floor = estimate_noise_floor(waveform)
    log.debug(f"  recording noise floor: {recording_noise_floor:.5f}")

    segments    = []
    metadata    = []
    valid_mask  = []

    for group in groups:
        t_start  = float(group["start_time"])
        t_stop   = float(group["stop_time"])
        duration = t_stop - t_start
        gid      = group.get("group_id", len(metadata))

        meta = {
            "group_id":      gid,
            "start_time":    round(t_start, 3),
            "stop_time":     round(t_stop,  3),
            "duration":      round(duration, 3),
            "used_zeros":    False,
            "low_speech":    False,
            "mean_pause_sec": 0.0, "max_pause_sec": 0.0,
            "n_pauses":      0,    "speech_ratio":  0.0,
            "mean_pitch_hz": -1.0, "std_pitch_hz":  -1.0,
            "mean_energy":   -1.0, "std_energy":    -1.0,
            "noise_floor":   round(recording_noise_floor, 5),
        }

        if duration < MIN_DURATION_SEC:
            meta["used_zeros"] = True
            segments.append(None)
            valid_mask.append(False)
            metadata.append(meta)
            continue

        segment = slice_segment(waveform, t_start, t_stop, max_segment_sec)
        if segment.numel() < int(TARGET_SR * MIN_DURATION_SEC):
            meta["used_zeros"] = True
            segments.append(None)
            valid_mask.append(False)
            metadata.append(meta)
            continue

        # Acoustic features (CPU)
        silence_feats = compute_silence_features(segment, recording_noise_floor)
        meta.update(silence_feats)
        meta.update(compute_prosody_features(segment))

        # FIX: quality gate on speech ratio
        if silence_feats["speech_ratio"] < MIN_SPEECH_RATIO:
            meta["used_zeros"] = True
            meta["low_speech"] = True
            segments.append(None)
            valid_mask.append(False)
            metadata.append(meta)
            log.debug(f"  Group {gid}: speech_ratio={silence_feats['speech_ratio']:.2f} < {MIN_SPEECH_RATIO} → zeros")
            continue

        segments.append(segment)
        valid_mask.append(True)
        metadata.append(meta)

    # Batch embed valid segments
    valid_segments = [s for s, v in zip(segments, valid_mask) if v]

    if valid_segments:
        try:
            valid_embeddings = embed_segments_batched(
                valid_segments, processor, model, device,
                batch_size=batch_size, use_fp16=use_fp16,
            )
        except Exception as e:
            log.error(f"Batch embedding failed: {e} — using zeros for this batch")
            valid_embeddings = [np.zeros(EMBED_DIM, dtype=np.float32)] * len(valid_segments)
            for i, v in enumerate(valid_mask):
                if v:
                    metadata[i]["used_zeros"] = True
    else:
        valid_embeddings = []

    # Reassemble
    embeddings  = []
    valid_iter  = iter(valid_embeddings)
    for v in valid_mask:
        if v:
            embeddings.append(next(valid_iter))
        else:
            embeddings.append(np.zeros(EMBED_DIM, dtype=np.float32))

    return np.vstack(embeddings).astype(np.float32), metadata


def process_participant(
    pid:             int,
    data_root:       str,
    waveform:        torch.Tensor,
    groups:          List[dict],
    processor,
    model:           WavLMModel,
    device:          torch.device,
    batch_size:      int,
    use_fp16:        bool,
    max_segment_sec: float,
    overwrite:       bool,
) -> bool:
    out_npy  = os.path.join(data_root, f"{pid}_audio_feats.npy")
    out_meta = os.path.join(data_root, f"{pid}_audio_feats_meta.json")

    if not overwrite and os.path.exists(out_npy):
        log.info(f"[{pid}] Already exists — skip (use --overwrite to redo)")
        return True

    try:
        embeddings, meta = extract_features(
            waveform, groups, processor, model, device,
            batch_size, use_fp16, max_segment_sec,
        )
    except Exception as e:
        log.error(f"[{pid}] Failed: {e}")
        return False

    np.save(out_npy, embeddings)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    n_groups    = len(meta)
    n_zeros     = sum(1 for m in meta if m.get("used_zeros"))
    n_low_speech = sum(1 for m in meta if m.get("low_speech"))
    log.info(
        f"[{pid}] Done — groups={n_groups} "
        f"(zeros={n_zeros}, low_speech={n_low_speech}) "
        f"shape={embeddings.shape}"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Prefetch helper
# ─────────────────────────────────────────────────────────────────────────────

def prefetch_audio(audio_path: str) -> tuple:
    try:
        waveform, sr = load_audio(audio_path)
        waveform = resample_if_needed(waveform, sr)
        return waveform, True
    except Exception as e:
        log.error(f"Prefetch failed: {audio_path}: {e}")
        return torch.zeros(1), False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract WavLM audio embeddings (v3 — adaptive VAD, layer 6-11 pool)"
    )
    parser.add_argument("--data_root",   default="daicwoz/")
    parser.add_argument("--split_csv",   required=True)
    parser.add_argument("--model_name",  default="microsoft/wavlm-base-plus",
                        help="Recommended: wavlm-base-plus (self-supervised, not ASR-finetuned)")
    parser.add_argument("--hf_token",    default=None)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--merge_gap",   type=float, default=DEFAULT_MERGE_GAP)
    parser.add_argument("--no_text_meta", action="store_true")
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--fp16",        action="store_true")
    parser.add_argument("--max_segment_sec", type=float, default=MAX_SEGMENT_SEC)
    parser.add_argument("--num_prefetch_workers", type=int, default=2)
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type == "cuda"

    log.info(f"Device      : {device}")
    log.info(f"Model       : {args.model_name}")
    log.info(f"FP16        : {use_fp16}")
    log.info(f"Layer pool  : [{WAVLM_LAYER_START},{WAVLM_LAYER_END}) (upper-mid layers)")
    log.info(f"VAD         : adaptive (noise_floor * {VAD_NOISE_MULTIPLIER:.1f})")
    log.info(f"Min speech  : {MIN_SPEECH_RATIO} ratio (below → zeros)")
    log.info(f"Librosa     : {'yes' if HAS_LIBROSA else 'no'}")

    hf_kwargs = {"token": args.hf_token} if args.hf_token else {}
    processor = AutoProcessor.from_pretrained(args.model_name, **hf_kwargs)
    model     = WavLMModel.from_pretrained(args.model_name, **hf_kwargs).to(device)
    model.eval()

    split_df = pd.read_csv(args.split_csv)
    split_df.columns = [c.strip() for c in split_df.columns]
    pids = split_df["Participant_ID"].astype(int).tolist()
    log.info(f"Participants: {len(pids)}")

    use_text_meta = not args.no_text_meta

    participant_info = []
    for pid in pids:
        transcript_path = os.path.join(args.data_root, f"{pid}_TRANSCRIPT.csv")
        text_meta_path  = os.path.join(args.data_root, f"{pid}_text_feats_meta.json")

        audio_path = find_audio_file(args.data_root, pid)
        if audio_path is None:
            log.warning(f"[{pid}] Audio file not found — skip")
            continue

        if use_text_meta and os.path.exists(text_meta_path):
            groups = load_text_meta_groups(text_meta_path)
        else:
            if use_text_meta:
                log.warning(f"[{pid}] text meta not found — fallback to transcript")
            if not os.path.exists(transcript_path):
                log.warning(f"[{pid}] Transcript not found — skip")
                continue
            groups = build_groups_from_transcript(transcript_path, args.merge_gap)

        participant_info.append((pid, audio_path, groups))

    if not participant_info:
        log.info("Nothing to process.")
        return

    executor = ThreadPoolExecutor(max_workers=args.num_prefetch_workers)
    futures  = {}
    for i, (pid, audio_path, _) in enumerate(participant_info):
        if i < args.num_prefetch_workers:
            futures[i] = executor.submit(prefetch_audio, audio_path)

    success = 0
    for i, (pid, audio_path, groups) in enumerate(participant_info):
        waveform, ok = futures[i].result() if i in futures else prefetch_audio(audio_path)

        next_i = i + args.num_prefetch_workers
        if next_i < len(participant_info):
            _, next_audio, _ = participant_info[next_i]
            futures[next_i] = executor.submit(prefetch_audio, next_audio)

        if not ok:
            log.warning(f"[{pid}] Audio load failed")
            continue

        log.info(f"[{pid}] Embedding {len(groups)} groups ({audio_path})")
        ok = process_participant(
            pid, args.data_root, waveform, groups,
            processor, model, device,
            args.batch_size, use_fp16, args.max_segment_sec,
            overwrite=args.overwrite,
        )
        if ok:
            success += 1

    executor.shutdown(wait=False)
    log.info(f"\nDone: {success}/{len(participant_info)} participants.")


if __name__ == "__main__":
    main()