"""
daicwoz_dataset.py (FIXED v2)

DaicWozDataset: PyG InMemoryDataset for the original DAIC-WOZ corpus.

Fixes vs original FIXED:
  1. [FIX-CRITICAL] n_utt now reads from text_feats_meta.json (or n_groups.txt)
     instead of counting raw participant rows in the transcript.
  2. [FIX-MAJOR] Hard assertion on text/audio alignment mismatch.
     Previously: silently used min() and continued.
     Now: logs a WARNING with participant details, tracks misaligned PIDs,
     and writes a {split}_alignment_report.json for post-hoc audit.
  3. [FIX-SHORTCUT] TEXT_CONTEXT_MODE controls which extracted embedding file
     to load: ctx_none / ctx_truncated / ctx_full.
     Set via environment variable DAICWOZ_CONTEXT_MODE or constructor arg.
     Default: "truncated" (backward-compatible).
  4. [FIX] Added _ALIGNMENT_REPORT path and per-process stats.

All other graph construction logic (edge types, unified features, etc.) preserved.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path.cwd()

DAICWOZ_DATA_DIR  = BASE_DIR / "daicwoz"
DAICWOZ_CACHE_DIR = BASE_DIR / "cache_daicwoz"

TEXT_DIM    = 768
AUDIO_DIM   = 768

TEXT_ACOUSTIC_DIM  = 8
AUDIO_ACOUSTIC_DIM = 8

TEXT_DIM_TOTAL  = TEXT_DIM  + TEXT_ACOUSTIC_DIM    # 776
AUDIO_DIM_TOTAL = AUDIO_DIM + AUDIO_ACOUSTIC_DIM   # 776
UNIFIED_DIM     = TEXT_DIM_TOTAL + AUDIO_DIM_TOTAL  # 1552

TEMPORAL_WINDOW = 5

DAICWOZ_PHQ8_COLS = [
    "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
    "PHQ8_Appetite", "PHQ8_Failure", "PHQ8_Concentrating", "PHQ8_Moving",
]

DAICWOZ_SPLIT_FILES = {
    "train": DAICWOZ_DATA_DIR / "train_split_Depression_AVEC2017.csv",
    "dev":   DAICWOZ_DATA_DIR / "dev_split_Depression_AVEC2017.csv",
    "test":  DAICWOZ_DATA_DIR / "full_test_split.csv",
}

# SHORTCUT ABLATION: which context mode embedding to load.
# Can be overridden by environment variable DAICWOZ_CONTEXT_MODE.
# Values: "none" | "truncated" | "full"
_DEFAULT_CONTEXT_MODE = os.environ.get("DAICWOZ_CONTEXT_MODE", "truncated")


# ─────────────────────────────────────────────────────────────────────────────
# Feature loading
# ─────────────────────────────────────────────────────────────────────────────

def _get_text_feats_path(pid: int, context_mode: str) -> Path:
    """
    Return path to text features for the given context mode.

    File naming convention (from extract_bert_v3_fixed.py):
      {pid}_text_feats_ctx_none.npy
      {pid}_text_feats_ctx_truncated.npy
      {pid}_text_feats_ctx_full.npy

    Fallback: if the mode-specific file doesn't exist, try the legacy
    name {pid}_text_feats.npy (for backward compatibility with old extractions).
    """
    mode_path = DAICWOZ_DATA_DIR / f"{pid}_text_feats_ctx_{context_mode}.npy"
    if mode_path.exists():
        return mode_path
    legacy_path = DAICWOZ_DATA_DIR / f"{pid}_text_feats.npy"
    if legacy_path.exists():
        warnings.warn(
            f"PID {pid}: mode-specific file for context_mode='{context_mode}' not found. "
            f"Falling back to legacy {pid}_text_feats.npy. "
            f"Re-run extract_bert_v3_fixed.py with --context_mode {context_mode} "
            f"to get proper ablation files."
        )
        return legacy_path
    raise FileNotFoundError(
        f"PID {pid}: No text features found for context_mode='{context_mode}'. "
        f"Tried: {mode_path} and {legacy_path}. "
        f"Run: python extract_bert_v3_fixed.py --context_mode {context_mode} ..."
    )


def _get_n_groups(pid: int) -> Optional[int]:
    """
    Get the number of utterance groups from the extraction pipeline.
    Priority: text_feats_meta.json > n_groups.txt > text_feats npy shape.
    """
    meta_path = DAICWOZ_DATA_DIR / f"{pid}_text_feats_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return len(meta)

    ngroups_path = DAICWOZ_DATA_DIR / f"{pid}_n_groups.txt"
    if ngroups_path.exists():
        with open(ngroups_path, "r") as f:
            return int(f.read().strip())

    # Fallback: infer from any available npy file
    for suffix in ["ctx_none", "ctx_truncated", "ctx_full", ""]:
        fname = f"{pid}_text_feats_{suffix}.npy" if suffix else f"{pid}_text_feats.npy"
        npy_path = DAICWOZ_DATA_DIR / fname
        if npy_path.exists():
            arr = np.load(str(npy_path))
            return arr.shape[0] if arr.ndim == 2 else 1

    return None


def _load_feats_from_path(path: Path, n_utt: int, dim: int, label: str) -> np.ndarray:
    """
    Load pre-extracted features. Returns (n_utt, dim). Truncates or zero-pads.
    """
    feats = np.load(str(path)).astype(np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    M = feats.shape[0]
    if M == n_utt:
        return feats
    if M > n_utt:
        return feats[:n_utt]
    if M < n_utt:
        warnings.warn(
            f"{label}: {path.name} has {M} rows but expected {n_utt}. Padding with zeros."
        )
    pad = np.zeros((n_utt - M, dim), dtype=np.float32)
    return np.concatenate([feats, pad], axis=0)


def _load_text_feats(pid: int, n_utt: int, context_mode: str) -> np.ndarray:
    path = _get_text_feats_path(pid, context_mode)
    return _load_feats_from_path(path, n_utt, TEXT_DIM, f"PID {pid} text")


def _load_audio_feats(pid: int, n_utt: int) -> np.ndarray:
    path = DAICWOZ_DATA_DIR / f"{pid}_audio_feats.npy"
    return _load_feats_from_path(path, n_utt, AUDIO_DIM, f"PID {pid} audio")


# ─────────────────────────────────────────────────────────────────────────────
# Alignment validation — FIX-MAJOR
# ─────────────────────────────────────────────────────────────────────────────

def _validate_alignment(
    pid: int,
    n_text: int,
    n_audio: int,
    n_groups: int,
    context_mode: str,
) -> Tuple[bool, int, str]:
    """
    Validate that text and audio feature counts are consistent.

    Returns (is_valid, effective_n_utt, warning_message).

    Policy (hardened vs original):
      - If counts differ by ≤ 2 utterances: warn + use min (minor sync drift)
      - If counts differ by > 2 utterances: FAIL the participant (skip graph)
        because same-utterance T→A / A→T edges would be semantically wrong.

    The threshold of 2 is a pragmatic allowance for edge cases where
    audio extraction drops 1-2 very short segments (< 0.1s) that BERT
    still embeds as zeros.
    """
    mismatch_abs = abs(n_text - n_audio)

    if mismatch_abs == 0:
        # Perfect alignment — best case
        return True, min(n_text, n_groups), ""

    msg = (
        f"PID {pid}: ALIGNMENT MISMATCH — "
        f"text={n_text}, audio={n_audio}, meta_groups={n_groups}, "
        f"context_mode={context_mode!r}. "
        f"|diff|={mismatch_abs}."
    )

    if mismatch_abs <= 2:
        effective = min(n_text, n_audio, n_groups)
        return True, effective, msg + f" Minor drift → using min={effective}."
    else:
        return False, 0, (
            msg + " CRITICAL MISMATCH (>2). Skipping participant. "
            "Re-run extract_bert and extract_wav2vec with the same merge_gap."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prosodic / acoustic feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_acoustics(meta: dict) -> np.ndarray:
    sr  = float(meta.get("speech_rate_wps", 0.0) or 0.0) / 5.0
    lat = float(meta.get("response_latency_sec", 0.0) or 0.0) / 10.0
    sigh   = float(bool(meta.get("has_sigh",        False)))
    laugh  = float(bool(meta.get("has_laughter",    False)))
    breath = float(bool(meta.get("has_breath",      False)))
    cry    = float(bool(meta.get("has_cry",         False)))
    cough  = float(bool(meta.get("has_cough",       False)))
    other  = float(bool(meta.get("has_other_sound", False)))
    return np.array([sr, lat, sigh, laugh, breath, cry, cough, other], dtype=np.float32)


def _extract_audio_acoustics(meta: dict) -> np.ndarray:
    mean_pause = float(meta.get("mean_pause_sec", 0.0) or 0.0) / 5.0
    max_pause  = float(meta.get("max_pause_sec",  0.0) or 0.0) / 10.0
    n_pauses   = float(meta.get("n_pauses",        0.0) or 0.0) / 10.0
    speech_rat = float(meta.get("speech_ratio",    0.5) or 0.5)
    mean_pitch = (float(meta.get("mean_pitch_hz",  200.0) or 200.0) - 50.0) / 450.0
    std_pitch  = float(meta.get("std_pitch_hz",    0.0) or 0.0) / 100.0
    mean_nrg   = float(meta.get("mean_energy",     0.0) or 0.0) / 0.2
    std_nrg    = float(meta.get("std_energy",      0.0) or 0.0) / 0.1
    return np.array(
        [mean_pause, max_pause, n_pauses, speech_rat, mean_pitch, std_pitch, mean_nrg, std_nrg],
        dtype=np.float32,
    )


def _load_text_acoustics(pid: int, n_utt: int) -> np.ndarray:
    path = DAICWOZ_DATA_DIR / f"{pid}_text_feats_meta.json"
    zeros = np.zeros((n_utt, TEXT_ACOUSTIC_DIM), dtype=np.float32)
    if not path.exists():
        return zeros
    with open(path, "r") as f:
        meta_list = json.load(f)
    rows = [_extract_text_acoustics(m) for m in meta_list]
    arr = np.stack(rows, axis=0) if rows else zeros
    M = arr.shape[0]
    if M >= n_utt:
        return arr[:n_utt]
    pad = np.zeros((n_utt - M, TEXT_ACOUSTIC_DIM), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def _load_audio_acoustics(pid: int, n_utt: int) -> np.ndarray:
    path = DAICWOZ_DATA_DIR / f"{pid}_audio_feats_meta.json"
    zeros = np.zeros((n_utt, AUDIO_ACOUSTIC_DIM), dtype=np.float32)
    if not path.exists():
        return zeros
    with open(path, "r") as f:
        meta_list = json.load(f)
    rows = [_extract_audio_acoustics(m) for m in meta_list]
    arr = np.stack(rows, axis=0) if rows else zeros
    M = arr.shape[0]
    if M >= n_utt:
        return arr[:n_utt]
    pad = np.zeros((n_utt - M, AUDIO_ACOUSTIC_DIM), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Edge construction
# ─────────────────────────────────────────────────────────────────────────────

def _temporal_edges(n: int, offset: int, win: int = TEMPORAL_WINDOW) -> np.ndarray:
    src, dst = [], []
    for i in range(n):
        lo = max(0, i - win)
        hi = min(n - 1, i + win)
        for j in range(lo, hi + 1):
            if j != i:
                src.append(offset + i)
                dst.append(offset + j)
    if not src:
        return np.empty((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)


def _same_utt_edges(n: int, offset_src: int, offset_dst: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    return np.stack([offset_src + idx, offset_dst + idx], axis=0)


def _build_edges(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    T_off = 0
    A_off = n
    edge_parts: List[np.ndarray] = []
    edge_types: List[np.ndarray] = []

    def add(edges: np.ndarray, etype: int):
        if edges.shape[1] > 0:
            edge_parts.append(edges)
            edge_types.append(np.full(edges.shape[1], etype, dtype=np.int64))

    add(_temporal_edges(n, T_off), 0)              # T→T
    add(_temporal_edges(n, A_off), 1)              # A→A
    add(_same_utt_edges(n, T_off, A_off), 2)       # T→A
    add(_same_utt_edges(n, A_off, T_off), 3)       # A→T

    if not edge_parts:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,),   dtype=torch.long))

    edge_index = torch.from_numpy(np.concatenate(edge_parts, axis=1))
    edge_type  = torch.from_numpy(np.concatenate(edge_types, axis=0))
    return edge_index, edge_type


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    pid: int,
    label: int,
    phq_score: float,
    phq8: np.ndarray,
    context_mode: str = _DEFAULT_CONTEXT_MODE,
    alignment_issues: Optional[List[dict]] = None,
) -> Optional[Data]:
    """
    Build a bimodal graph for one participant.

    FIXED:
      - n_utt from extraction metadata
      - Hard alignment validation (>2 mismatch → skip)
      - context_mode controls which text embedding file to load
    """
    n_groups = _get_n_groups(pid)
    if n_groups is None or n_groups == 0:
        warnings.warn(f"PID {pid}: cannot determine n_groups, skipping.")
        return None

    MAX_UTT = 512
    n_groups = min(n_groups, MAX_UTT)

    # Count features from actual files
    try:
        text_path = _get_text_feats_path(pid, context_mode)
    except FileNotFoundError as e:
        warnings.warn(str(e))
        return None

    audio_path = DAICWOZ_DATA_DIR / f"{pid}_audio_feats.npy"
    if not audio_path.exists():
        warnings.warn(f"PID {pid}: audio_feats.npy not found, skipping.")
        return None

    n_text  = np.load(str(text_path)).shape[0]
    n_audio = np.load(str(audio_path)).shape[0]

    # FIX-MAJOR: Hard alignment validation
    is_valid, n_utt, align_msg = _validate_alignment(
        pid, n_text, n_audio, n_groups, context_mode
    )

    if align_msg:
        warnings.warn(align_msg)
        if alignment_issues is not None:
            alignment_issues.append({
                "pid": pid, "n_text": n_text, "n_audio": n_audio,
                "n_groups": n_groups, "is_valid": is_valid,
                "effective_n_utt": n_utt, "message": align_msg,
            })

    if not is_valid:
        return None

    if n_utt == 0:
        warnings.warn(f"PID {pid}: n_utt resolved to 0, skipping.")
        return None

    # Load features
    text_feats  = _load_text_feats(pid, n_utt, context_mode)
    audio_feats = _load_audio_feats(pid, n_utt)

    text_acoustics  = _load_text_acoustics(pid, n_utt)
    audio_acoustics = _load_audio_acoustics(pid, n_utt)

    text_feats_aug  = np.concatenate([text_feats,  text_acoustics],  axis=1)
    audio_feats_aug = np.concatenate([audio_feats, audio_acoustics], axis=1)

    x_text = np.concatenate([
        text_feats_aug,
        np.zeros((n_utt, AUDIO_DIM_TOTAL), dtype=np.float32),
    ], axis=1)
    x_audio = np.concatenate([
        np.zeros((n_utt, TEXT_DIM_TOTAL), dtype=np.float32),
        audio_feats_aug,
    ], axis=1)
    x = np.concatenate([x_text, x_audio], axis=0)

    node_type = np.array([0]*n_utt + [1]*n_utt, dtype=np.int64)
    utt_pos   = np.linspace(0.0, 1.0, n_utt, dtype=np.float32)
    pos       = np.tile(utt_pos, 2)

    edge_index, edge_type = _build_edges(n_utt)

    return Data(
        x             = torch.from_numpy(x).float(),
        edge_index    = edge_index,
        edge_type     = edge_type,
        node_type     = torch.from_numpy(node_type),
        pos           = torch.from_numpy(pos),
        y             = torch.tensor([label],     dtype=torch.long),
        phq_score     = torch.tensor([phq_score], dtype=torch.float),
        phq8          = torch.from_numpy(phq8.astype(np.float32)),
        pid           = pid,
        n_utt         = n_utt,
        context_mode  = context_mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class DaicWozDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for DAIC-WOZ.

    Parameters
    ----------
    context_mode : str
        Which BERT embedding version to use for text nodes.
        "none"      → no interviewer context (anti-shortcut)
        "truncated" → truncated Ellie question (default)
        "full"      → full Ellie turn (max-shortcut baseline)
        Overrides DAICWOZ_CONTEXT_MODE env variable if set.
    """

    def __init__(
        self,
        split: str = "train",
        root: str = str(DAICWOZ_CACHE_DIR),
        context_mode: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        assert split in ("train", "dev", "test")
        self.split = split
        self._force_reload = force_reload

        # Resolve context mode: constructor > env var > default
        self.context_mode = context_mode if context_mode is not None else _DEFAULT_CONTEXT_MODE
        assert self.context_mode in ("none", "truncated", "full"), \
            f"Invalid context_mode: {self.context_mode!r}"

        # Include context_mode in the cache key so different modes get
        # separate cached files and don't overwrite each other.
        self._cache_subdir = os.path.join(root, f"ctx_{self.context_mode}")

        super().__init__(self._cache_subdir, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.split}_data.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        split_df = pd.read_csv(DAICWOZ_SPLIT_FILES[self.split])
        split_df.columns = split_df.columns.str.strip()

        if "PHQ8_Binary" in split_df.columns:
            label_col, score_col = "PHQ8_Binary", "PHQ8_Score"
        else:
            label_col, score_col = "PHQ_Binary", "PHQ_Score"

        phq8_dict: Dict[int, np.ndarray] = {}
        if all(c in split_df.columns for c in DAICWOZ_PHQ8_COLS):
            for _, row in split_df.iterrows():
                vals = row[DAICWOZ_PHQ8_COLS].values.astype(np.float32)
                vals = np.where(np.isnan(vals), 0.0, vals)
                phq8_dict[int(row["Participant_ID"])] = vals

        data_list = []
        skipped = 0
        alignment_issues: List[dict] = []

        for _, row in split_df.iterrows():
            pid       = int(row["Participant_ID"])
            label     = int(row[label_col])
            phq_score = float(row[score_col]) if score_col in split_df.columns else 0.0
            phq8      = phq8_dict.get(pid, np.zeros(8, dtype=np.float32))

            # Check required audio file (text file checked inside build_graph)
            if not (DAICWOZ_DATA_DIR / f"{pid}_audio_feats.npy").exists():
                warnings.warn(f"PID {pid}: missing audio_feats.npy, skipping.")
                skipped += 1
                continue

            try:
                graph = build_graph(
                    pid, label, phq_score, phq8,
                    context_mode=self.context_mode,
                    alignment_issues=alignment_issues,
                )
            except Exception as exc:
                warnings.warn(f"PID {pid}: graph build failed ({exc}), skipping.")
                skipped += 1
                continue

            if graph is None:
                skipped += 1
                continue

            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            data_list.append(graph)

        print(
            f"[DaicWozDataset] split={self.split} context_mode={self.context_mode!r}: "
            f"{len(data_list)} graphs built, {skipped} skipped."
        )

        # Write alignment report
        if alignment_issues:
            report_path = (
                Path(self._cache_subdir) /
                f"{self.split}_alignment_report.json"
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(alignment_issues, f, indent=2)
            n_critical = sum(1 for x in alignment_issues if not x["is_valid"])
            print(
                f"  [ALIGNMENT] {len(alignment_issues)} issues detected "
                f"({n_critical} critical → skipped). "
                f"Report: {report_path}"
            )

        self.save(data_list, self.processed_paths[0])

    @property
    def num_node_features(self) -> int:
        return UNIFIED_DIM

    @property
    def num_classes(self) -> int:
        return 2

    def pids(self) -> List[int]:
        return [d.pid for d in self]

    def labels(self) -> torch.Tensor:
        return torch.cat([d.y for d in self], dim=0)

    @property
    def class_weights(self) -> torch.Tensor:
        labels = torch.stack([d.y.squeeze() for d in self], dim=0).float()
        n_pos  = labels.sum().item()
        n_neg  = (1 - labels).sum().item()
        total  = n_pos + n_neg
        w_neg  = total / (2.0 * max(n_neg, 1))
        w_pos  = total / (2.0 * max(n_pos, 1))
        return torch.tensor([w_neg, w_pos], dtype=torch.float)


if __name__ == "__main__":
    for split in ("train", "dev", "test"):
        for mode in ("none", "truncated", "full"):
            try:
                ds = DaicWozDataset(split=split, context_mode=mode)
                print(f"  {split} / {mode}: {len(ds)} samples")
                if len(ds) > 0:
                    d = ds[0]
                    print(f"    x.shape={d.x.shape}, edges={d.edge_index.shape[1]}, "
                          f"y={d.y.item()}, n_utt={d.n_utt}, ctx={d.context_mode}")
            except Exception as e:
                print(f"  {split} / {mode}: FAILED — {e}")