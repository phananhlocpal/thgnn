"""
daicwoz_dataset.py (FIXED)

DaicWozDataset: PyG InMemoryDataset for the original DAIC-WOZ corpus.

Fixes vs original:
  1. [FIX-CRITICAL] n_utt now reads from text_feats_meta.json (or n_groups.txt)
     instead of counting raw participant rows in the transcript.
     Previously: n_utt = number of raw participant rows (e.g. 119)
     Now:        n_utt = number of utterance groups from extraction (e.g. 60)
     This was causing ~50% of nodes to be zero-padded garbage because
     _load_text_feats would pad 61-row embeddings to 119 rows.
  2. [FIX-MAJOR] Removed _load_transcript dependency for node counting.
     The transcript is no longer the source of truth for graph size —
     the pre-extracted features are.
  3. [FIX] Added validation: if text and audio feature counts don't match,
     use the minimum and log a warning (instead of silently misaligning).

All other graph construction logic (edge types, unified features, etc.) preserved.
"""

import os
import warnings
import json
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

# Per-utterance prosodic/acoustic features extracted from metadata JSONs
TEXT_ACOUSTIC_DIM  = 8   # from text_feats_meta.json (speech rate, latency, para-ling events)
AUDIO_ACOUSTIC_DIM = 8   # from audio_feats_meta.json (pauses, pitch, energy)

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


# ─────────────────────────────────────────────────────────────────────────────
# Feature loading — now driven by pre-extracted features, not transcript
# ─────────────────────────────────────────────────────────────────────────────

def _get_n_groups(pid: int) -> Optional[int]:
    """
    Get the number of utterance groups from the extraction pipeline.
    Priority: text_feats_meta.json > n_groups.txt > text_feats.npy shape.
    """
    # Method 1: Read from metadata JSON (most reliable)
    meta_path = DAICWOZ_DATA_DIR / f"{pid}_text_feats_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return len(meta)

    # Method 2: Read from side file
    ngroups_path = DAICWOZ_DATA_DIR / f"{pid}_n_groups.txt"
    if ngroups_path.exists():
        with open(ngroups_path, "r") as f:
            return int(f.read().strip())

    # Method 3: Infer from npy shape
    npy_path = DAICWOZ_DATA_DIR / f"{pid}_text_feats.npy"
    if npy_path.exists():
        feats = np.load(str(npy_path))
        if feats.ndim == 2:
            return feats.shape[0]
        return 1

    return None


def _load_feats(pid: int, n_utt: int, modality: str) -> np.ndarray:
    """
    Load pre-extracted features. Modality is 'text' or 'audio'.
    Returns (n_utt, 768). Truncates or zero-pads if needed.
    """
    dim = TEXT_DIM if modality == "text" else AUDIO_DIM
    fname = f"{pid}_{modality}_feats.npy"
    path = DAICWOZ_DATA_DIR / fname

    feats = np.load(str(path)).astype(np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    M = feats.shape[0]
    if M == n_utt:
        return feats
    if M > n_utt:
        return feats[:n_utt]

    # Padding needed — this should be rare with the fix
    if M < n_utt:
        warnings.warn(
            f"PID {pid}: {modality}_feats has {M} rows but expected {n_utt}. "
            f"Padding with zeros."
        )
    pad = np.zeros((n_utt - M, dim), dtype=np.float32)
    return np.concatenate([feats, pad], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Prosodic / acoustic feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_acoustics(meta: dict) -> np.ndarray:
    """
    Extract 8 normalised prosodic features from a text_feats_meta.json entry.
    Fields come from extract_bert_daicwoz.py.
    """
    sr  = float(meta.get("speech_rate_wps", 0.0) or 0.0) / 5.0          # ~[0,2]
    lat = float(meta.get("response_latency_sec", 0.0) or 0.0) / 10.0    # ~[0,1]
    sigh   = float(bool(meta.get("has_sigh",        False)))
    laugh  = float(bool(meta.get("has_laughter",    False)))
    breath = float(bool(meta.get("has_breath",      False)))
    cry    = float(bool(meta.get("has_cry",         False)))
    cough  = float(bool(meta.get("has_cough",       False)))
    other  = float(bool(meta.get("has_other_sound", False)))
    return np.array([sr, lat, sigh, laugh, breath, cry, cough, other], dtype=np.float32)


def _extract_audio_acoustics(meta: dict) -> np.ndarray:
    """
    Extract 8 normalised acoustic features from an audio_feats_meta.json entry.
    Fields come from extract_wav2vec_daicwoz.py.
    """
    mean_pause  = float(meta.get("mean_pause_sec", 0.0) or 0.0) / 5.0   # ~[0,1]
    max_pause   = float(meta.get("max_pause_sec",  0.0) or 0.0) / 10.0
    n_pauses    = float(meta.get("n_pauses",        0.0) or 0.0) / 10.0
    speech_rat  = float(meta.get("speech_ratio",    0.5) or 0.5)         # already [0,1]
    mean_pitch  = (float(meta.get("mean_pitch_hz",  200.0) or 200.0) - 50.0) / 450.0  # [0,1]
    std_pitch   = float(meta.get("std_pitch_hz",    0.0) or 0.0) / 100.0
    mean_nrg    = float(meta.get("mean_energy",     0.0) or 0.0) / 0.2
    std_nrg     = float(meta.get("std_energy",      0.0) or 0.0) / 0.1
    return np.array(
        [mean_pause, max_pause, n_pauses, speech_rat, mean_pitch, std_pitch, mean_nrg, std_nrg],
        dtype=np.float32,
    )


def _load_text_acoustics(pid: int, n_utt: int) -> np.ndarray:
    """
    Load per-utterance prosodic features from {pid}_text_feats_meta.json.
    Returns (n_utt, TEXT_ACOUSTIC_DIM).  Falls back to zeros if file missing.
    """
    path = DAICWOZ_DATA_DIR / f"{pid}_text_feats_meta.json"
    zeros = np.zeros((n_utt, TEXT_ACOUSTIC_DIM), dtype=np.float32)
    if not path.exists():
        return zeros
    with open(path, "r") as f:
        meta_list = json.load(f)
    rows = [_extract_text_acoustics(m) for m in meta_list]
    arr = np.stack(rows, axis=0) if rows else zeros
    # Align length
    M = arr.shape[0]
    if M >= n_utt:
        return arr[:n_utt]
    pad = np.zeros((n_utt - M, TEXT_ACOUSTIC_DIM), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def _load_audio_acoustics(pid: int, n_utt: int) -> np.ndarray:
    """
    Load per-utterance acoustic features from {pid}_audio_feats_meta.json.
    Returns (n_utt, AUDIO_ACOUSTIC_DIM).  Falls back to zeros if file missing.
    """
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

    add(_temporal_edges(n, T_off), 0)   # T→T
    add(_temporal_edges(n, A_off), 1)   # A→A
    add(_same_utt_edges(n, T_off, A_off), 2)  # T→A
    add(_same_utt_edges(n, A_off, T_off), 3)  # A→T

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
) -> Optional[Data]:
    """
    Build a bimodal graph for one participant.

    FIXED: n_utt is now determined by the pre-extracted feature files,
    not by counting raw transcript rows.
    """
    # FIX: Get n_utt from extraction metadata, not transcript
    n_utt = _get_n_groups(pid)
    if n_utt is None or n_utt == 0:
        warnings.warn(f"PID {pid}: cannot determine n_groups, skipping.")
        return None

    MAX_UTT = 512
    if n_utt > MAX_UTT:
        n_utt = MAX_UTT

    # Validate text and audio alignment
    text_path  = DAICWOZ_DATA_DIR / f"{pid}_text_feats.npy"
    audio_path = DAICWOZ_DATA_DIR / f"{pid}_audio_feats.npy"
    n_text  = np.load(str(text_path)).shape[0]  if text_path.exists()  else 0
    n_audio = np.load(str(audio_path)).shape[0] if audio_path.exists() else 0

    if n_text != n_audio:
        warnings.warn(
            f"PID {pid}: text has {n_text} groups, audio has {n_audio}. "
            f"Using min={min(n_text, n_audio)}. Re-run extraction pipelines "
            f"to ensure alignment."
        )
        n_utt = min(n_text, n_audio, n_utt)

    if n_utt == 0:
        warnings.warn(f"PID {pid}: no valid utterances, skipping.")
        return None

    text_feats  = _load_feats(pid, n_utt, "text")    # (n_utt, 768)
    audio_feats = _load_feats(pid, n_utt, "audio")   # (n_utt, 768)

    # Prosodic / acoustic features from metadata JSONs
    text_acoustics  = _load_text_acoustics(pid, n_utt)   # (n_utt, 8)
    audio_acoustics = _load_audio_acoustics(pid, n_utt)  # (n_utt, 8)

    # Augmented per-modality features
    text_feats_aug  = np.concatenate([text_feats,  text_acoustics],  axis=1)  # (n_utt, 776)
    audio_feats_aug = np.concatenate([audio_feats, audio_acoustics], axis=1)  # (n_utt, 776)

    # Unified feature matrix (2n, 1552)
    # Text nodes:  [text_aug | zeros_audio_dim]
    # Audio nodes: [zeros_text_dim | audio_aug]
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
        x          = torch.from_numpy(x).float(),
        edge_index = edge_index,
        edge_type  = edge_type,
        node_type  = torch.from_numpy(node_type),
        pos        = torch.from_numpy(pos),
        y          = torch.tensor([label],     dtype=torch.long),
        phq_score  = torch.tensor([phq_score], dtype=torch.float),
        phq8       = torch.from_numpy(phq8.astype(np.float32)),
        pid        = pid,
        n_utt      = n_utt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class DaicWozDataset(InMemoryDataset):

    def __init__(
        self,
        split: str = "train",
        root: str = str(DAICWOZ_CACHE_DIR),
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        assert split in ("train", "dev", "test")
        self.split = split
        self._force_reload = force_reload
        super().__init__(root, transform, pre_transform)
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

        for _, row in split_df.iterrows():
            pid       = int(row["Participant_ID"])
            label     = int(row[label_col])
            phq_score = float(row[score_col]) if score_col in split_df.columns else 0.0
            phq8      = phq8_dict.get(pid, np.zeros(8, dtype=np.float32))

            # Check required files
            missing = False
            for fname in [f"{pid}_text_feats.npy", f"{pid}_audio_feats.npy"]:
                if not (DAICWOZ_DATA_DIR / fname).exists():
                    warnings.warn(f"PID {pid}: missing {fname}, skipping.")
                    missing = True
                    break
            if missing:
                skipped += 1
                continue

            try:
                graph = build_graph(pid, label, phq_score, phq8)
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

        print(f"[DaicWozDataset] split={self.split}: "
              f"{len(data_list)} graphs built, {skipped} skipped.")

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
        ds = DaicWozDataset(split=split)
        print(f"  {split}: {len(ds)} samples")
        if len(ds) > 0:
            d = ds[0]
            print(f"    x.shape={d.x.shape}, edges={d.edge_index.shape[1]}, "
                  f"y={d.y.item()}, n_utt={d.n_utt}")