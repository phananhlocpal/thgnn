"""
daicwoz_dataset.py

DaicWozDataset: PyG InMemoryDataset for the original DAIC-WOZ corpus.

Graph construction (per participant / interview):
  Node types:
    0 = Text  : N utterances, feature dim 768  (pre-extracted BERT)
    1 = Audio : N utterances, feature dim 148  (74 COVAREP × mean+std)
    2 = Video : N utterances, feature dim 40   (20 CLNF_AUs × mean+std)

  Unified feature tensor x: shape (3N, 956)
    text  node i    : x[i]     = [text_feat(768) | zeros(148) | zeros(40)]
    audio node N+i  : x[N+i]   = [zeros(768)     | audio_feat(148) | zeros(40)]
    video node 2N+i : x[2N+i]  = [zeros(768)     | zeros(148)  | video_feat(40)]

  Edge types (9): identical to DepressionDataset (E-DAIC).
    0 T→T  temporal window ±3
    1 A→A  temporal window ±3
    2 V→V  temporal window ±3
    3 T→A  same-utterance cross-modal
    4 A→T  same-utterance cross-modal
    5 T→V  same-utterance cross-modal
    6 V→T  same-utterance cross-modal
    7 A→V  same-utterance cross-modal
    8 V→A  same-utterance cross-modal

  Additional tensors:
    node_type : LongTensor (3N,)   values {0,1,2}
    pos       : FloatTensor (3N,)  normalised utterance index ∈ [0,1]
    y         : LongTensor (1,)    binary depression label
    phq_score : FloatTensor (1,)   PHQ total score
    phq8      : FloatTensor (8,)   per-item PHQ-8 subscores (zeros when unavailable)
    pid       : int                participant id
    n_utt     : int                number of utterances
"""

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

DAICWOZ_DATA_DIR = Path("C:/Users/ezycloudx-admin/Desktop/thgnn/daicwoz")
DAICWOZ_CACHE_DIR = Path("C:/Users/ezycloudx-admin/Desktop/thgnn/cache_daicwoz")

TEXT_DIM   = 768
AUDIO_DIM  = 148   # 74 COVAREP features × (mean + std)
VIDEO_DIM  = 40    # 20 CLNF_AU features × (mean + std)
UNIFIED_DIM = TEXT_DIM + AUDIO_DIM + VIDEO_DIM   # 956

COVAREP_N_FEATS = 74    # all 74 columns are features; time = row_index / 100.0
COVAREP_SAMPLE_RATE = 100.0   # Hz

CLNF_AU_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r",
    "AU25_r", "AU26_r",
    "AU04_c", "AU12_c", "AU15_c", "AU23_c", "AU28_c", "AU45_c",
]   # 20 features

TEMPORAL_WINDOW = 5   # wider window: participant turns are sparser (Ellie speaks in between)

# PHQ-8 per-item column names in train/dev split CSVs
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
# Low-level feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_transcript(pid: int) -> pd.DataFrame:
    """
    Load DAIC-WOZ transcript (tab-separated).
    Filters to Participant utterances only — text_feats.npy is pre-extracted
    for participant turns, so including Ellie would misalign features.
    Returns DataFrame with columns Start_Time / End_Time (float), index reset.
    """
    path = DAICWOZ_DATA_DIR / f"{pid}_TRANSCRIPT.csv"
    df = pd.read_csv(str(path), sep="\t", engine="python")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"start_time": "Start_Time", "stop_time": "End_Time"})
    df["Start_Time"] = pd.to_numeric(df["Start_Time"], errors="coerce").fillna(0.0)
    df["End_Time"]   = pd.to_numeric(df["End_Time"],   errors="coerce").fillna(0.0)
    # Keep only participant turns — case-insensitive to match extract_bert.py
    df = df[df["speaker"].str.strip().str.lower() == "participant"].reset_index(drop=True)
    return df


def _load_text_feats(pid: int, n_utt: int) -> np.ndarray:
    """
    Load pre-extracted BERT utterance features.
    Returns array shape (n_utt, 768).
    Truncates or zero-pads if file N differs from n_utt.
    """
    path = DAICWOZ_DATA_DIR / f"{pid}_text_feats.npy"
    feats = np.load(str(path)).astype(np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    M = feats.shape[0]
    if M == n_utt:
        return feats
    if M > n_utt:
        return feats[:n_utt]
    pad = np.zeros((n_utt - M, TEXT_DIM), dtype=np.float32)
    return np.concatenate([feats, pad], axis=0)


def _window_stats(
    values: np.ndarray,     # (F_frames, n_feats) — float32
    times:  np.ndarray,     # (F_frames,) — seconds
    t_start: float,
    t_end:   float,
    n_feats: int,
) -> np.ndarray:
    """
    Select frames within [t_start, t_end] and return mean+std concatenation.
    Returns zeros (2*n_feats,) when no frames match.
    """
    mask = (times >= t_start) & (times <= t_end)
    subset = values[mask]
    if subset.shape[0] == 0:
        return np.zeros(2 * n_feats, dtype=np.float32)
    subset = np.where(subset == -9999.0, np.nan, subset)   # COVAREP missing sentinel
    col_mean = np.nanmean(subset, axis=0)
    col_std  = np.nanstd(subset,  axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    col_std  = np.nan_to_num(col_std,  nan=0.0)
    return np.concatenate([col_mean, col_std]).astype(np.float32)


def _load_audio_feats(pid: int, transcript: pd.DataFrame) -> np.ndarray:
    """
    Load COVAREP features (no header, 74 cols, 100 Hz).
    Returns array shape (n_utt, 148).
    """
    path = DAICWOZ_DATA_DIR / f"{pid}_COVAREP.csv"
    raw = pd.read_csv(str(path), header=None, dtype=np.float32).values   # (F, 74)
    n_frames = raw.shape[0]
    times = np.arange(n_frames, dtype=np.float32) / COVAREP_SAMPLE_RATE

    n_utt = len(transcript)
    feats = np.zeros((n_utt, AUDIO_DIM), dtype=np.float32)
    for i, (_, row) in enumerate(transcript.iterrows()):
        feats[i] = _window_stats(
            raw, times, float(row["Start_Time"]), float(row["End_Time"]),
            COVAREP_N_FEATS,
        )
    return feats


def _load_video_feats(pid: int, transcript: pd.DataFrame) -> np.ndarray:
    """
    Load CLNF_AUs features.
    Returns array shape (n_utt, 40).
    """
    path = DAICWOZ_DATA_DIR / f"{pid}_CLNF_AUs.txt"
    df = pd.read_csv(str(path))
    df.columns = df.columns.str.strip()
    # Keep only AU feature columns that exist in this file
    available = [c for c in CLNF_AU_COLS if c in df.columns]
    for c in available:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    raw   = df[available].values.astype(np.float32)   # (F, ≤20)
    times = df["timestamp"].values.astype(np.float32)

    n_avail = raw.shape[1]
    n_utt = len(transcript)
    # Output is always VIDEO_DIM=40; zero-pad missing features
    feats = np.zeros((n_utt, VIDEO_DIM), dtype=np.float32)
    for i, (_, row) in enumerate(transcript.iterrows()):
        stat = _window_stats(
            raw, times, float(row["Start_Time"]), float(row["End_Time"]),
            n_avail,
        )   # (2*n_avail,)
        feats[i, : len(stat)] = stat
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Edge construction (same topology as E-DAIC dataset)
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
    T_off, A_off, V_off = 0, n, 2 * n
    edge_parts: List[np.ndarray] = []
    edge_types: List[np.ndarray] = []

    def add(edges: np.ndarray, etype: int):
        if edges.shape[1] > 0:
            edge_parts.append(edges)
            edge_types.append(np.full(edges.shape[1], etype, dtype=np.int64))

    add(_temporal_edges(n, T_off), 0)
    add(_temporal_edges(n, A_off), 1)
    add(_temporal_edges(n, V_off), 2)
    add(_same_utt_edges(n, T_off, A_off), 3)
    add(_same_utt_edges(n, A_off, T_off), 4)
    add(_same_utt_edges(n, T_off, V_off), 5)
    add(_same_utt_edges(n, V_off, T_off), 6)
    add(_same_utt_edges(n, A_off, V_off), 7)
    add(_same_utt_edges(n, V_off, A_off), 8)

    if not edge_parts:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,),   dtype=torch.long))

    edge_index = torch.from_numpy(np.concatenate(edge_parts, axis=1))
    edge_type  = torch.from_numpy(np.concatenate(edge_types, axis=0))
    return edge_index, edge_type


# ─────────────────────────────────────────────────────────────────────────────
# Per-participant graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    pid: int,
    label: int,
    phq_score: float,
    phq8: np.ndarray,   # shape (8,)
) -> Optional[Data]:
    transcript = _load_transcript(pid)
    n_utt = len(transcript)
    if n_utt == 0:
        warnings.warn(f"PID {pid}: empty transcript, skipping.")
        return None

    MAX_UTT = 512
    if n_utt > MAX_UTT:
        transcript = transcript.iloc[:MAX_UTT].reset_index(drop=True)
        n_utt = MAX_UTT

    text_feats  = _load_text_feats(pid, n_utt)    # (n_utt, 768)
    audio_feats = _load_audio_feats(pid, transcript)   # (n_utt, 148)
    video_feats = _load_video_feats(pid, transcript)   # (n_utt, 40)

    # Unified feature matrix (3n, 956)
    x_text = np.concatenate([
        text_feats,
        np.zeros((n_utt, AUDIO_DIM + VIDEO_DIM), dtype=np.float32),
    ], axis=1)
    x_audio = np.concatenate([
        np.zeros((n_utt, TEXT_DIM),  dtype=np.float32),
        audio_feats,
        np.zeros((n_utt, VIDEO_DIM), dtype=np.float32),
    ], axis=1)
    x_video = np.concatenate([
        np.zeros((n_utt, TEXT_DIM + AUDIO_DIM), dtype=np.float32),
        video_feats,
    ], axis=1)
    x = np.concatenate([x_text, x_audio, x_video], axis=0)   # (3n, 956)

    node_type = np.array([0]*n_utt + [1]*n_utt + [2]*n_utt, dtype=np.int64)
    utt_pos   = np.linspace(0.0, 1.0, n_utt, dtype=np.float32)
    pos       = np.tile(utt_pos, 3)

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
    """
    PyG InMemoryDataset for the original DAIC-WOZ corpus.

    Parameters
    ----------
    split : str
        One of 'train', 'dev', 'test'.
    root : str | Path
        Directory where processed cache files are saved.
    force_reload : bool
        If True, re-process even if cache exists.
    """

    def __init__(
        self,
        split: str = "train",
        root: str = str(DAICWOZ_CACHE_DIR),
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        assert split in ("train", "dev", "test"), \
            f"split must be 'train'/'dev'/'test', got '{split}'"
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

        # Determine label / score column names (differ between train/dev and test)
        if "PHQ8_Binary" in split_df.columns:
            label_col = "PHQ8_Binary"
            score_col = "PHQ8_Score"
        else:
            label_col = "PHQ_Binary"
            score_col = "PHQ_Score"

        # Build PHQ8 per-item dict from split CSV when available
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
            for fname in [f"{pid}_text_feats.npy", f"{pid}_TRANSCRIPT.csv",
                          f"{pid}_COVAREP.csv", f"{pid}_CLNF_AUs.txt"]:
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

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def num_node_features(self) -> int:
        return UNIFIED_DIM   # 956

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


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for split in ("train", "dev", "test"):
        ds = DaicWozDataset(split=split)
        print(f"  {split}: {len(ds)} samples")
        if len(ds) > 0:
            d = ds[0]
            print(f"    x.shape={d.x.shape}, edges={d.edge_index.shape[1]}, "
                  f"y={d.y.item()}, phq_score={d.phq_score.item():.1f}, "
                  f"phq8={d.phq8.tolist()}")
