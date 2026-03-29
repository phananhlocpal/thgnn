"""
dataset.py

DepressionDataset: PyG InMemoryDataset for DAIC-WOZ / E-DAIC.

Graph construction (per participant / interview):
  Node types:
    0 = Text  : N utterances,  feature dim 768
    1 = Audio : N utterances,  feature dim 46  (mean+std of 23 eGeMAPS features)
    2 = Video : N utterances,  feature dim 98  (mean+std of 49 OpenFace features)

  Unified feature tensor x: shape (3N, 912)
    text  node i      : x[i]     = [text_feat(768) | zeros(46)  | zeros(98)]
    audio node N+i    : x[N+i]   = [zeros(768)     | audio_feat(46) | zeros(98)]
    video node 2N+i   : x[2N+i]  = [zeros(768)     | zeros(46)  | video_feat(98)]

  Edge types (9):
    0 T→T  temporal window ±3
    1 A→A  temporal window ±3
    2 V→V  temporal window ±3
    3 T→A  same-utterance cross-modal
    4 A→T  same-utterance cross-modal
    5 T→V  same-utterance cross-modal
    6 V→T  same-utterance cross-modal
    7 A→V  same-utterance cross-modal
    8 V→A  same-utterance cross-modal

  Additional tensors stored in Data:
    node_type : LongTensor (3N,)   values {0,1,2}
    pos       : FloatTensor (3N,)  normalised utterance index ∈ [0,1]
    y         : LongTensor (1,)    binary depression label
    phq_score : FloatTensor (1,)   PHQ total score
    phq8      : FloatTensor (8,)   per-item PHQ-8 subscores (NaN → 0)
    pid       : int                participant id (for debugging)
"""

import os
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("C:/Users/ezycloudx-admin/Desktop/thgnn/edaic_final")
CACHE_DIR  = Path("C:/Users/ezycloudx-admin/Desktop/thgnn/cache")

TEXT_DIM   = 768
AUDIO_DIM  = 46   # 23 features × (mean + std)
VIDEO_DIM  = 98   # 49 features × (mean + std)
UNIFIED_DIM = TEXT_DIM + AUDIO_DIM + VIDEO_DIM  # 912

TEMPORAL_WINDOW = 3   # ±k utterance neighbours for intra-modal temporal edges

AUDIO_FEAT_COLS: Optional[List[str]] = None   # resolved lazily
VIDEO_FEAT_COLS: Optional[List[str]] = None   # resolved lazily

PHQ8_COLS = [
    "PHQ_8NoInterest", "PHQ_8Depressed", "PHQ_8Sleep", "PHQ_8Tired",
    "PHQ_8Appetite", "PHQ_8Failure", "PHQ_8Concentrating", "PHQ_8Moving",
]

SPLIT_FILES = {
    "train": DATA_DIR / "train_split.csv",
    "dev":   DATA_DIR / "dev_split.csv",
    "test":  DATA_DIR / "test_split.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _audio_feature_cols() -> List[str]:
    """Return the 23 eGeMAPS feature column names (constant across files)."""
    global AUDIO_FEAT_COLS
    if AUDIO_FEAT_COLS is None:
        AUDIO_FEAT_COLS = [
            "Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3",
            "slope0-500_sma3", "slope500-1500_sma3", "spectralFlux_sma3",
            "mfcc1_sma3", "mfcc2_sma3", "mfcc3_sma3", "mfcc4_sma3",
            "F0semitoneFrom27.5Hz_sma3nz", "jitterLocal_sma3nz",
            "shimmerLocaldB_sma3nz", "HNRdBACF_sma3nz",
            "logRelF0-H1-H2_sma3nz", "logRelF0-H1-A3_sma3nz",
            "F1frequency_sma3nz", "F1bandwidth_sma3nz",
            "F1amplitudeLogRelF0_sma3nz", "F2frequency_sma3nz",
            "F2amplitudeLogRelF0_sma3nz", "F3frequency_sma3nz",
            "F3amplitudeLogRelF0_sma3nz",
        ]
    return AUDIO_FEAT_COLS


def _video_feature_cols() -> List[str]:
    """Return the 49 OpenFace feature column names (cols 4..)."""
    global VIDEO_FEAT_COLS
    if VIDEO_FEAT_COLS is None:
        VIDEO_FEAT_COLS = [
            "pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz",
            "gaze_0_x", "gaze_0_y", "gaze_0_z",
            "gaze_1_x", "gaze_1_y", "gaze_1_z",
            "gaze_angle_x", "gaze_angle_y",
            "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
            "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
            "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
            "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
            "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
            "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
        ]
    return VIDEO_FEAT_COLS


def _load_transcript(pid: int) -> pd.DataFrame:
    """Load utterance transcript, returning DataFrame with Start_Time / End_Time."""
    path = DATA_DIR / f"{pid}_Transcript.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Ensure numeric times
    df["Start_Time"] = pd.to_numeric(df["Start_Time"], errors="coerce").fillna(0.0)
    df["End_Time"]   = pd.to_numeric(df["End_Time"],   errors="coerce").fillna(0.0)
    return df


def _load_text_feats(pid: int, n_utt: int) -> np.ndarray:
    """
    Load pre-extracted BERT utterance features.
    Returns array shape (n_utt, 768).
    If file has different N, we truncate or zero-pad to n_utt.
    """
    path = DATA_DIR / f"{pid}_text_feats.npy"
    feats = np.load(str(path)).astype(np.float32)   # (M, 768)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    M = feats.shape[0]
    if M == n_utt:
        return feats
    if M > n_utt:
        return feats[:n_utt]
    # M < n_utt: zero-pad
    pad = np.zeros((n_utt - M, TEXT_DIM), dtype=np.float32)
    return np.concatenate([feats, pad], axis=0)


def _window_stats(df_frames: pd.DataFrame,
                  feat_cols: List[str],
                  t_start: float,
                  t_end: float) -> Optional[np.ndarray]:
    """
    Select rows whose frame time falls in [t_start, t_end].
    Return mean+std concatenated → shape (2*len(feat_cols),).
    Returns None if no frames are found (caller should use zeros).
    """
    mask = (df_frames["frameTime"] >= t_start) & (df_frames["frameTime"] <= t_end)
    subset = df_frames.loc[mask, feat_cols].values.astype(np.float32)
    if subset.shape[0] == 0:
        return None
    # Replace sentinel -201 with NaN, then fill with 0
    subset = np.where(subset == -201.0, np.nan, subset)
    col_mean = np.nanmean(subset, axis=0)
    col_std  = np.nanstd(subset,  axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    col_std  = np.nan_to_num(col_std,  nan=0.0)
    return np.concatenate([col_mean, col_std], axis=0)


def _load_audio_feats(pid: int, transcript: pd.DataFrame) -> np.ndarray:
    """
    Load eGeMAPS frame-level features and aggregate per utterance window.
    Returns array shape (n_utt, 46).
    """
    path = DATA_DIR / f"{pid}_OpenSMILE2.3.0_egemaps.csv"
    df = pd.read_csv(str(path), sep=";")
    df.columns = df.columns.str.strip()
    feat_cols = _audio_feature_cols()
    # Convert numeric, replace bad values
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    n_utt = len(transcript)
    feats = np.zeros((n_utt, AUDIO_DIM), dtype=np.float32)
    for i, row in transcript.iterrows():
        result = _window_stats(df, feat_cols, float(row["Start_Time"]), float(row["End_Time"]))
        if result is not None:
            feats[i] = result
    return feats


def _load_video_feats(pid: int, transcript: pd.DataFrame) -> np.ndarray:
    """
    Load OpenFace frame-level features and aggregate per utterance window.
    Returns array shape (n_utt, 98).
    OpenFace timestamps are in the 'timestamp' column (seconds).
    """
    path = DATA_DIR / f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"
    df = pd.read_csv(str(path))
    df.columns = df.columns.str.strip()
    feat_cols = _video_feature_cols()
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # Rename 'timestamp' to 'frameTime' for reuse of _window_stats
    df = df.rename(columns={"timestamp": "frameTime"})

    n_utt = len(transcript)
    feats = np.zeros((n_utt, VIDEO_DIM), dtype=np.float32)
    for i, row in transcript.iterrows():
        result = _window_stats(df, feat_cols, float(row["Start_Time"]), float(row["End_Time"]))
        if result is not None:
            feats[i] = result
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Edge construction
# ─────────────────────────────────────────────────────────────────────────────

def _temporal_edges(n: int, offset: int, win: int = TEMPORAL_WINDOW) -> np.ndarray:
    """
    Build temporal edges for a block of `n` nodes that starts at `offset`.
    Window ±`win`: connect node (offset+i) → (offset+j) for |i-j| ≤ win, i≠j.
    Returns edge array shape (2, E).
    """
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
    """
    Build same-utterance cross-modal edges: (offset_src+i) → (offset_dst+i)
    for i in 0..n-1. Returns (2, n).
    """
    idx = np.arange(n, dtype=np.int64)
    src = offset_src + idx
    dst = offset_dst + idx
    return np.stack([src, dst], axis=0)


def _build_edges(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build all 9 edge types for a graph with n utterances (3n nodes total).

    Node layout: [0..n-1] = text, [n..2n-1] = audio, [2n..3n-1] = video

    Returns:
        edge_index : LongTensor (2, E_total)
        edge_type  : LongTensor (E_total,)
    """
    T_off = 0
    A_off = n
    V_off = 2 * n

    edge_parts  = []
    edge_types  = []

    def add(edges: np.ndarray, etype: int):
        if edges.shape[1] > 0:
            edge_parts.append(edges)
            edge_types.append(np.full(edges.shape[1], etype, dtype=np.int64))

    # Intra-modal temporal
    add(_temporal_edges(n, T_off), 0)   # T→T
    add(_temporal_edges(n, A_off), 1)   # A→A
    add(_temporal_edges(n, V_off), 2)   # V→V

    # Cross-modal same-utterance (bidirectional pairs)
    add(_same_utt_edges(n, T_off, A_off), 3)   # T→A
    add(_same_utt_edges(n, A_off, T_off), 4)   # A→T
    add(_same_utt_edges(n, T_off, V_off), 5)   # T→V
    add(_same_utt_edges(n, V_off, T_off), 6)   # V→T
    add(_same_utt_edges(n, A_off, V_off), 7)   # A→V
    add(_same_utt_edges(n, V_off, A_off), 8)   # V→A

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
    phq8: np.ndarray,         # shape (8,) – per-item scores
    phq8_labels: Optional[pd.DataFrame] = None,
) -> Data:
    """
    Build a single PyG Data object for one participant.
    """
    # ── Transcript ────────────────────────────────────────────────────────────
    transcript = _load_transcript(pid)
    n_utt = len(transcript)
    if n_utt == 0:
        warnings.warn(f"PID {pid}: empty transcript, skipping.")
        return None

    # Clamp utterance count to protect against pathological transcripts
    MAX_UTT = 512
    if n_utt > MAX_UTT:
        transcript = transcript.iloc[:MAX_UTT].reset_index(drop=True)
        n_utt = MAX_UTT

    # ── Raw per-modality features ─────────────────────────────────────────────
    text_feats  = _load_text_feats(pid, n_utt)   # (n_utt, 768)
    audio_feats = _load_audio_feats(pid, transcript)  # (n_utt, 46)
    video_feats = _load_video_feats(pid, transcript)  # (n_utt, 98)

    # ── Unified feature matrix (3n, 912) ─────────────────────────────────────
    zeros_av   = np.zeros((n_utt, AUDIO_DIM + VIDEO_DIM), dtype=np.float32)
    zeros_tv   = np.zeros((n_utt, TEXT_DIM + VIDEO_DIM),  dtype=np.float32)
    zeros_ta   = np.zeros((n_utt, TEXT_DIM + AUDIO_DIM),  dtype=np.float32)

    # text  row: [text_feat(768) | zeros(46+98)]
    x_text  = np.concatenate([text_feats,  zeros_av], axis=1)
    # audio row: [zeros(768) | audio_feat(46) | zeros(98)]
    x_audio = np.concatenate([
        np.zeros((n_utt, TEXT_DIM),  dtype=np.float32),
        audio_feats,
        np.zeros((n_utt, VIDEO_DIM), dtype=np.float32),
    ], axis=1)
    # video row: [zeros(768+46) | video_feat(98)]
    x_video = np.concatenate([zeros_ta, video_feats], axis=1)

    x = np.concatenate([x_text, x_audio, x_video], axis=0)  # (3n, 912)

    # ── Node metadata ─────────────────────────────────────────────────────────
    node_type = np.array(
        [0] * n_utt + [1] * n_utt + [2] * n_utt, dtype=np.int64
    )
    utt_pos = np.linspace(0.0, 1.0, n_utt, dtype=np.float32)
    pos     = np.tile(utt_pos, 3)  # repeat for text / audio / video blocks

    # ── Edges ─────────────────────────────────────────────────────────────────
    edge_index, edge_type = _build_edges(n_utt)

    # ── Assemble PyG Data ─────────────────────────────────────────────────────
    data = Data(
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
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class DepressionDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for DAIC-WOZ / E-DAIC.

    Parameters
    ----------
    split : str
        One of 'train', 'dev', 'test'.
    root : str | Path
        Directory where processed cache files are saved.
    transform / pre_transform : callable, optional
        Standard PyG transforms.
    force_reload : bool
        If True, re-process even if cache exists.
    """

    def __init__(
        self,
        split: str = "train",
        root: str = str(CACHE_DIR),
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        assert split in ("train", "dev", "test"), \
            f"split must be one of 'train'/'dev'/'test', got '{split}'"
        self.split = split
        self._force_reload = force_reload
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    # ── PyG hooks ─────────────────────────────────────────────────────────────

    @property
    def raw_file_names(self) -> List[str]:
        return []   # we read directly from DATA_DIR

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.split}_data.pt"]

    def download(self) -> None:
        pass  # data already on disk

    def process(self) -> None:
        """Build graphs for every participant in this split and save."""
        split_df  = pd.read_csv(SPLIT_FILES[self.split])
        phq8_df   = pd.read_csv(DATA_DIR / "Detailed_PHQ8_Labels.csv")
        phq8_dict: Dict[int, np.ndarray] = {}
        for _, row in phq8_df.iterrows():
            vals = row[PHQ8_COLS].values.astype(np.float32)
            vals = np.where(np.isnan(vals), 0.0, vals)
            phq8_dict[int(row["Participant_ID"])] = vals

        data_list = []
        skipped   = 0

        for _, row in split_df.iterrows():
            pid       = int(row["Participant_ID"])
            label     = int(row["PHQ_Binary"])
            phq_score = float(row["PHQ_Score"])
            phq8      = phq8_dict.get(pid, np.zeros(8, dtype=np.float32))

            # Check that the required files exist
            if not (DATA_DIR / f"{pid}_text_feats.npy").exists():
                warnings.warn(f"PID {pid}: missing text_feats.npy, skipping.")
                skipped += 1
                continue
            if not (DATA_DIR / f"{pid}_Transcript.csv").exists():
                warnings.warn(f"PID {pid}: missing Transcript.csv, skipping.")
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

        print(f"[DepressionDataset] split={self.split}: "
              f"{len(data_list)} graphs built, {skipped} skipped.")

        self.save(data_list, self.processed_paths[0])

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def num_node_features(self) -> int:
        return UNIFIED_DIM  # 912

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for BCE loss."""
        labels = torch.stack([d.y.squeeze() for d in self], dim=0).float()
        n_pos  = labels.sum().item()
        n_neg  = (1 - labels).sum().item()
        total  = n_pos + n_neg
        w_neg  = total / (2.0 * max(n_neg, 1))
        w_pos  = total / (2.0 * max(n_pos, 1))
        return torch.tensor([w_neg, w_pos], dtype=torch.float)

    def pids(self) -> List[int]:
        return [d.pid for d in self]

    def labels(self) -> torch.Tensor:
        return torch.cat([d.y for d in self], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for split in ("train", "dev", "test"):
        ds = DepressionDataset(split=split)
        print(f"  {split}: {len(ds)} samples")
        if len(ds) > 0:
            d = ds[0]
            print(f"    x.shape={d.x.shape}, edge_index.shape={d.edge_index.shape}, "
                  f"n_edge_types={d.edge_type.unique().numel()}, "
                  f"node_type={d.node_type.unique().tolist()}, "
                  f"y={d.y.item()}, phq_score={d.phq_score.item():.1f}, "
                  f"phq8={d.phq8.tolist()}")
