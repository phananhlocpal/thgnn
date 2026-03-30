"""
edaic_dataset.py

DepressionDataset: PyG InMemoryDataset for E-DAIC.

Two-modal variant: Text (BERT) + Audio (wav2vec).

Graph construction (per participant / interview):
  Node types:
    0 = Text  : N utterances,  feature dim 768  (pre-extracted BERT)
    1 = Audio : N utterances,  feature dim 768  (pre-extracted wav2vec)

  Unified feature tensor x: shape (2N, 1536)
    text  node i  : x[i]    = [text_feat(768)  | zeros(768)]
    audio node N+i: x[N+i]  = [zeros(768)      | audio_feat(768)]

  Edge types (4):
    0 T→T  temporal window ±TEMPORAL_WINDOW
    1 A→A  temporal window ±TEMPORAL_WINDOW
    2 T→A  same-utterance cross-modal
    3 A→T  same-utterance cross-modal

  Additional tensors stored in Data:
    node_type : LongTensor (2N,)   values {0,1}
    pos       : FloatTensor (2N,)  normalised utterance index ∈ [0,1]
    y         : LongTensor (1,)    binary depression label
    phq_score : FloatTensor (1,)   PHQ total score
    phq8      : FloatTensor (8,)   per-item PHQ-8 subscores (NaN → 0)
    pid       : int                participant id (for debugging)

Audio features are expected as pre-extracted wav2vec embeddings saved as
{pid}_audio_feats.npy with shape (N_utts, 768).
"""

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

DATA_DIR   = Path("C:/Users/ezycloudx-admin/Desktop/thgnn/edaic_final")
CACHE_DIR  = Path("C:/Users/ezycloudx-admin/Desktop/thgnn/cache")

TEXT_DIM    = 768   # BERT
AUDIO_DIM   = 768   # wav2vec2
UNIFIED_DIM = TEXT_DIM + AUDIO_DIM  # 1536

TEMPORAL_WINDOW = 3   # ±k utterance neighbours for intra-modal temporal edges

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

def _load_transcript(pid: int) -> pd.DataFrame:
    """Load utterance transcript, returning DataFrame with Start_Time / End_Time."""
    path = DATA_DIR / f"{pid}_Transcript.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Start_Time"] = pd.to_numeric(df["Start_Time"], errors="coerce").fillna(0.0)
    df["End_Time"]   = pd.to_numeric(df["End_Time"],   errors="coerce").fillna(0.0)
    return df


def _load_text_feats(pid: int, n_utt: int) -> np.ndarray:
    """
    Load pre-extracted BERT utterance features.
    Returns array shape (n_utt, 768).
    Truncates or zero-pads to n_utt if needed.
    """
    path = DATA_DIR / f"{pid}_text_feats.npy"
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


def _load_audio_feats(pid: int, n_utt: int) -> np.ndarray:
    """
    Load pre-extracted wav2vec audio features.
    Expected file: {pid}_audio_feats.npy, shape (n_utt, 768).
    Truncates or zero-pads to n_utt if needed.
    """
    path = DATA_DIR / f"{pid}_audio_feats.npy"
    feats = np.load(str(path)).astype(np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    M = feats.shape[0]
    if M == n_utt:
        return feats
    if M > n_utt:
        return feats[:n_utt]
    pad = np.zeros((n_utt - M, AUDIO_DIM), dtype=np.float32)
    return np.concatenate([feats, pad], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Edge construction (2-modal: Text + Audio)
# ─────────────────────────────────────────────────────────────────────────────

def _temporal_edges(n: int, offset: int, win: int = TEMPORAL_WINDOW) -> np.ndarray:
    """
    Build temporal edges for a block of `n` nodes starting at `offset`.
    Window ±`win`: connect (offset+i) → (offset+j) for |i-j| ≤ win, i≠j.
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
    Build same-utterance cross-modal edges.
    Returns (2, n).
    """
    idx = np.arange(n, dtype=np.int64)
    return np.stack([offset_src + idx, offset_dst + idx], axis=0)


def _build_edges(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build all 4 edge types for a graph with n utterances (2n nodes total).

    Node layout: [0..n-1] = text, [n..2n-1] = audio

    Edge types:
      0  T→T  temporal
      1  A→A  temporal
      2  T→A  same-utterance
      3  A→T  same-utterance
    """
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
    add(_same_utt_edges(n, T_off, A_off), 2)   # T→A
    add(_same_utt_edges(n, A_off, T_off), 3)   # A→T

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
) -> Data:
    """Build a single PyG Data object for one participant."""
    transcript = _load_transcript(pid)
    n_utt = len(transcript)
    if n_utt == 0:
        warnings.warn(f"PID {pid}: empty transcript, skipping.")
        return None

    MAX_UTT = 512
    if n_utt > MAX_UTT:
        transcript = transcript.iloc[:MAX_UTT].reset_index(drop=True)
        n_utt = MAX_UTT

    text_feats  = _load_text_feats(pid, n_utt)   # (n_utt, 768)
    audio_feats = _load_audio_feats(pid, n_utt)  # (n_utt, 768)

    # Unified feature matrix (2n, 1536)
    x_text = np.concatenate([
        text_feats,
        np.zeros((n_utt, AUDIO_DIM), dtype=np.float32),
    ], axis=1)
    x_audio = np.concatenate([
        np.zeros((n_utt, TEXT_DIM), dtype=np.float32),
        audio_feats,
    ], axis=1)
    x = np.concatenate([x_text, x_audio], axis=0)   # (2n, 1536)

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

class DepressionDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for E-DAIC (2-modal: Text + Audio).

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

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.split}_data.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        """Build graphs for every participant in this split and save."""
        split_df = pd.read_csv(SPLIT_FILES[self.split])
        phq8_df  = pd.read_csv(DATA_DIR / "Detailed_PHQ8_Labels.csv")
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

            # Check required files (text + audio pre-extracted features)
            missing = False
            for fname in [f"{pid}_text_feats.npy", f"{pid}_audio_feats.npy",
                          f"{pid}_Transcript.csv"]:
                if not (DATA_DIR / fname).exists():
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

        print(f"[DepressionDataset] split={self.split}: "
              f"{len(data_list)} graphs built, {skipped} skipped.")

        self.save(data_list, self.processed_paths[0])

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def num_node_features(self) -> int:
        return UNIFIED_DIM  # 1536

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def class_weights(self) -> torch.Tensor:
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
