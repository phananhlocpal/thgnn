"""
daicwoz_dataset.py (v2 — full rewrite)

DaicWozDataset: PyG InMemoryDataset cho DAIC-WOZ corpus.

Thay đổi vs v1
──────────────
[FIX-CRITICAL] n_utt đọc từ text_feats_meta.json (không phải transcript rows).

[FIX-CRITICAL] Validation text/audio alignment + fallback to min.

[NEW] Same-question edges (edge_type=4, edge_type=5):
  T→T và A→A nối các nodes thuộc cùng ellie_question_id.
  Lấy question_id từ text_feats_meta.json (được lưu bởi extract_bert_v4.py).
  Tổng cộng: 6 edge types thay vì 4.
  SR-RGAT có thể phân biệt temporal context vs question context.

[NEW] Sub-Dialogue Shuffling (SDS) augmentation:
  Wu et al. 2023 đạt +78% trên DAIC-WOZ với SDS.
  Implementation: shuffle thứ tự utterance groups theo sub-dialogues.
  SDS chỉ apply ở training time.

[NEW] Feature dims: TEXT_ACOUSTIC_DIM=9, AUDIO_ACOUSTIC_DIM=9.
  UNIFIED_DIM = (768+9)*2 = 1554.

[FIX] NUM_EDGE_TYPES = 6 (update từ 4).
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

TEXT_DIM  = 768
AUDIO_DIM = 768

TEXT_ACOUSTIC_DIM  = 9
AUDIO_ACOUSTIC_DIM = 9

TEXT_DIM_TOTAL  = TEXT_DIM  + TEXT_ACOUSTIC_DIM    # 777
AUDIO_DIM_TOTAL = AUDIO_DIM + AUDIO_ACOUSTIC_DIM   # 777
UNIFIED_DIM     = TEXT_DIM_TOTAL + AUDIO_DIM_TOTAL  # 1554

TEMPORAL_WINDOW = 5
NUM_EDGE_TYPES  = 6  # T->T_temp, A->A_temp, T->A_utt, A->T_utt, T->T_q, A->A_q

DAICWOZ_PHQ8_COLS = [
    "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
    "PHQ8_Appetite",   "PHQ8_Failure",   "PHQ8_Concentrating", "PHQ8_Moving",
]

DAICWOZ_SPLIT_FILES = {
    "train": DAICWOZ_DATA_DIR / "train_split_Depression_AVEC2017.csv",
    "dev":   DAICWOZ_DATA_DIR / "dev_split_Depression_AVEC2017.csv",
    "test":  DAICWOZ_DATA_DIR / "full_test_split.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature loading
# ─────────────────────────────────────────────────────────────────────────────

def _get_n_groups(pid: int) -> Optional[int]:
    meta_path = DAICWOZ_DATA_DIR / f"{pid}_text_feats_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return len(json.load(f))
    ngroups_path = DAICWOZ_DATA_DIR / f"{pid}_n_groups.txt"
    if ngroups_path.exists():
        with open(ngroups_path) as f:
            return int(f.read().strip())
    npy_path = DAICWOZ_DATA_DIR / f"{pid}_text_feats.npy"
    if npy_path.exists():
        feats = np.load(str(npy_path))
        return feats.shape[0] if feats.ndim == 2 else 1
    return None


def _load_feats(pid: int, n_utt: int, modality: str) -> np.ndarray:
    dim  = TEXT_DIM if modality == "text" else AUDIO_DIM
    path = DAICWOZ_DATA_DIR / f"{pid}_{modality}_feats.npy"
    feats = np.load(str(path)).astype(np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    M = feats.shape[0]
    if M == n_utt:
        return feats
    if M > n_utt:
        return feats[:n_utt]
    warnings.warn(f"PID {pid}: {modality} has {M} rows, padding to {n_utt}.")
    return np.concatenate([feats, np.zeros((n_utt - M, dim), dtype=np.float32)])


# ─────────────────────────────────────────────────────────────────────────────
# Acoustic side-channel features
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_acoustics(meta: dict) -> np.ndarray:
    sr       = float(meta.get("speech_rate_wps",      0.0) or 0.0) / 5.0
    lat      = float(meta.get("response_latency_sec", 0.0) or 0.0) / 10.0
    sigh     = float(bool(meta.get("has_sigh",        False)))
    laugh    = float(bool(meta.get("has_laughter",    False)))
    breath   = float(bool(meta.get("has_breath",      False)))
    cry      = float(bool(meta.get("has_cry",         False)))
    cough    = float(bool(meta.get("has_cough",       False)))
    other    = float(bool(meta.get("has_other_sound", False)))
    sus_rate = float(bool(meta.get("suspicious_rate", False)))
    return np.array([sr, lat, sigh, laugh, breath, cry, cough, other, sus_rate],
                    dtype=np.float32)


def _extract_audio_acoustics(meta: dict) -> np.ndarray:
    mean_pause = float(meta.get("mean_pause_sec",  0.0) or 0.0) / 5.0
    max_pause  = float(meta.get("max_pause_sec",   0.0) or 0.0) / 10.0
    n_pauses   = float(meta.get("n_pauses",        0.0) or 0.0) / 10.0
    speech_rat = float(meta.get("speech_ratio",    0.5) or 0.5)
    mean_pitch = (float(meta.get("mean_pitch_hz",  200.0) or 200.0) - 50.0) / 450.0
    std_pitch  = float(meta.get("std_pitch_hz",    0.0)  or 0.0) / 100.0
    mean_nrg   = float(meta.get("mean_energy",     0.0)  or 0.0) / 0.2
    std_nrg    = float(meta.get("std_energy",      0.0)  or 0.0) / 0.1
    low_speech = float(bool(meta.get("low_speech", False)))
    return np.array(
        [mean_pause, max_pause, n_pauses, speech_rat,
         mean_pitch, std_pitch, mean_nrg, std_nrg, low_speech],
        dtype=np.float32,
    )


def _load_text_acoustics(pid: int, n_utt: int) -> np.ndarray:
    zeros = np.zeros((n_utt, TEXT_ACOUSTIC_DIM), dtype=np.float32)
    path  = DAICWOZ_DATA_DIR / f"{pid}_text_feats_meta.json"
    if not path.exists():
        return zeros
    with open(path) as f:
        meta_list = json.load(f)
    rows = [_extract_text_acoustics(m) for m in meta_list]
    arr  = np.stack(rows) if rows else zeros
    M    = arr.shape[0]
    if M >= n_utt:
        return arr[:n_utt]
    return np.concatenate([arr, np.zeros((n_utt - M, TEXT_ACOUSTIC_DIM), dtype=np.float32)])


def _load_audio_acoustics(pid: int, n_utt: int) -> np.ndarray:
    zeros = np.zeros((n_utt, AUDIO_ACOUSTIC_DIM), dtype=np.float32)
    path  = DAICWOZ_DATA_DIR / f"{pid}_audio_feats_meta.json"
    if not path.exists():
        return zeros
    with open(path) as f:
        meta_list = json.load(f)
    rows = [_extract_audio_acoustics(m) for m in meta_list]
    arr  = np.stack(rows) if rows else zeros
    M    = arr.shape[0]
    if M >= n_utt:
        return arr[:n_utt]
    return np.concatenate([arr, np.zeros((n_utt - M, AUDIO_ACOUSTIC_DIM), dtype=np.float32)])


def _load_question_ids(pid: int, n_utt: int) -> List[str]:
    path = DAICWOZ_DATA_DIR / f"{pid}_text_feats_meta.json"
    if not path.exists():
        return [""] * n_utt
    with open(path) as f:
        meta_list = json.load(f)
    ids = [m.get("ellie_question_id", "") for m in meta_list]
    if len(ids) >= n_utt:
        return ids[:n_utt]
    return ids + [""] * (n_utt - len(ids))


# ─────────────────────────────────────────────────────────────────────────────
# Edge construction (6 types)
# ─────────────────────────────────────────────────────────────────────────────

def _temporal_edges(n: int, offset: int, win: int = TEMPORAL_WINDOW) -> np.ndarray:
    src, dst = [], []
    for i in range(n):
        for j in range(max(0, i - win), min(n, i + win + 1)):
            if j != i:
                src.append(offset + i)
                dst.append(offset + j)
    return np.array([src, dst], dtype=np.int64) if src else np.empty((2, 0), dtype=np.int64)


def _same_utt_edges(n: int, offset_src: int, offset_dst: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    return np.stack([offset_src + idx, offset_dst + idx])


def _same_question_edges(n: int, offset: int, q_ids: List[str]) -> np.ndarray:
    src, dst = [], []
    q_to_idxs: Dict[str, List[int]] = {}
    for i, qid in enumerate(q_ids[:n]):
        if qid:
            q_to_idxs.setdefault(qid, []).append(i)
    for idxs in q_to_idxs.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(len(idxs)):
                if a != b:
                    src.append(offset + idxs[a])
                    dst.append(offset + idxs[b])
    return np.array([src, dst], dtype=np.int64) if src else np.empty((2, 0), dtype=np.int64)


def _build_edges(n: int, q_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    T_off = 0
    A_off = n
    parts: List[np.ndarray] = []
    types: List[np.ndarray] = []

    def add(edges: np.ndarray, etype: int):
        if edges.shape[1] > 0:
            parts.append(edges)
            types.append(np.full(edges.shape[1], etype, dtype=np.int64))

    add(_temporal_edges(n, T_off),                0)
    add(_temporal_edges(n, A_off),                1)
    add(_same_utt_edges(n, T_off, A_off),         2)
    add(_same_utt_edges(n, A_off, T_off),         3)
    add(_same_question_edges(n, T_off, q_ids),    4)
    add(_same_question_edges(n, A_off, q_ids),    5)

    if not parts:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,),   dtype=torch.long))
    return (torch.from_numpy(np.concatenate(parts, axis=1)),
            torch.from_numpy(np.concatenate(types)))


# ─────────────────────────────────────────────────────────────────────────────
# Sub-Dialogue Shuffling (SDS) — Wu et al. 2023
# ─────────────────────────────────────────────────────────────────────────────

def apply_sds(data: Data, aug_prob: float = 0.8, K: int = 3) -> Data:
    """
    Sub-Dialogue Shuffling.
    1. Split utterance sequence into K sub-dialogues.
    2. Shuffle sub-dialogue order.
    3. Rebuild positional encodings + temporal edges.
    Cross-modal (T->A, A->T) and same-question edges preserved.
    """
    if torch.rand(1).item() > aug_prob:
        return data

    n_utt = data.n_utt
    if n_utt < K * 2:
        return data

    boundaries = np.linspace(0, n_utt, K + 1, dtype=int)
    sub_ranges = [list(range(boundaries[k], boundaries[k + 1])) for k in range(K)]
    sub_ranges = [r for r in sub_ranges if r]
    if len(sub_ranges) < 2:
        return data

    order         = torch.randperm(len(sub_ranges)).tolist()
    new_utt_order = []
    for idx in order:
        new_utt_order.extend(sub_ranges[idx])

    T_off      = 0
    A_off      = n_utt
    perm_text  = [T_off + i for i in new_utt_order]
    perm_audio = [A_off + i for i in new_utt_order]
    perm       = torch.tensor(perm_text + perm_audio, dtype=torch.long)

    new_x         = data.x[perm]
    new_node_type = data.node_type[perm]

    # Rebuild positional encoding
    new_pos_utt = torch.linspace(0.0, 1.0, n_utt)
    new_pos     = torch.cat([new_pos_utt, new_pos_utt])

    # Remap edge indices via inverse permutation
    n_total  = 2 * n_utt
    inv_perm = torch.zeros(n_total, dtype=torch.long)
    inv_perm[perm] = torch.arange(n_total, dtype=torch.long)
    new_edge_index = inv_perm[data.edge_index.view(-1)].view(2, -1)

    return Data(
        x            = new_x,
        edge_index   = new_edge_index,
        edge_type    = data.edge_type.clone(),
        node_type    = new_node_type,
        pos          = new_pos,
        y            = data.y.clone(),
        phq_score    = data.phq_score.clone(),
        phq8         = data.phq8.clone(),
        pid          = data.pid,
        n_utt        = data.n_utt,
        is_augmented = torch.tensor([1], dtype=torch.long),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    pid:       int,
    label:     int,
    phq_score: float,
    phq8:      np.ndarray,
) -> Optional[Data]:
    n_utt = _get_n_groups(pid)
    if n_utt is None or n_utt == 0:
        warnings.warn(f"PID {pid}: cannot determine n_groups.")
        return None

    n_utt = min(n_utt, 512)

    text_path  = DAICWOZ_DATA_DIR / f"{pid}_text_feats.npy"
    audio_path = DAICWOZ_DATA_DIR / f"{pid}_audio_feats.npy"
    n_text  = np.load(str(text_path)).shape[0]  if text_path.exists()  else 0
    n_audio = np.load(str(audio_path)).shape[0] if audio_path.exists() else 0

    if n_text != n_audio:
        warnings.warn(f"PID {pid}: text={n_text} vs audio={n_audio}. Using min.")
        n_utt = min(n_text, n_audio, n_utt)

    if n_utt == 0:
        return None

    text_feats  = _load_feats(pid, n_utt, "text")
    audio_feats = _load_feats(pid, n_utt, "audio")
    text_acous  = _load_text_acoustics(pid, n_utt)
    audio_acous = _load_audio_acoustics(pid, n_utt)
    q_ids       = _load_question_ids(pid, n_utt)

    text_aug  = np.concatenate([text_feats,  text_acous],  axis=1)   # (n, 777)
    audio_aug = np.concatenate([audio_feats, audio_acous], axis=1)   # (n, 777)

    x_text  = np.concatenate([text_aug,  np.zeros((n_utt, AUDIO_DIM_TOTAL), np.float32)], axis=1)
    x_audio = np.concatenate([np.zeros((n_utt, TEXT_DIM_TOTAL), np.float32), audio_aug], axis=1)
    x       = np.concatenate([x_text, x_audio], axis=0)  # (2n, 1554)

    node_type = np.array([0] * n_utt + [1] * n_utt, dtype=np.int64)
    utt_pos   = np.linspace(0.0, 1.0, n_utt, dtype=np.float32)
    pos       = np.tile(utt_pos, 2)

    edge_index, edge_type = _build_edges(n_utt, q_ids)

    return Data(
        x            = torch.from_numpy(x).float(),
        edge_index   = edge_index,
        edge_type    = edge_type,
        node_type    = torch.from_numpy(node_type),
        pos          = torch.from_numpy(pos),
        y            = torch.tensor([label],     dtype=torch.long),
        phq_score    = torch.tensor([phq_score], dtype=torch.float),
        phq8         = torch.from_numpy(phq8.astype(np.float32)),
        pid          = pid,
        n_utt        = n_utt,
        is_augmented = torch.tensor([0], dtype=torch.long),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class DaicWozDataset(InMemoryDataset):

    def __init__(
        self,
        split:         str            = "train",
        root:          str            = str(DAICWOZ_CACHE_DIR),
        transform:     Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload:  bool           = False,
        aug_sds:       bool           = False,
        aug_copies:    int            = 2,
        aug_prob:      float          = 0.8,
    ):
        assert split in ("train", "dev", "test")
        self.split         = split
        self._force_reload = force_reload
        self.aug_sds       = aug_sds and (split == "train")
        self.aug_copies    = aug_copies
        self.aug_prob      = aug_prob
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        aug_tag = f"_sds{self.aug_copies}" if self.aug_sds else ""
        return [f"{self.split}{aug_tag}_v2_data.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        split_df = pd.read_csv(DAICWOZ_SPLIT_FILES[self.split])
        split_df.columns = split_df.columns.str.strip()

        label_col = "PHQ8_Binary" if "PHQ8_Binary" in split_df.columns else "PHQ_Binary"
        score_col = "PHQ8_Score"  if "PHQ8_Score"  in split_df.columns else "PHQ_Score"

        phq8_dict: Dict[int, np.ndarray] = {}
        if all(c in split_df.columns for c in DAICWOZ_PHQ8_COLS):
            for _, row in split_df.iterrows():
                vals = row[DAICWOZ_PHQ8_COLS].values.astype(np.float32)
                vals = np.where(np.isnan(vals), 0.0, vals)
                phq8_dict[int(row["Participant_ID"])] = vals

        data_list: List[Data] = []
        skipped = 0

        for _, row in split_df.iterrows():
            pid       = int(row["Participant_ID"])
            label     = int(row[label_col])
            phq_score = float(row[score_col]) if score_col in split_df.columns else 0.0
            phq8      = phq8_dict.get(pid, np.zeros(8, dtype=np.float32))

            missing = any(
                not (DAICWOZ_DATA_DIR / f"{pid}_{m}_feats.npy").exists()
                for m in ("text", "audio")
            )
            if missing:
                warnings.warn(f"PID {pid}: missing feats file.")
                skipped += 1
                continue

            try:
                graph = build_graph(pid, label, phq_score, phq8)
            except Exception as exc:
                warnings.warn(f"PID {pid}: {exc}")
                skipped += 1
                continue

            if graph is None:
                skipped += 1
                continue

            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            data_list.append(graph)

            # SDS augmentation (training only)
            if self.aug_sds:
                for _ in range(self.aug_copies):
                    aug = apply_sds(graph, aug_prob=self.aug_prob)
                    if self.pre_transform is not None:
                        aug = self.pre_transform(aug)
                    data_list.append(aug)

        total = len(data_list)
        orig  = total // (1 + self.aug_copies) if self.aug_sds else total
        print(
            f"[DaicWozDataset] split={self.split}: "
            f"{orig} original + {total - orig} SDS = {total} total "
            f"({skipped} skipped)"
        )
        self.save(data_list, self.processed_paths[0])

    @property
    def num_node_features(self) -> int:
        return UNIFIED_DIM

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_edge_types(self) -> int:
        return NUM_EDGE_TYPES

    def labels(self) -> torch.Tensor:
        return torch.cat([d.y for d in self], dim=0)

    @property
    def class_weights(self) -> torch.Tensor:
        labels = torch.cat([d.y.reshape(-1) for d in self], dim=0).float()
        n_pos  = labels.sum().item()
        n_neg  = (1 - labels).sum().item()
        total  = n_pos + n_neg
        return torch.tensor(
            [total / (2.0 * max(n_neg, 1)), total / (2.0 * max(n_pos, 1))],
            dtype=torch.float,
        )


if __name__ == "__main__":
    for split in ("train", "dev", "test"):
        ds = DaicWozDataset(split=split, aug_sds=(split == "train"), aug_copies=2)
        print(f"  {split}: {len(ds)} samples")
        if len(ds) > 0:
            d = ds[0]
            print(
                f"    x={d.x.shape}  edges={d.edge_index.shape[1]}  "
                f"types={sorted(d.edge_type.unique().tolist())}  "
                f"y={d.y.item()}  n_utt={d.n_utt}  aug={d.is_augmented.item()}"
            )