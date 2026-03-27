"""
DAIC-WOZ Dataset Loader for CDL Model

Data structure (flat, NOT in {ID}_P/ folders):
  data/daicwoz/
    train_split_Depression_AVEC2017.csv   ← labels (Participant_ID, PHQ8_Score, PHQ8_Binary)
    dev_split_Depression_AVEC2017.csv
    test_split_Depression_AVEC2017.csv
    {ID}_TRANSCRIPT.csv         ← columns: start_time, stop_time, speaker, value
    {ID}_CLNF_AUs.txt           ← OpenFace AU intensities (frame-level, 30 fps)
    {ID}_COVAREP.csv            ← acoustic features (COVAREP)
    {ID}_text_feats.npy         ← BERT embeddings per turn (from extract_bert.py)

Features:
- Text: 768-dim BERT embeddings (per turn)
- Non-verbal: AU_r + COVAREP (participant-dependent; 300 gives 14 + 74 = 88)
- Output: fixed `nonverbal_dim` after pad/truncate (default 88)

No synthetic data. All real data only.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────

def load_au_features(au_path, max_frames=3000):
    """
    Load OpenFace Action Unit intensities.
    Returns: (T, 17) numpy array — AU01–AU45 intensity columns.
    """
    try:
        df = pd.read_csv(au_path, sep=',', skipinitialspace=True)
        # OpenFace AU intensity columns: AU01_r ... AU45_r
        au_cols = [c for c in df.columns if c.strip().endswith('_r')]
        if not au_cols:
            raise ValueError(f"No AU columns found in {au_path}")
        au = df[au_cols].fillna(0).values.astype(np.float32)
        # Downsample if too long
        if len(au) > max_frames:
            idx = np.linspace(0, len(au) - 1, max_frames, dtype=int)
            au = au[idx]
        return au
    except Exception as e:
        print(f"[WARN] Could not load AUs from {au_path}: {e}")
        return np.zeros((50, 17), dtype=np.float32)


def load_covarep_features(cov_path, max_frames=3000):
    """
    Load COVAREP acoustic features (88-dim per frame).
    Returns: (T, 88) numpy array.
    """
    df = pd.read_csv(cov_path, header=None).fillna(0)
    feat = df.values.astype(np.float32)
    if len(feat) > max_frames:
        idx = np.linspace(0, len(feat) - 1, max_frames, dtype=int)
        feat = feat[idx]
    return feat


def build_nonverbal_feature(au_feat, covarep_feat=None):
    """
    Concatenate AU + acoustic features.
    Feature dimensions in DAIC-WOZ files can vary by extraction setup,
    so we keep whatever is available, then normalize to requested dim later.
    """
    if covarep_feat is not None:
        # Align lengths
        min_len = min(len(au_feat), len(covarep_feat))
        feat = np.concatenate([au_feat[:min_len], covarep_feat[:min_len]], axis=-1)
    else:
        feat = au_feat
    return feat


def load_transcript_features(transcript_path, text_feat_dim=768):
    """
    Load transcript BERT features.

    Requires pre-computed BERT embeddings as {ID}_text_feats.npy.

    Returns: (N_turns, text_feat_dim) numpy array
    """
    bert_path = transcript_path.replace('_TRANSCRIPT.csv', '_text_feats.npy')
    if not os.path.exists(bert_path):
        raise FileNotFoundError(
            f"BERT text features not found at {bert_path}. \n"
            f"Please run: python extract_bert.py"
        )
    return np.load(bert_path).astype(np.float32)


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class DAICWOZDataset(Dataset):
    """
    DAIC-WOZ Dataset for Contrastive Discrepancy Learning.

    Data directory structure (flat, not in {ID}_P folders):
      data/daicwoz/
        {ID}_TRANSCRIPT.csv         ← Transcript (TSV format)
        {ID}_text_feats.npy         ← BERT embeddings (required, from extract_bert.py)
        {ID}_CLNF_AUs.txt           ← OpenFace Action Units
        {ID}_COVAREP.csv            ← Acoustic features

    Args:
        label_csv:      Path to split CSV (train/dev/test) with columns:
                        Participant_ID, PHQ8_Score, PHQ8_Binary
        data_root:      Root directory containing participant data files
        text_feat_dim:  Dimension of text features (768 for BERT)
        nonverbal_dim:  Target dimension for non-verbal features after pad/truncate
        max_text_len:   Max # text turns to keep
        max_nv_len:     Max # non-verbal frames to keep
    """

    def __init__(
        self,
        label_csv,
        data_root="data/daicwoz",
        text_feat_dim=768,
        nonverbal_dim=88,
        max_text_len=100,
        max_nv_len=3000,
    ):
        self.data_root      = data_root
        self.text_feat_dim  = text_feat_dim
        self.nonverbal_dim  = nonverbal_dim
        self.max_text_len   = max_text_len
        self.max_nv_len     = max_nv_len

        self.labels = pd.read_csv(label_csv)
        # Normalize column names
        self.labels.columns = [c.strip() for c in self.labels.columns]
        assert 'Participant_ID' in self.labels.columns, \
            "CSV must have Participant_ID column"

        # PHQ8_Score: continuous; PHQ8_Binary: 0/1
        if 'PHQ8_Score' not in self.labels.columns:
            self.labels['PHQ8_Score'] = 0.0
        if 'PHQ8_Binary' not in self.labels.columns:
            self.labels['PHQ8_Binary'] = (self.labels['PHQ8_Score'] >= 10).astype(int)

    def __len__(self):
        return len(self.labels)

    def _match_feature_dim(self, feat):
        """Pad or truncate feature dim to `self.nonverbal_dim`."""
        cur_dim = feat.shape[-1]
        if cur_dim == self.nonverbal_dim:
            return feat
        if cur_dim < self.nonverbal_dim:
            pad = np.zeros((feat.shape[0], self.nonverbal_dim - cur_dim), dtype=np.float32)
            return np.concatenate([feat, pad], axis=-1)
        return feat[:, :self.nonverbal_dim]

    def _demo_sample(self, idx, pid, phq_score, phq_binary):
        """This method is no longer used."""
        raise NotImplementedError("Demo mode is removed. Use real data only.")

    def __getitem__(self, idx):
        row    = self.labels.iloc[idx]
        pid    = int(row['Participant_ID'])
        phq_s  = float(row['PHQ8_Score'])
        phq_b  = float(row['PHQ8_Binary'])

        # Load text features (BERT embeddings)
        transcript_path = os.path.join(self.data_root, f"{pid}_TRANSCRIPT.csv")
        text_feat = load_transcript_features(transcript_path, self.text_feat_dim)

        # Load non-verbal features (AUs + COVAREP)
        au_path  = os.path.join(self.data_root, f"{pid}_CLNF_AUs.txt")
        cov_path = os.path.join(self.data_root, f"{pid}_COVAREP.csv")
        au_feat  = load_au_features(au_path, self.max_nv_len)
        cov_feat = load_covarep_features(cov_path, self.max_nv_len)
        nv_feat  = build_nonverbal_feature(au_feat, cov_feat)
        nv_feat  = self._match_feature_dim(nv_feat)

        # Truncate
        text_feat = text_feat[:self.max_text_len]
        nv_feat   = nv_feat[:self.max_nv_len]

        return {
            "text_feat":      torch.tensor(text_feat,  dtype=torch.float32),
            "nonverbal_feat": torch.tensor(nv_feat,    dtype=torch.float32),
            "phq_score":      torch.tensor(phq_s,      dtype=torch.float32),
            "dep_label":      torch.tensor(phq_b,      dtype=torch.float32),
            "participant_id": pid,
        }


# ─────────────────────────────────────────
# Collate function (handles variable lengths)
# ─────────────────────────────────────────

def collate_fn(batch):
    text_feats  = [b["text_feat"]      for b in batch]
    nv_feats    = [b["nonverbal_feat"] for b in batch]
    phq_scores  = torch.stack([b["phq_score"]  for b in batch])
    dep_labels  = torch.stack([b["dep_label"]  for b in batch])
    pids        = [b["participant_id"] for b in batch]

    text_lengths = torch.tensor([len(t) for t in text_feats])
    nv_lengths   = torch.tensor([len(n) for n in nv_feats])

    text_padded  = pad_sequence(text_feats, batch_first=True)
    nv_padded    = pad_sequence(nv_feats,   batch_first=True)

    return {
        "text_feat":        text_padded,
        "nonverbal_feat":   nv_padded,
        "text_lengths":     text_lengths,
        "nonverbal_lengths": nv_lengths,
        "phq_score":        phq_scores,
        "dep_label":        dep_labels,
        "participant_id":   pids,
    }


def get_dataloader(label_csv, data_root="data/daicwoz", batch_size=16,
                   shuffle=True, text_feat_dim=768, nonverbal_dim=88, **kwargs):
    ds = DAICWOZDataset(
        label_csv, data_root,
        text_feat_dim=text_feat_dim,
        nonverbal_dim=nonverbal_dim,
        **kwargs
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0)