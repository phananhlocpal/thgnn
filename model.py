"""
Contrastive Discrepancy Learning (CDL) for Depression Detection
Dataset: DAIC-WOZ
Author: Based on user's idea
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
# 1. Text Encoder (Verbal Channel)
# ─────────────────────────────────────────
class TextEncoder(nn.Module):
    """
    Encodes transcript segments into a fixed-size embedding.
    Uses a 2-layer BiLSTM over token embeddings (or BERT features).
    Input: (B, T, text_feat_dim) — e.g. GloVe/BERT token embeddings
    Output: (B, hidden_dim)
    """
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),   # bounds output, prevents magnitude explosion, preserves variance
        )

    def forward(self, x, lengths=None):
        # x: (B, T, input_dim)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        out, _ = self.lstm(x)
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # Attention pooling with padding mask
        scores = self.attn(out).squeeze(-1)                          # (B, T)
        if lengths is not None:
            mask = torch.arange(scores.size(1), device=scores.device).unsqueeze(0) >= lengths.unsqueeze(1).to(scores.device)
            scores = scores.masked_fill(mask, -1e9)
        scores = F.softmax(scores, dim=-1).unsqueeze(-1)             # (B, T, 1)
        context = (out * scores).sum(dim=1)                          # (B, 2*hidden)
        return self.proj(context)                                    # (B, hidden)


# ─────────────────────────────────────────
# 2. Non-Verbal Encoder (Facial AUs + Audio)
# ─────────────────────────────────────────
class NonVerbalEncoder(nn.Module):
    """
    Encodes facial AUs (17 AUs from OpenFace) and/or acoustic features.
    Input: (B, T, nonverbal_feat_dim)
    Output: (B, hidden_dim)
    """
    def __init__(self, input_dim=88, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        # 1D CNN to capture local temporal patterns in AUs
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            128, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )
        self.norm = nn.Identity()  # kept for API compat, no-op

    def forward(self, x, lengths=None):
        # Data Augmentation: Thêm nhiễu lúc train
        if self.training:
            x = x + torch.randn_like(x) * 0.05

        # x: (B, T, input_dim)
        x = x.transpose(1, 2)          # (B, input_dim, T)
        x = self.conv(x)               # (B, 128, T)
        x = x.transpose(1, 2)          # (B, T, 128)

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        out, _ = self.lstm(x)
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        scores = self.attn(out).squeeze(-1)
        if lengths is not None:
            mask = torch.arange(scores.size(1), device=scores.device).unsqueeze(0) >= lengths.unsqueeze(1).to(scores.device)
            scores = scores.masked_fill(mask, -1e9)
        scores = F.softmax(scores, dim=-1).unsqueeze(-1)
        context = (out * scores).sum(dim=1)
        return self.proj(context)


# ─────────────────────────────────────────
# 3. Contrastive Discrepancy Module
# ─────────────────────────────────────────
class ContrastiveDiscrepancyModule(nn.Module):
    """
    Computes the discrepancy between verbal and non-verbal embeddings.

    Three discrepancy signals are combined:
      - Cosine distance: captures directional disagreement
      - L2 distance:     captures magnitude disagreement
      - Element-wise diff: rich local feature mismatch
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Projects rich discrepancy signal into a scalar + vector
        self.discrepancy_proj = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, z_text, z_nonverbal):
        """
        Args:
            z_text:      (B, H) verbal embedding
            z_nonverbal: (B, H) non-verbal embedding
        Returns:
            discrepancy_feat: (B, H//2) rich discrepancy features
            cosine_dist:      (B,) scalar discrepancy score [0, 2]
            l2_dist:          (B,) scalar L2 discrepancy
        """
        # --- Scalar distances ---
        cosine_sim  = F.cosine_similarity(z_text, z_nonverbal, dim=-1)   # (B,)
        cosine_dist = 1.0 - cosine_sim                                    # [0, 2]
        l2_dist     = (z_text - z_nonverbal).pow(2).sum(-1).sqrt()        # (B,)

        # --- Rich element-wise difference ---
        elem_diff = torch.abs(z_text - z_nonverbal)                       # (B, H)

        # Concatenate all signals
        scalars = torch.stack([cosine_dist, l2_dist], dim=-1)             # (B, 2)
        cat     = torch.cat([elem_diff, scalars], dim=-1)                 # (B, H+2)

        discrepancy_feat = self.discrepancy_proj(cat)                     # (B, H//2)
        return discrepancy_feat, cosine_dist, l2_dist


# ─────────────────────────────────────────
# 4. Full CDL Model
# ─────────────────────────────────────────
class CDLModel(nn.Module):
    """
    Contrastive Discrepancy Learning for Depression Severity Prediction.

    Pipeline:
      TextEncoder + NonVerbalEncoder → ContrastiveDiscrepancyModule
      → Fusion (discrepancy + both embeddings) → PHQ-8 score regression
        + binary depression classification

    Outputs:
      phq_score   : (B,) continuous PHQ-8 prediction
      dep_logit   : (B,) binary depression logit (PHQ >= 10)
      cosine_dist : (B,) discrepancy score (interpretable)
      l2_dist     : (B,)
    """
    def __init__(
        self,
        text_input_dim=768,
        nonverbal_input_dim=88,
        hidden_dim=128,
        dropout=0.3,
    ):
        super().__init__()
        self.text_encoder     = TextEncoder(text_input_dim, hidden_dim, dropout=dropout)
        self.nonverbal_encoder = NonVerbalEncoder(nonverbal_input_dim, hidden_dim, dropout=dropout)
        self.cdm              = ContrastiveDiscrepancyModule(hidden_dim)

        # Fusion: concat(z_text, z_nonverb, discrepancy_feat)
        fusion_dim = hidden_dim + hidden_dim + hidden_dim // 2

        self.fusion_bn = nn.LayerNorm(fusion_dim)  # normalize fusion input across batch
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),   # reduced dropout: 0.15 instead of 0.3
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Task heads
        self.phq_head = nn.Linear(64, 1)          # regression
        self.dep_head = nn.Linear(64, 1)          # binary classification

    def forward(self, text_feat, nonverbal_feat,
                text_lengths=None, nonverbal_lengths=None):
        z_text      = self.text_encoder(text_feat, text_lengths)
        z_nonverbal = self.nonverbal_encoder(nonverbal_feat, nonverbal_lengths)

        disc_feat, cosine_dist, l2_dist = self.cdm(z_text, z_nonverbal)

        fusion_input = torch.cat([z_text, z_nonverbal, disc_feat], dim=-1)
        fusion_input = F.dropout(fusion_input, p=0.3, training=self.training) # Thêm dòng này
        fusion_input = self.fusion_bn(fusion_input) 
        fused  = self.fusion(fusion_input)
        
        phq_score = self.phq_head(fused).squeeze(-1)   # (B,)
        dep_logit = self.dep_head(fused).squeeze(-1)   # (B,)
        # NOTE: Removed phq coupling — it injected noise before phq_head converges.

        return {
            "phq_score":   phq_score,
            "dep_logit":   dep_logit,
            "cosine_dist": cosine_dist,
            "l2_dist":     l2_dist,
            "z_text":      z_text,
            "z_nonverbal": z_nonverbal,
        }


# ─────────────────────────────────────────
# 5. Combined Loss
# ─────────────────────────────────────────
class CDLLoss(nn.Module):
    """
    Multi-objective loss:
      L = L_reg + λ1 * L_cls + λ2 * L_contrastive

    L_contrastive (Contrastive Loss):
      - Depressed patients (label=1): embeddings should be FAR apart
                                      (high discrepancy is a true signal)
      - Non-depressed (label=0):     embeddings should be CLOSE
                                      (congruent affect = healthy)
      This directly trains the model to use discrepancy as a diagnostic feature.
    """
    def __init__(
        self,
        margin=0.5,
        lambda_cls=1.0,
        lambda_contrast=0.5,
        pos_weight=None,
        contrastive_pos_weight=1.0,
        phq_max=27.0,
    ):
        super().__init__()
        self.margin          = margin
        self.lambda_cls      = lambda_cls
        self.lambda_contrast = lambda_contrast
        self.contrastive_pos_weight = contrastive_pos_weight
        self.phq_max         = phq_max
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, outputs, phq_labels, dep_labels):
        """
        Args:
            outputs:    dict from CDLModel.forward()
            phq_labels: (B,) float, PHQ-8 score (0-27)
            dep_labels: (B,) float, 0/1 binary label
        """
        # 1. Regression loss on normalized PHQ-8 score (stabilizes multi-task scale)
        l_reg = self.mse(outputs["phq_score"] / self.phq_max, phq_labels / self.phq_max)

        # 2. Binary classification loss
        l_cls = self.bce(outputs["dep_logit"], dep_labels)

        # 3. Contrastive discrepancy loss (cosine-based, scale-invariant)
        #    Depressed (1): cosine_dist should be HIGH → penalize if dist < margin
        #    Healthy   (0): cosine_dist should be LOW  → penalize if dist > margin*0.2
        #    Using cosine_dist (bounded [0,2]) instead of L2 avoids scale issues.
        cdist = outputs["cosine_dist"]  # already in [0, 2], no normalization needed
        pos_loss = dep_labels * self.contrastive_pos_weight * F.relu(self.margin - cdist).pow(2)
        neg_loss = (1 - dep_labels) * F.relu(cdist - self.margin * 0.3).pow(2)
        l_contrast = (pos_loss + neg_loss).mean()

        total = l_reg + self.lambda_cls * l_cls + self.lambda_contrast * l_contrast

        return {
            "total":      total,
            "l_reg":      l_reg.item(),
            "l_cls":      l_cls.item(),
            "l_contrast": l_contrast.item(),
        }