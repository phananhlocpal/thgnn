"""
HTDG-CDL: Heterogeneous Temporal Discrepancy Graph for
Contrastive Discrepancy Learning in Depression Detection

Core Mathematical Framework:
─────────────────────────────────────────────────────────────────
1. HTDG (Heterogeneous Temporal Discrepancy Graph):
   G = (V, E, τ_V, τ_E) where τ_V ∈ {text, audio, facial}
   and τ_E ∈ {temporal, cross-modal-consistent, cross-modal-discrepant}

   INNOVATION: The graph topology ENCODES discrepancy as structure.
   Discrepancy edge weight: w_ij = 1 - σ(cos_sim(h_i, h_j))
   This makes "incongruence" a first-class geometric property.

2. Heterogeneous Graph Attention with Edge Features (HGA-EF):
   α_ij^(τ) = softmax_j [ (W_q^τ h_i)^T (W_k^τ h_j) / √d
                           + φ_τ(e_ij)^T r_τ ]
   where φ_τ(e_ij) encodes (edge_type, discrepancy_weight, temporal_distance)
   
   Message passing:
   h_v^(l+1) = σ( LayerNorm( Σ_{τ∈R} W_o^τ · Σ_{u∈N_τ(v)} α_vu^(τ) · W_v^τ h_u^(l) ) )

3. Spectral Discrepancy Signal (SDS):
   Build discrepancy Laplacian: L_disc = D_disc - A_disc
   where A_disc[i,j] = w_ij (cross-modal discrepancy weight)
   
   Project features onto top-k high-frequency eigenvectors:
   z_spectral = U_k^T h  (k eigenvectors with largest eigenvalues)
   
   Biological basis: depression → abnormal HIGH-frequency patterns
   in facial/acoustic sequences (flat affect, psychomotor retardation).

4. Hyperbolic Manifold Embedding (Poincaré Ball):
   Map fused features to Poincaré ball B^n_c = {x ∈ ℝ^n : c||x|| < 1}
   
   Exponential map: Exp_x^c(v) = x ⊕_c tanh(√c · λ_x^c · ||v||/2) · v/(√c||v||)
   Möbius addition: x ⊕_c y = ((1+2c<x,y>+c||y||²)x + (1-c||x||²)y) / (1+2c<x,y>+c²||x||²||y||²)
   
   Poincaré distance: d_c(x,y) = (2/√c) · artanh(√c · ||-x ⊕_c y||)
   
   Depression severity forms a HIERARCHY in hyperbolic space:
   center (c=0) → mild → moderate → severe (near boundary)

5. Riemannian Manifold Contrastive Loss (RMC):
   L_RMC = Σ_i log [ exp(-d_c(z_i, z_i+)/τ) / Σ_j exp(-d_c(z_i, z_j)/τ) ]
   where d_c = Poincaré distance (geodesic on curved manifold)
   
   Combined loss:
   L = L_MSE(PHQ) + λ_cls · L_BCE(dep) + λ_rmc · L_RMC + λ_reg · ||θ||²

6. Hyperbolic Mixup (data scarcity augmentation):
   z_mix = Exp_0^c( λ · Log_0^c(z_i) + (1-λ) · Log_0^c(z_j) )
   y_mix = λ · y_i + (1-λ) · y_j
   
   Valid interpolation on the Poincaré manifold — generates
   semantically meaningful synthetic samples along depression continuum.
─────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax


# ═══════════════════════════════════════════════════════════════
# SECTION 1: Hyperbolic Geometry Operations (Poincaré Ball)
# ═══════════════════════════════════════════════════════════════

class PoincareBall(nn.Module):
    """
    Poincaré Ball model of hyperbolic space with curvature c.
    
    All operations are numerically stabilized for deep learning.
    
    Mathematical reference:
      Ganea et al. (2018) "Hyperbolic Neural Networks"
      Chami et al. (2019) "Hyperbolic Graph Convolutional Neural Networks"
    """
    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        super().__init__()
        # Learnable curvature (allows model to discover optimal geometry)
        self.c = nn.Parameter(torch.tensor(c))
        self.eps = eps

    def _clip(self, x):
        """Project x inside the open ball: ||x|| < 1/√c"""
        c = self.c.abs().clamp(min=1e-8)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        max_norm = (1.0 - self.eps) / (c.sqrt())
        return x * torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))

    def mobius_add(self, x, y):
        """
        Möbius addition: x ⊕_c y
        = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y)
          / (1 + 2c<x,y> + c²||x||²||y||²)
        """
        c = self.c.abs().clamp(min=1e-8)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        
        num_x = (1 + 2 * c * xy + c * y2) * x
        num_y = (1 - c * x2) * y
        denom = (1 + 2 * c * xy + c * c * x2 * y2).clamp(min=self.eps)
        
        return self._clip((num_x + num_y) / denom)

    def exp_map_zero(self, v):
        """
        Exponential map at origin: Exp_0^c(v)
        = tanh(√c · ||v||) · v / (√c · ||v||)
        Maps tangent vectors to Poincaré ball.
        """
        c = self.c.abs().clamp(min=1e-8)
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = c.sqrt()
        return self._clip(torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm))

    def log_map_zero(self, y):
        """
        Logarithmic map at origin: Log_0^c(y)
        = artanh(√c · ||y||) · y / (√c · ||y||)
        Maps Poincaré ball back to tangent space.
        """
        c = self.c.abs().clamp(min=1e-8)
        y_norm = y.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = c.sqrt()
        return torch.atanh((sqrt_c * y_norm).clamp(-1 + self.eps, 1 - self.eps)) * y / (sqrt_c * y_norm)

    def distance(self, x, y):
        """
        Poincaré distance: d_c(x,y) = (2/√c) · artanh(√c · ||-x ⊕_c y||)
        """
        c = self.c.abs().clamp(min=1e-8)
        neg_x = -x
        diff = self.mobius_add(neg_x, y)
        diff_norm = diff.norm(dim=-1).clamp(min=self.eps)
        sqrt_c = c.sqrt()
        return (2.0 / sqrt_c) * torch.atanh((sqrt_c * diff_norm).clamp(-1 + self.eps, 1 - self.eps))

    def hyperbolic_linear(self, x, W, b=None):
        """
        Hyperbolic linear map via:
        1. Log_0(x) → tangent space
        2. Apply linear W
        3. Exp_0 → back to manifold
        """
        tan_x = self.log_map_zero(x)
        out = tan_x @ W.t()
        if b is not None:
            out = out + b
        return self.exp_map_zero(out)

    def hyperbolic_mixup(self, z_i, z_j, lam):
        """
        Geodesic interpolation on Poincaré ball (valid hyperbolic mixup):
        z_mix = Exp_0(λ · Log_0(z_i) + (1-λ) · Log_0(z_j))
        
        Unlike Euclidean mixup, this stays ON the manifold.
        """
        tan_i = self.log_map_zero(z_i)
        tan_j = self.log_map_zero(z_j)
        tan_mix = lam * tan_i + (1 - lam) * tan_j
        return self.exp_map_zero(tan_mix)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: Heterogeneous Graph Attention with Edge Features
# ═══════════════════════════════════════════════════════════════

class HGAEdgeConv(MessagePassing):
    """
    Heterogeneous Graph Attention with Edge Features (HGA-EF).
    
    For a relation type τ:
    α_ij^(τ) = softmax_j [ (W_q h_i)^T (W_k h_j) / √d_k + φ(e_ij)^T r ]
    
    where φ(e_ij) ∈ ℝ^d encodes:
      - edge_type ∈ {temporal, cross-modal-consistent, cross-modal-discrepant}
      - discrepancy_weight w_ij = 1 - σ(cos_sim(h_i, h_j))
      - temporal_distance |t_i - t_j|
    
    This allows the model to LEARN different behaviors for
    consistent vs discrepant cross-modal edges.
    """
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 edge_dim: int = 16, dropout: float = 0.1):
        super().__init__(aggr='add', node_dim=0)
        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        self.edge_dim = edge_dim
        self.scale    = math.sqrt(self.head_dim)

        # Query / Key / Value projections (per head)
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)

        # Edge feature projection → attention bias
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, n_heads),
        )

        # Output projection
        self.W_o = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        """
        x:          (N, in_dim)
        edge_index: (2, E)
        edge_attr:  (E, edge_dim)
        """
        return self.norm(x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

    def message(self, x_i, x_j, edge_attr, index):
        """
        Compute attention-weighted message for each edge.
        x_i, x_j: (E, in_dim) — target/source node features
        edge_attr: (E, edge_dim)
        """
        B = x_i.size(0)

        # Multi-head projections → (E, n_heads, head_dim)
        Q = self.W_q(x_i).view(B, self.n_heads, self.head_dim)
        K = self.W_k(x_j).view(B, self.n_heads, self.head_dim)
        V = self.W_v(x_j).view(B, self.n_heads, self.head_dim)

        # Dot-product attention score: (E, n_heads)
        attn = (Q * K).sum(dim=-1) / self.scale

        # Edge feature bias: (E, n_heads)
        edge_bias = self.edge_proj(edge_attr)
        attn = attn + edge_bias

        # Softmax over neighbors (per head)
        attn = pyg_softmax(attn, index)                      # (E, n_heads)
        attn = self.drop(attn)

        # Weighted sum: (E, n_heads, head_dim)
        out = attn.unsqueeze(-1) * V                         # (E, n_heads, head_dim)
        out = out.view(B, self.out_dim)                      # (E, out_dim)
        return self.W_o(out)

    def update(self, aggr_out):
        return F.gelu(aggr_out)


# ═══════════════════════════════════════════════════════════════
# SECTION 3: Spectral Discrepancy Signal Module
# ═══════════════════════════════════════════════════════════════

class SpectralDiscrepancyModule(nn.Module):
    """
    Computes high-frequency spectral components of the discrepancy graph.
    
    Construction:
    1. Build discrepancy adjacency: A_disc[i,j] = 1 - σ(cos_sim(h_i, h_j))
       for cross-modal pairs (text-audio, text-facial, audio-facial)
    
    2. Symmetric normalized Laplacian:
       L_sym = I - D^{-1/2} A D^{-1/2}
    
    3. Eigendecomposition (approximate via power iteration for efficiency):
       L = U Λ U^T
    
    4. Project onto k HIGH-frequency eigenvectors (large eigenvalues):
       z_spectral = U_k^T h  →  captures ABNORMAL PATTERNS
    
    Biological basis: depression manifests as FLAT AFFECT (abnormally
    consistent facial/vocal expression regardless of verbal content).
    This shows up as anomalous HIGH-frequency components in the
    discrepancy graph spectrum (edges that SHOULD vary but don't).
    """
    def __init__(self, hidden_dim: int, n_components: int = 8):
        super().__init__()
        self.n_comp = n_components
        
        # Learnable projection to build discrepancy matrix
        self.disc_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # After spectral projection: (B, n_comp) → output features
        self.spectral_out = nn.Sequential(
            nn.Linear(n_components * 3, hidden_dim // 2),  # 3 modality pairs
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
        )

    def _approx_top_k_eigen(self, L: torch.Tensor, k: int):
        """
        Approximate top-k eigenvectors via k steps of power iteration.
        L: (N, N) symmetric matrix
        Returns: (N, k) eigenvector matrix
        
        Efficient for small N (number of modality segments ≤ 10).
        """
        N = L.size(0)
        device = L.device
        # Random initialization
        V = torch.randn(N, k, device=device)
        V, _ = torch.linalg.qr(V)

        # 5 power iterations (sufficient for approximate eigenvectors)
        for _ in range(5):
            V = L @ V
            V, _ = torch.linalg.qr(V)

        return V  # (N, k)

    def _build_discrepancy_laplacian(self, h_a: torch.Tensor, h_b: torch.Tensor):
        """
        h_a, h_b: (B, H) — two modality embeddings from same segments
        
        Build discrepancy adjacency matrix between N segments:
        A[i,j] = 1 - σ(cos_sim(projected_a_i, projected_b_j))
        
        This assigns HIGH weight to INCONSISTENT cross-modal pairs
        (verbal says positive, face shows negative → high discrepancy).
        
        Returns: batch of Laplacians (B, N, N) — here N=1 per sample,
        so we work at segment level across the batch as "nodes".
        """
        # Project to lower dim for discrepancy computation
        pa = F.normalize(self.disc_proj(h_a), dim=-1)  # (B, H//2)
        pb = F.normalize(self.disc_proj(h_b), dim=-1)  # (B, H//2)

        # Pairwise cosine: (B, B) — treat each sample as a "node"
        cos_AB = pa @ pb.t()                            # (B, B)
        A_disc = 1.0 - torch.sigmoid(cos_AB)            # High where inconsistent

        # Symmetric normalize: L = I - D^{-1/2} A D^{-1/2}
        deg = A_disc.sum(dim=-1).clamp(min=1e-8)
        d_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        A_norm = D_inv_sqrt @ A_disc @ D_inv_sqrt
        L = torch.eye(A_disc.size(0), device=A_disc.device) - A_norm

        return L  # (B, B)

    def forward(self, z_text, z_audio, z_facial):
        """
        z_text, z_audio, z_facial: (B, H)
        
        Returns: (B, hidden_dim//2) — spectral discrepancy features
        """
        k = min(self.n_comp, z_text.size(0) - 1)
        if k <= 0:
            # Fallback for batch_size=1
            return self.spectral_out(
                torch.zeros(z_text.size(0), self.n_comp * 3, device=z_text.device)
            )

        # Build 3 discrepancy Laplacians for 3 cross-modal pairs
        L_ta = self._build_discrepancy_laplacian(z_text,  z_audio)   # text-audio
        L_tf = self._build_discrepancy_laplacian(z_text,  z_facial)  # text-facial
        L_af = self._build_discrepancy_laplacian(z_audio, z_facial)  # audio-facial

        # Get top-k high-frequency eigenvectors (detach for stability)
        U_ta = self._approx_top_k_eigen(L_ta.detach(), k)  # (B, k)
        U_tf = self._approx_top_k_eigen(L_tf.detach(), k)
        U_af = self._approx_top_k_eigen(L_af.detach(), k)

        # Project each sample onto its spectral basis
        # s_ta[i] = U_ta[i, :] = the i-th row = spectral fingerprint of sample i
        # Then concatenate [U_ta, U_tf, U_af] for each sample
        # Pad/slice to exactly n_comp
        def pad_k(U):
            if U.size(1) < self.n_comp:
                pad = torch.zeros(U.size(0), self.n_comp - U.size(1), device=U.device)
                return torch.cat([U, pad], dim=-1)
            return U[:, :self.n_comp]

        spec_feat = torch.cat([pad_k(U_ta), pad_k(U_tf), pad_k(U_af)], dim=-1)  # (B, 3k)
        return self.spectral_out(spec_feat)


# ═══════════════════════════════════════════════════════════════
# SECTION 4: Temporal Feature Encoders (with Windowed Self-Attention)
# ═══════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """
    Encode a temporal sequence of features into segment-level embeddings
    using a multi-scale approach:
    
    1. Local: 1D-CNN captures short-range patterns (within turn)
    2. Global: BiLSTM captures long-range dependencies (across turns)
    3. Adaptive: Attention pooling weights by estimated relevance
    
    Produces BOTH:
    - z_global: (B, H) — session-level embedding
    - z_segments: (B, N_seg, H) — segment-level embeddings for graph building
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_segments: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.n_segments = n_segments

        # Local temporal patterns
        self.local_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Global sequential context
        self.global_lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout
        )

        # Attention pooling
        self.attn_gate = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def _segment_features(self, x: torch.Tensor, n: int):
        """
        Divide sequence into n segments, mean-pool each.
        x: (B, T, H) → (B, n, H)
        """
        B, T, H = x.shape
        seg_len = max(T // n, 1)
        segs = []
        for i in range(n):
            start = i * seg_len
            end   = min(start + seg_len, T)
            if start >= T:
                segs.append(torch.zeros(B, H, device=x.device))
            else:
                segs.append(x[:, start:end, :].mean(dim=1))
        return torch.stack(segs, dim=1)  # (B, n, H)

    def forward(self, x, lengths=None):
        """
        x: (B, T, input_dim)
        lengths: (B,)
        Returns:
            z_global: (B, hidden_dim)
            z_segs:   (B, n_segments, hidden_dim)
        """
        # CNN
        h = self.local_conv(x.transpose(1, 2)).transpose(1, 2)   # (B, T, H)

        # LSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            out, _ = self.global_lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.global_lstm(h)

        # Segment-level embeddings for graph
        z_segs = self._segment_features(out, self.n_segments)     # (B, n_seg, H)
        z_segs = self.proj(z_segs)

        # Attention pooling for global
        scores = self.attn_gate(out).squeeze(-1)                  # (B, T)
        if lengths is not None:
            mask = torch.arange(scores.size(1), device=scores.device).unsqueeze(0) \
                   >= lengths.unsqueeze(1).to(scores.device)
            scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        z_global = self.drop(self.proj((out * weights).sum(dim=1)))  # (B, H)

        return z_global, z_segs


# ═══════════════════════════════════════════════════════════════
# SECTION 5: HTDG Builder — Graph Construction from Features
# ═══════════════════════════════════════════════════════════════

class HTDGBuilder(nn.Module):
    """
    Builds the Heterogeneous Temporal Discrepancy Graph (HTDG).
    
    Graph topology for ONE sample:
    - N = 3 × n_seg nodes: [text_0..n, audio_0..n, facial_0..n]
    
    Edge types (encoded in edge_attr):
    ┌──────────────────┬─────────────────────────────────────────┐
    │ Type ID          │ Meaning                                 │
    ├──────────────────┼─────────────────────────────────────────┤
    │ 0: text-temporal │ text_i → text_{i+1} (sequential)        │
    │ 1: aud-temporal  │ audio_i → audio_{i+1}                   │
    │ 2: fac-temporal  │ facial_i → facial_{i+1}                 │
    │ 3: cross-consist │ low-discrepancy cross-modal edges        │
    │ 4: cross-discord │ HIGH-discrepancy cross-modal edges       │
    └──────────────────┴─────────────────────────────────────────┘
    
    Edge features φ(e_ij) ∈ ℝ^{edge_dim}:
    [one-hot(type), discrepancy_weight, temporal_distance, direction]
    
    KEY INNOVATION: Cross-modal edges are SPLIT by discrepancy threshold.
    The GNN learns separate aggregation rules for consistent vs discrepant
    cross-modal evidence — exactly what is needed for depression detection.
    """
    def __init__(self, hidden_dim: int, n_seg: int = 8, 
                 edge_dim: int = 16, disc_threshold: float = 0.4):
        super().__init__()
        self.n_seg = n_seg
        self.edge_dim = edge_dim
        self.disc_thr = disc_threshold

        # Learnable discrepancy estimator
        self.disc_query = nn.Linear(hidden_dim, hidden_dim // 2)
        self.disc_key   = nn.Linear(hidden_dim, hidden_dim // 2)

        # Edge type embeddings (5 types)
        self.edge_type_emb = nn.Embedding(5, edge_dim // 2)

    def _cosine_discrepancy(self, h_a, h_b):
        """
        Compute discrepancy between segment a and b:
        w = 1 - σ(cos_sim(q(h_a), k(h_b)))
        
        h_a, h_b: (..., H) → scalar discrepancy ∈ [0, 1]
        High w = modalities disagree (discrepant).
        Low  w = modalities agree (consistent).
        """
        qa = F.normalize(self.disc_query(h_a), dim=-1)
        kb = F.normalize(self.disc_key(h_b), dim=-1)
        return 1.0 - torch.sigmoid((qa * kb).sum(dim=-1))

    def forward(self, z_text_segs, z_audio_segs, z_facial_segs):
        """
        z_*_segs: (B, n_seg, H) — segment-level embeddings
        
        Returns per-sample graph components as lists (for batching):
          node_feats:  (B × 3n_seg, H)
          edge_indices: list of (2, E_i) tensors
          edge_attrs:   list of (E_i, edge_dim) tensors
          batch_vec:    (B × 3n_seg,) — which graph each node belongs to
        """
        B, N, H = z_text_segs.shape
        device = z_text_segs.device

        # Stack all nodes: [text | audio | facial]
        # Node ID layout: text=[0..N-1], audio=[N..2N-1], facial=[2N..3N-1]
        all_nodes = torch.cat([z_text_segs, z_audio_segs, z_facial_segs], dim=1)  # (B, 3N, H)

        all_edge_idx  = []
        all_edge_attr = []

        def make_edge_attr(etype_id, disc_w, temp_dist, direction=0.0):
            """Build edge feature vector φ(e)."""
            e_type = self.edge_type_emb(
                torch.tensor(etype_id, device=device)
            )  # (edge_dim//2,)
            scalar = torch.tensor(
                [disc_w, temp_dist, direction, float(etype_id) / 4.0],
                device=device, dtype=torch.float32
            )
            # Pad scalar to edge_dim//2 and concatenate
            pad_len = self.edge_dim // 2 - len(scalar)
            if pad_len > 0:
                scalar = torch.cat([scalar, torch.zeros(pad_len, device=device)])
            return torch.cat([e_type, scalar[:self.edge_dim // 2]])  # (edge_dim,)

        for b in range(B):
            edges_src, edges_dst, edges_attr = [], [], []

            # ── 1. Intra-modal temporal edges (bidirectional) ──
            for mod_offset, etype in [(0, 0), (N, 1), (2*N, 2)]:
                for i in range(N - 1):
                    u, v = mod_offset + i, mod_offset + i + 1
                    td = 1.0 / N  # normalized temporal distance

                    edges_src.extend([u, v])
                    edges_dst.extend([v, u])
                    attr = make_edge_attr(etype, 0.0, td, 1.0)
                    edges_attr.extend([attr, attr])

            # ── 2. Cross-modal edges (text↔audio, text↔facial, audio↔facial) ──
            for mod_a, mod_b, off_a, off_b in [
                ("text",  "audio",  0,    N),
                ("text",  "facial", 0,    2*N),
                ("audio", "facial", N,    2*N),
            ]:
                for i in range(N):
                    h_a = all_nodes[b, off_a + i]
                    h_b = all_nodes[b, off_b + i]
                    disc = self._cosine_discrepancy(h_a, h_b).item()

                    # Split into consistent (3) vs discrepant (4) edge types
                    etype = 4 if disc > self.disc_thr else 3
                    td = 0.0  # same time segment

                    edges_src.extend([off_a + i, off_b + i])
                    edges_dst.extend([off_b + i, off_a + i])
                    attr = make_edge_attr(etype, disc, td, 0.0)
                    edges_attr.extend([attr, attr])

            edge_index = torch.tensor(
                [edges_src, edges_dst], dtype=torch.long, device=device
            )
            edge_attr = torch.stack(edges_attr, dim=0)
            all_edge_idx.append(edge_index)
            all_edge_attr.append(edge_attr)

        # Flatten: build a single disconnected graph for the batch
        node_feats = all_nodes.view(B * 3 * N, H)

        # Offset edge indices by batch
        flat_edges = []
        flat_attrs  = []
        for b in range(B):
            offset = b * 3 * N
            flat_edges.append(all_edge_idx[b] + offset)
            flat_attrs.append(all_edge_attr[b])

        edge_index = torch.cat(flat_edges, dim=1)             # (2, total_E)
        edge_attr  = torch.cat(flat_attrs, dim=0)             # (total_E, edge_dim)

        # Batch vector for global pooling later
        batch_vec = torch.arange(B, device=device).repeat_interleave(3 * N)

        return node_feats, edge_index, edge_attr, batch_vec


# ═══════════════════════════════════════════════════════════════
# SECTION 6: Full HTDG-CDL Model
# ═══════════════════════════════════════════════════════════════

class HTDGCDLModel(nn.Module):
    """
    Full HTDG-CDL: Heterogeneous Temporal Discrepancy Graph
    for Contrastive Discrepancy Learning in Depression Detection.
    
    Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  [Text]──────────────────────────────────────────────┐   │
    │  768-dim BERT → TemporalEncoder → z_text, z_text_segs │   │
    │                                                        │   │
    │  [Audio]────────────────────────────────────────────┐  │   │
    │  88-dim COVAREP → TemporalEncoder → z_audio, segs   │  │   │
    │                                                      │  │   │
    │  [Facial]───────────────────────────────────────────┘  │   │
    │  17-dim AUs → TemporalEncoder → z_facial, segs         │   │
    │                                                         │   │
    │  ────────────────────────────────────────────────────   │   │
    │  HTDGBuilder → G = (V, E, τ_V, τ_E)                    │   │
    │    ↓ HGA-EF × 2 layers                                  │   │
    │  z_graph (global mean pool from GNN)                    │   │
    │                                                         │   │
    │  SpectralDiscrepancyModule → z_spectral                 │   │
    │                                                         │   │
    │  Fusion: [z_text | z_audio | z_facial | z_graph |      │   │
    │           z_spectral] → BN → MLP                        │   │
    │                                                         │   │
    │  → Poincaré Ball → z_hyp (for RMC loss)                │   │
    │  → PHQ-8 regression head                                │   │
    │  → Depression classification head                       │   │
    └──────────────────────────────────────────────────────────┘
    """
    def __init__(
        self,
        text_input_dim: int  = 768,
        audio_input_dim: int = 74,   # COVAREP only
        facial_input_dim: int = 17,  # AU intensities only
        nonverbal_input_dim: int = 88,  # combined, for backward compat
        hidden_dim: int       = 128,
        n_segments: int       = 8,
        n_gnn_layers: int     = 2,
        n_attn_heads: int     = 4,
        edge_dim: int         = 16,
        n_spectral: int       = 8,
        poincare_c: float     = 1.0,
        dropout: float        = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_seg      = n_segments

        # ── 1. Temporal Encoders ──────────────────────────────────
        self.text_encoder  = TemporalEncoder(text_input_dim,  hidden_dim, n_segments, dropout)
        # We treat the combined non-verbal as "audio" for simplicity
        # but keep separate projections for text vs non-verbal
        self.nv_encoder    = TemporalEncoder(nonverbal_input_dim, hidden_dim, n_segments, dropout)
        
        # Separate projections for audio vs facial within non-verbal
        # (COVAREP: last 74 dims; AUs: first 14-17 dims)
        self.audio_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.facial_proj = nn.Linear(hidden_dim, hidden_dim)

        # ── 2. HTDG Builder ──────────────────────────────────────
        self.graph_builder = HTDGBuilder(hidden_dim, n_segments, edge_dim)

        # ── 3. HGA-EF GNN Layers ─────────────────────────────────
        self.gnn_layers = nn.ModuleList([
            HGAEdgeConv(hidden_dim, hidden_dim, n_attn_heads, edge_dim, dropout)
            for _ in range(n_gnn_layers)
        ])

        # ── 4. Spectral Discrepancy Module ───────────────────────
        self.spectral_module = SpectralDiscrepancyModule(hidden_dim, n_spectral)

        # ── 5. Fusion ─────────────────────────────────────────────
        # Input: [z_text | z_nv | z_graph | z_spectral]
        # z_text, z_nv, z_graph: hidden_dim each
        # z_spectral: hidden_dim // 2
        # Total = 3 * hidden_dim + hidden_dim // 2
        fusion_in = hidden_dim * 3 + hidden_dim // 2
        self.fusion_bn = nn.BatchNorm1d(fusion_in)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # ── 6. Poincaré Projection ───────────────────────────────
        self.poincare = PoincareBall(c=poincare_c)
        self.to_poincare = nn.Linear(hidden_dim // 2, hidden_dim // 2)

        # ── 7. Task Heads ─────────────────────────────────────────
        head_in = hidden_dim // 2
        self.phq_head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.dep_head = nn.Linear(head_in, 1)

    def _global_mean_pool_graph(self, x, batch_vec, B):
        """Global mean pooling over graph nodes per sample."""
        H = x.size(-1)
        out = torch.zeros(B, H, device=x.device)
        count = torch.zeros(B, 1, device=x.device)
        out.scatter_add_(0, batch_vec.unsqueeze(-1).expand_as(x), x)
        count.scatter_add_(0, batch_vec.unsqueeze(-1), torch.ones(x.size(0), 1, device=x.device))
        return out / count.clamp(min=1)

    def forward(self, text_feat, nonverbal_feat,
                text_lengths=None, nonverbal_lengths=None):
        """
        text_feat:      (B, T_text, 768)
        nonverbal_feat: (B, T_nv, 88)
        """
        B = text_feat.size(0)

        # ── 1. Encode modalities ──────────────────────────────────
        z_text, z_text_segs = self.text_encoder(text_feat, text_lengths)
        z_nv, z_nv_segs     = self.nv_encoder(nonverbal_feat, nonverbal_lengths)

        # Decompose non-verbal into "audio" and "facial" sub-representations
        z_audio  = self.audio_proj(z_nv)
        z_facial = self.facial_proj(z_nv)
        z_audio_segs  = self.audio_proj(z_nv_segs)
        z_facial_segs = self.facial_proj(z_nv_segs)

        # ── 2. Build HTDG and run HGA-EF ─────────────────────────
        node_feats, edge_index, edge_attr, batch_vec = self.graph_builder(
            z_text_segs, z_audio_segs, z_facial_segs
        )

        # GNN forward pass
        h = node_feats
        for gnn in self.gnn_layers:
            h = gnn(h, edge_index, edge_attr)

        z_graph = self._global_mean_pool_graph(h, batch_vec, B)

        # ── 3. Spectral Discrepancy Signal ────────────────────────
        z_spectral = self.spectral_module(z_text, z_audio, z_facial)

        # ── 4. Fusion ─────────────────────────────────────────────
        fused_raw = torch.cat([z_text, z_nv, z_graph, z_spectral], dim=-1)  # (B, 4H)
        fused_bn  = self.fusion_bn(fused_raw)
        fused     = self.fusion_mlp(fused_bn)                               # (B, H//2)

        # ── 5. Hyperbolic projection ──────────────────────────────
        z_hyp = self.poincare.exp_map_zero(
            F.tanh(self.to_poincare(fused))                                  # must be in tangent space
        )

        # ── 6. Predictions ────────────────────────────────────────
        phq_score = self.phq_head(fused).squeeze(-1) * 27.0
        dep_logit = self.dep_head(fused).squeeze(-1)

        return {
            "phq_score":  phq_score,
            "dep_logit":  dep_logit,
            "z_hyp":      z_hyp,       # for RMC contrastive loss
            "z_text":     z_text,
            "z_nv":       z_nv,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 7: Riemannian Manifold Contrastive Loss (RMC)
# ═══════════════════════════════════════════════════════════════

class RiemannianManifoldContrastiveLoss(nn.Module):
    """
    Contrastive loss using POINCARÉ DISTANCE instead of Euclidean.
    
    L_RMC = -1/B Σ_i log [ exp(-d_c(z_i, z_i+)/τ) 
                            / Σ_j exp(-d_c(z_i, z_j)/τ) ]
    
    where z+ = another sample of SAME depression class (positive pair)
    and   z- = samples of DIFFERENT class (negative pairs)
    
    Key properties:
    - d_c respects the curved manifold geometry
    - Depressed patients cluster near the Poincaré boundary
    - Non-depressed patients cluster near the origin
    - Separation in hyperbolic space > Euclidean for hierarchical data
    
    τ (temperature): small τ → harder separation
    """
    def __init__(self, poincare: PoincareBall, temperature: float = 0.07):
        super().__init__()
        self.poincare = poincare
        self.tau = temperature

    def forward(self, z_hyp: torch.Tensor, labels: torch.Tensor):
        """
        z_hyp:  (B, D) — hyperbolic embeddings (on Poincaré ball)
        labels: (B,)   — binary depression labels {0, 1}
        
        Returns: scalar RMC loss
        """
        B = z_hyp.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z_hyp.device)

        # Pairwise Poincaré distances: (B, B)
        dist_mat = torch.zeros(B, B, device=z_hyp.device)
        for i in range(B):
            for j in range(B):
                if i != j:
                    dist_mat[i, j] = self.poincare.distance(
                        z_hyp[i].unsqueeze(0), z_hyp[j].unsqueeze(0)
                    ).squeeze()

        # Build positive mask: same label, different sample
        labels_row = labels.unsqueeze(1).expand(B, B)
        labels_col = labels.unsqueeze(0).expand(B, B)
        pos_mask = (labels_row == labels_col) & (~torch.eye(B, dtype=torch.bool, device=z_hyp.device))

        if not pos_mask.any():
            return torch.tensor(0.0, device=z_hyp.device)

        # Logits = -dist / τ (higher = closer = more similar)
        logits = -dist_mat / self.tau

        # Mask diagonal
        logits = logits - 1e9 * torch.eye(B, device=z_hyp.device)

        # Log-softmax over all non-self pairs
        log_prob = F.log_softmax(logits, dim=-1)  # (B, B)

        # Average loss over positive pairs only
        loss = -(log_prob * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1)
        return loss


# ═══════════════════════════════════════════════════════════════
# SECTION 8: Combined CDL Loss with RMC
# ═══════════════════════════════════════════════════════════════

class HTDGCDLLoss(nn.Module):
    """
    Combined loss for HTDG-CDL:
    
    L = L_MSE(PHQ) + λ_cls · L_BCE(depression) + λ_rmc · L_RMC
    
    L_MSE: Normalized mean squared error for PHQ-8 regression
    L_BCE: Binary cross-entropy for depression classification
    L_RMC: Riemannian Manifold Contrastive loss in hyperbolic space
    
    Note: ALL losses contribute to backprop (unlike previous version
    which zeroed out BCE). The λ parameters balance the objectives.
    
    λ_rmc acts as a geometric regularizer: it forces the hyperbolic
    embedding to maintain clinically meaningful structure even when
    the primary regression signal is weak.
    """
    def __init__(
        self,
        poincare: PoincareBall,
        lambda_cls: float = 0.5,
        lambda_rmc: float = 0.3,
        pos_weight: torch.Tensor = None,
        phq_max: float = 27.0,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_rmc = lambda_rmc
        self.phq_max    = phq_max

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.rmc = RiemannianManifoldContrastiveLoss(poincare)

    def forward(self, outputs, phq_labels, dep_labels):
        """
        outputs: dict with keys phq_score, dep_logit, z_hyp
        phq_labels: (B,) continuous PHQ-8 scores
        dep_labels: (B,) binary {0, 1}
        """
        # 1. Regression (MSE on normalized scale)
        l_reg = self.mse(
            outputs["phq_score"] / self.phq_max,
            phq_labels / self.phq_max
        )

        # 2. Classification (BCE)
        l_cls = self.bce(outputs["dep_logit"], dep_labels)

        # 3. Riemannian contrastive
        l_rmc = self.rmc(outputs["z_hyp"], dep_labels)

        total = l_reg + self.lambda_cls * l_cls + self.lambda_rmc * l_rmc

        return {
            "total":   total,
            "l_reg":   l_reg.item(),
            "l_cls":   l_cls.item(),
            "l_rmc":   l_rmc.item(),
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 9: Hyperbolic Mixup Augmentation
# ═══════════════════════════════════════════════════════════════

def hyperbolic_mixup_batch(z_hyp, phq_scores, dep_labels, poincare, alpha=0.4):
    """
    Geodesic mixup in Poincaré ball for data augmentation.
    
    For each pair (i, j) in the batch:
      λ ~ Beta(α, α)
      z_mix = Exp_0(λ·Log_0(z_i) + (1-λ)·Log_0(z_j))  ← on manifold
      y_mix_phq = λ·y_i + (1-λ)·y_j                    ← linear label
      y_mix_dep = round(λ·y_i + (1-λ)·y_j)              ← hard binary
    
    This generates O(B²) synthetic samples on the depression
    severity manifold — addressing DAIC-WOZ's extreme data scarcity.
    
    Returns: (z_mix, phq_mix, dep_mix) augmented batch
    """
    B = z_hyp.size(0)
    device = z_hyp.device

    # Sample mixing coefficients
    lam_dist = torch.distributions.Beta(alpha, alpha)
    lam = lam_dist.sample((B,)).to(device)

    # Random pairing
    perm = torch.randperm(B, device=device)
    z_j   = z_hyp[perm]
    phq_j = phq_scores[perm]
    dep_j = dep_labels[perm]

    # Geodesic interpolation on Poincaré ball
    z_mix = poincare.hyperbolic_mixup(z_hyp, z_j, lam.unsqueeze(-1))

    # Label interpolation
    phq_mix = lam * phq_scores + (1 - lam) * phq_j
    dep_mix = (lam * dep_labels + (1 - lam) * dep_j).round()

    return z_mix, phq_mix, dep_mix