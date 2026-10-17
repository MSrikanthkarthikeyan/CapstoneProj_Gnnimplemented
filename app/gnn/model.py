"""
GNN Model Architecture
=======================
Implements a two-layer GCN (Graph Convolutional Network) with an optional
GAT (Graph Attention) variant for prerequisite-aware representation learning.

Architecture:
  Input  : X ∈ R^{N×7}  (node features from graph_constructor)
  Layer 1: GCN/GAT → σ(AXW₁)   ∈ R^{N×hidden_dim}
  Layer 2: GCN/GAT → AXW₂       ∈ R^{N×embed_dim}
  Output : h_v ∈ R^{embed_dim}  (concept embeddings)

Scoring head:
  s(u, v) = q_u^T W_r h_v + b_v
  where q_u is the learner's mastery vector (dimension = n_concepts).

Supports both PyTorch Geometric (full GPU/CPU) and a pure-numpy fallback
that implements the spectral GCN convolution in closed form.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  PyTorch Geometric Model
# ══════════════════════════════════════════════════════════════════════════════
if _TORCH_AVAILABLE:
    class GCNRecommender(nn.Module):
        """
        Two-layer GCN for prerequisite-aware concept embedding.
        Forward returns node embeddings h ∈ R^{N × embed_dim}.
        """

        def __init__(
            self,
            in_channels: int = 7,
            hidden_dim:  int = 32,
            embed_dim:   int = 16,
            dropout:     float = 0.3,
        ):
            super().__init__()
            self.conv1   = GCNConv(in_channels, hidden_dim)
            self.conv2   = GCNConv(hidden_dim,  embed_dim)
            self.dropout = dropout
            self.embed_dim = embed_dim

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv2(h, edge_index)
            return h  # N × embed_dim

    class GATRecommender(nn.Module):
        """
        Two-layer GAT (Graph Attention Network) variant.
        Concatenates attention heads in layer-1, averages in layer-2.
        """

        def __init__(
            self,
            in_channels: int = 7,
            hidden_dim:  int = 16,
            embed_dim:   int = 16,
            heads:       int = 4,
            dropout:     float = 0.3,
        ):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_dim,
                                 heads=heads, dropout=dropout, concat=True)
            self.conv2 = GATConv(hidden_dim * heads, embed_dim,
                                 heads=1, dropout=dropout, concat=False)
            self.dropout  = dropout
            self.embed_dim = embed_dim

        def forward(self, x, edge_index):
            h = F.elu(self.conv1(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv2(h, edge_index)
            return h


    class ScoringHead(nn.Module):
        """
        Learner-concept scoring: s(u,v) = q_u^T W_r h_v + b_v
        """

        def __init__(self, n_concepts: int, embed_dim: int):
            super().__init__()
            self.W_r = nn.Linear(n_concepts, embed_dim, bias=False)
            self.b   = nn.Parameter(torch.zeros(n_concepts))

        def forward(self, learner_state: torch.Tensor, embeddings: torch.Tensor):
            """
            learner_state : (n_concepts,)   – mastery scores 0-1
            embeddings    : (N, embed_dim)  – GNN output
            Returns scores: (N,)
            """
            q_proj = self.W_r(learner_state.unsqueeze(0))  # (1, embed_dim)
            scores = (q_proj @ embeddings.T).squeeze(0)     # (N,)
            scores = scores + self.b
            return scores


# ══════════════════════════════════════════════════════════════════════════════
#  Pure-NumPy GCN fallback  (no PyTorch / PyG required)
# ══════════════════════════════════════════════════════════════════════════════
class NumpyGCN:
    """
    Spectral GCN in closed-form numpy:
        H^(l+1) = σ( D̃^{-1/2} Ã D̃^{-1/2} H^(l) W^(l) )
    Two layers with random-init weights (frozen – used for inference only).
    For training, swap to the PyG model above.
    """

    def __init__(
        self,
        in_dim:     int = 7,
        hidden_dim: int = 32,
        embed_dim:  int = 16,
        seed:       int = 42,
    ):
        rng = np.random.default_rng(seed)
        scale1 = math.sqrt(2.0 / (in_dim + hidden_dim))
        scale2 = math.sqrt(2.0 / (hidden_dim + embed_dim))
        self.W1 = rng.normal(0, scale1, (in_dim,     hidden_dim)).astype(np.float32)
        self.W2 = rng.normal(0, scale2, (hidden_dim, embed_dim )).astype(np.float32)
        self.embed_dim = embed_dim

    def _normalised_adj(self, edge_index: np.ndarray, n: int) -> np.ndarray:
        A = np.zeros((n, n), dtype=np.float32)
        if edge_index.shape[1] > 0:
            A[edge_index[0], edge_index[1]] = 1.0
        A_tilde = A + np.eye(n, dtype=np.float32)
        D       = np.diag(A_tilde.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-8)))
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def forward(self, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        A_norm = self._normalised_adj(edge_index, n)
        H1 = np.maximum(A_norm @ x  @ self.W1, 0)   # ReLU layer 1
        H2 = A_norm @ H1 @ self.W2                   # linear layer 2
        return H2

    def load_weights(self, W1: np.ndarray, W2: np.ndarray):
        self.W1 = W1.astype(np.float32)
        self.W2 = W2.astype(np.float32)


def build_model(
    in_channels: int = 7,
    hidden_dim:  int = 32,
    embed_dim:   int = 16,
    arch:        str = "gcn",   # "gcn" | "gat"
    seed:        int = 42,
) -> "GCNRecommender | GATRecommender | NumpyGCN":
    """Factory: returns a PyG model if torch is available, else NumpyGCN."""
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if arch == "gat":
            return GATRecommender(in_channels, hidden_dim, embed_dim)
        return GCNRecommender(in_channels, hidden_dim, embed_dim)
    return NumpyGCN(in_channels, hidden_dim, embed_dim, seed=seed)
