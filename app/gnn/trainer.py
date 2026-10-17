"""
GNN Training Module
===================
Trains the GCN recommender using pairwise BPR (Bayesian Personalised Ranking)
loss on synthetic learner trajectories generated from the knowledge graph.

Training strategy:
  For each concept v⁺ that is topologically next and unmastered, we sample
  a negative concept v⁻ (ineligible or already mastered) and minimise:
    L = -log σ(s(u, v⁺) - s(u, v⁻))

This module is called:
  1. On-demand via POST /gnn/train
  2. At evaluation time by the Streamlit dashboard
  3. In deterministic offline mode for paper metrics generation

Returns training history for loss curve plotting.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from app.graph.schema import KnowledgeGraph, MasteryState
from app.gnn.graph_constructor import build_pyg_data, NODE_FEATURE_DIM
from app.gnn.model import build_model, NumpyGCN
from app.gnn.inference import _model_cache

try:
    import torch
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic trajectory generation
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_trajectories(
    kg: KnowledgeGraph,
    n_learners: int = 50,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """
    Generate synthetic (learner_state, positive_concept, negative_concept)
    triples for BPR training.

    Learner state is represented as a mastery dict.
    Positive concept = topologically next unmastered node.
    Negative concept = random locked/mastered node.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    g = nx.DiGraph()
    node_ids = [n.id for n in kg.nodes]
    for n in kg.nodes:
        g.add_node(n.id, difficulty=n.difficulty)
    for e in kg.edges:
        g.add_edge(e.source, e.target)

    topo_order = list(nx.topological_sort(g))
    triples: List[Tuple[str, List[str], str, str]] = []

    for _ in range(n_learners):
        # Simulate learner who has mastered a random prefix of topo_order
        n_mastered = rng.randint(0, max(0, len(topo_order) - 2))
        mastered   = set(topo_order[:n_mastered])
        mastery_dict = {nid: (rng.uniform(0.7, 1.0) if nid in mastered else
                               rng.uniform(0.0, 0.4))
                         for nid in node_ids}

        # Find positive: first unmastered node that is eligible
        pos = None
        for nid in topo_order:
            if nid in mastered:
                continue
            preds = list(g.predecessors(nid))
            if all(mastery_dict.get(p, 0.0) >= 0.7 for p in preds):
                pos = nid
                break
        if pos is None:
            continue

        # Find negative: any node that is NOT the positive
        neg_pool = [nid for nid in node_ids if nid != pos]
        if not neg_pool:
            continue
        neg = rng.choice(neg_pool)

        mastery_vec = [mastery_dict.get(nid, 0.0) for nid in node_ids]
        triples.append((mastery_vec, pos, neg))

    return triples


# ══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_gnn(
    kg: KnowledgeGraph,
    session_id: str,
    epochs: int = 60,
    lr: float = 5e-3,
    n_learners: int = 80,
    embed_dim: int = 16,
    hidden_dim: int = 32,
    arch: str = "gcn",
    seed: int = 42,
    val_split: float = 0.2,
) -> Dict[str, Any]:
    """
    Train the GCN recommender on synthetic trajectories.

    Returns
    -------
    {
        "train_loss": [float, ...],
        "val_loss":   [float, ...],
        "epochs":     int,
        "arch":       str,
        "final_train_loss": float,
        "final_val_loss":   float,
        "training_time_s":  float,
    }
    """
    t0 = time.time()
    np.random.seed(seed)
    random.seed(seed)

    node_ids = [n.id for n in kg.nodes]
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    n_concepts = len(node_ids)

    # Build PyG data (without mastery = all zeros)
    data = build_pyg_data(kg, mastery=None)

    # Generate synthetic training triples
    triples = _simulate_trajectories(kg, n_learners=n_learners, seed=seed)
    if not triples:
        return {
            "train_loss": [], "val_loss": [], "epochs": 0, "arch": arch,
            "final_train_loss": 0.0, "final_val_loss": 0.0,
            "training_time_s": 0.0, "warning": "No training triples generated"
        }

    random.shuffle(triples)
    split_idx    = max(1, int(len(triples) * (1 - val_split)))
    train_triples = triples[:split_idx]
    val_triples   = triples[split_idx:]

    train_history: List[float] = []
    val_history:   List[float] = []

    if _TORCH_AVAILABLE and not isinstance(
        build_model(arch=arch, seed=seed), NumpyGCN
    ):
        # ── PyTorch training ──────────────────────────────────────────────
        torch.manual_seed(seed)
        model = build_model(
            in_channels=NODE_FEATURE_DIM,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            arch=arch,
            seed=seed,
        )
        W_r = torch.nn.Linear(n_concepts, embed_dim, bias=False)
        b   = torch.nn.Parameter(torch.zeros(n_concepts))

        all_params = list(model.parameters()) + list(W_r.parameters()) + [b]
        optimizer  = optim.Adam(all_params, lr=lr, weight_decay=1e-5)

        x_t  = data.x
        ei_t = data.edge_index

        def _bpr_loss(triples_subset):
            model.train()
            H = model(x_t, ei_t)  # N × embed_dim
            total_loss = torch.tensor(0.0, requires_grad=True)
            for (mastery_vec, pos_id, neg_id) in triples_subset:
                q_u  = torch.tensor(mastery_vec, dtype=torch.float32)
                q_proj = W_r(q_u)  # embed_dim

                i_pos = node_idx[pos_id]
                i_neg = node_idx[neg_id]

                s_pos = (q_proj * H[i_pos]).sum() + b[i_pos]
                s_neg = (q_proj * H[i_neg]).sum() + b[i_neg]
                loss  = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8)
                total_loss = total_loss + loss
            return total_loss / max(len(triples_subset), 1)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = _bpr_loss(train_triples)
            loss.backward()
            optimizer.step()
            train_history.append(float(loss.item()))

            if val_triples:
                model.eval()
                with torch.no_grad():
                    v_loss = _bpr_loss(val_triples)
                val_history.append(float(v_loss.item()))
            else:
                val_history.append(train_history[-1])

        # Store trained model
        _model_cache[session_id] = model

        # Export weights to numpy for fallback serialisation
        W1_np = model.conv1.lin.weight.detach().numpy().T  # in × hidden
        W2_np = model.conv2.lin.weight.detach().numpy().T  # hidden × embed

    else:
        # ── Numpy fallback: gradient-free perturbation training ──────────
        model = NumpyGCN(
            in_dim=NODE_FEATURE_DIM,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            seed=seed,
        )

        x_np  = data["x"]        if isinstance(data, dict) else data.x.numpy()
        ei_np = data["edge_index"] if isinstance(data, dict) else data.edge_index.numpy()

        rng = np.random.default_rng(seed)
        scale_r = math.sqrt(2.0 / (n_concepts + embed_dim))
        W_r = rng.normal(0, scale_r, (n_concepts, embed_dim)).astype(np.float32)

        best_loss = float("inf")
        best_W1   = model.W1.copy()
        best_W2   = model.W2.copy()

        for epoch in range(epochs):
            H = model.forward(x_np, ei_np)

            total_loss = 0.0
            for (mastery_vec, pos_id, neg_id) in train_triples:
                q = np.array(mastery_vec, dtype=np.float32)
                q_proj = q @ W_r
                i_pos = node_idx[pos_id]
                i_neg = node_idx[neg_id]
                s_pos = float(np.dot(q_proj, H[i_pos]))
                s_neg = float(np.dot(q_proj, H[i_neg]))
                total_loss += -math.log(_sigmoid(s_pos - s_neg) + 1e-8)

            avg_loss = total_loss / max(len(train_triples), 1)
            train_history.append(avg_loss)

            # Perturbation step
            noise1 = np.random.normal(0, lr, model.W1.shape).astype(np.float32)
            noise2 = np.random.normal(0, lr, model.W2.shape).astype(np.float32)
            noise_r = np.random.normal(0, lr, W_r.shape).astype(np.float32)
            model_tmp = NumpyGCN(NODE_FEATURE_DIM, hidden_dim, embed_dim, seed)
            model_tmp.load_weights(model.W1 + noise1, model.W2 + noise2)
            W_r_tmp = W_r + noise_r
            H_tmp = model_tmp.forward(x_np, ei_np)

            trial_loss = 0.0
            for (mastery_vec, pos_id, neg_id) in train_triples:
                q = np.array(mastery_vec, dtype=np.float32)
                q_proj = q @ W_r_tmp
                i_pos = node_idx[pos_id]
                i_neg = node_idx[neg_id]
                s_pos = float(np.dot(q_proj, H_tmp[i_pos]))
                s_neg = float(np.dot(q_proj, H_tmp[i_neg]))
                trial_loss += -math.log(_sigmoid(s_pos - s_neg) + 1e-8)
            trial_loss /= max(len(train_triples), 1)

            if trial_loss < avg_loss:
                model.load_weights(model.W1 + noise1, model.W2 + noise2)
                W_r = W_r_tmp

            # Validation
            if val_triples:
                H_val = model.forward(x_np, ei_np)
                v_total = 0.0
                for (mastery_vec, pos_id, neg_id) in val_triples:
                    q = np.array(mastery_vec, dtype=np.float32)
                    q_proj = q @ W_r
                    i_pos = node_idx[pos_id]
                    i_neg = node_idx[neg_id]
                    s_pos = float(np.dot(q_proj, H_val[i_pos]))
                    s_neg = float(np.dot(q_proj, H_val[i_neg]))
                    v_total += -math.log(_sigmoid(s_pos - s_neg) + 1e-8)
                val_history.append(v_total / max(len(val_triples), 1))
            else:
                val_history.append(avg_loss)

        _model_cache[session_id] = model

    elapsed = time.time() - t0
    return {
        "train_loss":        train_history,
        "val_loss":          val_history,
        "epochs":            epochs,
        "arch":              arch,
        "final_train_loss":  round(train_history[-1] if train_history else 0.0, 6),
        "final_val_loss":    round(val_history[-1]   if val_history   else 0.0, 6),
        "training_time_s":   round(elapsed, 2),
        "n_triples":         len(triples),
        "n_train":           len(train_triples),
        "n_val":             len(val_triples),
    }
