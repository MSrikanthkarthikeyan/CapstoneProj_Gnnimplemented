"""
GNN Inference Engine  –  Scoring & Path Ranking
================================================
Composite scoring function:

    Score(u, v) = α · PrerequisiteReadiness(u, v)
                + β · TargetRelevance(v, goal_embedding)
                + γ · MarginalInformationGain(u, v)

Where:
  PrerequisiteReadiness(u, v) = min mastery of v's direct prerequisites
  TargetRelevance(v)          = cosine similarity of h_v to goal direction
  MarginalInformationGain     = (1 - mastery_v) · difficulty_norm_v

The pipeline:
  1. build_pyg_data()   → PyG Data / numpy dict
  2. model.forward()    → node embeddings H
  3. score_concepts()   → ranked list of (node_id, score, breakdown)
  4. filter_eligible()  → remove locked / mastered nodes
  5. rank_path()        → final ordered recommendation

This module is the single source of truth consumed by the FastAPI /recommend
endpoint and the orchestrator.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from app.graph.schema import KnowledgeGraph, MasteryState
from app.gnn.graph_constructor import build_pyg_data, NODE_FEATURE_DIM
from app.gnn.model import build_model, NumpyGCN

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Composite score weights
ALPHA = 0.40   # Prerequisite Readiness
BETA  = 0.35   # Target Relevance
GAMMA = 0.25   # Marginal Information Gain

# Singleton model cache {session_id -> (model, embeddings)}
_model_cache: Dict[str, Any] = {}


def _get_embeddings(
    data: Any,
    model: Any,
    node_ids: List[str],
) -> np.ndarray:
    """Run forward pass and return numpy array H ∈ R^{N × embed_dim}."""
    if _TORCH_AVAILABLE and not isinstance(model, NumpyGCN):
        model.eval()
        with torch.no_grad():
            H = model(data.x, data.edge_index).numpy()
    else:
        X = data["x"] if isinstance(data, dict) else data.x.numpy()
        ei = (
            data["edge_index"]
            if isinstance(data, dict)
            else data.edge_index.numpy()
        )
        H = model.forward(X, ei)
    return H.astype(np.float32)


def _prerequisite_readiness(
    node_id: str,
    g: nx.DiGraph,
    mastery: MasteryState,
    threshold: float = 0.0,
) -> float:
    """
    Returns mean mastery of direct prerequisites.
    If no prerequisites → 1.0 (node is a root, always ready).
    """
    preds = list(g.predecessors(node_id))
    if not preds:
        return 1.0
    scores = [mastery.node_scores.get(p, 0.0) for p in preds]
    return float(np.mean(scores))


def _target_relevance(h_v: np.ndarray, goal_dir: np.ndarray) -> float:
    """Cosine similarity of concept embedding to goal direction vector."""
    norm_v    = np.linalg.norm(h_v)    + 1e-8
    norm_goal = np.linalg.norm(goal_dir) + 1e-8
    return float(np.dot(h_v, goal_dir) / (norm_v * norm_goal))


def _marginal_gain(
    node_id: str,
    kg: KnowledgeGraph,
    mastery: MasteryState,
) -> float:
    """
    Marginal information gain ≈ (1 - mastery) × difficulty_norm
    High for hard, unmastered concepts.
    """
    node = next((n for n in kg.nodes if n.id == node_id), None)
    if node is None:
        return 0.0
    current_mastery = mastery.node_scores.get(node_id, 0.0)
    difficulty_norm = node.difficulty / 5.0
    return float((1.0 - current_mastery) * difficulty_norm)


def run_gnn_inference(
    kg: KnowledgeGraph,
    mastery: MasteryState,
    session_id: str,
    force_refresh: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Core GNN inference pipeline.

    Returns a ranked list of concept recommendations:
    [
      {
        "node_id": str,
        "title": str,
        "score": float,
        "prerequisite_readiness": float,
        "target_relevance": float,
        "marginal_gain": float,
        "status": str,
        "eligible": bool,
      },
      ...
    ]
    Ordered by composite score descending; ineligible nodes appended last.
    """
    np.random.seed(seed)

    # ── 1. Build graph ───────────────────────────────────────────────────
    data = build_pyg_data(kg, mastery)
    node_ids: List[str] = (
        data.node_ids if not isinstance(data, dict) else data["node_ids"]
    )
    node_idx: Dict[str, int] = (
        data.node_idx if not isinstance(data, dict) else data["node_idx"]
    )

    # ── 2. Get / cache model ─────────────────────────────────────────────
    cache_key = session_id
    if force_refresh or cache_key not in _model_cache:
        model = build_model(
            in_channels=NODE_FEATURE_DIM,
            hidden_dim=32,
            embed_dim=16,
            arch="gcn",
            seed=seed,
        )
        _model_cache[cache_key] = model
    model = _model_cache[cache_key]

    # ── 3. Forward pass ──────────────────────────────────────────────────
    H = _get_embeddings(data, model, node_ids)  # N × 16

    # Goal direction = mean of all concept embeddings (proxy for goal space)
    goal_dir = H.mean(axis=0)

    # ── 4. Build networkx graph for topology queries ──────────────────────
    g = nx.DiGraph()
    for nid in node_ids:
        g.add_node(nid)
    for edge in kg.edges:
        g.add_edge(edge.source, edge.target)

    # ── 5. Score every concept ────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    for node in kg.nodes:
        nid = node.id
        i   = node_idx[nid]
        h_v = H[i]

        pr  = _prerequisite_readiness(nid, g, mastery)
        tr  = _target_relevance(h_v, goal_dir)
        mg  = _marginal_gain(nid, kg, mastery)

        # Normalise target relevance from [-1,1] → [0,1]
        tr_norm = (tr + 1.0) / 2.0

        # Fuse GNN embedding relevance with BM25 & TF-IDF semantic relevance from Phase B/Ranking
        bm25_s = getattr(node, "bm25_score", 0.0)
        tfidf_s = getattr(node, "tfidf_score", 0.0)
        res_q = getattr(node, "resource_quality", 0.0)

        if bm25_s > 0 or tfidf_s > 0:
            effective_relevance = 0.60 * tr_norm + 0.25 * tfidf_s + 0.15 * bm25_s
        else:
            effective_relevance = tr_norm

        composite = ALPHA * pr + BETA * effective_relevance + GAMMA * mg

        status  = mastery.node_status.get(nid, "locked")
        eligible = status in ("available", "in_progress")

        res_list = [r.model_dump() for r in getattr(node, "resources", [])]

        results.append({
            "node_id":                nid,
            "title":                  node.title,
            "difficulty":             node.difficulty,
            "source_url":             node.source_url,
            "score":                  round(float(composite), 4),
            "prerequisite_readiness": round(float(pr),      4),
            "target_relevance":       round(float(effective_relevance), 4),
            "marginal_gain":          round(float(mg),       4),
            "bm25_score":             round(float(bm25_s),   4),
            "tfidf_score":            round(float(tfidf_s),  4),
            "resource_quality":       round(float(res_q),    4),
            "status":                 status,
            "eligible":               eligible,
            "mastery":                round(mastery.node_scores.get(nid, 0.0), 4),
            "resources":              res_list,
        })

    # ── 6. Sort: eligible first (by score desc), then ineligible ─────────
    eligible_list   = sorted([r for r in results if r["eligible"]],
                              key=lambda x: x["score"], reverse=True)
    ineligible_list = sorted([r for r in results if not r["eligible"]],
                              key=lambda x: x["score"], reverse=True)
    return eligible_list + ineligible_list


def pick_gnn_next_node(
    kg: KnowledgeGraph,
    mastery: MasteryState,
    session_id: str,
) -> Optional[str]:
    """
    Drop-in replacement for the topology-only pick_next_node().
    Uses GNN composite scores to select the best eligible next concept.
    """
    ranked = run_gnn_inference(kg, mastery, session_id)
    for rec in ranked:
        if rec["eligible"] and rec["status"] != "mastered":
            return rec["node_id"]
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Baseline Scorers (for evaluation / comparison)
# ══════════════════════════════════════════════════════════════════════════════

def baseline_topological_sort(
    kg: KnowledgeGraph,
    mastery: MasteryState,
) -> List[Dict[str, Any]]:
    """Greedy topological sort baseline – ignores learner state."""
    g = nx.DiGraph()
    for node in kg.nodes:
        g.add_node(node.id)
    for edge in kg.edges:
        g.add_edge(edge.source, edge.target)

    order = list(nx.topological_sort(g))
    node_map = {n.id: n for n in kg.nodes}
    results = []
    for rank, nid in enumerate(order):
        node   = node_map[nid]
        status = mastery.node_status.get(nid, "locked")
        score  = 1.0 - rank / max(len(order) - 1, 1)
        results.append({
            "node_id":  nid,
            "title":    node.title,
            "score":    round(score, 4),
            "status":   status,
            "eligible": status in ("available", "in_progress"),
            "baseline": "topological",
        })
    return results


def baseline_heuristic_ranker(
    kg: KnowledgeGraph,
    mastery: MasteryState,
) -> List[Dict[str, Any]]:
    """
    Heuristic: score = (1 - mastery) / difficulty
    Prefers easy, unlearned concepts first.
    """
    results = []
    for node in kg.nodes:
        nid   = node.id
        m     = mastery.node_scores.get(nid, 0.0)
        score = (1.0 - m) / (node.difficulty / 5.0 + 1e-6)
        score = min(score, 1.0)
        status = mastery.node_status.get(nid, "locked")
        results.append({
            "node_id":  nid,
            "title":    node.title,
            "score":    round(score, 4),
            "status":   status,
            "eligible": status in ("available", "in_progress"),
            "baseline": "heuristic",
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)


def baseline_random_walk(
    kg: KnowledgeGraph,
    mastery: MasteryState,
    alpha: float = 0.85,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Random Walk with Restart (PageRank variant).
    Personalization vector = (1 - mastery) so unmastered nodes get
    more probability mass.
    """
    g = nx.DiGraph()
    for node in kg.nodes:
        g.add_node(node.id)
    for edge in kg.edges:
        g.add_edge(edge.source, edge.target)

    personalisation = {}
    total = 0.0
    for node in kg.nodes:
        w = 1.0 - mastery.node_scores.get(node.id, 0.0) + 1e-6
        personalisation[node.id] = w
        total += w
    personalisation = {k: v / total for k, v in personalisation.items()}

    try:
        pr = nx.pagerank(g, alpha=alpha, personalization=personalisation,
                         max_iter=200, tol=1e-6)
    except nx.exception.NetworkXError:
        pr = {n: 1.0 / len(kg.nodes) for n in [nd.id for nd in kg.nodes]}

    results = []
    for node in kg.nodes:
        nid    = node.id
        status = mastery.node_status.get(nid, "locked")
        results.append({
            "node_id":  nid,
            "title":    node.title,
            "score":    round(pr.get(nid, 0.0), 6),
            "status":   status,
            "eligible": status in ("available", "in_progress"),
            "baseline": "random_walk",
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)
