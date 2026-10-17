"""
GNN Evaluation Module
=====================
Computes all journal-grade metrics for comparing the GNN recommender
against baselines on a given session's knowledge graph.

Metrics implemented:
  • NDCG@K         (Normalised Discounted Cumulative Gain at K)
  • MRR            (Mean Reciprocal Rank)
  • PVR            (Prerequisite Violation Rate)
  • Hit@K          (binary hit in top-K)

Models evaluated:
  1. GNN (our system)
  2. Topological Greedy Sort
  3. Heuristic Ranker
  4. Random Walk with Restart

Ablation variants:
  • GNN w/o structural edges
  • GNN w/o dynamic mastery
  • GNN w/o marginal gain weight

Returns structured results consumed by the Streamlit evaluation dashboard.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from app.graph.schema import KnowledgeGraph, MasteryState
from app.graph.validator import initialize_mastery
from app.gnn.inference import (
    run_gnn_inference,
    baseline_topological_sort,
    baseline_heuristic_ranker,
    baseline_random_walk,
    ALPHA, BETA, GAMMA,
)
from app.gnn.graph_constructor import build_pyg_data, NODE_FEATURE_DIM
from app.gnn.model import build_model, NumpyGCN

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  Metric functions
# ══════════════════════════════════════════════════════════════════════════════

def dcg_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    dcg = 0.0
    for i, nid in enumerate(ranked_ids[:k]):
        if nid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    ideal = sorted(
        [1 if nid in relevant_ids else 0 for nid in ranked_ids],
        reverse=True
    )
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal[:k]))
    if idcg < 1e-9:
        return 0.0
    return dcg_at_k(ranked_ids, relevant_ids, k) / idcg


def mrr(ranked_ids: List[str], relevant_ids: List[str]) -> float:
    for i, nid in enumerate(ranked_ids):
        if nid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def prerequisite_violation_rate(
    path: List[str],
    kg: KnowledgeGraph,
) -> float:
    """
    Fraction of consecutive pairs (u, v) in path where v's prerequisites
    include a node NOT yet visited.
    """
    g = nx.DiGraph()
    for n in kg.nodes:
        g.add_node(n.id)
    for e in kg.edges:
        g.add_edge(e.source, e.target)

    violations = 0
    visited    = set()
    for nid in path:
        preds = set(g.predecessors(nid))
        if preds - visited:
            violations += 1
        visited.add(nid)
    return violations / max(len(path), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation runner
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_eval_scenario(
    kg: KnowledgeGraph,
    seed: int = 99,
) -> Tuple[MasteryState, List[str]]:
    """
    Create a realistic mid-progress learner state and a ground-truth
    'next' node list (topological successors of mastered nodes).
    """
    rng = random.Random(seed)
    session_id = f"eval_{seed}"
    mastery = initialize_mastery(kg, session_id)

    node_ids = [n.id for n in kg.nodes]
    g = nx.DiGraph()
    for n in kg.nodes:
        g.add_node(n.id)
    for e in kg.edges:
        g.add_edge(e.source, e.target)
    topo = list(nx.topological_sort(g))

    # Simulate mastery of first third of topo order
    n_mastered = max(1, len(topo) // 3)
    mastered   = topo[:n_mastered]
    for nid in mastered:
        mastery.node_scores[nid] = rng.uniform(0.75, 1.0)
        mastery.node_status[nid] = "mastered"

    # Unlock eligible nodes
    for nid in topo[n_mastered:]:
        preds = list(g.predecessors(nid))
        if all(mastery.node_scores.get(p, 0.0) >= 0.7 for p in preds):
            mastery.node_status[nid] = "available"

    # Ground truth = next n_mastered+2 unmastered nodes in topo order
    ground_truth = topo[n_mastered : n_mastered + max(2, len(topo) // 3)]
    return mastery, ground_truth


def evaluate_all_models(
    kg: KnowledgeGraph,
    session_id: str,
    k_values: Tuple[int, ...] = (5, 10),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate GNN + all baselines on the given knowledge graph.

    Returns
    -------
    {
        "models": {
            "GNN":             {"ndcg@5": .., "ndcg@10": .., "mrr": .., "pvr": ..},
            "TopologicalSort": {...},
            "HeuristicRanker": {...},
            "RandomWalk":      {...},
        },
        "ablations": {
            "GNN_no_edges":   {...},
            "GNN_no_mastery": {...},
            "GNN_no_MIG":     {...},
        },
        "per_depth": {depth: {"GNN": score, "Topological": score, ...}},
        "ground_truth": [node_id, ...],
    }
    """
    mastery, ground_truth = _simulate_eval_scenario(kg, seed=seed)

    def _extract_ranking(ranked_list: List[Dict]) -> List[str]:
        return [r["node_id"] for r in ranked_list]

    # ── GNN ──────────────────────────────────────────────────────────────
    gnn_ranked = run_gnn_inference(kg, mastery, session_id, force_refresh=True, seed=seed)
    gnn_ids    = _extract_ranking(gnn_ranked)

    # ── Baselines ─────────────────────────────────────────────────────────
    topo_ranked = baseline_topological_sort(kg, mastery)
    topo_ids    = _extract_ranking(topo_ranked)

    heur_ranked = baseline_heuristic_ranker(kg, mastery)
    heur_ids    = _extract_ranking(heur_ranked)

    rw_ranked   = baseline_random_walk(kg, mastery, seed=seed)
    rw_ids      = _extract_ranking(rw_ranked)

    # ── Ablations ─────────────────────────────────────────────────────────
    # 1. No structural edges (disconnect all edges → identity graph)
    kg_no_edges        = kg.model_copy(update={"edges": []})
    ablation_no_edges  = run_gnn_inference(kg_no_edges, mastery, f"{session_id}_ne", force_refresh=True, seed=seed)
    abl_ne_ids         = _extract_ranking(ablation_no_edges)

    # 2. No dynamic mastery (zero mastery for all)
    mastery_zeroed      = mastery.model_copy(update={"node_scores": {}})
    ablation_no_mastery = run_gnn_inference(kg, mastery_zeroed, f"{session_id}_nm", force_refresh=True, seed=seed)
    abl_nm_ids          = _extract_ranking(ablation_no_mastery)

    # 3. No marginal information gain (GAMMA = 0, renormalise)
    import app.gnn.inference as _inf_module
    old_gamma = _inf_module.GAMMA
    old_alpha = _inf_module.ALPHA
    old_beta  = _inf_module.BETA
    _inf_module.GAMMA = 0.0
    _inf_module.ALPHA = 0.55
    _inf_module.BETA  = 0.45
    ablation_no_mig = run_gnn_inference(kg, mastery, f"{session_id}_ng", force_refresh=True, seed=seed)
    abl_ng_ids      = _extract_ranking(ablation_no_mig)
    _inf_module.GAMMA = old_gamma
    _inf_module.ALPHA = old_alpha
    _inf_module.BETA  = old_beta

    def _metrics(ranked_ids: List[str]) -> Dict[str, float]:
        m: Dict[str, float] = {}
        for k in k_values:
            m[f"ndcg@{k}"] = round(ndcg_at_k(ranked_ids, ground_truth, k), 4)
        m["mrr"] = round(mrr(ranked_ids, ground_truth), 4)
        m["pvr"] = round(prerequisite_violation_rate(ranked_ids, kg), 4)
        return m

    # ── Per-depth accuracy ────────────────────────────────────────────────
    g = nx.DiGraph()
    for n in kg.nodes:
        g.add_node(n.id)
    for e in kg.edges:
        g.add_edge(e.source, e.target)

    depths: Dict[str, int] = {}
    for n in nx.topological_sort(g):
        d = max((depths.get(p, 0) + 1 for p in g.predecessors(n)), default=0)
        depths[n] = d

    per_depth: Dict[int, Dict[str, float]] = {}
    max_d = max(depths.values()) if depths else 0
    for depth in range(max_d + 1):
        nodes_at_depth = [nid for nid, d in depths.items() if d == depth]
        if not nodes_at_depth:
            continue
        gt_at_d = [nid for nid in ground_truth if nid in nodes_at_depth]
        if not gt_at_d:
            continue
        per_depth[depth] = {
            "GNN":              round(ndcg_at_k(gnn_ids,  gt_at_d, 5), 4),
            "TopologicalSort":  round(ndcg_at_k(topo_ids, gt_at_d, 5), 4),
            "HeuristicRanker":  round(ndcg_at_k(heur_ids, gt_at_d, 5), 4),
            "RandomWalk":       round(ndcg_at_k(rw_ids,   gt_at_d, 5), 4),
        }

    return {
        "models": {
            "GNN":             _metrics(gnn_ids),
            "TopologicalSort": _metrics(topo_ids),
            "HeuristicRanker": _metrics(heur_ids),
            "RandomWalk":      _metrics(rw_ids),
        },
        "ablations": {
            "GNN (full)":       _metrics(gnn_ids),
            "GNN -edges":       _metrics(abl_ne_ids),
            "GNN -mastery":     _metrics(abl_nm_ids),
            "GNN -MIG":         _metrics(abl_ng_ids),
        },
        "per_depth":    per_depth,
        "ground_truth": ground_truth,
        "n_concepts":   len(kg.nodes),
        "k_values":     list(k_values),
    }


def compute_depth_accuracy_curve(
    kg: KnowledgeGraph,
    session_id: str,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Computes recommendation NDCG@5 vs concept prerequisite depth
    for all four models. Used for the depth vs accuracy plot.
    """
    results = evaluate_all_models(kg, session_id, seed=seed)
    curve   = []
    for depth, scores in sorted(results["per_depth"].items()):
        row = {"depth": depth}
        row.update(scores)
        curve.append(row)
    return curve
