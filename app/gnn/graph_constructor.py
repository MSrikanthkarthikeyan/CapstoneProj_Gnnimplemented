"""
GNN Graph Construction Module
==============================
Converts a KnowledgeGraph (our Pydantic DAG) into a PyTorch Geometric
Data object with rich node features for GNN forward propagation.

Node features (per concept node):
  [0] difficulty       – normalised 0-1 (raw 1-5 / 5)
  [1] topo_depth       – normalised 0-1 (topological level / max_depth)
  [2] topic_cluster    – normalised 0-1 (cluster index / n_clusters, computed by degree community)
  [3] mastery          – learner mastery score 0-1 (dynamic, updated each quiz)
  [4] in_degree        – normalised number of prerequisite edges incoming
  [5] out_degree       – normalised number of dependent edges outgoing
  [6] prerequisite_ratio – in_degree / (in_degree + out_degree + 1e-6)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from app.graph.schema import KnowledgeGraph, MasteryState

# ──────────────────────────────────────────────────────────────────────────────
# Torch / PyG with graceful fallback (numpy-only path if torch unavailable)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch
    from torch_geometric.data import Data as PyGData
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    PyGData = None  # type: ignore

NODE_FEATURE_DIM = 10


def _topo_depths(g: nx.DiGraph) -> Dict[str, int]:
    """BFS-based topological depth (0 = root node)."""
    depths: Dict[str, int] = {}
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    queue = list(roots)
    for r in roots:
        depths[r] = 0
    while queue:
        current = queue.pop(0)
        for succ in g.successors(current):
            new_d = depths[current] + 1
            if depths.get(succ, -1) < new_d:
                depths[succ] = new_d
                queue.append(succ)
    return depths


def _topic_clusters(g: nx.DiGraph, n_nodes: int) -> Dict[str, int]:
    """
    Simple degree-based community approximation.
    Groups nodes by their strongly-connected component index (degrades to
    weakly connected for DAGs), giving each a cluster integer 0..K-1.
    """
    undirected = g.to_undirected()
    components = list(nx.connected_components(undirected))
    cluster_map: Dict[str, int] = {}
    for ci, comp in enumerate(components):
        for node in comp:
            cluster_map[node] = ci
    return cluster_map


def build_pyg_data(
    kg: KnowledgeGraph,
    mastery: Optional[MasteryState] = None,
) -> "PyGData | dict":
    """
    Construct node-feature matrix X and edge_index from a KnowledgeGraph.

    Returns either a PyG Data object (if torch_geometric is available) or a
    plain dict with numpy arrays for the numpy-only fallback path.

    Parameters
    ----------
    kg      : KnowledgeGraph pydantic model
    mastery : optional learner mastery state (updates feature col [3])
    """
    g = nx.DiGraph()
    for node in kg.nodes:
        g.add_node(node.id, **node.model_dump())
    for edge in kg.edges:
        g.add_edge(edge.source, edge.target)

    node_ids = [n.id for n in kg.nodes]
    node_idx: Dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    depths = _topo_depths(g)
    max_depth = max(depths.values()) if (depths and max(depths.values()) > 0) else 1
    clusters = _topic_clusters(g, n)
    n_clusters = max(clusters.values()) + 1 if (clusters and max(clusters.values()) >= 0) else 1

    max_in  = max((g.in_degree(nid)  for nid in node_ids), default=1) or 1
    max_out = max((g.out_degree(nid) for nid in node_ids), default=1) or 1

    X = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)
    for node in kg.nodes:
        nid = node.id
        i   = node_idx[nid]
        in_d  = g.in_degree(nid)
        out_d = g.out_degree(nid)
        mastery_score = (
            mastery.node_scores.get(nid, 0.0) if mastery else 0.0
        )
        bm25_val = getattr(node, "bm25_score", 0.0)
        tfidf_val = getattr(node, "tfidf_score", 0.0)
        res_qual = getattr(node, "resource_quality", 0.0)
        if getattr(node, "resources", None) and (bm25_val == 0.0 and tfidf_val == 0.0):
            bm25_val = float(np.mean([r.bm25_score for r in node.resources])) if node.resources else 0.0
            tfidf_val = float(np.mean([r.tfidf_score for r in node.resources])) if node.resources else 0.0
            res_qual = float(np.mean([r.source_authority for r in node.resources])) if node.resources else 0.0

        X[i, 0] = node.difficulty / 5.0
        X[i, 1] = depths.get(nid, 0) / max_depth
        X[i, 2] = clusters.get(nid, 0) / n_clusters
        X[i, 3] = float(mastery_score)
        X[i, 4] = in_d  / max_in
        X[i, 5] = out_d / max_out
        X[i, 6] = in_d  / (in_d + out_d + 1e-6)
        X[i, 7] = float(min(1.0, max(0.0, bm25_val)))
        X[i, 8] = float(min(1.0, max(0.0, tfidf_val)))
        X[i, 9] = float(min(1.0, max(0.0, res_qual)))

    # Edge index [2, E] – directed prerequisite edges
    edges_src, edges_dst = [], []
    for edge in kg.edges:
        if edge.source in node_idx and edge.target in node_idx:
            edges_src.append(node_idx[edge.source])
            edges_dst.append(node_idx[edge.target])

    if _TORCH_AVAILABLE:
        x_t = torch.tensor(X, dtype=torch.float32)
        if edges_src:
            edge_index = torch.tensor(
                [edges_src, edges_dst], dtype=torch.long
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = PyGData(x=x_t, edge_index=edge_index)
        data.node_ids  = node_ids
        data.node_idx  = node_idx
        data.num_nodes = n
        return data
    else:
        # numpy fallback
        return {
            "x": X,
            "edge_index": np.array([edges_src, edges_dst], dtype=np.int64)
            if edges_src else np.zeros((2, 0), dtype=np.int64),
            "node_ids":  node_ids,
            "node_idx":  node_idx,
            "num_nodes": n,
        }
