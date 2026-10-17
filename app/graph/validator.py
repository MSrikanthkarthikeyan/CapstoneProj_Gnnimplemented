"""
DAG validation and traversal utilities using networkx.
Ensures the LLM-generated graph is acyclic and computes topological
ordering + which nodes are currently "available" given mastery state.
"""
import networkx as nx
from app.graph.schema import KnowledgeGraph, MasteryState


class InvalidGraphError(Exception):
    pass


def to_networkx(kg: KnowledgeGraph) -> nx.DiGraph:
    g = nx.DiGraph()
    for node in kg.nodes:
        g.add_node(node.id, **node.model_dump())
    for edge in kg.edges:
        if edge.source not in kg.node_ids() or edge.target not in kg.node_ids():
            raise InvalidGraphError(
                f"Edge references unknown node: {edge.source} -> {edge.target}"
            )
        g.add_edge(edge.source, edge.target, relation=edge.relation)
    return g


def validate_dag(kg: KnowledgeGraph) -> nx.DiGraph:
    """Raises InvalidGraphError if the graph is not a valid DAG."""
    g = to_networkx(kg)
    if not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        raise InvalidGraphError(f"Generated graph contains cycles: {cycles}")
    return g


def topological_order(kg: KnowledgeGraph) -> list:
    g = validate_dag(kg)
    return list(nx.topological_sort(g))


def initialize_mastery(kg: KnowledgeGraph, session_id: str) -> MasteryState:
    """Root nodes (no prerequisites) start 'available'; everything else 'locked'."""
    g = validate_dag(kg)
    status = {}
    for node_id in g.nodes:
        preds = list(g.predecessors(node_id))
        status[node_id] = "available" if not preds else "locked"
    return MasteryState(
        session_id=session_id,
        goal=kg.goal,
        node_scores={},
        node_status=status,
        current_node=None,
    )


def unlock_next_nodes(kg: KnowledgeGraph, mastery: MasteryState, threshold: float) -> MasteryState:
    """
    After a node is scored, recompute which locked nodes become available:
    a node unlocks once ALL of its prerequisites are mastered (score >= threshold).
    """
    g = validate_dag(kg)
    for node_id in g.nodes:
        if mastery.node_status.get(node_id) == "locked":
            preds = list(g.predecessors(node_id))
            if all(mastery.node_scores.get(p, 0.0) >= threshold for p in preds):
                mastery.node_status[node_id] = "available"
    return mastery


def pick_next_node(kg: KnowledgeGraph, mastery: MasteryState) -> str | None:
    """Pick the next 'available' node in topological order that isn't mastered yet."""
    order = topological_order(kg)
    for node_id in order:
        status = mastery.node_status.get(node_id)
        if status in ("available", "in_progress"):
            return node_id
    return None
