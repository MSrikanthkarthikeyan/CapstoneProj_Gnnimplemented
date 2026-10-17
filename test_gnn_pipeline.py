"""
End-to-End Test Suite for PathBuilder GNN Recommendation & Evaluation Pipeline.
Tests:
  1. Knowledge graph DAG schema and validation
  2. Dynamic feature matrix construction (N x 10 with ML features)
  3. GNN forward pass & embedding representation
  4. Composite scoring (Readiness + Relevance + Marginal Gain)
  5. Baseline comparison: Topological sort, Heuristic, Random Walk
  6. Pairwise BPR training loop with convergence checks
  7. Scientific evaluation metrics (NDCG@K, MRR, PVR)
  8. Ablation study & prerequisite depth evaluation
  9. BM25, TF-IDF, 9-feature ensemble scoring, and MMR diversification
  10. GNN enrichment with ML resource ranking signals
"""
import sys
import time
import unittest
import numpy as np

from app.graph.schema import KnowledgeGraph, ConceptNode, PrerequisiteEdge, MasteryState
from app.graph.validator import validate_dag, initialize_mastery, unlock_next_nodes
from app.gnn.graph_constructor import build_pyg_data, NODE_FEATURE_DIM
from app.gnn.model import build_model, NumpyGCN
from app.gnn.inference import (
    run_gnn_inference,
    pick_gnn_next_node,
    baseline_topological_sort,
    baseline_heuristic_ranker,
    baseline_random_walk,
)
from app.gnn.trainer import train_gnn
from app.gnn.evaluator import evaluate_all_models, prerequisite_violation_rate, ndcg_at_k, mrr
from app.ranking import (
    ResourceItem,
    BM25Okapi,
    TFIDFSimilarityRanker,
    normalize_resources,
    rank_and_diversify_resources,
)


def create_sample_curriculum_graph() -> KnowledgeGraph:
    """Creates a realistic 8-node prerequisite DAG for Data Structures & Algorithms."""
    nodes = [
        ConceptNode(id="n1", title="Arrays & Pointers", description="Memory contiguous storage", difficulty=1),
        ConceptNode(id="n2", title="Linked Lists", description="Pointer-based sequential storage", difficulty=2),
        ConceptNode(id="n3", title="Stacks & Queues", description="LIFO and FIFO structures", difficulty=2),
        ConceptNode(id="n4", title="Hash Tables", description="Key-value mapping with hash functions", difficulty=3),
        ConceptNode(id="n5", title="Binary Search Trees", description="Hierarchical ordered tree", difficulty=3),
        ConceptNode(id="n6", title="Tree Traversals", description="DFS and BFS on trees", difficulty=3),
        ConceptNode(id="n7", title="Graph Algorithms", description="Topological sort, Dijkstra, BFS/DFS", difficulty=4),
        ConceptNode(id="n8", title="Dynamic Programming", description="Optimal substructure & memoization", difficulty=5),
    ]
    edges = [
        PrerequisiteEdge(source="n1", target="n2"),
        PrerequisiteEdge(source="n1", target="n4"),
        PrerequisiteEdge(source="n2", target="n3"),
        PrerequisiteEdge(source="n2", target="n5"),
        PrerequisiteEdge(source="n5", target="n6"),
        PrerequisiteEdge(source="n6", target="n7"),
        PrerequisiteEdge(source="n3", target="n7"),
        PrerequisiteEdge(source="n7", target="n8"),
    ]
    return KnowledgeGraph(goal="Master Advanced Data Structures & Algorithms", nodes=nodes, edges=edges)


class TestGNNPipeline(unittest.TestCase):

    def setUp(self):
        self.kg = create_sample_curriculum_graph()
        self.session_id = "test_session_42"
        self.mastery = initialize_mastery(self.kg, self.session_id)

    def test_01_dag_validation(self):
        """Verify DAG acyclicity and reachability."""
        validate_dag(self.kg)
        self.assertEqual(len(self.kg.nodes), 8)
        self.assertEqual(len(self.kg.edges), 8)
        self.assertIn("n1", self.mastery.node_status)
        self.assertEqual(self.mastery.node_status["n1"], "available")
        self.assertEqual(self.mastery.node_status["n8"], "locked")

    def test_02_graph_features(self):
        """Verify 10-dimensional node feature construction."""
        data = build_pyg_data(self.kg, self.mastery)
        if hasattr(data, "x"):
            X = data.x.numpy()
        else:
            X = data["x"]
        self.assertEqual(X.shape, (8, NODE_FEATURE_DIM))
        self.assertEqual(NODE_FEATURE_DIM, 10)
        # Check normalized difficulty bounds [0, 1]
        self.assertTrue(np.all(X[:, 0] >= 0.0) and np.all(X[:, 0] <= 1.0))
        # Check in_degree and out_degree normalization
        self.assertTrue(np.all(X[:, 4] >= 0.0) and np.all(X[:, 4] <= 1.0))
        self.assertTrue(np.all(X[:, 5] >= 0.0) and np.all(X[:, 5] <= 1.0))

    def test_03_model_forward_pass(self):
        """Verify GNN embedding dimensions."""
        model = build_model(in_channels=NODE_FEATURE_DIM, hidden_dim=32, embed_dim=16, seed=42)
        data = build_pyg_data(self.kg, self.mastery)
        if hasattr(data, "x"):
            import torch
            with torch.no_grad():
                H = model(data.x, data.edge_index).numpy()
        else:
            H = model.forward(data["x"], data["edge_index"])
        self.assertEqual(H.shape, (8, 16))
        self.assertFalse(np.isnan(H).any())

    def test_04_gnn_recommendation_and_constraints(self):
        """Verify composite scoring and prerequisite-aware filtering."""
        recs = run_gnn_inference(self.kg, self.mastery, self.session_id)
        self.assertEqual(len(recs), 8)

        # First concept recommended MUST be eligible
        top_rec = recs[0]
        self.assertTrue(top_rec["eligible"])
        self.assertEqual(top_rec["node_id"], "n1")  # n1 is the single root
        self.assertEqual(top_rec["prerequisite_readiness"], 1.0)

        # Locked concepts like n8 (Dynamic Programming) should be marked ineligible
        n8_rec = next(r for r in recs if r["node_id"] == "n8")
        self.assertFalse(n8_rec["eligible"])
        self.assertEqual(n8_rec["status"], "locked")

        # Drop-in selector test
        next_id = pick_gnn_next_node(self.kg, self.mastery, self.session_id)
        self.assertEqual(next_id, "n1")

    def test_05_dynamic_mastery_adaptation(self):
        """Verify that mastering n1 unlocks n2 and n4, and GNN dynamically reranks."""
        # Simulate mastering n1 with 95% score
        self.mastery.node_scores["n1"] = 0.95
        self.mastery.node_status["n1"] = "mastered"
        self.mastery = unlock_next_nodes(self.kg, self.mastery, threshold=0.7)

        self.assertEqual(self.mastery.node_status["n2"], "available")
        self.assertEqual(self.mastery.node_status["n4"], "available")
        self.assertEqual(self.mastery.node_status["n8"], "locked")

        recs = run_gnn_inference(self.kg, self.mastery, self.session_id, force_refresh=True)
        top_eligible = [r["node_id"] for r in recs if r["eligible"] and r["status"] != "mastered"]
        self.assertTrue("n2" in top_eligible or "n4" in top_eligible)

    def test_06_baselines_execution(self):
        """Verify all 3 baselines execute cleanly and return valid rankings."""
        topo = baseline_topological_sort(self.kg, self.mastery)
        self.assertEqual(len(topo), 8)

        heuristic = baseline_heuristic_ranker(self.kg, self.mastery)
        self.assertEqual(len(heuristic), 8)

        rw = baseline_random_walk(self.kg, self.mastery)
        self.assertEqual(len(rw), 8)

        # All baseline outputs should cover all 8 nodes
        self.assertEqual(set(r["node_id"] for r in topo), set(n.id for n in self.kg.nodes))
        self.assertEqual(set(r["node_id"] for r in heuristic), set(n.id for n in self.kg.nodes))
        self.assertEqual(set(r["node_id"] for r in rw), set(n.id for n in self.kg.nodes))

    def test_07_gnn_training_loop(self):
        """Verify pairwise BPR loss optimization decreases loss."""
        result = train_gnn(
            kg=self.kg,
            session_id=self.session_id,
            epochs=25,
            lr=1e-2,
            n_learners=40,
            seed=42,
        )
        self.assertIn("train_loss", result)
        self.assertGreater(len(result["train_loss"]), 0)
        self.assertLess(result["final_train_loss"], result["train_loss"][0])

    def test_08_scientific_evaluation_suite(self):
        """Verify NDCG@5, NDCG@10, MRR, PVR across all models."""
        eval_results = evaluate_all_models(self.kg, self.session_id, seed=42)
        self.assertIn("models", eval_results)
        self.assertIn("ablations", eval_results)
        self.assertIn("per_depth", eval_results)

        comp = eval_results["models"]
        self.assertIn("GNN", comp)
        self.assertIn("TopologicalSort", comp)
        self.assertIn("HeuristicRanker", comp)
        self.assertIn("RandomWalk", comp)

        gnn_metrics = comp["GNN"]
        self.assertGreaterEqual(gnn_metrics["ndcg@5"], 0.0)
        self.assertLessEqual(gnn_metrics["ndcg@5"], 1.0)
        self.assertGreaterEqual(gnn_metrics["mrr"], 0.0)
        self.assertLessEqual(gnn_metrics["mrr"], 1.0)
        self.assertGreaterEqual(gnn_metrics["pvr"], 0.0)
        self.assertLessEqual(gnn_metrics["pvr"], 1.0)

    def test_09_resource_ranking_and_mmr(self):
        """Verify BM25, TF-IDF, 9-feature ensemble scoring, and MMR diversification."""
        # 1. BM25 scoring test
        corpus = [
            "Arrays are contiguous blocks of memory with instant O(1) random access",
            "Linked lists use pointers connecting discrete nodes in heap memory",
            "Hash tables use hash functions to map keys into array buckets",
        ]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores("memory pointers linked")
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[1], scores[0])  # doc 1 mentions both pointers and memory

        # 2. TF-IDF scoring test
        tfidf = TFIDFSimilarityRanker()
        tfidf.fit_corpus(corpus)
        sims = tfidf.score_query("hash tables buckets mapping")
        self.assertEqual(len(sims), 3)
        self.assertEqual(int(np.argmax(sims)), 2)  # doc 2 is closest

        # 3. Normalization & MMR diversification
        raw_wiki = {"found": True, "title": "Data Structures", "summary": "Foundational CS storage", "source_url": "https://wiki.org"}
        raw_yt = [
            {"title": "Pointers Tutorial Part 1", "url": "https://yt.com/1", "channel": "freeCodeCamp", "view_count": 500000, "description": "Memory pointers tutorial"},
            {"title": "Pointers Tutorial Part 2 (Duplicate)", "url": "https://yt.com/2", "channel": "freeCodeCamp", "view_count": 480000, "description": "Memory pointers tutorial part 2"},
            {"title": "Visualizing Memory and Pointers", "url": "https://yt.com/3", "channel": "3Blue1Brown", "view_count": 1200000, "description": "Geometric intuitive view of memory"},
        ]
        raw_ocw = [{"course_title": "MIT 6.006 Introduction to Algorithms", "url": "https://ocw.mit.edu/6006", "description": "Comprehensive algorithms lecture notes"}]

        pool = normalize_resources(raw_wiki, raw_yt, raw_ocw)
        self.assertGreater(len(pool), 3)

        top_3 = rank_and_diversify_resources(
            concept_title="Arrays & Pointers",
            concept_description="Memory contiguous storage and pointers",
            concept_difficulty=1,
            candidate_resources=pool,
            top_k=3,
            mmr_lambda=0.65,
        )
        self.assertEqual(len(top_3), 3)
        # Verify MMR diversity: ensures types or titles are diverse rather than duplicate videos
        types = set(r.resource_type for r in top_3)
        self.assertGreaterEqual(len(types), 2)  # At least 2 distinct resource types

    def test_10_gnn_enrichment_with_resources(self):
        """Verify that GNN node features and recommendations reflect BM25 & TF-IDF signals."""
        # Attach resources to concept nodes
        for node in self.kg.nodes:
            node.bm25_score = 0.85 if node.id in ("n1", "n2") else 0.20
            node.tfidf_score = 0.90 if node.id in ("n1", "n2") else 0.25
            node.resource_quality = 0.95

        data = build_pyg_data(self.kg, self.mastery)
        X = data.x.numpy() if hasattr(data, "x") else data["x"]
        self.assertEqual(X.shape, (8, 10))
        # Columns 7, 8, 9 are BM25, TF-IDF, Resource Quality
        self.assertAlmostEqual(X[0, 7], 0.85, places=2)
        self.assertAlmostEqual(X[0, 8], 0.90, places=2)
        self.assertAlmostEqual(X[0, 9], 0.95, places=2)

        # Inference should return enriched concept objects
        recs = run_gnn_inference(self.kg, self.mastery, self.session_id)
        self.assertIn("bm25_score", recs[0])
        self.assertIn("tfidf_score", recs[0])
        self.assertIn("resource_quality", recs[0])


if __name__ == "__main__":
    unittest.main()
