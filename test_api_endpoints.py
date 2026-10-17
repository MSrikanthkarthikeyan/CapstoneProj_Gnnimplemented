"""
Integration tests for FastAPI endpoints with GNN engine.
"""
import unittest
from fastapi.testclient import TestClient

from app.api.main import app
from app.graph.schema import KnowledgeGraph, ConceptNode, PrerequisiteEdge
from app.agent.orchestrator import SessionStore

class TestFastAPIWithGNN(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        # Prepopulate a test session in SessionStore
        nodes = [
            ConceptNode(id="n1", title="Python Basics", description="Syntax and variables", difficulty=1),
            ConceptNode(id="n2", title="Control Flow", description="Loops and conditionals", difficulty=2),
            ConceptNode(id="n3", title="Functions & Scope", description="Defining functions", difficulty=2),
            ConceptNode(id="n4", title="Object Oriented Programming", description="Classes and objects", difficulty=3),
        ]
        edges = [
            PrerequisiteEdge(source="n1", target="n2"),
            PrerequisiteEdge(source="n2", target="n3"),
            PrerequisiteEdge(source="n3", target="n4"),
        ]
        kg = KnowledgeGraph(goal="Learn Python Programming", nodes=nodes, edges=edges)
        self.session_id = SessionStore.create(kg)

    def test_01_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_02_graph_endpoint(self):
        resp = self.client.get(f"/graph/{self.session_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["goal"], "Learn Python Programming")
        self.assertEqual(len(data["nodes"]), 4)
        self.assertEqual(len(data["edges"]), 3)
        self.assertIn("mermaid", data)

    def test_03_gnn_recommend_endpoint(self):
        resp = self.client.get(f"/gnn/recommend/{self.session_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], self.session_id)
        self.assertEqual(data["recommended_next"], "n1")
        self.assertEqual(len(data["ranked_concepts"]), 4)
        self.assertTrue(data["ranked_concepts"][0]["eligible"])

    def test_04_gnn_baselines_endpoint(self):
        resp = self.client.get(f"/gnn/baselines/{self.session_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("topological", data)
        self.assertIn("heuristic", data)
        self.assertIn("random_walk", data)

    def test_05_gnn_train_endpoint(self):
        resp = self.client.post(
            f"/gnn/train/{self.session_id}",
            json={"epochs": 15, "lr": 0.01, "n_learners": 30}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("training", data)
        self.assertEqual(data["training"]["epochs"], 15)

    def test_06_gnn_evaluate_endpoint(self):
        resp = self.client.get(f"/gnn/evaluate/{self.session_id}?seed=42")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)
        self.assertIn("ablations", data)
        self.assertIn("per_depth", data)


if __name__ == "__main__":
    unittest.main()
