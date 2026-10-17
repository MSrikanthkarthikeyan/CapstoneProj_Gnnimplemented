"""
FastAPI application – Learning Path Builder with GNN Recommendation Engine.

Endpoints:
  POST /build-path             -> ingest sources, build DAG, start session
  GET  /graph/{session_id}     -> graph + progress
  GET  /tutor/next/{session_id}-> lesson + quiz for current node
  POST /tutor/answer           -> submit answers, mastery update
  GET  /gnn/recommend/{sid}    -> GNN-ranked concept list
  POST /gnn/train/{sid}        -> train GCN on session graph
  GET  /gnn/evaluate/{sid}     -> full evaluation metrics (all models)
  GET  /health                 -> liveness check
"""
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.tools.wikipedia_tool import fetch_wikipedia_summary
from app.tools.youtube_tool import fetch_youtube_tutorials
from app.tools.mit_ocw_tool import fetch_mit_ocw_materials
from app.graph.builder import build_knowledge_graph
from app.graph.validator import validate_dag, InvalidGraphError
from app.agent.orchestrator import (
    SessionStore, get_lesson_and_quiz, submit_answers, get_progress,
)
from app.gnn.inference import (
    run_gnn_inference,
    pick_gnn_next_node,
    baseline_topological_sort,
    baseline_heuristic_ranker,
    baseline_random_walk,
)
from app.gnn.trainer import train_gnn
from app.gnn.evaluator import evaluate_all_models

app = FastAPI(
    title="Agentic AI Learning Path Builder",
    description="Graph-based, adaptive learning path generator with GNN recommendation.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Request / Response Models
# --------------------------------------------------------------------------

class BuildPathRequest(BaseModel):
    goal: str
    max_youtube_results: int = 3
    max_ocw_results: int = 5


class BuildPathResponse(BaseModel):
    session_id: str
    goal: str
    nodes: list
    edges: list
    mermaid: str


class AnswerRequest(BaseModel):
    session_id: str
    node_id: str
    user_answers: List[int]


class TrainRequest(BaseModel):
    epochs: int = 60
    lr: float = 5e-3
    n_learners: int = 80
    arch: str = "gcn"
    seed: int = 42


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _graph_to_mermaid(kg) -> str:
    lines = ["graph TD"]
    difficulty_labels = {1: "🟢", 2: "🟢", 3: "🟡", 4: "🟠", 5: "🔴"}
    for n in kg.nodes:
        label = f"{difficulty_labels.get(n.difficulty, '')} {n.title}".replace('"', "'")
        lines.append(f'    {n.id}["{label}"]')
    for e in kg.edges:
        lines.append(f"    {e.source} --> {e.target}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Core Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


@app.post("/build-path", response_model=BuildPathResponse)
def build_path(req: BuildPathRequest):
    """Ingest sources, build the prerequisite DAG, start a tutoring session."""
    goal = req.goal.strip()
    if not goal:
        raise HTTPException(400, "goal must not be empty")

    try:
        wiki = fetch_wikipedia_summary(goal)
    except Exception as e:
        wiki = {"found": False, "error": str(e)}

    try:
        youtube = fetch_youtube_tutorials(goal, max_results=req.max_youtube_results)
    except Exception:
        youtube = []

    try:
        ocw = fetch_mit_ocw_materials(goal, max_results=req.max_ocw_results)
    except Exception:
        ocw = []

    try:
        kg = build_knowledge_graph(goal, wiki, youtube, ocw)
        validate_dag(kg)
    except InvalidGraphError as e:
        raise HTTPException(422, f"Generated graph failed DAG validation: {e}")
    except Exception as e:
        raise HTTPException(502, f"Failed to build knowledge graph: {e}")

    session_id = SessionStore.create(kg)

    return BuildPathResponse(
        session_id=session_id,
        goal=kg.goal,
        nodes=[n.model_dump() for n in kg.nodes],
        edges=[e.model_dump() for e in kg.edges],
        mermaid=_graph_to_mermaid(kg),
    )


@app.get("/graph/{session_id}")
def get_graph(session_id: str):
    try:
        session  = SessionStore.get(session_id)
        progress = get_progress(session_id)
    except KeyError:
        raise HTTPException(404, "Unknown session_id")

    kg = session["kg"]
    return {
        "goal":    kg.goal,
        "nodes":   [n.model_dump() for n in kg.nodes],
        "edges":   [e.model_dump() for e in kg.edges],
        "mermaid": _graph_to_mermaid(kg),
        "progress": progress,
    }


@app.get("/tutor/next/{session_id}")
def tutor_next(session_id: str):
    try:
        return get_lesson_and_quiz(session_id)
    except KeyError:
        raise HTTPException(404, "Unknown session_id")
    except Exception as e:
        raise HTTPException(502, f"Failed to generate lesson/quiz: {e}")


@app.post("/tutor/answer")
def tutor_answer(req: AnswerRequest):
    try:
        return submit_answers(req.session_id, req.node_id, req.user_answers)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"Failed to grade answers: {e}")


# --------------------------------------------------------------------------
# GNN Endpoints
# --------------------------------------------------------------------------

@app.get("/gnn/recommend/{session_id}")
def gnn_recommend(session_id: str, force_refresh: bool = False):
    """
    GNN inference: returns composite-scored concept ranking.
    Score = alpha*PrerequisiteReadiness + beta*TargetRelevance + gamma*MIG
    """
    try:
        session = SessionStore.get(session_id)
    except KeyError:
        raise HTTPException(404, "Unknown session_id")

    kg      = session["kg"]
    mastery = session["mastery"]
    try:
        ranked    = run_gnn_inference(kg, mastery, session_id,
                                      force_refresh=force_refresh)
        next_node = pick_gnn_next_node(kg, mastery, session_id)
        return {
            "session_id":       session_id,
            "recommended_next": next_node,
            "ranked_concepts":  ranked,
        }
    except Exception as e:
        raise HTTPException(502, f"GNN inference failed: {e}")


@app.post("/gnn/train/{session_id}")
def gnn_train(session_id: str, req: TrainRequest):
    """Train the GCN on synthetic trajectories derived from the session graph."""
    try:
        session = SessionStore.get(session_id)
    except KeyError:
        raise HTTPException(404, "Unknown session_id")

    kg = session["kg"]
    try:
        history = train_gnn(
            kg=kg,
            session_id=session_id,
            epochs=req.epochs,
            lr=req.lr,
            n_learners=req.n_learners,
            arch=req.arch,
            seed=req.seed,
        )
        return {"session_id": session_id, "training": history}
    except Exception as e:
        raise HTTPException(502, f"GNN training failed: {e}")


@app.get("/gnn/evaluate/{session_id}")
def gnn_evaluate(session_id: str, seed: int = 42):
    """
    Full evaluation suite: GNN vs Topological Sort, Heuristic Ranker,
    Random Walk with Restart. Returns NDCG@5, NDCG@10, MRR, PVR,
    ablation study, and per-depth accuracy.
    """
    try:
        session = SessionStore.get(session_id)
    except KeyError:
        raise HTTPException(404, "Unknown session_id")

    kg = session["kg"]
    try:
        results = evaluate_all_models(kg, session_id, seed=seed)
        return {"session_id": session_id, **results}
    except Exception as e:
        raise HTTPException(502, f"Evaluation failed: {e}")


@app.get("/gnn/baselines/{session_id}")
def gnn_baselines(session_id: str):
    """All three baseline rankings side-by-side."""
    try:
        session = SessionStore.get(session_id)
    except KeyError:
        raise HTTPException(404, "Unknown session_id")

    kg      = session["kg"]
    mastery = session["mastery"]
    return {
        "topological": baseline_topological_sort(kg, mastery),
        "heuristic":   baseline_heuristic_ranker(kg, mastery),
        "random_walk": baseline_random_walk(kg, mastery),
    }
