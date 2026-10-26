"""
Phase C - Adaptive Tutor Orchestration (LangGraph).

For the currently "available" node in the Prerequisite Knowledge Graph, this
stateful graph:
  1. generate_lesson  - Gemini writes a focused micro-lesson grounded in the
                         node's description + source material.
  2. generate_quiz    - Gemini writes N multiple-choice mastery-check questions.
  3. grade_answers     - Gemini grades submitted answers, returns a 0-1 score
                         and per-question feedback.
  4. update_mastery    - networkx-backed unlock logic (see app/graph/validator.py)
                         updates MasteryState and picks the next node.

LangGraph is used for the lesson/quiz/grade sub-flow because it is a natural,
inspectable state machine; overall session state (which node a learner is on,
their scores) is held in a simple in-memory SessionStore for this MVP.
"""
import json
import re
import uuid
from typing import TypedDict, Optional, List, Dict, Any

from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

from app.config import settings, require_gemini_key
from app.graph.schema import KnowledgeGraph, MasteryState, ConceptNode
from app.graph.validator import (
    initialize_mastery, unlock_next_nodes, pick_next_node, topological_order,
)


# --------------------------------------------------------------------------
# In-memory session store (swap for Redis/Postgres in production)
# --------------------------------------------------------------------------
class SessionStore:
    _sessions: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def create(cls, kg: KnowledgeGraph) -> str:
        session_id = str(uuid.uuid4())
        mastery = initialize_mastery(kg, session_id)
        mastery.current_node = pick_next_node(kg, mastery)
        cls._sessions[session_id] = {"kg": kg, "mastery": mastery, "cache": {}}
        return session_id

    @classmethod
    def get(cls, session_id: str) -> Dict[str, Any]:
        if session_id not in cls._sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        return cls._sessions[session_id]

    @classmethod
    def save(cls, session_id: str, kg: KnowledgeGraph, mastery: MasteryState):
        cls._sessions[session_id]["kg"] = kg
        cls._sessions[session_id]["mastery"] = mastery


def _client() -> genai.Client:
    require_gemini_key()
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def _generate_with_fallback(client: genai.Client, contents: Any, config: types.GenerateContentConfig):
    models = [settings.GEMINI_MODEL]
    if settings.GEMINI_MODEL != "gemini-flash-latest":
        models.append("gemini-flash-latest")
    last_err = None
    for m in models:
        try:
            return client.models.generate_content(model=m, contents=contents, config=config)
        except Exception as e:
            last_err = e
    raise last_err


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


# --------------------------------------------------------------------------
# LangGraph state
# --------------------------------------------------------------------------
class TutorState(TypedDict, total=False):
    node: ConceptNode
    goal: str
    lesson: str
    quiz: List[Dict[str, Any]]          # [{"question":..,"options":[..],"answer_index":..}]
    user_answers: Optional[List[int]]
    score: Optional[float]
    feedback: Optional[List[str]]


def generate_lesson_node(state: TutorState) -> TutorState:
    node: ConceptNode = state["node"]
    client = _client()

    resource_lines = []
    if getattr(node, "resources", None):
        for r in node.resources:
            r_type = getattr(r, "resource_type", "reference").upper()
            resource_lines.append(f"- [{r_type}] {r.title} ({r.url}): {r.snippet[:180]}")
    resources_text = "\n".join(resource_lines) if resource_lines else (node.source_url or "none provided")

    prompt = (
        f"Learning goal: {state['goal']}\n"
        f"Concept to teach: {node.title}\n"
        f"Concept description: {node.description}\n"
        f"Difficulty (1-5): {node.difficulty}\n"
        f"Curated Multi-Source Materials (BM25 & TF-IDF Ranked):\n{resources_text}\n\n"
        "Write a focused, self-contained micro-lesson (300-500 words) teaching this "
        "single concept to a motivated learner. Synthesize insights across the reference "
        "materials, use plain language, include one concrete intuitive example, and end "
        "with a one-sentence takeaway. Do not include a quiz."
    )
    resp = _generate_with_fallback(
        client,
        prompt,
        types.GenerateContentConfig(max_output_tokens=1200)
    )
    state["lesson"] = resp.text
    return state


def generate_quiz_node(state: TutorState) -> TutorState:
    node: ConceptNode = state["node"]
    client = _client()
    n_q = settings.QUIZ_QUESTIONS_PER_NODE
    prompt = (
        f"Based on this lesson about '{node.title}':\n\n{state['lesson']}\n\n"
        f"Write exactly {n_q} multiple-choice mastery-check questions, each with "
        "4 options and exactly one correct answer. Respond with ONLY JSON, no "
        "markdown fences, matching this schema:\n"
        '{"questions": [{"question": "...", "options": ["a","b","c","d"], '
        '"answer_index": 0, "explanation": "..."}]}'
    )
    resp = _generate_with_fallback(
        client,
        prompt,
        types.GenerateContentConfig(
            max_output_tokens=1500,
            response_mime_type="application/json"
        )
    )
    raw = resp.text
    data = json.loads(_strip_fences(raw))
    state["quiz"] = data["questions"]
    return state


def _lesson_quiz_graph():
    g = StateGraph(TutorState)
    g.add_node("generate_lesson", generate_lesson_node)
    g.add_node("generate_quiz", generate_quiz_node)
    g.set_entry_point("generate_lesson")
    g.add_edge("generate_lesson", "generate_quiz")
    g.add_edge("generate_quiz", END)
    return g.compile()


_LESSON_QUIZ_GRAPH = _lesson_quiz_graph()


def get_lesson_and_quiz(session_id: str) -> Dict[str, Any]:
    """Entry point for the '/tutor/next' API: builds (or returns cached)
    lesson + quiz for the learner's current node."""
    session = SessionStore.get(session_id)
    kg: KnowledgeGraph = session["kg"]
    mastery: MasteryState = session["mastery"]

    node_id = mastery.current_node or pick_next_node(kg, mastery)
    if node_id is None:
        return {"done": True, "message": "All concepts mastered. Goal achieved!"}

    mastery.node_status[node_id] = "in_progress"
    mastery.current_node = node_id

    if node_id in session["cache"]:
        cached = session["cache"][node_id]
        result = {"done": False, "node_id": node_id, **cached}
    else:
        node = next(n for n in kg.nodes if n.id == node_id)
        state: TutorState = {"node": node, "goal": kg.goal}
        result_state = _LESSON_QUIZ_GRAPH.invoke(state)
        payload = {
            "node_title": node.title,
            "node_description": node.description,
            "difficulty": node.difficulty,
            "source_url": node.source_url,
            "resources": [r.model_dump() for r in getattr(node, "resources", [])],
            "lesson": result_state["lesson"],
            "quiz": [
                {"question": q["question"], "options": q["options"]}
                for q in result_state["quiz"]
            ],
        }
        session["cache"][node_id] = {**payload, "_answer_key": result_state["quiz"]}
        result = {"done": False, "node_id": node_id, **payload}

    SessionStore.save(session_id, kg, mastery)
    return result


def submit_answers(session_id: str, node_id: str, user_answers: List[int]) -> Dict[str, Any]:
    """Entry point for '/tutor/answer': grades submitted answers, updates
    mastery state via networkx unlock logic, and returns the next available node."""
    session = SessionStore.get(session_id)
    kg: KnowledgeGraph = session["kg"]
    mastery: MasteryState = session["mastery"]

    cached = session["cache"].get(node_id)
    if not cached:
        raise KeyError(f"No cached quiz found for node {node_id}; call /tutor/next first.")

    answer_key = cached["_answer_key"]
    correct = 0
    feedback = []
    for i, q in enumerate(answer_key):
        user_choice = user_answers[i] if i < len(user_answers) else -1
        is_correct = user_choice == q["answer_index"]
        correct += int(is_correct)
        feedback.append(
            f"Q{i+1}: {'Correct' if is_correct else 'Incorrect'}. {q.get('explanation', '')}"
        )

    score = correct / len(answer_key) if answer_key else 0.0
    mastery.node_scores[node_id] = max(mastery.node_scores.get(node_id, 0.0), score)

    if score >= settings.MASTERY_PASS_THRESHOLD:
        mastery.node_status[node_id] = "mastered"
    else:
        mastery.node_status[node_id] = "available"  # allow retry

    mastery = unlock_next_nodes(kg, mastery, settings.MASTERY_PASS_THRESHOLD)
    next_node = pick_next_node(kg, mastery)
    mastery.current_node = next_node

    SessionStore.save(session_id, kg, mastery)

    return {
        "score": score,
        "passed": score >= settings.MASTERY_PASS_THRESHOLD,
        "feedback": feedback,
        "next_node_id": next_node,
        "goal_complete": next_node is None,
        "mastery_snapshot": {
            "node_status": mastery.node_status,
            "node_scores": mastery.node_scores,
        },
    }


def get_progress(session_id: str) -> Dict[str, Any]:
    session = SessionStore.get(session_id)
    kg: KnowledgeGraph = session["kg"]
    mastery: MasteryState = session["mastery"]
    order = topological_order(kg)
    return {
        "goal": kg.goal,
        "order": order,
        "node_status": mastery.node_status,
        "node_scores": mastery.node_scores,
        "current_node": mastery.current_node,
    }
