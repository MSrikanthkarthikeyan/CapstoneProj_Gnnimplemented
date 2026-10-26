"""
Phase B - Auto-HKG: LLM-powered Prerequisite Knowledge Graph construction.

Aggregates ingested source material (Wikipedia + YouTube + MIT OCW) and prompts
Gemini to emit a structured DAG (nodes + prerequisite edges) as strict
JSON, which is then parsed into Pydantic models and validated with networkx.
"""
import json
import re
from typing import List, Dict

from google import genai
from google.genai import types

import numpy as np
from app.config import settings, require_gemini_key
from app.graph.schema import KnowledgeGraph
from app.graph.validator import validate_dag, InvalidGraphError
from app.ranking import normalize_resources, rank_and_diversify_resources

SYSTEM_PROMPT = """You are an expert curriculum architect and instructional designer.
You will be given a learning GOAL and a bundle of aggregated source material
(Wikipedia summaries/sections/links, YouTube tutorial titles/descriptions/transcript
snippets, and MIT OpenCourseWare course metadata).

Your job is to design a Prerequisite Knowledge Graph: a Directed Acyclic Graph (DAG)
of concept nodes a learner must master, in order, to achieve the GOAL.

Rules:
- Produce between 6 and {max_nodes} concept nodes. Fewer, well-scoped nodes are better
  than many shallow ones.
- Every node must have: id (short string like "c1","c2"...), title, description
  (2-4 sentences, self-contained, written for a learner), difficulty (integer 1-5),
  and source_url (pick the single most relevant URL from the provided material, or
  null if none fits).
- Edges represent "source is a prerequisite_for target". The graph MUST be acyclic.
- The graph must be connected enough that a topological sort reaches every node
  needed for the goal - avoid isolated, irrelevant nodes.
- Order difficulty roughly non-decreasing along any path from roots to the final
  concept(s) most representative of the GOAL.
- Ground node titles/descriptions in the provided source material wherever possible;
  do not invent source_urls that were not given to you.

Respond with ONLY a single JSON object matching this exact schema, and nothing else
(no markdown fences, no commentary, no preamble):

{{
  "goal": "<the learning goal, echoed back>",
  "nodes": [
    {{"id": "c1", "title": "...", "description": "...", "difficulty": 1, "source_url": "..." }}
  ],
  "edges": [
    {{"source": "c1", "target": "c2", "relation": "prerequisite_for"}}
  ]
}}
"""

MAX_REPAIR_ATTEMPTS = 2


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _build_source_bundle(
    wiki: Dict,
    youtube: List[Dict],
    ocw: List[Dict],
) -> str:
    parts = []

    if wiki and wiki.get("found"):
        parts.append(
            "### WIKIPEDIA\n"
            f"Title: {wiki.get('title')}\n"
            f"URL: {wiki.get('source_url')}\n"
            f"Summary: {wiki.get('summary')}\n"
            f"Sections: {', '.join(wiki.get('sections', []))}\n"
            f"Related links: {', '.join(wiki.get('links', [])[:25])}\n"
        )

    if youtube:
        yt_lines = ["### YOUTUBE TUTORIALS"]
        for v in youtube:
            yt_lines.append(
                f"- Title: {v.get('title')} | URL: {v.get('url')}\n"
                f"  Channel: {v.get('channel')} | Views: {v.get('view_count')}\n"
                f"  Description: {v.get('description', '')[:300]}\n"
                f"  Transcript snippet: {v.get('transcript_snippet', '')[:400]}"
            )
        parts.append("\n".join(yt_lines))

    if ocw:
        ocw_lines = ["### MIT OPENCOURSEWARE"]
        for c in ocw:
            ocw_lines.append(
                f"- Course: {c.get('course_title')} | URL: {c.get('url')}\n"
                f"  Description: {c.get('description', '')[:300]}"
            )
        parts.append("\n".join(ocw_lines))

    return "\n\n".join(parts) if parts else "(No source material was retrieved.)"


def _call_gemini(goal: str, source_bundle: str, repair_note: str = "") -> str:
    require_gemini_key()
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    system = SYSTEM_PROMPT.format(max_nodes=settings.MAX_GRAPH_NODES)
    user_content = (
        f"GOAL: {goal}\n\n"
        f"AGGREGATED SOURCE MATERIAL:\n{source_bundle}\n"
    )
    if repair_note:
        user_content += f"\n\nIMPORTANT - fix this issue from your previous attempt: {repair_note}"

    models_to_try = [settings.GEMINI_MODEL]
    if settings.GEMINI_MODEL != "gemini-flash-latest":
        models_to_try.append("gemini-flash-latest")

    last_err = None
    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=4000,
                    response_mime_type="application/json"
                )
            )
            return response.text
        except Exception as e:
            last_err = e
    raise last_err


def build_knowledge_graph(
    goal: str,
    wiki: Dict,
    youtube: List[Dict],
    ocw: List[Dict],
) -> KnowledgeGraph:
    """
    Orchestrates the Phase B pipeline:
      1. Aggregate source material into a single text bundle.
      2. Prompt Gemini for a strict-JSON DAG.
      3. Parse into Pydantic models.
      4. Validate acyclicity with networkx, retrying with a repair prompt on failure.
    """
    source_bundle = _build_source_bundle(wiki, youtube, ocw)
    repair_note = ""

    last_error = None
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        raw = _call_gemini(goal, source_bundle, repair_note)
        cleaned = _strip_code_fences(raw)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = f"Response was not valid JSON: {e}"
            repair_note = last_error
            continue

        try:
            kg = KnowledgeGraph(**data)
        except Exception as e:
            last_error = f"JSON did not match schema: {e}"
            repair_note = last_error
            continue

        try:
            validate_dag(kg)
            
            # ── Resource Ranking Phase (BM25, TF-IDF, Ensemble, MMR) ──
            candidate_pool = normalize_resources(wiki, youtube, ocw)
            if candidate_pool:
                for node in kg.nodes:
                    ranked_3 = rank_and_diversify_resources(
                        concept_title=node.title,
                        concept_description=node.description,
                        concept_difficulty=node.difficulty,
                        candidate_resources=candidate_pool,
                        top_k=3,
                        mmr_lambda=0.65,
                    )
                    node.resources = ranked_3
                    if ranked_3:
                        node.bm25_score = round(float(np.mean([r.bm25_score for r in ranked_3])), 4)
                        node.tfidf_score = round(float(np.mean([r.tfidf_score for r in ranked_3])), 4)
                        node.resource_quality = round(float(np.mean([r.source_authority for r in ranked_3])), 4)
                        if not node.source_url or node.source_url == "null":
                            node.source_url = ranked_3[0].url

            return kg
        except InvalidGraphError as e:
            last_error = str(e)
            repair_note = str(e)
            continue

    raise RuntimeError(
        f"Failed to build a valid knowledge graph after {MAX_REPAIR_ATTEMPTS + 1} attempts. "
        f"Last error: {last_error}"
    )
