"""
Resource Ranking Engine
=======================
Implements the full ML resource ranking pipeline from the system architecture:
1. Feature Extraction:
   - BM25 Relevance
   - TF-IDF Similarity
   - Difficulty Match
   - Freshness
   - Popularity
   - Source Authority
   - Graph Relevance
   - Duration Fit
   - Channel Authority
2. Ensemble Scoring:
   Weighted combination across all 9 extracted features.
3. MMR Diversification (Maximal Marginal Relevance):
   Selects Top-3 relevant and diverse resources per concept.
"""
from typing import List, Dict, Any, Optional
import math
import numpy as np

from app.ranking.schema import ResourceItem
from app.ranking.bm25 import BM25Okapi
from app.ranking.tfidf_ranker import TFIDFSimilarityRanker

# Recognized high-authority educational channels and publishers
TRUSTED_CHANNELS = {
    "mit opencourseware", "mit", "3blue1brown", "stanford", "harvard",
    "khan academy", "freecodecamp.org", "freecodecamp", "statquest with josh starmer",
    "statquest", "crashcourse", "computerphile", "two minute papers", "deeplearning.ai",
    "lex fridman", "cs50", "fireship", "sentdex", "edx", "coursera"
}

# Feature weights for ensemble scoring (sum to 1.0)
FEATURE_WEIGHTS = {
    "bm25": 0.22,
    "tfidf": 0.22,
    "difficulty_match": 0.15,
    "source_authority": 0.12,
    "channel_authority": 0.08,
    "graph_relevance": 0.08,
    "duration_fit": 0.05,
    "popularity": 0.05,
    "freshness": 0.03,
}


def normalize_resources(
    wiki_data: Optional[Dict],
    youtube_data: Optional[List[Dict]],
    ocw_data: Optional[List[Dict]],
) -> List[ResourceItem]:
    """
    Normalizes diverse ingested data (Phase A) into unified ResourceItem objects.
    Types: article, video, course, pdf.
    """
    resources: List[ResourceItem] = []
    r_idx = 1

    # 1. Wikipedia Article
    if wiki_data and wiki_data.get("found"):
        title = wiki_data.get("title") or "Wikipedia Overview"
        summary = wiki_data.get("summary") or ""
        url = wiki_data.get("source_url") or "https://en.wikipedia.org"
        resources.append(
            ResourceItem(
                id=f"res_{r_idx}",
                title=f"Wikipedia: {title}",
                url=url,
                resource_type="article",
                snippet=summary[:600],
                difficulty_level=2,
                duration_minutes=12.0,
                popularity_score=0.90,
                source_authority=0.92,
                channel_or_author="Wikimedia Foundation",
                published_date="2024",
            )
        )
        r_idx += 1

    # 2. YouTube Tutorials
    if youtube_data:
        for v in youtube_data:
            v_title = v.get("title") or "Video Tutorial"
            v_url = v.get("url") or ""
            channel = v.get("channel") or "YouTube Creator"
            views = v.get("view_count") or 1000
            desc = v.get("description", "") or ""
            transcript = v.get("transcript_snippet", "") or ""
            snippet = f"{desc[:300]} {transcript[:300]}".strip()

            # Compute normalized popularity log-scale (1k -> 0.4, 100k -> 0.7, 1M+ -> 0.95)
            pop = min(1.0, max(0.1, math.log10(max(10, views)) / 6.0))

            # Channel authority bonus
            ch_lower = channel.lower()
            ch_auth = 0.90 if any(tc in ch_lower for tc in TRUSTED_CHANNELS) else 0.65

            resources.append(
                ResourceItem(
                    id=f"res_{r_idx}",
                    title=v_title,
                    url=v_url,
                    resource_type="video",
                    snippet=snippet,
                    difficulty_level=3,
                    duration_minutes=18.0,
                    popularity_score=round(pop, 3),
                    source_authority=0.80,
                    channel_or_author=channel,
                    published_date=v.get("published_at", "2023"),
                )
            )
            r_idx += 1

    # 3. MIT OpenCourseWare Courses & Readings
    if ocw_data:
        for c in ocw_data:
            c_title = c.get("course_title") or "MIT OpenCourseWare Course"
            c_url = c.get("url") or "https://ocw.mit.edu"
            c_desc = c.get("description", "") or ""

            resources.append(
                ResourceItem(
                    id=f"res_{r_idx}",
                    title=f"MIT OCW: {c_title}",
                    url=c_url,
                    resource_type="course",
                    snippet=c_desc[:600],
                    difficulty_level=4,
                    duration_minutes=45.0,
                    popularity_score=0.75,
                    source_authority=1.00,
                    channel_or_author="Massachusetts Institute of Technology",
                    published_date=c.get("published", "2024"),
                )
            )
            r_idx += 1

    return resources


def _extract_duration_fit(duration_min: float) -> float:
    """Micro-learning sweet spot is 10-25 minutes."""
    if duration_min <= 0:
        return 0.5
    if 10 <= duration_min <= 25:
        return 1.0
    if duration_min < 10:
        return max(0.3, duration_min / 10.0)
    # > 25 mins: soft penalty
    return max(0.2, 1.0 - (duration_min - 25) / 60.0)


def rank_and_diversify_resources(
    concept_title: str,
    concept_description: str,
    concept_difficulty: int,
    candidate_resources: List[ResourceItem],
    top_k: int = 3,
    mmr_lambda: float = 0.65,
) -> List[ResourceItem]:
    """
    Ranks candidate resources for a single concept using all 9 features,
    applies Ensemble Scoring, and performs MMR Diversification to pick top_k.
    """
    if not candidate_resources:
        return []

    query = f"{concept_title} {concept_description}"
    corpus_texts = [f"{r.title} {r.snippet}" for r in candidate_resources]

    # 1. BM25 Relevance Feature
    bm25 = BM25Okapi(corpus_texts)
    bm25_raw = bm25.get_scores(query)
    max_bm25 = max(bm25_raw) if bm25_raw and max(bm25_raw) > 0 else 1.0
    bm25_norm = [min(1.0, s / max_bm25) for s in bm25_raw]

    # 2. TF-IDF Semantic Similarity Feature
    tfidf = TFIDFSimilarityRanker()
    tfidf.fit_corpus(corpus_texts)
    tfidf_sims = tfidf.score_query(query)
    doc_sim_matrix = tfidf.pairwise_similarity()

    # 3. Extract all 9 features & compute ensemble score
    scored_items: List[ResourceItem] = []
    for i, res in enumerate(candidate_resources):
        f_bm25 = float(bm25_norm[i]) if i < len(bm25_norm) else 0.0
        f_tfidf = float(tfidf_sims[i]) if i < len(tfidf_sims) else 0.0

        # Difficulty match [0, 1]
        f_diff = 1.0 - abs(concept_difficulty - res.difficulty_level) / 4.0

        # Freshness [0, 1]
        f_fresh = 0.85
        if res.published_date and any(yr in str(res.published_date) for yr in ("2024", "2025", "2026")):
            f_fresh = 1.0
        elif res.published_date and any(yr in str(res.published_date) for yr in ("2022", "2023")):
            f_fresh = 0.80

        # Popularity
        f_pop = res.popularity_score

        # Source Authority
        f_src = res.source_authority

        # Graph Relevance (title/topic alignment)
        f_graph = 0.5 * f_tfidf + 0.5 * f_bm25

        # Duration Fit
        f_dur = _extract_duration_fit(res.duration_minutes or 15.0)

        # Channel Authority
        f_chan = 0.90 if res.channel_or_author and any(tc in res.channel_or_author.lower() for tc in TRUSTED_CHANNELS) else 0.65

        # Ensemble Score
        ensemble = (
            FEATURE_WEIGHTS["bm25"] * f_bm25
            + FEATURE_WEIGHTS["tfidf"] * f_tfidf
            + FEATURE_WEIGHTS["difficulty_match"] * f_diff
            + FEATURE_WEIGHTS["source_authority"] * f_src
            + FEATURE_WEIGHTS["channel_authority"] * f_chan
            + FEATURE_WEIGHTS["graph_relevance"] * f_graph
            + FEATURE_WEIGHTS["duration_fit"] * f_dur
            + FEATURE_WEIGHTS["popularity"] * f_pop
            + FEATURE_WEIGHTS["freshness"] * f_fresh
        )

        item = res.model_copy()
        item.bm25_score = round(f_bm25, 4)
        item.tfidf_score = round(f_tfidf, 4)
        item.difficulty_match = round(f_diff, 4)
        item.freshness_score = round(f_fresh, 4)
        item.duration_fit = round(f_dur, 4)
        item.graph_relevance = round(f_graph, 4)
        item.channel_authority = round(f_chan, 4)
        item.ensemble_score = round(float(ensemble), 4)
        scored_items.append(item)

    # 4. Maximal Marginal Relevance (MMR) Diversification
    # Selects high-relevance items while penalizing redundancy (type & text overlap)
    selected_indices: List[int] = []
    remaining_indices = list(range(len(scored_items)))

    k_target = min(top_k, len(scored_items))

    while len(selected_indices) < k_target and remaining_indices:
        if not selected_indices:
            # Pick the highest ensemble scoring item first
            best_idx = max(remaining_indices, key=lambda idx: scored_items[idx].ensemble_score)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            mmr_scores = []
            for candidate_idx in remaining_indices:
                rel = scored_items[candidate_idx].ensemble_score
                cand_type = scored_items[candidate_idx].resource_type

                # Compute maximum similarity to any already-selected item
                max_sim = 0.0
                for sel_idx in selected_indices:
                    text_sim = float(doc_sim_matrix[candidate_idx, sel_idx]) if doc_sim_matrix.size > 0 else 0.0
                    # Additional penalty for matching resource types (e.g. 2 identical videos)
                    type_penalty = 0.25 if cand_type == scored_items[sel_idx].resource_type else 0.0
                    combined_sim = min(1.0, text_sim + type_penalty)
                    if combined_sim > max_sim:
                        max_sim = combined_sim

                mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
                mmr_scores.append((candidate_idx, mmr))

            best_candidate = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)

    final_results = []
    for idx in selected_indices:
        item = scored_items[idx]
        item.mmr_selected = True
        final_results.append(item)

    return final_results
