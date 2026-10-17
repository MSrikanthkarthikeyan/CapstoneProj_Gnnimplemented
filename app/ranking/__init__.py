"""
Resource Ranking and Diversification Module.
"""
from app.ranking.schema import ResourceItem
from app.ranking.bm25 import BM25Okapi
from app.ranking.tfidf_ranker import TFIDFSimilarityRanker
from app.ranking.resource_ranker import (
    normalize_resources,
    rank_and_diversify_resources,
    FEATURE_WEIGHTS,
)

__all__ = [
    "ResourceItem",
    "BM25Okapi",
    "TFIDFSimilarityRanker",
    "normalize_resources",
    "rank_and_diversify_resources",
    "FEATURE_WEIGHTS",
]
