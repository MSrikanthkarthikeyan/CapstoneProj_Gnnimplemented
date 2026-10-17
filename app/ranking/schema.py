"""
Schema definitions for Resource Ingestion and Ranking (Resource v2).
Normalizes resources across Wikipedia, YouTube, and MIT OCW.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ResourceItem(BaseModel):
    id: str = Field(..., description="Unique resource identifier")
    title: str
    url: str
    resource_type: Literal["video", "article", "pdf", "course"] = "article"
    snippet: str = ""
    difficulty_level: int = Field(default=3, ge=1, le=5)
    duration_minutes: Optional[float] = Field(default=15.0, description="Estimated duration in minutes")
    popularity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_authority: float = Field(default=0.7, ge=0.0, le=1.0)
    channel_or_author: Optional[str] = None
    published_date: Optional[str] = None

    # Ranking & ML Features
    bm25_score: float = 0.0
    tfidf_score: float = 0.0
    difficulty_match: float = 0.0
    freshness_score: float = 0.0
    duration_fit: float = 0.0
    graph_relevance: float = 0.0
    channel_authority: float = 0.0
    ensemble_score: float = 0.0
    mmr_selected: bool = False
