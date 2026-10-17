"""
Pydantic models for the Prerequisite Knowledge Graph (Auto-HKG).
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from app.ranking.schema import ResourceItem


class ConceptNode(BaseModel):
    id: str = Field(..., description="Short unique node id, e.g. 'c1'")
    title: str
    description: str
    difficulty: int = Field(..., ge=1, le=5, description="1=beginner ... 5=advanced")
    source_url: Optional[str] = None
    resources: List[ResourceItem] = Field(default_factory=list, description="Top-3 relevant and diverse resources")
    bm25_score: float = Field(default=0.0, description="Normalized BM25 relevance to concept")
    tfidf_score: float = Field(default=0.0, description="TF-IDF semantic similarity to concept")
    resource_quality: float = Field(default=0.0, description="Aggregate resource authority score")


class PrerequisiteEdge(BaseModel):
    source: str = Field(..., description="Prerequisite concept id")
    target: str = Field(..., description="Dependent concept id")
    relation: str = Field(default="prerequisite_for")

    @field_validator("relation")
    @classmethod
    def _relation_fixed(cls, v):
        return "prerequisite_for"


class KnowledgeGraph(BaseModel):
    goal: str
    nodes: List[ConceptNode]
    edges: List[PrerequisiteEdge]

    def node_ids(self) -> set:
        return {n.id for n in self.nodes}


class MasteryState(BaseModel):
    """Tracks a learner's progress through the graph."""
    session_id: str
    goal: str
    node_scores: dict = Field(default_factory=dict)   # node_id -> best score (0-1)
    node_status: dict = Field(default_factory=dict)   # node_id -> "locked"|"available"|"in_progress"|"mastered"
    current_node: Optional[str] = None
