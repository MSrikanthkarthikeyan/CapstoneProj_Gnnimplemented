"""
TF-IDF Semantic Similarity Ranker using Scikit-Learn.
Computes n-gram (1-2) TF-IDF representations and cosine similarities.
"""
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFSimilarityRanker:
    """Computes TF-IDF vector representations and semantic cosine similarity."""

    def __init__(self, ngram_range=(1, 2), min_df=1):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
            norm="l2",
            sublinear_tf=True,
        )
        self.doc_vectors = None
        self.fitted = False

    def fit_corpus(self, corpus: List[str]):
        """Fit vectorizer on the corpus of resources."""
        if not corpus or all(len(d.strip()) == 0 for d in corpus):
            self.fitted = False
            return
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        self.fitted = True

    def score_query(self, query: str) -> np.ndarray:
        """Compute cosine similarity of query against all corpus documents."""
        if not self.fitted or self.doc_vectors is None or not query.strip():
            return np.zeros(0 if self.doc_vectors is None else self.doc_vectors.shape[0])

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()
        return np.clip(sims, 0.0, 1.0)

    def pairwise_similarity(self) -> np.ndarray:
        """Compute inter-document similarity matrix for MMR diversification."""
        if not self.fitted or self.doc_vectors is None:
            return np.zeros((0, 0))
        return cosine_similarity(self.doc_vectors)
