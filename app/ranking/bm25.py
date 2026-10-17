"""
Okapi BM25 Implementation for Information Retrieval on Curriculum Resources.

Formula:
  IDF(q_i) = ln((N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1.0)
  BM25(D, Q) = sum_{q_i in Q} IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl)))
"""
import math
import re
from typing import List, Dict, Tuple

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """Tokenize and filter alphanumeric words."""
    words = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


class BM25Okapi:
    """Okapi BM25 indexer and ranker."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []

        if not corpus:
            self.avgdl = 0.0
            return

        total_tokens = 0
        for doc in corpus:
            tokens = tokenize(doc)
            self.doc_lengths.append(len(tokens))
            total_tokens += len(tokens)

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            for t in set(tokens):
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avgdl = total_tokens / max(self.corpus_size, 1)

    def _idf(self, term: str) -> float:
        n_q = self.doc_freqs.get(term, 0)
        return math.log((self.corpus_size - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score of a query against document at doc_idx."""
        if doc_idx < 0 or doc_idx >= self.corpus_size or self.avgdl == 0:
            return 0.0

        q_tokens = tokenize(query)
        if not q_tokens:
            return 0.0

        doc_len = self.doc_lengths[doc_idx]
        tf = self.term_freqs[doc_idx]
        score = 0.0

        for t in q_tokens:
            if t not in tf:
                continue
            freq = tf[t]
            idf = self._idf(t)
            numerator = freq * (self.k1 + 1.0)
            denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * (numerator / denominator)

        return max(0.0, score)

    def get_scores(self, query: str) -> List[float]:
        """Compute BM25 scores for all documents in the corpus."""
        return [self.score(query, i) for i in range(self.corpus_size)]
