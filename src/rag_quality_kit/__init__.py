"""rag_quality_kit -- heuristic quality metrics for RAG retrieval and grounded answers.

Public surface (Python port of @mukundakatta/rag-quality-kit):

    from rag_quality_kit import score, missing_evidence, QualityResult

* ``score(question, contexts, answer)`` -- four heuristic metrics in [0, 1].
* ``missing_evidence(answer, contexts)`` -- terms in the answer absent from any context.
* ``QualityResult`` -- dataclass: groundedness, context_relevance, answer_relevance,
  conciseness, plus the aggregate ``overall``.

The implementation is zero-dependency and uses token-overlap heuristics (mirrors the
JS sibling). For evaluation-grade scoring, plug an LLM judge on top.
"""

from .score import (
    QualityResult,
    missing_evidence,
    score,
)

__version__ = "0.1.0"
VERSION = __version__

__all__ = [
    "VERSION",
    "QualityResult",
    "missing_evidence",
    "score",
]
