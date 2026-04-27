"""Heuristic RAG quality scoring.

Four metrics, each in ``[0, 1]``:

* ``groundedness``       -- fraction of answer terms that appear in any context.
                            Mirrors the JS sibling's ``groundedness``.
* ``context_relevance``  -- fraction of (longer) question terms covered by the contexts.
                            Mirrors the JS sibling's ``retrievalCoverage``.
* ``answer_relevance``   -- fraction of question terms covered by the answer.
* ``conciseness``        -- 1.0 when the answer is roughly the size of the question or
                            shorter, decays as the answer balloons relative to the
                            question.

``overall`` is the unweighted mean of the four.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# Tokenizer mirrors the JS heuristic: lowercase + alphanumeric runs.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class QualityResult:
    """Heuristic RAG quality metrics in ``[0, 1]``."""

    groundedness: float
    context_relevance: float
    answer_relevance: float
    conciseness: float
    overall: float


def _normalize(text: object) -> str:
    return str(text).lower()


def _terms(text: object) -> set[str]:
    return set(_TOKEN_RE.findall(_normalize(text)))


def _context_text(contexts: Sequence[object]) -> str:
    """Each context can be a str or a dict with a ``text`` field (mirrors JS)."""
    parts: list[str] = []
    for c in contexts:
        if isinstance(c, dict):
            parts.append(str(c.get("text", "")))
        else:
            parts.append(str(c))
    return " ".join(parts)


def _ratio(num: int, den: int) -> float:
    # Mirrors JS: ``ratio(n, d) = d ? n/d : 1`` -- empty denominator is a free pass.
    if den <= 0:
        return 1.0
    return _round(num / den)


def _round(n: float) -> float:
    # Mirrors the JS ``round`` to 4 decimals.
    return round(n * 10000) / 10000


def score(
    question: str,
    contexts: Sequence[object],
    answer: str,
) -> QualityResult:
    """Score a (question, contexts, answer) triple with four heuristic metrics.

    Args:
        question: The original user query.
        contexts: List of retrieved chunks. Each entry may be a string or a dict
            with a ``text`` key (and optionally an ``id``).
        answer: The model's generated answer.

    Returns:
        QualityResult with four metrics and the unweighted aggregate ``overall``.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a str")
    if not isinstance(answer, str):
        raise TypeError("answer must be a str")
    if not isinstance(contexts, (list, tuple)):
        raise TypeError("contexts must be a list or tuple")

    q_terms = _terms(question)
    a_terms = _terms(answer)
    ctx_terms = _terms(_context_text(contexts))

    # ``required`` = JS heuristic: long-ish question terms (>3 chars).
    required = {t for t in q_terms if len(t) > 3}

    context_relevance = _ratio(
        sum(1 for t in required if t in ctx_terms),
        len(required),
    )
    groundedness = _ratio(
        sum(1 for t in a_terms if t in ctx_terms),
        len(a_terms),
    )
    answer_relevance = _ratio(
        sum(1 for t in q_terms if t in a_terms),
        len(q_terms),
    )

    # Conciseness: 1.0 when answer length <= 2x question length, decays linearly to 0
    # at 10x. This is heuristic; tune ``threshold`` knobs by reading the metric.
    q_len = max(len(q_terms), 1)
    a_len = len(a_terms)
    if a_len <= 2 * q_len:
        conciseness = 1.0
    elif a_len >= 10 * q_len:
        conciseness = 0.0
    else:
        # Linear decay from (2x, 1.0) to (10x, 0.0).
        conciseness = _round(1.0 - (a_len - 2 * q_len) / (8 * q_len))

    overall = _round(
        (groundedness + context_relevance + answer_relevance + conciseness) / 4
    )
    return QualityResult(
        groundedness=groundedness,
        context_relevance=context_relevance,
        answer_relevance=answer_relevance,
        conciseness=conciseness,
        overall=overall,
    )


def missing_evidence(answer: str, contexts: Sequence[object]) -> list[str]:
    """Return the answer terms (>4 chars) not present in any context.

    Mirrors the JS sibling's ``missingEvidence`` helper.
    """
    if not isinstance(answer, str):
        raise TypeError("answer must be a str")
    if not isinstance(contexts, (list, tuple)):
        raise TypeError("contexts must be a list or tuple")
    ctx_terms = _terms(_context_text(contexts))
    return sorted({t for t in _terms(answer) if len(t) > 4 and t not in ctx_terms})
