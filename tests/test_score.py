"""Tests for ``rag_quality_kit.score`` and the ``missing_evidence`` helper."""

from __future__ import annotations

import pytest

from rag_quality_kit import QualityResult, missing_evidence, score


def test_happy_path_perfect_alignment():
    """Question, contexts, and answer all overlap heavily -- expect high scores."""
    q = "Tell me about Hamlet Shakespeare written 1600."
    ctx = [{"id": "d1", "text": "Hamlet was written by William Shakespeare around 1600."}]
    a = "Hamlet was written by Shakespeare in 1600."
    r = score(q, ctx, a)
    assert isinstance(r, QualityResult)
    assert r.groundedness >= 0.8
    # All long-ish question terms (hamlet, shakespeare, written, about, tell) are
    # in the context except 'tell' and 'about' -> context_relevance should be high.
    assert r.context_relevance >= 0.5
    assert r.answer_relevance >= 0.5
    assert r.conciseness == 1.0
    assert 0.0 <= r.overall <= 1.0


def test_groundedness_drops_when_answer_invents_terms():
    """Answer adds facts absent from the contexts -> groundedness penalized."""
    q = "Where is the Eiffel Tower?"
    ctx = ["The Eiffel Tower is in Paris."]
    a_grounded = "The Eiffel Tower is in Paris."
    a_invented = "The Eiffel Tower is in Tokyo near Mount Fuji shrine."
    r_grounded = score(q, ctx, a_grounded)
    r_invented = score(q, ctx, a_invented)
    assert r_grounded.groundedness > r_invented.groundedness


def test_context_relevance_zero_when_contexts_unrelated():
    """When the contexts have no overlap with question terms, context_relevance = 0."""
    q = "What is the capital of France?"
    ctx = ["Photosynthesis converts sunlight into chemical energy."]
    a = "Paris."
    r = score(q, ctx, a)
    assert r.context_relevance == 0.0


def test_conciseness_penalizes_long_answers():
    """Conciseness drops as the answer's unique-term count balloons vs the question's."""
    q = "What is 2 plus 2?"
    ctx = ["The sum 2 + 2 equals 4."]
    a_short = "Four."
    # Many distinct tokens -> term-set is large -> conciseness drops.
    a_long = " ".join(f"word{i}" for i in range(200))
    r_short = score(q, ctx, a_short)
    r_long = score(q, ctx, a_long)
    assert r_short.conciseness == 1.0
    assert r_long.conciseness == 0.0


def test_empty_contexts_does_not_crash():
    """Empty contexts: ratio with zero denominator returns 1.0 (mirrors JS)."""
    r = score("hi", [], "hello")
    # No required terms (>3 chars) in 'hi', no context terms, no answer terms covered.
    assert isinstance(r, QualityResult)
    assert 0.0 <= r.overall <= 1.0


def test_string_contexts_supported():
    """Contexts can be plain strings, not just dicts."""
    r = score("hello world", ["hello universe"], "hello")
    assert r.groundedness == 1.0  # 'hello' is in the contexts


def test_missing_evidence_lists_uncovered_terms():
    a = "Tokyo Berlin Paris"
    ctx = ["Paris is wonderful."]
    out = missing_evidence(a, ctx)
    # 'tokyo' and 'berlin' are >4 chars and not in contexts; 'paris' is covered.
    assert "tokyo" in out
    assert "berlin" in out
    assert "paris" not in out


def test_missing_evidence_empty_when_all_grounded():
    out = missing_evidence("paris", ["paris paris paris"])
    assert out == []


def test_score_rejects_non_string_question():
    with pytest.raises(TypeError):
        score(123, [], "hi")  # type: ignore[arg-type]


def test_score_rejects_non_list_contexts():
    with pytest.raises(TypeError):
        score("q", "not a list", "a")  # type: ignore[arg-type]


def test_threshold_tuning_groundedness_partial():
    """Half the answer terms are in the contexts -> groundedness ~0.5."""
    q = "what?"
    ctx = ["alpha beta"]
    a = "alpha beta gamma delta"
    r = score(q, ctx, a)
    # 2 of 4 answer terms covered -> 0.5
    assert r.groundedness == 0.5


def test_overall_is_average_of_four():
    """The overall metric is the unweighted mean of the four."""
    q = "hello world"
    ctx = ["hello universe"]
    a = "hello"
    r = score(q, ctx, a)
    expected = round(
        (r.groundedness + r.context_relevance + r.answer_relevance + r.conciseness)
        / 4
        * 10000
    ) / 10000
    assert r.overall == expected
