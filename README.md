# rag-quality-kit

[![PyPI](https://img.shields.io/pypi/v/rag-quality-kit.svg)](https://pypi.org/project/rag-quality-kit/)
[![Python](https://img.shields.io/pypi/pyversions/rag-quality-kit.svg)](https://pypi.org/project/rag-quality-kit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Heuristic quality metrics for RAG retrieval and grounded answers.** Zero runtime dependencies, pure-Python.

Python port of [@mukundakatta/rag-quality-kit](https://github.com/MukundaKatta/rag-quality-kit). The JS sibling has the original heuristics; this README sticks to the Python API.

## Install

```bash
pip install rag-quality-kit
```

## Usage

```python
from rag_quality_kit import score, missing_evidence

question = "Who wrote Hamlet and when was it first performed?"
contexts = [
    {"id": "doc-1", "text": "Hamlet is a tragedy by William Shakespeare, written around 1600."},
    {"id": "doc-2", "text": "Records suggest Hamlet was first performed in 1602."},
]
answer = "Hamlet was written by Shakespeare and first performed in 1602."

r = score(question, contexts, answer)
r.groundedness         # 0..1 -- answer terms that appear in any context
r.context_relevance    # 0..1 -- question terms covered by the contexts
r.answer_relevance     # 0..1 -- question terms covered by the answer
r.conciseness          # 0..1 -- 1.0 if answer is roughly question-sized, decays as it balloons
r.overall              # unweighted mean of the four

missing_evidence(answer, contexts)   # -> list[str] of answer terms not in any context
```

## Metrics

| Metric | Range | Behavior |
|---|---|---|
| `groundedness` | 0..1 | Fraction of answer terms found in any context. |
| `context_relevance` | 0..1 | Fraction of (longer) question terms covered by the contexts. Mirrors the JS `retrievalCoverage`. |
| `answer_relevance` | 0..1 | Fraction of question terms that the answer addresses. |
| `conciseness` | 0..1 | 1.0 when the answer is up to ~2x the question's term count; linearly decays to 0 at 10x. |
| `overall` | 0..1 | Unweighted mean of the four. |

All metrics are heuristic and token-overlap based -- fast, deterministic, no LLM calls. For evaluation-grade scoring layer an LLM judge on top.

## API differences from the JS sibling

* Python signature is `score(question, contexts, answer)` (positional) instead of `scoreRag({ query, answer, contexts })`.
* Returns a `QualityResult` dataclass instead of a plain object.
* Metric names: `context_relevance` (was `retrievalCoverage`), `groundedness` is unchanged. Adds two extra heuristics: `answer_relevance` and `conciseness`. The aggregate is `overall` (was `score`) and now averages all four.
* Drops the `citationCoverage` metric -- it's heavily citation-format dependent and best owned by the calling app. Use `missing_evidence(answer, contexts)` for an analogous signal.

See the JS sibling for the original heuristics and broader design notes.
