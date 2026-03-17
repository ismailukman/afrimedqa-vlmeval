from __future__ import annotations

from sacrebleu.metrics import CHRF

_chrf = CHRF()


def score_chrf(hypothesis: str, reference: str) -> float:
    """Sentence-level chrF score (0–100)."""
    return _chrf.sentence_score(hypothesis, [reference]).score


def score_chrf_corpus(hypotheses: list[str], references: list[str]) -> float:
    """Corpus-level chrF score (0–100)."""
    return _chrf.corpus_score(hypotheses, [references]).score
