from __future__ import annotations

from typing import Iterable, Sequence


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        # For refuse questions, success is retrieving nothing critical — treat as 1.0 if empty relevant
        return 1.0
    top = list(retrieved)[:k]
    hits = sum(1 for r in relevant if r in top)
    return hits / len(relevant)


def hit_rate_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """1.0 if any relevant id appears in top-k; 0.0 otherwise. Empty relevant → 1.0."""
    if not relevant:
        return 1.0
    top = set(list(retrieved)[:k])
    return 1.0 if any(r in top for r in relevant) else 0.0


def mean_recall_at_k(
    cases: Iterable[tuple[Sequence[str], Sequence[str]]], k: int
) -> float:
    scores = [recall_at_k(rel, ret, k) for rel, ret in cases]
    return sum(scores) / len(scores) if scores else 0.0


def mean_hit_rate_at_k(
    cases: Iterable[tuple[Sequence[str], Sequence[str]]], k: int
) -> float:
    scores = [hit_rate_at_k(rel, ret, k) for rel, ret in cases]
    return sum(scores) / len(scores) if scores else 0.0


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def unsupported_claim_rate(
    answer: str,
    *,
    must_not_claim: Sequence[str],
    expect_uncertainty: bool,
    looks_uncertain: bool,
) -> float:
    """
    Return 1.0 if the answer hallucinates forbidden content, else 0.0.
    For adversarial/unknown questions, fabricating details counts as unsupported.
    """
    if not must_not_claim:
        return 0.0
    lower = answer.lower()
    invented = [c for c in must_not_claim if c.lower() in lower]
    if not invented:
        return 0.0
    # Uncertainty that still echoes the probed entity can be OK if clearly refusing
    if expect_uncertainty and looks_uncertain:
        # Still fail if it affirms the invented event (simple heuristic)
        affirm = any(
            phrase in lower
            for phrase in (
                "yes",
                "i remember",
                "we went",
                "i lived",
                "when she was",
            )
        )
        return 1.0 if affirm else 0.0
    return 1.0


def uncertainty_appropriate(
    *,
    expect_uncertainty: bool,
    must_refuse: bool,
    refused: bool,
    looks_uncertain: bool,
) -> float:
    """1.0 when uncertainty/refusal behavior matches the gold expectation."""
    if expect_uncertainty or must_refuse:
        return 1.0 if (refused or looks_uncertain) else 0.0
    # Answerable questions should not only refuse unless evidence missing
    return 1.0
