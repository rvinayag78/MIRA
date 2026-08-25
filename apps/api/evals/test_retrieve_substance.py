from __future__ import annotations

from uuid import UUID

from app.services.evidence import build_evidence_package, estimate_coverage
from app.services.retrieve import (
    RetrievedChunk,
    boost_chunk_score,
    diversify_chunks,
    understand_query,
)


def _chunk(cid: str, mid: str, text: str, score: float, meta: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        id=UUID(cid),
        memory_id=UUID(mid),
        text=text,
        speaker=None,
        ts_start=None,
        ts_end=None,
        score=score,
        meta=meta or {},
        salience=0.7,
    )


def test_understand_query_family_and_year():
    q = understand_query("What was dad like in 2008?")
    assert "dad" in q.people
    assert "2008" in q.time_hints


def test_diversify_limits_per_memory():
    m1 = "11111111-1111-1111-1111-111111111111"
    m2 = "22222222-2222-2222-2222-222222222222"
    chunks = [
        _chunk("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1", m1, "a1", 1.0),
        _chunk("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa2", m1, "a2", 0.9),
        _chunk("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa3", m1, "a3", 0.8),
        _chunk("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbb1", m2, "b1", 0.7),
    ]
    picked = diversify_chunks(chunks, final_k=3, max_per_memory=2)
    assert len(picked) == 3
    assert sum(1 for c in picked if str(c.memory_id) == m1) <= 2
    assert any(str(c.memory_id) == m2 for c in picked)


def test_boost_entity_overlap():
    q = understand_query("Tell me about Lena")
    chunk = _chunk(
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "11111111-1111-1111-1111-111111111111",
        "My sister Lena carved initials.",
        0.2,
        meta={"people": [{"name": "Lena", "relationship": "sister", "confidence": 0.9}]},
    )
    assert boost_chunk_score(chunk, q) > chunk.score


def test_conflict_detection_and_coverage():
    chunks = [
        _chunk(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "11111111-1111-1111-1111-111111111111",
            "We carved initials in 1998.",
            0.9,
            meta={
                "topics": ["family"],
                "time_periods": [{"text": "1998", "approx_year": 1998, "confidence": 0.9}],
            },
        ),
        _chunk(
            "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "22222222-2222-2222-2222-222222222222",
            "I think it was 1999.",
            0.8,
            meta={
                "topics": ["family"],
                "time_periods": [{"text": "1999", "approx_year": 1999, "confidence": 0.6}],
            },
        ),
    ]
    pkg = build_evidence_package("What year?", chunks)
    assert pkg.conflicts
    assert estimate_coverage("Yosemite with Sarah", [], []) == "none"


def test_unrelated_established_fact_does_not_force_strong_coverage():
    """Nearest-neighbor established facts must not mark an unanswered question as strong."""
    from app.services.memory_types import RetrievedFact

    fact = RetrievedFact(
        id=UUID("33333333-3333-3333-3333-333333333333"),
        statement="Has a sister named Priya.",
        category="relationship",
        confidence=0.9,
        status="established",
        people=["Priya"],
        places=[],
        topics=["family"],
        supporting_chunk_ids=[],
        supporting_memory_ids=[UUID("11111111-1111-1111-1111-111111111111")],
        conflict_note=None,
        score=0.42,  # plausible NN score for same-agent English text
    )
    assert (
        estimate_coverage("What is your favorite color?", [], [fact]) != "strong"
    )


def test_relevant_established_fact_can_be_strong():
    from app.services.memory_types import RetrievedFact

    fact = RetrievedFact(
        id=UUID("33333333-3333-3333-3333-333333333333"),
        statement="Has a sister named Priya.",
        category="relationship",
        confidence=0.9,
        status="established",
        people=["Priya"],
        places=[],
        topics=["family"],
        supporting_chunk_ids=[],
        supporting_memory_ids=[UUID("11111111-1111-1111-1111-111111111111")],
        conflict_note=None,
        score=0.55,
    )
    assert estimate_coverage("Tell me about your sister Priya", [], [fact]) == "strong"
