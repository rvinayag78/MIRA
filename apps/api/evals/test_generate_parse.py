from uuid import UUID

from app.services.generate import (
    REFUSAL,
    Citation,
    GroundedAnswer,
    _parse_json,
    check_citations_valid,
    broaden_retrieval_query,
    expand_retrieval_query,
    uncited_claims_should_refuse,
)
from app.services.retrieve import RetrievedChunk


def test_parse_json_fenced():
    raw = 'Here you go:\n{"answer": "hi", "citations": [], "confidence": 0.5}\n'
    parsed = _parse_json(raw)
    assert parsed["answer"] == "hi"


def test_citation_validation():
    cid = UUID("11111111-1111-1111-1111-111111111111")
    chunks = [
        RetrievedChunk(
            id=cid,
            memory_id=cid,
            text="blue bungalow",
            speaker=None,
            ts_start=None,
            ts_end=None,
            score=1.0,
        )
    ]
    ok = GroundedAnswer(
        answer="It was blue",
        citations=[Citation(chunk_id=str(cid), quote="blue bungalow", memory_id=str(cid))],
    )
    punctuated = GroundedAnswer(
        answer="It was a blue bungalow.",
        citations=[Citation(chunk_id=str(cid), quote="blue bungalow,")],
    )
    bad_id = GroundedAnswer(
        answer="Nope",
        citations=[Citation(chunk_id="00000000-0000-0000-0000-000000000000", quote="x")],
    )
    invented = GroundedAnswer(
        answer="It was a red victorian",
        citations=[Citation(chunk_id=str(cid), quote="red victorian mansion")],
    )
    assert check_citations_valid(ok, chunks)
    assert check_citations_valid(punctuated, chunks)
    assert not check_citations_valid(bad_id, chunks)
    assert not check_citations_valid(invented, chunks)


def test_expand_retrieval_query_follow_up():
    history = [{"role": "user", "content": "What was your childhood home like?"}]
    assert "childhood home" in expand_retrieval_query("tell me more", history)
    assert expand_retrieval_query("Where did you work?", history) == "Where did you work?"


def test_broaden_retrieval_query_family_and_early():
    dad = broaden_retrieval_query("What was dad like?")
    assert "father" in dad
    early = broaden_retrieval_query("what's an early memory?")
    assert "childhood" in early
    assert broaden_retrieval_query("Where did you work?") == "Where did you work?"


def test_refusal_constant():
    assert "recorded memories" in REFUSAL.lower()


def test_uncited_partial_refuses_even_when_model_marks_uncertain():
    """Hedged inventions must not bypass the citation guard.

    Concrete trigger: coverage=partial, empty citations, model sets
    uncertainty:true (or answer contains "not sure") while inventing a pet /
    event that was never recorded. Pre-fix, ``not uncertainty_flag`` exempted
    these answers and the keeper heard false maker memories.
    """
    assert uncited_claims_should_refuse(has_citations=False, coverage="partial")
    assert uncited_claims_should_refuse(has_citations=False, coverage="none")
    assert not uncited_claims_should_refuse(has_citations=True, coverage="partial")
    # strong-coverage uncited path left to the separate coverage-guard fix
    assert not uncited_claims_should_refuse(has_citations=False, coverage="strong")


def test_parse_contextualized_keeps_every_chunk():
    from app.services.voyage import _parse_contextualized

    data = {
        "data": [
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                    {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
                ],
            }
        ]
    }
    assert _parse_contextualized(data) == [[0.1, 0.2], [0.3, 0.4]]
