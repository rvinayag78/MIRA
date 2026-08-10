from app.services.generate import REFUSAL, _parse_json, check_citations_valid
from app.services.retrieve import RetrievedChunk
from uuid import UUID


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
    from app.services.generate import Citation, GroundedAnswer

    ok = GroundedAnswer(
        answer="It was blue",
        citations=[Citation(chunk_id=str(cid), quote="blue bungalow")],
    )
    bad = GroundedAnswer(
        answer="Nope",
        citations=[Citation(chunk_id="00000000-0000-0000-0000-000000000000", quote="x")],
    )
    assert check_citations_valid(ok, chunks)
    assert not check_citations_valid(bad, chunks)


def test_refusal_constant():
    assert "recorded memories" in REFUSAL.lower()
