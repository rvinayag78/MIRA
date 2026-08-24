from app.services.extract import (
    clean_transcript_light,
    fact_status_for_draft,
    heuristic_chunk_meta,
    heuristic_fact_drafts,
)
from app.services.memory_types import KnowledgeFactDraft


def test_clean_transcript_light_collapses_whitespace():
    assert clean_transcript_light("  hello   world \n") == "hello world"


def test_heuristic_extracts_sister_lena():
    text = "My sister Lena and I carved our initials into the porch railing in 1998."
    meta = heuristic_chunk_meta(text)
    assert any(p.name == "Lena" and p.relationship == "sister" for p in meta.people)
    assert any(t.approx_year == 1998 for t in meta.time_periods)


def test_heuristic_fact_drafts_relationship():
    text = "My sister Lena and I carved our initials into the porch railing in 1998."
    meta = heuristic_chunk_meta(text)
    drafts = heuristic_fact_drafts([meta], [text])
    assert any("Lena" in d.statement and d.category == "relationship" for d in drafts)


def test_fact_status_single_anecdote_stays_candidate():
    draft = KnowledgeFactDraft(
        statement="Loved hiking once on a trip.",
        category="preference",
        confidence=0.6,
        explicitly_stated=False,
    )
    assert fact_status_for_draft(draft, evidence_count=1) == "candidate"


def test_fact_status_explicit_relationship_established():
    draft = KnowledgeFactDraft(
        statement="Has a sister named Lena.",
        category="relationship",
        confidence=0.9,
        explicitly_stated=True,
    )
    assert fact_status_for_draft(draft, evidence_count=1) == "established"
