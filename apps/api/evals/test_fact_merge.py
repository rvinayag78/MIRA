"""Regression: distinct relationship facts must not collapse into one established claim."""

from __future__ import annotations

from app.db.queries import should_merge_knowledge_facts


def test_different_sister_names_do_not_merge_even_when_near_neighbors():
    """Templated 'Has a sister named X' facts sit close in embed space (~dist 0.1)."""
    assert not should_merge_knowledge_facts(
        distance=0.12,
        existing_statement="Has a sister named Lena.",
        new_statement="Has a sister named Laura.",
        existing_people=["Lena"],
        new_people=["Laura"],
        existing_places=[],
        new_places=[],
    )


def test_different_names_in_statement_alone_block_merge():
    """LLM drafts may omit people[]; statement text must still prevent collapse."""
    assert not should_merge_knowledge_facts(
        distance=0.1,
        existing_statement="Has a brother named Mark.",
        new_statement="Has a brother named Mike.",
        existing_people=[],
        new_people=[],
        existing_places=[],
        new_places=[],
    )


def test_same_person_near_neighbor_still_merges():
    assert should_merge_knowledge_facts(
        distance=0.08,
        existing_statement="Has a sister named Lena.",
        new_statement="Has a sister named Lena.",
        existing_people=["Lena"],
        new_people=["Lena"],
        existing_places=[],
        new_places=[],
    )


def test_conflicting_years_do_not_merge():
    assert not should_merge_knowledge_facts(
        distance=0.15,
        existing_statement="Moved to Portland in 1998 with family.",
        new_statement="Moved to Portland in 2001 with family.",
        existing_people=[],
        new_people=[],
        existing_places=["Portland"],
        new_places=["Portland"],
    )


def test_different_places_do_not_merge():
    assert not should_merge_knowledge_facts(
        distance=0.14,
        existing_statement="Connected to place: Portland.",
        new_statement="Connected to place: Seattle.",
        existing_people=[],
        new_people=[],
        existing_places=["Portland"],
        new_places=["Seattle"],
    )


def test_far_neighbors_never_merge():
    assert not should_merge_knowledge_facts(
        distance=0.4,
        existing_statement="Has a sister named Lena.",
        new_statement="Has a sister named Lena.",
        existing_people=["Lena"],
        new_people=["Lena"],
        existing_places=[],
        new_places=[],
    )
