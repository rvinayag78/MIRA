"""Evidence packaging for grounded generation.

Builds a structured package from retrieved episodic chunks + semantic facts,
detects simple conflicts, and preserves provenance for claim tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.memory_types import (
    ChunkMeta,
    ConflictNote,
    QueryUnderstanding,
    RetrievedFact,
)
from app.services.retrieve import RetrievedChunk


UNCERTAINTY_PHRASES = (
    "I don't think I ever recorded much about that.",
    "I remember talking about that a little, but I don't remember enough to say for sure.",
    "I don't have a clear memory of that.",
)


@dataclass
class EvidencePackage:
    question: str
    query: QueryUnderstanding | None
    episodic: list[RetrievedChunk] = field(default_factory=list)
    semantic: list[RetrievedFact] = field(default_factory=list)
    conflicts: list[ConflictNote] = field(default_factory=list)
    coverage: str = "none"  # none | partial | strong

    def to_prompt_dict(self) -> dict[str, Any]:
        episodic = []
        for c in self.episodic:
            meta = c.meta or {}
            episodic.append(
                {
                    "chunk_id": str(c.id),
                    "memory_id": str(c.memory_id),
                    "text": c.text,
                    "speaker": c.speaker,
                    "salience": c.salience,
                    "score": c.score,
                    "people": meta.get("people") or [],
                    "places": meta.get("places") or [],
                    "time_periods": meta.get("time_periods") or [],
                    "topics": meta.get("topics") or [],
                    "memorable_phrases": meta.get("memorable_phrases") or [],
                    "important_details": meta.get("important_details") or [],
                    "emotional_tone": meta.get("emotional_tone"),
                }
            )
        return {
            "question": self.question,
            "coverage": self.coverage,
            "episodic_memories": episodic,
            "semantic_knowledge": [f.to_dict() for f in self.semantic],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "guidance": {
                "supported": "State only when directly present in episodic text or established facts.",
                "inferred": "Allowed only when multiple evidence pieces clearly imply it; hedge.",
                "unknown": "Use natural uncertainty; do not invent experiences.",
            },
        }

    def chunk_id_to_memory(self) -> dict[str, str]:
        return {str(c.id): str(c.memory_id) for c in self.episodic}


def detect_conflicts(chunks: list[RetrievedChunk]) -> list[ConflictNote]:
    """Flag conflicting years/places mentioned across memories for the same topic."""
    by_topic: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        meta = ChunkMeta.from_dict(c.meta if isinstance(c.meta, dict) else {})
        for topic in meta.topics or ["general"]:
            by_topic.setdefault(topic.lower(), []).append(c)

    conflicts: list[ConflictNote] = []
    for topic, group in by_topic.items():
        if len({c.memory_id for c in group}) < 2:
            continue
        years: dict[str, list[RetrievedChunk]] = {}
        for c in group:
            meta = ChunkMeta.from_dict(c.meta if isinstance(c.meta, dict) else {})
            for tp in meta.time_periods:
                key = tp.text or str(tp.approx_year or "")
                if key:
                    years.setdefault(key, []).append(c)
        if len(years) >= 2:
            descriptions = [
                f"{year}: memory {c.memory_id} — {c.text[:120]}"
                for year, items in years.items()
                for c in items[:1]
            ]
            mem_ids = sorted({str(c.memory_id) for items in years.values() for c in items})
            conflicts.append(
                ConflictNote(topic=topic, descriptions=descriptions, memory_ids=mem_ids)
            )
    return conflicts


def estimate_coverage(
    question: str,
    chunks: list[RetrievedChunk],
    facts: list[RetrievedFact],
) -> str:
    if not chunks and not facts:
        return "none"
    q_tokens = {t for t in question.lower().split() if len(t) > 2}
    if not q_tokens:
        return "partial" if chunks or facts else "none"
    best = 0.0
    for c in chunks:
        c_tokens = set(c.text.lower().split())
        overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
        best = max(best, overlap)
        if c.score >= 0.15:
            best = max(best, 0.45)
    for f in facts:
        if f.status == "established" and f.score >= 0.1:
            best = max(best, 0.5)
        f_tokens = set(f.statement.lower().split())
        best = max(best, len(q_tokens & f_tokens) / max(1, len(q_tokens)))
    if best >= 0.45:
        return "strong"
    if best >= 0.15 or chunks:
        return "partial"
    return "none"


def build_evidence_package(
    question: str,
    chunks: list[RetrievedChunk],
    facts: list[RetrievedFact] | None = None,
    *,
    query: QueryUnderstanding | None = None,
) -> EvidencePackage:
    facts = facts or []
    # Prefer established, then disputed (for uncertainty), then high-confidence candidates
    ranked_facts = sorted(
        facts,
        key=lambda f: (
            {"established": 3, "disputed": 2, "candidate": 1}.get(f.status, 0),
            f.confidence,
            f.score,
        ),
        reverse=True,
    )[:6]
    conflicts = detect_conflicts(chunks)
    for f in ranked_facts:
        if f.status == "disputed" and f.conflict_note:
            conflicts.append(
                ConflictNote(
                    topic=f.category,
                    descriptions=[f.conflict_note, f.statement],
                    memory_ids=[str(m) for m in f.supporting_memory_ids],
                )
            )
    coverage = estimate_coverage(question, chunks, ranked_facts)
    return EvidencePackage(
        question=question,
        query=query,
        episodic=chunks,
        semantic=ranked_facts,
        conflicts=conflicts,
        coverage=coverage,
    )
