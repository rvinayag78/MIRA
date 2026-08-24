"""Structured memory types for the substance / grounding layer.

Episodic memory = specific experiences stored as transcript chunks (+ metadata).
Semantic / person knowledge = facts distilled only when explicitly supported.
Raw transcript text remains the source of truth; metadata is derivative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import UUID

SupportLevel = Literal["supported", "inferred", "unknown"]
FactCategory = Literal[
    "preference", "relationship", "belief", "attribute", "event_summary"
]
FactStatus = Literal["candidate", "established", "disputed"]


@dataclass
class MentionedPerson:
    name: str
    relationship: str | None = None
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MentionedPlace:
    name: str
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimePeriod:
    text: str
    approx_year: int | None = None
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MentionedEvent:
    summary: str
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkMeta:
    """Structured metadata extracted from a chunk. Never invent unsupported fields."""

    people: list[MentionedPerson] = field(default_factory=list)
    places: list[MentionedPlace] = field(default_factory=list)
    time_periods: list[TimePeriod] = field(default_factory=list)
    events: list[MentionedEvent] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    emotional_tone: str | None = None
    important_details: list[str] = field(default_factory=list)
    memorable_phrases: list[str] = field(default_factory=list)
    cleaned_text: str | None = None
    extraction_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "people": [p.to_dict() for p in self.people],
            "places": [p.to_dict() for p in self.places],
            "time_periods": [t.to_dict() for t in self.time_periods],
            "events": [e.to_dict() for e in self.events],
            "topics": list(self.topics),
            "emotional_tone": self.emotional_tone,
            "important_details": list(self.important_details),
            "memorable_phrases": list(self.memorable_phrases),
            "cleaned_text": self.cleaned_text,
            "extraction_confidence": self.extraction_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ChunkMeta:
        if not data:
            return cls()
        people = [
            MentionedPerson(
                name=str(p.get("name", "")).strip(),
                relationship=(str(p["relationship"]) if p.get("relationship") else None),
                confidence=float(p.get("confidence") or 0.5),
            )
            for p in (data.get("people") or [])
            if isinstance(p, dict) and str(p.get("name", "")).strip()
        ]
        places = [
            MentionedPlace(
                name=str(p.get("name", "")).strip(),
                confidence=float(p.get("confidence") or 0.5),
            )
            for p in (data.get("places") or [])
            if isinstance(p, dict) and str(p.get("name", "")).strip()
        ]
        time_periods = [
            TimePeriod(
                text=str(t.get("text", "")).strip(),
                approx_year=(
                    int(t["approx_year"])
                    if t.get("approx_year") is not None
                    else None
                ),
                confidence=float(t.get("confidence") or 0.5),
            )
            for t in (data.get("time_periods") or [])
            if isinstance(t, dict) and str(t.get("text", "")).strip()
        ]
        events = [
            MentionedEvent(
                summary=str(e.get("summary", "")).strip(),
                confidence=float(e.get("confidence") or 0.5),
            )
            for e in (data.get("events") or [])
            if isinstance(e, dict) and str(e.get("summary", "")).strip()
        ]
        return cls(
            people=people,
            places=places,
            time_periods=time_periods,
            events=events,
            topics=[str(t).strip() for t in (data.get("topics") or []) if str(t).strip()],
            emotional_tone=(
                str(data["emotional_tone"]).strip()
                if data.get("emotional_tone")
                else None
            ),
            important_details=[
                str(d).strip()
                for d in (data.get("important_details") or [])
                if str(d).strip()
            ],
            memorable_phrases=[
                str(p).strip()
                for p in (data.get("memorable_phrases") or [])
                if str(p).strip()
            ],
            cleaned_text=(
                str(data["cleaned_text"]).strip() if data.get("cleaned_text") else None
            ),
            extraction_confidence=float(data.get("extraction_confidence") or 0.0),
        )

    def entity_names(self) -> set[str]:
        names = {p.name.lower() for p in self.people}
        names |= {p.name.lower() for p in self.places}
        names |= {t.lower() for t in self.topics}
        return names


@dataclass
class KnowledgeFactDraft:
    """Candidate semantic / person-knowledge statement before persistence."""

    statement: str
    category: FactCategory = "attribute"
    people: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    confidence: float = 0.5
    salience: float = 0.5
    # Indices into the chunk list being extracted (resolved to UUIDs later)
    source_chunk_indices: list[int] = field(default_factory=list)
    explicitly_stated: bool = True


@dataclass
class MemoryExtraction:
    cleaned_transcript: str | None
    chunk_metas: list[ChunkMeta]
    chunk_saliences: list[float]
    fact_drafts: list[KnowledgeFactDraft]


@dataclass
class QueryUnderstanding:
    raw: str
    expanded: str
    semantic: str
    people: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    time_hints: list[str] = field(default_factory=list)
    wants_story: bool = False

    def entity_terms(self) -> set[str]:
        return {t.lower() for t in self.people + self.places + self.topics if t}


@dataclass
class ProvenanceLink:
    """Claim → supporting memory → original chunk/transcript."""

    chunk_id: str
    memory_id: str
    quote: str
    support_level: SupportLevel = "supported"
    fact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "memory_id": self.memory_id,
            "quote": self.quote,
            "support_level": self.support_level,
        }
        if self.fact_id:
            d["fact_id"] = self.fact_id
        return d


@dataclass
class ConflictNote:
    topic: str
    descriptions: list[str]
    memory_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedFact:
    id: UUID
    statement: str
    category: str
    confidence: float
    status: str
    people: list[str]
    places: list[str]
    topics: list[str]
    supporting_chunk_ids: list[UUID]
    supporting_memory_ids: list[UUID]
    conflict_note: str | None
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "statement": self.statement,
            "category": self.category,
            "confidence": self.confidence,
            "status": self.status,
            "people": self.people,
            "places": self.places,
            "topics": self.topics,
            "supporting_chunk_ids": [str(x) for x in self.supporting_chunk_ids],
            "supporting_memory_ids": [str(x) for x in self.supporting_memory_ids],
            "conflict_note": self.conflict_note,
            "score": self.score,
        }
