"""Conservative structured extraction from Maker transcripts.

Extract only what is explicitly supported by the recording. Fail open on errors
so ingest still succeeds without structured metadata.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic

from app.config import get_settings
from app.services.memory_types import (
    ChunkMeta,
    KnowledgeFactDraft,
    MemoryExtraction,
    MentionedEvent,
    MentionedPerson,
    MentionedPlace,
    TimePeriod,
)

logger = logging.getLogger(__name__)

_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_PROPER = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def _client() -> anthropic.AsyncAnthropic:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


def _parse_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def clean_transcript_light(text: str) -> str:
    """Light cleanup only: normalize whitespace. Do not rewrite meaning."""
    return " ".join((text or "").split()).strip()


def heuristic_chunk_meta(text: str) -> ChunkMeta:
    """Offline / fail-open extractor using only surface cues in the text."""
    people: list[MentionedPerson] = []
    seen: set[str] = set()
    rel_patterns = [
        (re.compile(r"\bmy sister\s+([A-Z][a-z]+)\b", re.I), "sister"),
        (re.compile(r"\bmy brother\s+([A-Z][a-z]+)\b", re.I), "brother"),
        (re.compile(r"\bmy (?:mom|mother)\s+([A-Z][a-z]+)\b", re.I), "mother"),
        (re.compile(r"\bmy (?:dad|father)\s+([A-Z][a-z]+)\b", re.I), "father"),
        (re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+)\b"), None),
    ]
    for pat, rel in rel_patterns:
        for m in pat.finditer(text):
            name = m.group(1).strip()
            key = name.lower()
            if key not in seen:
                seen.add(key)
                people.append(
                    MentionedPerson(name=name, relationship=rel, confidence=0.7)
                )

    places: list[MentionedPlace] = []
    place_pat = re.compile(
        r"\b(?:in|on|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    )
    for m in place_pat.finditer(text):
        name = m.group(1).strip()
        if name.lower() in {"i", "my", "the", "a", "an", "we", "saturday", "april"}:
            continue
        if name.lower() not in {p.name.lower() for p in places}:
            places.append(MentionedPlace(name=name, confidence=0.55))

    time_periods = [
        TimePeriod(text=y, approx_year=int(y), confidence=0.8)
        for y in _YEAR.findall(text)
    ]

    topics: list[str] = []
    topic_cues = [
        ("home", "home"),
        ("childhood", "childhood"),
        ("work", "work"),
        ("bakery", "work"),
        ("school", "school"),
        ("family", "family"),
    ]
    lower = text.lower()
    for cue, topic in topic_cues:
        if cue in lower and topic not in topics:
            topics.append(topic)

    salience = 0.5
    if people or places or time_periods:
        salience = 0.65
    if len(text) > 200:
        salience = max(salience, 0.6)

    return ChunkMeta(
        people=people,
        places=places,
        time_periods=time_periods,
        events=[],
        topics=topics,
        emotional_tone=None,
        important_details=[],
        memorable_phrases=[],
        cleaned_text=clean_transcript_light(text),
        extraction_confidence=0.4,
    )


def heuristic_fact_drafts(metas: list[ChunkMeta], texts: list[str]) -> list[KnowledgeFactDraft]:
    """Build conservative fact drafts from explicit relationship / attribute cues."""
    drafts: list[KnowledgeFactDraft] = []
    for i, (meta, text) in enumerate(zip(metas, texts)):
        for person in meta.people:
            if person.relationship and person.confidence >= 0.65:
                drafts.append(
                    KnowledgeFactDraft(
                        statement=f"Has a {person.relationship} named {person.name}.",
                        category="relationship",
                        people=[person.name],
                        confidence=min(0.9, person.confidence),
                        salience=0.7,
                        source_chunk_indices=[i],
                        explicitly_stated=True,
                    )
                )
        for place in meta.places:
            if place.confidence >= 0.7 and any(
                w in text.lower() for w in ("lived", "home", "grew up", "from")
            ):
                drafts.append(
                    KnowledgeFactDraft(
                        statement=f"Connected to place: {place.name}.",
                        category="attribute",
                        places=[place.name],
                        confidence=place.confidence * 0.85,
                        salience=0.55,
                        source_chunk_indices=[i],
                        explicitly_stated=True,
                    )
                )
        # Preferences only when explicitly worded
        pref = re.search(
            r"\b(?:my favorite|i (?:loved|love|liked|like|prefer))\b(.{5,80})",
            text,
            re.I,
        )
        if pref:
            snippet = pref.group(0).strip().rstrip(".")
            drafts.append(
                KnowledgeFactDraft(
                    statement=snippet[0].upper() + snippet[1:] + ".",
                    category="preference",
                    topics=list(meta.topics),
                    confidence=0.75,
                    salience=0.6,
                    source_chunk_indices=[i],
                    explicitly_stated=True,
                )
            )
    return drafts


async def extract_memory_structure(
    *,
    transcript: str,
    chunk_texts: list[str],
) -> MemoryExtraction:
    """Extract structured metadata + candidate facts. Falls back to heuristics."""
    cleaned = clean_transcript_light(transcript)
    if not chunk_texts:
        return MemoryExtraction(
            cleaned_transcript=cleaned or None,
            chunk_metas=[],
            chunk_saliences=[],
            fact_drafts=[],
        )

    settings = get_settings()
    if not settings.anthropic_api_key:
        metas = [heuristic_chunk_meta(t) for t in chunk_texts]
        return MemoryExtraction(
            cleaned_transcript=cleaned or None,
            chunk_metas=metas,
            chunk_saliences=[m.extraction_confidence * 0.5 + 0.4 for m in metas],
            fact_drafts=heuristic_fact_drafts(metas, chunk_texts),
        )

    try:
        return await _llm_extract(transcript=cleaned, chunk_texts=chunk_texts)
    except Exception:  # noqa: BLE001
        logger.exception("Structured extraction failed; using heuristics")
        metas = [heuristic_chunk_meta(t) for t in chunk_texts]
        return MemoryExtraction(
            cleaned_transcript=cleaned or None,
            chunk_metas=metas,
            chunk_saliences=[0.55 for _ in metas],
            fact_drafts=heuristic_fact_drafts(metas, chunk_texts),
        )


async def _llm_extract(*, transcript: str, chunk_texts: list[str]) -> MemoryExtraction:
    settings = get_settings()
    client = _client()
    system = (
        "You extract structured memory metadata from a Maker's recorded transcript.\n"
        "CRITICAL RULES:\n"
        "- Only extract information explicitly present in the text.\n"
        "- Do NOT infer sensitive personal facts, diagnoses, politics, or secrets.\n"
        "- Do NOT invent people, places, dates, feelings, or events.\n"
        "- Relationship labels only when explicitly stated (e.g. \"my sister Lena\").\n"
        "- Prefer lower confidence when unsure.\n"
        "- Semantic facts must be explicitly supported; a single anecdote is not a "
        "permanent preference unless stated as such.\n"
        "- cleaned_transcript: fix filler/disfluency lightly; never change meaning.\n"
        "Return strict JSON with keys:\n"
        "cleaned_transcript, chunks (array aligned to input chunks), facts.\n"
        "Each chunk: people[{name,relationship,confidence}], places[{name,confidence}], "
        "time_periods[{text,approx_year,confidence}], events[{summary,confidence}], "
        "topics[], emotional_tone, important_details[], memorable_phrases[], "
        "cleaned_text, salience (0-1), extraction_confidence (0-1).\n"
        "Each fact: statement, category "
        "(preference|relationship|belief|attribute|event_summary), "
        "people[], places[], topics[], confidence, salience, "
        "source_chunk_indices[] (0-based), explicitly_stated (bool)."
    )
    user = json.dumps(
        {"transcript": transcript, "chunks": chunk_texts},
        ensure_ascii=False,
    )
    msg = await client.messages.create(
        model=settings.anthropic_route_model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    parsed = _parse_json(raw)
    return _memory_extraction_from_parsed(parsed, chunk_texts, transcript)


def _memory_extraction_from_parsed(
    parsed: dict[str, Any],
    chunk_texts: list[str],
    transcript: str,
) -> MemoryExtraction:
    cleaned = parsed.get("cleaned_transcript")
    cleaned_transcript = (
        clean_transcript_light(str(cleaned))
        if cleaned
        else clean_transcript_light(transcript)
    )

    raw_chunks = parsed.get("chunks") or []
    metas: list[ChunkMeta] = []
    saliences: list[float] = []
    for i, text in enumerate(chunk_texts):
        item = raw_chunks[i] if i < len(raw_chunks) and isinstance(raw_chunks[i], dict) else {}
        meta = ChunkMeta.from_dict(item if isinstance(item, dict) else {})
        if not meta.cleaned_text:
            meta.cleaned_text = clean_transcript_light(text)
        # Guard: memorable phrases must appear in the chunk
        meta.memorable_phrases = [
            p for p in meta.memorable_phrases if _norm(p) in _norm(text)
        ][:5]
        metas.append(meta)
        try:
            sal = float(item.get("salience", 0.5)) if isinstance(item, dict) else 0.5
        except (TypeError, ValueError):
            sal = 0.5
        saliences.append(max(0.0, min(1.0, sal)))

    fact_drafts: list[KnowledgeFactDraft] = []
    for f in parsed.get("facts") or []:
        if not isinstance(f, dict):
            continue
        statement = str(f.get("statement") or "").strip()
        if len(statement) < 8:
            continue
        category = str(f.get("category") or "attribute")
        if category not in {
            "preference",
            "relationship",
            "belief",
            "attribute",
            "event_summary",
        }:
            category = "attribute"
        indices = []
        for idx in f.get("source_chunk_indices") or []:
            try:
                ii = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= ii < len(chunk_texts):
                indices.append(ii)
        if not indices:
            continue
        # Require statement overlap with at least one source chunk
        if not any(_soft_supported(statement, chunk_texts[i]) for i in indices):
            continue
        explicitly = bool(f.get("explicitly_stated", True))
        conf = float(f.get("confidence") or 0.5)
        if not explicitly:
            conf = min(conf, 0.55)
        fact_drafts.append(
            KnowledgeFactDraft(
                statement=statement,
                category=category,  # type: ignore[arg-type]
                people=[str(p).strip() for p in (f.get("people") or []) if str(p).strip()],
                places=[str(p).strip() for p in (f.get("places") or []) if str(p).strip()],
                topics=[str(t).strip() for t in (f.get("topics") or []) if str(t).strip()],
                confidence=max(0.0, min(1.0, conf)),
                salience=max(0.0, min(1.0, float(f.get("salience") or 0.5))),
                source_chunk_indices=indices,
                explicitly_stated=explicitly,
            )
        )

    if not metas:
        metas = [heuristic_chunk_meta(t) for t in chunk_texts]
        saliences = [0.55 for _ in metas]
        fact_drafts = heuristic_fact_drafts(metas, chunk_texts)

    return MemoryExtraction(
        cleaned_transcript=cleaned_transcript or None,
        chunk_metas=metas,
        chunk_saliences=saliences or [0.5 for _ in metas],
        fact_drafts=fact_drafts,
    )


def _norm(value: str) -> str:
    chars: list[str] = []
    for ch in value.lower():
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def _soft_supported(statement: str, chunk: str) -> bool:
    """Require meaningful token overlap between fact and source chunk."""
    s_tokens = set(_norm(statement).split()) - {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "was",
        "has",
        "have",
        "named",
        "my",
        "i",
    }
    c_tokens = set(_norm(chunk).split())
    if not s_tokens:
        return False
    overlap = len(s_tokens & c_tokens)
    return overlap >= max(2, len(s_tokens) // 3)


def fact_status_for_draft(draft: KnowledgeFactDraft, *, evidence_count: int = 1) -> str:
    """Single anecdotes stay candidates unless strongly and explicitly stated."""
    if evidence_count >= 2 and draft.confidence >= 0.55:
        return "established"
    if draft.explicitly_stated and draft.confidence >= 0.8 and evidence_count >= 1:
        if draft.category in {"relationship", "attribute"}:
            return "established"
    return "candidate"
