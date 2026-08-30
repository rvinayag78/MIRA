from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic

from app.config import get_settings
from app.services.evidence import (
    UNCERTAINTY_PHRASES,
    EvidencePackage,
    build_evidence_package,
)
from app.services.memory_types import ProvenanceLink, SupportLevel
from app.services.retrieve import RetrievedChunk, RetrievedFact

RouteIntent = Literal[
    "answerable_from_memories",
    "clarify",
    "refuse_out_of_scope",
    "smalltalk",
]


@dataclass
class Citation:
    chunk_id: str
    quote: str
    memory_id: str | None = None
    support_level: SupportLevel = "supported"
    fact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "quote": self.quote,
            "support_level": self.support_level,
        }
        if self.memory_id:
            d["memory_id"] = self.memory_id
        if self.fact_id:
            d["fact_id"] = self.fact_id
        return d


@dataclass
class GroundedAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    intent: RouteIntent = "answerable_from_memories"
    coverage: str = "none"
    provenance: list[ProvenanceLink] = field(default_factory=list)
    uncertainty: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "refused": self.refused,
            "intent": self.intent,
            "coverage": self.coverage,
            "uncertainty": self.uncertainty,
            "provenance": [p.to_dict() for p in self.provenance],
        }


def _client() -> anthropic.AsyncAnthropic:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


_FOLLOW_UP = re.compile(
    r"^(and |what about |tell me more|more about |why\b|how so|go on|continue|"
    r"that\b|those\b|it\b|them\b|and then)",
    re.I,
)


_THEME_TERMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(dad|daddy|father|papa)\b", re.I), "dad father papa parent"),
    (re.compile(r"\b(mom|mum|mama|mother)\b", re.I), "mom mother mama parent"),
    (
        re.compile(r"\b(early|childhood|young|growing up|first)\b", re.I),
        "early childhood young first growing up kid",
    ),
    (re.compile(r"\b(memory|memories|story|stories)\b", re.I), "memory story"),
]


def expand_retrieval_query(
    question: str, history: list[dict[str, str]] | None = None
) -> str:
    """Fold a short follow-up into the prior user turn so retrieval stays on-topic."""
    prior_user = None
    if history:
        for turn in reversed(history):
            if turn.get("role") == "user" and (turn.get("content") or "").strip():
                prior_user = turn["content"].strip()
                break
    q = question.strip()
    if prior_user and (len(q.split()) <= 3 or _FOLLOW_UP.search(q)):
        return f"{prior_user} {q}"
    return q


def broaden_retrieval_query(question: str) -> str:
    """Add everyday synonyms so dense search can match related memories."""
    extra: list[str] = []
    for pat, terms in _THEME_TERMS:
        if pat.search(question):
            extra.append(terms)
    if not extra:
        return question.strip()
    return f"{question.strip()} {' '.join(extra)}"


def _history_block(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""
    lines = []
    for turn in history[-6:]:
        role = turn.get("role") or "user"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return "Recent conversation:\n" + "\n".join(lines) + "\n\n"


async def route_intent(
    question: str,
    *,
    maker_name: str = "the maker",
    history: list[dict[str, str]] | None = None,
) -> RouteIntent:
    settings = get_settings()
    client = _client()
    prompt = (
        _history_block(history)
        + f"A keeper is talking to a memory of {maker_name}. The agent will answer AS {maker_name}.\n"
        "The keeper may say \"you\" (addressing the maker) or they/he/she (about the maker). Both are in-scope.\n"
        "Follow-ups like \"tell me more\" after a memory question are answerable_from_memories.\n"
        "Broad life questions are answerable even if they do not name a specific memory: "
        "\"What was dad like?\", \"What's an early memory?\", \"Tell me a story\".\n"
        "Return ONLY one label:\n"
        "- answerable_from_memories: anything about the maker's life, people, places, feelings, "
        "or a request to hear a recorded memory — including open prompts\n"
        "- clarify: empty or unintelligible only\n"
        "- refuse_out_of_scope: world knowledge, news, advice, or topics clearly not about "
        "this person's recorded life\n"
        "- smalltalk: greetings / thanks with no request for a memory\n"
        "Never refuse or clarify just because the question is broad or uses you/they/he/she.\n\n"
        f"User: {question}"
    )
    msg = await client.messages.create(
        model=settings.anthropic_route_model,
        max_tokens=32,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip().lower()
    for label in (
        "answerable_from_memories",
        "clarify",
        "refuse_out_of_scope",
        "smalltalk",
    ):
        if label in text:
            return label  # type: ignore[return-value]
    return "answerable_from_memories"


REFUSAL = "I don’t have that in the recorded memories."
DEFAULT_UNCERTAINTY = UNCERTAINTY_PHRASES[0]


def _uncertainty_answer(coverage: str) -> str:
    if coverage == "partial":
        return UNCERTAINTY_PHRASES[1]
    return UNCERTAINTY_PHRASES[0]


def uncited_claims_should_refuse(*, has_citations: bool, coverage: str) -> bool:
    """Whether an answer without grounded citations must be replaced.

    ``uncertainty_flag`` is intentionally not a parameter: models often set
    ``uncertainty: true`` (or write "not sure") while inventing personal details.
    That must not exempt uncited answers from the fail-closed path.

    ``coverage == "strong"`` remains exempt here so this guard stays scoped to the
    uncertainty bypass; strong-coverage inflation is tracked separately.
    """
    if has_citations:
        return False
    return coverage != "strong"


async def ground_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    maker_name: str = "the maker",
    history: list[dict[str, str]] | None = None,
    facts: list[RetrievedFact] | None = None,
    evidence: EvidencePackage | None = None,
) -> GroundedAnswer:
    package = evidence or build_evidence_package(question, chunks, facts)
    settings = get_settings()

    if package.coverage == "none" or (not package.episodic and not package.semantic):
        answer = _uncertainty_answer("none")
        return GroundedAnswer(
            answer=answer,
            citations=[],
            confidence=0.0,
            refused=True,
            intent="answerable_from_memories",
            coverage="none",
            uncertainty=True,
        )

    who = maker_name.strip() or "the maker"
    system = (
        f"You are {who}. A keeper is speaking with you through your recorded memories.\n"
        "Speak in first person (I, me, my) the way you actually talk in the evidence: "
        "your wording, pacing, warmth or bluntness, the details you linger on. "
        "Sound like a person remembering out loud, not a biography or a list.\n\n"
        "EVIDENCE RULES (substance grounding):\n"
        "- Episodic memories are specific recorded experiences (source of truth).\n"
        "- Semantic knowledge is distilled person-knowledge; treat 'candidate' as weak "
        "and 'disputed' as conflicting — never present disputed details as certain.\n"
        "- Internally classify every personal claim as:\n"
        "  SUPPORTED: directly present in episodic text or an established fact.\n"
        "  INFERRED: reasonably implied by multiple evidence pieces; hedge in wording.\n"
        "  UNKNOWN: not in evidence — do not invent; use natural uncertainty.\n"
        "- NEVER invent events, relationships, conversations, locations, dates, opinions, "
        "feelings, experiences, or personal details because they sound plausible.\n"
        "- If coverage is partial, say only what was recorded and acknowledge limits.\n"
        "- If evidence conflicts (see conflicts), preserve uncertainty "
        "(e.g. \"I think that was around 2008, although I might be mixing up the year.\").\n"
        "- Do not convert a single anecdote into a lifelong fact unless semantic status "
        "is established or the recording states it explicitly as a standing fact.\n"
        "- Recent conversation is only for follow-ups; it is not evidence.\n\n"
        "When you cannot answer from evidence, prefer natural uncertainty such as:\n"
        f'- "{UNCERTAINTY_PHRASES[0]}"\n'
        f'- "{UNCERTAINTY_PHRASES[1]}"\n'
        f'- "{UNCERTAINTY_PHRASES[2]}"\n'
        f'Hard out-of-scope / empty may still use: "{REFUSAL}"\n\n'
        "Citations: for each factual claim, include chunk_id and a short quote copied "
        "verbatim from that episodic chunk (substring). Optionally include memory_id "
        "and support_level (supported|inferred). Do not paraphrase the quote.\n"
        "Return strict JSON: "
        '{"answer": string, "citations": [{"chunk_id": string, "quote": string, '
        '"memory_id": string|null, "support_level": "supported"|"inferred"}], '
        '"confidence": number, "uncertainty": boolean}\n'
        "confidence is 0-1."
    )
    user = json.dumps(
        {
            "question": question,
            "recent_conversation": (history or [])[-6:],
            "evidence": package.to_prompt_dict(),
        },
        ensure_ascii=False,
    )

    client = _client()
    msg = await client.messages.create(
        model=settings.anthropic_ground_model,
        max_tokens=1536,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    parsed = _parse_json(raw)

    mem_by_chunk = package.chunk_id_to_memory()
    allowed = set(mem_by_chunk.keys())
    citations: list[Citation] = []
    raw_cites = parsed.get("citations") or []
    if isinstance(raw_cites, list):
        for c in raw_cites:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("chunk_id", ""))
            if cid not in allowed:
                continue
            level = str(c.get("support_level") or "supported")
            if level not in {"supported", "inferred"}:
                level = "supported"
            citations.append(
                Citation(
                    chunk_id=cid,
                    quote=str(c.get("quote", ""))[:500],
                    memory_id=str(c.get("memory_id") or mem_by_chunk.get(cid) or ""),
                    support_level=level,  # type: ignore[arg-type]
                )
            )

    answer = str(parsed.get("answer") or "").strip() or _uncertainty_answer(package.coverage)
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    uncertainty_flag = bool(parsed.get("uncertainty")) or _looks_uncertain(answer)

    citations = citations_grounded_in_chunks(citations, package.episodic)
    # Enrich memory_id if missing
    for cite in citations:
        if not cite.memory_id:
            cite.memory_id = mem_by_chunk.get(cite.chunk_id)

    hard_refuse = REFUSAL.lower() in answer.lower()

    if hard_refuse:
        return GroundedAnswer(
            answer=REFUSAL,
            citations=[],
            confidence=0.0,
            refused=True,
            intent="answerable_from_memories",
            coverage=package.coverage,
            uncertainty=False,
        )

    # Fail closed on uncited answers. uncertainty_flag must not bypass this:
    # models often set uncertainty:true (or emit "not sure") while still inventing
    # personal details. coverage=="strong" bypass is a separate guard path.
    if uncited_claims_should_refuse(
        has_citations=bool(citations),
        coverage=package.coverage,
    ):
        return GroundedAnswer(
            answer=_uncertainty_answer(package.coverage),
            citations=[],
            confidence=0.0,
            refused=True,
            intent="answerable_from_memories",
            coverage=package.coverage,
            uncertainty=True,
        )

    provenance = [
        ProvenanceLink(
            chunk_id=c.chunk_id,
            memory_id=c.memory_id or mem_by_chunk.get(c.chunk_id, ""),
            quote=c.quote,
            support_level=c.support_level,
            fact_id=c.fact_id,
        )
        for c in citations
    ]

    return GroundedAnswer(
        answer=answer,
        citations=citations,
        confidence=max(0.0, min(1.0, confidence)),
        refused=False,
        intent="answerable_from_memories",
        coverage=package.coverage,
        provenance=provenance,
        uncertainty=uncertainty_flag,
    )


def _looks_uncertain(answer: str) -> bool:
    lower = answer.lower()
    cues = (
        "don't think i ever recorded",
        "don't remember enough",
        "don't have a clear memory",
        "not sure",
        "might be mixing",
        "i'm not certain",
        "i do not remember",
        "can't say for sure",
        "cannot say for sure",
    )
    return any(c in lower for c in cues) or any(p.lower() in lower for p in UNCERTAINTY_PHRASES)


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
    return {"answer": REFUSAL, "citations": [], "confidence": 0.0}


async def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    maker_name: str = "the maker",
    history: list[dict[str, str]] | None = None,
    facts: list[RetrievedFact] | None = None,
    evidence: EvidencePackage | None = None,
) -> GroundedAnswer:
    intent = await route_intent(question, maker_name=maker_name, history=history)

    if intent == "smalltalk":
        return GroundedAnswer(
            answer="Hi — ask me about what I recorded.",
            citations=[],
            confidence=1.0,
            refused=False,
            intent=intent,
            coverage="n/a",
        )
    if intent == "refuse_out_of_scope":
        return GroundedAnswer(
            answer=REFUSAL,
            citations=[],
            confidence=1.0,
            refused=True,
            intent=intent,
            coverage="none",
        )

    return await ground_answer(
        question,
        chunks,
        maker_name=maker_name,
        history=history,
        facts=facts,
        evidence=evidence,
    )


def _norm_text(value: str) -> str:
    chars: list[str] = []
    for ch in value.lower():
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        elif ch in "'’":
            continue
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def citations_grounded_in_chunks(
    citations: list[Citation], chunks: list[RetrievedChunk]
) -> list[Citation]:
    """Keep citations whose quote is a verbatim substring of the cited chunk."""
    by_id = {str(c.id): c for c in chunks}
    kept: list[Citation] = []
    for cite in citations:
        source = by_id.get(cite.chunk_id)
        quote = (cite.quote or "").strip()
        if not source or len(quote) < 8:
            continue
        if _norm_text(quote) in _norm_text(source.text):
            if not cite.memory_id:
                cite.memory_id = str(source.memory_id)
            kept.append(cite)
    return kept


def check_citations_valid(answer: GroundedAnswer, chunks: list[RetrievedChunk]) -> bool:
    allowed = {str(c.id) for c in chunks}
    if not all(c.chunk_id in allowed for c in answer.citations):
        return False
    return len(citations_grounded_in_chunks(answer.citations, chunks)) == len(answer.citations)


async def faithfulness_score(answer: str, evidence_texts: list[str]) -> float:
    """LLM-as-judge: fraction of answer supported by evidence (0-1)."""
    if not answer.strip() or answer.strip() == REFUSAL or _looks_uncertain(answer):
        return 1.0
    if not evidence_texts:
        return 0.0

    settings = get_settings()
    client = _client()
    prompt = (
        "Score faithfulness of the ANSWER given EVIDENCE.\n"
        "Return JSON only: {\"score\": number between 0 and 1}.\n"
        "1 means fully supported; 0 means unsupported/hallucinated.\n"
        "Treat appropriate uncertainty about missing info as faithful (score 1).\n\n"
        f"EVIDENCE:\n{json.dumps(evidence_texts, ensure_ascii=False)}\n\n"
        f"ANSWER:\n{answer}"
    )
    msg = await client.messages.create(
        model=settings.anthropic_route_model,
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    parsed = _parse_json(raw)
    try:
        return max(0.0, min(1.0, float(parsed.get("score", 0.0))))
    except (TypeError, ValueError):
        return 0.0
