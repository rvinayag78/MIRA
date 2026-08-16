from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

import anthropic

from app.config import get_settings
from app.services.retrieve import RetrievedChunk

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


@dataclass
class GroundedAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    intent: RouteIntent = "answerable_from_memories"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [{"chunk_id": c.chunk_id, "quote": c.quote} for c in self.citations],
            "confidence": self.confidence,
            "refused": self.refused,
            "intent": self.intent,
        }


def _client() -> anthropic.AsyncAnthropic:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


async def route_intent(question: str) -> RouteIntent:
    settings = get_settings()
    client = _client()
    prompt = (
        "Classify the user message for a memory-grounded agent.\n"
        "Return ONLY one label:\n"
        "- answerable_from_memories: question about personal facts/stories that might be in recorded memories\n"
        "- clarify: too vague to retrieve\n"
        "- refuse_out_of_scope: asks for world knowledge, advice, or things not about the maker's memories\n"
        "- smalltalk: greetings / thanks\n\n"
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


async def ground_answer(question: str, chunks: list[RetrievedChunk]) -> GroundedAnswer:
    settings = get_settings()
    if not chunks:
        return GroundedAnswer(
            answer=REFUSAL,
            citations=[],
            confidence=0.0,
            refused=True,
            intent="answerable_from_memories",
        )

    evidence = [
        {
            "chunk_id": str(c.id),
            "text": c.text,
            "speaker": c.speaker,
        }
        for c in chunks
    ]
    system = (
        "You are a keeper-facing memory agent. Answer ONLY from the provided evidence chunks.\n"
        "Rules:\n"
        "1. Every factual claim must be supported by a cited chunk.\n"
        "2. If evidence is insufficient, refuse with exactly: "
        f'"{REFUSAL}"\n'
        "3. Do not use biography, world knowledge, or invent details.\n"
        "4. Return strict JSON: "
        '{"answer": string, "citations": [{"chunk_id": string, "quote": string}], "confidence": number}\n'
        "confidence is 0-1."
    )
    user = json.dumps({"question": question, "evidence": evidence}, ensure_ascii=False)

    client = _client()
    msg = await client.messages.create(
        model=settings.anthropic_ground_model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    parsed = _parse_json(raw)

    allowed = {str(c.id) for c in chunks}
    citations: list[Citation] = []
    raw_cites = parsed.get("citations") or []
    if isinstance(raw_cites, list):
        for c in raw_cites:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("chunk_id", ""))
            if cid in allowed:
                citations.append(Citation(chunk_id=cid, quote=str(c.get("quote", ""))[:500]))

    answer = str(parsed.get("answer") or "").strip() or REFUSAL
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    refused = REFUSAL.lower() in answer.lower() or not citations

    # Post-check: drop citations not in retrieved set (already filtered)
    if refused and REFUSAL not in answer:
        answer = REFUSAL

    return GroundedAnswer(
        answer=answer,
        citations=[] if refused and answer == REFUSAL else citations,
        confidence=0.0 if refused and answer == REFUSAL else max(0.0, min(1.0, confidence)),
        refused=refused and answer == REFUSAL,
        intent="answerable_from_memories",
    )


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
) -> GroundedAnswer:
    intent = await route_intent(question)

    if intent == "smalltalk":
        return GroundedAnswer(
            answer="Hi — ask me about the memories that were recorded.",
            citations=[],
            confidence=1.0,
            refused=False,
            intent=intent,
        )
    if intent == "clarify":
        return GroundedAnswer(
            answer="Could you ask a more specific question about a person, place, or event from the memories?",
            citations=[],
            confidence=1.0,
            refused=False,
            intent=intent,
        )
    if intent == "refuse_out_of_scope":
        return GroundedAnswer(
            answer=REFUSAL,
            citations=[],
            confidence=1.0,
            refused=True,
            intent=intent,
        )

    return await ground_answer(question, chunks)


def check_citations_valid(answer: GroundedAnswer, chunks: list[RetrievedChunk]) -> bool:
    allowed = {str(c.id) for c in chunks}
    return all(c.chunk_id in allowed for c in answer.citations)


async def faithfulness_score(answer: str, evidence_texts: list[str]) -> float:
    """LLM-as-judge: fraction of answer supported by evidence (0-1)."""
    if not answer.strip() or answer.strip() == REFUSAL:
        return 1.0
    if not evidence_texts:
        return 0.0

    settings = get_settings()
    client = _client()
    prompt = (
        "Score faithfulness of the ANSWER given EVIDENCE.\n"
        "Return JSON only: {\"score\": number between 0 and 1}.\n"
        "1 means fully supported; 0 means unsupported/hallucinated.\n\n"
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
