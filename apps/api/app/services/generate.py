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


_FOLLOW_UP = re.compile(
    r"^(and |what about |tell me more|more about |why\b|how so|go on|continue|"
    r"that\b|those\b|it\b|them\b|and then)",
    re.I,
)


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
        "Return ONLY one label:\n"
        "- answerable_from_memories: anything about the maker's life, stories, people, places, "
        "feelings, or what they recorded — e.g. \"What was your childhood home like?\", "
        "\"What did they say about their mom?\"\n"
        "- clarify: too vague to retrieve (e.g. \"tell me stuff\") with no prior thread\n"
        "- refuse_out_of_scope: world knowledge, news, advice, or topics clearly not about "
        "this person's recorded life\n"
        "- smalltalk: greetings / thanks\n"
        "Never refuse just because the question uses you/they/he/she.\n\n"
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


async def ground_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    maker_name: str = "the maker",
    history: list[dict[str, str]] | None = None,
) -> GroundedAnswer:
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
    who = maker_name.strip() or "the maker"
    system = (
        f"You are {who}. A keeper is speaking with you through your recorded memories.\n"
        "Speak in first person (I, me, my) the way you actually talk in the evidence: "
        "your wording, pacing, warmth or bluntness, the details you linger on. "
        "Sound like a person remembering out loud, not a biography or a list.\n"
        "You may weave several evidence chunks into one conversational reply and lightly "
        "paraphrase, but every fact must come from the evidence chunks.\n"
        "The keeper may say \"you\" or they/he/she — they mean you. "
        "Recent conversation is only for understanding follow-ups; it is not evidence. "
        "If an earlier reply said something that is not in this evidence, ignore it.\n"
        "Hard limits — no hallucinations:\n"
        "- Do not invent names, places, dates, feelings, jobs, or events missing from the evidence.\n"
        "- Do not fill gaps with what someone like you \"would\" have done or felt.\n"
        "- Do not complete a story the evidence does not finish.\n"
        "- If you only have a partial answer, say only what you recorded and stop.\n"
        "- If the evidence does not answer the question, refuse with exactly: "
        f'"{REFUSAL}"\n'
        "Citations: for each factual claim, include chunk_id and a short quote copied verbatim "
        "from that chunk (a substring of the chunk text). Do not paraphrase the quote.\n"
        "Return strict JSON: "
        '{"answer": string, "citations": [{"chunk_id": string, "quote": string}], "confidence": number}\n'
        "confidence is 0-1."
    )
    user = json.dumps(
        {
            "question": question,
            "recent_conversation": (history or [])[-6:],
            "evidence": evidence,
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
    citations = citations_grounded_in_chunks(citations, chunks)
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
    *,
    maker_name: str = "the maker",
    history: list[dict[str, str]] | None = None,
) -> GroundedAnswer:
    intent = await route_intent(question, maker_name=maker_name, history=history)

    if intent == "smalltalk":
        return GroundedAnswer(
            answer="Hi — ask me about what I recorded.",
            citations=[],
            confidence=1.0,
            refused=False,
            intent=intent,
        )
    if intent == "clarify":
        return GroundedAnswer(
            answer=(
                "Could you ask me something more specific — a person, place, "
                "or time I talked about?"
            ),
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

    return await ground_answer(question, chunks, maker_name=maker_name, history=history)


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
    by_id = {str(c.id): c.text for c in chunks}
    kept: list[Citation] = []
    for cite in citations:
        source = by_id.get(cite.chunk_id)
        quote = (cite.quote or "").strip()
        if not source or len(quote) < 8:
            continue
        if _norm_text(quote) in _norm_text(source):
            kept.append(cite)
    return kept


def check_citations_valid(answer: GroundedAnswer, chunks: list[RetrievedChunk]) -> bool:
    allowed = {str(c.id) for c in chunks}
    if not all(c.chunk_id in allowed for c in answer.citations):
        return False
    return len(citations_grounded_in_chunks(answer.citations, chunks)) == len(answer.citations)


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
