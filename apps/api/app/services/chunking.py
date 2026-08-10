from __future__ import annotations

from dataclasses import dataclass

from app.services.assemblyai import Utterance


@dataclass
class TextChunk:
    text: str
    speaker: str | None
    ts_start: float | None  # seconds
    ts_end: float | None


def chunk_utterances(
    utterances: list[Utterance],
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[TextChunk]:
    """Chunk by speaker turn, merging short turns up to ~400–800 tokens (~chars proxy)."""
    if not utterances:
        return []

    chunks: list[TextChunk] = []
    buffer_parts: list[str] = []
    buffer_speaker: str | None = None
    buffer_start: int | None = None
    buffer_end: int | None = None

    def flush() -> None:
        nonlocal buffer_parts, buffer_speaker, buffer_start, buffer_end
        if not buffer_parts:
            return
        text = " ".join(buffer_parts).strip()
        if text:
            chunks.append(
                TextChunk(
                    text=text,
                    speaker=buffer_speaker,
                    ts_start=(buffer_start / 1000.0) if buffer_start is not None else None,
                    ts_end=(buffer_end / 1000.0) if buffer_end is not None else None,
                )
            )
        buffer_parts = []
        buffer_speaker = None
        buffer_start = None
        buffer_end = None

    for u in utterances:
        text = (u.text or "").strip()
        if not text:
            continue

        # Speaker change → flush
        if buffer_parts and u.speaker is not None and buffer_speaker is not None:
            if u.speaker != buffer_speaker:
                flush()

        if not buffer_parts:
            buffer_speaker = u.speaker
            buffer_start = u.start_ms
            buffer_end = u.end_ms
            buffer_parts = [text]
            continue

        candidate = " ".join(buffer_parts + [text])
        if len(candidate) <= max_chars:
            buffer_parts.append(text)
            buffer_end = u.end_ms
        else:
            # overlap: keep tail of previous
            flush()
            if chunks and overlap_chars > 0:
                prev = chunks[-1].text
                overlap = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
                buffer_parts = [overlap, text]
            else:
                buffer_parts = [text]
            buffer_speaker = u.speaker
            buffer_start = u.start_ms
            buffer_end = u.end_ms

    flush()
    return chunks
