from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import assemblyai as aai

from app.config import get_settings


@dataclass
class Utterance:
    speaker: str | None
    text: str
    start_ms: int
    end_ms: int


@dataclass
class TranscriptResult:
    transcript_id: str
    text: str
    utterances: list[Utterance]


def _configure() -> None:
    settings = get_settings()
    if not settings.assemblyai_api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set")
    aai.settings.api_key = settings.assemblyai_api_key


def transcribe_file(audio_path: Path) -> TranscriptResult:
    """Batch transcription with speaker diarization (sync; call via asyncio.to_thread)."""
    _configure()
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(str(audio_path))

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI error: {transcript.error}")

    utterances: list[Utterance] = []
    if transcript.utterances:
        for u in transcript.utterances:
            utterances.append(
                Utterance(
                    speaker=getattr(u, "speaker", None),
                    text=u.text or "",
                    start_ms=int(u.start or 0),
                    end_ms=int(u.end or 0),
                )
            )
    elif transcript.text:
        utterances.append(
            Utterance(speaker=None, text=transcript.text, start_ms=0, end_ms=0)
        )

    return TranscriptResult(
        transcript_id=transcript.id or "",
        text=transcript.text or "",
        utterances=utterances,
    )
