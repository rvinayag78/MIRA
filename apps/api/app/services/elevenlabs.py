from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import httpx

from app.config import get_settings

EL_BASE = "https://api.elevenlabs.io/v1"


class ElevenLabsError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    settings = get_settings()
    if not settings.elevenlabs_api_key:
        raise ElevenLabsError("ELEVENLABS_API_KEY is not set")
    return {"xi-api-key": settings.elevenlabs_api_key}


async def clone_voice(name: str, audio_paths: list[Path]) -> str:
    """Instant Voice Clone from sample files. Returns voice_id."""
    if not audio_paths:
        raise ElevenLabsError("No audio samples for voice clone")

    files = []
    handles = []
    try:
        for p in audio_paths[:25]:
            fh = open(p, "rb")
            handles.append(fh)
            files.append(("files", (p.name, fh, "application/octet-stream")))

        data = {
            "name": name[:100],
            "description": "Ovyu demo memory voice",
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{EL_BASE}/voices/add",
                headers=_headers(),
                data=data,
                files=files,
            )
        if resp.status_code >= 400:
            raise ElevenLabsError(f"ElevenLabs clone error {resp.status_code}: {resp.text}")
        body = resp.json()
        voice_id = body.get("voice_id")
        if not voice_id:
            raise ElevenLabsError(f"No voice_id in response: {body}")
        return voice_id
    finally:
        for fh in handles:
            fh.close()


async def text_to_speech(voice_id: str, text: str) -> Path:
    """Synthesize speech; write mp3 under TTS_DIR; return path."""
    settings = get_settings()
    settings.tts_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.tts_dir / f"{uuid4()}.mp3"

    headers = {**_headers(), "Accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{EL_BASE}/text-to-speech/{voice_id}",
            headers=headers,
            json=payload,
        )
    if resp.status_code >= 400:
        raise ElevenLabsError(f"ElevenLabs TTS error {resp.status_code}: {resp.text}")
    out_path.write_bytes(resp.content)
    return out_path
