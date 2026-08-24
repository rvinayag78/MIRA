from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class AgentCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)


class AgentOut(BaseModel):
    id: UUID
    display_name: str
    elevenlabs_voice_id: str | None = None
    status: str
    created_at: datetime
    share_token: str | None = None
    keeper_url: str | None = None
    indexed_memories: int | None = None
    text_indexed: int = 0
    voice_indexed: int = 0
    text_required: int = 3
    voice_required: int = 3
    ready_for_keeper: bool = False


class MemoryOut(BaseModel):
    id: UUID
    agent_id: UUID
    kind: Literal["text", "voice"] = "voice"
    text_content: str | None = None
    audio_uri: str | None = None
    duration_ms: int | None = None
    status: str
    error_message: str | None = None
    assemblyai_transcript_id: str | None = None
    created_at: datetime


class TextMemoryCreate(BaseModel):
    agent_id: UUID
    token: str
    text: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    agent_id: UUID
    token: str
    message: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = None
    speak: bool = True


class CitationOut(BaseModel):
    chunk_id: str
    quote: str
    memory_id: str | None = None
    support_level: str | None = None


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    citations: list[CitationOut]
    confidence: float
    refused: bool
    intent: str
    audio_url: str | None = None
    coverage: str | None = None
    uncertainty: bool = False


class HealthOut(BaseModel):
    status: str
    rerank_enabled: bool


class ErrorOut(BaseModel):
    detail: str
    extras: dict[str, Any] | None = None
