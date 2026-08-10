from app.services.assemblyai import Utterance
from app.services.chunking import chunk_utterances


def test_chunk_by_speaker_and_merge():
    utts = [
        Utterance(speaker="A", text="Hello there friend.", start_ms=0, end_ms=1000),
        Utterance(speaker="A", text="I lived on Maple Street.", start_ms=1000, end_ms=2500),
        Utterance(speaker="B", text="Interesting.", start_ms=2500, end_ms=3000),
    ]
    chunks = chunk_utterances(utts, max_chars=200, overlap_chars=0)
    assert len(chunks) >= 2
    assert chunks[0].speaker == "A"
    assert "Maple Street" in chunks[0].text
    assert chunks[-1].speaker == "B"
