"""Whisper transcription helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .envinfo import EnvironmentInfo, append_env_details


@dataclass
class TranscriptionResult:
    segments: list[dict]
    text: str

    def as_json(self) -> dict:
        return {"text": self.text, "segments": self.segments}


def transcribe_audio(audio_path, model_size: str = "small", *, env_info: EnvironmentInfo | None = None) -> TranscriptionResult:
    try:
        import whisper  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            append_env_details(
                "openai-whisper is required for transcription. Install with `pip install -r requirements.txt`.", env_info
            )
        ) from exc

    model = whisper.load_model(model_size)
    result: Dict[str, Any] = model.transcribe(str(audio_path))
    segments = [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": str(seg["text"]).strip()}
        for seg in result.get("segments", [])
    ]
    return TranscriptionResult(segments=segments, text=str(result.get("text", "")).strip())
