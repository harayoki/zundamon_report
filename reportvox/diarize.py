"""Speaker diarization utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence

import numpy as np

SpeakerLabel = Literal["A", "B"]
Mode = Literal["auto", "1", "2"]


@dataclass
class DiarizedSegment:
    start: float
    end: float
    speaker: SpeakerLabel

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class AlignedSegment:
    start: float
    end: float
    text: str
    speaker: SpeakerLabel
    character: Optional[str] = None

    def with_character(self, character: str) -> "AlignedSegment":
        return AlignedSegment(
            start=self.start,
            end=self.end,
            text=self.text,
            speaker=self.speaker,
            character=character,
        )


def diarize_audio(
    audio_path,
    mode: Mode,
    hf_token: Optional[str],
    work_dir,
) -> list[DiarizedSegment]:
    if mode == "1":
        return [DiarizedSegment(start=0.0, end=1e9, speaker="A")]

    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("pyannote.audio is required for diarization. Please install dependencies.") from exc

    token = hf_token
    if token is None:
        raise RuntimeError("Hugging Face token is required for pyannote diarization (set --hf-token or PYANNOTE_TOKEN).")

    _configure_hf_auth(token)

    try:
        pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    except TypeError:
        # pyannote.audio 3.x removed use_auth_token. Authentication is handled via huggingface_hub login/env.
        pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(
        str(audio_path),
        min_speakers=1 if mode == "auto" else None,
        max_speakers=2 if mode == "auto" else None,
        num_speakers=2 if mode == "2" else None,
    )

    segments: list[DiarizedSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        label: SpeakerLabel = "A" if speaker == "SPEAKER_00" else "B"
        segments.append(DiarizedSegment(start=start, end=end, speaker=label))
    return segments


def save_diarization(segments: Sequence[DiarizedSegment], path) -> None:
    data = [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def align_segments(
    whisper_segments,
    diarization: Sequence[DiarizedSegment],
    min_duration: float = 0.4,
) -> list[AlignedSegment]:
    aligned: list[AlignedSegment] = []
    for seg in whisper_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        speaker = _pick_speaker(start, end, diarization)
        aligned.append(
            AlignedSegment(
                start=start,
                end=end,
                text=seg["text"].strip(),
                speaker=speaker,
            )
        )

    # simple smoothing: merge very short with previous speaker
    for idx in range(1, len(aligned)):
        prev = aligned[idx - 1]
        cur = aligned[idx]
        if (cur.end - cur.start) < min_duration:
            aligned[idx] = AlignedSegment(cur.start, cur.end, cur.text, prev.speaker, cur.character)
    return aligned


def _pick_speaker(start: float, end: float, diarization: Sequence[DiarizedSegment]) -> SpeakerLabel:
    overlaps: dict[SpeakerLabel, float] = {"A": 0.0, "B": 0.0}
    for seg in diarization:
        overlap = _overlap(start, end, seg.start, seg.end)
        overlaps[seg.speaker] += overlap
    return "A" if overlaps["A"] >= overlaps["B"] else "B"


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _configure_hf_auth(token: str) -> None:
    """Make HF token visible to pyannote.audio (supports 2.x/3.x)."""
    try:
        from huggingface_hub import login  # type: ignore
    except Exception:
        # Fall back to environment variables that huggingface_hub recognizes.
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        return

    login(token=token, add_to_git_credential=False)
