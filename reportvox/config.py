"""パイプラインの設定を定義するデータクラス。"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Literal, Optional


SpeakerMode = Literal["auto", "1", "2"]
SubtitleMode = Literal["off", "all", "split"]
TranscriptReviewMode = Literal["off", "manual", "llm"]
LLMBackend = Literal["none", "openai", "local"]


@dataclass
class PipelineConfig:
    input_audio: pathlib.Path | None
    voicevox_url: str
    speakers: SpeakerMode
    speaker1: str
    speaker2: str
    zunda_senior_job: Optional[str]
    zunda_junior_job: Optional[str]
    want_mp3: bool
    mp3_bitrate: str
    ffmpeg_path: str
    keep_work: bool
    output_name: Optional[str]
    force_overwrite: bool
    whisper_model: str
    llm_backend: LLMBackend
    hf_token: Optional[str] = None
    speed_scale: float = 1.1
    output_duration: Optional[float] = None
    resume_run_id: Optional[str] = None
    resume_from: Optional[str] = None
    subtitle_mode: SubtitleMode = "off"
    subtitle_max_chars: int = 25
    review_transcript: TranscriptReviewMode = "off"
