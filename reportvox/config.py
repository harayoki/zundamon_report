"""パイプラインの設定を定義するデータクラス。"""

from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Literal, Optional


SpeakerMode = Literal["auto", "1", "2"]
SubtitleMode = Literal["off", "all", "split"]
TranscriptReviewMode = Literal["off", "manual", "llm"]
LLMBackend = Literal["none", "openai", "ollama"]
KanaTargetLevel = Literal["none", "elementary", "junior", "high", "college"]


@dataclass
class PipelineConfig:
    input_audio: pathlib.Path | None
    voicevox_url: str
    speakers: SpeakerMode
    speaker1: str
    speaker2: str
    color1: Optional[str]
    color2: Optional[str]
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
    llm_host: Optional[str]
    llm_port: Optional[int]
    ollama_model: Optional[str] = None
    speed_scale: float = 1.1
    resume_run_id: Optional[str] = None
    resume_from: Optional[str] = None
    force_transcribe: bool = False
    force_diarize: bool = False
    subtitle_mode: SubtitleMode = "all"
    subtitle_max_chars: int = 25
    subtitle_font: Optional[str] = None
    subtitle_font_size: int = 42
    video_width: int = 1080
    video_height: int = 1920
    video_fps: int = 24
    output_mp4: bool = False
    output_mov: bool = False
    review_transcript: TranscriptReviewMode = "llm"
    style_with_llm: bool = True
    linebreak_with_llm: bool = True
    linebreak_min_chars: int = 40
    kana_level: KanaTargetLevel = "high"
    diarization_threshold: float = 0.8
    prepend_intro: bool = True
    intro1: Optional[str] = None
    intro2: Optional[str] = None
    max_pause_between_segments: float = 0.2
    video_images: list[pathlib.Path] = field(default_factory=list)
    video_image_scale: float = 0.45
    video_image_position: tuple[int, int] | None = None
    video_image_times: list[float] | None = None
    cli_args: list[str] | None = None
