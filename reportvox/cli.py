"""Command line interface for ReportVox."""

from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reportvox",
        description="Audio to diarized, character-styled VoiceVox report generator.",
    )
    parser.add_argument("input", help="Input audio/video file path (audio track required).")
    parser.add_argument("--voicevox-url", default="http://127.0.0.1:50021", help="VoiceVox Engine base URL.")
    parser.add_argument(
        "--speakers",
        choices=["auto", "1", "2"],
        default="auto",
        help="Number of speakers: auto detection, force 1, or force 2.",
    )
    parser.add_argument("--speaker1", default="zundamon", help="Character id for primary speaker.")
    parser.add_argument("--speaker2", default="metan", help="Character id for secondary speaker.")
    parser.add_argument("--zunda-senior-job", dest="zunda_senior_job", default=None, help="ずんだもんが憧れる職業を指定。")
    parser.add_argument("--zunda-junior-job", dest="zunda_junior_job", default=None, help="ずんだもんの現在の役割を指定。")
    parser.add_argument("--mp3", action="store_true", help="Generate mp3 if ffmpeg is available.")
    parser.add_argument("--bitrate", default="192k", help="Bitrate for mp3 output.")
    parser.add_argument("--keep-work", action="store_true", help="Keep intermediate files under work/.")
    parser.add_argument("--model", default="small", help="Whisper model size to use.")
    parser.add_argument(
        "--llm",
        choices=["none", "openai", "local"],
        default="none",
        help="LLM backend for style conversion.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for pyannote.audio if required (env PYANNOTE_TOKEN is also read).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PipelineConfig:
    args = build_parser().parse_args(argv)
    input_path = pathlib.Path(args.input).expanduser().resolve()
    return PipelineConfig(
        input_audio=input_path,
        voicevox_url=args.voicevox_url,
        speakers=args.speakers,
        speaker1=args.speaker1,
        speaker2=args.speaker2,
        zunda_senior_job=args.zunda_senior_job,
        zunda_junior_job=args.zunda_junior_job,
        want_mp3=args.mp3,
        mp3_bitrate=args.bitrate,
        keep_work=args.keep_work,
        whisper_model=args.model,
        llm_backend=args.llm,
        hf_token=args.hf_token,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    run_pipeline(config)
