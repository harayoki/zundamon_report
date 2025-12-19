"""Processing pipeline orchestration."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from . import diarize, transcribe, style_convert, voicevox, audio, characters


SpeakerMode = Literal["auto", "1", "2"]
LLMBackend = Literal["none", "openai", "local"]


@dataclass
class PipelineConfig:
    input_audio: pathlib.Path
    voicevox_url: str
    speakers: SpeakerMode
    speaker1: str
    speaker2: str
    want_mp3: bool
    mp3_bitrate: str
    keep_work: bool
    whisper_model: str
    llm_backend: LLMBackend
    hf_token: Optional[str] = None


def _ensure_paths() -> tuple[pathlib.Path, pathlib.Path]:
    work_dir = pathlib.Path("work")
    out_dir = pathlib.Path("out")
    work_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    return work_dir, out_dir


def _copy_input(input_audio: pathlib.Path, run_dir: pathlib.Path) -> pathlib.Path:
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio}")
    dest = run_dir / f"input{input_audio.suffix}"
    shutil.copy2(input_audio, dest)
    return dest


def _summarize_speaker_durations(aligned: Sequence[diarize.AlignedSegment]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for seg in aligned:
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + (seg.end - seg.start)
    return totals


def _map_speakers(
    aligned: Sequence[diarize.AlignedSegment],
    totals: Dict[str, float],
    mode: SpeakerMode,
    char1: characters.CharacterMeta,
    char2: characters.CharacterMeta,
) -> list[diarize.AlignedSegment]:
    mapped: list[diarize.AlignedSegment] = []
    if mode == "1":
        for seg in aligned:
            mapped.append(seg.with_character(char1.id))
        return mapped

    if mode == "auto" and totals:
        speaker, duration = max(totals.items(), key=lambda x: x[1])
        share = duration / sum(totals.values())
        if share >= 0.93:
            # treat as single speaker
            for seg in aligned:
                mapped.append(seg.with_character(char1.id))
            return mapped

    # two speaker mapping: longer becomes speaker1
    sorted_speakers = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_speakers[0][0] if sorted_speakers else "A"
    for seg in aligned:
        char = char1 if seg.speaker == primary else char2
        mapped.append(seg.with_character(char.id))
    return mapped


def run_pipeline(config: PipelineConfig) -> None:
    work_dir, out_dir = _ensure_paths()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = work_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[reportvox] checking ffmpeg availability...")
    audio.ensure_ffmpeg()

    print(f"[reportvox] run id: {run_id}")
    print(f"[reportvox] copying input...")
    input_path = _copy_input(config.input_audio, run_dir)
    normalized_input = run_dir / "input.wav"
    audio.normalize_to_wav(input_path, normalized_input)

    print(f"[reportvox] transcribing with Whisper ({config.whisper_model})...")
    whisper_result = transcribe.transcribe_audio(normalized_input, model_size=config.whisper_model)
    (run_dir / "transcript.json").write_text(json.dumps(whisper_result.as_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[reportvox] diarizing speakers ({config.speakers})...")
    diarization = diarize.diarize_audio(
        normalized_input,
        mode=config.speakers,
        hf_token=config.hf_token or os.environ.get("PYANNOTE_TOKEN"),
        work_dir=run_dir,
    )
    diarize.save_diarization(diarization, run_dir / "diarization.json")

    aligned = diarize.align_segments(whisper_result.segments, diarization)
    totals = _summarize_speaker_durations(aligned)
    print(f"[reportvox] speaker durations: {totals}")

    char1 = characters.load_character(config.speaker1)
    char2 = characters.load_character(config.speaker2)
    mapped = _map_speakers(aligned, totals, config.speakers, char1, char2)

    print("[reportvox] converting style and inserting phrases...")
    stylized = style_convert.apply_style(mapped, char1, char2, backend=config.llm_backend)

    print("[reportvox] synthesizing with VoiceVox...")
    synthesized_paths = voicevox.synthesize_segments(
        stylized,
        characters={char1.id: char1, char2.id: char2},
        base_url=config.voicevox_url,
        run_dir=run_dir,
    )

    output_wav = out_dir / f"{config.input_audio.stem}_report.wav"
    print(f"[reportvox] joining audio -> {output_wav}")
    audio.join_wavs(synthesized_paths, output_wav)

    if config.want_mp3:
        mp3_path = out_dir / f"{config.input_audio.stem}_report.mp3"
        print(f"[reportvox] generating mp3 -> {mp3_path}")
        audio.convert_to_mp3(output_wav, mp3_path, bitrate=config.mp3_bitrate)

    if not config.keep_work:
        shutil.rmtree(run_dir, ignore_errors=True)

    print("[reportvox] done.")
