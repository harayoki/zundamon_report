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
    input_audio: pathlib.Path | None
    voicevox_url: str
    speakers: SpeakerMode
    speaker1: str
    speaker2: str
    zunda_senior_job: Optional[str]
    zunda_junior_job: Optional[str]
    want_mp3: bool
    mp3_bitrate: str
    keep_work: bool
    whisper_model: str
    llm_backend: LLMBackend
    hf_token: Optional[str] = None
    resume_run_id: Optional[str] = None


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


def _find_existing_input(run_dir: pathlib.Path) -> pathlib.Path | None:
    for path in run_dir.glob("input.*"):
        return path
    return None


def _load_transcription(path: pathlib.Path) -> transcribe.TranscriptionResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        {"start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)), "text": str(seg.get("text", ""))}
        for seg in data.get("segments", [])
    ]
    text = str(data.get("text", ""))
    return transcribe.TranscriptionResult(segments=segments, text=text)


def _load_diarization(path: pathlib.Path) -> list[diarize.DiarizedSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: list[diarize.DiarizedSegment] = []
    for item in data:
        segments.append(
            diarize.DiarizedSegment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                speaker=item.get("speaker", "A"),  # type: ignore[arg-type]
            )
        )
    return segments


def _save_stylized(segments: Sequence[style_convert.StylizedSegment], path: pathlib.Path) -> None:
    data = [
        {"start": seg.start, "end": seg.end, "text": seg.text, "speaker": seg.speaker, "character": seg.character}
        for seg in segments
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_stylized(path: pathlib.Path) -> list[style_convert.StylizedSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: list[style_convert.StylizedSegment] = []
    for item in data:
        segments.append(
            style_convert.StylizedSegment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=str(item.get("text", "")),
                speaker=str(item.get("speaker", "")),
                character=str(item.get("character", "")),
            )
        )
    return segments


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
    resume = config.resume_run_id is not None
    run_id = config.resume_run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = work_dir / run_id

    if not resume:
        # Avoid accidental reuse when multiple runs start within the same second.
        suffix = 1
        base_id = run_id
        while run_dir.exists():
            run_id = f"{base_id}-{suffix}"
            run_dir = work_dir / run_id
            suffix += 1

    if resume:
        if not run_dir.exists():
            raise FileNotFoundError(f"指定された run_id が見つかりませんでした: {run_id}")
        print(f"[reportvox] resuming run id: {run_id}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    hf_token = config.hf_token or os.environ.get("PYANNOTE_TOKEN")
    diarization_path = run_dir / "diarization.json"
    need_new_diarization = config.speakers != "1" and not (resume and diarization_path.exists())
    allow_partial = False
    if need_new_diarization and hf_token is None:
        print("[reportvox] pyannote diarization requires a Hugging Face token (set PYANNOTE_TOKEN or --hf-token).")
        print("[reportvox] Without the token, processing can only proceed until transcription and will then stop.")
        reply = input("[reportvox] Continue with partial processing? [y/N]: ").strip().lower()
        if reply not in ("y", "yes"):
            print("[reportvox] Aborting. Please provide the Hugging Face token and re-run.")
            return
        allow_partial = True

    print("[reportvox] checking ffmpeg availability...")
    audio.ensure_ffmpeg()

    print(f"[reportvox] run id: {run_id}")
    if not resume and config.input_audio is None:
        raise ValueError("input_audio must be provided when not resuming")

    existing_input = _find_existing_input(run_dir) if resume else None
    if existing_input:
        input_path = existing_input
        print(f"[reportvox] found existing input copy -> {input_path.name}")
    elif config.input_audio is not None:
        print(f"[reportvox] copying input...")
        input_path = _copy_input(config.input_audio, run_dir)
    else:
        raise FileNotFoundError("入力ファイルが見つかりません (--resume 先に input.* が存在しません)")

    normalized_input = run_dir / "input.wav"
    if resume and normalized_input.exists():
        print("[reportvox] using existing normalized wav.")
    else:
        audio.normalize_to_wav(input_path, normalized_input)

    transcript_path = run_dir / "transcript.json"
    if resume and transcript_path.exists():
        print("[reportvox] loading existing transcript...")
        whisper_result = _load_transcription(transcript_path)
    else:
        print(f"[reportvox] transcribing with Whisper ({config.whisper_model})...")
        whisper_result = transcribe.transcribe_audio(normalized_input, model_size=config.whisper_model)
        transcript_path.write_text(json.dumps(whisper_result.as_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    if allow_partial:
        print("[reportvox] Hugging Face token was not provided; stopping after transcription as requested.")
        print(f"[reportvox] transcript saved -> {transcript_path}")
        print(f"[reportvox] Re-run with --resume {run_id} after setting PYANNOTE_TOKEN or --hf-token to continue.")
        return

    if resume and diarization_path.exists():
        print(f"[reportvox] loading existing diarization ({config.speakers})...")
        diarization = _load_diarization(diarization_path)
    else:
        print(f"[reportvox] diarizing speakers ({config.speakers})...")
        diarization = diarize.diarize_audio(
            normalized_input,
            mode=config.speakers,
            hf_token=hf_token,
            work_dir=run_dir,
        )
        diarize.save_diarization(diarization, diarization_path)

    aligned = diarize.align_segments(whisper_result.segments, diarization)
    totals = _summarize_speaker_durations(aligned)
    print(f"[reportvox] speaker durations: {totals}")

    char1 = characters.load_character(config.speaker1)
    char2 = characters.load_character(config.speaker2)
    mapped = _map_speakers(aligned, totals, config.speakers, char1, char2)

    stylized_path = run_dir / "stylized.json"
    if resume and stylized_path.exists():
        print("[reportvox] loading stylized segments...")
        stylized = _load_stylized(stylized_path)
    else:
        print("[reportvox] converting style and inserting phrases...")
        stylized = style_convert.apply_style(mapped, char1, char2, backend=config.llm_backend)
        stylized = _maybe_prepend_intro(
            stylized,
            char1=char1,
            senior_job=config.zunda_senior_job,
            junior_job=config.zunda_junior_job,
        )
        _save_stylized(stylized, stylized_path)

    print("[reportvox] synthesizing with VoiceVox...")
    synthesized_paths = voicevox.synthesize_segments(
        stylized,
        characters={char1.id: char1, char2.id: char2},
        base_url=config.voicevox_url,
        run_dir=run_dir,
        skip_existing=resume,
    )

    base_stem = (config.input_audio or input_path).stem
    output_wav = out_dir / f"{base_stem}_report.wav"
    print(f"[reportvox] joining audio -> {output_wav}")
    audio.join_wavs(synthesized_paths, output_wav)

    if config.want_mp3:
        mp3_path = out_dir / f"{base_stem}_report.mp3"
        print(f"[reportvox] generating mp3 -> {mp3_path}")
        audio.convert_to_mp3(output_wav, mp3_path, bitrate=config.mp3_bitrate)

    if not config.keep_work:
        shutil.rmtree(run_dir, ignore_errors=True)

    print("[reportvox] done.")


def _maybe_prepend_intro(
    segments: Sequence[style_convert.StylizedSegment],
    char1: characters.CharacterMeta,
    senior_job: Optional[str],
    junior_job: Optional[str],
) -> list[style_convert.StylizedSegment]:
    if not (senior_job and junior_job):
        return list(segments)

    speaker_label: str | None = None
    for seg in segments:
        if seg.character == char1.id:
            speaker_label = seg.speaker
            break

    if speaker_label is None and segments:
        speaker_label = segments[0].speaker
    if speaker_label is None:
        speaker_label = "A"

    intro_text = f"僕の名前はずんだもん、{senior_job}にあこがれる{junior_job}なのだ"
    intro_segment = style_convert.StylizedSegment(
        start=0.0,
        end=0.0,
        text=intro_text,
        speaker=speaker_label,
        character=char1.id,
    )
    return [intro_segment, *segments]
