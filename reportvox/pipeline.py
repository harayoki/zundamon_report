"""処理パイプラインのオーケストレーション。"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence

from . import diarize, transcribe, style_convert, voicevox, audio, characters, subtitles
from .envinfo import EnvironmentInfo, append_env_details


SpeakerMode = Literal["auto", "1", "2"]
LLMBackend = Literal["none", "openai", "local"]
SubtitleMode = Literal["off", "all", "split"]


class _ProgressReporter:
    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def _format_duration(self, seconds: float) -> str:
        seconds = max(0.0, seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def log(self, message: str, *, step_duration: float | None = None, remaining: float | None = None) -> None:
        parts = [f"[reportvox +{self._format_duration(self.elapsed())}] {message}"]
        if step_duration is not None:
            parts.append(f"(step {self._format_duration(step_duration)})")
        if remaining is not None:
            parts.append(f"(残り予想: {self._format_duration(remaining)})")
        print(" ".join(parts))

    def now(self) -> float:
        return time.monotonic()


def _estimate_remaining(total_steps: int, steps_done: int, elapsed: float) -> float | None:
    if steps_done <= 0 or total_steps <= steps_done:
        return None
    avg = elapsed / steps_done
    return avg * (total_steps - steps_done)


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
    resume_run_id: Optional[str] = None
    subtitle_mode: SubtitleMode = "off"
    subtitle_max_chars: int = 25


def _ensure_paths() -> tuple[pathlib.Path, pathlib.Path]:
    work_dir = pathlib.Path("work")
    out_dir = pathlib.Path("out")
    work_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    return work_dir, out_dir


def _resolve_output_stem(config: PipelineConfig, input_path: pathlib.Path) -> str:
    if config.output_name:
        candidate = pathlib.Path(config.output_name).name
        stem = pathlib.Path(candidate).stem
        return stem or candidate
    input_stem = (config.input_audio or input_path).stem
    return f"{input_stem}_report"


def _collect_existing_outputs(
    out_dir: pathlib.Path,
    base_name: str,
    *,
    want_mp3: bool,
    subtitle_mode: SubtitleMode,
) -> list[pathlib.Path]:
    existing: list[pathlib.Path] = []
    if want_mp3:
        mp3_path = out_dir / f"{base_name}.mp3"
        if mp3_path.exists():
            existing.append(mp3_path)
    else:
        wav_path = out_dir / f"{base_name}.wav"
        if wav_path.exists():
            existing.append(wav_path)

    if subtitle_mode == "all":
        srt_path = out_dir / f"{base_name}.srt"
        if srt_path.exists():
            existing.append(srt_path)
    elif subtitle_mode == "split":
        for path in sorted(out_dir.glob(f"{base_name}_*.srt")):
            if path.exists():
                existing.append(path)
    return existing


def _confirm_overwrite(paths: Sequence[pathlib.Path]) -> None:
    if not paths:
        return
    print("以下の出力ファイルが既に存在します。上書きしてよいですか? [y/N]")
    for path in paths:
        print(f"  - {path}")
    answer = input("> ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("中止しました。--force で上書きします。")


def _copy_input(input_audio: pathlib.Path, run_dir: pathlib.Path, *, env_info: EnvironmentInfo | None = None) -> pathlib.Path:
    if not input_audio.exists():
        raise FileNotFoundError(append_env_details(f"入力音声が見つかりません: {input_audio}", env_info))
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
            # 実質 1 話者として扱う
            for seg in aligned:
                mapped.append(seg.with_character(char1.id))
            return mapped

    # 2 話者の場合: 発話が長い方を speaker1 に割り当てる
    sorted_speakers = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_speakers[0][0] if sorted_speakers else "A"
    for seg in aligned:
        char = char1 if seg.speaker == primary else char2
        mapped.append(seg.with_character(char.id))
    return mapped


def run_pipeline(config: PipelineConfig) -> None:
    reporter = _ProgressReporter()
    work_dir, out_dir = _ensure_paths()
    resume = config.resume_run_id is not None
    run_id = config.resume_run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = work_dir / run_id
    env_info = EnvironmentInfo.collect(config.ffmpeg_path, config.hf_token, os.environ.get("PYANNOTE_TOKEN"))

    if not resume:
        # 同じ秒に実行開始した際の run_id 競合を避ける
        suffix = 1
        base_id = run_id
        while run_dir.exists():
            run_id = f"{base_id}-{suffix}"
            run_dir = work_dir / run_id
            suffix += 1

    if resume:
        if not run_dir.exists():
            raise FileNotFoundError(append_env_details(f"指定された run_id が見つかりませんでした: {run_id}", env_info))
        reporter.log(f"run_id {run_id} で再開します。")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    hf_token = config.hf_token or os.environ.get("PYANNOTE_TOKEN")
    diarization_path = run_dir / "diarization.json"
    need_new_diarization = config.speakers != "1" and not (resume and diarization_path.exists())
    if need_new_diarization and hf_token is None:
        raise RuntimeError(
            append_env_details(
                "pyannote の話者分離には Hugging Face Token が必要ですが、--hf-token と PYANNOTE_TOKEN のいずれも指定されていません。\n"
                f"{diarize._PYANNOTE_ACCESS_STEPS}\n{diarize._PYANNOTE_TOKEN_USAGE}",
                env_info,
            )
        )

    total_steps = 4  # ffmpeg 確認、入力コピー、正規化、文字起こし
    total_steps += 1  # 話者分離の読み込み/実行
    total_steps += 1  # 口調変換
    if config.subtitle_mode != "off":
        total_steps += 1  # 字幕生成
    total_steps += 1  # VOICEVOX 合成
    total_steps += 1  # WAV 結合
    if config.want_mp3:
        total_steps += 1
    if not config.keep_work:
        total_steps += 1
    steps_done = 0

    def _complete_step(message: str, start_time: float) -> None:
        nonlocal steps_done
        duration = reporter.now() - start_time
        steps_done += 1
        remaining = _estimate_remaining(total_steps, steps_done, reporter.elapsed())
        reporter.log(message, step_duration=duration, remaining=remaining)

    reporter.log("ffmpeg の利用可否を確認しています...")
    step_start = reporter.now()
    resolved_ffmpeg = audio.ensure_ffmpeg(config.ffmpeg_path, env_info=env_info)
    config.ffmpeg_path = resolved_ffmpeg
    _complete_step("ffmpeg の確認が完了しました。", step_start)

    reporter.log(f"run_id: {run_id}")
    if not resume and config.input_audio is None:
        raise ValueError(append_env_details("再開しない場合は input_audio が必須です。", env_info))

    existing_input = _find_existing_input(run_dir) if resume else None
    step_start = reporter.now()
    if existing_input:
        input_path = existing_input
        reporter.log(f"既存の入力コピーを使用します -> {input_path.name}")
    elif config.input_audio is not None:
        reporter.log("入力ファイルを作業ディレクトリへコピーしています...")
        input_path = _copy_input(config.input_audio, run_dir, env_info=env_info)
        reporter.log(f"入力をコピーしました -> {input_path.name}")
    else:
        raise FileNotFoundError(append_env_details("入力ファイルが見つかりません (--resume 先に input.* が存在しません)", env_info))
    _complete_step("入力準備が完了しました。", step_start)

    output_base = _resolve_output_stem(config, input_path)
    output_wav = out_dir / f"{output_base}.wav"
    output_mp3 = out_dir / f"{output_base}.mp3"
    existing_outputs = _collect_existing_outputs(
        out_dir,
        output_base,
        want_mp3=config.want_mp3,
        subtitle_mode=config.subtitle_mode,
    )
    if existing_outputs and not config.force_overwrite:
        _confirm_overwrite(existing_outputs)

    normalized_input = run_dir / "input.wav"
    step_start = reporter.now()
    if resume and normalized_input.exists():
        reporter.log("既存の正規化済み WAV を利用します。")
    else:
        audio.normalize_to_wav(input_path, normalized_input, ffmpeg_path=config.ffmpeg_path, env_info=env_info)
    _complete_step("WAV 正規化が完了しました。", step_start)

    transcript_path = run_dir / "transcript.json"
    step_start = reporter.now()
    if resume and transcript_path.exists():
        reporter.log("既存の文字起こし結果を読み込みます...")
        whisper_result = _load_transcription(transcript_path)
    else:
        reporter.log(f"Whisper ({config.whisper_model}) で文字起こし中です... この処理は数分かかる場合があります。")
        whisper_result = transcribe.transcribe_audio(
            normalized_input, model_size=config.whisper_model, env_info=env_info
        )
        transcript_path.write_text(json.dumps(whisper_result.as_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        reporter.log("文字起こしが完了し保存しました。")
    _complete_step("文字起こし工程が完了しました。", step_start)

    step_start = reporter.now()
    if resume and diarization_path.exists():
        reporter.log(f"既存の話者分離結果を読み込みます ({config.speakers})...")
        diarization = _load_diarization(diarization_path)
    else:
        reporter.log(f"話者分離を実行しています ({config.speakers})...")
        with diarize.torchcodec_warning_detector() as torchcodec_warning:
            diarization = diarize.diarize_audio(
                normalized_input,
                mode=config.speakers,
                hf_token=hf_token,
                work_dir=run_dir,
                env_info=env_info,
            )
        if torchcodec_warning.detected:
            reporter.log(
                "TorchCodec/libtorchcodec に関する警告を検出しました。FFmpeg の共有DLLや TorchCodec の互換バージョンが不足していると出ることがあります。"
                " 警告だけならそのまま続行しても構いませんが、話者分離に失敗する場合は README の TorchCodec 対処を確認してください。"
            )
        diarize.save_diarization(diarization, diarization_path)
    _complete_step("話者分離工程が完了しました。", step_start)

    aligned = diarize.align_segments(whisper_result.segments, diarization)
    totals = _summarize_speaker_durations(aligned)
    reporter.log(f"話者ごとの発話時間: {totals}")

    char1 = characters.load_character(config.speaker1)
    char2 = characters.load_character(config.speaker2)
    mapped = _map_speakers(aligned, totals, config.speakers, char1, char2)

    stylized_path = run_dir / "stylized.json"
    step_start = reporter.now()
    if resume and stylized_path.exists():
        reporter.log("既存のスタイル適用済みセグメントを読み込みます...")
        stylized = _load_stylized(stylized_path)
    else:
        reporter.log("口調変換と定型句の挿入を実行しています...")
        stylized = style_convert.apply_style(mapped, char1, char2, backend=config.llm_backend)
        stylized = _maybe_prepend_intro(
            stylized,
            char1=char1,
            senior_job=config.zunda_senior_job,
            junior_job=config.zunda_junior_job,
        )
        _save_stylized(stylized, stylized_path)
        reporter.log("口調変換が完了し保存しました。")
    _complete_step("口調変換工程が完了しました。", step_start)

    reporter.log("VOICEVOX で音声合成を実行しています...")
    step_start = reporter.now()
    synth_durations: list[float] = []

    def _synth_progress(done: int, total: int, duration: float) -> None:
        synth_durations.append(duration)
        avg = sum(synth_durations) / len(synth_durations)
        remaining_segments = max(0, total - done)
        remaining_time = avg * remaining_segments
        reporter.log(f"VOICEVOX 合成 {done}/{total} セグメント完了。", step_duration=duration, remaining=remaining_time)

    synthesized_paths = voicevox.synthesize_segments(
        stylized,
        characters={char1.id: char1, char2.id: char2},
        base_url=config.voicevox_url,
        run_dir=run_dir,
        speed_scale=config.speed_scale,
        skip_existing=resume,
        progress=_synth_progress,
        env_info=env_info,
    )
    _complete_step("VOICEVOX での合成が完了しました。", step_start)

    if config.subtitle_mode != "off":
        reporter.log("字幕ファイルを生成しています...")
        step_start = reporter.now()
        subtitle_segments = subtitles.align_segments_to_audio(stylized, synthesized_paths)
        subtitle_paths = subtitles.write_subtitles(
            subtitle_segments,
            out_dir=out_dir,
            base_stem=output_base,
            mode=config.subtitle_mode,
            characters={char1.id: char1, char2.id: char2},
            max_chars_per_line=config.subtitle_max_chars,
        )
        reporter.log(f"字幕を出力しました: {[p.name for p in subtitle_paths]}")
        _complete_step("字幕ファイルの生成が完了しました。", step_start)

    if config.want_mp3:
        temp_wav = run_dir / f"{output_base}.wav"
        reporter.log(f"音声を結合しています -> {temp_wav} (mp3 用の一時ファイル)")
        step_start = reporter.now()
        audio.join_wavs(synthesized_paths, temp_wav, env_info=env_info)
        _complete_step("音声の結合が完了しました。", step_start)

        reporter.log(f"mp3 を生成しています -> {output_mp3}")
        step_start = reporter.now()
        audio.convert_to_mp3(
            temp_wav,
            output_mp3,
            bitrate=config.mp3_bitrate,
            ffmpeg_path=config.ffmpeg_path,
            env_info=env_info,
        )
        _complete_step("mp3 生成が完了しました。", step_start)
        if temp_wav.exists():
            temp_wav.unlink()
    else:
        reporter.log(f"音声を結合しています -> {output_wav}")
        step_start = reporter.now()
        audio.join_wavs(synthesized_paths, output_wav, env_info=env_info)
        _complete_step("音声の結合が完了しました。", step_start)

    if not config.keep_work:
        reporter.log("作業ディレクトリをクリーンアップしています...")
        step_start = reporter.now()
        shutil.rmtree(run_dir, ignore_errors=True)
        _complete_step("作業ディレクトリを削除しました。", step_start)
    else:
        reporter.log(f"作業ディレクトリを保持します -> {run_dir}")

    reporter.log("すべての処理が完了しました。")


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
