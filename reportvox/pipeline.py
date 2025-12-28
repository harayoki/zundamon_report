"処理パイプラインのオーケストレーション。"

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import pathlib
import re
import random
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from . import audio, characters, diarize, style_convert, subtitles, transcribe, utils, video, voicevox
from .config import LLMBackend, PipelineConfig, SpeakerMode, SubtitleMode, TranscriptReviewMode
from .envinfo import EnvironmentInfo, append_env_details, resolve_hf_token
from .llm_client import chat_completion


class _ProgressReporter:
    def __init__(self) -> None:
        self._start = time.monotonic()
        self.timings: dict[str, float] = {}

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

    def record_step(self, name: str, duration: float) -> None:
        """ステップの処理時間を記録する。"""
        self.timings[name] = self.timings.get(name, 0.0) + duration

    def summarize(self) -> None:
        """処理時間のサマリーを表示する。"""
        print("\n--- 処理時間サマリー ---")
        total_duration = self.elapsed()
        # "finalize" ステップは変動が大きいため、それ以外の合計を計算
        relevant_total = sum(d for n, d in self.timings.items() if n != "finalize")

        sorted_steps = sorted(self.timings.items(), key=lambda item: item[1], reverse=True)

        for name, duration in sorted_steps:
            percentage = (duration / relevant_total) * 100 if relevant_total > 0 else 0
            print(f"- {name:<15}: {self._format_duration(duration)} ({percentage:.1f}%)")
        print("--------------------------")
        print(f"- {'合計処理時間':<13}: {self._format_duration(total_duration)}")
        print("--------------------------\n")


def _estimate_remaining(total_steps: int, steps_done: int, elapsed: float) -> float | None:
    if steps_done <= 0 or total_steps <= steps_done:
        return None
    avg = elapsed / steps_done
    return avg * (total_steps - steps_done)


def _extract_json_payload(text: str) -> str:
    """LLM 応答から JSON 本文を抽出する。

    Markdown のコードブロックや前後の文章に囲まれていても、最初の
    JSON オブジェクト部分を取り出す。
    """

    trimmed = text.strip()

    fence_match = re.fullmatch(r"```(?:json)?\n(.*?)\n```", trimmed, flags=re.DOTALL)
    if fence_match:
        trimmed = fence_match.group(1).strip()

    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed

    brace_match = re.search(r"\{.*\}", trimmed, flags=re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return trimmed


def _ensure_paths() -> tuple[pathlib.Path, pathlib.Path]:
    work_dir = pathlib.Path("work")
    out_dir = pathlib.Path("out")
    work_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    return work_dir, out_dir


def _slugify_for_cache(path: pathlib.Path) -> str:
    base = path.stem or "audio"
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", base)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{digest}"


def _create_prompt_logger(path: pathlib.Path, title: str) -> style_convert.LLMPromptLogger:
    path.parent.mkdir(parents=True, exist_ok=True)

    def log_prompt(system_prompt: str, user_prompt: str) -> Callable[[str | Exception], None]:
        with path.open("a", encoding="utf-8") as fp:
            fp.write(f"=== {title} ===\n")
            fp.write("[System]\n")
            fp.write(system_prompt.strip() + "\n\n")
            fp.write("[User]\n")
            fp.write(user_prompt.strip() + "\n\n")

        def log_response(response: str | Exception) -> None:
            with path.open("a", encoding="utf-8") as fp:
                fp.write("[Response]\n")
                if isinstance(response, Exception):
                    fp.write(f"[ERROR] {type(response).__name__}: {response}\n\n")
                else:
                    fp.write(str(response).strip() + "\n\n")

        return log_response

    return log_prompt


def _resolve_transcription_cache_path(config: PipelineConfig, out_dir: pathlib.Path) -> pathlib.Path | None:
    if config.input_audio is None:
        return None
    cache_dir = out_dir / "transcripts" / config.whisper_model
    return cache_dir / f"{_slugify_for_cache(config.input_audio)}.json"


def _resolve_diarization_cache_path(config: PipelineConfig, out_dir: pathlib.Path) -> pathlib.Path | None:
    if config.input_audio is None:
        return None
    threshold_label = str(config.diarization_threshold).replace(".", "_")
    cache_dir = out_dir / "diarization" / f"{config.speakers}_thr{threshold_label}"
    return cache_dir / f"{_slugify_for_cache(config.input_audio)}.json"


def _get_mtime(path: pathlib.Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _resolve_output_stem(config: PipelineConfig, input_path: pathlib.Path) -> str:
    if config.output_name:
        candidate = pathlib.Path(config.output_name).name
        stem = pathlib.Path(candidate).stem
        return stem or candidate
    input_stem = (config.input_audio or input_path).stem
    return f"{input_stem}_report"


def _estimate_total_duration(segments: Sequence[style_convert.StylizedSegment]) -> float:
    if not segments:
        return 0.0
    max_end = max(seg.end for seg in segments)
    if max_end > 0:
        return max_end
    return sum(max(0.0, seg.end - seg.start) for seg in segments)


def _resolve_temp_output_wav(run_dir: pathlib.Path) -> pathlib.Path:
    """作業ディレクトリに保持する一時的な出力 WAV のパスを返す。"""

    return run_dir / "output.wav"


def _build_target_durations(
    segments: Sequence[style_convert.StylizedSegment],
    *,
    max_pause: float,
    speed_scale: float = 1.0,
) -> list[float]:
    durations: list[float] = []
    speed_factor = 1.0 / max(speed_scale, 1e-6)
    for idx, seg in enumerate(segments):
        next_start = segments[idx + 1].start if idx + 1 < len(segments) else seg.end
        gap_to_next = max(0.0, next_start - seg.start)
        segment_length = max(0.0, seg.end - seg.start)
        expected_length = segment_length * speed_factor
        silence_to_next = max(0.0, gap_to_next - expected_length)
        if max_pause < 0:
            clamped_gap = expected_length + silence_to_next
        else:
            clamped_gap = expected_length + min(silence_to_next, max_pause)
        base = max(clamped_gap, expected_length, 0.05)
        durations.append(base)
    return durations


def _collect_existing_outputs(
    out_dir: pathlib.Path,
    base_name: str,
    *,
    want_mp3: bool,
    subtitle_mode: SubtitleMode,
    want_mp4: bool,
    want_mov: bool,
) -> list[pathlib.Path]:
    existing: list[pathlib.Path] = []
    want_wav_output = not want_mp3 and not (want_mp4 or want_mov)

    if want_mp3:
        mp3_path = out_dir / f"{base_name}.mp3"
        if mp3_path.exists():
            existing.append(mp3_path)
    elif want_wav_output:
        wav_path = out_dir / f"{base_name}.wav"
        if wav_path.exists():
            existing.append(wav_path)

    if want_mp4:
        mp4_path = out_dir / f"{base_name}.mp4"
        if mp4_path.exists():
            existing.append(mp4_path)
    if want_mov:
        mov_path = out_dir / f"{base_name}.mov"
        if mov_path.exists():
            existing.append(mov_path)

    if subtitle_mode == "all":
        srt_path = out_dir / f"{base_name}_report.srt"
        if srt_path.exists():
            existing.append(srt_path)
    elif subtitle_mode == "split":
        for path in sorted(out_dir.glob(f"{base_name}_report_*.srt")):
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


def _llm_review_transcription(
    result: transcribe.TranscriptionResult,
    *,
    config: PipelineConfig,
    run_dir: pathlib.Path,
    prompt_log_path: pathlib.Path | None = None,
    env_info: EnvironmentInfo | None = None,
) -> transcribe.TranscriptionResult:
    # system_prompt = (
    #     "あなたは、日本語の文字起こしを校正する専門家です。\n"
    #     "以下のルールを厳密に守ってください:\n"
    #     "- 応答は、修正後のJSONオブジェクトのみを含めてください。\n"
    #     "- 説明、前置き、言い訳、追加のテキストは一切不要です。\n"
    #     "- 明らかな誤字脱字や、不自然な句読点のみを修正してください。\n"
    #     "- 話し手の意図や発言内容、固有名詞を絶対に変えないでください。\n"
    #     "- segments配列のstart/endの値、および配列の長さは絶対に変更しないでください。\n"
    #     "- 英語に翻訳しないでください。\n"
    #     "- 応答のJSONは、元の構造 `{\"segments\": [...], \"text\": \"...\"}` を完全に維持してください。"
    # )
    system_prompt = (
        "あなたは、日本語の文字起こしを校正する専門家です。\n"
        "以下のルールを厳密に守ってください:\n"
        "- 応答は、修正後のJSONオブジェクトのみを含めてください。\n"
        "- 説明、前置き、言い訳、追加のテキストは一切不要です。\n"
        "- 修正を許可するのは、明らかな誤字脱字・句読点・空白・全角半角の誤りのみです。\n"
        "- 語尾（です/ます、だ/である、〜してください/〜しないでください等）を絶対に変更しないでください。\n"
        "- 助詞、活用形、敬語、言い回し、同義語への置換を一切禁止します。\n"
        "- 話し手の意図や発言内容、意味、ニュアンスを絶対に変えないでください。\n"
        "- 固有名詞を絶対に変更しないでください。\n"
        "- segments 配列の長さを絶対に変更しないでください（順序も固定）。\n"
        "- 削除・省略は禁止。繰り返し表現もそのまま残すこと。\n"
        "- 修正は置換のみ。削除・追加は禁止（句読点/空白/全角半角の修正は例外）。\n"
        "- 英語に翻訳しないでください。\n"
        "- 応答のJSONは、元の構造を完全に維持してください。"
    )
    conversation_lines = [seg["text"] for seg in result.segments]
    simplified = {"lines": conversation_lines}
    user_prompt = (
        "以下のJSONに含まれる会話テキスト（1 行につき 1 発話）を、上記のルールに従って校正してください。"
        "各行の順序と行数は絶対に変更しないでください。同じ JSON 構造で応答してください。\n"
        f"{json.dumps(simplified, ensure_ascii=False)}"
    )

    if prompt_log_path is None:
        prompt_log_path = run_dir / "prompt_transcript_review_llm.txt"

    prompt_log_path.write_text("", encoding="utf-8")
    log_prompt = _create_prompt_logger(prompt_log_path, "Transcript Review LLM Prompt")
    log_response = log_prompt(system_prompt, user_prompt)

    try:
        content = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt, config=config, env_info=env_info)
        log_response(content)
    except Exception as exc:
        log_response(exc)
        raise
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(append_env_details("LLM の応答を JSON として読み取れませんでした。", env_info)) from exc

    segments_in = data.get("lines")
    if not isinstance(segments_in, list) or len(segments_in) != len(result.segments):
        raise RuntimeError(
            append_env_details("LLM の応答形式が不正です（行数が一致しません）。", env_info)
        )

    reviewed_segments: list[dict] = []
    for original, revised in zip(result.segments, segments_in):
        text = revised if isinstance(revised, str) else original.get("text", "")
        reviewed_segments.append(
            {"start": float(original["start"]), "end": float(original["end"]), "text": str(text).strip()}
        )

    reviewed_text = "\n".join(str(seg.get("text", "")) for seg in reviewed_segments)
    reviewed_result = transcribe.TranscriptionResult(segments=reviewed_segments, text=reviewed_text)

    def _format_transcription_for_diff(segments: Sequence[dict]) -> list[str]:
        return [f"{str(seg.get('text', '')).strip()}\n" for seg in segments]

    diff = difflib.unified_diff(
        _format_transcription_for_diff(result.segments),
        _format_transcription_for_diff(reviewed_segments),
        fromfile="before_llm_review.txt",
        tofile="after_llm_review.txt",
    )
    diff_log_path = run_dir / "diff_llm_review.log"
    diff_log_path.write_text("".join(diff), encoding="utf-8")

    return reviewed_result


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


def _save_placements(placements: Sequence[tuple[float, float]], path: pathlib.Path) -> None:
    data = [{"start": start, "end": end} for start, end in placements]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_placements(path: pathlib.Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(float(item.get("start", 0.0)), float(item.get("end", 0.0))) for item in data]


def _write_cli_args_snapshot(run_dir: pathlib.Path, cli_args: list[str] | None) -> None:
    if cli_args is None:
        return
    metadata_path = run_dir / "metadata.json"
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    except json.JSONDecodeError:
        existing = {}

    existing["cli_args"] = list(cli_args)
    existing.setdefault("run_id", run_dir.name)
    metadata_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_run_metadata(state: "PipelineState") -> None:
    metadata = {
        "run_id": state.run_id,
        "output_base_name": state.output_base_name,
        "speaker1": state.config.speaker1,
        "speaker2": state.config.speaker2,
        "color1": state.config.color1,
        "color2": state.config.color2,
        "cli_args": state.config.cli_args,
        "subtitle_max_chars": state.config.subtitle_max_chars,
        "subtitle_font": state.config.subtitle_font,
        "subtitle_font_size": state.config.subtitle_font_size,
        "video_width": state.config.video_width,
        "video_height": state.config.video_height,
        "video_fps": state.config.video_fps,
        "video_images": [str(p) for p in state.config.video_images],
        "video_image_scale": state.config.video_image_scale,
        "video_image_position": state.config.video_image_position,
        "video_image_times": state.config.video_image_times,
        "ffmpeg_path": state.config.ffmpeg_path,
    }
    path = state.run_dir / "metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_segments_for_diff(
    segments: Sequence[diarize.AlignedSegment | style_convert.StylizedSegment],
) -> list[str]:
    lines: list[str] = []
    for seg in segments:
        label = getattr(seg, "character", None) or getattr(seg, "speaker", "")
        prefix = f"[{label}] " if label else ""
        lines.append(f"{prefix}{getattr(seg, 'text', '')}\n")
    return lines


def _log_style_diff(
    original: Sequence[diarize.AlignedSegment],
    stylized: Sequence[style_convert.StylizedSegment],
    path: pathlib.Path,
) -> None:
    before = _format_segments_for_diff(original)
    after = _format_segments_for_diff(stylized)
    diff = difflib.unified_diff(before, after, fromfile="before_stylize.txt", tofile="after_stylize.txt")
    path.write_text("".join(diff), encoding="utf-8")


def _summarize_speaker_durations(aligned: Sequence[diarize.AlignedSegment]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for seg in aligned:
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + (seg.end - seg.start)
    return totals


def _rgb_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = (value / 255.0 for value in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _spread_default_colors(color1: str, color2: str, *, threshold: float = 90.0) -> tuple[str, str, bool]:
    base1 = utils.hex_to_rgb(color1)
    base2 = utils.hex_to_rgb(color2)
    if _rgb_distance(base1, base2) >= threshold:
        return color1, color2, False

    def _lighten(rgb: tuple[int, int, int], amount: int = 28) -> tuple[int, int, int]:
        return tuple(min(255, value + amount) for value in rgb)  # type: ignore[return-value]

    def _darken(rgb: tuple[int, int, int], amount: int = 28) -> tuple[int, int, int]:
        return tuple(max(0, value - amount) for value in rgb)  # type: ignore[return-value]

    base2_luminance = _luminance(base2)
    base1_luminance = _luminance(base1)
    adjusted_second = _lighten(base2) if base2_luminance <= base1_luminance else _darken(base2)

    return color1, utils.rgb_to_hex(adjusted_second), True


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


def _ensure_character_colors(
    state: "PipelineState", char1: characters.CharacterMeta, char2: characters.CharacterMeta
) -> Dict[str, str]:
    if state.character_colors is not None:
        return state.character_colors

    config = state.config
    reporter = state.reporter

    color1 = config.color1 or char1.main_color
    color2 = config.color2 or char2.main_color

    try:
        color1 = utils.normalize_hex_color(color1)
        color2 = utils.normalize_hex_color(color2)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    color1_from_arg = config.color1 is not None
    color2_from_arg = config.color2 is not None

    if color1_from_arg and color2_from_arg and color1 == color2:
        reporter.log(
            f"警告: speaker1 と speaker2 のカラーコードが同じです ({color1})。指定された値をそのまま使用します。"
        )

    if not color1_from_arg and not color2_from_arg:
        spread1, spread2, adjusted = _spread_default_colors(color1, color2)
        if adjusted:
            reporter.log(
                "メインカラーが近いため二人目のカラーを調整しました: "
                f"{char2.display_name} {color2} -> {spread2} (基準 {char1.display_name} {color1})"
            )
            color1, color2 = spread1, spread2

    reporter.log(
        "キャラクターカラー: "
        f"{char1.display_name}={color1}{' (指定)' if color1_from_arg else ' (メインカラー)'} / "
        f"{char2.display_name}={color2}{' (指定)' if color2_from_arg else ' (メインカラー)'}"
    )

    colors = {char1.id: color1, char2.id: color2}
    state.character_colors = colors
    return colors


@dataclass
class PipelineState:
    """パイプラインの実行状態を管理する。"""

    config: PipelineConfig
    reporter: _ProgressReporter
    run_id: str
    run_dir: pathlib.Path
    out_dir: pathlib.Path
    env_info: EnvironmentInfo
    hf_token: Optional[str] = None
    steps_done: int = 0
    total_steps: int = 0
    # Step outputs
    normalized_input_path: Optional[pathlib.Path] = None
    output_base_name: Optional[str] = None
    transcription_result: Optional[transcribe.TranscriptionResult] = None
    diarization_result: Optional[list[diarize.DiarizedSegment]] = None
    stylized_segments: Optional[list[style_convert.StylizedSegment]] = None
    synthesized_paths: Optional[list[pathlib.Path]] = None
    placements: Optional[list[tuple[float, float]]] = None
    character_colors: Optional[Dict[str, str]] = None

    def complete_step(self, message: str, start_time: float) -> None:
        """ステップ完了を報告し、進捗を更新する。"""
        duration = self.reporter.now() - start_time
        self.steps_done += 1
        remaining = _estimate_remaining(self.total_steps, self.steps_done, self.reporter.elapsed())
        self.reporter.log(message, step_duration=duration, remaining=remaining)


def _step_prepare_input(state: PipelineState) -> None:
    """入力ファイルの準備、正規化、上書き確認などを行う。"""
    if state.normalized_input_path and (state.run_dir / "input.wav").exists():
        return

    config = state.config
    reporter = state.reporter
    run_dir = state.run_dir
    out_dir = state.out_dir
    env_info = state.env_info
    resume = config.resume_run_id is not None

    if config.video_images:
        missing_images = [path for path in config.video_images if not path.exists()]
        if missing_images:
            missing_list = ", ".join(str(path) for path in missing_images)
            raise FileNotFoundError(
                append_env_details(f"指定された動画用画像が見つかりません: {missing_list}", env_info)
            )

    reporter.log("ffmpeg の利用可否を確認しています...")
    step_start = reporter.now()
    resolved_ffmpeg = audio.ensure_ffmpeg(config.ffmpeg_path, env_info=env_info)
    config.ffmpeg_path = resolved_ffmpeg
    state.complete_step("ffmpeg の確認が完了しました。", step_start)

    reporter.log(f"run_id: {state.run_id}")
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
    state.complete_step("入力準備が完了しました。", step_start)

    output_base = _resolve_output_stem(config, input_path)
    state.output_base_name = output_base

    _write_run_metadata(state)

    existing_outputs = _collect_existing_outputs(
        out_dir,
        output_base,
        want_mp3=config.want_mp3,
        subtitle_mode=config.subtitle_mode,
        want_mp4=config.output_mp4,
        want_mov=config.output_mov,
    )
    if existing_outputs and not config.force_overwrite:
        _confirm_overwrite(existing_outputs)

    normalized_input = run_dir / "input.wav"
    step_start = reporter.now()
    if resume and normalized_input.exists():
        reporter.log("既存の正規化済み WAV を利用します。")
    elif input_path.suffix.lower() == ".wav":
        if input_path.resolve() == normalized_input.resolve():
            reporter.log("入力が既に目的のWAVファイルのため、正規化をスキップします。")
        else:
            reporter.log("入力が WAV ファイルのため、正規化をスキップしてコピーします。")
            shutil.copy2(input_path, normalized_input)
    else:
        reporter.log("WAV 形式に正規化しています...")
        audio.normalize_to_wav(input_path, normalized_input, ffmpeg_path=config.ffmpeg_path, env_info=env_info)
    state.complete_step("WAV 正規化が完了しました。", step_start)
    state.normalized_input_path = normalized_input


def _step_transcribe(state: PipelineState) -> None:
    """Whisper で文字起こしを行う。"""
    if state.transcription_result:
        return
    if not (state.run_dir / "transcript.json").exists():
        state.reporter.log("文字起こし結果が見つからないため、文字起こしステップから実行します。")
        _step_prepare_input(state)

    config = state.config
    reporter = state.reporter
    run_dir = state.run_dir
    out_dir = state.out_dir
    env_info = state.env_info
    resume = config.resume_run_id is not None

    transcript_path = run_dir / "transcript.json"
    cache_path = _resolve_transcription_cache_path(config, out_dir)
    step_start = reporter.now()
    cached_result: transcribe.TranscriptionResult | None = None

    if not resume and cache_path and cache_path.exists() and not config.force_transcribe:
        audio_mtime = _get_mtime(config.input_audio or state.normalized_input_path)
        cache_mtime = _get_mtime(cache_path)
        if cache_mtime is not None and (audio_mtime is None or cache_mtime >= audio_mtime):
            try:
                cache_label = str(cache_path.relative_to(out_dir))
            except ValueError:
                cache_label = str(cache_path)
            reporter.log(f"キャッシュ済みの文字起こしを使用します -> {cache_label}")
            cached_json = cache_path.read_text(encoding="utf-8")
            transcript_path.write_text(cached_json, encoding="utf-8")
            cached_result = _load_transcription(cache_path)

    if resume and transcript_path.exists():
        reporter.log("既存の文字起こし結果を読み込みます...")
        whisper_result = _load_transcription(transcript_path)
    else:
        if cached_result:
            whisper_result = cached_result
        else:
            reporter.log(f"Whisper ({config.whisper_model}) で文字起こし中です... この処理は数分かかる場合があります。")
            whisper_result = transcribe.transcribe_audio(
                state.normalized_input_path, model_size=config.whisper_model, env_info=env_info
            )
            transcript_json = json.dumps(whisper_result.as_json(), ensure_ascii=False, indent=2)
            transcript_path.write_text(transcript_json, encoding="utf-8")
            reporter.log("文字起こしが完了し保存しました。")
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(transcript_json, encoding="utf-8")
                try:
                    cache_label = str(cache_path.relative_to(out_dir))
                except ValueError:
                    cache_label = str(cache_path)
                reporter.log(f"文字起こし結果をキャッシュしました -> {cache_label}")
        if config.review_transcript == "manual":
            reporter.log("transcript.json を開いて明らかな誤字脱字を修正できます。修正後、次のコマンドで再開してください。")
            reporter.log(f"python -m reportvox --resume {state.run_id}")
            raise SystemExit(0)
        elif config.review_transcript == "llm":
            if config.llm_backend == "none":
                reporter.log("LLM バックエンドが指定されていないため、誤字脱字の自動校正をスキップします (--llm で設定可能)。")
            else:
                reporter.log("LLM で文字起こしを校正しています...")
                prompt_log_path = run_dir / "prompt_transcript_review_llm.txt"
                try:
                    whisper_result = _llm_review_transcription(
                        whisper_result,
                        config=config,
                        run_dir=run_dir,
                        prompt_log_path=prompt_log_path,
                        env_info=env_info,
                    )
                    transcript_path.write_text(
                        json.dumps(whisper_result.as_json(), ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    reporter.log("LLM による校正を完了し、transcript.json を更新しました。")
                    reporter.log(f"誤字脱字校正用プロンプトを {prompt_log_path.name} に保存しました。")
                except Exception as exc:
                    reporter.log(f"LLM 校正に失敗したため、元の文字起こしで続行します: {exc}")
    state.complete_step("文字起こし工程が完了しました。", step_start)
    state.transcription_result = whisper_result


def _step_diarize(state: PipelineState) -> None:
    """pyannote で話者分離を行う。"""
    if state.diarization_result:
        return
    if not (state.run_dir / "diarization.json").exists():
        state.reporter.log("話者分離結果が見つからないため、話者分離ステップから実行します。")
        _step_prepare_input(state)

    config = state.config
    reporter = state.reporter
    run_dir = state.run_dir
    out_dir = state.out_dir
    env_info = state.env_info
    resume = config.resume_run_id is not None

    hf_token = state.hf_token
    diarization_path = run_dir / "diarization.json"
    cache_path = _resolve_diarization_cache_path(config, out_dir)
    step_start = reporter.now()
    cached_result: list[diarize.DiarizedSegment] | None = None

    if not resume and cache_path and cache_path.exists() and not config.force_diarize:
        audio_mtime = _get_mtime(config.input_audio or state.normalized_input_path)
        cache_mtime = _get_mtime(cache_path)
        if cache_mtime is not None and (audio_mtime is None or cache_mtime >= audio_mtime):
            try:
                cache_label = str(cache_path.relative_to(out_dir))
            except ValueError:
                cache_label = str(cache_path)
            reporter.log(f"キャッシュ済みの話者分離を使用します -> {cache_label}")
            diarization_path.write_text(cache_path.read_text(encoding="utf-8"), encoding="utf-8")
            cached_result = _load_diarization(cache_path)

    if resume and diarization_path.exists():
        reporter.log(f"既存の話者分離結果を読み込みます ({config.speakers})...")
        diarization = _load_diarization(diarization_path)
    else:
        if cached_result:
            diarization = cached_result
        else:
            reporter.log(f"話者分離を実行しています ({config.speakers})...")
            with diarize.torchcodec_warning_detector() as torchcodec_warning:
                diarization = diarize.diarize_audio(
                    state.normalized_input_path,
                    mode=config.speakers,
                    hf_token=hf_token,
                    work_dir=run_dir,
                    env_info=env_info,
                    threshold=config.diarization_threshold,
                )
            if torchcodec_warning.detected:
                reporter.log(
                    "TorchCodec/libtorchcodec に関する警告を検出しました。FFmpeg の共有DLLや TorchCodec の互換バージョンが不足していると出ることがあります。"
                    " 警告だけならそのまま続行しても構いませんが、話者分離に失敗する場合は README の TorchCodec 対処を確認してください。"
                )
            diarize.save_diarization(diarization, diarization_path)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                diarize.save_diarization(diarization, cache_path)
                try:
                    cache_label = str(cache_path.relative_to(out_dir))
                except ValueError:
                    cache_label = str(cache_path)
                reporter.log(f"話者分離結果をキャッシュしました -> {cache_label}")
    state.complete_step("話者分離工程が完了しました。", step_start)
    state.diarization_result = diarization


def _step_stylize(state: PipelineState) -> None:
    """口調変換と定型句の挿入を行う。"""
    if state.stylized_segments:
        return
    if not (state.run_dir / "stylized.json").exists():
        state.reporter.log("口調変換済みセグメントが見つからないため、口調変換ステップから実行します。")
        _step_transcribe(state)
        _step_diarize(state)

    config = state.config
    reporter = state.reporter
    run_dir = state.run_dir
    resume = config.resume_run_id is not None

    whisper_result = state.transcription_result
    diarization = state.diarization_result
    aligned = diarize.align_segments(whisper_result.segments, diarization)
    totals = _summarize_speaker_durations(aligned)
    reporter.log(f"話者ごとの発話時間: {totals}")

    char1 = characters.load_character(config.speaker1)
    char2 = characters.load_character(config.speaker2)
    _ensure_character_colors(state, char1, char2)
    mapped = _map_speakers(aligned, totals, config.speakers, char1, char2)

    stylized_path = run_dir / "stylized.json"
    step_start = reporter.now()
    if resume and stylized_path.exists():
        reporter.log("既存のスタイル適用済みセグメントを読み込みます...")
        stylized = _load_stylized(stylized_path)
    else:
        reporter.log("口調変換と定型句の挿入を実行しています...")
        prompt_logger = None
        prompt_log_path = run_dir / "prompt_style_llm.log"
        if config.style_with_llm and config.llm_backend not in {"none", "gemini"}:
            prompt_log_path.write_text("", encoding="utf-8")
            reporter.log(f"口調変換用LLMプロンプトを {prompt_log_path.name} に記録します。")

            prompt_logger = _create_prompt_logger(prompt_log_path, "Style LLM Prompt")

        stylized = style_convert.apply_style(mapped, char1, char2, config=config, prompt_logger=prompt_logger)
        stylized = _prepend_introductions(
            stylized,
            char1=char1,
            char2=char2,
            config=config,
            env_info=state.env_info,
            transcript_text=whisper_result.text if whisper_result else None,
            reporter=reporter,
            run_dir=run_dir,
        )
        linebreak_prompt_logger = None
        linebreak_prompt_path = run_dir / "prompt_line_break_llm.log"
        if config.linebreak_with_llm and config.llm_backend != "none":
            linebreak_prompt_path.write_text("", encoding="utf-8")
            reporter.log(f"改行調整用LLMプロンプトを {linebreak_prompt_path.name} に記録します。")

            linebreak_prompt_logger = _create_prompt_logger(linebreak_prompt_path, "Line Break LLM Prompt")

        stylized = _insert_line_breaks(
            stylized,
            config=config,
            prompt_logger=linebreak_prompt_logger,
        )
        _save_stylized(stylized, stylized_path)
        _log_style_diff(mapped, stylized, run_dir / "diff_style.log")
        reporter.log("口調変換が完了し保存しました。")
    state.complete_step("口調変換工程が完了しました。", step_start)
    state.stylized_segments = stylized


def _step_synthesize(state: PipelineState) -> None:
    """VOICEVOX で音声合成を実行する。"""
    # このステップは再開時に既存ファイルを使うロジックが既にあるため、依存関係チェックのみ
    if state.stylized_segments is None:
        state.reporter.log("口調変換済みセグメントが見つからないため、口調変換ステップから実行します。")
        _step_stylize(state)

    config = state.config
    reporter = state.reporter
    run_dir = state.run_dir
    resume = config.resume_run_id is not None
    env_info = state.env_info

    char1 = characters.load_character(config.speaker1)
    char2 = characters.load_character(config.speaker2)

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
        state.stylized_segments,
        characters={char1.id: char1, char2.id: char2},
        base_url=config.voicevox_url,
        run_dir=run_dir,
        speed_scale=config.speed_scale,
        skip_existing=resume,
        progress=_synth_progress,
        env_info=env_info,
    )
    state.complete_step("VOICEVOX での合成が完了しました。", step_start)
    state.synthesized_paths = synthesized_paths


def _step_concatenate(state: PipelineState) -> None:
    """合成された WAV をタイムラインに配置して結合する。"""
    placements_path = state.run_dir / "placements.json"

    if state.placements:
        return

    if state.config.resume_run_id is not None and placements_path.exists():
        state.reporter.log("既存の配置情報を読み込みます...")
        state.placements = _load_placements(placements_path)
        return

    if state.synthesized_paths is None:
        state.reporter.log("合成済み音声が見つからないため、音声合成ステップから実行します。")
        _step_synthesize(state)

    config = state.config
    reporter = state.reporter
    run_dir = state.run_dir
    out_dir = state.out_dir
    env_info = state.env_info

    target_durations = _build_target_durations(
        state.stylized_segments,
        max_pause=config.max_pause_between_segments,
        speed_scale=config.speed_scale,
    )

    reporter.log("セグメントをタイムラインに配置しています...")
    step_start = reporter.now()
    temp_wav = _resolve_temp_output_wav(run_dir)
    placements = audio.join_wavs(
        state.synthesized_paths,
        temp_wav,
        target_durations=target_durations,
        env_info=env_info,
    )
    _save_placements(placements, placements_path)
    state.complete_step("音声の結合が完了しました。", step_start)
    state.placements = placements


def _step_finalize(state: PipelineState) -> None:
    """字幕生成、mp3 化、作業ディレクトリのクリーンアップなどを行う。"""
    if state.placements is None:
        state.reporter.log("配置情報が見つからないため、音声結合ステップから実行します。")
        _step_concatenate(state)

    config = state.config
    reporter = state.reporter
    run_dir = state.run_dir
    out_dir = state.out_dir
    env_info = state.env_info

    char1 = characters.load_character(config.speaker1)
    char2 = characters.load_character(config.speaker2)
    colors = _ensure_character_colors(state, char1, char2)

    need_video = config.output_mp4 or config.output_mov
    subtitle_segments_for_files: list[style_convert.StylizedSegment] | None = None
    subtitle_segments_for_video: list[style_convert.StylizedSegment] | None = None
    max_chars = config.subtitle_max_chars

    want_subtitle_files = config.subtitle_mode != "off"

    if want_subtitle_files or need_video:
        base_subtitle_segments = subtitles.align_segments_to_audio(
            state.stylized_segments, state.synthesized_paths, placements=state.placements
        )
        if want_subtitle_files:
            subtitle_segments_for_files = base_subtitle_segments
        if need_video:
            # 動画に焼き込む字幕は常にキャラクター別のトラックを使う
            # （--subtitles の指定は out/ へ保存する SRT のみを制御）。
            subtitle_segments_for_video = subtitles.merge_subtitle_segments(base_subtitle_segments)

    if config.subtitle_mode != "off" and subtitle_segments_for_files is not None:
        reporter.log("字幕ファイルを生成しています...")
        step_start = reporter.now()
        subtitle_paths = subtitles.write_subtitles(
            subtitle_segments_for_files,
            out_dir=out_dir,
            base_stem=state.output_base_name,
            mode=config.subtitle_mode,
            characters={char1.id: char1, char2.id: char2},
            max_chars_per_line=max_chars,
        )
        reporter.log(f"字幕を出力しました: {[p.name for p in subtitle_paths]}")
        state.complete_step("字幕ファイルの生成が完了しました。", step_start)

    output_wav = out_dir / f"{state.output_base_name}.wav"
    output_mp3 = out_dir / f"{state.output_base_name}.mp3"
    temp_wav = _resolve_temp_output_wav(run_dir)

    need_wav_output = not config.want_mp3 and not need_video

    if config.want_mp3:
        reporter.log(f"mp3 を生成しています -> {output_mp3}")
        step_start = reporter.now()
        audio.convert_to_mp3(
            temp_wav,
            output_mp3,
            bitrate=config.mp3_bitrate,
            ffmpeg_path=config.ffmpeg_path,
            env_info=env_info,
        )
        state.complete_step("mp3 生成が完了しました。", step_start)
        if need_wav_output:
            shutil.copy2(temp_wav, output_wav)
    else:
        if need_wav_output:
            shutil.copy2(temp_wav, output_wav)

    ass_path: pathlib.Path | None = None
    audio_for_video = temp_wav
    image_overlays: list[tuple[pathlib.Path, float, float]] = []

    if need_video:
        if subtitle_segments_for_video is None:
            raise RuntimeError("動画生成に必要な字幕データを準備できませんでした。")
        reporter.log("動画用の字幕ファイルを生成しています...")
        ass_path = run_dir / f"{state.output_base_name}.ass"
        subtitles.write_ass_subtitles(
            subtitle_segments_for_video,
            path=ass_path,
            characters={char1.id: char1, char2.id: char2},
            colors=colors,
            font=config.subtitle_font,
            font_size=config.subtitle_font_size,
            resolution=(config.video_width, config.video_height),
            max_chars_per_line=max_chars,
        )
        reporter.log(f"字幕ASSを生成しました -> {ass_path.name}")

        if config.video_images:
            video_duration = audio.read_wav_duration(audio_for_video)
            image_overlays = video.build_image_overlays(
                config.video_images,
                video_duration=video_duration,
                start_times=config.video_image_times,
                env_info=env_info,
            )

    if need_video and ass_path is not None:
        if not audio_for_video.exists():
            raise FileNotFoundError(append_env_details("動画出力用の音声ファイルが見つかりません。", env_info))
        if config.output_mp4:
            target = out_dir / f"{state.output_base_name}.mp4"
            reporter.log(f"mp4 を生成しています -> {target}")
            step_start = reporter.now()
            video.render_video_with_subtitles(
                audio=audio_for_video,
                subtitles=ass_path,
                output=target,
                ffmpeg_path=config.ffmpeg_path,
                width=config.video_width,
                height=config.video_height,
                fps=config.video_fps,
                transparent=False,
                env_info=env_info,
                overlays=image_overlays,
                image_scale=config.video_image_scale,
                image_position=config.video_image_position,
            )
            state.complete_step("mp4 生成が完了しました。", step_start)

        if config.output_mov:
            target = out_dir / f"{state.output_base_name}.mov"
            reporter.log(f"mov を生成しています -> {target}")
            step_start = reporter.now()
            video.render_video_with_subtitles(
                audio=audio_for_video,
                subtitles=ass_path,
                output=target,
                ffmpeg_path=config.ffmpeg_path,
                width=config.video_width,
                height=config.video_height,
                fps=config.video_fps,
                transparent=True,
                env_info=env_info,
                overlays=image_overlays,
                image_scale=config.video_image_scale,
                image_position=config.video_image_position,
            )
            state.complete_step("mov 生成が完了しました。", step_start)

    if not config.keep_work:
        reporter.log("作業ディレクトリをクリーンアップしています...")
        step_start = reporter.now()
        shutil.rmtree(run_dir, ignore_errors=True)
        state.complete_step("作業ディレクトリを削除しました。", step_start)
    else:
        reporter.log(f"作業ディレクトリを保持します -> {run_dir}")



_PIPELINE_STEPS: list[Callable[[PipelineState], None]] = [
    _step_prepare_input,
    _step_transcribe,
    _step_diarize,
    _step_stylize,
    _step_synthesize,
    _step_concatenate,
    _step_finalize,
]


def run_pipeline(config: PipelineConfig) -> None:
    reporter = _ProgressReporter()
    work_dir, out_dir = _ensure_paths()
    resume = config.resume_run_id is not None
    run_id = config.resume_run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = work_dir / run_id

    if config.llm_backend == "local":
        if config.llm_host is not None or config.llm_port is not None:
            host = config.llm_host or "127.0.0.1"
            port = config.llm_port or 11434
            os.environ["LOCAL_LLM_BASE_URL"] = f"http://{host}:{port}/v1"

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
            raise FileNotFoundError(f"指定された run_id が見つかりませんでした: {run_id}")
        reporter.log(f"run_id {run_id} で再開します。")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    _write_cli_args_snapshot(run_dir, config.cli_args)

    env_pyannote_token = os.environ.get("PYANNOTE_TOKEN")
    hf_token, _ = resolve_hf_token(env_pyannote_token)
    env_info = EnvironmentInfo.collect(
        config.ffmpeg_path, env_pyannote_token, config.llm_host, config.llm_port
    )

    if config.llm_backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            append_env_details(
                "OPENAI_API_KEY が未設定のため LLM バックエンド 'openai' を利用できません。環境変数を設定してから再実行してください。",
                env_info,
            )
        )

    state = PipelineState(
        config=config,
        reporter=reporter,
        run_id=run_id,
        run_dir=run_dir,
        out_dir=out_dir,
        env_info=env_info,
        hf_token=hf_token,
    )
    diarization_path = run_dir / "diarization.json"
    need_new_diarization = config.speakers != "1" and not (resume and diarization_path.exists())
    if need_new_diarization and hf_token is None:
        raise RuntimeError(
            append_env_details(
                "pyannote の話者分離には Hugging Face Token が必要ですが、HF_TOKEN/PYANNOTE_TOKEN のいずれも指定されていません。\n"
                f"{diarize._PYANNOTE_ACCESS_STEPS}\n{diarize._PYANNOTE_TOKEN_USAGE}",
                state.env_info,
            )
        )

    state.total_steps = len(_PIPELINE_STEPS)

    _run_pipeline_steps(state)


def _run_pipeline_steps(state: PipelineState) -> None:
    """パイプラインの各ステップを実行する。"""

    start_index = 0
    resume_from = state.config.resume_from
    step_names = [step.__name__.replace("_step_", "") for step in _PIPELINE_STEPS]

    if resume_from:
        if resume_from.isdigit():
            num = int(resume_from) - 1
            if 0 <= num < len(step_names):
                start_index = num
            else:
                state.reporter.log(f"警告: 不正なステップ番号 '{resume_from}' が指定されました。最初から実行します。")
        else:
            try:
                start_index = step_names.index(resume_from)
            except ValueError:
                state.reporter.log(f"警告: 不正なステップ名 '{resume_from}' が指定されました。最初から実行します。")

    for i in range(start_index, len(_PIPELINE_STEPS)):
        step_func = _PIPELINE_STEPS[i]
        step_name = step_names[i]
        step_start_time = state.reporter.now()
        try:
            step_func(state)
            duration = state.reporter.now() - step_start_time
            state.reporter.record_step(step_name, duration)
        except SystemExit:
            duration = state.reporter.now() - step_start_time
            state.reporter.record_step(step_name, duration)
            state.reporter.summarize()
            return
        except Exception:
            duration = state.reporter.now() - step_start_time
            state.reporter.record_step(step_name, duration)
            state.reporter.summarize()
            raise

    state.reporter.log("すべての処理が完了しました。")
    state.reporter.summarize()


def _decide_zunda_jobs(
    *,
    transcript_text: str | None,
    config: PipelineConfig,
    env_info: EnvironmentInfo | None,
    reporter: _ProgressReporter,
    prompt_log_dir: pathlib.Path | None = None,
) -> tuple[str, str] | None:
    if config.llm_backend == "none":
        reporter.log("LLM バックエンドが指定されていないため、ずんだもんの職業決定をスキップします (--llm で設定可能)。")
        return None

    context = (transcript_text or "").strip()
    if len(context) > 800:
        context = context[:800] + "…"

    system_prompt = (
        "あなたは、音声レポートのテーマに合う職業を提案する日本語アシスタントです。\n"
        "出力は JSON 配列で、要素は必ず 4 つにしてください。\n"
        "各要素はオブジェクトとし、キーは 'zunda_senior_job' と 'zunda_junior_job' のみを含めてください。\n"
        "説明文や余計な文字は入れず、必ず日本語で短い職業名だけを返してください。"
    )

    user_prompt = (
        "以下の内容に関連する、ずんだもんの職業設定を考えてください。\n"
        "- zunda_senior_job: レポート全体の話題に関連し、憧れの対象になりそうな華やかな職業。\n"
        "- zunda_junior_job: 上記と対比になる、現実がちょっぴり厳しそうでコミカルな仕事。\n"
        "短く親しみやすい言い回しで、聞いてクスッとできる組み合わせを 4 通り提案してください。\n"
        "会話内容の概要:\n"
        f"{context}"
    )

    prompt_log_path: pathlib.Path | None = None
    log_response: Callable[[str | Exception], None] | None = None
    if prompt_log_dir is not None:
        prompt_log_path = prompt_log_dir / "prompt_zunda_jobs.txt"
        prompt_log_path.write_text("", encoding="utf-8")
        prompt_logger = _create_prompt_logger(prompt_log_path, "Zunda Jobs Prompt")
        log_response = prompt_logger(system_prompt, user_prompt)

    content = ""
    try:
        content = chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
            env_info=env_info,
        )
        if log_response:
            log_response(content)
        payload = _extract_json_payload(content)
        data = json.loads(payload)
        raw_candidates: list[dict[str, str]] = []
        if isinstance(data, list):
            raw_candidates = data
        elif isinstance(data, dict):
            if isinstance(data.get("candidates"), list):
                raw_candidates = data.get("candidates", [])
            else:
                raw_candidates = [data]

        candidates: list[tuple[str, str]] = []
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            senior_job = str(item.get("zunda_senior_job", "")).strip()
            junior_job = str(item.get("zunda_junior_job", "")).strip()
            if senior_job and junior_job:
                candidates.append((senior_job, junior_job))
    except json.JSONDecodeError as exc:
        if log_response:
            log_response(exc)
        reporter.log(f"ずんだもんの職業応答を JSON として解釈できませんでした: {exc}")
        preview = (content or "").strip()
        if preview:
            preview = preview[:200].replace("\n", " ")
            reporter.log(f"LLM 応答の先頭部分: {preview}")
        raise RuntimeError("LLM 応答を解析できませんでした") from exc
    except Exception as exc:
        if log_response:
            log_response(exc)
        reporter.log(f"ずんだもんの職業生成に失敗したため、省略します: {exc}")
        raise

    if not candidates:
        reporter.log("LLM 応答に必要な職業情報が含まれていなかったため、省略します。")
        raise RuntimeError("LLM 応答に職業情報が含まれていません")

    selected_senior, selected_junior = random.choice(candidates)
    config.zunda_senior_job = selected_senior
    config.zunda_junior_job = selected_junior
    if prompt_log_path is not None:
        reporter.log(f"ずんだもん職業プロンプトを {prompt_log_path.name} に保存しました。")
    reporter.log(
        "LLM がずんだもんの職業候補を生成しました: "
        + " | ".join(f"憧れ={s} / 現在={j}" for s, j in candidates)
    )
    reporter.log(
        f"候補からランダムに選択しました: 憧れ={selected_senior} / 現在={selected_junior}"
    )
    return selected_senior, selected_junior


def _prepend_introductions(
    segments: Sequence[style_convert.StylizedSegment],
    char1: characters.CharacterMeta,
    char2: characters.CharacterMeta,
    config: PipelineConfig,
    *,
    env_info: EnvironmentInfo | None = None,
    transcript_text: str | None = None,
    reporter: _ProgressReporter | None = None,
    run_dir: pathlib.Path | None = None,
) -> list[style_convert.StylizedSegment]:
    if not config.prepend_intro:
        return list(segments)

    intro_segments: list[style_convert.StylizedSegment] = []
    zunda_intro_segment: style_convert.StylizedSegment | None = None

    def resolve_zunda_intro_text() -> str | None:
        def build_intro(senior_job: str, junior_job: str) -> str:
            return f"僕の名前はずんだもん、{senior_job}にあこがれる{junior_job}なのだ"

        if config.zunda_senior_job and config.zunda_junior_job:
            return build_intro(config.zunda_senior_job, config.zunda_junior_job)

        senior_job = config.zunda_senior_job
        junior_job = config.zunda_junior_job

        if reporter is not None and (senior_job is None or junior_job is None):
            if config.llm_backend == "none":
                reporter.log("職業が未指定ですが LLM が無効のため、職業トークを省略します。")
                return None

            decided = _decide_zunda_jobs(
                transcript_text=transcript_text,
                config=config,
                env_info=env_info,
                reporter=reporter,
                prompt_log_dir=run_dir,
            )
            if decided:
                senior_job, junior_job = decided
            else:
                return None

        if senior_job is None or junior_job is None:
            return None

        return build_intro(senior_job, junior_job)

    # 音声に登場するキャラクターとスピーカーラベルのマッピングを作成
    speaker_map: dict[str, str] = {}
    if segments:
        for seg in segments:
            if seg.character and seg.character not in speaker_map:
                speaker_map[seg.character] = seg.speaker
            if len(speaker_map) == 2:
                break
        # マッピングが空の場合（単一話者などでキャラクタ未割り当て）、char1に最初の話者ラベルを割り当てる
        if not speaker_map:
            speaker_map[char1.id] = segments[0].speaker

    # 話者1の挨拶を処理
    intro1_text = config.intro1
    # --intro1 がなく、かつ条件を満たす場合にずんだもんの職業挨拶を生成
    if intro1_text is None and char1.id == "zundamon":
        intro1_text = resolve_zunda_intro_text()

    if intro1_text and char1.id in speaker_map:
        target_segment = style_convert.StylizedSegment(
            start=0.0,
            end=0.0,
            text=intro1_text,
            speaker=speaker_map[char1.id],
            character=char1.id,
        )
        if char1.id == "zundamon" and config.intro1 is None:
            zunda_intro_segment = target_segment
        else:
            intro_segments.append(target_segment)

    # 話者2がずんだもんの場合、職業挨拶を生成して最優先で挿入する
    zunda_intro_text: str | None = None
    if char2.id == "zundamon" and config.intro2 is None:
        zunda_intro_text = resolve_zunda_intro_text()
    if zunda_intro_segment is None and zunda_intro_text and char2.id in speaker_map:
        zunda_intro_segment = style_convert.StylizedSegment(
            start=0.0,
            end=0.0,
            text=zunda_intro_text,
            speaker=speaker_map[char2.id],
            character=char2.id,
        )

    # 話者2の挨拶を処理
    if config.intro2 and char2.id in speaker_map:
        intro_segments.append(
            style_convert.StylizedSegment(
                start=0.0,
                end=0.0,
                text=config.intro2,
                speaker=speaker_map[char2.id],
                character=char2.id,
            )
        )

    ordered_segments: list[style_convert.StylizedSegment] = []
    if zunda_intro_segment is not None:
        ordered_segments.append(zunda_intro_segment)

    ordered_segments.extend(intro_segments)
    ordered_segments.extend(segments)
    return ordered_segments


def _insert_line_breaks(
    segments: Sequence[style_convert.StylizedSegment],
    *,
    config: PipelineConfig,
    prompt_logger: style_convert.LLMPromptLogger | None = None,
) -> list[style_convert.StylizedSegment]:
    if not config.linebreak_with_llm or config.llm_backend == "none":
        return list(segments)

    long_segments: list[tuple[int, style_convert.StylizedSegment]] = [
        (idx, seg) for idx, seg in enumerate(segments) if len(seg.text) > config.linebreak_min_chars
    ]

    if not long_segments:
        return list(segments)

    adjusted: list[style_convert.StylizedSegment] = []

    system_prompt = (
        "あなたは日本語のセリフを整形するアシスタントです。\n"
        "以下の規則に従ってください:\n"
        "- 語句や順序を一切変更せず、必要なら改行(\\n)を1〜2回だけ挿入するだけにしてください。\n"
        "- 改行は必ず実際の改行文字のみを使い、\\N や \\\\N のような文字列は絶対に使わないでください。\n"
        "- 整形後のセリフのみを返し、説明や番号付け、余計な記号は入れないでください。\n"
        "- 改行が不要なら元のセリフをそのまま返してください。\n"
        "- 各セグメントは <SEG id=...> と </SEG> で囲み、id は入力と同じものを使ってください。"
    )
    lines: list[str] = [
        f"次のセリフが{config.linebreak_min_chars}文字を超えて長い場合に、句読点や自然な区切りで最小限の改行を入れてください。",
        "1つのセグメントは2行程度までにとどめ、不要な連続改行は入れないでください。",
        "読みやすさが目的なので、言い回しや句読点は変えないでください。",
        "入力と同じ順序で、各セグメントを <SEG id=...> で囲んで返してください。",
    ]

    for idx, seg in long_segments:
        lines.append(f"<SEG id={idx}>")
        lines.append(seg.text)
        lines.append("</SEG>")

    user_prompt = "\n".join(lines)

    response_logger = prompt_logger(system_prompt, user_prompt) if prompt_logger else None

    response_map: dict[int, str] = {}

    try:
        content = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt, config=config)
        if response_logger:
            response_logger(content)
    except Exception as exc:
        if response_logger:
            response_logger(exc)
        content = ""

    if content and re.search(r"\\{1,2}N", content):
        raise ValueError("LLM 応答に許可されていないエスケープ表現 (\\N など) が含まれています。")

    if content:
        for match in re.finditer(r"<SEG id=(\d+)>\s*(.*?)\s*</SEG>", content, re.DOTALL):
            try:
                seg_id = int(match.group(1))
            except ValueError:
                continue
            text = match.group(2).strip()
            if text:
                response_map[seg_id] = text

    for idx, seg in enumerate(segments):
        new_text = response_map.get(idx, seg.text)
        lines = [line.strip() for line in new_text.splitlines()]
        lines = [line for line in lines if line]

        if not lines:
            lines = [seg.text]

        if len(lines) == 1:
            adjusted.append(
                style_convert.StylizedSegment(
                    start=seg.start,
                    end=seg.end,
                    text=lines[0],
                    speaker=seg.speaker,
                    character=seg.character,
                )
            )
            continue

        duration = seg.end - seg.start
        if duration > 0:
            step = duration / len(lines)
        else:
            step = 0.0

        for i, line in enumerate(lines):
            start = seg.start + step * i
            end = seg.start + step * (i + 1) if i < len(lines) - 1 else seg.end
            adjusted.append(
                style_convert.StylizedSegment(
                    start=start,
                    end=end,
                    text=line,
                    speaker=seg.speaker,
                    character=seg.character,
                )
            )

    return adjusted
