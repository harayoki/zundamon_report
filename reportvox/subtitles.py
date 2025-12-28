"""字幕データの生成ユーティリティ。"""

from __future__ import annotations

import pathlib
import wave
from typing import Dict, List, Literal, Sequence, Tuple

from . import utils
from .style_convert import StylizedSegment
from .characters import CharacterMeta

SubtitleMode = Literal["off", "all", "split"]


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    millis_total = int(round(seconds * 1000))
    hours, rem = divmod(millis_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _build_line_text(segment: StylizedSegment, characters: Dict[str, CharacterMeta], *, include_label: bool) -> str:
    """字幕用に文字列を整形する際に先頭ラベルを除去する。"""

    if include_label and segment.text.startswith("【"):
        end_idx = segment.text.find("】")
        if end_idx != -1:
            return segment.text[end_idx + 1 :]

    return segment.text


def _write_srt(segments: Sequence[StylizedSegment], path: pathlib.Path) -> None:
    lines: List[str] = []
    counter = 1
    for seg in segments:
        start = _format_timestamp(seg.start)
        end = _format_timestamp(seg.end)
        lines.append(str(counter))
        lines.append(f"{start} --> {end}")
        lines.append(seg.text)
        lines.append("")
        counter += 1
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _split_text(text: str, include_label: bool, *, max_chars: int | None = None) -> list[str]:
    """テキストを字幕用に分割する。

    include_label が True の場合は先頭のラベルを優先してまとめるため、ラベル部分を分割しない。
    """
    if not max_chars or max_chars <= 0:
        return [text]

    prefix = ""
    content = text
    if include_label and text.startswith("【"):
        end_idx = text.find("】")
        if end_idx != -1:
            prefix = text[: end_idx + 1]
            content = text[end_idx + 1 :]

    content = content.strip()
    if not content or len(content) + len(prefix) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = content
    is_first = True
    while remaining:
        limit = max_chars
        if is_first and prefix:
            limit = max(1, max_chars - len(prefix))
        if len(remaining) + (len(prefix) if is_first else 0) <= max_chars:
            chunk_text = remaining.strip()
            if is_first and prefix:
                chunk_text = f"{prefix}{chunk_text}"
            chunks.append(chunk_text)
            break
        candidate = remaining[:limit]
        split_at = max((candidate.rfind(p) for p in "。！？\n"), default=-1)
        if split_at <= 0:
            split_at = max(1, min(limit - 1, len(candidate) // 2))
        chunk_body = candidate[: split_at + 1].strip()
        remaining = remaining[split_at + 1 :].lstrip()
        if is_first and prefix:
            chunk_body = f"{prefix}{chunk_body}"
        chunks.append(chunk_body)
        is_first = False
    return chunks


def merge_subtitle_segments(segments: Sequence[StylizedSegment]) -> list[StylizedSegment]:
    """複数話者の字幕セグメントを時系列で統合する。"""

    merged = [
        StylizedSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker=seg.speaker,
            character=seg.character,
        )
        for seg in segments
    ]

    return sorted(merged, key=lambda s: (s.start, s.end))


def write_subtitles(
    segments: Sequence[StylizedSegment],
    *,
    out_dir: pathlib.Path,
    base_stem: str,
    mode: SubtitleMode,
    characters: Dict[str, CharacterMeta],
    max_chars_per_line: int | None = None,
) -> list[pathlib.Path]:
    """StylizedSegment から字幕ファイルを生成する。"""
    out_dir.mkdir(exist_ok=True)
    subtitle_paths: list[pathlib.Path] = []

    if mode == "off":
        return subtitle_paths

    include_label = mode == "all"
    sorted_segments = sorted(
        _explode_segments(segments, characters, include_label=include_label, max_chars=max_chars_per_line),
        key=lambda s: (s.start, s.end),
    )

    if mode == "all":
        path = out_dir / f"{base_stem}_report.srt"
        _write_srt(sorted_segments, path)
        subtitle_paths.append(path)
        return subtitle_paths

    # mode == "split"
    segments_by_character: Dict[str, list[StylizedSegment]] = {}
    for seg in sorted_segments:
        segments_by_character.setdefault(seg.character, []).append(seg)

    for character_id, char_segments in segments_by_character.items():
        suffix = character_id or "speaker"
        path = out_dir / f"{base_stem}_report_{suffix}.srt"
        _write_srt(char_segments, path)
        subtitle_paths.append(path)

    return subtitle_paths


def align_segments_to_audio(
    segments: Sequence[StylizedSegment],
    audio_paths: Sequence[pathlib.Path],
    *,
    placements: Sequence[tuple[float, float]] | None = None,
) -> list[StylizedSegment]:
    """VOICEVOX で生成された音声ファイルの長さに合わせてセグメントの時間を再配置する。"""
    if len(segments) != len(audio_paths):
        raise ValueError(f"セグメント数 ({len(segments)}) と音声ファイル数 ({len(audio_paths)}) が一致しません。")

    retimed: list[StylizedSegment] = []

    if placements is not None:
        if len(placements) != len(segments):
            raise ValueError(
                f"セグメント数 ({len(segments)}) と配置情報の数 ({len(placements)}) が一致しません。"
            )
        for segment, placement in zip(segments, placements):
            start, end = placement
            retimed.append(
                StylizedSegment(
                    start=start,
                    end=end,
                    text=segment.text,
                    speaker=segment.speaker,
                    character=segment.character,
                )
            )
        return retimed

    cursor = 0.0
    for segment, path in zip(segments, audio_paths):
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            framerate = wf.getframerate() or 1
            duration = frames / framerate
        start = cursor
        end = start + duration
        cursor = end
        retimed.append(
            StylizedSegment(
                start=start,
                end=end,
                text=segment.text,
                speaker=segment.speaker,
                character=segment.character,
            )
        )
    return retimed


def _explode_segments(
    segments: Sequence[StylizedSegment],
    characters: Dict[str, CharacterMeta],
    *,
    include_label: bool,
    max_chars: int | None,
) -> list[StylizedSegment]:
    if not max_chars or max_chars <= 0:
        return [
            StylizedSegment(
                start=seg.start,
                end=seg.end,
                text=_build_line_text(seg, characters, include_label=include_label),
                speaker=seg.speaker,
                character=seg.character,
            )
            for seg in segments
        ]

    exploded: list[StylizedSegment] = []
    for segment in segments:
        display_text = _build_line_text(segment, characters, include_label=include_label)
        chunks = _split_text(display_text, include_label=include_label, max_chars=max_chars)
        if len(chunks) == 1:
            exploded.append(
                StylizedSegment(
                    start=segment.start,
                    end=segment.end,
                    text=chunks[0],
                    speaker=segment.speaker,
                    character=segment.character,
                )
            )
            continue
        duration = max(0.0, segment.end - segment.start)
        piece = duration / len(chunks) if duration else 0.0
        for idx, chunk in enumerate(chunks):
            start = segment.start + piece * idx
            end = segment.start + piece * (idx + 1) if duration else segment.end
            if idx == len(chunks) - 1:
                end = segment.end
            exploded.append(
                StylizedSegment(
                    start=start,
                    end=end,
                    text=chunk,
                    speaker=segment.speaker,
                    character=segment.character,
                )
            )
    return exploded


def _ass_color_from_hex(color: str, alpha: int = 0) -> str:
    r, g, b = utils.hex_to_rgb(color)
    alpha_clamped = max(0, min(255, alpha))
    return f"&H{alpha_clamped:02X}{b:02X}{g:02X}{r:02X}"


def _format_ass_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_centis = int(round(seconds * 100))
    hours, rem = divmod(total_centis, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, centis = divmod(rem, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape_ass_text(text: str) -> str:
    normalized = text.replace("\\n", "\n").replace("\\N", "\n")
    return (
        normalized.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def _wrap_ass_text_to_two_lines(text: str, *, include_label: bool, max_chars: int | None, is_portrait: bool) -> str:
    """縦長動画向けに ASS 用字幕テキストを 2 行に折り返す。

    max_chars を超える長さのテキストを対象とし、ラベルは先頭行に残したまま
    2 行目に残りをまとめる。max_chars が未指定または 0 以下の場合や、
    縦長動画でない場合は折り返しを行わずそのまま返す。
    """

    if not is_portrait or not max_chars or max_chars <= 0:
        return text

    chunks = _split_text(text, include_label=include_label, max_chars=max_chars)
    if len(chunks) <= 1:
        return text

    first_line = chunks[0].strip()
    second_line = "".join(chunk.strip() for chunk in chunks[1:])
    if not second_line:
        return first_line

    return f"{first_line}\n{second_line}"


def write_ass_subtitles(
    segments: Sequence[StylizedSegment],
    *,
    path: pathlib.Path,
    characters: Dict[str, CharacterMeta],
    colors: Dict[str, str],
    font: str | None,
    font_size: int,
    resolution: Tuple[int, int],
    max_chars_per_line: int | None = None,
) -> pathlib.Path:
    """StylizedSegment から ASS 字幕ファイルを生成する。"""

    play_res_x, play_res_y = resolution
    font_name = font or "Noto Sans JP"
    is_portrait = play_res_y > play_res_x

    exploded = _explode_segments(
        segments,
        characters,
        include_label=True,
        max_chars=(None if is_portrait else max_chars_per_line),
    )

    primary_default = colors.get("default", "#ffffff")
    outline_color = _ass_color_from_hex("#111111", alpha=0x00)  # alphaは低い値ほど濃い
    shadow_color = _ass_color_from_hex("#000000", alpha=0x80)

    script_info = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "",
    ]

    styles = [
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
    ]

    outline_size = 6
    bold = 1

    default_style = _ass_color_from_hex(primary_default)
    styles.append(
        f"Style: default,{font_name},{font_size},{default_style},&H00FFFFFF,"
        f"{outline_color},{shadow_color},{bold},0,0,0,100,100,0,0,1,{outline_size}"
        f",0,2,20,20,40,1"
    )

    for char_id, meta in characters.items():
        primary = _ass_color_from_hex(colors.get(char_id, primary_default))
        style_name = meta.id or char_id or "default"
        styles.append(
            f"Style: {style_name},{font_name},{font_size},{primary},&H00FFFFFF,"
            f"{outline_color},{shadow_color},{bold},0,0,0,100,100,0,0,1,{outline_size}"
            f",0,2,20,20,40,1"
        )

    events = [
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for seg in exploded:
        style_name = seg.character if seg.character in characters else "default"
        wrapped = _wrap_ass_text_to_two_lines(
            seg.text,
            include_label=True,
            max_chars=max_chars_per_line,
            is_portrait=is_portrait,
        )
        text = _escape_ass_text(wrapped)
        events.append(
            "Dialogue: 0," f"{_format_ass_timestamp(seg.start)}," f"{_format_ass_timestamp(seg.end)}," f"{style_name},,0,0,0,,{text}"
        )

    path.write_text("\n".join(script_info + styles + events) + "\n", encoding="utf-8")
    return path
