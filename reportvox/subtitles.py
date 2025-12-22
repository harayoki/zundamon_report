"""字幕データの生成ユーティリティ。"""

from __future__ import annotations

import pathlib
import wave
from typing import Dict, List, Literal, Sequence

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
    if not include_label:
        return segment.text
    meta = characters.get(segment.character)
    label = meta.display_name if meta else segment.character
    if label:
        return f"【{label}】{segment.text}"
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
            split_at = limit
        chunk_body = candidate[: split_at + 1].strip()
        remaining = remaining[split_at + 1 :].lstrip()
        if is_first and prefix:
            chunk_body = f"{prefix}{chunk_body}"
        chunks.append(chunk_body)
        is_first = False
    return chunks


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
