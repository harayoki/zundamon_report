"""字幕データの生成ユーティリティ。"""

from __future__ import annotations

import pathlib
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


def _write_srt(segments: Sequence[StylizedSegment], path: pathlib.Path, characters: Dict[str, CharacterMeta], *, include_label: bool) -> None:
    lines: List[str] = []
    for idx, seg in enumerate(segments, 1):
        start = _format_timestamp(seg.start)
        end = _format_timestamp(seg.end)
        text = _build_line_text(seg, characters, include_label=include_label)
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_subtitles(
    segments: Sequence[StylizedSegment],
    *,
    out_dir: pathlib.Path,
    base_stem: str,
    mode: SubtitleMode,
    characters: Dict[str, CharacterMeta],
) -> list[pathlib.Path]:
    """StylizedSegment から字幕ファイルを生成する。"""
    out_dir.mkdir(exist_ok=True)
    subtitle_paths: list[pathlib.Path] = []

    if mode == "off":
        return subtitle_paths

    sorted_segments = sorted(segments, key=lambda s: (s.start, s.end))

    if mode == "all":
        path = out_dir / f"{base_stem}_report.srt"
        _write_srt(sorted_segments, path, characters, include_label=True)
        subtitle_paths.append(path)
        return subtitle_paths

    # mode == "split"
    segments_by_character: Dict[str, list[StylizedSegment]] = {}
    for seg in sorted_segments:
        segments_by_character.setdefault(seg.character, []).append(seg)

    for character_id, char_segments in segments_by_character.items():
        suffix = character_id or "speaker"
        path = out_dir / f"{base_stem}_report_{suffix}.srt"
        _write_srt(char_segments, path, characters, include_label=False)
        subtitle_paths.append(path)

    return subtitle_paths
