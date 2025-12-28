"""mp3 作成後に mp4 だけを生成するための補助 CLI。"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Sequence

from . import audio, characters, subtitles, utils, video
from .envinfo import EnvironmentInfo, append_env_details
from .pipeline import _load_placements, _load_stylized, _spread_default_colors


def _positive_nonzero_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(f"正の整数を指定してください: {value}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"正の整数を指定してください: {value}")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(f"整数を指定してください: {value}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"0 以上の整数を指定してください: {value}")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(f"数値を指定してください: {value}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"正の数を指定してください: {value}")
    return number


def _non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(f"0 以上の数を指定してください: {value}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"0 以上の数を指定してください: {value}")
    return number


def _xy_pair(value: str) -> tuple[int, int]:
    try:
        x_str, y_str = value.split(",", maxsplit=1)
        x_pos = int(x_str)
        y_pos = int(y_str)
    except Exception as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError("位置は 'X,Y' 形式の整数で指定してください。") from exc
    return (x_pos, y_pos)


def _load_metadata(run_dir: pathlib.Path) -> dict:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _resolve_from_metadata(args_value, meta_value, default=None):
    return args_value if args_value is not None else meta_value if meta_value is not None else default


def _discover_segments(directory: pathlib.Path) -> list[pathlib.Path]:
    segments = sorted(directory.glob("seg_*.wav"))
    if not segments:
        raise FileNotFoundError(f"セグメント音声が見つかりません: {directory}")
    return segments


def _ensure_wav(audio_path: pathlib.Path, work_dir: pathlib.Path, *, ffmpeg_path: str, env_info: EnvironmentInfo) -> pathlib.Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    target = work_dir / "normalized_audio.wav"
    audio.normalize_to_wav(audio_path, target, ffmpeg_path=ffmpeg_path, env_info=env_info)
    return target


def _resolve_colors(char1: characters.CharacterMeta, char2: characters.CharacterMeta, *, color1: str | None, color2: str | None) -> dict[str, str]:
    resolved1 = color1 or char1.main_color
    resolved2 = color2 or char2.main_color

    try:
        resolved1 = utils.normalize_hex_color(resolved1)
        resolved2 = utils.normalize_hex_color(resolved2)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    if color1 is None and color2 is None:
        resolved1, resolved2, _ = _spread_default_colors(resolved1, resolved2)

    return {char1.id: resolved1, char2.id: resolved2}


@dataclass
class VideoArgumentDefaults:
    subtitle_font: str | None = None
    subtitle_font_size: int | None = None
    subtitle_max_chars: int | None = None
    video_width: int | None = None
    video_height: int | None = None
    video_fps: int | None = None
    video_images: list[str] | None = field(default_factory=list)
    video_image_times: list[float] | None = None
    video_image_scale: float | None = None
    video_image_pos: tuple[int, int] | None = None


def add_video_arguments(parser: argparse.ArgumentParser, *, defaults: VideoArgumentDefaults) -> None:
    parser.add_argument("--subtitle-font", default=defaults.subtitle_font, help="動画字幕に使用するフォント名。")
    parser.add_argument(
        "--subtitle-font-size",
        type=_positive_nonzero_int,
        default=defaults.subtitle_font_size,
        help="動画字幕のフォントサイズ (pt)。",
    )
    parser.add_argument(
        "--subtitle-max-chars",
        type=_non_negative_int,
        default=defaults.subtitle_max_chars,
        help="字幕1枚あたりの最大文字数。",
    )
    parser.add_argument("--video-width", type=_positive_nonzero_int, default=defaults.video_width, help="動画出力時の横幅 (ピクセル)。")
    parser.add_argument("--video-height", type=_positive_nonzero_int, default=defaults.video_height, help="動画出力時の縦幅 (ピクセル)。")
    parser.add_argument("--video-fps", type=_positive_nonzero_int, default=defaults.video_fps, help="動画出力時のフレームレート。")
    parser.add_argument(
        "--video-images",
        nargs="+",
        default=defaults.video_images if defaults.video_images is not None else None,
        metavar="PATH",
        help="動画上に重ねる画像ファイルのパス。",
    )
    parser.add_argument(
        "--video-image-times",
        nargs="+",
        type=_non_negative_float,
        default=defaults.video_image_times,
        metavar="SECONDS",
        help="各画像の表示開始秒。--video-images と組み合わせて使用します。",
    )
    parser.add_argument(
        "--video-image-scale",
        type=_positive_float,
        default=defaults.video_image_scale,
        help="動画に重ねる画像の拡大率。",
    )
    parser.add_argument(
        "--video-image-pos",
        type=_xy_pair,
        default=defaults.video_image_pos,
        help="画像の表示位置 (左上基準 'X,Y')。省略時は自動配置。",
    )


def validate_video_image_args(
    parser: argparse.ArgumentParser, video_images: list[str] | None, video_image_times: list[float] | None
) -> None:
    if video_image_times is not None and not video_images:
        parser.error("--video-image-times を使用する場合は --video-images も指定してください。")
    if video_image_times is not None and video_images is not None and len(video_image_times) != len(video_images):
        parser.error("--video-image-times のは --video-images に指定した画像数と一致させてください。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reportvox-mp4",
        description="生成済みの音声と作業ディレクトリを使って字幕付き mp4 を作成します。",
    )
    parser.add_argument("--run-id", default=None, help="work/<run_id> を指定して作業情報を読み込みます。")
    parser.add_argument("--stylized", type=pathlib.Path, default=None, help="stylized.json へのパス。")
    parser.add_argument("--placements", type=pathlib.Path, default=None, help="placements.json へのパス。")
    parser.add_argument("--segments-dir", type=pathlib.Path, default=None, help="seg_*.wav が置かれたディレクトリ。")
    parser.add_argument("--audio", type=pathlib.Path, default=None, help="動画化に使う完成音声 (wav/mp3)。")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="出力する mp4 ファイルのパス。")
    parser.add_argument("--ffmpeg-path", default=None, help="ffmpeg 実行ファイルへのパス。")
    parser.add_argument("--speaker1", default=None, help="主話者に対応するキャラクター ID。")
    parser.add_argument("--speaker2", default=None, help="副話者に対応するキャラクター ID。")
    parser.add_argument("--color1", default=None, help="話者1に使うカラーコード (#RRGGBB)。")
    parser.add_argument("--color2", default=None, help="話者2に使うカラーコード (#RRGGBB)。")
    add_video_arguments(parser, defaults=VideoArgumentDefaults(video_images=None))
    return parser


def _resolve_audio_from_run(metadata: dict, run_dir: pathlib.Path | None) -> pathlib.Path | None:
    base_name = metadata.get("output_base_name")
    if not base_name:
        return None
    candidates = [
        pathlib.Path("out") / f"{base_name}.wav",
        pathlib.Path("out") / f"{base_name}.mp3",
    ]
    if run_dir is not None:
        candidates.append(run_dir / f"{base_name}.wav")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_output_path(args_output: pathlib.Path | None, metadata: dict, audio_path: pathlib.Path | None) -> pathlib.Path:
    if args_output is not None:
        return args_output
    base_name = metadata.get("output_base_name")
    if base_name:
        return pathlib.Path("out") / f"{base_name}.mp4"
    if audio_path is not None:
        return audio_path.with_suffix(".mp4")
    raise ValueError("出力ファイル名を決定できません。--output を指定してください。")


def _prepare_work_dir(run_dir: pathlib.Path | None) -> tuple[pathlib.Path, bool]:
    if run_dir is not None:
        return run_dir, False
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="reportvox_mp4_"))
    return tmp, True


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_dir = pathlib.Path("work") / args.run_id if args.run_id else None
    metadata = _load_metadata(run_dir) if run_dir else {}

    stylized_path = args.stylized or (run_dir / "stylized.json" if run_dir else None)
    placements_path = args.placements or (run_dir / "placements.json" if run_dir else None)
    segments_dir = args.segments_dir or run_dir

    if stylized_path is None:
        parser.error("--stylized か --run-id を指定してください。")
    if placements_path is None:
        parser.error("--placements か --run-id を指定してください。")
    if segments_dir is None:
        parser.error("--segments-dir か --run-id を指定してください。")

    stylized_path = stylized_path.expanduser().resolve()
    placements_path = placements_path.expanduser().resolve()
    segments_dir = segments_dir.expanduser().resolve()

    audio_path = args.audio or _resolve_audio_from_run(metadata, run_dir)
    if audio_path is None:
        parser.error("音声ファイルを特定できませんでした。--audio を指定してください。")
    audio_path = audio_path.expanduser().resolve()

    ffmpeg_path = _resolve_from_metadata(args.ffmpeg_path, metadata.get("ffmpeg_path"), "ffmpeg")

    output_path = _resolve_output_path(args.output, metadata, audio_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    speaker1 = _resolve_from_metadata(args.speaker1, metadata.get("speaker1"))
    speaker2 = _resolve_from_metadata(args.speaker2, metadata.get("speaker2"))

    subtitle_max_chars = _resolve_from_metadata(args.subtitle_max_chars, metadata.get("subtitle_max_chars"), 25)
    subtitle_font = _resolve_from_metadata(args.subtitle_font, metadata.get("subtitle_font"))
    subtitle_font_size = _resolve_from_metadata(args.subtitle_font_size, metadata.get("subtitle_font_size"), 84)

    video_width = _resolve_from_metadata(args.video_width, metadata.get("video_width"), 1080)
    video_height = _resolve_from_metadata(args.video_height, metadata.get("video_height"), 1920)
    video_fps = _resolve_from_metadata(args.video_fps, metadata.get("video_fps"), 24)

    video_images_arg = args.video_images
    video_images = video_images_arg if video_images_arg is not None else metadata.get("video_images")
    video_image_paths = [pathlib.Path(p).expanduser().resolve() for p in video_images] if video_images else []

    video_image_times = args.video_image_times if video_images_arg is not None else metadata.get("video_image_times")
    validate_video_image_args(parser, video_images, video_image_times)

    video_image_scale = _resolve_from_metadata(args.video_image_scale, metadata.get("video_image_scale"), 0.45)
    video_image_pos = args.video_image_pos if args.video_image_pos is not None else metadata.get("video_image_position")

    if not stylized_path.exists():
        parser.error(f"stylized.json が見つかりません: {stylized_path}")
    if not placements_path.exists():
        parser.error(f"placements.json が見つかりません: {placements_path}")
    if not segments_dir.exists():
        parser.error(f"セグメントディレクトリが見つかりません: {segments_dir}")
    if not audio_path.exists():
        parser.error(f"音声ファイルが見つかりません: {audio_path}")

    env_info = EnvironmentInfo.collect(ffmpeg_path, os.environ.get("PYANNOTE_TOKEN"), None, None)

    stylized_segments = _load_stylized(stylized_path)
    placements = _load_placements(placements_path)
    segment_paths = _discover_segments(segments_dir)

    if speaker1 is None or speaker2 is None:
        unique_chars = []
        for seg in stylized_segments:
            if seg.character not in unique_chars:
                unique_chars.append(seg.character)
        if len(unique_chars) < 2:
            parser.error("話者情報を特定できません。--speaker1 と --speaker2 を指定してください。")
        speaker1 = speaker1 or unique_chars[0]
        speaker2 = speaker2 or unique_chars[1]

    char1 = characters.load_character(speaker1)
    char2 = characters.load_character(speaker2)
    colors = _resolve_colors(char1, char2, color1=args.color1 or metadata.get("color1"), color2=args.color2 or metadata.get("color2"))

    aligned_segments = subtitles.align_segments_to_audio(stylized_segments, segment_paths, placements=placements)
    subtitle_segments = subtitles.merge_subtitle_segments(aligned_segments)

    work_dir, should_cleanup = _prepare_work_dir(run_dir)

    try:
        normalized_audio = _ensure_wav(audio_path, work_dir, ffmpeg_path=ffmpeg_path, env_info=env_info)
        video_duration = audio.read_wav_duration(normalized_audio)

        merged_srt_path = work_dir / "video_subtitles.srt"
        subtitles.write_merged_srt_for_video(
            subtitle_segments,
            path=merged_srt_path,
            characters={char1.id: char1, char2.id: char2},
            max_chars_per_line=subtitle_max_chars,
        )

        ass_path = work_dir / "video_subtitles.ass"
        subtitles.write_ass_subtitles(
            subtitle_segments,
            path=ass_path,
            characters={char1.id: char1, char2.id: char2},
            colors=colors,
            font=subtitle_font,
            font_size=subtitle_font_size,
            resolution=(video_width, video_height),
            max_chars_per_line=subtitle_max_chars,
        )

        overlays = []
        if video_image_paths:
            overlays = video.build_image_overlays(
                video_image_paths,
                video_duration=video_duration,
                start_times=video_image_times,
                env_info=env_info,
            )

        video.render_video_with_subtitles(
            audio=normalized_audio,
            subtitles=ass_path,
            output=output_path,
            ffmpeg_path=ffmpeg_path,
            width=video_width,
            height=video_height,
            fps=video_fps,
            transparent=False,
            env_info=env_info,
            overlays=overlays,
            image_scale=video_image_scale,
            image_position=video_image_pos,
        )
        print(f"mp4 を生成しました -> {output_path}")
    except Exception as exc:
        print(append_env_details(f"mp4 生成に失敗しました: {exc}", env_info), file=sys.stderr)
        sys.exit(1)
    finally:
        if should_cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

