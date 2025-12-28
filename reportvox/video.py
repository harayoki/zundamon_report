"""動画生成関連のユーティリティ。"""

from __future__ import annotations

import functools
import math
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from PIL import Image

from . import audio as audio_utils
from .envinfo import EnvironmentInfo, append_env_details


def _escape_filter_path(path: pathlib.Path) -> str:
    escaped = path.as_posix().replace("\\", r"\\").replace(":", r"\\:").replace(",", r"\\,")
    return escaped.replace("'", r"\'")


@dataclass(frozen=True)
class OverlayLayout:
    path: pathlib.Path
    start: float
    end: float
    x: int
    y: int
    width: int
    height: int
    scale: float
    source_width: int
    source_height: int


@functools.lru_cache(maxsize=None)
def _load_image_size(path: pathlib.Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def compute_overlay_layouts(
    overlays: Sequence[tuple[pathlib.Path, float, float]],
    *,
    video_width: int,
    video_height: int,
    image_scale: float,
    image_position: tuple[int, int] | None,
    image_size_lookup: Callable[[pathlib.Path], tuple[int, int]] | None = None,
) -> list[OverlayLayout]:
    """画像配置の具体的な位置・サイズを計算する。"""

    if not overlays:
        return []

    size_lookup = image_size_lookup or _load_image_size
    vertical_offset_px = 10
    margin_top_px = 8
    auto_position = image_position is None

    layouts: list[OverlayLayout] = []
    for path, start, end in overlays:
        src_w, src_h = size_lookup(path)
        if auto_position:
            base_height = src_h * image_scale
            target_bottom = video_height * 0.58 + base_height / 2
            target_height = max(base_height, target_bottom - margin_top_px)
            scale_factor = target_height / src_h

            scaled_w = math.ceil(src_w * scale_factor)
            scaled_h = math.ceil(src_h * scale_factor)
            x_pos = (video_width - scaled_w) / 2
            y_pos = target_bottom - scaled_h + vertical_offset_px
        else:
            scale_factor = image_scale
            scaled_w = math.ceil(src_w * scale_factor)
            scaled_h = math.ceil(src_h * scale_factor)
            x_pos, y_pos = image_position

        layouts.append(
            OverlayLayout(
                path=path,
                start=start,
                end=end,
                x=math.floor(x_pos),
                y=math.floor(y_pos),
                width=scaled_w,
                height=scaled_h,
                scale=scale_factor,
                source_width=src_w,
                source_height=src_h,
            )
        )

    return layouts


def build_image_overlays(
    image_paths: Sequence[pathlib.Path],
    *,
    video_duration: float,
    start_times: Sequence[float] | None = None,
    env_info: EnvironmentInfo | None = None,
) -> list[tuple[pathlib.Path, float, float]]:
    """動画尺に合わせた画像の表示区間を返す。

    start_times を指定しない場合は動画の尺を画像数で等分する。
    start_times を指定する場合、各画像の開始秒に加え、次の開始秒または
    動画尺のいずれか早い方を終了秒として扱う。
    """

    if not image_paths:
        return []
    if video_duration <= 0:
        raise ValueError(append_env_details("動画尺が 0 秒以下のため画像を配置できません。", env_info))

    overlays: list[tuple[pathlib.Path, float, float]] = []
    if start_times is not None:
        if len(start_times) != len(image_paths):
            raise ValueError(append_env_details("画像数と開始時刻の数が一致しません。", env_info))
        for idx, (path, start) in enumerate(zip(image_paths, start_times)):
            end = video_duration if idx == len(image_paths) - 1 else start_times[idx + 1]
            if start < 0:
                raise ValueError(append_env_details("画像の開始秒には 0 以上の値を指定してください。", env_info))
            if end < start:
                raise ValueError(append_env_details("画像の表示時間帯が逆転しています。開始秒を見直してください。", env_info))
            overlays.append((path, start, min(video_duration, end)))
        return overlays

    slice_length = video_duration / len(image_paths)
    for idx, path in enumerate(image_paths):
        start = slice_length * idx
        end = video_duration if idx == len(image_paths) - 1 else slice_length * (idx + 1)
        overlays.append((path, start, end))
    return overlays


def render_video_with_subtitles(
    *,
    audio: pathlib.Path,
    subtitles: pathlib.Path,
    output: pathlib.Path,
    ffmpeg_path: str = "ffmpeg",
    width: int = 1080,
    height: int = 1920,
    fps: int = 24,
    transparent: bool = False,
    env_info: EnvironmentInfo | None = None,
    overlays: Iterable[tuple[pathlib.Path, float, float]] | None = None,
    image_scale: float = 0.45,
    image_position: tuple[int, int] | None = None,
) -> None:
    """音声と字幕を組み合わせて動画を生成する。

    overlays で (画像パス, 開始秒, 終了秒) を指定すると、対応する時間帯で画像を重ねる。
    """

    if not audio.exists():
        raise FileNotFoundError(append_env_details(f"音声ファイルが見つかりません: {audio}", env_info))
    if not subtitles.exists():
        raise FileNotFoundError(append_env_details(f"字幕ファイルが見つかりません: {subtitles}", env_info))

    video_duration = audio_utils.read_wav_duration(audio)

    overlay_list = list(overlays or [])
    if overlay_list and image_scale <= 0:
        raise ValueError(append_env_details("画像の拡大率は 0 より大きい値を指定してください。", env_info))
    for path, start, end in overlay_list:
        if not path.exists():
            raise FileNotFoundError(append_env_details(f"画像ファイルが見つかりません: {path}", env_info))
        if end < start:
            raise ValueError(append_env_details("画像の開始秒より終了秒が短くなっています。設定を見直してください。", env_info))

    base_color = "black@0" if transparent else "black"
    x_expr = str(image_position[0]) if image_position else "(W-w)/2"
    y_expr = str(image_position[1]) if image_position else f"H*0.58 - h/2 + 10"

    overlay_layouts = compute_overlay_layouts(
        overlay_list,
        video_width=width,
        video_height=height,
        image_scale=image_scale,
        image_position=image_position,
    )

    filter_parts: list[str] = []
    last_stream = "[0:v]"
    for idx, (path, start, end) in enumerate(overlay_list):
        layout = overlay_layouts[idx] if overlay_layouts else None
        input_label = f"[{idx + 2}:v]"
        scaled_label = input_label
        overlay_x_expr = x_expr
        overlay_y_expr = y_expr
        if layout is not None:
            overlay_x_expr = str(layout.x)
            overlay_y_expr = str(layout.y)
            if layout.width != layout.source_width or layout.height != layout.source_height:
                filter_parts.append(
                    f"{input_label}scale={layout.width}:{layout.height}[img{idx}]"
                )
                scaled_label = f"[img{idx}]"
        elif image_scale != 1.0:
            filter_parts.append(
                f"{input_label}scale=ceil(iw*{image_scale}):ceil(ih*{image_scale})[img{idx}]"
            )
            scaled_label = f"[img{idx}]"
        output_label = "[v_pre_sub]" if idx == len(overlay_list) - 1 else f"[vimg{idx}]"
        start_ts = max(0.0, start)
        end_ts = max(0.0, end)
        filter_parts.append(
            f"{last_stream}{scaled_label}overlay=x={overlay_x_expr}:y={overlay_y_expr}:enable='between(t,{start_ts:.3f},{end_ts:.3f})'"
            f"{output_label}"
        )
        last_stream = output_label

    subtitle_filter = _escape_filter_path(subtitles)
    filter_parts.append(f"{last_stream}subtitles={subtitle_filter}[vout]")

    filter_complex = ";".join(filter_parts)

    cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={base_color}:s={width}x{height}:r={fps}",
        "-i",
        str(audio),
    ]

    for path, *_ in overlay_list:
        cmd += ["-loop", "1", "-i", str(path)]

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "1:a",
    ]

    if transparent:
        cmd += [
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4444",
            "-pix_fmt",
            "yuva444p10le",
            "-c:a",
            "pcm_s16le",
            "-shortest",
        ]
    else:
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
        ]
    cmd += ["-progress", "pipe:1", "-nostats", "-loglevel", "error", str(output)]
    print(" ".join(cmd))

    total_ms = max(1, int(video_duration * 1000))
    last_percent: float | None = None
    printed_progress = False

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key == "out_time_ms":
                try:
                    current_ms = int(value)
                except ValueError:
                    continue
                percent = min(current_ms / total_ms * 100, 100)
                if last_percent is None or percent - last_percent >= 0.5:
                    seconds = current_ms / 1000
                    print(
                        f"[ffmpeg] 動画生成進捗: {percent:5.1f}% ({seconds:.1f}/{video_duration:.1f}s)",
                        end="\r",
                        flush=True,
                    )
                    printed_progress = True
                    last_percent = percent
            elif key == "progress" and value == "end":
                print(
                    f"[ffmpeg] 動画生成進捗: 100.0% ({video_duration:.1f}/{video_duration:.1f}s)",
                    end="\r",
                    flush=True,
                )
                printed_progress = True
    finally:
        stderr_output = process.stderr.read() if process.stderr else ""
        return_code = process.wait()
        if printed_progress:
            print()

    if return_code != 0:
        error_output = stderr_output or ""
        message = f"ffmpeg による動画生成に失敗しました。ffmpegからのエラー:\n{error_output}"
        raise RuntimeError(append_env_details(message, env_info))


# def remux_subtitle_tracks(
#     source: pathlib.Path,
#     *,
#     subtitle: pathlib.Path | None,
#     ffmpeg_path: str = "ffmpeg",
#     env_info: EnvironmentInfo | None = None,
# ) -> None:
#     """動画ファイルから既存の字幕ストリームを削除し、必要に応じて統合字幕を追加する。"""
#
#     if not source.exists():
#         raise FileNotFoundError(append_env_details(f"動画ファイルが見つかりません: {source}", env_info))
#     if subtitle is not None and not subtitle.exists():
#         raise FileNotFoundError(append_env_details(f"字幕ファイルが見つかりません: {subtitle}", env_info))
#
#     # 元の拡張子を保った一時ファイル名にする（例: movie.mp4 -> movie_tmp.mp4）。
#     # .tmp だけを後ろに付けると拡張子が .tmp になり、ffmpeg がフォーマットを判別できない。
#     tmp_output = source.with_name(f"{source.stem}_tmp{source.suffix}")
#
#     cmd: list[str] = [
#         ffmpeg_path,
#         "-y",
#         "-i",
#         str(source),
#     ]
#     print(" ".join(cmd))
#
#     if subtitle is not None:
#         cmd += ["-i", str(subtitle)]
#
#     cmd += [
#         "-map",
#         "0:v",
#         "-map",
#         "0:a?",
#         "-c:v",
#         "copy",
#         "-c:a",
#         "copy",
#     ]
#
#     if subtitle is not None:
#         cmd += [
#             "-map",
#             "1:0",
#             "-c:s",
#             "mov_text",
#         ]
#
#     cmd.append(str(tmp_output))
#
#     try:
#         subprocess.run(
#             cmd,
#             check=True,
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.PIPE,
#             text=True,
#             encoding="utf-8",
#         )
#     except subprocess.CalledProcessError as exc:
#         error_output = exc.stderr or ""
#         message = f"ffmpeg での字幕差し替えに失敗しました。ffmpegからのエラー:\n{error_output}"
#         raise RuntimeError(append_env_details(message, env_info)) from exc
#
#     tmp_output.replace(source)
