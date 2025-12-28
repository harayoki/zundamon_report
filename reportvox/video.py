"""動画生成関連のユーティリティ。"""

from __future__ import annotations

import math
import pathlib
import subprocess
from typing import Iterable, Sequence

from PIL import Image

from .envinfo import EnvironmentInfo, append_env_details


def _escape_filter_path(path: pathlib.Path) -> str:
    escaped = path.as_posix().replace("\\", r"\\").replace(":", r"\\:").replace(",", r"\\,")
    return escaped.replace("'", r"\'")


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
    vertical_offset_px = 10
    y_expr = (
        str(image_position[1])
        if image_position
        else f"H*0.58 - h/2 + {vertical_offset_px}"
    )

    overlay_transforms: list[dict[str, float]] = []
    if overlay_list and image_position is None:
        margin_top_px = 8
        for path, *_ in overlay_list:
            with Image.open(path) as img:
                src_w, src_h = img.size

            base_height = src_h * image_scale
            target_bottom = height * 0.58 + base_height / 2
            target_height = max(base_height, target_bottom - margin_top_px)
            scale_factor = target_height / src_h

            scaled_w = math.ceil(src_w * scale_factor)
            scaled_h = math.ceil(src_h * scale_factor)
            x_pos = (width - scaled_w) / 2
            y_pos = target_bottom - scaled_h + vertical_offset_px

            overlay_transforms.append(
                {
                    "scale_w": scaled_w,
                    "scale_h": scaled_h,
                    "x": x_pos,
                    "y": y_pos,
                }
            )

    filter_parts: list[str] = []
    last_stream = "[0:v]"
    for idx, (path, start, end) in enumerate(overlay_list):
        input_label = f"[{idx + 2}:v]"
        scaled_label = input_label
        overlay_x_expr = x_expr
        overlay_y_expr = y_expr
        if overlay_transforms:
            transform = overlay_transforms[idx]
            scaled_label = f"[img{idx}]"
            overlay_x_expr = str(math.floor(transform["x"]))
            overlay_y_expr = str(math.floor(transform["y"]))
            filter_parts.append(
                f"{input_label}scale={transform['scale_w']}:{transform['scale_h']}[img{idx}]"
            )
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
            str(output),
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
            str(output),
        ]
    print(" ".join(cmd))
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        error_output = exc.stderr or ""
        message = f"ffmpeg による動画生成に失敗しました。ffmpegからのエラー:\n{error_output}"
        raise RuntimeError(append_env_details(message, env_info)) from exc


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
