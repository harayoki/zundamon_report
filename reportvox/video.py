"""動画生成関連のユーティリティ。"""

from __future__ import annotations

import pathlib
import subprocess

from .envinfo import EnvironmentInfo, append_env_details


def _escape_filter_path(path: pathlib.Path) -> str:
    escaped = path.as_posix().replace("\\", r"\\").replace(":", r"\\:").replace(",", r"\\,")
    return escaped.replace("'", r"\'")


def render_video_with_subtitles(
    *,
    audio: pathlib.Path,
    subtitles: pathlib.Path,
    output: pathlib.Path,
    ffmpeg_path: str = "ffmpeg",
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
    transparent: bool = False,
    env_info: EnvironmentInfo | None = None,
) -> None:
    """音声と字幕を組み合わせて動画を生成する。"""

    if not audio.exists():
        raise FileNotFoundError(append_env_details(f"音声ファイルが見つかりません: {audio}", env_info))
    if not subtitles.exists():
        raise FileNotFoundError(append_env_details(f"字幕ファイルが見つかりません: {subtitles}", env_info))

    base_color = "black@0" if transparent else "black"
    subtitle_filter = _escape_filter_path(subtitles)
    filter_complex = f"subtitles={subtitle_filter}"

    cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={base_color}:s={width}x{height}:r={fps}",
        "-i",
        str(audio),
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",
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

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(append_env_details("ffmpeg による動画生成に失敗しました。", env_info)) from exc
