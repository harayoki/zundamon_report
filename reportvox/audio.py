"""Audio utilities."""

from __future__ import annotations

import os
import pathlib
import subprocess
import wave
from typing import Sequence

from .envinfo import EnvironmentInfo, append_env_details, probe_ffmpeg


def ensure_ffmpeg(ffmpeg_path: str = "ffmpeg", *, env_info: EnvironmentInfo | None = None) -> str:
    """Ensure ffmpeg is available; return the executable path or raise RuntimeError if not."""
    probe = probe_ffmpeg(ffmpeg_path)
    if env_info is not None:
        env_info.update_ffmpeg(probe)
    if not probe.available:
        msg = f"ffmpeg必須です。README参照。ffmpeg が見つかりません (指定: {probe.path!r})。"
        if probe.error == "permission":
            msg = (
                "ffmpeg必須です。README参照。指定されたパスに実行権限がないかアクセスが拒否されました "
                f"(指定: {probe.path!r})。管理者権限やパスを確認してください。"
            )
        elif probe.error == "execution_failed":
            msg = f"ffmpeg必須です。README参照。ffmpeg の実行に失敗しました (指定: {probe.path!r})。"
        raise RuntimeError(append_env_details(msg, env_info))
    return str(probe.path)


def normalize_to_wav(
    src: pathlib.Path,
    dest: pathlib.Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    env_info: EnvironmentInfo | None = None,
) -> pathlib.Path:
    """Convert input audio to wav using ffmpeg."""
    cmd = [ffmpeg_path, "-y", "-i", str(src), str(dest)]
    print("[reportvox] wav正規化開始")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            append_env_details("ffmpegによるwav正規化に失敗しました。入力に音声トラックが見つからないため変換できませんでした。", env_info)
        ) from exc
    print("[reportvox] wav正規化完了")
    return dest


def _read_params(path: pathlib.Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as wf:
        return (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes())


def join_wavs(inputs: Sequence[pathlib.Path], output: pathlib.Path, *, env_info: EnvironmentInfo | None = None) -> None:
    if not inputs:
        raise ValueError(append_env_details("No input wavs to join.", env_info))

    params = _read_params(inputs[0])
    nchannels, sampwidth, framerate, _ = params

    with wave.open(str(output), "wb") as out_wf:
        out_wf.setnchannels(nchannels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(framerate)
        for path in inputs:
            p = _read_params(path)
            if p[:3] != params[:3]:
                raise ValueError(append_env_details("Input wav parameters do not match; cannot concatenate.", env_info))
            with wave.open(str(path), "rb") as in_wf:
                out_wf.writeframes(in_wf.readframes(in_wf.getnframes()))


def convert_to_mp3(
    src: pathlib.Path,
    dest: pathlib.Path,
    bitrate: str = "192k",
    *,
    ffmpeg_path: str = "ffmpeg",
    env_info: EnvironmentInfo | None = None,
) -> None:
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(src),
        "-b:a",
        bitrate,
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(append_env_details("ffmpegによるmp3生成に失敗しました。", env_info)) from exc
