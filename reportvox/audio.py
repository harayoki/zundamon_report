"""Audio utilities."""

from __future__ import annotations

import pathlib
import subprocess
import wave
from typing import Sequence


def ensure_ffmpeg(ffmpeg_path: str = "ffmpeg") -> None:
    """Ensure ffmpeg is available; raise RuntimeError if not."""
    ffmpeg_path_obj = pathlib.Path(ffmpeg_path)
    if ffmpeg_path_obj.exists() and ffmpeg_path_obj.is_dir():
        raise RuntimeError(
            f"ffmpeg必須です。README参照。指定されたパスはディレクトリです (指定: {ffmpeg_path!r})。ffmpegの実行ファイルを指定してください。"
        )
    try:
        subprocess.run(
            [ffmpeg_path, "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg必須です。README参照。ffmpeg が見つかりません (指定: {ffmpeg_path!r})。"
        ) from exc
    except PermissionError as exc:
        raise RuntimeError(
            f"ffmpeg必須です。README参照。指定されたパスに実行権限がないかアクセスが拒否されました (指定: {ffmpeg_path!r})。管理者権限やパスを確認してください。"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg必須です。README参照。ffmpeg の実行に失敗しました (指定: {ffmpeg_path!r})。"
        ) from exc


def normalize_to_wav(src: pathlib.Path, dest: pathlib.Path, *, ffmpeg_path: str = "ffmpeg") -> pathlib.Path:
    """Convert input audio to wav using ffmpeg."""
    cmd = [ffmpeg_path, "-y", "-i", str(src), str(dest)]
    print("[reportvox] wav正規化開始")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("ffmpegによるwav正規化に失敗しました。入力に音声トラックが見つからないため変換できませんでした。") from exc
    print("[reportvox] wav正規化完了")
    return dest


def _read_params(path: pathlib.Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as wf:
        return (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes())


def join_wavs(inputs: Sequence[pathlib.Path], output: pathlib.Path) -> None:
    if not inputs:
        raise ValueError("No input wavs to join.")

    params = _read_params(inputs[0])
    nchannels, sampwidth, framerate, _ = params

    with wave.open(str(output), "wb") as out_wf:
        out_wf.setnchannels(nchannels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(framerate)
        for path in inputs:
            p = _read_params(path)
            if p[:3] != params[:3]:
                raise ValueError("Input wav parameters do not match; cannot concatenate.")
            with wave.open(str(path), "rb") as in_wf:
                out_wf.writeframes(in_wf.readframes(in_wf.getnframes()))


def convert_to_mp3(src: pathlib.Path, dest: pathlib.Path, bitrate: str = "192k", *, ffmpeg_path: str = "ffmpeg") -> None:
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
        raise RuntimeError("ffmpegによるmp3生成に失敗しました。") from exc
