"""音声処理用のユーティリティ。"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import wave
from typing import Sequence

from .envinfo import EnvironmentInfo, append_env_details, probe_ffmpeg


_SUPPORTED_FFMPEG_MAJOR_VERSIONS = {4, 5, 6, 7}


def _parse_ffmpeg_major_version(version_line: str | None) -> int | None:
    if not version_line:
        return None
    match = re.search(r"ffmpeg version\s+([^\s]+)", version_line)
    if not match:
        return None
    version_token = re.sub(r"^[^\d]*", "", match.group(1))
    major_match = re.match(r"(\d+)", version_token)
    if not major_match:
        return None
    return int(major_match.group(1))


def ensure_ffmpeg(ffmpeg_path: str = "ffmpeg", *, env_info: EnvironmentInfo | None = None) -> str:
    """ffmpeg の存在を確認し、実行可能パスを返す（見つからなければ RuntimeError を送出）。"""
    probe = probe_ffmpeg(ffmpeg_path)
    if env_info is not None:
        env_info.update_ffmpeg(probe)

    if probe.available:
        major_version = _parse_ffmpeg_major_version(probe.version)
        if major_version is not None and major_version not in _SUPPORTED_FFMPEG_MAJOR_VERSIONS:
            supported = "/".join(str(v) for v in sorted(_SUPPORTED_FFMPEG_MAJOR_VERSIONS))
            msg = (
                "ffmpeg は見つかりましたがサポート外のバージョンでした。"
                f" 検出されたバージョン: {probe.version}。サポート対象: {supported} 系。"
            )
            raise RuntimeError(append_env_details(msg, env_info))

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
    """入力音声を ffmpeg で WAV に変換する。"""
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


def read_wav_duration(path: pathlib.Path) -> float:
    """WAV ファイルの長さ（秒）を取得する。"""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        framerate = wf.getframerate() or 1
    return frames / framerate


def join_wavs(inputs: Sequence[pathlib.Path], output: pathlib.Path, *, env_info: EnvironmentInfo | None = None) -> None:
    if not inputs:
        raise ValueError(append_env_details("結合対象の WAV がありません。", env_info))

    params = _read_params(inputs[0])
    nchannels, sampwidth, framerate, _ = params

    with wave.open(str(output), "wb") as out_wf:
        out_wf.setnchannels(nchannels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(framerate)
        for path in inputs:
            p = _read_params(path)
            if p[:3] != params[:3]:
                raise ValueError(append_env_details("入力 WAV のパラメーターが一致しません。結合できません。", env_info))
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


def time_stretch_wav(
    src: pathlib.Path,
    dest: pathlib.Path,
    *,
    target_duration: float,
    ffmpeg_path: str = "ffmpeg",
    env_info: EnvironmentInfo | None = None,
) -> None:
    """ffmpeg の atempo フィルターで WAV を時間伸縮する。"""
    current_duration = read_wav_duration(src)
    if target_duration <= 0 or current_duration <= 0:
        raise ValueError(append_env_details("時間伸縮には正の長さが必要です。", env_info))
    ratio = target_duration / current_duration
    if abs(ratio - 1.0) < 1e-3:
        # ほぼ同じ長さならコピーのみ
        dest.write_bytes(src.read_bytes())
        return

    filters: list[str] = []
    remaining = ratio
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    filter_arg = ",".join(filters)

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(src),
        "-filter:a",
        filter_arg,
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(append_env_details("ffmpegによる時間伸縮に失敗しました。", env_info)) from exc
