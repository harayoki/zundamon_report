"""音声処理用のユーティリティ。"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import wave
from typing import Sequence

import numpy as np

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
        process = subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8"
        )
    except subprocess.CalledProcessError as exc:
        error_output = exc.stderr
        error_message = f"ffmpegによるwav正規化に失敗しました。ffmpegからのエラー:\n{error_output}"
        raise RuntimeError(append_env_details(error_message, env_info)) from exc
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


def _validate_wav_params(paths: Sequence[pathlib.Path], *, env_info: EnvironmentInfo | None = None) -> tuple[int, int, int]:
    if not paths:
        raise ValueError(append_env_details("結合対象の WAV がありません。", env_info))

    first = _read_params(paths[0])
    for path in paths[1:]:
        params = _read_params(path)
        if params[:3] != first[:3]:
            raise ValueError(append_env_details("入力 WAV のパラメーターが一致しません。結合できません。", env_info))
    nchannels, sampwidth, framerate, _ = first
    return nchannels, sampwidth, framerate


def _dtype_from_sampwidth(sampwidth: int) -> np.dtype:
    if sampwidth == 1:
        # 8bit PCM (unsigned) を符号付きで扱い、クリッピングでサチらせる
        return np.int8
    if sampwidth == 2:
        return np.dtype("<i2")
    if sampwidth == 4:
        return np.dtype("<i4")
    raise ValueError(f"未対応のサンプル幅です: {sampwidth}")


def _calc_target_starts(target_durations: Sequence[float] | None, inputs: Sequence[pathlib.Path]) -> list[float]:
    starts: list[float] = []
    cursor = 0.0
    if target_durations is None:
        for path in inputs:
            starts.append(cursor)
            cursor += max(0.0, read_wav_duration(path))
    else:
        for duration in target_durations:
            starts.append(cursor)
            cursor += max(0.0, duration)
    return starts


def join_wavs(
    inputs: Sequence[pathlib.Path],
    output: pathlib.Path,
    *,
    target_durations: Sequence[float] | None = None,
    env_info: EnvironmentInfo | None = None,
) -> list[tuple[float, float]]:
    """WAV を指定したタイムラインに沿って結合する。

    target_durations が与えられた場合は各セグメントの開始位置を秒単位で計算し、
    間に無音を挿入したり、前のセグメントと重ね合わせたりしながら結合する。
    戻り値は各セグメントの (start, end) 秒を示すタプルのリスト。
    """

    if target_durations is not None and len(inputs) != len(target_durations):
        raise ValueError(append_env_details("音声ファイル数とターゲット長の数が一致しません。", env_info))

    nchannels, sampwidth, framerate = _validate_wav_params(inputs, env_info=env_info)
    dtype = _dtype_from_sampwidth(sampwidth)
    max_val = np.iinfo(dtype).max
    min_val = np.iinfo(dtype).min

    starts = _calc_target_starts(target_durations, inputs)
    buffer = np.zeros((0, nchannels), dtype=dtype)
    placements: list[tuple[float, float]] = []

    for path, start_sec in zip(inputs, starts):
        with wave.open(str(path), "rb") as in_wf:
            frames = in_wf.getnframes()
            raw = in_wf.readframes(frames)

        audio_data = np.frombuffer(raw, dtype=dtype)
        if nchannels > 1:
            audio_data = audio_data.reshape((-1, nchannels))
        else:
            audio_data = audio_data.reshape((-1, 1))

        # ターゲット尺より実際の音声が長い場合に後続が潰れないよう、
        # 既に書き込んだ末尾より手前には配置しない。
        planned_start_frame = max(0, int(round(start_sec * framerate)))
        start_frame = max(planned_start_frame, buffer.shape[0])
        end_frame = start_frame + audio_data.shape[0]

        if end_frame > buffer.shape[0]:
            pad = np.zeros((end_frame - buffer.shape[0], nchannels), dtype=dtype)
            buffer = np.concatenate([buffer, pad], axis=0)

        existing = buffer[start_frame:end_frame].astype(np.int64)
        mixed = existing + audio_data.astype(np.int64)
        np.clip(mixed, min_val, max_val, out=mixed)
        buffer[start_frame:end_frame] = mixed.astype(dtype)
        placements.append((start_frame / framerate, end_frame / framerate))

    with wave.open(str(output), "wb") as out_wf:
        out_wf.setnchannels(nchannels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(framerate)
        out_wf.writeframes(buffer.reshape(-1).astype(dtype).tobytes())

    return placements


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
