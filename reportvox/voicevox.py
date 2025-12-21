"""VOICEVOX で音声合成するためのクライアント。"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Callable, Dict, Iterable, Sequence

import requests

from .characters import CharacterMeta
from .envinfo import EnvironmentInfo, append_env_details
from .style_convert import StylizedSegment


class VoiceVoxError(RuntimeError):
    """VOICEVOX API 呼び出しに失敗した際の例外。"""


def _request_audio_query(text: str, speaker_id: int, base_url: str, env_info: EnvironmentInfo | None) -> dict:
    resp = requests.post(f"{base_url}/audio_query", params={"text": text, "speaker": speaker_id})
    if resp.status_code != 200:
        raise VoiceVoxError(append_env_details(f"VOICEVOX audio_query に失敗しました: {resp.status_code} {resp.text}", env_info))
    return resp.json()


def _request_synthesis(query: dict, speaker_id: int, base_url: str, env_info: EnvironmentInfo | None) -> bytes:
    headers = {"Content-Type": "application/json"}
    resp = requests.post(f"{base_url}/synthesis", params={"speaker": speaker_id}, data=json.dumps(query), headers=headers)
    if resp.status_code != 200:
        raise VoiceVoxError(append_env_details(f"VOICEVOX synthesis に失敗しました: {resp.status_code} {resp.text}", env_info))
    return resp.content


def synthesize_segments(
    segments: Sequence[StylizedSegment],
    characters: Dict[str, CharacterMeta],
    base_url: str,
    run_dir: pathlib.Path,
    skip_existing: bool = False,
    progress: Callable[[int, int, float], None] | None = None,
    *,
    env_info: EnvironmentInfo | None = None,
) -> list[pathlib.Path]:
    outputs: list[pathlib.Path] = []
    for idx, seg in enumerate(segments):
        meta = characters.get(seg.character)
        if meta is None:
            raise VoiceVoxError(append_env_details(f"{seg.character} のキャラクターメタ情報が見つかりません。", env_info))
        if meta.voicevox_speaker_id is None:
            raise VoiceVoxError(
                append_env_details(f"キャラクター {meta.id} に VOICEVOX の speaker_id が設定されていません。", env_info)
            )
        out_path = run_dir / f"seg_{idx:04d}.wav"
        if skip_existing and out_path.exists():
            if progress is not None:
                progress(idx + 1, len(segments), 0.0)
            outputs.append(out_path)
            continue
        seg_start = time.monotonic()
        query = _request_audio_query(seg.text, meta.voicevox_speaker_id, base_url, env_info)
        audio = _request_synthesis(query, meta.voicevox_speaker_id, base_url, env_info)
        out_path.write_bytes(audio)
        if progress is not None:
            progress(idx + 1, len(segments), time.monotonic() - seg_start)
        outputs.append(out_path)
    return outputs
