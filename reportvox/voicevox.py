"""VoiceVox synthesis client."""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Iterable, Sequence

import requests

from .characters import CharacterMeta
from .style_convert import StylizedSegment


class VoiceVoxError(RuntimeError):
    """Raised when VoiceVox API calls fail."""


def _request_audio_query(text: str, speaker_id: int, base_url: str) -> dict:
    resp = requests.post(f"{base_url}/audio_query", params={"text": text, "speaker": speaker_id})
    if resp.status_code != 200:
        raise VoiceVoxError(f"VoiceVox audio_query failed: {resp.status_code} {resp.text}")
    return resp.json()


def _request_synthesis(query: dict, speaker_id: int, base_url: str) -> bytes:
    headers = {"Content-Type": "application/json"}
    resp = requests.post(f"{base_url}/synthesis", params={"speaker": speaker_id}, data=json.dumps(query), headers=headers)
    if resp.status_code != 200:
        raise VoiceVoxError(f"VoiceVox synthesis failed: {resp.status_code} {resp.text}")
    return resp.content


def synthesize_segments(
    segments: Sequence[StylizedSegment],
    characters: Dict[str, CharacterMeta],
    base_url: str,
    run_dir: pathlib.Path,
    skip_existing: bool = False,
) -> list[pathlib.Path]:
    outputs: list[pathlib.Path] = []
    for idx, seg in enumerate(segments):
        meta = characters.get(seg.character)
        if meta is None:
            raise VoiceVoxError(f"Character metadata missing for {seg.character}.")
        if meta.voicevox_speaker_id is None:
            raise VoiceVoxError(f"VoiceVox speaker id is missing for character {meta.id}.")
        out_path = run_dir / f"seg_{idx:04d}.wav"
        if skip_existing and out_path.exists():
            outputs.append(out_path)
            continue
        query = _request_audio_query(seg.text, meta.voicevox_speaker_id, base_url)
        audio = _request_synthesis(query, meta.voicevox_speaker_id, base_url)
        out_path.write_bytes(audio)
        outputs.append(out_path)
    return outputs
