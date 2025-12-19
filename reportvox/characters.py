"""Character metadata loading."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml


CHARACTER_DIR = pathlib.Path(__file__).resolve().parent.parent / "characters"


@dataclass
class CharacterMeta:
    id: str
    display_name: str
    voicevox_speaker_id: Optional[int]
    style_first_person: str
    style_endings: list[str]
    role: str
    phrases: Dict[str, list[str]]
    examples: list[str]


def load_character(char_id: str) -> CharacterMeta:
    base = CHARACTER_DIR / char_id
    meta_path = base / "meta.yaml"
    examples_path = base / "examples.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Character meta not found for id '{char_id}' at {meta_path}")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    examples = []
    if examples_path.exists():
        examples = json.loads(examples_path.read_text(encoding="utf-8"))

    return CharacterMeta(
        id=meta.get("id", char_id),
        display_name=meta.get("display_name", char_id),
        voicevox_speaker_id=_read_voicevox(meta),
        style_first_person=meta.get("style", {}).get("first_person", ""),
        style_endings=meta.get("style", {}).get("endings", []),
        role=meta.get("style", {}).get("role", ""),
        phrases=meta.get("phrases", {}),
        examples=examples,
    )


def _read_voicevox(meta: Dict[str, Any]) -> Optional[int]:
    voicevox_info = meta.get("voicevox", {})
    speaker_id = voicevox_info.get("speaker_id")
    if speaker_id in (None, "", "null"):
        return None
    try:
        return int(speaker_id)
    except Exception:
        return None
