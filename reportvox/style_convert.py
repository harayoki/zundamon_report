"""Style conversion and phrase insertion."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence

from .diarize import AlignedSegment
from .characters import CharacterMeta

LLMBackend = Literal["none", "openai", "local"]


@dataclass
class StylizedSegment:
    start: float
    end: float
    text: str
    speaker: str
    character: str


def _insert_phrase(text: str, phrases: list[str]) -> str:
    if not phrases or len(text) < 8:
        return text
    # avoid multiple insertions; place at start or end randomly
    phrase = random.choice(phrases)
    if random.random() < 0.5:
        return f"{phrase}。{text}"
    return f"{text}。{phrase}"


def _heuristic_phrase(segment: AlignedSegment, meta: CharacterMeta, inserted: set[str]) -> str:
    text = segment.text
    if segment.character in inserted:
        return text
    if len(text) < 12:
        return text
    # simple heuristics: intro
    text = _insert_phrase(text, meta.phrases.get("idea_intro", []))
    inserted.add(segment.character or "")
    return text


def _llm_transform(text: str, meta: CharacterMeta, backend: LLMBackend) -> str:
    if backend == "none":
        return text
    # Placeholder for pluggable backends
    # For now, simply return text unchanged to avoid external dependencies.
    return text


def apply_style(
    segments: Sequence[AlignedSegment],
    char1: CharacterMeta,
    char2: CharacterMeta,
    backend: LLMBackend = "none",
) -> list[StylizedSegment]:
    stylized: list[StylizedSegment] = []
    inserted: set[str] = set()
    char_map = {char1.id: char1, char2.id: char2}
    for seg in segments:
        meta = char_map.get(seg.character or char1.id, char1)
        text = _llm_transform(seg.text, meta, backend)
        text = _heuristic_phrase(seg, meta, inserted)
        stylized.append(
            StylizedSegment(
                start=seg.start,
                end=seg.end,
                text=text,
                speaker=seg.speaker,
                character=seg.character or char1.id,
            )
        )
    return stylized
