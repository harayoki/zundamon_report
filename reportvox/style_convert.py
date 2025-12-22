"""口調変換と定型句挿入のロジック。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .diarize import AlignedSegment
from .characters import CharacterMeta
from .llm_client import LLMBackend, chat_completion


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
    # 複数回の挿入を避け、先頭か末尾にランダムで挿入する
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
    # 簡易ヒューリスティック（自己紹介など）
    text = _insert_phrase(text, meta.phrases.get("idea_intro", []))
    inserted.add(segment.character or "")
    return text


def _llm_transform(text: str, meta: CharacterMeta, backend: LLMBackend) -> str:
    if backend == "none":
        return text
    system_prompt = (
        "あなたは日本語の文体調整アシスタントです。"
        " 与えられた文章を話し言葉として自然にし、明らかな誤字脱字を直してください。"
        " 文意や固有名詞は変えず、1文で返してください。"
    )
    user_prompt = f"キャラクター: {meta.display_name or meta.id}\n文章: {text}"
    try:
        return chat_completion(system_prompt=system_prompt, user_prompt=user_prompt, backend=backend)
    except Exception:
        # LLM が利用できない場合は元のテキストを返して処理継続
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
