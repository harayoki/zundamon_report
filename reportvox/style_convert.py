"""口調変換と定型句挿入のロジック。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .diarize import AlignedSegment
from .characters import CharacterMeta
from .config import LLMBackend # <-- 追加
from .llm_client import chat_completion # <-- 修正


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


def _llm_transform(text: str, meta: CharacterMeta, backend: LLMBackend) -> list[str]:
    if backend == "none":
        return [text]

    system_prompt = (
        "あなたは、与えられたキャラクターになりきって文章のスタイルを変換する、プロの日本語文体アシスタントです。\n"
        "以下のルールを厳密に守ってください:\n"
        "- 応答は、変換後の日本語の文章のみを含めてください。\n"
        "- 説明、前置き、言い訳、追加のテキストは一切不要です。\n"
        "- 元の文章の意味、固有名詞、専門用語を絶対に変えないでください。\n"
        "- 英語に翻訳しないでください。\n"
        "- 文章が長い場合（目安として50文字以上）、文脈が自然に区切れる箇所で複数の短い文に分割し、各文を改行して出力してください。\n"
    )
    user_prompt = (
        f"キャラクター「{meta.display_name or meta.id}」の話し方を参考に、以下の文章を自然な話し言葉に変換してください。\n"
        f"口癖: {', '.join(meta.phrases.get('default', []))}\n"
        f"文章: {text}"
    )
    try:
        content = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt, backend=backend)
        # 応答が空行を含む場合があるので、空行は除去する
        return [line for line in content.splitlines() if line.strip()]
    except Exception:
        # LLM が利用できない場合は元のテキストを返して処理継続
        return [text]


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

        # LLM変換とヒューリスティックな口癖挿入を適用
        texts = _llm_transform(seg.text, meta, backend)
        if len(texts) == 1:
             # LLMが分割しなかった場合は、ヒューリスティックな口癖挿入を試みる
            texts = [_heuristic_phrase(seg, meta, inserted)]

        # 分割されたセグメントを生成
        if len(texts) == 1:
            stylized.append(
                StylizedSegment(
                    start=seg.start,
                    end=seg.end,
                    text=texts[0],
                    speaker=seg.speaker,
                    character=seg.character or char1.id,
                )
            )
            continue

        # LLMが文章を分割した場合、タイムスタンプを按分する
        duration = max(0.0, seg.end - seg.start)
        piece = duration / len(texts) if duration else 0.0
        for idx, chunk in enumerate(texts):
            start = seg.start + piece * idx
            end = seg.start + piece * (idx + 1) if duration else seg.end
            if idx == len(texts) - 1:
                end = seg.end  # 最後のセグメントは元の終了時間に合わせる
            stylized.append(
                StylizedSegment(
                    start=start,
                    end=end,
                    text=chunk,
                    speaker=seg.speaker,
                    character=seg.character or char1.id,
                )
            )
    return stylized