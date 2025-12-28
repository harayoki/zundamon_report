"""口調変換と定型句挿入のロジック。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from .characters import CharacterMeta
from .config import PipelineConfig
from .diarize import AlignedSegment
from .llm_client import chat_completion


LLMPromptLogger = Callable[[str, str], Callable[[str | Exception], None] | None]


@dataclass
class StylizedSegment:
    start: float
    end: float
    text: str
    speaker: str
    character: str


def _sanitize_subtitle_text(text: str) -> str:
    """字幕用のテキストを事前に正規化する。"""

    return text.replace("、", " ").replace("。", "")


def _heuristic_phrases(
    segment: AlignedSegment,
    meta: CharacterMeta,
    inserted: set[str],
    *,
    text_override: str | None = None,
) -> list[str]:
    text = text_override if text_override is not None else segment.text
    character_id = segment.character or meta.id
    if character_id in inserted:
        return [text]
    if len(text) < 12:
        return [text]

    ideas = meta.phrases.get("idea_intro", [])
    if not ideas:
        return [text]

    # 簡易ヒューリスティック（自己紹介など）
    phrase = random.choice(ideas)
    inserted.add(character_id)
    return [phrase, text]


def _character_traits(meta: CharacterMeta) -> list[str]:
    traits = []
    if meta.style_first_person:
        traits.append(f"一人称: {meta.style_first_person}")
    if meta.style_endings:
        traits.append(f"語尾: {', '.join(meta.style_endings)}")
    if meta.role:
        traits.append(f"役割: {meta.role}")
    default_phrases = meta.phrases.get("default", [])
    if default_phrases:
        traits.append(f"口癖: {', '.join(default_phrases)}")
    return traits


def _examples_block(meta: CharacterMeta) -> str:
    if not meta.examples:
        return ""

    examples = "\n".join(f"- {example}" for example in meta.examples)
    return f"口調の例:\n{examples}\n"


def _kana_instruction(level: str) -> str:
    level_labels = {
        "elementary": "小学生",
        "junior": "中学生",
        "high": "高校生",
        "college": "大学生",
    }
    if level == "none":
        return ""

    target = level_labels.get(level, level)
    return (
        f"- 読み手は{target}程度の漢字力と想定し、その水準を超える難しい漢字はひらがなまたはカタカナに置き換えてください。\n"
        "- 固有名詞や一般に漢字表記される専門用語は意味が変わらない範囲で残して構いません。\n"
    )


def _character_specific_rules(meta: CharacterMeta) -> str:
    if meta.id == "zundamon":
        return (
            "- 挨拶ではなく、何か考えた結果の意見を述べる文の直前に1回だけ「かしこいぼくはかんがえました。」を挿入してください。\n"
            "- 必要な箇所のみで1回だけ使い、同じセグメント内で繰り返さないでください。\n"
        )
    return ""


def _llm_transform_batch(
    texts: Sequence[str],
    meta: CharacterMeta,
    config: PipelineConfig,
    *,
    prompt_logger: LLMPromptLogger | None = None,
) -> list[list[str]] | None:
    if not config.style_with_llm:
        return None

    if config.llm_backend in {"none", "gemini"}:
        return None

    traits = _character_traits(meta)
    trait_block = "\n".join(traits) if traits else "特徴: なし"
    examples_block = _examples_block(meta)
    kana_rule = _kana_instruction(config.kana_level)

    system_prompt = (
        "あなたは、与えられたキャラクターになりきって文章のスタイルを変換する、プロの日本語文体アシスタントです。\n"
        "以下のルールを厳密に守ってください:\n"
        "- 応答は、入力行を同じ順序・同じ行数で並べた変換後の日本語のみを含めてください。\n"
        "- 各入力行を1行のまま返し、改行で分割・結合しないでください。\n"
        "- 説明、前置き、番号、余計な装飾や記号は不要です。\n"
        "- 元の文章の意味、固有名詞、専門用語を絶対に変えないでください。\n"
        "- 英語に翻訳しないでください。\n"
    )
    system_prompt += _character_specific_rules(meta)
    system_prompt += kana_rule

    user_prompt = (
        f"キャラクター『{meta.display_name or meta.id}』の口調に合わせてください。\n"
        f"{trait_block}\n"
        f"{examples_block}"
        "以下に並ぶ会話文を順番に口調変換し、対応する結果だけを改行区切りで出力してください。"
        "指示や特徴の説明は上記だけで十分なので、出力は会話文だけにしてください。\n"
        "会話文:\n"
        + "\n".join(texts)
    )

    response_logger = prompt_logger(system_prompt, user_prompt) if prompt_logger else None

    try:
        content = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt, config=config)
        if response_logger:
            response_logger(content)
    except Exception as exc:
        if response_logger:
            response_logger(exc)
        return None

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) != len(texts):
        return None

    return [[line] for line in lines]


def _llm_transform(
    text: str,
    meta: CharacterMeta,
    config: PipelineConfig,
    *,
    prompt_logger: LLMPromptLogger | None = None,
) -> list[str]:
    if not config.style_with_llm:
        return [text]

    if config.llm_backend == "none":
        return [text]
    if config.llm_backend == "gemini":
        # Gemini は文字起こし校正専用。口調変換は従来ロジックに任せる。
        return [text]

    examples_block = _examples_block(meta)
    kana_rule = _kana_instruction(config.kana_level)
    system_prompt = (
        "あなたは、与えられたキャラクターになりきって文章のスタイルを変換する、プロの日本語文体アシスタントです。\n"
        "以下のルールを厳密に守ってください:\n"
        "- 応答は、変換後の日本語の文章のみを含めてください。\n"
        "- 1行のまま出力し、改行で分割・結合しないでください。\n"
        "- 説明、前置き、言い訳、追加のテキストは一切不要です。\n"
        "- 元の文章の意味、固有名詞、専門用語を絶対に変えないでください。\n"
        "- 英語に翻訳しないでください。\n"
    )
    system_prompt += _character_specific_rules(meta)
    system_prompt += kana_rule
    user_prompt = (
        f"キャラクター「{meta.display_name or meta.id}」の話し方を参考に、以下の文章を自然な話し言葉に変換してください。\n"
        f"口癖: {', '.join(meta.phrases.get('default', []))}\n"
        f"{examples_block}"
        f"文章: {text}"
    )
    response_logger = prompt_logger(system_prompt, user_prompt) if prompt_logger else None
    try:
        content = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt, config=config)
        if response_logger:
            response_logger(content)
        # 応答が空行を含む場合があるので、空行は除去する
        lines = [line for line in content.splitlines() if line.strip()]
        # LLM が改行を挿入しても、1つのセグメントとして扱う
        if len(lines) > 1:
            return ["\n".join(lines)]
        return lines or [text]
    except Exception as exc:
        if response_logger:
            response_logger(exc)
        # LLM が利用できない場合は元のテキストを返して処理継続
        return [text]


def apply_style(
    segments: Sequence[AlignedSegment],
    char1: CharacterMeta,
    char2: CharacterMeta,
    config: PipelineConfig,
    *,
    prompt_logger: LLMPromptLogger | None = None,
) -> list[StylizedSegment]:
    stylized: list[StylizedSegment] = []
    inserted: set[str] = set()
    char_map = {char1.id: char1, char2.id: char2}
    llm_results: dict[int, list[str]] = {}
    use_batch_llm = config.style_with_llm and config.llm_backend not in {"none", "gemini"}

    if use_batch_llm:
        for character_id, meta in char_map.items():
            indexes = [idx for idx, seg in enumerate(segments) if (seg.character or char1.id) == character_id]
            if not indexes:
                continue
            texts = [_sanitize_subtitle_text(segments[idx].text) for idx in indexes]
            transformed = _llm_transform_batch(texts, meta, config, prompt_logger=prompt_logger)
            if transformed is None:
                continue
            for idx, lines in zip(indexes, transformed):
                llm_results[idx] = lines

    for idx, seg in enumerate(segments):
        meta = char_map.get(seg.character or char1.id, char1)
        sanitized_text = _sanitize_subtitle_text(seg.text)

        # LLM変換とヒューリスティックな口癖挿入を適用
        if use_batch_llm:
            texts = llm_results.get(idx, [sanitized_text])
        else:
            texts = _llm_transform(sanitized_text, meta, config, prompt_logger=prompt_logger)
        if len(texts) == 1:
            # LLMが分割しなかった場合は、ヒューリスティックな口癖挿入を試みる
            texts = _heuristic_phrases(seg, meta, inserted, text_override=texts[0])

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

