"""LLM 連携の共通クライアント。"""

from __future__ import annotations

import os
from typing import Literal

import httpx

from .envinfo import EnvironmentInfo, append_env_details

LLMBackend = Literal["none", "openai", "local"]


def chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    backend: LLMBackend,
    env_info: EnvironmentInfo | None = None,
    model: str | None = None,
    timeout: float = 60.0,
) -> str:
    if backend == "none":
        raise RuntimeError(
            append_env_details("LLM バックエンドが none のため呼び出せません。--llm openai などを指定してください。", env_info)
        )

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                append_env_details("OPENAI_API_KEY が未設定のため LLM を呼び出せません。", env_info)
            )
        base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
        model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        headers = {"Authorization": f"Bearer {api_key}"}
    else:
        base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
        model_name = model or os.environ.get("LOCAL_LLM_MODEL", "gpt-4o-mini")
        headers = {}

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM 応答に choices が含まれていません。")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("LLM 応答に content が含まれていません。")
        return str(content).strip()
    except Exception as exc:  # pragma: no cover - ネットワーク依存
        raise RuntimeError(append_env_details("LLM 呼び出しに失敗しました。", env_info)) from exc
