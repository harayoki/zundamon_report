"""LLM 連携の共通クライアント。"""

from __future__ import annotations

import os
from typing import Literal

import ollama
import httpx
import requests

from .config import LLMBackend
from .envinfo import EnvironmentInfo, append_env_details


def chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    backend: LLMBackend,
    env_info: EnvironmentInfo | None = None,
    model: str | None = None,
    ollama_options_overwrite: dict | None = None,
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
    elif backend == "ollama":
        ollama_host = os.environ.get("OLLAMA_HOST") or "localhost"
        if env_info and env_info.llm_host:
            ollama_host = env_info.llm_host
        
        ollama_port_str = os.environ.get("OLLAMA_PORT") or "11434"
        if env_info and env_info.llm_port:
            ollama_port_str = str(env_info.llm_port)
        
        ollama_base_url = f"http://{ollama_host}:{ollama_port_str}"
        # print("tags:", requests.get(f"{ollama_base_url}/api/tags").json())

        model_name = model or os.environ.get("OLLAMA_MODEL", "llama3")
        client = ollama.Client(host=ollama_base_url)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        options: dict = {
            "temperature": 0.0,
            "top_p": 1.0,
            # "num_predict": 2048,  # 短すぎると末尾が欠ける
            "repeat_penalty": 1.0,  # 1.0に近いほど繰り返し抑制しない
        }
        if ollama_options_overwrite:
            for k, v in ollama_options_overwrite:
                options[k] = v
        try:
            response = client.chat(model=model_name, messages=messages, options=options)
            content = response.get("message", {}).get("content")
            if not content:
                raise RuntimeError("Ollama 応答に content が含まれていません。")
            return str(content).strip()
        except ollama.ResponseError as exc:
            raise RuntimeError(
                append_env_details(
                    f"Ollama サーバーがエラーステータス {exc.status_code} を返しました。\n"
                    f"モデル名 '{model_name}' が正しいか、Ollamaでダウンロード済みか確認してください。\n"
                    f"サーバーからの詳細: {exc.error}",
                    env_info,
                )
            ) from exc
        except httpx.ConnectError as exc: # ollama-python は内部で httpx を使っているので、そのエラーも捕捉
            raise RuntimeError(
                append_env_details(
                    f"Ollama サーバーへの接続に失敗しました: {ollama_base_url}\n"
                    "Ollamaが起動しているか、--ollama-host/--ollama-port（または環境変数 OLLAMA_HOST/OLLAMA_PORT）が正しいか確認してください。",
                    env_info,
                )
            ) from exc
        except Exception as exc:  # pragma: no cover - ネットワーク依存
            raise RuntimeError(
                append_env_details(f"Ollama 呼び出し中に予期せぬエラーが発生しました: {type(exc).__name__}: {exc}", env_info)
            ) from exc
    else:
        raise RuntimeError(f"不明な LLM バックエンドが指定されました: {backend}")
