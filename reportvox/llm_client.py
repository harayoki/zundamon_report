"""LLM 連携の共通クライアント。"""

from __future__ import annotations

import os
from typing import Literal
import pathlib
import subprocess
import tempfile

import ollama
import httpx
import requests

from .config import LLMBackend
from .envinfo import EnvironmentInfo, append_env_details


def chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    config: PipelineConfig,
    env_info: EnvironmentInfo | None = None,
    ollama_options_overwrite: dict | None = None,
    timeout: float = 60.0,
) -> str:
    backend = config.llm_backend
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
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        headers = {"Authorization": f"Bearer {api_key}"}
        # ここに OpenAI の API 呼び出しロジックを後で実装する必要がある
        # 現状は Ollama のロジックのみが実装されているため、プレースホルダーとしてエラーを出す
        raise NotImplementedError("OpenAI バックエンドは現在実装されていません。")

    elif backend == "ollama":
        ollama_host = config.llm_host or os.environ.get("OLLAMA_HOST") or "localhost"
        ollama_port = config.llm_port or int(os.environ.get("OLLAMA_PORT") or 11434)
        
        ollama_base_url = f"http://{ollama_host}:{ollama_port}"

        model_name = config.ollama_model or os.environ.get("LOCAL_LLM_MODEL") or "llama3"
        client = ollama.Client(host=ollama_base_url)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        options: dict = {
            "temperature": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
        }
        if ollama_options_overwrite:
            options.update(ollama_options_overwrite)

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
        except httpx.ConnectError as exc:
            raise RuntimeError(
                append_env_details(
                    f"Ollama サーバーへの接続に失敗しました: {ollama_base_url}\n"
                    "Ollamaが起動しているか、--ollama-host/--ollama-port が正しいか確認してください。",
                    env_info,
                )
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                append_env_details(f"Ollama 呼び出し中に予期せぬエラーが発生しました: {type(exc).__name__}: {exc}", env_info)
            ) from exc
    elif backend == "gemini":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            input_path = tmp_path / "prompt.txt"
            prompt_content = f"[System]\n{system_prompt}\n\n[User]\n{user_prompt}\n"
            input_path.write_text(prompt_content, encoding="utf-8")

            prompt_arg = f"@\"{input_path}\""
            command = ["gemini", "-p", prompt_arg]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    append_env_details("Gemini CLI が見つかりません。gemini コマンドが PATH にあるか確認してください。", env_info)
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    append_env_details("Gemini CLI の実行がタイムアウトしました。", env_info)
                ) from exc
            except Exception as exc:  # pragma: no cover - unexpected subprocess errors
                raise RuntimeError(
                    append_env_details(f"Gemini CLI 実行中に予期せぬエラーが発生しました: {type(exc).__name__}: {exc}", env_info)
                ) from exc

            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                raise RuntimeError(
                    append_env_details(
                        f"Gemini CLI がエラー終了しました (exit {completed.returncode}). stderr: {stderr}", env_info
                    )
                )

            content = completed.stdout.strip()
            if not content:
                raise RuntimeError(append_env_details("Gemini CLI から応答が得られませんでした。", env_info))

            return content
    else:
        raise RuntimeError(f"不明な LLM バックエンドが指定されました: {backend}")
