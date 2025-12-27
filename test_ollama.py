import os
import sys
import pathlib

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from reportvox.cli import parse_args
from reportvox.llm_client import chat_completion
from reportvox.envinfo import EnvironmentInfo


def run_ollama_test():
    print("Ollamaテストを開始します...")

    config = parse_args(
        [
            "dummy.wav",
            "--llm",
            "ollama",
            "--ollama-host",
            "localhost",
            "--ollama-port",
            "11434",
            "--ollama-model",
            "llama3:latest",
            "--no-style-with-llm",
            "--review-transcript",
        ]
    )

    # EnvironmentInfo を模擬
    env_info = EnvironmentInfo.collect(
        ffmpeg_path=config.ffmpeg_path,
        env_token=os.environ.get("PYANNOTE_TOKEN"),
        llm_host=config.llm_host,
        llm_port=config.llm_port,
    )

    try:
        # llama3:latest モデルを使用
        response = chat_completion(
            system_prompt="あなたは役立つアシスタントです。",
            user_prompt="こんにちは",
            config=config,
            env_info=env_info,
            ollama_options_overwrite={"model": "llama3:latest"},
        )
        print(f"\nOllamaからの応答: {response}")
        print("Ollamaテストが成功しました。")
    except Exception as e:
        print(f"\nOllamaテスト中にエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_ollama_test()
