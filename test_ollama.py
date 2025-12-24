import os
import sys
import pathlib

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from reportvox.llm_client import chat_completion
from reportvox.config import PipelineConfig
from reportvox.envinfo import EnvironmentInfo


def run_ollama_test():
    print("Ollamaテストを開始します...")

    # PipelineConfig を模擬
    config = PipelineConfig(
        input_audio=None,
        voicevox_url="http://127.0.0.1:50021",
        speakers="auto",
        speaker1="zundamon",
        speaker2="metan",
        zunda_senior_job=None,
        zunda_junior_job=None,
        want_mp3=False,
        mp3_bitrate="192k",
        ffmpeg_path="ffmpeg",
        keep_work=False,
        output_name=None,
        force_overwrite=False,
        whisper_model="large-v3",
        llm_backend="ollama", # Ollama を指定
        llm_host="localhost", # Ollama ホスト
        llm_port=11434,       # Ollama ポート
        hf_token=None,
        speed_scale=1.1,
        output_duration=None,
        resume_run_id=None,
        resume_from=None,
        subtitle_mode="off",
        subtitle_max_chars=25,
        review_transcript="off",
        style_with_llm=False,
    )

    # EnvironmentInfo を模擬
    env_info = EnvironmentInfo.collect(
        ffmpeg_path=config.ffmpeg_path,
        hf_token_arg=config.hf_token,
        env_token=os.environ.get("PYANNOTE_TOKEN"),
        llm_host=config.llm_host,
        llm_port=config.llm_port
    )

    try:
        # llama3:latest モデルを使用
        response = chat_completion(
            system_prompt="あなたは役立つアシスタントです。",
            user_prompt="こんにちは",
            backend=config.llm_backend,
            env_info=env_info,
            model="llama3:latest"  # 明示的にモデルを指定
        )
        print(f"\nOllamaからの応答: {response}")
        print("Ollamaテストが成功しました。")
    except Exception as e:
        print(f"\nOllamaテスト中にエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_ollama_test()
