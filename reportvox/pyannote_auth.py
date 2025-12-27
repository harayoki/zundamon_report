"""pyannote/speaker-diarization への認証だけを確認するユーティリティ。"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from . import diarize
from .envinfo import EnvironmentInfo, append_env_details, resolve_hf_token


def authenticate_only(hf_token: Optional[str], ffmpeg_path: str) -> None:
    """pyannote/speaker-diarization の認証を試行し、結果を標準出力へ表示する。"""
    env_token = os.environ.get("PYANNOTE_TOKEN")
    token, _ = resolve_hf_token(hf_token, env_token)
    env_info = EnvironmentInfo.collect(ffmpeg_path, hf_token, env_token)

    if token is None:
        raise RuntimeError(
            append_env_details(
                "pyannote/speaker-diarization への認証には Hugging Face Token が必要です。\n"
                "環境変数 HF_TOKEN/PYANNOTE_TOKEN か --hf-token で指定してください。",
                env_info,
            )
        )

    try:
        from pyannote.audio import Pipeline as PyannotePipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            append_env_details(
                "pyannote.audio をインポートできませんでした。インストールと依存モジュールを確認してください。",
                env_info,
            )
        ) from exc

    diarize._configure_hf_auth(token)
    kwargs = diarize._build_pyannote_kwargs(PyannotePipeline, token)
    # kwargs["revision"] = "2.1"
    try:
        PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", **kwargs)
    except Exception as exc:  # pragma: no cover - network/auth errors
        raise RuntimeError(
            append_env_details(
                "pyannote/speaker-diarization への認証に失敗しました。\n"
                "HuggingFace で以下 4 つのページすべてにアクセスし、'Agree and access repository' をクリックして承諾済みか確認してください。\n"
                "# 1. https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "# 2. https://huggingface.co/pyannote/segmentation-3.0\n"
                "# 3. https://huggingface.co/pyannote/segmentation\n"
                "# 4. https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                "\n"
                "また、Token は 'Fine-grained' ではなく 'Classic (Read)' で作成したものを使用することを強く推奨します。\n"
                "Token 設定URL: https://huggingface.co/settings/tokens\n",
                env_info,
            )
        ) from exc

    print("pyannote/speaker-diarization への認証が成功しました。")
    print(env_info.format())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reportvox.pyannote_auth",
        description="環境変数 PYANNOTE_TOKEN などを用いて pyannote/speaker-diarization への認証だけを試行します。",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="pyannote.audio 用の Hugging Face Token（環境変数 PYANNOTE_TOKEN も参照）。",
    )
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="環境情報表示用に ffmpeg のパスを指定できます。")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    authenticate_only(args.hf_token, args.ffmpeg_path)


if __name__ == "__main__":
    main()
