"""ReportVox のコマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Sequence

from reportvox import utils
from reportvox.config import PipelineConfig
from reportvox.envinfo import resolve_hf_token
from reportvox.pipeline import run_pipeline
from reportvox.video_cli import (
    VideoArgumentDefaults,
    _non_negative_float,
    add_video_arguments,
    validate_video_image_args,
)


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(f"数値を指定してください: {value}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"正の数を指定してください: {value}")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(f"整数を指定してください: {value}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"0 以上の整数を指定してください: {value}")
    return number


def _port(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(f"ポート番号を指定してください: {value}") from exc
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError(f"1〜65535 のポート番号を指定してください: {value}")
    return number


def _hex_color(value: str) -> str:
    try:
        return utils.normalize_hex_color(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reportvox",
        description="音声を話者分離・キャラクター口調で整形し、VOICEVOX で読み上げるレポート生成ツール。",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="入力となる音声/動画ファイルのパス（音声トラック必須）。--resume を使わない場合は必須。",
    )
    parser.add_argument(
        "--voicevox-url",
        default="http://127.0.0.1:50021",
        help="VOICEVOX Engine のベースURL（例: http://127.0.0.1:50021）。",
    )
    parser.add_argument(
        "--speakers",
        choices=["auto", "1", "2"],
        default="auto",
        help="話者数の扱い: auto=自動判定 / 1=単一話者として処理 / 2=2人固定（pyannote 認証が必要）。",
    )
    parser.add_argument(
        "--diarization-threshold",
        type=float,
        default=0.8,
        help="話者分離のクラスタリング閾値（0.0-1.0）。値が低いほど別人として分離されやすくなります。",
    )
    parser.add_argument("--speaker1", default="zundamon", help="主話者に割り当てるキャラクターID（characters/ 以下のID）。")
    parser.add_argument("--speaker2", default="metan", help="副話者に割り当てるキャラクターID（characters/ 以下のID）。")
    parser.add_argument(
        "--color1",
        type=_hex_color,
        default=None,
        help="話者1に使うカラーコード（#RRGGBB）。省略時はキャラクターのメインカラーを使用します。",
    )
    parser.add_argument(
        "--color2",
        type=_hex_color,
        default=None,
        help="話者2に使うカラーコード（#RRGGBB）。省略時はキャラクターのメインカラーを使用します。",
    )
    parser.add_argument("--intro1", default=None, help="話者1の最初の挨拶文を上書きします。")
    parser.add_argument("--intro2", default=None, help="話者2の最初の挨拶文を上書きします。")
    parser.add_argument(
        "--no-intro",
        action="store_false",
        dest="prepend_intro",
        default=True,
        help="ずんだもんの自己紹介など、最初の挨拶文の自動挿入を無効化します。",
    )
    parser.add_argument(
        "--zunda-senior-job",
        dest="zunda_senior_job",
        default=None,
        help="ずんだもんが憧れる職業を指定（--zunda-junior-job と併用、省略時はLLMで自動決定）。",
    )
    parser.add_argument(
        "--zunda-junior-job",
        dest="zunda_junior_job",
        default=None,
        help="ずんだもんの現在の役割を指定（--zunda-senior-job と併用、省略時はLLMで自動決定）。",
    )
    parser.add_argument("--mp3", action="store_true", help="mp3 を生成（out/ には mp3 だけを出力）。")
    parser.add_argument("--bitrate", default="192k", help="mp3 出力時のビットレート（--mp3 使用時のみ）。")
    parser.add_argument("--mp4", action="store_true", help="字幕付きの mp4 動画を生成します。")
    parser.add_argument("--mov", action="store_true", help="字幕付きの透明 mov 動画を生成します。")
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="ffmpeg 実行ファイルへのパス（コマンド名またはフルパス）。")
    parser.add_argument(
        "--output-name",
        default=None,
        help="出力ファイル名のベース（拡張子不要）。例: --output-name demo => out/demo.wav|mp3|srt",
    )
    parser.add_argument("-f", "--force", action="store_true", help="既存の出力があっても確認せず上書きする。")
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="文字起こしキャッシュを無視して Whisper による文字起こしをやり直します。",
    )
    parser.add_argument(
        "--force-diarize",
        action="store_true",
        help="話者分離キャッシュを無視して pyannote による解析をやり直します。",
    )
    parser.add_argument("--keep-work", action="store_true", help="work/ 以下の中間ファイルを削除せず残す（開発/再出力向け）。")
    parser.add_argument("--model", default="large-v3", choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3'], help="利用する Whisper モデルサイズ（tiny/base/small/medium/large-v3）。")
    parser.add_argument("--speed-scale", type=_positive_float, default=1.1, help="VOICEVOX の speedScale を指定（デフォルト 1.1）。")
    parser.add_argument(
        "--max-pause",
        type=_non_negative_float,
        default=0.2,
        help="セリフ間の無音をこの秒数までに詰めて結合します（デフォルト 0.2）。0 で詰めない。",
    )
    parser.add_argument(
        "--resume",
        dest="resume_run_id",
        default=None,
        help="既存の work/<run_id> ディレクトリから処理を再開する（入力コピーや途中結果を再利用）。",
    )
    parser.add_argument(
        "--llm",
        choices=["none", "openai", "ollama"],
        default="none",
        help="口調変換に使う LLM バックエンド（none は変換なし）。",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Ollama のホスト名 (default: 127.0.0.1)。",
    )
    parser.add_argument(
        "--ollama-port",
        type=_port,
        default=None,
        help="Ollama のポート番号 (default: 11434)。",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama で使用するモデル名。デフォルトは環境変数 LOCAL_LLM_MODEL または 'llama3'。 ",
    )
    parser.add_argument(
        "--subtitles",
        choices=["off", "all", "split"],
        default="off",
        help="字幕データの出力モード: off=生成なし / all=すべての発話を1ファイル / split=話者ごとに別ファイル。",
    )
    add_video_arguments(
        parser,
        defaults=VideoArgumentDefaults(
            subtitle_max_chars=25,
            subtitle_font=None,
            subtitle_font_size=84,
            video_width=1080,
            video_height=1920,
            video_fps=24,
            video_images=[],
            video_image_scale=0.45,
            video_image_pos=None,
            video_image_times=None,
        ),
    )
    parser.add_argument(
        "--review-transcript",
        action="store_true",
        help="文字起こし保存後に処理を停止し、transcript.json を手動で修正するために終了します。再開用コマンドを表示します。",
    )
    parser.add_argument(
        "--review-transcript-llm",
        action="store_true",
        help="文字起こし保存後に LLM で明らかな誤字脱字を校正してから次の工程へ進みます (--llm でバックエンド指定)。",
    )
    parser.add_argument(
        "--skip-review-transcript",
        action="store_true",
        help="誤字脱字の自動校正を行わずに次の工程へ進みます。",
    )
    parser.add_argument(
        "--no-style-with-llm",
        action="store_false",
        dest="style_with_llm",
        default=True,
        help="口調変換で LLM を使用しない (--llm でバックエンド指定)。省略時は LLM による口調変換を行います。",
    )
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        default=None,
        help="指定したステップから処理を再開する（--resume と併用）。ステップ名または1から始まる番号で指定。",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PipelineConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input is None and args.resume_run_id is None:
        parser.error("--resume を指定しない場合は入力ファイルが必要です。")

    if args.resume_run_id is None and args.resume_from is not None:
        parser.error("--resume-from は --resume と一緒に使用する必要があります。")

    if args.review_transcript and args.review_transcript_llm:
        parser.error("--review-transcript と --review-transcript-llm は同時に指定できません。")
    if args.skip_review_transcript and (args.review_transcript or args.review_transcript_llm):
        parser.error("--skip-review-transcript は他の誤字脱字校正オプションと同時に指定できません。")
    validate_video_image_args(parser, args.video_images, args.video_image_times)

    input_path = pathlib.Path(args.input).expanduser().resolve() if args.input else None
    video_images = [pathlib.Path(p).expanduser().resolve() for p in args.video_images]
    review_mode = "llm"
    if args.skip_review_transcript:
        review_mode = "off"
    elif args.review_transcript:
        review_mode = "manual"
    return PipelineConfig(
        input_audio=input_path,
        voicevox_url=args.voicevox_url,
        speakers=args.speakers,
        speaker1=args.speaker1,
        speaker2=args.speaker2,
        color1=args.color1,
        color2=args.color2,
        zunda_senior_job=args.zunda_senior_job,
        zunda_junior_job=args.zunda_junior_job,
        want_mp3=args.mp3,
        mp3_bitrate=args.bitrate,
        ffmpeg_path=args.ffmpeg_path,
        keep_work=args.keep_work,
        output_name=args.output_name,
        force_transcribe=args.force_transcribe,
        force_diarize=args.force_diarize,
        force_overwrite=args.force,
        whisper_model=args.model,
        llm_backend=args.llm,
        llm_host=args.ollama_host,
        llm_port=args.ollama_port,
        ollama_model=args.ollama_model,
        speed_scale=args.speed_scale,
        max_pause_between_segments=args.max_pause,
        resume_run_id=args.resume_run_id,
        resume_from=args.resume_from,
        subtitle_mode=args.subtitles,
        subtitle_max_chars=args.subtitle_max_chars,
        subtitle_font=args.subtitle_font,
        subtitle_font_size=args.subtitle_font_size,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=args.video_fps,
        output_mp4=args.mp4,
        output_mov=args.mov,
        review_transcript=review_mode,
        style_with_llm=args.style_with_llm,
        diarization_threshold=args.diarization_threshold,
        intro1=args.intro1,
        intro2=args.intro2,
        prepend_intro=args.prepend_intro,
        video_images=video_images,
        video_image_scale=args.video_image_scale,
        video_image_position=args.video_image_pos,
        video_image_times=args.video_image_times,
    )


def _warn_missing_hf_token() -> None:
    token, _ = resolve_hf_token(os.environ.get("PYANNOTE_TOKEN"))
    if token is None:
        print(
            "警告: Hugging Face Token (HF_TOKEN/PYANNOTE_TOKEN) が設定されていません。"
            "pyannote の話者分離を行う場合に必要となりますが、このまま続行します。",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    _warn_missing_hf_token()
    try:
        run_pipeline(config)
    except (FileNotFoundError, RuntimeError) as e:
        # envinfo が詳細を追記してくれるのでそのまま出す
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)
