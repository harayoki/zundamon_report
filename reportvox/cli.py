"""ReportVox のコマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Sequence

from reportvox import utils
from reportvox.config import PipelineConfig
from reportvox.pipeline import run_pipeline


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
        "--zunda-senior-job", dest="zunda_senior_job", default=None, help="ずんだもんが憧れる職業を指定（--zunda-junior-job と併用）。"
    )
    parser.add_argument(
        "--zunda-junior-job", dest="zunda_junior_job", default=None, help="ずんだもんの現在の役割を指定（--zunda-senior-job と併用）。"
    )
    parser.add_argument("--mp3", action="store_true", help="mp3 を生成（out/ には mp3 だけを出力）。")
    parser.add_argument("--bitrate", default="192k", help="mp3 出力時のビットレート（--mp3 使用時のみ）。")
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="ffmpeg 実行ファイルへのパス（コマンド名またはフルパス）。")
    parser.add_argument(
        "--output-name",
        default=None,
        help="出力ファイル名のベース（拡張子不要）。例: --output-name demo => out/demo.wav|mp3|srt",
    )
    parser.add_argument("-f", "--force", action="store_true", help="既存の出力があっても確認せず上書きする。")
    parser.add_argument("--keep-work", action="store_true", help="work/ 以下の中間ファイルを削除せず残す（開発/再出力向け）。")
    parser.add_argument("--model", default="large-v3", choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3'], help="利用する Whisper モデルサイズ（tiny/base/small/medium/large-v3）。")
    parser.add_argument("--speed-scale", type=_positive_float, default=1.1, help="VOICEVOX の speedScale を指定（デフォルト 1.1）。")
    parser.add_argument(
        "--resume",
        dest="resume_run_id",
        default=None,
        help="既存の work/<run_id> ディレクトリから処理を再開する（入力コピーや途中結果を再利用）。",
    )
    parser.add_argument(
        "--llm",
        choices=["none", "openai", "ollama", "gemini"],
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
        "--hf-token",
        default=None,
        help="pyannote.audio 用の Hugging Face Token（auto/2 人話者分離時に必要、環境変数 PYANNOTE_TOKEN も参照）。",
    )
    parser.add_argument(
        "--subtitles",
        choices=["off", "all", "split"],
        default="off",
        help="字幕データの出力モード: off=生成なし / all=すべての発話を1ファイル / split=話者ごとに別ファイル。",
    )
    parser.add_argument(
        "--subtitle-max-chars",
        type=_positive_int,
        default=25,
        help="字幕1枚あたりの最大文字数。0 で制限なし。デフォルトは 25。",
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
        "--style-with-llm",
        action="store_true",
        help="口調変換で LLM を使用します (--llm でバックエンド指定)。",
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

    input_path = pathlib.Path(args.input).expanduser().resolve() if args.input else None
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
        force_overwrite=args.force,
        whisper_model=args.model,
        llm_backend=args.llm,
        llm_host=args.ollama_host,
        llm_port=args.ollama_port,
        ollama_model=args.ollama_model,
        hf_token=args.hf_token,
        speed_scale=args.speed_scale,
        resume_run_id=args.resume_run_id,
        resume_from=args.resume_from,
        subtitle_mode=args.subtitles,
        subtitle_max_chars=args.subtitle_max_chars,
        review_transcript=review_mode,
        style_with_llm=args.style_with_llm,
        diarization_threshold=args.diarization_threshold,
        intro1=args.intro1,
        intro2=args.intro2,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    try:
        run_pipeline(config)
    except (FileNotFoundError, RuntimeError) as e:
        # envinfo が詳細を追記してくれるのでそのまま出す
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)
