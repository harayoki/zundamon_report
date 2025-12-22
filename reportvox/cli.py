"""ReportVox のコマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

from .pipeline import PipelineConfig, run_pipeline


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
    parser.add_argument("--speaker1", default="zundamon", help="主話者に割り当てるキャラクターID（characters/ 以下のID）。")
    parser.add_argument("--speaker2", default="metan", help="副話者に割り当てるキャラクターID（characters/ 以下のID）。")
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
    parser.add_argument("--model", default="small", help="利用する Whisper モデルサイズ（tiny/base/small/medium/large）。")
    parser.add_argument("--speed-scale", type=_positive_float, default=1.1, help="VOICEVOX の speedScale を指定（デフォルト 1.1）。")
    parser.add_argument(
        "--resume",
        dest="resume_run_id",
        default=None,
        help="既存の work/<run_id> ディレクトリから処理を再開する（入力コピーや途中結果を再利用）。",
    )
    parser.add_argument(
        "--llm",
        choices=["none", "openai", "local"],
        default="none",
        help="口調変換に使う LLM バックエンド（none は変換なし）。",
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PipelineConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input is None and args.resume_run_id is None:
        parser.error("--resume を指定しない場合は入力ファイルが必要です。")

    input_path = pathlib.Path(args.input).expanduser().resolve() if args.input else None
    return PipelineConfig(
        input_audio=input_path,
        voicevox_url=args.voicevox_url,
        speakers=args.speakers,
        speaker1=args.speaker1,
        speaker2=args.speaker2,
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
        hf_token=args.hf_token,
        speed_scale=args.speed_scale,
        resume_run_id=args.resume_run_id,
        subtitle_mode=args.subtitles,
        subtitle_max_chars=args.subtitle_max_chars,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    run_pipeline(config)
