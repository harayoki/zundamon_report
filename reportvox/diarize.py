"""話者分離に関するユーティリティ。"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")  # すべての警告を無視する あまりにもたくさん出る

import os
import torch
import functools
import huggingface_hub
import contextlib
import json
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Any
from pathlib import Path
import platform

# --- 1. Windows用DLL読み込みパスの設定 ---
# FFmpeg 7.1 Shared版のパスを最優先で読み込む
# 環境変数 'FFMPEG_SHARED_PATH' が設定されている場合のみ DLL 探索パスに追加
# (例: F:\Program Files\ffmpeg-7.1.1-full_build-shared\bin)
ffmpeg_path = os.environ.get("FFMPEG_SHARED_PATH")

if platform.system() == "Windows" and ffmpeg_path:
    if os.path.isdir(ffmpeg_path):
        os.add_dll_directory(ffmpeg_path)

# --- 2. ライブラリのバグを起動時に即殺するパッチ ---

import torch.serialization

# セキュリティチェック（weights_only）を一時的に無効化するパッチ
# pyannoteの古いモデル構造を読み込むために必要です
_original_load = torch.load

@functools.wraps(_original_load)
def _patched_torch_load(*args, **kwargs):
    # weights_only 引数が明示されていない場合、または True の場合に False に書き換える
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)

# PyTorch のロード関数そのものを差し替える
torch.load = _patched_torch_load

# (既存の hf_hub_download パッチ)
_original_hf_download = huggingface_hub.hf_hub_download

@functools.wraps(_original_hf_download)
def _universal_hf_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _original_hf_download(*args, **kwargs)

huggingface_hub.hf_hub_download = _universal_hf_download

# --- 3. データクラス・ヘルパーの定義 ---

@dataclass
class TorchcodecWarningStatus:
    detected: bool = False

@contextlib.contextmanager
def torchcodec_warning_detector() -> Iterable[TorchcodecWarningStatus]:
    """libtorchcodec 関連の警告発生を検知するためのコンテキスト。
    pipeline.py から呼び出されるため必須です。
    """
    status = TorchcodecWarningStatus()
    original_showwarning = warnings.showwarning

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        if "torchcodec" in str(message).lower():
            status.detected = True
        original_showwarning(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _showwarning
    try:
        yield status
    finally:
        warnings.showwarning = original_showwarning

@dataclass
class DiarizedSegment:
    start: float
    end: float
    speaker: SpeakerLabel

@dataclass
class AlignedSegment:
    start: float
    end: float
    text: str
    speaker: SpeakerLabel
    character: Optional[str] = None

    def with_character(self, character: str) -> "AlignedSegment":
        """キャラクター名をセットした新しいインスタンスを返す。"""
        return AlignedSegment(
            start=self.start,
            end=self.end,
            text=self.text,
            speaker=self.speaker,
            character=character,
        )

_PYANNOTE_ACCESS_STEPS = """\
話者分離を有効化するには次を実施してください:
  1. https://huggingface.co/pyannote/speaker-diarization と https://huggingface.co/pyannote/segmentation で利用規約に同意する
  2. https://huggingface.co/settings/tokens でアクセストークンを発行する"""

_TORCHCODEC_TROUBLESHOOT = """TorchCodec/FFmpeg エラーの対処:
- FFmpeg 7.1 Shared版を使い、bin フォルダを PATH に含めてください。
- FFmpeg 8.x 系は現在ライブラリ側が未対応のため、必ず 7.x を使用してください。"""

SpeakerLabel = Literal["A", "B"]
Mode = Literal["auto", "1", "2"]

# --- 4. メイン関数 ---

def diarize_audio(
    audio_path,
    mode: Mode,
    hf_token: Optional[str],
    work_dir,
    *,
    env_info: Any = None,
) -> list[DiarizedSegment]:
    """話者分離を実行する。"""

    if mode == "1":
        return [DiarizedSegment(start=0.0, end=1e9, speaker="A")]

    # 認証設定
    token = hf_token or os.environ.get("PYANNOTE_TOKEN")
    if token is None:
        raise RuntimeError(f"Hugging Face Token が必要です。\n{_PYANNOTE_ACCESS_STEPS}")
    os.environ["HF_TOKEN"] = token

    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError:
        raise RuntimeError("pyannote.audio がインストールされていません。")

    # パイプラインの読み込み
    try:
        pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        device = env_info.device if hasattr(env_info, 'device') and env_info.device else torch.device("cpu")
        pipeline.to(device)
    except Exception as exc:
        raise RuntimeError(f"モデルの読み込みに失敗しました。規約同意を確認してください: {exc}")

    # 解析実行
    try:
        diarization = pipeline(
            str(audio_path),
            min_speakers=1 if mode == "auto" else None,
            max_speakers=2 if mode == "auto" else None,
            num_speakers=2 if mode == "2" else None,
        )
    except Exception as exc:
        raise RuntimeError(f"解析中にエラーが発生しました。FFmpegの共有DLLを確認してください: {exc}\n{_TORCHCODEC_TROUBLESHOOT}")

    # 結果の整形
    segments: list[DiarizedSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        label: SpeakerLabel = "A" if speaker == "SPEAKER_00" else "B"
        segments.append(DiarizedSegment(
            start=float(turn.start),
            end=float(turn.end),
            speaker=label
        ))
    return segments

# --- 5. 補助関数 ---

def align_segments(whisper_segments, diarization: Sequence[DiarizedSegment], min_duration: float = 0.4) -> list[AlignedSegment]:
    aligned: list[AlignedSegment] = []
    for seg in whisper_segments:
        start, end = float(seg["start"]), float(seg["end"])
        speaker = _pick_speaker(start, end, diarization)
        aligned.append(AlignedSegment(start=start, end=end, text=seg["text"].strip(), speaker=speaker))
    return aligned

def _pick_speaker(start: float, end: float, diarization: Sequence[DiarizedSegment]) -> SpeakerLabel:
    overlaps: dict[SpeakerLabel, float] = {"A": 0.0, "B": 0.0}
    for seg in diarization:
        overlap = max(0.0, min(end, seg.end) - max(start, seg.start))
        overlaps[seg.speaker] += overlap
    return "A" if overlaps["A"] >= overlaps["B"] else "B"

def save_diarization(segments: Sequence[DiarizedSegment], path: Path) -> None:
    data = [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")