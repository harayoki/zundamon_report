"""話者分離に関するユーティリティ。"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence

from .envinfo import EnvironmentInfo, append_env_details


_PYANNOTE_ACCESS_STEPS = """\
話者分離を有効化するには次を実施してください:
  1. https://huggingface.co/pyannote/speaker-diarization と https://huggingface.co/pyannote/segmentation で利用規約に同意する
  2. https://huggingface.co/settings/tokens でアクセストークンを発行する"""

_PYANNOTE_TOKEN_USAGE = "発行した Hugging Face Token を --hf-token または環境変数 PYANNOTE_TOKEN で指定してください。"

_TORCHCODEC_TROUBLESHOOT = """TorchCodec が FFmpeg の共有DLLを見つけられず失敗する場合の対処:
- PyTorch 2.8.0 と組み合わせる場合は torchcodec 0.6/0.7 系が互換です。例: `pip uninstall -y torchcodec && pip install \"torchcodec==0.7.*\"`
- Windows では FFmpeg の shared build（共有DLL同梱版）が必須です。Gyan の配布なら「shared」と書かれたものを選び、bin フォルダーを PATH か DLL 探索パスに含めてください。
- Python 実行前に DLL 探索パスへ追加する必要がある場合の例:
  import os
  os.add_dll_directory(r\"F:\\\\Program Files\\\\ffmpeg-shared\\\\bin\")
- VC++ 再頒布可能パッケージなど依存DLLが欠けていないか、Dependencies で libtorchcodec_core*.dll を開いて確認してください。
- どうしても解消しない場合は、soundfile / librosa などで WAV を読み込み、`{\"waveform\": テンソル, \"sample_rate\": int}` を `Pipeline.from_pretrained` に渡す回避策もあります。"""

SpeakerLabel = Literal["A", "B"]
Mode = Literal["auto", "1", "2"]


@dataclass
class DiarizedSegment:
    start: float
    end: float
    speaker: SpeakerLabel

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class AlignedSegment:
    start: float
    end: float
    text: str
    speaker: SpeakerLabel
    character: Optional[str] = None

    def with_character(self, character: str) -> "AlignedSegment":
        return AlignedSegment(
            start=self.start,
            end=self.end,
            text=self.text,
            speaker=self.speaker,
            character=character,
        )


def diarize_audio(
    audio_path,
    mode: Mode,
    hf_token: Optional[str],
    work_dir,
    *,
    env_info: EnvironmentInfo | None = None,
) -> list[DiarizedSegment]:
    if mode == "1":
        return [DiarizedSegment(start=0.0, end=1e9, speaker="A")]

    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            append_env_details(
                "pyannote.audio が必要です。TorchCodec が FFmpeg の DLL を見つけられない場合などは以下を確認してください。\n"
                f"{_PYANNOTE_ACCESS_STEPS}\n{_PYANNOTE_TOKEN_USAGE}\n{_TORCHCODEC_TROUBLESHOOT}",
                env_info,
            )
        ) from exc

    token = hf_token
    if token is None:
        raise RuntimeError(
            append_env_details(
                "pyannote で話者分離するには Hugging Face Token が必要です。\n"
                f"{_PYANNOTE_ACCESS_STEPS}\n{_PYANNOTE_TOKEN_USAGE}",
                env_info,
            )
        )

    _configure_hf_auth(token)

    pipeline_kwargs = _build_pyannote_kwargs(PyannotePipeline, token)

    try:
        pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization", **pipeline_kwargs)
    except Exception as exc:  # pragma: no cover - network/auth errors
        raise RuntimeError(
            append_env_details(
                "Hugging Face Token は指定されていますが pyannote/speaker-diarization への認証に失敗しました。\n"
                "Token を発行したアカウントで利用規約に同意しているか、権限が失効していないかを確認してください。\n"
                f"{_PYANNOTE_ACCESS_STEPS}\n{_PYANNOTE_TOKEN_USAGE}",
                env_info,
            )
        ) from exc
    try:
        diarization = pipeline(
            str(audio_path),
            min_speakers=1 if mode == "auto" else None,
            max_speakers=2 if mode == "auto" else None,
            num_speakers=2 if mode == "2" else None,
        )
    except Exception as exc:
        raise RuntimeError(
            append_env_details(
                "pyannote.audio の話者分離中にエラーが発生しました。TorchCodec が FFmpeg の共有DLLを見つけられない場合によく発生します。\n"
                f"{_TORCHCODEC_TROUBLESHOOT}",
                env_info,
            )
        ) from exc

    segments: list[DiarizedSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        label: SpeakerLabel = "A" if speaker == "SPEAKER_00" else "B"
        segments.append(DiarizedSegment(start=start, end=end, speaker=label))
    return segments


def save_diarization(segments: Sequence[DiarizedSegment], path) -> None:
    data = [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def align_segments(
    whisper_segments,
    diarization: Sequence[DiarizedSegment],
    min_duration: float = 0.4,
) -> list[AlignedSegment]:
    aligned: list[AlignedSegment] = []
    for seg in whisper_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        speaker = _pick_speaker(start, end, diarization)
        aligned.append(
            AlignedSegment(
                start=start,
                end=end,
                text=seg["text"].strip(),
                speaker=speaker,
            )
        )

    # 簡易スムージング: きわめて短いセグメントは直前の話者に合わせる
    for idx in range(1, len(aligned)):
        prev = aligned[idx - 1]
        cur = aligned[idx]
        if (cur.end - cur.start) < min_duration:
            aligned[idx] = AlignedSegment(cur.start, cur.end, cur.text, prev.speaker, cur.character)
    return aligned


def _pick_speaker(start: float, end: float, diarization: Sequence[DiarizedSegment]) -> SpeakerLabel:
    overlaps: dict[SpeakerLabel, float] = {"A": 0.0, "B": 0.0}
    for seg in diarization:
        overlap = _overlap(start, end, seg.start, seg.end)
        overlaps[seg.speaker] += overlap
    return "A" if overlaps["A"] >= overlaps["B"] else "B"


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _configure_hf_auth(token: str) -> None:
    """pyannote.audio から Hugging Face Token が見えるように設定する（2.x/3.x 両対応）。"""
    try:
        from huggingface_hub import login  # type: ignore
    except Exception:
        # huggingface_hub が認識する環境変数にフォールバック
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        return

    login(token=token, add_to_git_credential=False)


def _build_pyannote_kwargs(pyannote_pipeline_cls, token: str) -> dict:
    """インストール済みの pyannote バージョンで受け付けられる引数を判定する。"""
    signature = inspect.signature(pyannote_pipeline_cls.from_pretrained)
    params = signature.parameters

    if "token" in params:
        return {"token": token}

    if "use_auth_token" in params:
        return {"use_auth_token": token}

    if "hf_token" in params:
        return {"hf_token": token}

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return {"token": token}

    return {}
