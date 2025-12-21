"""小さなユーティリティ関数群。"""

from __future__ import annotations

import pathlib
import platform
import shutil


def detect_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def is_windows() -> bool:
    return platform.system().lower().startswith("win")
