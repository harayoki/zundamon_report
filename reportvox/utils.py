"""小さなユーティリティ関数群。"""

from __future__ import annotations

import pathlib
import platform
import shutil
import string


def normalize_hex_color(value: str) -> str:
    """#RRGGBB 形式のカラーコードを正規化する。"""

    if value is None:
        raise ValueError("カラーコードを指定してください。")
    raw = value.strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) != 6 or any(ch not in string.hexdigits for ch in raw):
        raise ValueError(f"カラーコードは #RRGGBB 形式で指定してください: {value}")
    return f"#{raw.lower()}"


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = normalize_hex_color(color)
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def detect_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def is_windows() -> bool:
    return platform.system().lower().startswith("win")
