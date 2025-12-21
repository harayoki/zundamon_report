"""Environment detection and reporting utilities."""

from __future__ import annotations

import os
import pathlib
import subprocess
from dataclasses import dataclass, field
from importlib import metadata
from typing import Dict, Optional

DEPENDENCIES: Dict[str, str] = {
    "pyannote.audio": "pyannote.audio",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "openai-whisper": "openai-whisper",
    "huggingface-hub": "huggingface-hub",
    "requests": "requests",
}


@dataclass
class FfmpegProbeResult:
    path: pathlib.Path
    available: bool
    version: str | None
    error: str | None = None


def probe_ffmpeg(ffmpeg_path: str) -> FfmpegProbeResult:
    """Check whether ffmpeg is available and return its version line if possible."""
    path_obj = pathlib.Path(ffmpeg_path)
    if path_obj.is_dir():
        exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        path_obj = path_obj / exe_name

    try:
        result = subprocess.run(
            [str(path_obj), "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
        version_line = None
        for line in result.stdout.splitlines():
            if line.strip():
                version_line = line.strip()
                break
        return FfmpegProbeResult(path_obj, True, version_line)
    except FileNotFoundError:
        return FfmpegProbeResult(path_obj, False, None, error="not_found")
    except PermissionError:
        return FfmpegProbeResult(path_obj, False, None, error="permission")
    except subprocess.CalledProcessError:
        return FfmpegProbeResult(path_obj, False, None, error="execution_failed")


def gather_dependency_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for display_name, dist_name in DEPENDENCIES.items():
        try:
            versions[display_name] = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return versions


@dataclass
class EnvironmentInfo:
    ffmpeg_path: str
    ffmpeg_available: bool
    ffmpeg_version: str | None
    dependency_versions: Dict[str, str] = field(default_factory=dict)
    hf_token_source: str = "absent"

    @classmethod
    def collect(cls, ffmpeg_path: str, hf_token_arg: Optional[str], env_token: Optional[str]) -> "EnvironmentInfo":
        ffmpeg_probe = probe_ffmpeg(ffmpeg_path)
        token_source = "cli" if hf_token_arg is not None else "env" if env_token else "absent"
        return cls(
            ffmpeg_path=str(ffmpeg_probe.path),
            ffmpeg_available=ffmpeg_probe.available,
            ffmpeg_version=ffmpeg_probe.version,
            dependency_versions=gather_dependency_versions(),
            hf_token_source=token_source,
        )

    def update_ffmpeg(self, probe: FfmpegProbeResult) -> None:
        self.ffmpeg_path = str(probe.path)
        self.ffmpeg_available = probe.available
        self.ffmpeg_version = probe.version

    def format(self) -> str:
        lines = []
        ffmpeg_line = f"ffmpeg: {'found' if self.ffmpeg_available else 'not found'} (path: {self.ffmpeg_path})"
        if self.ffmpeg_version:
            ffmpeg_line += f" / version: {self.ffmpeg_version}"
        lines.append(ffmpeg_line)
        if self.dependency_versions:
            for name, version in sorted(self.dependency_versions.items()):
                lines.append(f"{name}: {version}")
        else:
            lines.append("依存モジュール: 検出なし")

        if self.hf_token_source == "cli":
            token_line = "Hugging Face token source: CLI argument (--hf-token)"
        elif self.hf_token_source == "env":
            token_line = "Hugging Face token source: environment variable (PYANNOTE_TOKEN)"
        else:
            token_line = "Hugging Face token source: not provided"
        lines.append(token_line)
        return "環境情報:\n  " + "\n  ".join(lines)


def append_env_details(message: str, env_info: EnvironmentInfo | None) -> str:
    if env_info is None:
        return message
    return f"{message}\n{env_info.format()}"
