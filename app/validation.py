"""Input + output validation helpers (Phase 0/1).

Every check here is "safe to show to a user": messages never contain
filesystem paths or stack traces (PRD §10 Security). Full details belong in
the application log, written by the caller.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


class ValidationError(ValueError):
    """Script text failed validation; message is safe to show in the UI."""


class InvalidAudioError(ValueError):
    """Audio data failed validation; message is safe to show in the UI."""


# Phase 1 defaults; the Gradio app may pass looser/stricter values from config.
DEFAULT_MAX_CHARS = 5000
MIN_AUDIO_SECONDS = 0.1
MAX_AUDIO_SECONDS = 600.0
MIN_PEAK_AMPLITUDE = 1e-4  # below this the clip is effectively silence


def validate_script(
    text: str | None,
    *,
    min_chars: int = 1,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Validate + normalize a script. Returns trimmed text or raises.

    PRD §7 Phase 1 tasks: "Validate text length and empty input".
    """
    if text is None or not text.strip():
        raise ValidationError("Please enter a script before generating audio.")
    cleaned = " ".join(text.strip().split())  # collapse runs of whitespace
    if len(cleaned) < min_chars:
        raise ValidationError("The script is too short to narrate.")
    if len(cleaned) > max_chars:
        raise ValidationError(
            f"The script is too long ({len(cleaned)} characters). "
            f"The MVP limit is {max_chars} — please split it into shorter passages."
        )
    return cleaned


def validate_wav_file(path: str | Path) -> dict:
    """Validate a WAV file on disk. Returns audio info; raises InvalidAudioError.

    Checks the file parses, has audible content, and a sane duration — this is
    the gate that runs *before* any output is exposed to the user or renamed
    into place (PRD §10 Reliability).
    """
    p = Path(path)
    try:
        info = sf.info(str(p))
        data, sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception as exc:  # unreadable/truncated header etc.
        raise InvalidAudioError("The audio file could not be read.") from exc

    if data.size == 0:
        raise InvalidAudioError("The audio file is empty.")
    seconds = float(data.shape[0] / sr) if sr else 0.0
    if seconds < MIN_AUDIO_SECONDS:
        raise InvalidAudioError("The generated audio is too short.")
    if seconds > MAX_AUDIO_SECONDS:
        raise InvalidAudioError("The generated audio is unexpectedly long.")
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if not math.isfinite(peak) or peak < MIN_PEAK_AMPLITUDE:
        raise InvalidAudioError("The generated audio is silent.")
    return {"seconds": round(seconds, 3), "sample_rate": int(sr), "channels": int(info.channels)}


def atomic_move(src: str | Path, dst: str | Path) -> dict:
    """Validate audio at `src`, then atomically move it onto `dst`.

    Guarantees (PRD §7 Phase 1): a failed generation never leaves corrupt or
    partial audio at the destination path — either `dst` ends up with valid,
    fully written audio, or it does not exist at all.
    """
    info = validate_wav_file(src)  # raises before any rename happens
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # os.replace is atomic on the same filesystem; tmp lives in the same dir.
    src.replace(dst)
    return info


def sanitize_seed(value) -> int | None:
    """Coerce a UI seed into a non-negative int, or None (unseeded run)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        seed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValidationError("Seed must be an integer (or blank for a random seed).") from exc
    if seed < 0:
        raise ValidationError("Seed must be a non-negative integer.")
    return seed
