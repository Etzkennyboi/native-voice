"""Unit tests for app.validation — run without a GPU or the TTS model.

These tests let us verify Phase 0/1 logic locally (free) before spending any
cloud GPU time (PRD §11 cost discipline):

    python -m pytest tests/ -q
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import soundfile as sf

from app.validation import (
    InvalidAudioError,
    ValidationError,
    atomic_move,
    validate_script,
    validate_wav_file,
)


# ---------- validate_script ----------

def test_empty_script_rejected():
    with pytest.raises(ValidationError):
        validate_script("")


def test_whitespace_only_script_rejected():
    with pytest.raises(ValidationError):
        validate_script("   \n\t  ")


def test_overlong_script_rejected():
    with pytest.raises(ValidationError):
        validate_script("a " * 6000, max_chars=5000)


def test_valid_script_returned_cleaned():
    assert validate_script("  Hello narration.  ") == "Hello narration."


# ---------- validate_wav_file ----------

def _write_wav(directory: Path, name: str, data: np.ndarray, sr: int = 16000) -> Path:
    target = directory / name
    sf.write(target, data, sr, subtype="PCM_16")
    return target


def test_valid_wav_passes():
    with TemporaryDirectory() as tmp:
        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr  # 1 s of audible tone
        wav = _write_wav(Path(tmp), "ok.wav", 0.2 * np.sin(2 * np.pi * 220 * t), sr)
        info = validate_wav_file(wav)
        assert info["seconds"] > 0.5
        assert info["sample_rate"] == sr


def test_silent_wav_rejected():
    with TemporaryDirectory() as tmp:
        sr = 16000
        wav = _write_wav(Path(tmp), "silent.wav", np.zeros(sr, dtype=np.float32), sr)
        with pytest.raises(InvalidAudioError):
            validate_wav_file(wav)


def test_corrupt_wav_rejected():
    with TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.wav"
        bad.write_bytes(b"definitely not a wav file")
        with pytest.raises(InvalidAudioError):
            validate_wav_file(bad)


# ---------- atomic_move (failed generations must not corrupt outputs) ----------

def test_failed_move_leaves_no_destination_file():
    with TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.wav"
        src.write_bytes(b"not audio")
        dst = Path(tmp) / "final.wav"
        with pytest.raises(InvalidAudioError):
            atomic_move(src, dst)  # validates content before moving
        assert not dst.exists(), "invalid audio must never appear at the output path"
