#!/usr/bin/env python3
"""CLI wrapper around app.validation.validate_wav_file (Phase 0/1 utility).

Usage:
    python scripts/validate_audio.py path/to/file.wav [more.wav ...]

Exit codes: 0 = all files valid, 1 = at least one invalid, 2 = usage error.
This is the audio-check used before any generated file is handed to a user
(PRD §10 "Outputs must be checked for valid audio data").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.validation import InvalidAudioError, validate_wav_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate WAV file(s).")
    parser.add_argument("paths", nargs="+", help="One or more .wav paths")
    args = parser.parse_args()

    results: list[dict] = []
    any_invalid = False
    for raw in args.paths:
        path = Path(raw)
        try:
            info = validate_wav_file(path)
            results.append({"path": str(path), "valid": True, **info})
        except InvalidAudioError as exc:
            any_invalid = True
            results.append({"path": str(path), "valid": False, "reason": str(exc)})

    print(json.dumps(results, indent=2))
    return 1 if any_invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
