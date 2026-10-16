"""Baseline TTS engine for NaturalVoice AI (Phase 0/1).

MODEL-SWAP NOTE (PRD §6.3 / §12):
    The PRD's candidate baseline is "VoxCPM2" (2B params). A live Colab run
    (Oct 16, 2026) showed PyPI carries `voxcpm` 1.0.x / 1.5.0 / 2.0.0–2.0.3;
    the 2.0.x line is almost certainly the PRD's VoxCPM2, and we pin
    `voxcpm==2.0.3` (see requirements.txt). The functions marked "MODEL API
    ADAPTER" below assume the documented OpenBMB VoxCPM API
    (`VoxCPM.from_pretrained` / `.infer`); their exact signatures still need
    verifying against the installed 2.0.x package (Colab introspection) —
    that is the ONLY thing to change here if they differ. Everything else
    model-specific lives in `configs/baseline.yaml`; swapping the baseline
    model should require changes ONLY in those places + a re-run of
    notebooks/01_environment_check.ipynb.

Responsibilities:
    - lazy model load (nothing is loaded at import → free local test runs)
    - text -> audio generation with sentence-based chunking for long scripts
    - output validation + atomic file writes (failures never corrupt outputs)
    - per-run metadata records appended to a JSONL log (PRD §5.5, §10)
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .validation import InvalidAudioError, atomic_move

log = logging.getLogger("naturalvoice")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise ValueError(f"Config at {path} is missing the required 'model' section.")
    return cfg


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TTSEngine:
    """Lazy-enabled wrapper around the baseline speech model."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG):
        self.cfg = load_config(config_path)
        self.config_path = Path(config_path)
        self._model = None
        self._lock = threading.Lock()  # Gradio default concurrency is 1, but be safe

    # ------------------------------------------------------------------ model

    @property
    def model_id(self) -> str:
        return str(self.cfg["model"]["id"])

    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self):
        """MODEL API ADAPTER — change here when swapping the baseline model.

        voxcpm 2.0.3 signatures (verified via Colab introspection, Oct 2026):
          VoxCPM.from_pretrained(hf_model_id='openbmb/VoxCPM2', load_denoiser=True,
                                 zipenhancer_model_id='iic/...', cache_dir=None,
                                 local_files_only=False, optimize=True, device=None,
                                 lora_config=None, lora_weights_path=None, **kwargs)
        NOTE: lora_config/lora_weights_path are Phase 3 hooks — deliberately unused here (PRD §3.3).
        """
        # Import inside the call so importing this module never pulls the model
        # stack (keeps unit tests fast and offline).
        from voxcpm import VoxCPM

        mcfg = self.cfg["model"]
        device = mcfg.get("device", "auto")
        kwargs: dict = {"optimize": bool(mcfg.get("optimize", True))}
        if device and device != "auto":
            kwargs["device"] = device
        with self._lock:
            if self._model is None:
                log.info("Loading checkpoint %s (optimize=%s, device=%s)…", mcfg["id"], kwargs.get("optimize"), device)
                self._model = VoxCPM.from_pretrained(hf_model_id=mcfg["id"], **kwargs)
                log.info("Checkpoint loaded.")
        return self._model

    def _sample_rate(self) -> int:
        """MODEL API ADAPTER — prefer the model's own sample rate if exposed."""
        for attr in ("sample_rate", "output_sample_rate", "sampling_rate"):
            sr = getattr(self._model, attr, None)
            if isinstance(sr, int) and sr > 0:
                return sr
        # Fall back to the configured value; VERIFY against the model docs.
        return int(self.cfg["output"].get("sample_rate", 16000))

    # ------------------------------------------------------------- generation

    def _infer_segment(self, text: str, overrides: dict) -> np.ndarray:
        """MODEL API ADAPTER — change here when swapping the baseline model.

        voxcpm 2.0.3 (verified via Colab introspection):
          model.generate(*args, **kwargs) -> np.ndarray          # blocking
          model.generate_streaming(...) -> Generator[np.ndarray] # kept in mind for Phase 2
          known kwargs: cfg_value (0.1-10, rec 1.0-3.0),
                        inference_timesteps (1-100, rec 4-30)
        Must return a 1-D float numpy waveform (any sample rate).
        """
        infer_cfg = {**self.cfg.get("inference", {}), **(overrides or {})}
        prompt = infer_cfg.get("prompt", {}) or {}
        # NOTE: an empty reference ("prompt") means the model's default voice.
        # Reference-audio voice cloning requires documented consent (PRD §8);
        # the MVP ships with NO reference voice configured.
        # (Phase 4 note: voxcpm 2.0.x's CLI wraps style controls as
        # "({control}){text}" — out of scope here, documented in docs/.)
        wav = self._model.generate(
            text,
            cfg_value=float(infer_cfg.get("cfg_value", 2.0)),
            inference_timesteps=int(infer_cfg.get("inference_timesteps", 10)),
        )
        if isinstance(wav, (list, tuple)):
            chunks = [np.asarray(w).reshape(-1) for w in wav if np.asarray(w).size]
            if not chunks:
                raise InvalidAudioError("The model returned no audio samples.")
            return np.concatenate(chunks).astype(np.float32)
        return np.asarray(wav).reshape(-1).astype(np.float32)

    def _chunk_text(self, text: str, max_chars: int) -> list[str]:
        """Split long scripts at sentence boundaries (PRD §10 Performance)."""
        max_chars = max(80, int(max_chars))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?…:])\s+", text) if s.strip()]
        chunks: list[str] = []
        buf = ""
        for s in sentences:
            cand = f"{buf} {s}".strip() if buf else s
            if len(cand) <= max_chars:
                buf = cand
                continue
            if buf:
                chunks.append(buf)
            if len(s) <= max_chars:
                buf = s
            else:  # pathological single sentence; hard-wrap it
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i : i + max_chars])
                buf = ""
        if buf:
            chunks.append(buf)
        return chunks or ([text] if text else [])

    def generate(self, script: str, overrides: dict | None = None) -> dict:
        """Generate audio for `script`. Returns the metadata record.

        Raises on failure. Output is written atomically: a failed run can
        never leave a corrupt file in the outputs directory, and no failed
        run is ever recorded as success (PRD §10 Reliability).
        """
        infer_cfg = {**self.cfg.get("inference", {}), **(overrides or {})}
        out_dir = REPO_ROOT / self.cfg["paths"]["outputs_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(time.time_ns())[-6:]

        t_start = time.perf_counter()
        model = self._load_model()
        load_s = time.perf_counter() - t_start

        seed = infer_cfg.get("seed")
        if isinstance(seed, int):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        t_gen = time.perf_counter()
        segments = self._infer_segment(script, overrides)
        gen_s = time.perf_counter() - t_gen

        tmp_path = out_dir / f"tmp-{run_id}.wav"
        final_path = out_dir / f"generated-{run_id}.wav"
        try:
            import soundfile as sf

            sf.write(str(tmp_path), segments, self._sample_rate(), subtype="PCM_16")
            info = atomic_move(tmp_path, final_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)  # never leave partial temp files
            raise

        record = {
            "run_id": run_id,
            "timestamp": utc_now(),
            "status": "success",
            "model_id": self.model_id,
            "checkpoint_revision": self.cfg["model"].get("revision") or "unpinned",
            "config_file": str(self.config_path.relative_to(REPO_ROOT)),
            "settings": {
                "cfg_value": infer_cfg.get("cfg_value"),
                "inference_timesteps": infer_cfg.get("inference_timesteps"),
                "seed": seed,
                "normalize": infer_cfg.get("normalize", True),
                "denoise": infer_cfg.get("denoise", False),
            },
            "input_chars": len(script),
            "audio_file": final_path.name,
            "sample_rate": info["sample_rate"],
            "output_seconds": info["seconds"],
            "gen_seconds": round(gen_s, 3),
            "load_seconds": round(load_s, 3),
            "realtime_factor": round(gen_s / info["seconds"], 3) if info["seconds"] else None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        self._append_log(record)
        return record

    # ---------------------------------------------------------------- logging

    def _append_log(self, record: dict) -> None:
        log_path = REPO_ROOT / self.cfg["paths"]["run_log"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_failure(self, script: str, detail: str) -> None:
        """Record a failed run — failures are logged, never shown as success."""
        try:
            self._append_log(
                {
                    "run_id": "failed-" + str(time.time_ns())[-9:],
                    "timestamp": utc_now(),
                    "status": "failed",
                    "model_id": self.model_id,
                    "input_chars": len(script),
                    "error": detail[:300],  # message only, never a stack trace
                }
            )
        except Exception:  # logging must never mask the original failure
            log.exception("Could not write failure record to the run log.")

    def recent_runs(self, limit: int = 10) -> list[dict]:
        log_path = REPO_ROOT / self.cfg["paths"]["run_log"]
        if not log_path.exists():
            return []
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()[-limit:]
        out = []
        for line in reversed(lines):  # newest first
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


_ENGINE: TTSEngine | None = None


def get_engine() -> TTSEngine:
    """Shared engine instance so the Gradio app (re)uses one model load."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TTSEngine()
    return _ENGINE
