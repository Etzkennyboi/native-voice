# Architecture (Phase 0/1)

Baseline stack per PRD §6.1: Python + PyTorch, VoxCPM family checkpoint via
Hugging Face-compatible loading, Gradio UI, SoundFile WAV output, JSONL
run logs.

## Request flow (PRD §6.2, inference path only)

```
user text
  -> input validation          app/validation.py (user-safe messages)
  -> text chunking             app/tts_engine.py  (sentence-boundary split)
  -> TTS model                 app/tts_engine.py  (_infer_segment, adapter)
  -> audio validation          app/validation.py (must pass before exposing)
  -> metadata logging          outputs/run_log.jsonl (PRD §5.5 fields)
  -> playback + download       app/gradio_app.py
```

## Deliberate deviation from PRD §14 (flagged)

§14 lists a flat scaffold; we added two modules under `app/` so validation
and generation logic are shared by the app, the notebooks, and the tests
without duplication:

- `app/tts_engine.py` — model loading + generation + run logging
- `app/validation.py` — script/audio checks shared by app, scripts, tests

Everything else follows §14 exactly.

## Baseline model swap points (PRD §6.3)

The model is treated as swappable. Changing the baseline touches exactly:

1. `configs/baseline.yaml` — `model.id`, `model.revision`, dtype, inference params.
2. `app/tts_engine.py` — only the functions marked `MODEL API ADAPTER`
   (`_load_model`, `_sample_rate`, `_infer_segment`).
3. `notebooks/01_environment_check.ipynb` — dependency install cell + re-verify.

Phase 2+ tracks (benchmark, fine-tuning, NVAlign-style optimization) are NOT
implemented here. PRD §6.3's warning applies unchanged: no compatibility with
the NVAlign training method is assumed or implied by this scaffold.

## Reliability guarantees implemented (PRD §10)

- Outputs are written to a temp file, validated (`app/validation.py`), then
  atomically renamed — a failed generation cannot corrupt saved outputs.
- Failed runs are recorded with status `"failed"` and never marked successful.
- User-facing error messages are static, safe strings; stack traces go to
  `logs/gradio_app.log` only.
