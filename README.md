# NaturalVoice AI — Phase 0 & 1 (MVP)

Text-to-speech research platform. This repository currently contains
**Phase 0 (environment & feasibility)** and **Phase 1 (baseline TTS app)**
only — per PRD v1.0 §3.3/§7, fine-tuning, expressive tags, benchmark
building, and NVAlign-style optimization are explicitly NOT implemented yet.

## Repository layout (PRD §14)

```
naturalvoice-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── baseline.yaml        # model id + inference settings (the ONLY place to change the baseline model)
│   └── evaluation.yaml      # placeholder — filled in Phase 2
├── notebooks/
│   ├── 01_environment_check.ipynb
│   └── 02_first_generation.ipynb
├── app/
│   ├── gradio_app.py        # Phase 1 UI
│   ├── tts_engine.py        # model load/generate + JSONL run log (deviation, see below)
│   ├── validation.py        # script/audio checks shared everywhere (deviation, see below)
│   └── __init__.py
├── data/
│   ├── README.md
│   └── manifests/           # empty until Phase 3
├── scripts/
│   ├── validate_audio.py    # CLI wrapper around app/validation.py
│   ├── prepare_dataset.py   # STUB — Phase 3
│   └── evaluate_outputs.py  # STUB — Phase 2
├── experiments/
│   └── README.md            # explains env snapshots + run_log.jsonl
├── tests/
│   └── test_audio_validation.py
└── docs/
    ├── architecture.md
    ├── dataset_policy.md
    └── evaluation_protocol.md
```

## Flagged deviations from the PRD

1. **Baseline model = VoxCPM2-line package, `voxcpm==2.0.3`.**
   The live Colab run (Oct 16) showed PyPI carries `voxcpm` 1.0.x / 1.5.0 /
   2.0.0–2.0.3 — the 2.0.x line is almost certainly the PRD's VoxCPM2, so
   requirements pin `2.0.3` and notebook 01 installs it. Still to verify on a
   GPU session: exact API signatures (`from_pretrained` / `infer` kwargs),
   the default checkpoint id, and the equivalent of the 0.5B checkpoint for
   fast feasibility tests. Everything model-specific is isolated in
   `configs/baseline.yaml` + the `MODEL API ADAPTER` functions in
   `app/tts_engine.py`, so adapting is a config edit + one adapter file,
   followed by re-running notebook 01. Verify the checkpoint's license
   before any paid use (PRD §6.1).
2. **Two extra files under `app/` (`tts_engine.py`, `validation.py`).**
   §14's flat layout would duplicate validation/loading logic across the
   app, notebooks, and tests; these two modules are the single source of
   truth and are used by all three. No other structural changes were made.

## Setup

```bash
git clone <repo> && cd naturalvoice-ai
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt
```

For the model check + first generation use **Google Colab or a rented GPU
VM** (PRD §6.1). On Colab, `torch` comes preinstalled with the matching CUDA
build — do not force-reinstall it.

## How to run — Phase 0 (notebooks)

1. Open a cloud notebook (Colab GPU runtime is fine for the 0.5B baseline;
   a 2B-class checkpoint will need more VRAM — notebook 01 enforces the
   `min_vram` check from `configs/baseline.yaml` and stops you *before*
   you rent/burn anything).
2. Upload/clone the repo, then run `notebooks/01_environment_check.ipynb`
   top-to-bottom. It reports GPU/VRAM, Python/PyTorch/CUDA versions, disk
   space, installs pinned deps, and writes `.gen/env_check-<timestamp>.json`.
3. Run `notebooks/02_first_generation.ipynb`. It loads the checkpoint,
   generates ONE short test utterance, plays it inline, triggers a browser
   download, and appends the full settings record to
   `outputs/run_log.jsonl`.
4. **Cost discipline (PRD §11):** each notebook is one short pass — shut the
   GPU runtime down right after. Neither notebook runs anything in the
   background.

## How to run — Phase 1 (Gradio app)

```bash
python app/gradio_app.py
```

Then open the printed local URL. Features: script box, generate button with
progress, model-supported settings only (CFG, timesteps, seed — per PRD §5.2
unsupported controls are never simulated), inline playback, explicit WAV
download button, metadata panel (model id/revision/settings/timing), and a
recent-runs table backed by `outputs/run_log.jsonl`. On Colab add
`share=True` in `main()` for a public link.

Error behavior: empty/oversized/short scripts raise friendly messages;
generation failures show a generic user-safe message while the full
traceback goes to `logs/gradio_app.log` — never the browser (PRD §10).

## How to run — tests (free, no GPU needed)

```bash
python -m pytest tests/ -q
```

Covers input validation, WAV validation (valid/silent/corrupt), and the
atomic-output guarantee (failed generations cannot corrupt saved files).
Run these locally *before* renting GPU minutes.

Also useful, no GPU needed:

```bash
python scripts/validate_audio.py path/to/some.wav   # validates any WAV
```

## Where results are recorded (PRD §5.4/§5.5, §10)

- `outputs/*.wav` — generated audio (git-ignored).
- `outputs/run_log.jsonl` — every run: model id, checkpoint revision,
  settings, seed, input size, generation duration, output duration, RTF,
  device, status. Failed runs are logged as `"failed"` and never shown as
  success.
- `.gen/env_check-*.json` — environment snapshots from notebook 01
  (small JSON, safe to commit).

## What is intentionally NOT here (Phase 2+)

Benchmark set, evaluation reports, dataset prep, fine-tuning, expressive
tags, and any NVAlign-related code. PRD §6.3's NVAlign caveat stands
unchanged: nothing in this repo should be assumed compatible with that
training method. The recommended next step (PRD §15) is exactly what these
two notebooks do: confirm feasibility first, then build the evaluation set.
