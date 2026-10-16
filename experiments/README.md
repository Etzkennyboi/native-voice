# experiments/

Per-run record-keeping (PRD §5.5, §10 Reproducibility).

- `env_check-*.json` — environment snapshots written by notebooks/01 (fine to commit; small JSON).
- Generation runs are appended to `outputs/run_log.jsonl` by `app/tts_engine.py` and
  `notebooks/02_first_generation.ipynb` (git-ignored; copy highlights here if you want
  a result in history).

Every record includes: model id, checkpoint revision, settings, seed, generation
duration, output duration, and device — the fields PRD §5.5 requires for each run.
