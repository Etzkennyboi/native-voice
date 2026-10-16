# Evaluation protocol (summary of PRD §9 — formalized in Phase 2)

Core metric families the platform will report:

- **Intelligibility** — WER via an appropriate ASR + manual checks on names,
  numbers, difficult words.
- **Naturalness** — blind human ratings under a fixed listening protocol
  (MOS where the design supports it).
- **Speaker consistency** — speaker-embedding similarity + human identity-drift
  checks (relevant once reference-audio conditioning exists).
- **Expressive control** — tag precision/recall, false positives/negatives
  (Phase 4 only).
- **Performance** — time to first audio, total generation time, real-time
  factor, peak GPU memory, cost per generated minute. Phase 0/1 already
  records generation time, output duration, and RTF in `run_log.jsonl` so
  later runs are comparable.

Rules carried into Phase 2:

- Automatic metrics never substitute for listening tests (PRD §9).
- No metric is optimized in isolation (e.g. lower WER alone != more human).
- "Any unsupported controls must remain disabled rather than being simulated
  by UI elements" (PRD §5.2) — the Gradio app exposes only model-supported
  knobs (cfg, timesteps, seed).

Concrete test-pass definition, identifiers, and scoring sheets arrive with
Phase 2 (`scripts/evaluate_outputs.py`).
