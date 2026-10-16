# Dataset policy (summary of PRD §8 — enforced in Phase 2/3, asserted now)

- All training data must have documented provenance and permitted usage.
- Per-sample tracking required: path/identifier, transcript, duration,
  sample rate, speaker id, language/accent labels, vocalization tags,
  source + license, and an approved-for-training flag.
- Quality rules: clean recordings, corrected transcripts, no long trailing
  silence, separated train/test material, exclude unclear permissions.
- Voice cloning / reference-audio conditioning (PRD §5.3) requires documented
  consent and authorization. The Phase 0/1 app ships with NO reference voice
  configured for exactly this reason (`configs/baseline.yaml: inference.prompt`
  is empty by default).
- No private audio, model weights, or credentials are ever committed
  (`.gitignore` enforces the file-type and directory rules).

Full governance requirements: PRD §8. This file is a pointer, not the
authoritative policy; Phase 3 dataset work must re-read the PRD section.
