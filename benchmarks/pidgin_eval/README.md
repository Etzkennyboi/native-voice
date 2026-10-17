# Pidgin benchmark evaluation protocol (Phase 3 adapted, PRD §9 + §5.5)

## What gets compared

Untouched English baseline (`configs/baseline.yaml`, snapshot
`32279effe8c19989596f05d353d1447f51d9e915`) **vs** Pidgin-tuned checkpoint
(`configs/pidgin_baseline.yaml`, checkpoint_id `voxcpm2-pidgin-v0`).

Same 20 passages (`passages.json`), same generation settings
(`inference_timesteps: 10`, `cfg_value: 2.0`, seed fixed), only weights differ.
No improvement may be claimed without this A/B (task constraint).

## Metric 1 — Intelligibility (automatic, imperfect — flagged)

- ASR: `facebook/mms-1b-all` with the `pcm` adapter if available (CC-BY-NC 4.0 —
  research evaluation only, never for training).
- **Flag on record:** no strong Pidgin-native recognizer is confirmed; MMS-pcm WER
  is an *indication*, not proof. Human listening below outranks it.
- Report WER per passage, mean per system. Also count **probe-word errors** on the
  `pidgin_specific_probe_words` list (pronunciation accuracy proxy).

## Metric 2 — Naturalness (human, mandatory)

- ≥ 3 Pidgin speakers (recruited; consent for evaluation recorded, no voice output used).
- Blind randomized A/B: listener hears tuned vs baseline of the same passage, does not
  know which is which.
- Ratings 1–5 on: (a) naturalness, (b) Pidgin authenticity ("e sound like real
  Pidgin talk, no be English with accent").
- Win rate + mean scores per system. Any "sounds Pidgin" claim requires this.

## Exit decision rule reported in the merge summary

- Ship candidate → tuned beats baseline on human naturalness **and** authenticity by
  clear margins, WER not worse, and no English-regression spot check fails
  (5 English passages from Phase 1 tests re-generated with tuned weights as a
  regression guard).
- Otherwise → another iteration with stated failure mode (data? prosody? probe words?).
