# Phase 4 automatic metrics — WER + failure-mode audit (2026-10-23)

Status: automatic metrics complete; **human A/B pending**. No quality claim.

- Artifact: `voxcpm2-pidgin-v0`, step 1000, `lora_weights.safetensors` sha256 `edd8a79adbf134d989d5112b05c4d48411b7d50e8398263059691a949b13f9a7` (69.04 MiB), base `openbmb/VoxCPM2` @ `32279effe8c19989596f05d353d1447f51d9e915` (unchanged).
- Settings (both systems): cfg 2.0, `inference_timesteps` 10, seed 3407, `load_denoiser=False`, `optimize=False`; only the weights differ.
- Recognizer: `facebook/mms-1b-all` with the `pcm` / `eng` adapters (CC-BY-NC 4.0 — research evaluation only). WER is an *indication*, not proof.
- Stimuli: 20 Pidgin passages (`benchmarks/pidgin_eval/passages.json`) and 5 English regression passages (`english_regression.json`), blind A/B keyed on Drive at `eval-20261022-raw/key.json` and `eval-20261022/key.json`.
- **Primary run: raw voice (`--style ""`).** The tuned LoRA was trained on raw transcripts; the control-prefix run below is a diagnostic, not the primary evaluation.

## Pidgin (raw voice) — primary

| system | mean WER | median WER | probe-word hits |
|---|---|---|---|
| baseline | 0.410 | 0.413 | 11/27 (0.407) |
| tuned | 0.456 | 0.442 | 18/27 (0.667) |

- Per passage: tuned better 5, tie 2, worse 13.
- Largest gains: pidgin-010 (0.68 → 0.45), pidgin-009 (0.48 → 0.30), pidgin-014 (0.61 → 0.43), pidgin-001 (0.48 → 0.35), pidgin-011 (0.39 → 0.35).
- Largest regressions: pidgin-012 (0.39 → 0.70), pidgin-015 (0.43 → 0.71), pidgin-019 (0.54 → 0.71), pidgin-020 (0.57 → 0.74).
- Probe words: tuned 0.667 vs baseline 0.407 — the clear positive signal. Sample transcripts: tuned renders “wetin dey happen” and “abeg no vex” correctly where the baseline hears “wher din dey happen” / “i beg no vex”.

## Failure-mode audit (durations)

- Tuned is slower: 13/20 clips longer. Overruns > 1.35× baseline: pidgin-002 (1.74), 007 (1.51), 012 (1.91), 013 (1.42), 020 (1.56).
- Transcript audit:
  - 002 / 007 / 013 end cleanly — slower delivery, not degeneration.
  - **012 shows a clear babble tail**: “... na 12 000 nera boddo je but de towewr de jrupeboy”, 17.4 s vs 9.1 s baseline, plus a number error (12 000 vs two thousand).
  - 020 has only a trailing syllable (“... tumoro norday k”).
- Conclusion: one clear stop/termination failure in 20 (012), one mild (020); most duration growth is slower speech.

## English regression (raw voice)

| system | mean WER | median WER |
|---|---|---|
| baseline | 0.011 | 0.000 |
| tuned | 0.073 | 0.095 |

- Errors are small word/pronunciation substitutions (“eocean”, “moni”, “as seven”, “pleas”, “rainfol”), no babble; eng-003 tuned 6.2 s vs baseline 3.5 s (same words, slower).
- Absolute level stays low, but the relative degradation is real and touches the ship rule’s English-regression guard.

## Diagnostic: control-prefix (`(documentary)`) run

- Same 20 passages with the `({style})` prefix: Pidgin mean WER baseline 0.417 / tuned 0.541; pidgin-006 tuned 14.88 s, WER 1.00 and pidgin-009 15.68 s, WER 1.17 — both coherent for one sentence, then long gibberish tails.
- Raw re-generation of the same two passages: 9.28 s / WER 0.29 and 8.16 s / WER 0.30.
- Conclusion: the control prefix is out-of-distribution for the LoRA and triggers stop failure; it is not a base-checkpoint defect.
- **Limitation on record:** `voxcpm2-pidgin-v0` must be used with raw voice. Style/control tags are unsupported until a retrain that includes them.

## Gate status

- Automatic checks do **not** support a ship claim: mean WER worse, English regression present, one clear degeneration case.
- Human A/B (≥ 3 Pidgin speakers, blind) is mandatory and still pending; it is the deciding evidence. Listener kit: `eval-20261022-raw/listener_kit/` (audio + instructions + template only; `key.json` withheld).
