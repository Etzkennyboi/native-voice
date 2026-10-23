# Phase 4 pilot — author blind listening pass (2026-10-23)

Status: **pilot only, not the human gate.** One listener, preference-only, 13/20
passages. The formal gate still requires >= 3 complete listeners with both rating
scales.

## Method

- Stimuli: `eval-20261022-raw`, 20 Pidgin passages, raw voice (`--style ""`),
  baseline vs `voxcpm2-pidgin-v0`.
- Blind A/B per passage; the key was withheld until after the pass.
- Listener: project author (Pidgin target user; fluency not independently
  verified).
- Task: mark the recording that sounds best. 13 of 20 passages were marked with
  a side (A/B); 7 were left unselected.

## Result

- **13/13 picked sides decoded to the tuned system.**
- Sides were mixed (7 A, 6 B), so there is no A/B side bias.
- Under random guessing, 13/13 has one-sided p = 2^-13 ≈ **0.00012**.
- Caveat: 13/20 were selected, likely the cases the listener felt confident
  about; this is a preference signal on the clear cases, not a random sample of
  all 20.

## The WER dissociation

Of the 13 picks, 9 were passages where MMS scored the tuned model **worse**:

| passage | tuned - baseline WER | picked |
|---|---|---|
| pidgin-001 | -0.13 | tuned |
| pidgin-002 | +0.09 | tuned |
| pidgin-005 | 0.00 | tuned |
| pidgin-006 | +0.10 | tuned |
| pidgin-008 | +0.09 | tuned |
| pidgin-010 | -0.23 | tuned |
| pidgin-011 | -0.04 | tuned |
| pidgin-015 | +0.28 | tuned |
| pidgin-016 | +0.13 | tuned |
| pidgin-017 | +0.13 | tuned |
| pidgin-018 | +0.05 | tuned |
| pidgin-019 | +0.17 | tuned |
| pidgin-020 | +0.17 | tuned |

- 3 better, 1 tie, **9 worse** on WER — yet all 13 were preferred.
- The 7 unselected passages include the two largest WER *improvements* for tuned
  (pidgin-009 and pidgin-014, both -0.18) and the clear babble case
  (pidgin-012, +0.31). Preference is not a monotone function of WER in either
  direction.
- Interpretation: MMS-pcm is not measuring perceived Pidgin quality. The tuned
  model is slower (13/20 clips longer) and hits more probe words (18/27 vs
  11/27); listeners may reward both. WER remains an indication only.

## Limits

- One listener, not independent (project author), preference only (no 1-5
  scales), 13/20 coverage.
- Possible expectation bias despite the blind key; the formal independent
  listeners are still the deciding evidence.
- The tuned model is slower: pidgin-002 1.74x, 007 1.51x, 012 1.91x, 013 1.42x,
  020 1.56x. Slower delivery can read as more natural, independent of Pidgin
  authenticity.
- This does **not** satisfy the ship rule: raw mean WER is still worse
  (0.456 vs 0.410), English regressed (0.073 vs 0.011), and pidgin-012 babbles.

## Consequence

- Prior shifts: v0 likely has a real **perceived** Pidgin win despite the worse
  WER and English drift. A "ship with caveats" outcome is now plausible if the
  formal gate confirms it.
- The 13 picks are frozen as `golden-v0` in
  `benchmarks/pidgin_eval/golden_set.json`: 13 golden clips plus 7 coverage
  clips (pidgin-012 flagged as the primary regression target). Every later
  checkpoint must match or beat this bar.

## Next

1. 2-3 fresh Pidgin speakers rate all 20 passages on naturalness and authenticity
   (plus preference).
2. Run `score_human.py`; fold the result into the Phase 4 verdict.
