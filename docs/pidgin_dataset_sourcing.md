# Nigerian Pidgin dataset sourcing report (Phase 3 gate 1, PRD §8)

Date: 2026-10-17 · Author: Phase 3 adapted scope (approved scope addition; PRD §3.3 excluded
multi-language at launch — this track is isolated from the English baseline by design).

## Rule zero

No scraped audio (YouTube/social/etc.) without verified rights. All candidates below
carry publisher licenses; none are scraped. (Task constraint honored — explicitly noted.)

## Candidates considered

| # | Dataset | Content | License (as published) | Verdict |
|---|---|---|---|---|
| 1 | **Mozilla Common Voice 25.0 — Nigerian Pidgin (pcm)** ([Data Collective page](https://datacollective.mozillafoundation.org/datasets/…zg4mz)) | 8,944 read clips / 12.5 h validated / 59 speakers / 987 sentences | **CC0-1.0** | **CHOSEN (primary)** |
| 2a | **Naija-TTS-Dataset** (Mozilla Data Collective, 2026-04-14) | 1,987 TTS-suitable read clips / 5 h 55 m, utterances 1–38 s, 16 folders + prompt audio | **NOODL-1.0 (Nwulite Obodo Open Data Licence)** — terms not yet read | **SECONDARY — gated on NOODL-1.0 verification** |
| 2b | **Waxal-TTS / Media Trust — Nigerian Pidgin** (via BolajiJsAI/WaxalNLP) | TTS-grade single-speaker phonetically balanced reads | **CC-BY-SA-4.0** (per provider table) | **TERTIARY — verify actual clip availability for pcm** |
| 2c | **NAC-v2.0** (`voicedata/pidginData`, derived from UD_Naija-NSC) | Spontaneous conversational clips, 16 kHz mono, speaker codes, full provenance | **CC-BY-SA-4.0** | **RESERVE** for a future spontaneous-speech track. Caution: source recordings' consent was for corpus research — **do not assume it covers voice-replication use** without a consent review |
| 3 | **Wilderness / bible.is PCMTSC** (15 h 10 m) | Bible readings | Unclear (bible.is redistribution terms) | **REJECTED** — cannot document license/consent for voice use |
| 4 | ngano listings (e.g. `Twelve2five/nigerian-pidgin-tts-dataset`, `WAZOBIALABS/…`, `Rexe/…`) | Mixed | **Unstated** | **REJECTED** — license unknown by publisher's own admission |
| 5 | DexterityLearn pidgin voice v1 | 45 clips / ~5 min | Research-only, cloning prohibited | **REJECTED** — far too small, no voice-replication rights |
| 6 | Scraping YouTube/social | n/a | — | **PROHIBITED by policy** (task constraint; PRD §8) |

## Choice rationale (primary = Common Voice pcm)

1. **License quality:** CC0 = public domain — the cleanest possible ML-training posture;
   no attribution trap, no share-alike contagion to our tuned checkpoint.
2. **Consent posture:** recorded under Mozilla's structured consent + release flow with
   a public, documented ToS; speakers are anonymized (dataset forbids re-identifying them).
3. **Multi-speaker:** 59 speakers → speaker-disjoint splits are actually possible,
   which satisfies §8's train/test separation honestly.
4. **Volume:** 12.5 validated hours is comfortably above the ~6–8 h felt "sweet spot"
   for a LoRA-style adaptation on a 2B checkpoint.

**Residual consent questions still to be documented before training (hard gate — not hand-waved):**
- [ ] Read + file the CC0 licence text (trivially yes, but record WHERE it was read from).
- [ ] Confirm Mozilla's ToS wording covers **training generative speech models** (not just ASR).
- [ ] Note that CC0 + anonymized multi-speaker does NOT threaten to replicate any single
      person's voice (59 mixed speakers, no single-voice cloning). Record this reasoning.

## Open licensing items (BLOCKERS before any full fine-tune)

| ID | Item | Who unblocks it |
|---|---|---|
| L1 | Download + save-to-repo `configs` the CV pcm CC0 licence text | PM/Emmanuel (login on Data Collective, download TSVs) |
| L2 | Read NOODL-1.0 text → decide whether it clears TTS training + voice use | research + PM sign-off |
| L3 | Confirm VoxCPM2 base checkpoint's own license permits derivative fine-tuning at all | license check before ANY paid run |
| L4 | Fine-tuning API + manifest schema — must be introspected on the GPU runtime **before** manifests are built (see `notebooks/03_pidgin_finetune.ipynb`, verification cell; zero GPU cost) | next GPU session |

## What will NOT be done

- No unlicensed substitute will be used if a license check fails. Substitute is:
  **record our own** (consent forms, PRD §8 wording) — slower, smaller, but clean.
- No claim of "sounds Nigerian Pidgin" without human listening from ≥3 actual Pidgin
  speakers (hard constraint; PRD §5.5).
