# Phase 3 feasibility verdict — Tesla T4 (free Colab), 2026-10-17

- **Verdict: INSUFFICIENT** (seed 3407)
- Ceiling 1 (GPU): official fp32 path OOMs at 14.39 GiB allocated with batch 1x16 / max_batch_tokens 2048 — both already at their configured floors.
- Ceiling 2 (host RAM): even with VM-only patch v3 (GPU peak 6,879 MiB — fits), the loader's CPU stage reached ~10.8 GB RSS and was SIGKILLed twice; free Colab host RAM is ~12.7 GB.
- Patch lineage (VM-only, never committed): v1 VAE dtype mismatch -> v2 AMP scaler bf16-grad rejection -> v3 fit GPU, killed at load by host RAM.
- **Decision: full run moves to A100 (owner-approved) on the OFFICIAL unpatched fp32 path.**
- Probe GPU spend: ~10 min GPU reservation total across 7 short instrumented probes in one free-T4 session (sum of run train walls 491.6 s, plus model-load phases).

**Amendment 2026-10-21 — verdict superseded.** Both ceilings were later refuted on the same free T4 under VM-only patch v4: probe `experiments/2026-10-17_phase3-probe-completion.md` (50 iters, GPU peak 7,703 MiB, child RSS 9,260 MiB, driver returncode 0) and production run `experiments/2026-10-21_phase3-fullrun-complete.md` (step 1000 reached 2026-10-21 22:54:39). The A100 decision above was not executed; Phase 3 GPU spend stayed $0. The INSUFFICIENT text is retained unchanged as the record of the official-path result.
