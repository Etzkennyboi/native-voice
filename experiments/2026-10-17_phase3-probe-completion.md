# Phase 3 probe completion — patch v4 clears both T4 ceilings (2026-10-17)
- Official recipe on free T4 + VM-only patch v4 (bf16 frozen base / fp32 LoRA master / fp32 VAE):
  50 iters / 40 clips / batch 1x16 / max_batch_tokens 2048 -> driver_returncode 0.
- Peaks: GPU 7,703 MiB | child RSS 9,260 MiB (free-Colab 12.7 GB floor cleared; v3 died at 10,840).
- Loss trend: loss/diff 0.924 -> 0.777; grad_norm 0.051-0.109; ~16.7 s/step measured.
- Supersedes 'phase3-t4-verdict.md': Ceiling 2 (host RAM) is FALSE under patch v4; Ceiling 1 (GPU)
  is FALSE under patch v4 numerics. Production run therefore proceeds on FREE T4 at $0.
- Note: the probe resumed from an earlier interrupted attempt (step 17); the production run uses a
  fresh save_path (pidgin-v0) and starts at step 0.
- Full run: configs/pidgin_voxcpm2_ft_v0.yaml, 1000 iters, warmup 100, save every 250, ETA ~4.6 h.

## Addendum 2026-10-21

- The production run did not stay on one session: the original session was lost 2026-10-18 at
  ~step 340, a second resume was lost with its session (guardian not re-armed), and the third
  session (2026-10-19, guardian v2) completed from `step_0000250` to step 1000 on
  2026-10-21 22:54:39.
- Final artifact and full evidence: `experiments/2026-10-21_phase3-fullrun-complete.md`.
