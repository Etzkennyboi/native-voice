# Phase 3 API + schema verification (2026-10-17, zero GPU cost)

Purpose per PRD §9/§10: record what was verified before any fine-tune work. Owner
decisions recorded separately in `configs/pidgin_baseline.yaml` (decisions block).

## What we verified (package introspection + official repo)

| Fact | Source |
|---|---|
| Fine-tuning driver = `scripts/train_voxcpm_finetune.py --config_path <yaml>` | OpenBMB/VoxCPM GitHub README + repo tree `scripts/` |
| Training building blocks inside pip package: `voxcpm/training/` (data, packers, Accelerator, TrainingState, tracker, validate) | Stage 1 introspection on Colab runtime (pip voxcpm 2.0.3) |
| Manifest schema: `{"audio": <path>`, `"text": <str>}`; optional `duration` (s), `dataset_id`; loader casts audio to 16 kHz itself | `examples/train_data_example.jsonl` + `src/voxcpm/training/data.py` (`load_audio_text_datasets`) |
| LoRA recipe (VoxCPM2): r=32, α=32, dropout=0.0, enable_lm/enable_dit+, enable_proj−; batch 2 × accum 8; lr 1e-4; wd 0.01; warmup 100; 1000 iters; max_grad_norm 1.0; 16k in → 48k out | `conf/voxcpm_v2/voxcpm_finetune_lora.yaml` (fetched raw) |
| LoRA needs as little as **5–10 minutes** of audio; full SFT also supported (`voxcpm_finetune_all.yaml`) | https://huggingface.co/openbmb/VoxCPM2 model card |
| Inference-time LoRA loading verified path: `lora_weights_path` param of `VoxCPM.from_pretrained` + reference impl `scripts/test_voxcpm_lora_infer.py` | Phase 0/1 introspection + repo `scripts/` |
| **License gate L3 CLEARED**: base checkpoint + code = Apache-2.0 ("Fully Open-Source & Commercial-Ready"; fine-tuning officially supported) | HF model card front-matter `license: apache-2.0` + GitHub `LICENSE` full text |

## Gates still closed (hard, before Stage 3+)

- [ ] L1/L2: read the Common Voice pcm bundled licence/consent docs (they ship inside the
      download; Stage 2 `find` locates them) → then flip `license_cleared` +
      `consent_review_done` in `configs/pidgin_baseline.yaml`.

## Outcome

`notebooks/03_pidgin_finetune.ipynb` Stages 4–5 rebuilt around the verified driver:
feasibility = 40 clips / 50 iters / official recipe; full run = config
`configs/pidgin_voxcpm2_ft_v0.yaml` (written on the runtime from the training block of
`configs/pidgin_baseline.yaml`), gated manually on the feasibility record.
