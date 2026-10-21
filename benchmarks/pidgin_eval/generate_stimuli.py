#!/usr/bin/env python3
"""Phase 4 stimulus generator — blind A/B: untouched baseline vs voxcpm2-pidgin-v0.

Generates the 20 Pidgin passages and the 5 English regression passages with both
systems (sequential model loads to stay inside T4 VRAM), then renames outputs to
blind A/B letters per passage and writes the mapping to key.json.

Protocol: benchmarks/pidgin_eval/README.md
Settings mirror the app/production path: style token "(documentary)" (override
with --style ""), cfg 2.0, timesteps 10, seed 3407. Only the weights differ.

Usage (Colab, repo root):
    python benchmarks/pidgin_eval/generate_stimuli.py \
        --out /content/drive/MyDrive/naturalvoice-ai-phase3/eval-20261022 \
        --artifact /content/drive/MyDrive/naturalvoice-ai-phase3/voxcpm2-pidgin-v0

Resumable: --skip-existing reuses already-generated wavs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PASSAGES = REPO / "benchmarks" / "pidgin_eval" / "passages.json"
DEFAULT_ENGLISH = REPO / "benchmarks" / "pidgin_eval" / "english_regression.json"

INSTRUCTIONS = """# Blind listening test — NaturalVoice AI (Pidgin v0)

You are rating **20 short passages**. For each passage you hear two recordings,
**A** and **B**. They differ only in the model weights; one is the English
baseline, the other is the Pidgin-tuned model. You are not told which is which.

For every passage, rate each recording on two scales (1-5):

- **Naturalness** — does it sound like a human speaking, with natural rhythm?
  1 = very robotic/unnatural ... 5 = completely natural.
- **Pidgin authenticity** — does it sound like real Nigerian Pidgin rather than
  English read with an accent? 1 = pure English ... 5 = convincing Pidgin.

Then mark your **preference** for the passage: A, B, or tie.

Rules:
- Use headphones if possible; listen to both recordings fully before rating.
- Rate what you hear, not what you expect. There are no right answers.
- Do not open `key.json` or look for system names — the test is blind.

Save your filled sheet as `ratings_listener-<yourname>.csv` in the `ratings/`
folder (keep the template's column names unchanged). Send it back to the owner.

Consent: this is an evaluation of synthetic speech only. No recordings of your
voice are made or used. Your name is stored only to distinguish raters in the
analysis (use a pseudonym if you prefer).
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="output directory (Drive recommended)")
    p.add_argument("--artifact", required=True,
                   help="tuned checkpoint dir containing lora_config.json + lora_weights.safetensors")
    p.add_argument("--passages", default=str(DEFAULT_PASSAGES))
    p.add_argument("--english", default=str(DEFAULT_ENGLISH))
    p.add_argument("--model-id", default="openbmb/VoxCPM2")
    p.add_argument("--style", default="documentary",
                   help='narration style token; "" = raw model voice')
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--only", choices=["all", "pidgin", "english"], default="all")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_voxcpm():
    src = os.environ.get("VOXCPM_SRC", "/content/OpenBMB-VoxCPM/src")
    if Path(src).is_dir() and src not in sys.path:
        sys.path.insert(0, src)
    from voxcpm import VoxCPM  # noqa: WPS433 (deliberate late import)
    return VoxCPM


def model_sample_rate(model) -> int:
    for obj in (model, getattr(model, "tts_model", None)):
        if obj is None:
            continue
        for attr in ("sample_rate", "output_sample_rate", "sampling_rate"):
            value = getattr(obj, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    return 48000


def flatten(wav) -> np.ndarray:
    if isinstance(wav, (list, tuple)):
        parts = [np.asarray(w, dtype=np.float32).reshape(-1) for w in wav]
        parts = [p for p in parts if p.size]
        if not parts:
            raise RuntimeError("model returned no audio samples")
        return np.concatenate(parts)
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def generate_set(model, items, style, seed, timesteps, cfg, out_dir: Path,
                 skip_existing: bool) -> int:
    sr = model_sample_rate(model)
    for item in items:
        dst = out_dir / f"{item['id']}.wav"
        if skip_existing and dst.is_file():
            print(f"[skip] {dst.name}")
            continue
        text = f"({style}){item['text']}" if style else item["text"]
        t0 = time.time()
        wav = flatten(model.generate(text, cfg_value=cfg,
                                     inference_timesteps=timesteps, seed=seed))
        if wav.size == 0:
            raise RuntimeError(f"empty audio for {item['id']}")
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), wav, sr, subtype="PCM_16")
        print(f"[gen] {dst.name}: {wav.size / sr:.2f}s @ {sr} Hz "
              f"in {time.time() - t0:.1f}s (peak {abs(wav).max():.3f})")
    return sr


def blind_map(items, seed, tmp_root: Path, src_set: str, dst_dir: Path) -> dict:
    mapping = {}
    for item in items:
        rng = random.Random(f"{seed}:{item['id']}")
        tuned_is_a = rng.random() < 0.5
        systems = {"A": "tuned" if tuned_is_a else "baseline",
                   "B": "baseline" if tuned_is_a else "tuned"}
        mapping[item["id"]] = systems
        for letter, system in systems.items():
            src = tmp_root / system / src_set / f"{item['id']}.wav"
            if not src.is_file():
                raise FileNotFoundError(f"missing {src} — generation incomplete")
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / f"{item['id']}_{letter}.wav")
    return mapping


def write_listener_assets(out: Path, passages: list) -> None:
    rows = ["listener_id,passage_id,A_naturalness_1_5,A_authenticity_1_5,"
            "B_naturalness_1_5,B_authenticity_1_5,preference_A_B_tie,comments"]
    rows += [f",{p['id']},,,,,," for p in passages]
    (out / "ratings").mkdir(parents=True, exist_ok=True)
    (out / "ratings" / "ratings_TEMPLATE.csv").write_text("\n".join(rows) + "\n",
                                                          encoding="utf-8")
    (out / "instructions.md").write_text(INSTRUCTIONS, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    tmp = out / "tmp"
    passages = json.loads(Path(args.passages).read_text(encoding="utf-8"))["passages"]
    english = json.loads(Path(args.english).read_text(encoding="utf-8"))["passages"]
    artifact = Path(args.artifact)
    if not (artifact / "lora_weights.safetensors").is_file():
        raise SystemExit(f"artifact missing lora_weights.safetensors: {artifact}")

    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_id": args.model_id,
        "style": args.style,
        "seed": args.seed,
        "inference_timesteps": args.timesteps,
        "cfg_value": args.cfg,
        "artifact": str(artifact),
        "lora_sha256": sha256_file(artifact / "lora_weights.safetensors"),
        "repo_commit": repo_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    VoxCPM = load_voxcpm()
    systems = [("baseline", None), ("tuned", str(artifact))]
    sample_rate = 0
    for name, lora in systems:
        print(f"\n=== system: {name} (lora={lora}) ===")
        model = VoxCPM.from_pretrained(hf_model_id=args.model_id, load_denoiser=False,
                                       optimize=False, lora_weights_path=lora)
        if args.only in ("all", "pidgin"):
            sample_rate = generate_set(model, passages, args.style, args.seed,
                                       args.timesteps, args.cfg,
                                       tmp / name / "pidgin", args.skip_existing)
        if args.only in ("all", "english"):
            sample_rate = generate_set(model, english, args.style, args.seed,
                                       args.timesteps, args.cfg,
                                       tmp / name / "english", args.skip_existing)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    key = {"seed": args.seed,
           "note": "A/B mapping — owner only; never share with listeners",
           "pidgin": {}, "english": {}}
    if args.only in ("all", "pidgin"):
        key["pidgin"] = blind_map(passages, args.seed, tmp, "pidgin", out / "audio")
        write_listener_assets(out, passages)
    if args.only in ("all", "english"):
        key["english"] = blind_map(english, args.seed, tmp, "english",
                                   out / "english_audio")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")

    meta.update({
        "sample_rate": sample_rate,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_pidgin": len(passages) if args.only in ("all", "pidgin") else 0,
        "n_english": len(english) if args.only in ("all", "english") else 0,
    })
    (out / "manifest.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\ndone -> {out}")
    print("listeners get: audio/, instructions.md, ratings/ratings_TEMPLATE.csv")
    print("owner keeps: key.json, manifest.json, english_audio/, tmp/")


if __name__ == "__main__":
    main()
