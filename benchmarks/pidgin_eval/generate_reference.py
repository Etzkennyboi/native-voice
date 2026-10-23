#!/usr/bin/env python3
"""Generate Pidgin passages with a reference voice (zero-shot conditioning).

VoxCPM2 supports `reference_wav_path`: the clip is encoded as an isolated voice
prefix and the target text is generated in that voice. This is the fast way to
turn one of the golden-v0 clips into a consistent house voice without training.

This is a development experiment, not the human gate. Compare the output with
`golden-set-v0/audio/` by ear; if it wins, wire `reference_wav_path` into the
app config.

HOST-RAM NOTE: same free-Colab constraint as the Phase 4 generator — run on the
clone with the VM-only patch v4 (bf16 construction) restored.

Usage (Colab, repo root):
    python benchmarks/pidgin_eval/generate_reference.py \
        --out /content/drive/MyDrive/naturalvoice-ai-phase3/ref-008-20261023 \
        --artifact /content/drive/MyDrive/naturalvoice-ai-phase3/voxcpm2-pidgin-v0 \
        --reference /content/drive/MyDrive/naturalvoice-ai-phase3/golden-set-v0/audio/pidgin-008_tuned.wav
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
PASSAGES = REPO / "benchmarks" / "pidgin_eval" / "passages.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="output dir (Drive recommended)")
    p.add_argument("--artifact", required=True,
                   help="tuned checkpoint dir (lora_config.json + lora_weights.safetensors)")
    p.add_argument("--reference", required=True, help="reference wav (voice prefix)")
    p.add_argument("--passages", default=str(PASSAGES))
    p.add_argument("--ids", default="", help="comma-separated passage ids (default: all)")
    p.add_argument("--model-id", default="openbmb/VoxCPM2")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--cfg", type=float, default=2.0)
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


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    ref = Path(args.reference)
    artifact = Path(args.artifact)
    if not ref.is_file():
        raise SystemExit(f"reference wav missing: {ref}")
    if not (artifact / "lora_weights.safetensors").is_file():
        raise SystemExit(f"artifact missing lora_weights.safetensors: {artifact}")

    patched = Path(os.environ.get("VOXCPM_SRC", "/content/OpenBMB-VoxCPM/src")) / \
        "voxcpm" / "model" / "voxcpm2.py"
    if patched.is_file() and "ntv-t4-v4" not in patched.read_text(encoding="utf-8", errors="ignore"):
        print("[warn] patch v4 marker not found — the stock loader may exceed free-Colab RAM",
              flush=True)

    items = json.loads(Path(args.passages).read_text(encoding="utf-8"))["passages"]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [p for p in items if p["id"] in wanted]
        missing = wanted - {p["id"] for p in items}
        if missing:
            raise SystemExit(f"unknown passage ids: {sorted(missing)}")

    ref_info = sf.info(str(ref))
    print(f"[ref] {ref.name}: {ref_info.frames / ref_info.samplerate:.2f}s @ "
          f"{ref_info.samplerate} Hz, sha256 {sha256_file(ref)[:12]}", flush=True)

    VoxCPM = load_voxcpm()
    model = VoxCPM.from_pretrained(hf_model_id=args.model_id, load_denoiser=False,
                                   optimize=False, lora_weights_path=str(artifact))
    sr = model_sample_rate(model)
    clips = []
    for item in items:
        dst = out / "audio" / f"{item['id']}.wav"
        if args.skip_existing and dst.is_file():
            print(f"[skip] {dst.name}", flush=True)
        else:
            t0 = time.time()
            wav = flatten(model.generate(
                item["text"], reference_wav_path=str(ref),
                cfg_value=args.cfg, inference_timesteps=args.timesteps, seed=args.seed))
            if wav.size == 0:
                raise RuntimeError(f"empty audio for {item['id']}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), wav, sr, subtype="PCM_16")
            print(f"[gen] {dst.name}: {wav.size / sr:.2f}s in {time.time() - t0:.1f}s",
                  flush=True)
        info = sf.info(str(dst))
        clips.append({"id": item["id"], "audio": f"audio/{item['id']}.wav",
                      "sha256": sha256_file(dst),
                      "duration_s": round(info.frames / info.samplerate, 3)})

    manifest = {
        "kind": "reference_conditioning",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reference": {"path": str(ref), "sha256": sha256_file(ref),
                      "duration_s": round(ref_info.frames / ref_info.samplerate, 3)},
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact / "lora_weights.safetensors"),
        "model_id": args.model_id,
        "settings": {"style": "", "cfg_value": args.cfg,
                     "inference_timesteps": args.timesteps, "seed": args.seed},
        "repo_commit": repo_commit(),
        "n": len(clips),
        "clips": clips,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\ndone -> {out} ({len(clips)} clips)")
    print("compare by ear with golden-set-v0/audio/ before wiring anything into the app.")


if __name__ == "__main__":
    main()
