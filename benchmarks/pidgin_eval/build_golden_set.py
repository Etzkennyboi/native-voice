#!/usr/bin/env python3
"""Freeze the golden-v0 reference set from the Phase 4 blind evaluation.

The author's blind A/B pass over `eval-20261022-raw` marked 13 of 20 passages as
good-sounding; all 13 decoded to the tuned system. Those 13 clips become the
relative quality bar ("golden-v0") that every later checkpoint must match or
beat. The other 7 passages stay in the set as coverage checks (pidgin-012 is the
known tail-babble case).

This script reads the committed selection (`golden_set.json`), resolves the
tuned side of each passage from the evaluation `key.json`, copies that clip to
`<out>/audio/<passage>_tuned.wav`, and writes a manifest with sha256/durations.

Audio stays on Drive and is never committed (PRD §8). The selection manifest is
committed; the generated manifest is the evidence copy.

Usage (Colab, repo root):
    python benchmarks/pidgin_eval/build_golden_set.py \
        --eval-dir /content/drive/MyDrive/naturalvoice-ai-phase3/eval-20261022-raw \
        --out /content/drive/MyDrive/naturalvoice-ai-phase3/golden-set-v0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION = REPO / "benchmarks" / "pidgin_eval" / "golden_set.json"
PASSAGES = REPO / "benchmarks" / "pidgin_eval" / "passages.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", required=True,
                   help="evaluation dir containing key.json and audio/")
    p.add_argument("--out", required=True, help="golden-set output dir (Drive)")
    p.add_argument("--selection", default=str(DEFAULT_SELECTION))
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out = Path(args.out)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    key = json.loads((eval_dir / "key.json").read_text(encoding="utf-8"))["pidgin"]
    texts = {p["id"]: p["text"] for p in
             json.loads(PASSAGES.read_text(encoding="utf-8"))["passages"]}

    problems: list[str] = []
    clips: list[dict] = []
    for entry in selection["clips"]:
        passage = entry["id"]
        systems = key.get(passage)
        if not systems:
            problems.append(f"{passage}: absent from key.json")
            continue
        tuned_sides = [side for side, name in systems.items() if name == "tuned"]
        if len(tuned_sides) != 1:
            problems.append(f"{passage}: expected exactly one tuned side, got {systems}")
            continue
        side = tuned_sides[0]
        picked = entry.get("picked_side")
        if picked and systems.get(picked) != "tuned":
            problems.append(f"{passage}: picked side {picked} is "
                            f"{systems.get(picked)}, expected tuned")
            continue
        src = eval_dir / "audio" / f"{passage}_{side}.wav"
        if not src.is_file():
            problems.append(f"{passage}: missing {src}")
            continue
        dst = out / "audio" / f"{passage}_tuned.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        import soundfile as sf  # late import: metadata-only runs stay cheap

        info = sf.info(str(dst))
        clips.append({
            "id": passage,
            "tier": entry["tier"],
            "system": "tuned",
            "blind_side_in_source": side,
            "source": f"audio/{passage}_{side}.wav",
            "audio": f"audio/{passage}_tuned.wav",
            "sha256": sha256_file(dst),
            "bytes": dst.stat().st_size,
            "duration_s": round(info.frames / info.samplerate, 3),
            "sample_rate": info.samplerate,
            "text": texts.get(passage, ""),
            "note": entry.get("note", ""),
        })

    if problems:
        raise SystemExit("golden set incomplete:\n  " + "\n  ".join(problems))

    golden = [c for c in clips if c["tier"] == "golden"]
    coverage = [c for c in clips if c["tier"] == "coverage"]
    total_bytes = sum(c["bytes"] for c in clips)
    manifest = {
        "set": selection["set"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint_id": selection["checkpoint_id"],
        "artifact_sha256": selection["artifact_sha256"],
        "source_eval": str(eval_dir),
        "protocol": selection["protocol"],
        "pilot": selection["pilot"],
        "counts": {"golden": len(golden), "coverage": len(coverage),
                   "total": len(clips)},
        "total_mb": round(total_bytes / 2 ** 20, 2),
        "clips": clips,
    }
    (out / "golden_set_manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    readme = f"""# golden-v0 — perceived-quality bar for Pidgin checkpoints

Frozen {manifest['created']} from `{eval_dir.name}` (raw voice, no style tag).

- {len(golden)} **golden** clips: the blind pilot's good-sounding set (all tuned).
- {len(coverage)} **coverage** clips: not selected; `pidgin-012` is the known
  tail-babble case and the primary regression target.
- All {len(clips)} clips are `{selection['checkpoint_id']}` at raw voice,
  cfg {selection['protocol']['cfg_value']}, timesteps {selection['protocol']['inference_timesteps']},
  seed {selection['protocol']['seed']}.

Use: generate the same 20 passages with a candidate checkpoint, pair candidate
vs golden-v0 per passage, re-randomize A/B, and have >=3 Pidgin speakers rate
both. The candidate must match/beat v0 on the golden 13 and fix the coverage 7
(no babble, WER improves or holds). This is a relative bar, not a license for a
"sounds Pidgin" claim — that still needs the three-listener gate.

Audio is stored here, never committed. `golden_set_manifest.json` has the
sha256 and duration of every clip.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    print(f"golden-v0 -> {out}")
    print(f"counts   : {manifest['counts']} | {manifest['total_mb']} MB")
    print(f"golden   : {[c['id'] for c in golden]}")
    print(f"coverage : {[c['id'] for c in coverage]}")
    print(f"manifest : {out / 'golden_set_manifest.json'}")
    print("audio stays on Drive and is never committed (PRD §8).")


if __name__ == "__main__":
    main()
