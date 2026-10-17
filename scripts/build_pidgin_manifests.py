#!/usr/bin/env python3
"""Build Nigerian Pidgin (pcm) train/val/test manifests (PRD §8 / Phase 3).

HARD GATES — the script refuses to run until both are satisfied:
  1. License: configs/pidgin_baseline.yaml must mark the source dataset as
     license_cleared: true (done only after the licence text has actually been read, PRD §8).
  2. Schema: the manifest field names must match what VoxCPM2's fine-tuning pipeline
     expects. That is verified in notebooks/03_pidgin_finetune.ipynb (zero-GPU
     introspection cell) — then SCHEMA_KEYS below is updated. We never assume.

Input format: Common Voice-style validated TSV (auto-detects the
path/sentence/speaker columns), plus the audio directory the `path` column refers to.

Output (JSONL, one record per utterance):
  data/manifests/pidgin_train.jsonl / pidgin_val.jsonl / pidgin_test.jsonl
Splits are SPEAKER-DISJOINT (no speaker appears in two splits) and deterministic
(seed fixed in configs/pidgin_baseline.yaml).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Placeholder pending the verification cell — DO NOT assume these match VoxCPM2.
# Until verified, manifests use THIS repo's canonical schema (PRD §8 fields);
# after notebooks/03 introspection, either rename keys here or add an adapter that
# re-emits them in VoxCPM2's exact fine-tune format.
SCHEMA_KEYS = {
    "audio_path": "audio_path",   # UNVERIFIED — check against VoxCPM2 fine-tune schema
    "text": "text",               # UNVERIFIED
    "speaker_id": "speaker_id",   # UNVERIFIED
}


def load_split_config() -> dict:
    path = REPO / "configs" / "pidgin_baseline.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not cfg.get("data", {}).get("splits_locked", False):
        raise SystemExit(
            "GATE: configs/pidgin_baseline.yaml has splits_locked: false — "
            "set it true ONLY after the fine-tune schema verification cell has run."
        )
    if not cfg.get("license", {}).get("license_cleared", False):
        raise SystemExit(
            "GATE: license_cleared is false — the licence must be read and filed "
            "before manifests are built (PRD §8)."
        )
    return cfg


def detect_columns(header: list[str]) -> dict:
    low = [h.strip().lower() for h in header]
    def find(*aliases: str) -> str:
        for a in aliases:
            for h in low:
                if a == h:
                    return header[low.index(a)]
        for a in aliases:  # substring fallback
            for h in low:
                if a in h:
                    return header[low.index(a)]
        raise KeyError(f"no column matching {aliases} in {header}")
    return {"path": find("path", "filename", "clip"), "text": find("sentence", "transcript", "text"), "speaker": find("speaker", "client_id")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Common Voice-style validated .tsv")
    ap.add_argument("--audio-dir", required=True, help="directory holding clip files")
    ap.add_argument("--max-speakers", type=int, default=0, help="cap speakers (0 = all)")
    ap.add_argument("--align-schema-only", action="store_true",
                    help="print schema review summary and exit (verification aid)")
    args = ap.parse_args()

    cfg = load_split_config()
    with open(args.tsv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    cols = detect_columns(rows[0].keys() if rows else [])
    audio_dir = Path(args.audio_dir)
    seed = int(cfg["training"]["seed"])

    per_speaker: dict[str, list[dict]] = {}
    r = random.Random(seed)
    for row in rows:
        rel = row[cols["path"]]
        f = audio_dir / rel
        if not f.exists():
            continue
        try:
            info = sf.info(str(f))
        except Exception:
            continue  # unreadable/missing audio → skip, never crash the build
        if not row[cols["text"]].strip():
            continue
        per_speaker.setdefault(row[cols["speaker"]], []).append(
            {"file": rel, "text": row[cols["text"]].strip(),
             "seconds": round(info.duration, 3), "sr": info.samplerate}
        )

    speakers = sorted(per_speaker)
    r.shuffle(speakers)
    if args.max_speakers:
        speakers = speakers[: args.max_speakers]

    fr = cfg["data"].get("split_fractions", {"train": 0.85, "val": 0.07, "test": 0.08})
    counts = [
        max(1, int(len(speakers) * fr["train"])),
        max(1, int(len(speakers) * fr["val"])),
        max(1, int(len(speakers) * fr["test"])),
    ]
    while sum(counts) > len(speakers) and counts[0] > 1:
        counts[0] -= 1
    split_of = {}
    for spk, sp in zip(speakers, ["train"] * counts[0] + ["val"] * counts[1] + ["test"] * counts[2]):
        split_of[spk] = sp

    out = {"train": [], "val": [], "test": []}
    src = cfg["license"]["source_name"]
    lic = cfg["license"]["license_id"]
    for spk in speakers:
        sp = split_of[spk]
        for item in per_speaker[spk]:
            out[sp].append({
                "id": f"pcm-{spk}-{Path(item['file']).stem}",
                "audio_path": str(audio_dir / item["file"]),
                "text": item["text"],
                "speaker_id": spk,
                "duration_s": item["seconds"],
                "sampling_rate": item["sr"],
                "language": "pcm",
                "source": src,
                "license": lic,
                "approved_for_training": bool(cfg["license"].get("license_cleared", False)),
            })

    outdir = REPO / "data" / "manifests"
    outdir.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        target = outdir / f"pidgin_{sp}.jsonl"
        with open(target, "w", encoding="utf-8") as fh:
            for rec in out[sp]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        hours = sum(x["duration_s"] for x in out[sp]) / 3600
        print(f"pidgin_{sp}: {len(out[sp])} clips | ~{hours:.2f} h | {len({x['speaker_id'] for x in out[sp]})} speakers")

    train_spk = {x["speaker_id"] for x in out["train"]}
    test_spk = {x["speaker_id"] for x in out["test"]}
    val_spk = {x["speaker_id"] for x in out["val"]}
    overlap = (train_spk & val_spk) | (train_spk & test_spk) | (val_spk & test_spk)
    assert not overlap, "speaker overlap detected — split failed sanity check"
    print("speaker overlap check: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
