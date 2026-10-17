#!/usr/bin/env python3
"""Build Nigerian Pidgin (pcm) train/val/test manifests (PRD §8 / Phase 3).

HARD GATES — the script refuses to run until all are satisfied:
  1. license_cleared: true   in configs/pidgin_baseline.yaml (only after the licence
     documents have actually been read — PRD §8).
  2. consent_review_done: true  (same condition).
  3. splits_locked: true     (flipped after the Stage 1 schema verification, done 2026-10-17).

Two input modes:
  --tsv + --audio-dir        Common Voice-style validated TSV + its clips directory.
  --hf-parquet-dir           a downloaded HF-parquet dataset on disk (e.g. the pinned
                             snapshot of michaelodafe/pidgin-asr-combined). Audio bytes
                             are unpacked to --audio-dir. Rows whose `source` matches
                             --exclude-source are skipped (default: "Rexe" — license
                             unspecified in that source = hard exclusion, PRD §8).

Output: data/manifests/pidgin_{train,val,test}.jsonl in the VERIFIED VoxCPM2 format
  {"audio": <path>, "text": <str>, "duration": <seconds>} (+ traceability metadata).
Splits are SPEAKER-DISJOINT (no speaker or utterance appears in two splits) and
deterministic (seed pinned in configs/pidgin_baseline.yaml).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REQUIRED_GATES = ("license_cleared", "consent_review_done", "splits_locked")


def load_split_config() -> dict:
    import yaml
    path = REPO / "configs" / "pidgin_baseline.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    data_gates = {
        "license_cleared": bool(cfg["license"].get("license_cleared", False)),
        "consent_review_done": bool(cfg["license"].get("consent_review_done", False)),
        "splits_locked": bool(cfg["data"].get("splits_locked", False)),
    }
    for flag, ok in data_gates.items():
        if not ok:
            raise SystemExit(
                f"GATE: configs/pidgin_baseline.yaml has {flag}: false — "
                "flip it only after the required review/documentation (PRD §8)."
            )
    return cfg


def detect_columns(header: list[str]) -> dict:
    low = [h.strip().lower() for h in header]
    def find(*aliases: str) -> str:
        for a in aliases:
            if a in low:
                return header[low.index(a)]
        for a in aliases:
            for h in low:
                if a in h:
                    return header[low.index(a)]
        raise KeyError(f"no column matching {aliases} in {header}")
    return {"path": find("path", "filename", "clip"),
            "text": find("sentence", "transcript", "text"),
            "speaker": find("speaker", "client_id")}


def collect_from_tsv(tsv: str, audio_dir: Path) -> dict[str, list[dict]]:
    import soundfile as sf
    with open(tsv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    cols = detect_columns(rows[0].keys() if rows else [])
    per_speaker: dict[str, list[dict]] = {}
    for row in rows:
        f = audio_dir / row[cols["path"]]
        if not f.exists():
            continue
        try:
            info = sf.info(str(f))
        except Exception:
            continue  # unreadable/missing audio → skip, never crash the build
        if not row[cols["text"]].strip():
            continue
        per_speaker.setdefault(row[cols["speaker"]], []).append(
            {"file": row[cols["path"]], "text": row[cols["text"]].strip(),
             "seconds": round(info.duration, 3), "sr": info.samplerate}
        )
    return per_speaker


def collect_from_hf_parquet(parquet_dir: Path, audio_dir: Path, exclude_source: str) -> dict[str, list[dict]]:
    """Unpack HF 'Audio'-column parquet rows to real files + group by speaker.

    The michaelodafe/pidgin-asr-combined schema (audio/text/duration/speaker_id/source)
    is near-identical to the verified VoxCPM2 training format, so rows carry over
    essentially 1:1 after the license audit (see --exclude-source).
    """
    import pyarrow.parquet as pq
    audio_dir.mkdir(parents=True, exist_ok=True)
    per_speaker: dict[str, list[dict]] = {}
    skipped_source = 0
    for shard in sorted(Path(parquet_dir).rglob("*.parquet")):
        table = pq.read_table(shard)
        for row in table.to_pylist():
            if exclude_source and (row.get("source") or "") == exclude_source:
                skipped_source += 1
                continue
            text = (row.get("text") or "").strip()
            audio = row.get("audio") or {}
            blob = audio.get("bytes")
            if not blob or not text:
                continue
            target_rel = (
                f"{shard.stem}__{len(sum(per_speaker.values(), [])):06d}"
                f"_{abs(hash(text)) % 10_000_000:08d}.wav"
            )
            target = audio_dir / target_rel
            target.write_bytes(blob)
            dur = row.get("duration") or 0.0
            spk = (row.get("speaker_id") or "").strip() or f"hf-unknown-{shard.stem}"
            per_speaker.setdefault(spk, []).append(
                {"file": target_rel, "text": text, "seconds": round(float(dur), 3), "sr": 16000}
            )
    print(f"HF mode: skipped {skipped_source} clipped rows from excluded source '{exclude_source}'")
    return per_speaker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", help="Common Voice-style validated .tsv (mode 1)")
    ap.add_argument("--hf-parquet-dir", help="local dir with dataset parquet shards (mode 2)")
    ap.add_argument("--audio-dir", required=True, help="clips dir (mode 1) / unpack target (mode 2)")
    ap.add_argument("--exclude-source", default="Rexe",
                    help="skip rows whose source equals this (license audit; '' disables)")
    ap.add_argument("--max-speakers", type=int, default=0, help="cap speakers (0 = all)")
    args = ap.parse_args()

    cfg = load_split_config()
    audio_dir = Path(args.audio_dir)

    if args.hf_parquet_dir:
        per_speaker = collect_from_hf_parquet(Path(args.hf_parquet_dir), audio_dir, args.exclude_source)
    elif args.tsv:
        per_speaker = collect_from_tsv(args.tsv, audio_dir)
    else:
        ap.error("choose --tsv (mode 1) or --hf-parquet-dir (mode 2)")

    seed = int(cfg["training"]["seed"])
    r = random.Random(seed)
    speakers = sorted(per_speaker)
    r.shuffle(speakers)
    if args.max_speakers:
        speakers = speakers[: args.max_speakers]
    print(f"speakers: {len(speakers)} | seed {seed}")

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

    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    src_name = cfg["license"]["source_name"]
    lic_id = cfg["license"]["license_id"]
    for spk in speakers:
        sp = split_of[spk]
        for i, item in enumerate(per_speaker[spk]):
            out[sp].append({
                "id": f"pcm-{spk}-{i:04d}",
                "audio": str(audio_dir / item["file"]),
                "duration": item["seconds"],
                "text": item["text"],
                "speaker_id": spk,
                "sampling_rate": item["sr"],
                "language": "pcm",
                "source": src_name,
                "license": lic_id,
                "approved_for_training": bool(cfg["license"]["license_cleared"]
                                              and cfg["license"]["consent_review_done"]),
            })

    outdir = REPO / "data" / "manifests"
    outdir.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        target = outdir / f"pidgin_{sp}.jsonl"
        with open(target, "w", encoding="utf-8") as fh:
            for rec in out[sp]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        hours = sum(x["duration"] for x in out[sp]) / 3600
        print(f"pidgin_{sp}: {len(out[sp])} clips | ~{hours:.2f} h | "
              f"{len({x['speaker_id'] for x in out[sp]})} speakers")

    train_spk = {x["speaker_id"] for x in out["train"]}
    val_spk = {x["speaker_id"] for x in out["val"]}
    test_spk = {x["speaker_id"] for x in out["test"]}
    overlap = (train_spk & val_spk) | (train_spk & test_spk) | (val_spk & test_spk)
    assert not overlap, "speaker overlap detected — split failed sanity check"
    print("speaker overlap check: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
