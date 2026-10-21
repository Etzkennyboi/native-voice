#!/usr/bin/env python3
"""Phase 4 human-rating analysis — the mandatory quality gate.

Reads every ratings/ratings_listener-*.csv (template excluded), joins the blind
key, and reports per-system naturalness/authenticity means plus win rate.
Refuses to declare a gate pass with fewer than --min-listeners complete raters.

Usage (local or Colab, repo root):
    python benchmarks/pidgin_eval/score_human.py \
        --out /content/drive/MyDrive/naturalvoice-ai-phase3/eval-20261022
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

FIELDS = ["listener_id", "passage_id", "A_naturalness_1_5", "A_authenticity_1_5",
          "B_naturalness_1_5", "B_authenticity_1_5", "preference_A_B_tie", "comments"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--min-listeners", type=int, default=3)
    return p.parse_args()


def rating(value: str) -> int | None:
    try:
        x = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return x if 1 <= x <= 5 else None


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))["pidgin"]
    files = [f for f in sorted((out / "ratings").glob("ratings_listener-*.csv"))
             if "TEMPLATE" not in f.name.upper()]
    if not files:
        raise SystemExit("no ratings_listener-*.csv files found — run the listening test first")

    per_system = {"baseline": {"naturalness": [], "authenticity": []},
                  "tuned": {"naturalness": [], "authenticity": []}}
    preference = Counter()
    listeners, coverage, warnings = [], {}, []

    for path in files:
        name = path.stem.replace("ratings_listener-", "")
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        missing = [c for c in FIELDS if rows and c not in rows[0]]
        if missing:
            warnings.append(f"{name}: missing columns {missing}")
            continue
        filled = 0
        for row in rows:
            passage = (row.get("passage_id") or "").strip()
            if passage not in key:
                continue
            systems = key[passage]
            values = {
                ("A", "naturalness"): rating(row.get("A_naturalness_1_5")),
                ("A", "authenticity"): rating(row.get("A_authenticity_1_5")),
                ("B", "naturalness"): rating(row.get("B_naturalness_1_5")),
                ("B", "authenticity"): rating(row.get("B_authenticity_1_5")),
            }
            for (letter, metric), value in values.items():
                if value is None:
                    warnings.append(f"{name}/{passage}: missing {letter} {metric}")
                    continue
                per_system[systems[letter]][metric].append(value)
            pref = (row.get("preference_A_B_tie") or "").strip().upper()
            if pref in ("A", "B"):
                preference[systems[pref]] += 1
            elif pref == "TIE":
                preference["tie"] += 1
            else:
                warnings.append(f"{name}/{passage}: bad preference '{pref}'")
            filled += 1
        coverage[name] = filled
        listeners.append(name)

    complete = [name for name, n in coverage.items() if n == len(key)]
    result = {
        "listeners": listeners,
        "coverage": coverage,
        "complete_listeners": complete,
        "systems": {
            system: {
                "naturalness_mean": round(statistics.mean(v["naturalness"]), 3)
                if v["naturalness"] else None,
                "authenticity_mean": round(statistics.mean(v["authenticity"]), 3)
                if v["authenticity"] else None,
                "n_ratings": len(v["naturalness"]),
            }
            for system, v in per_system.items()
        },
        "preference": dict(preference),
        "preference_win_rate": {
            system: round(count / sum(preference.values()), 3)
            for system, count in preference.items()
        } if preference else {},
        "gate": {
            "min_listeners": args.min_listeners,
            "complete_listeners": len(complete),
            "met": len(complete) >= args.min_listeners,
            "note": "No 'sounds Pidgin' claim may be made unless met is true and the "
                    "tuned system wins on both scales (README exit rule).",
        },
        "warnings": warnings[:50],
    }

    path = out / "human_eval_summary.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"listeners: {listeners} | complete: {complete}")
    print("\n| system | naturalness (1-5) | authenticity (1-5) | ratings |")
    print("|---|---|---|---|")
    for system, s in result["systems"].items():
        print(f"| {system} | {s['naturalness_mean']} | {s['authenticity_mean']} | {s['n_ratings']} |")
    if preference:
        print("\npreference:", dict(preference))
    print(f"\nGATE: {result['gate']['met']} "
          f"({len(complete)}/{args.min_listeners} complete listeners)")
    print(f"wrote {path}")
    if not result["gate"]["met"]:
        print("HUMAN GATE NOT MET — do not claim Pidgin quality yet.")


if __name__ == "__main__":
    main()
