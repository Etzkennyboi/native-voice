#!/usr/bin/env python3
"""Paired significance check for a Phase 4 WER result (indication only).

The Phase 4 record reports mean WER per system. With 20 passages, a 0.046 mean
gap is not obviously distinguishable from noise. This script reads the
`per_passage` rows written by `score_asr.py`, computes the paired difference
(tuned - baseline; positive = tuned worse), a percentile bootstrap CI, an exact
sign test, and (if scipy is installed) a Wilcoxon signed-rank test.

It does not change the verdict: WER remains an indication, human ratings
outrank it. It only stops the record from overstating a noisy gap.

Usage:
    python benchmarks/pidgin_eval/wer_significance.py \
        --wer /content/drive/MyDrive/naturalvoice-ai-phase3/eval-20261022-raw/wer_pidgin_mms.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wer", required=True, help="wer_<set>_<recognizer>.json")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--out", default="", help="output JSON (default: next to input)")
    return p.parse_args()


def bootstrap_ci(diffs: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs)))
                   for _ in range(n_boot))
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return round(lo, 4), round(hi, 4)


def sign_test(diffs: list[float]) -> dict:
    worse = sum(d > 0 for d in diffs)
    better = sum(d < 0 for d in diffs)
    ties = sum(d == 0 for d in diffs)
    n = worse + better
    if n == 0:
        return {"worse": 0, "better": 0, "ties": ties, "p_two_sided": 1.0}
    k = min(worse, better)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"worse": worse, "better": better, "ties": ties,
            "p_two_sided": round(min(1.0, 2 * tail), 4)}


def wilcoxon(diffs: list[float]) -> dict | None:
    try:
        from scipy.stats import wilcoxon as _w
    except Exception:
        return None
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 6:
        return None
    try:
        stat, p = _w(nonzero, zero_method="wilcox")
        return {"statistic": round(float(stat), 4), "p_two_sided": round(float(p), 4)}
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    path = Path(args.wer)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["per_passage"]
    base = [float(r["systems"]["baseline"]["wer"]) for r in rows]
    tuned = [float(r["systems"]["tuned"]["wer"]) for r in rows]
    diffs = [t - b for b, t in zip(base, tuned)]

    lo, hi = bootstrap_ci(diffs, args.n_boot, args.seed)
    sign = sign_test(diffs)
    wil = wilcoxon(diffs)
    mean_diff = statistics.mean(diffs)
    ci_excludes_zero = lo > 0 or hi < 0

    result = {
        "source": str(path),
        "set": data.get("set"),
        "recognizer": data.get("recognizer"),
        "n": len(diffs),
        "mean_wer": {"baseline": round(statistics.mean(base), 4),
                     "tuned": round(statistics.mean(tuned), 4)},
        "paired_diff_tuned_minus_baseline": {
            "mean": round(mean_diff, 4),
            "median": round(statistics.median(diffs), 4),
            "sd": round(statistics.stdev(diffs), 4) if len(diffs) > 1 else None,
            "bootstrap_95_ci": [lo, hi],
        },
        "sign_test": sign,
        "wilcoxon": wil,
        "flag": "WER is an indication only; this test does not make it proof. "
                "Human ratings outrank it.",
        "reading": ("gap distinguishable from zero at 0.05"
                    if ci_excludes_zero and sign["p_two_sided"] < 0.05
                    else "gap not distinguishable from zero at 0.05"),
    }
    out = Path(args.out) if args.out else path.with_name("wer_significance.json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"n = {result['n']} | mean WER baseline {result['mean_wer']['baseline']:.3f} "
          f"vs tuned {result['mean_wer']['tuned']:.3f}")
    print(f"paired diff (tuned - baseline): mean {mean_diff:+.3f}, "
          f"median {statistics.median(diffs):+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"sign test: {sign['worse']} worse / {sign['better']} better / "
          f"{sign['ties']} tie, two-sided p = {sign['p_two_sided']:.3f}")
    if wil:
        print(f"wilcoxon: statistic {wil['statistic']:.3f}, p = {wil['p_two_sided']:.3f}")
    print(f"reading: {result['reading']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
