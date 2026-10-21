#!/usr/bin/env python3
"""Phase 4 automatic WER scoring — indication only (README Metric 1).

Transcribes the blind A/B wavs, maps them to systems via key.json, and reports
WER per passage / mean per system plus probe-word hit rates for the Pidgin set.

Recognizers:
  --recognizer mms            facebook/mms-1b-all with the pcm (or eng) adapter
                              CC-BY-NC 4.0 — research evaluation only
  --recognizer whisper-pidgin michaelodafe/whisper-pidgin-v1 (fallback when MMS
                              has no pcm adapter)

The WER is an *indication*, not proof: orthography and code-switching make it
noisy. Human listening (score_human.py) outranks it.

Usage (Colab, repo root):
    python benchmarks/pidgin_eval/score_asr.py \
        --out /content/drive/MyDrive/naturalvoice-ai-phase3/eval-20261022 \
        --set pidgin --recognizer mms --lang pcm
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PASSAGES = REPO / "benchmarks" / "pidgin_eval" / "passages.json"
DEFAULT_ENGLISH = REPO / "benchmarks" / "pidgin_eval" / "english_regression.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--set", choices=["pidgin", "english"], required=True)
    p.add_argument("--recognizer", choices=["mms", "whisper-pidgin"], default="mms")
    p.add_argument("--lang", default="", help="MMS adapter language (pcm/eng); default by set")
    p.add_argument("--passages", default=str(DEFAULT_PASSAGES))
    p.add_argument("--english", default=str(DEFAULT_ENGLISH))
    return p.parse_args()


def normalise(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_wav16k(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import torch
        import torchaudio
        data = torchaudio.functional.resample(
            torch.from_numpy(data), sr, 16000).numpy()
    return data


def make_transcriber(kind: str, lang: str):
    if kind == "mms":
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to("cuda").eval()
        try:
            processor.tokenizer.set_target_lang(lang)
            model.load_adapter(lang)
        except Exception as exc:  # adapter missing for this language
            raise SystemExit(
                f"MMS adapter '{lang}' unavailable ({exc}). "
                "Re-run with --recognizer whisper-pidgin.") from exc
        print(f"[asr] mms-1b-all adapter '{lang}' ready")

        def transcribe(path: Path) -> str:
            inputs = processor(read_wav16k(path), sampling_rate=16000,
                               return_tensors="pt")
            with torch.inference_mode():
                logits = model(inputs.input_values.to("cuda")).logits
            ids = logits.argmax(dim=-1)[0].cpu()
            return processor.decode(ids)

        return transcribe

    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition",
                   model="michaelodafe/whisper-pidgin-v1", device=0)
    print("[asr] whisper-pidgin-v1 ready")

    def transcribe(path: Path) -> str:
        return asr(str(path))["text"]

    return transcribe


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))[args.set]
    if args.set == "pidgin":
        data = json.loads(Path(args.passages).read_text(encoding="utf-8"))
        items, probe_words = data["passages"], data["pidgin_specific_probe_words"]
        audio_dir = out / "audio"
        lang = args.lang or "pcm"
    else:
        items = json.loads(Path(args.english).read_text(encoding="utf-8"))["passages"]
        probe_words = []
        audio_dir = out / "english_audio"
        lang = args.lang or "eng"

    import jiwer  # late import so --help works without deps

    transcribe = make_transcriber(args.recognizer, lang)
    rows, hypotheses = [], {}
    t_start = time.time()
    for item in items:
        ref = normalise(item["text"])
        entry = {"id": item["id"], "reference": ref, "systems": {}}
        for letter in ("A", "B"):
            system = key[item["id"]][letter]
            wav_path = audio_dir / f"{item['id']}_{letter}.wav"
            hyp = normalise(transcribe(wav_path))
            hypotheses[(item["id"], system)] = hyp
            entry["systems"][system] = {
                "wer": float(jiwer.wer(ref, hyp)) if hyp else 1.0,
                "hypothesis": hyp,
            }
            print(f"[wer] {item['id']} {system:8s}: "
                  f"{entry['systems'][system]['wer']:.3f} | {hyp[:90]}")
        rows.append(entry)

    summary = {}
    for system in ("baseline", "tuned"):
        wers = [r["systems"][system]["wer"] for r in rows]
        summary[system] = {
            "mean_wer": float(statistics.mean(wers)),
            "median_wer": float(statistics.median(wers)),
            "n": len(wers),
        }
    if probe_words:
        for system in ("baseline", "tuned"):
            hits = total = 0
            for item in items:
                ref = normalise(item["text"])
                hyp = hypotheses[(item["id"], system)]
                for word in probe_words:
                    if re.search(rf"\b{re.escape(word)}\b", ref):
                        total += 1
                        hits += int(bool(re.search(rf"\b{re.escape(word)}\b", hyp)))
            summary[system]["probe_word_hits"] = hits
            summary[system]["probe_word_total"] = total
            summary[system]["probe_word_rate"] = round(hits / total, 3) if total else None

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "set": args.set,
        "recognizer": args.recognizer,
        "language": lang,
        "elapsed_s": round(time.time() - t_start, 1),
        "flag": "WER is an indication only — no Pidgin-native recognizer is confirmed; "
                "human ratings outrank this metric.",
        "summary": summary,
        "per_passage": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"wer_{args.set}_{args.recognizer}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n| system | mean WER | median WER | n |")
    print("|---|---|---|---|")
    for system, s in summary.items():
        print(f"| {system} | {s['mean_wer']:.3f} | {s['median_wer']:.3f} | {s['n']} |")
    if probe_words:
        for system, s in summary.items():
            print(f"{system}: probe-word hit rate {s['probe_word_rate']:.3f} "
                  f"({s['probe_word_hits']}/{s['probe_word_total']})")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
