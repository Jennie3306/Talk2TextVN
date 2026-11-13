"""Benchmark Vietnamese ASR models on the public VIVOS test set (760 utterances, 19 speakers).

Compares the original multilingual Whisper (current baseline) with VinAI's PhoWhisper
models, using the SAME preprocessing as the app and the SAME text normalisation for
every model, and reports corpus-level WER / CER plus speed.

Usage (Google Colab with a T4 GPU is recommended):
    pip install -r requirements.txt
    python benchmark_vivos.py --limit 50          # quick check on 50 utterances
    python benchmark_vivos.py                     # full test set, default models
    python benchmark_vivos.py --models whisper:base phowhisper:vinai/PhoWhisper-small
    python benchmark_vivos.py --own               # also score the 20 self-recorded samples

Outputs (results/):
    benchmark_vivos.md                  summary table, ready to paste into the README
    benchmark_vivos.csv                 one row per model
    vivos_predictions_<model>.csv       every reference / hypothesis pair, for error analysis

Dataset: VIVOS (AILAB, VNUHCM - University of Science), CC BY-NC-SA 4.0.
Note: PhoWhisper's training data includes the VIVOS *train* split, so VIVOS *test* is
in-domain for PhoWhisper. Report it as such, and use --own for an out-of-domain check.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tarfile
import time
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
from jiwer import cer, wer
from scipy.io import wavfile

from config import SAMPLE_RATE
from preprocess import preprocess
from transcribe import transcribe

VIVOS_URL = "https://zenodo.org/record/7068130/files/vivos.tar.gz"
DEFAULT_MODELS = [
    "whisper:base",                       # current app baseline
    "phowhisper:vinai/PhoWhisper-tiny",
    "phowhisper:vinai/PhoWhisper-base",
    "phowhisper:vinai/PhoWhisper-small",
]
RESULTS = Path("results")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def download_vivos_test(data_dir: Path) -> Path:
    """Stream the VIVOS archive and extract only the test split (~45 min of audio)."""
    test_dir = data_dir / "vivos" / "test"
    if (test_dir / "prompts.txt").exists():
        return test_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading VIVOS from Zenodo (about 1.4 GB, only the test split is kept)...")
    with urllib.request.urlopen(VIVOS_URL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if member.name.startswith("vivos/test/"):
                try:
                    tar.extract(member, data_dir, filter="data")   # Python 3.12+: safe extraction
                except TypeError:
                    tar.extract(member, data_dir)
    if not (test_dir / "prompts.txt").exists():
        raise FileNotFoundError(f"Expected {test_dir / 'prompts.txt'} after extraction")
    return test_dir


def load_vivos_test(data_dir: Path, limit: int | None) -> list[dict]:
    test_dir = download_vivos_test(data_dir)
    items = []
    for line in (test_dir / "prompts.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt_id, text = line.split(" ", 1)
        speaker = utt_id.split("_")[0]
        items.append({"id": utt_id, "path": str(test_dir / "waves" / speaker / f"{utt_id}.wav"), "ref": text})
    items.sort(key=lambda x: x["id"])
    return items[:limit] if limit else items


def load_own_samples() -> list[dict]:
    """The 10 clean + 10 noisy self-recorded samples already in test_cases/."""
    items = []
    for cond in ("clean", "noisy"):
        folder = Path("test_cases") / cond
        refs = [l.strip() for l in (folder / "ground_truth.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        wavs = sorted(p for p in folder.glob("*.wav"))
        if not wavs:
            print(f"No .wav files in {folder} (they are not in the repo); skipping the {cond} set.")
            continue
        for wav_path, ref in zip(wavs, refs):
            items.append({"id": f"{cond}/{wav_path.name}", "path": str(wav_path), "ref": ref, "cond": cond})
    return items


def audio_seconds(path: str) -> float:
    sr, data = wavfile.read(path)
    return len(data) / sr


# --------------------------------------------------------------------------- #
# Text normalisation (applied identically to references and hypotheses)
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()   # one Unicode form for Vietnamese diacritics
    text = _PUNCT.sub(" ", text)                          # drop punctuation, keep letters and digits
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def run_model(spec: str, items: list[dict]) -> tuple[dict, list[dict]]:
    backend, model_name = spec.split(":", 1)
    refs, hyps, rows = [], [], []
    total_proc, total_audio = 0.0, 0.0

    for i, it in enumerate(items, 1):
        audio = preprocess(it["path"])            # same preprocessing as the app
        out = transcribe(audio, model_name=model_name, backend=backend)
        ref, hyp = normalize(it["ref"]), normalize(out["raw_text"])
        refs.append(ref)
        hyps.append(hyp)
        total_proc += out["latency_s"]
        total_audio += audio_seconds(it["path"])
        rows.append({"id": it["id"], "reference": ref, "hypothesis": hyp,
                     "wer": round(wer(ref, hyp), 4) if ref else None})
        if i % 50 == 0:
            print(f"  [{spec}] {i}/{len(items)}  running WER {wer(refs, hyps):.4f}")

    summary = {
        "model": spec,
        "n": len(items),
        "wer": round(wer(refs, hyps), 4),          # corpus-level: total errors / total reference words
        "cer": round(cer(refs, hyps), 4),
        "sec_per_utt": round(total_proc / len(items), 3),
        "rtf": round(total_proc / total_audio, 3),  # real-time factor: < 1 means faster than real time
    }
    return summary, rows


def write_outputs(summaries: list[dict], preds: dict[str, list[dict]], label: str, device: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"benchmark_{label}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader()
        w.writerows(summaries)
    for spec, rows in preds.items():
        safe = spec.replace(":", "_").replace("/", "_")
        with open(RESULTS / f"{label}_predictions_{safe}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    best = min(summaries, key=lambda s: s["wer"])
    lines = [
        f"# ASR benchmark: {label}",
        "",
        f"{summaries[0]['n']} utterances · device: {device} · same preprocessing and text normalisation for every model.",
        "",
        "| Model | WER | CER | sec / utterance | Real-time factor |",
        "|---|---|---|---|---|",
    ]
    for s in sorted(summaries, key=lambda s: s["wer"]):
        name = f"**{s['model']}**" if s is best else s["model"]
        lines.append(f"| {name} | {s['wer']:.3f} | {s['cer']:.3f} | {s['sec_per_utt']} | {s['rtf']} |")
    (RESULTS / f"benchmark_{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="backend:model, e.g. phowhisper:vinai/PhoWhisper-small")
    p.add_argument("--limit", type=int, default=None, help="only the first N VIVOS test utterances")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--own", action="store_true", help="also score the self-recorded clean/noisy samples")
    args = p.parse_args()

    try:
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except ImportError:
        device = "CPU"
    print(f"Device: {device}")

    suites = [("vivos", load_vivos_test(args.data_dir, args.limit))]
    if args.own:
        own = load_own_samples()
        suites += [("own_clean", [x for x in own if x["cond"] == "clean"]),
                   ("own_noisy", [x for x in own if x["cond"] == "noisy"])]

    for label, items in suites:
        if not items:
            continue
        print(f"\n===== {label}: {len(items)} utterances =====")
        summaries, preds = [], {}
        for spec in args.models:
            t0 = time.time()
            summary, rows = run_model(spec, items)
            print(f"{spec}: WER {summary['wer']:.4f}  CER {summary['cer']:.4f}  ({time.time() - t0:.0f}s)")
            summaries.append(summary)
            preds[spec] = rows
        write_outputs(summaries, preds, label, device)


if __name__ == "__main__":
    main()
