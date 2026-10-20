"""Run every trained digit model with every segmentation method on the benchmark set.

Each run goes through the real /api/predict route, so segmentation, preprocessing and
the classifier are all measured together. The tables reuse Person A's
preprocessing.benchmark_metrics so every result is summarised the same way.

Run from the repo root (after python -m preprocessing.build_benchmark_set):
  python -m cnn_model.benchmark_all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cnn_model.evaluate_uploads import evaluate_manifest
from preprocessing.benchmark_metrics import save_outputs, shared_segmentation_errors, summarise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark/all"))
    parser.add_argument("--models", nargs="+", help="Model keys to run; default: every trained digit model")
    parser.add_argument("--segmentation", nargs="+", help="Segmentation keys to run; default: all of them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import api.main as api

    models = args.models or [key for key, path in api.MODEL_PATHS.items() if path.is_file()]
    # Skip the learned segmenter until its model has been trained.
    methods = args.segmentation or [key for key in api.SEGMENTERS if api.segmenter_available(key)]
    if not models:
        raise SystemExit("No trained digit models found. Train one first, e.g. python -m cnn_model.train_compare")

    cases, summaries = {}, {}
    for model in models:
        for method in methods:
            name = f"{model} + {method}"
            print(f"\n=== {name}")
            started = time.perf_counter()
            path = args.output / "runs" / f"{model}__{method}.json"
            evaluate_manifest(args.manifest, path, model, method)
            seconds = time.perf_counter() - started
            cases[name] = json.loads(path.read_text(encoding="utf-8"))["cases"]
            summaries[name] = {**summarise(cases[name]), "model": model, "segmentation": method,
                               "seconds": round(seconds, 2)}

    # Errors every model makes with the same segmenter come from that segmenter, not a classifier.
    shared = {method: shared_segmentation_errors({name: rows for name, rows in cases.items()
                                                  if summaries[name]["segmentation"] == method})
              for method in methods}
    json_path, csv_path = save_outputs(summaries, shared, args.output)

    print(f"\n{'model + segmentation':38} {'exact':>7} {'digit':>7} {'CER':>6} {'seg err':>8} {'clf err':>8} {'secs':>6}")
    ranked = sorted(summaries.items(), key=lambda item: item[1]["overall"]["exact_accuracy"], reverse=True)
    for name, summary in ranked:
        overall = summary["overall"]
        digit = "n/a" if overall["digit_accuracy"] is None else f"{overall['digit_accuracy']:.2%}"
        print(f"{name:38} {overall['exact_accuracy']:>7.2%} {digit:>7} {overall['character_error_rate']:>6.3f} "
              f"{overall['segmentation_errors']:>8} {overall['classifier_errors']:>8} {summary['seconds']:>6}")
    for method, images in shared.items():
        print(f"{method}: {len(images)} images split wrongly by every model")
    print(f"\nSaved {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
