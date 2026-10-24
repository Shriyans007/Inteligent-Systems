"""
Task 4 - Benchmark metrics

Summarises one or more results files from cnn_model/evaluate_uploads.py:
exact accuracy, digit accuracy, character error rate and segmentation vs
classifier errors, split by number length. With several files (one per model)
it also builds a single comparison table.

Run from the repo root:
  python3 -m preprocessing.benchmark_metrics --results artifacts/benchmark_cnn.json artifacts/benchmark_svm.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

FIELDS = ["model", "digits", "images", "exact_accuracy", "digit_accuracy", "character_error_rate",
          "segmentation_errors", "classifier_errors"]


def edit_distance(expected: str, predicted: str) -> int:
    # levenshtein distance, so one missing digit counts as one error not a whole wrong number
    previous = list(range(len(predicted) + 1))
    for i, expected_digit in enumerate(expected, start=1):
        current = [i]
        for j, predicted_digit in enumerate(predicted, start=1):
            cost = 0 if expected_digit == predicted_digit else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _metrics_for(cases) -> dict:
    total = len(cases)
    exact = sum(case["expected"] == case["predicted"] for case in cases)
    length_errors = sum(len(case["expected"]) != len(case["predicted"]) for case in cases)
    # right digit count but still wrong means the classifier got a digit wrong
    classifier_errors = total - exact - length_errors

    # only count digits when the lengths match, otherwise they don't line up
    aligned = [case for case in cases if len(case["expected"]) == len(case["predicted"])]
    aligned_digits = sum(len(case["expected"]) for case in aligned)
    correct_digits = sum(e == p for case in aligned for e, p in zip(case["expected"], case["predicted"]))

    expected_digits = sum(len(case["expected"]) for case in cases)
    edits = sum(edit_distance(case["expected"], case["predicted"]) for case in cases)

    return {
        "images": total,
        "exact_matches": exact,
        "exact_accuracy": exact / total if total else 0.0,
        "digit_accuracy": correct_digits / aligned_digits if aligned_digits else None,
        "character_error_rate": edits / expected_digits if expected_digits else 0.0,
        "segmentation_errors": length_errors,
        "classifier_errors": classifier_errors,
    }


def summarise(cases) -> dict:
    if not cases:
        raise ValueError("No cases to summarise")
    by_length = {}
    for length in sorted({len(case["expected"]) for case in cases}):
        group = [case for case in cases if len(case["expected"]) == length]
        by_length[length] = _metrics_for(group)

    confusions = Counter()
    for case in cases:
        if len(case["expected"]) == len(case["predicted"]):
            for e, p in zip(case["expected"], case["predicted"]):
                if e != p:
                    confusions[f"{e}->{p}"] += 1

    return {
        "overall": _metrics_for(cases),
        "by_length": by_length,
        "top_confusions": confusions.most_common(10),
    }


def shared_segmentation_errors(results: dict) -> list:
    """Images with the wrong digit count for every model. Since all models share the
    same segmentation step, these point at segmentation rather than a classifier."""
    per_model = [
        {case["image"] for case in cases if len(case["expected"]) != len(case["predicted"])}
        for cases in results.values()
    ]
    return sorted(set.intersection(*per_model)) if per_model else []


def error_rows(results: dict) -> list:
    """Every wrong prediction, with whether segmentation or the classifier caused it."""
    rows = []
    for model, cases in results.items():
        for case in cases:
            if case["expected"] == case["predicted"]:
                continue
            same_length = len(case["expected"]) == len(case["predicted"])
            rows.append({"model": model, "image": case["image"], "expected": case["expected"],
                         "predicted": case["predicted"],
                         "error_type": "classifier" if same_length else "segmentation"})
    return rows


def save_errors(results: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "image", "expected", "predicted", "error_type"])
        writer.writeheader()
        writer.writerows(error_rows(results))
    return path


def comparison_rows(summaries: dict) -> list:
    rows = []
    for model, summary in summaries.items():
        groups = list(summary["by_length"].items()) + [("all", summary["overall"])]
        for length, metrics in groups:
            rows.append({"model": model, "digits": length, **{f: metrics[f] for f in FIELDS[2:]}})
    return rows


def save_outputs(summaries: dict, shared: list, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_summary.json"
    json_path.write_text(json.dumps({"models": summaries, "shared_segmentation_errors": shared}, indent=2),
                         encoding="utf-8")

    csv_path = output_dir / "benchmark_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in comparison_rows(summaries):
            writer.writerow({key: _round(value) for key, value in row.items()})
    return json_path, csv_path


def save_accuracy_plot(summaries: dict, path: Path) -> Path:
    """Bar chart of whole-number accuracy by number length, one bar per model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plot needs matplotlib. Run: python3 -m pip install matplotlib") from error

    lengths = sorted({length for summary in summaries.values() for length in summary["by_length"]})
    width = 0.8 / len(summaries)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (model, summary) in enumerate(summaries.items()):
        values = [summary["by_length"].get(length, {}).get("exact_accuracy", 0) * 100 for length in lengths]
        positions = [x + i * width - 0.4 + width / 2 for x in range(len(lengths))]
        bars = ax.bar(positions, values, width, label=model)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([f"{length}-digit" for length in lengths])
    ax.set_ylabel("Whole number correct (%)")
    ax.set_title("Benchmark accuracy by number length")
    # start a bit below the lowest bar so the differences are easy to see
    lowest = min(s["by_length"][l]["exact_accuracy"] for s in summaries.values() for l in s["by_length"]) * 100
    ax.set_ylim(max(0, lowest - 15), 102)
    ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _round(value):
    if value is None:
        return "n/a"
    return round(value, 4) if isinstance(value, float) else value


def _percent(value) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def load_results(paths) -> dict:
    results = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Can't find {path}. Run cnn_model.evaluate_uploads with --output {path} first.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "cases" not in data:
            raise ValueError(f"{path} has no 'cases', so it doesn't look like evaluate_uploads output")
        # older results files don't record the model, so fall back to the file name
        model = data.get("model") or path.stem
        results[model] = data["cases"]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise benchmark results from evaluate_uploads.py.")
    parser.add_argument("--results", type=Path, nargs="+", default=[Path("artifacts/benchmark_evaluation.json")])
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument("--plot", action="store_true", help="Also save a bar chart of accuracy by number length")
    args = parser.parse_args()

    try:
        results = load_results(args.results)
    except (FileNotFoundError, ValueError) as error:
        # print a short message instead of a full traceback
        raise SystemExit(f"Error: {error}")
    summaries = {model: summarise(cases) for model, cases in results.items()}
    shared = shared_segmentation_errors(results)
    json_path, csv_path = save_outputs(summaries, shared, args.output)

    print(f"{'model':>8} {'digits':>6} {'images':>6} {'exact':>7} {'digit':>7} {'CER':>7} {'seg err':>8} {'clf err':>8}")
    for row in comparison_rows(summaries):
        print(f"{row['model']:>8} {row['digits']:>6} {row['images']:>6} {_percent(row['exact_accuracy']):>7} "
              f"{_percent(row['digit_accuracy']):>7} {row['character_error_rate']:>7.3f} "
              f"{row['segmentation_errors']:>8} {row['classifier_errors']:>8}")
    for model, summary in summaries.items():
        if summary["top_confusions"]:
            top = ", ".join(f"{pair} ({count})" for pair, count in summary["top_confusions"][:5])
            print(f"Most common confusions for {model}: {top}")
    if len(results) > 1:
        print(f"Wrong digit count for every model: {len(shared)} image(s)")
    print(f"Saved {json_path} and {csv_path}")
    # handy for looking at the actual images that went wrong
    errors_path = save_errors(results, args.output / "benchmark_errors.csv")
    print(f"Saved {errors_path}")
    if args.plot:
        try:
            plot_path = save_accuracy_plot(summaries, args.output / "benchmark_accuracy.png")
        except RuntimeError as error:
            raise SystemExit(f"Error: {error}")
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
