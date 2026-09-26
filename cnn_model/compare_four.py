"""Collect measured results for the four lecturer-approved models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect(baseline: Path, lenet: Path, resnet: Path):
    old = json.loads(baseline.read_text(encoding="utf-8"))
    if not isinstance(old, list) or {row.get("model") for row in old} != {"MLP baseline", "CNN selected"}:
        raise ValueError("Baseline comparison must contain the MLP and CNN results")
    new = [json.loads(path.read_text(encoding="utf-8")) for path in (lenet, resnet)]
    if [row.get("model_key") for row in new] != ["lenet5", "resnet"]:
        raise ValueError("The new result files must be LeNet-5 then ResNet")
    if any(row.get("quick") or row.get("train_samples") != 60000 or row.get("test_samples") != 10000 for row in new):
        raise ValueError("Full-run comparison requires 60,000 train and 10,000 test samples")
    if new[0]["seed"] != new[1]["seed"] or new[0]["batch_size"] != new[1]["batch_size"]:
        raise ValueError("LeNet-5 and ResNet must use the same seed and batch size")
    # Legacy comparison.json has no run metadata: verify its provenance manually.
    return old + new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/comparison.json"))
    parser.add_argument("--lenet", type=Path, default=Path("artifacts/models/lenet5/result.json"))
    parser.add_argument("--resnet", type=Path, default=Path("artifacts/models/resnet/result.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/four_model_comparison.csv"))
    args = parser.parse_args()
    try:
        results = collect(args.baseline, args.lenet, args.resnet)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "test_accuracy", "test_loss", "training_seconds", "parameters", "epochs_completed"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in fields})
    print(f"Saved {args.output}. Confirm the legacy MLP/CNN file came from the full MNIST run with the same settings.")
    for row in results:
        print(f"{row['model']}: {row['test_accuracy']:.4%} accuracy, {row['test_loss']:.4f} loss")
    print("Select the final integrated model only after reviewing these results and real uploaded-image tests.")


if __name__ == "__main__":
    main()
