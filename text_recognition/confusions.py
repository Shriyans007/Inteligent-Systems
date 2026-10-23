"""Summarise which EMNIST labels a trained classifier mixes up."""

import argparse
import csv
from pathlib import Path


def analyse(matrix_path, output):
    with matrix_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    if not rows or rows[0][0] != "actual/predicted":
        raise ValueError(f"Expected an EMNIST confusion matrix: {matrix_path}")
    labels = rows[0][1:]
    if len(labels) != 62 or len(set(labels)) != 62 or len(rows) != 63:
        raise ValueError("Expected 62 distinct labels and 62 data rows.")
    counts = []
    for index, row in enumerate(rows[1:]):
        if len(row) != 63 or row[0] != labels[index]:
            raise ValueError(f"Invalid label or column count on row {index + 2}.")
        try:
            values = [int(value) for value in row[1:]]
        except ValueError as exc:
            raise ValueError(f"Non-integer count on row {index + 2}.") from exc
        if any(value < 0 for value in values):
            raise ValueError("Confusion counts cannot be negative.")
        counts.append(values)

    pairs = []
    for i, actual in enumerate(labels):
        total = sum(counts[i])
        for j, predicted in enumerate(labels):
            if i != j and counts[i][j]:
                pairs.append({"actual": actual, "predicted": predicted,
                              "count": counts[i][j], "actual_samples": total,
                              "error_rate": round(counts[i][j] / total, 4) if total else 0})
    pairs.sort(key=lambda pair: pair["count"], reverse=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "top_confusions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["actual", "predicted", "count", "actual_samples", "error_rate"])
        writer.writeheader()
        writer.writerows(pairs)

    # Both directions matter: c -> C can be frequent even if C -> c is rare.
    with (output / "case_pairs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["letter", "lower_to_upper", "lower_samples", "upper_to_lower", "upper_samples"])
        for letter in "abcdefghijklmnopqrstuvwxyz":
            lower, upper = labels.index(letter), labels.index(letter.upper())
            writer.writerow([letter, counts[lower][upper], sum(counts[lower]),
                             counts[upper][lower], sum(counts[upper])])
    for pair in pairs[:15]:
        print(f"{pair['actual']} -> {pair['predicted']}: {pair['count']}/{pair['actual_samples']} ({pair['error_rate']:.1%})")
    print(f"Saved {output / 'top_confusions.csv'} and {output / 'case_pairs.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("artifacts/text/character/confusion_matrix.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/text/character/diagnosis"))
    args = parser.parse_args()
    if not args.matrix.is_file():
        parser.error(f"Missing {args.matrix}; evaluate the trained model first.")
    analyse(args.matrix, args.output)


if __name__ == "__main__":
    main()
