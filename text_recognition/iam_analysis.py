"""Summarise an existing IAM predictions CSV without loading images or TensorFlow."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .iam import distance


def summarise(path):
    groups = {}
    mistakes = Counter()

    def add(group, actual, predicted):
        entry = groups.setdefault(group, {"samples": 0, "incorrect_words": 0,
                                          "character_errors": 0, "reference_characters": 0})
        entry["samples"] += 1
        entry["incorrect_words"] += actual != predicted
        entry["character_errors"] += distance(actual, predicted)
        entry["reference_characters"] += len(actual)

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"actual", "predicted"}.issubset(reader.fieldnames):
            raise ValueError("Expected a CSV with actual,predicted columns.")
        for row in reader:
            actual, predicted = row["actual"], row["predicted"]
            if actual is None or predicted is None:
                raise ValueError("A prediction row has a missing actual or predicted field.")
            add("all", actual, predicted)
            add(f"length_{'1-4' if len(actual) <= 4 else '5-8' if len(actual) <= 8 else '9+'}", actual, predicted)
            if actual.isupper() and any(c.isalpha() for c in actual): add("all_capitals", actual, predicted)
            if any(c.isdigit() for c in actual): add("contains_digit", actual, predicted)
            if any(not c.isalnum() for c in actual): add("punctuation_or_space", actual, predicted)
            if actual != predicted: mistakes[(actual, predicted)] += 1
    if not groups:
        raise ValueError("The predictions CSV contains no rows.")
    for entry in groups.values():
        entry["cer"] = entry["character_errors"]/entry["reference_characters"] if entry["reference_characters"] else None
        entry["exact_word_accuracy"] = 1-entry["incorrect_words"]/entry["samples"]
    return {"source": str(path), "groups": groups,
            "frequent_errors": [{"actual": a, "predicted": p, "count": n}
                                for (a, p), n in mistakes.most_common(20)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/text/word/test_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/text/word/error_analysis.json"))
    args = parser.parse_args()
    if not args.predictions.is_file():
        parser.error(f"IAM predictions missing: {args.predictions}; evaluate the existing model first.")
    result = summarise(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, group in result["groups"].items():
        print(f"{name}: {group['samples']} samples; CER {group['cer']}; exact-word accuracy {group['exact_word_accuracy']:.4f}")
    print(f"Saved {args.output}")


if __name__ == "__main__": main()
