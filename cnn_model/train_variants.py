"""Train LeNet-5 or small ResNet with the existing MNIST experiment settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import build_lenet5, build_resnet
from .train_compare import load_mnist, train_one

BUILDERS = {"lenet5": build_lenet5, "resnet": build_resnet}
NAMES = {"lenet5": "LeNet-5", "resnet": "Small ResNet"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=BUILDERS, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use 10,000/2,000 images to check the pipeline")
    parser.add_argument("--output", type=Path, help="Dedicated output directory (defaults to artifacts/models/<model>)")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.output is None:
        args.output = Path("artifacts") / ("quick_check" if args.quick else "models") / args.model
    # Do not accidentally replace the existing CNN/MLP model and comparison files.
    if args.output.resolve() == Path("artifacts").resolve():
        parser.error("Choose a separate --output folder, such as artifacts/models/lenet5")
    return args


def main():
    args = parse_args()
    data = load_mnist(args.quick, args.seed)
    model, history, record = train_one(NAMES[args.model], BUILDERS[args.model], data, args)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model}_mnist.keras"
    model.save(model_path)
    record.update({
        "model_key": args.model,
        "quick": args.quick,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "train_samples": len(data[0]),
        "test_samples": len(data[2]),
        "validation_split": 0.1,
        "model_path": str(model_path),
    })
    (args.output / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
