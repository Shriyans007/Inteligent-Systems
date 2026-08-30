# Task 1 - comparing preprocessing methods on MNIST before picking one

from pathlib import Path
import time

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from preprocessing import preprocess_pipeline

RESULTS_DIR = Path("results")
VARIANTS = [
    {"name": "Grayscale only", "binarize": False, "method": None},
    {"name": "Fixed threshold (128)", "binarize": True, "method": "fixed"},
    {"name": "Otsu threshold", "binarize": True, "method": "otsu"},
]


def load_mnist_subset():
    from mlxtend.data import mnist_data

    X, y = mnist_data()
    images = X.reshape(-1, 28, 28).astype(np.uint8)
    return images, y


def apply_variant(images: np.ndarray, variant: dict) -> np.ndarray:
    processed = np.empty((len(images), 28, 28), dtype=np.float32)
    for i, arr in enumerate(images):
        img = Image.fromarray(arr, mode="L")
        processed[i] = preprocess_pipeline(
            img,
            size=(28, 28),
            binarize=variant["binarize"],
            method=variant["method"] or "fixed",
        )
    return processed.reshape(len(images), -1)


def run_comparison():
    print("Loading MNIST subset (mlxtend, 5000 images)...")
    images, labels = load_mnist_subset()

    idx_train, idx_test = train_test_split(
        np.arange(len(images)), test_size=0.2, stratify=labels, random_state=42
    )

    rows = []
    for variant in VARIANTS:
        t0 = time.time()
        X_flat = apply_variant(images, variant)
        X_train, X_test = X_flat[idx_train], X_flat[idx_test]
        y_train, y_test = labels[idx_train], labels[idx_test]

        clf = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=40,
            random_state=42,
            early_stopping=True,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        elapsed = time.time() - t0

        print(f"{variant['name']:<24} accuracy = {acc*100:.2f}%  ({elapsed:.1f}s)")
        rows.append({"method": variant["name"], "accuracy_pct": round(acc * 100, 2)})

    return rows


def save_outputs(rows):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(exist_ok=True)

    csv_path = RESULTS_DIR / "preprocessing_compare.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "accuracy_pct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    names = [r["method"] for r in rows]
    accs = [r["accuracy_pct"] for r in rows]
    bars = ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("MNIST test accuracy by preprocessing method")
    ax.set_ylim(min(accs) - 5, 100)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.3, f"{acc:.2f}%", ha="center")
    plt.xticks(rotation=10)
    plt.tight_layout()
    png_path = RESULTS_DIR / "preprocessing_compare.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    results = run_comparison()
    save_outputs(results)