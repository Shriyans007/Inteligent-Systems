"""Tests for build_benchmark_set.py, using drawn digits so no MNIST download is needed."""

import csv

import cv2
import numpy as np
import pytest
from PIL import Image

from preprocessing.build_benchmark_set import build_benchmark_set, make_number_image
from segmentation import segment


def _fake_mnist(count=40):
    images, labels = [], []
    for i in range(count):
        tile = np.zeros((28, 28), dtype=np.uint8)
        cv2.putText(tile, str(i % 10), (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        images.append(tile)
        labels.append(i % 10)
    return np.array(images), np.array(labels)


def _read_manifest(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_manifest_has_one_row_per_image_with_matching_lengths(tmp_path):
    images, labels = _fake_mnist()
    manifest = build_benchmark_set(images, labels, tmp_path, counts={1: 3, 2: 2, 3: 2})
    rows = _read_manifest(manifest)
    assert len(rows) == 7
    assert [len(row["label"]) for row in rows] == [1, 1, 1, 2, 2, 3, 3]
    for row in rows:
        assert row["label"].isdigit()
        assert (tmp_path / row["image"]).exists()


def test_number_image_is_dark_ink_on_white(tmp_path):
    images, _ = _fake_mnist(3)
    number = np.array(make_number_image(list(images), height=56, margin=16))
    border = np.concatenate((number[0], number[-1], number[:, 0], number[:, -1]))
    assert border.min() == 255
    assert number.min() < 50
    assert number.shape[0] == 56 + 2 * 16


def test_same_seed_builds_the_same_labels(tmp_path):
    images, labels = _fake_mnist()
    first = _read_manifest(build_benchmark_set(images, labels, tmp_path / "a", counts={2: 5}, seed=7))
    second = _read_manifest(build_benchmark_set(images, labels, tmp_path / "b", counts={2: 5}, seed=7))
    assert [row["label"] for row in first] == [row["label"] for row in second]


def test_raises_when_not_enough_digits(tmp_path):
    images, labels = _fake_mnist(5)
    with pytest.raises(ValueError):
        build_benchmark_set(images, labels, tmp_path, counts={3: 2})


def test_segmenter_finds_one_crop_per_digit(tmp_path):
    # make sure the images actually work with segmentation, since that's the first step
    images, labels = _fake_mnist()
    rows = _read_manifest(build_benchmark_set(images, labels, tmp_path, counts={1: 2, 3: 2, 4: 2}))
    for row in rows:
        crops = segment(Image.open(tmp_path / row["image"]), split_touching=True)
        assert len(crops) == len(row["label"])


def test_jitter_zero_gives_the_same_image_as_before(tmp_path):
    images, _ = _fake_mnist(3)
    neat = np.array(make_number_image(list(images)))
    also_neat = np.array(make_number_image(list(images), jitter=0, rng=np.random.default_rng(1)))
    assert np.array_equal(neat, also_neat)


def test_jitter_keeps_the_same_digits_and_labels(tmp_path):
    images, labels = _fake_mnist()
    neat = _read_manifest(build_benchmark_set(images, labels, tmp_path / "neat", counts={3: 4}))
    messy = _read_manifest(build_benchmark_set(images, labels, tmp_path / "messy", counts={3: 4}, jitter=6))
    assert [r["label"] for r in neat] == [r["label"] for r in messy]
    first = np.array(Image.open(tmp_path / "neat" / neat[0]["image"]))
    first_messy = np.array(Image.open(tmp_path / "messy" / messy[0]["image"]))
    assert first.shape != first_messy.shape or not np.array_equal(first, first_messy)


def test_segmenter_still_splits_jittered_numbers(tmp_path):
    images, labels = _fake_mnist()
    rows = _read_manifest(build_benchmark_set(images, labels, tmp_path, counts={2: 3, 4: 3}, jitter=6))
    for row in rows:
        crops = segment(Image.open(tmp_path / row["image"]), split_touching=True)
        assert len(crops) == len(row["label"])
