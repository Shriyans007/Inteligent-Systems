"""Tests for the ResNet U-Net character segmenter (no EMNIST files needed).

The training check runs a few dozen steps on the CPU (about 15 s). The accuracy
tests use the saved model if you've trained one, otherwise they're skipped.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from segmentation.synthetic_lines import (StubSource, make_batch, make_targets, render_line,
                                          vertical_metrics)
from segmentation.unet_resnet import UNET_MODEL, build_unet_resnet, centre_loss
from segmentation.unet_segmenter import (CharBox, UNetSegmenter, decode_peaks, pad_to_multiple,
                                         remove_disconnected_neighbours, window_crop)


def test_model_is_small_enough_for_a_cpu_and_accepts_any_line_size():
    model = build_unet_resnet()
    assert model.count_params() < 1_000_000
    for h, w in ((64, 384), (48, 208), (96, 512)):
        out = model.predict(np.zeros((1, h, w, 1), np.float32), verbose=0)
        assert out.shape == (1, h // 2, w // 2, 2)


def test_pad_to_multiple_keeps_ink_and_sizes_are_divisible():
    padded, top, left = pad_to_multiple(np.ones((37, 101), np.float32))
    assert padded.shape[0] % 16 == 0 and padded.shape[1] % 16 == 0
    assert padded[top:top + 37, :101].min() == 1.0


def test_targets_have_exactly_one_peak_per_character():
    rng = np.random.default_rng(0)
    _, glyphs = render_line(StubSource(), "butterfly Hello", rng, 384, cap_height=36)
    heat = make_targets(glyphs)[..., 0]
    assert int((heat >= 0.999).sum()) == len(glyphs) == 14


def test_generator_really_produces_overlapping_letters():
    rng = np.random.default_rng(1)
    overlaps = [g.overlap for _ in range(20)
                for g in render_line(StubSource(), "butterfly bottle", rng)[1]]
    assert max(overlaps) > 0.3             # shared strokes occur
    assert sum(o > 0.05 for o in overlaps) > 40


def test_sizes_follow_case():
    assert vertical_metrics("C")[0] > vertical_metrics("c")[0]
    assert vertical_metrics("y")[1] < 0 and vertical_metrics("Y")[1] == 0


def test_decode_peaks_keeps_close_overlapping_peaks_but_drops_noise():
    heat = np.zeros((32, 96), np.float32)
    width = np.full((32, 96), 0.15, np.float32)       # half-width = 0.15 * 64 = ~10 px
    heat[16, 20] = 1.0                                # two characters whose centres are 10 px
    heat[16, 25] = 0.9                                # apart (5 columns at half resolution)
    heat[16, 70] = 0.1                                # weak noise
    peaks = decode_peaks(heat, width, threshold=0.3)
    assert len(peaks) == 2
    assert peaks[0][0] < peaks[1][0]


def test_decode_peaks_merges_two_peaks_inside_one_wide_character():
    heat = np.zeros((32, 96), np.float32)
    width = np.full((32, 96), 0.2, np.float32)
    heat[16, 20] = 1.0
    heat[16, 22] = 0.8                                # 4 px apart inside a ~26 px character
    assert len(decode_peaks(heat, width)) == 1


def test_window_crop_works_for_both_ink_polarities():
    line = np.zeros((64, 200), np.uint8)
    line[10:50, 90:110] = 255
    dark = window_crop(line, 100, 12)
    light = window_crop(255 - line, 100, 12)
    assert dark.shape == light.shape and dark.shape[0] <= 44


def test_disconnected_neighbour_is_removed_but_shared_stroke_remains():
    import cv2

    white = np.full((64, 140), 255, np.uint8)
    cv2.line(white, (18, 10), (30, 50), 0, 4)
    cv2.circle(white, (56, 36), 11, 0, 4)
    boxes = [CharBox(15, 10, 18, 41, 24, 25, 0.9),
             CharBox(45, 25, 23, 23, 56, 25, 0.9)]
    for line in (white, 255 - white):
        cleaned = remove_disconnected_neighbours(line, boxes, 1)
        assert np.all(cleaned[10:51, 16:33] == line[0, 0])
        assert np.array_equal(cleaned[25:48, 45:68], line[25:48, 45:68])

    joined = np.full((64, 140), 255, np.uint8)
    cv2.line(joined, (24, 35), (90, 35), 0, 4)
    shared = [CharBox(23, 33, 35, 5, 32, 25, 0.9),
              CharBox(56, 33, 36, 5, 80, 25, 0.9)]
    assert np.array_equal(remove_disconnected_neighbours(joined, shared, 0), joined)
    assert np.array_equal(remove_disconnected_neighbours(joined, shared, 1), joined)


def test_a_few_training_steps_reduce_the_loss():
    keras = pytest.importorskip("keras")
    rng = np.random.default_rng(0)
    model = build_unet_resnet()
    model.compile(optimizer=keras.optimizers.Adam(2e-3), loss=centre_loss)
    x, y, _ = make_batch(StubSource(), 4, rng, width=192)
    first = float(np.asarray(model.train_on_batch(x, y)).mean())
    for _ in range(25):
        x, y, _ = make_batch(StubSource(), 4, rng, width=192)
        last = float(np.asarray(model.train_on_batch(x, y)).mean())
    assert last < first


@pytest.mark.skipif(not (ROOT / UNET_MODEL).is_file(), reason="train the U-Net first")
def test_trained_unet_splits_shared_crossbar_and_overlapped_letters():
    segmenter = UNetSegmenter(ROOT / UNET_MODEL)
    rng = np.random.default_rng(5)
    exact = 0
    for _ in range(10):
        image, glyphs = render_line(StubSource(), "butter", rng, 384, cap_height=36, overlap=True)
        exact += len(segmenter.find_characters((image * 255).astype(np.uint8))) == len(glyphs)
    assert exact >= 6
