"""The experimental hard cuts must not pass a neighbouring glyph to the CNN."""
from types import SimpleNamespace

import numpy as np

from segmentation.unet_segmenter import tight_boundaries, tight_strip_crop


def test_tight_crops_partition_adjacent_ink_without_duplication():
    image = np.full((40, 70), 255, dtype=np.uint8)
    image[8:32, 10:23] = 0
    image[10:30, 42:55] = 0
    detections = [SimpleNamespace(cx=16, half_width=17),
                  SimpleNamespace(cx=48, half_width=17)]
    cuts = tight_boundaries(detections, image.shape[1])
    first, first_box = tight_strip_crop(image, cuts[0], cuts[1])
    second, second_box = tight_strip_crop(image, cuts[1], cuts[2])
    assert first_box == (10, 8, 13, 24)
    assert second_box == (42, 10, 13, 20)
    assert first.shape[1] < 25 and second.shape[1] < 25


def test_tight_crop_keeps_a_split_ink_stroke_within_its_strip():
    image = np.full((40, 60), 255, dtype=np.uint8)
    image[8:33, 10:24] = 0
    image[2:7, 16:19] = 0  # a disconnected dot belongs to the same glyph
    crop, box = tight_strip_crop(image, 0, 30)
    assert box == (10, 2, 14, 31)
    assert crop.shape[0] >= 31
