"""Training crops that look exactly like what the classifier receives at run time.

At run time the U-Net gives a window per character that is a little wider than
the character itself, so the crop often contains a sliver of a neighbour or a
shared crossbar. This module builds such crops from synthetic overlapping lines
(using the true character positions, with a bit of jitter to mimic the U-Net's
small errors) and runs them through the team's own ``prepare_character`` so the
classifier can be fine-tuned on them.
"""
import numpy as np
from PIL import Image
from types import SimpleNamespace

from segmentation.synthetic_lines import augment_line, random_text, render_line
from segmentation.unet_segmenter import window_crop, tight_boundaries, tight_strip_crop
from text_recognition.common import prepare_character


def overlap_crops(source, char_to_label, n_lines, seed=0, width=384, overlap=True,
                  crop_mode="window"):
    """Return (x float32 [N,28,28,1], y int64 [N]) of crops from overlapping lines."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for _ in range(n_lines):
        text = random_text(rng, source.available, max_chars=int(rng.integers(8, 28)))
        canvas, glyphs = render_line(source, text, rng, width, overlap=overlap)
        gray = (augment_line(canvas, rng) * 255).astype(np.uint8)
        if crop_mode not in ("window", "tight"):
            raise ValueError(f"Unknown crop mode: {crop_mode}")
        estimated = ([SimpleNamespace(cx=g.cx + rng.uniform(-0.08, 0.08) * g.w,
                                      half_width=g.w / 2 * rng.uniform(0.85, 1.15)) for g in glyphs]
                     if crop_mode == "tight" else None)
        cuts = tight_boundaries(estimated, gray.shape[1]) if estimated else None
        for index, g in enumerate(glyphs):
            # The U-Net is accurate to a few pixels; imitate that so training is not too clean.
            if crop_mode == "tight":
                item = tight_strip_crop(gray, cuts[index], cuts[index + 1])
                crop = item[0] if item is not None else None
            else:
                cx = g.cx + rng.uniform(-0.08, 0.08) * g.w
                half_width = g.w / 2 * rng.uniform(0.85, 1.15)
                crop = window_crop(gray, cx, half_width)
            if crop is None:
                continue
            try:
                images.append(prepare_character(Image.fromarray(crop)))
            except ValueError:
                continue
            labels.append(char_to_label[g.char])
    return np.asarray(images, np.float32), np.asarray(labels, np.int64)
