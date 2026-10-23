"""Small handwriting shifts for training only; input and output are 0..1."""

import numpy as np
from PIL import Image


def augment_batch(images, seed):
    rng = np.random.default_rng(seed)
    result = np.empty_like(images)
    for index, image in enumerate(images):
        angle = np.deg2rad(rng.uniform(-7, 7))
        shear = rng.uniform(-0.07, 0.07)
        shift_x, shift_y = rng.uniform(-1.5, 1.5, size=2)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        slant = np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1]])
        center = np.array([[1, 0, 13.5], [0, 1, 13.5], [0, 0, 1]])
        move = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]])
        forward = move @ center @ rotation @ slant @ np.linalg.inv(center)
        inverse = np.linalg.inv(forward)[:2].reshape(-1)
        source = Image.fromarray((image[:, :, 0] * 255).astype(np.uint8))
        transformed = source.transform((28, 28), Image.Transform.AFFINE, tuple(inverse),
                                       resample=Image.Resampling.BILINEAR, fillcolor=0)
        result[index, :, :, 0] = np.asarray(transformed, dtype=np.float32) / 255
    return result
