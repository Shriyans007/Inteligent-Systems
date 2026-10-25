"""Synthetic lines of overlapping handwriting for training the character U-Net.

Real handwriting with per-character boxes and heavy overlap barely exists, so
we build it: take real single-character images (EMNIST), scale each to the
right size for its case, then paste them into a line, deliberately overlapping
neighbours (an "a" over an "e", two "t"s sharing one crossbar). Because we know
exactly where every character went, we get perfect labels for free.

Sizes follow normal handwriting: capitals, digits and tall lowercase letters
(b d h k l f) are full height, most lowercase letters are about half height
(a c e m n o r s u v w x z), and g j p q y hang below the baseline. That is also
what lets the case resolver tell "c" from "C" later.
"""
from dataclasses import dataclass
import cv2
import numpy as np

LINE_H = 64
STRIDE = 2

UPPER_LIKE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
ASCENDER = set("bdhklf")
XHEIGHT = set("acemnorsuvwxz")
DESCENDER = set("gpqy")

WORDS = ["butterfly", "letter", "bottle", "attempt", "button", "little", "coffee", "apple",
         "hello", "world", "summer", "yellow", "pattern", "different", "network", "computer",
         "handwriting", "recognition", "segment", "digits", "Sydney", "Melbourne", "Thursday",
         "football", "kitten", "puzzle", "happy", "balloon", "pillow", "cotton", "scatter",
         "stuff", "effort", "offer", "afford", "buffer", "muffin", "tattoo", "settle", "attitude",
         "Swinburne", "Akhila", "Tuesday", "better", "matter", "pretty", "wallet", "hammer",
         "success", "address", "tennis", "following", "writing", "shuffle", "waffle", "kettle"]


def vertical_metrics(ch):
    """(top, bottom) of a character relative to the baseline, in cap-heights."""
    if ch in UPPER_LIKE or ch in ASCENDER:
        return 1.0, 0.0
    if ch in "ti":
        return 0.8, 0.0
    if ch in XHEIGHT:
        return 0.55, 0.0
    if ch in DESCENDER:
        return 0.55, -0.35
    if ch == "j":
        return 0.8, -0.35
    return 1.0, 0.0


@dataclass
class GlyphInfo:
    char: str
    x0: int
    x1: int
    y0: int
    y1: int
    overlap: float = 0.0   # fraction of the smaller glyph covered by its left neighbour

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0


def tight_crop(tile, threshold=0.2):
    """Crop a float tile to its ink. Returns None for an empty tile."""
    mask = tile > threshold * max(float(tile.max()), 1e-6)
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    return tile[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


class EmnistSource:
    """Draw random real character images (28x28, white ink on black) for a label."""

    def __init__(self, images, labels, mapping):
        self.images = images
        self.chars = {char: np.flatnonzero(labels == index) for index, char in mapping.items()}
        self.chars = {c: ids for c, ids in self.chars.items() if len(ids)}

    def sample(self, char, rng):
        ids = self.chars[char]
        image = self.images[ids[rng.integers(len(ids))]]
        return image.reshape(28, 28).astype(np.float32) / 255

    @property
    def available(self):
        return set(self.chars)


class StubSource:
    """EMNIST-like characters drawn with an OpenCV font.

    Used by the unit tests and for quick checks on machines without the EMNIST
    files. The look is cleaner than real handwriting, so accuracy measured with
    this source is only a smoke test, not a real result.
    """
    FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
             cv2.FONT_HERSHEY_TRIPLEX]

    @property
    def available(self):
        return set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

    def sample(self, char, rng):
        font = self.FONTS[rng.integers(len(self.FONTS))]
        canvas = np.zeros((96, 96), np.uint8)
        cv2.putText(canvas, char, (20, 70), font, rng.uniform(1.6, 2.2), 255,
                    int(rng.integers(4, 8)), cv2.LINE_AA)
        crop = tight_crop(canvas.astype(np.float32) / 255)
        h, w = crop.shape
        scale = 20 / max(h, w)
        crop = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))),
                          interpolation=cv2.INTER_AREA)
        tile = np.zeros((28, 28), np.float32)
        y, x = (28 - crop.shape[0]) // 2, (28 - crop.shape[1]) // 2
        tile[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        return cv2.GaussianBlur(tile, (3, 3), 0.6)


def _prepare_glyph(tile, char, cap_height, rng):
    """Scale a character tile to its size for this case, with a little rotation and slant."""
    crop = tight_crop(tile)
    if crop is None:
        return None
    top, bottom = vertical_metrics(char)
    target_h = max(5, int(round((top - bottom) * cap_height * rng.uniform(0.92, 1.08))))
    h, w = crop.shape
    new_w = int(np.clip(round(w * target_h / h * rng.uniform(0.85, 1.2)), 2, 1.3 * cap_height))
    interp = cv2.INTER_AREA if target_h < h else cv2.INTER_CUBIC
    glyph = cv2.resize(crop, (new_w, target_h), interpolation=interp)
    # Slight rotation and slant, like real pen strokes.
    pad = 6
    glyph = np.pad(glyph, pad)
    angle, shear = rng.uniform(-8, 8), rng.uniform(-0.22, 0.22)
    centre = (glyph.shape[1] / 2, glyph.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    matrix[0, 1] += shear
    glyph = cv2.warpAffine(glyph, matrix, (glyph.shape[1], glyph.shape[0]))
    return np.clip(tight_crop(glyph, 0.1), 0, 1)


def _sample_overlap(rng, left, right):
    """Fraction of the smaller glyph that the neighbour covers (negative = a gap)."""
    kind = rng.random()
    if left in "tf" and right in "tf":     # shared crossbar, as in "butterfly"
        return rng.uniform(0.3, 0.55) if kind < 0.7 else rng.uniform(-0.1, 0.3)
    if kind < 0.50:
        return rng.uniform(-0.25, 0.05)
    if kind < 0.82:
        return rng.uniform(0.05, 0.35)
    return rng.uniform(0.35, 0.55)


def random_text(rng, available, max_chars=40):
    """Mostly real words (many double letters), sometimes random letters and digits."""
    words = []
    while sum(len(w) + 1 for w in words) < max_chars:
        roll = rng.random()
        if roll < 0.55:
            word = WORDS[rng.integers(len(WORDS))]
            if rng.random() < 0.25:
                word = word.capitalize()
            elif rng.random() < 0.05:
                word = word.upper()
        elif roll < 0.9:
            size = int(rng.integers(2, 8))
            word = "".join(chr(97 + rng.integers(26)) for _ in range(size))
            if rng.random() < 0.3:
                word = word.capitalize()
        else:
            word = "".join(str(rng.integers(10)) for _ in range(int(rng.integers(1, 5))))
        words.append("".join(c for c in word if c in available))
    return " ".join(w for w in words if w)


def render_line(source, text, rng, width=384, cap_height=None, overlap=True):
    """Paste characters into a line. Returns (image float32 HxW, [GlyphInfo]).

    overlap=False leaves a clear gap between every pair of letters (used to measure
    how the model does on easy, separated writing).
    """
    cap_height = cap_height or rng.uniform(26, 42)
    canvas = np.zeros((LINE_H, width), np.float32)
    low, high = cap_height + 3, LINE_H - 0.35 * cap_height - 3
    baseline = rng.uniform(low, high) if high > low else low
    x, prev_right, prev_w, prev_char = rng.uniform(4, 14), None, 0, None
    glyphs = []
    for char in text:
        if char == " ":
            x += rng.uniform(0.5, 1.0) * cap_height
            prev_right, prev_char = None, None
            continue
        glyph = _prepare_glyph(source.sample(char, rng), char, cap_height, rng)
        if glyph is None:
            continue
        gh, gw = glyph.shape
        overlap_fraction = 0.0
        if prev_right is not None:
            overlap_fraction = _sample_overlap(rng, prev_char, char) if overlap else rng.uniform(-0.3, -0.08)
            x = prev_right - overlap_fraction * min(prev_w, gw)
        x0 = int(round(x))
        top, _ = vertical_metrics(char)
        y0 = int(round(baseline - top * cap_height + rng.uniform(-1.5, 1.5)))
        y0 = int(np.clip(y0, 0, LINE_H - gh))
        if x0 + gw > width - 2:
            break
        region = canvas[y0:y0 + gh, x0:x0 + gw]
        np.maximum(region, glyph, out=region)
        glyphs.append(GlyphInfo(char, x0, x0 + gw, y0, y0 + gh, max(0.0, overlap_fraction)))
        prev_right, prev_w, prev_char = x0 + gw, gw, char
        x = x0 + gw
    return canvas, glyphs


def augment_line(image, rng):
    """Stroke thickness, blur and noise so the net is not tuned to one pen."""
    if rng.random() < 0.35:
        image = cv2.dilate(image, np.ones((2, 2), np.uint8))
    if rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.3, 0.9))
    image = image * rng.uniform(0.75, 1.0) + rng.normal(0, rng.uniform(0, 0.03), image.shape)
    return np.clip(image, 0, 1).astype(np.float32)


def make_targets(glyphs, height=LINE_H, width=384, stride=STRIDE):
    """Heat-map of character centres, half-width map and its mask (all at 1/stride)."""
    h, w = height // stride, width // stride
    heat = np.zeros((h, w), np.float32)
    half_width = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for g in glyphs:
        cx = int(np.clip(round(g.cx / stride), 0, w - 1))
        cy = int(np.clip(round(g.cy / stride), 0, h - 1))
        sx, sy = max(1.2, g.w / stride / 8), max(1.5, g.h / stride / 6)
        peak = np.exp(-((xx - cx) ** 2 / (2 * sx ** 2) + (yy - cy) ** 2 / (2 * sy ** 2)))
        heat = np.maximum(heat, peak.astype(np.float32))
        window = (slice(max(0, cy - 1), cy + 2), slice(max(0, cx - 1), cx + 2))
        half_width[window] = (g.w / 2) / LINE_H
        mask[window] = 1
    return np.stack([heat, half_width, mask], axis=-1)


def make_batch(source, batch_size, rng, width=384):
    """Return x (B, 64, W, 1), y (B, 32, W/2, 3) and the glyph lists."""
    xs, ys, infos = [], [], []
    available = source.available
    for _ in range(batch_size):
        text = random_text(rng, available, max_chars=int(rng.integers(8, 28)))
        image, glyphs = render_line(source, text, rng, width)
        xs.append(augment_line(image, rng)[..., None])
        ys.append(make_targets(glyphs, LINE_H, width))
        infos.append(glyphs)
    return np.stack(xs), np.stack(ys), infos
