"""
segmentation/segmenter.py

Person B's segmentation component for the HNRS pipeline:

    image_acquisition.build_number_from_digits()  ->  [SEGMENTATION]  ->  cnn_model
    (or a single uploaded photo of handwriting)         (this module)      (per-digit prediction)

Verified against the repo
Verified with a real image built via the team's own build_number_from_digits()
-- see tests/test_segmenter.py.
"""

from dataclasses import dataclass
from typing import List, Tuple, Literal
import numpy as np
import cv2
from PIL import Image


@dataclass
class CharacterCrop:
    """A single segmented character. `image` is the RAW (non-binarized)
    crop from the original image -- deliberately unprocessed, so
    preprocessing.prepare_mnist_digit() has full pixel information
    to work with, same as it would for a directly-uploaded single digit."""
    image: np.ndarray                 # raw grayscale crop, 0-255
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in the original image
    index: int                        # left-to-right order


def _to_grayscale_array(image) -> np.ndarray:
    """Accept a file path, PIL Image, or numpy array."""
    if isinstance(image, str):
        image = np.array(Image.open(image).convert("L"))
    elif isinstance(image, Image.Image):
        image = np.array(image.convert("L"))
    elif image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _binary_for_detection(gray: np.ndarray) -> np.ndarray:
    """
    Binarize purely for FINDING character regions -- never used as the
    returned crop itself.

    Detect the canvas background from its border. Photos and tile composites
    have a light outer border, whereas MNIST-style single digits can have a
    dark outer border and bright strokes. Tiled inputs can have dark pixels
    along most of the outer edge; full-height light gaps reveal their canvas.
    The whole-image mean is unsuitable because dark tiles can cover most of it.
    """
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    # Composite tiles can cover the top and bottom edges, but their full-
    # height white gap columns reveal the light canvas between tiles.
    light_gaps = gray.shape[1] > gray.shape[0] and np.any(np.min(gray, axis=0) > 230)
    dark_background = np.median(border) < 128 and not light_gaps
    foreground = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV
    try:
        _, binary = cv2.threshold(gray, 0, 255, foreground + cv2.THRESH_OTSU)
    except cv2.error:
        # OpenCV sometimes fails here on Windows. The NumPy path still finds
        # the ink using Otsu's threshold.
        histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        counts = np.cumsum(histogram)
        weighted = np.cumsum(histogram * np.arange(256))
        valid = (counts > 0) & (counts < gray.size)
        variance = np.zeros(256, dtype=np.float64)
        variance[valid] = ((weighted[-1] * counts[valid] - weighted[valid] * gray.size) ** 2
                           / (counts[valid] * (gray.size - counts[valid])))
        threshold = int(np.argmax(variance))
        mask = gray > threshold if dark_background else gray <= threshold
        binary = np.where(mask, 255, 0).astype(np.uint8)
    return binary


def _filter_boxes(boxes, image_shape, min_area):
    h_img, w_img = image_shape
    kept = []
    for (x, y, w, h) in boxes:
        if w * h < min_area:
            continue
        if w > 0.98 * w_img and h > 0.98 * h_img:
            continue
        kept.append((x, y, w, h))
    return kept


def _merge_overlapping_boxes(boxes, overlap_thresh: float = 0.3):
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]
    for (x, y, w, h) in boxes[1:]:
        px, py, pw, ph = merged[-1]
        overlap = max(0, (px + pw) - x)
        smaller_w = min(pw, w)
        if smaller_w > 0 and overlap / smaller_w > overlap_thresh:
            nx, ny = min(px, x), min(py, y)
            nw = max(px + pw, x + w) - nx
            nh = max(py + ph, y + h) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append((x, y, w, h))
    return merged


def segment_connected_components(binary_image: np.ndarray, min_area: int = 20):
    try:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    except cv2.error:
        from skimage.measure import label, regionprops
        regions = regionprops(label(binary_image > 0, connectivity=2))
        return [(region.bbox[1], region.bbox[0], region.bbox[3] - region.bbox[1],
                 region.bbox[2] - region.bbox[0]) for region in regions]
    boxes = []
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, w, h))
    return boxes


def segment_contours(binary_image: np.ndarray, min_area: int = 20):
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    if hierarchy is None:
        return boxes
    hierarchy = hierarchy[0]
    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1:  # skip holes (e.g. the digit stroke inside a tile)
            continue
        boxes.append(cv2.boundingRect(contour))
    return boxes


def _split_touching_box(binary: np.ndarray, box: Tuple[int, int, int, int]):
    """Split a wide connected digit group only at a narrow ink bridge.

    A single wide stroke or loop must stay intact, so require a wide bounding
    box and substantial ink on either side of the proposed cut. This is only
    used for the number route; it cannot resolve every pair of touching digits.
    """
    x, y, w, h = box
    if w < 20 or w < 1.25 * h:
        return [box]
    ink = binary[y:y+h, x:x+w] > 0
    # A narrow column near the middle can be the join between two digits.
    projection = ink.sum(axis=0)
    left_limit = max(1, int(w * 0.26))
    right_limit = min(w - 1, int(w * 0.74))
    if left_limit >= right_limit:
        return [box]
    cut = min(range(left_limit, right_limit), key=lambda col: (projection[col], abs(col - w/2)))
    if projection[cut] > max(2, 0.10 * h) or projection[cut] > 0.4 * projection.max():
        return [box]
    left = ink[:, :cut]
    right = ink[:, cut:]
    # Avoid turning a long tail or stray mark into a separate digit.
    if left.sum() < 0.18 * ink.sum() or right.sum() < 0.18 * ink.sum():
        return [box]
    pieces = []
    for offset, part in ((0, left), (cut, right)):
        rows, cols = np.where(part)
        if not len(rows):
            return [box]
        pieces.extend(_split_touching_box(binary, (x + offset + int(cols.min()), y + int(rows.min()),
                         int(np.ptp(cols)) + 1, int(np.ptp(rows)) + 1)))
    return pieces


def segment(
    image,
    method: Literal["connected_components", "contours"] = "connected_components",
    min_area: int = 20,
    padding: int = 2,
    split_touching: bool = False,
) -> List[CharacterCrop]:
    """
    Main entry point. `image` can be a file path, PIL Image, or numpy array
    -- either a composite "number" image from build_number_from_digits(), or
    a single photo containing multiple handwritten characters.

    Returns ordered (left-to-right) CharacterCrop objects with RAW pixel
    data, ready to be passed straight into
    preprocessing.prepare_mnist_digit().
    """
    gray = _to_grayscale_array(image)
    binary = _binary_for_detection(gray)

    if method == "connected_components":
        boxes = segment_connected_components(binary, min_area=min_area)
    elif method == "contours":
        boxes = segment_contours(binary, min_area=min_area)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    boxes = _filter_boxes(boxes, binary.shape[:2], min_area=min_area)
    boxes = _merge_overlapping_boxes(boxes)
    if split_touching:
        separated = []
        for box in boxes:
            pieces = _split_touching_box(binary, box)
            separated.extend(pieces)
        boxes = separated
    boxes = sorted(boxes, key=lambda b: b[0])

    crops = []
    h_img, w_img = gray.shape[:2]
    for idx, (x, y, w, h) in enumerate(boxes):
        left_padding = right_padding = padding
        if split_touching:
            # Close crops must not include strokes from their neighbours.
            if idx:
                prev_x, _, prev_w, _ = boxes[idx-1]
                left_padding = min(padding, max(0, (x - prev_x - prev_w)//2))
            if idx + 1 < len(boxes):
                next_x, _, _, _ = boxes[idx+1]
                right_padding = min(padding, max(0, (next_x - x - w)//2))
        x0, y0 = max(0, x - left_padding), max(0, y - padding)
        x1, y1 = min(w_img, x + w + right_padding), min(h_img, y + h + padding)
        crop = gray[y0:y1, x0:x1]  # <-- raw grayscale, NOT the binary mask
        crops.append(CharacterCrop(image=crop, bbox=(x, y, w, h), index=idx))
    return crops


def crops_to_model_input(crops: List[CharacterCrop]) -> List[np.ndarray]:
    """
    Convert each raw crop to the (1,28,28,1) float32 tensor the CNN expects.
    This uses the shared preprocessing module so the full system has only one
    implementation of the image preparation logic.
    """
    from preprocessing import prepare_mnist_digit

    tensors = []
    for c in crops:
        pil_crop = Image.fromarray(c.image)
        tensors.append(prepare_mnist_digit(pil_crop))
    return tensors
