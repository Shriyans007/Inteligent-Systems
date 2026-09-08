import cv2
import numpy as np

def segment_digits(binary_img, method="connected_components", min_area=20):
    """
    binary_img: preprocessed binary image (digits=white, background=black)
    Returns: list of (x, y, w, h) boxes, sorted left-to-right
    """
    boxes = []

    if method == "connected_components":
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_img, connectivity=8
        )
        for i in range(1, num_labels):  # skip background label 0
            x, y, w, h, area = stats[i]
            if area >= min_area:
                boxes.append((x, y, w, h))

    elif method == "contours":
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= min_area:
                boxes.append((x, y, w, h))

    # sort left-to-right so digit order matches reading order
    boxes.sort(key=lambda b: b[0])
    return boxes


def extract_digit_crops(binary_img, boxes, target_size=28, padding=4):
    """
    Crops each box, pads to square, resizes to CNN input size (MNIST-style 28x28).
    """
    crops = []
    for (x, y, w, h) in boxes:
        crop = binary_img[y:y+h, x:x+w]

        # pad to square so resizing doesn't distort the digit
        side = max(w, h) + 2 * padding
        square = np.zeros((side, side), dtype=binary_img.dtype)
        x_off = (side - w) // 2
        y_off = (side - h) // 2
        square[y_off:y_off+h, x_off:x_off+w] = crop

        resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
        crops.append(resized)
    return crops