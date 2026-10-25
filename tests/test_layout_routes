import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_recognition import layout


def _two_separate_strokes():
    canvas = np.full((80, 200), 255, np.uint8)
    cv2.line(canvas, (30, 12), (30, 65), 0, 6)
    cv2.line(canvas, (150, 12), (150, 65), 0, 6)
    return Image.fromarray(canvas)


def test_auto_falls_back_to_the_classical_route_when_no_unet_is_trained(monkeypatch):
    monkeypatch.setattr(layout, "_get_unet", lambda: None)
    assert len(layout.segment_text(_two_separate_strokes())) == 2


def test_unet_route_reports_missing_weights(monkeypatch):
    monkeypatch.setattr(layout, "_get_unet", lambda: None)
    with pytest.raises(FileNotFoundError):
        layout.segment_text(_two_separate_strokes(), method="unet")


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        layout.segment_text(_two_separate_strokes(), method="magic")
