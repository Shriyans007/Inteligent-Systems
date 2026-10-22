"""Suggestions are separate from the measured character model output."""

from PIL import Image

from text_recognition.layout import Glyph
from text_recognition.suggestions import suggest_text


def reading(text, heights, separators=None):
    separators = separators or [""] * len(text)
    predictions = [{"character": c, "before": before} for c, before in zip(text, separators)]
    glyphs = [Glyph(Image.new("L", (20, height)), (i * 25, 0, 20, height), before)
              for i, (height, before) in enumerate(zip(heights, separators))]
    return suggest_text(predictions, glyphs)


def test_hello_can_be_suggested_without_replacing_raw_output():
    assert reading("Hell0", [62, 31, 62, 61, 32]) == "Hello"


def test_numbers_and_identifiers_stay_as_predicted():
    assert reading("0", [32]) == "0"
    assert reading("H20", [62, 31, 31]) == "H20"
    assert reading("abc123", [31, 62, 31, 62, 31, 31]) == "abc123"


def test_uppercase_and_separate_words_are_not_forced_lowercase():
    assert reading("COOL", [62, 62, 62, 62]) == "COOL"
    assert reading("ab0", [31, 62, 62]) == "ab0"
    assert reading("ab0", [31, 62, 31], ["", "", " "]) == "ab 0"
