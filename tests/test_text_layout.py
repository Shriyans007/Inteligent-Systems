"""Check text layout without training or relying on a saved character model."""

from PIL import Image, ImageDraw

from text_recognition.layout import segment_text


def test_close_horizontal_strokes_form_one_character():
    canvas = Image.new("L", (170, 80), 255)
    draw = ImageDraw.Draw(canvas)
    draw.line((15, 9, 15, 65), fill=0, width=5)
    draw.line((42, 9, 42, 65), fill=0, width=5)
    draw.line((21, 37, 42, 37), fill=0, width=5)  # short gap from the left stem
    draw.ellipse((100, 20, 130, 60), outline=0, width=5)
    glyphs = segment_text(canvas)
    assert len(glyphs) == 2
    assert glyphs[0].bbox[0] < 20 and glyphs[0].bbox[2] > 25
    assert glyphs[1].before == " "


def test_nearby_letters_do_not_turn_into_spaces():
    canvas = Image.new("L", (300, 100), 255)
    draw = ImageDraw.Draw(canvas)
    for x in (20, 66, 110, 155, 195):
        draw.rectangle((x, 20, x + 27, 70), fill=0)
    glyphs = segment_text(canvas)
    assert len(glyphs) == 5
    assert [glyph.before for glyph in glyphs] == [""] * 5


def test_wide_gap_and_second_line():
    canvas = Image.new("L", (280, 180), 255)
    draw = ImageDraw.Draw(canvas)
    for x, y in ((20, 15), (60, 15), (160, 15), (200, 15), (25, 105)):
        draw.rectangle((x, y, x + 22, y + 42), fill=0)
    glyphs = segment_text(canvas)
    assert len(glyphs) == 5
    assert [glyph.before for glyph in glyphs] == ["", "", " ", "", "\n"]
