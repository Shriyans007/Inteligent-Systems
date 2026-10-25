import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_recognition.case import resolve_case

# Boxes are (x, y, w, h); the baseline of the line is y = 100.
TALL = lambda x: (x, 60, 20, 40)        # full height, sits on the baseline
SHORT = lambda x: (x, 78, 20, 22)       # x-height, sits on the baseline


def test_short_o_after_capital_h_is_lowercase():
    chars = resolve_case(list("HellO"), [TALL(0), SHORT(30), TALL(60), TALL(90), SHORT(120)])
    assert "".join(chars) == "Hello"


def test_tall_c_before_short_letters_becomes_capital():
    chars = resolve_case(list("cat"), [TALL(0), SHORT(30), (60, 68, 20, 32)])
    assert chars[0] == "C" and chars[1] == "a"


def test_hanging_y_is_lowercase_but_full_height_y_is_capital():
    hanging = (30, 78, 20, 40)          # top at x-height, hangs 18 px below the baseline
    assert resolve_case(["H", "y"], [TALL(0), hanging])[1] == "y"
    assert resolve_case(["H", "y"], [TALL(0), TALL(30)])[1] == "Y"


def test_no_reference_letters_keeps_the_classifier_answer():
    assert resolve_case(["s", "O"], [SHORT(0), TALL(30)]) == ["s", "O"]
    assert resolve_case(["S"], [SHORT(0)]) == ["S"]


def test_each_line_is_judged_on_its_own_scale():
    boxes = [TALL(0), SHORT(30), (0, 260, 10, 20), (30, 270, 10, 10)]   # second line is smaller writing
    chars = resolve_case(["H", "o", "H", "O"], boxes, [0, 0, 1, 1])
    assert chars[1] == "o"
