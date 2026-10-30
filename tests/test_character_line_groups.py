"""Mixed character heights on one line should not add a newline."""

from text_recognition.layout import _line_groups


def test_capital_lowercase_and_dot_share_one_line():
    boxes = [(5, 12, 30, 48), (40, 29, 12, 30), (43, 15, 6, 7),
             (58, 27, 25, 33), (90, 31, 20, 29)]
    groups = _line_groups(boxes)
    assert len(groups) == 1
    assert len(groups[0]) == len(boxes)


def test_distinct_lines_remain_separate():
    boxes = [(5, 10, 25, 30), (36, 18, 16, 22),
             (5, 70, 25, 30), (36, 78, 16, 22)]
    groups = _line_groups(boxes)
    assert len(groups) == 2
    assert [len(group) for group in groups] == [2, 2]
