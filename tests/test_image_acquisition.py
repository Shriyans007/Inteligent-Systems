"""Basic tests for image_acquisition.py - covers load_number_image() and
build_number_from_digits(), since preprocess_pipeline() already had tests
but these two didn't have their own yet."""

from PIL import Image
import pytest

from preprocessing.image_acquisition import build_number_from_digits, load_number_image


def _make_digit_folder(tmp_path, widths, height=28):
    """Creates a folder of plain grayscale digit images with the given widths,
    named so their sorted filename order matches the order given here."""
    folder = tmp_path / "digits"
    folder.mkdir()
    for i, width in enumerate(widths):
        Image.new("L", (width, height), color=0).save(folder / f"digit_{i}.png")
    return folder


def test_load_number_image_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_number_image("this/path/does/not/exist.png")


def test_load_number_image_returns_the_saved_image(tmp_path):
    path = tmp_path / "number.png"
    Image.new("L", (60, 28), color=255).save(path)
    loaded = load_number_image(str(path))
    assert loaded.size == (60, 28)


def test_build_number_from_digits_raises_on_missing_folder():
    with pytest.raises(FileNotFoundError):
        build_number_from_digits("this/folder/does/not/exist")


def test_build_number_from_digits_raises_when_folder_has_no_images(tmp_path):
    empty_folder = tmp_path / "empty"
    empty_folder.mkdir()
    (empty_folder / "notes.txt").write_text("not an image")
    with pytest.raises(FileNotFoundError):
        build_number_from_digits(str(empty_folder))


def test_build_number_from_digits_resizes_to_requested_height(tmp_path):
    # digits start at height 10, well below the 28 the pipeline expects
    folder = _make_digit_folder(tmp_path, widths=[20, 40], height=10)
    composite = build_number_from_digits(str(folder), height=28, spacing=4)
    assert composite.height == 28


def test_build_number_from_digits_combines_expected_width_with_spacing(tmp_path):
    # digits are already at the target height, so their width shouldn't change
    folder = _make_digit_folder(tmp_path, widths=[20, 30], height=28)
    composite = build_number_from_digits(str(folder), height=28, spacing=5)
    assert composite.width == 20 + 30 + 5
