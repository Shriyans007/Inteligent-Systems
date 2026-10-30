"""Tests for build_photo_manifest.py, using blank images with photo-style names."""

import csv

import pytest
from PIL import Image

from preprocessing.build_photo_manifest import build_photo_manifest, label_from_name


def _make_photos(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name.endswith((".txt", ".heic")):
            (folder / name).write_text("not an image")
        else:
            Image.new("L", (40, 20), color=255).save(folder / name)
    return folder


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_label_comes_from_the_start_of_the_name():
    assert label_from_name("527.jpg") == "527"
    assert label_from_name("527_02.jpg") == "527"
    assert label_from_name("0042_b.png") == "0042"
    assert label_from_name("photo.jpg") is None
    assert label_from_name("12a.jpg") is None


def test_manifest_lists_each_named_photo_with_its_label(tmp_path):
    folder = _make_photos(tmp_path / "photos", ["5.png", "527_01.jpg", "527_02.jpg", "notes.txt"])
    manifest, rows, skipped = build_photo_manifest(folder)
    assert manifest == folder / "manifest.csv"
    assert _read(manifest) == [
        {"image": "5.png", "label": "5"},
        {"image": "527_01.jpg", "label": "527"},
        {"image": "527_02.jpg", "label": "527"},
    ]
    # a text file isn't a photo, so it's ignored rather than reported
    assert skipped == []


def test_badly_named_and_heic_photos_are_reported(tmp_path):
    folder = _make_photos(tmp_path / "photos", ["12.png", "IMG_1234.jpg", "34.heic"])
    _, rows, skipped = build_photo_manifest(folder)
    assert [row["label"] for row in rows] == ["12"]
    assert [name for name, _ in skipped] == ["34.heic", "IMG_1234.jpg"]


def test_paths_still_work_when_manifest_is_saved_elsewhere(tmp_path):
    folder = _make_photos(tmp_path / "photos", ["908.png"])
    manifest, _, _ = build_photo_manifest(folder, tmp_path / "manifests" / "real.csv")
    row = _read(manifest)[0]
    assert (manifest.parent / row["image"]).resolve() == (folder / "908.png").resolve()


def test_errors_for_missing_folder_or_no_named_photos(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_photo_manifest(tmp_path / "missing")
    folder = _make_photos(tmp_path / "photos", ["IMG_0001.jpg"])
    with pytest.raises(ValueError):
        build_photo_manifest(folder)
