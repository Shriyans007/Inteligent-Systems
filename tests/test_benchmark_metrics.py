"""Tests for benchmark_metrics.py with small results that are easy to check by hand."""

import csv
import json

import pytest

from preprocessing.benchmark_metrics import (edit_distance, load_results, save_accuracy_plot, save_outputs,
                                             shared_segmentation_errors, summarise)


def _case(expected, predicted):
    return {"image": f"{expected}.png", "expected": expected, "predicted": predicted}


def test_edit_distance_counts_swaps_missing_and_extra_digits():
    assert edit_distance("123", "123") == 0
    assert edit_distance("123", "193") == 1
    assert edit_distance("123", "13") == 1
    assert edit_distance("123", "1234") == 1
    assert edit_distance("12", "") == 2


def test_summary_splits_segmentation_and_classifier_errors():
    cases = [
        _case("5", "5"),
        _case("7", "1"),        # classifier error
        _case("42", "42"),
        _case("42", "49"),      # classifier error, 2 -> 9
        _case("308", "38"),     # segmentation missed a digit
    ]
    overall = summarise(cases)["overall"]
    assert overall["exact_matches"] == 2
    assert overall["segmentation_errors"] == 1
    assert overall["classifier_errors"] == 2
    # "308" is left out, so it's 4 right out of 6 lined-up digits
    assert overall["digit_accuracy"] == pytest.approx(4 / 6)
    # 3 edits over 9 expected digits
    assert overall["character_error_rate"] == pytest.approx(3 / 9)


def test_summary_groups_by_length_and_finds_confusions():
    summary = summarise([_case("4", "9"), _case("4", "9"), _case("1", "1"), _case("24", "29")])
    assert summary["by_length"][1]["exact_accuracy"] == pytest.approx(1 / 3)
    assert summary["by_length"][2]["exact_accuracy"] == 0.0
    assert summary["top_confusions"][0] == ("4->9", 3)


def test_digit_accuracy_is_none_when_nothing_lines_up():
    summary = summarise([_case("308", "38"), _case("5", "5")])
    assert summary["by_length"][3]["digit_accuracy"] is None


def test_summary_raises_on_empty_results():
    with pytest.raises(ValueError):
        summarise([])


def test_shared_segmentation_errors_only_keeps_images_wrong_for_every_model():
    results = {
        "cnn": [_case("308", "38"), _case("12", "123"), _case("5", "5")],
        "svm": [_case("308", "38"), _case("12", "17"), _case("5", "5")],
    }
    assert shared_segmentation_errors(results) == ["308.png"]


def test_load_results_uses_model_key_or_file_name(tmp_path):
    with_model = tmp_path / "a.json"
    with_model.write_text(json.dumps({"model": "svm", "cases": [_case("5", "5")]}))
    without_model = tmp_path / "benchmark_old.json"
    without_model.write_text(json.dumps({"cases": [_case("5", "3")]}))
    assert set(load_results([with_model, without_model])) == {"svm", "benchmark_old"}


def test_save_outputs_writes_one_comparison_csv_for_all_models(tmp_path):
    summaries = {
        "cnn": summarise([_case("5", "5"), _case("12", "12")]),
        "svm": summarise([_case("5", "5"), _case("12", "17")]),
    }
    json_path, csv_path = save_outputs(summaries, [], tmp_path)
    assert set(json.loads(json_path.read_text())["models"]) == {"cnn", "svm"}
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    # each model gets a row per number length plus an "all" row
    assert [(r["model"], r["digits"]) for r in rows] == [
        ("cnn", "1"), ("cnn", "2"), ("cnn", "all"), ("svm", "1"), ("svm", "2"), ("svm", "all")]
    assert rows[-1]["exact_accuracy"] == "0.5"


def test_load_results_explains_missing_file(tmp_path):
    missing = tmp_path / "benchmark_svm.json"
    with pytest.raises(FileNotFoundError, match="evaluate_uploads"):
        load_results([missing])


def test_load_results_rejects_file_without_cases(tmp_path):
    wrong = tmp_path / "comparison.json"
    wrong.write_text(json.dumps([{"model": "CNN selected", "test_accuracy": 0.99}]))
    with pytest.raises(ValueError, match="cases"):
        load_results([wrong])


def test_save_accuracy_plot_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    summaries = {
        "cnn": summarise([_case("5", "5"), _case("12", "12"), _case("308", "38")]),
        "svm": summarise([_case("5", "3"), _case("12", "12"), _case("308", "308")]),
    }
    path = save_accuracy_plot(summaries, tmp_path / "plots" / "accuracy.png")
    # every PNG file starts with the same 8 bytes
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
