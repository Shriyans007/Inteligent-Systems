import json

import pytest

from cnn_model.compare_four import collect


def test_comparison_rejects_quick_results(tmp_path):
    baseline = tmp_path / "comparison.json"
    lenet = tmp_path / "lenet.json"
    resnet = tmp_path / "resnet.json"
    baseline.write_text(json.dumps([{"model": "MLP baseline"}, {"model": "CNN selected"}]))
    lenet.write_text(json.dumps({"model_key": "lenet5", "quick": True,
                                 "train_samples": 10000, "test_samples": 2000,
                                 "seed": 42, "batch_size": 128}))
    resnet.write_text(json.dumps({"model_key": "resnet", "quick": False,
                                  "train_samples": 60000, "test_samples": 10000,
                                  "seed": 42, "batch_size": 128}))
    with pytest.raises(ValueError, match="Full-run"):
        collect(baseline, lenet, resnet)
