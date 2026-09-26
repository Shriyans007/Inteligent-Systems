"""Check that the two new classifiers can be saved for the API input contract."""

import numpy as np
import pytest


def test_new_models_build_predict_and_reload(tmp_path):
    keras = pytest.importorskip("tensorflow").keras
    from cnn_model.models import build_lenet5, build_resnet, compile_model

    for name, builder in (("lenet5", build_lenet5), ("resnet", build_resnet)):
        model = compile_model(builder())
        assert model.input_shape == (None, 28, 28, 1)
        assert model.output_shape == (None, 10)
        if name == "resnet":
            assert sum(isinstance(layer, keras.layers.Add) for layer in model.layers) == 4
        # One synthetic batch verifies the compiled training path; it is not
        # an accuracy experiment and produces no reportable model result.
        model.train_on_batch(np.zeros((2, 28, 28, 1), dtype=np.float32),
                             np.array([0, 1]))
        path = tmp_path / f"{name}.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        probabilities = loaded.predict(np.zeros((1, 28, 28, 1), dtype=np.float32), verbose=0)
        assert probabilities.shape == (1, 10)
        np.testing.assert_allclose(probabilities.sum(axis=1), [1], atol=1e-5)
