import io
import json
import base64
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
import api.main as api


class CharacterModel:
    def predict(self, x, verbose=0):
        assert x.ndim == 4 and x.shape[1:] == (28, 28, 1)
        probabilities = np.zeros((len(x), 62), dtype=np.float32)
        probabilities[:, 11] = 1
        return probabilities


def test_extension_availability_and_character_route(tmp_path, monkeypatch):
    char, meta = tmp_path/'char.keras', tmp_path/'mapping.json'
    char.touch(); meta.write_text(json.dumps({'11': 'B'}))
    monkeypatch.setattr(api, 'CHARACTER_PATHS', {'cnn': (char, meta), 'mlp': (tmp_path/'missing-mlp.keras', meta)})
    monkeypatch.setattr(api, '_extension_models', {'cnn': (CharacterModel(), {'11': 'B'})})
    canvas = Image.new('L', (70, 70), 255)
    ImageDraw.Draw(canvas).line((20, 10, 30, 55), fill=0, width=5)
    image = io.BytesIO(); canvas.save(image, 'PNG')
    with TestClient(api.app) as client:
        assert [x['available'] for x in client.get('/api/extension-models').json()['character_models']] == [True, False]
        response = client.post('/api/recognise-text', files={'file': ('b.png', image.getvalue(), 'image/png')})
        assert response.status_code == 200 and response.json()['text'] == 'B' and response.json()['model'] == 'cnn'
        assert response.json()['predictions'][0]['character'] == 'B'
        # Two strokes should come back in their left-to-right order.
        multiple = Image.new('L', (160, 70), 255)
        drawing = ImageDraw.Draw(multiple)
        drawing.line((20, 10, 30, 55), fill=0, width=5)
        drawing.line((105, 10, 115, 55), fill=0, width=5)
        image = io.BytesIO(); multiple.save(image, 'PNG')
        response = client.post('/api/recognise-text', files={'file': ('draw.png', image.getvalue(), 'image/png')})
        assert response.status_code == 200
        assert response.json()['text'] == 'B B'
        assert response.json()['predictions'][1]['before'] == ' '
        assert [item['position'] for item in response.json()['predictions']] == [0, 1]
        blank = io.BytesIO(); Image.new('L', (160, 70), 255).save(blank, 'PNG')
        assert client.post('/api/recognise-text', files={'file': ('blank.png', blank.getvalue(), 'image/png')}).status_code == 400
        assert client.post('/api/recognise-text', data={'model': 'mlp'}, files={'file': ('b.png', image.getvalue(), 'image/png')}).status_code == 503
        assert client.post('/api/recognise-text', data={'model': 'unknown'}, files={'file': ('b.png', image.getvalue(), 'image/png')}).status_code == 400
        assert client.post('/api/recognise-text', data={'mode': 'word'}, files={'file': ('b.png', image.getvalue(), 'image/png')}).status_code == 400


def test_character_api_preserves_letter_case(tmp_path, monkeypatch):
    class LowercaseModel:
        def predict(self, x, verbose=0):
            scores = np.zeros((len(x), 62), dtype=np.float32)
            scores[:, 38] = 0.9
            scores[:, 12] = 0.1
            return scores

    model_path = tmp_path/'lower.keras'
    mapping_path = tmp_path/'mapping.json'
    model_path.touch()
    mapping_path.write_text(json.dumps({'12': 'C', '38': 'c'}))
    monkeypatch.setattr(api, 'CHARACTER_PATHS', {'cnn': (model_path, mapping_path)})
    monkeypatch.setattr(api, '_extension_models', {'cnn': (LowercaseModel(), {'12': 'C', '38': 'c'})})
    drawing = Image.new('L', (70, 70), 255)
    ImageDraw.Draw(drawing).ellipse((15, 15, 50, 50), outline=0, width=5)
    buffer = io.BytesIO(); drawing.save(buffer, 'PNG')
    with TestClient(api.app) as client:
        response = client.post('/api/recognise-text', files={'file': ('c.png', buffer.getvalue(), 'image/png')})
    assert response.status_code == 200
    assert response.json()['text'] == 'c'


def test_character_segmenter_requires_explicit_opt_in(monkeypatch, tmp_path):
    from text_recognition.layout import Glyph

    methods = []
    crop = Image.new('L', (28, 28), 255)
    monkeypatch.setattr(api, 'segment_text', lambda image, method: (
        methods.append(method) or [Glyph(crop, (0, 0, 28, 28))]))
    monkeypatch.setattr(api, 'prepare_character', lambda image: np.zeros((28, 28, 1), np.float32))
    monkeypatch.setattr(api, 'get_extension_model', lambda key: (CharacterModel(), {'11': 'B'}))
    weights = tmp_path/'unet_resnet.keras'
    weights.touch()
    monkeypatch.setattr(api, 'UNET_MODEL', weights)
    data = io.BytesIO()
    crop.save(data, 'PNG')
    monkeypatch.delenv('HNRS_CHARACTER_SEGMENTER', raising=False)
    with TestClient(api.app) as client:
        response = client.post('/api/recognise-text', files={'file': ('c.png', data.getvalue(), 'image/png')})
        assert response.status_code == 200
        assert response.json()['segmenter'] == 'classical'
        monkeypatch.setenv('HNRS_CHARACTER_SEGMENTER', 'unet')
        response = client.post('/api/recognise-text', files={'file': ('c.png', data.getvalue(), 'image/png')})
        assert response.status_code == 200
        assert response.json()['segmenter'] == 'unet'
        response = client.post('/api/recognise-text', data={'segmenter': 'classical'},
                               files={'file': ('c.png', data.getvalue(), 'image/png')})
        assert response.status_code == 200
        assert response.json()['glyph_count'] == 1
    assert methods == ['classical', 'unet', 'classical']


def test_unet_selection_reports_missing_full_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(api, 'UNET_MODEL', tmp_path/'unet_resnet.keras')
    with TestClient(api.app) as client:
        assert client.get('/api/extension-models').json()['segmenters']['unet'] is False
        response = client.post('/api/recognise-text', data={'segmenter': 'unet'},
                               files={'file': ('c.png', b'ignored', 'image/png')})
    assert response.status_code == 503
    assert 'smoke model' in response.json()['detail']


def test_crop_inspection_returns_raw_and_classifier_input(monkeypatch):
    from text_recognition.layout import Glyph

    raw = Image.new('L', (40, 50), 255)
    ImageDraw.Draw(raw).line((9, 5, 24, 42), fill=0, width=4)
    monkeypatch.setattr(api, 'segment_text', lambda image, method: [Glyph(raw, (9, 5, 16, 38))])
    monkeypatch.setattr(api, 'get_extension_model', lambda key: (CharacterModel(), {'11': 'B'}))
    data = io.BytesIO(); raw.save(data, 'PNG')
    with TestClient(api.app) as client:
        response = client.post('/api/recognise-text', data={'show_crops': 'true'},
                               files={'file': ('draw.png', data.getvalue(), 'image/png')})
    assert response.status_code == 200
    prediction = response.json()['predictions'][0]
    assert prediction['bbox'] == [9, 5, 16, 38]
    for field, expected_size in [('crop_preview', (40, 50)), ('model_input_preview', (28, 28))]:
        prefix, payload = prediction[field].split(',', 1)
        assert prefix == 'data:image/png;base64'
        with Image.open(io.BytesIO(base64.b64decode(payload))) as preview:
            assert preview.size == expected_size
