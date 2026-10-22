import io
import json
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
import api.main as api
from text_recognition.layout import Glyph


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


def test_character_api_keeps_raw_prediction_without_trained_context(tmp_path, monkeypatch):
    class FixedModel:
        def predict(self, batch, verbose=0):
            scores = np.zeros((len(batch), 62), dtype=np.float32)
            for i, label in enumerate([17, 40, 47, 47, 0]):
                scores[i, label] = 0.9
            return scores

    model_path, metadata = tmp_path/'char.keras', tmp_path/'mapping.json'
    model_path.touch(); metadata.write_text('{}')
    monkeypatch.setattr(api, 'CHARACTER_PATHS', {'cnn': (model_path, metadata)})
    monkeypatch.setattr(api, '_extension_models', {'cnn': (FixedModel(),
        {'17': 'H', '40': 'e', '47': 'l', '0': '0'})})
    glyphs = []
    for i, height in enumerate([62, 31, 62, 61, 32]):
        crop = Image.new('L', (20, height), 255)
        ImageDraw.Draw(crop).rectangle((5, 3, 14, height-4), fill=0)
        glyphs.append(Glyph(crop, (i*25, 0, 20, height)))
    monkeypatch.setattr(api, 'segment_text', lambda image: glyphs)
    buffer = io.BytesIO(); Image.new('L', (70, 70), 255).save(buffer, 'PNG')
    with TestClient(api.app) as client:
        result = client.post('/api/recognise-text', files={'file': ('draw.png', buffer.getvalue(), 'image/png')})
    assert result.status_code == 200
    assert result.json()['text'] == 'Hell0'
    assert result.json()['raw_text'] == 'Hell0'
    assert result.json()['case_context'] is False
    assert result.json()['predictions'][-1]['character'] == '0'


def test_local_case_sample_requires_explicit_label_and_stays_under_data(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'CASE_SAMPLE_ROOT', tmp_path/'data'/'case_benchmark')
    image = io.BytesIO()
    Image.new('L', (70, 70), 255).save(image, 'PNG')
    with TestClient(api.app) as client:
        bad = client.post('/api/case-samples', data={'text': ' '},
                          files={'file': ('draw.png', image.getvalue(), 'image/png')})
        assert bad.status_code == 400
        response = client.post('/api/case-samples', data={'text': 'Room', 'writer': 'A'},
                               files={'file': ('draw.png', image.getvalue(), 'image/png')})
        assert response.status_code == 200
        assert response.json()['saved'] is True
        assert (api.CASE_SAMPLE_ROOT / response.json()['image']).read_bytes() == image.getvalue()
        assert 'Room,A' in (api.CASE_SAMPLE_ROOT/'labels.csv').read_text()


def test_context_is_opt_in_and_model_specific(tmp_path, monkeypatch):
    folder = tmp_path/'cnn'
    folder.mkdir()
    (folder/'case_context.joblib').touch()
    (folder/'metadata.json').write_text(json.dumps({'model': 'cnn', 'enabled': True}))
    monkeypatch.setattr(api, 'CASE_CONTEXT_ROOT', tmp_path)
    assert api.context_available('cnn') and not api.context_available('mlp')
    (folder/'metadata.json').write_text(json.dumps({'model': 'cnn', 'enabled': False}))
    assert not api.context_available('cnn')


def test_character_route_applies_trained_context_only_when_requested(tmp_path, monkeypatch):
    class Scores:
        def predict(self, batch, verbose=0):
            result = np.zeros((len(batch), 62), dtype=np.float32)
            result[np.arange(4), [0, 1, 1, 3]] = .9
            return result
    class CaseDecision:
        def predict_proba(self, rows):
            p = np.array([.9 if row[11] and row[3] < .6 else
                          .8 if row[10] and row[3] >= .6 else .1 for row in rows])
            return np.column_stack((1-p, p))
    path, metadata = tmp_path/'char.keras', tmp_path/'mapping.json'
    path.touch(); metadata.write_text('{}')
    monkeypatch.setattr(api, 'CHARACTER_PATHS', {'cnn': (path, metadata)})
    monkeypatch.setattr(api, '_extension_models', {'cnn': (Scores(),
        {'0': 'R', '1': 'O', '2': 'o', '3': 'M', '4': 'm', '5': 'r', '6': '0'})})
    monkeypatch.setattr(api, 'get_case_context', lambda key: CaseDecision())
    glyphs = []
    for i, h in enumerate([62, 30, 31, 32]):
        crop = Image.new('L', (20, h), 255)
        ImageDraw.Draw(crop).rectangle((4, 3, 15, h-4), fill=0)
        glyphs.append(Glyph(crop, (i*25, 0, 20, h)))
    monkeypatch.setattr(api, 'segment_text', lambda image: glyphs)
    buffer = io.BytesIO(); Image.new('L', (70, 70), 255).save(buffer, 'PNG')
    upload = {'file': ('image.png', buffer.getvalue(), 'image/png')}
    with TestClient(api.app) as client:
        raw = client.post('/api/recognise-text', files=upload).json()
        corrected = client.post('/api/recognise-text', files=upload,
                                data={'model': 'cnn', 'context': 'true'}).json()
    assert raw['text'] == 'ROOM'
    assert corrected['text'] == 'Room' and corrected['raw_text'] == 'ROOM'
    assert corrected['predictions'][1]['character'] == 'O'
