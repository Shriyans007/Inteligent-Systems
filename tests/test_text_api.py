import io
import json
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
import api.main as api


class CharacterModel:
    def predict(self, x, verbose=0):
        assert x.shape == (1, 28, 28, 1)
        probabilities = np.zeros((1, 62), dtype=np.float32)
        probabilities[0, 11] = 1
        return probabilities


def test_extension_availability_and_character_route(tmp_path, monkeypatch):
    char, meta = tmp_path/'char.keras', tmp_path/'mapping.json'
    char.touch(); meta.write_text(json.dumps({'11': 'B'}))
    monkeypatch.setattr(api, 'EXTENSION_PATHS', {'character': (char, meta), 'word': (tmp_path/'missing.keras', tmp_path/'vocab.json')})
    monkeypatch.setattr(api, '_extension_models', {'character': (CharacterModel(), {'11': 'B'})})
    canvas = Image.new('L', (70, 70), 255)
    ImageDraw.Draw(canvas).line((20, 10, 30, 55), fill=0, width=5)
    image = io.BytesIO(); canvas.save(image, 'PNG')
    with TestClient(api.app) as client:
        assert [x['available'] for x in client.get('/api/extension-models').json()['models']] == [True, False]
        response = client.post('/api/recognise-text', data={'mode': 'character'}, files={'file': ('b.png', image.getvalue(), 'image/png')})
        assert response.status_code == 200 and response.json()['text'] == 'B'
        assert client.post('/api/recognise-text', data={'mode': 'word'}, files={'file': ('b.png', image.getvalue(), 'image/png')}).status_code == 503
        assert client.post('/api/recognise-text', data={'mode': 'numbers'}, files={'file': ('b.png', image.getvalue(), 'image/png')}).status_code == 400
