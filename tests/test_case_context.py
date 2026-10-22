"""The context model uses layout and CNN scores, not a word lookup."""

import csv

import numpy as np
import pytest
from PIL import Image

from text_recognition.case_context import alternatives, decode, features, fit, manifest, split, measure


def test_candidate_families_and_feature_shapes():
    assert alternatives('C') == ('C', 'c')
    assert alternatives('0') == ('0', 'O', 'o')
    assert alternatives('1') == ('1', 'I', 'l')
    mapping = {str(i): c for i, c in enumerate('0Oo')}
    scores = np.zeros((4, 62), dtype=np.float32)
    scores[:, 1] = 0.9
    glyphs = [{'bbox': (i*25, 0, 20, h), 'before': ''}
              for i, h in enumerate([62, 30, 31, 32])]
    chars, candidates = features(scores, glyphs, mapping)
    assert chars == ['O'] * 4
    assert len(candidates) == 12 and len(candidates[0][2]) == 16
    class ByHeight:
        def predict_proba(self, rows):
            # The candidate type and relative size affect every glyph.
            p = np.array([.9 if row[11] and row[3] < .6 else
                          .8 if row[10] and row[3] >= .6 else .1 for row in rows])
            return np.column_stack((1-p, p))
    assert decode(chars, candidates, ByHeight()) == ['O', 'o', 'o', 'o']


def test_writer_split_and_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match='Labelled case examples missing'):
        manifest(tmp_path/'labels.csv')
    with (tmp_path/'labels.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'text', 'writer'])
        for n in range(10):
            Image.new('L', (5, 5)).save(tmp_path/f'{n}.png')
            writer.writerow([f'{n}.png', 'Room', f'writer_{n}'])
    rows = manifest(tmp_path/'labels.csv')
    train, validation, test, method = split(rows)
    assert method == 'writer'
    assert {r['writer'] for r in train}.isdisjoint({r['writer'] for r in validation + test})
    assert len(train) + len(validation) + len(test) == 10


def test_three_writers_with_multiple_images_are_enough_to_split():
    rows = [{'image': f'{i}.png', 'text': 'Room', 'writer': f'writer_{i // 4}'}
            for i in range(12)]
    train, validation, test, method = split(rows)
    assert method == 'writer'
    assert sorted([len(train), len(validation), len(test)]) == [4, 4, 4]
    assert len({r['writer'] for r in train + validation + test}) == 3


def test_informative_small_split_errors():
    few = [{'image': f'{i}.png', 'text': 'Room', 'writer': ''} for i in range(3)]
    with pytest.raises(ValueError, match='Found 3 labelled images; need at least 7'):
        split(few)
    two_writers = [{**row, 'writer': str(i % 2)} for i, row in enumerate(few * 3)]
    with pytest.raises(ValueError, match='Found 2 writer IDs'):
        split(two_writers)


def test_evaluation_counts_case_errors_without_hiding_segmentation():
    mapping = {str(i): c for i, c in enumerate('ROoM')}
    scores = np.zeros((4, 62))
    scores[np.arange(4), [0, 1, 1, 3]] = 1
    row = {'image': 'example.png', 'text': 'Room', 'writer': 'one', 'status': 'ok',
           'scores': scores, 'glyphs': [{'bbox': (i*20, 0, 15, h), 'before': ''}
                                       for i, h in enumerate([62, 30, 30, 32])]}
    result, examples = measure([row], mapping)
    assert examples[0]['raw'] == 'ROOM'
    assert result['raw']['cer'] == .75
    assert result['status'] == {'ok': 1}


def test_small_synthetic_fit_constructs_separate_context_model():
    mapping = {str(i): c for i, c in enumerate('0Oo')}
    rows = []
    for n in range(30):
        scores = np.zeros((2, 62))
        scores[:, 1] = .8
        scores[:, 2] = .15
        scores[:, 0] = .05
        tall = (n % 2 == 0)
        rows.append({'status': 'ok', 'text': 'OO' if tall else 'oo',
                     'scores': scores,
                     'glyphs': [{'bbox': (i*25, 0, 20, 60 if tall else 28), 'before': ''}
                                for i in range(2)]})
    context, skipped = fit(rows, mapping)
    assert skipped == {}
    chars, candidates = features(rows[0]['scores'], rows[0]['glyphs'], mapping)
    assert len(decode(chars, candidates, context)) == 2
