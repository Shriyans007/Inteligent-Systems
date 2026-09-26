"""Small fixtures exercise parsing and image contracts without downloading datasets."""
import gzip
import struct

import numpy as np
import pytest
from PIL import Image, ImageDraw

from text_recognition.common import HEIGHT, WIDTH, prepare_word
from text_recognition.emnist import orient, read_idx, read_mapping
from text_recognition.iam import distance, read_samples, split_writers


def test_emnist_mapping_and_idx(tmp_path):
    mapping = tmp_path/'mapping.txt'
    mapping.write_text('\n'.join(f'{i} {ord("0")+i}' for i in range(62)))
    assert len(read_mapping(mapping)) == 62
    path = tmp_path/'images.gz'
    with gzip.open(path, 'wb') as file:
        file.write(b'\0\0\x08\x03' + struct.pack('>III', 1, 28, 28) + bytes(28*28))
    assert read_idx(path).shape == (1, 28, 28)
    img = np.zeros((1, 28, 28), dtype=np.uint8)
    img[0, 2, 5] = 255
    assert orient(img).shape == (1, 28, 28, 1)
    assert orient(img)[0, 5, 2, 0] == 255  # transpose, without mirroring


def test_iam_writer_split_and_word_aspect(tmp_path):
    root = tmp_path/'iam'
    (root/'ascii').mkdir(parents=True)
    (root/'ascii'/'forms.txt').write_text('\n'.join(f'a0{i}-000u w{i}' for i in range(5)))
    records = []
    for i in range(5):
        identifier = f'a0{i}-000u-00-00'
        folder = root/'words'/f'a0{i}'/f'a0{i}-000u'
        folder.mkdir(parents=True)
        sample = Image.new('L', (50, 20), 255)
        ImageDraw.Draw(sample).line((4, 4, 42, 15), fill=0, width=2)
        sample.save(folder/f'{identifier}.png')
        records.append(f'{identifier} ok 0 0 0 0 0 AT Test{i}')
    (root/'ascii'/'words.txt').write_text('\n'.join(records))
    blank_id = 'a00-000u-00-01'
    blank_path = root/'words'/'a00'/'a00-000u'/f'{blank_id}.png'
    Image.new('L', (50, 20), 255).save(blank_path)
    corrupt_id = 'a00-000u-00-02'
    (blank_path.parent/f'{corrupt_id}.png').write_bytes(b'not a PNG')
    with (root/'ascii'/'words.txt').open('a') as labels:
        labels.write(f'\n{blank_id} ok 0 0 0 0 0 AT Empty')
        labels.write(f'\n{corrupt_id} ok 0 0 0 0 0 AT Broken')
    assert len(read_samples(root)) == 7
    parts = split_writers(read_samples(root, validate_images=True))
    assert sum(map(len, parts)) == 5
    assert len(set(p[2] for group in parts for p in group)) == 5
    image = Image.new('L', (120, 40), 255)
    ImageDraw.Draw(image).text((4, 4), 'Test42', fill=0)
    ready = prepare_word(image)
    assert ready.shape == (HEIGHT, WIDTH, 1)
    assert ready.dtype == np.float32 and ready.max() > 0
    assert distance('Room42', 'Room43') == 1
    with pytest.raises(ValueError, match='visible writing'):
        prepare_word(Image.new('L', (120, 40), 255))
