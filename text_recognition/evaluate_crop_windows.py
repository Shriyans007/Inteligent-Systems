"""Compare crop strategies on generated lines from EMNIST, per character.

Run on training characters to select a strategy; reserve the EMNIST test
characters for a final held-out check. The same U-Net detections are reused
for every strategy. Synthetic-line accuracy is not real-drawing accuracy.

    python -m text_recognition.evaluate_crop_windows train --repeats 5
    python -m text_recognition.evaluate_crop_windows test --repeats 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation.synthetic_lines import EmnistSource, render_line
from segmentation.segmenter import _binary_for_detection
from segmentation.unet_segmenter import UNetSegmenter, remove_disconnected_neighbours, window_crop
from text_recognition.common import prepare_character
from text_recognition.emnist import FILES, load, read_mapping


def run(split, repeats=5, seed=42, data=Path('data/emnist/byclass'),
        cnn_path=Path('artifacts/text/character/emnist_cnn.keras'),
        unet_path=Path('artifacts/text/segmentation/unet_resnet.keras'), output=None):
    folder = data
    mapping = read_mapping(folder/FILES['mapping'])
    images, labels = load(folder, split)
    source = EmnistSource(images, labels, mapping)
    import keras
    cnn = keras.models.load_model(cnn_path)
    segmenter = UNetSegmenter(unet_path)
    char_to_id = {v:k for k,v in mapping.items()}
    rng = np.random.default_rng(seed)
    methods = ['baseline', 'narrow', 'component', 'narrow_component']
    batches = {name:[] for name in methods}
    truth = []
    segments = {'lines':0,'exact':0,'matched':0,'glyphs':0,'predicted':0}
    alphabet = sorted(source.available)
    for overlap in (False, True):
        for repeat in range(repeats):
            for i, target in enumerate(mapping.values()):
                contexts = [str(rng.choice(alphabet)) for _ in range(4)]
                text = ''.join(contexts[:2])+target+''.join(contexts[2:])
                canvas, glyphs = render_line(source,text,rng,width=320,overlap=overlap)
                if len(glyphs) != 5:
                    continue
                raw = (255 - canvas*255).astype(np.uint8)
                ink = _binary_for_detection(raw)
                rows, cols = np.where(ink > 0)
                x0,x1=max(0,cols.min()-4),min(raw.shape[1],cols.max()+5)
                y0,y1=max(0,rows.min()-4),min(raw.shape[0],rows.max()+5)
                line=raw[y0:y1,x0:x1]
                boxes=segmenter.find_characters(line)
                segments['lines']+=1
                segments['exact']+=int(len(boxes)==len(glyphs))
                segments['glyphs']+=len(glyphs)
                segments['predicted']+=len(boxes)
                used=set()
                for g in glyphs:
                    candidates=[(abs(b.cx+x0-g.cx),j) for j,b in enumerate(boxes) if j not in used]
                    if not candidates:continue
                    dist,j=min(candidates)
                    if dist>0.4*g.w:continue
                    b=boxes[j]
                    cleaned = remove_disconnected_neighbours(line,boxes,j)
                    crops={'baseline':window_crop(line,b.cx,b.half_width,margin=.18),
                           'narrow':window_crop(line,b.cx,b.half_width,margin=.08),
                           'component':window_crop(cleaned,b.cx,b.half_width,margin=.18),
                           'narrow_component':window_crop(cleaned,b.cx,b.half_width,margin=.08)}
                    try:
                        prepared={key:prepare_character(Image.fromarray(crop)) for key,crop in crops.items()}
                    except (ValueError,TypeError):continue
                    for key in methods:batches[key].append(prepared[key])
                    truth.append((char_to_id[g.char],overlap))
                    used.add(j)
                segments['matched']+=len(used)
                if segments['lines']%50==0:print('lines',segments['lines'],flush=True)
    outputs={'split':split,'repeats':repeats,'segmentation':segments,'methods':{}}
    y=np.array([t[0] for t in truth]); overlap=np.array([t[1] for t in truth])
    for key in methods:
        x=np.stack(batches[key]).astype(np.float32)
        predicted=cnn.predict(x,batch_size=256,verbose=0).argmax(axis=1)
        per={}
        for idx,ch in mapping.items():
            select=y==idx
            per[ch]={'n':int(select.sum()),'accuracy':round(float((predicted[select]==y[select]).mean()),4) if select.any() else None}
        outputs['methods'][key]={'overall':round(float((predicted==y).mean()),4),
                                  'separate':round(float((predicted[~overlap]==y[~overlap]).mean()),4),
                                  'overlap':round(float((predicted[overlap]==y[overlap]).mean()),4),
                                  'per_class':per}
    print(json.dumps({k:v for k,v in outputs.items() if k!='methods'}))
    print(json.dumps({k:{q:v[q] for q in ('overall','separate','overlap')} for k,v in outputs['methods'].items()}))
    if output is None:
        output = Path('artifacts/text/segmentation')/f'crop_windows_{split}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(outputs,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('train','test'))
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--data',type=Path,default=Path('data/emnist/byclass'))
    parser.add_argument('--cnn',type=Path,default=Path('artifacts/text/character/emnist_cnn.keras'))
    parser.add_argument('--unet',type=Path,default=Path('artifacts/text/segmentation/unet_resnet.keras'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.split,args.repeats,args.seed,args.data,args.cnn,args.unet,args.output)
