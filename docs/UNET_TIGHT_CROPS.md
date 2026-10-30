# Experimental tight U-Net character crops

The existing U-Net predicts character centres and widths. Its default crop
windows can overlap, so a character classifier may see pieces of its neighbour.
`unet_tight` keeps the same trained U-Net detections, chooses one vertical cut
between adjacent predictions, limits each crop to its predicted character width,
then trims that region to its actual ink. Thus a narrow `l` gets a narrow crop,
while a wide `M` gets a wider one. Crops cannot borrow columns from a neighbour.
This is closer to the digit path's crop style. It can also cut a real character
stroke where two letters overlap; an improvement is not assumed.

The original U-Net weights, original EMNIST CNN and default GUI route are not
overwritten. A separate CNN is fine-tuned on synthetic EMNIST lines using the
new crop geometry, then saved as `emnist_cnn_unet_tight.keras`.

If you trained the tight-crop CNN before the adaptive-width update, retrain it:
that earlier model saw full centre-to-centre strips, which are different from
the crops now sent by the API. To preserve those older experimental weights,
copy the file before running the full command below.

From the repository root in the existing `(hnrs)` PowerShell environment:

```powershell
python -m pytest tests/test_unet_tight_crops.py
python -m text_recognition.finetune_tight_crops --data data/emnist/byclass
Get-Content artifacts/text/character/tight_crop_finetune_results.json
```

The command needs the original `artifacts/text/character/emnist_cnn.keras`
and the EMNIST ByClass files under `data/emnist/byclass`. It mixes clean
EMNIST characters with synthetic hard-cut crops, then scores held-out clean
characters and synthetic crops before and after fine-tuning. The synthetic
labels use the known character positions with simulated detection jitter;
they do not measure the error of the real U-Net or arbitrary handwriting.

For a small smoke run that saves **outside** the GUI path:

```powershell
python -m text_recognition.finetune_tight_crops --data data/emnist/byclass --lines 200 --clean 2000 --epochs 1 --output artifacts/text/character/tight_smoke.keras
```

Restart the API after full training and refresh the GUI. Choose **CNN for tight
U-Net crops** and **U-Net · tight character crops**. The normal EMNIST CNN and
normal U-Net segmentation remain available. Capture the input, intended text,
predicted text, character count and raw/28 × 28 crop previews for several
previously unseen words, including touching letters. Measure exact word and
character accuracy on the same labelled drawings before claiming a gain.

Do not copy the smoke model into the full model path or treat its result as a
held-out evaluation. The report should identify this as a separate experiment
only after a full run and labelled real-drawing checks.

## Paired check on your own drawings

Save the same untouched PNG/JPG files in a folder, for example
`data/character_examples/`, then create a UTF-8 CSV beside them:

```csv
image,label
word_01.png,Model
word_02.png,Work
```

Use the actual text in each image, including its case and spaces. Do not pick
only examples that one setup gets right. Aim for at least 20 varied drawings,
including connected and separated letters. Then run:

```powershell
python -m text_recognition.evaluate_tight_crops --manifest data/character_examples/manifest.csv
Get-Content artifacts/text/segmentation/tight_crop_paired.json
Import-Csv artifacts/text/segmentation/tight_crop_paired.csv | Format-Table -AutoSize
```

This uses the real FastAPI route in-process. It saves each prediction, exact
match, edit distance, crop count and API error. The JSON reports whole-word
accuracy, character error rate and wrong crop counts for both deployed setups.
An improvement here tests the **combined crop and separately trained CNN**;
the training JSON alone only measures held-out synthetic crops.
