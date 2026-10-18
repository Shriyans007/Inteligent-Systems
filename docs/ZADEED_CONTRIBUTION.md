Handwritten Number Recognition System (HNRP)

https://e6355ea0b4127434c5.gradio.live

COS30018 – Intelligent Systems | Swinburne University of Technology | Semester 2, 2026 Project Option B: Handwritten Number Recognition Problem

	
Team	Add team member names and student IDs
Tutor	Add tutor name
Notebook	notebooks/HandwrittenNumberRecog.ipynb
Best result	89.35% of multi-digit numbers recognised exactly (held-out test set)
1. Summary

This project builds a machine learning system that reads handwritten numbers (single digits and multi-digit numbers such as 16, 120, 281) from images.

The final system works in three stages:

Input image ──► 1. Segmentation ──► 2. Digit classification (CNN) ──► 3. Join digits ──► "120"
               (OpenCV)            (TensorFlow / Keras)

We started with a standard CNN trained on MNIST (98.88% on MNIST), found that it only reached 64.6% on our real handwritten-number dataset, and then improved the system step by step using error analysis, fine-tuning, segmentation tuning and a corrected evaluation metric, reaching 89.35%. We also built an interactive Gradio web app where users can draw or upload a number.

2. Tools and Environment
Tool	Purpose
Google Colab (T4 GPU)	Development and training
Python, NumPy, pandas	Data handling
TensorFlow / Keras	CNN models
scikit-learn	SVM baseline, metrics, train/test split
OpenCV	Thresholding, connected components, segmentation
Matplotlib	Plots, confusion matrix, error visualisation
Gradio	Interactive web interface
GitHub	Version control and progress tracking
3. Datasets
3.1 MNIST
60,000 training and 10,000 test images of single digits (28×28, white digit on black background).
Used to train the initial digit classifiers.
3.2 Handwritten Numbers dataset (handwritten_numbers_v1)
10,000 PNG images (0.png … 9999.png) of handwritten numbers, dark ink on a white background.
Variable image sizes (e.g. 75×26, 108×39, 108×40).
Contains both single digits and multi-digit numbers.
Labels are stored in labels.csv as one single row of 10,000 values; value i is the label for i.png.
Source: add dataset source here. (The dataset itself is not included in this repository.)

Data issues discovered:

pandas.read_csv treated the single row as column headers (duplicate values became 4.1, 4.2, …). Fixed with header=None, dtype=str.
Labels are stored as integers, so leading zeros were lost (an image showing 005 is labelled 5).
Some label noise exists (e.g. an image showing 167. is labelled 0).
3.3 Train / test split

The 10,000 images were split by image (not by digit) into 80% training / 20% test (random_state=42). This prevents data leakage: digits from the same image never appear in both training and testing. All dataset results below are measured on the same 2,000 held-out test images.

4. Methodology and Development Process
Step 1 – Baseline CNN on MNIST (CNN v1)

Architecture:

Conv2D(32, 3×3, ReLU) → MaxPool → Conv2D(64, 3×3, ReLU) → MaxPool
→ Flatten → Dense(128, ReLU) → Dropout(0.3) → Dense(10, Softmax)
Optimiser: Adam, loss: sparse categorical cross-entropy, 5 epochs, 10% validation split.
Test accuracy: 98.88% (loss 0.0360).
Observation: at epoch 5, training accuracy kept rising (98.98% → 99.11%) while validation accuracy dropped (99.03% → 98.77%), an early sign of overfitting.

Error analysis (confusion matrix):

Most common confusions	Count
5 predicted as 3	21
4 predicted as 9	16
4 predicted as 6	8
5 predicted as 6	7
7 predicted as 2	6

Digits 4 and 5 had the lowest recall (0.97). Many misclassified images were ambiguous even to a human.

Step 2 – Improved CNN (CNN v2)

Changes:

Data augmentation: random rotation (±8%), translation (10%) and zoom (10%).
Batch normalisation after each convolution layer.
Dropout increased to 0.4.
Early stopping (patience=3, restore best weights), up to 20 epochs.

Results:

Training stopped at epoch 14; best epoch was 11 (val loss 0.0400).
Test accuracy: 98.84% (loss 0.0389), about the same as v1.
Learning curves showed validation loss below training loss throughout. This is expected, because augmentation and dropout make training harder, and there was no sign of overfitting.
Conclusion: augmentation did not raise the MNIST score because MNIST test digits are clean and centred; its purpose is robustness on messier real-world input.
Step 3 – Classical ML baseline (SVM)
Support Vector Machine with RBF kernel, trained on 10,000 flattened MNIST images.
Test accuracy: 95.94%, training + prediction time 34.2 s.
Note: this comparison is not fully fair, because the SVM used 10k training images while the CNNs used 54k.
Step 4 – Testing on the real handwritten-numbers dataset

The dataset contains multi-digit numbers, so a segmentation pipeline was built:

Otsu thresholding + inversion → white digit on black (MNIST style).
Connected components to find separate ink blobs.
Remove tiny blobs (noise).
Merge blobs that overlap horizontally (e.g. a 4 written in two strokes).
Resize each digit into a 20×20 box, centred on a 28×28 canvas (same as MNIST).
Predict every digit with the CNN, then join the predictions left to right.

First results (all 10,000 images, string match):

Metric	Result
Digits found	19,476
Segmentation accuracy (correct digit count)	78.83%
CNN v1 exact match	64.63%
CNN v2 exact match	62.04%

Error analysis of the failures showed:

Domain shift: this writer draws 1 with a hook and some 7s crossed, so MNIST-trained models read 1 as 4 or 7 (e.g. 16 → 46, 114 → 774).
Segmentation errors: touching/overlapping digits were merged (e.g. 95 → 8, 120 → 42).
Label noise (e.g. image 000 labelled 0).
Data augmentation (CNN v2) did not help on this dataset; it scored lower than v1.
Step 5 – Fine-tuning on the target dataset

To adapt the CNN to this writer's style, a new digit-level training set was created from training images only: when segmentation found the correct number of digits, each piece was labelled with the matching character of the label (e.g. "16" → 1, 6). This produced 10,914 labelled digits.

CNN v2 was then fine-tuned with a small learning rate (Adam, 1e-4).

Model (2,000 test images)	Exact match
CNN v1	64.55%
CNN v2	61.35%
Fine-tuned (3 epochs)	69.15%
Fine-tuned (full training)	74.95%
Segmentation upper limit	78.15%

Bug found and fixed: the first fine-tuning run stopped after only 3 epochs because the early-stopping callback object was reused from CNN v2 and still remembered its best validation loss (0.040). A new callback was created and training continued for 20 epochs (digit validation accuracy about 97.3%).

Conclusion: when segmentation is correct, the classifier recognises about 96% of numbers (74.95 ÷ 78.15). Segmentation became the bottleneck.

Step 6 – Improving segmentation

Attempt A – splitting wide blobs. Blobs much wider than a typical digit were split at the columns with the least ink (vertical projection). Parameters were tuned on 2,000 training images:

merge_thr	ratio	Seg. accuracy
0.5	None (no split)	78.50%
0.5	1.3	78.80%
0.8	1.3	76.65%
Test exact match: 75.10% (only +0.15%).
New diagnosis on test images: too few pieces 1.20%, too many pieces 20.65%.
Key insight: the main problem was over-segmentation (too many pieces), not touching digits. The original assumption was wrong, and the data showed it.

Attempt B – noise filtering and more merging. 24 combinations were tested on 1,000 training images:

min_frac (ignore blobs smaller than a fraction of the largest): 0.05 / 0.15 / 0.3
merge_thr (negative = also merge small gaps): 0.5 / 0.2 / 0.0 / −0.2
dilate (thicken strokes before finding blobs): 0 / 1

Best: min_frac=0.3, merge_thr=0.2, dilate=0 → 83.9% on the tuning images.

Test metric	Result
Segmentation accuracy	84.00%
Too few pieces	2.30%
Too many pieces	13.70%
Exact match	79.75%

Dilation and negative merge thresholds made results worse. Note that the best min_frac was at the edge of the tested range, so larger values could be explored.

Step 7 – Fixing the evaluation metric

Looking at the remaining "too many pieces" images revealed that many were not errors: images showing 005, 00, 09 were labelled 5, 0, 9 because leading zeros were lost in the CSV. The system found the correct digits but was marked wrong by the string comparison.

The metric was changed to numeric comparison (int("005") == 5):

Version	Exact match
Segmentation v3 (string match)	79.75%
Segmentation v3 (numeric match)	89.25%

This shows how much the choice of evaluation metric affects reported performance.

Step 8 – Ablation: dot/underline filter (removed)

A filter was added to remove blobs shorter than 35% of the tallest blob (to ignore dots, dashes and underlines).

Version	Numeric match
Without filter	89.25%
With filter	88.65%

The filter hurt performance because it also removed real digit parts (e.g. the top stroke of a 5, or a small 2), so it was removed (h_frac=0.0).

Step 9 – Fine-tuning v2 (more data)

The fine-tuning set was rebuilt using the improved segmentation. Images with missing leading zeros were also included by padding the label with zeros (e.g. 5 → 005), but only when the model already predicted zeros for the extra pieces, to avoid adding noisy labels.

Fine-tuning digits: 13,870 (previously 10,914).
Learning rate 5e-5, early stopping at epoch 12.
Digit validation accuracy stayed flat at about 97.1–97.4%: the classifier has plateaued.
Model	Numeric match (no filter)
Fine-tuned v1	89.25%
Fine-tuned v2	89.35%

The +0.10% difference is only 2 of 2,000 images and is within random variation, so it should not be treated as a real improvement.

Step 10 – Interactive interface (Gradio)

A web app was built with Gradio inside Colab:

Draw tab: users write a number with the mouse on a sketchpad.
Upload tab: users upload an image of a number.
Output: the predicted number plus a gallery of the segmented digits with each digit's prediction and confidence, which makes the system's decisions easy to explain.
Uses the best model (fine-tuned v2) and the final segmentation settings.
5. Final Results
5.1 Single-digit classification (MNIST test set)
Model	Training data	Test accuracy
CNN v1	54k MNIST	98.88%
CNN v2 (augmentation + batch norm + early stopping)	54k MNIST	98.84%
SVM (RBF)	10k MNIST	95.94%
5.2 Multi-digit number recognition (2,000 held-out test images)
#	Version	Exact match
1	CNN v1 (MNIST only), string match	64.55%
2	CNN v2 (MNIST + augmentation), string match	61.35%
3	Fine-tuned CNN, string match	74.95%
4	+ wide-blob splitting	75.10%
5	+ noise filter & merge tuning (segmentation v3)	79.75%
6	Same, numeric match	89.25%
7	+ dot/underline filter	88.65% ✗ removed
8	Fine-tuned v2 (more data, restored zeros)	89.35% ✓ best
6. Discussion

What worked

Fine-tuning on the target dataset gave the largest model-side improvement (+10%), by fixing domain shift.
Error analysis at each stage pointed to the real problem (over-segmentation, not touching digits).
Tuning on training images only kept the test set unseen.
Choosing the right evaluation metric revealed the system's true performance.

What did not work

Data augmentation did not help on this dataset.
Dilation, negative merge thresholds and the height filter all reduced accuracy.
More fine-tuning data gave no meaningful gain once the classifier plateaued.

Remaining errors

Classifier confusions in this writer's style (e.g. 11 → 17, 7 → 4, 25 → 23, 47 → 42).
Digits written in two strokes being split (e.g. 34 → 361).
Label noise in the dataset.
7. Limitations
Some design decisions (removing the height filter, choosing the final model) were made by comparing test scores. Strictly, a separate validation set should be used for these decisions.
The dataset appears to come from a limited number of writers, so results may not generalise to other handwriting.
Label noise means the maximum achievable score is below 100%.
Segmentation is rule-based and struggles with touching or broken digits.
Mouse drawings in the app look different from pen handwriting (another domain shift).
8. Future Work
CRNN with CTC loss: recognise whole numbers without a separate segmentation step (uses RNN/LSTM concepts from Week 8).
A deeper CNN trained on MNIST and the target dataset together.
Extending to handwritten letters and maths symbols (e.g. EMNIST).
Cleaning label noise and using a proper train/validation/test split.
Deploying the app permanently (e.g. Hugging Face Spaces).
9. How to Run
Open notebooks/HandwrittenNumberRecog.ipynb in Google Colab.
Set Runtime → Change runtime type → GPU.
Upload handwritten_numbers_v1.rar to Google Drive (MyDrive/).
Run all cells in order. The notebook:
trains CNN v1, CNN v2 and the SVM on MNIST,
extracts and segments the dataset,
fine-tunes the CNN and evaluates on the test split,
launches the Gradio app (a public gradio.live link is printed).
Saved models
File	Description
digit_cnn.keras	CNN v1 (MNIST)
digit_cnn_v2.keras	CNN v2 (MNIST + augmentation)
digit_cnn_finetuned.keras	Fine-tuned v1
digit_cnn_finetuned_v2.keras	Fine-tuned v2 (best)
Final segmentation settings

merge_thr=0.2, ratio=1.3, min_frac=0.3, dilate=0, h_frac=0.0

10. Glossary
Term	Meaning
Epoch	One full pass through the training data
Overfitting	The model memorises training data and performs worse on new data
Data augmentation	Creating varied training images (rotate, shift, zoom)
Early stopping	Stop training when validation loss stops improving
Confusion matrix	Table showing which classes are confused with which
Precision / Recall	How often predictions are correct / how many real examples are found
Segmentation	Splitting an image into individual digits
Connected components	Groups of touching pixels, usually one digit each
Domain shift	Real data looks different from the training data
Fine-tuning	Continuing to train an existing model on new data with a small learning rate
Data leakage	Test data influencing training, which makes scores unrealistically high
Bottleneck	The part of a system that limits overall performance
Ablation study	Testing the effect of adding or removing one component at a time
Label noise	Incorrect labels in a dataset
Evaluation metric	The rule used to score a model
