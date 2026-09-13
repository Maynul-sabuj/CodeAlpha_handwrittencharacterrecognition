# CodeAlpha_HandwrittenCharacterRecognition

**Task 3 — Handwritten Character Recognition** (CodeAlpha Machine Learning Internship)

Identifies handwritten digits and/or letters using a Convolutional Neural
Network (CNN), trained on MNIST (digits) or EMNIST (letters/alphanumeric).
Includes a bonus script that extends single-character recognition to
segmenting and reading whole words.

## 🧠 Approach
1. **Data**: MNIST (10 digit classes) or EMNIST (26 letter classes, or the
   47-class "balanced" set with digits + letters).
2. **Model**: A CNN — two Conv2D blocks (32 then 64 filters) with batch
   normalization, max-pooling and dropout, followed by dense layers and a
   softmax output.
3. **Evaluation**: Accuracy, per-class precision/recall/F1, and a confusion
   matrix.
4. **Extension (bonus)**: `word_segmentation.py` uses OpenCV contour
   detection to split a word image into individual characters, then feeds
   each through the trained CNN and joins the predictions — a simple way
   to go from single-character to full-word recognition, as suggested in
   the task brief. A true sequence model for cursive/touching characters
   would use a CRNN (CNN + BiLSTM + CTC loss) instead — see "Next steps".

## 📁 Project Structure
```
CodeAlpha_HandwrittenCharacterRecognition/
├── data/                       # not needed — datasets auto-download
├── models/                     # trained model + plots saved here
├── src/
│   ├── data_loader.py          # loads MNIST or EMNIST, consistent format
│   ├── train.py                 # builds, trains, evaluates the CNN
│   ├── predict.py                # predicts a single character image
│   └── word_segmentation.py      # BONUS: segments + reads a word image
├── requirements.txt
└── README.md
```

## ⚙️ Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📦 Dataset
No manual download needed:
- `mnist` — downloads automatically via `tf.keras.datasets.mnist`.
- `emnist-letters` / `emnist-balanced` — downloads automatically via the
  `emnist` pip package (first run only, ~500MB cache).

## 🚀 Train
```bash
cd src
python train.py --dataset mnist --epochs 15
# or
python train.py --dataset emnist-letters --epochs 20
# or
python train.py --dataset emnist-balanced --epochs 25
```
Saves to `../models/`:
- `char_model.h5` — trained model
- `class_names.npy` — index → character mapping
- `training_curves.png`, `confusion_matrix.png`

## 🔮 Predict a single character
```bash
python predict.py --image ../data/sample_digit.png
```
Works with any image — it auto-converts to grayscale, resizes to 28×28,
and auto-inverts colors if needed to match the training data convention
(white stroke on black background).

## 🔤 Bonus: read a whole word
```bash
python word_segmentation.py --image ../data/sample_word.png
```
Segments the image into individual characters left-to-right and prints
the predicted string. Works best with clearly separated, non-cursive
characters (block letters or printed digits).

## 📊 Notes on improving accuracy
- Data augmentation (rotation, shear, elastic distortion) helps a lot on
  EMNIST, which is noisier than MNIST.
- EMNIST-balanced is genuinely hard for some letter pairs (e.g. 'O'/'0',
  'l'/'1') — check the confusion matrix to see which classes are confused.
- Try a deeper CNN (ResNet-style skip connections) for a few extra points
  of accuracy on EMNIST.

## 🔭 Next steps (full word/sentence recognition)
The segmentation approach in `word_segmentation.py` breaks down for
cursive handwriting or touching characters. The standard solution is a
**CRNN**: CNN layers extract visual features, a bidirectional LSTM reads
them left-to-right, and a **CTC (Connectionist Temporal Classification)**
loss lets the model learn alignment between image columns and output
characters without needing pre-segmented characters. Datasets like IAM
Handwriting or the EMNIST "byclass" split are good starting points for
that extension.

## ✅ CodeAlpha Submission Checklist
- [ ] Push this repo to GitHub as `CodeAlpha_HandwrittenCharacterRecognition`
- [ ] Record a short video walking through the code and results
- [ ] Post on LinkedIn tagging @CodeAlpha, with the GitHub link
- [ ] Submit via the CodeAlpha submission form
