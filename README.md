# Chest X-ray Pneumonia Detection

A beginner-friendly binary image classification project using a pretrained ResNet18 model.
The model predicts whether a chest X-ray shows **NORMAL** or **PNEUMONIA**.

No model comparisons, ensemble, or hyperparameter optimization are used. The goal is
to provide a small, understandable PyTorch implementation that can be explained in an
AI/ML interview.

## Project structure

```text
pneunomia/
├── data/
│   ├── raw/chest_xray/
│   └── splits/
├── models/
├── results/
│   ├── training/
│   └── evaluation/
├── src/
│   ├── create_splits.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── prepare_data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── gradcam.py
├── requirements.txt
└── README.md
```

## Dataset

Use the **Chest X-ray Images (Pneumonia)** dataset from Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Place it in this directory:

```text
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The original validation folder is too small for reliable model validation. The script
creates a new stratified validation set from the original training images:

```bash
python src/create_splits.py
```

This split uses random seed 42 and keeps 20% of the original training data for
validation. The test set remains completely separate and is not used during training.

### Data-split limitation

The split is performed at the **image level**, not the patient level. The commonly
distributed dataset does not provide a reliable patient identifier for every image.
Different views or repeated X-rays from the same patient could therefore appear in
both training and validation data. If patient identifiers become available, perform
the split using patient IDs instead of individual filenames to prevent patient-level
data leakage. The independent test set should remain untouched during both modes of
training and model selection.

## Setup

```bash
pip install -r requirements.txt
```

## Train

Full fine-tuning (the default):

```bash
python src/train.py --mode fine-tune --epochs 5 --batch-size 16 --learning-rate 0.0001
```

Frozen feature extractor:

```bash
python src/train.py --mode freeze --epochs 5 --batch-size 16 --learning-rate 0.0001
```

Full fine-tuning updates every ResNet18 weight. Frozen mode loads pretrained ImageNet
weights, freezes the convolutional feature extractor, replaces the 1000-class
fully connected layer with a 2-class layer, and trains only that new layer.

Training saves the best checkpoint for the selected mode:

- Full fine-tuning: `models/resnet18.pth`
- Frozen features: `models/resnet18_feature_extractor.pth`

It also saves mode-specific history and plot files under `results/training/`:

- Fine-tuning: `training_history_fine_tuned.csv`,
  `training_loss_fine_tuned.png`, `training_accuracy_fine_tuned.png`
- Frozen features: `training_history_feature_extractor.csv`,
  `training_loss_feature_extractor.png`, `training_accuracy_feature_extractor.png`

The checkpoint is saved when macro F1 on the validation set improves. Training history
also reports validation accuracy, precision, recall, and F1.

## Evaluate

```bash
python src/evaluate.py
python src/evaluate.py --model models/resnet18_feature_extractor.pth
```

Evaluation uses the independent test set and reports accuracy, precision, recall,
F1-score, and the confusion matrix. It saves:

- `results/evaluation/classification_report.txt`
- `results/evaluation/confusion_matrix.png`

## Predict

```bash
python src/predict.py path/to/xray.jpeg
python src/predict.py path/to/xray.jpeg --model models/resnet18_feature_extractor.pth
```

Example output:

```text
Prediction: PNEUMONIA
Probability: 98.42%
```

## Optional Grad-CAM

Grad-CAM is kept as an optional, separate explainability feature:

```bash
python src/gradcam.py
```

It creates visualizations under `results/explainability/`.

## Google Colab

1. Upload this repository or clone it.
2. Enable GPU under **Runtime > Change runtime type > T4 GPU**.
3. Download and place the Kaggle dataset under `data/raw/chest_xray/`.
4. Install dependencies:

```python
%pip install -r requirements.txt
```

5. Change to the project directory:

```python
%cd /content/pneunomia
```

6. Create the split and train:

```bash
!python src/create_splits.py
!python src/train.py --mode fine-tune --epochs 5 --batch-size 16 --learning-rate 0.0001
!python src/evaluate.py --model models/resnet18.pth
```

When running in Colab, the split CSV files and trained model can be saved in Google
Drive if the session is restarted. The Kaggle dataset itself must be downloaded again
unless it is stored persistently.

## Complete ML pipeline

1. **Dataset split:** Normal and pneumonia images are separated and split reproducibly.
   The test set is kept independent.
2. **Preprocessing:** Every image is converted to RGB, resized to 224 × 224, converted
   to a tensor, and normalized with ImageNet mean and standard deviation.
3. **Transfer learning:** ResNet18 is loaded with pretrained ImageNet weights.
4. **Final layer:** The original 1000-class fully connected layer is replaced by a
   2-class linear layer.
5. **Training:** Adam minimizes CrossEntropyLoss. Training and validation loss and
   accuracy are tracked each epoch.
6. **Model selection:** The model state with the highest macro-validation-F1 score is
   saved. This is preferable to checkpoint selection by loss or a single metric.
7. **Evaluation:** The saved model is tested on images it never saw during training
   or validation. Accuracy, precision, recall, F1-score, and a confusion matrix describe
   performance.
8. **Prediction:** A new X-ray is transformed in the same way as validation images,
   passed through ResNet18, and classified using softmax probabilities.

## Interview explanation

Chest X-ray images are loaded from `NORMAL` and `PNEUMONIA` folders and converted into
PyTorch tensors. The pretrained ResNet18 provides learned image features from ImageNet,
while the final fully connected layer is trained to produce exactly two outputs.
CrossEntropyLoss measures the difference between predicted class scores and true labels,
and Adam updates the network weights. The model is evaluated on the independent test
set using multiple metrics, not only accuracy, because class imbalance and false negatives
can make a simple accuracy score misleading.
