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
├── notebooks/
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

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Small configurable defaults are provided:

```bash
python src/train.py --epochs 5 --batch-size 16 --learning-rate 0.0001
```

Training saves:

- `models/resnet18.pth`
- `results/training/training_history.csv`
- `results/training/training_loss.png`
- `results/training/training_accuracy.png`

## Evaluate

```bash
python src/evaluate.py
```

Evaluation uses the independent test set and reports accuracy, precision, recall,
F1-score, and the confusion matrix. It saves:

- `results/evaluation/classification_report.txt`
- `results/evaluation/confusion_matrix.png`

## Predict

```bash
python src/predict.py path/to/xray.jpeg
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
!python src/train.py --epochs 5 --batch-size 16 --learning-rate 0.0001
!python src/evaluate.py
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
6. **Model selection:** The model state with the highest validation accuracy is saved.
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
