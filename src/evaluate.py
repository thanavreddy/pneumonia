from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from model import create_model
from prepare_data import create_dataloaders


BATCH_SIZE = 16
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw" / "chest_xray"
    model_path = project_root / "models" / "resnet18.pth"
    results_dir = project_root / "results" / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=BATCH_SIZE,
    )

    model = create_model(num_classes=2, use_pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        digits=4,
    )
    matrix = confusion_matrix(true_labels, predicted_labels)

    print(f"Test images: {len(true_labels)}")
    print(f"Test accuracy: {accuracy:.4%}")
    print(report)
    print("Confusion matrix:")
    print(matrix)

    (results_dir / "classification_report.txt").write_text(
        f"Test accuracy: {accuracy:.4%}\n\n"
        f"{report}\n\nConfusion matrix:\n{matrix}\n",
        encoding="utf-8",
    )

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    ).plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
