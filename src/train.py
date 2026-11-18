import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import precision_recall_fscore_support

from model import create_model
from prepare_data import create_dataloaders


BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4


def train_one_epoch(model, loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate_one_epoch(model, loader, loss_function, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    return (
        running_loss / total,
        correct / total,
        precision,
        recall,
        f1,
    )


def plot_history(history, results_dir, mode):
    epochs = [row["epoch"] for row in history]
    suffix = "feature_extractor" if mode == "freeze" else "fine_tuned"

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Training")
    plt.plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"training_loss_{suffix}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="Training")
    plt.plot(epochs, [row["validation_accuracy"] for row in history], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"training_accuracy_{suffix}.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--mode",
        choices=["fine-tune", "freeze"],
        default="fine-tune",
        help="fine-tune all weights or freeze the pretrained feature extractor",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw" / "chest_xray"
    models_dir = project_root / "models"
    results_dir = project_root / "results" / "training"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = create_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
    )
    model = create_model(
        num_classes=2,
        use_pretrained=True,
        freeze_features=args.mode == "freeze",
    ).to(device)
    loss_function = nn.CrossEntropyLoss()
    trainable_parameters = (
        model.parameters()
        if args.mode == "fine-tune"
        else model.fc.parameters()
    )
    optimizer = Adam(trainable_parameters, lr=args.learning_rate)

    history = []
    best_validation_f1 = -1.0
    best_validation_metrics = None
    model_name = (
        "resnet18_feature_extractor.pth"
        if args.mode == "freeze"
        else "resnet18.pth"
    )
    model_path = models_dir / model_name

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        validation_loss, validation_accuracy, precision, recall, f1 = validate_one_epoch(
            model, val_loader, loss_function, device
        )

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, "
            f"validation_loss: {validation_loss:.4f}, "
            f"validation_accuracy: {validation_accuracy:.4f}, "
            f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_f1": f1,
        })

        if f1 > best_validation_f1:
            best_validation_f1 = f1
            best_validation_metrics = {
                "accuracy": validation_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            torch.save(model.state_dict(), model_path)

    history_name = (
        "training_history_feature_extractor.csv"
        if args.mode == "freeze"
        else "training_history_fine_tuned.csv"
    )
    pd.DataFrame(history).to_csv(
        results_dir / history_name,
        index=False,
    )
    plot_history(history, results_dir, args.mode)

    print(
        "Best validation metrics: "
        f"accuracy={best_validation_metrics['accuracy']:.4%}, "
        f"precision={best_validation_metrics['precision']:.4f}, "
        f"recall={best_validation_metrics['recall']:.4f}, "
        f"f1={best_validation_metrics['f1']:.4f}"
    )
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
