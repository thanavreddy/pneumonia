import argparse
from pathlib import Path

import torch
from PIL import Image

from model import create_model
from transforms import get_eval_transforms


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def predict_image(image_path, model_path=None):
    project_root = Path(__file__).resolve().parents[1]
    if model_path is None:
        model_path = project_root / "models" / "resnet18.pth"
    else:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=2, use_pretrained=False).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = get_eval_transforms()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)
        predicted_index = probabilities.argmax(dim=1).item()

    predicted_class = CLASS_NAMES[predicted_index]
    predicted_probability = probabilities[0, predicted_index].item()
    return predicted_class, predicted_probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a chest X-ray image")
    parser.add_argument("--model", help="Optional path to the trained model")
    args = parser.parse_args()

    label, probability = predict_image(args.image, args.model)
    print(f"Prediction: {label}")
    print(f"Probability: {probability:.4%}")


if __name__ == "__main__":
    main()
