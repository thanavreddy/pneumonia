import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


NUM_CLASSES = 2


def create_model(
    num_classes: int = NUM_CLASSES,
    use_pretrained: bool = True,
) -> nn.Module:
    """Create a ResNet18 model for chest X-ray classification."""

    if use_pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)

    # ResNet18 原本最后输出 1000 个 ImageNet 类别。
    # 现在修改为 NORMAL 和 PNEUMONIA 两个类别。
    input_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features=input_features,
        out_features=num_classes,
    )

    return model


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model(
        num_classes=NUM_CLASSES,
        use_pretrained=True,
    )

    model = model.to(device)

    print("=" * 50)
    print("Chest X-ray Classification Model")
    print("=" * 50)

    print(f"Device: {device}")
    print(f"Final layer: {model.fc}")
    print()

    # 创建一个模拟批次：
    # 4 张图片，每张图片 shape 是 [3, 224, 224]
    dummy_images = torch.randn(
        4,
        3,
        224,
        224,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        outputs = model(dummy_images)

    print("=" * 50)
    print("Forward Pass Test")
    print("=" * 50)

    print(f"Input shape  : {dummy_images.shape}")
    print(f"Output shape : {outputs.shape}")
    print()
    print("Raw outputs:")
    print(outputs)

    predicted_labels = outputs.argmax(dim=1)

    print()
    print(f"Predicted labels: {predicted_labels}")


if __name__ == "__main__":
    main()