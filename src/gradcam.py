from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from torch import nn
from torch.utils.data import DataLoader

from model import create_model
from prepare_data import create_dataloaders


BATCH_SIZE = 16
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


class GradCAM:
    """
    Generate Grad-CAM heatmaps for a convolutional neural network.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor,
    ) -> None:
        """
        Save feature maps produced by the target convolutional layer.
        """

        self.activations = output

        output.register_hook(
            self._save_gradients
        )

    def _save_gradients(
        self,
        gradients: torch.Tensor,
    ) -> None:
        """
        Save gradients flowing through the target layer.
        """

        self.gradients = gradients

    def generate(
        self,
        image: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for one image.

        Parameters
        ----------
        image:
            Tensor with shape [1, 3, 224, 224].

        target_class:
            Class index whose Grad-CAM should be generated.
        """

        self.model.zero_grad()

        outputs = self.model(image)

        target_score = outputs[
            0,
            target_class,
        ]

        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Gradients were not captured."
            )

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        # Average each feature map's gradients across height and width.
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted combination of feature maps.
        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # Keep only positive contributions.
        cam = F.relu(cam)

        # Resize heatmap to the original model input size.
        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()

        # Normalize heatmap to the range 0–1.
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self) -> None:
        """
        Remove the registered forward hook.
        """

        self.forward_hook.remove()


def denormalize_image(
    image_tensor: torch.Tensor,
) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.
    """

    image = (
        image_tensor
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    image = (
        image * IMAGENET_STD
        + IMAGENET_MEAN
    )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return image


def find_correct_pneumonia_sample(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    int,
    int,
    float,
]:
    """
    Find a test image that is truly pneumonia and is correctly predicted.
    """

    model.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = outputs.argmax(
                dim=1
            )

            for index in range(
                images.size(0)
            ):
                true_label = labels[index].item()
                predicted_label = predictions[index].item()

                if (
                    true_label == 1
                    and predicted_label == 1
                ):
                    pneumonia_probability = (
                        probabilities[
                            index,
                            1,
                        ].item()
                    )

                    selected_image = (
                        images[index]
                        .unsqueeze(0)
                    )

                    return (
                        selected_image,
                        true_label,
                        predicted_label,
                        pneumonia_probability,
                    )

    raise RuntimeError(
        "No correctly predicted pneumonia image was found."
    )


def create_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.
    """

    heatmap_colored = colormaps["jet"](
        heatmap
    )[:, :, :3]

    overlay = (
        (1.0 - alpha) * original_image
        + alpha * heatmap_colored
    )

    overlay = np.clip(
        overlay,
        0.0,
        1.0,
    )

    return overlay


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    data_root = (
        project_root
        / "data"
        / "raw"
        / "chest_xray"
    )

    model_path = (
        project_root
        / "models"
        / "resnet18.pth"
    )

    figures_dir = (
        project_root
        / "results"
        / "explainability"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Chest X-ray Grad-CAM Analysis")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Model path : {model_path}")
    print()

    _, _, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=BATCH_SIZE,
    )

    model = create_model(
        num_classes=2,
        use_pretrained=False,
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    )

    model = model.to(device)
    model.eval()

    (
        selected_image,
        true_label,
        predicted_label,
        pneumonia_probability,
    ) = find_correct_pneumonia_sample(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    print(
        "True label       : "
        f"{CLASS_NAMES[true_label]}"
    )

    print(
        "Predicted label  : "
        f"{CLASS_NAMES[predicted_label]}"
    )

    print(
        "Pneumonia probability: "
        f"{pneumonia_probability:.4%}"
    )

    # ResNet18's final convolutional block.
    target_layer = model.layer4[-1]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    heatmap = gradcam.generate(
        image=selected_image,
        target_class=predicted_label,
    )

    gradcam.remove_hooks()

    original_image = denormalize_image(
        selected_image.squeeze(0)
    )

    overlay = create_overlay(
        original_image=original_image,
        heatmap=heatmap,
        alpha=0.40,
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(
        original_image,
        cmap="gray",
    )

    plt.axis("off")
    plt.title(
        "Original Chest X-ray\n"
        f"True: {CLASS_NAMES[true_label]} | "
        f"Predicted: {CLASS_NAMES[predicted_label]}"
    )

    plt.tight_layout()

    original_path = (
        figures_dir
        / "original_pneumonia.png"
    )

    plt.savefig(
        original_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(6, 6))

    plt.imshow(
        heatmap,
        cmap="jet",
    )

    plt.colorbar(
        label="Model attention"
    )

    plt.axis("off")
    plt.title(
        "Grad-CAM Heatmap"
    )

    plt.tight_layout()

    heatmap_path = (
        figures_dir
        / "gradcam_heatmap.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(6, 6))

    plt.imshow(
        overlay
    )

    plt.axis("off")
    plt.title(
        "Grad-CAM Overlay\n"
        f"Pneumonia probability: "
        f"{pneumonia_probability:.2%}"
    )

    plt.tight_layout()

    overlay_path = (
        figures_dir
        / "gradcam_overlay.png"
    )

    plt.savefig(
        overlay_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(
        f"Original image saved to: "
        f"{original_path}"
    )

    print(
        f"Heatmap saved to       : "
        f"{heatmap_path}"
    )

    print(
        f"Overlay saved to       : "
        f"{overlay_path}"
    )


if __name__ == "__main__":
    main()