from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


CLASS_TO_LABEL = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}


class ChestXrayDataset(Dataset):
    """Chest X-ray dataset loaded from a directory or a CSV file."""

    def __init__(
        self,
        data_dir: Path | None = None,
        csv_path: Path | None = None,
        transform=None,
    ) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if data_dir is None and csv_path is None:
            raise ValueError(
                "You must provide either data_dir or csv_path."
            )

        if data_dir is not None and csv_path is not None:
            raise ValueError(
                "Provide only one of data_dir or csv_path."
            )

        if csv_path is not None:
            self._load_from_csv(csv_path)

        if data_dir is not None:
            self._load_from_directory(data_dir)

        if not self.samples:
            raise RuntimeError("No images were found.")

    def _load_from_directory(
        self,
        data_dir: Path,
    ) -> None:
        """Load image paths from NORMAL and PNEUMONIA folders."""

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {data_dir}"
            )

        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = data_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class directory not found: {class_dir}"
                )

            image_paths = sorted(class_dir.rglob("*.jpeg"))

            for image_path in image_paths:
                self.samples.append(
                    (image_path, label)
                )

    def _load_from_csv(
        self,
        csv_path: Path,
    ) -> None:
        """Load image paths and labels from a CSV split file."""

        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}"
            )

        dataframe = pd.read_csv(csv_path)

        required_columns = {
            "image_path",
            "label",
        }

        missing_columns = (
            required_columns - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"CSV is missing columns: {missing_columns}"
            )

        for row in dataframe.itertuples(index=False):
            image_path = Path(row.image_path)
            label = int(row.label)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image listed in CSV was not found: "
                    f"{image_path}"
                )

            self.samples.append(
                (image_path, label)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        return image, label


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    train_csv_path = (
        project_root
        / "data"
        / "splits"
        / "train.csv"
    )

    from transforms import get_train_transforms

    dataset = ChestXrayDataset(
        csv_path=train_csv_path,
        transform=get_train_transforms(),
    )

    image, label = dataset[0]

    print("=" * 50)
    print("CSV Dataset Test")
    print("=" * 50)
    print(f"Number of images : {len(dataset)}")
    print(f"Image type       : {type(image)}")
    print(f"Image shape      : {image.shape}")
    print(f"Image dtype      : {image.dtype}")
    print(f"Label            : {label}")


if __name__ == "__main__":
    main()