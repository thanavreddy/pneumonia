from pathlib import Path

from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from transforms import get_eval_transforms, get_train_transforms


BATCH_SIZE = 16
NUM_WORKERS = 0


def create_dataloaders(
    data_root: Path,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""

    project_root = Path(__file__).resolve().parents[1]
    splits_dir = project_root / "data" / "splits"

    train_dataset = ChestXrayDataset(
        csv_path=splits_dir / "train.csv",
        transform=get_train_transforms(),
    )

    val_dataset = ChestXrayDataset(
        csv_path=splits_dir / "val.csv",
        transform=get_eval_transforms(),
    )

    test_dataset = ChestXrayDataset(
        data_dir=data_root / "test",
        transform=get_eval_transforms(),
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw" / "chest_xray"

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=BATCH_SIZE,
    )

    print("=" * 50)
    print("DataLoader Summary")
    print("=" * 50)

    print(f"Training images     : {len(train_loader.dataset)}")
    print(f"Validation images   : {len(val_loader.dataset)}")
    print(f"Test images         : {len(test_loader.dataset)}")
    print()

    print(f"Training batches    : {len(train_loader)}")
    print(f"Validation batches  : {len(val_loader)}")
    print(f"Test batches        : {len(test_loader)}")
    print()

    images, labels = next(iter(train_loader))

    print("=" * 50)
    print("First Training Batch")
    print("=" * 50)

    print(f"Images shape : {images.shape}")
    print(f"Labels shape : {labels.shape}")
    print(f"Labels       : {labels}")


if __name__ == "__main__":
    main()