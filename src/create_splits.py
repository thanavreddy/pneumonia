from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
VALIDATION_RATIO = 0.20

CLASS_TO_LABEL = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}


def collect_samples(train_dir: Path) -> pd.DataFrame:
    """Collect image paths and labels from the original training directory."""

    records: list[dict[str, str | int]] = []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = train_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        image_paths = sorted(class_dir.rglob("*.jpeg"))

        for image_path in image_paths:
            records.append(
                {
                    "image_path": str(image_path.resolve()),
                    "label": label,
                    "class_name": class_name,
                }
            )

    if not records:
        raise RuntimeError(
            f"No JPEG images found inside: {train_dir}"
        )

    return pd.DataFrame(records)


def print_split_summary(
    split_name: str,
    dataframe: pd.DataFrame,
) -> None:
    """Print class counts for one split."""

    print(split_name)
    print("-" * 40)

    counts = (
        dataframe["class_name"]
        .value_counts()
        .reindex(["NORMAL", "PNEUMONIA"])
    )

    print(f"NORMAL    : {counts['NORMAL']}")
    print(f"PNEUMONIA : {counts['PNEUMONIA']}")
    print(f"Total     : {len(dataframe)}")
    print()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    original_train_dir = (
        project_root
        / "data"
        / "raw"
        / "chest_xray"
        / "train"
    )

    splits_dir = project_root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = splits_dir / "train.csv"
    val_csv_path = splits_dir / "val.csv"

    dataframe = collect_samples(original_train_dir)

    train_dataframe, val_dataframe = train_test_split(
        dataframe,
        test_size=VALIDATION_RATIO,
        random_state=RANDOM_SEED,
        stratify=dataframe["label"],
    )

    # 重置行号并固定顺序，方便复现和检查。
    train_dataframe = (
        train_dataframe
        .sort_values("image_path")
        .reset_index(drop=True)
    )

    val_dataframe = (
        val_dataframe
        .sort_values("image_path")
        .reset_index(drop=True)
    )

    train_dataframe.to_csv(
        train_csv_path,
        index=False,
    )

    val_dataframe.to_csv(
        val_csv_path,
        index=False,
    )

    print("=" * 50)
    print("Stratified Dataset Split")
    print("=" * 50)
    print()

    print_split_summary(
        "New training split",
        train_dataframe,
    )

    print_split_summary(
        "New validation split",
        val_dataframe,
    )

    print(f"Train CSV saved to: {train_csv_path}")
    print(f"Val CSV saved to  : {val_csv_path}")


if __name__ == "__main__":
    main()