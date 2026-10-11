import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]

# Our preprocessing choice; these values are not specified in the paper.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class AnimalDataset(Dataset):
    def __init__(self, split, augment_train=True):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        manifest = ROOT / "data" / "splits" / f"{split}.csv"

        if not manifest.is_file():
            raise FileNotFoundError(f"Missing split manifest: {manifest}")

        with manifest.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            required_columns = {"path", "label"}
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Manifest must contain path and label columns: {manifest}"
                )

            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")

        for row in self.rows:
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"Invalid label in manifest: {row}")

            image_path = ROOT / row["path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image: {image_path}")

        self.split = split
        self.augmentation_enabled = (
            split == "train" and augment_train
        )

        operations = [
            transforms.Resize((224, 224)),
        ]

        # The paper mentions augmentation but does not give its settings.
        # Horizontal flips and rotations are our implementation choices.
        if self.augmentation_enabled:
            operations.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])

        operations.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = ROOT / row["path"]

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)

        label = int(row["label"])
        return tensor, label


def seed_worker(worker_id):
    """Seed Python and NumPy inside each data-loading worker."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loaders(
    batch_size=32,
    num_workers=0,
    seed=42,
    augment_train=True,
):
    """
    Return training, validation, and test loaders.

    augment_train=False: initial baseline.
    augment_train=True: augmented training experiment.

    Validation and test never receive random augmentation.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    loaders = {}

    for index, split in enumerate(("train", "val", "test")):
        dataset = AnimalDataset(
            split=split,
            augment_train=augment_train,
        )

        generator = torch.Generator()
        generator.manual_seed(seed + index)

        loaders[split] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

    return loaders["train"], loaders["val"], loaders["test"]