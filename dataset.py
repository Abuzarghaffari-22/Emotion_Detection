from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
except ImportError as e:
    raise ImportError(
        "PyTorch / torchvision are required but not installed.\n"
        "Run:  pip install -r requirements.txt\n"
        f"Original error: {e}"
    )

import config

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm"}


def locate_dataset(candidates: Optional[List[Path]] = None) -> Path:
    candidates = candidates or config.DATASET_CANDIDATES
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists() and (candidate / "train").exists():
            return candidate

    searched = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate the FER2013 dataset. Looked in:\n"
        f"  - {searched}\n\n"
        "Fix this by either:\n"
        "  1) Downloading it from Kaggle "
        "(https://www.kaggle.com/datasets/pankaj4321/fer-2013-facial-expression-dataset) "
        "and extracting it to ./dataset, or\n"
        "  2) Placing it in Google Drive and mounting Drive (Colab), or\n"
        "  3) Passing an explicit --data-dir path to the script you're running."
    )


def _discover_classes(split_dir: Path) -> Dict[str, str]:
    return {p.name.lower(): p.name for p in split_dir.iterdir() if p.is_dir()}


def verify_dataset_structure(data_dir: Path) -> Dict[str, Dict[str, str]]:
    data_dir = Path(data_dir)
    splits = ["train", "val", "test"]
    resolved: Dict[str, Dict[str, str]] = {}

    missing_splits = [s for s in splits if not (data_dir / s).is_dir()]
    if missing_splits:
        raise FileNotFoundError(
            f"Dataset at '{data_dir}' is missing split folder(s): {missing_splits}. "
            f"Expected structure: {data_dir}/{{train,val,test}}/<class_name>/*.png"
        )

    for split in splits:
        split_dir = data_dir / split
        class_map = _discover_classes(split_dir)
        missing_classes = [c for c in config.CLASS_NAMES if c not in class_map]
        if missing_classes:
            raise FileNotFoundError(
                f"Dataset split '{split}' is missing class folder(s): {missing_classes}. "
                f"Found folders: {sorted(class_map.values())}"
            )
        resolved[split] = {c: class_map[c] for c in config.CLASS_NAMES}

    return resolved


def check_corrupted_images(data_dir: Path, class_map: Dict[str, Dict[str, str]]) -> List[str]:
    data_dir = Path(data_dir)
    corrupted: List[str] = []

    for split, classes in class_map.items():
        for canonical_class, folder_name in classes.items():
            class_dir = data_dir / split / folder_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except (UnidentifiedImageError, OSError, ValueError):
                    corrupted.append(str(img_path))

    return corrupted


def get_class_distribution(data_dir: Path, class_map: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    data_dir = Path(data_dir)
    dist: Dict[str, Dict[str, int]] = {}

    for split, classes in class_map.items():
        dist[split] = {}
        for canonical_class, folder_name in classes.items():
            class_dir = data_dir / split / folder_name
            count = sum(1 for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
            dist[split][canonical_class] = count

    return dist


def print_dataset_statistics(data_dir: Optional[Path] = None, run_corruption_check: bool = True) -> Dict:
    data_dir = Path(data_dir) if data_dir else locate_dataset()
    print(f"\n{'=' * 70}\nDATASET REPORT — {data_dir}\n{'=' * 70}")

    class_map = verify_dataset_structure(data_dir)
    print("[OK] Folder structure verified: train/ val/ test/, 7 classes each.\n")

    dist = get_class_distribution(data_dir, class_map)
    header = f"{'Class':<10}" + "".join(f"{s:>10}" for s in dist.keys()) + f"{'Total':>10}"
    print(header)
    print("-" * len(header))
    totals_per_split = {s: 0 for s in dist}
    for cls in config.CLASS_NAMES:
        row_total = 0
        row = f"{cls:<10}"
        for split in dist:
            n = dist[split].get(cls, 0)
            row += f"{n:>10}"
            row_total += n
            totals_per_split[split] += n
        row += f"{row_total:>10}"
        print(row)
    print("-" * len(header))
    footer = f"{'TOTAL':<10}" + "".join(f"{totals_per_split[s]:>10}" for s in dist)
    footer += f"{sum(totals_per_split.values()):>10}"
    print(footer)

    if run_corruption_check:
        print("\nScanning for corrupted images (this reads every file once)...")
        corrupted = check_corrupted_images(data_dir, class_map)
        if corrupted:
            print(f"[WARN] Found {len(corrupted)} corrupted/unreadable image(s):")
            for p in corrupted[:20]:
                print(f"   - {p}")
            if len(corrupted) > 20:
                print(f"   ... and {len(corrupted) - 20} more.")
        else:
            print("[OK] No corrupted images found.")
    else:
        corrupted = []

    print(f"{'=' * 70}\n")
    return {"class_map": class_map, "distribution": dist, "corrupted": corrupted}


def get_transforms(split: str, img_size: int = config.IMG_SIZE) -> "transforms.Compose":
    normalize = transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)

    if split == "train":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size + config.AUG_RANDOM_CROP_PADDING * 2,) * 2),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=config.AUG_HFLIP_PROB),
            transforms.RandomRotation(config.AUG_ROTATION_DEGREES),
            transforms.ColorJitter(brightness=config.AUG_BRIGHTNESS, contrast=config.AUG_CONTRAST),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=config.AUG_RANDOM_ERASING_PROB),
        ])

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = config.BATCH_SIZE,
    img_size: int = config.IMG_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    data_dir = Path(data_dir) if data_dir else locate_dataset()
    class_map = verify_dataset_structure(data_dir)

    train_ds = ImageFolder(data_dir / "train", transform=get_transforms("train", img_size))
    val_ds = ImageFolder(data_dir / "val", transform=get_transforms("val", img_size))
    test_ds = ImageFolder(data_dir / "test", transform=get_transforms("test", img_size))

    class_names = train_ds.classes

    sampler = None
    shuffle = True
    if config.USE_CLASS_WEIGHTS:
        targets = np.array(train_ds.targets)
        class_counts = np.bincount(targets, minlength=len(class_names))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS and num_workers > 0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=True, **common_kwargs
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader, class_names


def get_class_weights_tensor(data_dir: Optional[Path] = None) -> "torch.Tensor":
    data_dir = Path(data_dir) if data_dir else locate_dataset()
    class_map = verify_dataset_structure(data_dir)
    dist = get_class_distribution(data_dir, class_map)["train"]
    counts = np.array([dist[c] for c in config.CLASS_NAMES], dtype=np.float32)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    try:
        print_dataset_statistics()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
