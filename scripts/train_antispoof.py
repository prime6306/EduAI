#!/usr/bin/env python
"""
Face Anti-Spoofing Model Training Script for EduAI
==================================================

Trains a deep-learning anti-spoofing ensemble (MobileNetV2 + ResNet18) for
real-time webcam face liveness verification.

The resulting artifact is saved to `models/antispoof_fullmodels.pkl` and directly
consumed by `app/modules/attendance/antispoof.py`.

Dataset Format Support:
-----------------------
1. Standard Image Folder (`--dataset-type folder`):
   root/
     train/
       real/  (or live/)
       spoof/ (or fake/)
     val/
       real/
       spoof/
   (If only a single directory with real/ and spoof/ is provided, `--split`
    will automatically split into train/val).

2. CelebA-Spoof (`--dataset-type celeba`):
   CelebA-Spoof root containing `Data/` and `metas/intra_test/train_label.txt`
   (Labels: 0 = live -> real (1), 1..43 = spoof -> spoof (0)).

3. Synthetic Demo Mode (`--demo`):
   Generates a synthetic dataset on the fly to verify training, evaluation,
   and artifact generation end-to-end without needing external downloads.

Usage Examples:
---------------
  # 1. Quick test with synthetic demo data (verifies full pipeline immediately):
  python scripts/train_antispoof.py --demo --epochs 2

  # 2. Train on standard folder dataset:
  python scripts/train_antispoof.py --data-dir path/to/dataset --epochs 15 --batch-size 32

  # 3. Train on CelebA-Spoof:
  python scripts/train_antispoof.py --dataset-type celeba --data-dir path/to/CelebA_Spoof --epochs 10

  # 4. Train only MobileNetV2 for low-resource environments:
  python scripts/train_antispoof.py --model mobilenet --data-dir path/to/dataset
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    resnet18,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_antispoof")

# Class labels: 0 = Spoof, 1 = Real
CLASS_NAMES = ["spoof", "real"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Datasets & Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(img_size: int = 160):
    """
    Standard image transformations for training and inference.
    Matches the normalization used in app/modules/attendance/antispoof.py.
    """
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        norm,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])

    return train_transform, val_transform


class SimpleImageDataset(Dataset):
    """Dataset from list of (file_path, label) tuples."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to open %s (%s); generating blank fallback.", path, exc)
            img = Image.new("RGB", (160, 160), color=(128, 128, 128))

        if self.transform:
            img = self.transform(img)
        return img, label


def scan_folder_dataset(data_dir: str, split: float = 0.8) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Scans a folder for real/spoof subdirectories.
    Supports either pre-split (train/val) or a flat directory.
    """
    p = Path(data_dir)
    real_aliases = {"real", "live", "bonafide", "bona_fide", "1"}
    spoof_aliases = {"spoof", "fake", "attack", "imposter", "0"}

    # Check if train/val subdirectories exist
    if (p / "train").is_dir() and ((p / "val").is_dir() or (p / "test").is_dir()):
        val_dir = p / "val" if (p / "val").is_dir() else p / "test"
        train_samples = _scan_classes(p / "train", real_aliases, spoof_aliases)
        val_samples = _scan_classes(val_dir, real_aliases, spoof_aliases)
        return train_samples, val_samples

    # Single directory: scan all and split
    all_samples = _scan_classes(p, real_aliases, spoof_aliases)
    if not all_samples:
        raise ValueError(
            f"No image files found in '{data_dir}'. Expected subfolders like 'real'/'live' and 'spoof'/'fake'."
        )

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * split)
    return all_samples[:split_idx], all_samples[split_idx:]


def _scan_classes(dir_path: Path, real_aliases, spoof_aliases) -> List[Tuple[str, int]]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = []
    for entry in dir_path.iterdir():
        if not entry.is_dir():
            continue
        name_lower = entry.name.lower()
        if name_lower in real_aliases:
            label = 1  # Real
        elif name_lower in spoof_aliases:
            label = 0  # Spoof
        else:
            continue

        for root, _, files in os.walk(entry):
            for file in files:
                if Path(file).suffix.lower() in valid_exts:
                    samples.append((os.path.join(root, file), label))
    return samples


def load_celeba_spoof(data_dir: str, split: float = 0.8) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Loads samples from CelebA-Spoof dataset.
    Reads `metas/intra_test/train_label.txt` or scans `Data/`.
    In CelebA-Spoof: label 0 = live (1), label > 0 = spoof (0).
    """
    p = Path(data_dir)
    train_label_file = p / "metas" / "intra_test" / "train_label.txt"
    test_label_file = p / "metas" / "intra_test" / "test_label.txt"

    if train_label_file.exists():
        logger.info("Found CelebA-Spoof label file: %s", train_label_file)
        train_samples = _parse_celeba_label_file(p, train_label_file)
        val_samples = (
            _parse_celeba_label_file(p, test_label_file)
            if test_label_file.exists()
            else []
        )
        if not val_samples:
            random.shuffle(train_samples)
            s_idx = int(len(train_samples) * split)
            return train_samples[:s_idx], train_samples[s_idx:]
        return train_samples, val_samples

    # Fallback to general folder scan
    return scan_folder_dataset(data_dir, split)


def _parse_celeba_label_file(root_dir: Path, label_path: Path) -> List[Tuple[str, int]]:
    samples = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rel_img_path, raw_label = parts[0], int(parts[1])
                full_path = str(root_dir / rel_img_path)
                # CelebA: 0 is live, >0 is spoof
                label = 1 if raw_label == 0 else 0
                samples.append((full_path, label))
    return samples


def create_synthetic_demo_data(tmp_dir: str, count: int = 120) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Generates synthetic real and spoof face-like image samples for testing
    the training script and verifying model artifact generation.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    samples = []
    rng = np.random.default_rng(42)

    for i in range(count):
        is_real = (i % 2 == 0)
        label = 1 if is_real else 0
        arr = np.zeros((160, 160, 3), dtype=np.uint8)

        # Base skin tone
        skin_color = [rng.integers(180, 240), rng.integers(130, 180), rng.integers(100, 150)]
        arr[:, :] = skin_color

        # Simulate eyes / face features
        arr[50:70, 40:60] = [30, 30, 30]
        arr[50:70, 100:120] = [30, 30, 30]
        arr[105:120, 55:105] = [180, 50, 50]

        if not is_real:
            # Simulate spoof attack: moiré pattern / border / color degradation
            grid_y, grid_x = np.mgrid[0:160, 0:160]
            moire = (np.sin(grid_x / 3.0) * 30 + np.cos(grid_y / 3.0) * 30).astype(np.int16)
            arr = np.clip(arr.astype(np.int16) + moire[:, :, None], 0, 255).astype(np.uint8)
            # Add photo print border
            arr[:6, :] = [255, 255, 255]
            arr[-6:, :] = [255, 255, 255]
            arr[:, :6] = [255, 255, 255]
            arr[:, -6:] = [255, 255, 255]

        fname = f"{'real' if is_real else 'spoof'}_{i:04d}.jpg"
        fpath = os.path.join(tmp_dir, fname)
        Image.fromarray(arr).save(fpath, "JPEG")
        samples.append((fpath, label))

    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    return samples[:split_idx], samples[split_idx:]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_mobilenetv2(pretrained: bool = True) -> nn.Module:
    """Builds MobileNetV2 with a 2-class classifier (0=spoof, 1=real)."""
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2),
    )
    return model


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """Builds ResNet18 with a 2-class classifier (0=spoof, 1=real)."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training & Evaluation Engine
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), (correct / max(total, 1)) * 100.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """
    Evaluates standard classification accuracy plus PAD metrics:
    - APCER: Attack Presentation Classification Error Rate (False Acceptance of Spoof)
    - BPCER: Bona Fide Presentation Classification Error Rate (False Rejection of Real)
    - ACER: (APCER + BPCER) / 2
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total = len(all_labels)

    acc = (all_preds == all_labels).sum() / max(total, 1) * 100.0

    # PAD Metrics
    spoof_mask = (all_labels == 0)
    real_mask = (all_labels == 1)

    # APCER: Spoofs incorrectly classified as Real (predicted == 1)
    apcer = (all_preds[spoof_mask] == 1).mean() * 100.0 if np.any(spoof_mask) else 0.0
    # BPCER: Reals incorrectly classified as Spoof (predicted == 0)
    bpcer = (all_preds[real_mask] == 0).mean() * 100.0 if np.any(real_mask) else 0.0
    acer = (apcer + bpcer) / 2.0

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": acc,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
    }


def train_single_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> nn.Module:
    logger.info("Training %s on %s for %d epochs...", name, device, epochs)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acer = 999.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            "[%s] Epoch %02d/%02d (%.1fs) | Train Loss: %.4f, Acc: %.1f%% | "
            "Val Acc: %.1f%%, APCER: %.1f%%, BPCER: %.1f%%, ACER: %.2f%%",
            name, epoch, epochs, elapsed, train_loss, train_acc,
            val_metrics["accuracy"], val_metrics["apcer"], val_metrics["bpcer"], val_metrics["acer"],
        )

        if val_metrics["acer"] <= best_acer:
            best_acer = val_metrics["acer"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)
    logger.info("[%s] Completed. Best ACER: %.2f%%", name, best_acer)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Face Anti-Spoofing Deep Learning Model for EduAI")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Path to dataset root (containing real/spoof folders or CelebA-Spoof directory).",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["folder", "celeba", "synthetic"],
        default="folder",
        help="Dataset format: 'folder' (real/spoof subdirectories), 'celeba', or 'synthetic'.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demo training on synthetic data (verifies full pipeline immediately).",
    )
    parser.add_argument(
        "--model",
        choices=["ensemble", "mobilenet", "resnet"],
        default="ensemble",
        help="Model architecture: 'ensemble' (MobileNetV2 + ResNet18), 'mobilenet', or 'resnet'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 0.0003).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/antispoof_fullmodels.pkl",
        help="Destination path for trained model artifact (default: ./models/antispoof_fullmodels.pkl).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: 'auto', 'cuda', or 'cpu'.",
    )
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Prepare datasets
    train_transform, val_transform = get_transforms(img_size=160)

    if args.demo or args.dataset_type == "synthetic":
        logger.info("Creating synthetic demo dataset...")
        demo_dir = os.path.join(os.path.dirname(__file__), "..", "scratch", "demo_antispoof_data")
        train_samples, val_samples = create_synthetic_demo_data(demo_dir, count=160)
    elif args.dataset_type == "celeba":
        if not args.data_dir:
            logger.error("Must provide --data-dir when using CelebA-Spoof.")
            sys.exit(1)
        train_samples, val_samples = load_celeba_spoof(args.data_dir)
    else:
        if not args.data_dir:
            logger.error("Must provide --data-dir or use --demo.")
            sys.exit(1)
        train_samples, val_samples = scan_folder_dataset(args.data_dir)

    logger.info("Dataset loaded: %d train samples, %d val samples.", len(train_samples), len(val_samples))
    if not train_samples:
        logger.error("Train dataset is empty. Check your dataset path and directory structure.")
        sys.exit(1)

    train_ds = SimpleImageDataset(train_samples, transform=train_transform)
    val_ds = SimpleImageDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Initialize model(s)
    trained_artifacts = {}

    if args.model in ("ensemble", "mobilenet"):
        mb_model = build_mobilenetv2(pretrained=True).to(device)
        mb_model = train_single_model(
            "MobileNetV2", mb_model, train_loader, val_loader, args.epochs, args.lr, device
        )
        trained_artifacts["mobilenetv2"] = mb_model.cpu()

    if args.model in ("ensemble", "resnet"):
        rn_model = build_resnet18(pretrained=True).to(device)
        rn_model = train_single_model(
            "ResNet18", rn_model, train_loader, val_loader, args.epochs, args.lr, device
        )
        trained_artifacts["resnet18"] = rn_model.cpu()

    # Save artifact compatible with app/modules/attendance/antispoof.py
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving trained model artifact to: %s", out_path)
    torch.save(trained_artifacts, str(out_path))
    logger.info("Artifact saved successfully! Size: %.2f MB", out_path.stat().st_size / (1024 * 1024))

    # Verify compatibility with app
    logger.info("Verifying artifact compatibility with EduAI antispoof engine...")
    try:
        test_load = torch.load(str(out_path), map_location="cpu", weights_only=False)
    except TypeError:
        test_load = torch.load(str(out_path), map_location="cpu")
    assert isinstance(test_load, dict), "Saved artifact must be a dictionary."
    for name, m in test_load.items():
        m.eval()
        dummy = torch.randn(1, 3, 160, 160)
        out = m(dummy)
        assert out.shape == (1, 2), f"Sub-model {name} output shape should be (1, 2)."
        logger.info("Sub-model '%s' verified: output shape (1, 2), 2-class softmax output.", name)

    logger.info("Verification passed! The model is ready to use with EduAI.")


if __name__ == "__main__":
    main()
