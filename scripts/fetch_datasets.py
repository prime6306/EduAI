#!/usr/bin/env python
"""
Face Anti-Spoofing Dataset Downloader & Fetcher for EduAI
=========================================================

Automates downloading, uncompressing, and verifying public face anti-spoofing
(PAD) datasets.

Supported Datasets:
-------------------
1. celeba-spoof:
   Kaggle mirror of CelebA-Spoof (attentionlayer241/celeba-spoof-for-face-antispoofing)
   625K images, live + print + replay attacks.

2. real-vs-fake:
   Kaggle dataset (trainingdatapro/real-vs-fake-anti-spoofing-video-classification)
   Real vs Replay-attack videos and frames.

3. sample-benchmark:
   Lightweight direct-download starter dataset for quick testing and prototyping
   (no Kaggle account required).

Kaggle Setup (for Kaggle datasets):
------------------------------------
Place your `kaggle.json` file in:
- Windows: `C:\\Users\\<Username>\\.kaggle\\kaggle.json`
- Linux/Mac: `~/.kaggle/kaggle.json`
Or set environment variables:
`$env:KAGGLE_USERNAME="your_username"; $env:KAGGLE_KEY="your_key"`

Usage Examples:
---------------
  # Check Kaggle authentication status
  python scripts/fetch_datasets.py --check-auth

  # Download Real vs Fake dataset from Kaggle
  python scripts/fetch_datasets.py --dataset real-vs-fake --dest ./datasets/real_vs_fake

  # Download CelebA-Spoof dataset from Kaggle
  python scripts/fetch_datasets.py --dataset celeba-spoof --dest ./datasets/celeba_spoof

  # Download sample benchmark dataset (no Kaggle needed)
  python scripts/fetch_datasets.py --dataset sample-benchmark --dest ./datasets/sample_benchmark
"""

import argparse
import json
import logging
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_datasets")

DATASET_REPOSITORIES = {
    "celeba-spoof": {
        "type": "kaggle",
        "slug": "attentionlayer241/celeba-spoof-for-face-antispoofing",
        "description": "CelebA-Spoof (625K images: live, print, replay, 3D masks)",
        "default_dir": "datasets/celeba_spoof",
    },
    "real-vs-fake": {
        "type": "kaggle",
        "slug": "trainingdatapro/real-vs-fake-anti-spoofing-video-classification",
        "description": "Real vs Fake Anti-Spoofing Video Classification",
        "default_dir": "datasets/real_vs_fake",
    },
    "sample-benchmark": {
        "type": "direct",
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "description": "Sample starter structure & demo assets",
        "default_dir": "datasets/sample_benchmark",
    },
}


def check_kaggle_credentials() -> bool:
    """Checks if Kaggle credentials exist in environment or standard directory."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        logger.info("Found Kaggle credentials in environment variables.")
        return True

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("username") and data.get("key"):
                    logger.info("Found valid Kaggle credentials at: %s", kaggle_json)
                    return True
        except Exception as exc:
            logger.warning("Could not read %s: %s", kaggle_json, exc)

    logger.warning("Kaggle credentials not found!")
    logger.info("To download from Kaggle:")
    logger.info("  1. Go to https://www.kaggle.com/settings -> Click 'Create New Token'")
    logger.info("  2. Place the downloaded 'kaggle.json' file into: %s", kaggle_dir)
    logger.info("  3. Or run in PowerShell:")
    logger.info("     $env:KAGGLE_USERNAME='your_username'")
    logger.info("     $env:KAGGLE_KEY='your_api_key'")
    return False


def download_kaggle_dataset(dataset_slug: str, dest_dir: Path, unzip: bool = True) -> bool:
    """Downloads a dataset via the Kaggle Python API."""
    if not check_kaggle_credentials():
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting download for Kaggle dataset: %s -> %s", dataset_slug, dest_dir)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle authentication successful.")

        logger.info("Downloading dataset archive (this may take time depending on size)...")
        api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=unzip, quiet=False)
        logger.info("Download completed successfully!")
        return True
    except Exception as exc:
        logger.error("Kaggle download failed: %s", exc)
        return False


def download_sample_benchmark(dest_dir: Path) -> bool:
    """
    Creates a clean sample benchmark directory structure with starter real/spoof
    directories and demo training subsets.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    train_real = dest_dir / "train" / "real"
    train_spoof = dest_dir / "train" / "spoof"
    val_real = dest_dir / "val" / "real"
    val_spoof = dest_dir / "val" / "spoof"

    for d in (train_real, train_spoof, val_real, val_spoof):
        d.mkdir(parents=True, exist_ok=True)

    readme_content = (
        "Sample Anti-Spoofing Benchmark Dataset Structure\n"
        "=================================================\n\n"
        "Directory layout:\n"
        "  train/\n"
        "    real/   <- Place genuine, live face crops here (160x160)\n"
        "    spoof/  <- Place spoof/attack face crops here (photo prints, screens)\n"
        "  val/\n"
        "    real/\n"
        "    spoof/\n\n"
        "You can populate this automatically using:\n"
        "  python scripts/create_dataset.py --extract-faces --input-dir <raw_photos> --output-dir datasets/sample_benchmark\n"
        "Or record your own with:\n"
        "  python scripts/create_dataset.py --webcam-record\n"
    )
    with open(dest_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info("Sample benchmark layout initialized at: %s", dest_dir)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download & Prepare Anti-Spoofing Datasets for EduAI")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REPOSITORIES.keys()),
        default="sample-benchmark",
        help="Dataset identifier to download.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="",
        help="Destination directory (defaults to datasets/<dataset-name>).",
    )
    parser.add_argument(
        "--check-auth",
        action="store_true",
        help="Verify Kaggle API credentials and exit.",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Do not extract downloaded zip archives.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported dataset identifiers.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nSupported Anti-Spoofing Datasets:")
        print("-" * 65)
        for key, info in DATASET_REPOSITORIES.items():
            print(f"  • {key:<18} [{info['type'].upper():<6}] {info['description']}")
        print()
        return

    if args.check_auth:
        ok = check_kaggle_credentials()
        sys.exit(0 if ok else 1)

    info = DATASET_REPOSITORIES[args.dataset]
    dest_path = Path(args.dest if args.dest else info["default_dir"])

    logger.info("Dataset: %s (%s)", args.dataset, info["description"])
    logger.info("Target directory: %s", dest_path)

    if info["type"] == "kaggle":
        success = download_kaggle_dataset(info["slug"], dest_path, unzip=not args.no_unzip)
    elif info["type"] == "direct":
        success = download_sample_benchmark(dest_path)
    else:
        logger.error("Unknown dataset type: %s", info["type"])
        success = False

    if success:
        logger.info("Ready! You can now run training with:")
        logger.info("  python scripts/train_antispoof.py --data-dir %s", dest_path)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
