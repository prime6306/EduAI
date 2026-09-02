#!/usr/bin/env python
"""
Face Anti-Spoofing Dataset Creation & Preprocessing Tool for EduAI
==================================================================

Prepares, formats, and standardizes face anti-spoofing training data into the
160x160 RGB face crop format expected by MobileNetV2 + ResNet18.

Supported Modes:
----------------
1. Extract Frames & Crop from Videos (`--mode videos`):
   Smart recursive processor for video datasets (Real vs Fake, Replay-Attack,
   OULU-NPU, etc.). Supports:
   - CSV metadata annotations (e.g. `real_and_fake.csv`, `metadata.csv`)
   - Auto-unzipping if archives are present
   - Nested directory structures (e.g. `videos/real`, `data/attack`, etc.)
   - Filename-based classification (e.g. `real_01.mp4`, `attack_02.mp4`)

2. Extract & Crop Faces from Images (`--mode images`):
   Detects faces in raw uncropped photos, extracts 160x160 crops with
   margin padding, and organizes into real/spoof classes.

3. CelebA-Spoof Subset Extractor (`--mode celeba-subset`):
   Creates a balanced, lightweight subset (e.g. 2,500 real + 2,500 spoof)
   from full CelebA-Spoof to train quickly without needing 77GB in RAM.

4. Interactive Webcam Collector (`--mode webcam`):
   Interactively captures live faces and simulated spoof attacks (screen/print)
   from your local webcam to train on your real hardware.

Usage Examples:
---------------
  # Process Real vs Fake videos (auto-detects real_and_fake.csv and nested folders):
  python scripts/create_dataset.py --mode videos --input-dir datasets/real_vs_fake --output-dir datasets/processed_pad --frame-interval 8

  # Crop faces from raw image folders into standardized dataset:
  python scripts/create_dataset.py --mode images --input-dir raw_photos --output-dir datasets/custom_pad

  # Create a balanced 5,000 image subset from CelebA-Spoof:
  python scripts/create_dataset.py --mode celeba-subset --input-dir datasets/celeba_spoof --output-dir datasets/celeba_mini --max-per-class 2500
"""

import argparse
import csv
import logging
import os
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("create_dataset")

# Built-in OpenCV Haar Cascade for face detection
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_DETECTOR = None

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REAL_ALIASES = {"real", "live", "bonafide", "bona_fide", "genuine", "1"}
SPOOF_ALIASES = {"spoof", "fake", "attack", "imposter", "replay", "print", "0"}


def get_face_detector():
    """Initializes OpenCV's default Haar Cascade frontal face detector if available."""
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        try:
            cascade_cls = getattr(cv2, "CascadeClassifier", None)
            cv2_data = getattr(cv2, "data", None)
            haarcascades_dir = getattr(cv2_data, "haarcascades", "") if cv2_data else ""
            cascade_path = os.path.join(haarcascades_dir, "haarcascade_frontalface_default.xml") if haarcascades_dir else ""

            if cascade_cls is not None and os.path.exists(cascade_path):
                detector = cascade_cls(cascade_path)
                if not detector.empty():
                    _FACE_DETECTOR = detector
                    logger.info("Initialized OpenCV Haar Cascade face detector.")
                    return _FACE_DETECTOR
        except Exception as exc:
            logger.debug("Haar cascade face detector unavailable: %s", exc)

        logger.info("OpenCV CascadeClassifier unavailable in current environment. Using smart portrait face cropping.")
        _FACE_DETECTOR = False  # Mark as checked but unavailable

    return _FACE_DETECTOR if _FACE_DETECTOR is not False else None


def crop_face_160(
    img_bgr: np.ndarray,
    target_size: int = 160,
    margin_ratio: float = 0.2,
) -> Optional[Image.Image]:
    """
    Extracts a 160x160 RGB face crop from an image.
    Uses Haar Cascade detection if available; otherwise uses a smart portrait center crop.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None

    h, w = img_bgr.shape[:2]
    detector = get_face_detector()
    face_crop = None

    if detector is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            if len(faces) > 0:
                # Pick largest detected face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, fw, fh = faces[0]

                margin_x = int(fw * margin_ratio)
                margin_y = int(fh * margin_ratio)

                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(w, x + fw + margin_x)
                y2 = min(h, y + fh + margin_y)
                face_crop = img_bgr[y1:y2, x1:x2]
        except Exception:
            face_crop = None

    # Fallback: Smart Portrait Center-Crop
    # In anti-spoofing and biometric video datasets, the subject's face is
    # centered in the upper-middle portion of the camera frame.
    if face_crop is None or face_crop.size == 0:
        side = min(h, w)
        crop_size = int(side * 0.75)
        cy = int(h * 0.45)  # face is centered slightly above vertical midpoint
        cx = int(w * 0.5)

        y1 = max(0, cy - crop_size // 2)
        y2 = min(h, y1 + crop_size)
        x1 = max(0, cx - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        face_crop = img_bgr[y1:y2, x1:x2]

    if face_crop is None or face_crop.size == 0:
        return None

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb).resize((target_size, target_size), Image.Resampling.BILINEAR)
    return pil_img


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Auto-Unzip
# ─────────────────────────────────────────────────────────────────────────────

def auto_extract_zips(directory: Path) -> None:
    """Finds and unpacks any .zip files present in directory."""
    zip_files = list(directory.glob("*.zip"))
    for zf in zip_files:
        logger.info("Found zip archive '%s'. Unpacking...", zf.name)
        try:
            with zipfile.ZipFile(zf, "r") as zip_ref:
                zip_ref.extractall(directory)
            logger.info("Extracted %s successfully.", zf.name)
        except Exception as exc:
            logger.warning("Could not extract %s: %s", zf.name, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Metadata CSV Parsing (e.g. real_and_fake.csv, labels.csv)
# ─────────────────────────────────────────────────────────────────────────────

def parse_metadata_csv(
    csv_path: Path,
    root_dir: Path,
    target_exts: set,
) -> List[Tuple[Path, str, Optional[str]]]:
    """
    Parses metadata CSV files (such as `real_and_fake.csv` in TrainingData.pro datasets).
    Looks for columns: (file/path/video, type/label/category, split).
    Returns [(file_path, category, split_name), ...]
    """
    results = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        cols = {c.lower().strip(): c for c in reader.fieldnames}
        file_col = next((cols[k] for k in ["file", "filepath", "path", "video", "filename", "name"] if k in cols), None)
        type_col = next((cols[k] for k in ["type", "label", "class", "category", "attack_type"] if k in cols), None)
        split_col = next((cols[k] for k in ["split", "subset", "partition"] if k in cols), None)

        if not file_col or not type_col:
            return []

        for row in reader:
            raw_file = row.get(file_col, "").strip()
            raw_type = row.get(type_col, "").strip().lower()
            raw_split = row.get(split_col, "").strip().lower() if split_col else None

            if not raw_file or not raw_type:
                continue

            # Determine category: real vs spoof
            if any(alias in raw_type for alias in REAL_ALIASES):
                category = "real"
            elif any(alias in raw_type for alias in SPOOF_ALIASES):
                category = "spoof"
            else:
                continue

            # Determine split
            split_name = "train"
            if raw_split:
                if any(k in raw_split for k in ("val", "test", "valid")):
                    split_name = "val"

            # Resolve file path on disk
            candidates = [
                root_dir / raw_file,
                csv_path.parent / raw_file,
                root_dir / "videos" / raw_file,
                root_dir / "data" / raw_file,
                csv_path.parent / Path(raw_file).name,
            ]
            matched_path = next((p for p in candidates if p.is_file()), None)

            # Recursive fallback by filename if not directly found
            if not matched_path:
                matches = list(root_dir.rglob(Path(raw_file).name))
                if matches:
                    matched_path = matches[0]

            if matched_path and matched_path.suffix.lower() in target_exts:
                results.append((matched_path, category, split_name))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Smart Recursive File Finder
# ─────────────────────────────────────────────────────────────────────────────

def discover_media_items(
    input_dir: Path,
    target_exts: set,
    split_ratio: float = 0.8,
) -> List[Tuple[Path, str, str]]:
    """
    Intelligently discovers real and spoof files across the directory tree.
    Priority:
      1. Check for CSV metadata (e.g. real_and_fake.csv)
      2. Check directory names (train/real, val/attack, etc.)
      3. Check filenames (real_01.mp4, attack_02.mp4)
    """
    auto_extract_zips(input_dir)

    # 1. Search for metadata CSV
    for csv_candidate in list(input_dir.rglob("*.csv")):
        logger.info("Found metadata CSV: %s. Inspecting...", csv_candidate)
        items = parse_metadata_csv(csv_candidate, input_dir, target_exts)
        if items:
            logger.info("Successfully matched %d media items from %s", len(items), csv_candidate.name)
            return items

    # 2. Recursive scan of files matching target extensions
    all_files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in target_exts]
    logger.info("Found %d media file(s) across '%s'. Classifying...", len(all_files), input_dir)

    classified_real, classified_spoof = [], []

    for fpath in all_files:
        path_str = str(fpath).lower()
        parts = [p.lower() for p in fpath.parts]

        # Check path parts
        is_real = any(any(alias == p or f"/{alias}/" in path_str or f"\\{alias}\\" in path_str for alias in REAL_ALIASES) for p in parts)
        is_spoof = any(any(alias == p or f"/{alias}/" in path_str or f"\\{alias}\\" in path_str for alias in SPOOF_ALIASES) for p in parts)

        # Check filename
        if not is_real and not is_spoof:
            name_lower = fpath.stem.lower()
            if any(alias in name_lower for alias in REAL_ALIASES):
                is_real = True
            elif any(alias in name_lower for alias in SPOOF_ALIASES):
                is_spoof = True

        # Check split
        split_name = "val" if any(k in parts for k in ("val", "test", "valid")) else "train"

        if is_real and not is_spoof:
            classified_real.append((fpath, "real", split_name))
        elif is_spoof:
            classified_spoof.append((fpath, "spoof", split_name))

    # If splits weren't explicit in folder names, assign train/val using split_ratio
    final_items = []
    for group in (classified_real, classified_spoof):
        if not group:
            continue
        # Check if already partitioned
        splits_present = {item[2] for item in group}
        if len(splits_present) > 1:
            final_items.extend(group)
        else:
            random.shuffle(group)
            split_idx = int(len(group) * split_ratio)
            for i, (path, cat, _) in enumerate(group):
                final_items.append((path, cat, "train" if i < split_idx else "val"))

    return final_items


def print_directory_diagnostics(input_dir: Path) -> None:
    """Prints diagnostic information about the directory tree when no samples are found."""
    logger.warning("--- Directory Diagnostics for '%s' ---", input_dir)
    if not input_dir.exists():
        logger.error("Directory '%s' does not exist!", input_dir)
        return

    ext_counts = Counter(p.suffix.lower() for p in input_dir.rglob("*") if p.is_file())
    top_entries = list(input_dir.iterdir())

    logger.info("Top-level contents (%d items):", len(top_entries))
    for e in top_entries[:15]:
        logger.info("  • %s %s", "[DIR] " if e.is_dir() else "[FILE]", e.name)

    logger.info("File types found across all subdirectories:")
    for ext, count in ext_counts.most_common(10):
        logger.info("  %s: %d file(s)", ext if ext else "[no-ext]", count)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: Video Processor
# ─────────────────────────────────────────────────────────────────────────────

def process_video_folder(
    input_dir: Path,
    output_dir: Path,
    frame_interval: int = 8,
    max_frames_per_video: int = 30,
    split_ratio: float = 0.8,
) -> None:
    """Processes videos into 160x160 face crops."""
    train_real = output_dir / "train" / "real"
    train_spoof = output_dir / "train" / "spoof"
    val_real = output_dir / "val" / "real"
    val_spoof = output_dir / "val" / "spoof"

    for d in (train_real, train_spoof, val_real, val_spoof):
        d.mkdir(parents=True, exist_ok=True)

    items = discover_media_items(input_dir, VIDEO_EXTENSIONS, split_ratio=split_ratio)

    # Auto-fallback: if 0 videos found but image files exist, process as images
    if not items:
        img_check = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if img_check:
            logger.info("No videos found, but found %d image file(s)! Switching to image processing...", len(img_check))
            process_image_folder(input_dir, output_dir, split_ratio=split_ratio)
            return

        logger.error("No video files found to process.")
        print_directory_diagnostics(input_dir)
        return

    logger.info("Discovered %d video(s) to process. Extracting face crops...", len(items))
    counts = {"real": 0, "spoof": 0}

    for v_idx, (v_path, category, split_name) in enumerate(items):
        dest_dir = (
            (train_real if category == "real" else train_spoof)
            if split_name == "train"
            else (val_real if category == "real" else val_spoof)
        )

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            continue

        frame_num = 0
        saved_from_video = 0

        while cap.isOpened() and saved_from_video < max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                face = crop_face_160(frame)
                if face is not None:
                    out_name = f"{category}_v{v_idx:04d}_f{frame_num:05d}.jpg"
                    face.save(dest_dir / out_name, "JPEG", quality=95)
                    counts[category] += 1
                    saved_from_video += 1

            frame_num += 1
        cap.release()

        if (v_idx + 1) % 20 == 0 or (v_idx + 1) == len(items):
            logger.info("Progress: %d/%d videos processed (%d real crops, %d spoof crops)", v_idx + 1, len(items), counts["real"], counts["spoof"])

    logger.info("Video processing finished: %d real crops, %d spoof crops saved to '%s'.", counts["real"], counts["spoof"], output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: Raw Images Processor
# ─────────────────────────────────────────────────────────────────────────────

def process_image_folder(
    input_dir: Path,
    output_dir: Path,
    split_ratio: float = 0.8,
) -> None:
    """Processes raw uncropped photos into 160x160 face crops."""
    train_real = output_dir / "train" / "real"
    train_spoof = output_dir / "train" / "spoof"
    val_real = output_dir / "val" / "real"
    val_spoof = output_dir / "val" / "spoof"

    for d in (train_real, train_spoof, val_real, val_spoof):
        d.mkdir(parents=True, exist_ok=True)

    items = discover_media_items(input_dir, IMAGE_EXTENSIONS, split_ratio=split_ratio)
    if not items:
        logger.error("No image files found to process.")
        print_directory_diagnostics(input_dir)
        return

    logger.info("Processing %d images into 160x160 face crops...", len(items))
    counts = {"real": 0, "spoof": 0}

    for idx, (img_path, category, split_name) in enumerate(items):
        dest_dir = (
            (train_real if category == "real" else train_spoof)
            if split_name == "train"
            else (val_real if category == "real" else val_spoof)
        )

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        face = crop_face_160(img_bgr)
        if face is not None:
            out_name = f"{category}_{counts[category]:05d}.jpg"
            face.save(dest_dir / out_name, "JPEG", quality=95)
            counts[category] += 1

        if (idx + 1) % 500 == 0 or (idx + 1) == len(items):
            logger.info("Progress: %d/%d images processed (%d real crops, %d spoof crops)", idx + 1, len(items), counts["real"], counts["spoof"])

    logger.info("Image processing finished: %d real crops, %d spoof crops saved to '%s'.", counts["real"], counts["spoof"], output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3: CelebA-Spoof Balanced Subset Extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_celeba_subset(
    celeba_root: Path,
    output_dir: Path,
    max_per_class: int = 2500,
    split_ratio: float = 0.8,
) -> None:
    """Extracts a clean, balanced subset from CelebA-Spoof."""
    auto_extract_zips(celeba_root)
    label_file = celeba_root / "metas" / "intra_test" / "train_label.txt"
    if not label_file.exists():
        # Search for any label file
        candidates = list(celeba_root.rglob("*train_label*.txt"))
        if candidates:
            label_file = candidates[0]
        else:
            raise FileNotFoundError(f"CelebA-Spoof label file not found in: {celeba_root}")

    train_real = output_dir / "train" / "real"
    train_spoof = output_dir / "train" / "spoof"
    val_real = output_dir / "val" / "real"
    val_spoof = output_dir / "val" / "spoof"

    for d in (train_real, train_spoof, val_real, val_spoof):
        d.mkdir(parents=True, exist_ok=True)

    real_paths, spoof_paths = [], []

    logger.info("Reading CelebA-Spoof annotations from %s...", label_file)
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rel_path, raw_label = parts[0], int(parts[1])
                full_path = celeba_root / rel_path
                if full_path.exists():
                    if raw_label == 0:
                        real_paths.append(full_path)
                    else:
                        spoof_paths.append(full_path)

    logger.info("Found %d real and %d spoof samples in dataset.", len(real_paths), len(spoof_paths))

    random.shuffle(real_paths)
    random.shuffle(spoof_paths)

    selected_real = real_paths[:max_per_class]
    selected_spoof = spoof_paths[:max_per_class]

    logger.info("Extracting %d real + %d spoof samples...", len(selected_real), len(selected_spoof))

    for category, paths in [("real", selected_real), ("spoof", selected_spoof)]:
        split_idx = int(len(paths) * split_ratio)
        for idx, src_path in enumerate(paths):
            dest_folder = (
                (train_real if category == "real" else train_spoof)
                if idx < split_idx
                else (val_real if category == "real" else val_spoof)
            )

            img_bgr = cv2.imread(str(src_path))
            if img_bgr is not None:
                face = crop_face_160(img_bgr)
                if face is not None:
                    out_name = f"{category}_{idx:05d}.jpg"
                    face.save(dest_folder / out_name, "JPEG", quality=95)

    logger.info("CelebA-Spoof subset created successfully at: %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 4: Interactive Webcam Collector
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam_collector(output_dir: Path, class_label: str = "real") -> None:
    """Captures live webcam faces and simulated attacks (requires local GUI)."""
    try:
        cv2.imshow("test", np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.destroyAllWindows()
    except Exception:
        logger.error(
            "GUI/Display not available (e.g. Google Colab or headless server). "
            "Webcam recording requires running locally on your laptop/PC with a physical camera. "
            "For Colab, use '--mode videos', '--mode images', or '--mode celeba-subset'."
        )
        return

    target_dir = output_dir / ("real" if class_label == "real" else "spoof")
    target_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam (camera index 0).")
        return

    logger.info("Webcam collector started for class '%s'.", class_label)
    logger.info("Keys: [SPACE] Capture frame | [A] Toggle auto-capture | [Q/ESC] Quit")

    auto_capture = False
    last_auto_time = 0.0
    count = len(list(target_dir.glob("*.jpg")))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preview = frame.copy()
        detector = get_face_detector()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status_text = f"Class: {class_label.upper()} | Saved: {count} | Auto: {'ON' if auto_capture else 'OFF'}"
        cv2.putText(preview, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("EduAI Face Anti-Spoof Dataset Collector", preview)

        now = cv2.getTickCount() / cv2.getTickFrequency()
        capture_now = False

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            capture_now = True
        elif key in (ord('a'), ord('A')):
            auto_capture = not auto_capture
            logger.info("Auto-capture toggled: %s", auto_capture)

        if auto_capture and (now - last_auto_time) >= 1.0:
            capture_now = True
            last_auto_time = now

        if capture_now:
            face_img = crop_face_160(frame)
            if face_img is not None:
                filename = f"{class_label}_{count:05d}.jpg"
                face_img.save(target_dir / filename, "JPEG", quality=95)
                count += 1
                logger.info("Captured [%s] -> %s", class_label, filename)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam session finished. %d total samples in %s.", count, target_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create & Preprocess Face Anti-Spoofing Datasets for EduAI")
    parser.add_argument(
        "--mode",
        choices=["videos", "images", "celeba-subset", "webcam"],
        default="videos",
        help="Processing mode (default: videos).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Input raw dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets/processed_pad",
        help="Output directory for processed 160x160 face crops.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/Val split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2500,
        help="Max samples per class for celeba-subset mode (default: 2500).",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=8,
        help="Sample 1 frame every N frames in video mode (default: 8).",
    )
    parser.add_argument(
        "--class-label",
        choices=["real", "spoof"],
        default="real",
        help="Class to record in webcam collector mode ('real' or 'spoof').",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)

    if args.mode == "videos":
        if not args.input_dir:
            logger.error("Please specify --input-dir pointing to your video dataset.")
            sys.exit(1)
        process_video_folder(
            Path(args.input_dir),
            out_path,
            frame_interval=args.frame_interval,
            split_ratio=args.split,
        )

    elif args.mode == "images":
        if not args.input_dir:
            logger.error("Please specify --input-dir with raw images.")
            sys.exit(1)
        process_image_folder(Path(args.input_dir), out_path, split_ratio=args.split)

    elif args.mode == "celeba-subset":
        if not args.input_dir:
            logger.error("Please specify --input-dir pointing to CelebA-Spoof root.")
            sys.exit(1)
        extract_celeba_subset(
            Path(args.input_dir),
            out_path,
            max_per_class=args.max_per_class,
            split_ratio=args.split,
        )

    elif args.mode == "webcam":
        run_webcam_collector(out_path, class_label=args.class_label)


if __name__ == "__main__":
    main()
