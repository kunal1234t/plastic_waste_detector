#!/usr/bin/env python3
"""
=============================================================================
 Model Download Utility — Hybrid Plastic Waste Detection
=============================================================================

 Downloads pre-trained model weights for the hybrid detection pipeline.

 Available models:
   1. TACO Mask R-CNN        — waste detection (60 TACO categories)
   2. EfficientNet-B3        — resin type classifier (7 classes)
   3. TrashNet ResNet-50     — waste material classifier (6 classes)

 Usage:
   python download_models.py --all
   python download_models.py --taco
   python download_models.py --classifier
   python download_models.py --trashnet

 Notes:
   - TACO weights from https://github.com/pedropro/TACO/releases
   - Downloads are saved to the plastic_detection/model/ subdirectories
   - Existing files are skipped unless --force is specified

=============================================================================
"""

import argparse
import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("download_models")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "taco": {
        "display_name": "TACO Mask R-CNN (waste detection)",
        "dir": MODEL_DIR / "taco",
        "files": [
            {
                "filename": "taco_maskrcnn.pth",
                # Primary: TACO project release (converted to PyTorch)
                # The original TACO weights are in TF/Keras .h5 format.
                # We provide a converted version via GitHub releases.
                # If unavailable, user must convert manually.
                "url": "https://github.com/pedropro/TACO/releases/download/1.0/mask_rcnn_taco_0100.h5",
                "description": (
                    "TACO Mask R-CNN weights (TF/Keras .h5 original). "
                    "NOTE: These are the original TF/Keras format weights. "
                    "For PyTorch use, convert with the included conversion script, "
                    "or the system will automatically use COCO-pretrained torchvision "
                    "Mask R-CNN as fallback."
                ),
                "size_mb": 256,
                "optional": True,
            },
        ],
    },
    "classifier": {
        "display_name": "EfficientNet-B3 Resin Classifier",
        "dir": MODEL_DIR / "classifier",
        "files": [
            {
                "filename": "resin_classifier.pth",
                "url": None,  # No pre-trained weights available publicly
                "description": (
                    "EfficientNet-B3 trained on plastic resin types "
                    "(PET/HDPE/PVC/LDPE/PP/PS/OTHER). "
                    "Custom training required. The system will automatically use "
                    "ImageNet-pretrained EfficientNet-B3 backbone as fallback."
                ),
                "size_mb": 48,
                "optional": True,
            },
        ],
    },
    "trashnet": {
        "display_name": "TrashNet ResNet-50 Classifier",
        "dir": MODEL_DIR / "trashnet",
        "files": [
            {
                "filename": "trashnet_resnet50.pth",
                "url": None,  # Original TrashNet is Lua/Torch
                "description": (
                    "ResNet-50 trained on TrashNet dataset "
                    "(glass/paper/cardboard/plastic/metal/trash). "
                    "Original dataset: https://github.com/garythung/trashnet. "
                    "Custom PyTorch training required. The system will use "
                    "ImageNet-pretrained ResNet-50 backbone as fallback."
                ),
                "size_mb": 98,
                "optional": True,
            },
        ],
    },
    "yolov4": {
        "display_name": "YOLOv4 (already present)",
        "dir": MODEL_DIR,
        "files": [
            {
                "filename": "yolov4.weights",
                "url": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
                "description": "YOLOv4 COCO weights (ultimate fallback)",
                "size_mb": 246,
                "optional": False,
            },
            {
                "filename": "yolov4.cfg",
                "url": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
                "description": "YOLOv4 configuration",
                "size_mb": 0.012,
                "optional": False,
            },
            {
                "filename": "yolov4-tiny.weights",
                "url": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
                "description": "YOLOv4-tiny weights (lightweight fallback)",
                "size_mb": 22,
                "optional": False,
            },
            {
                "filename": "yolov4-tiny.cfg",
                "url": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
                "description": "YOLOv4-tiny configuration",
                "size_mb": 0.003,
                "optional": False,
            },
            {
                "filename": "coco.names",
                "url": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
                "description": "COCO class names (80 classes)",
                "size_mb": 0.001,
                "optional": False,
            },
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print download progress."""
    if total_size > 0:
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 / total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  ({mb_done:.1f}/{mb_total:.1f} MB)"
        )
        sys.stdout.flush()
    else:
        downloaded = block_num * block_size
        mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded {mb:.1f} MB...")
        sys.stdout.flush()


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """
    Download a file from URL to dest path.

    Returns True if download was successful, False otherwise.
    """
    if dest.exists() and not force:
        log.info("  ✓ Already exists: %s", dest.name)
        return True

    if not url:
        log.warning("  ⚠ No download URL available for %s", dest.name)
        log.info("    → Custom training required; see description above")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        log.info("  ↓ Downloading: %s", url.split("/")[-1])
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
        print()  # newline after progress bar
        size_mb = dest.stat().st_size / (1024 * 1024)
        log.info("  ✓ Saved: %s (%.1f MB)", dest.name, size_mb)
        return True
    except Exception as e:
        log.error("  ✗ Download failed: %s", e)
        if dest.exists():
            dest.unlink()
        return False


def download_model_group(group_key: str, force: bool = False) -> dict:
    """
    Download all files for a model group.

    Returns dict with 'success' and 'failed' counts.
    """
    if group_key not in MODELS:
        log.error("Unknown model group: %s", group_key)
        return {"success": 0, "failed": 1}

    group = MODELS[group_key]
    log.info("━" * 60)
    log.info("Model: %s", group["display_name"])
    log.info("━" * 60)

    # Ensure directory exists
    group["dir"].mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for file_info in group["files"]:
        dest = group["dir"] / file_info["filename"]
        log.info("")
        log.info("  File: %s", file_info["filename"])
        log.info("  %s", file_info["description"])

        if file_info.get("optional"):
            log.info("  (optional — system will use fallback if unavailable)")

        ok = download_file(file_info.get("url"), dest, force=force)
        if ok:
            success += 1
        else:
            failed += 1

    return {"success": success, "failed": failed}


def list_models() -> None:
    """Print available models and their download status."""
    print("\n" + "=" * 60)
    print(" Available Models for Hybrid Detection Pipeline")
    print("=" * 60)

    for key, group in MODELS.items():
        print(f"\n  [{key}] {group['display_name']}")
        print(f"       Directory: {group['dir']}")
        for file_info in group["files"]:
            dest = group["dir"] / file_info["filename"]
            status = "✓ downloaded" if dest.exists() else "✗ missing"
            opt = " (optional)" if file_info.get("optional") else ""
            print(f"         {file_info['filename']}: {status}{opt}")
            if file_info.get("size_mb"):
                print(f"           Size: ~{file_info['size_mb']} MB")

    print("\n" + "=" * 60)
    print(" Use --all to download all models, or --<model> for specific ones")
    print("=" * 60 + "\n")


def check_status() -> dict:
    """Check which models are available."""
    status = {}
    for key, group in MODELS.items():
        files_ok = True
        for file_info in group["files"]:
            dest = group["dir"] / file_info["filename"]
            if not dest.exists() and not file_info.get("optional"):
                files_ok = False
                break
        status[key] = files_ok
    return status


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for hybrid plastic waste detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py --list          Show all models and status
  python download_models.py --all           Download all available models
  python download_models.py --taco          Download TACO Mask R-CNN weights
  python download_models.py --classifier    Download EfficientNet-B3 classifier
  python download_models.py --trashnet      Download TrashNet ResNet-50 weights
  python download_models.py --yolov4        Download YOLOv4 base weights
  python download_models.py --all --force   Re-download everything
        """,
    )

    parser.add_argument(
        "--list", action="store_true",
        help="List all available models and their download status",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all available models",
    )
    parser.add_argument(
        "--taco", action="store_true",
        help="Download TACO Mask R-CNN waste detection weights",
    )
    parser.add_argument(
        "--classifier", action="store_true",
        help="Download EfficientNet-B3 resin classifier weights",
    )
    parser.add_argument(
        "--trashnet", action="store_true",
        help="Download TrashNet ResNet-50 weights",
    )
    parser.add_argument(
        "--yolov4", action="store_true",
        help="Download YOLOv4 base model weights",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Check and print model availability status",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.status:
        status = check_status()
        print("\nModel Status:")
        for key, ok in status.items():
            icon = "✓" if ok else "✗"
            print(f"  {icon} {MODELS[key]['display_name']}")
        return

    # Determine which models to download
    targets = []
    if args.all:
        targets = list(MODELS.keys())
    else:
        if args.taco:
            targets.append("taco")
        if args.classifier:
            targets.append("classifier")
        if args.trashnet:
            targets.append("trashnet")
        if args.yolov4:
            targets.append("yolov4")

    if not targets:
        parser.print_help()
        print("\n  ⚠ Specify at least one model to download (or use --all)")
        return

    # Do the downloads
    total_ok, total_fail = 0, 0
    for t in targets:
        result = download_model_group(t, force=args.force)
        total_ok += result["success"]
        total_fail += result["failed"]

    print("\n" + "=" * 60)
    print(f" Summary: {total_ok} succeeded, {total_fail} failed/unavailable")
    print("=" * 60)

    if total_fail > 0:
        print("\n  ℹ Models marked 'optional' will fall back to pre-trained")
        print("    ImageNet/COCO weights automatically.\n")


if __name__ == "__main__":
    main()
