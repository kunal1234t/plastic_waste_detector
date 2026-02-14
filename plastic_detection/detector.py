import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("detector")

RESIN_CODES = {
    "PET":   {"code": 1, "name": "Polyethylene Terephthalate", "recyclable": True},
    "HDPE":  {"code": 2, "name": "High-Density Polyethylene",  "recyclable": True},
    "PVC":   {"code": 3, "name": "Polyvinyl Chloride",         "recyclable": False},
    "LDPE":  {"code": 4, "name": "Low-Density Polyethylene",   "recyclable": True},
    "PP":    {"code": 5, "name": "Polypropylene",              "recyclable": True},
    "PS":    {"code": 6, "name": "Polystyrene",                "recyclable": False},
    "OTHER": {"code": 7, "name": "Other Plastics",             "recyclable": False},
}

# ---------------------------------------------------------------------------
# Sub-type rules: items that can be waste (disposable) or reusable.
# Each entry has a "default" profile and optional sub-type overrides. The
# WasteSubClassifier picks the right sub-type at runtime using visual cues.
# ---------------------------------------------------------------------------

# Per-label waste bias: offset added to the raw waste_score.
# Positive = biased toward waste, negative = biased toward reusable.
# Most bottles, cups, bags detected by YOLO in urban settings ARE waste.
WASTE_BIAS: dict[str, float] = {
    "bottle":   0.20,   # most bottles are disposable PET (Bisleri, Aquafina)
    "cup":      0.15,   # most cups seen are disposable
    "handbag":  0.15,   # thin carry bags are usually single-use
    "backpack": 0.10,   # large polythene bags
    "bowl":     0.10,
    "vase":     0.05,
}

SUB_TYPE_RULES: dict[str, dict] = {
    "bottle": {
        "sub_types": {
            "disposable": {
                "item_type": "Water Bottle",
                "material": "PET", "resin_code": 1, "recyclable": True,
                "waste_category": "waste",
                "description": "Single-use PET mineral water / soda bottle (e.g. Bisleri, Aquafina)",
            },
            "reusable": {
                "item_type": "Water Bottle",
                "material": "PP", "resin_code": 5, "recyclable": True,
                "waste_category": "reusable",
                "description": "Reusable PP / Tritan / HDPE daily-use water bottle",
            },
        },
    },
    "cup": {
        "sub_types": {
            "disposable": {
                "item_type": "Plastic Cup",
                "material": "PS", "resin_code": 6, "recyclable": False,
                "waste_category": "waste",
                "description": "Polystyrene / PET single-use disposable cup",
            },
            "reusable": {
                "item_type": "Plastic Cup",
                "material": "PP", "resin_code": 5, "recyclable": True,
                "waste_category": "reusable",
                "description": "Polypropylene reusable tumbler / cup",
            },
        },
    },
    "handbag": {
        "sub_types": {
            "disposable": {
                "item_type": "Polythene Bag",
                "material": "LDPE", "resin_code": 4, "recyclable": True,
                "waste_category": "waste",
                "description": "LDPE / HDPE single-use polythene carry bag",
            },
            "reusable": {
                "item_type": "Shopping Bag",
                "material": "PP", "resin_code": 5, "recyclable": True,
                "waste_category": "reusable",
                "description": "Woven PP / non-woven fabric reusable bag",
            },
        },
    },
    "backpack": {
        "sub_types": {
            "disposable": {
                "item_type": "Polythene Bag (Large)",
                "material": "LDPE", "resin_code": 4, "recyclable": True,
                "waste_category": "waste",
                "description": "Large LDPE / HDPE polythene carry bag",
            },
            "reusable": {
                "item_type": "Reusable Bag",
                "material": "PP", "resin_code": 5, "recyclable": True,
                "waste_category": "reusable",
                "description": "Reusable bag / backpack — woven or fabric",
            },
        },
    },
    "bowl": {
        "sub_types": {
            "disposable": {
                "item_type": "Plastic Bowl",
                "material": "PS", "resin_code": 6, "recyclable": False,
                "waste_category": "waste",
                "description": "Polystyrene / thin PP single-use food bowl",
            },
            "reusable": {
                "item_type": "Plastic Bowl",
                "material": "PP", "resin_code": 5, "recyclable": True,
                "waste_category": "reusable",
                "description": "Polypropylene reusable food container",
            },
        },
    },
    "vase": {
        "sub_types": {
            "disposable": {
                "item_type": "Plastic Container",
                "material": "PET", "resin_code": 1, "recyclable": True,
                "waste_category": "waste",
                "description": "Thin PET clamshell / disposable container",
            },
            "reusable": {
                "item_type": "Plastic Container",
                "material": "PP", "resin_code": 5, "recyclable": True,
                "waste_category": "reusable",
                "description": "Rigid PP / HDPE reusable container",
            },
        },
    },
}

PLASTIC_MAP: dict[str, dict] = {
    "bottle": {
        "item_type": "Plastic Bottle",
        "is_plastic": True,
        "material": "PET",
        "resin_code": 1,
        "recyclable": True,
        "description": "PET water/soda bottles",
    },
    "cup": {
        "item_type": "Plastic Cup",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": True,
        "description": "Polypropylene cups",
    },
    "wine glass": {
        "item_type": "Plastic Cup/Glass",
        "is_plastic": True,
        "material": "PS",
        "resin_code": 6,
        "recyclable": False,
        "description": "Polystyrene disposable glass",
    },
    "bowl": {
        "item_type": "Plastic Bowl",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": True,
        "description": "Polypropylene food container",
    },
    "vase": {
        "item_type": "Plastic Container",
        "is_plastic": True,
        "material": "PET",
        "resin_code": 1,
        "recyclable": True,
        "description": "PET container",
    },
    "handbag": {
        "item_type": "Plastic Bag",
        "is_plastic": True,
        "material": "LDPE",
        "resin_code": 4,
        "recyclable": True,
        "description": "LDPE shopping/carry bag",
    },
    "backpack": {
        "item_type": "Plastic Bag (Large)",
        "is_plastic": True,
        "material": "LDPE",
        "resin_code": 4,
        "recyclable": True,
        "description": "LDPE/HDPE large carry bag",
    },
    "suitcase": {
        "item_type": "Plastic Container (Large)",
        "is_plastic": True,
        "material": "HDPE",
        "resin_code": 2,
        "recyclable": True,
        "description": "HDPE rigid container",
    },
    "umbrella": {
        "item_type": "Plastic Item",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "Mixed plastic/nylon",
    },
    "cell phone": {
        "item_type": "Plastic Wrapper/Casing",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "Mixed plastic casing — e-waste",
    },
    "remote": {
        "item_type": "Plastic Casing",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": True,
        "description": "Polypropylene plastic casing",
    },
    "toothbrush": {
        "item_type": "Plastic Hygiene Item",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": False,
        "description": "Polypropylene handle — mixed materials",
    },
    "scissors": {
        "item_type": "Plastic Handle Item",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": False,
        "description": "Polypropylene handles — mixed materials",
    },
    "mouse": {
        "item_type": "Plastic Casing",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "ABS/mixed plastic casing — e-waste",
    },
    "keyboard": {
        "item_type": "Plastic Casing",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "ABS/mixed plastic casing — e-waste",
    },
    "fork": {
        "item_type": "Plastic Cutlery",
        "is_plastic": True,
        "material": "PS",
        "resin_code": 6,
        "recyclable": False,
        "description": "Polystyrene single-use cutlery",
    },
    "knife": {
        "item_type": "Plastic Cutlery",
        "is_plastic": True,
        "material": "PS",
        "resin_code": 6,
        "recyclable": False,
        "description": "Polystyrene single-use cutlery",
    },
    "spoon": {
        "item_type": "Plastic Cutlery",
        "is_plastic": True,
        "material": "PS",
        "resin_code": 6,
        "recyclable": False,
        "description": "Polystyrene single-use cutlery",
    },
    "refrigerator": {
        "item_type": "Plastic Appliance",
        "is_plastic": True,
        "material": "HDPE",
        "resin_code": 2,
        "recyclable": True,
        "description": "HDPE panels — industrial recycling",
    },
    "microwave": {
        "item_type": "Plastic Appliance",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": False,
        "description": "Polypropylene casing — e-waste",
    },
    "sink": {
        "item_type": "Plastic Fixture",
        "is_plastic": True,
        "material": "PVC",
        "resin_code": 3,
        "recyclable": False,
        "description": "PVC plumbing fixture",
    },
    "toilet": {
        "item_type": "Plastic Fixture",
        "is_plastic": True,
        "material": "PVC",
        "resin_code": 3,
        "recyclable": False,
        "description": "PVC/HDPE plumbing",
    },
    "book": {
        "item_type": "Plastic-Wrapped Item",
        "is_plastic": True,
        "material": "LDPE",
        "resin_code": 4,
        "recyclable": True,
        "description": "LDPE shrink-wrap/lamination",
    },
    "laptop": {
        "item_type": "Plastic Casing",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "ABS/polycarbonate casing — e-waste",
    },
    "tvmonitor": {
        "item_type": "Plastic Casing",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "ABS/polycarbonate casing — e-waste",
    },
    "hair drier": {
        "item_type": "Plastic Appliance",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": False,
        "description": "Polypropylene casing — e-waste",
    },
    "tie": {
        "item_type": "Polythene Wrapper / Film",
        "is_plastic": True,
        "material": "LDPE",
        "resin_code": 4,
        "recyclable": True,
        "description": "LDPE polythene wrap / thin plastic film",
    },
    "kite": {
        "item_type": "Plastic Film / Sheet",
        "is_plastic": True,
        "material": "LDPE",
        "resin_code": 4,
        "recyclable": True,
        "description": "LDPE / HDPE plastic film or sheet waste",
    },
    "frisbee": {
        "item_type": "Plastic Disc Waste",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": True,
        "description": "Polypropylene plastic disc",
    },
    "oven": {
        "item_type": "Plastic Appliance",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": False,
        "description": "Polypropylene casing — e-waste",
    },
    "toaster": {
        "item_type": "Plastic Appliance",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": False,
        "description": "Polypropylene casing — e-waste",
    },
}


def _get_plastic_category(info: dict) -> str:
    item = info["item_type"].lower()
    if "bottle" in item or "cup" in item or "glass" in item or "bowl" in item or "container" in item:
        return "plastic_bottle"
    elif "bag" in item or "carrier" in item or "polythene" in item:
        return "plastic_bag"
    elif "cutlery" in item:
        return "plastic_cutlery"
    elif "wrapper" in item or "film" in item or "sheet" in item or "wrap" in item:
        return "plastic_wrapper"
    elif "casing" in item or "appliance" in item or "fixture" in item:
        return "plastic_casing"
    else:
        return "plastic_wrapper"


COLORS: dict[str, tuple[int, int, int]] = {
    "plastic_bottle":  (0, 255, 0),
    "plastic_bag":     (0, 165, 255),
    "plastic_wrapper": (0, 0, 255),
    "plastic_cutlery": (255, 0, 255),
    "plastic_casing":  (255, 255, 0),
}

WASTE_COLORS: dict[str, tuple[int, int, int]] = {
    "waste":    (0, 0, 255),      # red
    "reusable": (0, 200, 0),      # green
    "unknown":  (0, 165, 255),    # orange
}

COLOR_RECYCLABLE = (0, 200, 0)
COLOR_NOT_RECYCLABLE = (0, 0, 200)
DEFAULT_COLOR = (255, 255, 0)
COLOR_SCANNING = (200, 200, 0)   # teal for "scanning" state


class DetectionTracker:
    """Temporal persistence filter — confirms detections over time.

    Only reports an item once it has been **consistently present** for
    ``confirm_secs`` (default 3 s).  Items that appear for less than
    ``min_presence`` (default 2 s) are silently discarded.

    While an item is being analysed (between min_presence and confirm_secs)
    it is returned with ``tracking_status = "scanning"`` so the UI can show
    a progress indicator.  Once confirmed it switches to ``"confirmed"``.
    """

    def __init__(self, confirm_secs: float = 3.0, min_presence: float = 2.0,
                 expire_secs: float = 2.0, iou_thresh: float = 0.35):
        self.confirm_secs = confirm_secs
        self.min_presence = min_presence
        self.expire_secs  = expire_secs
        self.iou_thresh   = iou_thresh
        self._tracks: list[dict] = []       # active tracked objects
        self._next_id: int = 0
        self._confirmed_labels: set[str] = set()  # labels confirmed at least once
        log.info("DetectionTracker ready  (confirm=%.1fs, min_presence=%.1fs, "
                 "expire=%.1fs)", confirm_secs, min_presence, expire_secs)

    # -------------------------------------------------------------- #
    #  Spatial helpers                                                  #
    # -------------------------------------------------------------- #
    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx); y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center_dist(a: tuple, b: tuple) -> float:
        """Euclidean distance between centers of two (x, y, w, h) boxes."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ca = (ax + aw / 2, ay + ah / 2)
        cb = (bx + bw / 2, by + bh / 2)
        return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

    @staticmethod
    def _update_track(track: dict, det: dict, now: float) -> None:
        """Merge a new detection into an existing track."""
        track["last_seen"] = now
        track["hit_count"] += 1
        track["box"] = det["box"]
        track["conf_sum"] += det["confidence"]
        track["best_det"] = det.copy()

    # -------------------------------------------------------------- #
    #  Public API                                                      #
    # -------------------------------------------------------------- #
    def update(self, detections: list[dict]) -> list[dict]:
        """Feed raw detections, get back filtered list with tracking info.

        Returns only detections that have been present >= min_presence.
        Each returned detection has extra keys:
          * ``tracking_status``: ``"scanning"`` | ``"confirmed"``
          * ``tracking_progress``: 0.0 → 1.0 (fraction toward confirm)
          * ``tracking_secs``: seconds this item has been tracked
        """
        now = time.time()

        # --- Pass 1: Match by IoU (object barely moved) ---
        matched_tracks: set[int] = set()
        matched_dets:   set[int] = set()

        for track in sorted(self._tracks, key=lambda t: t["first_seen"]):
            best_iou = 0.0
            best_det_idx = -1
            for di, det in enumerate(detections):
                if di in matched_dets:
                    continue
                if det["label"] != track["label"]:
                    continue
                iou = self._iou(det["box"], track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = di
            if best_iou >= self.iou_thresh and best_det_idx >= 0:
                self._update_track(track, detections[best_det_idx], now)
                matched_tracks.add(id(track))
                matched_dets.add(best_det_idx)

        # --- Pass 2: Label + nearest-center fallback (object moved) ---
        # For tracks that didn't get an IoU match, find the closest
        # unmatched detection with the same label.
        for track in self._tracks:
            if id(track) in matched_tracks:
                continue
            best_dist = float("inf")
            best_det_idx = -1
            for di, det in enumerate(detections):
                if di in matched_dets:
                    continue
                if det["label"] != track["label"]:
                    continue
                dist = self._center_dist(det["box"], track["box"])
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = di
            if best_det_idx >= 0:
                # Same label still in frame → keep the timer going
                self._update_track(track, detections[best_det_idx], now)
                matched_tracks.add(id(track))
                matched_dets.add(best_det_idx)

        # --- Create new tracks for unmatched detections ---
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            # If this label was confirmed before, skip scanning entirely
            if det["label"] in self._confirmed_labels:
                first = now - self.confirm_secs  # instant confirm
            else:
                first = now
            self._tracks.append({
                "id":         self._next_id,
                "label":      det["label"],
                "box":        det["box"],
                "first_seen": first,
                "last_seen":  now,
                "hit_count":  1,
                "conf_sum":   det["confidence"],
                "best_det":   det.copy(),
            })
            self._next_id += 1

        # --- Expire stale tracks ---
        self._tracks = [t for t in self._tracks
                        if (now - t["last_seen"]) < self.expire_secs]

        # --- Build output list ---
        output: list[dict] = []
        for track in self._tracks:
            age = now - track["first_seen"]
            if age < self.min_presence:
                # Too brief — don't show at all
                continue

            det = track["best_det"].copy()
            det["box"] = track["box"]   # latest position

            # Average confidence across all hits
            avg_conf = track["conf_sum"] / max(track["hit_count"], 1)
            det["confidence"] = round(min(avg_conf, 0.99), 4)

            # Tracking metadata
            progress = min(age / self.confirm_secs, 1.0)
            det["tracking_status"]   = "confirmed" if progress >= 1.0 else "scanning"
            det["tracking_progress"] = round(progress, 3)
            det["tracking_secs"]     = round(age, 1)
            det["tracking_id"]       = track["id"]

            # Remember confirmed labels so re-detections skip scanning
            if progress >= 1.0:
                self._confirmed_labels.add(track["label"])

            output.append(det)

        return output


class WasteSubClassifier:
    """Visual heuristic sub-classifier for waste vs reusable items.

    Analyzes the cropped bounding-box region to estimate whether a
    detected plastic item is single-use waste or a reusable object.

    Heuristics (applied per COCO label):
      * Transparency   — high transparency  → disposable PET
      * Saturation     — low saturation     → clear disposable
      * Color variance — varied colours     → printed label (disposable)
      * Aspect ratio   — tall & thin        → mineral water bottle
      * Edge density   — crumpled / crushed → waste
      * Size           — very small on screen → likely disposable
    """

    # Weights per signal (tuned empirically)
    _W_TRANSPARENCY  = 0.20
    _W_SATURATION    = 0.15
    _W_COLOR_VAR     = 0.20   # NEW: label / print detection
    _W_ASPECT        = 0.15
    _W_EDGES         = 0.15
    _W_SIZE          = 0.15
    _BASE_THRESHOLD  = 0.45   # base threshold before per-label bias

    @staticmethod
    def _crop_roi(frame: np.ndarray, box: tuple) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x, y, bw, bh = box
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return frame[y1:y2, x1:x2]

    @classmethod
    def classify(cls, frame: np.ndarray, det: dict) -> dict:
        """Return updated detection dict with waste_category & sub_type fields."""
        label = det["label"]
        rules = SUB_TYPE_RULES.get(label)

        # Items always classified as waste (cutlery, wine glasses, etc.)
        if rules is None and label in ("wine glass", "fork", "knife", "spoon"):
            det["waste_category"] = "waste"
            det["sub_type"] = "disposable"
            return det

        # Items always classified as waste (polythene film, wrappers)
        if rules is None and label in ("tie", "kite"):
            det["waste_category"] = "waste"
            det["sub_type"] = "disposable"
            return det

        if rules is None:
            det["waste_category"] = "unknown"
            det["sub_type"] = "default"
            return det

        roi = cls._crop_roi(frame, det["box"])
        if roi is None:
            # Can't analyze — use bias to decide
            bias = WASTE_BIAS.get(label, 0.0)
            if bias > 0.10:
                det["waste_category"] = "waste"
                det["sub_type"] = "disposable"
                sub = rules["sub_types"]["disposable"]
            else:
                det["waste_category"] = "unknown"
                det["sub_type"] = "default"
                return det
        else:
            score = cls._compute_waste_score(roi, det["box"], frame.shape, label)
            is_waste = score >= cls._BASE_THRESHOLD
            sub_key = "disposable" if is_waste else "reusable"
            sub = rules["sub_types"][sub_key]
            det["waste_score"] = round(score, 3)

        det["item_type"]       = sub["item_type"]
        det["material"]        = sub["material"]
        det["material_name"]   = RESIN_CODES[sub["material"]]["name"]
        det["resin_code"]      = sub["resin_code"]
        det["recyclable"]      = sub["recyclable"]
        det["description"]     = sub["description"]
        det["waste_category"]  = sub["waste_category"]
        det["sub_type"]        = sub_key
        return det

    @classmethod
    def _compute_waste_score(cls, roi: np.ndarray, box: tuple,
                             frame_shape: tuple, label: str) -> float:
        """0.0 = definitely reusable, 1.0 = definitely waste."""
        scores: list[tuple[float, float]] = []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # --- 1. Transparency: clear/light pixels → disposable PET ---
        bright_ratio = float(np.mean(gray > 200))
        transparency_score = min(bright_ratio * 2.0, 1.0)
        scores.append((cls._W_TRANSPARENCY, transparency_score))

        # --- 2. Saturation: vivid solid colours → reusable, pale → disposable ---
        mean_sat = float(np.mean(hsv[:, :, 1])) / 255.0
        saturation_score = 1.0 - min(mean_sat * 1.8, 1.0)
        scores.append((cls._W_SATURATION, saturation_score))

        # --- 3. Color variance: high variance → printed label → disposable ---
        # Reusable bottles are usually uniform colour; disposable have labels
        b_ch, g_ch, r_ch = cv2.split(roi)
        color_std = float(np.mean([np.std(b_ch), np.std(g_ch), np.std(r_ch)])) / 128.0
        color_var_score = min(color_std * 1.5, 1.0)
        scores.append((cls._W_COLOR_VAR, color_var_score))

        # --- 4. Aspect ratio: tall thin → mineral water bottle (waste) ---
        _, _, bw, bh = box
        aspect = bh / max(bw, 1)
        if aspect > 2.5:
            aspect_score = 0.95   # very tall & thin = almost certainly disposable
        elif aspect > 1.8:
            aspect_score = 0.7
        elif aspect > 1.2:
            aspect_score = 0.4
        else:
            aspect_score = 0.15   # wide / squat = likely reusable
        scores.append((cls._W_ASPECT, aspect_score))

        # --- 5. Edge density: crushed / crumpled → waste ---
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.mean(edges > 0))
        edge_score = min(edge_ratio * 2.5, 1.0)
        scores.append((cls._W_EDGES, edge_score))

        # --- 6. Relative size: small on-screen → disposable ---
        frame_area = frame_shape[0] * frame_shape[1]
        det_area = max(bw * bh, 1)
        area_ratio = det_area / frame_area
        if area_ratio < 0.02:
            size_score = 0.8
        elif area_ratio < 0.06:
            size_score = 0.5
        else:
            size_score = 0.15
        scores.append((cls._W_SIZE, size_score))

        raw = sum(w * s for w, s in scores)

        # Apply per-label waste bias
        bias = WASTE_BIAS.get(label, 0.0)
        total = min(raw + bias, 1.0)
        return total

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CFG_PATH = MODEL_DIR / "yolov4.cfg"
WEIGHTS_PATH = MODEL_DIR / "yolov4.weights"
NAMES_PATH = MODEL_DIR / "coco.names"
CFG_PATH_TINY = MODEL_DIR / "yolov4-tiny.cfg"
WEIGHTS_PATH_TINY = MODEL_DIR / "yolov4-tiny.weights"

BACKEND_URL = "http://localhost:8000"
REPORT_ENDPOINT = "/report"
BATCH_ENDPOINT = "/report/batch"


class PlasticDetector:
    """YOLOv4 detector that filters for plastic-related COCO objects."""

    def __init__(
        self,
        cfg: str | Path = CFG_PATH,
        weights: str | Path = WEIGHTS_PATH,
        names: str | Path = NAMES_PATH,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        input_size: int = 608,
        use_gpu: bool = False,
        multi_scale: bool = True,
        temporal_smooth: bool = True,
        roi_margin: float = 0.10,
        confirm_secs: float = 3.0,
        min_presence: float = 2.0,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.multi_scale = multi_scale
        self.temporal_smooth = temporal_smooth
        self.roi_margin = roi_margin
        self._history: list[list[dict]] = []
        self._history_len = 5
        self._tracker = DetectionTracker(
            confirm_secs=confirm_secs, min_presence=min_presence,
        )

        self.class_names = self._load_names(names)
        log.info("Loaded %d COCO class names", len(self.class_names))

        self.index_to_plastic: dict[int, dict] = {}
        for idx, name in enumerate(self.class_names):
            if name in PLASTIC_MAP:
                info = PLASTIC_MAP[name].copy()
                info["plastic_type"] = _get_plastic_category(info)
                self.index_to_plastic[idx] = info
        log.info("Mapped %d COCO classes to plastic categories", len(self.index_to_plastic))

        cfg_str, weights_str = str(cfg), str(weights)
        if not os.path.isfile(cfg_str):
            log.error("Config not found: %s", cfg_str)
            sys.exit(1)
        if not os.path.isfile(weights_str):
            log.error("Weights not found: %s", weights_str)
            sys.exit(1)

        self.net = cv2.dnn.readNetFromDarknet(cfg_str, weights_str)
        log.info("Loaded YOLOv4 network")

        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            log.info("Using CUDA backend")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        try:
            out_indices = self.net.getUnconnectedOutLayers().flatten()
        except AttributeError:
            out_indices = self.net.getUnconnectedOutLayers()
        self.output_layers = [layer_names[i - 1] for i in out_indices]

    @staticmethod
    def _load_names(path: str | Path) -> list[str]:
        p = Path(path)
        if not p.is_file():
            log.error("Names file not found: %s", p)
            sys.exit(1)
        with open(p, "r") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _detect_at_scale(self, frame: np.ndarray, input_size: int) -> tuple[
        list[list[int]], list[float], list[int]
    ]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (input_size, input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if class_id not in self.index_to_plastic:
                    continue
                if confidence < self.conf_threshold:
                    continue

                cx, cy, bw, bh = detection[0:4]
                cx = int(cx * w)
                cy = int(cy * h)
                bw = int(bw * w)
                bh = int(bh * h)
                x = cx - bw // 2
                y = cy - bh // 2

                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        return boxes, confidences, class_ids

    def _iou(self, box_a: tuple, box_b: tuple) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _apply_temporal_smoothing(self, detections: list[dict]) -> list[dict]:
        self._history.append(detections)
        if len(self._history) > self._history_len:
            self._history.pop(0)

        if len(self._history) < 2:
            return detections

        for det in detections:
            match_count = 0
            for past_frame in self._history[:-1]:
                for past_det in past_frame:
                    if (past_det["plastic_type"] == det["plastic_type"]
                            and self._iou(det["box"], past_det["box"]) > 0.3):
                        match_count += 1
                        break
            if match_count >= 2:
                det["confidence"] = min(det["confidence"] * 1.15, 0.99)
                det["confidence"] = round(det["confidence"], 4)

        current_boxes = [d["box"] for d in detections]
        for past_det in (self._history[-2] if len(self._history) >= 2 else []):
            matched = any(
                self._iou(past_det["box"], cb) > 0.3 for cb in current_boxes
            )
            if not matched and past_det["confidence"] > 0.35:
                recovered = dict(past_det)
                recovered["confidence"] = round(past_det["confidence"] * 0.85, 4)
                detections.append(recovered)

        return detections

    def detect(self, frame: np.ndarray) -> list[dict]:
        enhanced = self._preprocess(frame)

        all_boxes: list[list[int]] = []
        all_confs: list[float] = []
        all_cids: list[int] = []

        if self.multi_scale:
            scales = [self.input_size, self.input_size - 192, self.input_size + 192]
            scales = [max(320, min(832, (s // 32) * 32)) for s in scales]
            scales = sorted(set(scales))
        else:
            scales = [self.input_size]

        for scale in scales:
            b, c, ids = self._detect_at_scale(enhanced, scale)
            all_boxes.extend(b)
            all_confs.extend(c)
            all_cids.extend(ids)

        results: list[dict] = []
        if all_boxes:
            indices = cv2.dnn.NMSBoxes(
                all_boxes, all_confs, self.conf_threshold, self.nms_threshold,
            )
            if len(indices) > 0:
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                for i in indices:
                    cid = all_cids[i]
                    info = self.index_to_plastic[cid]
                    resin = RESIN_CODES.get(info["material"], RESIN_CODES["OTHER"])
                    results.append({
                        "label": self.class_names[cid],
                        "plastic_type": info["plastic_type"],
                        "item_type": info["item_type"],
                        "is_plastic": info["is_plastic"],
                        "material": info["material"],
                        "material_name": resin["name"],
                        "resin_code": info["resin_code"],
                        "recyclable": info["recyclable"],
                        "description": info["description"],
                        "confidence": round(all_confs[i], 4),
                        "box": tuple(all_boxes[i]),
                    })

        if self.temporal_smooth:
            results = self._apply_temporal_smoothing(results)

        # Sub-classify waste vs reusable using visual heuristics
        for det in results:
            WasteSubClassifier.classify(frame, det)

        # ── Center-frame ROI filter ──
        # Drop detections whose center falls outside the main region
        # (removes edge/peripheral false positives).
        if self.roi_margin > 0:
            h, w = frame.shape[:2]
            mx = int(w * self.roi_margin)
            my = int(h * self.roi_margin)
            filtered: list[dict] = []
            for det in results:
                bx, by, bw, bh = det["box"]
                cx = bx + bw // 2
                cy = by + bh // 2
                if mx <= cx <= w - mx and my <= cy <= h - my:
                    filtered.append(det)
            results = filtered

        # ── Temporal persistence filter ──
        # Only report items present long enough to be properly identified.
        results = self._tracker.update(results)

        return results

    @staticmethod
    def draw(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        for det in detections:
            x, y, bw, bh = det["box"]
            ptype = det["plastic_type"]
            material = det.get("material", "?")
            resin = det.get("resin_code", "?")
            recyclable = det.get("recyclable", False)
            item_type = det.get("item_type", det["label"])
            conf = det["confidence"]
            waste_cat = det.get("waste_category", "unknown")
            status = det.get("tracking_status", "confirmed")
            progress = det.get("tracking_progress", 1.0)
            track_secs = det.get("tracking_secs", 0)

            is_scanning = (status == "scanning")

            # Box colour: teal while scanning, normal once confirmed
            color = COLOR_SCANNING if is_scanning else COLORS.get(ptype, DEFAULT_COLOR)
            recycle_tag = "RECYCLABLE" if recyclable else "NOT RECYCLABLE"
            recycle_color = COLOR_RECYCLABLE if recyclable else COLOR_NOT_RECYCLABLE
            waste_tag = {"waste": "WASTE", "reusable": "REUSABLE"}.get(waste_cat, "")

            # --- Line 1: Item name + confidence (TOP of box) ---
            if is_scanning:
                line1 = f"Scanning... {progress:.0%} ({track_secs:.0f}s)"
            else:
                line1 = f"{item_type} {conf:.0%}"
            (tw1, th1), bl1 = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - th1 - bl1 - 4), (x + tw1 + 4, y), color, -1)
            cv2.putText(frame, line1, (x + 2, y - bl1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # --- Bounding box ---
            thickness = 1 if is_scanning else 2
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, thickness)

            # --- Progress bar (scanning only) ---
            if is_scanning:
                bar_y = y + bh + 2
                bar_h = 6
                bar_w_full = bw
                bar_w_fill = int(bar_w_full * progress)
                cv2.rectangle(frame, (x, bar_y), (x + bar_w_full, bar_y + bar_h),
                              (80, 80, 80), -1)
                cv2.rectangle(frame, (x, bar_y), (x + bar_w_fill, bar_y + bar_h),
                              COLOR_SCANNING, -1)
                continue   # don't draw material/waste rows while scanning

            # --- Line 2: Material info (BELOW box, row 1) ---
            line2 = f"{material} (#{resin}) | {recycle_tag}"
            (tw2, th2), bl2 = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y2_top = y + bh + 2
            y2_bot = y2_top + th2 + bl2 + 4
            cv2.rectangle(frame, (x, y2_top), (x + tw2 + 4, y2_bot), recycle_color, -1)
            cv2.putText(frame, line2, (x + 2, y2_bot - bl2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # --- Line 3: Waste/Reusable badge (BELOW material, row 2) ---
            if waste_tag:
                badge_color = WASTE_COLORS.get(waste_cat, DEFAULT_COLOR)
                (tw3, th3), bl3 = cv2.getTextSize(waste_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                y3_top = y2_bot + 2
                y3_bot = y3_top + th3 + bl3 + 4
                cv2.rectangle(frame, (x, y3_top), (x + tw3 + 6, y3_bot), badge_color, -1)
                cv2.putText(frame, waste_tag, (x + 3, y3_bot - bl3 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return frame


class BackendReporter:
    """Sends detection events to the FastAPI backend via HTTP."""

    def __init__(
        self,
        url: str = BACKEND_URL,
        zone_id: str = "Z-101",
        batch_size: int = 5,
        cooldown: float = 2.0,
    ):
        if not HAS_REQUESTS:
            log.warning("'requests' not installed — backend reporting disabled")
        self.url = url.rstrip("/")
        self.zone_id = zone_id
        self.batch_size = batch_size
        self.cooldown = cooldown
        self._last_report: dict[str, float] = {}
        self._buffer: list[dict] = []

    def _should_report(self, plastic_type: str) -> bool:
        now = time.time()
        last = self._last_report.get(plastic_type, 0)
        if now - last >= self.cooldown:
            self._last_report[plastic_type] = now
            return True
        return False

    def report(self, detections: list[dict]) -> None:
        if not HAS_REQUESTS:
            return

        for det in detections:
            if not self._should_report(det["plastic_type"]):
                continue
            event = {
                "zoneId": self.zone_id,
                "plasticType": det["plastic_type"],
                "itemType": det.get("item_type", ""),
                "material": det.get("material", ""),
                "materialName": det.get("material_name", ""),
                "resinCode": det.get("resin_code", 7),
                "recyclable": det.get("recyclable", False),
                "wasteCategory": det.get("waste_category", "unknown"),
                "subType": det.get("sub_type", "default"),
                "wasteScore": det.get("waste_score", 0),
                "confidence": det["confidence"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._buffer.append(event)

        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        try:
            if len(self._buffer) == 1:
                r = requests.post(
                    f"{self.url}{REPORT_ENDPOINT}",
                    json=self._buffer[0],
                    timeout=3,
                )
            else:
                r = requests.post(
                    f"{self.url}{BATCH_ENDPOINT}",
                    json=self._buffer,
                    timeout=3,
                )
            if r.ok:
                log.debug("Reported %d event(s) to backend", len(self._buffer))
            else:
                log.warning("Backend responded %d: %s", r.status_code, r.text[:200])
        except requests.ConnectionError:
            log.warning("Backend unreachable at %s", self.url)
        except Exception as exc:
            log.warning("Report failed: %s", exc)
        finally:
            self._buffer.clear()

    def flush_remaining(self) -> None:
        self._flush()


def _overlay_info(frame: np.ndarray, fps: float, count: int, zone: str) -> None:
    info_lines = [
        f"FPS: {fps:.1f}",
        f"Detections: {count}",
        f"Zone: {zone}",
    ]
    for i, line in enumerate(info_lines):
        y = 22 + i * 22
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def run_webcam(
    detector: PlasticDetector,
    camera_idx: int = 0,
    zone: str = "Z-101",
    reporter: Optional[BackendReporter] = None,
    no_display: bool = False,
    save_output: Optional[str] = None,
) -> None:
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        log.error("Cannot open webcam (index %d)", camera_idx)
        sys.exit(1)

    log.info("Webcam opened — press 'q' to quit")
    writer = None
    fps = 0.0
    frame_count = 0
    total_detections = 0

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame — retrying")
                time.sleep(0.1)
                continue

            detections = detector.detect(frame)
            total_detections += len(detections)
            frame_count += 1

            detector.draw(frame, detections)
            fps = 1.0 / max(time.time() - t0, 1e-6)
            _overlay_info(frame, fps, len(detections), zone)

            if reporter and detections:
                reporter.report(detections)

            if save_output and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_output, fourcc, 20, (w, h))
            if writer:
                writer.write(frame)

            if not no_display:
                cv2.imshow("Plastic Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            else:
                if frame_count % 100 == 0:
                    log.info("Frame %d | FPS %.1f | Detections: %d",
                             frame_count, fps, len(detections))
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        if reporter:
            reporter.flush_remaining()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        log.info("Done — %d frames, %d total detections", frame_count, total_detections)


def run_video(
    detector: PlasticDetector,
    source: str,
    zone: str = "Z-101",
    reporter: Optional[BackendReporter] = None,
    no_display: bool = False,
    save_output: Optional[str] = None,
) -> None:
    if not os.path.isfile(source):
        log.error("Video file not found: %s", source)
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open video: %s", source)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    log.info("Video: %s (%d frames @ %.1f FPS)", source, total_frames, input_fps)

    writer = None
    frame_count = 0
    total_detections = 0

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            confirmed = [d for d in detections if d.get("tracking_status") == "confirmed"]
            total_detections += len(confirmed)
            frame_count += 1

            detector.draw(frame, detections)
            fps = 1.0 / max(time.time() - t0, 1e-6)
            _overlay_info(frame, fps, len(confirmed), zone)

            if reporter and confirmed:
                reporter.report(confirmed)

            if save_output and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_output, fourcc, input_fps, (w, h))
            if writer:
                writer.write(frame)

            if not no_display:
                cv2.imshow("Plastic Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            if frame_count % 50 == 0:
                pct = frame_count / max(total_frames, 1) * 100
                log.info("Progress: %d/%d (%.0f%%)", frame_count, total_frames, pct)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        if reporter:
            reporter.flush_remaining()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        log.info("Done — %d/%d frames, %d total detections",
                 frame_count, total_frames, total_detections)


def run_image(
    detector: PlasticDetector,
    source: str,
    zone: str = "Z-101",
    reporter: Optional[BackendReporter] = None,
    no_display: bool = False,
    save_output: Optional[str] = None,
) -> None:
    paths: list[str] = []
    if os.path.isdir(source):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(
            str(p) for p in Path(source).iterdir() if p.suffix.lower() in exts
        )
        if not paths:
            log.error("No images found in %s", source)
            sys.exit(1)
        log.info("Found %d images in %s", len(paths), source)
    elif os.path.isfile(source):
        paths = [source]
    else:
        log.error("Image not found: %s", source)
        sys.exit(1)

    for img_path in paths:
        frame = cv2.imread(img_path)
        if frame is None:
            log.warning("Cannot read image: %s — skipping", img_path)
            continue

        detections = detector.detect(frame)
        detector.draw(frame, detections)
        log.info("%s — %d detection(s): %s", img_path, len(detections),
                 [d["plastic_type"] for d in detections] if detections else "none")

        if reporter and detections:
            reporter.report(detections)

        out_path = save_output
        if out_path is None:
            stem = Path(img_path).stem
            out_path = str(BASE_DIR / f"{stem}_detected.jpg")
        cv2.imwrite(out_path, frame)
        log.info("Saved: %s", out_path)

        if not no_display:
            cv2.imshow("Plastic Detection", frame)
            log.info("Press any key to continue (or 'q' to quit)")
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == 27:
                break

    if reporter:
        reporter.flush_remaining()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plastic Waste Detection Engine (YOLOv4 / Hybrid Pipeline)",
    )
    p.add_argument("--mode", choices=["webcam", "video", "image"], default="webcam")
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--size", type=int, default=608)
    p.add_argument("--no-multiscale", action="store_true")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--zone", type=str, default="Z-101")
    p.add_argument("--send", action="store_true")
    p.add_argument("--backend-url", type=str, default=BACKEND_URL)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--use-taco", action="store_true")
    p.add_argument("--use-classifier", action="store_true")
    p.add_argument("--taco-weights", type=str, default=None)
    p.add_argument("--classifier-weights", type=str, default=None)
    p.add_argument("--trashnet-weights", type=str, default=None)
    p.add_argument("--detection-threshold", type=float, default=0.5)
    p.add_argument("--classification-threshold", type=float, default=0.7)
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--roi-margin", type=float, default=0.10,
                   help="Fraction of frame edge to ignore (0.0-0.4, default 0.10)")
    p.add_argument("--confirm-secs", type=float, default=3.0,
                   help="Seconds an item must be present before reporting (default 3)")
    p.add_argument("--min-presence", type=float, default=2.0,
                   help="Minimum seconds before showing scanning indicator (default 2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.mode in ("video", "image") and not args.source:
        log.error("--source is required for %s mode", args.mode)
        sys.exit(1)

    use_hybrid = args.use_taco or args.use_classifier
    hybrid_detector = None

    if use_hybrid:
        try:
            from hybrid_detector import HybridPlasticDetector
            log.info("Hybrid detection pipeline enabled")
        except ImportError:
            log.error("hybrid_detector module not found.")
            sys.exit(1)

    if args.tiny:
        cfg = CFG_PATH_TINY
        weights = WEIGHTS_PATH_TINY
        log.info("Using YOLOv4-tiny (fast mode)")
    else:
        cfg = CFG_PATH
        weights = WEIGHTS_PATH
        log.info("Using full YOLOv4 (high accuracy mode)")

    yolo_detector = PlasticDetector(
        cfg=cfg,
        weights=weights,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        input_size=args.size,
        use_gpu=args.gpu,
        multi_scale=not args.no_multiscale,
        temporal_smooth=not args.no_smooth,
        roi_margin=args.roi_margin,
        confirm_secs=args.confirm_secs,
        min_presence=args.min_presence,
    )

    if use_hybrid:
        try:
            hybrid_detector = HybridPlasticDetector(
                use_taco=args.use_taco,
                taco_weights=args.taco_weights,
                detection_threshold=args.detection_threshold,
                use_classifier=args.use_classifier,
                classifier_weights=args.classifier_weights,
                classification_threshold=args.classification_threshold,
                trashnet_weights=args.trashnet_weights,
                yolo_fallback=yolo_detector,
                frame_skip=args.frame_skip,
            )
            log.info("Hybrid pipeline status: %s", hybrid_detector.pipeline_status)
        except Exception as e:
            log.warning("Failed to initialize hybrid pipeline: %s", e)
            log.warning("Falling back to YOLO-only detection")
            hybrid_detector = None

    active_detector = hybrid_detector if hybrid_detector else yolo_detector

    reporter = None
    if args.send:
        if not HAS_REQUESTS:
            log.error("Install 'requests' to use --send: pip install requests")
            sys.exit(1)
        reporter = BackendReporter(url=args.backend_url, zone_id=args.zone)
        log.info("Backend reporting enabled: %s (zone %s)", args.backend_url, args.zone)

    if args.mode == "webcam":
        run_webcam(active_detector, camera_idx=args.camera, zone=args.zone,
                   reporter=reporter, no_display=args.no_display, save_output=args.save)
    elif args.mode == "video":
        run_video(active_detector, source=args.source, zone=args.zone,
                  reporter=reporter, no_display=args.no_display, save_output=args.save)
    elif args.mode == "image":
        run_image(active_detector, source=args.source, zone=args.zone,
                  reporter=reporter, no_display=args.no_display, save_output=args.save)


if __name__ == "__main__":
    main()
