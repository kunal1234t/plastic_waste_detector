# Plastic Waste Detector — Reference

## Quick Start

```bash
# Activate venv
venv\Scripts\activate

# Live webcam (balanced speed/accuracy)
python detector.py --mode webcam --tiny --size 416 --conf 0.25 --verbose

# Best accuracy (slow, ~2-4 FPS on CPU)
python detector.py --mode webcam --size 608 --conf 0.25 --verbose

# Fastest (~20+ FPS, weaker detection)
python detector.py --mode webcam --tiny --size 320 --no-multiscale --conf 0.3

# Image / Video
python detector.py --mode image --source photo.jpg --size 608 --verbose
python detector.py --mode video --source clip.mp4 --tiny --size 416 --save output.avi
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `webcam` | `webcam`, `video`, or `image` |
| `--source` | — | Path to image/video file (required for video/image mode) |
| `--camera` | `0` | Webcam index |
| `--tiny` | off | Use YOLOv4-tiny (fast) instead of full YOLOv4 |
| `--size` | `608` | Input resolution (320/416/608). Higher = more accurate, slower |
| `--conf` | `0.25` | Confidence threshold |
| `--nms` | `0.5` | Non-max suppression threshold |
| `--no-multiscale` | off | Disable multiscale detection (faster) |
| `--no-smooth` | off | Disable temporal smoothing |
| `--gpu` | off | Use CUDA backend (requires OpenCV with CUDA) |
| `--send` | off | Report detections to backend server |
| `--backend-url` | `http://localhost:3000` | Backend API URL |
| `--zone` | `Z-101` | Zone ID for reporting |
| `--save` | — | Save output to file (e.g. `output.avi`) |
| `--no-display` | off | Run headless (no window) |
| `--verbose` | off | Debug logging |
| `--use-taco` | off | Enable TACO Mask R-CNN (hybrid pipeline) |
| `--use-classifier` | off | Enable resin classifier (hybrid pipeline) |
| `--roi-margin` | `0.10` | Fraction of frame edge to ignore (0.0–0.4). Keeps only items in the center of frame |
| `--confirm-secs` | `3` | Seconds an item must stay in frame before it's reported |
| `--min-presence` | `2` | Minimum seconds before showing “Scanning…” indicator |

---

## Detection JSON Structure

Each detection from `detect()`:

```json
{
  "label": "bottle",
  "plastic_type": "plastic_bottle",
  "item_type": "Water Bottle",
  "is_plastic": true,
  "material": "PET",
  "material_name": "Polyethylene Terephthalate",
  "resin_code": 1,
  "recyclable": true,
  "description": "Single-use PET mineral water / soda bottle (e.g. Bisleri, Aquafina)",
  "confidence": 0.8723,
  "box": [120, 45, 80, 200],
  "waste_category": "waste",
  "sub_type": "disposable",
  "waste_score": 0.712
}
```

Backend event (sent with `--send`):

```json
{
  "zoneId": "Z-101",
  "plasticType": "plastic_bottle",
  "itemType": "Water Bottle",
  "material": "PET",
  "materialName": "Polyethylene Terephthalate",
  "resinCode": 1,
  "recyclable": true,
  "wasteCategory": "waste",
  "subType": "disposable",
  "wasteScore": 0.712,
  "confidence": 0.8723,
  "timestamp": "2026-02-14T15:38:12.000000+00:00"
}
```

---

## Waste Sub-Classification (NEW)

6 items are automatically sub-classified as **waste (disposable)** or **reusable** using visual analysis of the cropped region:

| COCO Label | Item Name | Waste Material | Reusable Material |
|---|---|---|---|
| `bottle` | Water Bottle | PET (#1) | PP (#5) |
| `cup` | Plastic Cup | PS (#6) | PP (#5) |
| `handbag` | Polythene Bag / Shopping Bag | LDPE (#4) | PP (#5) |
| `backpack` | Polythene Bag (Large) / Reusable Bag | LDPE (#4) | PP (#5) |
| `bowl` | Plastic Bowl | PS (#6) | PP (#5) |
| `vase` | Plastic Container | PET (#1) | PP (#5) |

The **item name stays the same** (e.g. "Water Bottle") — it's the **material line** and **waste badge** below the box that distinguish them.

**On-screen layout (3 rows below/above bounding box):**

```
  [Water Bottle 87%]          ← Row 1: Name + confidence (TOP of box)
  ┌──────────────────┐
  │  (bounding box)  │
  └──────────────────┘
  [PET (#1) | RECYCLABLE]     ← Row 2: Material + recyclable status
  [WASTE]                     ← Row 3: Waste/Reusable badge (separate)
```

**Visual heuristics used:**

| Signal | Weight | Waste indicator |
|---|---|---|
| Transparency | 20% | Clear/bright pixels → disposable PET |
| Saturation | 15% | Low colour saturation → clear disposable |
| Color variance | 20% | High variance → printed label (disposable) |
| Aspect ratio | 15% | Tall & thin → mineral water bottle |
| Edge density | 15% | Crushed/crumpled texture → waste |
| Relative size | 15% | Small on screen → disposable |

`waste_score` ≥ 0.45 (+ per-label bias) → **waste**, below → **reusable**

All other items keep `waste_category: "unknown"` unless they are always-waste.

---

## Temporal Tracking

Detections are **not reported instantly**. Each item goes through a tracking pipeline:

```
   Item enters frame
        │
   0–2 s  →  Hidden (too brief, could be noise)
        │
   2–3 s  →  "Scanning..." shown with progress bar
        │      (teal bounding box, no material info yet)
        │
   3 s+  →  ✅ Confirmed — full info displayed
        │      (item name, material, waste badge)
        │
   Gone for 2 s → Track expired & removed
```

| Parameter | Default | What it does |
|---|---|---|
| `--confirm-secs` | 3 | Seconds until full report (higher = more accurate, slower) |
| `--min-presence` | 2 | Seconds before showing scanning indicator |

**Why?** Prevents random misdetections from flashing on screen. Only items held steadily in front of the camera for 10+ seconds get fully identified and reported.

**On-screen while scanning:**
```
  [Scanning... 45% (5s)]      ← progress toward confirmation
  ┌──────────────────┐
  │  (teal dashed box)  │
  └──────────────────┘
  [██████████░░░░░░░░░░]   ← progress bar

---

## All 31 Detectable Items

| COCO Label | Item Type | Material | Resin | Recyclable | Waste Category |
|---|---|---|---|---|---|
| `bottle` | Water Bottle | PET / PP | 1 / 5 | Yes | sub-classified |
| `cup` | Plastic Cup | PS / PP | 6 / 5 | No / Yes | sub-classified |
| `wine glass` | Plastic Cup/Glass | PS | 6 | No | always waste |
| `bowl` | Plastic Bowl | PS / PP | 6 / 5 | No / Yes | sub-classified |
| `vase` | Plastic Container | PET / PP | 1 / 5 | Yes | sub-classified |
| `handbag` | Polythene Bag / Shopping Bag | LDPE / PP | 4 / 5 | Yes | sub-classified |
| `backpack` | Polythene Bag (Large) / Reusable Bag | LDPE / PP | 4 / 5 | Yes | sub-classified |
| `suitcase` | Plastic Container (Large) | HDPE | 2 | Yes | unknown |
| `umbrella` | Plastic Item | OTHER | 7 | No | unknown |
| `cell phone` | Plastic Wrapper/Casing | OTHER | 7 | No | unknown |
| `remote` | Plastic Casing | PP | 5 | Yes | unknown |
| `toothbrush` | Plastic Hygiene Item | PP | 5 | No | unknown |
| `scissors` | Plastic Handle Item | PP | 5 | No | unknown |
| `mouse` | Plastic Casing | OTHER | 7 | No | unknown |
| `keyboard` | Plastic Casing | OTHER | 7 | No | unknown |
| `fork` | Plastic Cutlery | PS | 6 | No | always waste |
| `knife` | Plastic Cutlery | PS | 6 | No | always waste |
| `spoon` | Plastic Cutlery | PS | 6 | No | always waste |
| `tie` | Polythene Wrapper / Film | LDPE | 4 | Yes | always waste |
| `kite` | Plastic Film / Sheet | LDPE | 4 | Yes | always waste |
| `frisbee` | Plastic Disc Waste | PP | 5 | Yes | unknown |
| `oven` | Plastic Appliance | PP | 5 | No | unknown |
| `toaster` | Plastic Appliance | PP | 5 | No | unknown |
| `refrigerator` | Plastic Appliance | HDPE | 2 | Yes | unknown |
| `microwave` | Plastic Appliance | PP | 5 | No | unknown |
| `sink` | Plastic Fixture | PVC | 3 | No | unknown |
| `toilet` | Plastic Fixture | PVC | 3 | No | unknown |
| `book` | Plastic-Wrapped Item | LDPE | 4 | Yes | unknown |
| `laptop` | Plastic Casing | OTHER | 7 | No | unknown |
| `tvmonitor` | Plastic Casing | OTHER | 7 | No | unknown |
| `hair drier` | Plastic Appliance | PP | 5 | No | unknown |

---

## Resin Codes

| Code | Symbol | Full Name | Recyclable |
|---|---|---|---|
| 1 | PET | Polyethylene Terephthalate | Yes |
| 2 | HDPE | High-Density Polyethylene | Yes |
| 3 | PVC | Polyvinyl Chloride | No |
| 4 | LDPE | Low-Density Polyethylene | Yes |
| 5 | PP | Polypropylene | Yes |
| 6 | PS | Polystyrene | No |
| 7 | OTHER | Other Plastics | No |

---

## Model Files (`model/`)

| File | Size | Required |
|---|---|---|
| `yolov4.weights` | 246 MB | Yes (full model) |
| `yolov4-tiny.weights` | 23 MB | Yes (fast model) |
| `yolov4.cfg` | 13 KB | Yes |
| `yolov4-tiny.cfg` | 4 KB | Yes |
| `coco.names` | 1 KB | Yes |
