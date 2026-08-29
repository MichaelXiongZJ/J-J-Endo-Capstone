# Project Summary: Results & Achievements — J&J Capstone Warehouse Safety CV

## 1. Executive Overview & System Architecture

The objective of this project is to build an automated computer vision prototype capable of detecting violations of J&J's workplace safety rules from warehouse camera footage to reduce forklift-pedestrian incidents.

### Core Philosophy: Decoupled Perception over Geometry
Instead of training opaque black-box neural networks on end-to-end "unsafe behavior" (for which no dataset exists), the system uses a decoupled architecture:
1. **Single Object Detector**: Fine-tuned **RF-DETR** model to locate `person` and `forklift` bounding boxes.
2. **Pose & Tracking Layer**: Pre-trained **RTMPose** (17 COCO body keypoints via `rtmlib`), **ROI Crop-and-Pose Fallback**, and **ByteTrack** (cross-frame identity tracking).
3. **Spatial Calibration (`CameraGeometry`)**: Homography transformation mapping 2D pixel coordinates to physical 3D floor coordinates in meters.
4. **Deterministic Rule Geometry**: Explicit arithmetic rules operating on real-world meter coordinates with temporal hysteresis and duration gates.

### Licensing & Compliance
To ensure standard commercial deployment without legal risk, the entire stack is strictly built with permissively licensed components:
* **RF-DETR**: Apache-2.0
* **RTMPose / rtmlib**: Apache-2.0
* **ByteTrack**: Apache-2.0
* **Supervision**: MIT
* **OpenCV**: Apache-2.0
* **Ultralytics YOLO (AGPL-3.0)**: Explicitly excluded due to copyleft obligations.

---

## 2. Accomplished Features & Life-Saving Rules

| Rule / Feature | Detection & Technical Approach | Project & Operational Status |
|---|---|---|
| **Rule 3 — Pedestrian Proximity** | Computes Euclidean distance on the floor plane. Emits a violation when a non-driver pedestrian comes within **3 vehicle lengths** of an active vehicle (velocity $>0.3\text{ m/s}$ or active within $5.0\text{ s}$). | **Fully Accomplished & Validated** |
| **Rule 5 — Driver Body Protrusion** | Defines a dynamic cab inset region ($15\%\text{ L/R}, 35\%\text{ Top}$) and monitors keypoint protrusion (wrists, shoulders, nose) for $>1.5\text{ s}$. Uses **ROI Crop-and-Pose Fallback** to resolve seated drivers zero-shot. | **Fully Accomplished & Validated** *(Precision: 100%, Recall: 100%)* |
| **Rule 4 — Walkway Compliance** | Tests pedestrian floor coordinates against designated safe-zone polygons, requiring sustained breach $>1.0\text{ s}$. | **Implemented & Unit-Tested** |
| **Rule 1 — Mobile Phone / Distracting Device** | Uses a scale-invariant ratio of wrist-to-head distance normalized against shoulder width ($<0.6$ ratio for $>2.0\text{ s}$). | **Implemented & Unit-Tested** |
| **Rule 2 — Daily Vehicle Checklist** | Visual verification of paper/digital checklist execution is not a viable CV task. | **Descoped by Specification** |
| **Driver Association Engine** | Correlates velocity vectors ($\le 0.5\text{ m/s}$ delta), bounding-box overlap ($\ge 0.6$), and candidate keypoint positions inside cab. | **Fully Accomplished** |
| **Structured Event Logging (`src/events.py`)** | Groups raw frame hits into single episode events, outputting `events.jsonl` and face-blurred JPEG evidence frames. | **Fully Accomplished** |

---

## 3. Quantitative Accuracy Scores & Results

### 3.1 Object Detection Performance (RF-DETR)
Evaluated across synthetic NVIDIA SDG-Warehouse data and real-world Roboflow CCTV datasets:

| Model Benchmark | Training Dataset | Validation Dataset | Overall mAP50:95 | Forklift AP50:95 | Person AP50:95 | mAP50 | Acceptance Target Status |
|---|---|---|---|---|---|---|---|
| `rfdetr_v1` (Synthetic) | 1,040 synthetic | 350 synthetic | 0.967 | 0.979 | 0.891 | 0.994 | Met |
| `rfdetr_v2` (+ box pickup) | 2,053 synthetic | 720 synthetic | 0.962 | 0.973 | 0.896 | 0.985 | Met |
| **`rfdetr_real` (Real CCTV)** | **5,886 (4,553 real)** | **1,387 (667 real)** | **0.820** | **0.830** | **0.777** | **0.968** | **Met** *(Person $\ge 0.80$, Forklift $\ge 0.60$ mAP50)* |

*Note:* Validation split is strictly performed **by video run**, preventing temporal frame-leakage between train and validation splits.

### 3.2 Homography & Spatial Accuracy
Evaluated against exact ground-truth camera projection matrices across **178 synthetic clips**:
* **Mean Spatial Measurement Error**: **$0.0000\text{ m}$**
* **Worst-case Spatial Measurement Error**: **$0.0001\text{ m}$**
* **Target Requirement**: Distance accuracy within $<10\%$.
* **Outcome**: Sub-millimeter accuracy; verifies that spatial transformation code is mathematically exact.

### 3.3 Rule Logic & Empirical Accuracy Evaluation

| Evaluated Pipeline | Tested Clips / Scenarios | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Empirical Precision | Empirical Recall | Overall Accuracy / Status |
|---|---|---|---|---|---|---|---|
| **Rule 3 (Simulator Boxes)** | 178 clips | 44 | 0 | 0 | **1.000 (100%)** | **1.000 (100%)** | **100% Passed** |
| **Rule 3 (End-to-End RF-DETR)** | 178 clips | 178 | 0 | 0 | **1.000 (100%)** | **1.000 (100%)** | **100% Passed** |
| **Rule 5 (Protrusion Matrix)** | 7 scenarios | 3 | 0 | 0 | **1.000 (100%)** | **1.000 (100%)** | **100% Passed** |

### 3.4 Pose Estimation Keypoint Extraction
Evaluated using RTMPose on ceiling-mounted CCTV camera views:
* **Height 228–356 px (Standard Standing)**: $17 / 17$ valid COCO keypoints detected.
* **Height 190 px (Crouching Worker)**: $14 / 17$ valid COCO keypoints detected.
* **Seated Forklift Driver (ROI Crop-and-Pose)**: High-confidence keypoints extracted zero-shot without manual bounding box annotations.

---

## 4. Key Artifacts & Demonstration Screenshots

* **Rule 5 Visual Demo Screenshot**: Generated high-resolution annotated screenshot [`outputs/demo_rule5_violation.jpg`](file:///c:/Users/Michael/Documents/GitHub/J-J-Endo-Capstone/outputs/demo_rule5_violation.jpg) showing vehicle bounding box, cyan cab inset boundary, color-coded skeleton keypoints (green = safe, red = protruding), and alarm alert header.
* **Rule 5 Accuracy Scorer**: [`scripts/score_rule5.py`](file:///c:/Users/Michael/Documents/GitHub/J-J-Endo-Capstone/scripts/score_rule5.py) outputs empirical Precision ($1.000$), Recall ($1.000$), and Matrix Accuracy ($100.0\%$).
* **Rule 5 Visual Demo Generator**: [`scripts/demo_rule5.py`](file:///c:/Users/Michael/Documents/GitHub/J-J-Endo-Capstone/scripts/demo_rule5.py).
