# Warehouse Safety CV — Prototype Implementation Guide (RF-DETR)

**Goal:** A working proof-of-concept that detects **Rule 3** (pedestrian too close to a working vehicle) and **Rule 5** (driver's body outside the vehicle), plus **Rule 4** (off-walkway) and **Rule 1** (phone use) as cheap add-ons.

**Stack (fixed, all Apache-2.0 — license-clean for the J&J handoff by construction):**

| Component | Library | Trained by you? |
|---|---|---|
| Object detector (person, forklift) | **RF-DETR** (`rfdetr`) | **Yes** — fine-tuned once |
| Tracking (persistent IDs) | **ByteTrack** via `supervision` | No |
| Pose (17 body keypoints) | **RTMPose** via `rtmlib` | No |
| Geometry / video I/O | OpenCV + NumPy | No |

**Environment:** Google Colab (T4 GPU) for training and heavy inference. Local machine for labeling and light scripts.

---

## 0. How to Use This Guide

- Work **top to bottom**. Each phase ends with an **ACCEPTANCE CHECK** — do not continue until it passes.
- Every function used later is defined in this document. If a name is undefined, you skipped a section.
- **If you are an LLM agent executing this:** run each code block exactly as written, in order, substituting only paths marked `<LIKE THIS>`. After each ACCEPTANCE CHECK, verify the stated condition programmatically before proceeding. If a library call raises `TypeError`/`AttributeError`, the package API has drifted: run `help()` on the function, consult the package README, and adapt only the failing call — do not redesign the pipeline. Do not substitute different libraries or models; RF-DETR is a hard requirement.

### The big picture (read even if experienced)

You are **not** training a model to recognize "unsafe behavior" — no such dataset exists, and none is needed. You train exactly one thing: a detector that draws boxes around **people** and **forklifts**. Pose and tracking are pretrained/algorithmic. The rules themselves are **geometry over tracked objects**:

- Rule 3 = "are these two floor positions closer than 3 vehicle lengths, while the vehicle is working?"
- Rule 5 = "are the driver's keypoints outside the cab region for more than ~1.5 s?"

The one concept that makes this possible is **homography**: a one-time, ~10-minute per-camera calibration that converts image pixels (where distance is meaningless) into floor metres (where distance is real). Section 7 does this.

---

## 1. Setup

### 1.1 Accounts

- Google account + Drive (Colab's disk is wiped on disconnect; everything persistent lives in Drive)
- Google Colab — use the **T4 GPU** runtime only. Burn rates: T4 ≈ 2 credits/hr, A100 ≈ 12/hr. The whole prototype fits in **12–18 credits** of your 50. Never select A100.
- Roboflow free account (browser labeling + COCO export)
- Local Python 3.10+

### 1.2 Project folder

Create this exactly; later code assumes these paths.

```
warehouse-safety/
├── data/
│   ├── raw_videos/           # source clips
│   ├── frames/               # extracted stills
│   ├── dataset/              # labeled data (COCO format, from Roboflow)
│   └── calibration/          # one JSON per camera
├── models/                   # trained checkpoints
├── notebooks/                # Colab notebooks
├── src/
│   ├── extract_frames.py
│   ├── geometry.py
│   ├── pose_utils.py
│   ├── rules.py
│   └── run_pipeline.py
├── outputs/
│   ├── events/               # evidence frames + events.jsonl
│   └── videos/               # annotated output videos
└── requirements.txt
```

`git init` and push to GitHub on day one. Mirror the whole folder to Google Drive at `MyDrive/warehouse-safety` so Colab sees the same layout.

### 1.3 requirements.txt

```
rfdetr
supervision
rtmlib
onnxruntime
opencv-python
numpy
matplotlib
```

Local install: `pip install -r requirements.txt`
Colab install (first cell of every notebook):

```python
!pip install -q rfdetr supervision rtmlib onnxruntime-gpu opencv-python
```

### 1.4 Colab survival rules

```python
from google.colab import drive
drive.mount('/content/drive')
PROJECT = '/content/drive/MyDrive/warehouse-safety'
```

- Save **all** outputs (checkpoints, events, videos) under `{PROJECT}/...`, never only `/content`.
- **Copy training data to the local Colab disk first** — reading thousands of images through the Drive mount is 10–50× slower:
  ```python
  !cp -r "{PROJECT}/data/dataset" /content/dataset
  ```
- Disconnect the runtime when not actively using the GPU (Runtime → Disconnect). Idle sessions burn credits.

---

## 2. Phase 1 — Footage and Frames (Day 1)

### 2.1 Get video

Request from J&J: (a) several hours of normal-operations footage across cameras/lighting, and (b) **safety training videos** — these often contain staged demonstrations of exactly these violations and are the highest-value data you can obtain. If neither arrives by Day 2, film any accessible warehouse-like space, or start with public forklift videos from YouTube for pipeline development only (never for reported metrics).

Place clips in `data/raw_videos/`.

### 2.2 Extract frames — `src/extract_frames.py`

Adjacent video frames are near-identical; training wants diverse stills. One frame per second:

```python
import cv2, os, sys, glob

def extract_frames(video_path, out_dir, every_n_seconds=1.0):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps * every_n_seconds))
    name = os.path.splitext(os.path.basename(video_path))[0]
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"{name}_{idx:06d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved

if __name__ == "__main__":
    total = 0
    for vp in glob.glob("data/raw_videos/*"):
        n = extract_frames(vp, "data/frames")
        print(f"{vp}: {n} frames")
        total += n
    print("TOTAL:", total)
```

Run: `python src/extract_frames.py`

**ACCEPTANCE CHECK:** `data/frames/` contains 800–1,500 jpgs spanning every camera and lighting condition you have. Fewer than ~300 → extract at 2 frames/sec (`every_n_seconds=0.5`) or get more video.

---

## 3. Phase 2 — Zero-Shot Baseline (Day 1–2, Colab, ~1 credit)

Before training, measure what pretrained RF-DETR already does. Notebook `notebooks/01_baseline.ipynb`:

```python
!pip install -q rfdetr supervision opencv-python
from google.colab import drive; drive.mount('/content/drive')
PROJECT = '/content/drive/MyDrive/warehouse-safety'

import cv2, glob, supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES   # dict: class_id -> name

model = RFDETRBase()          # downloads COCO-pretrained weights on first run

box_ann, lab_ann = sv.BoxAnnotator(), sv.LabelAnnotator()

for path in sorted(glob.glob(f"{PROJECT}/data/frames/*.jpg"))[:30]:
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)      # CRITICAL: RF-DETR expects RGB
    dets = model.predict(rgb, threshold=0.5)
    labels = [f"{COCO_CLASSES[c]} {conf:.2f}"
              for c, conf in zip(dets.class_id, dets.confidence)]
    out = lab_ann.annotate(box_ann.annotate(bgr.copy(), dets), dets, labels)
    cv2.imwrite(f"{PROJECT}/outputs/baseline_{path.split('/')[-1]}", out)
    print(path.split('/')[-1], labels)
```

> **The #1 silent bug in this whole project:** OpenCV loads images as **BGR**; RF-DETR expects **RGB**. Forgetting `cvtColor` doesn't crash — it just quietly degrades accuracy. Every RF-DETR call in this guide converts first. (rtmlib, by contrast, takes BGR directly — Section 8.)

**Expected result:** people detected reliably (`person` is COCO class 1); forklifts **missed or mislabeled** as `truck`/`car` — COCO has no forklift class. That gap is precisely what Phase 3 closes.

**ACCEPTANCE CHECK:** in the saved annotated images, ≥80% of visible people have boxes; forklifts are wrong or absent (expected). If *people* are badly missed, your footage is unusual (extreme angle/dark) — collect more representative video before training.

---

## 4. Phase 3 — Label and Fine-Tune (Day 2–5, the big phase, 4–8 credits)

~60% of total prototype effort is here. All five teammates label.

### 4.1 Labeling rules (write these down before anyone draws a box)

Classes: `person`, `forklift`. Conventions — inconsistency silently corrupts the dataset:

1. **Pallet jacks / tuggers:** ask J&J whether they count as Rule-3 "vehicles." Until answered, label them `forklift` (you can filter later; you cannot un-merge).
2. Box the **whole vehicle including mast and forks**.
3. Label people **≥30% visible**, box the visible extent only.
4. **Always label the driver as `person`** — driver detection is what Rules 3 and 5 hang on.
5. Skip motion-blurred frames entirely rather than guessing.

**Volume:** 300–500 images = working prototype; 1,000+ = good one. **Label the validation set first** (~150 images) so every later training run is measurable.

### 4.2 The split rule that decides whether your metrics are real

**Never split train/val randomly by frame.** Frames seconds apart are near-duplicates; a random split leaks them across both sides and produces beautiful, fictional metrics that collapse on new footage. **Split by source video / camera / day:** e.g. videos A–C → train, video D (a camera the model never saw) → valid. In Roboflow: upload each video's frames as a separate batch and assign whole batches to splits manually.

### 4.3 Export in COCO format

RF-DETR trains on **COCO JSON**, not YOLO txt. In Roboflow: Generate → Export → format **"COCO"** → download. You get:

```
dataset/
├── train/   _annotations.coco.json + images
├── valid/   _annotations.coco.json + images
└── test/    _annotations.coco.json + images
```

Place at `data/dataset/` and mirror to Drive.

### 4.4 Discover your class IDs (do not hardcode from memory)

Roboflow's COCO export sometimes inserts a dummy category 0. Read the truth from the JSON:

```python
import json
with open('/content/dataset/train/_annotations.coco.json') as f:
    cats = json.load(f)['categories']
print({c['id']: c['name'] for c in cats})
# e.g. {0: 'objects', 1: 'forklift', 2: 'person'}
```

Record the two numbers; they parameterize everything downstream:

```python
FORKLIFT_ID = 1     # ← from YOUR print output
PERSON_ID   = 2     # ← from YOUR print output
```

### 4.5 Train — `notebooks/02_train.ipynb`

```python
!pip install -q rfdetr supervision
from google.colab import drive; drive.mount('/content/drive')
PROJECT = '/content/drive/MyDrive/warehouse-safety'
!cp -r "{PROJECT}/data/dataset" /content/dataset          # local disk = fast

from rfdetr import RFDETRBase
model = RFDETRBase()                     # start from COCO weights = transfer learning
model.train(
    dataset_dir='/content/dataset',
    epochs=25,
    batch_size=4,                        # T4-safe
    grad_accum_steps=4,                  # effective batch 16
    lr=1e-4,
    output_dir=f'{PROJECT}/models/rfdetr_v1',   # checkpoints straight to Drive
)
```

Notes:

- **Why pretrained + fine-tune works with only ~500 images:** the COCO weights already encode generic visual features; you're teaching one new class and adapting to your cameras, not learning vision from scratch. From scratch would need ~100× the data.
- **CUDA out-of-memory** → `batch_size=2, grad_accum_steps=8` (same effective batch).
- Small/distant people problem → add `RFDETRBase(resolution=728)` (must be divisible by 56; more compute).
- Validation mAP prints each epoch. **1.5–3 h on T4** for ~500 images.
- Best weights: `checkpoint_best_ema.pth` in the output dir (EMA = smoothed weights; use this one).

### 4.6 Verify the fine-tuned model

```python
from rfdetr import RFDETRBase
import cv2, glob, supervision as sv

model = RFDETRBase(pretrain_weights=f'{PROJECT}/models/rfdetr_v1/checkpoint_best_ema.pth')
model.optimize_for_inference()

CLASS_NAMES = {1: 'forklift', 2: 'person'}          # from YOUR 4.4 output
box_ann, lab_ann = sv.BoxAnnotator(), sv.LabelAnnotator()

for path in sorted(glob.glob('/content/dataset/valid/*.jpg'))[:20]:
    bgr = cv2.imread(path); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dets = model.predict(rgb, threshold=0.5)
    labels = [f"{CLASS_NAMES.get(c,'?')} {conf:.2f}"
              for c, conf in zip(dets.class_id, dets.confidence)]
    cv2.imwrite(f"{PROJECT}/outputs/val_{path.split('/')[-1]}",
                lab_ann.annotate(box_ann.annotate(bgr.copy(), dets), dets, labels))
```

**Important:** after fine-tuning, `class_id` indexes **your** dataset's categories (4.4), not COCO's 80.

**ACCEPTANCE CHECK:** final validation mAP50 ≥ **0.80 person**, ≥ **0.60 forklift** (PoC bar; 0.8+ forklift is normal with ~500 good images). If forklift is low: almost always a **data** problem — too few forklift instances, near-duplicate images, or label inconsistency. Inspect the annotated val images and fix data before touching any hyperparameter. If **person dropped** vs. the Phase-2 baseline, that's catastrophic forgetting → retrain with `lr=5e-5`.

---

## 5. Phase 4 — Tracking (Day 5, ~1 hour)

A detector is amnesiac — frame 100 doesn't know its forklift is frame 99's forklift. ByteTrack (via `supervision`) matches boxes across frames and assigns persistent integer IDs. No training.

```python
import supervision as sv

PROC_FPS = 10                                  # we process every 3rd frame of 30fps video
tracker  = sv.ByteTrack(frame_rate=PROC_FPS)   # MUST match the *processed* rate, not the video's

dets = model.predict(rgb, threshold=0.5)
dets = tracker.update_with_detections(dets)
# dets.xyxy (N,4) | dets.class_id (N,) | dets.confidence (N,) | dets.tracker_id (N,)
```

Two gotchas:

1. **`frame_rate` must equal the rate you feed frames at.** We'll process every 3rd frame (Section 10), so `frame_rate=10`. Mismatch → motion prediction misjudges gaps → constant ID switches.
2. The first ~3 processed frames may return **zero tracks** while ByteTrack confirms new objects. Normal; code must tolerate empty arrays.

**ACCEPTANCE CHECK:** run detector+tracker over a 10-second clip, drawing `tracker_id` on each box (use the Section-10 pipeline in detect-only mode). A person walking across the frame keeps **one** ID. Brief occlusion (behind a pillar) ideally resumes the same ID; a single switch after full occlusion is acceptable at PoC.

---

## 6. Phase 5 — Calibration: Pixels → Metres (Day 6, ~2 hours)

Unlocks Rule 3, Rule 4, and speed. Skipping it and using pixel distances is **wrong**: two boxes 200 px apart can be 1 m or 20 m apart depending on depth. Never use bounding-box overlap or pixel gaps as a proxy for distance.

### 6.1 Collect 4+ point correspondences (no GUI needed)

1. Pick one representative frame per camera: `data/calibration/cam1_ref.jpg`.
2. Choose ≥4 floor points visible in it whose real positions you can measure — walkway-marking corners are ideal. **Spread them across the whole area you care about**; homography accuracy degrades outside the calibrated region.
3. Get their pixel coordinates by opening the frame in any image viewer that shows cursor position (GIMP, Preview, Windows Photos), or run locally:
   ```python
   import cv2
   pts = []
   def click(e, x, y, *_):
       if e == cv2.EVENT_LBUTTONDOWN: pts.append((x, y)); print(x, y)
   img = cv2.imread('data/calibration/cam1_ref.jpg')
   cv2.imshow('click floor points, q to quit', img)
   cv2.setMouseCallback('click floor points, q to quit', click)
   cv2.waitKey(0)
   ```
4. Measure the same points in metres (tape measure or floor plan), any consistent origin/axes.

### 6.2 Camera config — `data/calibration/cam1.json`

One file per camera; the pipeline loads everything from here:

```json
{
  "camera_id": "cam1",
  "image_points": [[412, 880], [1180, 875], [1420, 620], [290, 625]],
  "floor_points": [[0.0, 0.0], [6.0, 0.0], [6.0, 8.0], [0.0, 8.0]],
  "walkways":    [[[0.0, 0.0], [2.0, 0.0], [2.0, 20.0], [0.0, 20.0]]],
  "vehicle_length_m": 2.7
}
```

`image_points[i]` **must** correspond to `floor_points[i]` — same order. Mixed-up ordering is the most common calibration failure and produces silently absurd distances. `walkways` = list of polygons in floor metres (for Rule 4). `vehicle_length_m`: measure a real J&J forklift; 2.7 m is a typical counterbalance body — **confirm with J&J**.

### 6.3 `src/geometry.py` (complete)

```python
import json, cv2, numpy as np
from matplotlib.path import Path

class CameraGeometry:
    def __init__(self, config_path):
        cfg = json.load(open(config_path))
        self.camera_id = cfg['camera_id']
        img = np.float32(cfg['image_points'])
        flr = np.float32(cfg['floor_points'])
        assert len(img) == len(flr) >= 4, "need >=4 matched point pairs"
        self.H, _ = cv2.findHomography(img, flr)
        self.walkways = [Path(p) for p in cfg.get('walkways', [])]
        self.vehicle_length_m = cfg.get('vehicle_length_m', 2.7)

    def to_floor(self, x, y):
        """Image point ON THE GROUND PLANE -> floor metres (x, y)."""
        out = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def floor_position(self, box):
        """box = (x1,y1,x2,y2). Uses BOTTOM-CENTRE — the ground-contact point.
        Using the box centre projects a point floating in mid-air: garbage."""
        x1, y1, x2, y2 = box
        return self.to_floor((x1 + x2) / 2.0, y2)

    def on_walkway(self, floor_xy):
        return any(w.contains_point(floor_xy) for w in self.walkways)

def floor_dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
```

### 6.4 Sanity check (mandatory)

```python
from src.geometry import CameraGeometry, floor_dist
g = CameraGeometry('data/calibration/cam1.json')
p1 = g.to_floor(412, 880)   # a point you measured
p2 = g.to_floor(1180, 875)  # another, known distance away
print(floor_dist(p1, p2))   # must be within ~10% of the tape-measured distance
```

**ACCEPTANCE CHECK:** computed distance within **±10%** of ground truth for at least two point pairs. Way off → point ordering mismatch (most likely) or heavy wide-angle lens distortion (undistort with `cv2.undistortPoints` first, or avoid fisheye cameras for the PoC).

---

## 7. Phase 6 — Pose Keypoints (Day 7, ~1 hour)

Rule 5 needs limb positions; Rule 1 needs wrist-to-head distance. **You do not train pose** — human anatomy is universal, so pretrained RTMPose works on warehouse workers out of the box. `rtmlib` runs it via ONNX with no heavyweight dependencies and auto-downloads weights on first use.

### 7.1 `src/pose_utils.py` (complete)

```python
import numpy as np
from rtmlib import Body

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_WRIST, R_WRIST, L_HIP, R_HIP = 9, 10, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16

_body = None
def get_pose_model(device='cuda'):          # device='cpu' if no GPU locally
    global _body
    if _body is None:
        _body = Body(mode='balanced', backend='onnxruntime', device=device)
    return _body

def run_pose(frame_bgr, device='cuda'):
    """NOTE: rtmlib takes the OpenCV BGR frame DIRECTLY (unlike RF-DETR, which needs RGB).
    Returns kpts: (N, 17, 3) = x, y, confidence per keypoint, N = people found."""
    keypoints, scores = get_pose_model(device)(frame_bgr)
    if keypoints is None or len(keypoints) == 0:
        return np.zeros((0, 17, 3), dtype=np.float32)
    return np.concatenate([np.asarray(keypoints),
                           np.asarray(scores)[..., None]], axis=-1).astype(np.float32)

def valid(kpt, thresh=0.5):
    """ALWAYS gate on confidence. Occluded joints (driver's legs behind the cab)
    come back with LOW confidence and a GUESSED position — trusting them means
    violations triggered by hallucinated limbs."""
    return kpt[2] > thresh

def match_pose_to_boxes(kpts_all, person_boxes):
    """Assign each pose to the person box containing its torso centre.
    Returns {box_index: (17,3) keypoints}."""
    def torso_centre(kp):
        pts = [kp[i][:2] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP) if valid(kp[i])]
        return np.mean(pts, axis=0) if pts else None
    def area(b): return max(1.0, (b[2]-b[0]) * (b[3]-b[1]))

    out = {}
    for kp in kpts_all:
        c = torso_centre(kp)
        if c is None:
            continue
        candidates = [i for i, b in enumerate(person_boxes)
                      if b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]]
        if candidates:
            best = min(candidates, key=lambda i: area(person_boxes[i]))  # smallest containing box
            if best not in out:
                out[best] = kp
    return out
```

Why matching is needed: rtmlib's `Body` finds people with its own internal detector, so its outputs must be re-associated with **your** tracked RF-DETR person boxes. Torso-centre-in-box with a smallest-box tie-break is simple and robust.

**ACCEPTANCE CHECK:** run `run_pose` on 5 frames with people and draw circles at each keypoint with `conf > 0.5`. Skeletons land on bodies; wrists on wrists. A driver's occluded lower body shows few/no valid leg keypoints — that is the confidence gate working, not a bug.

---

## 8. Phase 7 — The Rules — `src/rules.py` (complete, Day 8–11)

Everything below is geometry over tracked objects. All thresholds sit in `CFG` at the top — tuning (Section 11) means editing these numbers only.

```python
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np
from src.geometry import floor_dist
from src.pose_utils import (valid, NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER,
                            L_WRIST, R_WRIST)

CFG = {
    'PROC_FPS':              10,     # processed frames/sec (must match pipeline + ByteTrack)
    'MOVING_MS':             0.3,    # m/s above which a vehicle is "moving"
    'RECENT_MOVE_S':         5.0,    # a vehicle that moved in the last 5 s is still "working"
    'R3_VEHICLE_LENGTHS':    3.0,    # from the rule text
    'DRIVER_OVERLAP':        0.6,    # person-box fraction inside vehicle box to be driver-candidate
    'DRIVER_VEL_MATCH_MS':   0.5,    # velocity agreement (m/s) => moving together
    'R5_CAB_FRACTIONS':      (0.15, 0.35, 0.15, 0.0),  # cab inset: left, top, right, bottom
    'R5_MIN_S':              1.5,    # body-outside duration before violation
    'R4_MIN_S':              1.0,    # off-walkway duration before violation
    'R1_WRIST_HEAD_RATIO':   0.6,    # wrist-to-head dist / shoulder width
    'R1_MIN_S':              2.0,
    'KPT_CONF':              0.5,
}

@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    box: tuple                      # (x1, y1, x2, y2) pixels
    floor_xy: tuple                 # (x, y) metres
    keypoints: np.ndarray = None    # (17,3) for persons, else None

# ---------- generic helpers ----------

def dist2d(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def point_in_box(pt, box):
    x1, y1, x2, y2 = box
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

def overlap_ratio(inner, outer):
    """Fraction of `inner` box's area inside `outer` box."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a = max(1.0, (inner[2]-inner[0]) * (inner[3]-inner[1]))
    return inter / a

# ---------- motion state (shared by everything) ----------

class MotionState:
    """Rolling floor-position history per track -> velocity/speed.
    Feed REAL timestamps from the video (Section 10) — never frame_number/30:
    dropped frames silently corrupt every velocity otherwise."""
    def __init__(self):
        n = int(1.0 * CFG['PROC_FPS'])                 # ~1 s window
        self.hist = defaultdict(lambda: deque(maxlen=n))
        self.last_moving_t = {}

    def update(self, obj: TrackedObject, t: float):
        self.hist[obj.track_id].append((t, obj.floor_xy))
        s = self.speed(obj.track_id)
        if s is not None and s > CFG['MOVING_MS']:
            self.last_moving_t[obj.track_id] = t

    def velocity(self, tid):
        h = self.hist[tid]
        if len(h) < 4:
            return None
        (t0, p0), (t1, p1) = h[0], h[-1]
        dt = t1 - t0
        if dt <= 1e-3:
            return None
        return ((p1[0]-p0[0])/dt, (p1[1]-p0[1])/dt)

    def speed(self, tid):
        v = self.velocity(tid)
        return None if v is None else float(np.hypot(*v))

    def is_working(self, tid, t):
        s = self.speed(tid)
        if s is not None and s > CFG['MOVING_MS']:
            return True
        return (t - self.last_moving_t.get(tid, -1e9)) < CFG['RECENT_MOVE_S']

# ---------- driver association (feeds Rules 3 AND 5) ----------

def find_driver(vehicle, people, motion: MotionState):
    """The driver is the person who MOVES WITH the vehicle — not merely the one
    whose box overlaps it. Containment alone misfires: a pedestrian occluded
    BEHIND a forklift appears fully 'inside' its box."""
    v_vel = motion.velocity(vehicle.track_id)
    best, best_score = None, 0.0
    for p in people:
        if overlap_ratio(p.box, vehicle.box) < CFG['DRIVER_OVERLAP']:
            continue
        p_vel = motion.velocity(p.track_id)
        if v_vel is None or p_vel is None:
            score = 0.5                                # stationary: containment evidence only
        else:
            dv = np.hypot(v_vel[0]-p_vel[0], v_vel[1]-p_vel[1])
            score = 1.0 if dv < CFG['DRIVER_VEL_MATCH_MS'] else 0.0
        if score > best_score:
            best, best_score = p, score
    return best

# ---------- Rule 3: pedestrian near working vehicle ----------

def check_rule3(people, vehicles, driver_ids, motion, geom, t):
    """Metric distance on the FLOOR — never pixel gaps or box overlap.
    Known limitation: the rule's 'unless signaled for recognition' exception
    is not visually detectable; all proximity events are flagged for human review."""
    radius = CFG['R3_VEHICLE_LENGTHS'] * geom.vehicle_length_m
    out = []
    for v in vehicles:
        if not motion.is_working(v.track_id, t):
            continue
        for p in people:
            if p.track_id in driver_ids:
                continue                               # the driver is not a pedestrian
            d = floor_dist(p.floor_xy, v.floor_xy)
            if d < radius:
                out.append({'rule': 3, 'person_track': p.track_id,
                            'vehicle_track': v.track_id,
                            'distance_m': round(d, 2), 'threshold_m': round(radius, 2),
                            'vehicle_speed_ms': round(motion.speed(v.track_id) or 0.0, 2)})
    return out

# ---------- Rule 5: driver body outside vehicle ----------

class Rule5State:
    def __init__(self):
        self.frames_outside = defaultdict(int)

    @staticmethod
    def cab_region(vbox):
        """Cab approximated as an inset of the vehicle box (the raw box is mostly
        mast/forks — empty space — so raw-box containment MISSES real lean-outs).
        Tune fractions per camera against footage."""
        l, tp, r, b = CFG['R5_CAB_FRACTIONS']
        x1, y1, x2, y2 = vbox
        w, h = x2-x1, y2-y1
        return (x1 + l*w, y1 + tp*h, x2 - r*w, y2 - b*h)

    def check(self, driver, vehicle):
        if driver is None or driver.keypoints is None:
            return None
        cab = self.cab_region(vehicle.box)
        CHECK = (NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST)
        outside = any(valid(driver.keypoints[i], CFG['KPT_CONF'])
                      and not point_in_box(driver.keypoints[i][:2], cab)
                      for i in CHECK)
        tid = driver.track_id
        self.frames_outside[tid] = self.frames_outside[tid] + 1 if outside else 0
        if self.frames_outside[tid] >= int(CFG['R5_MIN_S'] * CFG['PROC_FPS']):
            return {'rule': 5, 'driver_track': tid,
                    'seconds_outside': round(self.frames_outside[tid] / CFG['PROC_FPS'], 1)}
        return None

# ---------- Rule 4: pedestrians off walkways ----------

class Rule4State:
    def __init__(self):
        self.frames_off = defaultdict(int)

    def check(self, people, driver_ids, geom):
        out = []
        for p in people:
            if p.track_id in driver_ids:
                continue
            off = not geom.on_walkway(p.floor_xy)
            self.frames_off[p.track_id] = self.frames_off[p.track_id] + 1 if off else 0
            if self.frames_off[p.track_id] >= int(CFG['R4_MIN_S'] * CFG['PROC_FPS']):
                out.append({'rule': 4, 'person_track': p.track_id,
                            'floor_xy': [round(c, 2) for c in p.floor_xy]})
        return out

# ---------- Rule 1: phone use (weakest rule — expect false positives) ----------

class Rule1State:
    def __init__(self):
        self.frames_raised = defaultdict(int)

    def check(self, people):
        out = []
        for p in people:
            kp = p.keypoints
            if kp is None or not (valid(kp[L_SHOULDER]) and valid(kp[R_SHOULDER])):
                continue
            shoulder_w = max(1.0, dist2d(kp[L_SHOULDER], kp[R_SHOULDER]))
            heads = [kp[i] for i in (NOSE, L_EAR, R_EAR) if valid(kp[i])]
            if not heads:
                continue
            raised = any(valid(kp[w]) and
                         min(dist2d(kp[w], h) for h in heads) / shoulder_w
                         < CFG['R1_WRIST_HEAD_RATIO']
                         for w in (L_WRIST, R_WRIST))
            self.frames_raised[p.track_id] = self.frames_raised[p.track_id] + 1 if raised else 0
            if self.frames_raised[p.track_id] >= int(CFG['R1_MIN_S'] * CFG['PROC_FPS']):
                out.append({'rule': 1, 'person_track': p.track_id})
        return out
```

**Rule 2** (daily pre-use inspection record) is **not a vision task** — a camera cannot see whether a checklist was completed. Descoped; say so explicitly to J&J.

---

## 9. Phase 8 — The Main Pipeline — `src/run_pipeline.py` (complete, Day 8–11)

Ties everything together: video in → annotated video + `events.jsonl` out. Runs on recorded clips (deterministic, replayable); pointing it at a live RTSP URL later is a one-line change and is J&J's integration concern, not yours.

```python
import argparse, json, os, cv2, numpy as np, supervision as sv
from rfdetr import RFDETRBase
from src.geometry import CameraGeometry
from src.pose_utils import run_pose, match_pose_to_boxes
from src.rules import (CFG, TrackedObject, MotionState, Rule1State, Rule4State,
                       Rule5State, find_driver, check_rule3)

PERSON_ID, FORKLIFT_ID = 2, 1        # ← from Section 4.4 — EDIT to your values

def main(video, calib, weights, outdir, device='cuda'):
    os.makedirs(f'{outdir}/events', exist_ok=True)
    os.makedirs(f'{outdir}/videos', exist_ok=True)

    geom  = CameraGeometry(calib)
    model = RFDETRBase(pretrain_weights=weights)
    model.optimize_for_inference()

    cap      = cv2.VideoCapture(video)
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride   = max(1, round(src_fps / CFG['PROC_FPS']))
    tracker  = sv.ByteTrack(frame_rate=CFG['PROC_FPS'])

    W, H = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(f'{outdir}/videos/annotated.mp4',
                             cv2.VideoWriter_fourcc(*'mp4v'), CFG['PROC_FPS'], (W, H))

    motion, r1, r4, r5 = MotionState(), Rule1State(), Rule4State(), Rule5State()
    box_ann, lab_ann = sv.BoxAnnotator(), sv.LabelAnnotator()
    events_path = f'{outdir}/events/events.jsonl'
    ev_file = open(events_path, 'a')
    frame_idx = ev_count = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride:
            continue
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0     # REAL timestamps, not frame/fps

        # 1. detect (RGB!) + track
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = tracker.update_with_detections(model.predict(rgb, threshold=0.5))
        if dets.tracker_id is None or len(dets) == 0:
            writer.write(bgr); continue

        # 2. pose (BGR!), matched back onto OUR person boxes
        person_boxes = [tuple(dets.xyxy[i]) for i in range(len(dets))
                        if dets.class_id[i] == PERSON_ID]
        pose_by_box = match_pose_to_boxes(run_pose(bgr, device), person_boxes)

        # 3. unify into TrackedObjects with floor positions
        people, vehicles, pi = [], [], 0
        for i in range(len(dets)):
            box = tuple(map(float, dets.xyxy[i]))
            obj = TrackedObject(int(dets.tracker_id[i]), int(dets.class_id[i]),
                                box, geom.floor_position(box))
            if obj.class_id == PERSON_ID:
                obj.keypoints = pose_by_box.get(pi); pi += 1
                people.append(obj)
            elif obj.class_id == FORKLIFT_ID:
                vehicles.append(obj)
            motion.update(obj, t)

        # 4. driver association, then rules
        driver_ids, events = set(), []
        for v in vehicles:
            d = find_driver(v, people, motion)
            if d:
                driver_ids.add(d.track_id)
                e5 = r5.check(d, v)
                if e5: events.append(e5)
        events += check_rule3(people, vehicles, driver_ids, motion, geom, t)
        events += r4.check(people, driver_ids, geom)
        events += r1.check(people)

        # 5. persist events (+ evidence frame) and annotate
        for e in events:
            e.update({'timestamp_s': round(t, 2), 'camera_id': geom.camera_id,
                      'video': os.path.basename(video)})
            ev_path = f'{outdir}/events/evt_{ev_count:05d}.jpg'
            cv2.imwrite(ev_path, bgr); e['evidence_frame'] = ev_path
            ev_file.write(json.dumps(e) + '\n'); ev_count += 1

        labels = [f"id{dets.tracker_id[i]} "
                  f"{'P' if dets.class_id[i]==PERSON_ID else 'F'}"
                  f"{' DRV' if int(dets.tracker_id[i]) in driver_ids else ''}"
                  for i in range(len(dets))]
        out = lab_ann.annotate(box_ann.annotate(bgr.copy(), dets), dets, labels)
        if events:
            cv2.putText(out, f"VIOLATION rule(s): {sorted({e['rule'] for e in events})}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        writer.write(out)

    cap.release(); writer.release(); ev_file.close()
    print(f"done: {ev_count} events -> {events_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--outdir', default='outputs')
    ap.add_argument('--device', default='cuda')
    main(**vars(ap.parse_args()))
```

Run (Colab or local-with-GPU):

```bash
python -m src.run_pipeline \
  --video data/raw_videos/test_clip.mp4 \
  --calib data/calibration/cam1.json \
  --weights models/rfdetr_v1/checkpoint_best_ema.pth
```

**ACCEPTANCE CHECK:** `outputs/videos/annotated.mp4` shows stable-ID boxes with the driver tagged `DRV`; `outputs/events/events.jsonl` contains one JSON per line. If the *driver* triggers Rule 3 against their own forklift, driver association failed — check that `DRIVER_OVERLAP` isn't too high and that the driver is being detected as a person at all (Section 4.1, convention 4).

---

## 10. Phase 9 — Validation (Day 12–13, non-negotiable)

You cannot measure a violation detector without violations, and no dataset of them exists — so **stage them**. In a controlled area, forklift parked or slow, safety supervisor present, record ~20–30 short clips with ground-truth times noted:

| Clip type | Tests |
|---|---|
| Person walks to marked 10 m / 8 m / 6 m / 3 m from a slowly moving forklift | Rule 3 fires below threshold, not above |
| Driver leans out: arm only / head only / torso | Rule 5 sensitivity |
| Driver turns head to look behind while reversing | Rule 5 must **NOT** fire — the make-or-break false-positive case |
| Person walks beside (off) the walkway | Rule 4 |
| Person holds phone to ear; person scratches head | Rule 1 + its false-positive twin |
| Plain safe operation | Nothing fires at all |

Score **per event** (not per frame): TP = real violation flagged, FP = flag with no violation, FN = violation missed. Precision = TP/(TP+FP); Recall = TP/(TP+FN). Match events to ground truth by time-window overlap (an event within the noted violation interval = TP).

**PoC targets: precision ≥ 0.8 on Rules 3 and 5.** Precision beats recall at this stage — a system that cries wolf gets muted within a week and is then worse than nothing. Nearly all tuning = editing `CFG` numbers (thresholds and `*_MIN_S` durations) and re-running clips.

---

## 11. Timeline and Credit Budget

5 people, part-time; days = working days.

| Days | Phase | Credits |
|---|---|---|
| 1 | Setup, footage, frame extraction | 0 |
| 1–2 | Zero-shot baseline | ~1 |
| 2–5 | Label ~500 images (all hands), fine-tune RF-DETR | 4–8 |
| 5 | Tracking | ~1 |
| 6 | Calibration | 0 |
| 7 | Pose | ~1 |
| 8–9 | Rule 3 + driver association | ~1 |
| 10–11 | Rule 5, then Rules 4 & 1 | ~1 |
| 12–13 | Staged validation, tune CFG | ~2 |
| 14 | Demo video, write-up | ~1 |

**Total ≈ 12–16 of your 50 credits**, leaving headroom for 2–3 re-training runs.

**Build order (each step verifiable before the next depends on it):** detector → tracker → homography → **Rule 3** → pose → driver association → Rule 5 → Rules 4/1. **If time runs short, stop after Rule 3** — one rule with correct metric distance beats four half-working ones.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| CUDA out of memory during training | T4's 16 GB exceeded | `batch_size=2, grad_accum_steps=8`; keep `resolution` at default |
| Detections mysteriously poor at inference | **BGR fed to RF-DETR** | `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before every `predict` |
| Wrong/blank class names after fine-tune | Hardcoded COCO ids | Re-read ids from `_annotations.coco.json` (§4.4); edit `PERSON_ID`/`FORKLIFT_ID` |
| Great val mAP, terrible on new video | Frame-level train/val leak | Re-split **by source video** (§4.2); metrics were fiction |
| Constant ID switches | ByteTrack `frame_rate` ≠ processed rate | Set both to `CFG['PROC_FPS']` |
| No `tracker_id` early in clip | Track confirmation lag | Normal for first ~3 processed frames; code already tolerates it |
| Absurd distances (100s of metres) | Calibration point-order mismatch | `image_points[i]` must pair with `floor_points[i]`; redo §6.4 check |
| Distances drift near frame edges | Outside calibrated region / lens distortion | Spread calibration points wider; undistort first |
| Driver flagged by Rule 3 | Driver association failing | Driver labeled as person? Lower `DRIVER_OVERLAP`; check velocities exist |
| Rule 5 fires on reversing driver | Head-turn counted as violation | Remove `NOSE` from `CHECK` set, or widen cab top inset — and ask J&J for the ruling |
| Training crawls on Colab | Reading data through Drive mount | Copy dataset to `/content` first (§1.4) |
| rtmlib download fails | Restricted network | Run once on open network; weights cache in `~/.cache` |

---

## 13. Known Limitations (state these openly — it adds credibility)

1. **Rule 3's "unless signaled for recognition" exception is not visually detectable.** All proximity events are flagged; humans dismiss acknowledged ones.
2. **Homography assumes a flat floor** — invalid on ramps; calibrate per zone and document valid regions.
3. **Single camera, no occlusion recovery** — people behind racking are invisible; multi-camera fusion is out of scope.
4. **Rule 1 confuses phones with radios and head-scratching** — conservative thresholds; the detector's phone class could boost it later.
5. **Cab region is a tuned approximation**, per-camera.
6. **Validation is staged, not real incidents** — real-world numbers may differ; request incident footage from J&J.

## 14. Week-1 Questions for J&J (send before labeling starts)

1. Do pallet jacks / tuggers count as Rule-3 "vehicles"?
2. Actual forklift length (sets the 3-vehicle-length radius)?
3. Rule 5: does a head-turn while reversing count as a violation?
4. Do safety training videos with staged violations exist? May we use them?
5. Floor plans / measurable dimensions for calibration?
6. Target precision/recall — what false-alarm rate will the safety team tolerate?
7. Confirm Rule 2 descope (not a vision task).
8. Restrictions on retaining/sharing footage with identifiable workers?
