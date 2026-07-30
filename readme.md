# Warehouse Safety CV — J&J Capstone

Computer vision detection of J&J Life-Saving Rule violations from warehouse
camera footage. We deliver **models and detection logic that emit structured
violation events**; J&J's engineer owns deployment, alerting, and dashboards.

**Read [`context.md`](context.md) first** (what and why), then
[`rule3-rule5-prototype-implementation.md`](rule3-rule5-prototype-implementation.md)
(how). This file is just how to run the code.

## The core idea

We do **not** train a model to recognise "unsafe behaviour" — no such dataset
exists. We train exactly one thing (a detector for `person` and `forklift`),
download two (pose, tracking), and express the rules as **geometry over tracked
positions**:

- **Rule 3** — is the floor distance between a pedestrian and a *working* vehicle
  under 3 vehicle lengths?
- **Rule 5** — are the driver's keypoints outside the cab region for >1.5 s?
- **Rule 4** — is a pedestrian's floor position outside the walkway polygons?
- **Rule 1** — is a wrist sustained near the head?
- **Rule 2** — descoped. A camera cannot see whether a checklist was completed.

Distances are measured in **floor metres via homography, never in pixels**. Two
boxes 200 px apart can be 1 m or 20 m apart depending on depth — this is the most
common way projects like this fail.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
# CUDA build of torch (skip on Colab, where torch is preinstalled):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify: `python -m pytest -q` → 67 tests, no network or GPU needed.

## Synthetic data: NVIDIA PhysicalAI SDG-Warehouse

Real warehouse CCTV of people near forklifts is effectively unobtainable, so the
detector's training data and Rule 3's validation set come from NVIDIA's
[SDG-Warehouse](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes)
simulation (OpenMDW 1.1 — **commercial use permitted**, so the licensing posture
holds).

```bash
python -m scripts.fetch_sdg --shards 1 2 3 4   # ~270 MiB/shard, 3 runs each
python -m scripts.sdg_calibration              # exact calibration + Rule 3 ground truth
python -m scripts.sdg_to_coco                  # -> data/dataset/, split by run
python -m scripts.sdg_validate_rules           # score rules with a perfect detector
```

We read the **`artifacts/` tier, not `rgb/`**: it ships `rgb.mp4` alongside the
annotations, so pixels and boxes provably come from the same render (verified
pixel-exact). The `rgb/` tier is Cosmos world-model output whose alignment with
these boxes is unverified. Only `ceiling_*` cameras are kept — the elevated
CCTV-like viewpoint that matches where J&J's cameras are. The tars are read with
HTTP random access, so we pull 272 MiB per shard instead of 4.6 GiB.

### What this buys us

**Calibration without a tape measure, and a way to grade our own homography.**
The simulator's camera matrices project known floor points exactly, so §6.1's
manual clicking-and-measuring is replaced by generated correspondences — and we
can compare `CameraGeometry.to_floor()` against exact truth. Measured error:
**0.0001 m mean, 0.0001 m worst across 44 clips.** Nothing else in the project
can check this; on real footage there is no truth to compare against.

**Rule 3 measured with a perfect detector.** Feeding the simulator's exact boxes
through the real pipeline isolates rule logic from detector error, setting the
ceiling on achievable accuracy:

| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Rule 3, 44 clips | 44 | 0 | 0 | **1.000** | **1.000** |

So the geometry, driver association, duration gates, and event aggregation are
sound. All remaining risk sits in detection quality and domain shift.

**560 labeled images** (`data/dataset/`, 8 train runs / 4 valid runs, split by
run per §4.2), replacing most of Phase 3's labeling effort.

### What it does not cover — read before trusting it

1. **The vehicles are stand-on reach trucks with no enclosed cab**, not sit-down
   counterbalance forklifts. So **Rule 5 cannot be exercised at all** here, and
   the vehicle type sits uncomfortably close to the pallet-jack family J&J just
   ruled out of scope. **Worth confirming with J&J.**
2. **No annotated driver.** Every `character` is a pedestrian, so this data
   cannot validate driver association — which is load-bearing for Rules 3 and 5.
3. **No walkways or phone use**, so Rules 4 and 1 are unexercised.
4. **No negative clips**: every near-miss clip contains a genuine violation.
   Precision is still measured (events outside a ground-truth interval count as
   FP), but for true negatives fetch the routine-operations scenario by setting
   `SCENARIO = 'warehouse_box_pickup'` in `scripts/fetch_sdg.py`.
5. **Domain shift is real** — CG humans, no hi-vis PPE, simulated lighting. Treat
   this as pretraining to be fine-tuned on real J&J footage, not as a substitute.

A dataset defect worth knowing: `metro_agent_data.world_position` is **frozen at
its initial value in ~40% of runs** while the boxes move correctly. We read
positions from `bounding_box_3d_fast.transform` instead, which is correct in
every run checked and agrees exactly where both work. Trusting `world_position`
silently marks those runs "vehicle never moved" and turns every real violation
into a scored false positive.

## Status

| Phase | State |
|---|---|
| Scaffold, geometry, tracking, all 4 rules, pipeline, event output | **Done, tested** |
| Validation scorer (§10 precision/recall) | **Done, tested** |
| Training data — 560 labeled images from SDG-Warehouse | **Done** |
| Rule 3 validated with a perfect detector (P=1.00, R=1.00) | **Done** |
| Fine-tune RF-DETR on the synthetic set | **Ready to run** (local 3070 or Colab T4) |
| Rules 5/4/1 validation, real footage, domain-shift fine-tune | **Blocked on J&J data** |

The rule logic is verified end-to-end against a synthetic clip with exact
arithmetic ground truth (`scripts/make_synthetic_clip.py`). **That is a
correctness test, not an accuracy measurement** — reported accuracy comes only
from staged footage per §10.

## Running it

```bash
# 1. Frames for labeling (after placing clips in data/raw_videos/)
python -m src.extract_frames

# 2. Class IDs — read from the dataset, never hardcoded
python -m scripts.discover_class_ids data/dataset

# 3. Calibrate a camera, then VALIDATE it (mandatory, §6.4)
python -m scripts.pick_calibration_points data/calibration/cam1_ref.jpg --camera-id cam1
python -m src.geometry data/calibration/cam1.json --pair 412 880 1180 875 6.0

# 4. Run the pipeline -> annotated video + events.jsonl
python -m src.run_pipeline \
  --video data/raw_videos/test_clip.mp4 \
  --calib data/calibration/cam1.json \
  --weights models/rfdetr_v1/checkpoint_best_ema.pth \
  --person-id 2 --forklift-id 1

# 5. Score against staged-clip ground truth
python -m scripts.score_events --events outputs/events/events.jsonl \
                               --truth data/validation/ground_truth.json
```

Try the whole pipeline right now, with no weights and no footage:

```bash
python -m scripts.make_synthetic_clip
python -m pytest tests/test_pipeline_synthetic.py -q
```

## Layout

```
src/
  extract_frames.py   frame sampling for labeling
  geometry.py         homography: pixels -> floor metres  (CameraGeometry)
  detector.py         RF-DETR wrapper + StubDetector for testing
  pose_utils.py       RTMPose via rtmlib; COCO-17 keypoints
  rules.py            all four rules + driver association; thresholds in CFG
  events.py           per-frame hits -> one event per episode
  run_pipeline.py     video in -> annotated video + events.jsonl
scripts/
  make_synthetic_clip.py     synthetic scene with exact ground truth
  discover_class_ids.py      read class IDs from COCO JSON
  pick_calibration_points.py click floor points -> calibration JSON
  score_events.py            §10 precision/recall scorer
notebooks/  01_baseline (zero-shot), 02_train (fine-tune)  — Colab, T4 only
tests/      67 tests
```

**All tuning is editing `CFG` in [`src/rules.py`](src/rules.py)** — thresholds and
duration gates. Never retrain to change rule behaviour.

## Licensing

Every component is permissively licensed, which is a hard requirement for
commercial handoff: RF-DETR (Apache-2.0), supervision (MIT), trackers
(Apache-2.0), rtmlib/RTMPose (Apache-2.0), OpenCV (Apache-2.0).

**Do not add Ultralytics YOLO.** It is AGPL-3.0, whose copyleft obligations are
incompatible with closed-source commercial deployment — not even for "quick
testing", because prototype code becomes deliverable code.

## Deviations from the implementation guide

Three, all deliberate and documented in the relevant module:

1. **`src/events.py` is new.** The guide emits one JSONL row and one JPEG per
   *frame*; §10 scores per *event*. Without grouping, precision measures frames,
   not violations.
2. **`trackers.ByteTrackTracker` replaces `sv.ByteTrack`**, which supervision
   deprecated in 0.28 and removes in 0.30. Same algorithm, still Apache-2.0.
3. **Two `rfdetr` renames**: `rfdetr.util.coco_classes` →
   `rfdetr.assets.coco_classes`, and `optimize_for_inference()` → `inference()`.
   `RFDETRBase` still works but is removed in rfdetr 2.0.0, hence the version pin.

## Rulings from J&J (2026-07-29)

Two of `context.md` §9's blocking questions are now answered. **Both override the
guide**, which assumed the opposite in each case.

**Q1 — Pallet jacks and tuggers are OUT OF SCOPE.** They do not count as Rule-3
vehicles. This reverses guide §4.1 convention 1 ("until answered, label them
`forklift`"). Labeling convention is now:

> Label `forklift` only for sit-down / stand-on powered forklifts. Do **not**
> label pallet jacks, tuggers, or hand trucks at all — not as `forklift`, not as
> a separate class. Leave them unlabeled background.

**Q3 — A head-turn while reversing IS a Rule 5 violation.** The head is an
important body part and must stay inside the forklift at all times; drivers can
see behind them from inside the cab. Consequences:

- `NOSE` **stays** in `R5_CHECK_KEYPOINTS` ([src/rules.py](src/rules.py)).
- Guide §12's troubleshooting row — *"Rule 5 fires on reversing driver → remove
  `NOSE` from the CHECK set"* — is now **wrong**. That is the specified behaviour,
  not a bug.
- The staged reversing head-turn clip flips from the set's most important
  *negative* to a *positive*, with a new paired negative (head stays inside).
  Both are in `data/validation/ground_truth.example.json`.

Still open, and now the main blocker: **actual forklift length** (sets the
3-vehicle-length radius; 2.7 m is a placeholder), plus the footage questions.

## Privacy

Workforce monitoring, treated as sensitive: no biometric identity storage, track
IDs are ephemeral per-video integers, faces blurred in any evidence frames shared
outside the team. Footage and weights are gitignored — they live in Drive.
