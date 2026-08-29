# Integration guide

For teammates adding a detection module to this pipeline. Answers two questions
that came up:

1. [Which datasets were used for fine-tuning?](#1-exactly-which-data-the-model-was-trained-on)
2. [How do I plug my module in?](#3-two-ways-to-integrate)

---

## 1. Exactly which data the model was trained on

The shipped model is `models/rfdetr_real/checkpoint_best_ema.pth`
(mAP50:95 **0.820**). It was trained on `data/dataset_v3`, which is three sources
merged:

| Source | Where it came from | Train | Valid |
|---|---|---|---|
| `forklift2-z6zww_v5` | [universe.roboflow.com/pdf-ih16p/forklift2-z6zww/dataset/5](https://universe.roboflow.com/pdf-ih16p/forklift2-z6zww/dataset/5) | 3192 | 275 |
| `forklift-and-human_v2` | [universe.roboflow.com/hitsz/forklift-and-human/dataset/2](https://universe.roboflow.com/hitsz/forklift-and-human/dataset/2) | 1361 | 392 |
| SDG-Warehouse (synthetic) | [nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes) | 1333 | 720 |
| **Total** | | **5886** | **1387** |

Boxes: 5393 forklift + 4892 person (train), 1437 + 1136 (valid). 4553 of the
training images are real photographs; the rest are simulator renders.

**Licences** — all commercial-use-safe, which is a hard project requirement since
J&J deploys this. Both Roboflow sets are **CC BY 4.0** (attribution required);
SDG-Warehouse is **OpenMDW 1.1**. Provenance is recorded in
`data/real/*/LICENCE.txt` next to each download.

### A third dataset was downloaded and deliberately excluded

[`paft/forklift-model`](https://universe.roboflow.com/paft/forklift-model) is on
disk but **not** in `dataset_v3`. It has 135 person boxes across 8076 warehouse
images, and `scripts/audit_labels.py` found visible workers — including a seated
forklift driver — all unlabeled. In detection training an unlabeled object is
supervised as *background*, so including it would teach the model that a seated
driver is not a person. Do not add it back without relabelling.

### Reproducing the dataset

```bash
export ROBOFLOW_API_KEY=<your key>          # app.roboflow.com -> Settings -> API
python -m scripts.fetch_roboflow --url https://universe.roboflow.com/pdf-ih16p/forklift2-z6zww/dataset/5
python -m scripts.fetch_roboflow --url https://universe.roboflow.com/hitsz/forklift-and-human/dataset/2
python -m scripts.fetch_sdg --scenario nearmiss --shards 1 2 3 4
python -m scripts.sdg_to_coco --root data/sdg --out data/dataset_v2
python -m scripts.real_to_coco --root data/real/forklift2-z6zww_v5 data/real/forklift-and-human_v2 \
       --merge-with data/dataset_v2 --out data/dataset_v3
python -m scripts.train_detector --dataset-dir data/dataset_v3 --output-dir models/rfdetr_real --epochs 15
```

Roughly 4 hours on an RTX 3070. `data/` and `models/` are gitignored — the repo
holds code, not gigabytes.

### One caveat you should know before trusting the model

It detects forklifts well and **cannot detect seated drivers at all** (0 of 16 on
clean test images). The training data labels standing pedestrians but not drivers.
If your module depends on finding the person *operating* a vehicle, it will get
nothing. See RESULTS.md §0.

---

## 2. How the pipeline is put together

One pass per processed frame, in `src/run_pipeline.py`:

```
video frame
   ↓  detector          RF-DETR → boxes + class ids            (src/detector.py)
   ↓  tracker           ByteTrack → persistent track_id
   ↓  pose  [optional]  RTMPose → 17 keypoints per person      (src/pose_utils.py)
   ↓  geometry          bottom-centre of box → floor metres    (src/geometry.py)
   ↓  motion            floor positions over time → velocity   (MotionState)
   ↓  RULES  ← your module goes here                           (src/rules.py)
   ↓  aggregation       per-frame hits → one event per episode (src/events.py)
events.jsonl + annotated video
```

We process **every 3rd frame** (30 fps source → 10 fps), set by `CFG['PROC_FPS']`.

### The one object you need to understand

Every rule receives lists of `TrackedObject` ([src/rules.py:52](src/rules.py)):

```python
@dataclass
class TrackedObject:
    track_id: int        # persistent across frames — this is what makes speed measurable
    class_id: int
    box: tuple           # (x1, y1, x2, y2) in PIXELS
    floor_xy: tuple      # (x, y) in METRES on the warehouse floor
    keypoints: np.ndarray = None   # (17, 3) = x, y, confidence — persons only, None if pose is off
    worker_detections: dict = field(default_factory=dict)  # {'phone': [...], 'helmet': [...], 'vest': [...]}
```

`floor_xy` is the important one. It is the box's bottom-centre projected through
the camera homography, so it is a real position on the floor plane in metres.
**Never compute distance or speed from pixels** — two boxes 200 px apart can be
1 m or 20 m apart depending on depth.

---

## 3. Two ways to integrate

### Option A — downstream of `events.jsonl` (loose coupling)

If your module can work from our output, just read the file. No code changes, no
merge conflicts, and you can develop independently.

```python
import json
events = [json.loads(l) for l in open('outputs/events/events.jsonl')]
```

Best when your logic is about reporting, aggregation, or dashboards.

### Option B — a rule inside the pipeline (tight coupling)

Pick this if your logic needs per-frame state — which speed detection does. Four
small edits, described below.

---

## 4. Worked example: adding a speed-limit rule

Assume your module flags a vehicle exceeding a speed limit. This is the exact
shape a new rule takes.

### Heads-up: speed already exists

`MotionState` ([src/rules.py:94](src/rules.py)) already tracks floor positions per
`track_id` and derives velocity:

```python
motion.velocity(track_id)   # (vx, vy) in metres/second, or None if <4 samples yet
motion.speed(track_id)      # scalar m/s, or None
motion.is_working(track_id, t)   # moving now, or moved within the last 5 s
```

These are **real metres per second**, because they are computed from `floor_xy`
and real frame timestamps. Before porting your own speed code, check whether you
can just consume `motion.speed()` — if so your rule is about ten lines. If your
module measures something ours does not (wheel rotation, acceleration, per-axis
velocity), keep yours and feed it the same `floor_xy` history.

Two things `MotionState` gets right that are easy to get wrong alone:

- it uses **real frame timestamps** (`cap.get(cv2.CAP_PROP_POS_MSEC)`), never
  `frame_number / fps` — videos drop frames silently, and assuming a fixed rate
  corrupts every velocity
- it returns **`None` rather than `0`** when there is not enough history, so
  "unknown" and "stopped" stay distinguishable

### Step 1 — write the rule in `src/rules.py`

Follow the shape of `Rule4State`. Every rule needs a **duration gate**: single-frame
decisions flicker constantly from occlusion and detection noise.

```python
# add to CFG at the top of src/rules.py
CFG = {
    ...
    'R6_SPEED_LIMIT_MS': 2.0,   # m/s; ~7 km/h. Confirm the real limit with J&J
    'R6_MIN_S':          1.0,   # must be sustained this long before reporting
}


class Rule6State:
    """Vehicle exceeding the site speed limit.

    Speed comes from MotionState, so it is floor metres/second rather than pixels
    per frame, and is therefore comparable to a posted limit.
    """

    def __init__(self):
        self.frames_over = defaultdict(int)

    def check(self, vehicles, motion):
        out = []
        for v in vehicles:
            speed = motion.speed(v.track_id)
            if speed is None:            # not enough history yet — not the same as stopped
                continue
            over = speed > CFG['R6_SPEED_LIMIT_MS']
            self.frames_over[v.track_id] = self.frames_over[v.track_id] + 1 if over else 0
            if self.frames_over[v.track_id] >= int(CFG['R6_MIN_S'] * CFG['PROC_FPS']):
                out.append({
                    'rule': 6,
                    'vehicle_track': v.track_id,
                    'speed_ms': round(speed, 2),
                    'limit_ms': CFG['R6_SPEED_LIMIT_MS'],
                })
        return out
```

Return a **list of dicts**. Each must have `'rule'` and enough identity fields to
tell one violation apart from another. Include the measurement *and* the threshold
— every event should explain itself without a lookup.

### Step 2 — register the rule in `src/events.py`

Otherwise every frame becomes its own event instead of one episode:

```python
KEY_FIELDS = {1: ('person_track',),
              3: ('person_track', 'vehicle_track'),
              4: ('person_track',),
              5: ('driver_track',),
              6: ('vehicle_track',)}      # <-- what identifies YOUR episode

SEVERITY = {3: ('distance_m', min),
            5: ('seconds_outside', max),
            6: ('speed_ms', max)}         # <-- peak severity; the evidence frame is taken here
```

`KEY_FIELDS` decides what counts as "the same ongoing violation". `SEVERITY` picks
which frame is worth saving — for speed, the fastest moment.

### Step 3 — call it in `src/run_pipeline.py`

Around [line 150](src/run_pipeline.py), where the other rules run:

```python
from src.rules import (..., Rule6State)

motion, r1, r4, r5 = MotionState(), Rule1State(), Rule4State(), Rule5State()
r6 = Rule6State()                                   # <-- construct once, outside the loop

# ... inside the frame loop, after driver association:
hits += check_rule3(people, vehicles, driver_ids, motion, geom, t)
hits += r4.check(people, driver_ids, geom)
hits += r1.check(people)
hits += r6.check(vehicles, motion)                  # <-- yours
```

Construct state objects **outside** the loop — they hold the per-track frame
counters that make duration gates work.

### Step 4 — test it

Rules are pure functions over `TrackedObject`, so they test without video, models
or a GPU. Copy the pattern in `tests/test_rules.py`:

```python
def test_rule6_fires_only_above_the_limit_and_only_when_sustained():
    m = MotionState()
    v = TrackedObject(1, 1, (0, 0, 10, 10), (0.0, 0.0))
    for i in range(10):                      # 3 m/s along +x, sampled at 10 Hz
        v.floor_xy = (i * 0.3, 0.0)
        m.update(v, i * 0.1)

    r6 = Rule6State()
    need = int(CFG['R6_MIN_S'] * CFG['PROC_FPS'])
    for _ in range(need - 1):
        assert r6.check([v], m) == []        # duration gate not yet satisfied
    assert len(r6.check([v], m)) == 1
```

Always test the **negative** case too — a vehicle under the limit must produce
nothing. Precision is the metric this project is graded on: a system that cries
wolf gets muted within a week, and a muted system has zero recall on everything.

Run `python -m pytest -q` — 91 tests, about 6 seconds, no GPU needed.

### Step 5 — see it end to end

```bash
python -m scripts.make_synthetic_clip     # scene with exact known ground truth
python -m src.run_pipeline \
  --video outputs/synthetic/synthetic_cam1.mp4 \
  --calib data/calibration/synthetic_cam1.json \
  --weights models/rfdetr_real/checkpoint_best_ema.pth
```

In that scene the forklift travels at a known 0.9 m/s, so you can check your rule
against arithmetic rather than eyeballing it.

---

## 5. Gotchas that will cost you a day

| Problem | Why | Fix |
|---|---|---|
| Detector returns nothing useful | Fine-tuned RF-DETR predicts **0-based contiguous** ids, not your COCO category ids. Ours: `forklift=0, person=1` | `from src.detector import model_class_ids` — never hardcode |
| Accuracy quietly poor | OpenCV loads **BGR**, RF-DETR wants **RGB**. It does not crash, it just degrades | `RFDetrDetector` converts internally. `rtmlib` takes BGR directly |
| Track IDs keep changing | ByteTrack `frame_rate` must match the rate frames are *fed*, not the video's native rate | both are `CFG['PROC_FPS']` |
| Velocities are wrong | `frame_number / fps` assumes no dropped frames | use the real timestamp, as `MotionState` does |
| Distances absurd (100s of metres) | `image_points[i]` must pair with `floor_points[i]` in the calibration | re-run `python -m src.geometry <calib> --pair ...` |
| Rule fires every frame | Rule not registered in `KEY_FIELDS` | add it (step 2) |
| Keypoints in impossible places | Occluded joints return **invented** coordinates with low confidence | gate with `valid(kp[i])` before using any keypoint |

**Do not add Ultralytics YOLO.** It is AGPL-3.0, which is incompatible with a
commercial handoff to J&J. Every current dependency is Apache-2.0 or MIT. YOLO
*format* files are fine — the licence attaches to their code, not the format.

---

## 6. Working together

- Rules are self-contained, so two people editing different rules rarely conflict.
  The shared files are `run_pipeline.py` (one line each) and `events.py` (two).
- Branch per module (`feature/speed-rule`), and run `pytest` before pushing.
- Tune behaviour by editing `CFG`, never by retraining. Thresholds are meant to
  move; the model is not.
- Reserve `rule` numbers so we do not collide: 1–5 are J&J's Life-Saving Rules,
  so start additions at **6**.

Questions worth asking before you build: does your module need per-frame state
(Option B) or can it read `events.jsonl` (Option A)? And does it need real-world
metres — because if so, the camera needs a calibration, which takes about ten
minutes per camera and is described in [DEMO.md](DEMO.md) §4.
