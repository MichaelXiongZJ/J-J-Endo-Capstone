# Project Context

**Read this before `rule3-rule5-prototype-implementation.md`.** This document explains *what* the project is and *why* every major decision was made. The implementation guide explains *how* to build it. If you find yourself wanting to change an approach, check Section 7 first — most alternatives were already considered and rejected for reasons that are not obvious from the code.

---

## 1. What This Project Is

A university capstone project sponsored by **Johnson & Johnson (J&J)**. A team of five engineering students builds computer vision models that detect violations of J&J's workplace safety rules from warehouse camera footage, with the goal of reducing accidents involving forklifts and pedestrians.

**This is not a research project.** The goal is a working, demonstrable prototype with honest measured accuracy — not a novel technique or a state-of-the-art benchmark result.

### Who does what

| Party | Responsibility |
|---|---|
| **Student team (us)** | Trained models, per-rule detection logic, evaluation, documentation, a violation-event output format |
| **J&J's engineer (not us)** | Camera deployment, edge hardware, alerting, dashboards, production integration |

This division was set by J&J and it matters enormously. **We deliver models and detection logic that emit structured violation events. We do not build a deployed system.** Do not spend effort on dashboards, alert routing, live-stream infrastructure, or multi-camera fusion — someone else owns those, and building them wastes the limited time available.

---

## 2. The Five Life-Saving Rules

J&J's existing internal safety rules, given to us verbatim as the specification:

1. No mobile phone / distracting device
2. All vehicle drivers have daily pre-use inspection record on vehicle
3. No pedestrian within 3 vehicle lengths of working vehicles unless signaled for recognition
4. Pedestrians keep to walkways and marked crossings
5. Vehicle drivers keep body inside vehicle at all times

### How each maps to computer vision

| Rule | Detection approach | Priority |
|---|---|---|
| **3** — pedestrian too close to working vehicle | Detect + track people and forklifts; convert positions to real-world floor metres; measure distance; "working" = vehicle is moving | **Primary target** |
| **5** — driver's body outside vehicle | Identify which person is the driver; run pose estimation; check whether keypoints fall outside the cab region | **Primary target** |
| **4** — pedestrians keep to walkways | Person floor position tested against walkway polygons | Cheap add-on (reuses Rule 3's calibration) |
| **1** — no phone use | Pose: wrist near head, sustained | Cheap add-on (reuses Rule 5's pose model); **lowest accuracy** |
| **2** — pre-use inspection record | **Not a computer vision task.** Whether a paper/digital checklist was completed is invisible to a camera. | **Descoped** |

Rule 2 must be explicitly descoped in conversation with J&J. Do not attempt to build it. If pressed, the only vaguely visual reinterpretation is action-recognition of a walk-around inspection being performed, which is hard, ambiguous, and not what the rule actually says.

### Rule 3 has an exception we cannot detect

The rule says "**unless signaled for recognition**" — meaning it is acceptable for a pedestrian to be close if the driver has acknowledged them. Detecting that acknowledgment visually (eye contact, hand signal) is beyond this project's scope. **The system flags all proximity events and leaves the exception to human review.** This is a documented limitation, not a bug.

---

## 3. Constraints

These shape every decision and cannot be traded away:

- **Time:** 8 weeks total, **part-time** (students have other coursework). The prototype in the implementation guide is scoped at ~14 working days of effort.
- **Team:** 5 students. Only some have CV experience; assume beginners.
- **Compute:** Google Colab with **~50 credits remaining**. Use the **T4** GPU only (≈2 credits/hour). Never A100 (≈12/hour). Total prototype budget: 12–16 credits.
- **Data:** No existing dataset of safety violations. None will appear. Validation footage must be **staged by the team** (see Section 6).
- **Licensing:** The deliverable goes to a large corporation for commercial use. **Every component must be permissively licensed** (Apache-2.0 / MIT). This is a hard requirement, not a preference.

---

## 4. The Central Insight

**We are not training a model to recognize "unsafe behavior."** That is the intuitive approach and it is wrong here, because it would require thousands of labeled examples of violations, which do not exist and cannot be cheaply created.

Instead:

```
Train ONE thing:     a detector that draws boxes around people and forklifts
Download TWO things: a pose model (body keypoints) and a tracker (persistent IDs)
Write the rules as:  GEOMETRY over tracked positions
```

Concretely:
- Rule 3 is "is the distance between these two floor positions less than 8 metres, while the vehicle is moving?"
- Rule 5 is "are the driver's keypoints outside the cab region for more than 1.5 seconds?"

Neither is machine learning. Both are arithmetic on top of detections.

**Consequences of this design, which explain the whole project's feasibility:**
- Only ~500 labeled images are needed (not thousands), because only the detector is trained.
- Adding rules is cheap — the shared perception layer is reused.
- Behavior is tunable by editing threshold numbers, not by retraining.
- The system's decisions are explainable to a safety team ("flagged because distance was 5.2 m, threshold 8.1 m"), which matters for adoption.

---

## 5. Glossary

Assume no prior CV knowledge. Terms used throughout both documents:

**Bounding box (bbox)** — A rectangle around a detected object, given as `(x1, y1, x2, y2)` pixel corners.

**Object detection** — Finding and classifying objects in an image, outputting boxes + class labels + confidence scores.

**Class / class ID** — The category of a detected object (`person`, `forklift`), represented internally as an integer. **These integers differ between datasets** and must always be read from the dataset file, never assumed.

**Confidence / threshold** — Every detection has a 0–1 certainty score. A threshold (e.g. 0.5) filters weak detections. Raising it increases precision, lowers recall.

**COCO** — A large public dataset of 80 everyday object classes, and also the JSON annotation format. Models "pretrained on COCO" already know `person` (and `cell phone`) but **not** `forklift`.

**Transfer learning / fine-tuning** — Starting from a model pretrained on a large dataset and continuing training on your small dataset. Reuses generic learned visual features (edges, textures, object parts), so ~500 images suffice instead of ~100,000. Fine-tuning is the specific technique; transfer learning is the umbrella term. In this project they refer to the same step.

**Catastrophic forgetting** — When fine-tuning overwrites useful pretrained knowledge (e.g. forklift accuracy rises while person accuracy collapses). Fixed by lowering the learning rate.

**mAP / mAP50** — Mean Average Precision, the standard 0–1 detection accuracy metric. mAP50 counts a detection as correct if its box overlaps ground truth by ≥50%.

**Precision** — Of all alerts raised, what fraction were real? Low precision = false alarms = the system gets muted and ignored.

**Recall** — Of all real violations, what fraction were caught? Low recall = missed hazards.

**Tracking / track ID** — A detector is amnesiac: it doesn't know frame 100's forklift is frame 99's forklift. A tracker matches boxes across frames and assigns each object a persistent integer ID, which is what makes velocity and duration measurable.

**ByteTrack** — The tracking algorithm used. Purely algorithmic; requires no training.

**Pose estimation / keypoints** — Locating body joints (nose, shoulders, wrists, hips, knees…). The standard COCO format gives **17 keypoints** per person, each as `(x, y, confidence)`. Occluded joints return low confidence and a *guessed* position — always gate on confidence before trusting one.

**Homography** — A 3×3 matrix mapping points on a flat plane (the warehouse floor) between the camera image and real-world coordinates. Computed once per camera from ≥4 point correspondences. **This is the single most important concept in the project** — see Section 7.2.

**Ground plane / floor coordinates** — Real-world positions in metres, obtained by projecting image points through the homography. All distance logic operates here, never in pixels.

**RF-DETR** — The object detector used (Apache-2.0, from Roboflow). Transformer-based, real-time, strong on domain shift.

**RTMPose** — The pose model used (Apache-2.0). Accessed via the `rtmlib` package, which runs it through ONNX without heavyweight dependencies.

**supervision** — A utility library (Apache-2.0) providing the `Detections` container, ByteTrack, and annotation drawing.

**Domain shift** — The gap between a model's training data and your actual footage (different camera angles, lighting, equipment). The reason pretrained models underperform on J&J video and need fine-tuning.

---

## 6. Data Situation

**What exists:** Whatever normal-operations footage J&J provides.

**What does not exist:** Any dataset of people violating these rules. This will not change.

**Implications:**

1. **The detector is trained on ordinary footage** — people and forklifts doing normal things. That is all it needs, because it only learns to find objects, not to judge behavior.
2. **Violation examples are only needed for *validation*,** and the team must **stage them**: ~20–30 short clips filmed deliberately in a controlled setting with a safety supervisor present. This is mandatory — without violation examples there is no way to measure whether the rules work.
3. **Negative cases matter as much as positive ones.** The most important single clip is a driver turning their head to look behind while reversing, which is *correct* behavior that naive Rule 5 logic will wrongly flag.
4. **Ask J&J for safety training videos.** These routinely contain staged demonstrations of exactly these violations and would be the highest-value data available.

---

## 7. Key Decisions and Their Rationale

**Do not reverse these without understanding why they were made.** Each was chosen against plausible-looking alternatives.

### 7.1 RF-DETR, not YOLO

Ultralytics YOLO is more popular, better documented, and easier for beginners — and it is **AGPL-3.0 licensed**, which imposes copyleft obligations incompatible with closed-source commercial deployment. J&J will deploy this. **RF-DETR is Apache-2.0 and is a hard project requirement.** Do not substitute YOLO, even for "quick testing," because prototype code has a way of becoming deliverable code.

The full stack is license-clean by construction: RF-DETR (Apache-2.0) + supervision/ByteTrack (Apache-2.0/MIT) + rtmlib/RTMPose (Apache-2.0).

### 7.2 Distance is measured on the floor, never in pixels

This is the most important technical decision and the most common way similar projects fail.

A camera flattens 3D space into 2D, destroying distance information. **Two bounding boxes 200 pixels apart can be 1 metre or 20 metres apart in reality**, depending on their distance from the camera. Worse, a pedestrian standing 15 m *behind* a forklift will often have an *overlapping* box, purely from occlusion geometry.

Therefore: **never use bounding-box overlap, pixel gaps, or box size as a proxy for real distance.** Project each object's ground-contact point (the bottom-centre of its box) through the homography into floor metres, then measure there.

The one-time ~10-minute-per-camera calibration this requires also provides forklift speed and Rule 4's walkway zones for free.

### 7.3 Driver association uses motion, not containment

A naive Rule 3 implementation flags the forklift driver as "a pedestrian standing 0 metres from a moving forklift." The driver must be identified and excluded.

The obvious approach — "the person whose box is inside the forklift's box is the driver" — fails, because a pedestrian standing *behind* a forklift also appears fully inside its box.

**The robust signal is that the driver moves *with* the vehicle.** Correlate the person's floor velocity with the forklift's over ~1 second. Someone walking past has uncorrelated velocity. Containment is used only as a candidate filter.

This one function is load-bearing for two rules: it prevents Rule 3 false positives *and* identifies whose pose to check for Rule 5.

### 7.4 Rule 5 uses pose keypoints, not bounding boxes

A forklift's axis-aligned bounding box includes the mast and overhead guard, so it is mostly empty space. A driver leaning well out of the cab is still comfortably "inside" that box — so box containment would **miss real violations**.

Instead, define the cab as an inset sub-region of the vehicle box and test whether specific *keypoints* (nose, shoulders, wrists) fall outside it. Pose gives limb-level precision that boxes fundamentally cannot, and it matches how the rule is actually written ("body inside the vehicle").

### 7.5 Every violation requires duration

Single-frame decisions flicker constantly due to occlusion, motion blur, and detection noise. Every rule requires a violating condition to persist for a minimum number of consecutive frames (1–2 seconds) before emitting an event.

This is also semantically correct: a driver briefly reaching for a control is not "keeping their body outside the vehicle."

### 7.6 Precision is prioritized over recall

For a *safety* system this seems backwards — surely missing a hazard is worse than a false alarm? In deployment it is not. **A system that cries wolf gets muted within a week**, and a muted system has zero recall on everything. Worse, people then trust the absence of alarms.

PoC target: **precision ≥ 0.8** on Rules 3 and 5. Tune thresholds accordingly.

### 7.7 Train/validation split by video, never by frame

Frames one second apart are near-identical. Splitting randomly by frame puts near-duplicates on both sides of the split, so the model is effectively evaluated on data it memorized. **Metrics will look excellent and be entirely fictional**, and this is discovered only when the model fails on new footage.

Always split by source video, camera, or day.

### 7.8 Recorded clips, not live streams

All development runs against recorded video: deterministic, replayable, and the same code path. Live-stream handling (RTSP buffering, dropped frames, keeping up with 30 fps) belongs to J&J's integration engineer. Do not build it.

---

## 8. Non-Obvious Technical Gotchas

Collected because each has silently broken similar projects:

1. **RF-DETR expects RGB; OpenCV loads BGR.** Missing the `cvtColor` call does not crash — it quietly degrades accuracy. Meanwhile `rtmlib` takes BGR directly. Convert per call site, deliberately.
2. **Class IDs must be read from the dataset's JSON, never hardcoded from memory.** Roboflow's COCO export sometimes inserts a dummy category at index 0, shifting everything.
3. **Use real frame timestamps, never `frame_number / fps`.** Videos drop frames silently; assuming fixed fps corrupts every velocity computed.
4. **The tracker's `frame_rate` parameter must match the rate frames are actually fed at** (we process every 3rd frame → 10 fps), not the video's native rate. Mismatch causes constant ID switches.
5. **Keypoint confidence must be checked before use.** Occluded joints return plausible-looking but invented coordinates.
6. **Homography point correspondences must be in matching order.** `image_points[i]` pairs with `floor_points[i]`. Mis-ordering produces silently absurd distances.
7. **On Colab, copy the dataset to local disk before training.** Reading thousands of images through the mounted Drive is 10–50× slower.

---

## 9. Open Questions for J&J

Unresolved; several block work and should be asked in week 1 (labeling in particular should not start before Q1 is answered, since relabeling is expensive):

1. Do pallet jacks and tuggers count as "vehicles" for Rule 3? *(Blocks labeling.)*
2. What is the actual length of J&J's forklifts? *(Sets the 3-vehicle-length threshold; ~2.7 m assumed as a placeholder.)*
3. For Rule 5, does turning the head to look behind while reversing count as a violation? *(Determines which keypoints are checked; flagging correct reversing behavior would destroy trust in the system.)*
4. Do safety training videos containing staged violations exist, and may we use them?
5. Are floor plans or measured floor dimensions available for calibration?
6. What precision/recall targets, and what false-alarm rate will the safety team tolerate?
7. Confirmation that Rule 2 is descoped as a non-vision task.
8. Any restrictions on retaining or sharing footage containing identifiable workers?

---

## 10. Privacy Posture

This is workforce monitoring and is treated as sensitive:

- **No biometric identity storage.** Track IDs are ephemeral integers that reset per video; no face recognition, no persistent worker identification.
- **Aggregate over individual.** Reporting is about violation counts and locations, not about named people.
- **Faces blurred** in any evidence frames shared outside the team.
- Footage handled under whatever retention and access rules J&J specifies (Question 8 above).

---

## 11. Document Map

| Document | Purpose |
|---|---|
| **`context.md`** (this file) | Background, rationale, glossary. Read first. |
| **`rule3-rule5-prototype-implementation.md`** | Step-by-step build guide with complete code. Read second, follow top to bottom. |

Earlier proposal documents exist in Google Drive covering scope negotiation and model selection. They are historical context, superseded by these two files on any point of conflict.

---

## 12. Definition of Done (prototype)

1. A fine-tuned RF-DETR model detecting `person` and `forklift`, with measured validation mAP.
2. A pipeline that consumes a video file and emits `events.jsonl` — one structured JSON violation event per line, with evidence frames.
3. Rules 3 and 5 working, validated against staged clips, at **precision ≥ 0.8**.
4. Rules 4 and 1 implemented (lower accuracy acceptable).
5. An annotated demo video showing boxes, track IDs, the identified driver, and violation banners.
6. A short write-up: measured results per rule, known limitations, next steps.

**If time runs short, a single well-validated Rule 3 with correct metric distance is a better outcome than four half-working rules.** Build order is dependency-driven and given in the implementation guide: detector → tracker → homography → Rule 3 → pose → driver association → Rule 5 → Rules 4/1.
