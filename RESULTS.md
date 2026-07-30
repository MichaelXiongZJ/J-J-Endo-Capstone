# Prototype Results

Measured outcomes for the warehouse safety CV prototype. Item 6 of the Definition
of Done (`context.md` §12).

**One-line summary:** the perception stack and Rule 3 work and are measured;
Rule 5 — the rule J&J cares most about — has validated *logic* but **no
validation data**, and closing that is the single most important next step.

Everything below is measured on **synthetic** footage (NVIDIA PhysicalAI
SDG-Warehouse). No J&J footage has been available at any point. Numbers are real
but the domain is narrow, and §10's staged clips remain mandatory before any of
this is quoted to a safety team.

---

## 0. The finding that blocks the MVP

**The detector cannot see seated forklift drivers, and fine-tuning is what
destroyed that ability.**

Measured on 16 clean validation images from `forklift2`, every one containing a
plainly visible driver in a hard hat and hi-vis vest:

| Model | Forklifts with the driver detected |
|---|---|
| COCO-pretrained RF-DETR (before any fine-tuning) | **5 / 16** |
| Our fine-tuned detector (`rfdetr_real`, mAP50:95 0.820) | **0 / 16** |

The fine-tuned model finds every forklift and none of their drivers. The COCO
model it started from could find some, with no false positives.

**Cause:** the real datasets label standing pedestrians but not seated drivers —
only 4.9% of `forklift2`'s forklift boxes contain a labeled person, against a
driver visibly present in most of them. In training, an unlabeled object is
supervised as *background*, so we spent 222 GPU-minutes teaching the model that a
seated driver is not a person.

This is catastrophic forgetting (context.md §5), but caused by **label gaps
rather than learning rate**, so the guide's remedy — lower `lr` to 5e-5 — would
not touch it.

**Consequence for Rule 5:** Rule 5 identifies the driver, then tests whether
their keypoints leave the cab. With no driver detection there is no driver to
test, so Rule 5 cannot function on real footage with this detector, regardless of
how good its rule logic is. **The MVP is not met.**

**Fix:** label the drivers. The images are already right — real factory, J&J's
exact Toyota counterbalance forklifts, drivers present. Only one class in one
region is missing. Roughly 800 unique source images sit behind the ~2600
augmented copies. COCO pseudo-labelling recovers about 31% of them accurately
(`scripts/label_drivers.py`); the rest needs human labelling, which is a few
hours across five people and exactly the task §4.1 assumes.

Until that is done, every Rule 5 number in this document describes logic that has
no working input.

## 1. Detector

Fine-tuned RF-DETR (Apache-2.0), one model for `person` + `forklift`.

| | v1 (synthetic) | v2 (+ box_pickup) | **real (+ 2 real datasets)** |
|---|---|---|---|
| Training images | 1040 | 2053 | **5886 (4553 real)** |
| Validation images | 350 | 720 | **1387 (667 real)** |
| Best EMA mAP50:95 | 0.967 | 0.962 | **0.820** |
| Final-epoch mAP50 | 0.994 | 0.985 | 0.968 |
| forklift AP50:95 | 0.979 | 0.973 | **0.830** |
| person AP50:95 | 0.891 | 0.896 | **0.777** |
| Wall time (RTX 3070) | 52.8 min | 100.3 min | 221.9 min |

**`rfdetr_real` is the only meaningful number.** v1 and v2 scored ~0.96 against
synthetic validation drawn from the same simulator as their training data � an
easy, in-distribution task. 0.820 against a validation set containing 667 real
CCTV images is the first figure in this project that reflects anything like
deployment. The drop is not a regression; it is the synthetic numbers being
exposed as optimistic.

And per �0, that 0.820 hides a total failure on seated drivers.

§4.6's acceptance bar is person ≥ 0.80 and forklift ≥ 0.60 mAP50. Both models
clear it comfortably.

**v1 and v2 are not directly comparable** — they have different validation sets
(v2's includes `box_pickup` aisle scenes, which are visually unlike the near-miss
floor shots). v2's slightly lower mAP reflects a harder, more varied validation
set, not a worse model. Neither is "better" in a way that matters yet; both are
measuring an easy in-distribution task.

**Do not read these as a forecast of J&J performance.** The task here is easy:
one warehouse, two object classes, clean synthetic renders, no clutter or
occlusion variety, and a validation split drawn from the same simulator. The
honest claim is *"the training pipeline works and the data is learnable"*, not
*"the detector is 97% accurate"*.

Split is **by run**, never by frame (§4.2) — including across the 5 ceiling
cameras of a run, which show the same moment from different angles and would
otherwise leak.

## 2. Rule 3 — pedestrian within 3 vehicle lengths of a working vehicle

Scored per event against ground truth derived from simulator state, over **178
clips**.

| Detector | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Perfect (simulator boxes) | 44 | 0 | 0 | **1.000** | **1.000** |
| Fine-tuned RF-DETR v1, end to end | 178 | 0 | 0 | **1.000** | **1.000** |
| Fine-tuned RF-DETR v2, end to end | 178 | 0 | 0 | **1.000** | **1.000** |

PoC target is precision ≥ 0.8. Both runs pass with margin.

The perfect-detector run is the more informative one: it isolates the rule logic
from detector error, so it says the geometry, driver association, duration gates
and event aggregation are correct. The end-to-end run matching it says detection
is not the limiting factor *on this data*.

**Caveat that matters:** every near-miss clip contains a genuine violation, so
there are no all-negative clips. Precision is still measured — an event outside a
ground-truth interval scores as a false positive, and none occurred — but a
precision figure carries more weight when some clips should produce nothing at
all. The `warehouse_box_pickup` scenario is the natural source of those.

## 3. Homography — the number the project depends on

`CameraGeometry` graded against the simulator's exact camera matrices across 178
clips:

| Mean error | Worst error |
|---|---|
| **0.0000 m** | **0.0001 m** |

This is the only place in the project where ground truth for the homography
exists at all; on real footage there is nothing to compare against. §6.4 asks for
distances within 10%. Sub-millimetre error means the homography and the
pixels→metres conversion are correct, and that on real cameras the limiting factor
will be the tape measure, not this code.

## 4. Rule 5 — driver's body outside the vehicle

**Logic validated. Accuracy unmeasured. This is the project's biggest gap.**

The §10 validation matrix is encoded as executable scenarios
(`tests/test_rule5_scenarios.py`), all passing:

| Scenario | Expected | Result |
|---|---|---|
| Arm only outside cab | fires | pass |
| Torso outside cab | fires | pass |
| Head outside while reversing | **fires** (J&J ruling) | pass |
| Head-turn, head inside cab | silent | pass |
| Normal seated driving | silent | pass |
| Brief reach for a control | silent | pass |
| Occluded low-confidence leg joints | silent | pass |

**Pose estimation partially de-risked.** RTMPose was run on warehouse workers at
ceiling-camera distance (`outputs/pose_check.jpg`):

| Clip | Person height | Valid keypoints |
|---|---|---|
| nearmiss ceiling_00 | 228 px | 17 / 17 |
| nearmiss ceiling_00 | 356 px | 17 / 17 |
| box_pickup cam_00 (crouching) | 190 px | 14 / 17 |
| box_pickup cam_04 | 251 px | 17 / 17 |

Skeletons land correctly on the bodies, including a crouching worker — a hard
pose. So pose estimation is viable at these distances and the confidence gate
behaves as designed.

What is still **not** validated: RTMPose on *real* people, and on a driver seen
*through a cab frame*, where the mast, overhead guard and seat back occlude the
torso. Those are the conditions Rule 5 actually runs in, and no synthetic data
here reproduces them.

Why there is no data: **no SDG-Warehouse scenario contains a driver seated in a
forklift.** The `nearmiss` vehicles are stand-on reach trucks with no cab; the
`box_pickup` forklift is the right type but parked and empty. Rule 5 needs a
seated driver who leans out, and that requires staged footage or authored
character animation.

## 5. Rules 4 and 1

Implemented, unit-tested, and unexercised on real data.

- **Rule 4 (walkways)** needs walkway polygons in floor metres. None of the
  synthetic scenes has marked walkways, so the rule is skipped rather than
  flagging every pedestrian.
- **Rule 1 (phone use)** is the weakest rule by design and needs staged clips
  including its false-positive twins (scratching head, adjusting a hard hat,
  holding a radio).

## 6. Vehicle-type gap

J&J operates **sit-down counterbalance forklifts**. The detector is trained
mostly on **stand-on reach trucks**, because that is what the near-miss scenario
provides. A detector trained on reach trucks alone may not recognise J&J's
vehicles — different silhouette entirely.

Attempted mitigation in v2: sit-down forklift labels were recovered from
`box_pickup` instance segmentation (`scripts/sdg_extract_forklifts.py`), since
that scenario renders the right vehicle but leaves it unlabeled. The resulting
coverage is too thin to matter:

| Split | nearmiss (reach truck) images | box_pickup images | **sit-down forklift boxes** |
|---|---|---|---|
| train | 1123 | 210 | **28** |
| valid | 650 | 70 | **0** |

Only 2 of 20 cameras show a forklift at all — in the box-picking aisles it is
usually out of frame — and it is parked, so those 28 boxes come from **2 static
viewpoints of 2 parked vehicles**. Worse, the validation split contains **zero**,
so sit-down forklift accuracy is not merely low, it is **unmeasurable**.

**Treat the vehicle-type gap as open.** v2 is not meaningfully better than v1 for
J&J's equipment, and no amount of additional SDG data will fix it: the scenario
that renders the right forklift does not label it, and the scenario that labels
its vehicle renders the wrong one.

A secondary risk worth recording: the extractor ignores masks under 1500 px, so a
*distant* forklift could be present-but-unlabeled in a `box_pickup` frame, which
is the poisoning case. The 18 cameras with no forklift were confirmed empty by
segmentation (authoritative, since it covers every pixel), so the exposure is
limited to small/distant instances.

## 7. Known limitations

1. **No real footage anywhere in these results.** Every number is synthetic.
2. **Rule 5 has no validation data**, and it is the most important rule.
3. **Driver association is unvalidated on real data.** The synthetic scenes have
   no annotated driver, so the motion-correlation logic (context.md §7.3) is
   exercised only by unit tests and the synthetic clip generator.
4. **No negative clips** for Rule 3 precision.
5. **Vehicle type mismatch** (§6 above).
6. **Domain shift**: CG humans, mostly no hi-vis PPE, simulated lighting, one
   warehouse geometry.
7. **Rule 3's "unless signaled for recognition" exception is undetectable.** All
   proximity events are flagged for human review.
8. **Homography assumes a flat floor** — invalid on ramps.
9. **Single camera, no occlusion recovery.**
10. **`vehicle_length_m` is a 2.7 m placeholder.** It scales every Rule 3
    decision linearly and J&J has not yet supplied the real figure.

## 8. Next steps, in priority order

1. **Film the §10 staged clips**, Rule 5 first. An afternoon with a forklift, a
   volunteer and a safety supervisor unblocks the most important rule. The
   expected outcome of each clip is already written down in
   `tests/test_rule5_scenarios.py`.
2. **Get J&J footage** for a domain-shift fine-tune and for sit-down forklift
   coverage. Even 100–200 labeled real frames would be worth more than more
   synthetic data.
3. **Get the real forklift length** and set `vehicle_length_m`.
4. **Fetch `box_pickup` clips for negative-clip precision** on Rule 3.
5. **Calibrate a real camera** and re-run the §6.4 check with a tape measure.
6. Only then consider Isaac Sim for Rule 5 — and note the driver would need
   authored lean-out animation, which is the expensive part.

## 9. Reproducing these numbers

```bash
python -m scripts.fetch_sdg --scenario nearmiss --shards 1 2 3 4
python -m scripts.sdg_calibration                  # calibration + Rule 3 truth
python -m scripts.sdg_to_coco --root data/sdg --out data/dataset
python -m scripts.train_detector --epochs 25
python -m scripts.sdg_validate_rules               # perfect-detector ceiling
python -m scripts.sdg_validate_rules --weights models/rfdetr_v1/checkpoint_best_ema.pth
python -m scripts.make_demo --weights models/rfdetr_v1/checkpoint_best_ema.pth
python -m pytest -q                                # 78 tests
```
