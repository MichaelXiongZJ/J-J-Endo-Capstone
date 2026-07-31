# How to demo this

Four ways to show the work, from "no setup at all" to "full pipeline". Pick by
audience and how long you have.

Shareable summary page (no setup, send a link):
**https://claude.ai/code/artifact/d56a0af5-74b8-4c12-8987-696a0fb7d5af**
Private until you share it from the page's share menu.

---

## Which rules each demo actually shows

Be upfront about this — someone will ask.

| | `demo.mp4` (real footage) | `rules3and4.mp4` (synthetic) |
|---|---|---|
| Rule 3 — proximity | **fires** | **fires** |
| Rule 4 — off walkway | no walkways in the footage | **fires** |
| Driver association (`DRV` tag) | no driver in the footage | **visible** |
| Rule 5 — body outside cab | blocked (no driver detected) | n/a (no cab) |
| Rule 1 — phone use | nobody uses a phone | n/a |

Why the gaps are real rather than laziness:

- **Rule 4 needs walkway polygons in floor metres.** The real clips have no marked
  walkways to calibrate against, and the rule deliberately *skips* rather than
  flagging every pedestrian when none is configured. The synthetic scene has one
  by construction, which is why it fires there.
- **Rule 5 needs a driver.** The real clips are reach trucks with no cab, and the
  detector cannot see seated drivers anyway (see below).
- **Rule 1 needs someone holding a phone to their ear.** No such footage exists.

The honest framing: *"we can demonstrate 2 of the 4 rules on the footage we have;
the other two need staged clips, which is the next task."*

## 1. The 60-second reel — best for any audience

`outputs/demo/demo.mp4` — 1080p, six different warehouse scenes.

Just play it. Say what to look for:

- **orange boxes** = forklift, **red boxes** = person, detected by our trained model
- **`id0`, `id1`** = persistent track IDs; the same object keeps its number across
  frames, which is what makes speed and duration measurable
- **red banner** = Rule 3 firing: a pedestrian is within 3 vehicle lengths of a
  moving forklift
- the lighting changes between scenes because the training data is deliberately
  varied

The one line worth saying out loud: **the distance test happens in floor metres,
not pixels.** Two boxes 200 px apart can be 1 m or 20 m apart depending on depth,
which is how projects like this usually go wrong.

## 1b. The two-rule clip — when asked "does it do more than Rule 3?"

`outputs/demo/rules3and4.mp4` — schematic rather than photoreal, but it shows
three things the real-footage reel cannot:

- the banner reads **`VIOLATION rule(s): [3, 4]`** — two rules at once
- the **yellow polygon** is the walkway; the pedestrian inside it is never flagged,
  the one outside it is
- **`id1 P DRV`** — the driver, correctly identified as riding the vehicle and
  correctly *excluded* from Rule 3. Without that, the driver would read as "a
  pedestrian standing 0 m from a moving forklift" and the system would alarm
  constantly

It looks abstract because the scene is generated, and that is the point: every
position is known exactly, so the events can be checked against arithmetic rather
than judged by eye.

## 2. Run it live in six seconds — best for technical audiences

No footage, no model weights, no GPU. Builds a synthetic warehouse whose ground
truth is exact arithmetic, then runs the real pipeline against it.

```bash
python -m scripts.make_synthetic_clip
python -m pytest -q                       # 91 tests, ~6 s
```

Then prove the geometry is real, which is the claim people doubt:

```bash
python -m src.geometry data/calibration/synthetic_cam1.json \
       --pair 140 700 1140 700 24.0
# (140,700)->(1140,700): computed 24.00 m, true 24.00 m, error 0.0%  [PASS]
```

The synthetic scene is built so Rule 3 *must* fire at t=7.0 s by arithmetic. The
pipeline fires at 6.97 s — one processed frame away. That is the demo: not "it
looks right", but "it agrees with the maths".

## 3. Show the output format — best for J&J's integration engineer

We deliver structured events, not a dashboard. One line of `events.jsonl`:

```json
{ "rule": 3, "person_track": 0, "vehicle_track": 1,
  "distance_m": 2.21, "threshold_m": 8.1, "vehicle_speed_ms": 0.73,
  "start_s": 0.47, "end_s": 3.67, "duration_s": 3.2,
  "camera_id": "ceiling_00", "evidence_frame": "evt_00000_rule3.jpg" }
```

The point: every alert is explainable. "A pedestrian came within 2.21 m of a
moving forklift, threshold 8.1 m, for 3.2 seconds" — with the frame attached.
A safety team can audit it, which is what makes it adoptable.

Evidence frames sit next to the JSONL in the same folder.

## 4. Run the pipeline on your own clip — needs a calibration

```bash
python -m src.run_pipeline \
  --video <your_clip.mp4> \
  --calib data/calibration/<cam>.json \
  --weights models/rfdetr_real/checkpoint_best_ema.pth
```

Calibration first, ~10 minutes per camera, and the validation step is mandatory:

```bash
python -m scripts.pick_calibration_points <ref_frame.jpg> --camera-id cam1
python -m src.geometry data/calibration/cam1.json --pair X1 Y1 X2 Y2 TRUE_METRES
```

---

## Numbers you can quote

| | |
|---|---|
| Detector | mAP50:95 **0.820** on real CCTV (1387 val images, 667 real) |
| Rule 3 | precision **1.000**, recall **1.000** over 178 clips |
| Homography | **0.0001 m** worst error vs exact camera matrices |
| Tests | **91** passing |

## Do not demo Rule 5

It will fail visibly. The rule logic is correct and tested (7/7 scenarios), but
the detector finds **0 of 16** seated drivers, so there is no driver for the rule
to check. Say this openly if asked — it is the honest state, the cause is
understood (drivers are unlabelled in the training data, so they trained as
background), and the fix is scoped.

Being straight about it lands better than being caught by the question.

## Two caveats to state up front

1. **No J&J footage has been used.** Everything is public and synthetic proxies,
   so numbers may shift on real site data.
2. **The 3-vehicle-length threshold uses a 2.7 m placeholder** for forklift
   length. It scales every Rule 3 decision linearly and needs J&J's real figure.

## Attribution, if the demo is public

Training data: NVIDIA PhysicalAI SDG-Warehouse (OpenMDW 1.1); Roboflow Universe
datasets `hitsz/forklift-and-human` and `pdf-ih16p/forklift2` (CC BY 4.0 —
attribution required).
