"""Measure Rule 3 against SDG ground truth using a PERFECT detector.

This answers a question nothing else in the project can: how good is the rule
logic itself, separated from detector error?

By feeding the simulator's exact ground-truth boxes into the real pipeline
(tracking, homography, driver association, duration gates, event aggregation) and
scoring against simulator-derived ground truth, any error that shows up belongs to
the *geometry and rules*, not to RF-DETR. That sets the ceiling: a fine-tuned
detector can only do worse.

Interpreting the result:
  * Precision/recall near 1.0 -> the rule logic is sound; remaining risk is all
    in detection quality and domain shift.
  * Poor numbers here -> a real bug in geometry, driver association, or the
    duration gates. Fix that before spending a single Colab credit.

Not a substitute for §10's staged clips: this footage is synthetic, contains only
the near-miss scenario, and cannot exercise Rules 5/4/1 at all.

Usage:
    python -m scripts.sdg_validate_rules
"""

import argparse
import json
import os

import numpy as np
import supervision as sv

from scripts.sdg_common import SDGClip, find_runs
from scripts.score_events import report, score
from src.rules import CFG
from src.run_pipeline import run

SRC_FPS = 30
FORKLIFT_ID, PERSON_ID = 1, 2
CLASS_ID = {'forklift': FORKLIFT_ID, 'person': PERSON_ID}


class GroundTruthDetector:
    """A detector that cannot be wrong: replays the simulator's exact boxes.

    The pipeline reads every frame, increments from 1, and processes when
    frame_idx % stride == 0 with stride = SRC_FPS / PROC_FPS. So processed call k
    (0-based) corresponds to source frame stride * (k + 1).
    """

    def __init__(self, clip, stride):
        self.clip = clip
        self.stride = stride
        self.i = -1

    def __call__(self, frame_bgr):
        self.i += 1
        frame_idx = self.stride * (self.i + 1)
        if frame_idx >= len(self.clip):
            return sv.Detections.empty()
        rows = [(box, CLASS_ID[cls])
                for cls, box, _w in self.clip.agents(frame_idx) if cls in CLASS_ID]
        if not rows:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array([b for b, _ in rows], dtype=np.float32),
            class_id=np.array([c for _, c in rows], dtype=int),
            confidence=np.ones(len(rows), dtype=np.float32),
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/sdg')
    ap.add_argument('--calib-dir', default='data/calibration')
    ap.add_argument('--truth', default='data/validation/sdg_ground_truth.json')
    ap.add_argument('--outdir', default='outputs/sdg_validation')
    ap.add_argument('--tolerance', type=float, default=2.0)
    ap.add_argument('--weights', default=None,
                    help='fine-tuned RF-DETR checkpoint. Omit to use the '
                         'simulator ground-truth boxes (the perfect-detector '
                         'ceiling); supply it for the honest end-to-end number')
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    detector = None
    if args.weights:
        from src.detector import RFDetrDetector
        detector = RFDetrDetector(weights=args.weights, threshold=args.threshold)
        print(f'using fine-tuned detector: {args.weights}\n')
    else:
        print('using ground-truth boxes (perfect detector)\n')

    if not os.path.exists(args.truth):
        raise SystemExit(f'{args.truth} missing — run scripts.sdg_calibration first')
    with open(args.truth) as f:
        truth = json.load(f)
    wanted = {c['video'] for c in truth['clips']}

    stride = max(1, round(SRC_FPS / CFG['PROC_FPS']))
    os.makedirs(args.outdir, exist_ok=True)
    all_events, skipped = [], []

    for run_dir, cams in find_runs(args.root):
        for cam in cams:
            clip = SDGClip(run_dir, cam)
            label = f'{clip.run_id}_{cam}.mp4'
            if label not in wanted:
                continue                     # degenerate calibration, excluded upstream
            calib = os.path.join(args.calib_dir,
                                 f'sdg_{clip.run_id[:20]}_{cam}.json')
            if not os.path.exists(calib):
                skipped.append(label)
                continue
            res = run(clip.video, calib,
                      detector or GroundTruthDetector(clip, stride),
                      outdir=os.path.join(args.outdir, f'{clip.run_id[:12]}_{cam}'),
                      use_pose=False,            # no pose: Rules 5 and 1 are N/A here
                      write_video=False, save_evidence=False, verbose=False,
                      person_id=PERSON_ID, forklift_id=FORKLIFT_ID,
                      video_label=label)
            with open(res['events_path']) as f:
                evs = [json.loads(line) for line in f]
            all_events.extend(evs)
            n_gt = len(next(c for c in truth['clips'] if c['video'] == label)['violations'])
            print(f'{label[:46]:46} events={len(evs):3d}  gt={n_gt}')

    if skipped:
        print(f'\nskipped {len(skipped)} clip(s) with no calibration')

    merged = os.path.join(args.outdir, 'events.jsonl')
    with open(merged, 'w') as f:
        for e in all_events:
            f.write(json.dumps(e) + '\n')
    print(f'\nmerged {len(all_events)} events -> {merged}')

    if args.weights:
        print('\n=== Rule 3 END TO END with the fine-tuned detector ===')
        print('Compare against the perfect-detector run: the gap between them is '
              'attributable to detection, everything else is shared.')
    else:
        print('\n=== Rule 3 with a PERFECT detector (ceiling on achievable accuracy) ===')
    counts, fps, fns = score(all_events, truth, args.tolerance, rules={3})
    report(counts, fps, fns, targets=(3,))

    n_neg = sum(1 for c in truth['clips'] if not c['violations'])
    if n_neg == 0:
        print('\nNOTE: every clip in this scenario contains a genuine near-miss, so '
              'there are no all-negative clips. Precision is still measured (events '
              'outside a ground-truth interval count as FP), but for true negatives '
              'fetch the routine-operations scenario:\n'
              '  edit SCENARIO in scripts/fetch_sdg.py to warehouse_box_pickup')


if __name__ == '__main__':
    main()
