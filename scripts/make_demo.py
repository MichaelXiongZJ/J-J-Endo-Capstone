"""Render the demo video: boxes, track IDs, the identified driver, violation banners.

Item 5 of the Definition of Done (context.md §12). Picks clips that actually
contain a Rule 3 violation — a demo of nothing happening proves nothing — runs
the real pipeline on each, and concatenates the annotated results.

Usage:
    python -m scripts.make_demo --weights models/rfdetr_v1/checkpoint_best_ema.pth
    python -m scripts.make_demo --weights ... --clips 4 --pose
"""

import argparse
import json
import os

import cv2

from scripts.sdg_common import SDGClip, find_runs
from src.detector import RFDetrDetector
from src.run_pipeline import run

PERSON_ID, FORKLIFT_ID = 2, 1


def pick_clips(truth_path, root, calib_dir, n):
    """Clips with the longest ground-truth violation, which demo most clearly."""
    with open(truth_path) as f:
        truth = json.load(f)
    best = {}
    for c in truth['clips']:
        v = [x for x in c['violations'] if x['rule'] == 3]
        if v:
            best[c['video']] = max(x['end_s'] - x['start_s'] for x in v)

    out = []
    for run_dir, cams in find_runs(root):
        for cam in cams:
            run_id = os.path.basename(run_dir)
            label = f'{run_id}_{cam}.mp4'
            calib = os.path.join(calib_dir, f'sdg_{run_id[:20]}_{cam}.json')
            if label in best and os.path.exists(calib):
                out.append((best[label], run_dir, cam, calib, label))
    out.sort(reverse=True)
    return out[:n]


def concat(paths, dest, fps=10):
    """Join annotated clips into one file, with a divider between them."""
    writer = None
    for p in paths:
        cap = cv2.VideoCapture(p)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = fr.shape[:2]
                writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (w, h))
            writer.write(fr)
        cap.release()
    if writer:
        writer.release()
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--root', default='data/sdg')
    ap.add_argument('--calib-dir', default='data/calibration')
    ap.add_argument('--truth', default='data/validation/sdg_ground_truth.json')
    ap.add_argument('--outdir', default='outputs/demo')
    ap.add_argument('--clips', type=int, default=4)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--pose', action='store_true',
                    help='enable RTMPose (Rules 5 and 1). Off by default: these '
                         'reach trucks have no cab, so Rule 5 cannot fire and '
                         'pose only costs time')
    args = ap.parse_args()

    picks = pick_clips(args.truth, args.root, args.calib_dir, args.clips)
    if not picks:
        raise SystemExit('no clips with Rule 3 ground truth — run scripts.sdg_calibration')

    print(f'rendering {len(picks)} clip(s):')
    for dur, _, cam, _, label in picks:
        print(f'  {dur:5.1f}s violation  {label[:52]} ({cam})')

    detector = RFDetrDetector(weights=args.weights, threshold=args.threshold)
    os.makedirs(args.outdir, exist_ok=True)

    parts, total_events = [], 0
    for _dur, run_dir, cam, calib, label in picks:
        clip = SDGClip(run_dir, cam)
        sub = os.path.join(args.outdir, f'{clip.run_id[:12]}_{cam}')
        res = run(clip.video, calib, detector, outdir=sub,
                  use_pose=args.pose, write_video=True, save_evidence=True,
                  person_id=PERSON_ID, forklift_id=FORKLIFT_ID,
                  video_label=label, verbose=False)
        parts.append(os.path.join(sub, 'videos', 'annotated.mp4'))
        total_events += res['events']
        print(f'  {label[:52]}: {res["events"]} event(s)')

    dest = os.path.join(args.outdir, 'demo.mp4')
    concat(parts, dest)
    size = os.path.getsize(dest) / 2**20 if os.path.exists(dest) else 0
    print(f'\ndemo -> {dest} ({size:.1f} MiB), {total_events} events total')
    print('Shows: boxes, track IDs, DRV tag on the identified driver, and a '
          'red VIOLATION banner on frames where a rule fires.')


if __name__ == '__main__':
    main()
