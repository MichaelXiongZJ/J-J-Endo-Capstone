"""Derive calibration and Rule 3 ground truth from SDG camera parameters.

Two things fall out of the simulator's camera matrices and world positions, both
of which normally cost real effort:

1. **Calibration without a tape measure.** §6.1 has you clicking floor points and
   measuring them physically. Here we project known floor points (z=0) through the
   exact camera matrices to get perfect correspondences, then write a normal
   `data/calibration/*.json` that `CameraGeometry` consumes unchanged.

   This also lets us *grade our own homography*: we fit `CameraGeometry` from 4
   points and compare its `to_floor()` against the true projection over the whole
   visible floor. Error in metres is reported. Nothing else in the project can
   check this — on real footage there is no ground truth to compare against.

2. **Rule 3 ground truth for free.** Agent world positions are exact metres, so
   the intervals when a pedestrian is genuinely within 3 vehicle lengths of a
   moving vehicle are computable, not staged. Output is in the exact format
   `scripts/score_events.py` expects, so precision/recall can be measured on
   thousands of clips instead of 20-30 hand-filmed ones.

Two honest caveats on that ground truth:

  * The vehicle's `world_position` is its pivot, not the centre of its footprint;
    on this reach truck the pivot sits ~1 m from the box bottom-centre our
    pipeline uses. So GT distance and measured distance have a systematic offset
    of that order. Fine for scoring event *timing*; do not quote sub-metre
    agreement.
  * "Working" here means the vehicle's own speed exceeds MOVING_MS, computed from
    position deltas. The `speed` field in the data is inconsistent between
    characters and vehicles and is ignored.

Usage:
    python -m scripts.sdg_calibration
    python -m scripts.sdg_calibration --validate-only
"""

import argparse
import json
import os

import numpy as np

from scripts.sdg_common import SDGClip, find_runs, spread_quad
from src.events import COOLDOWN_S
from src.geometry import CameraGeometry, floor_dist
from src.rules import CFG

SRC_FPS = 30.0
VEHICLE_LENGTH_M = 2.7          # placeholder; J&J has not given the real figure


def write_calibration(clip, out_dir='data/calibration', extent=14.0, n_points=8):
    """Write a CameraGeometry-compatible calibration for one clip.

    n_points defaults to 8 rather than the minimum 4. With exactly 4 points the
    homography is an exact fit, so a near-collinear quad — which happens when a
    camera looks down a narrow aisle and the visible floor is a sliver — yields a
    wildly wrong but error-free-looking result. Extra points make findHomography
    least-squares and well-conditioned, and make reprojection error meaningful.
    Same advice applies on real cameras: click more than four points.
    """
    floor, image = clip.visible_floor_points(extent=extent)
    if len(floor) < n_points:
        return None, f'only {len(floor)} visible floor points, need {n_points}'
    fq, iq = spread_quad(floor, image, n=n_points)
    if len(fq) < 4:
        return None, 'could not find 4 spread points'

    name = f'sdg_{clip.run_id[:20]}_{clip.camera}'
    path = os.path.join(out_dir, f'{name}.json')
    os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'camera_id': name,
            '_comment': 'Derived from SDG-Warehouse camera matrices by '
                        'scripts/sdg_calibration.py — exact, not tape-measured. '
                        'Synthetic data; do not present as a real J&J camera.',
            'image_points': [[round(u, 2), round(v, 2)] for u, v in iq],
            'floor_points': [[round(x, 3), round(y, 3)] for x, y in fq],
            'walkways': [],
            'vehicle_length_m': VEHICLE_LENGTH_M,
        }, f, indent=2)
    return path, None


def grade_homography(clip, calib_path, extent=14.0):
    """Compare CameraGeometry's fitted homography against exact projection.

    Returns (mean_err_m, p95_err_m, n_points).
    """
    geom = CameraGeometry(calib_path)
    floor, image = clip.visible_floor_points(extent=extent)
    errs = [floor_dist(geom.to_floor(u, v), (fx, fy))
            for (fx, fy), (u, v) in zip(floor, image)]
    errs = np.asarray(errs)
    return float(errs.mean()), float(np.percentile(errs, 95)), len(errs)


def rule3_intervals(clip, radius_m=None, min_seconds=0.5):
    """Intervals where a pedestrian is within `radius_m` of a WORKING vehicle.

    Two definitions here must match `src/rules.py` exactly, or the scorer measures
    a disagreement about wording rather than a detection error:

    * "Working" means moving now OR moved within RECENT_MOVE_S — same as
      `MotionState.is_working`. Using instantaneous speed instead fragments one
      real violation into several whenever the vehicle pauses mid-manoeuvre, and
      the pipeline (correctly) reports one continuous event, scoring the rest as
      false negatives.
    * `min_seconds` drops sub-half-second slivers. A 2-frame proximity at a clip
      boundary is not a violation anyone would want reported, and no
      duration-gated system can detect one.

    Also note the pipeline cannot fire in roughly the first 0.5 s of any clip:
    ByteTrack needs ~3 frames to confirm a track and MotionState needs 4 samples
    before a velocity exists. Violations that begin at t=0 are unavoidably missed.
    """
    radius_m = radius_m or CFG['R3_VEHICLE_LENGTHS'] * VEHICLE_LENGTH_M
    min_frames = max(1, int(round(min_seconds * SRC_FPS)))
    n = len(clip)

    # All people and all vehicles per frame, not just the first of each: a clip
    # with two workers would otherwise have half its violations invisible.
    #
    # Every `character` is treated as a pedestrian. That is correct for this
    # scenario — the reach trucks here have no annotated operator riding them — but
    # it also means this data CANNOT exercise driver association (context.md §7.3).
    people, vehicles = [], []
    for fr in range(n):
        p, v = [], []
        for cls, _box, world in clip.agents(fr):
            if world is None:
                continue
            (p if cls == 'person' else v).append(world)
        people.append(p)
        vehicles.append(v)

    def min_dist(fr):
        if not people[fr] or not vehicles[fr]:
            return None
        return min(float(np.hypot(a[0] - b[0], a[1] - b[1]))
                   for a in people[fr] for b in vehicles[fr])

    def veh_speed(fr, win=5):
        a, b = max(0, fr - win), min(n - 1, fr + win)
        if not vehicles[a] or not vehicles[b] or b == a:
            return None
        dt = (b - a) / SRC_FPS
        # Fastest vehicle in view: any working vehicle makes the scene hazardous.
        return max(float(np.hypot(q[0] - p[0], q[1] - p[1]) / dt)
                   for p in vehicles[a] for q in vehicles[b])

    # "Working" with the same recent-movement grace as MotionState.is_working.
    last_moving = -1e9
    working = []
    for fr in range(n):
        sp = veh_speed(fr)
        t = fr / SRC_FPS
        if sp is not None and sp > CFG['MOVING_MS']:
            last_moving = t
        working.append((t - last_moving) < CFG['RECENT_MOVE_S'])

    flags = []
    for fr in range(n):
        d = min_dist(fr)
        flags.append(d is not None and d < radius_m and working[fr])

    # Merge lapses shorter than the aggregator's cooldown. src/events.py
    # deliberately treats a brief dropout as one continuous episode, so ground
    # truth must segment episodes the same way. Otherwise a correctly-merged
    # single event matches only the first GT interval and every other fragment
    # scores as a false negative — measuring a disagreement about episode
    # boundaries rather than a detection failure.
    gap_frames = int(round(COOLDOWN_S * SRC_FPS))
    run_len, i = 0, 0
    while i < len(flags):
        if not flags[i]:
            run_len += 1
        else:
            if 0 < run_len <= gap_frames and i - run_len > 0:
                for j in range(i - run_len, i):
                    flags[j] = True
            run_len = 0
        i += 1

    intervals, start = [], None
    for fr, f in enumerate(flags + [False]):
        if f and start is None:
            start = fr
        elif not f and start is not None:
            if fr - start >= min_frames:
                seg = [d for d in (min_dist(i) for i in range(start, fr)) if d is not None]
                intervals.append({
                    'rule': 3,
                    'start_s': round(start / SRC_FPS, 2),
                    'end_s': round((fr - 1) / SRC_FPS, 2),
                    'min_distance_m': round(min(seg), 2) if seg else None,
                    'note': 'derived from simulator 3D box positions',
                })
            start = None
    return intervals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/sdg')
    ap.add_argument('--calib-dir', default='data/calibration')
    ap.add_argument('--truth-out', default='data/validation/sdg_ground_truth.json')
    ap.add_argument('--extent', type=float, default=14.0,
                    help='half-size of the floor grid searched, in metres')
    ap.add_argument('--points', type=int, default=8,
                    help='correspondences per camera; >4 keeps the fit well-conditioned')
    ap.add_argument('--max-err-m', type=float, default=0.05,
                    help='flag a calibration as degenerate above this mean error')
    ap.add_argument('--validate-only', action='store_true',
                    help='grade homographies without rewriting anything')
    args = ap.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f'no runs under {args.root}/ — run scripts.fetch_sdg first')

    clips, errors, degenerate, moving_cams = [], [], [], 0
    print(f'{"clip":52} {"mean_err_m":>11} {"p95_err_m":>10} {"pts":>5}')
    print('-' * 82)

    for run_dir, cams in runs:
        for cam in cams:
            clip = SDGClip(run_dir, cam)
            if not clip.is_static:
                moving_cams += 1
            name = f'sdg_{clip.run_id[:20]}_{clip.camera}'
            path = os.path.join(args.calib_dir, f'{name}.json')
            if not args.validate_only or not os.path.exists(path):
                path, err = write_calibration(clip, args.calib_dir, args.extent, args.points)
                if err:
                    print(f'{name:52} SKIPPED: {err}')
                    continue
            mean_e, p95_e, npts = grade_homography(clip, path, args.extent)
            flag = '' if mean_e <= args.max_err_m else '  <-- DEGENERATE, excluded'
            print(f'{name:52} {mean_e:11.4f} {p95_e:10.4f} {npts:5d}{flag}')
            if mean_e > args.max_err_m:
                # Near-collinear visible floor: the homography is unusable, so do
                # not let it pollute the aggregate or get used downstream.
                degenerate.append(name)
                os.remove(path)
                continue
            errors.append(mean_e)
            clips.append((clip, path))

    if errors:
        print('-' * 82)
        print(f'homography error across {len(errors)} clips: '
              f'mean {np.mean(errors):.4f} m, worst {np.max(errors):.4f} m')
        print('§6.4 wants distances within 10%. Sub-centimetre error here means the '
              'homography and CameraGeometry are correct; on real footage the limit '
              'will be measurement accuracy, not this code.')
    if degenerate:
        joined = ', '.join(degenerate)
        print(f'\n{len(degenerate)} camera(s) excluded as degenerate (near-collinear '
              f'visible floor, typically a view straight down an aisle): {joined}')
    if moving_cams:
        print(f'\nNOTE: {moving_cams} camera(s) move during their clip, so a single '
              'homography is only approximate for those.')

    if args.validate_only:
        return

    truth = {'_comment': 'Rule 3 ground truth derived from SDG simulator world '
                         'positions by scripts/sdg_calibration.py. Synthetic; '
                         'complements but does not replace staged footage (§10).',
             'clips': []}
    n_viol = 0
    for clip, _ in clips:
        iv = rule3_intervals(clip)
        n_viol += len(iv)
        truth['clips'].append({
            'video': f'{clip.run_id}_{clip.camera}.mp4',
            'violations': iv,
        })
    os.makedirs(os.path.dirname(args.truth_out), exist_ok=True)
    with open(args.truth_out, 'w') as f:
        json.dump(truth, f, indent=2)
    n_neg = sum(1 for c in truth['clips'] if not c['violations'])
    print(f'\nwrote {args.truth_out}: {len(truth["clips"])} clips, '
          f'{n_viol} Rule 3 violations, {n_neg} negative clips')


if __name__ == '__main__':
    main()
