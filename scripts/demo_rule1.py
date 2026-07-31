"""Rule 1 demo — run the real phone-use rule on a clip you film yourself.

Rule 1 is the only rule that can be demonstrated without a warehouse, a forklift,
or a camera calibration. It works purely on body keypoints, so the geometry that
Rules 3 and 4 depend on is irrelevant here: a laptop webcam and a phone are the
entire rig.

That matters, because no footage we could obtain contains phone use. Measured
across 109 pose samples of SDG warehouse workers, the closest anyone came to the
threshold was a wrist-to-head ratio of 1.17 against a threshold of 0.6 — they are
carrying boxes, not making calls.

WHAT TO FILM (about 40 seconds, one take)
    1. stand normally, arms down            -> must stay silent
    2. hold a phone to your ear for ~5 s    -> must fire
    3. arms down again                      -> must clear
    4. scratch your head for ~5 s           -> the interesting one, see below
    5. briefly touch your ear, under 2 s    -> must stay silent (duration gate)

Step 4 is the honest part of the demo. Rule 1 measures wrist-to-head distance
normalised by shoulder width; it cannot tell a phone from a hand. Scratching your
head will probably fire, and that is exactly the documented weakness
(context.md §13.4). Showing it deliberately is far better than being caught by it,
and it motivates why Rule 1 is the lowest-confidence rule of the four.

The overlay prints the live ratio and the duration gate filling up, so the rule is
not a black box — a viewer watches the number cross the threshold and the counter
run out before anything is reported.

Usage:
    python -m scripts.demo_rule1 --video my_clip.mp4
    python -m scripts.demo_rule1 --video my_clip.mp4 --device cuda
"""

import argparse
import json
import os

import cv2
import numpy as np

from src.pose_utils import (L_EAR, L_SHOULDER, L_WRIST, NOSE, R_EAR, R_SHOULDER,
                            R_WRIST, draw_pose, run_pose, valid)
from src.rules import CFG, dist2d

THRESH = CFG['R1_WRIST_HEAD_RATIO']
NEED = int(CFG['R1_MIN_S'] * CFG['PROC_FPS'])


def ratio_for(kp):
    """Wrist-to-head distance / shoulder width, or None if the pose is unusable.

    Normalising by shoulder width is what makes the rule scale-free: a 40 px gap
    means something different at 3 m and 30 m from the camera.
    """
    if not (valid(kp[L_SHOULDER], CFG['KPT_CONF']) and valid(kp[R_SHOULDER], CFG['KPT_CONF'])):
        return None, None, None
    shoulder_w = max(1.0, dist2d(kp[L_SHOULDER], kp[R_SHOULDER]))
    heads = [kp[i] for i in (NOSE, L_EAR, R_EAR) if valid(kp[i], CFG['KPT_CONF'])]
    if not heads:
        return None, None, None
    best, best_pair = None, None
    for w in (L_WRIST, R_WRIST):
        if not valid(kp[w], CFG['KPT_CONF']):
            continue
        for h in heads:
            r = dist2d(kp[w], h) / shoulder_w
            if best is None or r < best:
                best, best_pair = r, (kp[w][:2], h[:2])
    return best, best_pair, shoulder_w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', default='outputs/demo/rule1.mp4')
    ap.add_argument('--events', default='outputs/demo/rule1_events.jsonl')
    ap.add_argument('--device', default='cpu',
                    help='cuda if onnxruntime-gpu has a working CUDA provider')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {args.video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src_fps / CFG['PROC_FPS']))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                             CFG['PROC_FPS'], (W, H))

    scale = max(0.6, W / 1280.0)
    raised_frames, events, idx, processed = 0, [], 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        processed += 1
        t = idx / src_fps

        kpts = run_pose(frame, args.device)
        vis = draw_pose(frame.copy(), kpts, CFG['KPT_CONF'])

        ratio, pair, _ = (None, None, None)
        if len(kpts):
            ratio, pair, _ = ratio_for(kpts[0])

        raised = ratio is not None and ratio < THRESH
        raised_frames = raised_frames + 1 if raised else 0
        firing = raised_frames >= NEED
        if firing and (not events or events[-1]['end_s'] < t - 1.0):
            events.append({'rule': 1, 'start_s': round(t, 2), 'end_s': round(t, 2),
                           'ratio': round(ratio, 2)})
        elif firing:
            events[-1]['end_s'] = round(t, 2)

        if pair is not None:
            (wx, wy), (hx, hy) = pair
            col = (0, 0, 255) if raised else (0, 200, 255)
            cv2.line(vis, (int(wx), int(wy)), (int(hx), int(hy)), col, max(2, int(3 * scale)))

        # Readout: the live measurement, the threshold it is compared against, and
        # the duration gate filling. The rule should never look like a black box.
        pad = int(16 * scale)
        cv2.rectangle(vis, (pad, pad), (int(pad + 430 * scale), int(pad + 118 * scale)),
                      (24, 24, 24), -1)
        txt = 'no usable pose' if ratio is None else f'wrist/head ratio {ratio:5.2f}'
        cv2.putText(vis, txt, (int(pad + 14 * scale), int(pad + 34 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, (255, 255, 255), max(1, int(2 * scale)))
        cv2.putText(vis, f'threshold      {THRESH:5.2f}', (int(pad + 14 * scale), int(pad + 64 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * scale, (170, 170, 170), max(1, int(2 * scale)))
        bw = int(300 * scale)
        bx, by = int(pad + 14 * scale), int(pad + 84 * scale)
        cv2.rectangle(vis, (bx, by), (bx + bw, by + int(16 * scale)), (70, 70, 70), 1)
        fill = int(bw * min(1.0, raised_frames / NEED))
        if fill:
            cv2.rectangle(vis, (bx, by), (bx + fill, by + int(16 * scale)),
                          (0, 0, 255) if firing else (0, 190, 255), -1)
        cv2.putText(vis, f'{CFG["R1_MIN_S"]:.0f}s gate', (bx + bw + int(10 * scale), by + int(14 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (170, 170, 170), max(1, int(1 * scale)))

        if firing:
            cv2.putText(vis, 'VIOLATION rule 1 - phone use', (pad, int(H - 26 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (0, 0, 255), max(2, int(3 * scale)))
        writer.write(vis)

    cap.release()
    writer.release()
    with open(args.events, 'w') as f:
        for e in events:
            f.write(json.dumps(e) + '\n')

    print(f'{processed} frames processed -> {args.out}')
    print(f'{len(events)} Rule 1 event(s) -> {args.events}')
    for e in events:
        print(f"   {e['start_s']}s - {e['end_s']}s  (closest ratio {e['ratio']})")
    if not events:
        print('   nothing fired. Hold the phone against your ear for a clear '
              f'{CFG["R1_MIN_S"]:.0f}+ seconds, facing the camera.')
    print('\nIf scratching your head also fired, that is the documented weakness '
          '(context.md §13.4), not a bug: the rule measures wrist-to-head distance '
          'and cannot see the phone. Show it deliberately.')


if __name__ == '__main__':
    main()
