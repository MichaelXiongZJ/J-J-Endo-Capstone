"""Generate a synthetic warehouse clip with exactly known ground truth.

NOT part of the implementation guide, and NEVER a source of reported metrics —
§10's real validation is staged footage filmed with a safety supervisor. This
exists to answer a different question: "is the geometry and rule code correct?"

It builds a scene from floor coordinates in metres, projects it into a camera
image through a known homography, and emits:

  * outputs/synthetic/synthetic_cam1.mp4   — renderable video
  * data/calibration/synthetic_cam1.json   — calibration matching that camera
  * outputs/synthetic/detections.json      — per-frame boxes for StubDetector
  * outputs/synthetic/ground_truth.json    — when each rule should fire, and why

Because the floor positions are exact, the expected Rule 3 trigger time is
arithmetic, not a guess — which makes it a real regression test for the
homography, the tracker, the driver association, and the duration gates.

The scene (floor is 24 m x 30 m, origin at the near-left corner):

  forklift  drives (12, 26) -> (12, 8) over 20 s = 0.9 m/s  ["working"]
  driver    rides on the forklift                            [must be excluded]
  ped A     stands still at (14.5, 12), OFF the walkway      [Rule 3 + Rule 4]
  ped B     walks the walkway at x=1.5, y 4 -> 26            [must trigger NOTHING]

Ped B is the important one. A test where everything fires proves very little;
B is >= 10.5 m from the forklift at all times and always inside the walkway
polygon, so any event naming B is a false positive.

Usage:
    python -m scripts.make_synthetic_clip
"""

import json
import os

import cv2
import numpy as np

W, H = 1280, 720
SRC_FPS = 30
DURATION_S = 20.0

# Floor extent used for the camera model, in metres.
FLOOR_CORNERS = [[0.0, 0.0], [24.0, 0.0], [24.0, 30.0], [0.0, 30.0]]
# Where those corners land in the image: a plausible perspective trapezoid for
# a camera mounted high on a wall looking down the aisle.
IMAGE_CORNERS = [[140, 700], [1140, 700], [880, 250], [400, 250]]

WALKWAY = [[0.0, 0.0], [3.0, 0.0], [3.0, 30.0], [0.0, 30.0]]
VEHICLE_LENGTH_M = 2.7

PERSON_ID, FORKLIFT_ID = 2, 1

OUT_DIR = 'outputs/synthetic'
CALIB_PATH = 'data/calibration/synthetic_cam1.json'

# floor metres -> image pixels (the inverse of what the pipeline computes)
H_FLOOR_TO_IMG, _ = cv2.findHomography(np.float32(FLOOR_CORNERS), np.float32(IMAGE_CORNERS))


def to_img(x, y):
    p = cv2.perspectiveTransform(np.float32([[[x, y]]]), H_FLOOR_TO_IMG)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def px_per_metre(x, y):
    """Lateral image scale at this floor position, used to size objects.

    A homography maps the FLOOR plane only; it says nothing about vertical
    extent. Approximating object height with the local lateral scale is
    geometrically crude but perfectly adequate here — the rules only ever use
    the bottom-centre of a box, which IS on the floor and IS exact.
    """
    a = to_img(x - 0.5, y)
    b = to_img(x + 0.5, y)
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def box_at(x, y, width_m, height_m):
    """Axis-aligned image box for an object standing on the floor at (x, y)."""
    cx, by = to_img(x, y)
    s = px_per_metre(x, y)
    half_w = 0.5 * width_m * s
    return (cx - half_w, by - height_m * s, cx + half_w, by)


# ---------- the scene ----------

def forklift_pos(t):
    return (12.0, 26.0 - 0.9 * t)          # 0.9 m/s, "moving" (> MOVING_MS 0.3)


def driver_pos(t):
    # The driver rides the vehicle. Their box bottom is placed at ground level:
    # a real detector boxing a seated driver whose legs are visible down to the
    # footplate gives roughly this. What matters for find_driver is that the
    # driver's floor velocity MATCHES the forklift's, which it does exactly.
    return forklift_pos(t)


PED_A = (14.5, 12.0)                        # static, off walkway


def ped_b_pos(t):
    return (1.5, 4.0 + 1.1 * t)             # on the walkway, far from the forklift


def scene_at(t):
    """Returns list of (x1, y1, x2, y2, class_id, confidence)."""
    fx, fy = forklift_pos(t)
    dx, dy = driver_pos(t)
    bx, by = ped_b_pos(t)
    return [
        (*box_at(fx, fy, 1.4, 2.3), FORKLIFT_ID, 0.95),
        (*box_at(dx, dy, 0.6, 1.75), PERSON_ID, 0.92),
        (*box_at(*PED_A, 0.6, 1.75), PERSON_ID, 0.90),
        (*box_at(bx, by, 0.6, 1.75), PERSON_ID, 0.88),
    ]


COLORS = {FORKLIFT_ID: (40, 140, 240), PERSON_ID: (90, 200, 90)}


def render(frame, rows):
    # Floor grid, so the perspective is visually obvious in the output video.
    for gx in range(0, 25, 3):
        cv2.line(frame, tuple(map(int, to_img(gx, 0))), tuple(map(int, to_img(gx, 30))),
                 (60, 60, 60), 1)
    for gy in range(0, 31, 3):
        cv2.line(frame, tuple(map(int, to_img(0, gy))), tuple(map(int, to_img(24, gy))),
                 (60, 60, 60), 1)
    poly = np.int32([to_img(*p) for p in WALKWAY])
    cv2.polylines(frame, [poly], True, (0, 220, 220), 2)
    for x1, y1, x2, y2, cid, _ in rows:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      COLORS[cid], -1)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
    return frame


def expected_rule3_start():
    """Solve for when ped A first comes within 3 vehicle lengths of the forklift."""
    radius = 3.0 * VEHICLE_LENGTH_M
    ax, ay = PED_A
    for i in range(int(DURATION_S * SRC_FPS) + 1):
        t = i / SRC_FPS
        fx, fy = forklift_pos(t)
        if np.hypot(ax - fx, ay - fy) < radius:
            return round(t, 2)
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)

    with open(CALIB_PATH, 'w') as f:
        json.dump({
            'camera_id': 'synthetic_cam1',
            '_comment': 'Synthetic test camera generated by scripts/make_synthetic_clip.py. '
                        'Not a real calibration; do not use for reported metrics.',
            'image_points': IMAGE_CORNERS,
            'floor_points': FLOOR_CORNERS,
            'walkways': [WALKWAY],
            'vehicle_length_m': VEHICLE_LENGTH_M,
        }, f, indent=2)

    video_path = f'{OUT_DIR}/synthetic_cam1.mp4'
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), SRC_FPS, (W, H))
    n_frames = int(DURATION_S * SRC_FPS)
    for i in range(n_frames):
        frame = np.full((H, W, 3), 25, np.uint8)
        writer.write(render(frame, scene_at(i / SRC_FPS)))
    writer.release()

    # Detections keyed by PROCESSED frame index. The pipeline reads every frame,
    # increments frame_idx from 1, and processes when frame_idx % stride == 0
    # (stride = 30/10 = 3). So processed index k <-> source frame 3*(k+1).
    stride = SRC_FPS // 10
    script = {}
    k = 0
    for frame_idx in range(1, n_frames + 1):
        if frame_idx % stride:
            continue
        script[k] = [list(map(float, r)) for r in scene_at(frame_idx / SRC_FPS)]
        k += 1

    with open(f'{OUT_DIR}/detections.json', 'w') as f:
        json.dump(script, f)

    gt = {
        'video': video_path,
        'calib': CALIB_PATH,
        'duration_s': DURATION_S,
        'processed_frames': k,
        'rule3_expected_start_s': expected_rule3_start(),
        'rule3_expected_min_distance_m': round(
            min(float(np.hypot(PED_A[0] - forklift_pos(i / SRC_FPS)[0],
                               PED_A[1] - forklift_pos(i / SRC_FPS)[1]))
                for i in range(n_frames)), 2),
        'rule3_threshold_m': round(3.0 * VEHICLE_LENGTH_M, 2),
        'notes': [
            'Ped A (static, 14.5/12) must trigger Rule 3 and Rule 4.',
            'Ped B (walkway, x=1.5) must trigger NOTHING — it is the false-positive check.',
            'The driver must be identified and excluded from Rules 3 and 4.',
            'Rules 5 and 1 cannot fire: rendered rectangles have no pose. Run with --no-pose.',
        ],
    }
    with open(f'{OUT_DIR}/ground_truth.json', 'w') as f:
        json.dump(gt, f, indent=2)

    print(f'video       : {video_path}  ({n_frames} frames @ {SRC_FPS}fps)')
    print(f'calibration : {CALIB_PATH}')
    print(f'detections  : {OUT_DIR}/detections.json  ({k} processed frames)')
    print(f'ground truth: {OUT_DIR}/ground_truth.json')
    print(f'\nRule 3 should first fire at t={gt["rule3_expected_start_s"]}s '
          f'(threshold {gt["rule3_threshold_m"]} m, closest approach '
          f'{gt["rule3_expected_min_distance_m"]} m)')


if __name__ == '__main__':
    main()
