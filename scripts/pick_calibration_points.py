"""Click floor points to build a calibration file (implementation guide §6.1).

Run locally (needs a display). Click >= 4 points on the floor whose real-world
positions you can measure, then type the metre coordinates for each.

Choose points that SPREAD ACROSS the whole area you care about — homography
accuracy degrades outside the calibrated region, so four points clustered in one
corner will give good numbers there and nonsense everywhere else. Walkway-marking
corners are ideal: they are flat, high-contrast, and easy to tape-measure.

The order you click is the order you enter metres, and image_points[i] must pair
with floor_points[i]. Mis-ordering is the most common calibration failure and
produces silently absurd distances (context.md §8.6) — this script keeps them
paired for you, which is most of why it exists.

Usage:
    python -m scripts.pick_calibration_points data/calibration/cam1_ref.jpg \
        --camera-id cam1 --out data/calibration/cam1.json

Controls:  left-click = add point | u = undo | q / Enter = done | Esc = abort
"""

import argparse
import json
import os

import cv2

points = []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image', help='one representative frame from the camera')
    ap.add_argument('--camera-id', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--vehicle-length-m', type=float, default=2.7,
                    help='measure a real J&J forklift; 2.7 is a placeholder (§14 Q2)')
    args = ap.parse_args()

    out_path = args.out or f'data/calibration/{args.camera_id}.json'

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f'cannot read image: {args.image}')
    win = f'{args.camera_id}: click floor points (u=undo, q=done, Esc=abort)'

    def redraw():
        disp = img.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(disp, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
        if len(points) > 1:
            for a, b in zip(points, points[1:]):
                cv2.line(disp, a, b, (0, 255, 255), 1)
        cv2.imshow(win, disp)

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f'  point {len(points) - 1}: ({x}, {y})')
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)
    redraw()

    print('Click floor points, spread widely across the area of interest.')
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 13):
            break
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit('aborted; nothing written')
        if k == ord('u') and points:
            print(f'  undo point {len(points) - 1}')
            points.pop()
            redraw()
    cv2.destroyAllWindows()

    if len(points) < 4:
        raise SystemExit(f'need >= 4 points, got {len(points)}')

    print(f'\nNow enter the real floor position of each point, in metres.')
    print('Use any consistent origin and axes — e.g. a corner of the aisle as (0, 0).')
    floor = []
    for i, (x, y) in enumerate(points):
        while True:
            raw = input(f'  point {i} at pixel ({x}, {y}) -> floor "X Y" in metres: ')
            try:
                fx, fy = (float(v) for v in raw.replace(',', ' ').split())
                floor.append([fx, fy])
                break
            except ValueError:
                print('    expected two numbers, e.g. "6.0 8.0"')

    cfg = {
        'camera_id': args.camera_id,
        'image_points': [[int(x), int(y)] for x, y in points],
        'floor_points': floor,
        'walkways': [],
        'vehicle_length_m': args.vehicle_length_m,
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'\nwrote {out_path}')

    print('\nNEXT — both steps are mandatory:')
    print(f'  1. Validate against a distance you did NOT calibrate on (§6.4):')
    print(f'       python -m src.geometry {out_path} --pair X1 Y1 X2 Y2 TRUE_METRES')
    print(f'     Must be within 10%. If not, suspect point-order mismatch first.')
    print(f'  2. Add walkway polygons in floor metres to "walkways" for Rule 4. '
          f'Left empty, Rule 4 is skipped rather than flagging everyone.')


if __name__ == '__main__':
    main()
