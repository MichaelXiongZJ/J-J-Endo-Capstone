"""Pixels -> floor metres via homography (implementation guide §6.3).

This is the single most important module in the project. A camera flattens 3D
space into 2D, destroying distance: two boxes 200 px apart may be 1 m or 20 m
apart depending on depth, and a pedestrian 15 m BEHIND a forklift often has an
overlapping box purely from occlusion geometry.

So: never use box overlap, pixel gaps, or box size as a proxy for real
distance. Project each object's ground-contact point through the homography and
measure in metres (context.md §7.2).
"""

import json

import cv2
import numpy as np
from matplotlib.path import Path


class CameraGeometry:
    """One instance per camera, built from data/calibration/<cam>.json."""

    def __init__(self, config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        self.camera_id = cfg["camera_id"]
        img = np.float32(cfg["image_points"])
        flr = np.float32(cfg["floor_points"])
        assert len(img) == len(flr) >= 4, "need >=4 matched point pairs"
        self.image_points = img
        self.floor_points = flr
        self.H, _ = cv2.findHomography(img, flr)
        if self.H is None:
            raise ValueError(
                f"{config_path}: findHomography failed. Points are probably "
                "collinear — pick floor points that span an area, not a line."
            )
        self.walkways = []
        for i, poly in enumerate(cfg.get("walkways", [])):
            if len(poly) < 3:
                raise ValueError(
                    f"{config_path}: walkway {i} must have ≥3 points, got {len(poly)}."
                )
            for j, pt in enumerate(poly):
                if len(pt) != 2 or not all(isinstance(c, (int, float)) for c in pt):
                    raise ValueError(
                        f"{config_path}: walkway {i}, point {j} must be [x, y] numeric."
                    )
            self.walkways.append(Path(poly))
        self.vehicle_length_m = cfg.get("vehicle_length_m", 2.7)

    def to_floor(self, x, y):
        """Image point ON THE GROUND PLANE -> floor metres (x, y).

        Only valid for points actually on the floor. Feeding it a point in
        mid-air (e.g. a box centre) returns a confident, meaningless number.
        """
        out = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def floor_position(self, box):
        """box = (x1,y1,x2,y2). Uses BOTTOM-CENTRE — the ground-contact point.

        Using the box centre projects a point floating in mid-air: garbage.
        """
        x1, y1, x2, y2 = box
        return self.to_floor((x1 + x2) / 2.0, y2)

    def on_walkway(self, floor_xy):
        """True if this floor position is inside any walkway polygon (Rule 4).

        With no walkways configured this returns False for everything, which
        would make Rule 4 flag every pedestrian. run_pipeline skips Rule 4
        entirely when `walkways` is empty — see `has_walkways`.
        """
        return any(w.contains_point(floor_xy) for w in self.walkways)

    @property
    def has_walkways(self):
        return len(self.walkways) > 0

    def reprojection_error_m(self):
        """Mean error, in metres, of the calibration points through their own
        homography. This does NOT validate the calibration — with exactly 4
        points the fit is exact and the error is ~0 by construction even if
        every measurement is wrong. It only catches gross inconsistency when
        you supply 5+ points. The real check is §6.4: measure a distance you
        did not calibrate on.
        """
        proj = cv2.perspectiveTransform(self.image_points.reshape(-1, 1, 2), self.H)
        return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - self.floor_points, axis=1)))


def floor_dist(a, b):
    """Euclidean distance in floor metres between two (x, y) positions."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _cli():
    """Sanity check of §6.4 — mandatory before trusting any Rule 3 number.

    Usage:
        python -m src.geometry data/calibration/cam1.json
        python -m src.geometry data/calibration/cam1.json --pair 412 880 1180 875 6.0
    """
    import argparse

    ap = argparse.ArgumentParser(description=_cli.__doc__)
    ap.add_argument("config")
    ap.add_argument("--pair", nargs=5, type=float, action="append", metavar=("X1", "Y1", "X2", "Y2", "TRUE_M"),
                    help="two image points and their tape-measured distance in metres; repeatable")
    args = ap.parse_args()

    g = CameraGeometry(args.config)
    print(f"camera_id        : {g.camera_id}")
    print(f"point pairs      : {len(g.image_points)}")
    print(f"vehicle_length_m : {g.vehicle_length_m}")
    print(f"walkways         : {len(g.walkways)}")
    print(f"reprojection err : {g.reprojection_error_m():.3f} m "
          f"({'exact by construction — not a validation' if len(g.image_points) == 4 else 'meaningful'})")
    print(f"Rule 3 radius    : {3.0 * g.vehicle_length_m:.2f} m")

    if not args.pair:
        print("\nNo --pair given, so nothing was actually validated. Supply at least two "
              "pairs of points whose real separation you measured (§6.4 ACCEPTANCE CHECK).")
        return

    ok = True
    for x1, y1, x2, y2, true_m in args.pair:
        d = floor_dist(g.to_floor(x1, y1), g.to_floor(x2, y2))
        err = abs(d - true_m) / true_m if true_m else float("inf")
        flag = "PASS" if err <= 0.10 else "FAIL"
        if err > 0.10:
            ok = False
        print(f"({x1:.0f},{y1:.0f})->({x2:.0f},{y2:.0f}): computed {d:.2f} m, "
              f"true {true_m:.2f} m, error {err * 100:.1f}%  [{flag}]")
    if not ok:
        print("\nFAILED (>10% error). Most likely cause: image_points[i] does not "
              "correspond to floor_points[i] — same order is mandatory (§6.2). "
              "Second most likely: wide-angle lens distortion.")


if __name__ == "__main__":
    _cli()
