"""Shared helpers for the SDG-Warehouse slice: projection, clip loading, classes.

Isaac Sim / Omniverse conventions that matter here:

  * Matrices use the ROW-VECTOR convention, so `clip = [x,y,z,1] @ V @ P`, not
    `P @ V @ p`. Getting this backwards yields plausible-looking nonsense.
  * `metersPerSceneUnit: 1.0` — scene units are metres, so world coordinates are
    directly comparable to our floor metres. (The per-agent `speed` field is NOT
    consistent: characters report ~150 and vehicles ~1.5 for similar motion, so
    speed is derived from world_position deltas instead and `speed` is ignored.)
  * The floor is the z = 0 plane, and the camera is Z-up.
  * NDC y is flipped relative to image rows.

Validated against ground truth: projecting a character's `world_position` with
z forced to 0 lands within ~3 px of the bottom-centre of its own 2D box.
"""

import json
import os

import numpy as np

# The dataset labels exactly two things. They map 1:1 onto our classes, which is
# most of why this dataset is usable at all.
#
# CAVEAT: `robot` is a stand-on reach truck / walkie stacker with a mast and
# forks — NOT a sit-down counterbalance forklift. It has no enclosed cab, so
# Rule 5 ("driver keeps body inside vehicle") has no meaning on this footage.
SDG_CLASS_MAP = {'character': 'person', 'robot': 'forklift'}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class SDGClip:
    """One camera's worth of a run: rgb.mp4 + boxes + camera params."""

    def __init__(self, run_dir, camera):
        self.run_dir = run_dir
        self.camera = camera
        self.run_id = os.path.basename(run_dir.rstrip('/\\'))
        self.video = os.path.join(run_dir, f'{camera}.rgb.mp4')
        self.dets = load_jsonl(os.path.join(run_dir, f'{camera}.object_detection.jsonl'))
        self.cams = load_jsonl(os.path.join(run_dir, f'{camera}.camera_params.jsonl'))
        self.width, self.height = self.cams[0]['renderProductResolution']

    def __len__(self):
        return min(len(self.dets), len(self.cams))

    @property
    def is_static(self):
        """True if the camera never moves, so one calibration covers the clip."""
        first = self.cams[0]['cameraViewTransform']
        return all(c['cameraViewTransform'] == first for c in self.cams)

    def agents(self, frame):
        """[(our_class, (x1,y1,x2,y2), world_xy)] for one frame.

        world_xy is the agent's ground position in metres, from the simulator.

        IMPORTANT — it is read from `bounding_box_3d_fast.transform`, NOT from
        `metro_agent_data.world_position`. In 3 of the first 7 runs surveyed,
        world_position is frozen at its initial value for the whole clip while the
        2D boxes move hundreds of pixels; the 3D box transform tracks correctly in
        every run, and the two agree exactly wherever world_position is not
        broken. Trusting world_position silently marks ~40% of runs as "vehicle
        never moved", so every genuine violation in them becomes a false positive
        when scoring.
        """
        out = []
        d = self.dets[frame]
        for group in ('agents', 'objects'):
            for v in d.get(group, {}).values():
                cls = SDG_CLASS_MAP.get(v.get('label', {}).get('class'))
                if cls is None:
                    continue
                ann = v.get('annotators', {})
                bb = ann.get('bounding_box_2d_tight_fast')
                if not bb:
                    continue
                box = (float(bb['x_min']), float(bb['y_min']),
                       float(bb['x_max']), float(bb['y_max']))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                world_xy = None
                bb3 = ann.get('bounding_box_3d_fast')
                if bb3 and 'transform' in bb3:
                    t = np.asarray(bb3['transform'], dtype=np.float64).reshape(4, 4)
                    world_xy = (float(t[3, 0]), float(t[3, 1]))   # row-vector: translation in row 3
                out.append((cls, box, world_xy))
        return out

    def world_to_pixel(self, pts_xyz, frame=0):
        """(N,3) world metres -> (N,2) pixels. Row-vector convention."""
        cp = self.cams[frame]
        V = np.asarray(cp['cameraViewTransform'], dtype=np.float64).reshape(4, 4)
        P = np.asarray(cp['cameraProjection'], dtype=np.float64).reshape(4, 4)
        pts = np.atleast_2d(np.asarray(pts_xyz, dtype=np.float64))
        homog = np.hstack([pts, np.ones((len(pts), 1))])
        clip = homog @ V @ P
        w = clip[:, 3:4]
        ndc = clip[:, :2] / w
        u = (ndc[:, 0] * 0.5 + 0.5) * self.width
        v = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.height
        # Points behind the camera come back with w <= 0 and are meaningless.
        uv = np.stack([u, v], axis=1)
        uv[(w[:, 0] <= 0)] = np.nan
        return uv

    def floor_to_pixel(self, pts_xy, frame=0):
        """(N,2) floor metres (z=0) -> (N,2) pixels."""
        pts = np.atleast_2d(np.asarray(pts_xy, dtype=np.float64))
        return self.world_to_pixel(np.hstack([pts, np.zeros((len(pts), 1))]), frame)

    def visible_floor_points(self, extent=14.0, step=1.0, margin=60, frame=0):
        """Floor grid points (metres) that project inside the image.

        Used to choose calibration correspondences that actually span the
        visible floor — homography accuracy degrades outside the calibrated
        region, so spread matters more than count (§6.1).
        """
        g = np.arange(-extent, extent + 1e-9, step)
        grid = np.array([(x, y) for x in g for y in g])
        uv = self.floor_to_pixel(grid, frame)
        ok = (~np.isnan(uv).any(axis=1)
              & (uv[:, 0] > margin) & (uv[:, 0] < self.width - margin)
              & (uv[:, 1] > margin) & (uv[:, 1] < self.height - margin))
        return grid[ok], uv[ok]


def spread_quad(floor_pts, image_pts, n=4):
    """Pick `n` well-separated correspondences via farthest-point sampling.

    Points clustered together give an excellent fit locally and nonsense
    everywhere else, which is the most common calibration mistake after
    mis-ordering.

    Farthest-point sampling rather than the four axis extremes: when the visible
    floor is a narrow sliver (a camera aimed along an aisle), several axis
    extremes collide on the same point and you end up with fewer than 4 distinct
    correspondences. Greedy max-min distance always returns n distinct points
    when n are available, and spans the region regardless of its shape.
    """
    f = np.asarray(floor_pts, dtype=np.float64)
    img = np.asarray(image_pts, dtype=np.float64)
    if len(f) < n:
        return f.tolist(), img.tolist()

    # Seed with the point farthest from the centroid — a hull vertex.
    chosen = [int(np.argmax(np.linalg.norm(f - f.mean(axis=0), axis=1)))]
    while len(chosen) < n:
        d = np.min(np.linalg.norm(f[:, None, :] - f[chosen][None, :, :], axis=2), axis=1)
        nxt = int(np.argmax(d))
        if d[nxt] <= 1e-9:          # every remaining point duplicates one chosen
            break
        chosen.append(nxt)

    # Order them counter-clockwise so the correspondence order is stable and
    # human-readable in the JSON. Pairing is preserved either way.
    pts = f[chosen]
    ang = np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean())
    order = [chosen[i] for i in np.argsort(ang)]
    return f[order].tolist(), img[order].tolist()


def find_runs(root='data/sdg'):
    if not os.path.isdir(root):
        return []
    out = []
    for run in sorted(os.listdir(root)):
        d = os.path.join(root, run)
        if not os.path.isdir(d):
            continue
        cams = sorted({f.split('.')[0] for f in os.listdir(d) if f.endswith('.rgb.mp4')})
        cams = [c for c in cams
                if os.path.exists(os.path.join(d, f'{c}.object_detection.jsonl'))
                and os.path.exists(os.path.join(d, f'{c}.camera_params.jsonl'))]
        if cams:
            out.append((d, cams))
    return out
