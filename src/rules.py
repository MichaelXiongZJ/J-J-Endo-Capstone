"""The five rules as geometry over tracked objects (implementation guide §8).

None of this is machine learning. Rule 3 is "is the distance between these two
floor positions less than 8 metres, while the vehicle is moving?"; Rule 5 is
"are the driver's keypoints outside the cab region for more than 1.5 seconds?"
Both are arithmetic on top of detections (context.md §4).

All thresholds live in CFG. Tuning (§10) means editing these numbers only —
never retraining.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from src.geometry import floor_dist
from src.pose_utils import (NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER,
                            L_WRIST, R_WRIST, valid)

CFG = {
    'PROC_FPS':              10,     # processed frames/sec (must match pipeline + ByteTrack)
    'MOVING_MS':             0.3,    # m/s above which a vehicle is "moving"
    'RECENT_MOVE_S':         5.0,    # a vehicle that moved in the last 5 s is still "working"
    'R3_VEHICLE_LENGTHS':    3.0,    # from the rule text
    'DRIVER_OVERLAP':        0.6,    # person-box fraction inside vehicle box to be driver-candidate
    'DRIVER_VEL_MATCH_MS':   0.5,    # velocity agreement (m/s) => moving together
    'R5_CAB_FRACTIONS':      (0.15, 0.35, 0.15, 0.0),  # cab inset: left, top, right, bottom
    'R5_MIN_S':              1.5,    # body-outside duration before violation
    'R4_MIN_S':              1.0,    # off-walkway duration before violation
    'R1_WRIST_HEAD_RATIO':   0.6,    # wrist-to-head dist / shoulder width
    'R1_MIN_S':              2.0,
    'KPT_CONF':              0.5,
}

# Keypoints checked for Rule 5.
#
# NOSE is in, and this is now a settled ruling from J&J (2026-07-29), not a
# placeholder: the head is an important body part and must stay inside the
# forklift at all times. Drivers can see behind them from inside the cab, so a
# head-turn that puts the head outside the vehicle IS a violation.
#
# This reverses the guide's own advice (§12 troubleshooting: "Rule 5 fires on
# reversing driver -> remove NOSE from the CHECK set") and inverts the staged
# reversing head-turn clip from a negative case into a positive one. Do not
# "fix" a reversing-driver detection by dropping NOSE — that is now the
# specified behaviour.
R5_CHECK_KEYPOINTS = (NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST)


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    box: tuple                      # (x1, y1, x2, y2) pixels
    floor_xy: tuple                 # (x, y) metres
    keypoints: np.ndarray = None    # (17,3) for persons, else None
    worker_detections: dict = field(default_factory=dict)


# ---------- generic helpers ----------

def dist2d(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def point_in_box(pt, box):
    x1, y1, x2, y2 = box
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


def overlap_ratio(inner, outer):
    """Fraction of `inner` box's area inside `outer` box."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / a


# ---------- motion state (shared by everything) ----------

class MotionState:
    """Rolling floor-position history per track -> velocity/speed.

    Feed REAL timestamps from the video (§9) — never frame_number/30: dropped
    frames silently corrupt every velocity otherwise.
    """

    def __init__(self):
        n = int(1.0 * CFG['PROC_FPS'])                 # ~1 s window
        self.hist = defaultdict(lambda: deque(maxlen=n))
        self.last_moving_t = {}

    def update(self, obj: TrackedObject, t: float):
        self.hist[obj.track_id].append((t, obj.floor_xy))
        s = self.speed(obj.track_id)
        if s is not None and s > CFG['MOVING_MS']:
            self.last_moving_t[obj.track_id] = t

    def velocity(self, tid):
        """Mean floor velocity (m/s) over the window, or None if too few samples.

        Returning None rather than 0 matters: "we don't know yet" and "stopped"
        must be distinguishable, because find_driver treats them differently.
        """
        h = self.hist[tid]
        if len(h) < 4:
            return None
        (t0, p0), (t1, p1) = h[0], h[-1]
        dt = t1 - t0
        if dt <= 1e-3:
            return None
        return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)

    def speed(self, tid):
        v = self.velocity(tid)
        return None if v is None else float(np.hypot(*v))

    def is_working(self, tid, t):
        """A vehicle is "working" if moving now, or moved recently. A forklift
        paused for two seconds mid-manoeuvre is still a hazard.
        """
        s = self.speed(tid)
        if s is not None and s > CFG['MOVING_MS']:
            return True
        return (t - self.last_moving_t.get(tid, -1e9)) < CFG['RECENT_MOVE_S']


# ---------- driver association (feeds Rules 3 AND 5) ----------

def find_driver(vehicle, people, motion: MotionState):
    """The driver is the person who MOVES WITH the vehicle — not merely the one
    whose box overlaps it. Containment alone misfires: a pedestrian occluded
    BEHIND a forklift appears fully 'inside' its box.

    This one function is load-bearing for two rules: it prevents Rule 3 false
    positives AND identifies whose pose to check for Rule 5 (context.md §7.3).
    """
    v_vel = motion.velocity(vehicle.track_id)
    best, best_score = None, 0.0
    for p in people:
        if overlap_ratio(p.box, vehicle.box) < CFG['DRIVER_OVERLAP']:
            continue
        p_vel = motion.velocity(p.track_id)
        if v_vel is None or p_vel is None:
            score = 0.5                                # stationary: containment evidence only
        else:
            dv = np.hypot(v_vel[0] - p_vel[0], v_vel[1] - p_vel[1])
            score = 1.0 if dv < CFG['DRIVER_VEL_MATCH_MS'] else 0.0
        if score > best_score:
            best, best_score = p, score
    return best


# ---------- Rule 3: pedestrian near working vehicle ----------

def check_rule3(people, vehicles, driver_ids, motion, geom, t):
    """Metric distance on the FLOOR — never pixel gaps or box overlap.

    Known limitation: the rule's "unless signaled for recognition" exception is
    not visually detectable; all proximity events are flagged for human review
    (context.md §2).
    """
    radius = CFG['R3_VEHICLE_LENGTHS'] * geom.vehicle_length_m
    out = []
    for v in vehicles:
        if not motion.is_working(v.track_id, t):
            continue
        for p in people:
            if p.track_id in driver_ids:
                continue                               # the driver is not a pedestrian
            d = floor_dist(p.floor_xy, v.floor_xy)
            if d < radius:
                out.append({'rule': 3, 'person_track': p.track_id,
                            'vehicle_track': v.track_id,
                            'distance_m': round(d, 2), 'threshold_m': round(radius, 2),
                            'vehicle_speed_ms': round(motion.speed(v.track_id) or 0.0, 2)})
    return out


# ---------- Rule 5: driver body outside vehicle ----------

class Rule5State:
    def __init__(self):
        self.frames_outside = defaultdict(int)

    @staticmethod
    def cab_region(vbox):
        """Cab approximated as an inset of the vehicle box (the raw box is mostly
        mast/forks — empty space — so raw-box containment MISSES real lean-outs).
        Tune fractions per camera against footage.
        """
        l, tp, r, b = CFG['R5_CAB_FRACTIONS']
        x1, y1, x2, y2 = vbox
        w, h = x2 - x1, y2 - y1
        return (x1 + l * w, y1 + tp * h, x2 - r * w, y2 - b * h)

    def check(self, driver, vehicle):
        if driver is None or driver.keypoints is None:
            return None
        cab = self.cab_region(vehicle.box)
        outside = any(valid(driver.keypoints[i], CFG['KPT_CONF'])
                      and not point_in_box(driver.keypoints[i][:2], cab)
                      for i in R5_CHECK_KEYPOINTS)
        tid = driver.track_id
        self.frames_outside[tid] = self.frames_outside[tid] + 1 if outside else 0
        if self.frames_outside[tid] >= int(CFG['R5_MIN_S'] * CFG['PROC_FPS']):
            return {'rule': 5, 'driver_track': tid,
                    'vehicle_track': vehicle.track_id,
                    'seconds_outside': round(self.frames_outside[tid] / CFG['PROC_FPS'], 1)}
        return None


# ---------- Rule 4: pedestrians off walkways ----------

class Rule4State:
    def __init__(self):
        self.frames_off = defaultdict(int)

    def check(self, people, driver_ids, geom):
        # With no walkway polygons configured, on_walkway() is False everywhere
        # and every pedestrian would be flagged. Skip rather than lie.
        if not geom.has_walkways:
            return []
        out = []
        for p in people:
            if p.track_id in driver_ids:
                continue
            off = not geom.on_walkway(p.floor_xy)
            self.frames_off[p.track_id] = self.frames_off[p.track_id] + 1 if off else 0
            if self.frames_off[p.track_id] >= int(CFG['R4_MIN_S'] * CFG['PROC_FPS']):
                out.append({'rule': 4, 'person_track': p.track_id,
                            'floor_xy': [round(c, 2) for c in p.floor_xy]})
        return out


# ---------- Rule 1: phone use (weakest rule — expect false positives) ----------

class Rule1State:
    def __init__(self):
        self.frames_raised = defaultdict(int)

    def check(self, people):
        out = []
        for p in people:
            kp = p.keypoints
            if kp is None or not (valid(kp[L_SHOULDER], CFG['KPT_CONF'])
                                  and valid(kp[R_SHOULDER], CFG['KPT_CONF'])):
                continue
            # Shoulder width normalises for distance from camera: a wrist-to-head
            # gap of 40 px means different things at 3 m and 30 m.
            shoulder_w = max(1.0, dist2d(kp[L_SHOULDER], kp[R_SHOULDER]))
            heads = [kp[i] for i in (NOSE, L_EAR, R_EAR) if valid(kp[i], CFG['KPT_CONF'])]
            if not heads:
                continue
            raised = any(valid(kp[w], CFG['KPT_CONF']) and
                         min(dist2d(kp[w], h) for h in heads) / shoulder_w
                         < CFG['R1_WRIST_HEAD_RATIO']
                         for w in (L_WRIST, R_WRIST))
            self.frames_raised[p.track_id] = self.frames_raised[p.track_id] + 1 if raised else 0
            if self.frames_raised[p.track_id] >= int(CFG['R1_MIN_S'] * CFG['PROC_FPS']):
                out.append({'rule': 1, 'person_track': p.track_id})
        return out


# Rule 2 (daily pre-use inspection record) is NOT a vision task — a camera
# cannot see whether a checklist was completed. Descoped; say so explicitly to
# J&J (context.md §2).
