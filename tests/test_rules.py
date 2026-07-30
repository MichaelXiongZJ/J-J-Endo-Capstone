"""Rule logic tests.

Every rule's *negative* case is tested alongside its positive one, because
precision is prioritised over recall (context.md §7.6): a system that cries
wolf gets muted within a week, and a muted system has zero recall on everything.
"""

import pytest

from src.rules import (CFG, MotionState, Rule1State, Rule4State, Rule5State,
                       TrackedObject, check_rule3, find_driver, overlap_ratio,
                       point_in_box)


class FakeGeom:
    """Identity 'homography': floor coords are handed in directly, so these
    tests exercise rule logic in isolation from calibration."""

    camera_id = 'fake'
    vehicle_length_m = 2.7

    def __init__(self, walkways=None):
        self._w = walkways or []

    @property
    def has_walkways(self):
        return bool(self._w)

    def on_walkway(self, xy):
        return any(x1 <= xy[0] <= x2 and y1 <= xy[1] <= y2 for x1, y1, x2, y2 in self._w)


def obj(tid, cid, box, floor, kp=None):
    return TrackedObject(tid, cid, box, floor, kp)


def feed(motion, o, positions, t0=0.0, dt=0.1):
    """Push a floor-position history so velocity() has enough samples."""
    for i, p in enumerate(positions):
        o.floor_xy = p
        motion.update(o, t0 + i * dt)
    return o


# ---------- helpers ----------

def test_overlap_ratio_is_fraction_of_inner_not_iou():
    inner, outer = (10, 10, 20, 20), (0, 0, 100, 100)
    assert overlap_ratio(inner, outer) == pytest.approx(1.0)
    assert overlap_ratio((0, 0, 20, 20), (10, 10, 30, 30)) == pytest.approx(0.25)
    assert overlap_ratio((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_point_in_box():
    assert point_in_box((5, 5), (0, 0, 10, 10))
    assert not point_in_box((15, 5), (0, 0, 10, 10))


# ---------- motion ----------

def test_velocity_is_none_until_enough_samples():
    m = MotionState()
    o = obj(1, 2, (0, 0, 1, 1), (0.0, 0.0))
    m.update(o, 0.0)
    assert m.velocity(1) is None, "must distinguish 'unknown' from 'stopped'"
    feed(m, o, [(0, 0), (0, 0.1), (0, 0.2), (0, 0.3)])
    assert m.velocity(1) is not None


def test_speed_from_real_timestamps():
    m = MotionState()
    o = obj(1, 1, (0, 0, 1, 1), (0.0, 0.0))
    # 1 m/s along +x, sampled at 10 Hz
    feed(m, o, [(i * 0.1, 0.0) for i in range(10)])
    assert m.speed(1) == pytest.approx(1.0, abs=0.05)


def test_is_working_covers_recently_stopped_vehicle():
    """A forklift paused mid-manoeuvre is still a hazard (RECENT_MOVE_S).

    Note the stationary frames fed after the moving ones: is_working() reads a
    rolling history, so "has stopped" only becomes visible once enough
    stationary samples have pushed the moving ones out of the window. The
    pipeline calls motion.update() every frame for every visible track, so this
    is what actually happens — a test that just advances `t` without feeding
    frames would leave stale motion in the deque and report movement forever.
    """
    m = MotionState()
    v = obj(7, 1, (0, 0, 1, 1), (0.0, 0.0))
    feed(m, v, [(i * 0.1, 0.0) for i in range(10)])          # moving, t = 0.0..0.9
    assert m.is_working(7, 0.9)

    feed(m, v, [(0.9, 0.0)] * 20, t0=1.0)                     # parked, t = 1.0..2.9
    assert m.speed(7) == pytest.approx(0.0, abs=0.05)
    stopped_at = m.last_moving_t[7]
    assert m.is_working(7, stopped_at + CFG['RECENT_MOVE_S'] - 0.5)
    assert not m.is_working(7, stopped_at + CFG['RECENT_MOVE_S'] + 0.5)


def test_stationary_vehicle_is_not_working():
    m = MotionState()
    v = obj(7, 1, (0, 0, 1, 1), (5.0, 5.0))
    feed(m, v, [(5.0, 5.0)] * 10)
    assert not m.is_working(7, 1.0)


# ---------- driver association (context.md §7.3) ----------

VBOX = (100, 100, 300, 400)


def test_driver_identified_by_moving_with_vehicle():
    m = MotionState()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    d = obj(2, 2, (150, 150, 250, 380), (0.0, 0.0))
    for i in range(10):
        p = (i * 0.1, 0.0)
        v.floor_xy = p
        d.floor_xy = (p[0] + 0.05, p[1])
        m.update(v, i * 0.1)
        m.update(d, i * 0.1)
    assert find_driver(v, [d], m) is d


def test_pedestrian_occluded_behind_vehicle_is_not_the_driver():
    """THE case containment-only logic gets wrong: a pedestrian standing behind
    a forklift appears fully 'inside' its box, but moves independently.
    """
    m = MotionState()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    ped = obj(2, 2, (150, 150, 250, 380), (0.0, 0.0))     # fully contained
    assert overlap_ratio(ped.box, v.box) >= CFG['DRIVER_OVERLAP']
    for i in range(10):
        v.floor_xy = (i * 0.1, 0.0)                        # vehicle +x at 1 m/s
        ped.floor_xy = (0.0, i * 0.15)                     # pedestrian +y at 1.5 m/s
        m.update(v, i * 0.1)
        m.update(ped, i * 0.1)
    assert find_driver(v, [ped], m) is None


def test_non_overlapping_person_is_never_driver_candidate():
    m = MotionState()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    far = obj(2, 2, (600, 100, 700, 400), (9.0, 9.0))
    assert find_driver(v, [far], m) is None


def test_driver_chosen_over_bystander_when_both_overlap():
    m = MotionState()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    drv = obj(2, 2, (150, 150, 250, 380), (0.0, 0.0))
    bystander = obj(3, 2, (110, 150, 210, 380), (0.0, 0.0))
    for i in range(10):
        t = i * 0.1
        v.floor_xy = (t, 0.0)
        drv.floor_xy = (t, 0.0)
        bystander.floor_xy = (0.0, t * 2)
        for o in (v, drv, bystander):
            m.update(o, t)
    assert find_driver(v, [bystander, drv], m) is drv


# ---------- Rule 3 ----------

def test_rule3_fires_inside_radius_and_not_outside():
    g = FakeGeom()
    radius = CFG['R3_VEHICLE_LENGTHS'] * g.vehicle_length_m      # 8.1 m
    m = MotionState()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    feed(m, v, [(i * 0.1, 0.0) for i in range(10)])
    v.floor_xy = (0.0, 0.0)

    near = obj(2, 2, (500, 100, 550, 400), (radius - 1.0, 0.0))
    far = obj(3, 2, (900, 100, 950, 400), (radius + 1.0, 0.0))

    hits = check_rule3([near, far], [v], set(), m, g, 0.9)
    assert [h['person_track'] for h in hits] == [2]
    assert hits[0]['distance_m'] == pytest.approx(radius - 1.0, abs=0.01)
    assert hits[0]['threshold_m'] == pytest.approx(radius, abs=0.01)


def test_rule3_excludes_the_driver():
    """Without this the driver reads as 'a pedestrian 0 m from a moving forklift'."""
    g = FakeGeom()
    m = MotionState()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    feed(m, v, [(i * 0.1, 0.0) for i in range(10)])
    v.floor_xy = (0.0, 0.0)
    drv = obj(2, 2, (150, 150, 250, 380), (0.0, 0.0))
    assert check_rule3([drv], [v], {2}, m, g, 0.9) == []
    assert len(check_rule3([drv], [v], set(), m, g, 0.9)) == 1   # proves exclusion did it


def test_rule3_ignores_parked_vehicle():
    """'No pedestrian within 3 vehicle lengths of WORKING vehicles.' A parked
    forklift is not a hazard, and flagging it would flood the system."""
    g = FakeGeom()
    m = MotionState()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    feed(m, v, [(0.0, 0.0)] * 10)
    ped = obj(2, 2, (500, 100, 550, 400), (1.0, 0.0))
    assert check_rule3([ped], [v], set(), m, g, 0.9) == []


# ---------- Rule 5 ----------

def inside_cab_kpts(kpts):
    """All checked joints comfortably inside cab_region(VBOX) = (130,205,270,400)."""
    return kpts(nose=(200, 250), l_shoulder=(180, 280), r_shoulder=(220, 280),
                l_wrist=(175, 330), r_wrist=(225, 330))


def test_rule5_requires_sustained_duration(kpts):
    r5 = Rule5State()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    leaning = inside_cab_kpts(kpts)
    leaning[10] = (60, 330, 0.9)                  # R_WRIST far outside the cab
    d = obj(2, 2, (150, 150, 250, 380), (0.0, 0.0), leaning)

    need = int(CFG['R5_MIN_S'] * CFG['PROC_FPS'])
    for _ in range(need - 1):
        assert r5.check(d, v) is None, 'must not fire before the duration gate'
    ev = r5.check(d, v)
    assert ev is not None and ev['rule'] == 5 and ev['driver_track'] == 2
    assert ev['seconds_outside'] == pytest.approx(CFG['R5_MIN_S'], abs=0.11)


def test_rule5_silent_when_driver_inside_cab(kpts):
    r5 = Rule5State()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    d = obj(2, 2, (150, 150, 250, 380), (0.0, 0.0), inside_cab_kpts(kpts))
    for _ in range(60):
        assert r5.check(d, v) is None


def test_rule5_counter_resets_on_brief_reach(kpts):
    """'A driver briefly reaching for a control is not keeping their body
    outside the vehicle' (context.md §7.5)."""
    r5 = Rule5State()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    out = inside_cab_kpts(kpts); out[10] = (60, 330, 0.9)
    inside = inside_cab_kpts(kpts)
    d = obj(2, 2, (150, 150, 250, 380), (0.0, 0.0))

    for _ in range(3):                      # reach out briefly...
        d.keypoints = out
        r5.check(d, v)
    d.keypoints = inside                    # ...then back in
    r5.check(d, v)
    assert r5.frames_outside[2] == 0


def test_rule5_ignores_low_confidence_keypoints(kpts):
    """Occluded joints return invented coordinates. Trusting them means
    violations triggered by hallucinated limbs (context.md §8.5)."""
    r5 = Rule5State()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    k = inside_cab_kpts(kpts)
    k[10] = (60, 330, 0.1)                  # way outside, but unreliable
    d = obj(2, 2, (150, 150, 250, 380), (0.0, 0.0), k)
    for _ in range(60):
        assert r5.check(d, v) is None


def test_rule5_no_pose_means_no_decision(kpts):
    r5 = Rule5State()
    v = obj(1, 1, VBOX, (0.0, 0.0))
    assert r5.check(obj(2, 2, VBOX, (0, 0), None), v) is None
    assert r5.check(None, v) is None


def test_cab_region_is_inset_of_vehicle_box():
    """The raw box includes mast and overhead guard — mostly empty space — so
    raw-box containment would MISS real lean-outs (context.md §7.4)."""
    cab = Rule5State.cab_region(VBOX)
    assert cab[0] > VBOX[0] and cab[2] < VBOX[2]
    assert cab[1] > VBOX[1]
    assert cab[3] == VBOX[3]                # bottom not inset


# ---------- Rule 4 ----------

def test_rule4_fires_off_walkway_after_duration():
    g = FakeGeom(walkways=[(0, 0, 2, 20)])
    r4 = Rule4State()
    ped = obj(2, 2, (0, 0, 1, 1), (5.0, 5.0))
    need = int(CFG['R4_MIN_S'] * CFG['PROC_FPS'])
    for _ in range(need - 1):
        assert r4.check([ped], set(), g) == []
    assert len(r4.check([ped], set(), g)) == 1


def test_rule4_silent_on_walkway():
    g = FakeGeom(walkways=[(0, 0, 2, 20)])
    r4 = Rule4State()
    ped = obj(2, 2, (0, 0, 1, 1), (1.0, 5.0))
    for _ in range(60):
        assert r4.check([ped], set(), g) == []


def test_rule4_skipped_when_no_walkways_configured():
    """Otherwise on_walkway() is False everywhere and EVERY pedestrian is flagged."""
    r4 = Rule4State()
    ped = obj(2, 2, (0, 0, 1, 1), (5.0, 5.0))
    for _ in range(60):
        assert r4.check([ped], set(), FakeGeom(walkways=[])) == []


def test_rule4_excludes_driver():
    g = FakeGeom(walkways=[(0, 0, 2, 20)])
    r4 = Rule4State()
    drv = obj(2, 2, (0, 0, 1, 1), (5.0, 5.0))
    for _ in range(60):
        assert r4.check([drv], {2}, g) == []


# ---------- Rule 1 ----------

def test_rule1_fires_on_wrist_at_head(kpts):
    r1 = Rule1State()
    k = kpts(l_shoulder=(180, 280), r_shoulder=(220, 280), nose=(200, 250),
             r_wrist=(205, 255))
    p = obj(2, 2, (150, 200, 250, 400), (0, 0), k)
    need = int(CFG['R1_MIN_S'] * CFG['PROC_FPS'])
    for _ in range(need - 1):
        assert r1.check([p]) == []
    assert len(r1.check([p])) == 1


def test_rule1_silent_when_wrist_at_side(kpts):
    r1 = Rule1State()
    k = kpts(l_shoulder=(180, 280), r_shoulder=(220, 280), nose=(200, 250),
             r_wrist=(230, 400))
    p = obj(2, 2, (150, 200, 250, 400), (0, 0), k)
    for _ in range(60):
        assert r1.check([p]) == []


def test_rule1_normalises_by_shoulder_width(kpts):
    """A 40 px wrist-to-head gap means different things at 3 m and 30 m from the
    camera. Two geometrically identical people at different scales must agree.
    """
    r1a, r1b = Rule1State(), Rule1State()
    near = kpts(l_shoulder=(100, 200), r_shoulder=(200, 200), nose=(150, 150),
                r_wrist=(160, 160))
    small = kpts(l_shoulder=(100, 200), r_shoulder=(110, 200), nose=(105, 195),
                 r_wrist=(106, 196))
    need = int(CFG['R1_MIN_S'] * CFG['PROC_FPS'])
    for _ in range(need):
        a = r1a.check([obj(2, 2, (0, 0, 1, 1), (0, 0), near)])
        b = r1b.check([obj(3, 2, (0, 0, 1, 1), (0, 0), small)])
    assert bool(a) == bool(b) is True


def test_rule1_needs_both_shoulders_and_a_head_point(kpts):
    r1 = Rule1State()
    no_shoulders = kpts(nose=(200, 250), r_wrist=(205, 255))
    no_head = kpts(l_shoulder=(180, 280), r_shoulder=(220, 280), r_wrist=(205, 255))
    for _ in range(60):
        assert r1.check([obj(2, 2, (0, 0, 1, 1), (0, 0), no_shoulders)]) == []
        assert r1.check([obj(3, 2, (0, 0, 1, 1), (0, 0), no_head)]) == []
