"""Rule 5 validation matrix from §10, as executable scenarios.

Rule 5 ("vehicle drivers keep body inside vehicle at all times") is the project's
most important rule and the one with no footage behind it: the SDG-Warehouse
slice has stand-on reach trucks with no cab, so it cannot exercise Rule 5 at all.

These scenarios script keypoint sequences directly, which validates everything
downstream of pose estimation — the cab region, the duration gate, the confidence
gate, and the J&J head-turn ruling — before any imagery exists. What they do NOT
test is RTMPose itself: whether a real driver's joints are found accurately at
warehouse camera distances is an empirical question that needs real frames.

Each scenario mirrors one row of §10's staged-clip table, so when the footage is
filmed these tests say what the expected outcome is.
"""

import numpy as np
import pytest

from src.pose_utils import (L_SHOULDER, L_WRIST, NOSE, R_SHOULDER, R_WRIST)
from src.rules import CFG, Rule5State, TrackedObject

# A forklift box and the cab region it implies, with CFG's default insets
# (0.15, 0.35, 0.15, 0.0): cab = (130, 205, 270, 400).
VBOX = (100.0, 100.0, 300.0, 400.0)
CAB = Rule5State.cab_region(VBOX)

FRAMES_TO_FIRE = int(CFG['R5_MIN_S'] * CFG['PROC_FPS'])      # 15 frames = 1.5 s


def kp(joints):
    """(17,3) keypoints from {index: (x, y[, conf])}.

    Unspecified joints stay at zero confidence, i.e. unusable — which mirrors
    reality, where most joints on a seated, partly occluded driver genuinely are.
    """
    a = np.zeros((17, 3), dtype=np.float32)
    for idx, val in joints.items():
        a[idx] = (val[0], val[1], val[2] if len(val) > 2 else 0.9)
    return a


def seated_inside():
    """A driver seated normally, every checked joint well inside the cab."""
    return kp({NOSE: (200, 250), L_SHOULDER: (180, 285), R_SHOULDER: (220, 285),
               L_WRIST: (170, 330), R_WRIST: (230, 330)})


def play(frames, vbox=VBOX):
    """Run a keypoint sequence through Rule5State; return the first event or None."""
    r5 = Rule5State()
    vehicle = TrackedObject(1, 1, vbox, (0.0, 0.0))
    first = None
    for k in frames:
        driver = TrackedObject(2, 2, (150, 150, 250, 380), (0.0, 0.0), k)
        ev = r5.check(driver, vehicle)
        if ev and first is None:
            first = ev
    return first


def held(mutate, n=FRAMES_TO_FIRE + 5):
    """A pose held for n frames. `mutate` edits a seated baseline."""
    out = []
    for _ in range(n):
        k = seated_inside()
        mutate(k)
        out.append(k)
    return out


# ---------- positives: the violation, in its three filmed variants ----------

def test_arm_only_out_fires():
    """§10 'Driver leans out: arm only'. The weakest positive — a single wrist
    outside the cab is still a body part outside the vehicle."""
    ev = play(held(lambda k: k.__setitem__(R_WRIST, (60, 330, 0.9))))
    assert ev is not None and ev['rule'] == 5


def test_torso_out_fires():
    """§10 'Driver leans out: torso'. Shoulders and head swing outside together."""
    def lean(k):
        k[NOSE] = (70, 250, 0.9)
        k[L_SHOULDER] = (95, 285, 0.9)
        k[R_SHOULDER] = (120, 285, 0.9)
    assert play(held(lean)) is not None


def test_head_out_while_reversing_fires():
    """THE ruling case (J&J, 2026-07-29): a head-turn while reversing that puts
    the head outside the vehicle IS a violation — drivers can see behind them
    from inside the cab.

    This reverses the guide, which treats reversing head-turns as the
    make-or-break FALSE-POSITIVE case and advises removing NOSE from the checked
    keypoints. If this test ever starts failing because NOSE was dropped, the
    ruling was overridden by accident.
    """
    ev = play(held(lambda k: k.__setitem__(NOSE, (105, 240, 0.9))))
    assert ev is not None, 'head outside the cab must fire — see R5_CHECK_KEYPOINTS'
    assert ev['rule'] == 5


# ---------- negatives: precision is the PoC gate, so these matter more ----------

def test_head_turn_inside_cab_does_not_fire():
    """The paired control: driver reverses and looks behind while keeping the head
    INSIDE the cab. The gap between this and the test above is the tightest
    tolerance Rule 5 has to resolve, and it is why both clips must be filmed from
    the same camera position."""
    assert play(held(lambda k: k.__setitem__(NOSE, (150, 240, 0.9)))) is None


def test_normal_seated_driving_does_not_fire():
    """§10 'Plain safe operation' — nothing fires at all."""
    assert play([seated_inside() for _ in range(120)]) is None


def test_brief_reach_for_control_does_not_fire():
    """context.md §7.5: 'a driver briefly reaching for a control is not keeping
    their body outside the vehicle.' Reaching out for under R5_MIN_S must not
    fire, however far out the wrist goes."""
    frames = []
    for i in range(60):
        k = seated_inside()
        if i % 20 < FRAMES_TO_FIRE - 1:        # always shorter than the gate
            k[R_WRIST] = (55, 330, 0.9)
        frames.append(k)
    assert play(frames) is None


def test_occluded_lower_body_does_not_fire():
    """A seated driver's legs are hidden by the cab, so leg joints return low
    confidence and invented coordinates. Those must never trigger a violation
    (context.md §8.5) — this is the confidence gate doing its job."""
    def occluded(k):
        k[13] = (20, 700, 0.12)                # left knee, "outside", unreliable
        k[15] = (10, 750, 0.08)                # left ankle
    assert play(held(occluded, n=120)) is None


# ---------- the duration gate itself ----------

def test_fires_exactly_at_the_duration_threshold():
    """Not one frame early, not one late — the gate is what separates a genuine
    lean-out from detection flicker."""
    lean = lambda k: k.__setitem__(R_WRIST, (60, 330, 0.9))
    assert play(held(lean, n=FRAMES_TO_FIRE - 1)) is None
    ev = play(held(lean, n=FRAMES_TO_FIRE))
    assert ev is not None
    assert ev['seconds_outside'] == pytest.approx(CFG['R5_MIN_S'], abs=0.11)


def test_event_reports_both_tracks_for_review():
    """A safety reviewer needs to know which driver and which vehicle."""
    ev = play(held(lambda k: k.__setitem__(R_WRIST, (60, 330, 0.9))))
    assert ev['driver_track'] == 2 and ev['vehicle_track'] == 1


# ---------- the cab region is the main per-camera tuning knob ----------

def test_cab_region_geometry():
    """The raw vehicle box is mostly mast and overhead guard — empty space — so a
    driver leaning well out is still 'inside' it. The cab is an inset of that box
    (context.md §7.4)."""
    left, top, right, bottom = CAB
    assert (left, top, right, bottom) == (130.0, 205.0, 270.0, 400.0)
    assert left > VBOX[0] and right < VBOX[2] and top > VBOX[1]
    assert bottom == VBOX[3], 'cab floor is the vehicle floor; no bottom inset'


def test_wider_cab_inset_suppresses_a_marginal_lean():
    """R5_CAB_FRACTIONS is the per-camera knob tuning will actually turn, so pin
    its direction: a wider inset makes the cab SMALLER and the rule MORE
    sensitive; a narrower one suppresses marginal leans."""
    marginal = held(lambda k: k.__setitem__(R_WRIST, (125, 330, 0.9)))
    assert play(marginal) is not None, 'just outside the default cab'
    wide_box = (0.0, 100.0, 400.0, 400.0)      # same cab fractions, bigger vehicle
    assert play(marginal, vbox=wide_box) is None, 'now comfortably inside'
