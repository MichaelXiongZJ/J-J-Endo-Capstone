"""End-to-end pipeline test against the synthetic clip.

Exercises the real code path — decode, track, project to floor metres, associate
the driver, evaluate rules, aggregate events — with a scripted detector standing
in for RF-DETR. Ground truth is arithmetic (see scripts/make_synthetic_clip.py),
so the expected Rule 3 trigger time is derived, not guessed.

These are NOT accuracy metrics. Reported accuracy comes only from staged footage
per §10.
"""

import json

import pytest

from src.detector import StubDetector
from src.run_pipeline import run

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def result(synthetic_module, tmp_path_factory):
    out = tmp_path_factory.mktemp('synthrun')
    res = run(synthetic_module['video'], synthetic_module['calib'],
              StubDetector(synthetic_module['script']),
              outdir=str(out), use_pose=False, write_video=False,
              save_evidence=False, verbose=False)
    with open(res['events_path']) as f:
        res['records'] = [json.loads(line) for line in f]
    res['gt'] = synthetic_module['gt']
    return res


def test_all_frames_processed(result):
    assert result['frames_processed'] == result['gt']['processed_frames']


def test_rule3_fires_once_at_the_predicted_time(result):
    r3 = [e for e in result['records'] if e['rule'] == 3]
    assert len(r3) == 1, f'expected exactly one Rule 3 episode, got {r3}'
    e = r3[0]
    # Within one processed frame (0.1 s) of the arithmetic prediction.
    assert e['start_s'] == pytest.approx(result['gt']['rule3_expected_start_s'], abs=0.15)
    assert e['distance_m'] == pytest.approx(
        result['gt']['rule3_expected_min_distance_m'], abs=0.1)
    assert e['threshold_m'] == pytest.approx(result['gt']['rule3_threshold_m'], abs=0.01)


def test_rule3_does_not_fire_before_the_threshold_is_crossed(result):
    e = [x for x in result['records'] if x['rule'] == 3][0]
    assert e['start_s'] > 5.0, 'fired far too early — check homography scale'


def test_walkway_pedestrian_triggers_nothing(result):
    """The false-positive check. Ped B is >=10.5 m from the forklift at all times
    and always inside the walkway polygon, so ANY event naming it is an FP.

    Its track id is found by elimination: it is the only person track that
    should appear in no event at all.
    """
    flagged = {e.get('person_track') for e in result['records']}
    flagged |= {e.get('driver_track') for e in result['records']}
    flagged.discard(None)
    # Exactly one person (ped A) should ever be flagged; driver and ped B never.
    assert len(flagged) == 1, f'expected only ped A flagged, got tracks {flagged}'


def test_driver_is_never_flagged_as_a_pedestrian(result):
    """The driver rides the forklift at 0 m separation. If driver association
    failed, Rule 3 would fire against their own vehicle from frame ~4 onward at
    distance ~0 — the single most likely pipeline bug (§9 acceptance check).
    """
    for e in result['records']:
        if e['rule'] == 3:
            assert e['distance_m'] > 1.0, (
                'a ~0 m Rule 3 event means the driver was treated as a pedestrian')


def test_rule4_flags_the_off_walkway_pedestrian(result):
    r4 = [e for e in result['records'] if e['rule'] == 4]
    assert len(r4) == 1
    assert r4[0]['floor_xy'] == pytest.approx([14.5, 12.0], abs=0.1), \
        'floor position must round-trip through the homography exactly'


def test_rule4_respects_its_duration_gate(result):
    """Ped A is off-walkway from frame 0, so the event must start at R4_MIN_S,
    not at t=0."""
    from src.rules import CFG
    e = [x for x in result['records'] if x['rule'] == 4][0]
    assert e['start_s'] >= CFG['R4_MIN_S'] - 0.15


def test_pose_rules_are_silent_without_pose(result):
    """Rendered rectangles have no anatomy; run_pose was disabled. Rules 5 and 1
    must therefore produce nothing rather than guessing."""
    assert not [e for e in result['records'] if e['rule'] in (1, 5)]


def test_event_count_is_per_episode_not_per_frame(result):
    """Without src/events.py this clip would emit ~330 rows for 2 violations."""
    assert len(result['records']) == 2, result['records']
