"""Event aggregation tests.

The point of src/events.py is that §10 scores precision/recall PER EVENT. If one
20-second violation emitted 200 rows, precision would be measuring frames, not
violations.
"""

import json

from src.events import EventAggregator


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def agg(tmp_path, **kw):
    return EventAggregator(str(tmp_path / 'ev'), 'cam1', 'clip.mp4',
                           save_frames=False, **kw)


def r3(dist, person=5, vehicle=9):
    return {'rule': 3, 'person_track': person, 'vehicle_track': vehicle,
            'distance_m': dist, 'threshold_m': 8.1}


def test_sustained_condition_becomes_one_event(tmp_path):
    a = agg(tmp_path)
    for i in range(200):                       # 20 s at 10 fps
        a.add([r3(5.0)], i * 0.1)
    n = a.close()
    assert n == 1
    e = read(a.path)[0]
    assert e['frames'] == 200
    assert e['start_s'] == 0.0
    assert e['duration_s'] == round(199 * 0.1, 2)


def test_peak_severity_is_kept_for_rule3(tmp_path):
    """Rule 3's severity is the CLOSEST approach — that is the instant a
    reviewer needs to see, so it is also the evidence frame chosen."""
    a = agg(tmp_path)
    for i, d in enumerate([7.0, 5.0, 2.3, 4.0, 6.0]):
        a.add([r3(d)], i * 0.1)
    a.close()
    e = read(a.path)[0]
    assert e['distance_m'] == 2.3
    assert e['peak_s'] == 0.2


def test_peak_severity_is_max_for_rule5(tmp_path):
    a = agg(tmp_path)
    for i, s in enumerate([1.5, 2.0, 3.4, 1.8]):
        a.add([{'rule': 5, 'driver_track': 3, 'seconds_outside': s}], i * 0.1)
    a.close()
    assert read(a.path)[0]['seconds_outside'] == 3.4


def test_gap_longer_than_cooldown_splits_into_two_events(tmp_path):
    a = agg(tmp_path, cooldown_s=2.0)
    for i in range(10):
        a.add([r3(5.0)], i * 0.1)
    a.add([], 10.0)                            # 9 s of silence
    for i in range(10):
        a.add([r3(4.0)], 10.0 + i * 0.1)
    assert a.close() == 2


def test_brief_dropout_does_not_split(tmp_path):
    """Detection flicker and occlusion must not inflate the event count."""
    a = agg(tmp_path, cooldown_s=2.0)
    for i in range(10):
        a.add([r3(5.0)], i * 0.1)
    a.add([], 1.5)                             # 0.5 s dropout, within cooldown
    for i in range(10):
        a.add([r3(4.0)], 2.0 + i * 0.1)
    assert a.close() == 1


def test_distinct_participants_are_distinct_events(tmp_path):
    a = agg(tmp_path)
    for i in range(20):
        a.add([r3(5.0, person=1), r3(6.0, person=2)], i * 0.1)
    assert a.close() == 2
    assert {e['person_track'] for e in read(a.path)} == {1, 2}


def test_same_person_different_vehicle_is_a_separate_event(tmp_path):
    a = agg(tmp_path)
    for i in range(20):
        a.add([r3(5.0, person=1, vehicle=8), r3(6.0, person=1, vehicle=9)], i * 0.1)
    assert a.close() == 2


def test_different_rules_never_merge(tmp_path):
    a = agg(tmp_path)
    for i in range(20):
        a.add([r3(5.0, person=1), {'rule': 4, 'person_track': 1, 'floor_xy': [1, 2]}],
              i * 0.1)
    assert a.close() == 2
    assert {e['rule'] for e in read(a.path)} == {3, 4}


def test_close_flushes_open_episodes(tmp_path):
    a = agg(tmp_path)
    a.add([r3(5.0)], 0.0)
    assert read(a.path) == []                  # still open, nothing written
    assert a.close() == 1
    assert len(read(a.path)) == 1


def test_records_carry_provenance(tmp_path):
    a = agg(tmp_path)
    a.add([r3(5.0)], 1.0)
    a.close()
    e = read(a.path)[0]
    for field in ('event_id', 'camera_id', 'video', 'start_s', 'end_s',
                  'duration_s', 'timestamp_s', 'rule'):
        assert field in e, field
    assert e['camera_id'] == 'cam1' and e['video'] == 'clip.mp4'


def test_active_rules_reported_for_banner(tmp_path):
    a = agg(tmp_path)
    assert a.add([r3(5.0)], 0.0) == {3}
    assert a.add([], 0.1) == set()
    a.close()
