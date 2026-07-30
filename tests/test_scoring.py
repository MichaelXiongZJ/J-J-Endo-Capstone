"""Tests for the §10 precision/recall scorer.

A scoring bug is uniquely dangerous: it produces confident numbers that go into
the final write-up and cannot be sanity-checked by eye.
"""

from scripts.score_events import score


def truth(*clips):
    return {'clips': list(clips)}


def clip(video, *violations):
    return {'video': video, 'violations': list(violations)}


def gt(rule, start, end):
    return {'rule': rule, 'start_s': start, 'end_s': end}


def ev(rule, start, end, video='a.mp4', **kw):
    return {'rule': rule, 'start_s': start, 'end_s': end, 'video': video, **kw}


def test_overlapping_event_is_a_true_positive():
    c, fp, fn = score([ev(3, 5.0, 8.0)], truth(clip('a.mp4', gt(3, 4.0, 9.0))))
    assert c[3] == {'TP': 1, 'FP': 0, 'FN': 0}
    assert not fp and not fn


def test_event_on_negative_clip_is_a_false_positive():
    """The whole point of filming negative clips."""
    c, fp, fn = score([ev(5, 3.0, 5.0, video='reversing.mp4')],
                      truth(clip('reversing.mp4')))
    assert c[5] == {'TP': 0, 'FP': 1, 'FN': 0}
    assert len(fp) == 1


def test_unflagged_violation_is_a_false_negative():
    c, fp, fn = score([], truth(clip('a.mp4', gt(3, 4.0, 9.0))))
    assert c[3] == {'TP': 0, 'FP': 0, 'FN': 1}
    assert len(fn) == 1


def test_wrong_rule_does_not_match():
    """A Rule 4 flag does not excuse a missed Rule 3 violation."""
    c, fp, fn = score([ev(4, 5.0, 8.0)], truth(clip('a.mp4', gt(3, 4.0, 9.0))))
    assert c[3]['FN'] == 1
    assert c[4]['FP'] == 1


def test_events_are_matched_within_their_own_clip_only():
    c, fp, fn = score([ev(3, 5.0, 8.0, video='b.mp4')],
                      truth(clip('a.mp4', gt(3, 4.0, 9.0))))
    assert c[3]['FN'] == 1, 'must not match a violation in a different clip'
    assert c[3]['FP'] == 0, 'event in an unlisted clip is not scored at all'


def test_fragmented_detections_count_as_one_true_positive():
    """Several events for one violation is a fragmentation problem, not several
    successes — otherwise precision could be inflated by splitting events."""
    c, fp, fn = score([ev(3, 4.5, 5.0), ev(3, 6.0, 6.5), ev(3, 7.0, 8.0)],
                      truth(clip('a.mp4', gt(3, 4.0, 9.0))))
    assert c[3] == {'TP': 1, 'FP': 0, 'FN': 0}


def test_tolerance_accommodates_duration_gates():
    """Rule 5 cannot fire until R5_MIN_S has elapsed, so detections land after
    the violation starts. Within tolerance that is a TP, not FP+FN."""
    late = ev(5, 9.4, 10.0)
    window = truth(clip('a.mp4', gt(5, 6.0, 8.0)))
    assert score([late], window, tolerance_s=2.0)[0][5]['TP'] == 1
    strict = score([late], window, tolerance_s=0.0)[0][5]
    assert strict['TP'] == 0 and strict['FP'] == 1 and strict['FN'] == 1


def test_two_separate_violations_scored_independently():
    c, _, _ = score([ev(3, 4.5, 5.5)],
                    truth(clip('a.mp4', gt(3, 4.0, 6.0), gt(3, 20.0, 25.0))))
    assert c[3] == {'TP': 1, 'FP': 0, 'FN': 1}


def test_legacy_timestamp_only_events_are_supported():
    """Events written in the guide's original single-timestamp format."""
    c, _, _ = score([{'rule': 3, 'timestamp_s': 5.0, 'video': 'a.mp4'}],
                    truth(clip('a.mp4', gt(3, 4.0, 9.0))))
    assert c[3]['TP'] == 1


def test_rule_filter():
    events = [ev(3, 5.0, 8.0), ev(1, 5.0, 8.0)]
    t = truth(clip('a.mp4', gt(3, 4.0, 9.0)))
    c, _, _ = score(events, t, rules={3})
    assert 1 not in c and c[3]['TP'] == 1


def test_precision_arithmetic_matches_definition():
    """1 TP, 2 FP -> precision 1/3; 1 TP, 1 FN -> recall 1/2."""
    events = [ev(3, 4.5, 5.5), ev(3, 40.0, 41.0), ev(3, 50.0, 51.0)]
    c, fp, fn = score(events, truth(clip('a.mp4', gt(3, 4.0, 6.0), gt(3, 90.0, 95.0))))
    assert c[3] == {'TP': 1, 'FP': 2, 'FN': 1}
    tp, fpc, fnc = c[3]['TP'], c[3]['FP'], c[3]['FN']
    assert tp / (tp + fpc) == 1 / 3
    assert tp / (tp + fnc) == 1 / 2
