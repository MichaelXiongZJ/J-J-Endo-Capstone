"""Homography tests. The failure modes here are silent, hence the coverage."""

import json

import pytest

from src.geometry import CameraGeometry, floor_dist


@pytest.fixture
def cam(tmp_path):
    """A 10 m x 20 m floor seen through a plausible perspective trapezoid."""
    cfg = {
        'camera_id': 'test',
        'image_points': [[200, 700], [1080, 700], [860, 300], [420, 300]],
        'floor_points': [[0, 0], [10, 0], [10, 20], [0, 20]],
        'walkways': [[[0, 0], [2, 0], [2, 20], [0, 20]]],
        'vehicle_length_m': 2.7,
    }
    p = tmp_path / 'cam.json'
    p.write_text(json.dumps(cfg))
    return CameraGeometry(str(p))


def test_calibration_points_map_to_their_floor_coords(cam):
    for img, flr in zip(cam.image_points, cam.floor_points):
        got = cam.to_floor(*img)
        assert floor_dist(got, flr) < 1e-3


def test_known_distance_within_tolerance(cam):
    """The §6.4 acceptance check: a measured distance must come back within 10%."""
    near = floor_dist(cam.to_floor(200, 700), cam.to_floor(1080, 700))
    assert abs(near - 10.0) / 10.0 < 0.10


def test_floor_position_uses_bottom_centre_not_box_centre(cam):
    """A box's bottom edge is the ground-contact point; its centre floats in
    mid-air. Using the centre is the classic silent error, so pin the behaviour.
    """
    box = (600, 500, 700, 700)
    assert cam.floor_position(box) == pytest.approx(cam.to_floor(650, 700))
    assert cam.floor_position(box) != pytest.approx(cam.to_floor(650, 600))


def test_perspective_is_actually_nonlinear(cam):
    """Guards against someone replacing the homography with a linear scale.

    Equal pixel gaps must map to UNEQUAL floor distances — that non-linearity is
    the entire reason pixel distance cannot substitute for metres.
    """
    near = floor_dist(cam.to_floor(600, 700), cam.to_floor(700, 700))
    far = floor_dist(cam.to_floor(600, 320), cam.to_floor(700, 320))
    assert far > near * 1.5


def test_on_walkway(cam):
    assert cam.on_walkway((1.0, 10.0)) is True
    assert cam.on_walkway((5.0, 10.0)) is False
    assert cam.has_walkways


def test_no_walkways_configured_is_reported(tmp_path):
    cfg = {'camera_id': 'c', 'image_points': [[0, 0], [1, 0], [1, 1], [0, 1]],
           'floor_points': [[0, 0], [1, 0], [1, 1], [0, 1]]}
    p = tmp_path / 'c.json'
    p.write_text(json.dumps(cfg))
    g = CameraGeometry(str(p))
    assert not g.has_walkways
    assert g.vehicle_length_m == 2.7        # documented default


def test_rejects_fewer_than_four_points(tmp_path):
    cfg = {'camera_id': 'c', 'image_points': [[0, 0], [1, 0], [1, 1]],
           'floor_points': [[0, 0], [1, 0], [1, 1]]}
    p = tmp_path / 'c.json'
    p.write_text(json.dumps(cfg))
    with pytest.raises(AssertionError):
        CameraGeometry(str(p))


def test_collinear_points_raise_rather_than_silently_producing_garbage(tmp_path):
    cfg = {'camera_id': 'c',
           'image_points': [[0, 0], [10, 0], [20, 0], [30, 0]],
           'floor_points': [[0, 0], [1, 0], [2, 0], [3, 0]]}
    p = tmp_path / 'c.json'
    p.write_text(json.dumps(cfg))
    with pytest.raises((ValueError, Exception)):
        CameraGeometry(str(p))


def test_swapped_point_order_produces_detectably_wrong_distance(tmp_path):
    """context.md §8.6: image_points[i] must pair with floor_points[i].
    Mis-ordering is the most common calibration failure. Confirm the §6.4 check
    actually catches it rather than passing quietly.
    """
    good = {'camera_id': 'c',
            'image_points': [[200, 700], [1080, 700], [860, 300], [420, 300]],
            'floor_points': [[0, 0], [10, 0], [10, 20], [0, 20]]}
    bad = dict(good)
    bad['floor_points'] = [[10, 0], [0, 0], [10, 20], [0, 20]]   # first two swapped

    (tmp_path / 'g.json').write_text(json.dumps(good))
    (tmp_path / 'b.json').write_text(json.dumps(bad))
    g = CameraGeometry(str(tmp_path / 'g.json'))
    b = CameraGeometry(str(tmp_path / 'b.json'))

    d_good = floor_dist(g.to_floor(640, 500), g.to_floor(700, 500))
    d_bad = floor_dist(b.to_floor(640, 500), b.to_floor(700, 500))
    assert abs(d_bad - d_good) / d_good > 0.10
