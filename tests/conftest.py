import os
import sys

import numpy as np
import pytest

# Tests import `src.*`, which requires the repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def kpts():
    """Factory for a (17,3) COCO keypoint array.

    Every joint starts at confidence 0.0 (i.e. invalid), so a test only has to
    specify the joints it cares about. This mirrors reality: most joints on a
    seated, partly occluded driver are genuinely unusable.
    """
    def make(**joints):
        from src import pose_utils as pu
        arr = np.zeros((17, 3), dtype=np.float32)
        for name, val in joints.items():
            idx = getattr(pu, name.upper())
            x, y = val[0], val[1]
            conf = val[2] if len(val) > 2 else 0.9
            arr[idx] = (x, y, conf)
        return arr
    return make


def _load_synthetic():
    """Paths + ground truth for the synthetic clip, generating it if absent."""
    import json
    import subprocess
    video = os.path.join(ROOT, 'outputs/synthetic/synthetic_cam1.mp4')
    if not os.path.exists(video):
        subprocess.run([sys.executable, '-m', 'scripts.make_synthetic_clip'],
                       cwd=ROOT, check=True, capture_output=True)
    return {
        'video': video,
        'calib': os.path.join(ROOT, 'data/calibration/synthetic_cam1.json'),
        'script': {int(k): v for k, v in json.load(
            open(os.path.join(ROOT, 'outputs/synthetic/detections.json'))).items()},
        'gt': json.load(open(os.path.join(ROOT, 'outputs/synthetic/ground_truth.json'))),
    }


@pytest.fixture
def synthetic():
    return _load_synthetic()


@pytest.fixture(scope='module')
def synthetic_module():
    """Module-scoped, so the 20 s clip is decoded once per test module."""
    return _load_synthetic()
