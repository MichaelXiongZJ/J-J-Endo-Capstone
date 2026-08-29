import json, os
from pathlib import Path

import cv2
import numpy as np

from src.detector import StubDetector
from src.run_pipeline import run
from src.worker_detector import WorkerSafetyDetector


ROOT = Path('outputs/smoke_test')
ROOT.mkdir(parents=True, exist_ok=True)

video = ROOT / 'test.avi'
calib = ROOT / 'calib.json'


# Create a 5-second synthetic 640x480 video at 10 FPS.
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))
assert writer.isOpened(), 'Could not create synthetic test video'

for _ in range(50):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 100), (300, 420), (255, 255, 255), -1)
    writer.write(frame)
writer.release()


# Simple calibration so geometry.py can run.
calib.write_text(json.dumps({
    'camera_id': 'smoke_cam',
    'image_points': [[0, 0], [640, 0], [640, 480], [0, 480]],
    'floor_points': [[0, 0], [10, 0], [10, 8], [0, 8]],
    'vehicle_length_m': 2.7,
    'walkways': []
}))


# Person class = 2 in your current pipeline.
script = {
    i: [(200, 100, 300, 420, 2, 0.95)]
    for i in range(50)
}

detector = StubDetector(script)
worker_detector = WorkerSafetyDetector()

result = run(
    video=str(video),
    calib=str(calib),
    detector=detector,
    worker_detector=worker_detector,
    outdir=str(ROOT),
    device='cpu',
    person_id=2,
    forklift_id=1,
    use_pose=False,
    max_frames=50
)

print(result)
print(f'Annotated video: {ROOT / "videos/annotated.mp4"}')
print(f'Events: {ROOT / "events"}')
print(f'Person crops: {ROOT / "person_crops"}')