"""RTMPose keypoints via rtmlib (implementation guide §7.1).

You do NOT train pose. Human anatomy is universal, so a COCO-pretrained model
works on warehouse workers out of the box. rtmlib runs RTMPose through ONNX
with no heavyweight dependencies and auto-downloads weights on first use.
"""

import numpy as np

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_WRIST, R_WRIST, L_HIP, R_HIP = 9, 10, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO-17 skeleton edges, for drawing only.
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]

_body = None


def get_pose_model(device="cuda"):
    """Lazily construct and cache the RTMPose wrapper. device='cpu' if no GPU."""
    global _body
    if _body is None:
        from rtmlib import Body  # imported lazily so geometry/rules tests need no onnxruntime
        _body = Body(mode="balanced", backend="onnxruntime", device=device)
    return _body


def run_pose(frame_bgr, device="cuda"):
    """NOTE: rtmlib takes the OpenCV BGR frame DIRECTLY (unlike RF-DETR, which
    needs RGB). Returns kpts: (N, 17, 3) = x, y, confidence per keypoint,
    N = people found.
    """
    keypoints, scores = get_pose_model(device)(frame_bgr)
    if keypoints is None or len(keypoints) == 0:
        return np.zeros((0, 17, 3), dtype=np.float32)
    return np.concatenate(
        [np.asarray(keypoints), np.asarray(scores)[..., None]], axis=-1
    ).astype(np.float32)


def valid(kpt, thresh=0.5):
    """ALWAYS gate on confidence. Occluded joints (driver's legs behind the cab)
    come back with LOW confidence and a GUESSED position — trusting them means
    violations triggered by hallucinated limbs.
    """
    return kpt[2] > thresh


def match_pose_to_boxes(kpts_all, person_boxes):
    """Assign each pose to the person box containing its torso centre.
    Returns {box_index: (17,3) keypoints}.

    Needed because rtmlib's `Body` finds people with its own internal detector,
    so its outputs must be re-associated with OUR tracked RF-DETR person boxes.
    """
    def torso_centre(kp):
        pts = [kp[i][:2] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP) if valid(kp[i])]
        return np.mean(pts, axis=0) if pts else None

    def area(b):
        return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))

    out = {}
    for kp in kpts_all:
        c = torso_centre(kp)
        if c is None:
            continue
        candidates = [i for i, b in enumerate(person_boxes)
                      if b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]]
        if candidates:
            best = min(candidates, key=lambda i: area(person_boxes[i]))  # smallest containing box
            if best not in out:
                out[best] = kp
    return out


def draw_pose(frame_bgr, kpts_all, conf=0.5):
    """Draw keypoints and skeleton for visual verification (§7 ACCEPTANCE CHECK).

    A driver's occluded lower body showing few/no valid leg keypoints is the
    confidence gate working, not a bug.
    """
    import cv2

    for kp in kpts_all:
        for a, b in SKELETON:
            if valid(kp[a], conf) and valid(kp[b], conf):
                cv2.line(frame_bgr, tuple(map(int, kp[a][:2])), tuple(map(int, kp[b][:2])),
                         (0, 255, 255), 2)
        for i in range(17):
            if valid(kp[i], conf):
                cv2.circle(frame_bgr, tuple(map(int, kp[i][:2])), 4, (0, 0, 255), -1)
    return frame_bgr
