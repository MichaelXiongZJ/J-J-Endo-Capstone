"""Empirical Accuracy Scorer for Rule 5 (Driver Body Protrusion).

Evaluates Rule 5 precision, recall, F1 score, and scenario pass rate across the complete
Rule 5 validation matrix (positive lean-out scenarios and negative safe-operation controls).
"""

import sys
import numpy as np

from src.pose_utils import (L_SHOULDER, L_WRIST, NOSE, R_SHOULDER, R_WRIST)
from src.rules import CFG, Rule5State, TrackedObject

VBOX = (100.0, 100.0, 300.0, 400.0)
FRAMES_TO_FIRE = int(CFG['R5_MIN_S'] * CFG['PROC_FPS'])


def kp(joints):
    a = np.zeros((17, 3), dtype=np.float32)
    for idx, val in joints.items():
        a[idx] = (val[0], val[1], val[2] if len(val) > 2 else 0.9)
    return a


def seated_inside():
    return kp({NOSE: (200, 250), L_SHOULDER: (180, 285), R_SHOULDER: (220, 285),
               L_WRIST: (170, 330), R_WRIST: (230, 330)})


def play(frames, vbox=VBOX):
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
    out = []
    for _ in range(n):
        k = seated_inside()
        mutate(k)
        out.append(k)
    return out


def evaluate_rule5_matrix():
    scenarios = [
        # (Name, Frames_Function, Expected_Violation_Boolean)
        ("Arm-only protrusion outside cab", held(lambda k: k.__setitem__(R_WRIST, (60, 330, 0.9))), True),
        ("Torso protrusion outside cab", held(lambda k: {k.__setitem__(NOSE, (70, 250, 0.9)), k.__setitem__(L_SHOULDER, (95, 285, 0.9))}), True),
        ("Head protrusion while reversing", held(lambda k: k.__setitem__(NOSE, (105, 240, 0.9))), True),
        ("Head-turn inside cab (Control)", held(lambda k: k.__setitem__(NOSE, (150, 240, 0.9))), False),
        ("Normal seated driving (Control)", [seated_inside() for _ in range(60)], False),
        ("Brief reach for control (<1.5s)", [seated_inside() if i % 20 >= 14 else (lambda k: (k.__setitem__(R_WRIST, (55, 330, 0.9)), k)[1])(seated_inside()) for i in range(60)], False),
        ("Occluded lower body (Low conf leg)", held(lambda k: {k.__setitem__(13, (20, 700, 0.12)), k.__setitem__(15, (10, 750, 0.08))}), False),
    ]

    tp, fp, tn, fn = 0, 0, 0, 0
    results = []

    print("=" * 70)
    print("RULE 5 EMPIRICAL ACCURACY & SCENARIO EVALUATION SUMMARY")
    print("=" * 70)

    for name, frames, expected in scenarios:
        ev = play(frames)
        fired = ev is not None
        correct = (fired == expected)

        if expected and fired:
            tp += 1
            status = "TP (Passed)"
        elif not expected and not fired:
            tn += 1
            status = "TN (Passed)"
        elif expected and not fired:
            fn += 1
            status = "FN (Failed)"
        else:
            fp += 1
            status = "FP (Failed)"

        results.append((name, expected, fired, status, correct))
        print(f"Scenario: {name:<38} | Expected: {str(expected):<5} | Fired: {str(fired):<5} | Status: {status}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(scenarios)

    print("-" * 70)
    print(f"Total Scenarios Evaluated : {len(scenarios)}")
    print(f"True Positives (TP)      : {tp}")
    print(f"True Negatives (TN)      : {tn}")
    print(f"False Positives (FP)     : {fp}")
    print(f"False Negatives (FN)     : {fn}")
    print(f"Precision Score          : {precision:.4f} ({precision * 100:.1f}%)")
    print(f"Recall Score             : {recall:.4f} ({recall * 100:.1f}%)")
    print(f"F1 Score                 : {f1:.4f}")
    print(f"Overall Matrix Accuracy  : {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print("=" * 70)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'scenarios_passed': tp + tn,
        'scenarios_total': len(scenarios)
    }


if __name__ == '__main__':
    metrics = evaluate_rule5_matrix()
    if metrics['accuracy'] < 1.0:
        sys.exit(1)
