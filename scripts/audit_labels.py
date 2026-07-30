"""Detect objects that are present but UNLABELED in a dataset.

The most damaging defect a detection dataset can have is a missing label. In
training, an unlabeled object is supervised as *background*, so the model is
explicitly taught that the thing is not the thing. One badly under-labeled source
can undo a good one.

This is not hypothetical here: `box_pickup` renders J&J's exact forklift and
labels none of them, and `forklift-model` carries 135 person boxes across 8076
warehouse images.

Method: run an existing trained detector over a sample, and count confident
detections that overlap no ground-truth box of that class. A high rate means the
dataset systematically omits that class.

The detector is imperfect, so treat the output as a *signal*, not a verdict —
and read the contact sheet it writes before acting on it.

Usage:
    python -m scripts.audit_labels --root data/real/forklift-model_v3 \
        --weights models/rfdetr_v2/checkpoint_best_ema.pth --dataset-dir data/dataset_v2
"""

import argparse
import json
import os
import random

import cv2
import numpy as np


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--dataset-dir', default='data/dataset_v2',
                    help='resolves the model class-id mapping')
    ap.add_argument('--sample', type=int, default=120)
    ap.add_argument('--threshold', type=float, default=0.6,
                    help='detector confidence; kept high so misses are real')
    ap.add_argument('--iou', type=float, default=0.3)
    ap.add_argument('--out', default=None, help='contact sheet of worst offenders')
    args = ap.parse_args()

    from src.detector import RFDetrDetector, model_class_ids

    ids = model_class_ids(args.dataset_dir)
    inv = {v: k for k, v in ids.items()}
    det = RFDetrDetector(weights=args.weights, threshold=args.threshold)

    ann_path = os.path.join(args.root, args.split, '_annotations.coco.json')
    with open(ann_path) as f:
        data = json.load(f)
    cats = {c['id']: c['name'].strip().lower() for c in data['categories']}
    by_img = {}
    for a in data['annotations']:
        by_img.setdefault(a['image_id'], []).append(a)

    # Dataset class names -> ours, so ground truth and predictions are comparable.
    from scripts.fetch_roboflow import NAME_MAP

    imgs = list(data['images'])
    random.Random(0).shuffle(imgs)
    imgs = imgs[:args.sample]

    stats = {'person': {'gt': 0, 'pred': 0, 'unlabeled': 0},
             'forklift': {'gt': 0, 'pred': 0, 'unlabeled': 0}}
    worst = []

    for im in imgs:
        p = os.path.join(args.root, args.split, im['file_name'])
        frame = cv2.imread(p)
        if frame is None:
            continue
        gt = {'person': [], 'forklift': []}
        for a in by_img.get(im['id'], []):
            name = NAME_MAP.get(cats.get(a['category_id'], ''), None)
            if name is None:
                continue
            x, y, w, h = a['bbox']
            gt[name].append((x, y, x + w, y + h))
            stats[name]['gt'] += 1

        d = det(frame)
        missing = 0
        for i in range(len(d)):
            cls = inv.get(int(d.class_id[i]))
            if cls not in stats:
                continue
            box = tuple(float(v) for v in d.xyxy[i])
            stats[cls]['pred'] += 1
            if not any(iou(box, g) >= args.iou for g in gt[cls]):
                stats[cls]['unlabeled'] += 1
                missing += 1
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 0, 255), 3)
                cv2.putText(frame, f'UNLABELED {cls}', (int(box[0]), max(20, int(box[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        for cls, boxes in gt.items():
            for g in boxes:
                cv2.rectangle(frame, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])),
                              (0, 255, 0), 2)
        if missing:
            worst.append((missing, frame))

    print(f'\n{args.root} [{args.split}], {len(imgs)} images sampled, '
          f'detector threshold {args.threshold}\n')
    print(f'{"class":10}{"gt boxes":>10}{"detected":>10}{"unlabeled":>11}{"rate":>8}')
    for cls, s in stats.items():
        rate = s['unlabeled'] / s['pred'] if s['pred'] else 0.0
        print(f'{cls:10}{s["gt"]:>10}{s["pred"]:>10}{s["unlabeled"]:>11}{rate:>8.1%}')

    print('\nA high unlabeled rate means the dataset omits that class, and training '
          'on it teaches the model the class is background. Prefer excluding such a '
          'source over "fixing" it with a lower threshold.')

    name = os.path.basename(os.path.normpath(args.root))
    out = args.out or os.path.join('outputs', f'audit_{name}_{args.split}.jpg')
    if worst:
        worst.sort(key=lambda t: -t[0])
        tiles = [cv2.resize(f, (480, 360)) for _n, f in worst[:6]]
        while len(tiles) % 3:
            tiles.append(np.zeros((360, 480, 3), np.uint8))
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 3])
                                    for i in range(0, len(tiles), 3)]))
        print(f'\nworst offenders -> {out} (red = detected but unlabeled, green = labeled)')


if __name__ == '__main__':
    main()
