"""Recover missing seated-driver labels with COCO-pretrained person detection.

THE PROBLEM
-----------
`pdf-ih16p/forklift2` is the best imagery in the project: a real factory, Toyota
sit-down counterbalance forklifts (J&J's actual equipment), drivers in hi-vis,
elevated cameras. In most images containing a forklift, a driver is visibly
seated in it.

But only **4.9%** of its forklift boxes contain a labeled person. The drivers are
almost all unlabeled — and an unlabeled object is supervised as *background*, so
training on it teaches the detector that a seated driver is not a person. That
breaks Rule 5 at its root: Rule 5 identifies the driver, then checks whether the
driver's keypoints leave the cab. No driver detection, no Rule 5.

THE FIX
-------
This is a bounded recovery task, not labeling from scratch: one class, in a known
region. COCO-pretrained RF-DETR already detects `person` well on real imagery
(that is the one class COCO does best), so we run it, keep only detections that
sit inside an existing forklift box, and add them as `person`.

Guards, because a bad auto-label is worse than a missing one:
  * only inside a labeled forklift box (>= MIN_INSIDE of the person box)
  * high confidence (default 0.6)
  * discard anything overlapping an existing person label (no duplicates)
  * discard implausible sizes for a seated driver relative to the vehicle

Always run --review first and look at the sheet. These labels are machine-made;
they are good enough to train on but should never be described to J&J as
hand-verified ground truth.

Usage:
    python -m scripts.label_drivers --root data/real/forklift2-z6zww_v5 --review
    python -m scripts.label_drivers --root data/real/forklift2-z6zww_v5 --apply
"""

import argparse
import json
import os
import random

import cv2
import numpy as np

from scripts.fetch_roboflow import NAME_MAP

COCO_PERSON = 1          # class id for `person` in COCO-pretrained RF-DETR
MIN_INSIDE = 0.6         # fraction of the person box inside the forklift box
MAX_DUP_IOU = 0.3        # above this it duplicates an existing person label
MIN_REL_AREA = 0.01      # driver must be at least this fraction of the vehicle box
MAX_REL_AREA = 0.9       # ...and no more than this


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inside_frac(inner, outer):
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / a


def area(b):
    return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def process_split(root, split, det, threshold, limit=None, review=False, seed=0):
    ann_path = os.path.join(root, split, '_annotations.coco.json')
    if not os.path.exists(ann_path):
        return None
    with open(ann_path) as f:
        data = json.load(f)

    cats = {c['id']: NAME_MAP.get(c['name'].strip().lower()) for c in data['categories']}
    name_to_id = {}
    for cid, nm in cats.items():
        if nm and nm not in name_to_id:
            name_to_id[nm] = cid
    if 'person' not in name_to_id:
        # dataset has no person category at all; create one
        new_id = max(cats) + 1
        data['categories'].append({'id': new_id, 'name': 'person',
                                   'supercategory': 'person'})
        name_to_id['person'] = new_id

    by_img = {}
    for a in data['annotations']:
        by_img.setdefault(a['image_id'], []).append(a)

    images = list(data['images'])
    if review:
        random.Random(seed).shuffle(images)
    if limit:
        images = images[:limit]

    next_ann = max((a['id'] for a in data['annotations']), default=0) + 1
    added, scanned, with_forklift = 0, 0, 0
    tiles = []

    for im in images:
        anns = by_img.get(im['id'], [])
        forklifts = [a for a in anns if cats.get(a['category_id']) == 'forklift']
        if not forklifts:
            continue
        with_forklift += 1
        path = os.path.join(root, split, im['file_name'])
        frame = cv2.imread(path)
        if frame is None:
            continue
        scanned += 1

        existing = []
        for a in anns:
            if cats.get(a['category_id']) == 'person':
                x, y, w, h = a['bbox']
                existing.append((x, y, x + w, y + h))
        fboxes = []
        for a in forklifts:
            x, y, w, h = a['bbox']
            fboxes.append((x, y, x + w, y + h))

        d = det(frame)
        new_boxes = []
        for i in range(len(d)):
            if int(d.class_id[i]) != COCO_PERSON:
                continue
            box = tuple(float(v) for v in d.xyxy[i])
            host = None
            for fb in fboxes:
                if inside_frac(box, fb) >= MIN_INSIDE:
                    host = fb
                    break
            if host is None:
                continue
            rel = area(box) / area(host)
            if not (MIN_REL_AREA <= rel <= MAX_REL_AREA):
                continue
            if any(iou(box, e) > MAX_DUP_IOU for e in existing):
                continue
            new_boxes.append(box)
            existing.append(box)

        for box in new_boxes:
            x1, y1, x2, y2 = box
            data['annotations'].append({
                'id': next_ann, 'image_id': im['id'],
                'category_id': name_to_id['person'],
                'bbox': [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],
                'area': round((x2 - x1) * (y2 - y1), 2),
                'iscrowd': 0, 'segmentation': [], 'auto_labeled': True,
            })
            next_ann += 1
            added += 1

        if review and new_boxes and len(tiles) < 12:
            vis = frame.copy()
            for fb in fboxes:
                cv2.rectangle(vis, (int(fb[0]), int(fb[1])), (int(fb[2]), int(fb[3])),
                              (0, 140, 255), 2)
            for b in new_boxes:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (255, 0, 255), 3)
                cv2.putText(vis, 'AUTO driver', (int(b[0]), max(20, int(b[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            x1 = int(max(0, min(fb[0] for fb in fboxes) - 60))
            y1 = int(max(0, min(fb[1] for fb in fboxes) - 60))
            x2 = int(min(vis.shape[1], max(fb[2] for fb in fboxes) + 60))
            y2 = int(min(vis.shape[0], max(fb[3] for fb in fboxes) + 60))
            crop = vis[y1:y2, x1:x2]
            if crop.size:
                tiles.append(cv2.resize(crop, (400, 320)))

    return {'data': data, 'added': added, 'scanned': scanned,
            'with_forklift': with_forklift, 'tiles': tiles, 'path': ann_path}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'valid'])
    ap.add_argument('--threshold', type=float, default=0.6)
    ap.add_argument('--review', action='store_true',
                    help='sample a few hundred images, write a sheet, change nothing')
    ap.add_argument('--apply', action='store_true',
                    help='rewrite _annotations.coco.json with the recovered drivers')
    ap.add_argument('--review-limit', type=int, default=300)
    args = ap.parse_args()

    if not (args.review or args.apply):
        raise SystemExit('pass --review first, then --apply')

    from src.detector import RFDetrDetector

    # COCO-pretrained, NOT our fine-tuned model: we want generic `person`
    # knowledge from real photographs, which is exactly what COCO provides and
    # what our synthetic-trained checkpoint lacks.
    det = RFDetrDetector(weights=None, threshold=args.threshold)

    total_added = 0
    for split in args.splits:
        res = process_split(args.root, split, det, args.threshold,
                            limit=args.review_limit if args.review else None,
                            review=args.review)
        if res is None:
            continue
        rate = res['added'] / res['scanned'] if res['scanned'] else 0
        print(f"{split:6} images with a forklift: {res['with_forklift']:5d} | "
              f"scanned {res['scanned']:5d} | drivers recovered {res['added']:5d} "
              f"({rate:.2f}/image)")
        total_added += res['added']

        if args.review and res['tiles']:
            tiles = res['tiles']
            while len(tiles) % 4:
                tiles.append(np.zeros((320, 400, 3), np.uint8))
            out = os.path.join('outputs', f'drivers_{split}.jpg')
            os.makedirs('outputs', exist_ok=True)
            cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4])
                                        for i in range(0, len(tiles), 4)]))
            print(f'       review sheet -> {out} (magenta = auto driver)')

        if args.apply:
            with open(res['path'], 'w') as f:
                json.dump(res['data'], f)
            print(f"       wrote {res['path']}")

    if args.review:
        print('\nReview the sheet before --apply. These are machine-made labels: '
              'good enough to train on, never describable to J&J as hand-verified.')
    else:
        print(f'\nadded {total_added} driver labels, tagged "auto_labeled": true')


if __name__ == '__main__':
    main()
