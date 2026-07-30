"""Convert the SDG-Warehouse slice into a COCO dataset RF-DETR can train on.

Replaces Phase 3's labeling effort (~60% of total prototype effort) for the two
classes we need. The simulator's boxes are exact, so there is no label noise and
no inter-annotator inconsistency.

Two conventions are enforced here because they decide whether the resulting
metrics mean anything:

  * **Split by RUN, never by frame** (§4.2). Frames a second apart are
    near-duplicates, and the 5 ceiling cameras of one run show the SAME moment
    from different angles — so they are correlated too. Whole runs, all their
    cameras, go to exactly one side of the split. A random frame split here would
    produce excellent and entirely fictional mAP.
  * **Sample sparsely in time.** A 10 s clip at 30 fps is 300 near-identical
    frames; we take one every `--stride` frames (default 30 = one per second).

Class IDs are written as 1=forklift, 2=person to match the defaults in
`src/run_pipeline.py`. Always re-check with `scripts.discover_class_ids`.

Usage:
    python -m scripts.sdg_to_coco
    python -m scripts.sdg_to_coco --stride 15 --val-fraction 0.3
"""

import argparse
import json
import os
import random

import cv2

from scripts.sdg_common import SDGClip, find_runs

CATEGORIES = [{'id': 1, 'name': 'forklift', 'supercategory': 'vehicle'},
              {'id': 2, 'name': 'person', 'supercategory': 'person'}]
CAT_ID = {c['name']: c['id'] for c in CATEGORIES}

# Boxes smaller than this are unusable supervision at 1080p and mostly hurt.
MIN_BOX_AREA = 24 * 24
# Require most of the box to be inside the frame. The guide's "label people >=30%
# visible" cannot be applied directly because the dataset's occlusionRatio is
# -1.0 (not computed), so frame-edge truncation is the proxy we can compute.
MIN_INSIDE_FRACTION = 0.6


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(w), x2), min(float(h), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, 0.0
    full = (x2 - x1) * (y2 - y1)
    inside = (cx2 - cx1) * (cy2 - cy1)
    return (cx1, cy1, cx2, cy2), (inside / full if full > 0 else 0.0)


def export(runs, out_root, stride, val_fraction, seed, max_frames_per_clip):
    random.Random(seed).shuffle(runs)
    n_val = max(1, int(round(len(runs) * val_fraction))) if len(runs) > 1 else 0
    split_of = {}
    for i, (run_dir, _) in enumerate(runs):
        split_of[run_dir] = 'valid' if i < n_val else 'train'

    data = {s: {'images': [], 'annotations': [], 'categories': CATEGORIES}
            for s in ('train', 'valid')}
    counters = {'img': 0, 'ann': 0}
    stats = {s: {'images': 0, 'person': 0, 'forklift': 0, 'empty': 0}
             for s in ('train', 'valid')}

    for run_dir, cams in runs:
        split = split_of[run_dir]
        out_dir = os.path.join(out_root, split)
        os.makedirs(out_dir, exist_ok=True)
        for cam in cams:
            clip = SDGClip(run_dir, cam)
            cap = cv2.VideoCapture(clip.video)
            if not cap.isOpened():
                print(f'  cannot open {clip.video}')
                continue
            taken = 0
            for frame_idx in range(0, len(clip), stride):
                if max_frames_per_clip and taken >= max_frames_per_clip:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, bgr = cap.read()
                if not ok:
                    continue
                h, w = bgr.shape[:2]

                anns = []
                for cls, box, _world in clip.agents(frame_idx):
                    cb, inside = clamp_box(box, w, h)
                    if cb is None or inside < MIN_INSIDE_FRACTION:
                        continue
                    bw, bh = cb[2] - cb[0], cb[3] - cb[1]
                    if bw * bh < MIN_BOX_AREA:
                        continue
                    anns.append({
                        'id': counters['ann'],
                        'image_id': counters['img'],
                        'category_id': CAT_ID[cls],
                        'bbox': [round(cb[0], 2), round(cb[1], 2),
                                 round(bw, 2), round(bh, 2)],   # COCO: x,y,w,h
                        'area': round(bw * bh, 2),
                        'iscrowd': 0,
                        'segmentation': [],
                    })
                    counters['ann'] += 1
                    stats[split][cls] += 1

                fname = f'{clip.run_id[:20]}_{cam}_{frame_idx:04d}.jpg'
                cv2.imwrite(os.path.join(out_dir, fname), bgr)
                data[split]['images'].append({
                    'id': counters['img'], 'file_name': fname,
                    'width': w, 'height': h,
                    # Provenance, so a suspicious image can be traced back.
                    'sdg_run': clip.run_id, 'sdg_camera': cam,
                    'sdg_frame': frame_idx,
                })
                data[split]['annotations'].extend(anns)
                if not anns:
                    stats[split]['empty'] += 1
                counters['img'] += 1
                stats[split]['images'] += 1
                taken += 1
            cap.release()
            print(f'  {split:5} {clip.run_id[:20]}/{cam}: {taken} frames')

    for split in ('train', 'valid'):
        if not data[split]['images']:
            continue
        p = os.path.join(out_root, split, '_annotations.coco.json')
        with open(p, 'w') as f:
            json.dump(data[split], f)
        print(f'wrote {p}')
    return stats, split_of


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', nargs='+', default=['data/sdg'],
                    help='one or more source roots. Pass both scenarios to mix '
                         'reach trucks (nearmiss) with the sit-down counterbalance '
                         'forklift recovered from box_pickup segmentation')
    ap.add_argument('--out', default='data/dataset')
    ap.add_argument('--stride', type=int, default=30,
                    help='frames between samples (30 = one per second)')
    ap.add_argument('--val-fraction', type=float, default=0.34,
                    help='fraction of RUNS (not frames) held out for validation')
    ap.add_argument('--max-frames-per-clip', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    runs = []
    for root in args.root:
        found = find_runs(root)
        print(f'{root}: {len(found)} runs, {sum(len(c) for _, c in found)} clips')
        runs.extend(found)
    if not runs:
        raise SystemExit(f'no runs under {args.root} — run scripts.fetch_sdg first')
    print(f'total {len(runs)} runs, {sum(len(c) for _, c in runs)} clips')

    stats, split_of = export(runs, args.out, args.stride, args.val_fraction,
                             args.seed, args.max_frames_per_clip)

    print('\nsplit by run (§4.2):')
    for run_dir, split in sorted(split_of.items(), key=lambda kv: kv[1]):
        print(f'  {split:5} {os.path.basename(run_dir)[:40]}')

    print(f'\n{"split":8}{"images":>8}{"person":>8}{"forklift":>10}{"empty":>7}')
    for s in ('train', 'valid'):
        v = stats[s]
        print(f'{s:8}{v["images"]:>8}{v["person"]:>8}{v["forklift"]:>10}{v["empty"]:>7}')

    total = sum(stats[s]['images'] for s in stats)
    print(f'\ntotal {total} images')
    if total < 300:
        print('Under the ~300-image floor for a working prototype (§4.1) — fetch '
              'more shards: python -m scripts.fetch_sdg --shards 5 6 7 8')
    if stats['valid']['images'] == 0:
        print('WARNING: empty validation split — need at least 2 runs.')
    print('\nNext: python -m scripts.discover_class_ids ' + args.out)


if __name__ == '__main__':
    main()
