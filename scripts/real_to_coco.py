"""Normalise a downloaded Roboflow dataset into our COCO layout, and merge.

Handles both export formats:

  * **COCO** (`train/_annotations.coco.json`) — remap category names onto ours.
  * **YOLO** (`data.yaml` + `train/images/`, `train/labels/`) — convert
    normalised cx,cy,w,h to absolute COCO x,y,w,h, then remap.

Class names differ between datasets ('human', 'worker', 'people' all mean
`person`). NAME_MAP in scripts/fetch_roboflow.py handles the common cases;
anything unmapped is REPORTED AND DROPPED rather than folded into a class it is
not — a mislabeled class is worse than a missing one.

Output uses our fixed categories, 1 = forklift, 2 = person, so a merged dataset
keeps one consistent id space. (Remember the model then predicts 0-based:
forklift = 0, person = 1. See src.detector.model_class_ids.)

Usage:
    python -m scripts.real_to_coco --root data/real/forklift-and-human_v2
    python -m scripts.real_to_coco --root data/real/a data/real/b \
        --merge-with data/dataset_v2 --out data/dataset_v3
"""

import argparse
import json
import os
import shutil

from scripts.fetch_roboflow import NAME_MAP

CATEGORIES = [{'id': 1, 'name': 'forklift', 'supercategory': 'vehicle'},
              {'id': 2, 'name': 'person', 'supercategory': 'person'}]
CAT_ID = {c['name']: c['id'] for c in CATEGORIES}
SPLITS = ('train', 'valid', 'test')


def read_yolo_names(root):
    """Class names from data.yaml without requiring a YAML dependency."""
    for fn in ('data.yaml', 'data.yml'):
        p = os.path.join(root, fn)
        if not os.path.exists(p):
            continue
        text = open(p).read()
        i = text.find('names:')
        if i < 0:
            continue
        chunk = text[i + 6:]
        if '[' in chunk.split('\n')[0]:
            inside = chunk[chunk.find('[') + 1:chunk.find(']')]
            return [n.strip().strip('\'"') for n in inside.split(',') if n.strip()]
        names = []
        for line in chunk.splitlines():
            s = line.strip()
            if s.startswith('-'):
                names.append(s[1:].strip().strip('\'"'))
            elif names and s and not s.startswith('#'):
                break
        return names
    return []


def load_yolo_split(root, split, names, unmapped):
    img_dir = os.path.join(root, split, 'images')
    lbl_dir = os.path.join(root, split, 'labels')
    if not os.path.isdir(img_dir):
        return []
    import cv2

    out = []
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        ipath = os.path.join(img_dir, fn)
        img = cv2.imread(ipath)
        if img is None:
            continue
        h, w = img.shape[:2]
        anns = []
        lpath = os.path.join(lbl_dir, os.path.splitext(fn)[0] + '.txt')
        if os.path.exists(lpath):
            for line in open(lpath):
                parts = line.split()
                if len(parts) < 5:
                    continue
                ci = int(float(parts[0]))
                raw = names[ci] if ci < len(names) else str(ci)
                mapped = NAME_MAP.get(raw.strip().lower())
                if mapped is None:
                    unmapped[raw] = unmapped.get(raw, 0) + 1
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
                anns.append((mapped, [round(x, 2), round(y, 2),
                                      round(bw * w, 2), round(bh * h, 2)]))
        out.append((ipath, fn, w, h, anns))
    return out


def load_coco_split(root, split, unmapped):
    d = os.path.join(root, split)
    ann_path = os.path.join(d, '_annotations.coco.json')
    if not os.path.exists(ann_path):
        return []
    with open(ann_path) as f:
        data = json.load(f)
    cats = {c['id']: c['name'] for c in data['categories']}
    by_img = {}
    for a in data['annotations']:
        raw = cats.get(a['category_id'], '?')
        mapped = NAME_MAP.get(raw.strip().lower())
        if mapped is None:
            unmapped[raw] = unmapped.get(raw, 0) + 1
            continue
        by_img.setdefault(a['image_id'], []).append((mapped, a['bbox']))
    out = []
    for im in data['images']:
        p = os.path.join(d, im['file_name'])
        if os.path.exists(p):
            out.append((p, im['file_name'], im['width'], im['height'],
                        by_img.get(im['id'], [])))
    return out


def write_split(records, out_dir, prefix, start_img=0, start_ann=0):
    os.makedirs(out_dir, exist_ok=True)
    images, annotations = [], []
    iid, aid = start_img, start_ann
    for src, fn, w, h, anns in records:
        name = f'{prefix}_{fn}'
        shutil.copy2(src, os.path.join(out_dir, name))
        images.append({'id': iid, 'file_name': name, 'width': w, 'height': h,
                       'source': prefix})
        for cls, bbox in anns:
            annotations.append({'id': aid, 'image_id': iid,
                                'category_id': CAT_ID[cls], 'bbox': bbox,
                                'area': round(bbox[2] * bbox[3], 2),
                                'iscrowd': 0, 'segmentation': []})
            aid += 1
        iid += 1
    return images, annotations, iid, aid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', nargs='+', required=True, help='downloaded dataset root(s)')
    ap.add_argument('--out', default='data/dataset_real')
    ap.add_argument('--merge-with', default=None,
                    help='existing COCO dataset dir to combine with (e.g. data/dataset_v2)')
    args = ap.parse_args()

    unmapped = {}
    per_split = {s: [] for s in SPLITS}

    for root in args.root:
        names = read_yolo_names(root)
        kind = 'yolo' if names else 'coco'
        prefix = os.path.basename(root.rstrip('/\\'))
        print(f'\n{prefix}: format={kind}' + (f' names={names}' if names else ''))
        for split in SPLITS:
            recs = (load_yolo_split(root, split, names, unmapped) if kind == 'yolo'
                    else load_coco_split(root, split, unmapped))
            if recs:
                n_ann = sum(len(r[4]) for r in recs)
                print(f'  {split:6} {len(recs):5d} images, {n_ann:5d} boxes')
                per_split[split].append((prefix, recs))

    if unmapped:
        print(f'\nDROPPED unmapped classes (add to NAME_MAP if any are ours): {unmapped}')

    # Roboflow exports often have no valid split, or a tiny one. Fall back to
    # borrowing from test so validation is never empty.
    if not per_split['valid'] and per_split['test']:
        print('\nno valid split; using test as validation')
        per_split['valid'] = per_split.pop('test')

    os.makedirs(args.out, exist_ok=True)
    totals = {}
    for split in ('train', 'valid'):
        out_dir = os.path.join(args.out, split)
        images, annotations, iid, aid = [], [], 0, 0

        if args.merge_with:
            src_ann = os.path.join(args.merge_with, split, '_annotations.coco.json')
            if os.path.exists(src_ann):
                with open(src_ann) as f:
                    base = json.load(f)
                recs = []
                by_img = {}
                for a in base['annotations']:
                    by_img.setdefault(a['image_id'], []).append(a)
                os.makedirs(out_dir, exist_ok=True)
                for im in base['images']:
                    p = os.path.join(args.merge_with, split, im['file_name'])
                    if not os.path.exists(p):
                        continue
                    shutil.copy2(p, os.path.join(out_dir, im['file_name']))
                    images.append({'id': iid, 'file_name': im['file_name'],
                                   'width': im['width'], 'height': im['height'],
                                   'source': 'synthetic'})
                    for a in by_img.get(im['id'], []):
                        annotations.append({'id': aid, 'image_id': iid,
                                            'category_id': a['category_id'],
                                            'bbox': a['bbox'], 'area': a['area'],
                                            'iscrowd': 0, 'segmentation': []})
                        aid += 1
                    iid += 1
                print(f'\nmerged {len(images)} synthetic images into {split}')

        for prefix, recs in per_split[split]:
            imgs, anns, iid, aid = write_split(recs, out_dir, prefix, iid, aid)
            images += imgs
            annotations += anns

        if not images:
            continue
        with open(os.path.join(out_dir, '_annotations.coco.json'), 'w') as f:
            json.dump({'images': images, 'annotations': annotations,
                       'categories': CATEGORIES}, f)
        n_p = sum(1 for a in annotations if a['category_id'] == 2)
        n_f = sum(1 for a in annotations if a['category_id'] == 1)
        n_real = sum(1 for i in images if i.get('source') != 'synthetic')
        totals[split] = (len(images), n_real, n_f, n_p)
        print(f'{split}: {len(images)} images ({n_real} real), '
              f'forklift={n_f} person={n_p} -> {out_dir}')

    print(f'\nwrote {args.out}')
    if totals.get('valid', (0, 0, 0, 0))[1] == 0:
        print('WARNING: validation contains no REAL images, so real-world accuracy '
              'stays unmeasurable. Move a real subset into valid before quoting a '
              'number to J&J.')
    print(f'\nNext:\n  python -m scripts.train_detector --dataset-dir {args.out} '
          f'--output-dir models/rfdetr_real --epochs 25')


if __name__ == '__main__':
    main()
