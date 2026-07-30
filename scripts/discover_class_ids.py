"""Read class IDs from a COCO annotation file (implementation guide §4.4).

Never hardcode these from memory. Roboflow's COCO export sometimes inserts a
dummy category at index 0, which shifts everything — and the symptom is blank or
swapped class names at inference, not a crash (context.md §8.2).

Also reports per-class instance counts and per-split image counts, because when
forklift mAP comes back low the cause is almost always visible here: too few
forklift instances, or a split that is far from the intended proportion.

Usage:
    python -m scripts.discover_class_ids data/dataset
"""

import argparse
import json
import os
from collections import Counter


def inspect_split(path):
    with open(path) as f:
        data = json.load(f)
    cats = {c['id']: c['name'] for c in data['categories']}
    counts = Counter(a['category_id'] for a in data['annotations'])
    return cats, counts, len(data['images'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset_dir', nargs='?', default='data/dataset')
    args = ap.parse_args()

    splits = {}
    for split in ('train', 'valid', 'test'):
        p = os.path.join(args.dataset_dir, split, '_annotations.coco.json')
        if os.path.exists(p):
            splits[split] = inspect_split(p)

    if not splits:
        raise SystemExit(
            f'No _annotations.coco.json under {args.dataset_dir}/(train|valid|test)/.\n'
            'Export from Roboflow as format "COCO" (not YOLO) and unpack there (§4.3).')

    cats = splits[next(iter(splits))][0]
    print('Categories (from the dataset, authoritative):')
    for cid, name in sorted(cats.items()):
        print(f'  {cid}: {name}')

    print('\nInstances per class:')
    header = f"  {'class':<16}" + ''.join(f'{s:>10}' for s in splits)
    print(header)
    for cid, name in sorted(cats.items()):
        row = f'  {name:<16}' + ''.join(f'{splits[s][1].get(cid, 0):>10}' for s in splits)
        print(row)
    print(f"  {'IMAGES':<16}" + ''.join(f'{splits[s][2]:>10}' for s in splits))

    lower = {v.lower(): k for k, v in cats.items()}
    person = lower.get('person')
    forklift = lower.get('forklift')

    print('\nPaste into src/run_pipeline.py (or pass --person-id/--forklift-id):')
    print(f'  PERSON_ID   = {person if person is not None else "?  # NOT FOUND"}')
    print(f'  FORKLIFT_ID = {forklift if forklift is not None else "?  # NOT FOUND"}')

    dummy = [n for i, n in cats.items() if i == 0 and n.lower() not in ('person', 'forklift')]
    if dummy:
        print(f'\nNOTE: category 0 is "{dummy[0]}" — the Roboflow dummy category. This is '
              'exactly the shift that makes hardcoded IDs wrong; the values above account '
              'for it.')

    if person is None or forklift is None:
        print('\nWARNING: expected classes named exactly "person" and "forklift". '
              'Rename them in Roboflow and re-export, or the pipeline defaults will '
              'silently address the wrong classes.')

    for split in splits:
        cats_s, counts_s, n_img = splits[split]
        if n_img == 0:
            print(f'\nWARNING: split "{split}" has no images.')
        for cid, name in sorted(cats_s.items()):
            if name.lower() in ('person', 'forklift') and counts_s.get(cid, 0) == 0:
                print(f'\nWARNING: no "{name}" instances in "{split}" — mAP for it will '
                      'be meaningless (§4.6 blames data first, and rightly).')

    if 'valid' in splits and splits['valid'][2] < 100:
        print(f"\nNOTE: valid split has {splits['valid'][2]} images. §4.1 suggests ~150 "
              'labeled first so every training run is measurable.')

    print('\nReminder: confirm the split was made BY SOURCE VIDEO, not randomly by frame. '
          'A random frame-level split leaks near-duplicates across train/valid and '
          'produces beautiful, fictional metrics (§4.2).')


if __name__ == '__main__':
    main()
