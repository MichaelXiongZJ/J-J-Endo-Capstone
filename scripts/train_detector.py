"""Fine-tune RF-DETR on the COCO dataset built by scripts/sdg_to_coco.py (§4.5).

Headless equivalent of notebooks/02_train.ipynb, so it can run unattended.

Starting from COCO weights is why ~500 images suffice instead of ~100,000: the
pretrained weights already encode generic visual features, so we are teaching one
new class and adapting to these cameras, not learning vision from scratch.

Defaults are sized for an 8 GB GPU (RTX 3070): batch_size 2 with grad_accum 8
gives the same effective batch of 16 as the guide's T4 settings.

Usage:
    python -m scripts.train_detector --epochs 25
    python -m scripts.train_detector --epochs 1 --smoke   # API check, ~2 min
"""

import argparse
import json
import os
import shutil
import sys
import time

# RF-DETR prints its per-epoch metrics through `rich`, which emits box-drawing
# characters. On Windows the default console encoding is cp1252 and cannot encode
# them, so training crashes at the END of an epoch — after the expensive part,
# with the weights not yet saved. Force UTF-8 before anything prints.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


def ensure_test_split(dataset_dir):
    """RF-DETR's Roboflow loader expects train/valid/test. sdg_to_coco writes only
    train and valid (a third split of the same few runs would be noise, not
    signal), so mirror valid into test if it is missing. Nothing is reported from
    the test split — validation mAP is the number that matters.
    """
    test_dir = os.path.join(dataset_dir, 'test')
    ann = os.path.join(test_dir, '_annotations.coco.json')
    if os.path.exists(ann):
        return
    valid_dir = os.path.join(dataset_dir, 'valid')
    os.makedirs(test_dir, exist_ok=True)
    for f in os.listdir(valid_dir):
        src, dst = os.path.join(valid_dir, f), os.path.join(test_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f'mirrored valid -> test ({len(os.listdir(test_dir))} files)')


def describe(dataset_dir):
    for split in ('train', 'valid'):
        p = os.path.join(dataset_dir, split, '_annotations.coco.json')
        with open(p) as f:
            d = json.load(f)
        cats = {c['id']: c['name'] for c in d['categories']}
        from collections import Counter
        n = Counter(a['category_id'] for a in d['annotations'])
        counts = ', '.join(f'{cats[k]}={v}' for k, v in sorted(n.items()))
        print(f'  {split:6} {len(d["images"]):4d} images | {counts}')
    return cats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-dir', default='data/dataset')
    ap.add_argument('--output-dir', default='models/rfdetr_v1')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=2, help='2 is 8 GB-safe')
    ap.add_argument('--grad-accum-steps', type=int, default=8,
                    help='batch_size * this = effective batch (16)')
    ap.add_argument('--lr', type=float, default=1e-4,
                    help='drop to 5e-5 if person accuracy regresses '
                         '(catastrophic forgetting, §4.5)')
    ap.add_argument('--resolution', type=int, default=None,
                    help='must be divisible by 56; raises cost, helps small/distant people')
    ap.add_argument('--smoke', action='store_true', help='1 epoch, tiny, just checks the API')
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f'dataset: {dataset_dir}')
    cats = describe(dataset_dir)
    print(f'categories: {cats}')
    ensure_test_split(dataset_dir)

    import torch
    print(f'torch {torch.__version__} | cuda {torch.cuda.is_available()} | '
          f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')

    from rfdetr import RFDETRBase
    kwargs = {'resolution': args.resolution} if args.resolution else {}
    model = RFDETRBase(**kwargs)          # COCO weights = transfer learning

    epochs = 1 if args.smoke else args.epochs
    print(f'\ntraining {epochs} epoch(s), batch {args.batch_size} x '
          f'{args.grad_accum_steps} accum, lr {args.lr}')
    t0 = time.time()
    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=output_dir,
        tensorboard=False,
        wandb=False,
    )
    mins = (time.time() - t0) / 60
    print(f'\ntraining finished in {mins:.1f} min -> {output_dir}')
    best = os.path.join(output_dir, 'checkpoint_best_ema.pth')
    print(f'best weights: {best} (exists: {os.path.exists(best)})')
    print('EMA = smoothed weights; use this checkpoint for inference.')


if __name__ == '__main__':
    main()
