"""Fetch what we need from NVIDIA PhysicalAI SDG-Warehouse.

Dataset: nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes
Licence: OpenMDW 1.1 — commercial use permitted, so it is compatible with the
J&J handoff (context.md §3).

The full dataset is 18.57 TB. We take a thin slice:

  * the `forklift_human_nearmiss` scenario only — it is the one that stages
    pedestrian/vehicle proximity, i.e. Rule 3;
  * `ceiling_*` cameras only — the elevated CCTV-like viewpoint that matches
    where J&J's cameras actually are (`eye_*` is head-height and unrepresentative);
  * three files per clip: rgb.mp4, object_detection.jsonl, camera_params.jsonl.

Everything else in the tar (depth, edges, segmentation, and a 96 MB
instance-id mapping per camera) is skipped. That is a 17x saving: 272 MiB per
shard instead of 4.6 GiB.

We read from the `artifacts/` tier, NOT `rgb/`. The artifacts tar carries
rgb.mp4 *alongside* the annotations, so the pixels and the boxes are known to
come from the same render — verified pixel-exact. The separate `rgb/` tier is
Cosmos world-model output (see the `cosmos_generation` S3 paths) and its
alignment with these boxes is unverified.

The tar is opened over HTTP with random access so unwanted members are seeked
past rather than downloaded. Indexing a shard's headers costs ~3 min; that is
still far cheaper than pulling the whole shard.

Usage:
    python -m scripts.fetch_sdg --shards 1 2 3
    python -m scripts.fetch_sdg --shards 1 --cameras ceiling_00 ceiling_01
"""

import argparse
import os
import tarfile
import time

REPO = 'datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes'
SCENARIO = 'forklift_human_nearmiss'
SHARD_FMT = 'nearmiss-artifacts-{:05d}.tar'
WANTED_SUFFIXES = ('.rgb.mp4', '.object_detection.jsonl', '.camera_params.jsonl')

# Shard 0 is a 73 GiB outlier (the rest are ~4.6 GiB) and indexes very slowly.
# Skip it unless explicitly asked for.
ODD_SHARDS = {0}


def parse_member(name):
    """`<hash>_<run_id>.<camera>.<type>` -> (run_id, camera, type)."""
    head, camera, kind = name.split('.', 2)
    run_id = head.split('_', 1)[1] if '_' in head else head
    return run_id, camera, kind


def fetch_shard(fs, shard_idx, out_root, cameras, dry_run=False):
    path = f'{REPO}/artifacts/{SCENARIO}/{SHARD_FMT.format(shard_idx)}'
    print(f'\n=== shard {shard_idx} ===')
    t0 = time.time()
    with fs.open(path, 'rb') as f:
        tf = tarfile.open(fileobj=f, mode='r')       # random access
        members = tf.getmembers()
        print(f'indexed {len(members)} members in {time.time() - t0:.0f}s')

        wanted = []
        for m in members:
            if not m.name.endswith(WANTED_SUFFIXES):
                continue
            run_id, camera, kind = parse_member(m.name)
            if cameras and camera not in cameras:
                continue
            wanted.append((m, run_id, camera, kind))

        total = sum(m.size for m, *_ in wanted)
        print(f'{len(wanted)} files, {total / 2**20:.0f} MiB')
        if dry_run:
            return 0

        written = 0
        for m, run_id, camera, kind in wanted:
            dest_dir = os.path.join(out_root, run_id)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, f'{camera}.{kind}')
            if os.path.exists(dest) and os.path.getsize(dest) == m.size:
                continue                              # resumable
            with open(dest, 'wb') as o:
                o.write(tf.extractfile(m).read())
            written += 1
            print(f'  {m.size / 2**20:7.1f} MiB  {run_id[:28]}/{camera}.{kind}')
        return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', type=int, nargs='+', default=[1],
                    help='artifacts shard indices (each holds ~3 runs x 5 ceiling cameras)')
    ap.add_argument('--out', default='data/sdg', help='output root')
    ap.add_argument('--cameras', nargs='*',
                    default=['ceiling_00', 'ceiling_01', 'ceiling_02',
                             'ceiling_03', 'ceiling_04'],
                    help='camera aliases to keep; pass empty to keep all')
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would be fetched without downloading')
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem

    for s in args.shards:
        if s in ODD_SHARDS:
            print(f'shard {s} is the 73 GiB outlier; skipping. Remove it from '
                  '--shards or edit ODD_SHARDS if you really want it.')
    shards = [s for s in args.shards if s not in ODD_SHARDS]

    fs = HfFileSystem()
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for s in shards:
        total += fetch_shard(fs, s, args.out, set(args.cameras), args.dry_run)

    runs = [d for d in os.listdir(args.out)
            if os.path.isdir(os.path.join(args.out, d))] if os.path.exists(args.out) else []
    print(f'\nwrote {total} new files; {len(runs)} runs now under {args.out}/')
    print('Next: python -m scripts.sdg_to_coco')


if __name__ == '__main__':
    main()
