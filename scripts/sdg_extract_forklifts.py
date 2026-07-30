"""Recover sit-down forklift labels from SDG instance segmentation.

WHY THIS EXISTS
---------------
J&J operates sit-down counterbalance forklifts. The `nearmiss` scenario we train
on uses stand-on reach trucks, so a detector trained on it alone may not
recognise J&J's actual vehicle.

`box_pickup` contains the right forklift — but its `object_detection.jsonl`
annotates only `character`, so the forklift is unlabeled. Training on those
frames as-is would teach the detector that J&J's vehicle is *background*, which
is worse than omitting them.

The instance-segmentation tier does register it, as
`/Root/forklift/S_ForkliftBody` and `/Root/forklift/S_ForkliftFork`. So exact
boxes are recoverable: take the union of those prims' pixels per frame and bound
it. Union of body + forks matches labeling convention 2 ("box the whole vehicle
including mast and forks").

WHAT THIS CANNOT DO
-------------------
The `box_pickup` forklift is **parked and empty** — no driver in the cab. So this
recovers *appearance* data for the detector and nothing else. It does not give
Rule 5 a seated driver, and no SDG scenario does. Rule 5 still needs staged
footage or custom simulation.

The forklift is also stationary, so every frame of a clip yields an identical
box. Diversity comes from cameras and runs, never from sampling more frames of
one clip.

OUTPUT
------
`<run>/<cam>.forklift_boxes.json`, which `SDGClip.agents()` picks up
automatically, so `scripts/sdg_to_coco.py` needs no changes.

Usage:
    python -m scripts.sdg_extract_forklifts --shards 1 --cameras cam_00 cam_01
"""

import argparse
import json
import os
import tarfile

import numpy as np

REPO = 'datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes'
HF_DIR = 'warehouse_box_pickup'
SHARD_FMT = 'box_pickup-artifacts-{:05d}.tar'
SEG_SUFFIXES = ('.instance_id_segmentation.npz',
                '.instance_id_segmentation_mapping.jsonl')

PRIM_PATTERN = '/forklift'
MIN_PIXELS = 1500          # below this the vehicle is too small/occluded to teach


def parse_member(name):
    head, camera, kind = name.split('.', 2)
    run_id = head.split('_', 1)[1] if '_' in head else head
    return run_id, camera, kind


def fetch_segmentation(shard_idx, out_root, cameras):
    """Download only the segmentation pair for the requested cameras."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    path = f'{REPO}/artifacts/{HF_DIR}/{SHARD_FMT.format(shard_idx)}'
    print(f'=== shard {shard_idx}: indexing ===')
    written = []
    with fs.open(path, 'rb') as f:
        tf = tarfile.open(fileobj=f, mode='r')
        members = [m for m in tf.getmembers() if m.name.endswith(SEG_SUFFIXES)]
        wanted = []
        for m in members:
            run_id, camera, kind = parse_member(m.name)
            if cameras and camera not in cameras:
                continue
            wanted.append((m, run_id, camera, kind))
        print(f'{len(wanted)} files, {sum(m.size for m, *_ in wanted) / 2**20:.0f} MiB')
        for m, run_id, camera, kind in wanted:
            d = os.path.join(out_root, run_id)
            os.makedirs(d, exist_ok=True)
            dest = os.path.join(d, f'{camera}.{kind}')
            if os.path.exists(dest) and os.path.getsize(dest) == m.size:
                written.append(dest)
                continue
            with open(dest, 'wb') as o:
                o.write(tf.extractfile(m).read())
            print(f'  {m.size / 2**20:7.1f} MiB  {run_id[:24]}/{camera}.{kind}')
            written.append(dest)
    return written


def extract_boxes(run_dir, camera):
    """Union-of-forklift-prims bounding box per frame index."""
    npz = os.path.join(run_dir, f'{camera}.instance_id_segmentation.npz')
    mapping = os.path.join(run_dir, f'{camera}.instance_id_segmentation_mapping.jsonl')
    if not (os.path.exists(npz) and os.path.exists(mapping)):
        return None

    z = np.load(npz)
    frames, frame_indices = z['frames'], z['frame_indices']
    maps = [json.loads(line) for line in open(mapping)]

    out = {}
    for i in range(len(frames)):
        m = maps[i] if i < len(maps) else maps[-1]
        colours = [eval(k) if isinstance(k, str) else k
                   for k, v in m.items() if PRIM_PATTERN in str(v)]
        if not colours:
            continue
        seg = frames[i]
        mask = np.zeros(seg.shape[:2], dtype=bool)
        for c in colours:
            mask |= np.all(seg == np.array(c, dtype=np.uint8), axis=-1)
        n = int(mask.sum())
        if n < MIN_PIXELS:
            continue
        ys, xs = np.where(mask)
        out[int(frame_indices[i])] = {
            'box': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            'pixels': n,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', type=int, nargs='+', default=[1])
    ap.add_argument('--root', default='data/sdg_box_pickup')
    ap.add_argument('--cameras', nargs='*',
                    default=['cam_00', 'cam_01', 'cam_02', 'cam_03', 'cam_04'])
    ap.add_argument('--skip-fetch', action='store_true',
                    help='use segmentation files already on disk')
    ap.add_argument('--keep-npz', action='store_true',
                    help='keep the ~140 MB/camera segmentation files after extraction')
    args = ap.parse_args()

    if not args.skip_fetch:
        for s in args.shards:
            fetch_segmentation(s, args.root, set(args.cameras))

    total_cams = with_forklift = total_frames = 0
    for run in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, run)
        if not os.path.isdir(run_dir):
            continue
        cams = sorted({f.split('.')[0] for f in os.listdir(run_dir)
                       if f.endswith('.instance_id_segmentation.npz')})
        for cam in cams:
            total_cams += 1
            boxes = extract_boxes(run_dir, cam)
            if not boxes:
                print(f'  {run[:24]}/{cam}: no forklift visible')
                continue
            with_forklift += 1
            total_frames += len(boxes)
            dest = os.path.join(run_dir, f'{cam}.forklift_boxes.json')
            with open(dest, 'w') as f:
                json.dump(boxes, f)
            sample = next(iter(boxes.values()))
            print(f'  {run[:24]}/{cam}: {len(boxes)} frames, e.g. {sample["box"]} '
                  f'({sample["pixels"]} px) -> {os.path.basename(dest)}')

            if not args.keep_npz:
                for suffix in SEG_SUFFIXES:
                    p = os.path.join(run_dir, f'{cam}{suffix}')
                    if os.path.exists(p):
                        os.remove(p)

    print(f'\n{with_forklift}/{total_cams} cameras show a forklift; '
          f'{total_frames} annotated frames')
    if with_forklift:
        print('SDGClip.agents() now returns these boxes, so re-export with:\n'
              f'  python -m scripts.sdg_to_coco --root {args.root} --out data/dataset_bp')
    print('\nReminder: this forklift is PARKED AND EMPTY. It teaches the detector '
          'what a sit-down forklift looks like; it gives Rule 5 nothing.')


if __name__ == '__main__':
    main()
