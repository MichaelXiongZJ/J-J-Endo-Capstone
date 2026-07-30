"""Main pipeline (implementation guide §9): video in -> annotated video + events.jsonl.

Runs on recorded clips: deterministic and replayable. Pointing it at a live
RTSP URL later is a one-line change and is J&J's integration concern, not ours
(context.md §7.8).

Run:
    python -m src.run_pipeline \
        --video data/raw_videos/test_clip.mp4 \
        --calib data/calibration/cam1.json \
        --weights models/rfdetr_v1/checkpoint_best_ema.pth
"""

import argparse
import os

import cv2
import supervision as sv
from trackers import ByteTrackTracker

from src.events import EventAggregator
from src.geometry import CameraGeometry
from src.pose_utils import match_pose_to_boxes, run_pose
from src.rules import (CFG, MotionState, Rule1State, Rule4State, Rule5State,
                       TrackedObject, check_rule3, find_driver)

# From §4.4 — read these from your dataset's _annotations.coco.json, never from
# memory. Roboflow sometimes inserts a dummy category at index 0, shifting
# everything. Override at the command line with --person-id / --forklift-id.
PERSON_ID, FORKLIFT_ID = 2, 1


def run(video, calib, detector, outdir='outputs', device='cuda', person_id=PERSON_ID,
        forklift_id=FORKLIFT_ID, use_pose=True, max_frames=None, write_video=True,
        threshold=0.5, save_evidence=True, verbose=True, video_label=None):
    """Core loop, decoupled from argument parsing so tests can drive it directly.

    video_label overrides the `video` field written into events. Needed when many
    clips share a basename (the SDG slice stores every run's cameras as
    `ceiling_00.rgb.mp4`), since score_events matches events to ground truth by
    clip name and would otherwise merge every run into one.
    """
    os.makedirs(f'{outdir}/events', exist_ok=True)
    os.makedirs(f'{outdir}/videos', exist_ok=True)

    geom = CameraGeometry(calib)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f'cannot open video: {video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src_fps / CFG['PROC_FPS']))
    # ByteTrack: persistent IDs across frames, no training. The guide calls
    # sv.ByteTrack, which supervision deprecated in 0.28 and removes in 0.30;
    # the implementation moved to Roboflow's `trackers` package (also Apache-2.0,
    # so the licence posture of context.md §7.1 is unchanged).
    #
    # frame_rate MUST match the rate frames are actually FED, not the video's
    # native rate. Mismatch makes the motion model misjudge how far objects
    # travel between updates, causing constant ID switches (§5 gotcha 1).
    tracker = ByteTrackTracker(frame_rate=CFG['PROC_FPS'])

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if write_video:
        writer = cv2.VideoWriter(f'{outdir}/videos/annotated.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'), CFG['PROC_FPS'], (W, H))

    motion, r1, r4, r5 = MotionState(), Rule1State(), Rule4State(), Rule5State()
    # supervision's defaults are tuned for ~640px images and are illegible at
    # 1080p, which matters because the annotated video is a deliverable people
    # actually watch (Definition of Done item 5).
    box_ann = sv.BoxAnnotator(thickness=3)
    lab_ann = sv.LabelAnnotator(text_scale=0.8, text_thickness=2, text_padding=6)
    agg = EventAggregator(f'{outdir}/events', geom.camera_id,
                          video_label or os.path.basename(video),
                          save_frames=save_evidence)

    frame_idx = processed = 0
    fake_ts_warned = False

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride:
            continue

        # REAL timestamps, never frame_number/fps: videos drop frames silently,
        # and assuming fixed fps corrupts every velocity computed (context.md §8.3).
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and processed > 0:
            # Some codecs/containers do not report position. Fall back, loudly.
            t = frame_idx / src_fps
            if not fake_ts_warned:
                print('WARNING: container reports no frame timestamps; falling back to '
                      'frame_idx/fps. Velocities will be wrong if frames were dropped.')
                fake_ts_warned = True

        processed += 1
        if max_frames and processed > max_frames:
            break

        # 1. detect + track. (The detector handles BGR->RGB internally.)
        dets = tracker.update(detector(bgr))
        if dets.tracker_id is None or len(dets) == 0:
            # Normal for the first ~3 processed frames while ByteTrack confirms
            # new tracks (§5 gotcha 2).
            agg.add([], t, bgr)
            if writer:
                writer.write(bgr)
            continue

        # 2. pose (BGR!), matched back onto OUR tracked person boxes.
        person_idx = [i for i in range(len(dets)) if int(dets.class_id[i]) == person_id]
        pose_by_box = {}
        if use_pose and person_idx:
            person_boxes = [tuple(map(float, dets.xyxy[i])) for i in person_idx]
            pose_by_box = match_pose_to_boxes(run_pose(bgr, device), person_boxes)

        # 3. unify into TrackedObjects with floor positions.
        people, vehicles = [], []
        for local, i in enumerate(person_idx):
            box = tuple(map(float, dets.xyxy[i]))
            obj = TrackedObject(int(dets.tracker_id[i]), person_id, box,
                                geom.floor_position(box))
            obj.keypoints = pose_by_box.get(local)
            people.append(obj)
        for i in range(len(dets)):
            if int(dets.class_id[i]) != forklift_id:
                continue
            box = tuple(map(float, dets.xyxy[i]))
            vehicles.append(TrackedObject(int(dets.tracker_id[i]), forklift_id, box,
                                          geom.floor_position(box)))
        for obj in people + vehicles:
            motion.update(obj, t)

        # 4. driver association, then rules.
        driver_ids, hits = set(), []
        driver_of = {}
        for v in vehicles:
            d = find_driver(v, people, motion)
            if d:
                driver_ids.add(d.track_id)
                driver_of[v.track_id] = d.track_id
                e5 = r5.check(d, v)
                if e5:
                    hits.append(e5)
        hits += check_rule3(people, vehicles, driver_ids, motion, geom, t)
        hits += r4.check(people, driver_ids, geom)
        hits += r1.check(people)

        # 5. aggregate into episodes (see src/events.py) and annotate.
        active = agg.add(hits, t, bgr)

        if writer:
            labels = []
            for i in range(len(dets)):
                tid = int(dets.tracker_id[i])
                cid = int(dets.class_id[i])
                kind = 'P' if cid == person_id else ('F' if cid == forklift_id else '?')
                labels.append(f'id{tid} {kind}' + (' DRV' if tid in driver_ids else ''))
            out = lab_ann.annotate(box_ann.annotate(bgr.copy(), dets), dets, labels)
            if active:
                cv2.putText(out, f'VIOLATION rule(s): {sorted(active)}', (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            writer.write(out)

        if verbose and processed % 100 == 0:
            print(f'  {processed} frames processed, t={t:.1f}s, {agg.count} events closed')

    cap.release()
    if writer:
        writer.release()
    n = agg.close()
    if verbose:
        print(f'done: {processed} frames processed, {n} events -> {agg.path}')
    return {'frames_processed': processed, 'events': n, 'events_path': agg.path}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--weights', required=True,
                    help='path to checkpoint_best_ema.pth, or "coco" for the '
                         'un-fine-tuned baseline (finds people, not forklifts)')
    ap.add_argument('--outdir', default='outputs')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--person-id', type=int, default=PERSON_ID)
    ap.add_argument('--forklift-id', type=int, default=FORKLIFT_ID)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--no-pose', action='store_true',
                    help='skip pose; disables Rules 5 and 1, ~3x faster')
    ap.add_argument('--max-frames', type=int, default=None)
    args = ap.parse_args()

    from src.detector import RFDetrDetector
    detector = RFDetrDetector(weights=None if args.weights == 'coco' else args.weights,
                              threshold=args.threshold)
    run(args.video, args.calib, detector, outdir=args.outdir, device=args.device,
        person_id=args.person_id, forklift_id=args.forklift_id,
        use_pose=not args.no_pose, max_frames=args.max_frames, threshold=args.threshold)


if __name__ == '__main__':
    main()
