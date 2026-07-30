"""Extract diverse training stills from raw video (implementation guide §2.2).

Adjacent video frames are near-identical; a detector trained on them learns
nothing new per frame. One frame per second gives genuine visual diversity.

Usage:
    python -m src.extract_frames
    python -m src.extract_frames --videos data/raw_videos --out data/frames --every 0.5
"""

import argparse
import glob
import os

import cv2


def extract_frames(video_path, out_dir, every_n_seconds=1.0):
    """Write one JPEG every `every_n_seconds` of `video_path` into `out_dir`.

    Filenames embed the source video name, which is what makes the
    split-by-video rule of §4.2 possible later: you can tell at a glance which
    batch an image belongs to, and Roboflow batches follow the same grouping.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps * every_n_seconds))
    name = os.path.splitext(os.path.basename(video_path))[0]
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"{name}_{idx:06d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", default="data/raw_videos", help="directory of source clips")
    ap.add_argument("--out", default="data/frames", help="output directory for stills")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between saved frames")
    args = ap.parse_args()

    paths = [p for p in sorted(glob.glob(os.path.join(args.videos, "*")))
             if p.lower().endswith(VIDEO_EXTS)]
    if not paths:
        print(f"No video files found in {args.videos}/ — place clips there first (§2.1).")
        return

    total = 0
    for vp in paths:
        n = extract_frames(vp, args.out, args.every)
        print(f"{vp}: {n} frames")
        total += n
    print("TOTAL:", total)

    # ACCEPTANCE CHECK (§2.2): 800-1500 frames spanning every camera/lighting
    # condition. Under ~300 means re-extract at --every 0.5 or get more video.
    if total < 300:
        print(f"\nWARNING: only {total} frames. Re-run with --every 0.5, or obtain "
              "more footage — 300 is the floor for a working prototype (§4.1).")
    elif total > 1500:
        print(f"\nNOTE: {total} frames is more than 5 people can label. Consider "
              "--every 2.0, or label a diverse subset.")


if __name__ == "__main__":
    main()
