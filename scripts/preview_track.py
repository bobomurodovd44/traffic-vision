"""Dev-only sanity check: run YOLO detection + built-in tracker over a short
slice of a video and dump a few annotated frames + a summary, so we can
eyeball detection/tracking quality before writing solution.py for real.

Usage:
    python scripts/preview_track.py dev_data/C3896_proxy.mp4 --seconds 20
"""
from __future__ import annotations

import argparse
import pathlib
import time

import cv2
from ultralytics import YOLO

# COCO ids we care about for traffic scenes
COCO_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                 5: "bus", 7: "truck", 9: "traffic light"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=pathlib.Path)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("dev_data"))
    ap.add_argument("--sample-every-sec", type=float, default=3.0,
                     help="save an annotated frame this often for inspection")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(args.seconds * fps)

    args.out_dir.mkdir(exist_ok=True)
    next_sample_t = 0.0
    max_track_id = -1
    det_counts = []
    t0 = time.perf_counter()

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        results = model.track(frame, persist=True, verbose=False,
                               classes=list(COCO_CLASSES.keys()))
        r = results[0]
        n_det = 0 if r.boxes is None else len(r.boxes)
        det_counts.append(n_det)
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            if ids:
                max_track_id = max(max_track_id, max(ids))

        if t >= next_sample_t:
            annotated = r.plot()
            out_path = args.out_dir / f"preview_t{int(t):03d}.jpg"
            cv2.imwrite(str(out_path), annotated)
            next_sample_t += args.sample_every_sec

    cap.release()
    elapsed = time.perf_counter() - t0
    n = len(det_counts)
    print(f"processed {n} frames ({n / fps:.1f}s of video) in {elapsed:.1f}s "
          f"({elapsed / max(n, 1) * 1000:.0f} ms/frame)")
    print(f"detections/frame: min={min(det_counts)} max={max(det_counts)} "
          f"avg={sum(det_counts) / max(n, 1):.1f}")
    print(f"max track id seen: {max_track_id}")


if __name__ == "__main__":
    main()
