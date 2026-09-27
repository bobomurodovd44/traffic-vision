"""Live-ish visualization: YOLO+ByteTrack boxes, calibration zones, and
currently-active rule events drawn on top of the video, in a window and/or
saved to an annotated output file.

Processing is CPU-bound with no realtime guarantee -- frames are shown as
fast as they're processed, not paced to the source video's timestamps. On
this dev machine that's noticeably slower than real playback speed; the
organizers' GPU eval machine would be much faster.

Usage:
    python scripts/visualize.py --video dev_data/C3896_proxy.mp4
    python scripts/visualize.py --video samples/C3896.MP4 --save dev_data/demo.mp4 --no-display

Press q to quit the live window.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.calibration import SceneCalibration
from src.detection import TrafficDetector, probe_frame_size
from src.rules import detect_congestion, detect_jaywalking, detect_stopped_vehicle

CLASS_COLORS = {
    "person": (255, 255, 0),
    "car": (0, 255, 0),
    "truck": (0, 255, 0),
    "bus": (0, 255, 0),
    "motorcycle": (0, 200, 255),
    "bicycle": (255, 0, 200),
}
EVENT_COLOR = (0, 0, 255)

ZONE_STYLES = [
    ("carriageway", (255, 0, 0), 1),
    ("parking_zones", (0, 0, 255), 2),
    ("crosswalks", (0, 255, 255), 2),
]


def draw_calibration(frame: np.ndarray, cal) -> None:
    for attr, color, thickness in ZONE_STYLES:
        for zone in getattr(cal, attr):
            pts = np.array(zone.polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, thickness)
    for sl in cal.stop_lines:
        p1 = (int(sl.p1[0]), int(sl.p1[1]))
        p2 = (int(sl.p2[0]), int(sl.p2[1]))
        cv2.line(frame, p1, p2, (0, 128, 255), 2)


def draw_tracks(frame: np.ndarray, tracks) -> None:
    for tr in tracks:
        color = CLASS_COLORS.get(tr.cls, (255, 255, 255))
        x1, y1, x2, y2 = (int(v) for v in tr.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{tr.cls}#{tr.id}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_event_banner(frame: np.ndarray, t: float, events: dict) -> None:
    cv2.putText(frame, f"t={t:6.1f}s", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y = 56
    for label, segs in events.items():
        if any(s <= t <= e for s, e, _ in segs):
            cv2.putText(frame, f"** {label.upper()} **", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, EVENT_COLOR, 2)
            y += 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="dev_data/C3896_proxy.mp4")
    ap.add_argument("--target-fps", type=float, default=8.0,
                     help="sampling rate for detection, independent of source fps")
    ap.add_argument("--save", default=None, help="optional output annotated mp4 path")
    ap.add_argument("--no-display", action="store_true", help="skip the live window, just save")
    ap.add_argument("--recompute-every", type=float, default=2.0,
                     help="seconds of video between re-running the event rules on history so far")
    ap.add_argument("--max-seconds", type=float, default=None,
                     help="stop after this many seconds of video (for a quick preview)")
    args = ap.parse_args()

    detector = TrafficDetector()
    width, height = probe_frame_size(args.video)
    calibration = SceneCalibration().for_frame(width, height)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / args.target_fps))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.target_fps, (width, height))

    samples: list = []
    events = {"stopped_vehicle": [], "congestion": [], "jaywalking": []}
    last_recompute_t = -1e9
    idx = 0
    detector.reset()
    wall_start = time.time()

    try:
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                t = idx / fps
                if args.max_seconds is not None and t > args.max_seconds:
                    break
                tracks = detector.track_frame(frame)
                samples.append((t, tracks))

                if t - last_recompute_t >= args.recompute_every:
                    events["stopped_vehicle"] = detect_stopped_vehicle(samples, calibration=calibration)
                    events["congestion"] = detect_congestion(samples, calibration=calibration)
                    events["jaywalking"] = detect_jaywalking(samples, calibration=calibration)
                    last_recompute_t = t

                vis = frame.copy()
                draw_calibration(vis, calibration)
                draw_tracks(vis, tracks)
                draw_event_banner(vis, t, events)

                if writer is not None:
                    writer.write(vis)
                if not args.no_display:
                    cv2.imshow("traffic-vision", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                if not cap.grab():
                    break
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    wall_elapsed = time.time() - wall_start
    video_elapsed = idx / fps
    print(f"processed {video_elapsed:.1f}s of video in {wall_elapsed:.1f}s wall-clock "
          f"({video_elapsed / max(wall_elapsed, 1e-6):.2f}x realtime)")
    for label, segs in events.items():
        print(f"{label}: {segs}")


if __name__ == "__main__":
    main()
