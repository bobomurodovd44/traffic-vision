"""Dev-only helper: cut a short, downscaled proxy clip from a full-size sample
video so we can iterate quickly on a CPU dev machine. Not part of the
submission - the real solution.py must still work on full-res/full-length
videos at eval time.

Usage:
    python scripts/make_dev_proxy.py samples/C3896.MP4 --seconds 90 --width 960
"""
from __future__ import annotations

import argparse
import pathlib

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("dev_data"))
    args = ap.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    out_path = args.out_dir / f"{args.src.stem}_proxy.mp4"

    cap = cv2.VideoCapture(str(args.src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.width / src_w
    out_w, out_h = args.width, int(round(src_h * scale / 2) * 2)  # even height

    n_frames = int(args.seconds * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    written = 0
    while written < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    print(f"wrote {out_path}: {written} frames @ {fps:.2f} fps, {out_w}x{out_h} "
          f"(~{written / fps:.1f}s)")


if __name__ == "__main__":
    main()
