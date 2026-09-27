"""Interactive scene calibration: click zones on a reference frame, save to
calibration/scene.json in the schema src/calibration.py expects.

Run locally (needs a display -- not usable over a headless SSH session):

    python scripts/calibrate_scene.py --frame dev_data/frame_t45.jpg

Controls:
    left click     add a point to the shape currently being drawn
    n              finish current shape, start a new one of the same kind
    1 / 2 / 3 / 4  switch kind: parking zone / crosswalk / carriageway / stop line
    u              undo last point
    z              undo last finished shape (of the current kind)
    s              save calibration/scene.json
    q / ESC        quit (prompts to save if there are unsaved changes)

A stop line needs exactly 2 points for the line, then 1 more point placed on
the "approach" side (the side vehicles are on before they cross it) --
the tool prompts for this automatically after the 2nd point.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2

KIND_NAMES = {1: "parking_zones", 2: "crosswalks", 3: "carriageway", 4: "stop_lines"}
KIND_COLORS = {1: (0, 0, 255), 2: (0, 255, 255), 3: (255, 0, 0), 4: (0, 128, 255)}

OUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "calibration" / "scene.json"


def load_existing(path: pathlib.Path, frame_w: int, frame_h: int) -> dict:
    if not path.exists():
        return {"parking_zones": [], "crosswalks": [], "carriageway": [], "stop_lines": []}
    raw = json.loads(path.read_text())
    rw, rh = raw.get("ref_frame_size", [frame_w, frame_h])
    if [rw, rh] != [frame_w, frame_h]:
        print(f"warning: existing calibration ref size {rw}x{rh} != this frame "
              f"{frame_w}x{frame_h}; points will look wrong until re-saved")
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="path to a reference frame (jpg/png)")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    img = cv2.imread(args.frame)
    if img is None:
        raise SystemExit(f"cannot read {args.frame}")
    h, w = img.shape[:2]

    out_path = pathlib.Path(args.out)
    existing = load_existing(out_path, w, h)
    shapes = {k: list(existing.get(k, [])) for k in KIND_NAMES.values()}

    kind = 1
    current_pts: list[list[int]] = []
    dirty = False

    def redraw():
        canvas = img.copy()
        for k_id, k_name in KIND_NAMES.items():
            color = KIND_COLORS[k_id]
            for shape in shapes[k_name]:
                if k_name == "stop_lines":
                    p1 = (int(shape["p1"][0] * w), int(shape["p1"][1] * h))
                    p2 = (int(shape["p2"][0] * w), int(shape["p2"][1] * h))
                    ap_ = (int(shape["approach_point"][0] * w), int(shape["approach_point"][1] * h))
                    cv2.line(canvas, p1, p2, color, 2)
                    cv2.circle(canvas, ap_, 5, color, -1)
                else:
                    pts = [(int(px * w), int(py * h)) for px, py in shape["polygon"]]
                    cv2.polylines(canvas, [cv2_points(pts)], True, color, 2)
        color = KIND_COLORS[kind]
        for i, (px, py) in enumerate(current_pts):
            cv2.circle(canvas, (px, py), 4, color, -1)
            if i > 0:
                cv2.line(canvas, tuple(current_pts[i - 1]), (px, py), color, 1)
        label = f"kind={KIND_NAMES[kind]} pts={len(current_pts)} dirty={dirty}"
        cv2.putText(canvas, label, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("calibrate", canvas)

    def cv2_points(pts):
        import numpy as np
        return np.array(pts)

    def finish_shape():
        nonlocal current_pts, dirty
        k_name = KIND_NAMES[kind]
        if k_name == "stop_lines":
            if len(current_pts) != 3:
                print("stop line needs exactly 3 points: p1, p2, approach_point -- ignoring")
                current_pts = []
                return
            (x1, y1), (x2, y2), (ax, ay) = current_pts
            shapes[k_name].append({
                "name": f"{k_name}_{len(shapes[k_name])}",
                "p1": [round(x1 / w, 4), round(y1 / h, 4)],
                "p2": [round(x2 / w, 4), round(y2 / h, 4)],
                "approach_point": [round(ax / w, 4), round(ay / h, 4)],
            })
        else:
            if len(current_pts) < 3:
                print("polygon needs >= 3 points -- ignoring")
                current_pts = []
                return
            shapes[k_name].append({
                "name": f"{k_name}_{len(shapes[k_name])}",
                "polygon": [[round(px / w, 4), round(py / h, 4)] for px, py in current_pts],
            })
        current_pts = []
        dirty = True

    def on_mouse(event, x, y, flags, userdata):
        nonlocal current_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append([x, y])
            if kind == 4 and len(current_pts) == 3:
                finish_shape()
            redraw()

    def save():
        nonlocal dirty
        payload = {
            "video_ref": existing.get("video_ref", "unspecified"),
            "ref_frame_size": [w, h],
            "notes": existing.get("notes", "Calibrated with scripts/calibrate_scene.py"),
            **{k: shapes[k] for k in KIND_NAMES.values()},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"saved {out_path}")
        dirty = False

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    redraw()
    print(__doc__)

    while True:
        redraw()
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            if dirty:
                print("unsaved changes -- press s to save before quitting, or q again to discard")
                key2 = cv2.waitKey(0) & 0xFF
                if key2 in (ord("q"), 27):
                    break
                elif key2 == ord("s"):
                    save()
                    break
            else:
                break
        elif key == ord("s"):
            save()
        elif key == ord("n"):
            finish_shape()
        elif key == ord("u") and current_pts:
            current_pts.pop()
        elif key == ord("z"):
            k_name = KIND_NAMES[kind]
            if shapes[k_name]:
                shapes[k_name].pop()
                dirty = True
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            if current_pts:
                print("finish or discard the current shape before switching kind")
            else:
                kind = int(chr(key))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
