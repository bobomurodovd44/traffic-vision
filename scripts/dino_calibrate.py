"""Zero-shot crosswalk + road-border detection with Grounding DINO, as an aid
for calibration/scene.json instead of hand-guessing polygons from screenshots.

Deliberately does NOT ask the model for "road" as a single class -- a box
detector answering "road" just returns one giant box covering most of the
frame, which throws away exactly the boundary information calibration needs.
Instead:
    - crosswalk-ish prompts  -> boxes converted straight to crosswalk polygons
    - border-ish prompts (curb / road edge / lane line / road marking)
      -> boxes are line-like slivers strung along the road's actual edges;
         we keep their centers and let the caller trace a polyline through
         them rather than pretending a bounding box is a boundary.

Usage:
    python scripts/dino_calibrate.py --frame dev_data/frame_clean_t8.jpg
    python scripts/dino_calibrate.py --frame dev_data/frame_clean_t8.jpg --write-crosswalks

Always writes an annotated preview image so results can be checked visually
before anything touches calibration/scene.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

CROSSWALK_PROMPTS = ["crosswalk", "zebra crossing", "pedestrian crossing"]
BORDER_PROMPTS = ["curb", "road edge", "road shoulder", "sidewalk edge", "lane line"]

CROSSWALK_COLOR = (0, 255, 255)   # yellow, matches ZONE_STYLES in scripts/visualize.py
BORDER_COLOR = (255, 0, 0)        # blue, matches the "carriageway" style there

SCENE_PATH = pathlib.Path(__file__).resolve().parent.parent / "calibration" / "scene.json"


def build_prompt(phrases: list[str]) -> str:
    # Grounding DINO expects lowercase phrases separated by ". ".
    return ". ".join(phrases).lower() + "."


def run_detection(image_bgr: np.ndarray, prompt: str, box_threshold: float, text_threshold: float):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    model.eval()

    from PIL import Image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    h, w = image_bgr.shape[:2]
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]
    return results


def to_xyxy_list(results) -> list[tuple[list[float], str, float]]:
    out = []
    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = results.get("text_labels") or results.get("labels")
    for box, score, label in zip(boxes, scores, labels):
        out.append((box, str(label), float(score)))
    return out


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def dedup_by_containment(dets: list[tuple[list[float], str, float]],
                          iom_threshold: float = 0.7) -> list[tuple[list[float], str, float]]:
    """Drop boxes that mostly overlap a higher-scoring box (same detected
    crosswalk found twice at slightly different extents), keeping the
    highest-scoring survivor per cluster rather than every raw hit.
    """
    ordered = sorted(dets, key=lambda d: d[2], reverse=True)
    kept: list[tuple[list[float], str, float]] = []
    for box, label, score in ordered:
        area = box_area(box)
        suppressed = False
        for kbox, _klabel, _kscore in kept:
            inter = intersection_area(box, kbox)
            iom = inter / max(1.0, min(area, box_area(kbox)))
            if iom > iom_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append((box, label, score))
    return kept


def box_to_polygon(box: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def normalize_polygon(polygon: list[list[float]], w: int, h: int) -> list[list[float]]:
    return [[round(x / w, 4), round(y / h, 4)] for x, y in polygon]


def draw_detections(image: np.ndarray, dets: list[tuple[list[float], str, float]], color, label_prefix: str) -> None:
    for box, label, score in dets:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label_prefix}:{label} {score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="path to a reference frame (jpg/png)")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--out", default="dev_data/dino_calibration_preview.jpg")
    ap.add_argument("--write-crosswalks", action="store_true",
                     help="overwrite the 'crosswalks' entries in calibration/scene.json "
                          "with the detected boxes (carriageway is left alone -- border "
                          "detections are boxes, not a traced boundary)")
    args = ap.parse_args()

    image = cv2.imread(args.frame)
    if image is None:
        raise SystemExit(f"cannot read {args.frame}")
    h, w = image.shape[:2]

    crosswalk_prompt = build_prompt(CROSSWALK_PROMPTS)
    border_prompt = build_prompt(BORDER_PROMPTS)

    print(f"running grounding-dino-tiny against {args.frame} ({w}x{h})")
    print(f"  crosswalk prompt: {crosswalk_prompt!r}")
    raw_crosswalk = to_xyxy_list(run_detection(image, crosswalk_prompt, args.box_threshold, args.text_threshold))
    # keep only boxes actually labeled "crosswalk" -- phrase-grouped queries
    # sometimes echo back a *different* input phrase (e.g. "pedestrian") tied
    # to an unrelated detection (a person), which is noise for this purpose.
    crosswalk_results = dedup_by_containment(
        [d for d in raw_crosswalk if "crosswalk" in d[1].lower()]
    )
    print(f"  -> {len(raw_crosswalk)} raw, {len(crosswalk_results)} after filtering to 'crosswalk' label + dedup")

    print(f"  border prompt: {border_prompt!r}")
    raw_border = to_xyxy_list(run_detection(image, border_prompt, args.box_threshold, args.text_threshold))
    # a box detector answering "curb"/"road edge" tends to do one of two
    # useless things: return the median/traffic-island (a real object, but
    # not the road's border) or a box spanning most of the frame -- exactly
    # the "road area" result this script exists to avoid. Drop anything that
    # swallows more than a third of the frame; what's left is reported as
    # low-confidence, not written anywhere automatically.
    frame_area = w * h
    border_results = dedup_by_containment(
        [d for d in raw_border if box_area(d[0]) < 0.35 * frame_area]
    )
    print(f"  -> {len(raw_border)} raw, {len(border_results)} after dropping whole-frame boxes + dedup")

    vis = image.copy()
    draw_detections(vis, crosswalk_results, CROSSWALK_COLOR, "crosswalk")
    draw_detections(vis, border_results, BORDER_COLOR, "border")
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"saved preview -> {out_path}")

    print("\ncrosswalk detections:")
    for box, label, score in crosswalk_results:
        print(f"  {label!r} score={score:.2f} box={[round(v) for v in box]}")
    print("border detections (unfiltered by category -- inspect before trusting):")
    for box, label, score in border_results:
        print(f"  {label!r} score={score:.2f} box={[round(v) for v in box]}")
    if not border_results:
        print("  (none survived the whole-frame filter)")
    print("\nNOTE: grounding-dino-tiny does not reliably localize curbs/road edges as "
          "line features -- in testing it either boxes the raised median/traffic "
          "island or collapses to a box spanning most of the frame. Border boxes "
          "above are experimental and are never auto-written to scene.json; treat "
          "the carriageway boundary as still needing manual tracing "
          "(scripts/calibrate_scene.py) guided by the crosswalk boxes as anchors.")

    if args.write_crosswalks:
        if not crosswalk_results:
            print("\nno crosswalk detections -- not touching calibration/scene.json")
        else:
            raw = json.loads(SCENE_PATH.read_text()) if SCENE_PATH.exists() else {
                "video_ref": "unspecified", "ref_frame_size": [w, h], "notes": "",
                "parking_zones": [], "crosswalks": [], "carriageway": [], "stop_lines": [],
            }
            raw["crosswalks"] = [
                {"name": f"dino_crosswalk_{i}", "polygon": normalize_polygon(box_to_polygon(box), w, h)}
                for i, (box, _label, _score) in enumerate(crosswalk_results)
            ]
            SCENE_PATH.write_text(json.dumps(raw, indent=2))
            print(f"\nwrote {len(crosswalk_results)} crosswalk polygon(s) -> {SCENE_PATH}")


if __name__ == "__main__":
    main()
