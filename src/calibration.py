"""Scene geometry for one fixed camera: parking pockets, stop line, crosswalks.

The task description is explicit that this is "a fixed road camera" — the
eval set is more clips from the same physical install, not different
cameras. So one hand-calibrated layout (calibration/scene.json), built by
eye against dev_data/frame_t45_grid.jpg, is reused for every video. Points
are stored normalized (0..1 of frame width/height) so the same file works
whether a given video is the 1280x720 dev proxy or the native 4K source.

If a future sample turns out to be a different camera, recalibrate against
that specific frame — nothing here is video-content-derived.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib

DEFAULT_CALIBRATION_PATH = pathlib.Path(__file__).resolve().parent.parent / "calibration" / "scene.json"

Point = tuple[float, float]


@dataclasses.dataclass
class Zone:
    name: str
    polygon: list[Point]  # pixel coords, already scaled to the target frame

    def contains(self, x: float, y: float) -> bool:
        return _point_in_polygon(x, y, self.polygon)


@dataclasses.dataclass
class StopLine:
    name: str
    p1: Point
    p2: Point
    # a point on the "approach" side, so we can tell which side of the line
    # a vehicle is on (i.e. has it crossed past the line or not).
    approach_point: Point

    def signed_side(self, x: float, y: float) -> float:
        """>0 on the approach side, <0 past the line, 0 on it."""
        ax, ay = self.p1
        bx, by = self.p2
        px, py = self.approach_point
        line_sign = _cross(ax, ay, bx, by, px, py)
        pt_sign = _cross(ax, ay, bx, by, x, y)
        if line_sign == 0:
            return 0.0
        return pt_sign if line_sign > 0 else -pt_sign

    def has_crossed(self, x: float, y: float) -> bool:
        return self.signed_side(x, y) < 0

    def distance_to_line(self, x: float, y: float) -> float:
        """Perpendicular distance from (x, y) to the (infinite) line through
        p1/p2 -- fine near the stop line itself, which is all callers use it
        for (e.g. "is this vehicle within N px of the line").
        """
        ax, ay = self.p1
        bx, by = self.p2
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return math.hypot(x - ax, y - ay)
        t = ((x - ax) * dx + (y - ay) * dy) / length_sq
        proj_x, proj_y = ax + t * dx, ay + t * dy
        return math.hypot(x - proj_x, y - proj_y)


def _cross(ax, ay, bx, by, px, py) -> float:
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _point_in_polygon(x: float, y: float, polygon: list[Point]) -> bool:
    """Standard ray-casting test. polygon: [(x,y), ...], not necessarily closed."""
    inside = False
    n = len(polygon)
    x1, y1 = polygon[-1]
    for i in range(n):
        x2, y2 = polygon[i]
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
        x1, y1 = x2, y2
    return inside


class SceneCalibration:
    """Loads calibration/scene.json (normalized coords) and scales it to a
    given frame size on demand, since dev proxies and eval videos may not
    share the source's native resolution.
    """

    def __init__(self, path: str | pathlib.Path = DEFAULT_CALIBRATION_PATH):
        raw = json.loads(pathlib.Path(path).read_text())
        self._ref_w, self._ref_h = raw["ref_frame_size"]
        self._raw = raw

    def _scale_point(self, pt: list[float], w: float, h: float) -> Point:
        return (pt[0] * w, pt[1] * h)

    def for_frame(self, width: int, height: int) -> "ScaledCalibration":
        parking_zones = [
            Zone(z["name"], [self._scale_point(p, width, height) for p in z["polygon"]])
            for z in self._raw.get("parking_zones", [])
        ]
        crosswalks = [
            Zone(z["name"], [self._scale_point(p, width, height) for p in z["polygon"]])
            for z in self._raw.get("crosswalks", [])
        ]
        carriageway = [
            Zone(z["name"], [self._scale_point(p, width, height) for p in z["polygon"]])
            for z in self._raw.get("carriageway", [])
        ]
        stop_lines = [
            StopLine(
                sl["name"],
                self._scale_point(sl["p1"], width, height),
                self._scale_point(sl["p2"], width, height),
                self._scale_point(sl["approach_point"], width, height),
            )
            for sl in self._raw.get("stop_lines", [])
        ]
        scale = width / self._ref_w if self._ref_w else 1.0
        return ScaledCalibration(parking_zones, crosswalks, carriageway, stop_lines, scale)


@dataclasses.dataclass
class ScaledCalibration:
    parking_zones: list[Zone]
    crosswalks: list[Zone]
    carriageway: list[Zone]
    stop_lines: list[StopLine]
    # actual_frame_width / calibration/scene.json's ref_frame_size width.
    # src/rules.py and src/risk.py tune their px/s and px-distance constants
    # against that reference width (1280) -- a 3840-wide eval video moves
    # ~3x as many pixels per second for the same real-world speed, so those
    # constants get multiplied by this before comparison. See the "scale"
    # docstring on detect_stopped_vehicle et al.
    scale: float = 1.0

    def in_parking_zone(self, x: float, y: float) -> bool:
        return any(z.contains(x, y) for z in self.parking_zones)

    def in_crosswalk(self, x: float, y: float) -> bool:
        return any(z.contains(x, y) for z in self.crosswalks)

    def in_carriageway(self, x: float, y: float) -> bool:
        if not self.carriageway:
            return True  # no calibration -> don't filter anything out
        return any(z.contains(x, y) for z in self.carriageway)
