"""Trajectory + scene-geometry rules.

stopped_vehicle and congestion build on per-track speed history from
src.detection.TrafficDetector.track_video(); both are refined with
src/calibration.py zones (calibration/scene.json) to tell a genuinely
stopped vehicle apart from a parked car or a normal red-light queue.
"""
from __future__ import annotations

import bisect
import math
from collections import defaultdict

from src.risk import TTCRiskEstimator
from src.segments import flags_to_segments

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
PERSON_CLASSES = {"person"}

# --- stopped_vehicle ---------------------------------------------------
STOPPED_SPEED_PXPS = 6.0       # below this, a vehicle counts as "not moving"
STOPPED_MIN_SEC = 10.0         # official definition: stationary >= 10s
STOPPED_MERGE_GAP_SEC = 2.0    # bridge brief occlusion / detection jitter
# A "stop" needs a preceding moving sample, or every legally-parked car in
# frame at t=0 gets flagged as having "stopped" for the entire clip. Without
# this we saw a parked curb car reported as stopped_vehicle for 120/120s.
STOPPED_PARK_MARGIN_SEC = 1.0

# A stop counts as "queued at signal" (excluded, per the official
# definition) rather than a real stopped_vehicle event if enough OTHER
# vehicles, ANYWHERE in the scene, were also stopped at overlapping times.
# Two earlier attempts failed: (1) a fixed pixel-distance cluster test --
# camera perspective makes near-lane cars tens of pixels apart while
# far-lane cars sit within a few pixels of each other, so no single radius
# covers both; (2) a "fraction of scene moving slowly at time t" threshold
# -- a queue builds up gradually (measured: scene-slow fraction ramps
# smoothly from 0.15 to 0.9 over ~25s as more cars join), so no single
# instant cleanly separates "queue forming" from "not a queue" and an
# early-arriving queue member sits below threshold until later cars join.
# Just counting simultaneously-stopped tracks, with no spatial or instant-
# in-time restriction, sidesteps both: diagnosably, ~15 tracks were stopped
# with heavily overlapping windows around t~40-110s -- an ordinary
# red-light queue, not a violation.
QUEUE_MIN_SIMULTANEOUS = 3   # >= this many other overlapping stops -> queue

# --- congestion ----------------------------------------------------------
# Restricted to the calibrated carriageway zone so a stray detection off the
# road surface (sidewalk, parked row) can't count toward "most vehicles are
# barely moving". Still scene-wide across both directions since we don't
# have per-lane-direction calibration -- see src/calibration.py notes.
CONGESTION_SPEED_PXPS = 10.0
CONGESTION_MIN_VEHICLES = 3
CONGESTION_MIN_FRACTION = 0.6
CONGESTION_MIN_SEC = 4.0
CONGESTION_MERGE_GAP_SEC = 2.0

# --- jaywalking ------------------------------------------------------------
# A pedestrian on the carriageway but outside any crosswalk zone, sustained
# long enough to rule out detection jitter as they step off a curb.
JAYWALK_MIN_SEC = 1.0
JAYWALK_MERGE_GAP_SEC = 1.5

# --- red_light / stop_line -------------------------------------------------
# Neither rule reads the traffic light's actual color: the light head in
# this footage is small, distant, and gets blocked by buses/trucks
# constantly, so a pixel-color classifier would be noisy. Instead both infer
# "the signal is red for this approach" from a directly observable proxy:
# vehicles queued stopped at the calibrated stop line -- exactly what a
# human glancing at the video would use without seeing the light itself.
STOP_LINE_QUEUE_BAND_PX = 70.0   # "at the line", approach side; ref-scale, x calibration.scale
RED_PHASE_MIN_SEC = 3.0          # a queue must persist this long to count as a real red phase
RED_PHASE_MERGE_GAP_SEC = 2.0    # bridge brief gaps where the band's single occupant blips
# A red_light violation is a vehicle crossing the line that was NEVER seen
# stopped in the queue band beforehand (i.e. it drove straight through
# rather than joining the queue), while >=1 *other* vehicle is stopped in
# the band at that moment. That second condition is what excludes the
# ordinary case of the front car legitimately pulling away on green -- it
# was itself part of the (now ex-)queue, so its own "ever stopped" flag is
# set and it's excluded regardless of what trailing cars are still doing.
RED_LIGHT_TRAILING_SEC = 4.0     # how long after crossing to keep reporting the event, absent better track data

# stop_line: a vehicle stopped just past the line (encroaching, not fully
# into the intersection) while a red phase (as inferred above) is active.
STOP_LINE_PAST_BAND_PX = 90.0    # "just past the line" vs. having actually entered the intersection
STOP_LINE_MIN_SEC = 3.0

# --- accident / near_miss ---------------------------------------------------
# No learned contact/damage model -- the task description itself flags these
# two as where one would help most. Both reuse src.risk.TTCRiskEstimator's
# signal (TTC + hard braking) replayed over the whole video: Part A has no
# causality constraint, and the task FAQ explicitly allows Part A to reuse
# Part B's risk curve. Told apart by a simple, explainable proxy for "did
# they actually touch": whether the pair driving the risk score has
# overlapping bounding boxes at or shortly after the danger window. A single
# track braking hard with no identified collision partner (e.g. avoiding
# something off-screen or just aggressive driving) has nothing to check
# contact against, so it always reads as near_miss, never accident.
# Stricter than Part B's own alarm threshold (theta=0.5): Part B is scored
# on catching real accidents early and reacting fast is worth some nuisance
# alarms, but Part A only benefits from a segment that's actually there --
# a first full-video check against ordinary (non-accident) proxy footage at
# theta=0.5/min_sec=0.3 fired a "near_miss" roughly every 13s, which would
# wreck precision on a real test set. Traded recall for precision here.
DANGER_ALARM_THRESHOLD = 0.65
DANGER_MIN_SEC = 1.0
DANGER_MERGE_GAP_SEC = 1.0
ACCIDENT_CONTACT_IOU = 0.10       # bbox overlap this high between the pair counts as contact
ACCIDENT_CONTACT_WINDOW_SEC = 2.0  # look this far past a danger window's end for contact

# --- failure_to_yield --------------------------------------------------------
# "Driving through a crossing while a pedestrian is on it": a vehicle inside
# a calibrated crosswalk polygon, moving (not stopped -- a car stopped for a
# pedestrian is the correct behaviour, not a violation), while a pedestrian
# is within FAILURE_TO_YIELD_PROXIMITY_PX in that same crosswalk polygon at
# the same sampled instant. Both sides come from the same `samples`
# timestamps, so "same instant" is exact, not approximate. See
# detect_failure_to_yield's own docstring for why zone co-membership alone
# isn't enough and a proximity check was added.
FAILURE_TO_YIELD_MIN_SPEED_PXPS = 6.0   # same "not stopped" bar as STOPPED_SPEED_PXPS
FAILURE_TO_YIELD_PROXIMITY_PX = 180.0   # ref-scale; roughly one crosswalk-width
FAILURE_TO_YIELD_MIN_SEC = 0.5
FAILURE_TO_YIELD_MERGE_GAP_SEC = 1.0

# --- illegal_u_turn ------------------------------------------------------------
# No calibration of where U-turns are actually prohibited exists, so
# (matching the red_light pattern above) this flags the observable maneuver
# itself -- a heading reversal performed while continuously in the
# carriageway -- as a proxy for the violation, the same way red_light infers
# "red" from queue behaviour rather than reading the light. Heading is
# measured over a multi-second window on each side of a point (not
# frame-to-frame, which is too noisy mid-turn when a vehicle briefly slows
# close to STOPPED_SPEED_PXPS) and both windows must show real displacement,
# so a vehicle jittering near a standstill can't trip it.
#
# wrong_way (driving against the dominant direction of a carriageway zone)
# was also implemented and tested this way, then dropped: the circular
# concentration of vehicle headings per zone measured 0.11-0.24 on the dev
# clip (1.0 = one consistent direction), meaning these zones genuinely mix
# multiple legitimate directions (both sides of the two-way boulevard plus
# a perpendicular side street) rather than having one to compare against,
# and it badly over-fired in testing as a result. Not in CLASSES.
U_TURN_WINDOW_SEC = 3.0
U_TURN_ANGLE_DEG = 130.0
U_TURN_MIN_DISPLACEMENT_PX = 40.0   # ref-scale; each side of the comparison must move at least this far
U_TURN_MIN_SEC = 1.0
U_TURN_MERGE_GAP_SEC = 2.0


def _angle_diff(a_deg: float, b_deg: float) -> float:
    """Smallest angle (0-180) between two headings in degrees."""
    d = abs(a_deg - b_deg) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _track_histories(samples, classes=VEHICLE_CLASSES):
    """samples: list of (t_sec, list[Track]) -> dict[id] -> [(t, cx, cy, cls)]"""
    hist = defaultdict(list)
    for t, tracks in samples:
        for tr in tracks:
            if tr.cls in classes:
                hist[tr.id].append((t, tr.center[0], tr.center[1], tr.cls))
    return hist


def _speeds(track_samples):
    """track_samples: sorted [(t,cx,cy,cls), ...] for one track.
    -> [(t, speed_pxps), ...] aligned to the *later* sample of each pair.
    """
    out = []
    for (t0, x0, y0, _), (t1, x1, y1, _) in zip(track_samples, track_samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        dist = math.hypot(x1 - x0, y1 - y0)
        out.append((t1, dist / dt))
    return out


def _scene_slow_fraction_series(hist, calibration, speed_threshold, min_vehicles):
    """-> (sorted_times, {t: fraction of tracked vehicles moving < speed_threshold}).

    Restricted to the calibrated carriageway when available, so a stray
    detection off the road surface can't count toward "the scene is slow".
    Shared by congestion (is this fraction high for a while) and
    stopped_vehicle (was the scene already slow while this vehicle stopped,
    i.e. is it a signal queue rather than an isolated violation).
    """
    speed_by_t = defaultdict(list)
    for pts in hist.values():
        pts.sort(key=lambda p: p[0])
        pos_by_t = {t: (x, y) for t, x, y, _ in pts}
        for t, spd in _speeds(pts):
            if calibration is not None:
                x, y = pos_by_t[t]
                if not calibration.in_carriageway(x, y):
                    continue
            speed_by_t[t].append(spd)

    times = sorted(speed_by_t)
    fractions = {}
    for t in times:
        speeds = speed_by_t[t]
        if len(speeds) < min_vehicles:
            fractions[t] = 0.0
            continue
        slow = sum(1 for s in speeds if s < speed_threshold)
        fractions[t] = slow / len(speeds)
    return times, fractions


def merge_intervals(intervals: list[list[float]]) -> list[list[float]]:
    """Union of possibly-overlapping [start, end] intervals.

    Needed because two different vehicles being stopped at the same time
    would otherwise produce two overlapping same-class segments, which the
    harness collapses to one anyway (see FAQ: simultaneous same-class events
    are reported as a single segment covering both).
    """
    if not intervals:
        return []
    ivs = sorted(intervals)
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def detect_stopped_vehicle(samples, calibration=None) -> list[list]:
    """calibration: src.calibration.ScaledCalibration, or None to skip
    scene-aware filtering (raw trajectory-only behavior).

    The *_PXPS constants above are tuned against calibration/scene.json's
    ref_frame_size (1280 wide). A 3840-wide native eval video moves ~3x as
    many pixels/sec for the same real-world speed, so every constant gets
    multiplied by calibration.scale (actual_width / ref_width) before use --
    without calibration (scale defaults to 1.0) they're used as-is.
    """
    scale = calibration.scale if calibration is not None else 1.0
    hist = _track_histories(samples)

    # Pass 1: per-track candidate stop segments, minus parked-curb vehicles.
    candidates = []  # [[s, e], ...]
    for tid, pts in hist.items():
        pts.sort(key=lambda p: p[0])
        if len(pts) < 2:
            continue
        speeds = _speeds(pts)
        times = [t for t, _ in speeds]
        flags = [spd < STOPPED_SPEED_PXPS * scale for _, spd in speeds]
        first_seen_t = pts[0][0]
        segs = flags_to_segments(times, flags, min_duration=STOPPED_MIN_SEC,
                                  merge_gap=STOPPED_MERGE_GAP_SEC)
        segs = [s for s in segs if s[0] > first_seen_t + STOPPED_PARK_MARGIN_SEC]
        for s, e in segs:
            xs = [x for t, x, y, _ in pts if s <= t <= e]
            ys = [y for t, x, y, _ in pts if s <= t <= e]
            if not xs:
                continue
            avg_xy = (sum(xs) / len(xs), sum(ys) / len(ys))
            if calibration is not None and calibration.in_parking_zone(*avg_xy):
                continue
            candidates.append([s, e])

    # Pass 2: drop signal-queue stops -- a stop with enough OTHER vehicles
    # also stopped somewhere in the scene at overlapping times, regardless
    # of where. No spatial or single-instant test involved (see comment on
    # QUEUE_MIN_SIMULTANEOUS for why those failed).
    kept = []
    for i, (s, e) in enumerate(candidates):
        simultaneous = sum(
            1 for j, (s2, e2) in enumerate(candidates)
            if j != i and s <= e2 and s2 <= e
        )
        if simultaneous >= QUEUE_MIN_SIMULTANEOUS:
            continue
        kept.append([s, e])

    merged = merge_intervals(kept)
    return [[round(s, 2), round(e, 2), "stopped_vehicle"] for s, e in merged]


def detect_congestion(samples, calibration=None) -> list[list]:
    scale = calibration.scale if calibration is not None else 1.0
    hist = _track_histories(samples)
    times, fractions = _scene_slow_fraction_series(
        hist, calibration, CONGESTION_SPEED_PXPS * scale, CONGESTION_MIN_VEHICLES)
    flags = [fractions[t] >= CONGESTION_MIN_FRACTION for t in times]
    segs = flags_to_segments(times, flags, min_duration=CONGESTION_MIN_SEC,
                              merge_gap=CONGESTION_MERGE_GAP_SEC)
    return [[round(s, 2), round(e, 2), "congestion"] for s, e in segs]


def detect_jaywalking(samples, calibration=None) -> list[list]:
    """Requires calibration -- without carriageway/crosswalk zones there's no
    way to tell "on the road, not at a crossing" from "on the sidewalk".
    """
    if calibration is None:
        return []

    hist = _track_histories(samples, classes=PERSON_CLASSES)
    per_track_segments = []
    for pts in hist.values():
        pts.sort(key=lambda p: p[0])
        times = [t for t, _, _, _ in pts]
        flags = [calibration.in_carriageway(x, y) and not calibration.in_crosswalk(x, y)
                 for _, x, y, _ in pts]
        per_track_segments += flags_to_segments(times, flags, min_duration=JAYWALK_MIN_SEC,
                                                  merge_gap=JAYWALK_MERGE_GAP_SEC)

    merged = merge_intervals(per_track_segments)
    return [[round(s, 2), round(e, 2), "jaywalking"] for s, e in merged]


def _stop_line_queue_series(hist, stop_line, scale):
    """-> (sorted times, {t: bool}) -- True if >=1 vehicle is stopped within
    STOP_LINE_QUEUE_BAND_PX of the line, on its approach side, at time t.
    This is the whole "is it red" signal both rules below build on.
    """
    speed_by_t = defaultdict(list)
    for pts in hist.values():
        pts.sort(key=lambda p: p[0])
        pos_by_t = {t: (x, y) for t, x, y, _ in pts}
        for t, spd in _speeds(pts):
            x, y = pos_by_t[t]
            if stop_line.has_crossed(x, y):
                continue  # only the approach side forms a queue for this line
            if stop_line.distance_to_line(x, y) > STOP_LINE_QUEUE_BAND_PX * scale:
                continue
            speed_by_t[t].append(spd)
    times = sorted(speed_by_t)
    queue = {t: any(s < STOPPED_SPEED_PXPS * scale for s in speed_by_t[t]) for t in times}
    return times, queue


def detect_red_light(samples, calibration=None) -> list[list]:
    """Requires calibration with at least one stop line -- see the module
    docstring above for why this infers "red" from queue behaviour instead
    of the light's actual color.
    """
    if calibration is None or not calibration.stop_lines:
        return []
    scale = calibration.scale

    hist = _track_histories(samples)
    events: list[list] = []
    for stop_line in calibration.stop_lines:
        q_times, queue = _stop_line_queue_series(hist, stop_line, scale)
        red_phases = flags_to_segments(q_times, [queue[t] for t in q_times],
                                        min_duration=RED_PHASE_MIN_SEC,
                                        merge_gap=RED_PHASE_MERGE_GAP_SEC)
        if not red_phases:
            continue

        for tid, pts in hist.items():
            pts.sort(key=lambda p: p[0])
            if len(pts) < 2:
                continue
            ever_stopped_in_band = False
            speeds = dict(_speeds(pts))  # t -> speed, aligned to the later sample
            for (t0, x0, y0, _), (t1, x1, y1, _) in zip(pts, pts[1:]):
                spd = speeds.get(t1)
                crossed_now = (not stop_line.has_crossed(x0, y0)) and stop_line.has_crossed(x1, y1)
                in_band_stopped = (
                    spd is not None and spd < STOPPED_SPEED_PXPS * scale
                    and not stop_line.has_crossed(x0, y0)
                    and stop_line.distance_to_line(x0, y0) <= STOP_LINE_QUEUE_BAND_PX * scale
                )
                if in_band_stopped:
                    ever_stopped_in_band = True
                if not crossed_now or ever_stopped_in_band:
                    continue  # only flag a straight-through crossing, never a queue member
                if not any(s <= t1 < e for s, e in red_phases):
                    continue  # crossed while no queue was active for this line -- not on red
                # queue[t1] can't be true *because of* this vehicle: it's already
                # past the line at t1, so _stop_line_queue_series already excludes
                # it -- a True here means some other vehicle is still stopped.
                if not queue.get(t1, False):
                    continue  # nobody else stopped right then either -- ordinary green flow
                end = min(t1 + RED_LIGHT_TRAILING_SEC, pts[-1][0])
                events.append([t1, max(end, t1 + 0.1)])

    merged = merge_intervals(events)
    return [[round(s, 2), round(e, 2), "red_light"] for s, e in merged]


def detect_stop_line_violation(samples, calibration=None) -> list[list]:
    """A vehicle stopped just past the line (not fully into the
    intersection) for the duration of an inferred red phase.
    """
    if calibration is None or not calibration.stop_lines:
        return []
    scale = calibration.scale

    hist = _track_histories(samples)
    events: list[list] = []
    for stop_line in calibration.stop_lines:
        q_times, queue = _stop_line_queue_series(hist, stop_line, scale)
        red_phases = flags_to_segments(q_times, [queue[t] for t in q_times],
                                        min_duration=RED_PHASE_MIN_SEC,
                                        merge_gap=RED_PHASE_MERGE_GAP_SEC)
        if not red_phases:
            continue

        for pts in hist.values():
            pts.sort(key=lambda p: p[0])
            if len(pts) < 2:
                continue
            times = [t for t, _, _, _ in pts]
            speeds = _speeds(pts)
            speed_by_t = dict(speeds)
            pos_by_t = {t: (x, y) for t, x, y, _ in pts}
            flags = []
            for t in times:
                x, y = pos_by_t[t]
                spd = speed_by_t.get(t)
                past_line = stop_line.has_crossed(x, y)
                near_line = stop_line.distance_to_line(x, y) <= STOP_LINE_PAST_BAND_PX * scale
                stopped = spd is not None and spd < STOPPED_SPEED_PXPS * scale
                flags.append(past_line and near_line and stopped)
            for s, e in flags_to_segments(times, flags, min_duration=STOP_LINE_MIN_SEC,
                                           merge_gap=STOPPED_MERGE_GAP_SEC):
                phase = next((p for p in red_phases if p[0] <= s <= p[1]), None)
                if phase is None:
                    continue  # stopped past the line, but no red phase covers the start
                events.append([s, min(e, phase[1])])

    merged = merge_intervals(events)
    return [[round(s, 2), round(e, 2), "stop_line"] for s, e in merged]


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_accident_near_miss(samples, calibration=None) -> list[list]:
    """See the module-level comment above ACCIDENT_CONTACT_IOU for the
    approach; this is a coarse stand-in for a real contact/damage model and
    is expected to be the weakest of the implemented classes.
    """
    estimator = TTCRiskEstimator()
    estimator.reset()
    times: list[float] = []
    scores: list[float] = []
    culprits: list[tuple[int, int] | int | None] = []
    bbox_by_t_id: dict[tuple[float, int], tuple[float, float, float, float]] = {}

    for t, tracks in samples:
        for tr in tracks:
            bbox_by_t_id[(t, tr.id)] = tr.bbox
        score = estimator.update(t, tracks, calibration=calibration)
        times.append(t)
        scores.append(score)
        culprits.append(estimator.last_risk_pair or estimator.last_risk_track)

    flags = [s >= DANGER_ALARM_THRESHOLD for s in scores]
    danger_windows = flags_to_segments(times, flags, min_duration=DANGER_MIN_SEC,
                                        merge_gap=DANGER_MERGE_GAP_SEC)

    accident_events, near_miss_events = [], []
    for s, e in danger_windows:
        idxs = [i for i, t in enumerate(times) if s <= t <= e]
        if not idxs:
            continue
        peak_i = max(idxs, key=lambda i: scores[i])
        culprit = culprits[peak_i]

        contact = False
        if isinstance(culprit, tuple):
            a, b = culprit
            for t in times:
                if not (s <= t <= e + ACCIDENT_CONTACT_WINDOW_SEC):
                    continue
                ba, bb = bbox_by_t_id.get((t, a)), bbox_by_t_id.get((t, b))
                if ba is not None and bb is not None and _bbox_iou(ba, bb) >= ACCIDENT_CONTACT_IOU:
                    contact = True
                    break
        (accident_events if contact else near_miss_events).append([s, e])

    out = [[round(s, 2), round(e, 2), "accident"] for s, e in merge_intervals(accident_events)]
    out += [[round(s, 2), round(e, 2), "near_miss"] for s, e in merge_intervals(near_miss_events)]
    return out


def detect_failure_to_yield(samples, calibration=None) -> list[list]:
    """Requires calibration with at least one crosswalk zone.

    "In the same crosswalk zone" alone isn't enough: this scene's crosswalk
    polygons run the full width of the road, so a vehicle and a pedestrian
    can both be technically inside one at the same instant while many
    lanes apart, and a busy crossing has *someone* on it most of the time
    (measured: a pedestrian is in some crosswalk zone in 77% of sampled
    frames of the dev clip) -- zone co-membership alone flagged nearly
    every vehicle that ever touched a crosswalk. Also require the nearest
    same-zone pedestrian to be within FAILURE_TO_YIELD_PROXIMITY_PX, a
    proxy for "actually in this vehicle's path" rather than "somewhere on
    the same marked crossing".
    """
    if calibration is None or not calibration.crosswalks:
        return []
    scale = calibration.scale

    last_pos: dict[int, tuple[float, float, float]] = {}
    flags_by_vehicle: dict[int, list[tuple[float, bool]]] = defaultdict(list)

    for t, tracks in samples:
        peds_by_zone: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for tr in tracks:
            if tr.cls not in PERSON_CLASSES:
                continue
            x, y = tr.center
            for z in calibration.crosswalks:
                if z.contains(x, y):
                    peds_by_zone[z.name].append((x, y))
        for tr in tracks:
            if tr.cls not in VEHICLE_CLASSES:
                continue
            x, y = tr.center
            prev = last_pos.get(tr.id)
            last_pos[tr.id] = (t, x, y)
            moving = True
            if prev is not None:
                pt, px, py = prev
                dt = t - pt
                if dt > 0:
                    spd = math.hypot(x - px, y - py) / dt
                    moving = spd >= FAILURE_TO_YIELD_MIN_SPEED_PXPS * scale
            near_pedestrian = False
            for z in calibration.crosswalks:
                if not z.contains(x, y):
                    continue
                near_pedestrian = any(
                    math.hypot(x - px_, y - py_) <= FAILURE_TO_YIELD_PROXIMITY_PX * scale
                    for px_, py_ in peds_by_zone.get(z.name, [])
                )
                if near_pedestrian:
                    break
            flags_by_vehicle[tr.id].append((t, moving and near_pedestrian))

    segments = []
    for series in flags_by_vehicle.values():
        series.sort(key=lambda p: p[0])
        times = [t for t, _ in series]
        flags = [f for _, f in series]
        segments += flags_to_segments(times, flags, min_duration=FAILURE_TO_YIELD_MIN_SEC,
                                       merge_gap=FAILURE_TO_YIELD_MERGE_GAP_SEC)

    merged = merge_intervals(segments)
    return [[round(s, 2), round(e, 2), "failure_to_yield"] for s, e in merged]


def detect_illegal_u_turn(samples, calibration=None) -> list[list]:
    """See the module docstring above U_TURN_* for why this flags the raw
    heading-reversal maneuver rather than an actually-prohibited one.
    """
    scale = calibration.scale if calibration is not None else 1.0
    hist = _track_histories(samples)

    events = []
    for pts in hist.values():
        pts.sort(key=lambda p: p[0])
        if len(pts) < 3:
            continue
        times = [p[0] for p in pts]
        flags = []
        for i, (t, x, y, _) in enumerate(pts):
            j = bisect.bisect_right(times, t - U_TURN_WINDOW_SEC) - 1
            k = bisect.bisect_right(times, t - 2 * U_TURN_WINDOW_SEC) - 1
            if j < 0 or k < 0:
                flags.append(False)
                continue
            _, jx, jy, _ = pts[j]
            _, kx, ky, _ = pts[k]
            recent_dx, recent_dy = x - jx, y - jy
            past_dx, past_dy = jx - kx, jy - ky
            recent_disp = math.hypot(recent_dx, recent_dy)
            past_disp = math.hypot(past_dx, past_dy)
            if (recent_disp < U_TURN_MIN_DISPLACEMENT_PX * scale
                    or past_disp < U_TURN_MIN_DISPLACEMENT_PX * scale):
                flags.append(False)
                continue
            heading_recent = math.degrees(math.atan2(recent_dy, recent_dx))
            heading_past = math.degrees(math.atan2(past_dy, past_dx))
            reversed_ = _angle_diff(heading_recent, heading_past) >= U_TURN_ANGLE_DEG
            in_carriageway = calibration is None or calibration.in_carriageway(x, y)
            flags.append(reversed_ and in_carriageway)
        events += flags_to_segments(times, flags, min_duration=U_TURN_MIN_SEC,
                                     merge_gap=U_TURN_MERGE_GAP_SEC)

    merged = merge_intervals(events)
    return [[round(s, 2), round(e, 2), "illegal_u_turn"] for s, e in merged]
