"""Causal accident-risk estimation from a stream of per-frame tracks.

Part B must be causal (RiskEstimator.step() only ever sees past frames), so
this keeps its own short rolling per-track position history and estimates
time-to-collision (TTC) between currently-visible track pairs under a
constant-velocity assumption -- no training data or accident model needed,
matching the task description's own suggested baseline signal. A sudden
deceleration check adds a second, independent signal for hard braking that
doesn't reach an actual collision course (a near_miss precursor).

Radii for the collision test come from each track's own bounding box, so the
threshold shrinks for distant, perspective-foreshortened vehicles the same
way their pixel footprint does, without a lane/depth calibration.

The *_PXPS speed constants below are tuned against calibration/scene.json's
ref_frame_size (1280 wide); update() multiplies them by the calibration's
scale (actual_frame_width / 1280) before use, same convention as
src/rules.py, since a 3840-wide native eval video moves ~3x as many
pixels/sec as the 960-1280px dev proxy for the same real-world speed.
"""
from __future__ import annotations

import math
from collections import deque

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
PERSON_CLASSES = {"person"}

HISTORY_SEC = 1.0          # window used to estimate a track's velocity
MIN_VELOCITY_DT = 0.15     # need at least this much time spread to trust it
STALE_SEC = 1.5            # drop a track's history if unseen this long

TTC_HORIZON_SEC = 3.0        # ignore projected collisions further out than this
COLLISION_MARGIN_PX = 3.0    # padding added to the two tracks' combined radius
# Minimum relative closing speed to treat a pair as "closing" at all -- used
# for pairs genuinely heading in different/opposing directions (crossing or
# head-on paths), where any real closing is inherently more dangerous.
#
# TTC_HORIZON_SEC is deliberately short (not the metric's own H=5s) because
# this is a signal-controlled intersection: vehicles arriving from different
# approaches routinely have projected paths that "cross" 6-8s out under a
# naive straight-line extrapolation and never actually meet, because right-
# of-way/signal timing resolves it well before then -- something this purely
# kinematic model has no way to see. Restricting to a few seconds out keeps
# projections short enough that a real crossing risk mostly hasn't yet been
# resolved by a turn, a stop, or a light change.
MIN_CLOSING_SPEED_PXPS = 45.0
# Below this, a track's own speed is dominated by ByteTrack box jitter, not
# real motion -- its "heading" is noise, not a direction. Below this floor a
# track is treated as effectively stationary (parked/idle/queued) rather
# than compared for heading, which matters a lot in dense traffic: most
# vehicles in a jam sit a few px/s from motionless.
RELIABLE_HEADING_MIN_SPEED_PXPS = 12.0
# Two vehicles heading the *same* way at different speeds (one catching up
# to another ahead of it in the same lane) is ordinary car-following, not a
# collision course -- ubiquitous in queued/congested traffic. Only count it
# once the closing rate reaches tailgating-into-a-rear-end territory.
SAME_DIRECTION_COS_THRESHOLD = 0.5   # within ~60 degrees of heading
SAME_DIRECTION_MIN_CLOSING_SPEED_PXPS = 110.0
# One side of the pair isn't reliably moving (see RELIABLE_HEADING_MIN_SPEED_
# PXPS): most likely a parked/queued vehicle, so require the moving side to
# be closing distinctly faster than normal creep-forward-in-a-queue speed
# before this reads as "not slowing down for stopped traffic ahead".
STATIONARY_SIDE_MIN_CLOSING_SPEED_PXPS = 120.0
# A pair must look like it's on a collision course for this many consecutive
# update() calls before it contributes to the output risk, so a single noisy
# velocity estimate (tracker jitter, a box that jumped) can't spike the score.
# A full-video check against ordinary (non-accident) footage found the
# original value of 2 still left risk >= the alarm threshold (0.5) on ~75%
# of frames -- raised substantially since real accidents are rare and this
# is meant to be quiet almost all the time.
MIN_PERSISTENCE_HITS = 4

DECEL_MIN_SPEED_PXPS = 20.0  # must have been genuinely moving, not jitter
DECEL_DROP_RATIO = 0.5       # lost at least half its speed in one update
# Below the 0.5 alarm threshold on its own -- a single sharp-looking
# deceleration sample is corroborating evidence, not proof by itself
# (tracker jitter on one bad box looks identical to real braking). Needs
# DECEL_MIN_PERSISTENCE_HITS consecutive confirmations to count at full
# weight; see update().
DECEL_RISK = 0.3
DECEL_RISK_CONFIRMED = 0.6
DECEL_MIN_PERSISTENCE_HITS = 2


def _radius(bbox: tuple[float, float, float, float], scale: float = 1.0) -> float:
    x1, y1, x2, y2 = bbox
    return max(4.0 * scale, ((x2 - x1) + (y2 - y1)) / 4.0)


def _time_to_collision(xa, ya, va, radius_a, xb, yb, vb, radius_b, scale: float = 1.0) -> float | None:
    """Time to closest approach of two constant-velocity points, or None if
    they're not on a collision course within TTC_HORIZON_SEC.

    Two guards keep ordinary dense/queued traffic from reading as imminent
    collisions, where vehicles sit within a bbox-radius of each other as a
    matter of course: a minimum relative closing speed (so near-identical
    lane-following speed / tracking jitter doesn't count as "closing"), and
    a requirement that the pair starts out actually separated (already-
    touching, steady-state proximity like a queue is not a *newly emerging*
    collision course).
    """
    dx, dy = xb - xa, yb - ya
    dvx, dvy = vb[0] - va[0], vb[1] - va[1]
    rel_speed2 = dvx * dvx + dvy * dvy

    speed_a, speed_b = math.hypot(*va), math.hypot(*vb)
    if speed_a < RELIABLE_HEADING_MIN_SPEED_PXPS * scale or speed_b < RELIABLE_HEADING_MIN_SPEED_PXPS * scale:
        # one side isn't reliably moving -- its heading is jitter, not
        # signal, so don't trust a same/opposite-direction classification
        # built from it. Most likely a parked/queued vehicle.
        min_closing_speed = STATIONARY_SIDE_MIN_CLOSING_SPEED_PXPS * scale
    else:
        cos_heading = (va[0] * vb[0] + va[1] * vb[1]) / (speed_a * speed_b)
        min_closing_speed = (SAME_DIRECTION_MIN_CLOSING_SPEED_PXPS * scale
                              if cos_heading > SAME_DIRECTION_COS_THRESHOLD
                              else MIN_CLOSING_SPEED_PXPS * scale)
    if rel_speed2 < min_closing_speed ** 2:
        return None  # not closing meaningfully faster than normal drift/following

    threshold = radius_a + radius_b + COLLISION_MARGIN_PX * scale
    if math.hypot(dx, dy) <= threshold:
        return None  # already at contact range -- steady proximity, not new risk
    t_closest = -(dx * dvx + dy * dvy) / rel_speed2
    if t_closest <= 0 or t_closest > TTC_HORIZON_SEC:
        return None  # already past closest approach, or too far out to matter
    close_x, close_y = dx + dvx * t_closest, dy + dvy * t_closest
    dist_at_closest = math.hypot(close_x, close_y)
    if dist_at_closest > threshold:
        return None  # paths don't actually come close enough
    return t_closest


class TTCRiskEstimator:
    """Feed it one sampled frame's tracks at a time; get back an instantaneous
    risk score in [0, 1]. Purely reactive to what it's been shown so far.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._history: dict[int, deque] = {}
        self._last_bbox: dict[int, tuple[float, float, float, float]] = {}
        self._last_cls: dict[int, str] = {}
        self._last_seen: dict[int, float] = {}
        self._streaks: dict[tuple[int, int], int] = {}  # pair key -> consecutive hit count
        self._decel_streaks: dict[int, int] = {}  # track id -> consecutive "still braking" count
        self.last_risk_pair: tuple[int, int] | None = None  # set by update(); see its end
        self.last_risk_track: int | None = None

    def _velocity(self, tid: int) -> tuple[float, float] | None:
        hist = self._history.get(tid)
        if not hist or len(hist) < 2:
            return None
        t0, x0, y0 = hist[0]
        t1, x1, y1 = hist[-1]
        dt = t1 - t0
        if dt < MIN_VELOCITY_DT:
            return None
        return (x1 - x0) / dt, (y1 - y0) / dt

    def update(self, t_sec: float, tracks, calibration=None) -> float:
        scale = calibration.scale if calibration is not None else 1.0
        for tid in list(self._history):
            if t_sec - self._last_seen.get(tid, t_sec) > STALE_SEC:
                del self._history[tid]
                self._last_bbox.pop(tid, None)
                self._last_cls.pop(tid, None)
                self._last_seen.pop(tid, None)

        decel_risk = 0.0
        decel_track = None
        for tr in tracks:
            if tr.cls not in VEHICLE_CLASSES and tr.cls not in PERSON_CLASSES:
                continue
            hist = self._history.setdefault(tr.id, deque())
            prev_v = self._velocity(tr.id)
            prev_speed = math.hypot(*prev_v) if prev_v is not None else None

            hist.append((t_sec, tr.center[0], tr.center[1]))
            while hist and t_sec - hist[0][0] > HISTORY_SEC:
                hist.popleft()
            self._last_bbox[tr.id] = tr.bbox
            self._last_cls[tr.id] = tr.cls
            self._last_seen[tr.id] = t_sec

            braking_now = False
            if prev_speed is not None and prev_speed > DECEL_MIN_SPEED_PXPS * scale and tr.cls in VEHICLE_CLASSES:
                new_v = self._velocity(tr.id)
                if new_v is not None and math.hypot(*new_v) < prev_speed * DECEL_DROP_RATIO:
                    braking_now = True
            if braking_now:
                streak = self._decel_streaks.get(tr.id, 0) + 1
                self._decel_streaks[tr.id] = streak
                this_risk = DECEL_RISK_CONFIRMED if streak >= DECEL_MIN_PERSISTENCE_HITS else DECEL_RISK
                if this_risk > decel_risk:
                    decel_risk = this_risk
                    decel_track = tr.id
            else:
                self._decel_streaks.pop(tr.id, None)

        ids = [tid for tid in self._history if self._velocity(tid) is not None]
        hit_pairs: set[tuple[int, int]] = set()
        min_ttc = None
        min_ttc_pair = None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                cls_a, cls_b = self._last_cls[a], self._last_cls[b]
                if cls_a not in VEHICLE_CLASSES and cls_b not in VEHICLE_CLASSES:
                    continue  # need at least one vehicle for this to be collision-relevant
                pa, pb = self._history[a][-1], self._history[b][-1]
                # a pedestrian off the carriageway (e.g. on a sidewalk, just
                # walking near roadside parked cars) isn't a collision risk --
                # skip the pair rather than let it inflate min_ttc.
                if calibration is not None:
                    if cls_a in PERSON_CLASSES and not calibration.in_carriageway(pa[1], pa[2]):
                        continue
                    if cls_b in PERSON_CLASSES and not calibration.in_carriageway(pb[1], pb[2]):
                        continue
                va, vb = self._velocity(a), self._velocity(b)
                ttc = _time_to_collision(pa[1], pa[2], va, _radius(self._last_bbox[a], scale),
                                          pb[1], pb[2], vb, _radius(self._last_bbox[b], scale),
                                          scale)
                if ttc is None:
                    continue
                key = (a, b)
                hit_pairs.add(key)
                streak = self._streaks.get(key, 0) + 1
                self._streaks[key] = streak
                if streak < MIN_PERSISTENCE_HITS:
                    continue  # one-off velocity noise -- not counted yet
                if min_ttc is None or ttc < min_ttc:
                    min_ttc = ttc
                    min_ttc_pair = key
        # a pair that didn't look collision-bound this round resets its streak
        for key in list(self._streaks):
            if key not in hit_pairs:
                del self._streaks[key]

        ttc_risk = 0.0 if min_ttc is None else max(0.0, min(1.0, 1.0 - min_ttc / TTC_HORIZON_SEC))
        # Exposed for detect_accident_near_miss (src/rules.py), which reuses
        # this estimator over a whole video (Part A has no causality
        # constraint) and needs to know *which* pair/track drove the score
        # to check for actual bbox contact, not just a risk number.
        if ttc_risk >= decel_risk:
            self.last_risk_pair = min_ttc_pair
            self.last_risk_track = None
        else:
            self.last_risk_pair = None
            self.last_risk_track = decel_track
        return max(ttc_risk, decel_risk)
