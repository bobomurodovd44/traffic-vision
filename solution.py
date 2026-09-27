"""
solution.py — the ONLY file a team has to implement.

The organizers' harness (run_submission.py) imports this module and calls:

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B (optional)

Keep the names and signatures exactly as they are. Everything else — models,
tracking, rules, helper modules under src/ — is up to you.

Labels must come from CLASSES. You may REMOVE classes you never predict;
do not add new ids.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch

from src.calibration import SceneCalibration
from src.detection import CALIBRATION_WINDOW, TrafficDetector, adaptive_stride, probe_video_meta
from src.risk import TTCRiskEstimator
from src.rules import (
    detect_accident_near_miss,
    detect_congestion,
    detect_failure_to_yield,
    detect_illegal_u_turn,
    detect_jaywalking,
    detect_red_light,
    detect_stop_line_violation,
    detect_stopped_vehicle,
)

# Official class ids (14). See the task description for definitions and
# start/end conventions. Remove entries you never predict; never add.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
]
# Removed per the module docstring's "may REMOVE classes you never predict"
# rule: illegal_turn and solid_line_crossing would need lane-level marking
# and turn-restriction calibration this project never built; road_obstacle
# and fire_smoke would need object classes the YOLO model (COCO-pretrained,
# see src/detection.py) was never trained to see -- emitting guesses for
# either would be noise, not signal. wrong_way was implemented and tested
# but dropped: it needs a "correct direction" to compare against, and this
# scene's carriageway zones measurably mix multiple legitimate directions
# (both sides of a two-way boulevard plus a perpendicular side street --
# circular concentration of vehicle headings per zone measured at 0.11-0.24
# on the dev clip, where 1.0 would mean one consistent direction), so it
# had no reliable signal to work from and badly over-fired in testing.

# Anticipation horizon used by the metric (seconds). step() should return
# P(an `accident` starts within the next RISK_HORIZON_SEC seconds).
RISK_HORIZON_SEC = 5.0

# run_submission.py's official time budget is 3x a video's duration for Part
# A + Part B *together*, but it only checks that budget after detect_events
# returns -- an unbounded Part A on a slower-than-expected machine would
# burn the whole thing and get the video scored empty (events AND risk).
# detect_events polices its own share of the budget instead of trusting the
# harness to cut it off. Mirrors run_submission.TIME_FACTOR_DEFAULT; can't
# import that constant directly since detect_events must keep working when
# run standalone (examples/, tests) without run_submission.py involved.
TIME_FACTOR = 3.0
# Leaves the rest of the combined budget for Part B. Part B's own deadline
# check (run_submission.run_risk) only fires every 100 raw frames, so it can
# overshoot its share by however long those ~100 frames take to process --
# on a slow-enough machine that's several seconds. Biasing the split below
# 0.5 gives Part B's checker more slack to catch an overshoot before the
# harness's outer deadline does, which zeroes the whole video either way.
PART_A_BUDGET_FRACTION = 0.5

# RiskEstimator.step() is called for every frame, but running detection that
# often is wasteful -- throttle actual YOLO+track work to ~this many fps and
# hold the last score in between (explicitly allowed by the task spec). Like
# detect_events, this is a ceiling, not a fixed rate: RiskEstimator adapts
# its own stride from measured per-frame cost against the SAME overall
# deadline detect_events computed (see _overall_deadline below), so Part B
# doesn't blindly assume this fps fits in whatever budget Part A left it.
RISK_TARGET_FPS = 6.0
# See track_video's min_fps docstring (src/detection.py) for why this isn't
# lower: below ~1fps ByteTrack's frame-to-frame association assumption
# breaks down and produces unreliable track identities, which matters more
# than an occasional slow/long video overshooting the outer budget and
# scoring empty.
MIN_RISK_FPS = 1.0
# Fast-rise / slow-decay envelope on top of the instantaneous per-sample risk:
# react immediately to a newly-dangerous situation, but don't let a single
# noisy low sample between two high ones collapse the score back to zero.
RISK_DECAY_PER_SAMPLE = 0.85

# Neither YOLO11s inference nor ByteTrack's matching has any randomness on
# their actual code path here given fixed weights/config -- but this seeds
# the standard RNGs anyway as a guard against any framework-internal
# fallback that does sample (e.g. an Ultralytics codepath probed even at
# eval time), rather than relying on that staying true across library
# versions. Deliberately does NOT set torch.backends.cudnn.deterministic =
# True: that can meaningfully slow GPU convolution, risking the time budget
# detect_events already has to self-police (see TIME_FACTOR below), for a
# guarantee this pipeline doesn't otherwise need.
SEED = 0


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_seed_everything()

# One YOLO+ByteTrack model and one calibration file, reused across every
# video in a run_submission.py process instead of reloading per call.
_detector: TrafficDetector | None = None
_calibration: SceneCalibration | None = None

# The combined Part A + Part B deadline for whatever video detect_events was
# last called on (time.perf_counter() timestamp), mirroring run_submission.
# run_risk's own math exactly since both start from the same t0. Lets
# RiskEstimator.reset() know how much of the shared budget Part A already
# spent instead of assuming it owns the full 3x-duration allowance itself.
_overall_deadline: float | None = None


def _get_detector() -> TrafficDetector:
    global _detector
    if _detector is None:
        _detector = TrafficDetector()
    return _detector


def _get_calibration() -> SceneCalibration:
    global _calibration
    if _calibration is None:
        _calibration = SceneCalibration()
    return _calibration


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection.

    Pipeline: YOLO detection + ByteTrack tracking (src/detection.py) sampled
    at ~8 fps, then rules (src/rules.py) turn track histories into segments,
    refined with the fixed camera's scene geometry (src/calibration.py,
    calibration/scene.json) to tell a genuinely stopped vehicle apart from a
    parked car or a normal red-light queue. All 9 classes still in CLASSES
    are implemented; red_light/stop_line infer "the signal is red" from
    queued-vehicle behavior at the calibrated stop line rather than reading
    the light's actual color, and illegal_u_turn similarly flags the raw
    heading-reversal maneuver rather than an actually-prohibited one, since
    this project has no per-lane/turn-restriction calibration — see
    src/rules.py's module and per-rule docstrings for why.
    """
    global _overall_deadline
    _overall_deadline = None
    detector = _get_detector()
    meta = probe_video_meta(video_path)
    run_start = time.perf_counter()
    total_budget = TIME_FACTOR * meta["duration"]
    _overall_deadline = run_start + total_budget
    deadline = run_start + PART_A_BUDGET_FRACTION * total_budget
    samples = list(detector.track_video(video_path, target_fps=8.0, deadline=deadline))
    calibration = _get_calibration().for_frame(meta["width"], meta["height"])

    events: list[list] = []
    events += detect_stopped_vehicle(samples, calibration=calibration)
    events += detect_congestion(samples, calibration=calibration)
    events += detect_jaywalking(samples, calibration=calibration)
    events += detect_red_light(samples, calibration=calibration)
    events += detect_stop_line_violation(samples, calibration=calibration)
    events += detect_accident_near_miss(samples, calibration=calibration)
    events += detect_failure_to_yield(samples, calibration=calibration)
    events += detect_illegal_u_turn(samples, calibration=calibration)
    return events


class RiskEstimator:
    """Part B — causal accident anticipation (optional, bonus).

    The harness calls ``reset(meta)`` once per video and then ``step`` for
    EVERY frame, in order. ``step`` must use only the frames it has seen so
    far: do not open the video file inside this class, and do not reuse
    Part A results that were computed with access to future frames.

    Signal: time-to-collision between tracked road users under a
    constant-velocity assumption (src.risk.TTCRiskEstimator), plus a sudden-
    deceleration check for hard braking that doesn't reach a collision
    course. Reuses the same YOLO+ByteTrack model as Part A (solution._get_
    detector()) rather than loading a second copy -- safe because
    run_submission.py finishes Part A for a video before Part B starts on
    it, and reset() below drops any leftover tracker state either way.
    """

    def __init__(self) -> None:
        self._risk = TTCRiskEstimator()
        self._calibration = None
        self._stride = 1
        self._min_stride = 1
        self._max_stride = 1
        self._frame_idx = 0
        self.last_score = 0.0

    def reset(self, meta: dict) -> None:
        """Called once before the first frame of each video.

        meta = {"video_id": str, "fps": float, "width": int, "height": int,
                "n_frames": int}
        """
        self.meta = meta
        fps = meta.get("fps") or 25.0
        width, height = meta.get("width"), meta.get("height")
        n_frames = meta.get("n_frames") or 0
        self._fps = fps
        self._duration = n_frames / fps if fps else 0.0
        self._calibration = _get_calibration().for_frame(width, height) if width and height else None
        self._min_stride = max(1, round(fps / RISK_TARGET_FPS))
        self._max_stride = max(self._min_stride, round(fps / MIN_RISK_FPS))
        self._stride = self._min_stride
        # _overall_deadline is set by detect_events for this same video; if
        # Part B is ever driven standalone (no prior detect_events call),
        # fall back to assuming it owns the whole budget itself.
        self._deadline = _overall_deadline if _overall_deadline is not None else (
            time.perf_counter() + TIME_FACTOR * self._duration)
        self._window_start = time.perf_counter()
        self._window_samples = 0
        self._frame_idx = 0
        self.last_score = 0.0
        self._risk.reset()
        _get_detector().reset()

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within the next RISK_HORIZON_SEC s).

        Args:
            frame: BGR uint8 array of shape (H, W, 3) — OpenCV convention.
            t_sec: timestamp of this frame in seconds.

        Returns:
            A float in [0, 1]. Skipping frames internally and returning the
            previous score is fine; the harness still expects a value for
            every call.
        """
        run_now = self._frame_idx % self._stride == 0
        self._frame_idx += 1
        if not run_now:
            return self.last_score

        tracks = _get_detector().track_frame(frame)
        instant = self._risk.update(t_sec, tracks, calibration=self._calibration)
        self.last_score = max(instant, self.last_score * RISK_DECAY_PER_SAMPLE)

        self._window_samples += 1
        if self._window_samples >= CALIBRATION_WINDOW:
            now = time.perf_counter()
            per_sample = (now - self._window_start) / self._window_samples
            self._stride = adaptive_stride(self._fps, per_sample, self._duration - t_sec,
                                            self._deadline - now, self._min_stride, self._max_stride)
            self._window_start = now
            self._window_samples = 0
        return self.last_score
