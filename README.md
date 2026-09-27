# traffic-vision — WIUT Hackathon 2026, CV track

## Team

- **Dilshod Bobomurodov** (captain) — detection, tracking, rules, scene
  calibration, risk estimation, labeling, evaluation harness testing.
- **Sevara Saidova** — team website design and layout, EDA presentation.
- **Muhammadjon Ismoilov** — technical report and results write-up.

(Contact details are on file with the organizers rather than published
here.)

Submission for the WIUT Hackathon 2026 Computer Vision track elimination
task: watch a fixed road-camera video, emit traffic events as
`[start_sec, end_sec, label]` (Part A, mandatory), and optionally emit a
per-frame accident-risk score (Part B, bonus). Full spec:
`WIUT Hackathon _ CV Track Elimination Task.pdf`. `STARTER_KIT_README.md` is
the organizers' own quickstart/reference for the harness and metric — this
file is the team's writeup.

## Install & run

```bash
pip install -r requirements.txt
bash weights/download.sh                 # fetches weights/yolo11s.pt (~19 MB, gitignored)
python scripts/fetch_samples.py          # optional: pulls the 4 sample videos into samples/

python run_submission.py --videos samples --out predictions_samples.json --team dilshod
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
```

Requires Python 3.10+ (developed/tested on 3.14) and the packages in
`requirements.txt` (numpy, opencv-python-headless, ultralytics, scipy).
Torch is pulled in transitively by `ultralytics`; a CUDA-capable GPU is not
required to run but is strongly recommended (see Runtime, below).

## Approach

`solution.py` wires together:

- **`src/detection.py`** — YOLO11s (COCO-pretrained, `imgsz=1280` for
  pedestrian recall from this elevated camera angle) + Ultralytics'
  built-in ByteTrack, sampled adaptively up to ~8 fps for Part A and ~6 fps
  for Part B.
- **`src/calibration.py`** — loads `calibration/scene.json`, a single
  hand-calibrated scene layout (parking pockets, 2 crosswalks, carriageway
  polygons, 1 stop line) built against one sample video and stored as
  normalized coordinates so it rescales to any frame size. The task
  description treats this as one fixed physical camera install, so one
  calibration file is reused for every video.
- **`src/rules.py`** — trajectory + scene-geometry rules turning track
  histories into event segments (stopped vehicles, congestion, jaywalking,
  red-light/stop-line inference from queued-vehicle behavior, U-turns,
  failure-to-yield, accident/near-miss).
- **`src/risk.py`** — a causal time-to-collision + hard-braking estimator
  (`TTCRiskEstimator`) driving both Part B directly and Part A's
  `accident`/`near_miss` classes (the task FAQ explicitly allows Part A to
  reuse Part B's risk signal).

### Classes implemented

9 of the 14 official classes have a real detector behind them:
`accident`, `near_miss`, `red_light`, `illegal_u_turn`, `stopped_vehicle`,
`jaywalking`, `failure_to_yield`, `stop_line`, `congestion`.

The other 5 are deliberately **not** predicted (a class with no real
detector still gets scored — and dragged toward 0 — under Score_A's
macro-F1, so predicting one on a guess only hurts):

- `illegal_turn`, `solid_line_crossing` — need lane-level marking / turn
  restriction calibration this project never built.
- `road_obstacle`, `fire_smoke` — need object classes the COCO-pretrained
  YOLO model was never trained to see.
- `wrong_way` — implemented and tested, then removed: this scene's
  carriageway zones measurably mix multiple legitimate traffic directions
  (a two-way boulevard plus a perpendicular side street; per-zone heading
  concentration measured 0.11–0.24 on the dev clip, where 1.0 would mean a
  single consistent direction), so there was no reliable "correct
  direction" to compare against and it badly over-fired in testing.

**What's learned vs. rule-based:** the only learned component is YOLO11s
itself (COCO-pretrained, unmodified — no fine-tuning, no training was
performed for this submission). Detection → tracking → every event class →
the risk score are all hand-written rules and geometry over the tracker's
output; see `src/rules.py` and `src/risk.py`.

Neither `red_light` nor `stop_line` reads the traffic light's actual color
— the light head in this footage is small, distant, and frequently
occluded by buses/trucks. Both infer "the signal is red for this approach"
from a directly observable proxy: vehicles queued stopped at the
calibrated stop line, the same way a human glancing at the footage would
without seeing the light itself. `illegal_u_turn` similarly flags the
observable heading-reversal maneuver rather than an actually-prohibited
one, for the same reason (no per-lane/turn-restriction calibration). See
`src/rules.py`'s module and per-rule docstrings for the full reasoning
behind each class.

### Runtime / time budget

`run_submission.py` enforces a 3×-video-duration budget for Part A + Part B
together, checked only *after* `detect_events` returns — so an unbounded
Part A on slower-than-expected hardware could burn the whole budget and
score the video empty. `detect_events` and `RiskEstimator` both police
their own share of the combined budget instead of trusting the harness to
cut them off, using an **adaptive frame stride**: every ~15 samples they
recompute how densely they can afford to sample from the just-measured
per-frame cost and the time actually remaining. This means sampling stays
at the intended ~8 fps / ~6 fps ceilings on fast hardware (the organizers'
stated eval machine is a T4-class GPU, 8 CPU cores, 32 GB RAM) and
gracefully drops to whatever rate the hardware can sustain
otherwise — covering the *entire* video at a lower frame rate rather than
sampling densely until time runs out and silently truncating the back of
the clip. Verified on a CPU-only dev machine: a fixed 8 fps target only
ever covered the first ~35% of a 120s clip before the old fixed-deadline
cutoff kicked in; the adaptive version covers 100% of the clip within the
same budget.

The sampling floor (`min_fps`) is deliberately kept at ~1 fps rather than
lower: below that, per-sample cost measured to climb enough on a long run
that the video could still miss budget anyway (plausibly ByteTrack's
lost-track buffer aging by call count, not real time — sparser sampling
lets more stale tracks linger, making each call slower still), *and*
sampling below ~1 fps breaks ByteTrack's frame-to-frame association
assumption outright, producing unreliable track identities (observed: a
259-second single "jaywalking" segment that was clearly several unrelated
pedestrians merged under one track ID). A clean partial-video result is
preferred over a full-video result built on unreliable tracking, so an
unusually long/slow video can still occasionally exceed the outer budget
and score empty. **On this CPU-only dev machine, `samples/C3896.MP4`
(340s, native 4K) is one such case** — `predictions_samples.json` in this
repo currently reflects that (empty for this video, `total_sec` a few
seconds over its 1021s budget). This is a dev-hardware artifact, not a
correctness bug: the same code produces real, sane events well within
budget on shorter/lower-resolution inputs (verified on a 120s 720p proxy of
the same footage), and the organizers' actual eval machine is a T4-class
GPU with far more headroom than this dev box.

### Determinism

`solution.py` seeds Python's `random`, `numpy`, and `torch` (plus
`torch.cuda` when available) at import time. In practice neither YOLO11s
inference nor ByteTrack's matching has any randomness on this code path
given fixed weights and config — the seeding is a guard against any
framework-internal fallback that samples, rather than a guarantee this
pipeline actually needs. We deliberately do **not** set
`torch.backends.cudnn.deterministic = True`: it can meaningfully slow GPU
convolution, which would eat into the time budget above for a guarantee
that isn't otherwise necessary here. Expect run-to-run repeatability on
the same hardware/software stack, modulo the ordinary floating-point
non-associativity of parallel GPU reduction.

## Weights

`weights/yolo11s.pt` — Ultralytics YOLO11 small, COCO-pretrained, fetched
by `weights/download.sh` from Ultralytics' official GitHub release assets
(`ultralytics/assets` v8.4.0). Not fine-tuned — used as a general-purpose
person/vehicle detector feeding the tracker and rules. Ultralytics'
YOLO11 weights are distributed under AGPL-3.0 (see
[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)); the
`ultralytics` package itself is likewise AGPL-3.0-licensed.

## External data and models

- **Models**: Ultralytics YOLO11s, COCO-pretrained, open weights (see
  Weights, below). No other model, and no paid/hosted API, is used at any
  stage of inference.
- **Training data**: none. No training or fine-tuning was performed for
  this submission — YOLO11s is used exactly as released. No public
  accident/traffic dataset (DoTA, CCD, DAD, CADP, UCF-Crime, UA-DETRAC,
  BDD100K, etc.) was used.
- **Video data**: `samples/C3896.MP4`, one of the organizers' own sample
  videos from the target camera (see Data & labels). No footage was
  scraped or collected from the camera by any other means.

## Data & labels

- `samples/C3896.MP4` — one of the organizers' 4 sample videos (see
  `scripts/fetch_samples.py`; only this one has been fetched so far).
- `my_labels.json` / `my_labels_NOTES.md` — self-annotated dev ground truth
  for `C3896.MP4` only, built by visual review of the full video plus
  tracker-assisted candidate generation, used for local `evaluate.py` runs.
  **Not an official label set.** `my_labels_NOTES.md` documents the full
  methodology, known bugs found and fixed along the way (native-resolution
  speed scaling, two crosswalk-polygon calibration issues), and a later
  verification pass that spot-checked every event against the source video.
- No external training data was used — the only model in the pipeline is
  the off-the-shelf COCO-pretrained YOLO11s above.

## Repo layout

```
solution.py           the submission interface (CLASSES, detect_events, RiskEstimator)
run_submission.py     organizers' harness (do not modify)
evaluate.py           organizers' metric (do not modify)
src/                  detection, tracking glue, calibration, rules, risk estimation
calibration/          hand-built scene.json (zones, stop line) for the one calibrated camera
scripts/              dev-only tooling (calibration helpers, proxy clips, preview/visualize) —
                       not part of the submission interface
examples/             ground_truth.json / predictions.json in the exact expected format
my_labels.json        self-annotated dev ground truth (see Data & labels)
weights/download.sh   fetches weights/*.pt (gitignored — too large to commit)
```
