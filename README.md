# depal-al

Depalletizing vision for robotic suction picking: an RGB-D camera over a pallet
detects each item's **top-face geometric center + long/short axis in 3D** (ready
to use as a pick pose), signals when the pallet is empty, and feeds an
**active-learning loop** that incrementally improves the detector from field
hard-cases.

Two halves:

- **`dr_onsite_adap.runtime`** — live RGB-D detection (RF-DETR segmentation),
  top-face pick-pose estimation, occlusion filtering, SAM labeling, 2D HUD +
  3D snapshot.
- **`dr_onsite_adap.pipeline`** — the active-learning training loop: ingest
  labels → build a replay-mixed dataset → LoRA fine-tune → merge → evaluate.

## How it works

**Top-face estimation (`runtime/geometry.py`).** Per detected item: take the
mask contour, pick the top depth layer with a trimmed median of per-pixel
*height along world-up* (tilt-invariant), build the horizontal plane, ray-project
the contour onto it, run `minAreaRect` **in plane coordinates** (so perspective
foreshortening is corrected), and lift the rectangle back to 3D. Output: center,
long/short axis, side lengths (mm). Sub-mm in synthetic tilt tests.

**Active learning (`pipeline/`).** Field hard-cases are labeled with SAM in the
live app ('t'), ingested into a COCO set, then mixed with **pallet-bearing
synthetic replay** before a LoRA round. The replay is what makes a small real
batch deployable — it teaches the new appearance without the catastrophic
forgetting (e.g. pallet head collapse) seen when training on a pallet-free batch.

## Install

```bash
pip install -e ".[label,train,orbbec]"     # or [realsense] for a RealSense camera
```

Camera SDKs are optional extras (install only the backend you use). `pyorbbecsdk`
often has no PyPI wheel — build it from
<https://github.com/orbbec/pyorbbecsdk> if `pip` can't find it.

## Configuration

Everything is driven by environment variables (no hard-coded paths). Defaults
are relative to `DEPAL_HOME` (current dir). Common ones:

| Variable | Meaning |
|---|---|
| `DEPAL_HOME` | base dir for relative defaults |
| `DEPAL_WEIGHTS` | detection weights `.pth` (runtime) |
| `DEPAL_NUM_CLASSES` | class count the weights expect (default 4) |
| `DEPAL_DEVICE` | `cpu` / `cuda` |
| `DEPAL_SAM_WEIGHTS` | SAM checkpoint (labeling) |
| `DEPAL_DATA_DIR` | active-learning COCO dataset root |
| `DEPAL_CAMERA_BACKEND` | `auto` / `realsense` / `orbbec` |
| `DEPAL_UP_DIR` | world-up in camera frame, e.g. `-0.005,-0.428,-0.904` (fixed-mount fallback when no IMU) |
| `DEPAL_Z_MAX` | max valid depth (m); raise for high mounts |

Pipeline paths: `DEPAL_BASE_WEIGHTS`, `DEPAL_TRAIN_DATA`, `DEPAL_LORA_SAFE`,
`DEPAL_LORA_RUN`, `DEPAL_FINAL`, `DEPAL_PALLET_DATA`, `DEPAL_REPLAY_OUT`.
See `src/dr_onsite_adap/config.py` for the full list and defaults.

## Runtime — live app

```bash
DEPAL_WEIGHTS=/path/to/model.pth DEPAL_DEVICE=cuda depal-run
```

Keys: `d` (re)detect · `v` 3D snapshot · `c` clear · `r` set/clear ROI crop ·
`t` SAM label · `q` quit. An empty pallet (pallet visible, zero pick targets)
shows a **PALLET EMPTY — DONE** banner.

## Pipeline — active-learning round

```bash
# 0. (one-off) assemble the pallet-labeled synthetic dataset + declare category
depal-dataset build-pallet --src-base <real_coco> --synth <synth1> <synth2> --out $DEPAL_PALLET_DATA
depal-dataset declare-pallet --data-dir $DEPAL_DATA_DIR

# 1. ingest field labels collected via the live app's 't' key
depal-ingest

# 2. build a replay-mixed training set (real + pallet-bearing synthetic replay)
depal-replay

# 3. run the round (LoRA train → merge), then evaluate
DEPAL_TRAIN_DATA=$DEPAL_REPLAY_OUT depal-round --replay
depal-eval --models prev=/path/old.pth new=$DEPAL_FINAL --data $DEPAL_PALLET_DATA/valid

# 4. deploy: point DEPAL_WEIGHTS at $DEPAL_FINAL on the edge device.
#    Next round: set DEPAL_BASE_WEIGHTS to this result.
```

`depal-round` stops if the training set has 0 pallet annotations (a pallet-free
batch can weaken the pallet head); use `--replay` to mix in pallet-bearing
synthetic frames, or `--force` to override.

## Layout

```
src/dr_onsite_adap/
├── config.py            env-driven settings (paths, classes, geometry, hyperparams)
├── runtime/
│   ├── app.py           live loop + keys
│   ├── detect.py        run_detection / draw_state
│   ├── geometry.py      top-face estimation (plane-space minAreaRect)
│   ├── occlusion.py     self-hull occlusion filter
│   ├── inference.py     RF-DETR segmentation wrapper
│   ├── viz.py viz3d.py  2D HUD + 3D snapshot
│   ├── labeler.py       embedded SAM labeling
│   └── cameras/         realsense + orbbec backends, auto-detect
└── pipeline/
    ├── ingest.py        sam_hard triples → COCO (non-destructive append)
    ├── replay.py        build replay-mixed training set
    ├── train_lora.py    LoRA fine-tune
    ├── merge.py         fuse adapter → single .pth
    ├── eval.py          per-class AP@0.5 comparison
    ├── dataset.py       build-pallet / declare-pallet helpers
    └── round.py         orchestrator (replay → train → merge)
```
