---
name: render-disintegration
description: Render a smooth disintegration video — a prompt's image dissolving from clean representation into abstraction — and encode it with ffmpeg. Use when asked to make a disintegration video/sequence, tune the decay curve, calibrate how far to degrade, or re-encode existing rendered frames.
---

# Rendering a disintegration video

Workflow for `smooth_disintegrate.py`. Always use `./venv/bin/python` — the venv is not
auto-activated.

## 1. Calibrate before committing to a long render

You cannot guess the α (perturbation magnitude) that means "fully abstract" for a given
prompt — it shifts with prompt, resolution, and step count. Render a ladder first:

```bash
./venv/bin/python smooth_disintegrate.py --calibrate --alpha-max 0.6 --calibrate-steps 8
```

This writes `calibration/ladder.jpg`, a labelled contact sheet. Read that image and pick
the α where the subject is genuinely gone, not merely damaged. That value becomes
`--alpha-max` for the real render. Calibration costs ~8 renders (well under a minute).

## 2. Render

```bash
./venv/bin/python smooth_disintegrate.py \
    --alpha-max <calibrated> --duration 60 --fps 30 --keyframes 450 --size 720
```

Rendering is the slow part (~2 s per keyframe on the 3060). Run it in the background and
poll the frame count rather than blocking:

```bash
ls renders_*/keyframes/ | wc -l
```

## 3. Encode (or re-encode) without re-rendering

```bash
./venv/bin/python smooth_disintegrate.py --encode-only renders_<timestamp>
```

Re-encoding is cheap, so iterate on fps/interpolation here rather than re-rendering.

## Why it looks smooth

The naive approach — re-randomizing weight noise every frame — flickers, because each
frame is a fresh sample. This script instead draws **one** perturbation direction and
scales it by a monotonically increasing α, so the model traces a continuous path through
weight space. Combined with a fixed seed and a fixed initial latent, consecutive frames
are near-identical and the dissolve reads as a morph rather than a strobe.

Non-negotiables for smoothness:
- One fixed direction tensor, never resampled mid-render.
- Weights rebuilt each frame as `w = w0 + α·d` from pristine `w0`, never accumulated
  (accumulation compounds float error and drifts off the intended path).
- Same seed and same latent for every frame.
- Save keyframes as PNG; JPEG ringing between frames shows up as shimmer.

## Tuning the curve

`--curve-power` shapes α over time (default 2.2):
- Higher (3.0+) — long legible opening, violent late collapse.
- Lower (1.5) — decay starts eating the image almost immediately.
- 1.0 — linear; usually feels front-loaded because early damage is perceptually cheap.

If the ending saturates into flat static too early, lower `--alpha-max`; the most
interesting territory is usually just before total noise.

## Gotchas

- SD 1.5 duplicates subjects above 512x512. Render at 512 and let the script resample to
  the target size; do not raise `--render-size`.
- Holding `w0` + direction + model costs ~5 GB on top of the pipeline. It fits in 12 GB at
  512x512, but not alongside another CUDA process.
- The safety checker is disabled on purpose — abstract noise trips it and yields black
  frames mid-sequence.
- Every run writes `metadata.json`. Keep it with the frames; the project treats decay
  parameters as data, not incidental config.
