# Disintegrator — working notes for Claude

Artistic research project: degrade Stable Diffusion's weights to study how synthesized
reality unravels. Companion to the essay "In Defense of Disintegration" (Grayson Earle).

## Ground rules
1. **Empirical documentation** — every major run writes a `metadata.json` next to its frames
   with seeds, prompt, schedule, and disintegration parameters. This is not "just glitch art";
   the rate of decay is the subject.
2. **GPU first** — CUDA is assumed (RTX 3060, 12 GB). Use fp16.
3. **Reproducibility** — fixed seeds and fixed initial latents, always.
4. **Keep the docs current** — new decay patterns go in the README.

## Environment
- Interpreter is `./venv/bin/python` (the venv is *not* auto-activated).
- torch 2.11 / cu130, diffusers 0.37.
- `ffmpeg` is available at `/usr/bin/ffmpeg`.

## Model availability (checked 2026-07-20)
- Use `stable-diffusion-v1-5/stable-diffusion-v1-5`. The old `runwayml/...` id is a
  leftover from before that org's 2024 takedown — all scripts have been repointed.
- `stabilityai/stable-diffusion-2-1` is **gone** from the Hub (404). The README's
  768x768 suggestions are no longer runnable as written.
- SD 1.5 duplicates subjects above ~512x512. For clean high-res output, render at 512
  and resample up rather than generating large.

## Two decay modes
- **U-Net** → structural collapse. Form dissolves; use for "representation → abstraction".
- **Text encoder** → semantic drift. The model misreads the prompt while staying coherent.
  `disintegrate_sd.py` can map corrupted encoder states back to a clean vocabulary.

## Smoothness
Per-frame *re-randomized* noise (the `degrade_all_tensors` approach in the older scripts)
makes consecutive frames flicker — fine for stills, bad for video. For motion work use
`smooth_disintegrate.py`, which samples **one** fixed perturbation direction and scales it
by a smoothly increasing α. See that file's docstring.

## Housekeeping
- `pi_clone.img` (7.9 GB) is an unrelated Raspberry Pi disk image living in this repo.
  It is gitignored, not deleted — confirm with Grayson before removing it.
- Output/measurement dirs are gitignored; they hold prior runs and are not disposable.
