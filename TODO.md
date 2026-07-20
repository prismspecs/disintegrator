# TODO

## Recover the "majestic" starting composition

The 60s 720x720 run (`renders_mountains_720`, commit 7a6d150) opens on a hazy
blue-teal valley with distant peaks. The remembered image — see
`outputs_unet_20260505_141358/frame_0000.jpg` — is a completely different
composition: dense snow-capped peaks filling the frame, lit by pink/red
alpenglow. Same prompt, same seed value (42).

The seed *number* was never the difference. Two things diverged:

**1. Generator device.** The old script does
`torch.Generator(pipe.device).manual_seed(42)` — a CUDA generator.
`smooth_disintegrate.py:fixed_latents` builds latents with a CPU generator
(chosen so directions reproduce across machines). CUDA and CPU RNG produce
*entirely different noise for the same seed*, so the composition is unrelated.

**2. Render resolution.** The old frame is 2048x2160 = a 1024x1024 base at 2x
upscale plus the text band. The new run generates at 512. At 1024, SD 1.5's
subject-duplication tendency is what packs the frame edge-to-edge with peaks —
the "majestic" density is partly a duplication artifact, not just a better seed.

Either difference alone changes the image completely.

### Options next time
- Add `--latent-device {cpu,cuda}` to `smooth_disintegrate.py` and use `cuda`
  to reproduce the old noise exactly.
- Render at 1024 and downscale to 720, accepting (or exploiting) the
  duplication that produces the packed-peaks look.
- Add a `--seed-sheet N` mode: render a contact sheet of N seeds at low step
  count to pick a composition before committing to a full run. Same idea as
  `--calibrate`, applied to seeds.

Cheapest first move: re-run `--calibrate --calibrate-steps 1` at 1024 with a
CUDA generator and confirm the old composition comes back before rendering
anything long.

## Smaller items
- `GEMINI.md` and `CLAUDE.md` now overlap; consolidate into one.
- The amber/orange drift of the decay is a property of `--direction-seed 7`.
  Try other direction seeds for different collapse palettes.
- `--curve-power 2.2` makes the last 10s move ~5x faster than the first 10s.
  Try 1.6 for a more evenly paced dissolve.
