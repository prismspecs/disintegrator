"""
Smooth structural disintegration.

The older scripts in this repo re-randomize the weight perturbation at every step. That is
correct for still sequences but it flickers in motion: each frame is an independent sample,
so the image strobes even when the *amount* of damage barely changes.

This script instead samples ONE fixed perturbation direction `d` (per parameter, scaled by
that parameter's own std) and rebuilds the weights each frame as

    w(t) = w0 + alpha(t) * d

from the pristine `w0`. The model then walks a continuous path through weight space, and
with a fixed seed and a fixed initial latent, consecutive frames differ only by the smooth
growth of alpha. The result reads as a morph rather than a strobe.

alpha(t) = alpha_max * u**curve_power, u in [0, 1]. The power curve holds the clean image
legible for the opening stretch, then accelerates into collapse — early weight damage is
perceptually cheap, so a linear ramp feels front-loaded.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
from datetime import datetime

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_PROMPT = (
    "A high-resolution professional photograph of a majestic mountain range at sunrise, "
    "cinematic lighting, sharp focus, 8k"
)


def load_pipe(model_id=MODEL_ID):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16",
    )
    # Disabled deliberately: abstract noise trips the filter and yields black frames.
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def snapshot(module, seed, skip_norm=False):
    """Capture pristine weights and sample the fixed perturbation direction."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base, direction = {}, {}
    for name, param in module.named_parameters():
        if skip_norm and param.ndim == 1:
            continue
        # .item() keeps the scalar on CPU so it can scale the CPU-generated noise;
        # the CPU generator is what makes the direction reproducible across machines.
        std = param.data.float().std().item()
        if not math.isfinite(std) or std <= 1e-12:
            continue
        base[name] = param.data.clone()
        noise = torch.randn(param.shape, generator=gen, dtype=torch.float32)
        direction[name] = (noise * std).to(param.dtype).to(param.device)
    return base, direction


@torch.no_grad()
def apply_alpha(module, base, direction, alpha):
    """Rebuild weights from pristine base — never accumulate, it drifts off the path."""
    params = dict(module.named_parameters())
    for name, w0 in base.items():
        params[name].data.copy_(w0 + alpha * direction[name])


def alpha_at(u, alpha_max, curve_power):
    return alpha_max * (u ** curve_power)


def render(pipe, prompt, latents, steps, height, width):
    return pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        latents=latents,
    ).images[0]


def fixed_latents(pipe, seed, height, width):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, pipe.unet.config.in_channels, height // 8, width // 8)
    lat = torch.randn(shape, generator=gen, dtype=torch.float32)
    return lat.to(pipe.device, dtype=pipe.unet.dtype)


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def calibrate(args):
    """Render an alpha ladder so the collapse point can be chosen by eye, not guessed."""
    pipe = load_pipe(args.model_id)
    latents = fixed_latents(pipe, args.seed, args.render_size, args.render_size)
    base, direction = snapshot(pipe.unet, args.direction_seed, args.skip_norm)

    n = args.calibrate_steps
    alphas = [args.alpha_max * (i / (n - 1)) for i in range(n)]
    tiles = []
    for i, a in enumerate(alphas):
        apply_alpha(pipe.unet, base, direction, a)
        img = render(pipe, args.prompt, latents, args.inference_steps,
                     args.render_size, args.render_size)
        img = img.resize((256, 256), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        label = f"a={a:.3f}"
        draw.text((9, 9), label, fill=(0, 0, 0), font=_font(20))
        draw.text((8, 8), label, fill=(0, 255, 65), font=_font(20))
        tiles.append(img)
        print(f"  calibrate {i+1}/{n}  alpha={a:.4f}", flush=True)

    cols = min(4, n)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * 256, rows * 256), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 256, (i // cols) * 256))
    os.makedirs("calibration", exist_ok=True)
    out = os.path.join("calibration", "ladder.jpg")
    sheet.save(out, quality=92)
    print(f"\nLadder written to {out} — pick the alpha where the subject is truly gone.")


def render_run(args):
    out_dir = args.out or f"renders_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    kf_dir = os.path.join(out_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)

    pipe = load_pipe(args.model_id)
    latents = fixed_latents(pipe, args.seed, args.render_size, args.render_size)
    base, direction = snapshot(pipe.unet, args.direction_seed, args.skip_norm)
    print(f"Perturbing {len(base)} U-Net tensors along a fixed direction.")

    schedule = []
    for i in range(args.keyframes):
        u = i / (args.keyframes - 1)
        a = alpha_at(u, args.alpha_max, args.curve_power)
        apply_alpha(pipe.unet, base, direction, a)
        img = render(pipe, args.prompt, latents, args.inference_steps,
                     args.render_size, args.render_size)
        if img.size != (args.size, args.size):
            img = img.resize((args.size, args.size), Image.LANCZOS)
        img.save(os.path.join(kf_dir, f"kf_{i:05d}.png"))
        schedule.append({"index": i, "u": round(u, 6), "alpha": round(a, 6)})
        if i % 10 == 0 or i == args.keyframes - 1:
            print(f"  {i+1}/{args.keyframes}  alpha={a:.4f}", flush=True)

    meta = {
        "prompt": args.prompt,
        "model_id": args.model_id,
        "target": "unet",
        "method": "fixed-direction weight perturbation, w = w0 + alpha*d",
        "seed": args.seed,
        "direction_seed": args.direction_seed,
        "alpha_max": args.alpha_max,
        "curve_power": args.curve_power,
        "skip_norm": args.skip_norm,
        "inference_steps": args.inference_steps,
        "render_size": args.render_size,
        "output_size": args.size,
        "keyframes": args.keyframes,
        "duration_s": args.duration,
        "fps": args.fps,
        "created": datetime.now().isoformat(timespec="seconds"),
        "schedule": schedule,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nKeyframes in {kf_dir}")
    encode(out_dir, args.fps, args.duration, args.size)
    return out_dir


def encode(out_dir, fps, duration, size):
    """Blend keyframes up to the target frame count, then encode."""
    kf_dir = os.path.join(out_dir, "keyframes")
    keys = sorted(f for f in os.listdir(kf_dir) if f.endswith(".png"))
    if not keys:
        raise SystemExit(f"No keyframes in {kf_dir}")

    total = int(round(fps * duration))
    frames_dir = os.path.join(out_dir, "frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)

    print(f"Interpolating {len(keys)} keyframes -> {total} frames...")
    cache_idx, cache_arr = None, None
    for n in range(total):
        pos = n * (len(keys) - 1) / max(total - 1, 1)
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, len(keys) - 1)
        t = pos - i0

        if cache_idx != i0:
            cache_arr = np.asarray(Image.open(os.path.join(kf_dir, keys[i0])), dtype=np.float32)
            cache_idx = i0
        a = cache_arr
        if i1 == i0 or t == 0.0:
            blended = a
        else:
            b = np.asarray(Image.open(os.path.join(kf_dir, keys[i1])), dtype=np.float32)
            blended = a + (b - a) * t
        Image.fromarray(blended.round().clip(0, 255).astype(np.uint8)).save(
            os.path.join(frames_dir, f"f_{n:06d}.png")
        )
        if n % 200 == 0:
            print(f"  {n}/{total}", flush=True)

    video = os.path.join(out_dir, "disintegration.mp4")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "f_%06d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-vf", f"scale={size}:{size}",
        video,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Video: {video}")
    return video


def main():
    p = argparse.ArgumentParser(description="Smooth structural disintegration video.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--size", type=int, default=720, help="output resolution")
    p.add_argument("--render-size", type=int, default=512,
                   help="generation resolution; SD1.5 duplicates subjects above 512")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--keyframes", type=int, default=450, help="frames actually diffused")
    p.add_argument("--alpha-max", type=float, default=0.35)
    p.add_argument("--curve-power", type=float, default=2.2)
    p.add_argument("--inference-steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42, help="latent seed")
    p.add_argument("--direction-seed", type=int, default=7, help="perturbation direction")
    p.add_argument("--skip-norm", action="store_true",
                   help="leave 1-D (norm/bias) params intact")
    p.add_argument("--out", default=None)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calibrate-steps", type=int, default=8)
    p.add_argument("--encode-only", default=None, metavar="RUN_DIR")
    args = p.parse_args()

    if args.encode_only:
        meta_path = os.path.join(args.encode_only, "metadata.json")
        fps, duration, size = args.fps, args.duration, args.size
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                m = json.load(f)
            fps = m.get("fps", fps)
            duration = m.get("duration_s", duration)
            size = m.get("output_size", size)
        encode(args.encode_only, fps, duration, size)
    elif args.calibrate:
        calibrate(args)
    else:
        render_run(args)


if __name__ == "__main__":
    main()
