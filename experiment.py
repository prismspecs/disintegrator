"""Measured modelbending run: uniform vs gaussian, with per-step metrics.

Produces, for each distribution, a frame sequence plus a CSV recording how far
the model has moved in weight space and how far its denoising direction has
rotated away from the pristine model (the disorientation angle).
"""

import argparse, csv, json, math, os, time

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from bending import modelbend, make_generator

MODEL_IDS = ["stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"]
PROMPT = ("A high-resolution professional photograph of a majestic mountain range "
          "at sunrise, cinematic lighting, 8k")


def load_pipe(dtype):
    last = None
    for mid in MODEL_IDS:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                mid, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
            print(f"loaded {mid} ({dtype})")
            return pipe
        except Exception as e:                                   # noqa: BLE001
            last = e
            print(f"  {mid} failed: {str(e)[:120]}")
    raise SystemExit(f"no model could be loaded: {last}")


def high_freq_energy(img):
    """Fraction of spectral energy above half-Nyquist. Structure collapses -> falls."""
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    f = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    total = f.sum()
    return float(f[r > min(cy, cx) * 0.5].sum() / total) if total > 0 else 0.0


@torch.no_grad()
def weight_state(unet, pristine):
    """Relative L2 drift and global cosine against the pristine parameters.

    Reductions run on-device and only scalars come back; moving ~860M
    parameters to the CPU each step would dominate the run.
    """
    dd = ww = cw = c0 = 0.0
    for name, p in unet.named_parameters():
        w = p.detach().float()
        w0 = pristine[name].float()
        dd += (w - w0).pow(2).sum().item()
        ww += w0.pow(2).sum().item()
        cw += (w * w0).sum().item()
        c0 += w.pow(2).sum().item()
    return math.sqrt(dd / ww), cw / math.sqrt(c0 * ww)


@torch.no_grad()
def eps_predictions(unet, probes):
    return [unet(lat, t, encoder_hidden_states=cond).sample.float().cpu().flatten()
            for lat, t, cond in probes]


def disorientation_angle(bent, clean):
    """Mean angle in degrees between bent and pristine noise predictions."""
    out = []
    for a, b in zip(bent, clean):
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).clamp(-1, 1).item()
        out.append(math.degrees(math.acos(cos)))
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ratio", type=float, default=0.01)
    ap.add_argument("--percent", type=float, default=0.05)
    ap.add_argument("--inference_steps", type=int, default=25)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bend_seed", type=int, default=7)
    ap.add_argument("--dists", type=str, default="uniform,gaussian")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one image only, verify the stack")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if (args.fp32 or device == "cpu") else torch.float16
    pipe = load_pipe(dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet

    def render():
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        return pipe(PROMPT, height=args.size, width=args.size,
                    num_inference_steps=args.inference_steps, generator=g).images[0]

    if args.smoke:
        t0 = time.time()
        img = render()
        a = np.asarray(img.convert("L"), dtype=np.float32)
        print(f"smoke: {time.time()-t0:.1f}s  mean={a.mean():.1f}  std={a.std():.1f}  "
              f"hf={high_freq_energy(img):.4f}  {'OK' if a.std() > 5 else 'BLANK/NaN -- try --fp32'}")
        os.makedirs(args.out, exist_ok=True)
        img.save(f"{args.out}/smoke.jpg", quality=90)
        return

    os.makedirs(args.out, exist_ok=True)
    pristine = {k: v.detach().clone().cpu() for k, v in unet.state_dict().items()}
    # kept on-device so the per-step drift reductions never leave the GPU
    pristine_params = {k: v.detach().clone() for k, v in unet.named_parameters()}

    # Fixed probes for the disorientation angle: same latents, timesteps, prompt
    # embedding for every measurement, so only the weights vary.
    pg = torch.Generator(device="cpu").manual_seed(1234)
    cond = pipe.encode_prompt(PROMPT, device, 1, False)[0]
    lat_ch, lat_hw = unet.config.in_channels, args.size // 8
    probes = [(torch.randn(1, lat_ch, lat_hw, lat_hw, generator=pg).to(device, dtype),
               torch.tensor(t, device=device), cond)
              for t in (800, 500, 200)]
    eps_clean = eps_predictions(unet, probes)
    print(f"probes: {len(probes)} timesteps, latent {lat_ch}x{lat_hw}x{lat_hw}")

    meta = dict(vars(args), device=device, dtype=str(dtype), prompt=PROMPT,
                model=MODEL_IDS[0], torch=torch.__version__)
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=1)

    base = render()
    base.save(f"{args.out}/base.jpg", quality=92)
    base_hf = high_freq_energy(base)

    for dist in args.dists.split(","):
        unet.load_state_dict(pristine)                     # restore for a fair comparison
        gen = make_generator(args.bend_seed, device)
        d_out = f"{args.out}/{dist}"
        os.makedirs(d_out, exist_ok=True)
        base.save(f"{d_out}/frame_00.jpg", quality=92)

        rows = [dict(step=0, coverage=0.0, rel_drift=0.0, cos_w=1.0, theta_deg=0.0,
                     hf_energy=base_hf, jpg_bytes=os.path.getsize(f"{d_out}/frame_00.jpg"))]
        print(f"\n=== {dist} ===")
        for i in range(1, args.steps + 1):
            t0 = time.time()
            modelbend(unet, ratio=args.ratio, max_percent=args.percent,
                      dist=dist, generator=gen)
            drift, cos_w = weight_state(unet, pristine_params)
            theta = disorientation_angle(eps_predictions(unet, probes), eps_clean)

            img = render()
            path = f"{d_out}/frame_{i:02d}.jpg"
            img.save(path, quality=92)
            rows.append(dict(step=i, coverage=1 - (1 - args.ratio) ** i,
                             rel_drift=drift, cos_w=cos_w, theta_deg=theta,
                             hf_energy=high_freq_energy(img),
                             jpg_bytes=os.path.getsize(path)))
            print(f"  {i:2d}/{args.steps}  theta={theta:6.2f}deg  cos_w={cos_w:.5f}  "
                  f"drift={drift*100:5.2f}%  hf={rows[-1]['hf_energy']:.4f}  ({time.time()-t0:.1f}s)")

        with open(f"{args.out}/{dist}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print("\ndone ->", args.out)


if __name__ == "__main__":
    main()
