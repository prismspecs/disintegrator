import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
import numpy as np
import os
import argparse
import json
import textwrap
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def render_overlay(image, text_lines, font_path):
    w, h = image.size
    
    # Try to find a working font
    font = None
    for path in [font_path, "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 
                 "/usr/share/fonts/liberation/LiberationMono-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"]:
        try:
            font = ImageFont.truetype(path, 20)
            break
        except: continue
    if not font: font = ImageFont.load_default()

    line_h = 30
    total_text_h = len(text_lines) * line_h + 40
    new_img = Image.new('RGB', (w, h + int(total_text_h)), (0, 0, 0))
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    current_y = h + 20
    for line in text_lines:
        draw.text((20, current_y), line, fill=(0, 255, 65), font=font)
        current_y += line_h

    return new_img

def upscale_image(image, scale=2):
    w, h = image.size
    return image.resize((w * scale, h * scale), resample=Image.LANCZOS)

def degrade_module(module, ratio, max_percent):
    total_abs_drift = 0
    total_params = 0
    for name, param in module.named_parameters():
        with torch.no_grad():
            p_min, p_max = param.data.min(), param.data.max()
            p_range = p_max - p_min
            if p_range <= 1e-12: continue
            
            mask = (torch.rand(param.shape, device=param.device) < ratio)
            delta = max_percent * p_range
            shifts = torch.empty(param.shape, device=param.device).uniform_(-delta, delta)
            
            applied_shifts = shifts * mask.float()
            param.data += applied_shifts
            
            total_abs_drift += applied_shifts.abs().sum().item()
            total_params += param.numel()
            
    return total_abs_drift / total_params if total_params > 0 else 0

def measure(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Mandate violation: CUDA/GPU is required for this project.")
    
    device = "cuda"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"measurements_{args.target}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model {args.model_id} on {device}...")
    load_kwargs = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
    }
    if os.path.exists(args.model_id):
        load_kwargs["local_files_only"] = True

    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, **load_kwargs).to(device)
    pipe.safety_checker = None
    
    target_module = pipe.unet if args.target == "unet" else pipe.text_encoder

    if args.target_drift is not None:
        print(f"Calculating required ratio for target drift {args.target_drift} over {args.steps} steps...")
        total_range_weighted = 0
        total_params = 0
        for name, param in target_module.named_parameters():
            p_range = (param.data.max() - param.data.min()).item()
            total_range_weighted += p_range * param.numel()
            total_params += param.numel()
        
        mean_range = total_range_weighted / total_params
        required_ratio = (args.target_drift / args.steps) / ((args.percent / 2) * mean_range)
        args.ratio = required_ratio
        print(f"  Computed Ratio: {args.ratio:.8f} ({args.ratio*100:.6f}%)")

    # Initial metadata save for reproducibility in case of failure
    metadata = {
        "timestamp": timestamp,
        "mandates_followed": True,
        "status": "in_progress",
        "configuration": vars(args),
        "results": []
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    generator = torch.Generator(device).manual_seed(args.seed)
    latent_h, latent_w = args.height // 8, args.width // 8
    latents = torch.randn((1, 4, latent_h, latent_w), generator=generator, device=device, dtype=pipe.unet.dtype)
    
    print("Computing baseline...")
    with torch.no_grad():
        timestep_idx = 999
        timestep = torch.tensor([timestep_idx], device=device) 
        text_inputs = pipe.tokenizer(args.prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to(device)
        encoder_hidden_states = pipe.text_encoder(text_input_ids)[0]
        initial_noise_pred = pipe.unet(latents, timestep, encoder_hidden_states).sample

        if args.save_visuals:
            print("Generating baseline image...")
            image = pipe(args.prompt, height=args.height, width=args.width, num_inference_steps=args.inference_steps, generator=torch.Generator(device).manual_seed(args.seed)).images[0]
            if args.upscale > 1:
                image = upscale_image(image, args.upscale)
            info = [f"STEP: 0 (Baseline)", f"DRIFT: 0.000000", f"REL DIST: 0.000000"]
            image = render_overlay(image, info, args.font_path)
            image.save(os.path.join(output_dir, "frame_0000.jpg"))

    results = []
    cumulative_drift = 0
    
    print(f"\nDisintegration Parameters:")
    print(f"  Target: {args.target} | Ratio: {args.ratio*100:.6f}% | Max Shift: {args.percent*100:.1f}%")
    
    for step in range(1, args.steps + 1):
        step_drift = degrade_module(target_module, args.ratio, args.percent)
        cumulative_drift += step_drift
        
        with torch.no_grad():
            if args.target == "text_encoder":
                encoder_hidden_states = pipe.text_encoder(text_input_ids)[0]
            noise_pred = pipe.unet(latents, timestep, encoder_hidden_states).sample
            rel_dist = (torch.norm(noise_pred - initial_noise_pred) / torch.norm(initial_noise_pred)).item()

        results.append({"step": step, "cum_param_drift": cumulative_drift, "output_rel_dist": rel_dist})
        
        if step % args.interval == 0 or step == 1:
            print(f"Step {step:4d} | Drift: {cumulative_drift:.6f} | Dist: {rel_dist:.6f}")
            if args.save_visuals:
                image = pipe(args.prompt, height=args.height, width=args.width, num_inference_steps=args.inference_steps, generator=torch.Generator(device).manual_seed(args.seed)).images[0]
                if args.upscale > 1:
                    image = upscale_image(image, args.upscale)
                info = [
                    f"STEP: {step}",
                    f"CUMULATIVE DRIFT (Damage): {cumulative_drift:.6f}",
                    f"OUTPUT RELATIVE DISTANCE: {rel_dist:.6f}"
                ]
                image = render_overlay(image, info, args.font_path)
                image.save(os.path.join(output_dir, f"frame_{step:04d}.jpg"))

    # Final metadata update
    metadata["status"] = "complete"
    metadata["results"] = results
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSession complete. Saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure and visualize Stable Diffusion disintegration.")
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--target", type=str, default="unet", choices=["unet", "text_encoder"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--interval", type=int, default=10, help="Save/print every N steps")
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--target_drift", type=float, default=None, help="Automatically set ratio to reach this total drift by the final step")
    parser.add_argument("--percent", type=float, default=0.05)
    parser.add_argument("--prompt", type=str, default="A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference_steps", type=int, default=25)
    parser.add_argument("--save_visuals", action="store_true", help="Generate and save images with data overlays")
    parser.add_argument("--upscale", type=int, default=1, help="Upscale factor for visual output")
    parser.add_argument("--font_path", type=str, default="/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Bold.ttf")
    
    args = parser.parse_args()
    measure(args)
