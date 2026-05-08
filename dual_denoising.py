import torch
from diffusers import StableDiffusionPipeline
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def latents_to_pil(latents, vae):
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    if image.ndim == 3:
        image = image[None, ...]
    image = (image * 255).round().astype("uint8")
    if image.shape[0] == 1:
        return Image.fromarray(image[0])
    return [Image.fromarray(img) for img in image]

def get_text_overlay(image, text, font_size=20):
    draw = ImageDraw.Draw(image)
    # Use a basic font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw black shadow for readability
    draw.text((12, 12), text, font=font, fill="black")
    draw.text((10, 10), text, font=font, fill="white")
    return image

def run_dual_denoising(
    prompt1="A majestic mountain range at sunrise, cinematic lighting",
    prompt2="A futuristic cyberpunk city with neon lights, rain on pavement",
    model_id="runwayml/stable-diffusion-v1-5",
    num_inference_steps=50,
    output_dir="dual_denoising",
    height=512,
    width=512,
    seed=42
):
    print(f"Loading model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16" if torch.cuda.is_available() else None
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    else:
        pipe = pipe.to(device)

    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    generator = torch.Generator(device).manual_seed(seed)
    batch_size = 1
    shape = (batch_size, pipe.unet.config.in_channels, height // 8, width // 8)
    shared_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    
    def capture_denoising_path(prompt, latents):
        frames = []
        def callback(pipe, step, timestep, callback_kwargs):
            l = callback_kwargs["latents"]
            save_interval = max(1, num_inference_steps // 20)
            if step % save_interval == 0 or step == num_inference_steps - 1:
                # We can't easily extract internal attention maps here without deep monkeypatching,
                # but we can show the current "predicted clean image" or just the latent state.
                pil_img = latents_to_pil(l, pipe.vae)
                frames.append((step, timestep.item(), pil_img))
            return callback_kwargs

        pipe(
            prompt, 
            latents=latents,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps, 
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"]
        )
        return frames

    print(f"Path 1: {prompt1}")
    path1_frames = capture_denoising_path(prompt1, shared_noise.clone())
    
    print(f"Path 2: {prompt2}")
    path2_frames = capture_denoising_path(prompt2, shared_noise.clone())
    
    print("Combining frames...")
    # Initial noise frame
    init_noise_pil = latents_to_pil(shared_noise, pipe.vae)
    combined_init = Image.new('RGB', (width * 2 + 10, height + 60), (30, 30, 30))
    combined_init.paste(init_noise_pil, (0, 0))
    combined_init.paste(init_noise_pil, (width + 10, 0))
    get_text_overlay(combined_init, f"STEP 0: Starting Noise (Seed {seed})", font_size=24)
    combined_init.save(os.path.join(output_dir, "step_000_shared_noise.png"))

    for i in range(len(path1_frames)):
        step, t, img1 = path1_frames[i]
        _, _, img2 = path2_frames[i]
        
        # Create a side-by-side canvas
        combined = Image.new('RGB', (width * 2 + 10, height + 80), (20, 20, 20))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (width + 10, 0))
        
        # Add labels
        label1 = prompt1[:50] + "..." if len(prompt1) > 50 else prompt1
        label2 = prompt2[:50] + "..." if len(prompt2) > 50 else prompt2
        
        draw = ImageDraw.Draw(combined)
        get_text_overlay(combined, f"Step {step} | Timestep {int(t)}", font_size=20)
        
        # Bottom labels
        draw.text((10, height + 10), f"Prompt 1: {label1}", fill="white")
        draw.text((width + 20, height + 10), f"Prompt 2: {label2}", fill="white")
        
        filename = f"step_{i+1:03d}_denoising.png"
        combined.save(os.path.join(output_dir, filename))

    print(f"Dual denoising complete. Saved to '{output_dir}/'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", type=str, default="A majestic mountain range at sunrise, cinematic lighting")
    parser.add_argument("--p2", type=str, default="A futuristic cyberpunk city with neon lights, rain on pavement")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--output", type=str, default="dual_denoising")
    args = parser.parse_args()
    
    run_dual_denoising(
        prompt1=args.p1,
        prompt2=args.p2,
        num_inference_steps=args.steps,
        height=args.res,
        width=args.res,
        output_dir=args.output
    )
