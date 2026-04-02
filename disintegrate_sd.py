import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
import os
import shutil
from PIL import Image

def degrade_all_tensors(module, ratio, max_percent):
    """
    Programmatically degrades a PyTorch module's weights and buffers.
    """
    for name, param in module.named_parameters():
        with torch.no_grad():
            p_min, p_max = param.data.min(), param.data.max()
            p_range = p_max - p_min
            if p_range <= 1e-12: continue
            mask = (torch.rand(param.shape, device=param.device) < ratio)
            delta = max_percent * p_range
            shifts = torch.empty(param.shape, device=param.device).uniform_(-delta, delta)
            param.data[mask] += shifts[mask]

    for name, buf in module.named_buffers():
        if not torch.is_floating_point(buf): continue
        with torch.no_grad():
            b_min, b_max = buf.min(), buf.max()
            b_range = b_max - b_min
            if b_range <= 1e-12: continue
            mask = (torch.rand(buf.shape, device=buf.device) < ratio)
            delta = max_percent * b_range
            shifts = torch.empty(buf.shape, device=buf.device).uniform_(-delta, delta)
            buf[mask] += shifts[mask]

def upscale_image(image, scale=2):
    """
    Upscales a PIL image using Lanczos interpolation for high quality.
    """
    w, h = image.size
    return image.resize((w * scale, h * scale), resample=Image.LANCZOS)

def run_disintegration(target="unet", 
                       num_steps=20, 
                       ratio=0.01, 
                       max_percent=0.05,
                       height=512,
                       width=512,
                       upscale=True,
                       prompt="A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k",
                       model_id="runwayml/stable-diffusion-v1-5"):
    """
    Progressively degrades the chosen component(s) between full generation runs.
    """
    print(f"Loading model: {model_id} | Targeting: {target}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True, 
        variant="fp16"
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        # Standard Stable Diffusion memory optimizations (no xformers needed)
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
    
    res_label = "1024x1024" if upscale else f"{width}x{height}"
    output_dir = f"outputs_{target}_{res_label}"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

    print(f"Generating base image (Step 0) in {output_dir}...")
    image = pipe(prompt, height=height, width=width, generator=generator).images[0]
    if upscale: image = upscale_image(image)
    image.save(f"{output_dir}/step_000_clean.png")

    for i in range(1, num_steps + 1):
        print(f"[{target}] Disintegration Step {i}/{num_steps}...")
        
        if target in ["unet", "both"]:
            degrade_all_tensors(pipe.unet, ratio=ratio, max_percent=max_percent)
        if target in ["text_encoder", "both"]:
            degrade_all_tensors(pipe.text_encoder, ratio=ratio, max_percent=max_percent)
        
        # Free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
        image = pipe(prompt, height=height, width=width, num_inference_steps=30, generator=generator).images[0]
        if upscale: image = upscale_image(image)
        image.save(f"{output_dir}/step_{i:03d}_degraded.png")
        
    print(f"Disintegration of {target} complete. Saved to '{output_dir}/'.")

if __name__ == "__main__":
    prompt = "A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k"
    
    # 512x512 base + 1024x1024 upscale (Stable and high res)
    run_disintegration(
        target="unet", 
        num_steps=20, 
        ratio=0.01, 
        max_percent=0.05, 
        height=512, 
        width=512, 
        upscale=True,
        prompt=prompt
    )
