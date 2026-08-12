import torch
from diffusers import StableDiffusionPipeline
import os
import shutil
from PIL import Image

from bending import modelbend

def upscale_image(image, scale=2):
    w, h = image.size
    return image.resize((w * scale, h * scale), resample=Image.LANCZOS)

def run_disintegration(target="unet",
                       num_steps=100,
                       ratio=0.01,
                       max_percent=0.05,
                       dist="uniform",
                       height=512, 
                       width=512,
                       upscale=True, 
                       low_vram=True,
                       prompt="A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k",
                       model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                       output_dir="outputs_recreated_1024"):
    
    print(f"Loading model: {model_id} | Targeting: {target} | Resolution: {width}x{height} (Upscale: {upscale})")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True, variant="fp16"
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    if torch.cuda.is_available():
        if low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
    
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

    print(f"Generating base image (Step 0) in {output_dir}...")
    image = pipe(prompt, height=height, width=width, generator=generator).images[0]
    if upscale: image = upscale_image(image)
    image.save(f"{output_dir}/step_000_clean.png")

    for i in range(1, num_steps + 1):
        print(f"[{target}] Disintegration Step {i}/{num_steps}...")
        if target in ["unet", "both"]: modelbend(pipe.unet, ratio=ratio, max_percent=max_percent, dist=dist)
        if target in ["text_encoder", "both"]: modelbend(pipe.text_encoder, ratio=ratio, max_percent=max_percent, dist=dist)
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
        image = pipe(prompt, height=height, width=width, num_inference_steps=30, generator=generator).images[0]
        if upscale: image = upscale_image(image)
        image.save(f"{output_dir}/step_{i:03d}_degraded.png")
        
    print(f"Disintegration complete. Saved to '{output_dir}/'.")

if __name__ == "__main__":
    prompt = "A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k"
    
    run_disintegration(
        target="unet", 
        num_steps=100, 
        ratio=0.01, 
        max_percent=0.05, 
        height=512, 
        width=512, 
        upscale=True,
        prompt=prompt,
        output_dir="outputs_recreated_1024"
    )
