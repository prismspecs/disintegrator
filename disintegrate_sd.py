import torch
from diffusers import StableDiffusionPipeline
import os
import shutil

def degrade_all_tensors(module, ratio, max_percent):
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

def run_disintegration(model_id="runwayml/stable-diffusion-v1-5", 
                       prompt="A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k",
                       num_steps=20, 
                       ratio=0.005, 
                       max_percent=0.02):
    print(f"Loading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True, variant="fp16"
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    if torch.cuda.is_available(): pipe = pipe.to("cuda")
    
    # Clear old outputs to avoid confusion
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs", exist_ok=True)
    
    # Use a fixed seed so we see the SAME image disintegrate
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

    print("Generating base image (Step 0)...")
    image = pipe(prompt, generator=generator).images[0]
    image.save(f"outputs/step_00_clean.png")

    for i in range(1, num_steps + 1):
        print(f"Disintegration Step {i}/{num_steps}...")
        degrade_all_tensors(pipe.unet, ratio=ratio, max_percent=max_percent)
        
        # Reset generator for consistency
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
        image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
        image.save(f"outputs/step_{i:02d}_degraded.png")
        
    print(f"Disintegration complete. 20 steps saved to 'outputs/'.")

def run_dynamic_disintegration(model_id="runwayml/stable-diffusion-v1-5", 
                               prompt="A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k",
                               ratio=0.001, 
                               max_percent=0.01):
    print(f"Loading fresh model for Dynamic Disintegration...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True, variant="fp16"
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    if torch.cuda.is_available(): pipe = pipe.to("cuda")
    
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

    def callback_fn(step: int, timestep: int, latents: torch.FloatTensor):
        # We degrade at a very low rate during the 50 steps
        degrade_all_tensors(pipe.unet, ratio=ratio, max_percent=max_percent)
        return latents

    print("Generating image with dynamic disintegration (U-Net unravelling during creation)...")
    image = pipe(prompt, num_inference_steps=50, callback=callback_fn, callback_steps=1, generator=generator).images[0]
    image.save(f"outputs/z_dynamic_unravelling.png")

if __name__ == "__main__":
    prompt = "A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k"
    
    # 20 steps, low intensity for "motion"
    run_disintegration(prompt=prompt, num_steps=20, ratio=0.008, max_percent=0.03)
    
    # Sync prompt for comparison
    run_dynamic_disintegration(prompt=prompt, ratio=0.002, max_percent=0.01)
