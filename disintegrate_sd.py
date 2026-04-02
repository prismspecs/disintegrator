import torch
from diffusers import StableDiffusionPipeline
import os

def degrade_all_tensors(module, ratio, max_percent):
    """
    Programmatically degrades a PyTorch module's weights and buffers.
    Adapted from Grayson Earle's "In Defense of Disintegration".
    """
    # Parameters (Weights/Biases)
    for name, param in module.named_parameters():
        with torch.no_grad():
            p_min, p_max = param.data.min(), param.data.max()
            p_range = p_max - p_min
            
            # Skip if the parameter is essentially constant
            if p_range <= 1e-12:
                continue
                
            # Create a mask for which weights to affect
            mask = (torch.rand(param.shape, device=param.device) < ratio)
            
            # Calculate the range of the shift
            delta = max_percent * p_range
            
            # Generate random shifts in range [-delta, delta]
            shifts = torch.empty(param.shape, device=param.device).uniform_(-delta, delta)
            
            # Apply shifts to the masked weights
            param.data[mask] += shifts[mask]

    # Buffers (Running means, etc.)
    for name, buf in module.named_buffers():
        # Only process floating point buffers
        if not torch.is_floating_point(buf):
            continue
            
        with torch.no_grad():
            b_min, b_max = buf.min(), buf.max()
            b_range = b_max - b_min
            
            if b_range <= 1e-12:
                continue
                
            mask = (torch.rand(buf.shape, device=buf.device) < ratio)
            delta = max_percent * b_range
            shifts = torch.empty(buf.shape, device=buf.device).uniform_(-delta, delta)
            buf[mask] += shifts[mask]

def run_disintegration(model_id="runwayml/stable-diffusion-v1-5", 
                       prompt="A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k",
                       num_steps=10, 
                       ratio=0.01, 
                       max_percent=0.05):
    """
    Loads a Stable Diffusion model and progressively disintegrates the U-Net.
    """
    print(f"Loading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16"
    )
    
    # Disable the Safety Checker to see raw disintegration
    # Note: As weights degrade, the model may produce noise that triggers the safety filter.
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    os.makedirs("outputs", exist_ok=True)
    
    # Generate initial "clean" image
    print("Generating base image...")
    image = pipe(prompt).images[0]
    image.save(f"outputs/step_0_clean.png")

    # Progressively disintegrate the U-Net
    for i in range(1, num_steps + 1):
        print(f"Disintegration Step {i}/{num_steps}...")
        
        # Apply disintegration specifically to the U-Net
        degrade_all_tensors(pipe.unet, ratio=ratio, max_percent=max_percent)
        
        # Generate image with the now-degraded U-Net
        image = pipe(prompt, num_inference_steps=30).images[0]
        image.save(f"outputs/step_{i}_degraded.png")
        
    print("Disintegration complete. Check the 'outputs' folder.")

def run_dynamic_disintegration(model_id="runwayml/stable-diffusion-v1-5", 
                               prompt="A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k",
                               ratio=0.001, 
                               max_percent=0.01):
    """
    Degrades the U-Net *during* the diffusion process at each step.
    This simulates an "unraveling" as the image is being formed.
    """
    print(f"Loading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16"
    )
    
    # Disable the Safety Checker to see raw disintegration
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    os.makedirs("outputs", exist_ok=True)

    def callback_fn(step: int, timestep: int, latents: torch.FloatTensor):
        print(f"Step {step}: Degrading U-Net...")
        degrade_all_tensors(pipe.unet, ratio=ratio, max_percent=max_percent)
        return latents

    print("Generating image with dynamic disintegration...")
    image = pipe(prompt, num_inference_steps=50, callback=callback_fn, callback_steps=1).images[0]
    image.save(f"outputs/dynamic_disintegration.png")
    print("Complete. Check outputs/dynamic_disintegration.png")

if __name__ == "__main__":
    print("Starting Disintegration Demo...")
    photorealistic_prompt = "A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k"
    
    # 1. Run a longer progressive disintegration (6 steps)
    run_disintegration(
        prompt=photorealistic_prompt,
        num_steps=6, 
        ratio=0.03,
        max_percent=0.15
    )
    
    # 2. Run a dynamic disintegration on the SAME prompt for comparison
    run_dynamic_disintegration(
        prompt=photorealistic_prompt,
        ratio=0.005,
        max_percent=0.02
    )
    print("Demo complete. Check the 'outputs' folder for images.")
