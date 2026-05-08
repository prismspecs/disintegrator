import torch
from diffusers import StableDiffusionPipeline
import os
import shutil
from PIL import Image
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

def run_denoising_visualization(
    prompt="A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k",
    model_id="runwayml/stable-diffusion-v1-5",
    num_inference_steps=50,
    output_dir="denoising_steps",
    height=512,
    width=512,
    seed=42
):
    print(f"Loading model: {model_id} | Native Resolution: {width}x{height}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        use_safetensors=True, 
        variant="fp16" if torch.cuda.is_available() else None
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
    else:
        pipe = pipe.to(device)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    generator = torch.Generator(device).manual_seed(seed)
    
    intermediate_images = []
    
    # Define a callback to save latents
    def callback(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        save_interval = max(1, num_inference_steps // 20)
        if step % save_interval == 0 or step == num_inference_steps - 1:
            print(f"Captured step {step}")
            pil_img = latents_to_pil(latents, pipe.vae)
            intermediate_images.append((step, pil_img))
        return callback_kwargs

    print(f"Generating image with {num_inference_steps} steps...")
    
    batch_size = 1
    
    # Get initial noise
    shape = (batch_size, pipe.unet.config.in_channels, height // 8, width // 8)
    init_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    
    print("Saving initial noise...")
    noise_img = latents_to_pil(init_noise, pipe.vae)
    noise_img.save(os.path.join(output_dir, "step_000_noise.png"))
    
    # Run the pipeline
    pipe(
        prompt, 
        latents=init_noise,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps, 
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"]
    )
    
    print("Saving intermediate images...")
    for i, (step, img) in enumerate(intermediate_images):
        filename = f"step_{i+1:03d}_denoising_step_{step}.png"
        img.save(os.path.join(output_dir, filename))
        
    print(f"Denoising visualization complete. Saved to '{output_dir}/'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize the Stable Diffusion denoising process.")
    parser.add_argument("--prompt", type=str, default="A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k", help="The text prompt to generate.")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--output", type=str, default="denoising_steps", help="Output directory.")
    parser.add_argument("--height", type=int, default=512, help="Native height.")
    parser.add_argument("--width", type=int, default=512, help="Native width.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()
    
    run_denoising_visualization(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        output_dir=args.output,
        height=args.height,
        width=args.width,
        seed=args.seed
    )
