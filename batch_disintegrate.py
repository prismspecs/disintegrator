import torch
from diffusers import StableDiffusionPipeline
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import textwrap

def degrade_all_tensors(module, ratio, max_percent):
    device = next(module.parameters()).device
    if device.type == 'meta': return

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
    w, h = image.size
    return image.resize((w * scale, h * scale), resample=Image.LANCZOS)

def create_title_frame(prompt, width=1024, height=1024):
    image = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
        
    # Wrap text
    lines = textwrap.wrap(prompt, width=40)
    
    # Calculate text height for centering
    line_spacing = 10
    total_height = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        total_height += (bbox[3] - bbox[1]) + line_spacing
    
    current_y = (height - total_height) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x = (width - line_width) // 2
        draw.text((x, current_y), line, font=font, fill="yellow")
        current_y += (bbox[3] - bbox[1]) + line_spacing
        
    return image

def run_batch_disintegration(configs, 
                             num_steps=500, 
                             ratio=0.002,  # Slower decay: 0.2% instead of 1%
                             max_percent=0.02, # Slower decay: 2% instead of 5%
                             height=512, 
                             width=512,
                             upscale=True, 
                             low_vram=True,
                             model_id="runwayml/stable-diffusion-v1-5",
                             base_path="/mnt/coolrunnings/mega/projects/disintegration"):
    
    print(f"Loading model: {model_id}")
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

    # Store clean state for reloading
    import copy
    clean_unet_state = copy.deepcopy(pipe.unet.state_dict())
    if hasattr(pipe, "text_encoder"):
        clean_text_encoder_state = copy.deepcopy(pipe.text_encoder.state_dict())

    for config in configs:
        prompt = config['prompt']
        folder_name = config['folder']
        output_dir = os.path.join(base_path, folder_name)
        
        print(f"\n--- Starting Disintegration: {folder_name} ---")
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 000: Title Frame
        print(f"Creating title frame...")
        title_img = create_title_frame(prompt)
        title_img.save(f"{output_dir}/step_000_title.png")
        
        print("Reloading clean weights from memory...")
        pipe.unet.load_state_dict(clean_unet_state)
        if hasattr(pipe, "text_encoder"):
             pipe.text_encoder.load_state_dict(clean_text_encoder_state)

        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

        print(f"Generating base image...")
        image = pipe(prompt, height=height, width=width, generator=generator).images[0]
        if upscale: image = upscale_image(image)
        image.save(f"{output_dir}/step_001_clean.png")

        for i in range(1, num_steps + 1):
            if i % 10 == 0:
                print(f"Disintegration Step {i}/{num_steps}...")
            
            degrade_all_tensors(pipe.unet, ratio=ratio, max_percent=max_percent)
            
            if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            image = pipe(prompt, height=height, width=width, num_inference_steps=30, generator=generator).images[0]
            if upscale: image = upscale_image(image)
            image.save(f"{output_dir}/step_{i+1:03d}_degraded.png")
        
    print(f"\nAll disintegrations complete. Results in {base_path}")

if __name__ == "__main__":
    configs = [
        {
            "prompt": "A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k",
            "folder": "mountains"
        },
        {
            "prompt": "A classic 1960s luxury automobile parked on a cliffside, golden hour sun rays reflecting off the polished chrome and windshield, lens flare, highly detailed",
            "folder": "classic_car"
        },
        {
            "prompt": "A beautiful couple walking hand-in-hand on a serene tropical beach, soft sunset light, waves gently crashing, romantic atmosphere, cinematic 8k",
            "folder": "beach_couple"
        }
    ]
    
    run_batch_disintegration(configs)
