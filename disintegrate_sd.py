import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import argparse
import textwrap
from datetime import datetime

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

def precompute_semantic_anchors(pipe, device):
    """
    Precomputes the 'clean' output hidden states for a legible vocabulary.
    These act as anchors to map degraded states back to human concepts.
    """
    print("Precomputing semantic anchors (identifying legible concepts)...")
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    
    valid_ids = []
    valid_tokens = []
    
    # Filter vocab for legible words: alpha only, length >= 3
    for i in range(tokenizer.vocab_size):
        token = tokenizer.convert_ids_to_tokens([i])[0]
        clean_token = token.replace('</w>', '')
        if len(clean_token) >= 3 and clean_token.isalpha():
            valid_ids.append(i)
            valid_tokens.append(clean_token.upper())
            
    anchors = []
    batch_size = 512
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    
    text_encoder.eval()
    with torch.no_grad():
        for i in range(0, len(valid_ids), batch_size):
            batch = valid_ids[i:i+batch_size]
            # Construct [SOS, T, EOS] sequences
            input_ids = torch.full((len(batch), 77), eos, dtype=torch.long, device=device)
            input_ids[:, 0] = bos
            for j, tid in enumerate(batch):
                input_ids[j, 1] = tid
            
            # We take the hidden state at the EOS position (index 2)
            # This contains the 'conceptual' summary of the token
            outputs = text_encoder(input_ids)[0]
            concept_vectors = outputs[:, 2, :]
            anchors.append(concept_vectors.cpu())
            
    anchors = torch.cat(anchors, dim=0)
    # Normalize for cosine similarity via dot product
    anchors = F.normalize(anchors, p=2, dim=1)
    return anchors, valid_tokens

def get_semantic_drift(pipe, anchors, valid_tokens, prompt_ids, top_k=6):
    """
    Maps the corrupted encoder's output for the prompt back to the nearest clean concepts.
    """
    pipe.text_encoder.eval()
    with torch.no_grad():
        input_ids = prompt_ids.to(pipe.device)
        outputs = pipe.text_encoder(input_ids)[0]
        
        # Focus on the EOT (End of Text) token hidden state
        eos_token_id = pipe.tokenizer.eos_token_id
        eos_indices = (input_ids[0] == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            idx = eos_indices[0].item()
            concept_vector = outputs[0, idx].unsqueeze(0)
        else:
            concept_vector = outputs[0].mean(dim=0, keepdim=True)

        concept_vector = F.normalize(concept_vector, p=2, dim=1)
        
        # Compute similarities with all clean anchors
        sim = torch.mm(concept_vector, anchors.to(pipe.device).T)[0]
        
        values, indices = torch.topk(sim, top_k)
        drifted = [valid_tokens[idx] for idx in indices.tolist()]
        
        return " | ".join(drifted)

def render_overlay(image, text, font_path, label="CONCEPT"):
    w, h = image.size
    
    # Try to find a working font
    font = None
    for path in [font_path, "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 
                 "/usr/share/fonts/liberation/LiberationMono-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"]:
        try:
            font = ImageFont.truetype(path, 24)
            break
        except: continue
    if not font: font = ImageFont.load_default()

    max_chars = 80 if w >= 1024 else 40
    wrapped_lines = textwrap.wrap(f"{label}: {text}", width=max_chars)

    line_h = 35 
    try:
        bbox = font.getbbox("Ayj")
        line_h = (bbox[3] - bbox[1]) + 15
    except: pass

    total_text_h = len(wrapped_lines) * line_h + 40
    new_img = Image.new('RGB', (w, h + int(total_text_h)), (0, 0, 0))
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    current_y = h + 20
    for line in wrapped_lines:
        draw.text((20, current_y), line, fill=(0, 255, 65), font=font)
        current_y += line_h

    return new_img

def upscale_image(image, scale=2):
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
                       model_id="runwayml/stable-diffusion-v1-5",
                       font_path="/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Bold.ttf",
                       show_drift=None,
                       inference_steps=25):

    # Auto-determine if we should show semantic drift
    if show_drift is None:
        show_drift = (target != "unet")

    print(f"Loading model: {model_id}")
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
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()

    prompt_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
    
    anchors, valid_tokens = None, None
    if show_drift:
        anchors, valid_tokens = precompute_semantic_anchors(pipe, pipe.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs_{target}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    generator = torch.Generator(pipe.device).manual_seed(42)

    print(f"Step 0: Generating clean base...")
    image = pipe(prompt, height=height, width=width, num_inference_steps=inference_steps, generator=generator).images[0]
    if upscale: image = upscale_image(image)
    
    if show_drift:
        image = render_overlay(image, prompt, font_path, label="ORIGINAL PROMPT")
    
    image.save(f"{output_dir}/frame_0000.jpg", quality=90)

    for i in range(1, num_steps + 1):
        if target in ["unet", "both"]:
            degrade_all_tensors(pipe.unet, ratio=ratio, max_percent=max_percent)
        if target in ["text_encoder", "both"]:
            degrade_all_tensors(pipe.text_encoder, ratio=ratio, max_percent=max_percent)

        drift_text = ""
        if show_drift:
            drift_text = get_semantic_drift(pipe, anchors, valid_tokens, prompt_ids)

        generator = torch.Generator(pipe.device).manual_seed(42)
        image = pipe(prompt, height=height, width=width, num_inference_steps=inference_steps, generator=generator).images[0]

        if upscale: image = upscale_image(image)
        if show_drift:
            image = render_overlay(image, drift_text, font_path, label="SEMANTIC DRIFT")
        
        image.save(f"{output_dir}/frame_{i:04d}.jpg", quality=90)
        
        if i % 5 == 0 or i == num_steps:
            status = f"Step {i}/{num_steps}"
            if show_drift: status += f" | Interpretation: {drift_text}"
            print(status)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    print(f"Disintegration complete. Saved to '{output_dir}/'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disintegrate SD with Semantic Readout.")
    parser.add_argument("--target", type=str, default="text_encoder", choices=["unet", "text_encoder", "both"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ratio", type=float, default=0.005)
    parser.add_argument("--percent", type=float, default=0.05)
    parser.add_argument("--prompt", type=str, default="A high-resolution professional photograph of a majestic mountain range at sunrise, cinematic lighting, 8k")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--inference_steps", type=int, default=25)
    parser.add_argument("--show_drift", action="store_true", default=None, help="Force display of semantic drift overlay")
    parser.add_argument("--no_drift", action="store_false", dest="show_drift", help="Force disable semantic drift overlay")
    
    args = parser.parse_args()
    
    run_disintegration(
        target=args.target, 
        num_steps=args.steps, 
        ratio=args.ratio, 
        max_percent=args.percent, 
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        inference_steps=args.inference_steps,
        show_drift=args.show_drift
    )
