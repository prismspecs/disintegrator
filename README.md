# Disintegrator

A tool to programmatically degrade generative models, specifically Stable Diffusion's U-Net, exploring the aesthetics and politics of technological entropy.

## Overview

Based on the theory "In Defense of Disintegration" by Grayson Earle, this script introduces randomness into a model's weights and buffers, simulating decay. Over time, the model's capacity to synthesize coherent images erodes, transitioning from structured outputs to incoherent fragments.

## Usage

1. **Setup:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install torch diffusers transformers accelerate hf_transfer
   ```

2. **Run:**
   ```bash
   python3 disintegrate_sd.py
   ```

## Disintegration Methods

- **Progressive:** Degrades the model between full generation runs.
- **Dynamic:** Degrades the model at each denoising step during a single image generation.
