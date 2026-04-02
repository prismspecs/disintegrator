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

- **Progressive Disintegration:** Degrades the U-Net *between* full generation runs. This simulates a model losing its "memory" or training over time.
- **Dynamic Disintegration:** Degrades the U-Net *during* the diffusion process (at every denoising step). This simulates a model "unraveling" while it is in the middle of forming a single image.

## Note on the Safety Checker

The Stable Diffusion `SafetyChecker` has been disabled in this script. As the model's weights disintegrate, it often produces abstract noise that the safety filter misinterprets as NSFW content, returning black squares. To see the raw aesthetic of the disintegration, the filter is bypassed.
