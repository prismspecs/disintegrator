# Disintegrator

A suite of tools to programmatically degrade generative models, specifically Stable Diffusion, exploring the aesthetics and politics of technological entropy.

## Overview

Based on the theory "In Defense of Disintegration" by Grayson Earle, this project introduces controlled randomness into a model's weights and buffers to simulate decay. As the model's parameters are corrupted, its capacity to synthesize coherent images erodes, transitioning from high-fidelity representation to abstract, incoherent fragments.

This repository provides multiple methods for exploring this "digital dementia," targeting different components of the diffusion pipeline.

## Core Disintegration Methods

### 1. Structural Disintegration (U-Net)
By targeting the **U-Net**, the project degrades the model's ability to denoise and structure visual information. 
- **Aesthetic:** Results in "glitch" artifacts, structural collapse, and eventually, pure chromatic noise.
- **Mechanism:** Random noise is injected into the convolutional layers and attention blocks of the U-Net between generation runs.

### 2. Semantic Disintegration (Text Encoder)
By targeting the **CLIP Text Encoder**, the project explores the breakdown of meaning and language.
- **Aesthetic:** The model begins to misinterpret the prompt, drifting into unrelated conceptual territories while maintaining some structural coherence (as the U-Net remains intact).
- **Semantic Drift:** This method features a "Semantic Readout." The script maps the corrupted internal states of the encoder back to a "clean" vocabulary, allowing us to see how the model's "understanding" of a word like "Mountain" might drift into "FLUID" or "VIBRATION" as it decays.

### 3. Smooth Disintegration (Motion)
For video work, structural decay is driven along a **single fixed direction** in weight space
rather than re-randomized each frame. See `smooth_disintegrate.py` below.

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch diffusers transformers accelerate hf_transfer
```

## Usage

### Main Disintegration Script (`disintegrate_sd.py`)

Run the progressive disintegration loop:

```bash
# Target the U-Net (Structural decay)
python3 disintegrate_sd.py --target unet --steps 50 --ratio 0.01

# Target the Text Encoder (Semantic drift)
python3 disintegrate_sd.py --target text_encoder --steps 100 --ratio 0.005

# Target Both simultaneously
python3 disintegrate_sd.py --target both --steps 50
```

**Key Arguments:**
- `--target`: Choose `unet`, `text_encoder`, or `both`.
- `--steps`: Number of images to generate in the sequence.
- `--ratio`: The percentage of weights to corrupt at each step (e.g., `0.01` for 1%).
- `--percent`: The magnitude of the corruption (how far a weight can shift).
- `--prompt`: The starting concept to be disintegrated.

### Smooth Disintegration Video (`smooth_disintegrate.py`)

Produces a continuous dissolve from clean representation into abstraction, suitable for
projection or screening.

```bash
# 1. Calibrate: render an alpha ladder and pick the collapse point by eye
python3 smooth_disintegrate.py --calibrate --alpha-max 0.6 --calibrate-steps 8

# 2. Render a 60s 720x720 sequence
python3 smooth_disintegrate.py --alpha-max 0.30 --duration 60 --fps 30 \
    --keyframes 450 --size 720

# 3. Re-encode without re-rendering
python3 smooth_disintegrate.py --encode-only renders_<timestamp>
```

**Why it is smooth.** The methods above re-sample the weight perturbation at every step, so
consecutive frames are independent draws and the sequence strobes. This script samples one
perturbation direction `d` and rebuilds the weights each frame as `w = w0 + α·d` from the
pristine `w0`, with a fixed seed and a fixed initial latent. The model then traverses a
continuous path through weight space and the decay reads as a morph rather than a flicker.

**The α curve.** `α(u) = α_max · u^curve_power` (default power 2.2). Early weight damage is
perceptually cheap, so a linear ramp feels front-loaded; the power curve holds the image
legible, then accelerates into collapse.

**Calibration is not optional.** The α that means "fully abstract" moves with prompt,
resolution, and step count. For the mountain prompt at 512/25 steps, representation dies
between α 0.12 and 0.20, and past α≈0.34 the output saturates into flat static that no
longer evolves — the live range ends around 0.30.

### Measurement and Analysis (`measure_disintegration.py`)

The `measure_disintegration.py` script provides a quantitative look at how the model's output diverges as its weights are degraded. It measures the L2 distance between the original noise prediction and the degraded noise prediction at a fixed timestep.

#### Usage:
```bash
# Basic measurement at default 1024x1024
python3 measure_disintegration.py --target unet --steps 100 --ratio 0.01

# Measurement with visual output (saves images with data overlays)
python3 measure_disintegration.py --target unet --steps 100 --ratio 0.005 --interval 5 --save_visuals

# High-resolution run (see model notes below before raising resolution)
python3 measure_disintegration.py --height 768 --width 768 --save_visuals
```

#### Model Selection Notes:
- **SD v1.5 (Default)**: Best for observing local feature breakdown. At the 1024x1024
  default, it exhibits subject duplication, which provides a unique look at how the UNet
  handles repeated structural motifs during decay. Where duplication is *not* wanted,
  render at 512 and resample up rather than generating large.
- **Model id**: use `stable-diffusion-v1-5/stable-diffusion-v1-5`. The original
  `runwayml/...` id dates from before that org was taken down in 2024.
- **SD v2.1-768**: previously recommended here for duplication-free high-res work, but
  `stabilityai/stable-diffusion-2-1` now 404s on the Hub (checked 2026-07-20). Those
  instructions are retained in git history only.

#### Key Metrics:
- **Cumulative Parameter Drift (Damage):** The average absolute change applied to the model's parameters. This represents the total "physical" damage to the network.
- **Output Relative Distance:** The ratio of the norm of the difference between original and degraded noise predictions to the norm of the original prediction. This represents the "behavioral" divergence.

Results are saved as a JSON file for further analysis, including "milestones" that identify at which step the model reaches specific levels of distortion (e.g., "Structural failure" at 50% divergence).

---

## Auxiliary Tools

### Dual Denoising Visualization (`dual_denoising.py`)
Visualizes the "branching" of reality from a single point of origin. This script generates two different prompts starting from the *exact same* initial Gaussian noise, showing how the model carves different meanings out of the same entropy.

```bash
python3 dual_denoising.py --p1 "A mountain range" --p2 "A cyberpunk city"
```

### Denoising Step Visualization (`visualize_denoising.py`)
Captures the internal state of the model at every step of the diffusion process, allowing you to see how an image "precipitates" out of the noise.

```bash
python3 visualize_denoising.py --prompt "A majestic forest" --steps 50
```

---

## Theoretical Background: In Defense of Disintegration

This project is an artistic exploration of **Technological Entropy**. In the rush to create "perfect" generative models that are increasingly high-fidelity and "aligned" with human intent, we often ignore the inherent fragility and materiality of these systems.

By forcing a model into a state of decay, we reveal its internal architecture and the "latent biases" that usually remain hidden.
- **Structural decay (U-Net)** shows us how the model "hallucinates" form from noise.
- **Semantic decay (Text Encoder)** shows us how the model "conceptually" maps the world, and how fragile the link between a word (e.g., "Peace") and its mathematical representation really is.

Through disintegration, we find a new aesthetic that is not defined by the model's success, but by its spectacular failure.

---

## Technical Note: Safety Checker

The Stable Diffusion `SafetyChecker` is disabled by default. As the model disintegrates, it frequently produces abstract noise patterns that trigger false-positives in the safety filter, resulting in blacked-out frames. To observe the raw aesthetic of the disintegration, the filter is bypassed.
