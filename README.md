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
