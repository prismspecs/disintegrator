# Project Disintegrator: Ground Rules

## Objective
The goal of this project is to investigate the internal logic and structural limits of generative models (specifically Stable Diffusion) by programmatically introducing entropy. We learn how these models synthesize reality by observing how that reality unravels.

## Operational Mandates
1. **Empirical Documentation**: Every major run should be accompanied by its measurement data. We do not just make "glitch art"; we measure the rate of decay.
2. **GPU Priority**: This codebase is designed for CUDA-enabled environments. Always utilize the GPU to ensure precision and speed.
3. **High Fidelity**: Disintegration is most revealing at high resolutions. Always aim for the highest resolution the hardware permits.
    - *Default*: 1024x1024 (Note: SDv1.5 may show repetition artifacts at this size).
    - *Optional alternative*: `stabilityai/stable-diffusion-2-1` (Native 768x768 support, providing a more coherent base for high-res decay studies).
4. **Reproducibility**: Every output folder must contain a `data.json` or equivalent metadata file containing the exact seeds, prompts, and disintegration parameters required to reconstruct the results.
5. **Continuous Learning**: Update the README and other documentation as new patterns of disintegration are discovered.
