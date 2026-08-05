"""Modelbending: perturbing a trained network's weights so it loses its bearings.

The network is left structurally intact -- no parameters removed, no layers
disabled, every step still executed -- but the values encoding which direction
leads toward a coherent output are displaced.

Shared by disintegrate_sd.py, batch_disintegrate.py and recreate_outputs.py so
the bending logic exists in exactly one place.
"""

import math
import torch

DISTRIBUTIONS = ("uniform", "gaussian")


def make_generator(seed, device):
    """Generator for reproducible bends. Must sit on the device being bent."""
    return torch.Generator(device=device).manual_seed(seed)


def _draw_shifts(shape, device, dtype, delta, dist, generator):
    """Draw zero-centered perturbations.

    Both distributions carry the same variance, delta**2 / 3, so `dist` changes
    the *shape* of the damage rather than its overall magnitude. Uniform spreads
    an even drift with a hard bound at +/-delta: a shift of 0.9*delta is exactly
    as likely as one of 0.1*delta, and nothing ever exceeds delta. The gaussian
    tail is unbounded -- mostly smaller shifts, occasionally far larger ones --
    so for the same variance it concentrates its damage instead of spreading it.
    """
    if dist == "uniform":
        return torch.empty(shape, device=device, dtype=dtype).uniform_(
            -delta, delta, generator=generator
        )
    if dist == "gaussian":
        sigma = delta / math.sqrt(3.0)  # variance-matched to uniform(-delta, delta)
        return torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        ) * sigma
    raise ValueError(f"unknown distribution {dist!r}; expected one of {DISTRIBUTIONS}")


def modelbend(module, ratio, max_percent, dist="uniform", generator=None,
              include_buffers=True):
    """Bend `module` in place, perturbing a random fraction of its values.

    ratio            fraction of values perturbed on this pass. Note that passes
                     compound: after N passes the fraction touched at least once
                     is 1 - (1 - ratio)**N, not ratio.
    max_percent      perturbation scale as a fraction of each tensor's own range.
    dist             "uniform" or "gaussian", variance-matched (see _draw_shifts).
    generator        torch.Generator for reproducible bends, on the same device
                     as the module. None uses the global RNG and is not
                     reproducible.
    include_buffers  also bend floating-point buffers (running statistics etc).
    """
    if dist not in DISTRIBUTIONS:
        raise ValueError(f"unknown distribution {dist!r}; expected one of {DISTRIBUTIONS}")

    try:
        if next(module.parameters()).device.type == "meta":
            return
    except StopIteration:
        return

    def bend(tensor):
        t_range = (tensor.max() - tensor.min()).item()
        if t_range <= 1e-12:
            return
        delta = max_percent * t_range
        mask = torch.rand(tensor.shape, device=tensor.device, generator=generator) < ratio
        shifts = _draw_shifts(
            tensor.shape, tensor.device, tensor.dtype, delta, dist, generator
        )
        tensor[mask] += shifts[mask]

    with torch.no_grad():
        for _, param in module.named_parameters():
            bend(param.data)
        if include_buffers:
            for _, buf in module.named_buffers():
                if torch.is_floating_point(buf):
                    bend(buf)


# Previous name, kept so older scripts and notebooks keep working.
degrade_all_tensors = modelbend
