"""Small shared helpers."""

from __future__ import annotations

import os

import torch


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve a device and fail loudly on an invalid CUDA_VISIBLE_DEVICES.

    Without the guard, an out-of-range GPU index makes
    ``torch.cuda.is_available()`` false and the experiment silently runs on
    CPU (effectively hanging on a 3090 workload).
    """
    if device:
        return torch.device(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if visible not in (None, "", "-1"):
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES={visible!r} is set but CUDA is not "
            f"available. Check `nvidia-smi -L` and use a valid index, or "
            f"unset it for CPU.")
    return torch.device("cpu")
