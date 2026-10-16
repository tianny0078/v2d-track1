"""check_tensor, transcribed from kaolin.utils.testing so the assertion still fires."""
from __future__ import annotations
import torch


def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
    def fail(msg):
        if throw:
            raise ValueError(msg)
        return False
    if not torch.is_tensor(tensor):
        return fail(f"expected a torch.Tensor, got {type(tensor)}")
    if shape is not None:
        if len(tensor.shape) != len(shape):
            return fail(f"expected {len(shape)} dims, got {len(tensor.shape)} ({tuple(tensor.shape)})")
        for i, (got, want) in enumerate(zip(tensor.shape, shape)):
            if want is not None and got != want:
                return fail(f"dim {i}: expected {want}, got {got} (shape {tuple(tensor.shape)})")
    if dtype is not None and tensor.dtype != dtype:
        return fail(f"expected dtype {dtype}, got {tensor.dtype}")
    if device is not None and torch.device(device).type != tensor.device.type:
        return fail(f"expected device {device}, got {tensor.device}")
    return True
