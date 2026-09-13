# Minimal functional module for torch.nn

# Provide softmax and interpolate directly to avoid circular imports.
import numpy as np

def softmax(tensor, dim: int = -1):
    """Compute softmax for a Tensor stub or NumPy array.
    If `tensor` is an instance of the stub ``torch.Tensor``, we operate on its
    underlying ``_data`` attribute (a NumPy array). Otherwise we coerce the
    input to a NumPy array as before.
    """
    # Support our minimal Tensor stub
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            arr = tensor._data.astype(np.float32)
        else:
            arr = np.array(tensor, dtype=np.float32)
    except Exception:
        arr = np.array(tensor, dtype=np.float32)
    max_val = np.max(arr, axis=dim, keepdims=True)
    exp = np.exp(arr - max_val)
    sum_exp = np.sum(exp, axis=dim, keepdims=True)
    return exp / sum_exp


def interpolate(tensor, size, mode='bilinear', align_corners=False):
    arr = np.array(tensor)
    if arr.ndim == 3:
        C, H, W = arr.shape
        new_H, new_W = size
        scale_H = max(1, new_H // H)
        scale_W = max(1, new_W // W)
        return np.repeat(np.repeat(arr, scale_H, axis=1), scale_W, axis=2)
    elif arr.ndim == 4:
        N, C, H, W = arr.shape
        new_H, new_W = size
        scale_H = max(1, new_H // H)
        scale_W = max(1, new_W // W)
        return np.repeat(np.repeat(arr, scale_H, axis=2), scale_W, axis=3)
    else:
        raise ValueError("Unsupported tensor shape for interpolate stub")

__all__ = ["softmax", "interpolate"]
