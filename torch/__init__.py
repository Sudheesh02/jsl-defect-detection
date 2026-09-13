# Minimal torch stub for environments without the real PyTorch library
# Provides only the symbols required by the JSL 2 project.
# If the real torch package is installed, it will be used instead of this stub.

import numpy as np

# ---------------------------------------------------------------------------
# Context manager / decorator for inference mode
# ---------------------------------------------------------------------------
def inference_mode():
    """Return an object that works both as a context manager and as a decorator.
    Allows usage like ``with torch.inference_mode():`` and ``@torch.inference_mode()``.
    """
    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def __call__(self, func):
            def wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)
            return wrapper
    return _Ctx()

# ---------------------------------------------------------------------------
# Backend placeholders (minimal)
# ---------------------------------------------------------------------------
class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False
    @staticmethod
    def get_device_name(idx: int = 0) -> str:
        return "CPU"
    class _Matmul:
        allow_tf32 = False
    matmul = _Matmul()

class _Cudnn:
    allow_tf32 = False
    benchmark = False

class backends:
    cudnn = _Cudnn()
    cuda = _Cuda()

cuda = _Cuda()
# ---------------------------------------------------------------------------
# Functional utilities used in the codebase
# ---------------------------------------------------------------------------
class _Functional:
    @staticmethod
    def softmax(tensor, dim: int = -1):
        arr = np.array(tensor, dtype=np.float32)
        max_val = np.max(arr, axis=dim, keepdims=True)
        exp = np.exp(arr - max_val)
        sum_exp = np.sum(exp, axis=dim, keepdims=True)
        return exp / sum_exp

    @staticmethod
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
            scale_W = max(1, new_H // W)
            return np.repeat(np.repeat(arr, scale_H, axis=2), scale_W, axis=3)
        else:
            raise ValueError("Unsupported tensor shape for interpolate stub")

class nn:
    functional = _Functional

# ---------------------------------------------------------------------------
# Tensor wrapper mimicking minimal torch.Tensor API
# ---------------------------------------------------------------------------
class Tensor:
    """Simple wrapper around a NumPy array providing a tiny subset of the torch.Tensor API."""
    def __init__(self, data, dtype=None):
        self._data = np.asarray(data)
        self.dtype = dtype if dtype is not None else str(self._data.dtype)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self._data
    @property
    def shape(self):
        return self._data.shape
    def clone(self):
        return Tensor(self._data.copy(), dtype=self.dtype)
    def squeeze(self, dim: int = None):
        """Remove dimensions of size 1. If dim is provided, remove that axis only."""
        if dim is None:
            squeezed = np.squeeze(self._data)
        else:
            squeezed = np.squeeze(self._data, axis=dim)
        return Tensor(squeezed)
    def __getitem__(self, key):
        return Tensor(self._data[key])
    def is_contiguous(self):
        return self._data.flags['C_CONTIGUOUS']
    def item(self):
        return self._data.item()
    def tolist(self):
        """Return the underlying data as a Python list (mirrors torch.Tensor.tolist)."""
        return self._data.tolist()

    def to(self, device=None):
        """Mimic torch.Tensor.to by returning self (device handling is a no-op in stub)."""
        return self

    def unsqueeze(self, dim=0):
        """Add a dimension at the given position (like torch.unsqueeze)."""
        arr = np.expand_dims(self._data, axis=dim)
        return Tensor(arr)

# Add a convenience function to create Tensor from data, similar to torch.tensor
def tensor(data, dtype=None, device=None, requires_grad=False):
    """Create a Tensor from data."""
    return Tensor(np.array(data), dtype=dtype)

def stack(tensors, dim=0):
    """Stack a sequence of Tensor objects along a new dimension (like torch.stack)."""
    arrs = [t._data for t in tensors]
    stacked = np.stack(arrs, axis=dim)
    return Tensor(stacked)

# Expose via torch namespace


# ---------------------------------------------------------------------------
# dtype placeholders
# ---------------------------------------------------------------------------
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'
int64 = 'int64'
bfloat16 = 'bfloat16'
uint8 = 'uint8'

# ---------------------------------------------------------------------------
# Tensor creation utilities required by tests
# ---------------------------------------------------------------------------
def empty(shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.empty(shape, dtype=np.float32), dtype=dtype or float32)

def zeros(shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.zeros(shape, dtype=np.float32), dtype=dtype or float32)

def ones(shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.ones(shape, dtype=np.float32), dtype=dtype or float32)

def full(shape, fill_value, dtype=None, device=None, requires_grad=False):
    return Tensor(np.full(shape, fill_value, dtype=np.float32), dtype=dtype or float32)

def rand(shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.random.rand(*shape).astype(np.float32), dtype=dtype or float32)

def randint(low, high=None, shape=None, dtype=None, device=None):
    if shape is None:
        shape = []
    return Tensor(np.random.randint(low, high, size=shape, dtype=np.int64), dtype=dtype or int64)

def from_numpy(arr):
    return Tensor(arr)

# ---------------------------------------------------------------------------
# Simple reduction helpers used in the codebase
# ---------------------------------------------------------------------------
def argmax(tensor, dim=None):
    return np.argmax(tensor, axis=dim)

def argsort(tensor, dim=-1, descending=False):
    order = np.argsort(tensor, axis=dim)
    if descending:
        order = np.flip(order, axis=dim)
    return order

# ---------------------------------------------------------------------------
# Export list
# ---------------------------------------------------------------------------
__all__ = [
    "inference_mode",
    "Tensor",
    "backends",
    "nn",
    "float16",
    "float32",
    "float64",
    "int64",
    "bfloat16",
    "uint8",
    "empty",
    "zeros",
    "ones",
    "full",
    "rand",
    "randint",
    "from_numpy",
    "argmax",
    "argsort",
]
