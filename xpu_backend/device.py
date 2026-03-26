"""
Device utilities for Intel XPU.

Replaces nunchaku.utils CUDA-specific functions with XPU equivalents.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def get_torch_device():
    """Get the best available torch device (prefers XPU)."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_device_count():
    """Get number of available accelerator devices."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.device_count()
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_device(device_id=0):
    """Get a torch.device for the given device ID."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device(f"xpu:{device_id}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def get_device_properties(device_id=0):
    """Get device properties (memory, name, etc.)."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(device_id)
        return props
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(device_id)
    return None


def get_device_name(device_id=0):
    """Get device name string."""
    props = get_device_properties(device_id)
    if props is not None:
        return props.name
    return "cpu"


def get_gpu_memory(device=None, unit="GiB"):
    """Get total GPU memory in the specified unit."""
    divisor = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[unit]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        if device is None:
            device = "xpu"
        if isinstance(device, str):
            device = torch.device(device)
        idx = device.index if device.index is not None else 0
        props = torch.xpu.get_device_properties(idx)
        return props.total_memory / divisor
    if torch.cuda.is_available():
        if device is None:
            device = "cuda"
        if isinstance(device, str):
            device = torch.device(device)
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        return props.total_mem / divisor
    return 0


def empty_cache():
    """Clear GPU memory cache."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def current_stream(device=None):
    """Get the current compute stream."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.current_stream(device)
    if torch.cuda.is_available():
        return torch.cuda.current_stream(device)
    return None


def stream_context(stream):
    """Create a stream context manager."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.stream(stream)
    if torch.cuda.is_available():
        return torch.cuda.stream(stream)
    import contextlib
    return contextlib.nullcontext()


def synchronize(device=None):
    """Synchronize the current device."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize(device)
    elif torch.cuda.is_available():
        torch.cuda.synchronize(device)


def is_turing(device=None):
    """Check if the device is an NVIDIA Turing GPU. Always False on XPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return False
    if torch.cuda.is_available():
        try:
            from nunchaku.utils import is_turing as _is_turing
            return _is_turing(device)
        except ImportError:
            if device is None:
                device = "cuda"
            cap = torch.cuda.get_device_capability(device)
            return cap[0] == 7 and cap[1] == 5
    return False


def get_precision(precision="auto", device=None):
    """Get the quantization precision for the current device."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        # Intel XPU uses int4 precision
        return "int4"
    if torch.cuda.is_available():
        if device is None:
            device = "cuda"
        cap = torch.cuda.get_device_capability(device)
        sm = f"{cap[0]}{cap[1]}"
        return "fp4" if sm == "120" else "int4"
    return "int4"


def check_hardware_compatibility(quantization_config, device=None):
    """Check if the hardware is compatible with the quantization config.

    On XPU, we accept int4 precision. On CUDA, defer to nunchaku's check.
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        precision = quantization_config.get("quant_type", "int4")
        if precision == "nvfp4":
            logger.warning(
                "nvfp4 precision is NVIDIA-specific. On Intel XPU, int4 will be used instead. "
                "Model quality may differ slightly."
            )
        return  # XPU is compatible with int4
    # On CUDA, use nunchaku's check
    try:
        from nunchaku.utils import check_hardware_compatibility as _check
        _check(quantization_config, device)
    except ImportError:
        pass


def get_precision_from_quantization_config(quantization_config):
    """Extract precision from quantization config, adapting for XPU if needed."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        precision = quantization_config.get("quant_type", "int4")
        # Map NVIDIA-specific precisions to XPU equivalents
        if precision == "nvfp4":
            return "int4"
        return precision
    try:
        from nunchaku.utils import get_precision_from_quantization_config as _get
        return _get(quantization_config)
    except ImportError:
        return quantization_config.get("quant_type", "int4")
