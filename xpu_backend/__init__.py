"""
XPU Backend for ComfyUI-nunchaku.

Provides Intel XPU implementations of nunchaku's CUDA operations using omni_xpu_kernel.
Detects available backend (XPU via omni_xpu_kernel, or CUDA via nunchaku) at import time.
"""

import logging
import torch

logger = logging.getLogger(__name__)

# Backend detection
_backend = None  # "xpu", "cuda", or None


def _detect_backend():
    global _backend
    if _backend is not None:
        return _backend

    # Try XPU first
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            import omni_xpu_kernel

            if omni_xpu_kernel.is_available():
                _backend = "xpu"
                logger.info("XPU backend detected: using omni_xpu_kernel")
                return _backend
        except ImportError:
            pass

    # Fallback to CUDA
    if torch.cuda.is_available():
        try:
            import nunchaku  # noqa: F401

            _backend = "cuda"
            logger.info("CUDA backend detected: using nunchaku")
            return _backend
        except ImportError:
            pass

    _backend = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    logger.warning(f"No GPU kernel library found, backend set to: {_backend}")
    return _backend


def get_backend():
    """Return the detected backend: 'xpu', 'cuda', or 'cpu'."""
    return _detect_backend()


def is_xpu():
    """Return True if running on Intel XPU."""
    return get_backend() == "xpu"


def is_cuda():
    """Return True if running on CUDA."""
    return get_backend() == "cuda"
