"""
XPU-compatible CPU offload manager.

Replaces nunchaku.models.utils.CPUOffloadManager with XPU stream support.
"""

import logging

import torch
import torch.nn as nn

from .device import current_stream, empty_cache, stream_context

logger = logging.getLogger(__name__)


class CPUOffloadManager:
    """Manages CPU offloading of transformer blocks for memory-constrained XPU/CUDA inference.

    Maintains a sliding window of blocks on GPU, pre-fetching the next block
    while the current block is computing.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        use_pin_memory: bool = True,
        on_gpu_modules: list = None,
        num_blocks_on_gpu: int = 1,
    ):
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.num_blocks_on_gpu = num_blocks_on_gpu
        self.use_pin_memory = use_pin_memory
        self.on_gpu_modules = on_gpu_modules or []
        self.device = None
        self.transfer_stream = None
        self._current_block_idx = 0

        # Move blocks to CPU, pin memory if requested
        for block in self.blocks:
            block.to("cpu")
            if use_pin_memory:
                for param in block.parameters():
                    if param.data.is_contiguous():
                        param.data = param.data.pin_memory()
                for buf in block.buffers():
                    if buf.is_contiguous():
                        try:
                            buf.data = buf.data.pin_memory()
                        except RuntimeError:
                            pass

    def set_device(self, device):
        """Set the target compute device."""
        self.device = device

        # Move non-block modules to GPU
        for m in self.on_gpu_modules:
            m.to(device)

        # Pre-load first few blocks
        for i in range(min(self.num_blocks_on_gpu, self.num_blocks)):
            self.blocks[i].to(device, non_blocking=True)

    def initialize(self, compute_stream):
        """Initialize offloading with the given compute stream."""
        self._current_block_idx = 0

        # Create transfer stream for async prefetch
        device = self.device
        if device is not None and hasattr(torch, "xpu") and device.type == "xpu":
            self.transfer_stream = torch.xpu.Stream(device)
        elif device is not None and device.type == "cuda":
            self.transfer_stream = torch.cuda.Stream(device)
        else:
            self.transfer_stream = None

    def get_block(self, idx):
        """Get block at index, ensuring it's on the compute device."""
        block = self.blocks[idx]
        if next(block.parameters()).device != self.device:
            block.to(self.device, non_blocking=True)
            if hasattr(torch, "xpu") and self.device.type == "xpu":
                torch.xpu.synchronize(self.device)
            elif self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        return block

    def step(self, compute_stream):
        """Called after each block computation. Manages block prefetch and eviction."""
        idx = self._current_block_idx

        # Evict old blocks (keep only num_blocks_on_gpu blocks on GPU)
        evict_idx = idx - self.num_blocks_on_gpu
        if evict_idx >= 0:
            self.blocks[evict_idx].to("cpu", non_blocking=True)

        # Prefetch next block
        prefetch_idx = idx + self.num_blocks_on_gpu
        if prefetch_idx < self.num_blocks:
            if self.transfer_stream is not None:
                with stream_context(self.transfer_stream):
                    self.blocks[prefetch_idx].to(self.device, non_blocking=True)
            else:
                self.blocks[prefetch_idx].to(self.device, non_blocking=True)

        self._current_block_idx = idx + 1
