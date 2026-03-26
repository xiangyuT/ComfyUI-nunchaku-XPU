"""
XPU-compatible quantized linear layers.

Provides SVDQW4A4Linear and AWQW4A16Linear that use omni_xpu_kernel
instead of nunchaku's CUDA kernels.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import is_xpu

logger = logging.getLogger(__name__)


class SVDQW4A4Linear(nn.Module):
    """SVDQuant W4A4 quantized linear layer for Intel XPU.

    Uses omni_xpu_kernel.svdq for INT4 weight GEMM on XPU.
    Maintains the same interface as nunchaku.models.linear.SVDQW4A4Linear.
    """

    def __init__(
        self,
        in_features,
        out_features,
        rank=32,
        bias=True,
        precision="int4",
        act_unsigned=False,
        torch_dtype=torch.bfloat16,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.precision = precision
        self.act_unsigned = act_unsigned
        self.dtype = torch_dtype

        # Quantized weight: [out_features, in_features // 2] uint8
        self.register_buffer(
            "qweight",
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8, device=device),
        )
        # Per-group weight scales: [num_groups, out_features]
        num_groups = in_features // 64  # group_size = 64
        self.register_buffer(
            "wscales",
            torch.zeros(num_groups, out_features, dtype=torch_dtype, device=device),
        )
        # Optional per-channel scales
        self.register_buffer("wcscales", None)
        # Weight-LoRA scale
        self.register_buffer(
            "wtscale", torch.ones(1, dtype=torch_dtype, device=device)
        )
        # Smooth factor for activation
        self.register_buffer("smooth_factor", None)
        self.register_buffer("smooth_factor_orig", None)
        # LoRA projections (packed format)
        self.register_buffer(
            "proj_down",
            torch.zeros(rank, in_features, dtype=torch_dtype, device=device),
        )
        self.register_buffer(
            "proj_up",
            torch.zeros(out_features, rank, dtype=torch_dtype, device=device),
        )
        # Bias
        if bias:
            self.register_buffer(
                "bias", torch.zeros(out_features, dtype=torch_dtype, device=device)
            )
        else:
            self.register_buffer("bias", None)

        # Pre-converted weights for oneDNN (populated on first forward or explicitly)
        self._qweight_u4 = None
        self._wscales_f16 = None
        self._rcp_smooth = None

    def _ensure_prepared(self):
        """Prepare weights for oneDNN on first use."""
        if self._qweight_u4 is None and self.qweight is not None:
            try:
                from omni_xpu_kernel.svdq import prepare_onednn_weights
                self._qweight_u4, self._wscales_f16 = prepare_onednn_weights(
                    self.qweight, self.wscales
                )
            except ImportError:
                # Fallback: manual conversion
                self._qweight_u4 = (self.qweight ^ 0x88).contiguous()
                self._wscales_f16 = self.wscales.to(torch.float16).contiguous()
        if self._rcp_smooth is None and self.smooth_factor is not None:
            self._rcp_smooth = (1.0 / self.smooth_factor.float()).to(torch.float16)

    def prepare_weights(self):
        """Explicitly prepare weights for optimized inference. Call after loading."""
        self._qweight_u4 = None
        self._wscales_f16 = None
        self._rcp_smooth = None
        self._ensure_prepared()

    @classmethod
    def from_linear(cls, linear, **kwargs):
        """Create SVDQW4A4Linear from a standard nn.Linear (for model patching).

        The actual quantized weights should be loaded separately from checkpoint.
        """
        rank = kwargs.pop("rank", 32)
        precision = kwargs.pop("precision", "int4")
        torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
        device = linear.weight.device

        m = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            precision=precision,
            torch_dtype=torch_dtype,
            device=device,
        )
        return m

    def quantize(self, x, pad_size=256):
        """Quantize activations (XPU-compatible).

        Returns (quantized_act, ascales, lora_act) compatible with forward_quant.
        """
        self._ensure_prepared()
        M, K = x.shape

        # Apply smoothing
        if self.smooth_factor is not None and self._rcp_smooth is not None:
            try:
                from omni_xpu_kernel.svdq import fused_smooth_mul_convert
                x_f16 = fused_smooth_mul_convert(x, self._rcp_smooth)
            except ImportError:
                x_f16 = (x.float() * self._rcp_smooth.float()).to(torch.float16)
        else:
            x_f16 = x.to(torch.float16)

        # Pad to multiple of pad_size
        padded_M = ((M + pad_size - 1) // pad_size) * pad_size
        if padded_M > M:
            x_padded = torch.zeros(padded_M, K, dtype=torch.float16, device=x.device)
            x_padded[:M] = x_f16
        else:
            x_padded = x_f16.contiguous()

        # Quantize activations to INT4
        try:
            from omni_xpu_kernel.svdq import quantize_act_int4
            packed_act, ascales = quantize_act_int4(x_padded, group_size=64)
        except ImportError:
            # Fallback: keep as f16 (no actual quantization)
            packed_act = x_padded
            ascales = torch.ones(K // 64, padded_M, dtype=torch.float16, device=x.device)

        # LoRA down projection
        lora_act = None
        if self.proj_down is not None:
            lora_act = x.to(torch.float16) @ self.proj_down.to(torch.float16).T

        return packed_act, ascales, lora_act

    def forward_quant(self, quantized_x, ascales, lora_act, output=None):
        """Run quantized GEMM with LoRA correction."""
        self._ensure_prepared()

        M = quantized_x.shape[0]
        try:
            from omni_xpu_kernel import svdq

            # Dequantize activations back to f16 for oneDNN GEMM
            # quantized_x: [M, K/2] uint8, ascales: [num_groups, M]
            act_f16 = svdq.dequantize_w4(quantized_x, ascales.T, torch.float16)

            # W4A(f16) GEMM
            result = svdq.onednn_int4_gemm_preconverted(
                act_f16, self._qweight_u4, self._wscales_f16
            )
        except (ImportError, RuntimeError):
            # Fallback: dequantize weights and do standard matmul
            result = self._fallback_forward(quantized_x, ascales)

        # LoRA correction
        if lora_act is not None and self.proj_up is not None:
            lora_out = lora_act @ self.proj_up.to(torch.float16).T
            lora_out = lora_out * self.wtscale.to(torch.float16)
            result = result + lora_out

        # Channel-wise scales
        if self.wcscales is not None:
            result = result * self.wcscales.to(result.dtype)

        # Bias
        if self.bias is not None:
            result = result + self.bias.to(result.dtype)

        if output is not None:
            output.copy_(result[:output.shape[0], :output.shape[1]].to(output.dtype))
            return output
        return result.to(self.dtype)

    def _fallback_forward(self, quantized_x, ascales):
        """Fallback: dequantize everything and use standard matmul."""
        from omni_xpu_kernel.svdq import dequantize_w4
        act_f16 = dequantize_w4(quantized_x, ascales.T, torch.float16)
        wgt_f16 = dequantize_w4(self.qweight, self.wscales, torch.float16)
        return act_f16 @ wgt_f16.T

    def forward(self, x):
        """Forward pass through the quantized linear layer."""
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)

        self._ensure_prepared()

        try:
            from omni_xpu_kernel import svdq

            # Apply smoothing
            if self.smooth_factor is not None and self._rcp_smooth is not None:
                x_f16 = svdq.fused_smooth_mul_convert(x, self._rcp_smooth)
            else:
                x_f16 = x.to(torch.float16)

            # Direct W4A(f16) GEMM - skip activation quantization for efficiency
            result = svdq.onednn_int4_gemm_preconverted(
                x_f16, self._qweight_u4, self._wscales_f16
            )

            # LoRA correction
            if self.proj_down is not None and self.proj_up is not None:
                lora_act = x.to(torch.float16) @ self.proj_down.to(torch.float16).T
                lora_out = lora_act @ self.proj_up.to(torch.float16).T
                lora_out = lora_out * self.wtscale.to(torch.float16)
                result = result + lora_out

            # Channel-wise scales
            if self.wcscales is not None:
                result = result * self.wcscales.to(result.dtype)

            # Bias
            if self.bias is not None:
                result = result + self.bias.to(result.dtype)

            result = result.to(self.dtype)

        except (ImportError, RuntimeError) as e:
            logger.warning(f"XPU GEMM failed, using dequant fallback: {e}")
            result = self._dequant_fallback_forward(x)

        return result.reshape(*orig_shape[:-1], self.out_features)

    def _dequant_fallback_forward(self, x):
        """Full dequantization fallback (slow but always works)."""
        try:
            from omni_xpu_kernel.svdq import dequantize_w4
            weight = dequantize_w4(self.qweight, self.wscales, self.dtype)
        except ImportError:
            weight = self._manual_dequant()

        result = x.to(self.dtype) @ weight.T

        if self.proj_down is not None and self.proj_up is not None:
            lora_act = x.to(self.dtype) @ self.proj_down.to(self.dtype).T
            lora_out = lora_act @ self.proj_up.to(self.dtype).T * self.wtscale
            result = result + lora_out

        if self.bias is not None:
            result = result + self.bias

        return result

    def _manual_dequant(self):
        """Manually dequantize INT4 weights (pure PyTorch, no kernels)."""
        N, half_K = self.qweight.shape
        K = half_K * 2
        group_size = 64

        # Unpack INT4: low nibble = even indices, high nibble = odd indices
        low = (self.qweight & 0x0F).to(torch.int8) - 8  # signed
        high = ((self.qweight >> 4) & 0x0F).to(torch.int8) - 8

        # Interleave
        weight_int = torch.stack([low, high], dim=-1).reshape(N, K)

        # Apply per-group scales
        weight_float = weight_int.to(self.dtype)
        num_groups = K // group_size
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            weight_float[:, start:end] = weight_float[:, start:end] * self.wscales[g:g+1]

        return weight_float


class AWQW4A16Linear(nn.Module):
    """AWQ W4A16 quantized linear layer for Intel XPU.

    Dequantizes INT4 weights to float and performs standard matmul.
    On CUDA, nunchaku uses optimized AWQ GEMV kernels.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        group_size=64,
        torch_dtype=torch.bfloat16,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = torch_dtype

        # AWQ weight format
        self.register_buffer(
            "qweight",
            torch.zeros(in_features // 8, out_features, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                in_features // group_size, out_features, dtype=torch_dtype, device=device
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                in_features // group_size, out_features // 8, dtype=torch.int32, device=device
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros(out_features, dtype=torch_dtype, device=device)
            )
        else:
            self.register_buffer("bias", None)

        self._dequantized_weight = None

    def _dequantize(self):
        """Dequantize AWQ weights to float. Cached after first call."""
        if self._dequantized_weight is not None:
            return self._dequantized_weight

        K = self.in_features
        N = self.out_features

        # Unpack qweight: int32 -> 8 x int4
        weight = torch.zeros(K, N, dtype=self.dtype, device=self.qweight.device)
        for i in range(8):
            weight[i::8] = ((self.qweight >> (i * 4)) & 0xF).to(self.dtype)

        # Unpack qzeros
        num_groups = K // self.group_size
        zeros = torch.zeros(num_groups, N, dtype=self.dtype, device=self.qzeros.device)
        for i in range(8):
            zeros[:, i::8] = ((self.qzeros >> (i * 4)) & 0xF).to(self.dtype)

        # Dequantize: weight = (int4 - zero) * scale
        for g in range(num_groups):
            start = g * self.group_size
            end = start + self.group_size
            weight[start:end] = (weight[start:end] - zeros[g:g+1]) * self.scales[g:g+1]

        self._dequantized_weight = weight
        return weight

    def invalidate_cache(self):
        """Invalidate cached dequantized weights (call after weight update)."""
        self._dequantized_weight = None

    def forward(self, x):
        """Forward pass: dequantize weights and matmul."""
        weight = self._dequantize()
        result = x.to(self.dtype) @ weight

        if self.bias is not None:
            result = result + self.bias

        return result
