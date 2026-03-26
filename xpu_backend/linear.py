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

    Nunchaku weight key mapping:
        qweight     -> self.qweight       (torch.int8, [out_features, in_features//2])
        wscales     -> self.wscales        (bf16, [in_features//64, out_features])
        smooth      -> self.smooth_factor  (bf16, [in_features])
        smooth_orig -> self.smooth_factor_orig (bf16, [in_features])
        lora_down   -> self.proj_down      (bf16, [in_features, rank])
        lora_up     -> self.proj_up        (bf16, [out_features, rank])
        bias        -> self.bias           (bf16, [out_features])
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

        # Quantized weight: [out_features, in_features // 2] int8 (packed INT4)
        self.register_buffer(
            "qweight",
            torch.zeros(out_features, in_features // 2, dtype=torch.int8, device=device),
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
        # Smooth factor for activation (nunchaku key: "smooth")
        self.register_buffer("smooth_factor", None)
        # Smooth factor original (nunchaku key: "smooth_orig")
        self.register_buffer("smooth_factor_orig", None)
        # LoRA projections (nunchaku keys: "lora_down", "lora_up")
        self.register_buffer(
            "proj_down",
            torch.zeros(in_features, rank, dtype=torch_dtype, device=device),
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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to handle nunchaku weight key aliases during load_state_dict.

        Maps:
            smooth      -> smooth_factor
            smooth_orig -> smooth_factor_orig
            lora_down   -> proj_down
            lora_up     -> proj_up

        Also handles the case where smooth_factor / smooth_factor_orig are
        registered as ``None`` buffers: we must re-register them with a real
        tensor *before* the parent ``_load_from_state_dict`` runs, otherwise
        PyTorch skips loading into ``None`` buffers.
        """
        alias_map = {
            "smooth": "smooth_factor",
            "smooth_orig": "smooth_factor_orig",
            "lora_down": "proj_down",
            "lora_up": "proj_up",
        }
        # Remap aliased keys before calling parent
        for alias, canonical in alias_map.items():
            alias_key = prefix + alias
            canonical_key = prefix + canonical
            if alias_key in state_dict and canonical_key not in state_dict:
                state_dict[canonical_key] = state_dict.pop(alias_key)

        # If smooth_factor / smooth_factor_orig are currently None but the
        # state_dict has values for them, pre-register empty buffers so that
        # the parent _load_from_state_dict can assign into them.
        none_buffers = {"smooth_factor", "smooth_factor_orig"}
        for buf_name in none_buffers:
            key = prefix + buf_name
            if key in state_dict and getattr(self, buf_name) is None:
                self.register_buffer(buf_name, torch.empty_like(state_dict[key]))

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _ensure_prepared(self):
        """Prepare weights for oneDNN on first use."""
        if self._qweight_u4 is None and self.qweight is not None:
            # Convert int8 -> uint8 first (nunchaku stores as int8, oneDNN needs uint8)
            qweight_u8 = self.qweight.view(torch.uint8)
            try:
                from omni_xpu_kernel.svdq import prepare_onednn_weights
                self._qweight_u4, self._wscales_f16 = prepare_onednn_weights(
                    qweight_u8, self.wscales
                )
            except ImportError:
                # Fallback: manual conversion (XOR 0x88 for signed->unsigned INT4)
                self._qweight_u4 = (qweight_u8 ^ 0x88).contiguous()
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
            lora_act = x.to(torch.float16) @ self.proj_down.to(torch.float16)

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
        wgt_f16 = dequantize_w4(self.qweight.to(torch.uint8), self.wscales, torch.float16)
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
                lora_act = x.to(torch.float16) @ self.proj_down.to(torch.float16)
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
            weight = dequantize_w4(self.qweight.to(torch.uint8), self.wscales, self.dtype)
        except ImportError:
            weight = self._manual_dequant()

        result = x.to(self.dtype) @ weight.T

        if self.proj_down is not None and self.proj_up is not None:
            lora_act = x.to(self.dtype) @ self.proj_down.to(self.dtype)
            lora_out = lora_act @ self.proj_up.to(self.dtype).T * self.wtscale
            result = result + lora_out

        if self.bias is not None:
            result = result + self.bias

        return result

    def _manual_dequant(self):
        """Manually dequantize INT4 weights (pure PyTorch, no kernels).

        qweight is int8, each byte holds 2 packed int4 values (signed).
        """
        N, half_K = self.qweight.shape
        K = half_K * 2
        group_size = 64

        # Unpack INT4 from int8: low nibble = even indices, high nibble = odd indices
        qw = self.qweight.to(torch.int16)  # widen to avoid overflow
        low = (qw & 0x0F).to(torch.int8)
        high = ((qw >> 4) & 0x0F).to(torch.int8)
        # Convert unsigned nibble to signed: if >= 8, subtract 16
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)

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

    Supports the nunchaku safetensors format where:
        qweight: int32, [out_features//4, in_features//2] - packed INT4 weights
        wscales: bf16, [in_features//group_size, out_features] - per-group scales
        wzeros:  bf16, [in_features//group_size, out_features] - per-group zeros (float)
        bias:    bf16, [out_features]

    Packing: each int32 stores 4 int4 values along out_features, and in_features is
    packed 2:1 (each position stores 2 int4 values), so total 8 int4 per int32.
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

        # AWQ weight: [out_features // 4, in_features // 2] int32
        self.register_buffer(
            "qweight",
            torch.zeros(out_features // 4, in_features // 2, dtype=torch.int32, device=device),
        )
        # Per-group scales: [in_features // group_size, out_features]
        self.register_buffer(
            "wscales",
            torch.zeros(
                in_features // group_size, out_features, dtype=torch_dtype, device=device
            ),
        )
        # Per-group zeros as bf16 float (nunchaku format)
        self.register_buffer(
            "wzeros",
            torch.zeros(
                in_features // group_size, out_features, dtype=torch_dtype, device=device
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros(out_features, dtype=torch_dtype, device=device)
            )
        else:
            self.register_buffer("bias", None)

        self._dequantized_weight = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle legacy key names and shape differences during load."""
        # Handle legacy 'scales' -> 'wscales'
        scales_key = prefix + "scales"
        wscales_key = prefix + "wscales"
        if scales_key in state_dict and wscales_key not in state_dict:
            state_dict[wscales_key] = state_dict.pop(scales_key)

        # Handle legacy int32 'qzeros' -> convert to float 'wzeros'
        qzeros_key = prefix + "qzeros"
        wzeros_key = prefix + "wzeros"
        if qzeros_key in state_dict and wzeros_key not in state_dict:
            qzeros = state_dict.pop(qzeros_key)
            if qzeros.dtype == torch.int32:
                num_groups = qzeros.shape[0]
                N = self.out_features
                zeros = torch.zeros(num_groups, N, dtype=self.dtype, device=qzeros.device)
                for i in range(8):
                    zeros[:, i::8] = ((qzeros >> (i * 4)) & 0xF).to(self.dtype)
                state_dict[wzeros_key] = zeros
            else:
                state_dict[wzeros_key] = qzeros

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _dequantize(self):
        """Dequantize AWQ weights to float. Cached after first call.

        qweight layout: [out_features//4, in_features//2] int32
        Each int32 packs 8 int4 values: 4 along out_features × 2 along in_features.
        Unpacked weight is [out_features, in_features].
        """
        if self._dequantized_weight is not None:
            return self._dequantized_weight

        # qweight: [N//4, K//2] int32, where N=out_features, K=in_features
        N_packed = self.qweight.shape[0]  # out_features // 4
        K_packed = self.qweight.shape[1]  # in_features // 2
        N = N_packed * 4
        K = K_packed * 2

        # Unpack qweight: [N//4, K//2] int32 -> [N, K]
        # Each int32 at [n, k] packs 4 output features × 2 input features:
        #   bits [4*i + 4*2*j] for output 4n+j, input 2k+i
        # Standard AWQ unpack: 8 int4 values per int32, interleaved
        weight = torch.zeros(N, K, dtype=self.dtype, device=self.qweight.device)
        for j in range(4):    # output sub-index
            for i in range(2):  # input sub-index
                shift = (j * 2 + i) * 4
                vals = ((self.qweight >> shift) & 0xF).to(self.dtype)  # [N//4, K//2]
                weight[j::4, i::2] = vals

        # wzeros is already float: [K//group_size, N]
        zeros = self.wzeros.to(self.dtype)

        # Dequantize: weight[n, k] = (int4[n,k] - zeros[g, n]) * scales[g, n]
        # wscales, wzeros: [K//group_size, N]
        num_groups = K // self.group_size
        for g in range(num_groups):
            start = g * self.group_size
            end = start + self.group_size
            weight[:, start:end] = (weight[:, start:end] - zeros[g:g+1].T) * self.wscales[g:g+1].T

        self._dequantized_weight = weight
        return weight

    def invalidate_cache(self):
        """Invalidate cached dequantized weights (call after weight update)."""
        self._dequantized_weight = None

    def forward(self, x):
        """Forward pass: dequantize weights and matmul.

        x: [..., in_features]
        output: [..., out_features]
        """
        weight = self._dequantize()  # [out_features, in_features]
        result = x.to(self.dtype) @ weight.T

        if self.bias is not None:
            result = result + self.bias

        return result
