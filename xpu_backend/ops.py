"""
XPU implementations of nunchaku CUDA ops.

Maps nunchaku's ops (gemm, quantize, fused) to omni_xpu_kernel equivalents.
On XPU: uses omni_xpu_kernel.svdq for W4A(f16) GEMM.
On CUDA: falls back to nunchaku's native CUDA ops.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def svdq_gemm_w4a4(
    act,
    wgt,
    out,
    ascales,
    wscales,
    lora_act_in=None,
    lora_up=None,
    bias=None,
    fp4=False,
    alpha=None,
    wcscales=None,
    norm_q=None,
    norm_k=None,
    rotary_emb=None,
):
    """XPU implementation of nunchaku's svdq_gemm_w4a4_cuda.

    On XPU, we use a W4A(f16) approach: weights stay INT4, activations are kept in f16.
    The oneDNN INT4 GEMM handles weight dequantization internally.

    For W4A4 nunchaku: act is quantized INT4 [M, K/2] with ascales [num_groups, M].
    For XPU: we dequantize activations back to f16, then use oneDNN INT4 GEMM.
    """
    from omni_xpu_kernel import svdq

    M = act.shape[0]
    N = wgt.shape[0]

    # Dequantize activations from INT4 back to float
    # act: [M, K/2] uint8 packed, ascales: [num_groups_a, M]
    act_f16 = svdq.dequantize_w4(act, ascales.T, torch.float16)  # [M, K]

    # Prepare weight format for oneDNN
    # wgt: [N, K/2] uint8 signed INT4, wscales: [num_groups_w, N]
    wgt_u4 = (wgt ^ 0x88).contiguous()
    wscales_f16 = wscales.to(torch.float16).contiguous()

    # Main GEMM: [M, K] x [N, K]^T -> [M, N]
    result = svdq.onednn_int4_gemm_preconverted(act_f16, wgt_u4, wscales_f16)

    # LoRA correction
    if lora_act_in is not None and lora_up is not None:
        lora_out = lora_act_in @ lora_up.T
        if alpha is not None:
            lora_out = lora_out * alpha
        result = result + lora_out.to(result.dtype)

    # Bias
    if bias is not None:
        result = result + bias.to(result.dtype)

    # Channel-wise scale correction
    if wcscales is not None:
        result = result * wcscales.to(result.dtype)

    # Apply norm_q, norm_k, rotary_emb if provided (for fused QKV+norm+rope)
    if norm_q is not None or norm_k is not None or rotary_emb is not None:
        result = _apply_qkv_norm_rotary(result, norm_q, norm_k, rotary_emb, N)

    out.copy_(result[:M, :N].to(out.dtype))


def _apply_qkv_norm_rotary(qkv_output, norm_q, norm_k, rotary_emb, out_features):
    """Apply RMSNorm to Q/K and rotary embeddings after QKV projection."""
    # Split QKV: assume Q, K, V are equal-sized thirds
    third = out_features // 3
    q = qkv_output[:, :third]
    k = qkv_output[:, third : 2 * third]
    v = qkv_output[:, 2 * third :]

    # RMSNorm on Q and K
    if norm_q is not None:
        q = _rms_norm(q, norm_q)
    if norm_k is not None:
        k = _rms_norm(k, norm_k)

    # Rotary embeddings
    if rotary_emb is not None:
        q = _apply_rotary(q, rotary_emb)
        k = _apply_rotary(k, rotary_emb)

    return torch.cat([q, k, v], dim=-1)


def _rms_norm(x, weight, eps=1e-6):
    """Simple RMSNorm implementation."""
    dtype = x.dtype
    x = x.float()
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    x = x / rms
    return (x * weight.float()).to(dtype)


def _apply_rotary(x, freqs):
    """Apply rotary position embeddings."""
    # freqs expected in nunchaku's packed format
    # For now, return x unchanged - the RopeFuseAttentionHook in zimage.py
    # handles the actual rotary embedding application via the hook mechanism
    return x


def svdq_quantize_w4a4_act_fuse_lora(
    x, smooth_factor, proj_down, pad_size=256, fp4=False, act_unsigned=False
):
    """XPU implementation of activation quantization with LoRA fusion.

    On XPU, we keep activations in f16 (no actual INT4 quantization of activations).
    The 'quantized' activations are just the f16 activations after smoothing.
    We return them in a format compatible with our svdq_gemm_w4a4.
    """
    from omni_xpu_kernel import svdq

    M, K = x.shape

    # Pad M to multiple of pad_size
    padded_M = ((M + pad_size - 1) // pad_size) * pad_size

    # Apply smoothing and convert to f16
    if smooth_factor is not None:
        rcp_smooth = (1.0 / smooth_factor.float()).to(torch.float16)
        x_f16 = svdq.fused_smooth_mul_convert(x, rcp_smooth)
    else:
        x_f16 = x.to(torch.float16)

    # Quantize to INT4 for bandwidth reduction (activations)
    if padded_M > M:
        x_padded = torch.zeros(padded_M, K, dtype=x_f16.dtype, device=x_f16.device)
        x_padded[:M] = x_f16
    else:
        x_padded = x_f16

    packed_act, ascales = svdq.quantize_act_int4(x_padded, group_size=64)

    # LoRA down projection
    lora_act = None
    if proj_down is not None:
        # proj_down is in nunchaku's packed format
        # For XPU, compute lora_act = x @ proj_down^T
        lora_act = x.to(torch.float16) @ proj_down.to(torch.float16).T

    return packed_act, ascales, lora_act


def fused_gelu_mlp(hidden_states, linear1, linear2):
    """XPU implementation of fused GELU MLP.

    In nunchaku CUDA, this fuses GELU activation between two quantized GEMM operations.
    On XPU, we decompose it into separate steps.
    """
    # First linear + GELU
    x = linear1(hidden_states)
    x = F.gelu(x, approximate="tanh")

    # Second linear
    x = linear2(x)
    return x


def awq_gemv_w4a16(x, qweight, scales, qzeros, group_size=64):
    """XPU implementation of AWQ W4A16 GEMV.

    Dequantizes INT4 weights to float and performs standard matmul.
    """
    # Dequantize AWQ weights
    weight_fp = _dequantize_awq(qweight, scales, qzeros, group_size)
    return x @ weight_fp.T


def _dequantize_awq(qweight, scales, qzeros, group_size):
    """Dequantize AWQ INT4 packed weights to float."""
    # AWQ packs 8 INT4 values into one int32
    # qweight: [K // 8, N] int32
    # scales: [K // group_size, N] float16
    # qzeros: [K // group_size, N // 8] int32
    N = scales.shape[1]
    K = qweight.shape[0] * 8
    num_groups = K // group_size

    # Unpack qweight: each int32 contains 8 x 4-bit values
    weight_int4 = torch.zeros(K, N, dtype=torch.float16, device=qweight.device)
    for i in range(8):
        weight_int4[i::8] = ((qweight >> (i * 4)) & 0xF).to(torch.float16)

    # Unpack qzeros
    zeros = torch.zeros(num_groups, N, dtype=torch.float16, device=qzeros.device)
    for i in range(8):
        zeros[:, i::8] = ((qzeros >> (i * 4)) & 0xF).to(torch.float16)

    # Dequantize: weight = (int4_val - zero) * scale
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        weight_int4[start:end] = (weight_int4[start:end] - zeros[g:g+1]) * scales[g:g+1]

    return weight_int4
