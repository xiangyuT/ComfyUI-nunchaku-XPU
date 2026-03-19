"""
This module patches the original Z-Image transformer model by injecting into Nunchaku optimized attention and feed forward submodules.

Note
----
The injected transformer model is `comfy.ldm.lumina.model.NextDiT` in ComfyUI repository. Codes are adapted from:
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py#Lumina2
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/lumina/model.py#NextDiT

"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.lumina.model import FeedForward, JointAttention, JointTransformerBlock, NextDiT, clamp_fp16
from comfy.ldm.modules.attention import optimized_attention_masked

from nunchaku_torch.models.embeddings import pack_rotemb
from nunchaku_torch.models.linear import SVDQW4A4Linear
from nunchaku_torch.utils import pad_tensor

# Conditionally import XPU-specific ops
_xpu_available = False
try:
    from nunchaku_torch.ops.xpu_ops import svdq_gemm_w4a4_xpu_bf16act, is_available as xpu_ops_available
    _xpu_available = xpu_ops_available()
except ImportError:
    pass

# Conditionally import ESIMD kernels for XPU norm/rotary
_esimd_norm = None
_esimd_rotary = None
try:
    from omni_xpu_kernel.norm import ESIMDRMSNorm
    from omni_xpu_kernel.rotary import ESIMDRotaryEmb
    _esimd_norm = ESIMDRMSNorm()
    _esimd_rotary = ESIMDRotaryEmb()
except ImportError:
    pass

# Import CUDA-specific ops only if available
_cuda_gemm_available = False
try:
    from nunchaku_torch.ops.gemm import svdq_gemm_w4a4_cuda
    _cuda_gemm_available = True
except ImportError:
    pass


def _is_xpu_device(device: torch.device) -> bool:
    """Check if the device is an XPU device."""
    return device.type == "xpu"


def add_comfy_cast_weights_attr(svdq_linear: SVDQW4A4Linear, comfy_linear: nn.Linear):
    """
    Add dummy `comfy_cast_weights` and `weight`fields to a SVDQW4A4Linear module.

    Make it compatible with offloading mechanism in ModelPatcher class in a lower vram condition.

    Note
    ----
    See method `comfy.model_patcher.ModelPatcher#_load_list` and method `comfy.model_patcher.ModelPatcher#load`
    """
    if hasattr(comfy_linear, "comfy_cast_weights"):
        svdq_linear.comfy_cast_weights = comfy_linear.comfy_cast_weights
        svdq_linear.weight = None


def fuse_to_svdquant_linear(comfy_linear1: nn.Linear, comfy_linear2: nn.Linear, **kwargs) -> SVDQW4A4Linear:
    """
    Fuse two linear modules into one SVDQW4A4Linear.

    The two linear modules MUST have equal `in_features`.

    Parameters
    ----------
    comfy_linear1 : nn.Linear
        linear module 1
    comfy_linear2 : nn.Linear
        linear module 2

    Returns
    -------
    SVDQW4A4Linear:
        The fused module.
    """
    assert comfy_linear1.in_features == comfy_linear2.in_features
    assert comfy_linear1.bias is None and comfy_linear2.bias is None
    torch_dtype = kwargs.pop("torch_dtype", comfy_linear1.weight.dtype)
    svdq_linear = SVDQW4A4Linear(
        comfy_linear1.in_features,
        comfy_linear1.out_features + comfy_linear2.out_features,
        bias=False,
        torch_dtype=torch_dtype,
        device=comfy_linear1.weight.device,
        **kwargs,
    )
    add_comfy_cast_weights_attr(svdq_linear, comfy_linear1)
    return svdq_linear


def fused_qkv_norm_rotary(
    x: torch.Tensor,
    qkv: SVDQW4A4Linear,
    q_norm_weight: nn.Parameter,
    k_norm_weight: nn.Parameter,
    freqs_cis: torch.Tensor,
):
    """
    Fused quantized QKV projection with RMSNorm and rotary embeddings.

    On CUDA: uses fused CUDA kernel (svdq_gemm_w4a4_cuda with norm_q, norm_k, rotary_emb args).
    On XPU: uses bf16act GEMM + separate ESIMD RMSNorm + ESIMD Rotary kernels.

    Adapted from nunchaku.ops.fused#fused_qkv_norm_rottary.

    Manual cast `q_norm` weight and `k_norm` weight from `torch.bfloat16` to `torch.float16` when `x` dtype is `torch.float16`.
    """
    batch_size, seq_len, channels = x.shape
    x_dtype = x.dtype
    device_type = x.device.type

    if device_type == "xpu" and _xpu_available:
        # --- XPU path: bf16act GEMM + separate ESIMD norm/rotary ---
        return _xpu_fused_qkv_norm_rotary(x, qkv, q_norm_weight, k_norm_weight, freqs_cis)

    # --- CUDA path: fused kernel ---
    assert _cuda_gemm_available, "CUDA GEMM kernel not available"
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = qkv.quantize(x)
    output = torch.empty(batch_size * seq_len, qkv.out_features, dtype=x.dtype, device=x.device)
    if (q_norm_weight is not None) and (x_dtype != q_norm_weight.dtype):
        assert x_dtype == torch.float16
        assert q_norm_weight.dtype == torch.bfloat16
        assert k_norm_weight.dtype == torch.bfloat16
        q_norm_weight = torch.nan_to_num(q_norm_weight.to(dtype=torch.float16), nan=0.0, posinf=65504, neginf=-65504)
        k_norm_weight = torch.nan_to_num(k_norm_weight.to(dtype=torch.float16), nan=0.0, posinf=65504, neginf=-65504)
    svdq_gemm_w4a4_cuda(
        act=quantized_x,
        wgt=qkv.qweight,
        out=output,
        ascales=ascales,
        wscales=qkv.wscales,
        lora_act_in=lora_act,
        lora_up=qkv.proj_up,
        bias=qkv.bias,
        fp4=qkv.precision == "nvfp4",
        alpha=qkv.wtscale,
        wcscales=qkv.wcscales,
        norm_q=q_norm_weight if q_norm_weight is not None else None,
        norm_k=k_norm_weight if k_norm_weight is not None else None,
        rotary_emb=freqs_cis,
    )
    output = output.view(batch_size, seq_len, -1)
    return output


def _xpu_fused_qkv_norm_rotary(
    x: torch.Tensor,
    qkv: SVDQW4A4Linear,
    q_norm_weight: nn.Parameter,
    k_norm_weight: nn.Parameter,
    freqs_cis: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    XPU path for QKV projection + RMSNorm + Rotary.

    Uses bf16act GEMM (oneDNN INT4) + separate ESIMD RMSNorm + ESIMD Rotary kernels.
    Adapted from NunchakuZImageFusedModule.forward() in nunchaku_torch.
    """
    batch_size, seq_len, channels = x.shape
    x_2d = x.view(batch_size * seq_len, channels)

    output = torch.empty(batch_size * seq_len, qkv.out_features, dtype=x.dtype, device=x.device)

    # Prepare oneDNN weights (lazy cache)
    if not hasattr(qkv, '_wgt_u4'):
        try:
            from omni_xpu_kernel.svdq import prepare_onednn_weights
            u4, sf16 = prepare_onednn_weights(qkv.qweight.view(torch.uint8), qkv.wscales)
            qkv._wgt_u4 = u4
            qkv._wscales_f16 = sf16
            # Free original weight storage
            qkv.qweight.data = torch.empty(0, dtype=torch.int8, device=x.device)
            qkv.wscales.data = torch.empty(0, dtype=qkv.wscales.dtype, device=x.device)
        except (ImportError, AttributeError):
            qkv._wgt_u4 = None
            qkv._wscales_f16 = None

    # Step 1: LoRA down-proj + smooth + bf16act GEMM
    lora_act_out = x_2d @ qkv.proj_down
    act_smoothed = x_2d / qkv.smooth_factor
    svdq_gemm_w4a4_xpu_bf16act(
        act_bf16=act_smoothed,
        wgt=qkv.qweight,
        wscales=qkv.wscales,
        out=output,
        lora_act_in=lora_act_out,
        lora_up=qkv.proj_up,
        bias=qkv.bias,
        alpha=1.0 if qkv.precision == "nvfp4" else None,
        wcscales=qkv.wcscales if qkv.precision == "nvfp4" else None,
        wgt_u4=getattr(qkv, '_wgt_u4', None),
        wscales_f16=getattr(qkv, '_wscales_f16', None),
    )

    output = output.view(batch_size, seq_len, -1)

    # Step 2: Split QKV
    query, key, value = output.chunk(3, dim=-1)
    head_dim = int(q_norm_weight.numel())
    heads = query.shape[-1] // head_dim

    query = query.view(batch_size, seq_len, heads, head_dim)
    key = key.view(batch_size, seq_len, heads, head_dim)

    # Step 3: ESIMD RMSNorm
    if _esimd_norm is not None:
        flat_q = query.reshape(-1, head_dim).contiguous()
        flat_k = key.reshape(-1, head_dim).contiguous()
        query = _esimd_norm.rms_norm(q_norm_weight, flat_q, 1e-6).reshape(batch_size, seq_len, heads, head_dim)
        key = _esimd_norm.rms_norm(k_norm_weight, flat_k, 1e-6).reshape(batch_size, seq_len, heads, head_dim)
    else:
        # Fallback: PyTorch RMSNorm
        def _apply_rmsnorm(x_in, weight, eps=1e-6):
            variance = x_in.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            out = x_in * torch.rsqrt(variance + eps)
            return (out.to(weight.dtype) * weight).to(x_in.dtype)
        query = _apply_rmsnorm(query, q_norm_weight)
        key = _apply_rmsnorm(key, k_norm_weight)

    # Step 4: ESIMD Rotary
    if freqs_cis is not None:
        if _esimd_rotary is not None:
            # freqs_cis is [B, S, HD/2] complex64; extract cos/sin as [S, HD/2] f32
            cos_cache = freqs_cis[0].real.contiguous()
            sin_cache = freqs_cis[0].imag.contiguous()
            flat_q = query.reshape(-1, head_dim).contiguous()
            flat_k = key.reshape(-1, head_dim).contiguous()
            query = _esimd_rotary.rotary_emb(flat_q, cos_cache, sin_cache, seq_len, heads).reshape(
                batch_size, seq_len, heads, head_dim
            )
            key = _esimd_rotary.rotary_emb(flat_k, cos_cache, sin_cache, seq_len, heads).reshape(
                batch_size, seq_len, heads, head_dim
            )
        else:
            # Fallback: PyTorch rotary
            def _apply_rotary_emb(x_in, fc):
                device_t = x_in.device.type if x_in.device.type != "meta" else "cpu"
                with torch.amp.autocast(device_t, enabled=False):
                    xc = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                    x_out = torch.view_as_real(xc * fc.unsqueeze(2)).flatten(3)
                    return x_out.type_as(x_in)
            query = _apply_rotary_emb(query, freqs_cis)
            key = _apply_rotary_emb(key, freqs_cis)

    # Recombine for return — on XPU we return the full (Q, K, V) concatenated like CUDA path
    value = value.view(batch_size, seq_len, heads, head_dim)
    # Return concatenated [Q, K, V] along last dim, then reshaped to [B, S, 3*heads*head_dim]
    return torch.cat([
        query.view(batch_size, seq_len, -1),
        key.view(batch_size, seq_len, -1),
        value.view(batch_size, seq_len, -1),
    ], dim=-1)


class ComfyNunchakuZImageAttention(JointAttention):
    """
    Nunchaku optimized attention module for ZImage.
    """

    def __init__(self, orig_attn: JointAttention, **kwargs):
        nn.Module.__init__(self)
        self.n_kv_heads = orig_attn.n_kv_heads
        self.n_local_heads = orig_attn.n_local_heads
        self.n_local_kv_heads = orig_attn.n_local_kv_heads
        self.n_rep = orig_attn.n_rep
        self.head_dim = orig_attn.head_dim

        self.qkv = SVDQW4A4Linear.from_linear(orig_attn.qkv, **kwargs)
        add_comfy_cast_weights_attr(self.qkv, orig_attn.qkv)
        self.out = SVDQW4A4Linear.from_linear(orig_attn.out, **kwargs)
        add_comfy_cast_weights_attr(self.out, orig_attn.out)

        self.q_norm = orig_attn.q_norm
        self.k_norm = orig_attn.k_norm

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        transformer_options={},
    ) -> torch.Tensor:
        """
        Adapted from comfy.ldm.lumina.model.JointAttention#forward

        Optimized with fusing qkv projection, q_norm, k_norm and RoPE in one kernel (CUDA)
        or separate ESIMD kernels (XPU).
        """
        logging.debug(f"x shape: {x.shape}, freqs_cis shape: {freqs_cis.shape}")
        bsz, seqlen, _ = x.shape
        qkv = fused_qkv_norm_rotary(
            x,
            self.qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            freqs_cis,
        )

        xq, xk, xv = torch.split(
            qkv,
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        output = optimized_attention_masked(
            xq.movedim(1, 2),
            xk.movedim(1, 2),
            xv.movedim(1, 2),
            self.n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        return self.out(output)


class ComfyNunchakuZImageFeedForward(nn.Module):
    """
    Nunchaku optimized feed forward module for ZImage.
    """

    def __init__(self, orig_ff: FeedForward, **kwargs):
        super().__init__()
        self.w13 = fuse_to_svdquant_linear(orig_ff.w1, orig_ff.w3, **kwargs)
        self.w2 = SVDQW4A4Linear.from_linear(orig_ff.w2, **kwargs)
        add_comfy_cast_weights_attr(self.w2, orig_ff.w2)

    def _forward_silu_gating(self, x1, x3):
        return clamp_fp16(F.silu(x1) * x3)

    def forward(self, x: torch.Tensor):
        x = self.w13(x)
        x3, x1 = x.chunk(2, dim=-1)
        return self.w2(self._forward_silu_gating(x1, x3))


class RopeFuseAttentionHook:
    """
    This class acts as a pre-forward hook on `ComfyNunchakuZImageAttention`,
    replace the `freqs_cis` tensor with packed one to align with the input format of the fused kernel.

    On CUDA: packs freqs_cis via pack_rotemb for the fused CUDA kernel.
    On XPU: converts freqs_cis to complex64 format for the ESIMD rotary kernel.
    """

    def __init__(self):
        self.packed_freqs_cis_cache = {}
        self.hook_handles = []

    def pre_forward(self, module: ComfyNunchakuZImageAttention, input_args: tuple, input_kwargs: dict):
        """
        Pre-forward hook method.
        Transform freqs_cis to the format expected by the device-specific kernel.
        """
        new_input_args = list(input_args)
        freqs_cis: torch.Tensor = new_input_args[2]
        if freqs_cis is None:
            return None
        cache_key = (freqs_cis.data_ptr(), freqs_cis.shape, module.qkv.qweight.device.type)
        packed_freqs_cis = self.packed_freqs_cis_cache.get(cache_key, None)
        if packed_freqs_cis is None:
            orig_shape = freqs_cis.shape
            device_type = module.qkv.qweight.device.type

            if device_type == "xpu":
                # XPU path: convert to complex64 for ESIMD rotary kernel
                # freqs_cis shape: [1, seq_len, 1, 64, 2, 2] float32
                # The rotation matrix has shape [..., 2, 2] = [[cos, -sin], [sin, cos]]
                # We select row 1 ([sin, cos]) and form complex: cos + j*sin
                # Using [..., 1, :] (without brackets) to avoid keeping dim
                fc = freqs_cis[..., 1, :].squeeze(2)
                # fc shape: [1, seq_len, 64, 2] — last dim is [sin, cos]
                # Swap to [cos, sin] order for view_as_complex (real=cos, imag=sin)
                fc = fc.flip(-1)
                packed_freqs_cis = torch.view_as_complex(fc.contiguous())
                # Shape: [1, seq_len, 64] complex64
            else:
                # CUDA path: pack for fused kernel
                fc = freqs_cis[..., [1], :].squeeze(2)  # See comfy.ldm.flux.math#rope, #apply_rope
                packed_freqs_cis = pack_rotemb(pad_tensor(fc, 256, 1))

            self.packed_freqs_cis_cache[cache_key] = packed_freqs_cis
            logging.debug(
                f"cache miss and created, cache_key: {cache_key}, orig shape: {orig_shape}, "
                f"packed shape: {packed_freqs_cis.shape}, device_type: {device_type}"
            )
        else:
            logging.debug(
                f"cache hit, cache_key: {cache_key}, packed shape: {packed_freqs_cis.shape}"
            )
        new_input_args[2] = packed_freqs_cis
        return tuple(new_input_args), input_kwargs

    def hook(self, module: ComfyNunchakuZImageAttention):
        assert isinstance(module, ComfyNunchakuZImageAttention)
        self.hook_handles.append(module.register_forward_pre_hook(self.pre_forward, with_kwargs=True))

    def unhook(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()
        self.packed_freqs_cis_cache.clear()


class RopeFuseTransformerHook:
    """
    This class acts as a (both pre- and post- forward) hook on `comfy.ldm.lumina.model.NextDiT`,
    add hooks to its attention submodules for qkv/norm/RoPE fusion before forward pass,
    and remove the hooks added afterwards.
    """

    def __init__(self, skip_refiners: bool):
        self.skip_refiners = skip_refiners

    def pre_forward(self, module: NextDiT, input_args: tuple):
        """
        Pre-forward hook method.
        Add a newly created `RopeFuseAttentionHook` object to all the attention submodules in the `NextDiT` model.
        """
        self.attn_hook = RopeFuseAttentionHook()
        for _, ly in enumerate(module.layers):
            self.attn_hook.hook(ly.attention)
        if not self.skip_refiners:
            for _, nr in enumerate(module.noise_refiner):
                self.attn_hook.hook(nr.attention)
            for _, cr in enumerate(module.context_refiner):
                self.attn_hook.hook(cr.attention)
        return None

    def post_forward(self, module: NextDiT, input_args: tuple, output: tuple):
        """
        Post-forward hook method.
        Remove the hooks added by `pre_forward` method and clear the cached items.
        """
        self.attn_hook.unhook()
        logging.debug("RopeFuseTransformerHook post_forward called")
        return None

    def hook(self, model: NextDiT):
        assert isinstance(model, NextDiT)
        self.pre_handle = model.register_forward_pre_hook(self.pre_forward)
        self.post_handle = model.register_forward_hook(self.post_forward, always_call=True)


def patch_model(diffusion_model: NextDiT, skip_refiners: bool, **kwargs):
    """
    Patch the ZImage diffusion model by replacing the attention and feed forward modules with Nunchaku optimized ones in the transformer blocks.

    Parameters
    ----------
    diffusion_model : NextDiT
        The ZImage diffusion model to be patched.
    skip_refiners : bool
        If true, `noise_refiner` and `context_refiner` will NOT be replaced.
    """

    def _patch_transformer_block(block_list: List[JointTransformerBlock]):
        for _, block in enumerate(block_list):
            block.attention = ComfyNunchakuZImageAttention(block.attention, **kwargs)
            block.feed_forward = ComfyNunchakuZImageFeedForward(block.feed_forward, **kwargs)

    _patch_transformer_block(diffusion_model.layers)
    if not skip_refiners:
        _patch_transformer_block(diffusion_model.noise_refiner)
        _patch_transformer_block(diffusion_model.context_refiner)
    RopeFuseTransformerHook(skip_refiners).hook(diffusion_model)

    # `norm_final` is not used in Z-Image-Turbo, prevent state dict loading by setting it to None
    # Maybe remove this line in the future if future Z-Image models use `norm_final`
    diffusion_model.norm_final = None
