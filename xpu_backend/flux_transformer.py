"""
Pure-Python FLUX Transformer for Intel XPU.

Implements the FLUX diffusion transformer architecture using XPU-compatible
quantized linear layers. This replaces nunchaku's C++ QuantizedFluxModel.

The module attribute names match the nunchaku safetensors weight keys exactly
so that ``load_state_dict`` works without key remapping.

Weight files expected:
    unquantized_layers.safetensors   -- embedders, norms, proj_out
    transformer_blocks.safetensors   -- double blocks (transformer_blocks.{i}.*)
                                        and single blocks (single_transformer_blocks.{i}.*)

Architecture reference:
- https://github.com/black-forest-labs/flux
- diffusers FluxTransformer2DModel
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import AWQW4A16Linear, SVDQW4A4Linear

logger = logging.getLogger(__name__)


@dataclass
class FluxOutput:
    sample: torch.Tensor


# ---------------------------------------------------------------------------
# Embedding Layers
# ---------------------------------------------------------------------------


class TimestepEmbedding(nn.Module):
    """MLP to embed timesteps.

    Weight keys: ``timestep_embedder.linear_1.*``, ``timestep_embedder.linear_2.*``
    (or ``text_embedder.linear_1.*``, etc.)
    """

    def __init__(self, in_channels, time_embed_dim, dtype=None, device=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, dtype=dtype, device=device)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(self, num_channels, dtype=torch.float32):
        super().__init__()
        self.num_channels = num_channels
        self.dtype = dtype

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=self.dtype, device=timesteps.device) / half_dim
        emb = timesteps.float().unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combined timestep + text projection embeddings.

    Weight keys under ``time_text_embed``:
        time_text_embed.timestep_embedder.linear_1.weight/bias
        time_text_embed.timestep_embedder.linear_2.weight/bias
        time_text_embed.text_embedder.linear_1.weight/bias
        time_text_embed.text_embedder.linear_2.weight/bias
    """

    def __init__(self, embedding_dim, pooled_projection_dim, dtype=None, device=None):
        super().__init__()
        self.time_proj = Timesteps(256)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim, dtype=dtype, device=device)
        self.text_embedder = TimestepEmbedding(pooled_projection_dim, embedding_dim, dtype=dtype, device=device)

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    """Same as above but with an additional guidance embedder."""

    def __init__(self, embedding_dim, pooled_projection_dim, dtype=None, device=None):
        super().__init__()
        self.time_proj = Timesteps(256)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim, dtype=dtype, device=device)
        self.guidance_embedder = TimestepEmbedding(256, embedding_dim, dtype=dtype, device=device)
        self.text_embedder = TimestepEmbedding(pooled_projection_dim, embedding_dim, dtype=dtype, device=device)

    def forward(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def rope(pos, dim, theta):
    """Compute rotary position embeddings."""
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.float(), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.float()


def apply_rope(xq, freqs_cis):
    """Apply rotary embeddings to queries/keys.

    Args:
        xq: [B, L, num_heads, head_dim]
        freqs_cis: [B, 1, L, head_dim/2, 2, 2] from EmbedND
    """
    if xq.ndim == 3:
        return xq

    B, L, H, D = xq.shape
    # Reshape xq to [..., head_dim/2, 1, 2] for 2x2 rotation matrix multiply
    xq_ = xq.float().reshape(B, L, H, D // 2, 1, 2)
    # freqs_cis: [B, 1, L, D//2, 2, 2] -> broadcast over heads
    freqs = freqs_cis.unsqueeze(3)  # [B, 1, L, 1, D//2, 2, 2]
    # Apply rotation: out = R @ x for each 2-element pair
    xq_out = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    return xq_out.reshape(B, L, H, D).to(xq.dtype)


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class FluxAttention(nn.Module):
    """Standard scaled dot-product attention."""

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        B, L, D = q.shape
        head_dim = D // self.num_heads
        q = q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(B, L, D)


# ---------------------------------------------------------------------------
# Modulation (uses AWQ quantized linear)
# ---------------------------------------------------------------------------


class FluxModulationOut(nn.Module):
    """Container whose only submodule is ``linear`` -- an AWQ quantized linear.

    This exists so the weight key path is ``norm1.linear.qweight``,
    ``norm1.linear.wscales``, etc., matching nunchaku's format.
    """

    def __init__(self, hidden_size, out_multiplier, dtype=None, device=None, group_size=64):
        super().__init__()
        self.linear = AWQW4A16Linear(
            hidden_size, out_multiplier * hidden_size,
            bias=True, group_size=group_size,
            torch_dtype=dtype, device=device,
        )

    def forward(self, x):
        return self.linear(F.silu(x))


# ---------------------------------------------------------------------------
# Double Stream Block
# ---------------------------------------------------------------------------


class FluxDoubleStreamBlock(nn.Module):
    """FLUX double-stream transformer block with joint attention.

    Attribute names match nunchaku weight keys under ``transformer_blocks.{i}.*``:

        norm1.linear.*          -- AWQ modulation for img (6 * hidden_size output)
        norm1_context.linear.*  -- AWQ modulation for txt
        qkv_proj.*              -- SVDQ img QKV [9216, 1536]
        qkv_proj_context.*      -- SVDQ txt QKV
        out_proj.*              -- SVDQ img attn output [3072, 1536]
        out_proj_context.*      -- SVDQ txt attn output
        mlp_fc1.*               -- SVDQ img MLP up [12288, 1536]
        mlp_fc2.*               -- SVDQ img MLP down [3072, 6144]
        mlp_context_fc1.*       -- SVDQ txt MLP up
        mlp_context_fc2.*       -- SVDQ txt MLP down
        norm_q.weight           -- RMSNorm [128]
        norm_k.weight           -- RMSNorm [128]
        norm_added_q.weight     -- RMSNorm txt Q [128]
        norm_added_k.weight     -- RMSNorm txt K [128]
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 dtype=None, device=None, **kwargs):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        head_dim = hidden_size // num_heads

        quant_kwargs = {k: v for k, v in kwargs.items() if k in ("rank", "precision")}

        # --- Image stream ---
        # Pre-norm (not elementwise-affine, just standardization)
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )

        # Modulation -- output 6 * hidden_size (shift, scale, gate for attn + mlp)
        self.norm1 = FluxModulationOut(hidden_size, 6, dtype=dtype, device=device)
        # QKV projection
        self.qkv_proj = SVDQW4A4Linear(
            hidden_size, hidden_size * 3,
            bias=qkv_bias, torch_dtype=dtype, device=device, **quant_kwargs,
        )
        # Q / K RMSNorm (per-head)
        self.norm_q = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        self.norm_k = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        # Output projection
        self.out_proj = SVDQW4A4Linear(
            hidden_size, hidden_size,
            torch_dtype=dtype, device=device, **quant_kwargs,
        )
        # MLP
        self.mlp_fc1 = SVDQW4A4Linear(
            hidden_size, mlp_hidden_dim,
            torch_dtype=dtype, device=device, **quant_kwargs,
        )
        self.mlp_fc2 = SVDQW4A4Linear(
            mlp_hidden_dim, hidden_size,
            torch_dtype=dtype, device=device, **quant_kwargs,
        )

        # --- Text / context stream ---
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )

        self.norm1_context = FluxModulationOut(hidden_size, 6, dtype=dtype, device=device)
        self.qkv_proj_context = SVDQW4A4Linear(
            hidden_size, hidden_size * 3,
            bias=qkv_bias, torch_dtype=dtype, device=device, **quant_kwargs,
        )
        self.norm_added_q = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        self.norm_added_k = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        self.out_proj_context = SVDQW4A4Linear(
            hidden_size, hidden_size,
            torch_dtype=dtype, device=device, **quant_kwargs,
        )
        self.mlp_context_fc1 = SVDQW4A4Linear(
            hidden_size, mlp_hidden_dim,
            torch_dtype=dtype, device=device, **quant_kwargs,
        )
        self.mlp_context_fc2 = SVDQW4A4Linear(
            mlp_hidden_dim, hidden_size,
            torch_dtype=dtype, device=device, **quant_kwargs,
        )

        # Joint attention helper
        self.attn = FluxAttention(num_heads)

    def forward(self, img, txt, vec, pe):
        # --- Modulation ---
        img_mod_out = self.norm1(vec)[:, None, :]  # [B, 1, 6*H]
        (img_mod1_shift, img_mod1_scale, img_mod1_gate,
         img_mod2_shift, img_mod2_scale, img_mod2_gate) = img_mod_out.chunk(6, dim=-1)

        txt_mod_out = self.norm1_context(vec)[:, None, :]
        (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate,
         txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = txt_mod_out.chunk(6, dim=-1)

        # --- Image attention ---
        img_modulated = self.img_norm1(img) * (1 + img_mod1_scale) + img_mod1_shift
        img_qkv = self.qkv_proj(img_modulated)
        img_q, img_k, img_v = img_qkv.chunk(3, dim=-1)
        img_q = self.norm_q(img_q.unflatten(-1, (self.num_heads, -1)))
        img_k = self.norm_k(img_k.unflatten(-1, (self.num_heads, -1)))

        # --- Text attention ---
        txt_modulated = self.txt_norm1(txt) * (1 + txt_mod1_scale) + txt_mod1_shift
        txt_qkv = self.qkv_proj_context(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.chunk(3, dim=-1)
        txt_q = self.norm_added_q(txt_q.unflatten(-1, (self.num_heads, -1)))
        txt_k = self.norm_added_k(txt_k.unflatten(-1, (self.num_heads, -1)))

        # --- Joint attention with RoPE ---
        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)

        if pe is not None:
            q = apply_rope(q, pe)
            k = apply_rope(k, pe)

        q = q.flatten(-2)
        k = k.flatten(-2)
        v = torch.cat([txt_v, img_v], dim=1)

        attn = self.attn(q, k, v)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        # --- Output projections ---
        img = img + img_mod1_gate * self.out_proj(img_attn)
        txt = txt + txt_mod1_gate * self.out_proj_context(txt_attn)

        # --- MLP ---
        img_ff = self.mlp_fc1(self.img_norm2(img) * (1 + img_mod2_scale) + img_mod2_shift)
        img_ff = F.gelu(img_ff, approximate="tanh")
        img_ff = self.mlp_fc2(img_ff)
        img = img + img_mod2_gate * img_ff

        txt_ff = self.mlp_context_fc1(self.txt_norm2(txt) * (1 + txt_mod2_scale) + txt_mod2_shift)
        txt_ff = F.gelu(txt_ff, approximate="tanh")
        txt_ff = self.mlp_context_fc2(txt_ff)
        txt = txt + txt_mod2_gate * txt_ff

        return img, txt


# ---------------------------------------------------------------------------
# Single Stream Block
# ---------------------------------------------------------------------------


class FluxSingleStreamBlock(nn.Module):
    """FLUX single-stream transformer block.

    Attribute names match nunchaku weight keys under
    ``single_transformer_blocks.{i}.*``:

        norm.linear.*     -- AWQ modulation (3 * hidden_size)
        qkv_proj.*        -- SVDQ QKV [9216, 1536]
        mlp_fc1.*         -- SVDQ MLP up [12288, 1536]  (SEPARATE from qkv!)
        mlp_fc2.*         -- SVDQ MLP down [3072, 6144]
        out_proj.*        -- SVDQ output [3072, 1536] (SEPARATE!)
        norm_q.weight     -- RMSNorm [128]
        norm_k.weight     -- RMSNorm [128]
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 dtype=None, device=None, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        quant_kwargs = {k: v for k, v in kwargs.items() if k in ("rank", "precision")}

        # Pre-norm
        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )

        # Modulation (3 * hidden_size: shift, scale, gate)
        self.norm = FluxModulationOut(hidden_size, 3, dtype=dtype, device=device)

        # QKV projection (SEPARATE from MLP)
        self.qkv_proj = SVDQW4A4Linear(
            hidden_size, hidden_size * 3,
            torch_dtype=dtype, device=device, **quant_kwargs,
        )

        # MLP up (SEPARATE from QKV)
        self.mlp_fc1 = SVDQW4A4Linear(
            hidden_size, mlp_hidden_dim,
            torch_dtype=dtype, device=device, **quant_kwargs,
        )

        # MLP down
        self.mlp_fc2 = SVDQW4A4Linear(
            mlp_hidden_dim, hidden_size,
            torch_dtype=dtype, device=device, **quant_kwargs,
        )

        # Output projection (SEPARATE)
        self.out_proj = SVDQW4A4Linear(
            hidden_size, hidden_size,
            torch_dtype=dtype, device=device, **quant_kwargs,
        )

        # Q / K RMSNorm
        self.norm_q = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        self.norm_k = nn.RMSNorm(head_dim, dtype=dtype, device=device)

        self.attn = FluxAttention(num_heads)

    def forward(self, x, vec, pe):
        mod_out = self.norm(vec)[:, None, :]  # [B, 1, 3*H]
        mod_shift, mod_scale, mod_gate = mod_out.chunk(3, dim=-1)

        x_mod = self.pre_norm(x) * (1 + mod_scale) + mod_shift

        # QKV (separate projection)
        qkv = self.qkv_proj(x_mod)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.norm_q(q.unflatten(-1, (self.num_heads, -1)))
        k = self.norm_k(k.unflatten(-1, (self.num_heads, -1)))

        if pe is not None:
            q = apply_rope(q, pe)
            k = apply_rope(k, pe)

        q = q.flatten(-2)
        k = k.flatten(-2)

        attn = self.attn(q, k, v)

        # MLP (separate projection)
        mlp_out = self.mlp_fc1(x_mod)
        mlp_out = F.gelu(mlp_out, approximate="tanh")
        mlp_out = self.mlp_fc2(mlp_out)

        # Combine attention + MLP through output projection
        # In FLUX single blocks the residual is: x + gate * (attn_out + mlp_out)
        # where attn_out and mlp_out go through separate out_proj
        # Actually, the standard FLUX single block does:
        #   output = out_proj(attn) + mlp_fc2(gelu(mlp_fc1(x_mod)))
        output = self.out_proj(attn) + mlp_out
        return x + mod_gate * output


# ---------------------------------------------------------------------------
# Final Layer (norm_out)
# ---------------------------------------------------------------------------


class FluxFinalLayer(nn.Module):
    """Final adaptive-norm + linear projection.

    Weight keys:
        norm_out.linear.weight/bias  -- modulation linear (2 * hidden_size)
        proj_out.weight/bias         -- final projection
    """

    def __init__(self, hidden_size, out_channels, dtype=None, device=None):
        super().__init__()
        self.norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        # The modulation linear lives at norm_out.linear in the weight file
        self.linear = nn.Linear(hidden_size, hidden_size * 2, dtype=dtype, device=device)

    def forward(self, x, vec):
        mod = self.linear(F.silu(vec))[:, None, :]
        shift, scale = mod.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


# ---------------------------------------------------------------------------
# Main Transformer
# ---------------------------------------------------------------------------


class XPUFluxTransformer2dModel(nn.Module):
    """FLUX transformer model for Intel XPU.

    Implements the same interface as NunchakuFluxTransformer2dModel
    but uses pure Python + omni_xpu_kernel instead of C++ CUDA kernels.

    All attribute names are chosen to match the nunchaku safetensors weight keys
    exactly so that ``load_state_dict`` works without key remapping.
    """

    # Default FLUX architecture config
    DEFAULT_CONFIG = {
        "in_channels": 64,       # after patch embedding (not raw 16)
        "out_channels": 64,
        "hidden_size": 3072,
        "num_heads": 24,
        "num_double_blocks": 19,
        "num_single_blocks": 38,
        "mlp_ratio": 4.0,
        "guidance_embed": True,
        "axes_dim": [16, 56, 56],
        "theta": 10000,
        "pooled_projection_dim": 768,
        "context_in_dim": 4096,
        "qkv_bias": True,
    }

    def __init__(self, config=None, torch_dtype=torch.bfloat16, device=None, **kwargs):
        super().__init__()
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.config = cfg
        self.dtype = torch_dtype
        self.hidden_size = cfg["hidden_size"]
        self.num_heads = cfg["num_heads"]
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg.get("out_channels", cfg["in_channels"])

        quant_kwargs = {k: v for k, v in kwargs.items() if k in ("rank", "precision")}

        # --- Embeddings ---
        # Weight keys: x_embedder.weight/bias
        self.x_embedder = nn.Linear(
            cfg["in_channels"], cfg["hidden_size"], dtype=torch_dtype, device=device
        )

        # Weight keys: time_text_embed.timestep_embedder.*, time_text_embed.text_embedder.*
        if cfg.get("guidance_embed", False):
            self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
                cfg["hidden_size"], cfg["pooled_projection_dim"],
                dtype=torch_dtype, device=device,
            )
        else:
            self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                cfg["hidden_size"], cfg["pooled_projection_dim"],
                dtype=torch_dtype, device=device,
            )

        # Weight keys: context_embedder.weight/bias
        self.context_embedder = nn.Linear(
            cfg["context_in_dim"], cfg["hidden_size"], dtype=torch_dtype, device=device
        )

        # Positional embeddings (no learnable parameters)
        self.pe_embedder = EmbedND(
            cfg["hidden_size"] // cfg["num_heads"], cfg["theta"], cfg["axes_dim"]
        )

        # --- Double-stream transformer blocks ---
        # Weight keys: transformer_blocks.{i}.*
        self.transformer_blocks = nn.ModuleList([
            FluxDoubleStreamBlock(
                cfg["hidden_size"], cfg["num_heads"], cfg["mlp_ratio"],
                qkv_bias=cfg.get("qkv_bias", True),
                dtype=torch_dtype, device=device, **quant_kwargs,
            )
            for _ in range(cfg["num_double_blocks"])
        ])

        # --- Single-stream transformer blocks ---
        # Weight keys: single_transformer_blocks.{i}.*
        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleStreamBlock(
                cfg["hidden_size"], cfg["num_heads"], cfg["mlp_ratio"],
                dtype=torch_dtype, device=device, **quant_kwargs,
            )
            for _ in range(cfg["num_single_blocks"])
        ])

        # --- Output ---
        # Weight keys: norm_out.linear.weight/bias
        self.norm_out = FluxFinalLayer(
            cfg["hidden_size"], cfg["in_channels"], dtype=torch_dtype, device=device,
        )
        # Weight keys: proj_out.weight/bias
        self.proj_out = nn.Linear(
            cfg["hidden_size"], cfg["in_channels"], dtype=torch_dtype, device=device
        )

        # --- LoRA support ---
        self.comfy_lora_meta_list = []
        self.comfy_lora_sd_list = []

        # --- Caching support ---
        self.residual_diff_threshold_multi = 0
        self._is_cached = False

        # Attention implementation (unused on XPU, always uses PyTorch SDPA)
        self._attention_impl = "sdpa"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_attention_impl(self, impl):
        """Set attention implementation (for API compatibility)."""
        self._attention_impl = impl
        logger.info(f"Attention implementation set to: {impl} (XPU uses PyTorch SDPA)")

    def reset_lora(self):
        """Reset all LoRA parameters."""
        for block in list(self.transformer_blocks) + list(self.single_transformer_blocks):
            for module in block.modules():
                if isinstance(module, SVDQW4A4Linear):
                    if module.proj_down is not None:
                        module.proj_down.zero_()
                    if module.proj_up is not None:
                        module.proj_up.zero_()

    def reset_x_embedder(self):
        """Reset x_embedder to default."""
        pass

    def update_lora_params(self, lora_sd):
        """Update LoRA parameters from state dict."""
        for name, param in lora_sd.items():
            parts = name.split(".")
            try:
                module = self
                for part in parts[:-1]:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
                setattr(module, parts[-1], nn.Parameter(param.to(self.dtype)))
            except (AttributeError, IndexError):
                logger.debug(f"Skipping LoRA param: {name}")

    def prepare_all_weights(self):
        """Prepare all quantized linear weights for oneDNN inference."""
        for module in self.modules():
            if isinstance(module, SVDQW4A4Linear):
                module.prepare_weights()
            elif isinstance(module, AWQW4A16Linear):
                module.invalidate_cache()
        logger.info("All quantized weights prepared for XPU inference")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        **kwargs,
    ):
        # Embed inputs
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Time + text embedding
        if guidance is not None and hasattr(self.time_text_embed, 'guidance_embedder'):
            vec = self.time_text_embed(timestep, guidance, pooled_projections)
        else:
            vec = self.time_text_embed(timestep, pooled_projections)

        # Positional embeddings
        ids = torch.cat([txt_ids, img_ids], dim=1)
        pe = self.pe_embedder(ids)

        # Double stream blocks
        img = hidden_states
        txt = encoder_hidden_states
        for i, block in enumerate(self.transformer_blocks):
            img, txt = block(img, txt, vec, pe)
            if controlnet_block_samples is not None and i < len(controlnet_block_samples):
                ctrl = controlnet_block_samples[i]
                if ctrl is not None:
                    img = img + ctrl.to(img.dtype)

        # Concatenate for single stream
        x = torch.cat([txt, img], dim=1)

        # Single stream blocks
        for i, block in enumerate(self.single_transformer_blocks):
            x = block(x, vec, pe)
            if controlnet_single_block_samples is not None and i < len(controlnet_single_block_samples):
                ctrl = controlnet_single_block_samples[i]
                if ctrl is not None:
                    x[:, txt.shape[1]:] = x[:, txt.shape[1]:] + ctrl.to(x.dtype)

        # Extract image tokens
        img = x[:, txt.shape[1]:]

        # Final layer
        img = self.norm_out(img, vec)
        img = self.proj_out(img)

        return FluxOutput(sample=img)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        path,
        offload=False,
        device=None,
        torch_dtype=torch.bfloat16,
        return_metadata=False,
        **kwargs,
    ):
        """Load a FLUX model from nunchaku-format safetensors.

        The model directory should contain:
            - unquantized_layers.safetensors (embeddings, norms, proj_out)
            - transformer_blocks.safetensors (double + single blocks)
            - Optionally: comfy_config.json

        Or a single safetensors file combining everything.

        Parameters
        ----------
        path : str
            Path to model directory or safetensors file.
        offload : bool
            Whether to enable CPU offloading.
        device : torch.device
            Target device.
        torch_dtype : torch.dtype
            Model dtype.
        return_metadata : bool
            If True, return (model, metadata) tuple.

        Returns
        -------
        model or (model, metadata)
        """
        from safetensors.torch import load_file

        metadata = {}

        if os.path.isdir(path):
            # Collect state dict from all safetensors files in directory
            state_dict = {}

            # Try loading separate files
            unquant_path = os.path.join(path, "unquantized_layers.safetensors")
            blocks_path = os.path.join(path, "transformer_blocks.safetensors")

            if os.path.exists(unquant_path) and os.path.exists(blocks_path):
                logger.info(f"Loading unquantized layers from {unquant_path}")
                state_dict.update(load_file(unquant_path, device="cpu"))
                logger.info(f"Loading transformer blocks from {blocks_path}")
                state_dict.update(load_file(blocks_path, device="cpu"))
            else:
                # Fallback: look for any safetensors files
                candidates = ["quantized_model.safetensors", "model.safetensors"]
                safetensors_path = None
                for c in candidates:
                    p = os.path.join(path, c)
                    if os.path.exists(p):
                        safetensors_path = p
                        break
                if safetensors_path is None:
                    for f in sorted(os.listdir(path)):
                        if f.endswith(".safetensors"):
                            safetensors_path = os.path.join(path, f)
                            break
                if safetensors_path is None:
                    raise FileNotFoundError(f"No safetensors file found in {path}")
                logger.info(f"Loading FLUX model from {safetensors_path}")
                state_dict = load_file(safetensors_path, device="cpu")

            # Read metadata from any safetensors file
            try:
                from safetensors import safe_open
                for fname in ["transformer_blocks.safetensors", "unquantized_layers.safetensors"]:
                    fpath = os.path.join(path, fname)
                    if os.path.exists(fpath):
                        with safe_open(fpath, framework="pt") as f:
                            if f.metadata():
                                metadata.update(dict(f.metadata()))
                        break
            except Exception:
                pass

            # Load config
            config_path = os.path.join(path, "comfy_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    comfy_config = json.load(f)
            else:
                comfy_config = None
        else:
            logger.info(f"Loading FLUX model from {path}")
            state_dict = load_file(path, device="cpu")
            comfy_config = None
            try:
                from safetensors import safe_open
                with safe_open(path, framework="pt") as f:
                    if f.metadata():
                        metadata = dict(f.metadata())
            except Exception:
                pass

        # Detect model config from state dict
        config = cls._detect_config(state_dict, comfy_config)

        # Extract quantization config
        quant_config = {}
        if metadata and "quantization_config" in metadata:
            quant_config = json.loads(metadata["quantization_config"])

        rank = quant_config.get("rank", 32)
        precision = quant_config.get("quant_type", "int4")

        # Create model
        model = cls(
            config=config,
            torch_dtype=torch_dtype,
            device="cpu",
            rank=rank,
            precision=precision,
        )

        # Load state dict with strict=False (allows missing/unexpected keys to be reported)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug(f"Missing keys ({len(missing)}): {missing[:20]}...")
        if unexpected:
            logger.debug(f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}...")
        logger.info(
            f"Loaded state dict: {len(state_dict)} keys, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )

        # Move to device and prepare weights
        if device is not None:
            model = model.to(device)
        model.prepare_all_weights()

        if return_metadata:
            return model, metadata
        return model

    @staticmethod
    def _detect_config(state_dict, comfy_config=None):
        """Detect FLUX model configuration from state dict keys."""
        config = dict(XPUFluxTransformer2dModel.DEFAULT_CONFIG)

        if comfy_config and "model_config" in comfy_config:
            mc = comfy_config["model_config"]
            if "hidden_size" in mc:
                config["hidden_size"] = mc["hidden_size"]
            if "num_heads" in mc:
                config["num_heads"] = mc["num_heads"]

        # Count blocks
        double_blocks = set()
        single_blocks = set()
        for key in state_dict.keys():
            if key.startswith("transformer_blocks."):
                idx = int(key.split(".")[1])
                double_blocks.add(idx)
            elif key.startswith("single_transformer_blocks."):
                idx = int(key.split(".")[1])
                single_blocks.add(idx)

        if double_blocks:
            config["num_double_blocks"] = max(double_blocks) + 1
        if single_blocks:
            config["num_single_blocks"] = max(single_blocks) + 1

        # Detect hidden_size from embedder weight
        if "x_embedder.weight" in state_dict:
            config["hidden_size"] = state_dict["x_embedder.weight"].shape[0]
            config["in_channels"] = state_dict["x_embedder.weight"].shape[1]
        if "context_embedder.weight" in state_dict:
            config["context_in_dim"] = state_dict["context_embedder.weight"].shape[1]

        config["num_heads"] = config["hidden_size"] // 128  # head_dim=128 for FLUX

        # Detect guidance embed
        has_guidance = any(
            k.startswith("time_text_embed.guidance_embedder.") for k in state_dict
        )
        config["guidance_embed"] = has_guidance

        # Detect pooled_projection_dim from text_embedder
        text_emb_key = "time_text_embed.text_embedder.linear_1.weight"
        if text_emb_key in state_dict:
            config["pooled_projection_dim"] = state_dict[text_emb_key].shape[1]

        return config
