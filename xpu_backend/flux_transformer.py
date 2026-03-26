"""
Pure-Python FLUX Transformer for Intel XPU.

Implements the FLUX diffusion transformer architecture using XPU-compatible
quantized linear layers. This replaces nunchaku's C++ QuantizedFluxModel.

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


# --- Embedding Layers ---


class TimestepEmbedding(nn.Module):
    """MLP to embed timesteps."""

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


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim, dtype=None, device=None):
        super().__init__()
        self.time_proj = Timesteps(256)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim, dtype=dtype, device=device)
        self.guidance_embedder = TimestepEmbedding(256, embedding_dim, dtype=dtype, device=device)
        self.text_embedder = nn.Linear(pooled_projection_dim, embedding_dim, dtype=dtype, device=device)

    def forward(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim, dtype=None, device=None):
        super().__init__()
        self.time_proj = Timesteps(256)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim, dtype=dtype, device=device)
        self.text_embedder = nn.Linear(pooled_projection_dim, embedding_dim, dtype=dtype, device=device)

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


# --- RoPE ---


def rope(pos, dim, theta):
    """Compute rotary position embeddings."""
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.float(), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.float()


def apply_rope(xq, freqs_cis):
    """Apply rotary embeddings to queries/keys.

    Args:
        xq: [B, L, num_heads, head_dim] or [B, L, D] (will be reshaped per-head)
        freqs_cis: [B, 1, L, head_dim/2, 2, 2] from EmbedND
    """
    # xq: [B, L, n_heads, head_dim]
    orig_shape = xq.shape
    if xq.ndim == 3:
        # Already flattened - can't apply per-head, just return
        return xq

    B, L, H, D = xq.shape
    # Reshape xq to [..., head_dim/2, 1, 2] for 2x2 rotation matrix multiply
    xq_ = xq.float().reshape(B, L, H, D // 2, 1, 2)
    # freqs_cis: [B, 1, L, D//2, 2, 2] -> broadcast over heads
    freqs = freqs_cis.unsqueeze(3)  # [B, 1, L, 1, D//2, 2, 2]
    # Apply rotation: out = R @ x for each 2-element pair
    xq_out = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    # xq_out: [B, L, H, D//2, 2]
    return xq_out.reshape(orig_shape).to(xq.dtype)


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


# --- Attention ---


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


# --- Modulation ---


class FluxModulation(nn.Module):
    def __init__(self, dim, double, dtype=None, device=None):
        super().__init__()
        self.is_double = double
        multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, multiplier * dim, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, vec):
        out = self.lin(self.act(vec))[:, None, :]
        return out.chunk(6 if self.is_double else 3, dim=-1)


# --- Transformer Blocks ---


class FluxDoubleStreamBlock(nn.Module):
    """FLUX double-stream transformer block with joint attention."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True, dtype=None, device=None, **kwargs):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        head_dim = hidden_size // num_heads

        self.img_mod = FluxModulation(hidden_size, double=True, dtype=dtype, device=device)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_attn_qkv = SVDQW4A4Linear(hidden_size, hidden_size * 3, bias=qkv_bias, torch_dtype=dtype, device=device, **kwargs)
        self.img_attn_norm_q = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        self.img_attn_norm_k = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        self.img_attn_proj = SVDQW4A4Linear(hidden_size, hidden_size, torch_dtype=dtype, device=device, **kwargs)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_mlp_0 = SVDQW4A4Linear(hidden_size, mlp_hidden_dim, torch_dtype=dtype, device=device, **kwargs)
        self.img_mlp_2 = SVDQW4A4Linear(mlp_hidden_dim, hidden_size, torch_dtype=dtype, device=device, **kwargs)

        self.txt_mod = FluxModulation(hidden_size, double=True, dtype=dtype, device=device)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_attn_qkv = SVDQW4A4Linear(hidden_size, hidden_size * 3, bias=qkv_bias, torch_dtype=dtype, device=device, **kwargs)
        self.txt_attn_norm_q = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        self.txt_attn_norm_k = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        self.txt_attn_proj = SVDQW4A4Linear(hidden_size, hidden_size, torch_dtype=dtype, device=device, **kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_mlp_0 = SVDQW4A4Linear(hidden_size, mlp_hidden_dim, torch_dtype=dtype, device=device, **kwargs)
        self.txt_mlp_2 = SVDQW4A4Linear(mlp_hidden_dim, hidden_size, torch_dtype=dtype, device=device, **kwargs)

        self.attn = FluxAttention(num_heads)

    def forward(self, img, txt, vec, pe):
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.img_mod(vec)
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = self.txt_mod(vec)

        # Image attention
        img_modulated = self.img_norm1(img) * (1 + img_mod1_scale) + img_mod1_shift
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.chunk(3, dim=-1)
        # [B, L, D] -> [B, L, H, D/H] for norm, keep per-head for RoPE
        img_q = self.img_attn_norm_q(img_q.unflatten(-1, (self.num_heads, -1)))
        img_k = self.img_attn_norm_k(img_k.unflatten(-1, (self.num_heads, -1)))

        # Text attention
        txt_modulated = self.txt_norm1(txt) * (1 + txt_mod1_scale) + txt_mod1_shift
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.chunk(3, dim=-1)
        txt_q = self.txt_attn_norm_q(txt_q.unflatten(-1, (self.num_heads, -1)))
        txt_k = self.txt_attn_norm_k(txt_k.unflatten(-1, (self.num_heads, -1)))

        # Joint attention with RoPE (applied per-head)
        # [B, L_txt, H, D/H] cat [B, L_img, H, D/H] -> [B, L_total, H, D/H]
        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)

        if pe is not None:
            q = apply_rope(q, pe)
            k = apply_rope(k, pe)

        # Flatten back to [B, L, D] for attention
        q = q.flatten(-2)
        k = k.flatten(-2)
        v = torch.cat([txt_v, img_v], dim=1)

        attn = self.attn(q, k, v)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        # Output projections
        img = img + img_mod1_gate * self.img_attn_proj(img_attn)
        txt = txt + txt_mod1_gate * self.txt_attn_proj(txt_attn)

        # MLP
        img_mlp = self.img_mlp_0(self.img_norm2(img) * (1 + img_mod2_scale) + img_mod2_shift)
        img_mlp = F.gelu(img_mlp, approximate="tanh")
        img_mlp = self.img_mlp_2(img_mlp)
        img = img + img_mod2_gate * img_mlp

        txt_mlp = self.txt_mlp_0(self.txt_norm2(txt) * (1 + txt_mod2_scale) + txt_mod2_shift)
        txt_mlp = F.gelu(txt_mlp, approximate="tanh")
        txt_mlp = self.txt_mlp_2(txt_mlp)
        txt = txt + txt_mod2_gate * txt_mlp

        return img, txt


class FluxSingleStreamBlock(nn.Module):
    """FLUX single-stream transformer block."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dtype=None, device=None, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.mod = FluxModulation(hidden_size, double=False, dtype=dtype, device=device)
        self.linear1 = SVDQW4A4Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim, torch_dtype=dtype, device=device, **kwargs)
        self.linear2 = SVDQW4A4Linear(hidden_size + mlp_hidden_dim, hidden_size, torch_dtype=dtype, device=device, **kwargs)
        self.norm_q = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        self.norm_k = nn.RMSNorm(head_dim, dtype=dtype, device=device)
        self.attn = FluxAttention(num_heads)

    def forward(self, x, vec, pe):
        mod_shift, mod_scale, mod_gate = self.mod(vec)
        x_mod = self.norm(x) * (1 + mod_scale) + mod_shift

        qkv_mlp = self.linear1(x_mod)
        qkv, mlp = qkv_mlp.split([self.hidden_size * 3, qkv_mlp.shape[-1] - self.hidden_size * 3], dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, L, D] -> [B, L, H, D/H] for norm + RoPE
        q = self.norm_q(q.unflatten(-1, (self.num_heads, -1)))
        k = self.norm_k(k.unflatten(-1, (self.num_heads, -1)))

        if pe is not None:
            q = apply_rope(q, pe)
            k = apply_rope(k, pe)

        q = q.flatten(-2)
        k = k.flatten(-2)

        attn = self.attn(q, k, v)
        output = self.linear2(torch.cat([attn, F.gelu(mlp, approximate="tanh")], dim=-1))
        return x + mod_gate * output


# --- Main Transformer ---


class XPUFluxTransformer2dModel(nn.Module):
    """FLUX transformer model for Intel XPU.

    Implements the same interface as NunchakuFluxTransformer2dModel
    but uses pure Python + omni_xpu_kernel instead of C++ CUDA kernels.
    """

    # Default FLUX architecture config
    DEFAULT_CONFIG = {
        "in_channels": 64,
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

        # Embeddings
        self.x_embedder = nn.Linear(cfg["in_channels"], cfg["hidden_size"], dtype=torch_dtype, device=device)
        self.t_embedder = TimestepEmbedding(256, cfg["hidden_size"], dtype=torch_dtype, device=device)
        self.time_proj = Timesteps(256)

        if cfg.get("guidance_embed", False):
            self.guidance_embedder = TimestepEmbedding(256, cfg["hidden_size"], dtype=torch_dtype, device=device)
        else:
            self.guidance_embedder = None

        self.y_embedder = nn.Linear(cfg["pooled_projection_dim"], cfg["hidden_size"], dtype=torch_dtype, device=device)
        self.context_embedder = nn.Linear(cfg["context_in_dim"], cfg["hidden_size"], dtype=torch_dtype, device=device)

        self.pe_embedder = EmbedND(cfg["hidden_size"] // cfg["num_heads"], cfg["theta"], cfg["axes_dim"])

        # Transformer blocks
        self.double_blocks = nn.ModuleList([
            FluxDoubleStreamBlock(
                cfg["hidden_size"], cfg["num_heads"], cfg["mlp_ratio"],
                qkv_bias=cfg.get("qkv_bias", True),
                dtype=torch_dtype, device=device, **quant_kwargs,
            )
            for _ in range(cfg["num_double_blocks"])
        ])
        self.single_blocks = nn.ModuleList([
            FluxSingleStreamBlock(
                cfg["hidden_size"], cfg["num_heads"], cfg["mlp_ratio"],
                dtype=torch_dtype, device=device, **quant_kwargs,
            )
            for _ in range(cfg["num_single_blocks"])
        ])

        # Output
        self.norm_out = nn.LayerNorm(cfg["hidden_size"], elementwise_affine=False, eps=1e-6, dtype=torch_dtype, device=device)
        self.proj_out = nn.Linear(cfg["hidden_size"], cfg["in_channels"], dtype=torch_dtype, device=device)
        # Modulation for final layer
        self.final_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg["hidden_size"], cfg["hidden_size"] * 2, dtype=torch_dtype, device=device),
        )

        # LoRA support
        self.comfy_lora_meta_list = []
        self.comfy_lora_sd_list = []

        # Caching support
        self.residual_diff_threshold_multi = 0
        self._is_cached = False

        # Attention implementation (unused on XPU, always uses PyTorch SDPA)
        self._attention_impl = "sdpa"

    def set_attention_impl(self, impl):
        """Set attention implementation (for API compatibility)."""
        self._attention_impl = impl
        logger.info(f"Attention implementation set to: {impl} (XPU uses PyTorch SDPA)")

    def reset_lora(self):
        """Reset all LoRA parameters."""
        for block in list(self.double_blocks) + list(self.single_blocks):
            for module in block.modules():
                if isinstance(module, SVDQW4A4Linear):
                    # Reset LoRA projections to zero effect
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
        logger.info("All SVDQW4A4Linear weights prepared for XPU inference")

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

        timesteps_proj = self.time_proj(timestep)
        vec = self.t_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            vec = vec + self.guidance_embedder(guidance_proj.to(dtype=hidden_states.dtype))
        vec = vec + self.y_embedder(pooled_projections)

        # Positional embeddings
        ids = torch.cat([txt_ids, img_ids], dim=1)
        pe = self.pe_embedder(ids)

        # Double stream blocks
        img = hidden_states
        txt = encoder_hidden_states
        for i, block in enumerate(self.double_blocks):
            img, txt = block(img, txt, vec, pe)
            if controlnet_block_samples is not None and i < len(controlnet_block_samples):
                ctrl = controlnet_block_samples[i]
                if ctrl is not None:
                    img = img + ctrl.to(img.dtype)

        # Concatenate for single stream
        x = torch.cat([txt, img], dim=1)

        # Single stream blocks
        for i, block in enumerate(self.single_blocks):
            x = block(x, vec, pe)
            if controlnet_single_block_samples is not None and i < len(controlnet_single_block_samples):
                ctrl = controlnet_single_block_samples[i]
                if ctrl is not None:
                    x[:, txt.shape[1]:] = x[:, txt.shape[1]:] + ctrl.to(x.dtype)

        # Extract image tokens
        img = x[:, txt.shape[1]:]

        # Final layer
        final_mod = self.final_mod(vec)[:, None, :]
        shift, scale = final_mod.chunk(2, dim=-1)
        img = self.norm_out(img) * (1 + scale) + shift
        img = self.proj_out(img)

        return FluxOutput(sample=img)

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

        if os.path.isdir(path):
            # Look for safetensors file in directory
            candidates = ["quantized_model.safetensors", "model.safetensors"]
            safetensors_path = None
            for c in candidates:
                p = os.path.join(path, c)
                if os.path.exists(p):
                    safetensors_path = p
                    break
            if safetensors_path is None:
                # Try any .safetensors file
                for f in os.listdir(path):
                    if f.endswith(".safetensors"):
                        safetensors_path = os.path.join(path, f)
                        break
            if safetensors_path is None:
                raise FileNotFoundError(f"No safetensors file found in {path}")

            # Load config
            config_path = os.path.join(path, "comfy_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    comfy_config = json.load(f)
            else:
                comfy_config = None
        else:
            safetensors_path = path
            comfy_config = None

        logger.info(f"Loading FLUX model from {safetensors_path}")
        state_dict = load_file(safetensors_path, device="cpu")

        # Detect model config from state dict
        config = cls._detect_config(state_dict, comfy_config)

        # Read metadata from safetensors
        metadata = None
        try:
            from safetensors import safe_open
            with safe_open(safetensors_path, framework="pt") as f:
                metadata = dict(f.metadata()) if f.metadata() else {}
        except Exception:
            metadata = {}

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
            device="cpu",  # Load on CPU first
            rank=rank,
            precision=precision,
        )

        # Load state dict
        model._load_nunchaku_state_dict(state_dict, torch_dtype)

        # Prepare weights for oneDNN
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
            if key.startswith("double_blocks."):
                idx = int(key.split(".")[1])
                double_blocks.add(idx)
            elif key.startswith("single_blocks."):
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
        config["guidance_embed"] = any(k.startswith("guidance_embedder.") or k.startswith("guidance_in.") for k in state_dict)

        return config

    def _load_nunchaku_state_dict(self, state_dict, torch_dtype):
        """Load nunchaku-format quantized state dict into the model."""
        loaded = set()

        for name, module in self.named_modules():
            if isinstance(module, SVDQW4A4Linear):
                prefix = name + "."
                # Load quantized params
                for param_name in ["qweight", "wscales", "wcscales", "wtscale",
                                   "smooth_factor", "smooth_factor_orig",
                                   "proj_down", "proj_up", "bias"]:
                    key = prefix + param_name
                    if key in state_dict:
                        tensor = state_dict[key]
                        if param_name in ("qweight",):
                            # Keep uint8
                            getattr(module, param_name).data = tensor
                        elif param_name in ("wcscales", "wtscale"):
                            setattr(module, param_name, tensor.to(torch_dtype))
                        else:
                            buf = getattr(module, param_name, None)
                            if buf is not None:
                                buf.data = tensor.to(buf.dtype)
                            else:
                                setattr(module, param_name, tensor.to(torch_dtype))
                        loaded.add(key)

            elif isinstance(module, (nn.Linear, nn.LayerNorm, nn.RMSNorm)):
                prefix = name + "."
                for param_name in ["weight", "bias"]:
                    key = prefix + param_name
                    if key in state_dict:
                        param = getattr(module, param_name, None)
                        if param is not None:
                            param.data = state_dict[key].to(param.dtype)
                        loaded.add(key)

        # Handle modulation layers (may be AWQ quantized in nunchaku)
        for name, module in self.named_modules():
            if isinstance(module, FluxModulation):
                key = name + ".lin.weight"
                if key in state_dict:
                    module.lin.weight.data = state_dict[key].to(module.lin.weight.dtype)
                    loaded.add(key)
                key = name + ".lin.bias"
                if key in state_dict:
                    module.lin.bias.data = state_dict[key].to(module.lin.bias.dtype)
                    loaded.add(key)

        unloaded = set(state_dict.keys()) - loaded
        if unloaded:
            logger.debug(f"Unloaded keys ({len(unloaded)}): {list(unloaded)[:20]}...")
        logger.info(f"Loaded {len(loaded)}/{len(state_dict)} keys from state dict")
