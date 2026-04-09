"""
This module provides nodes and utilities for loading and Nunchaku text encoders within ComfyUI.
"""

import gc
import logging
import os
import types
from typing import Callable

import comfy
import comfy.sd
import comfy.sd1_clip
import torch
from comfy.text_encoders.flux import FluxClipModel
from torch import nn

from nunchaku_torch.device import has_xpu, resolve_device
def get_torch_device(): return resolve_device("auto")
def empty_cache():
    import torch
    if has_xpu(): torch.xpu.empty_cache()
    elif torch.cuda.is_available(): torch.cuda.empty_cache()

if has_xpu():
    # On XPU, T5 encoder runs in standard PyTorch mode (no nunchaku C++ kernels)
    # We still import NunchakuT5EncoderModel for compatibility but it will use
    # standard PyTorch ops on XPU
    try:
        NunchakuT5EncoderModel = None  # Not yet in nunchaku_torch
    except ImportError:
        NunchakuT5EncoderModel = None
else:
    NunchakuT5EncoderModel = None  # Not yet in nunchaku_torch

from ..utils import folder_paths, get_filename_list, get_full_path_or_raise

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuTextEncoderLoaderV2:
    """
    Node for loading Nunchaku text encoders. It also supports 16-bit and FP8 variants.

    .. note::
        On Intel XPU, the T5 encoder runs on the XPU device using standard PyTorch ops.
    """

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Text Encoder Loader V2"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["flux.1"],),
                "text_encoder1": (get_filename_list("text_encoders"),),
                "text_encoder2": (get_filename_list("text_encoders"),),
                "t5_min_length": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 1024,
                        "step": 128,
                        "display": "number",
                        "lazy": True,
                        "tooltip": "Minimum sequence length for the T5 encoder.",
                    },
                ),
            }
        }

    def load_text_encoder(self, model_type: str, text_encoder1: str, text_encoder2: str, t5_min_length: int):
        text_encoder_path1 = get_full_path_or_raise("text_encoders", text_encoder1)
        text_encoder_path2 = get_full_path_or_raise("text_encoders", text_encoder2)
        if model_type == "flux.1":
            clip_type = comfy.sd.CLIPType.FLUX
        else:
            raise ValueError(f"Unknown type {model_type}")

        clip = load_text_encoder_state_dicts(
            [text_encoder_path1, text_encoder_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options={},
        )
        clip.tokenizer.t5xxl.min_length = t5_min_length
        return (clip,)


def nunchaku_t5_forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask,
    embeds=None,
    intermediate_output=None,
    final_layer_norm_intermediate=True,
    dtype: str | torch.dtype = torch.bfloat16,
    **kwargs,
):
    """
    Forward function wrapper for NunchakuT5EncoderModel to be compatible with ComfyUI.

    On XPU, moves tensors to XPU device instead of CUDA.
    """
    assert attention_mask is None
    assert intermediate_output is None
    assert final_layer_norm_intermediate

    def get_device(tensors: list[torch.Tensor]) -> torch.device:
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")

    target_device = get_torch_device()
    original_device = None
    input_device = get_device([input_ids, attention_mask, embeds])

    if input_device.type not in ("cuda", "xpu") or input_device != target_device:
        original_device = input_device
        logger.info(
            f"Nunchaku T5 encoder: moving inputs from {original_device} to {target_device}"
        )
        input_ids = input_ids.to(target_device) if input_ids is not None else None
        embeds = embeds.to(target_device) if embeds is not None else None
        attention_mask = attention_mask.to(target_device) if attention_mask is not None else None
        self.encoder = self.encoder.to(target_device)

    outputs = self.encoder(input_ids=input_ids, inputs_embeds=embeds, attention_mask=attention_mask)

    hidden_states = outputs["last_hidden_state"]
    hidden_states = hidden_states.to(dtype=dtype)
    if original_device is not None:
        hidden_states = hidden_states.to(original_device)
        self.encoder = self.encoder.to(original_device)

        gc.collect()
        empty_cache()

    return hidden_states, None


class WrappedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input: torch.Tensor, out_dtype: torch.dtype | None = None):
        return self.embedding(input)

    @property
    def weight(self):
        return self.embedding.weight


def nunchaku_flux_clip(nunchaku_t5_path: str | os.PathLike[str], dtype_t5=None) -> Callable:
    class NunchakuFluxClipModel(FluxClipModel):
        def __init__(
            self,
            dtype_t5=None,
            device="cpu",
            dtype=None,
            model_options={},
        ):
            super(FluxClipModel, self).__init__()
            dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
            self.clip_l = comfy.sd1_clip.SDClipModel(
                device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options
            )

            # Use meta device for T5XXL to avoid loading into memory before replacement
            self.t5xxl = comfy.text_encoders.sd3_clip.T5XXLModel(
                device="meta", dtype=dtype_t5, model_options=model_options
            )

            if NunchakuT5EncoderModel is not None:
                transformer = NunchakuT5EncoderModel.from_pretrained(nunchaku_t5_path, device=device, torch_dtype=dtype_t5)
            else:
                # On XPU without nunchaku: load T5 in standard mode
                from transformers import T5EncoderModel
                transformer = T5EncoderModel.from_pretrained(nunchaku_t5_path, torch_dtype=dtype_t5)
                transformer = transformer.to(device)

            transformer.forward = types.MethodType(nunchaku_t5_forward, transformer)
            if hasattr(transformer, 'shared'):
                transformer.shared = WrappedEmbedding(transformer.shared)
            self.t5xxl.transformer = transformer
            self.t5xxl.logit_scale = nn.Parameter(torch.zeros_like(self.t5xxl.logit_scale, device=device))

            self.dtypes = set([dtype, dtype_t5])

    return NunchakuFluxClipModel


def load_text_encoder_state_dicts(
    paths: list[str | os.PathLike[str]],
    embedding_directory: str | os.PathLike[str] | None = None,
    clip_type=comfy.sd.CLIPType.FLUX,
    model_options: dict = {},
):
    state_dicts, metadata_list = [], []

    for p in paths:
        sd, metadata = comfy.utils.load_torch_file(p, safe_load=True, return_metadata=True)
        state_dicts.append(sd)
        metadata_list.append(metadata)

    class EmptyClass:
        pass

    for i in range(len(state_dicts)):
        if "transformer.resblocks.0.ln_1.weight" in state_dicts[i]:
            state_dicts[i] = comfy.utils.clip_text_transformers_convert(state_dicts[i], "", "")
        else:
            if "text_projection" in state_dicts[i]:
                state_dicts[i]["text_projection.weight"] = state_dicts[i]["text_projection"].transpose(0, 1)

    tokenizer_data = {}
    clip_target = EmptyClass()
    clip_target.params = {}

    nunchaku_model_id = None
    for i, metadata in enumerate(metadata_list):
        if metadata is not None and metadata.get("model_class", None) == "NunchakuT5EncoderModel":
            nunchaku_model_id = i
            break

    if len(state_dicts) == 2:
        if clip_type == comfy.sd.CLIPType.FLUX:
            if nunchaku_model_id is None:
                clip_target.clip = comfy.text_encoders.flux.flux_clip(**comfy.sd.t5xxl_detect(state_dicts))
            else:
                clip_target.clip = nunchaku_flux_clip(nunchaku_t5_path=paths[nunchaku_model_id], dtype_t5=torch.float16)
            clip_target.tokenizer = comfy.text_encoders.flux.FluxTokenizer
    else:
        raise NotImplementedError(f"Clip type {clip_type} not implemented.")

    parameters = 0
    for c in state_dicts:
        parameters += comfy.utils.calculate_parameters(c)
        tokenizer_data, model_options = comfy.text_encoders.long_clipl.model_options_long_clip(
            c, tokenizer_data, model_options
        )
    clip = comfy.sd.CLIP(
        clip_target,
        embedding_directory=embedding_directory,
        parameters=parameters,
        tokenizer_data=tokenizer_data,
        model_options=model_options,
    )
    for state_dict, metadata in zip(state_dicts, metadata_list):
        if metadata is not None and metadata.get("model_class", None) == "NunchakuT5EncoderModel":
            continue
        m, u = clip.load_sd(state_dict)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))

    return clip


class NunchakuTextEncoderLoader:
    """
    Node for loading Nunchaku text encoders (deprecated).
    """

    @classmethod
    def INPUT_TYPES(s):
        prefixes = folder_paths.folder_names_and_paths["text_encoders"][0]
        local_folders = set()
        for prefix in prefixes:
            if os.path.exists(prefix) and os.path.isdir(prefix):
                local_folders_ = os.listdir(prefix)
                local_folders_ = [
                    folder
                    for folder in local_folders_
                    if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
                ]
                local_folders.update(local_folders_)
        model_paths = ["none"] + sorted(list(local_folders))
        return {
            "required": {
                "model_type": (["flux"],),
                "text_encoder1": (get_filename_list("text_encoders"),),
                "text_encoder2": (get_filename_list("text_encoders"),),
                "t5_min_length": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 1024,
                        "step": 128,
                        "display": "number",
                        "lazy": True,
                        "tooltip": "Minimum sequence length for the T5 encoder.",
                    },
                ),
                "use_4bit_t5": (["disable", "enable"],),
                "int4_model": (
                    model_paths,
                    {"tooltip": "The name of the 4-bit T5 model."},
                ),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Text Encoder Loader (Deprecated)"

    def load_text_encoder(
        self,
        model_type: str,
        text_encoder1: str,
        text_encoder2: str,
        t5_min_length: int,
        use_4bit_t5: str,
        int4_model: str,
    ):
        logger.warning(
            "Nunchaku Text Encoder Loader will be deprecated in v0.4. "
            "Please use the Nunchaku Text Encoder Loader V2 node instead."
        )
        text_encoder_path1 = get_full_path_or_raise("text_encoders", text_encoder1)
        text_encoder_path2 = get_full_path_or_raise("text_encoders", text_encoder2)
        if model_type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX
        else:
            raise ValueError(f"Unknown type {model_type}")

        clip = comfy.sd.load_clip(
            ckpt_paths=[text_encoder_path1, text_encoder_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )

        if model_type == "flux":
            clip.tokenizer.t5xxl.min_length = t5_min_length

        if use_4bit_t5 == "enable":
            assert int4_model != "none", "Please select a 4-bit T5 model."
            transformer = clip.cond_stage_model.t5xxl.transformer
            param = next(transformer.parameters())
            dtype = param.dtype
            device = param.device

            prefixes = folder_paths.folder_names_and_paths["text_encoders"][0]
            model_path = None
            for prefix in prefixes:
                if os.path.exists(os.path.join(prefix, int4_model)):
                    model_path = os.path.join(prefix, int4_model)
                    break
            if model_path is None:
                model_path = int4_model

            if NunchakuT5EncoderModel is not None:
                transformer = NunchakuT5EncoderModel.from_pretrained(model_path)
            else:
                from transformers import T5EncoderModel
                transformer = T5EncoderModel.from_pretrained(model_path, torch_dtype=dtype)

            transformer.forward = types.MethodType(nunchaku_t5_forward, transformer)
            if hasattr(transformer, 'shared'):
                transformer.shared = WrappedEmbedding(transformer.shared)

            target_device = get_torch_device()
            clip.cond_stage_model.t5xxl.transformer = (
                transformer.to(device=device, dtype=dtype) if device.type in ("cuda", "xpu") else transformer
            )

        return (clip,)
