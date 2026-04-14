"""
This module provides the :class:`NunchakuFluxDiTLoader` class for loading Nunchaku FLUX models.
It also supports attention implementation selection, CPU offload, and first-block caching.
"""

import gc
import json
import logging
import os

import comfy.model_management
import comfy.model_patcher
import torch
from comfy.supported_models import Flux, FluxSchnell
from nunchaku_torch import NunchakuFluxTransformer2DModel as FluxTransformerModel
from nunchaku_torch.utils import is_turing, get_gpu_memory

from ...wrappers.flux import ComfyFluxWrapper
from ..utils import get_filename_list, get_full_path_or_raise


def get_device_count():
    """Get number of available XPU/CUDA devices."""
    if torch.xpu.is_available():
        return torch.xpu.device_count()
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def get_device(device_id: int = 0):
    """Get torch device by ID."""
    if torch.xpu.is_available():
        return torch.device(f"xpu:{device_id}")
    elif torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def get_device_name(device_id: int = 0):
    """Get device name string."""
    if torch.xpu.is_available():
        return torch.xpu.get_device_name(device_id)
    elif torch.cuda.is_available():
        return torch.cuda.get_device_name(device_id)
    return "cpu"

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuFluxDiTLoader:
    """
    Loader for Nunchaku FLUX.1 models.
    Supports both CUDA (via nunchaku) and Intel XPU (via omni_xpu_kernel).
    """

    def __init__(self):
        self.transformer = None
        self.metadata = None
        self.model_path = None
        self.device = None
        self.cpu_offload = None
        self.data_type = None
        self.patcher = None
        self.device = comfy.model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        safetensor_files = get_filename_list("diffusion_models")

        ngpus = max(get_device_count(), 1)

        device = comfy.model_management.get_torch_device()
        if device.type == "xpu":
            attention_options = ["nunchaku-fp16"]
            dtype_options = ["bfloat16", "float16"]
        else:
            all_turing = True
            for i in range(get_device_count()):
                if not is_turing(f"cuda:{i}"):
                    all_turing = False

            if all_turing:
                attention_options = ["nunchaku-fp16"]
                dtype_options = ["float16"]
            else:
                attention_options = ["nunchaku-fp16", "flash-attention2"]
                dtype_options = ["bfloat16", "float16"]

        return {
            "required": {
                "model_path": (
                    safetensor_files,
                    {"tooltip": "The Nunchaku FLUX model."},
                ),
                "cache_threshold": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1,
                        "step": 0.001,
                        "tooltip": "Adjusts the first-block caching tolerance"
                        "like `residual_diff_threshold` in WaveSpeed. "
                        "Increasing the value enhances speed at the cost of quality. "
                        "A typical setting is 0.12. Setting it to 0 disables the effect.",
                    },
                ),
                "attention": (
                    attention_options,
                    {
                        "default": attention_options[0],
                        "tooltip": (
                            "Attention implementation. "
                            "On XPU, PyTorch SDPA is used. "
                            "On CUDA, `flash-attention2` or `nunchaku-fp16` are available."
                        ),
                    },
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "Whether to enable CPU offload for the transformer model."
                        "auto' will enable it if the GPU memory is less than 14G.",
                    },
                ),
                "device_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": ngpus - 1,
                        "step": 1,
                        "display": "number",
                        "lazy": True,
                        "tooltip": "The GPU device ID to use for the model.",
                    },
                ),
                "data_type": (
                    dtype_options,
                    {
                        "default": dtype_options[0],
                        "tooltip": "Specifies the model's data type. Default is `bfloat16`. "
                        "For 20-series GPUs, which do not support `bfloat16`, use `float16` instead.",
                    },
                ),
            },
            "optional": {
                "i2f_mode": (
                    ["enabled", "always"],
                    {
                        "default": "enabled",
                        "tooltip": "The GEMM implementation for 20-series GPUs"
                        "— this option is only applicable to these GPUs.",
                    },
                )
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku FLUX DiT Loader"

    def load_model(
        self,
        model_path: str,
        attention: str,
        cache_threshold: float,
        cpu_offload: str,
        device_id: int,
        data_type: str,
        **kwargs,
    ):
        device = get_device(device_id)

        model_path = get_full_path_or_raise("diffusion_models", model_path)

        # Check if the device_id is valid
        if device_id >= get_device_count():
            raise ValueError(f"Invalid device_id: {device_id}. Only {get_device_count()} devices available.")

        # Get the GPU properties
        gpu_memory = get_gpu_memory(device, unit="MiB")
        gpu_name = get_device_name(device_id)
        logger.debug(f"Device {device_id} ({gpu_name}) Memory: {gpu_memory} MiB")

        # Check if CPU offload needs to be enabled
        if cpu_offload == "auto":
            if gpu_memory < 14336:  # 14GB threshold
                cpu_offload_enabled = True
                logger.debug("VRAM < 14GiB, enabling CPU offload")
            else:
                cpu_offload_enabled = False
                logger.debug("VRAM > 14GiB, disabling CPU offload")
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
            logger.debug("Enabling CPU offload")
        else:
            cpu_offload_enabled = False
            logger.debug("Disabling CPU offload")

        if (
            self.model_path != model_path
            or self.device != device
            or self.cpu_offload != cpu_offload_enabled
            or self.data_type != data_type
        ):
            if self.transformer is not None:
                model_size = comfy.model_management.module_size(self.transformer)
                transformer = self.transformer
                self.transformer = None
                transformer.to("cpu")
                del transformer
                gc.collect()
                comfy.model_management.cleanup_models_gc()
                comfy.model_management.soft_empty_cache()
                comfy.model_management.free_memory(model_size, device)

            self.transformer, self.metadata = FluxTransformerModel.from_pretrained(
                model_path,
                offload=cpu_offload_enabled,
                device=device,
                torch_dtype=torch.float16 if data_type == "float16" else torch.bfloat16,
                return_metadata=True,
            )
            self.model_path = model_path
            self.device = device
            self.cpu_offload = cpu_offload_enabled
            self.data_type = data_type

        if False:  # caching not supported
            self.transformer = apply_cache_on_transformer(
                transformer=self.transformer, residual_diff_threshold=cache_threshold
            )

        transformer = self.transformer
        if False:  # removed
            transformer.set_attention_impl("sdpa")
        elif attention == "nunchaku-fp16":
            transformer.set_attention_impl("nunchaku-fp16")
        else:
            assert attention == "flash-attention2"
            transformer.set_attention_impl("flashattn2")

        if self.metadata is None:
            if os.path.exists(os.path.join(model_path, "comfy_config.json")):
                config_path = os.path.join(model_path, "comfy_config.json")
            else:
                default_config_root = os.path.join(os.path.dirname(__file__), "configs")
                config_name = os.path.basename(model_path).replace("svdq-int4-", "").replace("svdq-fp4-", "")
                config_path = os.path.join(default_config_root, f"{config_name}.json")
                assert os.path.exists(config_path), f"Config file not found: {config_path}"

            logger.info(f"Loading ComfyUI model config from {config_path}")
            comfy_config = json.load(open(config_path, "r"))
        else:
            comfy_config_str = self.metadata.get("comfy_config", None)
            comfy_config = json.loads(comfy_config_str)
        model_class_name = comfy_config["model_class"]

        model_config = comfy_config["model_config"]
        if "disable_unet_model_creation" not in model_config:
            model_config["disable_unet_model_creation"] = True

        if model_class_name == "FluxSchnell":
            model_class = FluxSchnell
        else:
            assert model_class_name == "Flux", f"Unknown model class {model_class_name}."
            model_class = Flux
        model_config = model_class(comfy_config["model_config"])
        model_config.set_inference_dtype(torch.bfloat16, None)
        model_config.custom_operations = None
        model = model_config.get_model({})
        model.diffusion_model = ComfyFluxWrapper(
            transformer,
            config=comfy_config["model_config"],
            ctx_for_copy={
                "comfy_config": comfy_config,
                "model_config": model_config,
                "device": device,
                "device_id": device_id,
            },
        )
        model = comfy.model_patcher.ModelPatcher(model, device, device_id)
        return (model,)
