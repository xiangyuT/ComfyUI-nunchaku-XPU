import logging
import os
from pathlib import Path

import torch
import yaml
from packaging.version import InvalidVersion, Version

from nunchaku_torch.device import has_xpu
from nunchaku_torch.utils import get_precision

# vanilla and LTS compatibility snippet
try:
    from comfy_compatibility.vanilla import prepare_vanilla_environment

    prepare_vanilla_environment()

    from comfy.model_downloader import add_known_models
    from comfy.model_downloader_types import HuggingFile

    precision = get_precision()

    # add known models

    models_yaml_path = Path(__file__).parent / "test_data" / "models.yaml"
    with open(models_yaml_path, "r") as f:
        nunchaku_models_yaml = yaml.safe_load(f)

    NUNCHAKU_SVDQ_MODELS = []
    for model in nunchaku_models_yaml["models"]:
        filename = model["filename"]
        if not filename.startswith("svdq-"):
            continue
        if "{precision}" in filename:
            filename = filename.format(precision=precision)
        NUNCHAKU_SVDQ_MODELS.append(HuggingFile(repo_id=model["repo_id"], filename=filename))

    NUNCHAKU_SVDQ_TEXT_ENCODER_MODELS = [
        HuggingFile(repo_id="nunchaku-tech/nunchaku-t5", filename="awq-int4-flux.1-t5xxl.safetensors"),
    ]

    add_known_models("diffusion_models", *NUNCHAKU_SVDQ_MODELS)
    add_known_models("text_encoders", *NUNCHAKU_SVDQ_TEXT_ENCODER_MODELS)
except (ImportError, ModuleNotFoundError):
    pass

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-nunchaku-XPU Initialization " + "=" * 40)

logger.info(f"XPU available: {has_xpu()}")

from .utils import get_plugin_version

try:
    import nunchaku_torch
    logger.info(f"nunchaku_torch version: {nunchaku_torch.__version__}")
except ImportError:
    logger.warning("nunchaku_torch not found - install from analytics-zoo/nunchaku-torch")

logger.info(f"ComfyUI-nunchaku-XPU version: {get_plugin_version()}")

NODE_CLASS_MAPPINGS = {}

try:
    from .nodes.models.flux import NunchakuFluxDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"] = NunchakuFluxDiTLoader
except ImportError:
    logger.exception("Node `NunchakuFluxDiTLoader` import failed:")

try:
    from .nodes.models.qwenimage import NunchakuQwenImageDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuQwenImageDiTLoader"] = NunchakuQwenImageDiTLoader
except ImportError:
    logger.exception("Node `NunchakuQwenImageDiTLoader` import failed:")

try:
    from .nodes.lora.flux import NunchakuFluxLoraLoader, NunchakuFluxLoraStack

    NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"] = NunchakuFluxLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuFluxLoraStack"] = NunchakuFluxLoraStack
except ImportError:
    logger.exception("Nodes `NunchakuFluxLoraLoader` and `NunchakuFluxLoraStack` import failed:")

try:
    from .nodes.models.text_encoder import NunchakuTextEncoderLoader, NunchakuTextEncoderLoaderV2

    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoader"] = NunchakuTextEncoderLoader
    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoaderV2"] = NunchakuTextEncoderLoaderV2
except ImportError:
    logger.exception("Nodes `NunchakuTextEncoderLoader` and `NunchakuTextEncoderLoaderV2` import failed:")

try:
    from .nodes.preprocessors.depth import FluxDepthPreprocessor

    NODE_CLASS_MAPPINGS["NunchakuDepthPreprocessor"] = FluxDepthPreprocessor
except ImportError:
    logger.exception("Node `NunchakuDepthPreprocessor` import failed:")

try:
    from .nodes.models.pulid import (
        NunchakuFluxPuLIDApplyV2,
        NunchakuPulidApply,
        NunchakuPulidLoader,
        NunchakuPuLIDLoaderV2,
    )

    NODE_CLASS_MAPPINGS["NunchakuPulidApply"] = NunchakuPulidApply
    NODE_CLASS_MAPPINGS["NunchakuPulidLoader"] = NunchakuPulidLoader
    NODE_CLASS_MAPPINGS["NunchakuPuLIDLoaderV2"] = NunchakuPuLIDLoaderV2
    NODE_CLASS_MAPPINGS["NunchakuFluxPuLIDApplyV2"] = NunchakuFluxPuLIDApplyV2
except ImportError:
    logger.exception(
        "Nodes `NunchakuPulidApply`,`NunchakuPulidLoader`, "
        "`NunchakuPuLIDLoaderV2` and `NunchakuFluxPuLIDApplyV2` import failed:"
    )
try:
    from .nodes.models.ipadapter import NunchakuFluxIPAdapterApply, NunchakuIPAdapterLoader

    NODE_CLASS_MAPPINGS["NunchakuFluxIPAdapterApply"] = NunchakuFluxIPAdapterApply
    NODE_CLASS_MAPPINGS["NunchakuIPAdapterLoader"] = NunchakuIPAdapterLoader
except ImportError:
    logger.exception("Nodes `NunchakuFluxIPAdapterApply` and `NunchakuIPAdapterLoader` import failed:")

try:
    from .nodes.models.zimage import NunchakuZImageDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuZImageDiTLoader"] = NunchakuZImageDiTLoader
except ImportError:
    logger.exception("Nodes `NunchakuZImageDiTLoader` import failed:")

try:
    from .nodes.tools.merge_safetensors import NunchakuModelMerger

    NODE_CLASS_MAPPINGS["NunchakuModelMerger"] = NunchakuModelMerger
except ImportError:
    logger.exception("Node `NunchakuModelMerger` import failed:")

try:
    from .nodes.tools.installers import NunchakuWheelInstaller

    NODE_CLASS_MAPPINGS["NunchakuWheelInstaller"] = NunchakuWheelInstaller
except ImportError:
    logger.exception("Node `NunchakuWheelInstaller` import failed:")

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
logger.info("=" * (80 + len(" ComfyUI-nunchaku-XPU Initialization ")))
