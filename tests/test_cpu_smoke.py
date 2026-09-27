from types import SimpleNamespace

import pytest
import torch

import nunchaku_torch.zimage as runtime_zimage
from nunchaku_torch import GenerationConfig
from nunchaku_torch.models.linear import SVDQW4A4Linear
from nunchaku_torch.models.transformers import NunchakuZImageTransformer2DModel
from nunchaku_torch.models.transformers import transformer_zimage


def test_generation_config_defaults():
    config = GenerationConfig(
        quant_path="quant.safetensors",
        base_model="base-model",
        prompt="a cat",
    )
    assert config.device == "auto"
    assert config.steps == 9


def test_svdq_from_linear_accepts_comfy_pruned_weight_with_explicit_dtype():
    linear = torch.nn.Linear(16, 32, bias=False)
    linear.register_parameter("weight", None)

    converted = SVDQW4A4Linear.from_linear(
        linear,
        torch_dtype=torch.bfloat16,
    )

    assert converted.in_features == 16
    assert converted.out_features == 32
    assert converted.torch_dtype == torch.bfloat16
    assert converted.qweight.device.type == "cpu"


def test_svdq_from_linear_requires_dtype_for_comfy_pruned_weight():
    linear = torch.nn.Linear(16, 32, bias=False)
    linear.register_parameter("weight", None)

    with pytest.raises(ValueError, match="torch_dtype is required"):
        SVDQW4A4Linear.from_linear(linear)


def test_generate_image_builds_explicit_device_inputs(monkeypatch):
    config = GenerationConfig(
        quant_path="quant.safetensors",
        base_model="base-model",
        prompt="a cat",
        device="cpu",
        height=32,
        width=48,
        steps=2,
        guidance=0.0,
        seed=7,
    )

    class FakePipe:
        def __init__(self):
            _fake_param = torch.zeros(1)  # cpu parameter for offload_text_encoder
            self.transformer = SimpleNamespace(
                config=SimpleNamespace(in_channels=4),
                parameters=lambda: iter([_fake_param]),
            )
            self.text_encoder = SimpleNamespace()  # needed by offload_text_encoder
            self.encode_kwargs = None
            self.latent_kwargs = None
            self.call_kwargs = None

        def encode_prompt(self, **kwargs):
            self.encode_kwargs = kwargs
            return torch.ones(1, 2, 3), None

        def prepare_latents(self, **kwargs):
            self.latent_kwargs = kwargs
            return torch.zeros(1, 4, 4, 4, dtype=kwargs["dtype"])

        def __call__(self, **kwargs):
            self.call_kwargs = kwargs
            return SimpleNamespace(images=["fake-image"])

    fake_pipe = FakePipe()
    monkeypatch.setattr(
        runtime_zimage,
        "load_pipeline",
        lambda cfg: (fake_pipe, torch.device("cpu"), torch.bfloat16),
    )

    image = runtime_zimage.generate_image(config)

    assert image == "fake-image"
    assert fake_pipe.encode_kwargs["prompt"] == "a cat"
    assert fake_pipe.encode_kwargs["device"] == torch.device("cpu")
    assert fake_pipe.encode_kwargs["do_classifier_free_guidance"] is False
    assert fake_pipe.latent_kwargs["device"] == "cpu"
    assert fake_pipe.latent_kwargs["height"] == 32
    assert fake_pipe.latent_kwargs["width"] == 48
    assert fake_pipe.call_kwargs["prompt"] is None
    assert fake_pipe.call_kwargs["generator"] is None
    assert fake_pipe.call_kwargs["guidance_scale"] == 0.0


def test_load_pipeline_uses_vendored_runtime(monkeypatch):
    config = GenerationConfig(
        quant_path="quant.safetensors",
        base_model="base-model",
        prompt="a cat",
        device="cpu",
    )

    fake_transformer = SimpleNamespace(config=SimpleNamespace(in_channels=4))

    class FakePipe:
        def __init__(self):
            self.to_device = None

        def to(self, device: str):
            self.to_device = device
            return self

    fake_pipe = FakePipe()
    captured: dict[str, object] = {}

    def fake_transformer_from_pretrained(path: str, **kwargs):
        captured["transformer_path"] = path
        captured["transformer_kwargs"] = kwargs
        return fake_transformer

    def fake_pipe_from_pretrained(base_model: str, **kwargs):
        captured["base_model"] = base_model
        captured["pipe_kwargs"] = kwargs
        return fake_pipe

    monkeypatch.setattr(
        NunchakuZImageTransformer2DModel,
        "from_pretrained",
        fake_transformer_from_pretrained,
    )
    monkeypatch.setattr(
        runtime_zimage.ZImagePipeline,
        "from_pretrained",
        fake_pipe_from_pretrained,
    )

    original_precision = transformer_zimage.get_precision
    try:
        pipe, device, dtype = runtime_zimage.load_pipeline(config)
    finally:
        transformer_zimage.get_precision = original_precision

    assert pipe is fake_pipe
    assert device == torch.device("cpu")
    assert dtype == torch.bfloat16
    assert captured["transformer_path"] == "quant.safetensors"
    assert captured["transformer_kwargs"] == {"torch_dtype": torch.bfloat16}
    assert captured["base_model"] == "base-model"
    assert captured["pipe_kwargs"]["transformer"] is fake_transformer
    assert fake_pipe.to_device == "cpu"
