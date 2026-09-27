"""ComfyUI ModelPatcher clone compatibility for the Z-Image custom node."""

import importlib.util
from pathlib import Path

import comfy.model_management
import torch
from comfy.model_patcher import ModelPatcher

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if not (SOURCE_ROOT / "model_patcher" / "zimage.py").is_file():
    SOURCE_ROOT = Path("/llm/ComfyUI/custom_nodes/ComfyUI-nunchaku-XPU")
spec = importlib.util.spec_from_file_location(
    "_nunchaku_zimage_patcher_under_test",
    SOURCE_ROOT / "model_patcher" / "zimage.py",
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ZImageModelPatcher = module.ZImageModelPatcher


def _patcher(*, fast_disk=False):
    device = torch.device("cpu")
    return ZImageModelPatcher(
        torch.nn.Linear(4, 4),
        device,
        device,
        fast_disk=fast_disk,
    )


def test_clone_accepts_comfyui_fast_disk_and_preserves_backup(monkeypatch):
    monkeypatch.setattr(comfy.model_management.args, "disable_fast_disk", False)
    monkeypatch.setattr(comfy.model_management.args, "fast_disk", False)
    patcher = _patcher(fast_disk=True)
    assert patcher.fast_disk is True
    backup = {"linear.qweight": (torch.ones(1), torch.ones(1))}
    patcher.svdq_backup = backup

    cloned = patcher.clone()

    assert isinstance(cloned, ZImageModelPatcher)
    assert cloned.fast_disk is True
    assert cloned.svdq_backup is backup
    assert cloned.pinned == set()


def test_constructor_accepts_older_base_without_fast_disk(monkeypatch):
    original_init = ModelPatcher.__init__

    def legacy_init(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        original_init(self, model, load_device, offload_device, size, weight_inplace_update=weight_inplace_update)

    monkeypatch.setattr(ModelPatcher, "__init__", legacy_init)
    patcher = _patcher(fast_disk=True)

    cloned = patcher.clone()

    assert isinstance(cloned, ZImageModelPatcher)
    assert cloned.svdq_backup is patcher.svdq_backup
