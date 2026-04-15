# Intel XPU Support

This fork adds Intel XPU (Arc GPU) support via [omni_xpu_kernel](https://github.com/intel/llm-scaler) ESIMD/oneDNN kernels and the [nunchaku-torch](https://github.com/xiangyuT/nunchaku-torch) runtime.

## Supported Workflows

All sample images generated on **Intel Arc B580** (11 GB VRAM), 512x512, prompt: *"a cute cat sitting on a windowsill, highly detailed, 4k photography"*.

| Z-Image Turbo (9 steps) | FLUX.1-schnell (4 steps) | Qwen-Image Lightning (4 steps) |
|:---:|:---:|:---:|
| ![Z-Image](images/zimage_xpu_sample.png) | ![FLUX](images/flux_xpu_sample.png) | ![QwenImage](images/qwenimage_xpu_sample.png) |

## Requirements

- Intel Arc GPU (A-series or B-series) with XPU support
- PyTorch with XPU backend (`torch.xpu`)
- [omni_xpu_kernel](https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel) for ESIMD/oneDNN INT4 GEMM kernels
- [nunchaku-torch](https://github.com/xiangyuT/nunchaku-torch) runtime package

## Text Encoders

| Model | Text Encoder | Loader Node |
|-------|-------------|-------------|
| Z-Image Turbo | `Qwen3-4B-Q4_K_M.gguf` | CLIPLoaderGGUF, type=lumina2 |
| FLUX.1-schnell | `clip_l.safetensors` + `t5-v1_1-xxl-encoder-Q4_K_M.gguf` | DualCLIPLoaderGGUF, type=flux |
| Qwen-Image | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | CLIPLoader, type=qwen_image |

## VAE

| Model | VAE File |
|-------|----------|
| Z-Image / FLUX | `ae.safetensors` |
| Qwen-Image | `qwen_image_vae.safetensors` ([download](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors)) |
