# SD1.5_rknn_3588_euler

Stable Diffusion 1.5 inference runtime optimized for Rockchip RK3588 / RK3588S using RKNN and Euler scheduler.

This repository contains only code and runtime logic.  
All heavy assets (RKNN models, Super-Resolution, GFPGAN weights) are hosted separately on Hugging Face.

Since it is NOT LCM but Euler- takes much more time to generate -approximately 4-5 minutes , depending on number of steps etc. 
but also high quality images.  Right now only 512x512 generation is possible but can upscale 4X using the built in fucntions. 

For faster but a little not lifelike images LCM method can be used . url : https://docs.radxa.com/en/rock5/rock5b/app-development/rknn_toolkit_lite2_stable-diffusion

---

## Hardware target

- Rockchip RK3588 / RK3588S
- Tested on: Rock 5B / Rock 5B+
- Runtime: RKNN Toolkit Lite 2
- Scheduler: Euler
- Precision: pre-converted RKNN binaries (FP16 / INT8)

---

## What this repository contains

- RKNN-based Stable Diffusion 1.5 runtime
- Command-line interface (rkimg.py)
- Image-to-image and inpainting support
- Real-ESRGAN RKNN upscaler
- Lightweight Flask-based WebUI
- Hugging Face asset downloader

This repository does NOT store large binaries.

---

## Repository structure (after setup)

sd15_rknn_euler/
├── scripts/
│   ├── fetch_assets.py
│   ├── txt2img_rknn_sd15_euler.py
│   ├── img2img_inpaint_unet.py
│   └── upscale_realesrgan_rknn.py
├── models/                         (downloaded from HF)
│   └── business_rknn/
│       └── unet/
│           └── model.rknn
├── model/                          (downloaded from HF)
│   └── sr/realesrgan/
│       └── realesrgan_x4plus_tile128_fp16.rknn
├── gfpgan/
│   └── weights/                    (downloaded from HF)
│       ├── GFPGANv1.4.pth
│       ├── detection_Resnet50_Final.pth
│       └── parsing_parsenet.pth
├── images/
├── outputs/
├── out/
├── webui/
│   ├── app.py
│   ├── templates/index.html
│   └── data/
│       └── masks/
├── rkimg.py
├── requirements.lock.txt
└── README.md

---

## Installation

Clone the repository:

git clone https://github.com/Mojo24x7/SD1.5_rknn_3588_euler
cd SD1.5_rknn_3588_euler

Create and activate virtual environment:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt

---

## Download models (mandatory)

All RKNN models and weights are hosted on Hugging Face.  
You must run this once before using the runtime.

python3 scripts/fetch_assets.py

Assets are downloaded from:

https://huggingface.co/datasets/Mojo24x7/SD1.5_rknn_3588_euler

---

## Usage examples

Text to Image (Euler):

python3 scripts/txt2img_rknn_sd15_euler.py \
  --prompt "a cinematic ultra realistic portrait photo" \
  --steps 30

Image to Image / Inpainting:

python3 scripts/img2img_inpaint_unet.py \
  --init-image images/input.png \
  --mask webui/data/masks/mask.png \
  --prompt "In masked area ONLY: lush green grass" \
  --negative-prompt "snow, white" \
  --steps 30

Upscale (Real-ESRGAN RKNN):

python3 scripts/upscale_realesrgan_rknn.py \
  --rknn model/sr/realesrgan/realesrgan_x4plus_tile128_fp16.rknn \
  --in images/input.png \
  --out outputs/upscaled.png \
  --tile 128 \
  --overlap 64 \
  --scale 4

IMPORTANT: Upscaling works ONLY at x4. Other scales are not supported.

---

## WebUI

python3 webui/app.py

Open in browser:

http://<rock-ip>:7860

---

## Model details

- Base architecture: Stable Diffusion 1.5
- Main checkpoint style: Realistic Vision (SD1.5)
- Scheduler: Euler
- UNet: single-batch RKNN
- Target hardware: RK3588 NPU

---

## Credits and acknowledgements

This project builds upon the work of:

- Stable Diffusion 1.5 by CompVis and Stability AI
- Realistic Vision (SD1.5 checkpoint)
- Real-ESRGAN
- GFPGAN
- RKNN Toolkit and RKNN Toolkit Lite 2 by Rockchip
- Hugging Face Hub

All model weights remain under their original upstream licenses.  
This repository provides pre-converted RKNN binaries for convenience only.

---

## Disclaimer

This project is intended for research, experimentation, and personal use.

The author is not affiliated with Stability AI, CompVis, Rockchip, Real-ESRGAN, GFPGAN, or Hugging Face.
