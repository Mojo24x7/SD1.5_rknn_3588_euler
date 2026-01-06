#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
img2img_inpaint_unet.py (Euler + classic SD1.5 UNet) — CONTROLLED inpaint/img2img for RKNN

UPDATED (fix pack):
✅ Default --noise is now 1.0 (more correct diffusers-like behavior; avoids "too weak edits")
✅ Tokenizer loader prefers SD1.5 local folder relative to SCRIPT DIR (UI-safe)
✅ Writes DEBUG_mask.png when --mask is provided (verifies alignment/inversion)
✅ Refuses CLIP-L/14 fallback (wrong vocab/merges for SD1.5)
✅ Deterministic RKNN input ordering (prompt no longer ignored)
✅ Diffusers-correct img2img timesteps (strength controls steps_run)
✅ Optional preserve anchoring and debug roundtrip
✅ Masked inpaint blend locks unmasked region to original latents at each timestep

Behavior:
- If --mask is provided => inpaint-like masked edit (white=edit, black=keep)
- If no --mask => normal controlled img2img
"""

import argparse
import json
import logging
import os
import time
from typing import Tuple, Optional

import numpy as np
import torch
import types, torch
if not hasattr(torch, "xpu"):
    torch.xpu = types.SimpleNamespace(
        empty_cache=lambda: None,
        synchronize=lambda *a, **k: None,
        device_count=lambda: 0,
        is_available=lambda: False,
        current_device=lambda: 0,
        manual_seed=lambda *a, **k: None,
        manual_seed_all=lambda *a, **k: None,
    )
from PIL import Image, ImageFilter

from transformers import CLIPTokenizer
from diffusers.schedulers import EulerDiscreteScheduler
from rknnlite.api import RKNNLite

logging.basicConfig()
logger = logging.getLogger("img2img_inpaint_unet")
logger.setLevel(logging.INFO)


def load_sd15_tokenizer():
    """
    MUST match SD1.5 tokenizer.
    Refuse CLIP-ViT-L/14 tokenizer fallback because it breaks vocab/merges vs SD1.5.
    UI-safe: tries script-relative local folder first, then cwd, then HF cache.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # Best: local folder you created (recommended)
        os.path.join(script_dir, "models", "sd15_tokenizer"),
        os.path.join(os.getcwd(), "models", "sd15_tokenizer"),
        # Optional: if you keep a local diffusers repo layout
        os.path.join(script_dir, "models", "runwayml-stable-diffusion-v1-5"),
        os.path.join(os.getcwd(), "models", "runwayml-stable-diffusion-v1-5"),
        # Cached HF repos (offline-friendly if already cached)
        "runwayml/stable-diffusion-v1-5",
        "CompVis/stable-diffusion-v1-4",
    ]

    last_err = None
    for c in candidates:
        try:
            tok = CLIPTokenizer.from_pretrained(c, local_files_only=True)
            logger.info(f"Tokenizer loaded (SD1.5 local/cache): {c}")
            # Sanity: SD1.x tokenizer vocab is typically 49408
            if getattr(tok, "vocab_size", None) not in (49408, 49409):
                logger.warning(f"Tokenizer vocab_size looks unusual: {tok.vocab_size} (expected ~49408).")
            return tok
        except Exception as e:
            last_err = e

    # final fallback: allow download (if internet available)
    for c in ["runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4"]:
        try:
            tok = CLIPTokenizer.from_pretrained(c)
            logger.info(f"Tokenizer loaded (SD1.5 download): {c}")
            return tok
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Could not load SD1.5 tokenizer. "
        "Do NOT use openai/clip-vit-large-patch14 here. "
        f"Last error: {last_err}"
    )


def parse_size(s: str) -> Tuple[int, int]:
    w_s, h_s = s.lower().split("x")
    return int(w_s), int(h_s)


def nchw_to_nhwc(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (0, 2, 3, 1))


def nhwc_to_nchw(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (0, 3, 1, 2))


def as_contig(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a)


class RKNN2Model:
    """
    RKNN wrapper with deterministic input ordering (CRITICAL).
    Handles NCHW<->NHWC consistently for unet/vae models.
    """

    def __init__(self, model_dir: str):
        self.model_dir = os.path.abspath(model_dir)
        self.modelname = os.path.basename(self.model_dir.rstrip("/"))

        cfg_path = os.path.join(self.model_dir, "config.json")
        rknn_path = os.path.join(self.model_dir, "model.rknn")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing {cfg_path}")
        if not os.path.exists(rknn_path):
            raise FileNotFoundError(f"Missing {rknn_path}")

        logger.info(f"Loading {self.model_dir}")
        t0 = time.time()
        self.config = json.load(open(cfg_path, "r"))

        self.rknnlite = RKNNLite()
        self.rknnlite.load_rknn(rknn_path)
        self.rknnlite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

        # In your repo: UNet/VAE want NHWC buffers; text_encoder stays NCHW.
        self.use_nhwc = self.modelname in ("unet", "vae_decoder", "vae_encoder")
        logger.info(f"{self.modelname}: using {'NHWC' if self.use_nhwc else 'NCHW'}")
        logger.info(f"Done. Took {time.time() - t0:.1f}s")

    def release(self):
        try:
            self.rknnlite.release()
        except Exception:
            pass

    def _ordered_inputs(self, kwargs):
        # Deterministic inputs (CRITICAL)
        if self.modelname == "text_encoder":
            return [kwargs["input_ids"]]
        if self.modelname == "unet":
            return [kwargs["sample"], kwargs["timestep"], kwargs["encoder_hidden_states"]]
        if self.modelname == "vae_encoder":
            return [kwargs["sample"]]
        if self.modelname == "vae_decoder":
            return [kwargs["latent_sample"]]
        return list(kwargs.values())

    def __call__(self, **kwargs):
        inputs = self._ordered_inputs(kwargs)
        data_format = "nhwc" if self.use_nhwc else "nchw"

        # If model expects NHWC and first input looks like NCHW (C in {3,4}), convert
        if self.use_nhwc and inputs and isinstance(inputs[0], np.ndarray):
            x0 = inputs[0]
            if x0.ndim == 4 and x0.shape[1] in (3, 4):
                inputs[0] = as_contig(nchw_to_nhwc(x0))

        inputs = [as_contig(x) if isinstance(x, np.ndarray) else x for x in inputs]
        outputs = self.rknnlite.inference(inputs=inputs, data_format=data_format)

        # Convert NHWC outputs back to NCHW when they look image/latent-like
        if self.use_nhwc:
            fixed = []
            for o in outputs:
                if isinstance(o, np.ndarray) and o.ndim == 4 and o.shape[-1] in (3, 4):
                    o = nhwc_to_nchw(o)
                fixed.append(o)
            return fixed

        return outputs


def load_init_image(path: str, width: int, height: int) -> np.ndarray:
    """
    Load init image, center-crop to aspect, resize, output NCHW float32 in [-1,1].
    """
    im = Image.open(path).convert("RGB")
    src_w, src_h = im.size
    target_ratio = width / height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        im = im.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        im = im.crop((0, top, src_w, top + new_h))

    im = im.resize((width, height), resample=Image.LANCZOS)
    arr = np.array(im).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
    arr = (arr * 2.0) - 1.0
    return arr


def load_mask_latents(mask_path: str, width: int, height: int, blur_px: int) -> np.ndarray:
    """
    Load mask as (1,1,H/8,W/8) float32 in [0,1], where 1=EDIT, 0=KEEP.
    White=edit, black=keep.
    """
    m = Image.open(mask_path).convert("L").resize((width, height), resample=Image.LANCZOS)
    if blur_px and blur_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    m_np = (np.array(m).astype(np.float32) / 255.0).clip(0.0, 1.0)

    h8, w8 = height // 8, width // 8
    m_small = Image.fromarray((m_np * 255).astype("uint8")).resize((w8, h8), resample=Image.LANCZOS)
    m_lat = (np.array(m_small).astype(np.float32) / 255.0).clip(0.0, 1.0)
    return m_lat[None, None, ...]  # (1,1,h8,w8)


def denormalize_to_pil(nchw: np.ndarray) -> Image.Image:
    img = np.clip(nchw / 2.0 + 0.5, 0, 1)
    img = (img * 255).round().astype("uint8")
    img = img.transpose(0, 2, 3, 1)  # NCHW->NHWC
    return Image.fromarray(img[0])


def _encode(tokenizer: CLIPTokenizer, text_encoder: RKNN2Model, prompt: str) -> np.ndarray:
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    return text_encoder(input_ids=text_inputs.input_ids.astype(np.int32))[0]  # (1,77,768)


def cfg_rescale(noise_pred: np.ndarray, noise_text: np.ndarray, factor: float) -> np.ndarray:
    """
    Simple CFG rescale: match std of guided pred to std of text pred (per-sample).
    factor in [0..1]: 0 disables, 1 full rescale. Typically 0.2-0.6.
    """
    if factor <= 0:
        return noise_pred
    eps = 1e-6
    std_text = noise_text.std(axis=(1, 2, 3), keepdims=True) + eps
    std_cfg = noise_pred.std(axis=(1, 2, 3), keepdims=True) + eps
    scaled = noise_pred * (std_text / std_cfg)
    return scaled * factor + noise_pred * (1.0 - factor)


def main():
    p = argparse.ArgumentParser(
        description="Inpaint/img2img (mask optional) using SD1.5 UNet + Euler scheduler (RKNN) — controlled"
    )

    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="ugly, deformed, extra fingers, bad anatomy, worst quality, lowres")

    p.add_argument("--init-image", required=True, help="Input image path")
    p.add_argument("--mask", default=None, help="Optional mask path (white=edit, black=keep)")
    p.add_argument("--mask-blur", type=int, default=18, help="Mask feather blur radius in pixels (default: 18)")

    p.add_argument("-i", required=True, help="Base model dir containing text_encoder/unet/vae_encoder/vae_decoder")
    p.add_argument("--unet-dir-name", default="unet", help="Directory name for UNet inside base dir (default: unet)")

    p.add_argument("-o", required=True, help="Output directory")
    p.add_argument("--out-name", default="result_inpaint_euler.png")
    p.add_argument("-s", "--size", default="512x512", help="WIDTHxHEIGHT")

    p.add_argument("--seed", type=int, default=93)

    # UI-friendly total steps
    p.add_argument("--steps", type=int, default=30, help="Total diffusion steps (strength decides steps_run)")
    p.add_argument("--num-inference-steps", type=int, default=None, help="Alias for --steps (compat)")

    p.add_argument("--guidance-scale", type=float, default=6.5)

    # strength/denoise controls
    p.add_argument("--strength", type=float, default=0.75, help="0..1 (higher = more change)")
    p.add_argument("--denoise", type=float, default=None, help="0..1 diffusers-style; overrides --strength")

    # anchoring / randomness controls
    # FIX: default noise to 1.0 (0.6 made edits too weak)
    p.add_argument("--noise", type=float, default=1.0, help="Noise scale injected into init latents. 1.0 is standard; >1 boosts change.")
    p.add_argument("--preserve", type=float, default=0.0, help="0..1 extra latent preservation each step (0 off). 0.08-0.20 helps keep composition.")

    # Euler tweaks
    p.add_argument("--cfg-rescale", type=float, default=0.6, help="0 disables. Try 0.2-0.6")
    p.add_argument("--karras", action="store_true", help="Use Karras sigmas if supported by your diffusers")

    # debug
    p.add_argument("--debug-roundtrip", action="store_true",
                   help="Write DEBUG_roundtrip.png (VAE encode/decode of init) to verify VAE alignment")

    args = p.parse_args()

    base_dir = os.path.abspath(args.i)
    w, h = parse_size(args.size)

    # Model subdirs
    text_encoder_dir = os.path.join(base_dir, "text_encoder")
    unet_dir = os.path.join(base_dir, args.unet_dir_name)
    vae_enc_dir = os.path.join(base_dir, "vae_encoder")
    vae_dec_dir = os.path.join(base_dir, "vae_decoder")

    for d in (text_encoder_dir, unet_dir, vae_enc_dir, vae_dec_dir):
        if not os.path.exists(os.path.join(d, "model.rknn")):
            raise FileNotFoundError(f"Missing model.rknn in {d}")

    # Scheduler: Euler
    try:
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type="epsilon",
            use_karras_sigmas=bool(args.karras),
        )
    except TypeError:
        if args.karras:
            logger.warning("EulerDiscreteScheduler doesn't support use_karras_sigmas; continuing without it.")
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type="epsilon",
        )

    # Load models
    text_encoder = RKNN2Model(text_encoder_dir)
    unet = RKNN2Model(unet_dir)
    vae_encoder = RKNN2Model(vae_enc_dir)
    vae_decoder = RKNN2Model(vae_dec_dir)

    # Warn if UNet looks LCM-ish
    if unet.config.get("time_cond_proj_dim", None) is not None:
        logger.warning(
            "UNet config has time_cond_proj_dim != null (often LCM-style). "
            "Euler works best with classic SD1.5 UNet (time_cond_proj_dim null)."
        )

    tokenizer = load_sd15_tokenizer()

    strength = args.strength if args.denoise is None else args.denoise
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        raise ValueError("--strength/--denoise must be > 0")

    steps = int(args.steps if args.num_inference_steps is None else args.num_inference_steps)
    steps = max(1, steps)

    # Encode prompts
    t0 = time.time()
    prompt_embeds = _encode(tokenizer, text_encoder, args.prompt)
    do_cfg = float(args.guidance_scale) > 1.0
    uncond_embeds = _encode(tokenizer, text_encoder, args.negative_prompt) if do_cfg else None
    logger.info(f"Prompt encode: {time.time() - t0:.2f}s")

    # Init image -> latents
    init = load_init_image(args.init_image, width=w, height=h)
    init_latents = vae_encoder(sample=init)[0].astype(np.float32)

    scaling_factor = float(vae_decoder.config.get("scaling_factor", 0.18215))
    init_latents = init_latents * scaling_factor

    os.makedirs(args.o, exist_ok=True)

    if args.debug_roundtrip:
        rt = init_latents / scaling_factor
        rt_img = vae_decoder(latent_sample=rt)[0]
        denormalize_to_pil(rt_img).save(os.path.join(args.o, "DEBUG_roundtrip.png"))
        logger.info("Wrote DEBUG_roundtrip.png (should resemble init image)")

    # Mask (optional)
    mask_lat: Optional[np.ndarray] = None
    if args.mask:
        mask_lat = load_mask_latents(args.mask, width=w, height=h, blur_px=args.mask_blur)
        mask_lat = np.repeat(mask_lat, 4, axis=1).astype(np.float32)
        logger.info(f"Mask enabled: {args.mask} (blur={args.mask_blur}px)")

        # FIX: write DEBUG_mask.png to verify mask alignment/inversion
        try:
            m = mask_lat[0, 0]  # (h8,w8)
            m_img = Image.fromarray((m * 255.0).clip(0, 255).astype("uint8")).resize((w, h), Image.NEAREST)
            m_img.save(os.path.join(args.o, "DEBUG_mask.png"))
            logger.info("Wrote DEBUG_mask.png")
        except Exception as e:
            logger.warning(f"Could not write DEBUG_mask.png: {e}")

    # Diffusers img2img timesteps
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps  # length=steps

    init_timestep = int(steps * strength)
    init_timestep = max(1, min(init_timestep, steps))
    t_start = steps - init_timestep
    timesteps = timesteps[t_start:]
    logger.info(f"Schedule: steps(total)={steps}, strength={strength:.3f}, steps_run={len(timesteps)}")

    # Seeded noise (constant)
    rng = np.random.RandomState(args.seed)
    noise = rng.randn(*init_latents.shape).astype(np.float32) * float(args.noise)

    # Initial noisy latents
    t0_tensor = timesteps[0:1]
    latents = scheduler.add_noise(
        torch.from_numpy(init_latents),
        torch.from_numpy(noise),
        t0_tensor,
    ).numpy().astype(np.float32)

    preserve = float(np.clip(args.preserve, 0.0, 1.0))

    unet_total = 0.0
    mask_blend_total = 0.0
    preserve_total = 0.0

    for _, t in enumerate(timesteps):
        tt = int(t.item()) if hasattr(t, "item") else int(t)

        latents_in = scheduler.scale_model_input(torch.from_numpy(latents), t).numpy().astype(np.float32)
        timestep = np.array([tt], dtype=np.int64)

        if do_cfg:
            tA = time.time()
            noise_uncond = unet(sample=latents_in, timestep=timestep, encoder_hidden_states=uncond_embeds)[0]
            noise_text = unet(sample=latents_in, timestep=timestep, encoder_hidden_states=prompt_embeds)[0]
            unet_total += (time.time() - tA)

            noise_pred = noise_uncond + float(args.guidance_scale) * (noise_text - noise_uncond)
            if args.cfg_rescale and args.cfg_rescale > 0:
                noise_pred = cfg_rescale(noise_pred, noise_text, float(args.cfg_rescale))
        else:
            tA = time.time()
            noise_pred = unet(sample=latents_in, timestep=timestep, encoder_hidden_states=prompt_embeds)[0]
            unet_total += (time.time() - tA)

        latents_next = scheduler.step(
            torch.from_numpy(noise_pred),
            t,
            torch.from_numpy(latents),
            return_dict=False,
        )[0].numpy().astype(np.float32)

        # Extra anchoring (optional)
        if preserve > 0:
            tp = time.time()
            latents_orig = scheduler.add_noise(
                torch.from_numpy(init_latents),
                torch.from_numpy(noise),
                t.view(1) if hasattr(t, "view") else torch.tensor([t]),
            ).numpy().astype(np.float32)
            latents_next = latents_next * (1.0 - preserve) + latents_orig * preserve
            preserve_total += (time.time() - tp)

        # Masked blend (if mask provided)
        if mask_lat is not None:
            tb = time.time()
            latents_orig = scheduler.add_noise(
                torch.from_numpy(init_latents),
                torch.from_numpy(noise),
                t.view(1) if hasattr(t, "view") else torch.tensor([t]),
            ).numpy().astype(np.float32)

            inv = (1.0 - mask_lat)
            latents_next = latents_orig * inv + latents_next * mask_lat
            mask_blend_total += (time.time() - tb)

        latents = latents_next

    logger.info(
        f"UNet total: {unet_total:.2f}s steps_run={len(timesteps)}"
        + (f" | mask_blend={mask_blend_total:.2f}s" if mask_lat is not None else "")
        + (f" | preserve={preserve_total:.2f}s (preserve={preserve:.2f})" if preserve > 0 else "")
    )

    # Decode
    latents = latents / scaling_factor
    img_nchw = vae_decoder(latent_sample=latents)[0]
    out_img = denormalize_to_pil(img_nchw)

    out_path = os.path.join(args.o, args.out_name)
    out_img.save(out_path)
    logger.info(f"Saved: {out_path}")

    for m in (text_encoder, unet, vae_encoder, vae_decoder):
        m.release()


if __name__ == "__main__":
    main()
