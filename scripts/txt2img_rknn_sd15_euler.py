#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
txt2img_rknn_sd15_euler_b2_optional.py

SCRIPT_VERSION = "2026-01-05a"

Behavior:
- Prefer unet_b2_static/ if present (static-shape batch2 CFG in one call)
- Else use unet_b2/ if present (dynamic-shape batch2 CFG in one call)
- Else use legacy unet/ (two calls per step for CFG)

New (safe, default-off):
- --cfg-last-n N : if N>0, run CFG only on the last N steps (legacy UNet path).
  Early steps use 1 UNet call/step (text only), last N steps use full CFG (2 calls/step).

Progress UX:
- tqdm progress bar with per-step UNet timing in postfix (can disable via --no-progress).
"""

import argparse
import json
import time
import logging
import os
from typing import Optional, Tuple
from tqdm.auto import tqdm

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from transformers import CLIPTokenizer

from rknnlite.api import RKNNLite

SCRIPT_VERSION = "2026-01-05a"

logging.basicConfig()
logger = logging.getLogger("sd15_rknn_timed")
logger.setLevel(logging.INFO)


def load_sd15_tokenizer():
    """
    Load the tokenizer that MATCHES SD1.5 text_encoder.
    Tries local common locations first (offline-friendly).
    """
    candidates = [
        os.path.join(os.getcwd(), "models", "sd15_tokenizer"),
        os.path.join(os.getcwd(), "tokenizer"),
        "runwayml/stable-diffusion-v1-5",
        "CompVis/stable-diffusion-v1-4",
        "openai/clip-vit-large-patch14",
    ]

    last_err = None
    for c in candidates:
        try:
            tok = CLIPTokenizer.from_pretrained(c, local_files_only=True)
            logger.info(f"Tokenizer loaded (local): {c}")
            return tok
        except Exception as e:
            last_err = e

    for c in ["runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4", "openai/clip-vit-large-patch14"]:
        try:
            tok = CLIPTokenizer.from_pretrained(c)
            logger.info(f"Tokenizer loaded (download): {c}")
            return tok
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not load SD tokenizer. Last error: {last_err}")


class Timings:
    def __init__(self):
        self.t = {}
        self._start = {}

    def start(self, k):
        self._start[k] = time.time()

    def stop(self, k):
        dt = time.time() - self._start.get(k, time.time())
        self.t[k] = self.t.get(k, 0.0) + dt
        return dt

    def set(self, k, v):
        self.t[k] = float(v)

    def report(self):
        keys = [
            "load_text_encoder",
            "load_unet",
            "load_unet_b2",
            "load_vae_decoder",
            "encode_prompt",
            "encode_negative",
            "prepare_latents",
            "unet_total",
            "scheduler_total",
            "vae_decode",
            "postprocess",
            "total",
        ]
        print("\n========== TIMING REPORT ==========")
        print(f"{'script_version':>18s}: {SCRIPT_VERSION}")
        for k in keys:
            if k in self.t:
                print(f"{k:>18s}: {self.t[k]:8.3f} s")
        print("===================================\n")


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
    def __init__(self, model_dir, timings: Timings = None):
        self.timings = timings
        self.modelname = os.path.basename(model_dir)

        logger.info(f"Loading {model_dir}")
        t0 = time.time()

        cfg_path = os.path.join(model_dir, "config.json")
        rknn_path = os.path.join(model_dir, "model.rknn")
        assert os.path.exists(cfg_path), f"Missing {cfg_path}"
        assert os.path.exists(rknn_path), f"Missing {rknn_path}"

        self.config = json.load(open(cfg_path))

        self.rknnlite = RKNNLite()
        self.rknnlite.load_rknn(rknn_path)
        self.rknnlite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

        dt = time.time() - t0
        logger.info(f"Done. Took {dt:.1f}s")

        if self.timings:
            if self.modelname == "text_encoder":
                self.timings.set("load_text_encoder", dt)
            elif self.modelname == "unet":
                self.timings.set("load_unet", dt)
            elif self.modelname in ("unet_b2", "unet_b2_static"):
                self.timings.set("load_unet_b2", dt)
            elif self.modelname == "vae_decoder":
                self.timings.set("load_vae_decoder", dt)

    def release(self):
        try:
            self.rknnlite.release()
        except Exception:
            pass

    def _ordered_inputs(self, kwargs):
        if self.modelname == "text_encoder":
            return [kwargs["input_ids"]]
        if self.modelname == "unet":
            return [kwargs["sample"], kwargs["timestep"], kwargs["encoder_hidden_states"]]
        if self.modelname in ("unet_b2", "unet_b2_static"):
            return [kwargs["sample"], kwargs["timestep_4d"], kwargs["encoder_hidden_states_4d"]]
        if self.modelname == "vae_decoder":
            return [kwargs["latent_sample"]]
        return list(kwargs.values())

    def __call__(self, **kwargs):
        inputs = self._ordered_inputs(kwargs)

        data_format = "nchw"

        # Legacy: unet/vae -> transpose first 4D input to NHWC
        if self.modelname in ("unet", "vae_decoder", "vae_encoder"):
            if inputs and isinstance(inputs[0], np.ndarray):
                x = inputs[0]
                if x.ndim == 4 and x.shape[1] in (3, 4):
                    inputs[0] = as_contig(nchw_to_nhwc(x))
                    data_format = "nhwc"

        # Batch2 models: we feed NHWC already
        if self.modelname in ("unet_b2", "unet_b2_static"):
            data_format = "nhwc"

        inputs = [as_contig(x) if isinstance(x, np.ndarray) else x for x in inputs]
        outputs = self.rknnlite.inference(inputs=inputs, data_format=data_format)

        fixed = []
        for o in outputs:
            if isinstance(o, np.ndarray) and o.ndim == 4 and o.shape[-1] in (3, 4):
                o = nhwc_to_nchw(o)
            fixed.append(o)
        return fixed


def pick_unet_mode(
    base_model_dir: str,
    force_legacy: bool,
    force_b2: bool,
    prefer_b2_static: bool,
    force_b2_static: bool,
):
    base_model_dir = os.path.abspath(base_model_dir)

    unet_dir = os.path.join(base_model_dir, "unet")
    unet_rknn = os.path.join(unet_dir, "model.rknn")

    b2s_dir = os.path.join(base_model_dir, "unet_b2_static")
    b2s_rknn = os.path.join(b2s_dir, "model.rknn")

    b2_dir = os.path.join(base_model_dir, "unet_b2")
    b2_rknn = os.path.join(b2_dir, "model.rknn")

    if not os.path.exists(unet_rknn):
        raise RuntimeError(f"Missing legacy UNet: {unet_rknn}")

    has_b2s = os.path.exists(b2s_rknn)
    has_b2 = os.path.exists(b2_rknn)

    if force_legacy:
        logger.warning("force_legacy=True -> using legacy unet (batch1)")
        return unet_dir, (b2s_dir if has_b2s else (b2_dir if has_b2 else None)), False

    if force_b2_static:
        if not has_b2s:
            raise RuntimeError(f"--force-b2-static set but missing {b2s_rknn}")
        logger.info(f"Using unet_b2_static: {b2s_rknn}")
        return None, b2s_dir, True

    if prefer_b2_static and has_b2s:
        logger.info(f"Found unet_b2_static: {b2s_rknn} -> will use it (and NOT load legacy unet)")
        return None, b2s_dir, True

    if has_b2:
        logger.info(f"Found unet_b2: {b2_rknn} -> will use it (and NOT load legacy unet)")
        return None, b2_dir, True

    if force_b2:
        raise RuntimeError("--force-b2 set but neither unet_b2_static nor unet_b2 found")

    logger.info("No unet_b2* found -> using legacy unet")
    return unet_dir, None, False


class SD15StaticCFGEulerPipeline(DiffusionPipeline):
    def __init__(
        self,
        text_encoder: RKNN2Model,
        unet: Optional[RKNN2Model],
        unet_b2: Optional[RKNN2Model],
        use_unet_b2: bool,
        scheduler,
        tokenizer,
        timings: Timings,
        vae_dir: str,
        show_progress: bool = True,
    ):
        super().__init__()
        self.register_modules(tokenizer=tokenizer, scheduler=scheduler)

        self.text_encoder = text_encoder
        self.unet = unet
        self.unet_b2 = unet_b2
        self.use_unet_b2 = bool(use_unet_b2 and (unet_b2 is not None))

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.timings = timings

        self.vae_dir = vae_dir
        self.vae_decoder: Optional[RKNN2Model] = None

        self.show_progress = bool(show_progress)

    def _encode(self, prompt: str, tag: str) -> np.ndarray:
        self.timings.start(tag)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        out = self.text_encoder(input_ids=text_inputs.input_ids.astype(np.int32))[0]
        self.timings.stop(tag)
        return out  # (1,77,768)

    def prepare_latents(self, height, width, dtype, generator):
        self.timings.start("prepare_latents")
        cfg = self.unet_b2.config if self.use_unet_b2 else self.unet.config
        shape = (1, int(cfg["in_channels"]), height // 8, width // 8)
        latents = generator.randn(*shape).astype(dtype)
        latents *= float(self.scheduler.init_noise_sigma)
        self.timings.stop("prepare_latents")
        return latents

    @torch.no_grad()
    def denoise(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        generator,
        cfg_last_n: int = 0,
        print_each_step: bool = False,
    ) -> np.ndarray:
        # global CFG intent
        do_cfg = guidance_scale > 1.0

        prompt_embeds = self._encode(prompt, "encode_prompt")
        uncond_embeds = self._encode(negative_prompt or "", "encode_negative") if do_cfg else None

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        total_steps = len(timesteps)

        latents = self.prepare_latents(height, width, np.float32, generator)

        unet_calls = 0
        unet_total = 0.0
        sched_total = 0.0

        # rolling stats
        last_unet_dt = 0.0

        pbar = tqdm(
            enumerate(timesteps),
            total=total_steps,
            desc="Denoising",
            unit="step",
            dynamic_ncols=True,
            disable=not self.show_progress,
        )

        # sanitize cfg_last_n
        try:
            cfg_last_n = int(cfg_last_n)
        except Exception:
            cfg_last_n = 0
        if cfg_last_n < 0:
            cfg_last_n = 0
        if cfg_last_n > total_steps:
            cfg_last_n = total_steps

        for step_idx, t in pbar:
            t_i = int(t)
            latents_in = self.scheduler.scale_model_input(torch.from_numpy(latents), t).numpy()

            # per-step CFG activation:
            # - if cfg_last_n == 0: legacy behavior (CFG everywhere)
            # - else: CFG only in last cfg_last_n steps
            cfg_this_step = do_cfg and (cfg_last_n == 0 or step_idx >= (total_steps - cfg_last_n))

            if self.use_unet_b2:
                # For b2 CFG, it is already "one call". If user requests cfg_last_n,
                # we keep behavior stable (still uses the b2 path).
                # (We do NOT try to run "text-only" b2 because these exported b2 models
                # typically expect batch=2 signatures.)
                if do_cfg:
                    lat_b2_nchw = np.concatenate([latents_in, latents_in], axis=0).astype(np.float32)  # (2,4,H,W)
                    lat_b2_nhwc = as_contig(nchw_to_nhwc(lat_b2_nchw))  # (2,H,W,4)

                    enc_b2 = np.concatenate([uncond_embeds, prompt_embeds], axis=0).astype(np.float32)  # (2,77,768)
                    enc_4d = as_contig(np.transpose(enc_b2, (0, 2, 1))[:, :, None, :])  # (2,768,1,77)

                    timestep_4d = as_contig(np.full((2, 1, 1, 1), t_i, dtype=np.int64))

                    t0 = time.time()
                    noise_b2 = self.unet_b2(
                        sample=lat_b2_nhwc,
                        timestep_4d=timestep_4d,
                        encoder_hidden_states_4d=enc_4d,
                    )[0]
                    last_unet_dt = time.time() - t0
                    unet_total += last_unet_dt
                    unet_calls += 1

                    noise_uncond = noise_b2[0:1]
                    noise_text = noise_b2[1:2]
                    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
                else:
                    # no cfg: cannot use b2 model reliably; require legacy unet for non-cfg.
                    if self.unet is None:
                        raise RuntimeError("No-CFG requested but legacy unet is not loaded (using unet_b2*).")
                    timestep = np.array([t_i], dtype=np.int64)
                    t0 = time.time()
                    noise_pred = self.unet(sample=latents_in, timestep=timestep, encoder_hidden_states=prompt_embeds)[0]
                    last_unet_dt = time.time() - t0
                    unet_total += last_unet_dt
                    unet_calls += 1
            else:
                # legacy UNet path
                if self.unet is None:
                    raise RuntimeError("Legacy unet is None but unet_b2 not active.")
                timestep = np.array([t_i], dtype=np.int64)

                if cfg_this_step:
                    # full CFG (2 calls)
                    t0 = time.time()
                    noise_uncond = self.unet(sample=latents_in, timestep=timestep, encoder_hidden_states=uncond_embeds)[0]
                    dt1 = time.time() - t0
                    unet_total += dt1
                    unet_calls += 1

                    t0 = time.time()
                    noise_text = self.unet(sample=latents_in, timestep=timestep, encoder_hidden_states=prompt_embeds)[0]
                    dt2 = time.time() - t0
                    unet_total += dt2
                    unet_calls += 1

                    last_unet_dt = dt1 + dt2
                    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

                    if print_each_step:
                        print(f"[step {step_idx:02d}] UNet CFG on  dt={last_unet_dt:.3f}s (uncond {dt1:.3f}s + text {dt2:.3f}s)")
                else:
                    # text-only (1 call) – THIS is the speed win
                    t0 = time.time()
                    noise_pred = self.unet(sample=latents_in, timestep=timestep, encoder_hidden_states=prompt_embeds)[0]
                    last_unet_dt = time.time() - t0
                    unet_total += last_unet_dt
                    unet_calls += 1

                    if print_each_step:
                        print(f"[step {step_idx:02d}] UNet CFG off dt={last_unet_dt:.3f}s (text-only)")

            # scheduler step
            t0 = time.time()
            latents = self.scheduler.step(
                torch.from_numpy(noise_pred),
                t,
                torch.from_numpy(latents),
                return_dict=False,
            )[0].numpy()
            sched_total += (time.time() - t0)

            # progress postfix
            if self.show_progress:
                avg = (unet_total / unet_calls) if unet_calls else 0.0
                pbar.set_postfix_str(
                    f"CFG:{'on' if (do_cfg and (self.use_unet_b2 or cfg_this_step)) else 'off'} "
                    f"last_unet:{last_unet_dt:.2f}s avg_unet:{avg:.2f}s calls:{unet_calls}"
                )

        self.timings.set("unet_total", unet_total)
        self.timings.set("scheduler_total", sched_total)

        if unet_calls:
            print(f"\nUNet calls: {unet_calls}  |  avg per call: {unet_total/unet_calls:.3f}s")

        return latents

    def free_unet_memory(self):
        if self.unet_b2 is not None:
            self.unet_b2.release()
            self.unet_b2 = None
        if self.unet is not None:
            self.unet.release()
            self.unet = None

    def decode_latents(self, latents: np.ndarray):
        if self.vae_decoder is None:
            self.vae_decoder = RKNN2Model(self.vae_dir, timings=self.timings)

        self.timings.start("vae_decode")
        latents = latents / float(self.vae_decoder.config.get("scaling_factor", 0.18215))
        image = self.vae_decoder(latent_sample=latents)[0]
        self.timings.stop("vae_decode")
        return image


def build_scheduler(args):
    SchedulerCls = EulerDiscreteScheduler
    if getattr(args, "sampler", "euler") == "euler_a":
        SchedulerCls = EulerAncestralDiscreteScheduler

    scheduler = SchedulerCls(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
    )
    return scheduler


def main(args):
    timings = Timings()
    scheduler = build_scheduler(args)

    base_dir = os.path.abspath(args.i)
    vae_dir = os.path.join(base_dir, "vae_decoder")

    text_encoder = RKNN2Model(os.path.join(base_dir, "text_encoder"), timings=timings)

    unet_dir, b2_dir, use_unet_b2 = pick_unet_mode(
        base_dir,
        force_legacy=args.force_legacy_unet,
        force_b2=args.force_b2,
        prefer_b2_static=args.prefer_b2_static,
        force_b2_static=args.force_b2_static,
    )

    unet = RKNN2Model(unet_dir, timings=timings) if unet_dir else None
    unet_b2 = RKNN2Model(b2_dir, timings=timings) if (use_unet_b2 and b2_dir) else None

    pipe = SD15StaticCFGEulerPipeline(
        text_encoder=text_encoder,
        unet=unet,
        unet_b2=unet_b2,
        use_unet_b2=use_unet_b2,
        scheduler=scheduler,
        tokenizer=load_sd15_tokenizer(),
        timings=timings,
        vae_dir=vae_dir,
        show_progress=(not args.no_progress),
    )

    w, h = parse_size(args.size)
    t0_total = time.time()

    latents = pipe.denoise(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=h,
        width=w,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=np.random.RandomState(args.seed),
        cfg_last_n=args.cfg_last_n,
        print_each_step=args.print_each_step,
    )

    pipe.free_unet_memory()
    image = pipe.decode_latents(latents)

    timings.start("postprocess")
    image = (image / 2 + 0.5).clip(0, 1)
    image = image.transpose(0, 2, 3, 1)
    images = [Image.fromarray((image[0] * 255).astype("uint8"))]
    timings.stop("postprocess")

    timings.set("total", time.time() - t0_total)

    os.makedirs(args.o, exist_ok=True)
    out_path = os.path.join(args.o, args.out_name)
    images[0].save(out_path)
    logger.info(f"Saved image to {out_path}")

    timings.report()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="ugly,deformed")
    p.add_argument("-i", required=True, help="Base dir: text_encoder/, unet/, vae_decoder/, optional unet_b2*/")
    p.add_argument("-o", required=True)
    p.add_argument("--out-name", default="result_sd15_euler.png")
    p.add_argument("--seed", type=int, default=93)
    p.add_argument("-s", "--size", default="512x512", help="WIDTHxHEIGHT")
    p.add_argument("--num-inference-steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--print-each-step", action="store_true")

    # UX
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")

    # NEW speed feature (default 0 keeps legacy behavior)
    p.add_argument(
        "--cfg-last-n",
        type=int,
        default=10,
        help="If >0: apply CFG only on last N steps (legacy UNet path). 0 = CFG on all steps (default 10).",
    )

    # sampler choice
    p.add_argument("--sampler", choices=["euler", "euler_a"], default="euler",
                   help="Denoiser sampler: euler (default) or euler_a (more texture/realism)")

    # existing behavior
    p.add_argument("--force-legacy-unet", dest="force_legacy_unet", action="store_true")
    p.add_argument("--force-b2", action="store_true", help="Must use a batch2 model (static or dynamic).")

    # new options
    p.add_argument("--prefer-b2-static", dest="prefer_b2_static", action="store_true", default=True,
                   help="Prefer unet_b2_static/ over unet_b2/ if present (default: on).")
    p.add_argument("--no-prefer-b2-static", dest="prefer_b2_static", action="store_false",
                   help="Disable preference for unet_b2_static/.")
    p.add_argument("--force-b2-static", action="store_true",
                   help="Must use unet_b2_static/ (fail if missing).")

    args = p.parse_args()
    main(args)
