#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""rkimg.py

A thin CLI wrapper around the repo scripts.

Key design decisions (aligned with your requirements):
- **No LCM**: generation uses the Euler pipeline script.
- **Steps mean what users think**: for img2img, the value you pass as --steps
  is the *actual number of denoising steps executed*. Internally, the script
  computes the scheduler's total steps to achieve that.
- **No automatic strength/denoise guessing from prompt**.
- **Negative prompt is opt-in**: nothing is injected unless the user supplies
  --negative-prompt (or you set a preset that includes it).
- Keep things ControlNet-compatible: rkimg.py just passes flags through; the
  actual ControlNet integration belongs inside the pipeline scripts.
"""

import argparse
import math
import shutil
import subprocess
import sys
                        
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# Always use the classic/business RKNN model set for SD1.5 Euler
BUSINESS_MODELS_DIR = ROOT / "models" / "business_rknn"

# -------------------------
# EDIT (img2img) presets
# -------------------------
# These are *sane defaults* for typical tasks.
# You can always override with explicit flags.
EDIT_PRESETS = {
    # Minimal structural change, good for recolor / small edits.
    "recolor": {
        "denoise": 0.28,
        "noise": 0.25,
        "guidance": 6.5,
        "cfg_rescale": 0.45,
        "karras": True,
    },
    # Medium change, good for accessories / clothes tweaks (usually with a mask).
    "outfit": {
        "denoise": 0.55,
        "noise": 0.60,
        "guidance": 7.5,
        "cfg_rescale": 0.40,
        "karras": True,
    },
    # Heavy change (still Euler). Expect more geometry drift.
    "strong": {
        "denoise": 0.75,
        "noise": 0.80,
        "guidance": 7.5,
        "cfg_rescale": 0.35,
        "karras": True,
    },
}

                   
                                                                       
                                                                         
                                                                            
  
                   
                                                                             
                                                                             
  

# -------------------------
# GENERATE (txt2img) styles
# -------------------------
GEN_STYLES = {
    "balanced": {"steps": 25, "guidance": 7.0, "suffix": ""},
                 
                   
                        
                     
      
                                              
    "photoreal": {
        "steps": 28,
        "guidance": 7.5,
        "suffix": (
            "ultra realistic photograph, natural lighting, real world photo, "
            "high dynamic range, detailed textures, sharp focus, DSLR photo"
                                                        
        ),
    },
                                       
    "cinematic": {
        "steps": 30,
        "guidance": 8.0,
        "suffix": (
            "cinematic photograph, dramatic lighting, ultra realistic, "
            "highly detailed, depth of field, 35mm lens"
                                        
        ),
    },
                                           
    "creative": {
        "steps": 26,
        "guidance": 7.0,
                   
        "suffix": "beautiful composition, artistic, stylized, highly detailed",
          
    },
}

# -------------------------
# helpers
# -------------------------
_HELP_CACHE: dict[str, str] = {}


def run(cmd: list[str]):
    print("[rkimg]", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)

                                                
                              
                                         

                                                                                      
                                                                    
                                                                   
                                                        

                     
                                  
                                            
                            
                                   
                                            
                                   
                                   

               

def _pick_newest_png(out_dir: Path, since_ts: float | None = None) -> Path | None:
    newest = None
    newest_mtime = -1
    for p in out_dir.rglob("*.png"):
        try:
            m = p.stat().st_mtime
        except FileNotFoundError:
            continue
        if since_ts is not None and m < since_ts:
            continue
        if m > newest_mtime:
            newest_mtime = m
            newest = p
    return newest


def _force_output_png(out_path: Path, out_dir: Path, before_ts: float):
    newest = _pick_newest_png(out_dir, since_ts=before_ts) or _pick_newest_png(out_dir)
                      
                                                         
    if newest is None:
        raise SystemExit(f"[rkimg] No PNG output found under: {out_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(newest, out_path)
    print(f"[rkimg] saved: {out_path}  (from {newest})")


def _script_help(script_path: Path) -> str:
    key = str(script_path)
    if key in _HELP_CACHE:
        return _HELP_CACHE[key]
    try:
        cp = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        txt = cp.stdout or ""
    except Exception:
        txt = ""
    _HELP_CACHE[key] = txt
    return txt


def _supports(script_path: Path, flag: str) -> bool:
    return flag in _script_help(script_path)


def _maybe_add(cmd: list[str], script_path: Path, flag: str, value):
    """Add flag/value only if underlying script supports it.

    - For value flags pass the value (or None to skip)
    - For bool flags pass True/False
    """
    if value is None:
        return
    if not _supports(script_path, flag):
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
    else:
        cmd.extend([flag, str(value)])


def _build_prompt(prompt: str, style: str, suffix_extra: str | None = None) -> str:
    prompt = (prompt or "").strip()
    style_cfg = GEN_STYLES.get(style, GEN_STYLES["balanced"])
    suffix = (style_cfg.get("suffix") or "").strip()

    parts = [p for p in [prompt, suffix, (suffix_extra or "").strip()] if p]
    return ", ".join(parts)
                            

                                                                                              
                                                                          
                
                                          
                 
                                                   

# -------------------------
# commands
# -------------------------

def generate(a):
    # Euler generation only.
                                  
                                                                                                   
    if a.backend == "onnx":
        script = ROOT / "scripts" / "txt2img_onnx_cpu.py"
        models_dir = ROOT / "models"
    else:
                                 
        script = ROOT / "scripts" / "txt2img_rknn_sd15_euler.py"
        models_dir = BUSINESS_MODELS_DIR
             
                                                     
                                                     
                                      

    style_cfg = GEN_STYLES.get(a.style, GEN_STYLES["balanced"])
    steps = a.steps if a.steps is not None else int(style_cfg["steps"])
    guidance = a.guidance if a.guidance is not None else float(style_cfg["guidance"])

    full_prompt = _build_prompt(a.prompt, style=a.style, suffix_extra=a.suffix)
                 
                      
                                     
                        
     

    out_path = Path(a.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    before = time.time()

    cmd = [
        sys.executable,
        str(script),
        "--prompt",
        full_prompt,
        "-i",
        str(models_dir),
        "-o",
        str(out_dir),
        "--num-inference-steps",
        str(int(steps)),
        "--guidance-scale",
        str(float(guidance)),
        "-s",
        a.size,
    ]

    _maybe_add(cmd, script, "--seed", a.seed)
    _maybe_add(cmd, script, "--cfg-last-n", a.cfg_last_n)

    # Optional negative prompt (ONLY if user provided it; no injection)
    neg = a.negative_prompt if a.negative_prompt else a.neg
    _maybe_add(cmd, script, "--negative-prompt", neg)

    # UNet selection passthrough for the Euler generator
    if a.backend == "rknn":
        if a.unet == "legacy":
            _maybe_add(cmd, script, "--force-legacy-unet", True)
        elif a.unet == "b2":
            _maybe_add(cmd, script, "--force-b2", True)
            _maybe_add(cmd, script, "--no-prefer-b2-static", True)
        elif a.unet == "b2_static":
            _maybe_add(cmd, script, "--force-b2-static", True)
        else:
                                                                                                  
                                                  
            if a.no_prefer_b2_static:
                _maybe_add(cmd, script, "--no-prefer-b2-static", True)

    run(cmd)

                                                                                             
    _force_output_png(out_path, out_dir, before_ts=before)


def edit(a):
    """Euler img2img wrapper.

    For steps meaning:
    - User provides --steps (desired denoising steps)
    - Underlying script computes its internal total timesteps so *exactly* that
      many denoise steps are executed.
    """

    preset = EDIT_PRESETS.get(a.preset)

    denoise = a.denoise
    noise = a.noise
    guidance = a.guidance_scale
    cfg_rescale = a.cfg_rescale
    karras = a.karras

    if preset:
        if denoise is None:
            denoise = preset["denoise"]
        if noise is None:
            noise = preset["noise"]
        if guidance is None:
            guidance = preset["guidance"]
        if cfg_rescale is None:
            cfg_rescale = preset["cfg_rescale"]
        if not a.karras:
            karras = bool(preset.get("karras", False))

    # Sensible hard defaults if not set at all
    if denoise is None:
        denoise = 0.35
    if noise is None:
        noise = 0.60
    if guidance is None:
        guidance = 7.0
    if cfg_rescale is None:
        cfg_rescale = 0.0

    steps = int(a.steps)
    if steps < 1:
        raise SystemExit("--steps must be >= 1")

    out_path = Path(a.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    before = time.time()

    # Single script handles both masked + unmasked.
    script = ROOT / "scripts" / "img2img_inpaint_unet.py"

    cmd = [
        sys.executable,
        str(script),
        "--prompt",
        a.prompt,
        "--init-image",
        a.init,
        "-i",
        str(BUSINESS_MODELS_DIR),
        "-o",
        str(out_dir),
        "-s",
        a.size,
        "--steps",
        str(steps),
        "--denoise",
        str(float(denoise)),
        "--noise",
        str(float(noise)),
        "--guidance-scale",
        str(float(guidance)),
    ]

    _maybe_add(cmd, script, "--seed", a.seed)
    _maybe_add(cmd, script, "--cfg-rescale", cfg_rescale)
    _maybe_add(cmd, script, "--karras", bool(karras))

    if a.mask:
        cmd += ["--mask", a.mask]
        _maybe_add(cmd, script, "--mask-blur", a.mask_blur)

    # Important: do NOT pass negative prompt unless user provided it.
    _maybe_add(cmd, script, "--negative-prompt", a.negative_prompt)

    run(cmd)
    _force_output_png(out_path, out_dir, before_ts=before)


#def upscale(a):
#    rknn = a.rknn or (ROOT / "models" / "sr" / "realesrgan" / "realesrgan_x4plus_tile128_fp16.rknn")
#    script = ROOT / "scripts" / "upscale_realesrgan_rknn.py"
#    cmd = [
#        sys.executable,
#        str(script),
#        "--rknn",
#        str(rknn),
#        "--in",
#        a.input,
#        "--out",
#        a.out,
#        "--tile",
#        str(a.tile),
#        "--overlap",
#        str(a.overlap),
#        "--scale",
#        str(a.scale),
#    ]
#    if a.cores:
#        cmd += ["--cores", a.cores]
#    run(cmd)
def upscale(a):
    rknn = a.rknn or (ROOT / "model/sr/realesrgan/realesrgan_x4plus_tile128_fp16.rknn")
    cmd = [
#        sys.executable, ROOT / "tools/realesrgan_rknn_tile_upscale.py",
        sys.executable, ROOT / "scripts/upscale_realesrgan_rknn.py",
        "--rknn", str(rknn),
        "--in", a.input,
        "--out", a.out,
        "--tile", str(a.tile),
        "--overlap", str(a.overlap),
        "--scale", str(a.scale),
    ]
    if a.cores:
        cmd += ["--cores", a.cores]
    run(cmd)

def facefix(a):
    script = ROOT / "tools" / "gfpgan_facefix.py"
    cmd = [
        sys.executable,
        str(script),
        "--input",
        a.input,
        "--out",
        a.out,
        "--upscale",
        str(a.upscale),
    ]
    if a.bg_upscale:
        cmd.append("--bg-upscale")
    run(cmd)


# -------------------------
# CLI
# -------------------------

def main():
    p = argparse.ArgumentParser("rkimg")
    sp = p.add_subparsers(dest="cmd", required=True)

    # txt2img
    g = sp.add_parser("generate", help="Generate an image from text (txt2img) [Euler]")
    g.add_argument("--prompt", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--size", default="512x512")
    g.add_argument("--backend", choices=["rknn", "onnx"], default="rknn")
                                                                         
                                                                                                          
                                                                                           
                                                                          
                                                                
                                                                                             

    g.add_argument("--unet", choices=["auto", "legacy", "b2", "b2_static"], default="auto")
    g.add_argument("--no-prefer-b2-static", action="store_true")

    g.add_argument("--style", choices=list(GEN_STYLES.keys()), default="photoreal")
    g.add_argument("--suffix", default=None, help="Extra comma-separated text appended to the prompt")
    g.add_argument("--steps", type=int, default=None)
    g.add_argument("--cfg-last-n", type=int, default=10,
               help="Apply CFG only for the last N steps (legacy UNet big speed win)")

         
                                                                                   
                                                                    
                                                                                        
    g.add_argument("--guidance", type=float, default=None)
    g.add_argument("--seed", type=int, default=None)

    # NEW: Negative prompt support for generate (opt-in only)
    g.add_argument("--negative-prompt", default=None, help="Optional negative prompt (not injected unless set)")
    g.add_argument("--neg", dest="negative_prompt", help=argparse.SUPPRESS)


    g.set_defaults(func=generate)

    # img2img
    e = sp.add_parser("edit", help="Edit an image with a prompt (img2img) [Euler]")
    e.add_argument("--prompt", required=True)
    e.add_argument("--init", required=True, help="Input image path")
    e.add_argument("--out", required=True)
    e.add_argument("--size", default="512x512")

    e.add_argument("--preset", choices=list(EDIT_PRESETS.keys()), default="recolor",
                   help="Convenience preset (you can override any value)")
    e.add_argument("--steps", type=int, default=30, help="Desired denoising steps (actual loop iterations)")
    e.add_argument("--denoise", type=float, default=None, help="0..1, higher = more change")
                                                                            
    e.add_argument("--noise", type=float, default=None, help="Noise injected into init latents (0..2 typical)")
    e.add_argument("--seed", type=int, default=93)
    e.add_argument("--guidance-scale", type=float, default=None)
    e.add_argument("--cfg-rescale", type=float, default=None, help="0 disables. Try 0.3-0.7")
    e.add_argument("--karras", action="store_true", help="Use Karras sigmas when supported")

    e.add_argument("--negative-prompt", default=None, help="Optional negative prompt (not injected unless set)")

    e.add_argument("--mask", default=None, help="Optional mask path (white=edit, black=keep)")
    e.add_argument("--mask-blur", type=int, default=8)

    e.set_defaults(func=edit)

    # upscale
    u = sp.add_parser("upscale", help="Upscale an image (Real-ESRGAN x4)")
    u.add_argument("--input", required=True)
    u.add_argument("--out", required=True)
    u.add_argument("--rknn", default=None)
    u.add_argument("--tile", type=int, default=128)
    u.add_argument("--overlap", type=int, default=32)
    u.add_argument("--scale", type=int, default=4)
    u.add_argument("--cores", default=None)
    u.set_defaults(func=upscale)

    # facefix
    f = sp.add_parser("facefix", help="Restore faces (GFPGAN)")
    f.add_argument("--input", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--upscale", type=int, default=2)
    f.add_argument("--bg-upscale", action="store_true")
    f.set_defaults(func=facefix)



    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
