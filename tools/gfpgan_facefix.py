#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import urllib.request


# ---- helpers ----
def _repo_root() -> Path:
    # tools/gfpgan_facefix.py -> repo root is parents[1]
    return Path(__file__).resolve().parents[1]


def _ensure_file(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    print(f"[gfpgan] downloading model:\n  {url}\n  -> {dst}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, dst)  # simple + works on stock python
    except Exception as e:
        raise SystemExit(f"[gfpgan] failed to download model: {e}")
    if not dst.exists() or dst.stat().st_size == 0:
        raise SystemExit("[gfpgan] download finished but file is missing/empty")
    return dst


def _resolve_gfpgan_model(args) -> str:
    """
    GFPGAN 1.3.8 does NOT accept model_path=None.
    Return a valid local path (download if missing).
    """
    name = (args.gfpgan_model or "GFPGANv1.4").strip()

    # Known-good release URLs
    urls = {
        "GFPGANv1.4": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "GFPGANv1.3": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    }

    # If user passed a path instead of a name
    p = Path(name)
    if p.suffix.lower() == ".pth":
        if p.is_absolute():
            if not p.exists():
                raise SystemExit(f"[gfpgan] model file not found: {p}")
            return str(p)
        # relative path: resolve from repo root
        rp = (_repo_root() / p).resolve()
        if not rp.exists():
            raise SystemExit(f"[gfpgan] model file not found: {rp}")
        return str(rp)

    # Otherwise treat it as a model name
    if name not in urls:
        raise SystemExit(
            f"[gfpgan] unknown model '{name}'. Use GFPGANv1.4, GFPGANv1.3, or pass a .pth path."
        )

    weights_dir = _repo_root() / "gfpgan" / "weights"
    local = weights_dir / f"{name}.pth"
    _ensure_file(urls[name], local)
    return str(local)


def main():
    p = argparse.ArgumentParser(description="GFPGAN face restoration wrapper")
    p.add_argument("--input", required=True, help="Input image path")
    p.add_argument("--out", required=True, help="Output image path")
    p.add_argument("--upscale", type=int, default=2, help="Internal GFPGAN upscale factor (1/2/4)")
    p.add_argument("--bg-upscale", action="store_true", help="Also upscale background via RealESRGAN (if available)")
    p.add_argument("--bg-upscale-model", default="RealESRGAN_x4plus", help="RealESRGAN model name")
    p.add_argument("--gfpgan-model", default="GFPGANv1.4", help="GFPGAN model name or path to .pth")
    p.add_argument("--weight", type=float, default=0.7, help="CodeFormer only; kept for future compatibility")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Import here so install errors show cleanly
    try:
        from gfpgan import GFPGANer
    except Exception as e:
        raise SystemExit(
            "GFPGAN not installed. Activate venv and run:\n"
            "  pip install gfpgan opencv-python\n"
            f"\nOriginal error: {e}"
        )

    # Optional background upscaler via realesrgan (pip package)
    # NOTE: Many realesrgan installs also expect an explicit model_path.
    # We keep this best-effort; if it fails, background upscaling is disabled.
    bg_upsampler = None
    if args.bg_upscale:
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )

            # Try to locate a local RealESRGAN weight if user has one in repo/realesrgan/weights
            # Otherwise disable (avoids passing None and crashing some versions)
            re_dir = _repo_root() / "realesrgan" / "weights"
            re_local = re_dir / f"{args.bg_upscale_model}.pth"
            re_model_path = str(re_local) if re_local.exists() else None

            if re_model_path:
                bg_upsampler = RealESRGANer(
                    scale=4,
                    model_path=re_model_path,
                    model=model,
                    tile=256,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                )
            else:
                bg_upsampler = None
        except Exception:
            bg_upsampler = None

    # ✅ FIX: model_path must be a string path (not None)
    model_path = _resolve_gfpgan_model(args)

    # Use CPU (safe default on RK3588). If you later have CUDA on x86, set device="cuda".
    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=bg_upsampler,
    )

    import cv2

    img = cv2.imread(str(inp), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {inp}")

    # has_aligned=False => detect faces automatically
    _, _, restored = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    ok = cv2.imwrite(str(out), restored)
    if not ok:
        raise SystemExit(f"Failed to write output: {out}")
    print(f"[gfpgan] saved: {out}")


if __name__ == "__main__":
    main()
