#!/usr/bin/env python3
import argparse, os
import numpy as np
from PIL import Image
from rknnlite.api import RKNNLite

def to_nhwc_f32_01(tile_rgb: Image.Image) -> np.ndarray:
    arr = np.asarray(tile_rgb.convert("RGB"), dtype=np.float32) / 255.0  # HWC 0..1
    return arr[None, ...]  # NHWC

def out_to_hwc_f32(out: np.ndarray) -> np.ndarray:
    y = np.array(out)
    # common shapes: (1,H,W,3) or (1,3,H,W) or (H,W,3) or (3,H,W)
    if y.ndim == 4 and y.shape[0] == 1:
        y = y[0]
    if y.ndim == 3 and y.shape[0] == 3 and y.shape[-1] != 3:
        y = np.transpose(y, (1,2,0))  # CHW -> HWC
    return y.astype(np.float32)

def make_feather_mask(h, w, feather):
    # feather in pixels at output scale
    if feather <= 0:
        return np.ones((h, w), dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    x = np.linspace(0, 1, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # distance to nearest edge (0 at edge, 0.5 at center-ish)
    d = np.minimum.reduce([yy, 1-yy, xx, 1-xx])
    # map to 0..1 with feather region
    # feather is in pixels, so convert to normalized distance
    dn = d * min(h, w)
    m = np.clip(dn / feather, 0.0, 1.0)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=128, help="model input tile size (static)")
    ap.add_argument("--overlap", type=int, default=16, help="overlap in input pixels (reduces seams)")
    ap.add_argument("--scale", type=int, default=4, help="upscale factor (usually 4 for realesrgan_x4plus)")
    ap.add_argument("--cores", default="auto", choices=["auto","0","1","2"], help="NPU core mask")
    args = ap.parse_args()

    tile = args.tile
    ov = max(0, min(args.overlap, tile-1))
    step = tile - ov
    scale = args.scale

    im = Image.open(args.inp).convert("RGB")
    W, H = im.size

    # Pad to cover whole image with tiles
    pad_w = (step - (W - tile) % step) % step if W > tile else (tile - W)
    pad_h = (step - (H - tile) % step) % step if H > tile else (tile - H)
    pad_w = max(pad_w, 0)
    pad_h = max(pad_h, 0)

    if pad_w or pad_h:
        arr = np.asarray(im, dtype=np.uint8)
        arr = np.pad(arr, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
        im_pad = Image.fromarray(arr, "RGB")
    else:
        im_pad = im

    Wp, Hp = im_pad.size
    outW, outH = Wp * scale, Hp * scale

    # output accumulation buffers
    acc = np.zeros((outH, outW, 3), dtype=np.float32)
    wsum = np.zeros((outH, outW), dtype=np.float32)

    # Load RKNN
    r = RKNNLite()
    r.load_rknn(args.rknn)

    if args.cores == "auto":
        r.init_runtime()
    else:
        core_map = {"0": RKNNLite.NPU_CORE_0, "1": RKNNLite.NPU_CORE_1, "2": RKNNLite.NPU_CORE_2}
        r.init_runtime(core_mask=core_map[args.cores])

    # feather in output pixels
    feather_out = ov * scale // 2  # half-overlap feather
    # precompute mask at output tile size
    out_tile = tile * scale
    mask = make_feather_mask(out_tile, out_tile, feather_out)[..., None]  # HWC1

    for y in range(0, Hp - tile + 1, step):
        for x in range(0, Wp - tile + 1, step):
            tile_img = im_pad.crop((x, y, x+tile, y+tile))
            inp = to_nhwc_f32_01(tile_img)

            out = r.inference(inputs=[inp], data_format="nhwc")[0]
            ytile = out_to_hwc_f32(out)  # expect 0..1 (your “01_rgb”)

            # if model outputs 0..255 by any chance, normalize back to 0..1 safely
            if ytile.max() > 2.0:
                ytile = ytile / 255.0

            y0, x0 = y * scale, x * scale
            acc[y0:y0+out_tile, x0:x0+out_tile, :] += ytile * mask
            wsum[y0:y0+out_tile, x0:x0+out_tile] += mask[..., 0]

    # Normalize
    wsum = np.clip(wsum, 1e-6, None)
    out_img = acc / wsum[..., None]
    out_u8 = (np.clip(out_img, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    # Crop back to original size * scale (remove padding area)
    out_u8 = out_u8[:H*scale, :W*scale, :]
    Image.fromarray(out_u8, "RGB").save(args.out)
    print(f"[ok] saved: {args.out}  ({W*scale}x{H*scale})")

if __name__ == "__main__":
    main()
