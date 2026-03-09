#!/usr/bin/env python3
"""
Enhance a scanned map WITHOUT OCR (keeps all words authentic),
then export 600 DPI: PNG, JPG, PDF, EPS.

Pipeline (conservative, text-safe):
1) Mild denoise (NLMeans)
2) CLAHE on luminance only (prevents color shifts)
3) Edge-preserving sharpen (unsharp mask + small edge protection)
4) Optional upscale to a target print width at 600 DPI
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def unsharp_mask(img_bgr: np.ndarray, amount: float = 1.15, radius: float = 1.4, threshold: int = 3) -> np.ndarray:
    """
    Unsharp mask in a conservative way:
    - radius ~ Gaussian blur sigma
    - threshold avoids sharpening flat noise areas
    """
    blur = cv2.GaussianBlur(img_bgr, ksize=(0, 0), sigmaX=radius, sigmaY=radius)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0)

    if threshold > 0:
        # Only apply where there is an actual difference (edges), to protect paper texture/noise
        diff = cv2.absdiff(img_bgr, blur)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mask = diff_gray > threshold
        out = img_bgr.copy()
        out[mask] = sharp[mask]
        return out

    return sharp


def enhance_map(img_bgr: np.ndarray,
                denoise_h: int = 5,
                denoise_hColor: int = 5,
                clahe_clip: float = 2.0,
                clahe_grid: int = 8,
                sharpen_amount: float = 1.10,
                sharpen_radius: float = 1.35,
                sharpen_threshold: int = 3) -> np.ndarray:
    # 1) Mild denoise (keeps text strokes)
    den = cv2.fastNlMeansDenoisingColored(img_bgr, None,
                                         h=denoise_h, hColor=denoise_hColor,
                                         templateWindowSize=7, searchWindowSize=21)

    # 2) CLAHE on luminance only (LAB L channel) to improve readability without recoloring
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_grid), int(clahe_grid)))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    contrast = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 3) Edge-preserving slight smooth (optional but helpful for JPEG artifacts)
    # Keep it gentle; too strong will smear text.
    smooth = cv2.bilateralFilter(contrast, d=5, sigmaColor=30, sigmaSpace=30)

    # 4) Conservative sharpen
    sharp = unsharp_mask(smooth,
                         amount=sharpen_amount,
                         radius=sharpen_radius,
                         threshold=sharpen_threshold)

    return sharp


def resize_for_print(img_bgr: np.ndarray, dpi: int, width_in: float | None) -> np.ndarray:
    """
    If width_in is provided, upscale/downscale so the exported image is width_in inches at dpi.
    Otherwise, keep original pixels and only set DPI metadata in outputs.
    """
    if width_in is None:
        return img_bgr

    target_w = int(round(width_in * dpi))
    h, w = img_bgr.shape[:2]
    if w == target_w:
        return img_bgr

    scale = target_w / float(w)
    target_h = int(round(h * scale))

    # Lanczos is best for text/maps when upscaling
    resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return resized


def save_rasters(img_bgr: np.ndarray, out_base: Path, dpi: int, jpg_quality: int = 95) -> tuple[Path, Path]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)

    out_png = out_base.with_suffix(".png")
    out_jpg = out_base.with_suffix(".jpg")

    im.save(out_png, dpi=(dpi, dpi), optimize=True)
    im.save(out_jpg, dpi=(dpi, dpi), quality=jpg_quality, subsampling=0, optimize=True)

    return out_png, out_jpg


def save_pdf_eps(img_bgr: np.ndarray, out_base: Path, dpi: int) -> tuple[Path, Path]:
    """
    PDF/EPS here are "vector containers with embedded raster".
    This preserves original wording (no OCR) and keeps 600 dpi bitmap inside.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Figure size in inches so bitmap maps to 600 DPI
    fig_w = w / float(dpi)
    fig_h = h / float(dpi)

    # PDF
    out_pdf = out_base.with_suffix(".pdf")
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="lanczos")
    ax.axis("off")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # EPS (raster embedded)
    out_eps = out_base.with_suffix(".eps")
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="lanczos")
    ax.axis("off")
    fig.savefig(out_eps, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return out_pdf, out_eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input image (png/jpg/tif)")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--stem", default="map_enhanced_600dpi", help="Output base filename (no extension)")
    ap.add_argument("--dpi", type=int, default=600, help="Output DPI")
    ap.add_argument("--width_in", type=float, default=None,
                    help="Optional target print width in inches at DPI (e.g., 7.5). If omitted, keeps pixel size.")
    # Conservative defaults (text-safe)
    ap.add_argument("--denoise_h", type=int, default=5)
    ap.add_argument("--denoise_hColor", type=int, default=5)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)
    ap.add_argument("--sharpen_amount", type=float, default=1.10)
    ap.add_argument("--sharpen_radius", type=float, default=1.35)
    ap.add_argument("--sharpen_threshold", type=int, default=3)
    ap.add_argument("--jpg_quality", type=int, default=95)
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / args.stem

    img_bgr = cv2.imread(str(inp), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Could not read input image: {inp}")

    enhanced = enhance_map(
        img_bgr,
        denoise_h=args.denoise_h,
        denoise_hColor=args.denoise_hColor,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        sharpen_amount=args.sharpen_amount,
        sharpen_radius=args.sharpen_radius,
        sharpen_threshold=args.sharpen_threshold,
    )

    final_img = resize_for_print(enhanced, dpi=args.dpi, width_in=args.width_in)

    out_png, out_jpg = save_rasters(final_img, out_base, dpi=args.dpi, jpg_quality=args.jpg_quality)
    out_pdf, out_eps = save_pdf_eps(final_img, out_base, dpi=args.dpi)

    print("Saved:")
    print(out_png)
    print(out_jpg)
    print(out_pdf)
    print(out_eps)


if __name__ == "__main__":
    main()
