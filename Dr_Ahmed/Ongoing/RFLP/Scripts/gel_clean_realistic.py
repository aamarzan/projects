#!/usr/bin/env python3
"""
gel_clean_realistic.py
Realistic gel image cleanup:
- mild denoise (non-local means)
- remove small speckle "dust" without smearing bands
- gentle background flattening (uneven illumination correction)
- mild contrast enhancement (CLAHE)
- optional light sharpening (unsharp mask)
Designed to look natural (not AI) and preserve overall band relationships.

Usage:
  python gel_clean_realistic.py --in f7.png --out f7_clean.png
  python gel_clean_realistic.py --in_dir ./imgs --out_dir ./out --ext png --batch
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

import cv2


def percentile_rescale(gray: np.ndarray, lo=0.5, hi=99.7) -> np.ndarray:
    g = gray.astype(np.float32)
    a = np.percentile(g, lo)
    b = np.percentile(g, hi)
    if b <= a + 1e-6:
        return np.clip(g, 0, 255).astype(np.uint8)
    g = (g - a) * (255.0 / (b - a))
    return np.clip(g, 0, 255).astype(np.uint8)


def flatfield_correct(gray: np.ndarray, ksize: int = 121) -> np.ndarray:
    """
    Estimate smooth background via large-kernel morphological opening,
    then subtract (illumination correction).
    ksize should be odd and relatively large (depends on image size).
    """
    ksize = max(ksize, 31)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    corrected = cv2.subtract(gray, bg)
    return corrected


def remove_speckles(gray: np.ndarray, strength: int = 1) -> np.ndarray:
    """
    Remove tiny dust speckles without harming bands:
    - a small median blur
    - then a very small morphological open/close
    strength: 0(off), 1(mild), 2(medium)
    """
    if strength <= 0:
        return gray

    # Median removes isolated hot pixels gently
    k = 3 if strength == 1 else 5
    g = cv2.medianBlur(gray, k)

    # Morph ops to reduce isolated tiny dots (keep bands)
    mk = 3 if strength == 1 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel, iterations=1)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel, iterations=1)
    return g


def mild_denoise(gray: np.ndarray, h: int = 6) -> np.ndarray:
    """
    Non-local means denoise: preserves edges better than Gaussian blur.
    Lower h = more realistic (less "plastic").
    """
    h = int(np.clip(h, 0, 15))
    if h == 0:
        return gray
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def clahe_contrast(gray: np.ndarray, clip: float = 1.6, tile: int = 8) -> np.ndarray:
    """
    Gentle local contrast enhancement; keep clip low to avoid fake look.
    """
    clip = float(np.clip(clip, 1.0, 3.0))
    tile = int(np.clip(tile, 4, 16))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def unsharp_mask(gray: np.ndarray, amount: float = 0.25, sigma: float = 1.2) -> np.ndarray:
    """
    Very light sharpening. Keep amount small to avoid halos.
    amount: 0(off) to ~0.4 (safe range)
    """
    amount = float(np.clip(amount, 0.0, 0.6))
    if amount <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def maybe_upscale(gray: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return gray
    h, w = gray.shape[:2]
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def process_one(
    in_path: Path,
    out_path: Path,
    scale: float,
    bg_ksize: int,
    denoise_h: int,
    speckle_strength: int,
    clahe_clip: float,
    clahe_tile: int,
    sharpen_amount: float,
) -> None:
    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {in_path}")

    # Convert to grayscale if needed (gel images usually grayscale)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Optional upscale first (helps denoise/cleanup)
    gray = maybe_upscale(gray, scale=scale)

    # Normalize dynamic range gently
    gray = percentile_rescale(gray, lo=0.5, hi=99.7)

    # Flatten uneven illumination (smoke/haze background)
    corrected = flatfield_correct(gray, ksize=bg_ksize)
    corrected = percentile_rescale(corrected, lo=0.5, hi=99.7)

    # Mild denoise + speckle cleanup
    cleaned = mild_denoise(corrected, h=denoise_h)
    cleaned = remove_speckles(cleaned, strength=speckle_strength)

    # Gentle contrast + very light sharpening
    cleaned = clahe_contrast(cleaned, clip=clahe_clip, tile=clahe_tile)
    cleaned = unsharp_mask(cleaned, amount=sharpen_amount, sigma=1.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cleaned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_file", type=str, default=None, help="Input image file")
    ap.add_argument("--out", dest="out_file", type=str, default=None, help="Output image file")
    ap.add_argument("--in_dir", type=str, default=None, help="Input directory for batch")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory for batch")
    ap.add_argument("--ext", type=str, default="png", help="Batch extension (png/jpg/tif)")
    ap.add_argument("--batch", action="store_true", help="Process a directory (batch mode)")

    # Realism-tuned defaults (safe)
    ap.add_argument("--scale", type=float, default=2.0, help="Upscale factor (1.0 = no upscale)")
    ap.add_argument("--bg_ksize", type=int, default=121, help="Background kernel size (odd, larger=more flatten)")
    ap.add_argument("--denoise_h", type=int, default=6, help="Non-local means strength (0-15; lower=more real)")
    ap.add_argument("--speckle", type=int, default=1, help="Speckle removal: 0 off, 1 mild, 2 medium")
    ap.add_argument("--clahe_clip", type=float, default=1.6, help="CLAHE clipLimit (1.0-3.0)")
    ap.add_argument("--clahe_tile", type=int, default=8, help="CLAHE tile grid (4-16)")
    ap.add_argument("--sharpen", type=float, default=0.25, help="Sharpen amount (0-0.6; keep low)")

    args = ap.parse_args()

    if args.batch:
        if not args.in_dir or not args.out_dir:
            raise SystemExit("Batch mode requires --in_dir and --out_dir")
        in_dir = Path(args.in_dir)
        out_dir = Path(args.out_dir)
        files = sorted(in_dir.glob(f"*.{args.ext}"))
        if not files:
            raise SystemExit(f"No *.{args.ext} files found in {in_dir}")
        for f in files:
            out = out_dir / f"{f.stem}_clean.{args.ext}"
            process_one(
                f, out,
                scale=args.scale,
                bg_ksize=args.bg_ksize,
                denoise_h=args.denoise_h,
                speckle_strength=args.speckle,
                clahe_clip=args.clahe_clip,
                clahe_tile=args.clahe_tile,
                sharpen_amount=args.sharpen,
            )
        print(f"Done. Processed {len(files)} file(s) -> {out_dir}")
        return

    if not args.in_file or not args.out_file:
        raise SystemExit("Single-file mode requires --in and --out")

    process_one(
        Path(args.in_file),
        Path(args.out_file),
        scale=args.scale,
        bg_ksize=args.bg_ksize,
        denoise_h=args.denoise_h,
        speckle_strength=args.speckle,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        sharpen_amount=args.sharpen,
    )
    print(f"Saved: {args.out_file}")


if __name__ == "__main__":
    main()
