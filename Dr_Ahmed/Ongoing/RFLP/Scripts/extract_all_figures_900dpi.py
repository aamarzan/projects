#!/usr/bin/env python3
"""
Extract ALL images/figures from a PDF with maximum fidelity.

Outputs (under --out):
  1) embedded_images/        -> lossless extraction of embedded raster images (native pixels)
  2) blocks_900dpi/          -> 900-dpi crops of detected image blocks (good for "figures" embedded as images)
  3) pages_900dpi/           -> full pages rendered at 900 dpi (backup: nothing missed)
  4) pages_svg/              -> page SVG (vector crispness; best for line art/text)
  5) manifest.csv            -> mapping of everything extracted

IMPORTANT TRUTH:
- You cannot create detail beyond the PDF’s embedded raster resolution.
- SVG export preserves vector sharpness (ideal for text/lines).
"""

import argparse
from pathlib import Path
import csv
import re

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


# ------------------------- IMAGE ENHANCEMENT ------------------------- #
def enhance_pil(im: Image.Image, mode: str) -> Image.Image:
    """
    Enhance raster images mildly. This cannot invent detail; it only improves perceived crispness.
    mode: "none" | "mild" | "strong"
    """
    if mode == "none":
        return im

    # Work in RGB for consistency
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")

    # Gentle autocontrast (avoid blowing highlights)
    if mode in ("mild", "strong"):
        im = ImageOps.autocontrast(im, cutoff=1)

    # Unsharp mask (controlled)
    if mode == "mild":
        im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
        im = ImageEnhance.Sharpness(im).enhance(1.10)
    elif mode == "strong":
        im = im.filter(ImageFilter.UnsharpMask(radius=1.8, percent=170, threshold=2))
        im = ImageEnhance.Sharpness(im).enhance(1.25)

    return im


def save_png_with_dpi(im: Image.Image, out_path: Path, dpi: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="PNG", dpi=(dpi, dpi), optimize=True)


# ------------------------- PDF EXTRACTION ------------------------- #
def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip())
    return s[:120] if len(s) > 120 else s


def extract_embedded_images(doc: fitz.Document, out_dir: Path, dpi_meta: int, enhance: str, manifest_rows: list):
    """
    Extract embedded images by xref (native pixels). This is the closest you can get to the original raster.
    """
    out = out_dir / "embedded_images"
    out.mkdir(parents=True, exist_ok=True)

    seen = set()
    for page_i in range(len(doc)):
        page = doc.load_page(page_i)
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)

            info = doc.extract_image(xref)
            img_bytes = info["image"]
            ext = info.get("ext", "bin")
            w = info.get("width", None)
            h = info.get("height", None)

            # Save original bytes (lossless relative to what’s inside the PDF)
            raw_path = out / f"xref_{xref:06d}_native.{ext}"
            raw_path.write_bytes(img_bytes)

            # Also save a PNG preview (optionally enhanced) with 300/900 dpi metadata
            try:
                pil = Image.open(raw_path)
                pil = pil.convert("RGB") if pil.mode != "RGB" else pil
                pil = enhance_pil(pil, enhance)
                png_path = out / f"xref_{xref:06d}_native.png"
                save_png_with_dpi(pil, png_path, dpi_meta)

                manifest_rows.append({
                    "type": "embedded_native",
                    "page": page_i + 1,
                    "block_index": "",
                    "xref": xref,
                    "bbox_pts": "",
                    "out_file": str(png_path),
                    "native_file": str(raw_path),
                    "width_px": pil.width,
                    "height_px": pil.height,
                })
            except Exception:
                manifest_rows.append({
                    "type": "embedded_native_raw",
                    "page": page_i + 1,
                    "block_index": "",
                    "xref": xref,
                    "bbox_pts": "",
                    "out_file": "",
                    "native_file": str(raw_path),
                    "width_px": w,
                    "height_px": h,
                })


def render_page_png(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return im


def extract_image_blocks(doc: fitz.Document, out_dir: Path, dpi: int, pad_px: int, enhance: str, manifest_rows: list):
    """
    Find image blocks on each page (type==1), render page at DPI, crop bbox to separate PNG.
    This captures figures embedded as images and placed on pages.
    """
    out = out_dir / f"blocks_{dpi}dpi"
    out.mkdir(parents=True, exist_ok=True)

    for page_i in range(len(doc)):
        page = doc.load_page(page_i)
        d = page.get_text("dict")
        blocks = d.get("blocks", [])

        img_blocks = [(bi, b) for bi, b in enumerate(blocks) if b.get("type") == 1]
        if not img_blocks:
            continue

        page_im = render_page_png(page, dpi=dpi)
        zoom = dpi / 72.0

        for bi, b in img_blocks:
            x0, y0, x1, y1 = b["bbox"]  # points
            px0 = max(0, int(x0 * zoom) - pad_px)
            py0 = max(0, int(y0 * zoom) - pad_px)
            px1 = min(page_im.width, int(x1 * zoom) + pad_px)
            py1 = min(page_im.height, int(y1 * zoom) + pad_px)

            crop = page_im.crop((px0, py0, px1, py1))
            crop = enhance_pil(crop, enhance)

            fn = out / f"p{page_i+1:04d}_block{bi:03d}.png"
            save_png_with_dpi(crop, fn, dpi)

            manifest_rows.append({
                "type": f"image_block_{dpi}dpi",
                "page": page_i + 1,
                "block_index": bi,
                "xref": "",
                "bbox_pts": f"{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}",
                "out_file": str(fn),
                "native_file": "",
                "width_px": crop.width,
                "height_px": crop.height,
            })


def export_pages_png(doc: fitz.Document, out_dir: Path, dpi: int, enhance: str, manifest_rows: list):
    """
    Full-page renders at DPI. This ensures nothing is missed (even vector content becomes raster here).
    """
    out = out_dir / f"pages_{dpi}dpi"
    out.mkdir(parents=True, exist_ok=True)

    for page_i in range(len(doc)):
        page = doc.load_page(page_i)
        im = render_page_png(page, dpi=dpi)
        im = enhance_pil(im, enhance)
        fn = out / f"page_{page_i+1:04d}.png"
        save_png_with_dpi(im, fn, dpi)

        manifest_rows.append({
            "type": f"page_{dpi}dpi",
            "page": page_i + 1,
            "block_index": "",
            "xref": "",
            "bbox_pts": "",
            "out_file": str(fn),
            "native_file": "",
            "width_px": im.width,
            "height_px": im.height,
        })


def export_pages_svg(doc: fitz.Document, out_dir: Path, manifest_rows: list):
    """
    Export each page as SVG. This preserves vector sharpness for line/text figures.
    """
    out = out_dir / "pages_svg"
    out.mkdir(parents=True, exist_ok=True)

    for page_i in range(len(doc)):
        page = doc.load_page(page_i)
        svg = page.get_svg_image(text_as_path=False)  # keep real text when possible
        fn = out / f"page_{page_i+1:04d}.svg"
        fn.write_text(svg, encoding="utf-8")

        manifest_rows.append({
            "type": "page_svg",
            "page": page_i + 1,
            "block_index": "",
            "xref": "",
            "bbox_pts": "",
            "out_file": str(fn),
            "native_file": "",
            "width_px": "",
            "height_px": "",
        })


def write_manifest(out_dir: Path, rows: list):
    fn = out_dir / "manifest.csv"
    with fn.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "type", "page", "block_index", "xref", "bbox_pts",
            "out_file", "native_file", "width_px", "height_px"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------------- MAIN ------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Input PDF path (e.g., RFLP.pdf)")
    ap.add_argument("--out", default="extracted_figures", help="Output folder")
    ap.add_argument("--dpi", type=int, default=900, help="Render DPI for block/page PNGs")
    ap.add_argument("--dpi_meta", type=int, default=300, help="DPI metadata to store in extracted embedded PNGs")
    ap.add_argument("--pad", type=int, default=40, help="Padding (pixels) around cropped image blocks")
    ap.add_argument("--enhance", choices=["none", "mild", "strong"], default="mild",
                    help="Raster enhancement level (cannot invent detail)")
    ap.add_argument("--no_embedded", action="store_true", help="Skip embedded image extraction")
    ap.add_argument("--no_blocks", action="store_true", help="Skip image-block crops")
    ap.add_argument("--no_pages", action="store_true", help="Skip full-page PNGs")
    ap.add_argument("--no_svg", action="store_true", help="Skip per-page SVG export")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    manifest_rows = []

    if not args.no_embedded:
        extract_embedded_images(doc, out_dir, dpi_meta=args.dpi_meta, enhance=args.enhance, manifest_rows=manifest_rows)

    if not args.no_blocks:
        extract_image_blocks(doc, out_dir, dpi=args.dpi, pad_px=args.pad, enhance=args.enhance, manifest_rows=manifest_rows)

    if not args.no_pages:
        export_pages_png(doc, out_dir, dpi=args.dpi, enhance=args.enhance, manifest_rows=manifest_rows)

    if not args.no_svg:
        export_pages_svg(doc, out_dir, manifest_rows=manifest_rows)

    write_manifest(out_dir, manifest_rows)
    doc.close()

    print(f"Done. Outputs saved to: {out_dir.resolve()}")
    print(f"Manifest: { (out_dir / 'manifest.csv').resolve() }")


if __name__ == "__main__":
    main()
