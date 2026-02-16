import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image, UnidentifiedImageError

# ReportLab for precise page sizing/DPI control
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---------- Config defaults ----------
DEFAULT_INPUT_DIR = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\convert_to_pdf"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".jfif", ".heic", ".heif"}
# -------------------------------------


def find_images(folder: Path) -> List[Path]:
    imgs = [p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    # Sort for stable ordering in combined PDF
    imgs.sort(key=lambda p: p.name.lower())
    return imgs


def pil_to_rgb(img: Image.Image) -> Image.Image:
    """Ensure an RGB image with white background for transparency."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        base = Image.new("RGB", img.size, (255, 255, 255))
        base.paste(img, mask=img.split()[-1] if "A" in img.getbands() else None)
        return base
    return img.convert("RGB")


def draw_image_page(c: canvas.Canvas, img: Image.Image, dpi: int) -> None:
    """Add a page to the PDF sized to the image at the specified DPI."""
    img = pil_to_rgb(img)
    w_px, h_px = img.size
    # PDF points: 72 points == 1 inch
    w_pt = (w_px * 72.0) / dpi
    h_pt = (h_px * 72.0) / dpi
    c.setPageSize((w_pt, h_pt))
    c.drawImage(ImageReader(img), 0, 0, width=w_pt, height=h_pt)
    c.showPage()


def export_single_pdf(image_path: Path, out_path: Path, dpi: int) -> None:
    c = canvas.Canvas(str(out_path))
    with Image.open(image_path) as im:
        draw_image_page(c, im, dpi)
    c.save()


def export_combined_pdf(image_paths: List[Path], out_path: Path, dpi: int) -> None:
    c = canvas.Canvas(str(out_path))
    for p in image_paths:
        try:
            with Image.open(p) as im:
                draw_image_page(c, im, dpi)
        except UnidentifiedImageError:
            print(f"[WARN] Skipping unsupported/corrupt file: {p}", file=sys.stderr)
    c.save()


def main():
    parser = argparse.ArgumentParser(
        description="Convert images in a folder to high-resolution PDF(s)."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Folder containing images (default is the path in the script).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF page sizing (300=print quality; try 600 for very high).",
    )
    parser.add_argument(
        "--combine",
        metavar="OUTPUT_PDF",
        help="If set, write all images into a single combined PDF at this path.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output folder for per-image PDFs (default: '<input>/PDF_out'). Ignored if --combine is used.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[ERROR] Input folder does not exist:\n{input_dir}", file=sys.stderr)
        sys.exit(1)

    images = find_images(input_dir)
    if not images:
        print(f"[INFO] No images found in: {input_dir}")
        sys.exit(0)

    if args.combine:
        out_pdf = Path(args.combine)
        if not out_pdf.parent.exists():
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Creating combined PDF ({args.dpi} DPI): {out_pdf}")
        export_combined_pdf(images, out_pdf, args.dpi)
        print("[DONE] Combined PDF saved.")
    else:
        out_dir = Path(args.out_dir) if args.out_dir else (input_dir / "PDF_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Writing one PDF per image to: {out_dir}  (DPI={args.dpi})")
        for p in images:
            try:
                out_pdf = out_dir / (p.stem + ".pdf")
                export_single_pdf(p, out_pdf, args.dpi)
                print(f"  - {p.name} → {out_pdf.name}")
            except UnidentifiedImageError:
                print(f"[WARN] Skipping unsupported/corrupt file: {p}", file=sys.stderr)
        print("[DONE] All PDFs saved.")


if __name__ == "__main__":
    main()
