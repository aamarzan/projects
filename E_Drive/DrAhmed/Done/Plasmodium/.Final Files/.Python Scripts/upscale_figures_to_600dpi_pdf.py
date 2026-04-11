import sys
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

# Optional: If you have input PDFs and want to rasterize them at 600 DPI
# pip install pymupdf
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# Optional but recommended for embedding images into PDFs with correct physical size
# pip install reportlab
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False


# -----------------------------
# CONFIG (EDIT IF NEEDED)
# -----------------------------
SCRIPT_DIR = Path(r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\plasmodium\.Final Files\.Figure PDFs")
INPUT_DIR  = Path(r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\plasmodium\.Final Files\.Figure")
OUTPUT_DIR = SCRIPT_DIR / "upscaled_600dpi_output"

RECURSIVE = True  # True = include subfolders, False = only files directly inside INPUT_DIR

# File types to process
TARGET_EXTS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
    ".pdf"
}

DESIRED_DPI = 600

# If the source image has no DPI metadata, assume it was exported at 300 DPI
ASSUME_INPUT_DPI_IF_MISSING = 300

# Sharpening controls (keep modest to avoid halos)
APPLY_SHARPEN = True
UNSHARP_RADIUS = 2
UNSHARP_PERCENT = 140
UNSHARP_THRESHOLD = 3


# -----------------------------
# HELPERS
# -----------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_input_dpi(img: Image.Image) -> int:
    dpi = None
    if "dpi" in img.info and isinstance(img.info["dpi"], tuple) and len(img.info["dpi"]) >= 1:
        try:
            dpi = int(round(img.info["dpi"][0]))
        except Exception:
            dpi = None
    return dpi if dpi and dpi > 0 else ASSUME_INPUT_DPI_IF_MISSING


def flatten_alpha_to_white(img: Image.Image) -> Image.Image:
    """Flatten transparency to white background for clean print/PDF."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def upscale_image_to_dpi(img: Image.Image, desired_dpi: int) -> tuple[Image.Image, int]:
    """
    Upscale based on DPI ratio (desired / input).
    If scale <= 1.0, keep original pixels (no downscale).
    Returns (upscaled_img, input_dpi_used).
    """
    input_dpi = get_input_dpi(img)
    scale = desired_dpi / float(input_dpi)

    if scale <= 1.0:
        return img, input_dpi

    new_w = int(round(img.size[0] * scale))
    new_h = int(round(img.size[1] * scale))
    up = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return up, input_dpi


def apply_premium_sharpen(img: Image.Image) -> Image.Image:
    """Mild unsharp mask to restore edge crispness after upscaling."""
    if not APPLY_SHARPEN:
        return img
    return img.filter(
        ImageFilter.UnsharpMask(
            radius=UNSHARP_RADIUS,
            percent=UNSHARP_PERCENT,
            threshold=UNSHARP_THRESHOLD,
        )
    )


def save_png_600dpi(img: Image.Image, out_png: Path, dpi: int = 600) -> None:
    safe_mkdir(out_png.parent)
    img.save(out_png, format="PNG", dpi=(dpi, dpi), optimize=True)


def save_pdf_with_true_page_scale(img: Image.Image, out_pdf: Path, dpi: int = 600) -> None:
    """
    Creates a PDF with page size matching the image physical size at `dpi`.
    This yields predictable, crisp output in proofs.
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab is not installed. Run: pip install reportlab")

    safe_mkdir(out_pdf.parent)

    width_px, height_px = img.size
    width_in = width_px / dpi
    height_in = height_px / dpi
    width_pt = width_in * 72.0
    height_pt = height_in * 72.0

    c = canvas.Canvas(str(out_pdf), pagesize=(width_pt, height_pt))
    c.drawImage(ImageReader(img), 0, 0, width=width_pt, height=height_pt, mask="auto")
    c.showPage()
    c.save()


def rasterize_pdf_pages(pdf_path: Path, dpi: int = 600) -> list[Image.Image]:
    """
    Rasterize all pages of a PDF at target DPI.
    Requires PyMuPDF (pymupdf).
    """
    if not HAS_PYMUPDF:
        raise RuntimeError("pymupdf is not installed. Run: pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    if doc.page_count < 1:
        doc.close()
        raise ValueError(f"No pages found in {pdf_path.name}")

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)

    doc.close()
    return images


def iter_input_files(input_dir: Path):
    if RECURSIVE:
        yield from input_dir.rglob("*")
    else:
        yield from input_dir.glob("*")


# -----------------------------
# MAIN
# -----------------------------
def main():
    safe_mkdir(OUTPUT_DIR)

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input folder not found:\n  {INPUT_DIR}")
        sys.exit(1)

    files = []
    for p in iter_input_files(INPUT_DIR):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in TARGET_EXTS:
            continue

        # Optional: skip already-processed files if they exist in the same input directory
        if p.stem.lower().endswith("_600dpi"):
            continue

        files.append(p)

    if not files:
        print("[INFO] No supported figure files found in:")
        print(f"  {INPUT_DIR}")
        print(f"Supported extensions: {sorted(TARGET_EXTS)}")
        sys.exit(0)

    print(f"[INFO] Found {len(files)} file(s) to process.")
    print(f"       Output -> {OUTPUT_DIR}\n")

    for fp in sorted(files):
        try:
            rel = fp.relative_to(INPUT_DIR)
            out_dir = OUTPUT_DIR / rel.parent
            safe_mkdir(out_dir)

            print(f"--- Processing: {rel.as_posix()}")

            if fp.suffix.lower() == ".pdf":
                # Rasterize all PDF pages at 600 DPI
                pages = rasterize_pdf_pages(fp, dpi=DESIRED_DPI)

                for page_i, img in enumerate(pages, start=1):
                    img = flatten_alpha_to_white(img)
                    img = apply_premium_sharpen(img)  # already at 600 DPI

                    # Name outputs: originalname_p1_600dpi, etc.
                    suffix = f"_p{page_i}_600dpi" if len(pages) > 1 else "_600dpi"
                    out_png = out_dir / f"{fp.stem}{suffix}.png"
                    out_pdf = out_dir / f"{fp.stem}{suffix}.pdf"

                    save_png_600dpi(img, out_png, dpi=DESIRED_DPI)
                    save_pdf_with_true_page_scale(img, out_pdf, dpi=DESIRED_DPI)

                print(f"   PDF pages processed: {len(pages)}")
                print("")

            else:
                img = Image.open(fp)
                img = ImageOps.exif_transpose(img)
                img = flatten_alpha_to_white(img)

                # Upscale to 600 DPI equivalent if needed
                up_img, in_dpi = upscale_image_to_dpi(img, DESIRED_DPI)
                up_img = apply_premium_sharpen(up_img)

                out_png = out_dir / f"{fp.stem}_600dpi.png"
                out_pdf = out_dir / f"{fp.stem}_600dpi.pdf"

                save_png_600dpi(up_img, out_png, dpi=DESIRED_DPI)
                save_pdf_with_true_page_scale(up_img, out_pdf, dpi=DESIRED_DPI)

                print(f"   Input DPI used/assumed: {in_dpi}")
                print(f"   Saved PNG: {out_png.name}")
                print(f"   Saved PDF: {out_pdf.name}\n")

        except Exception as e:
            print(f"[ERROR] Failed on {fp.name}: {e}\n")

    print("[DONE] All figures processed.")
    print("Note: If a figure is still blurry after upscaling, the best fix is re-exporting the original as vector PDF "
          "(for plots/maps) or re-saving the original image at higher resolution.")


if __name__ == "__main__":
    main()