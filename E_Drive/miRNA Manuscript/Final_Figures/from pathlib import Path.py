from __future__ import annotations

from pathlib import Path
from io import BytesIO

from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# ================== SETTINGS ==================
INPUT_DIR = Path(r"E:\miRNA Manuscript\Final_Figures")
OUT_DIR = INPUT_DIR / "pdf_600dpi_under1mb"
OUT_DIR.mkdir(exist_ok=True)

DPI = 600  # fixed: not less, not more
MAX_PDF_BYTES = 1_000_000  # 1 MB
MARGIN_IN = 0.5  # white margin around the image on the PDF page (inches)

# JPEG quality search range (auto-adjusted to get under 1MB)
START_QUALITY = 92
MIN_QUALITY = 35
STEP = 5

# Supported raster inputs
SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# ==============================================


def flatten_to_rgb(im: Image.Image) -> Image.Image:
    """Ensure RGB; if transparency exists, composite on white."""
    im = ImageOps.exif_transpose(im)  # respect camera/EXIF rotation
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    return im


def resize_to_fit_letter_at_600dpi(im: Image.Image) -> tuple[Image.Image, float, float]:
    """
    Enforce: effective DPI exactly 600 AND not exceeding letter page usable area.
    If the image would exceed the page at 600 dpi, downsample so it fits.
    Returns (resized_image, placed_width_in, placed_height_in).
    """
    page_w_pt, page_h_pt = letter
    page_w_in = page_w_pt / 72.0
    page_h_in = page_h_pt / 72.0

    max_w_in = page_w_in - 2 * MARGIN_IN
    max_h_in = page_h_in - 2 * MARGIN_IN

    w_px, h_px = im.size
    w_in_600 = w_px / DPI
    h_in_600 = h_px / DPI

    # If it already fits at 600 dpi, don't resample (keeps best sharpness)
    if w_in_600 <= max_w_in and h_in_600 <= max_h_in:
        return im, w_in_600, h_in_600

    # Otherwise, we must downsample so that (pixels / 600) fits the page box
    s = min(max_w_in / w_in_600, max_h_in / h_in_600)  # < 1
    new_w = max(1, int(round(w_px * s)))
    new_h = max(1, int(round(h_px * s)))

    im2 = im.resize((new_w, new_h), resample=Image.LANCZOS)
    return im2, new_w / DPI, new_h / DPI


def make_pdf_bytes_from_jpeg(jpeg_bytes: bytes, place_w_in: float, place_h_in: float) -> bytes:
    """Create a single-page Letter PDF with the JPEG placed at exact physical size."""
    page_w_pt, page_h_pt = letter
    w_pt = place_w_in * 72.0
    h_pt = place_h_in * 72.0

    # Center on page (prevents any width overflow)
    x = (page_w_pt - w_pt) / 2.0
    y = (page_h_pt - h_pt) / 2.0

    pdf_io = BytesIO()
    c = canvas.Canvas(pdf_io, pagesize=letter)

    img_reader = ImageReader(BytesIO(jpeg_bytes))
    c.drawImage(img_reader, x, y, width=w_pt, height=h_pt, preserveAspectRatio=True, anchor='c')

    c.showPage()
    c.save()
    return pdf_io.getvalue()


def compress_to_under_1mb(im: Image.Image, place_w_in: float, place_h_in: float) -> tuple[bytes, int]:
    """
    Convert image to JPEG and wrap into PDF.
    Iteratively reduce JPEG quality until PDF < 1MB (or hits MIN_QUALITY).
    """
    q = START_QUALITY
    best_pdf = None
    best_q = q

    while q >= MIN_QUALITY:
        jpg_io = BytesIO()

        # subsampling=1 is a good compromise (4:2:2). Use 0 for max text sharpness but larger files.
        im.save(
            jpg_io,
            format="JPEG",
            quality=q,
            optimize=True,
            progressive=True,
            subsampling=1,
        )
        pdf_bytes = make_pdf_bytes_from_jpeg(jpg_io.getvalue(), place_w_in, place_h_in)

        if len(pdf_bytes) <= MAX_PDF_BYTES:
            return pdf_bytes, q

        # keep last attempt as best fallback
        best_pdf = pdf_bytes
        best_q = q
        q -= STEP

    # If we can't get under 1MB even at MIN_QUALITY, return the smallest we reached.
    return best_pdf, best_q


def main():
    imgs = sorted(
        [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED],
        key=lambda p: p.name.lower()
    )
    if not imgs:
        print(f"No supported images found in: {INPUT_DIR}")
        return

    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"Target: Letter page, {DPI} dpi fixed, PDF < {MAX_PDF_BYTES/1_000_000:.1f} MB each\n")

    too_big = []

    for p in imgs:
        out_pdf = OUT_DIR / f"{p.stem}.pdf"

        try:
            with Image.open(p) as im:
                im = flatten_to_rgb(im)
                im, w_in, h_in = resize_to_fit_letter_at_600dpi(im)

                pdf_bytes, used_q = compress_to_under_1mb(im, w_in, h_in)
                out_pdf.write_bytes(pdf_bytes)

                size_mb = out_pdf.stat().st_size / 1_000_000
                status = "OK" if out_pdf.stat().st_size <= MAX_PDF_BYTES else "WARN"

                print(f"[{status}] {p.name} -> {out_pdf.name} | {size_mb:.2f} MB | JPEG q={used_q} | placed={w_in:.2f}x{h_in:.2f} in @600dpi")

                if out_pdf.stat().st_size > MAX_PDF_BYTES:
                    too_big.append(out_pdf.name)

        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

    if too_big:
        print("\nThese PDFs could not be pushed under 1MB even at minimum quality:")
        for name in too_big:
            print(" -", name)
        print("\nIf you want, I can modify the script to use a stricter fit (smaller placed size) OR allow a lower MIN_QUALITY.")


if __name__ == "__main__":
    main()
