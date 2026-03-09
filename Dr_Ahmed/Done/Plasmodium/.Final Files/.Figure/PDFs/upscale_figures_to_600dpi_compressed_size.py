import sys
import io
from pathlib import Path

from PIL import Image, ImageOps

# pip install pymupdf
try:
    import fitz  # PyMuPDF
except Exception:
    print("[ERROR] PyMuPDF not installed. Run: pip install pymupdf")
    raise

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = Path(r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\plasmodium\.Final Files\.Figure")

OUTPUT_DIR = Path(
    r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\plasmodium\.Final Files\.Figure\PDFs\compressed pdf"
)

RECURSIVE = True

TARGET_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# For page size / "physical" scaling in PDF
DESIRED_DPI = 600

# Hard target: maximum PDF size per file
MAX_PDF_MB = 1.0

# Compression search space (quality: higher = better, larger file)
JPEG_QUALITIES = [92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50]

# If quality alone doesn't reach the size cap, try light downscales
DOWNSCALE_STEPS = [1.00, 0.95, 0.90, 0.85, 0.80]

# If you want to preserve sharp text/lines, keep downscale limited (above).
# -----------------------------
# HELPERS
# -----------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def flatten_alpha_to_white(img: Image.Image) -> Image.Image:
    """Flatten transparency onto white background for clean PDFs."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def pdf_page_size_points(width_px: int, height_px: int, dpi: int) -> tuple[float, float]:
    """Convert pixel dimensions to PDF points using a chosen DPI."""
    width_in = width_px / float(dpi)
    height_in = height_px / float(dpi)
    return width_in * 72.0, height_in * 72.0


def build_pdf_bytes_from_jpeg(jpeg_bytes: bytes, page_w_pt: float, page_h_pt: float) -> bytes:
    """Create a single-page PDF containing the JPEG image."""
    doc = fitz.open()
    page = doc.new_page(width=page_w_pt, height=page_h_pt)
    rect = fitz.Rect(0, 0, page_w_pt, page_h_pt)
    page.insert_image(rect, stream=jpeg_bytes)

    out_mem = io.BytesIO()
    # deflate/clean reduces size a bit
    doc.save(out_mem, garbage=4, deflate=True, clean=True)
    doc.close()
    return out_mem.getvalue()


def compress_image_to_pdf_under_limit(img: Image.Image, out_pdf: Path, max_mb: float) -> dict:
    """
    Try JPEG qualities + downscale to produce a PDF under max_mb.
    Writes the best match; returns chosen params and final size.
    """
    safe_mkdir(out_pdf.parent)
    target_bytes = int(max_mb * 1024 * 1024)

    # Page size should correspond to the *final* pixel dimensions at DESIRED_DPI.
    # We keep physical size correct for 600 DPI.
    best = None  # (size_bytes, scale, quality, pdf_bytes)

    for scale in DOWNSCALE_STEPS:
        if scale == 1.0:
            scaled = img
        else:
            new_w = max(1, int(round(img.size[0] * scale)))
            new_h = max(1, int(round(img.size[1] * scale)))
            scaled = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        page_w_pt, page_h_pt = pdf_page_size_points(scaled.size[0], scaled.size[1], DESIRED_DPI)

        for q in JPEG_QUALITIES:
            bio = io.BytesIO()
            # Progressive + optimize tends to reduce size while keeping quality decent
            scaled.save(bio, format="JPEG", quality=q, optimize=True, progressive=True)
            jpeg_bytes = bio.getvalue()

            pdf_bytes = build_pdf_bytes_from_jpeg(jpeg_bytes, page_w_pt, page_h_pt)
            size_b = len(pdf_bytes)

            if best is None or size_b < best[0]:
                best = (size_b, scale, q, pdf_bytes)

            if size_b <= target_bytes:
                out_pdf.write_bytes(pdf_bytes)
                return {"hit_target": True, "size_bytes": size_b, "scale": scale, "quality": q}

    # If we can't hit the target, write the smallest we found
    size_b, scale, q, pdf_bytes = best
    out_pdf.write_bytes(pdf_bytes)
    return {"hit_target": False, "size_bytes": size_b, "scale": scale, "quality": q}


def iter_input_files(input_dir: Path):
    if RECURSIVE:
        yield from input_dir.rglob("*")
    else:
        yield from input_dir.glob("*")


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input folder not found:\n  {INPUT_DIR}")
        sys.exit(1)

    safe_mkdir(OUTPUT_DIR)

    files = []
    for p in iter_input_files(INPUT_DIR):
        if p.is_file() and p.suffix.lower() in TARGET_EXTS:
            files.append(p)

    if not files:
        print("[INFO] No image files found to process.")
        sys.exit(0)

    print(f"[INFO] Found {len(files)} image(s).")
    print(f"[INFO] Output PDFs -> {OUTPUT_DIR}")
    print(f"[INFO] Target size <= {MAX_PDF_MB:.2f} MB each\n")

    for fp in sorted(files):
        try:
            rel = fp.relative_to(INPUT_DIR)
            out_dir = OUTPUT_DIR / rel.parent
            safe_mkdir(out_dir)

            out_pdf = out_dir / f"{fp.stem}_compressed.pdf"

            img = Image.open(fp)
            img = ImageOps.exif_transpose(img)
            img = flatten_alpha_to_white(img)

            info = compress_image_to_pdf_under_limit(img, out_pdf, max_mb=MAX_PDF_MB)

            size_mb = info["size_bytes"] / (1024 * 1024)
            status = "OK" if info["hit_target"] else "BEST-EFFORT"
            print(f"{status}: {rel.as_posix()} -> {out_pdf.name} | {size_mb:.2f} MB | "
                  f"scale={info['scale']}, q={info['quality']}")

        except Exception as e:
            print(f"[ERROR] Failed on {fp.name}: {e}")

    print("\n[DONE] Compressed PDFs created.")
    print("Note: If some PDFs remain >1 MB, the script still writes the smallest possible version.")
    print("      For perfect quality + small files, vector export (PDF/SVG) is always best, but this is the best option for PNG inputs.")


if __name__ == "__main__":
    main()
