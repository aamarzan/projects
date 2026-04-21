from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject

MM_TO_PT = 72.0 / 25.4
A4_PORTRAIT = (210.0 * MM_TO_PT, 297.0 * MM_TO_PT)
A4_LANDSCAPE = (297.0 * MM_TO_PT, 210.0 * MM_TO_PT)


def wrap_pdf_page_to_a4(
    input_pdf: str | Path,
    output_pdf: str | Path,
    *,
    landscape: bool = True,
    margin_mm: float = 12.0,
) -> Path:
    """
    Wrap a one-page PDF onto a fixed A4 page while preserving vector content.

    The source page is scaled uniformly and centered inside the usable area.
    This standardizes the MediaBox/CropBox across figures without altering the
    native plotting layout that generated the original PDF.
    """
    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf)

    page_w, page_h = A4_LANDSCAPE if landscape else A4_PORTRAIT
    margin_pt = float(margin_mm) * MM_TO_PT
    usable_w = page_w - 2.0 * margin_pt
    usable_h = page_h - 2.0 * margin_pt

    reader = PdfReader(str(input_pdf))
    if len(reader.pages) != 1:
        raise ValueError(f"Expected a single-page PDF: {input_pdf}")
    src = reader.pages[0]

    src_w = float(src.mediabox.width)
    src_h = float(src.mediabox.height)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source PDF page size for {input_pdf}")

    scale = min(usable_w / src_w, usable_h / src_h)
    draw_w = src_w * scale
    draw_h = src_h * scale
    tx = (page_w - draw_w) / 2.0
    ty = (page_h - draw_h) / 2.0

    blank = PageObject.create_blank_page(width=page_w, height=page_h)
    blank.merge_transformed_page(
        src,
        Transformation().scale(scale, scale).translate(tx=tx, ty=ty),
    )

    writer = PdfWriter()
    writer.add_page(blank)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdf, "wb") as f:
        writer.write(f)
    return output_pdf


def save_figure_pdf_a4(
    fig,
    output_pdf: str | Path,
    *,
    facecolor: str = "white",
    landscape: bool = True,
    margin_mm: float = 12.0,
    native_savefig_kwargs: Optional[Dict[str, Any]] = None,
    cleanup_native: bool = True,
) -> Path:
    """
    Save a matplotlib figure as PDF, then normalize it onto a fixed A4 page.

    The native PDF is written first (preserving the original layout/crop rules),
    then wrapped into a standardized A4 portrait/landscape page.
    """
    output_pdf = Path(output_pdf)
    native_pdf = output_pdf.with_name(output_pdf.stem + "__native__.pdf")

    save_kwargs: Dict[str, Any] = {"facecolor": facecolor}
    if native_savefig_kwargs:
        save_kwargs.update(native_savefig_kwargs)

    fig.savefig(native_pdf, **save_kwargs)
    wrap_pdf_page_to_a4(native_pdf, output_pdf, landscape=landscape, margin_mm=margin_mm)

    if cleanup_native:
        try:
            native_pdf.unlink()
        except FileNotFoundError:
            pass

    return output_pdf
