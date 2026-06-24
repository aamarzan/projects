import fitz  # PyMuPDF
import argparse

A4_PORTRAIT = (595.276, 841.89)  # points
A4_LANDSCAPE = (841.89, 595.276)

def fit_to_a4(in_pdf, out_pdf, landscape=True, margin_pt=18):
    src = fitz.open(in_pdf)
    dst = fitz.open()

    W, H = (A4_LANDSCAPE if landscape else A4_PORTRAIT)

    for i in range(src.page_count):
        sp = src.load_page(i)
        dp = dst.new_page(width=W, height=H)

        target = fitz.Rect(margin_pt, margin_pt, W - margin_pt, H - margin_pt)
        r = sp.rect

        s = min(target.width / r.width, target.height / r.height)

        new_w = r.width * s
        new_h = r.height * s

        x0 = target.x0 + (target.width - new_w) / 2
        y0 = target.y0 + (target.height - new_h) / 2
        dest = fitz.Rect(x0, y0, x0 + new_w, y0 + new_h)

        dp.show_pdf_page(dest, src, i)

    dst.save(out_pdf, deflate=True)
    dst.close()
    src.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_pdf")
    ap.add_argument("out_pdf")
    ap.add_argument("--portrait", action="store_true")
    ap.add_argument("--margin", type=float, default=18, help="margin in points (~18pt = 0.25in)")
    args = ap.parse_args()

    fit_to_a4(args.in_pdf, args.out_pdf, landscape=not args.portrait, margin_pt=args.margin)
