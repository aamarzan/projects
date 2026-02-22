#!/usr/bin/env python3
"""
Figure 3 — PCR–RFLP Signature Atlas (bioinformatics + wet-lab hybrid)

What it does (world-class, not a montage):
1) Parses your manuscript DOCX to extract, per locus:
   - Mutation name (e.g., T19I)
   - Gene (S gene / N gene / ORF1ab)
   - PCR amplicon size (bp)
   - Expected WT bands (bp list)
   - Expected Mut bands (bp list)
   - gain-of-site / loss-of-site (if stated)
2) Builds a top coordinate strip showing targeted loci across S / ORF1ab / N (using AA position from mutation code).
3) Builds a tiled atlas. Each tile shows a clean, synthetic "expected gel":
   Ladder | WT | Mut with straight bands and controlled shine on a crystal-black background.
4) Outputs: PNG (600 dpi), PDF, EPS.

Optional:
- Provide --gel_manifest CSV to inset real representative gel images per locus (without altering their band geometry).

Run example (Windows):
python fig3_rflp_signature_atlas.py ^
  --docx "E:\\Dr. Ahmed\\Ongoing\\RFLP\\SARS_Verification_RFLP_frontiers_v4_ready.docx" ^
  --outdir "E:\\Dr. Ahmed\\Ongoing\\RFLP\\Final_Figures" ^
  --cols 4 --dpi 600 --eps
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

try:
    import docx  # python-docx
except ImportError as e:
    raise SystemExit("python-docx is required. Install with: pip install python-docx") from e


# -----------------------------
# Styling (journal-safe)
# -----------------------------
def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.facecolor": "black",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


@dataclass
class Assay:
    locus: str           # e.g., "T19I", "HV69-70del"
    gene: str            # e.g., "S gene", "ORF1ab", "N gene"
    aa_pos: Optional[int]  # parsed from locus where possible
    pcr_bp: Optional[int]
    wt_frags: List[int]
    mut_frags: List[int]
    mechanism: str       # "gain-of-site" / "loss-of-site" / "unspecified"
    notes: str           # any extra info


# -----------------------------
# DOCX parsing
# -----------------------------
HDR_RE = re.compile(r"^3\.\d+\.\d+\s+(.+?)\s+\(([^)]+)\)\s*$")

def _first_int(s: str) -> Optional[int]:
    m = re.search(r"(\d{2,5})", s)
    return int(m.group(1)) if m else None

def parse_aa_pos(locus: str) -> Optional[int]:
    """
    Extract AA position from mutation code.
    Examples:
      T19I -> 19
      HV69-70del -> 69 (first)
      ERS31-33del -> 31
      SGF3675-3677del -> 3675
      R5716C -> 5716
    """
    m = re.search(r"(\d{1,5})", locus)
    return int(m.group(1)) if m else None

def extract_sizes_from_sentence(sent: str) -> List[int]:
    # collect numbers like "531 bp", "77 bp"
    nums = re.findall(r"(\d{2,5})\s*bp", sent)
    return [int(x) for x in nums]

def parse_block_for_assay(locus: str, gene: str, block: str) -> Assay:
    # Normalize whitespace
    t = " ".join(block.split())

    mechanism = "unspecified"
    if re.search(r"gain-?of-?site", t, re.IGNORECASE):
        mechanism = "gain-of-site"
    elif re.search(r"loss-?of-?site", t, re.IGNORECASE):
        mechanism = "loss-of-site"

    # PCR amplicon size: prefer "... produced a XXX bp PCR product" or "single XXX bp band"
    pcr_bp = None
    for pat in [
        r"produced a\s+(\d{2,5})\s*bp\s+PCR product",
        r"yielding a single\s+(\d{2,5})\s*bp\s+band",
        r"as a single\s+(\d{2,5})\s*bp\s+band",
        r"(\d{2,5})\s*bp\s+PCR product",
    ]:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            pcr_bp = int(m.group(1))
            break

    wt_frags: List[int] = []
    mut_frags: List[int] = []

    # Split into sentences for targeted extraction
    sentences = re.split(r"(?<=[.!?])\s+", t)

    # Find WT sentence(s)
    for s in sentences:
        if re.search(r"\bwild-?type\b", s, re.IGNORECASE):
            if re.search(r"undigested", s, re.IGNORECASE) and pcr_bp:
                wt_frags = [pcr_bp]
            elif re.search(r"(digested|cleaved)\s+into", s, re.IGNORECASE):
                wt_frags = extract_sizes_from_sentence(s)

    # Find Mut sentence(s)
    for s in sentences:
        if re.search(r"\bmutant\b|\bvariant\b|\bdeletion\b", s, re.IGNORECASE):
            # avoid grabbing WT-only sentences
            if re.search(r"\bwild-?type\b", s, re.IGNORECASE):
                continue
            if re.search(r"undigested", s, re.IGNORECASE) and pcr_bp:
                mut_frags = [pcr_bp]
            elif re.search(r"(digested|cleaved)\s+into", s, re.IGNORECASE):
                mut_frags = extract_sizes_from_sentence(s)

    # Fallback: if one side missing but we have pcr_bp, keep it readable
    if pcr_bp and not wt_frags:
        # if text mentions WT undigested anywhere
        if re.search(r"wild-?type.*undigested", t, re.IGNORECASE):
            wt_frags = [pcr_bp]
    if pcr_bp and not mut_frags:
        if re.search(r"(mutant|variant|deletion).*undigested", t, re.IGNORECASE):
            mut_frags = [pcr_bp]

    notes = ""
    # Known tricky locus: K417N sometimes has site-dependent outcomes; keep note if seen
    if locus.upper() == "K417N":
        notes = "K417N can be site-dependent in text; verify lane interpretation."

    return Assay(
        locus=locus.strip(),
        gene=gene.strip(),
        aa_pos=parse_aa_pos(locus),
        pcr_bp=pcr_bp,
        wt_frags=[x for x in wt_frags if x > 0],
        mut_frags=[x for x in mut_frags if x > 0],
        mechanism=mechanism,
        notes=notes,
    )

def parse_docx_assays(docx_path: Path) -> List[Assay]:
    d = docx.Document(str(docx_path))
    paras = [p.text.strip() for p in d.paragraphs]

    # find headers and blocks
    idxs: List[Tuple[int, str, str]] = []  # (i, locus, gene)
    for i, tx in enumerate(paras):
        m = HDR_RE.match(tx)
        if m:
            locus, gene = m.group(1), m.group(2)
            idxs.append((i, locus, gene))

    assays: List[Assay] = []
    for k, (i, locus, gene) in enumerate(idxs):
        j = idxs[k + 1][0] if k + 1 < len(idxs) else len(paras)
        block = "\n".join([p for p in paras[i + 1:j] if p])
        if not block:
            continue
        assays.append(parse_block_for_assay(locus, gene, block))

    # Keep only assays where we extracted at least PCR size OR fragments
    assays = [a for a in assays if (a.pcr_bp is not None) or a.wt_frags or a.mut_frags]
    return assays


# -----------------------------
# Optional real-gel insets
# -----------------------------
def read_gel_manifest(csv_path: Path) -> Dict[str, Path]:
    """
    CSV format (header required):
      locus,image_path
      T19I,E:\\...\\f79.png
    """
    out: Dict[str, Path] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            locus = (row.get("locus") or "").strip()
            ip = (row.get("image_path") or "").strip()
            if locus and ip:
                out[locus] = Path(ip)
    return out


# -----------------------------
# Mini synthetic gel renderer
# -----------------------------
def bp_to_y(bp: int, bp_min: int, bp_max: int) -> float:
    """
    Map bp to y in [0,1], with log scaling (big bp higher).
    """
    bp = max(bp_min, min(bp, bp_max))
    lo = math.log10(bp_min)
    hi = math.log10(bp_max)
    v = (math.log10(bp) - lo) / (hi - lo) if hi > lo else 0.5
    return 0.92 - 0.82 * v  # top margin to bottom margin

def draw_band(ax, x0: float, x1: float, y: float, core_h: float, glow_h: float, alpha: float):
    # glow
    ax.add_patch(Rectangle((x0, y - glow_h / 2), x1 - x0, glow_h,
                           facecolor="white", edgecolor="none", alpha=max(0.08, alpha * 0.35)))
    # core
    ax.add_patch(Rectangle((x0, y - core_h / 2), x1 - x0, core_h,
                           facecolor="white", edgecolor="none", alpha=alpha))

def draw_lane(ax, x_center: float, bands_bp: List[int], bp_min: int, bp_max: int, lane_w: float = 0.18):
    x0, x1 = x_center - lane_w / 2, x_center + lane_w / 2
    for bp in sorted(bands_bp, reverse=True):
        y = bp_to_y(bp, bp_min, bp_max)
        # intensity: bigger fragments slightly brighter; small fragments fainter
        rel = (bp - bp_min) / max(1, (bp_max - bp_min))
        alpha = 0.55 + 0.35 * rel
        alpha = max(0.35, min(alpha, 0.95))
        core_h = 0.028 + 0.015 * (1 - rel)
        glow_h = core_h * 2.2
        draw_band(ax, x0, x1, y, core_h=core_h, glow_h=glow_h, alpha=alpha)

def draw_tile(ax, assay: Assay, bp_min: int, bp_max: int, gel_img: Optional[Image.Image] = None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # tile background (pure black gel area)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="none"))

    # Header text
    locus = assay.locus
    gene = assay.gene
    mech = assay.mechanism
    header = f"{locus}  |  {gene}"
    ax.text(0.02, 0.98, header, ha="left", va="top",
            color="white", fontsize=8.5, weight="bold")

    # Mechanism badge
    badge = mech if mech != "unspecified" else ""
    if badge:
        ax.text(0.98, 0.98, badge, ha="right", va="top", color="white", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="white", lw=0.6, alpha=0.85))

    # Gel drawing region
    # Option A: inset real gel
    if gel_img is not None:
        # preserve original geometry; just fit it in the tile
        ax.imshow(gel_img, extent=(0.05, 0.95, 0.12, 0.88), aspect="auto")
    else:
        # Synthetic expected gel
        # Ladder lane (generic)
        ladder = [1000, 700, 500, 400, 300, 200, 150, 100, 75, 50]
        draw_lane(ax, 0.20, ladder, bp_min, bp_max, lane_w=0.20)

        wt = assay.wt_frags[:] if assay.wt_frags else ([assay.pcr_bp] if assay.pcr_bp else [])
        mut = assay.mut_frags[:] if assay.mut_frags else ([assay.pcr_bp] if assay.pcr_bp else [])

        draw_lane(ax, 0.52, wt, bp_min, bp_max, lane_w=0.22)
        draw_lane(ax, 0.82, mut, bp_min, bp_max, lane_w=0.22)

        # Lane labels
        ax.text(0.20, 0.08, "L", ha="center", va="bottom", color="white", fontsize=7, alpha=0.85)
        ax.text(0.52, 0.08, "WT", ha="center", va="bottom", color="white", fontsize=7, alpha=0.85)
        ax.text(0.82, 0.08, "Mut", ha="center", va="bottom", color="white", fontsize=7, alpha=0.85)

    # Footer: band signature
    def sig(frags: List[int]) -> str:
        if not frags:
            return "?"
        return "+".join(str(x) for x in frags)

    wt_sig = sig(assay.wt_frags if assay.wt_frags else ([assay.pcr_bp] if assay.pcr_bp else []))
    mut_sig = sig(assay.mut_frags if assay.mut_frags else ([assay.pcr_bp] if assay.pcr_bp else []))
    ax.text(0.02, 0.02, f"WT: {wt_sig} bp   |   Mut: {mut_sig} bp",
            ha="left", va="bottom", color="white", fontsize=7.2, alpha=0.92)

    if assay.notes:
        ax.text(0.98, 0.02, assay.notes, ha="right", va="bottom", color="white",
                fontsize=6.5, alpha=0.70)


# -----------------------------
# Coordinate strip (bioinformatics)
# -----------------------------
def draw_coordinate_strip(ax, assays: List[Assay]):
    ax.set_axis_off()

    # group by gene
    groups: Dict[str, List[Assay]] = {}
    for a in assays:
        g = a.gene
        groups.setdefault(g, []).append(a)

    # Order (nice for your study)
    order = []
    for key in ["S gene", "ORF1ab", "ORF1ab gene", "N gene"]:
        if key in groups:
            order.append(key)
    # add anything else
    for k in groups:
        if k not in order:
            order.append(k)

    y0 = 0.80
    dy = 0.24 if len(order) <= 3 else 0.18

    for gi, g in enumerate(order):
        items = [a for a in groups[g] if a.aa_pos is not None]
        if not items:
            continue
        items.sort(key=lambda x: x.aa_pos or 0)

        aa_min = min(a.aa_pos for a in items if a.aa_pos is not None)
        aa_max = max(a.aa_pos for a in items if a.aa_pos is not None)
        if aa_max == aa_min:
            aa_max = aa_min + 1

        y = y0 - gi * dy
        # baseline
        ax.add_patch(Rectangle((0.08, y - 0.02), 0.84, 0.04, facecolor="#111111", edgecolor="#000000"))

        ax.text(0.02, y, g, ha="left", va="center", fontsize=9, weight="bold", color="black")

        # ticks
        for a in items:
            x = 0.08 + 0.84 * ((a.aa_pos - aa_min) / (aa_max - aa_min))
            ax.add_patch(Rectangle((x - 0.0015, y - 0.06), 0.003, 0.12, facecolor="black", edgecolor="none"))
            ax.text(x, y + 0.08, a.locus, ha="center", va="bottom", fontsize=7, rotation=45, color="black")

        ax.text(0.08, y - 0.10, f"AA {aa_min}", ha="left", va="top", fontsize=7, color="black")
        ax.text(0.92, y - 0.10, f"AA {aa_max}", ha="right", va="top", fontsize=7, color="black")


# -----------------------------
# Main
# -----------------------------
def main(docx_path: Path,
         outdir: Path,
         cols: int,
         dpi: int,
         make_eps: bool,
         gel_manifest: Optional[Path],
         max_tiles: Optional[int]):

    apply_style()
    assays = parse_docx_assays(docx_path)

    if not assays:
        raise SystemExit("No assays parsed from DOCX. Check that section headers (3.x.x ...) exist in the document.")

    # optionally limit tiles
    if max_tiles is not None:
        assays = assays[:max_tiles]

    # Determine bp scale for synthetic gels
    all_sizes = []
    for a in assays:
        if a.pcr_bp:
            all_sizes.append(a.pcr_bp)
        all_sizes.extend(a.wt_frags)
        all_sizes.extend(a.mut_frags)
    bp_max = int(max(all_sizes)) if all_sizes else 1000
    bp_min = 40  # lower bound for display

    # Optional gel insets
    gels: Dict[str, Path] = {}
    if gel_manifest:
        gels = read_gel_manifest(gel_manifest)

    # Layout
    n = len(assays)
    cols = max(2, cols)
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(8.3, 10.8))  # A4-ish, journal friendly
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 9.0], hspace=0.15)

    # Panel A: coordinate strip
    axA = fig.add_subplot(gs[0, 0])
    draw_coordinate_strip(axA, assays)

    # Panel B: atlas grid
    gsB = gs[1, 0].subgridspec(rows, cols, hspace=0.18, wspace=0.12)

    for i, a in enumerate(assays):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gsB[r, c])

        gel_img = None
        if a.locus in gels and gels[a.locus].exists():
            im = Image.open(gels[a.locus])
            if im.mode != "RGB":
                im = im.convert("RGB")
            gel_img = im

        draw_tile(ax, a, bp_min=bp_min, bp_max=bp_max, gel_img=gel_img)

    # turn off unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gsB[r, c])
        ax.set_axis_off()

    fig.suptitle(
        "Figure 3. PCR–RFLP signature atlas (in-silico fragment predictions + expected gel readout)",
        x=0.01, ha="left", fontsize=12, weight="bold"
    )

    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "Figure_3_RFLP_Signature_Atlas"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".pdf"))
    if make_eps:
        fig.savefig(stem.with_suffix(".eps"))
    plt.close(fig)

    # also write a debug CSV of extracted signatures
    dbg = outdir / "Figure_3_RFLP_Signature_Atlas_extracted_signatures.csv"
    with dbg.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["locus", "gene", "aa_pos", "pcr_bp", "wt_frags", "mut_frags", "mechanism", "notes"])
        for a in assays:
            w.writerow([a.locus, a.gene, a.aa_pos or "", a.pcr_bp or "",
                        "+".join(map(str, a.wt_frags)), "+".join(map(str, a.mut_frags)),
                        a.mechanism, a.notes])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True, help="Path to SARS_Verification_RFLP_frontiers_v4_ready.docx")
    ap.add_argument("--outdir", default="out_figures", help="Output directory")
    ap.add_argument("--cols", type=int, default=4, help="Number of atlas columns")
    ap.add_argument("--dpi", type=int, default=600, help="PNG DPI")
    ap.add_argument("--eps", action="store_true", help="Also export EPS")
    ap.add_argument("--gel_manifest", default=None,
                    help="Optional CSV mapping locus->image_path to inset real gels (keeps original geometry intact)")
    ap.add_argument("--max_tiles", type=int, default=None, help="Debug: limit number of tiles")
    args = ap.parse_args()

    main(
        docx_path=Path(args.docx),
        outdir=Path(args.outdir),
        cols=args.cols,
        dpi=args.dpi,
        make_eps=args.eps,
        gel_manifest=Path(args.gel_manifest) if args.gel_manifest else None,
        max_tiles=args.max_tiles
    )