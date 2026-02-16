# Figure_2_final_v2.py
# Tweaks added:
# 1) Left panel major tick label shows 2016 (instead of 2015)
# 2) Optional paraquat bridge across the axis break:
#    - default: broken (same as now)
#    - --paraquat_no_gap: draws a continuous paraquat line across the gap (2013→2024)

from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PRISM_BLUE = "#00008B"
PRISM_ORANGE = "#FF7F0E"
PRISM_RED = "#8B0000"


def linreg_p_text(x, y):
    import numpy as np
    from scipy.stats import t as student_t

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    resid = y - yhat

    n = len(x)
    sse = np.sum(resid**2)
    s2 = sse / (n - 2)
    xbar = np.mean(x)
    sxx = np.sum((x - xbar) ** 2)
    se_slope = np.sqrt(s2 / sxx) if sxx != 0 else 0
    if se_slope == 0:
        return "P <0.0001"

    t = slope / se_slope
    df = n - 2
    p = 2 * student_t.sf(abs(t), df=df)

    return "P <0.0001" if p < 0.0001 else f"P = {p:.4f}"


def bold_ticklabels(ax):
    for lab in ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"):
        lab.set_fontweight("bold")


def nice_upper(v: float, step: int) -> int:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return step
    v = float(v)
    if v <= 0:
        return step
    return int(math.ceil(v / step) * step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("excel", type=Path)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--out", type=Path, default=Path("Figure2_final_v2"))

    ap.add_argument("--year_col", default="Year")
    ap.add_argument("--paraquat_col", default="Agriculture use of paraquat")
    ap.add_argument("--cases_col", default="No of cases")
    ap.add_argument("--deaths_col", default="No of deaths")

    ap.add_argument("--left_start", type=int, default=2013)
    ap.add_argument("--left_end", type=int, default=2016)
    ap.add_argument("--right_start", type=int, default=2021)
    ap.add_argument("--right_end", type=int, default=2024)

    # NEW: if set, draws a continuous paraquat line across the broken axis
    ap.add_argument("--paraquat_no_gap", action="store_true",
                    help="Draw paraquat line continuously across the break (bridge 2016→2021).")

    args = ap.parse_args()

    sheet_to_read = 0 if args.sheet is None else args.sheet
    df = pd.read_excel(args.excel, sheet_name=sheet_to_read)

    needed = [args.year_col, args.paraquat_col, args.cases_col, args.deaths_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    df = df[needed].copy()
    df[args.year_col] = pd.to_numeric(df[args.year_col], errors="coerce")
    for c in [args.paraquat_col, args.cases_col, args.deaths_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[args.year_col]).sort_values(args.year_col)

    left = df[(df[args.year_col] >= args.left_start) & (df[args.year_col] <= args.left_end)].copy()
    right = df[(df[args.year_col] >= args.right_start) & (df[args.year_col] <= args.right_end)].copy()

    if left.empty or right.empty:
        raise ValueError(
            "Split produced empty panel. Check split years vs your data. "
            f"Left: {args.left_start}-{args.left_end}, Right: {args.right_start}-{args.right_end}"
        )

    plot_df = pd.concat([left, right], ignore_index=True)

    pq_max = float(np.nanmax(plot_df[args.paraquat_col].to_numpy(float)))
    left_lim = nice_upper(pq_max, step=1000)

    cd_max = float(
        np.nanmax(
            np.concatenate(
                [
                    plot_df[args.cases_col].to_numpy(float),
                    plot_df[args.deaths_col].to_numpy(float),
                ]
            )
        )
    )
    right_step = 50 if cd_max <= 250 else 100
    right_lim = nice_upper(cd_max, step=right_step)
    if right_lim == cd_max:
        right_lim += right_step

    left_span = max(args.left_end - args.left_start, 1)
    right_span = max(args.right_end - args.right_start, 1)

    fig = plt.figure(figsize=(14.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[left_span, right_span], wspace=0.55)

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], sharey=axL)
    axL2 = axL.twinx()
    axR2 = axR.twinx()

    para_marker = "|"
    para_ms = 16
    para_mew = 3.0

    # ---- Left fragment ----
    axL.plot(
        left[args.year_col], left[args.paraquat_col],
        color=PRISM_BLUE, linewidth=3.0,
        marker=para_marker, markersize=para_ms, markeredgewidth=para_mew,
    )
    axL2.plot(
        left[args.year_col], left[args.cases_col],
        color=PRISM_ORANGE, linewidth=3.0,
        marker="s", markersize=8,
    )
    axL2.plot(
        left[args.year_col], left[args.deaths_col],
        color=PRISM_RED, linewidth=3.0,
        marker="^", markersize=9,
    )

    # ---- Right fragment ----
    axR.plot(
        right[args.year_col], right[args.paraquat_col],
        color=PRISM_BLUE, linewidth=3.0,
        marker=para_marker, markersize=para_ms, markeredgewidth=para_mew,
    )
    axR2.plot(
        right[args.year_col], right[args.cases_col],
        color=PRISM_ORANGE, linewidth=3.0,
        marker="s", markersize=8,
    )
    axR2.plot(
        right[args.year_col], right[args.deaths_col],
        color=PRISM_RED, linewidth=3.0,
        marker="^", markersize=9,
    )

    # ---- X axis limits ----
    axL.set_xlim(args.left_start, args.left_end + 0.7)
    axR.set_xlim(args.right_start - 0.7, args.right_end + 0.7)

    # ---- X ticks ----
    # Tweak 1: show 2016 (NOT 2015) as the second major tick on left panel
    axL.set_xticks([args.left_start, args.left_end])
    axL.set_xticks(np.arange(args.left_start, args.left_end + 1, 1), minor=True)

    axR.set_xticks([args.right_start, args.right_end])
    axR.set_xticks(np.arange(args.right_start, args.right_end + 1, 1), minor=True)

    # ---- Y axes ----
    axL.set_ylim(0, left_lim)
    axL.set_yticks(list(range(0, left_lim + 1, 1000)))

    axR2.set_ylim(0, right_lim)
    axR2.set_yticks(list(range(0, right_lim + 1, right_step)))

    axL2.set_ylim(0, right_lim)
    axL2.set_yticks(list(range(0, right_lim + 1, right_step)))

    axL.set_ylabel("Annual use of Paraquat in Ton/KL", fontsize=14, fontweight="bold")

    axR2.set_ylabel("")
    axR2.tick_params(axis="y", pad=10)

    axL2.tick_params(axis="y", labelright=False, right=False)
    axL2.spines["right"].set_visible(False)

    axR.tick_params(axis="y", left=False, labelleft=False)
    axR.spines["left"].set_visible(False)
    axR.spines["right"].set_visible(False)
    axR2.spines["left"].set_visible(False)
    axR2.tick_params(axis="y", left=False, labelleft=False)

    for ax in (axL, axR, axL2, axR2):
        ax.grid(False)
        ax.spines["top"].set_visible(False)

    axL.spines["right"].set_visible(False)
    axL2.spines["bottom"].set_visible(False)
    axR2.spines["bottom"].set_visible(False)

    axL.spines["left"].set_linewidth(2.2)
    axL.spines["bottom"].set_linewidth(2.2)
    axR.spines["bottom"].set_linewidth(2.2)
    axR2.spines["right"].set_linewidth(2.2)

    for ax in (axL, axR, axR2):
        ax.tick_params(axis="both", which="major", labelsize=12, width=2.0, length=8)
        ax.tick_params(axis="both", which="minor", width=1.5, length=4)

    bold_ticklabels(axL)
    bold_ticklabels(axR)
    bold_ticklabels(axR2)

    legend_items = [
        Line2D([0], [0], color=PRISM_BLUE, lw=3.0, marker=para_marker,
               markersize=para_ms, markeredgewidth=para_mew, label="Agricultural use of\nparaquat"),
        Line2D([0], [0], color=PRISM_ORANGE, lw=3.0, marker="s", markersize=8, label="Number of cases"),
        Line2D([0], [0], color=PRISM_RED, lw=3.0, marker="^", markersize=9, label="Number of deaths"),
    ]

    # If you previously tuned legend position, keep your preferred X here (e.g., 1.60)
    axR.legend(handles=legend_items, loc="center left", bbox_to_anchor=(1.60, 0.55),
              frameon=False, fontsize=14)

    fig.subplots_adjust(left=0.08, right=0.52, bottom=0.22, top=0.78)

    fig.canvas.draw()
    inv = fig.transFigure.inverted()

    # Dots between fragments
    x1_disp, y1_disp = axL.transAxes.transform((1.0, 0.0))
    x2_disp, y2_disp = axR.transAxes.transform((0.0, 0.0))
    x1_fig, y1_fig = inv.transform((x1_disp, y1_disp))
    x2_fig, y2_fig = inv.transform((x2_disp, y2_disp))
    mid_x = (x1_fig + x2_fig) / 2
    base_y = (y1_fig + y2_fig) / 2
    fig.text(mid_x, base_y + 0.006, "···", ha="center", va="center",
             fontsize=20, fontweight="bold", zorder=10)

    # Right Y label (outside ticks)
    posR2 = axR2.get_position()
    fig.text(
        posR2.x1 + 0.065,
        (posR2.y0 + posR2.y1) / 2,
        "Number of cases and deaths",
        rotation=270,
        va="center",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    # Centered "Year" across both fragments
    posL = axL.get_position()
    posR = axR.get_position()
    x_mid_all = (posL.x0 + posR.x1) / 2
    y_year = posL.y0 - 0.055
    fig.text(x_mid_all, y_year, "Year", ha="center", va="center",
             fontsize=14, fontweight="bold")

    # Footnote (close)
    y_note = y_year - 0.040
    fig.text(posL.x0, y_note, "Note: Clinical data for 2017-2020 are not available. Data for 2021 are partial due to the COVID-19 pandemic. Data for 2024 are available upto June",
             ha="left", va="center", fontsize=11)

    # P-value (plotted years only)
    ptxt = linreg_p_text(plot_df[args.year_col].to_numpy(float),
                         plot_df[args.paraquat_col].to_numpy(float))
    if ptxt:
        axL.text(0.42, 0.94, ptxt, transform=axL.transAxes,
                 fontsize=14, fontstyle="italic", fontweight="bold")

    # ---- Tweak 2: Optional continuous paraquat line across the gap ----
    if args.paraquat_no_gap:
        # connect last point of LEFT paraquat to first point of RIGHT paraquat
        xL = float(left[args.year_col].iloc[-1])
        yL = float(left[args.paraquat_col].iloc[-1])
        xR = float(right[args.year_col].iloc[0])
        yR = float(right[args.paraquat_col].iloc[0])

        # Convert those data points to FIGURE coordinates and draw a line in figure space
        pL_disp = axL.transData.transform((xL, yL))
        pR_disp = axR.transData.transform((xR, yR))
        pL_fig = inv.transform(pL_disp)
        pR_fig = inv.transform(pR_disp)

        bridge = Line2D(
            [pL_fig[0], pR_fig[0]],
            [pL_fig[1], pR_fig[1]],
            transform=fig.transFigure,
            color=PRISM_BLUE,
            linewidth=3.0,
            zorder=2  # behind the "···" marker
        )
        fig.add_artist(bridge)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)

    print("Saved:")
    print(out.with_suffix(".png"))
    print(out.with_suffix(".pdf"))
    print(out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
