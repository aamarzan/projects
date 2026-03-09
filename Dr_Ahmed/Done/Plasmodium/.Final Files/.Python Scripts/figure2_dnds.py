#!/usr/bin/env python
"""
Make Figure 2: premium dN/dS summary plot.

- Expects a file called "dNdS summary.txt" in the same folder,
  containing lines like:
      Mean dN:                        0.002783
      Mean dS:                        0.007634
      Global omega (dN/dS):           0.3646
"""

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


def parse_dnds_summary(path: Path):
    """Parse mean dN, mean dS, and omega from dNdS summary.txt."""
    text = path.read_text(encoding="utf-8")

    def grab(label):
        pattern = rf"{re.escape(label)}\s*([0-9.eE+-]+)"
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"Could not find '{label}' in {path.name}")
        return float(m.group(1))

    mean_dn = grab("Mean dN:")
    mean_ds = grab("Mean dS:")
    omega = grab("Global omega (dN/dS):")

    return mean_dn, mean_ds, omega


def main():
    here = Path(__file__).resolve().parent
    summary_path = here / "dNdS summary.txt"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Could not find {summary_path.name} in {here}.\n"
            "Make sure dNdS summary.txt is in the same folder as this script."
        )

    mean_dn, mean_ds, omega = parse_dnds_summary(summary_path)
    print(f"Parsed from {summary_path.name}:")
    print(f"  mean dN  = {mean_dn:.6f}")
    print(f"  mean dS  = {mean_ds:.6f}")
    print(f"  omega    = {omega:.4f}")

    # --------------------------------------------------------------
    # Figure settings
    # --------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
    })

    # Show neutral threshold at 1.0 clearly
    x_max = 1.5 if omega <= 1.0 else max(omega * 1.2, 1.5)

    fig, ax = plt.subplots(figsize=(8.5, 3))

    # Custom blue gradient for the bar (very light -> deep blue)
    omega_cmap = LinearSegmentedColormap.from_list(
        "omega_blue",
        ["#e8f1ff", "#7aa7e6", "#0b3c85"]
    )

    # Background: purifying (blue tint) vs positive selection (peach tint)
    purifying_color = "#d5e3ff"
    positive_color = "#ffdfc6"

    # Purifying region: 0–1
    ax.add_patch(
        Rectangle(
            (0, 0.36),
            min(1.0, x_max),
            0.30,
            facecolor=purifying_color,
            edgecolor="none",
            zorder=0,
        )
    )
    # Positive region: 1–x_max
    if x_max > 1.0:
        ax.add_patch(
            Rectangle(
                (1.0, 0.36),
                x_max - 1.0,
                0.30,
                facecolor=positive_color,
                edgecolor="none",
                zorder=0,
            )
        )

    # Neutral threshold (ω = 1)
    if x_max >= 1.0:
        ax.axvline(
            1.0,
            ymin=0.28,
            ymax=0.78,
            color="0.4",
            linestyle=(0, (3, 3)),
            linewidth=1,
            zorder=2,
        )
        ax.text(
            1.0,
            0.82,
            "Neutral threshold\n(ω = 1.0)",
            ha="center",
            va="bottom",
            fontsize=9,
            color="0.4",
        )

    # Gradient bar for omega
    y_center = 0.51
    bar_height = 0.22
    n_segments = 80
    xs = np.linspace(0, omega, n_segments + 1)

    for i in range(n_segments):
        left = xs[i]
        width = xs[i + 1] - left
        frac = (i + 0.5) / n_segments
        color = omega_cmap(frac)
        ax.barh(
            y_center,
            width=width,
            left=left,
            height=bar_height,
            color=color,
            edgecolor="none",
            zorder=3,
        )

    # Outline of the bar
    ax.barh(
        y_center,
        width=omega,
        left=0,
        height=bar_height,
        facecolor="none",
        edgecolor="#17335f",
        linewidth=1.3,
        zorder=4,
    )

    # Point marker at exact omega
    ax.scatter(
        [omega],
        [y_center],
        s=46,
        color="black",
        zorder=5,
    )
    ax.text(
        omega,
        y_center + 0.20,
        f"ω = {omega:.2f}",
        ha="center",
        va="bottom",
        fontsize=10.5,
    )

    # Labels of regimes
    ax.text(
        0.18 * x_max,
        0.17,
        "Purifying selection\n(ω < 1)",
        ha="center",
        va="center",
        color="#184a87",
        fontsize=9.5,
    )
    if x_max > 1.0:
        ax.text(
            0.78 * x_max,
            0.17,
            "Positive selection\n(ω > 1)",
            ha="center",
            va="center",
            color="#a04a00",
            fontsize=9.5,
        )

    # Axes formatting
    ax.set_yticks([y_center])
    ax.set_yticklabels(["Mean dN/dS ratio"])
    ax.set_xlabel("Mean dN/dS (ω)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)

    # Subtle vertical gridlines on x-axis
    xticks = np.linspace(0, x_max, 6)
    ax.set_xticks(xticks)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, color="0.8", zorder=0)
    ax.set_axisbelow(True)

    # Clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Extra note with mean dN and mean dS
    ax.text(
        0.99,
        0.03,
        f"mean dN = {mean_dn:.4g},  mean dS = {mean_ds:.4g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="0.35",
    )

    ax.set_title("Gene-wide dN/dS summary for CelTOS", pad=14)

    fig.tight_layout()
    out_path = here / "Figure2_dnds_summary.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
