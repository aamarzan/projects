from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap  # ⬅ add LSC


# ----------------------------------------------------------------------
# 1. Load mismatch.txt (must be in the same folder as this script)
# ----------------------------------------------------------------------
here = Path(__file__).resolve().parent
mismatch_path = here / "mismatch.txt"

text = mismatch_path.read_text(encoding="utf-8")

diffs = []
obs = []
exp = []

# Very tolerant parser: any line that starts with an integer followed by
# two numbers will be treated as (Number_of_differences, Observed, Expected)
for line in text.splitlines():
    line = line.strip()
    if not line:
        continue

    parts = line.replace("\t", " ").split()
    if len(parts) < 3:
        continue

    try:
        d = int(parts[0])
        o = float(parts[1])
        e = float(parts[2])
    except ValueError:
        # Not a data line (likely header or text) → skip
        continue

    diffs.append(d)
    obs.append(o)
    exp.append(e)

diffs = np.array(diffs, dtype=float)
obs = np.array(obs, dtype=float)
exp = np.array(exp, dtype=float)

if diffs.size == 0:
    raise RuntimeError(
        "Could not parse any mismatch classes from mismatch.txt.\n"
        "First column must be classes (0,1,2,...) and next two columns "
        "Observed and Expected frequencies."
    )

# ----------------------------------------------------------------------
# 2. Restrict plot to classes with information (here 0–15)
# ----------------------------------------------------------------------
mask = diffs <= 15
d_plot = diffs[mask]
o_plot = obs[mask]
e_plot = exp[mask]

# ----------------------------------------------------------------------
# 3. Build a SMOOTH expected curve (no plotting yet)
# ----------------------------------------------------------------------
x_smooth = np.linspace(d_plot.min(), d_plot.max(), 400)

try:
    # Preferred: cubic spline (very smooth) if SciPy is available
    from scipy.interpolate import make_interp_spline

    spline = make_interp_spline(d_plot, e_plot, k=3)  # cubic spline
    y_smooth = spline(x_smooth)

except Exception:
    # Fallback: interpolate + gentle 5-point smoothing (no SciPy needed)
    y_raw = np.interp(x_smooth, d_plot, e_plot)
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel /= kernel.sum()
    y_smooth = np.convolve(y_raw, kernel, mode="same")

# Avoid tiny negative values from smoothing
y_smooth = np.clip(y_smooth, 0.0, None)

# ----------------------------------------------------------------------
# 4. Create the premium-style mismatch figure
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titleweight": "bold"
})

fig, ax = plt.subplots(figsize=(9, 5))

# === Premium BLUE gradient based on OBSERVED frequency ===
# Light blue (low freq) → deep navy (high freq)
cmap = LinearSegmentedColormap.from_list(
    "premium_blues",
    ["#e7f1ff", "#4c8ae7", "#215ec7", "#004C98"]
)
norm = Normalize(vmin=o_plot.min(), vmax=o_plot.max())
bar_colors = cmap(norm(o_plot))

bars = ax.bar(
    d_plot,
    o_plot,
    width=0.85,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.6,
    label="Observed"
)

# Colourbar on the RIGHT using the same blue gradient
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, location="right")
cbar.set_label("Observed frequency level")

# Plot *smoothed* expected curve + shaded area
ax.plot(
    x_smooth,
    y_smooth,
    color="#b22222",
    linewidth=3,
    label="Expected (expansion)",
    zorder=3
)
ax.fill_between(
    x_smooth,
    y_smooth,
    0,
    color="#b22222",
    alpha=0.08,
    zorder=2
)

# Labels and title
ax.set_xlabel("Number of pairwise differences")
ax.set_ylabel("Frequency")
ax.set_title("Mismatch distribution of pairwise nucleotide differences")

# Annotate SSD and raggedness (updated values)
ax.text(
    0.98,
    0.93,
    "SSD = 0.029\nRaggedness = 0.0487",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    color="dimgray"
)

# Legend and styling
ax.legend(frameon=False, loc="upper left")

ax.set_xlim(d_plot.min() - 0.6, d_plot.max() + 0.6)
ax.set_ylim(0, max(o_plot.max(), y_smooth.max()) * 1.15)

# Modern look: remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
out_path = here / "Figure1_mismatch_premium.png"
fig.savefig(out_path, dpi=600)
plt.close(fig)

print(f"Saved figure to: {out_path}")
