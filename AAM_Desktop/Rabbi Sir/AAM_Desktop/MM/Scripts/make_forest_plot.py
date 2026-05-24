#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dersimonian_laird(y, v):
    # Random effects meta-analysis (DL). y: effect, v: within-study var
    w = 1.0 / v
    ybar = np.sum(w*y) / np.sum(w)
    Q = np.sum(w*(y - ybar)**2)
    df = len(y) - 1
    C = np.sum(w) - np.sum(w**2)/np.sum(w)
    tau2 = max(0.0, (Q - df)/C) if C > 0 else 0.0
    w_star = 1.0 / (v + tau2)
    mu = np.sum(w_star*y) / np.sum(w_star)
    se = np.sqrt(1.0 / np.sum(w_star))
    return mu, se, tau2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="forest_template.csv (filled)")
    ap.add_argument("--outdir", default="FIGS_FOREST")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pool", action="store_true", help="Add random-effects pooled estimate (DL)")
    ap.add_argument("--log", action="store_true", help="Interpret estimates on log-scale and exponentiate on axis")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    # required columns
    req = {"Outcome","Study","Estimate","CI_low","CI_high"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    for outcome, sub in df.groupby("Outcome", dropna=False):
        sub = sub.dropna(subset=["Estimate","CI_low","CI_high"], how="any").copy()
        if sub.empty:
            continue

        # handle log option
        est = sub["Estimate"].astype(float).to_numpy()
        lo  = sub["CI_low"].astype(float).to_numpy()
        hi  = sub["CI_high"].astype(float).to_numpy()

        if args.log:
            # assume inputs already log-scale; axis shows exp()
            x_est, x_lo, x_hi = np.exp(est), np.exp(lo), np.exp(hi)
        else:
            x_est, x_lo, x_hi = est, lo, hi

        studies = sub["Study"].astype(str).tolist()
        n = len(studies)

        fig_h = max(4, 0.35*n + 2.5)
        fig, ax = plt.subplots(figsize=(9, fig_h))

        y = np.arange(n)[::-1]  # top to bottom

        ax.hlines(y, x_lo, x_hi, linewidth=1.5)
        ax.plot(x_est, y, "o", markersize=6)

        # pooled
        if args.pool:
            # approximate variance from CI: (hi-lo)/(2*1.96)
            if args.log:
                # work on log-scale for pooling; convert back only on plotting
                y_eff = est
                se = (hi - lo) / (2*1.96)
                v = se**2
                mu, se_mu, tau2 = dersimonian_laird(y_eff, v)
                mu_p, lo_p, hi_p = np.exp(mu), np.exp(mu - 1.96*se_mu), np.exp(mu + 1.96*se_mu)
            else:
                y_eff = est
                se = (hi - lo) / (2*1.96)
                v = se**2
                mu, se_mu, tau2 = dersimonian_laird(y_eff, v)
                mu_p, lo_p, hi_p = mu, mu - 1.96*se_mu, mu + 1.96*se_mu

            ax.axhline(-1, color="none")
            ax.hlines(-1, lo_p, hi_p, linewidth=4)
            ax.plot(mu_p, -1, "D", markersize=7)
            studies.append("Pooled (random-effects)")
            y = np.append(y, -1)
            ax.set_yticks(y)
            ax.set_yticklabels(studies, fontsize=9)

        else:
            ax.set_yticks(y)
            ax.set_yticklabels(studies, fontsize=9)

        ax.set_xlabel("Effect estimate")
        ax.set_title(f"Forest plot — {outcome}", fontsize=14, pad=12)
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()

        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(outcome))[:80]
        fig.savefig(outdir / f"Forest_{safe}.png", dpi=args.dpi)
        fig.savefig(outdir / f"Forest_{safe}.pdf")
        plt.close(fig)

    print(f"Saved forest plots in: {outdir}")

if __name__ == "__main__":
    import re
    main()
