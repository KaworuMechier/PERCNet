#!/usr/bin/env python3
"""Draw Fig. 1 (architecture) as a vector figure.

Grey blocks are inherited from the FESTRNN backbone; orange blocks are the
contributions of this work.  Output: results/figures/architecture.{svg,pdf,png}

    python scripts/draw_architecture.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

GREY_F, GREY_E = "#ECECEC", "#5A5A5A"
ORAN_F, ORAN_E = "#FCE3CC", "#C8641E"
INK = "#222222"

FIG_W, FIG_H = 9.0, 5.15


def box(ax, x, y, w, h, text, new=False, fs=7.4):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.07",
        fc=ORAN_F if new else GREY_F, ec=ORAN_E if new else GREY_E,
        lw=1.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, linespacing=1.35)


def arrow(ax, p1, p2, dashed=False, color=INK, lw=1.0):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle="--" if dashed else "-",
                                shrinkA=1.5, shrinkB=1.5,
                                mutation_scale=9), zorder=2)


def elbow(ax, pts, dashed=False, color=INK):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.0,
                linestyle="--" if dashed else "-", zorder=2)
    arrow(ax, pts[-2], pts[-1], dashed=dashed, color=color)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 18)
    ax.set_ylim(0.4, 9.6)
    ax.axis("off")

    # ── input ────────────────────────────────────────────────────────────
    box(ax, 1.5, 6.0, 2.4, 0.95,
        "history window\n168 × 20", fs=7.6)

    # three branches out of the history
    ax.plot([2.7, 3.5], [6.3, 8.3], color=INK, lw=1.0, zorder=2)
    arrow(ax, (3.5, 8.3), (3.95, 8.3))
    ax.text(3.55, 9.02, "3 load channels", fontsize=6.6, ha="left",
            color=INK)

    arrow(ax, (2.7, 6.0), (3.95, 5.35), dashed=True)
    ax.plot(3.05, 5.85, marker="x", ms=6, mew=1.6, color=ORAN_E, zorder=5)
    ax.text(3.30, 4.62, "9 exogenous", fontsize=6.6, ha="left", color=INK)

    arrow(ax, (2.7, 5.7), (3.95, 2.45))
    ax.text(3.0, 3.0, "calendar (4)", fontsize=6.6, ha="left", color=INK)

    # ── inherited spine (grey) ───────────────────────────────────────────
    y1 = 8.3
    box(ax, 4.95, y1, 1.7, 0.95, "RevIN\n(loads only)")
    box(ax, 7.05, y1, 1.75, 0.95, "spectral\nenhancement")
    box(ax, 9.25, y1, 1.75, 0.95, "moving-average\ndecomposition")
    box(ax, 11.65, y1, 2.05, 0.95, "linear specialists\nseasonal / trend")

    arrow(ax, (5.8, y1), (6.18, y1))
    arrow(ax, (7.93, y1), (8.38, y1))
    arrow(ax, (10.13, y1), (10.62, y1))

    # patch-GRU lane, fed from the enhanced residual
    y2 = 6.05
    box(ax, 9.25, y2, 1.85, 0.95, "patch-GRU\n(local stream)")
    elbow(ax, [(7.05, 7.82), (7.05, y2), (8.3, y2)])

    # stacked streams
    box(ax, 12.85, 7.2, 1.25, 0.8, "s + t + $\\ell$")
    arrow(ax, (12.68, y1), (12.85, 7.62))
    arrow(ax, (10.18, y2), (12.6, 6.95))

    # ── new blocks (orange) ──────────────────────────────────────────────
    box(ax, 5.55, 5.35, 2.45, 0.95,
        "trailing-24 h context\n5 stats × 9 = 45\n(no parameters)", new=True,
        fs=7.0)
    arrow(ax, (4.6, 5.35), (4.32, 5.35)) if False else None

    box(ax, 5.6, 2.45, 2.7, 0.95,
        "daily / weekly cycle queue\n24×3 and 168×3, learned", new=True,
        fs=7.0)

    box(ax, 14.3, 5.6, 2.35, 1.15,
        "residual fusion\n[s ; t ; $\\ell$ ; c] → 54-64-32-3\nzero-initialised",
        new=True, fs=6.9)
    arrow(ax, (13.5, 6.8), (13.5, 6.18))
    arrow(ax, (6.8, 5.35), (13.1, 5.6))

    box(ax, 14.3, 3.3, 2.35, 1.0,
        "horizon-conditioned head\n9 quantiles + scale", new=True, fs=7.0)
    arrow(ax, (14.3, 5.0), (14.3, 3.82))

    box(ax, 17.15, 4.45, 1.45, 1.5,
        "$\\mu$\n$q_{1..9}$\n$\\sigma$", fs=7.6)
    arrow(ax, (15.48, 5.6), (16.42, 5.0))
    arrow(ax, (15.48, 3.3), (16.42, 3.9))

    # cycle queue: subtracted from the history, added back to the forecast
    elbow(ax, [(4.25, 2.72), (4.25, 7.55), (4.1, 7.83)], dashed=True,
          color=ORAN_E)
    ax.text(4.06, 6.95, "subtract", fontsize=6.3,
            color=ORAN_E, ha="center", va="center")
    elbow(ax, [(6.95, 2.45), (16.4, 2.45), (16.4, 3.72)], dashed=True,
          color=ORAN_E)
    ax.text(11.2, 2.22, "add back", fontsize=6.3, color=ORAN_E, ha="center")

    # ── legend ───────────────────────────────────────────────────────────
    ax.legend(handles=[
        Patch(fc=GREY_F, ec=GREY_E, label="inherited from FESTRNN backbone"),
        Patch(fc=ORAN_F, ec=ORAN_E, label="contribution of this work"),
        Line2D([0], [0], color=ORAN_E, ls="--", lw=1.1,
               label="cycle profile removed / re-added"),
    ], loc="lower left", bbox_to_anchor=(0.005, 0.0), fontsize=6.8,
        frameon=False, handlelength=1.6, labelspacing=0.35)

    ax.text(17.15, 5.42, "outputs", fontsize=6.8, ha="center", color=INK)

    fig.tight_layout(pad=0.2)
    out = Path(__file__).resolve().parents[1] / "results" / "figures"
    os.makedirs(out, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out / f"architecture.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    print(f"wrote {out}/architecture.{{svg,pdf,png}}")


if __name__ == "__main__":
    main()
