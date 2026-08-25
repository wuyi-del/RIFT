#!/usr/bin/env python3
"""Generate the three compact empirical figures used by the main paper."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent

BLUE = "#315C9B"
GREEN = "#3E7C59"
ORANGE = "#C87524"
GRAY = "#6A6A6A"
LIGHT = "#D7DCE3"
DARK = "#262626"

mpl.rcParams.update(
    {
        # Render chart text through the same NewTX family as the AAAI paper.
        # This avoids the previous Times/DejaVu math-font mixture.
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
        "font.family": "serif",
        "font.size": 7.0,
        "axes.titlesize": 7.8,
        "axes.labelsize": 6.9,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
    }
)


def clean_axis(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.4, pad=1.4)
    if grid_axis:
        ax.grid(axis=grid_axis, color=LIGHT, linewidth=0.55, alpha=0.8, zorder=0)


def save(fig, stem):
    pdf = ROOT / f"{stem}.pdf"
    png = ROOT / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=320)
    plt.close(fig)
    print(pdf)
    print(png)


def performance_robustness():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.08, 1.82),
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.33},
    )
    fig.subplots_adjust(left=0.078, right=0.992, bottom=0.25, top=0.84)

    # (a) Seed-level paired inference.
    ax = axes[0]
    labels = ["Seed 42", "Seed 43", "Seed 44", "Seeds 42/43 pooled"]
    mean = np.array([1.0185, 1.7593, 2.4074, 1.3889])
    lo = np.array([-0.8333, 0.0926, 0.3704, 0.2315])
    hi = np.array([2.8704, 3.4259, 4.4444, 2.5463])
    y = np.arange(len(labels))[::-1]
    colors = [GRAY, GREEN, GREEN, GREEN]
    markers = ["o", "o", "o", "D"]
    ax.axvline(0, color=DARK, linewidth=0.75, zorder=1)
    for yi, m, l, h, color, marker in zip(y, mean, lo, hi, colors, markers):
        ax.errorbar(
            m,
            yi,
            xerr=np.array([[m - l], [h - m]]),
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=4.2,
            elinewidth=1.05,
            capsize=2.2,
            markerfacecolor="white" if l <= 0 else color,
            markeredgewidth=0.9,
            zorder=3,
        )
        ax.text(h + 0.10, yi, f"{m:+.2f}", va="center", ha="left", fontsize=6.4)
    ax.set_yticks(y, labels)
    ax.set_xlim(-1.2, 5.0)
    ax.set_xlabel(r"RIFT $-$ Matched OPSD (percentage points)")
    ax.set_title(r"\textbf{(a)} Positive gain in all training seeds", loc="left", pad=3)
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # (b) Cross-scale matched gains from the final paired table.
    ax = axes[1]
    labels = ["1.7B s42", "4B s42--44", "8B s42"]
    gains = np.array([1.76, 1.69, 2.13])
    y = np.arange(len(labels))[::-1]
    colors = [BLUE, GREEN, BLUE]
    ax.barh(y, gains, height=0.48, color=colors, alpha=0.92, zorder=2)
    for yi, value in zip(y, gains):
        ax.text(value + 0.04, yi, f"{value:+.2f}", va="center", ha="left", fontsize=6.3)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 2.65)
    ax.set_xlabel("Matched gain (percentage points)")
    ax.set_title(r"\textbf{(b)} Gain transfers across model scale", loc="left", pad=3)
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    save(fig, "rift_performance_robustness")


def routing_diagnostics():
    # AAAI-27 requires all text inside figures to be at least 9 pt.  Keep this
    # override local so the supplementary figures retain their own layout.
    old_rc = mpl.rcParams.copy()
    mpl.rcParams.update(
        {
            "font.size": 9.3,
            "axes.titlesize": 9.3,
            "axes.labelsize": 9.3,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
        }
    )
    routing_blue = "#004C99"
    routing_green = "#006D2C"
    routing_gray = "#555555"
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.08, 2.18),
        gridspec_kw={"width_ratios": [1.08, 0.86, 0.92], "wspace": 0.55},
    )
    fig.subplots_adjust(left=0.087, right=0.993, bottom=0.27, top=0.80)

    # (a) Incremental recovery-signal value. Short row labels prevent panel collisions.
    ax = axes[0]
    rows = ["Future", "Future + current"]
    # Leave a dedicated top band for the legend so it cannot cover an interval.
    ybase = np.array([0.55, -0.25])
    auprc = np.array([0.00331, 0.01191])
    auprc_lo = np.array([-0.0012, 0.0049])
    auprc_hi = np.array([0.0078, 0.0188])
    auroc = np.array([0.00249, 0.00689])
    auroc_lo = np.array([-0.0036, 0.0015])
    auroc_hi = np.array([0.0084, 0.0123])
    ax.axvline(0, color=DARK, linewidth=0.75, zorder=1)
    for metric, vals, los, his, offset, color, marker in [
        ("AUPRC", auprc, auprc_lo, auprc_hi, 0.10, routing_green, "s"),
        ("AUROC", auroc, auroc_lo, auroc_hi, -0.10, routing_blue, "o"),
    ]:
        for yi, m, l, h in zip(ybase + offset, vals, los, his):
            ax.errorbar(
                m,
                yi,
                xerr=np.array([[m - l], [h - m]]),
                fmt=marker,
                color=color,
                ecolor=color,
                markersize=5.2,
                elinewidth=1.25,
                capsize=2.6,
                markerfacecolor=color if l > 0 else "white",
                markeredgewidth=1.0,
                zorder=3,
            )
        ax.plot([], [], marker=marker, color=color, linestyle="None", label=metric)
    ax.set_yticks(ybase, rows)
    ax.set_ylim(-0.60, 1.35)
    ax.set_xlim(-0.0065, 0.021)
    ax.set_xlabel("Increment over current-only")
    ax.set_title(r"\textbf{(a)} Future-signal value", loc="left", pad=4)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # (b) Calibration quality: lower route CV and tie excess are both better.
    ax = axes[1]
    short = ["FT", "GQ", "PT", "ER"]
    route_cv = np.array([0.05308, 0.04957, 0.04791, 0.04276]) * 100
    tie_excess = np.array([15.4588, 3.0763, 2.0363, 0.0])
    colors = [routing_gray, routing_gray, routing_gray, routing_green]
    markers = ["o", "o", "o", "D"]
    label_offsets = {
        "FT": (-13, -10),
        "GQ": (6, 5),
        "PT": (-16, -11),
        "ER": (6, 4),
    }
    for name, x, yv, color, marker in zip(short, route_cv, tie_excess, colors, markers):
        ax.scatter(
            x,
            yv,
            s=38,
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.65,
            zorder=3,
        )
        dx, dy = label_offsets[name]
        ax.annotate(name, (x, yv), xytext=(dx, dy), textcoords="offset points", fontsize=9.0)
    ax.set_xlim(4.05, 5.55)
    ax.set_ylim(-1.2, 17.3)
    ax.set_xlabel(r"Route CV (\%)")
    ax.set_ylabel("Tie excess")
    ax.set_title(r"\textbf{(b)} Exact-rank calibration", loc="left", pad=4)
    clean_axis(ax, grid_axis="both")

    # (c) Context-conditioned target arbitration.
    ax = axes[2]
    labels = [r"Reversed", r"Uniform $q^0$", r"Uniform $q^+$", "RIFT"]
    correct = np.array([679, 681, 686, 691])
    y = np.arange(len(labels))[::-1]
    colors = [routing_gray, routing_gray, routing_gray, routing_green]
    widths = [4.8, 4.8, 4.8, 6.2]
    markers = ["o", "o", "o", "D"]
    for yi, value, color, width, marker in zip(y, correct, colors, widths, markers):
        ax.hlines(yi, 676, value, color=color, linewidth=width, alpha=1.0, zorder=2)
        ax.scatter(
            value,
            yi,
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.55,
            s=30 if marker == "o" else 42,
            zorder=3,
        )
        ax.text(value + 0.65, yi, str(value), va="center", ha="left", fontsize=9.0)
    ax.set_yticks(y, labels)
    ax.set_xlim(676, 694.5)
    ax.set_xlabel("Correct / 1,080")
    ax.set_title(r"\textbf{(c)} Target arbitration", loc="left", pad=4)
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    save(fig, "rift_routing_diagnostics")
    mpl.rcParams.update(old_rc)


def transfer_and_budget():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.08, 1.82),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.34},
    )
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.25, top=0.82)

    # (a) ReGap pooled difference-in-differences. Scaling is endpoint-specific.
    ax = axes[0]
    labels = ["P-N NLL", "P-V NLL", "P-N JSD", "P-V JSD"]
    mean = np.array([-1.361, -2.072, -2.242, -3.374])
    lo = np.array([-2.632, -5.023, -3.221, -4.655])
    hi = np.array([-0.1439, 1.011, -1.323, -2.188])
    y = np.arange(4)[::-1]
    colors = [BLUE, BLUE, ORANGE, ORANGE]
    significant = [True, False, True, True]
    ax.axvline(0, color=DARK, linewidth=0.75, zorder=1)
    for yi, m, l, h, color, sig in zip(y, mean, lo, hi, colors, significant):
        ax.errorbar(
            m,
            yi,
            xerr=np.array([[m - l], [h - m]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.0,
            elinewidth=1.0,
            capsize=2.1,
            markerfacecolor=color if sig else "white",
            markeredgewidth=0.9,
            zorder=3,
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(-5.5, 1.5)
    ax.set_title(r"\textbf{(a)} ReGap held-out transfer", loc="left", pad=3)
    ax.set_xlabel(r"Endpoint-scaled pooled DiD (NLL $\times 10^{-4}$; JSD $\times 10^{-5}$)")
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.axhline(1.5, color=LIGHT, linewidth=0.65)

    # (b) Exploratory long-budget behavior.
    ax = axes[1]
    x = np.arange(3)
    budget_labels = ["4k", "8k", "16k"]
    series = [
        ("Base", [4.44, 22.22, 47.78], BLUE, "o", "-"),
        ("Continued OPSD", [2.22, 20.00, 45.56], GRAY, "^", "--"),
        ("AD-risk-only", [2.22, 22.22, 47.78], ORANGE, "s", ":"),
        ("RIFT", [6.67, 28.89, 53.33], GREEN, "D", "-"),
    ]
    for name, vals, color, marker, linestyle in series:
        ax.plot(
            x,
            vals,
            label=name,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.25 if name in {"Base", "RIFT"} else 0.9,
            markersize=3.6,
            zorder=3,
        )
    ax.set_xticks(x, budget_labels)
    ax.set_ylim(0, 58)
    ax.set_ylabel(r"Pass@1 (\%)")
    ax.set_xlabel("Generation budget")
    ax.set_title(r"\textbf{(b)} Exploratory long-budget behavior", loc="left", pad=3)
    ax.legend(frameon=False, ncol=2, loc="upper left", columnspacing=0.8, handlelength=1.5)
    clean_axis(ax, grid_axis="y")

    save(fig, "rift_transfer_budget")


if __name__ == "__main__":
    performance_robustness()
    routing_diagnostics()
    transfer_and_budget()
