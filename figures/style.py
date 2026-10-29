"""Shared figure style, matched to rspcunningham.com: change here, then re-render.

Figures sit in the site's 578 px figure column on its light page colour, so they
are sized to that width with text near the caption's size. Text is Geist in the site's
ink; images are greyscale and one hue, stepped in OKLCH lightness, is the accent.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap

DPI = 300
COLUMN_WIDTH_IN = 6.0  # displayed at 578 px, so 9 pt text shows at ~12 px

# Site tokens (--background, --foreground, --muted-foreground, --border, --ring).
PAPER = "#fdfdfd"
INK = "#0a0a0a"
INK_MUTED = "#636363"
RULE = "#dedede"
RING = "#a1a1a1"
# Physical parts (sample, lens, sensor) are drawn light so the signal outweighs them.
PART_FILL = "#f0efec"

# One hue (OKLCH h=245), lightness 0.18 -> 0.96, darkest first.
RAMP = ["#021322", "#15364f", "#315c7f", "#5985aa", "#8bafcf", "#c8daea", "#ecf3f8"]
# Three items told apart by lightness alone; always pair with a direct label.
SHADES = [RAMP[1], RAMP[3], RAMP[4]]
# Block diagrams: the sample, and the freedom the full model adds to the microscope.
LEARNED = RAMP[1]
ADDED = RAMP[3]
# The accent marks the signal: what gets through, and what to look at.
SIGNAL = RAMP[1]

# Data stays neutral so the hue reads as an accent on the marks.
# Intensity images: dark is no light. Spectra: dark is strong, on the page colour.
IMAGE_CMAP = LinearSegmentedColormap.from_list("ink", [INK, PAPER])
SPECTRUM_CMAP = LinearSegmentedColormap.from_list("spectrum", [PAPER, RING])


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "Geist",
            "mathtext.fontset": "cm",  # Computer Modern, like the site's KaTeX
            "font.size": 9,
            "text.color": INK,
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.titlepad": 5,
            "axes.labelsize": 8,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": PAPER,
            "xtick.color": RULE,
            "ytick.color": RULE,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "lines.linewidth": 1.0,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def image_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def scale_bar(ax: plt.Axes, um_per_px: float, length_um: float) -> None:
    width = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    height = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    length_px = length_um / um_per_px
    x0, y0 = width * 0.05, height * 0.93
    halo = [patheffects.withStroke(linewidth=2.5, foreground=PAPER)]
    ax.plot(
        [x0, x0 + length_px],
        [y0, y0],
        color=INK,
        lw=2.5,
        solid_capstyle="butt",
        path_effects=[patheffects.withStroke(linewidth=4.5, foreground=PAPER)],
    )
    ax.text(
        x0 + length_px / 2,
        y0 - height * 0.03,
        f"{length_um:g} µm",
        color=INK,
        fontsize=8,
        ha="center",
        va="bottom",
        path_effects=halo,
    )
