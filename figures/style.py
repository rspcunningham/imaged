"""Shared figure style. Placeholder defaults: change here, then re-render."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 200
COLUMN_WIDTH_IN = 7.2
IMAGE_CMAP = "gray"
PHASE_CMAP = "twilight"


def apply() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def image_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def scale_bar(ax: plt.Axes, um_per_px: float, length_um: float) -> None:
    width = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    height = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    length_px = length_um / um_per_px
    x0, y0 = width * 0.05, height * 0.93
    ax.plot([x0, x0 + length_px], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    ax.text(
        x0 + length_px / 2,
        y0 - height * 0.03,
        f"{length_um:g} µm",
        color="white",
        ha="center",
        va="bottom",
    )
