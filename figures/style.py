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
