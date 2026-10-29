"""Figure 2: a lens is a low-pass filter on the sample's spectrum.

Top: diffraction orders leave the sample at angles set by their spatial frequency;
the single lens catches only those inside its numerical aperture and brings them
together on the sensor; steeper ones miss the glass.
Bottom: the same thing as arrays, on a synthetic bar target.
"""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
from matplotlib.transforms import Bbox

from figures import style
from figures.common import figure_path, load_data, save_data

NAME = "optics"
GRID = 256
BAR_PERIODS_PX = [32, 10, 6]  # coarse passes the lens, the middle one barely, fine not
CUTOFF_CYC_PER_PX = 0.11
# Schematic geometry: sin(angle) of order n is n * ORDER_SIN, lens accepts sin < NA.
ORDER_SIN = 0.15
NA = 0.25
OBJECT_DISTANCE = 2.2
IMAGE_DISTANCE = 4.4


def compute() -> None:
    y, x = np.mgrid[:GRID, :GRID]
    sample = np.ones((GRID, GRID))
    width = GRID // len(BAR_PERIODS_PX)
    for column, period in enumerate(BAR_PERIODS_PX):
        start = column * width + 8
        bars = ((x - start) // (period // 2)) % 2 == 0
        block = (x >= start) & (x < (column + 1) * width - 8)
        sample[block & bars & (y > 40) & (y < GRID - 40)] = 0.15
    spectrum = np.fft.fft2(sample)
    k = np.fft.fftfreq(GRID)
    mask = np.hypot(k[:, None], k[None, :]) <= CUTOFF_CYC_PER_PX
    image = np.abs(np.fft.ifft2(spectrum * mask)) ** 2
    save_data(
        NAME,
        {"cutoff_cyc_per_px": CUTOFF_CYC_PER_PX},
        sample=sample,
        spectrum=np.log10(np.abs(np.fft.fftshift(spectrum)) + 1),
        image=image,
    )


def render() -> None:
    style.apply()
    data, arrays = load_data(NAME)
    fig = plt.figure(figsize=(style.COLUMN_WIDTH_IN, 4.7))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.22, wspace=0.12)

    ax = fig.add_subplot(grid[0, :])
    # datalim: the axes fill the row and the x range stretches to match, so the
    # schematic can be centred on the figure below.
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    led_x, lens_x, sensor_x = -1.6, OBJECT_DISTANCE, OBJECT_DISTANCE + IMAGE_DISTANCE
    rim = OBJECT_DISTANCE * np.tan(np.arcsin(NA))

    # Illumination: a plane wave from the LED.
    ax.add_patch(Circle((led_x, 0), 0.08, color=style.INK))
    for h in np.linspace(-0.3, 0.3, 3):
        ax.annotate(
            "",
            xy=(-0.1, h),
            xytext=(led_x + 0.25, h),
            arrowprops={"arrowstyle": "-|>", "color": style.RING, "lw": 0.7},
        )

    # Diffraction orders leave a point on the sample at angles set by their spatial
    # frequency. The lens only catches angles inside its numerical aperture: those
    # converge to the image point on the sensor, steeper ones miss the glass.
    lost_height = 0.0
    for n in range(-3, 4):
        height = OBJECT_DISTANCE * np.tan(np.arcsin(n * ORDER_SIN))
        if abs(height) <= rim:
            ax.plot(
                [0, lens_x, sensor_x],
                [0, height, 0],
                color=style.SIGNAL,
                lw=1.1,
                solid_capstyle="round",
                zorder=4,
            )
        else:
            ax.plot([0, lens_x], [0, height], color=style.RING, lw=0.7, ls=":")
            ax.plot(lens_x, height, "x", color=style.RING, ms=4, mew=1, zorder=5)
            lost_height = max(lost_height, height)

    def part(x: float, half_height: float) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.05, -half_height),
                0.1,
                2 * half_height,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=style.PART_FILL,
                edgecolor=style.RING,
                lw=0.7,
                zorder=3,
            )
        )

    part(0, 0.6)
    part(sensor_x, 0.6)
    ax.add_patch(
        Ellipse(
            (lens_x, 0),
            0.26,
            2 * rim,
            facecolor=style.PART_FILL,
            edgecolor=style.RING,
            lw=0.7,
            zorder=2,
        )
    )
    # The lens mount: everything outside the glass is blocked.
    for sign in (-1, 1):
        ax.plot(
            [lens_x, lens_x],
            [sign * rim, sign * (lost_height + 0.25)],
            color=style.RING,
            lw=2.2,
            solid_capstyle="round",
        )
    ax.text(
        lens_x + 0.3,
        rim + 0.08,
        "inside the NA",
        color=style.SIGNAL,
        fontsize=8,
        va="bottom",
    )
    ax.text(
        lens_x - 0.25,
        lost_height + 0.05,
        "outside the NA: lost",
        color=style.INK_MUTED,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    # One baseline for every component label.
    for x, label in (
        (led_x, "LED"),
        (0, "sample"),
        (lens_x, "lens"),
        (sensor_x, "sensor"),
    ):
        ax.text(x, -lost_height - 0.42, label, ha="center", va="top")

    cutoff = data["cutoff_cyc_per_px"]
    panels = [
        ("sample $s$", "sample"),
        ("spectrum $S$, lens $M$", "spectrum"),
        ("image $\\hat{y} = |u|^2$", "image"),
    ]
    for column, (title, key) in enumerate(panels):
        image_ax = fig.add_subplot(grid[1, column])
        image = arrays[key]
        if key == "spectrum":
            half = GRID // 2
            zoom = int(0.25 * GRID)
            image_ax.imshow(
                image[half - zoom : half + zoom, half - zoom : half + zoom],
                cmap=style.SPECTRUM_CMAP,
                extent=(-0.25, 0.25, 0.25, -0.25),
                vmin=np.percentile(image, 30),
            )
            image_ax.add_patch(
                Circle((0, 0), cutoff, fill=False, lw=1.2, color=style.SIGNAL)
            )
        else:
            image_ax.imshow(image, cmap=style.IMAGE_CMAP, interpolation="nearest")
        style.image_axis(image_ax)
        image_ax.set_xlabel(title, fontsize=8, labelpad=4)

    # Centre the schematic's drawn extent, labels included, over the panel row.
    fig.canvas.draw()
    content = cast(Bbox, ax.get_tightbbox()).transformed(ax.transData.inverted())
    shift = (content.x0 + content.x1 - sum(ax.get_xlim())) / 2
    ax.set_xlim(ax.get_xlim()[0] + shift, ax.get_xlim()[1] + shift)
    fig.savefig(figure_path(NAME))
    plt.close(fig)
