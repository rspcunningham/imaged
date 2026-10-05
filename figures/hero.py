"""Figure 1: one raw capture vs the reconstruction, zoomed on bar groups 7-8."""

import matplotlib.pyplot as plt
import numpy as np

from figures import style
from figures.common import (
    OBJECT_TO_CAPTURE_RATIO,
    capture_pixel_um,
    figure_path,
    load_data,
    load_study,
    reconstruct,
    save_data,
)

NAME = "hero"
RUN = "bar-single-patch"
CAPTURE_INDEX = 0  # centre LED, brightfield
# Groups 7 and 8 of the target, in capture pixels of the single-patch crop.
TOP, BOTTOM, LEFT, RIGHT = 52, 366, 89, 326


def compute() -> None:
    study = load_study("bar_single_patch")
    obj = reconstruct(RUN, "bar_single_patch")
    r = OBJECT_TO_CAPTURE_RATIO
    save_data(
        NAME,
        {
            "run": RUN,
            "capture_index": CAPTURE_INDEX,
            "region_capture_px": [TOP, BOTTOM, LEFT, RIGHT],
            "object_pixel_um": capture_pixel_um(study) / r,
        },
        # Raw capture repeated onto the object grid so both panels share a pixel scale.
        capture=np.kron(
            study.captures[CAPTURE_INDEX, TOP:BOTTOM, LEFT:RIGHT].numpy(),
            np.ones((r, r)),
        ),
        reconstruction=np.abs(obj[r * TOP : r * BOTTOM, r * LEFT : r * RIGHT]),
    )


def render() -> None:
    style.apply()
    data, arrays = load_data(NAME)
    fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN_WIDTH_IN, 4.9))
    for ax, (title, key) in zip(
        axes,
        [
            ("Before: one capture", "capture"),
            ("After: reconstruction", "reconstruction"),
        ],
    ):
        image = arrays[key]
        ax.imshow(
            image,
            cmap=style.IMAGE_CMAP,
            vmin=np.percentile(image, 1),
            vmax=np.percentile(image, 99),
            interpolation="nearest",
        )
        ax.set_title(title)
        style.image_axis(ax)
        style.scale_bar(ax, data["object_pixel_um"], 10)
    fig.savefig(figure_path(NAME))
    plt.close(fig)
