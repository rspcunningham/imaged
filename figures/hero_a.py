"""Figure 1, option A: a sample of the 145 captures, then the reconstruction (malaria)."""

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

NAME = "hero_a"
RUN = "malaria-full"
# Two LEDs per ring, centre outwards (rings hold 1, 8, 12, 16, 24, 36, 48 LEDs).
CAPTURE_INDICES = [0, 1, 5, 9, 15, 21, 29, 37, 49, 61, 79, 97]
RING_STARTS = np.cumsum([0, 1, 8, 12, 16, 24, 36])
# Central region of the malaria crop, in capture pixels.
TOP, BOTTOM, LEFT, RIGHT = 208, 408, 208, 408


def compute() -> None:
    study = load_study("malaria")
    obj = reconstruct(RUN, "malaria")
    r = OBJECT_TO_CAPTURE_RATIO
    save_data(
        NAME,
        {
            "run": RUN,
            "capture_indices": CAPTURE_INDICES,
            "rings": [
                int(np.searchsorted(RING_STARTS, i, side="right") - 1)
                for i in CAPTURE_INDICES
            ],
            "num_captures": int(study.captures.shape[0]),
            "region_capture_px": [TOP, BOTTOM, LEFT, RIGHT],
            "object_pixel_um": capture_pixel_um(study) / r,
        },
        captures=study.captures[CAPTURE_INDICES, TOP:BOTTOM, LEFT:RIGHT].numpy(),
        reconstruction=np.abs(obj[r * TOP : r * BOTTOM, r * LEFT : r * RIGHT]),
    )


def render() -> None:
    style.apply()
    data, arrays = load_data(NAME)
    fig = plt.figure(figsize=(style.COLUMN_WIDTH_IN, 3.4))
    grid = fig.add_gridspec(3, 8, wspace=0.05, hspace=0.05)
    for k, (capture, ring) in enumerate(
        zip(arrays["captures"], data["rings"], strict=True)
    ):
        ax = fig.add_subplot(grid[k // 4, k % 4])
        # Each thumbnail on its own scale: darkfield captures are orders of magnitude dimmer.
        ax.imshow(
            capture,
            cmap=style.IMAGE_CMAP,
            vmin=np.percentile(capture, 1),
            vmax=np.percentile(capture, 99.5),
        )
        ax.set_title(f"ring {ring}", fontsize=6, pad=2)
        style.image_axis(ax)
    fig.text(0.5, 0.5, "→", fontsize=28, ha="center", va="center")
    ax = fig.add_subplot(grid[:, 5:])
    image = arrays["reconstruction"]
    ax.imshow(
        image,
        cmap=style.IMAGE_CMAP,
        vmin=np.percentile(image, 1),
        vmax=np.percentile(image, 99),
    )
    ax.set_title("Reconstruction")
    style.image_axis(ax)
    style.scale_bar(ax, data["object_pixel_um"], 10)
    fig.text(
        0.27,
        0.0,
        f"{len(data['capture_indices'])} of {data['num_captures']} captures",
        ha="center",
    )
    fig.savefig(figure_path(NAME))
    plt.close(fig)
