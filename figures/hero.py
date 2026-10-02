"""Figure 1: one raw capture vs the reconstructed amplitude and phase (malaria)."""

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
RUN = "malaria-full"
CAPTURE_INDEX = 0  # centre LED, brightfield


def compute() -> None:
    study = load_study("malaria")
    obj = reconstruct(RUN, "malaria")
    phase = np.angle(obj * np.exp(-1j * np.angle(obj.mean())))
    save_data(
        NAME,
        {
            "run": RUN,
            "dataset": "malaria",
            "capture_index": CAPTURE_INDEX,
            "capture_pixel_um": capture_pixel_um(study),
            "object_pixel_um": capture_pixel_um(study) / OBJECT_TO_CAPTURE_RATIO,
        },
        capture=study.captures[CAPTURE_INDEX].numpy(),
        amplitude=np.abs(obj),
        phase=phase,
    )


def render() -> None:
    style.apply()
    data, arrays = load_data(NAME)
    fig, axes = plt.subplots(1, 3, figsize=(style.COLUMN_WIDTH_IN, 2.6))
    panels = [
        ("One capture", arrays["capture"], style.IMAGE_CMAP),
        ("Reconstructed amplitude", arrays["amplitude"], style.IMAGE_CMAP),
        ("Reconstructed phase", arrays["phase"], style.PHASE_CMAP),
    ]
    for ax, (title, image, cmap) in zip(axes, panels, strict=True):
        ax.imshow(image, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        style.image_axis(ax)
    fig.savefig(figure_path(NAME))
    plt.close(fig)
