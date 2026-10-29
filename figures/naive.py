"""Figure 5: the naive model's texture against the full model.

Both runs reconstruct the hero patch with identical settings except which error
terms are learned. A frozen term keeps its initial value (ideal disc pupil, unit
LED gains); a removed background is held at exactly zero.
"""

from contextlib import nullcontext
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import torch

from figures import style
from figures.common import LEARNING_RATES, figure_path, load_data, load_study, save_data
from imaged import SolverLearningRates, solve_study
from imaged.core.nuisance import Backgrounds

NAME = "naive"
# (name, label, learn lens, learn gains, learn background)
CONFIGS = [
    ("naive", "naive model", False, False, False),
    ("full", "full model", True, True, True),
]
# The hero figure's region (groups 7-9), in reconstruction pixels.
TOP, BOTTOM, LEFT, RIGHT = 208, 1464, 356, 1304


def solve(lens: bool, gains: bool, background: bool) -> np.ndarray:
    rates = SolverLearningRates(
        object=LEARNING_RATES.object,
        pupil=LEARNING_RATES.pupil if lens else 0.0,
        illumination_gains=LEARNING_RATES.illumination_gains if gains else 0.0,
        backgrounds=LEARNING_RATES.backgrounds if background else 0.0,
    )
    no_background = mock.patch.object(
        Backgrounds,
        "incoherent_intensity",
        lambda self: torch.zeros_like(self.raw_backgrounds),
    )
    with nullcontext() if background else no_background:
        result = solve_study(
            load_study("bar_single_patch"),
            patch_size=416,
            object_to_capture_ratio=4,
            pupil_phase_radial_order=3,
            pupil_amplitude_radial_order=0,
            epochs=200,
            patch_batch_size=1,
            illumination_chunk_size=145,
            learning_rates=rates,
        )
    return np.abs(result.object.numpy()[TOP:BOTTOM, LEFT:RIGHT])


def compute() -> None:
    save_data(
        NAME,
        {"labels": {name: label for name, label, *_ in CONFIGS}},
        **{name: solve(*flags) for name, _, *flags in CONFIGS},
    )


def show(ax: plt.Axes, image: np.ndarray, label: str, vmin: float, vmax: float) -> None:
    ax.imshow(
        image, cmap=style.IMAGE_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    style.image_axis(ax)
    ax.set_xlabel(label, fontsize=8, labelpad=4)


def render() -> None:
    style.apply()
    data, arrays = load_data(NAME)
    labels = data["labels"]
    # One shared display range, from the full model, so texture differences are real.
    vmin, vmax = np.percentile(arrays["full"], [1, 99.5])

    fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN_WIDTH_IN, 4.4))
    for ax, name in zip(axes, ("naive", "full")):
        show(ax, arrays[name], labels[name], vmin, vmax)
    fig.savefig(figure_path("naive-pair"))
    plt.close(fig)
