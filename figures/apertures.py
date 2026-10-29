"""Figure 3: each LED's aperture on the sample spectrum, with the captures it makes."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Circle

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

NAME = "apertures"
RUN = "bar-full"
# Centre LED (brightfield), then the ring just outside the pupil (darkfield) and
# the outermost ring, both along +x.
RINGS_CYC_PER_PX = [0.0, 0.161, 0.25]
# Bar groups 7-8, in capture pixels of the "bar" crop.
TOP, LEFT, SIZE = 150, 340, 200
SPECTRUM_LIMIT_CYC_PER_UM = 0.95
SMOOTH_PX = 4


def compute() -> None:
    study = load_study("bar")
    obj = reconstruct(RUN, "bar")
    pixel_um = capture_pixel_um(study)
    kx = study.illumination_kx.numpy()
    ky = study.illumination_ky.numpy()
    radius = np.hypot(kx, ky)
    # Nearest LED to each ring on the +x axis.
    indices = [
        int(np.argmin(np.abs(radius - ring) + np.abs(ky) + (kx < -1e-6)))
        for ring in RINGS_CYC_PER_PX
    ]

    # Hann window: the bright cross from the frame's non-periodic edges hides the sample.
    window = np.hanning(obj.shape[0])
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(obj * np.outer(window, window))))
    df = 1 / (obj.shape[0] * pixel_um / OBJECT_TO_CAPTURE_RATIO)
    half = int(SPECTRUM_LIMIT_CYC_PER_UM / df) // SMOOTH_PX * SMOOTH_PX
    centre = obj.shape[0] // 2
    spectrum = spectrum[centre - half : centre + half, centre - half : centre + half]
    # Block-average: the background should read as faint context, not speckle.
    bins = 2 * half // SMOOTH_PX
    spectrum = spectrum.reshape(bins, SMOOTH_PX, bins, SMOOTH_PX).mean(axis=(1, 3))

    save_data(
        NAME,
        {
            "run": RUN,
            "capture_indices": indices,
            "pupil_cutoff_cyc_per_um": study.manifest.numerical_aperture
            / (study.capture_metadata[0].wavelength * 1e6),
            "spectrum_extent_cyc_per_um": half * df,
        },
        # U_j(k) = M(k) S(k - k_j): LED j reads the sample spectrum around -k_j.
        aperture_kx=-kx / pixel_um,
        aperture_ky=-ky / pixel_um,
        spectrum=np.log10(spectrum + spectrum.max() * 1e-5),
        captures=study.captures[indices, TOP : TOP + SIZE, LEFT : LEFT + SIZE].numpy(),
    )


def render() -> None:
    style.apply()
    data, arrays = load_data(NAME)
    cutoff = data["pupil_cutoff_cyc_per_um"]
    extent = data["spectrum_extent_cyc_per_um"]
    indices = data["capture_indices"]
    kx, ky = arrays["aperture_kx"], arrays["aperture_ky"]
    halo = [patheffects.withStroke(linewidth=3, foreground=style.PAPER)]

    fig = plt.figure(figsize=(style.COLUMN_WIDTH_IN, 4.2))
    grid = fig.add_gridspec(3, 2, width_ratios=[2.5, 1], wspace=0.2, hspace=0.32)
    ax = fig.add_subplot(grid[:, 0])
    spectrum = arrays["spectrum"]
    ax.imshow(
        spectrum,
        cmap=style.SPECTRUM_CMAP,
        extent=(-extent, extent, extent, -extent),
        vmin=np.percentile(spectrum, 30),
        vmax=np.percentile(spectrum, 99.9),
        interpolation="bilinear",
    )
    ax.scatter(kx, ky, s=7, color=style.INK_MUTED, edgecolors=style.PAPER, lw=0.5)
    synthetic = np.hypot(kx, ky).max() + cutoff
    ax.add_patch(
        Circle(
            (0, 0), synthetic, fill=False, lw=0.8, ls=(0, (4, 3)), color=style.INK_MUTED
        )
    )
    ax.text(
        0,
        -synthetic + 0.11,
        "synthetic aperture",
        ha="center",
        va="top",
        fontsize=8,
        color=style.INK_MUTED,
    )
    for number, (color, index) in enumerate(zip(style.SHADES, indices), start=1):
        x, y = kx[index], ky[index]
        ax.add_patch(Circle((x, y), cutoff, fill=False, lw=1.4, color=color))
        ax.text(
            x,
            y + cutoff + 0.025,
            str(number),
            ha="center",
            va="top",
            fontsize=8.5,
            color=color,
            path_effects=halo,
        )
    ax.text(
        kx[indices[0]] + cutoff * 0.72 + 0.03,
        ky[indices[0]] - cutoff * 0.72 - 0.03,
        "lens aperture",
        ha="left",
        va="bottom",
        fontsize=8,
        color=style.SIGNAL,
        path_effects=halo,
    )
    ax.set_xlim(-extent, extent)
    ax.set_ylim(extent, -extent)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([-0.5, 0, 0.5])
    ax.tick_params(labelfontfamily="Geist Mono")
    ax.set_xlabel("spatial frequency $k_x$ (cycles/µm)")
    ax.set_ylabel("spatial frequency $k_y$ (cycles/µm)")

    labels = ["centre LED", "ring 4", "outer ring"]
    for row, (color, label) in enumerate(zip(style.SHADES, labels)):
        image_ax = fig.add_subplot(grid[row, 1])
        image = arrays["captures"][row]
        image_ax.imshow(
            image,
            cmap=style.IMAGE_CMAP,
            vmin=np.percentile(image, 1),
            vmax=np.percentile(image, 99.5),
            interpolation="nearest",
        )
        style.image_axis(image_ax)
        for spine in image_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(1.5)
        image_ax.text(
            0.0,
            1.0,
            str(row + 1),
            transform=image_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=color,
            bbox={"boxstyle": "square,pad=0.3", "fc": style.PAPER, "ec": "none"},
        )
        image_ax.set_xlabel(label, fontsize=7.5, labelpad=3)
    fig.savefig(figure_path(NAME))
    plt.close(fig)
