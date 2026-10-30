"""Figure 7: MTF and NPS of one capture vs the reconstruction, from image-eval.

compute runs the image-eval CLI on the centre-LED capture (the base image, which
mtf_template.json is drawn on) and on the hero reconstruction, and keeps its raw JSON
reports. render plots straight from those reports. The NPS is from the flat black
region, each image out to its own sampling Nyquist, relative to the capture's mean NPS.
"""

import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from figures import style
from figures.common import DATASETS, OUT, ROOT, figure_path, reconstruct
from imaged import PtychStudy

NAME = "mtf"
RUN = "bar-single-patch"
TEMPLATE = ROOT / "figures" / "mtf_template.json"
# image-eval lives in a sibling checkout; run it from there so local changes apply.
IMAGE_EVAL = ["uv", "run", "--project", str(ROOT.parent / "image-eval"), "image-eval"]


def compute() -> None:
    directory = OUT / NAME
    directory.mkdir(parents=True, exist_ok=True)
    study = PtychStudy.load(ROOT / DATASETS["bar"][0])
    base = directory / "capture.npy"
    reconstruction = directory / "reconstruction.npy"
    np.save(base, study.captures[0].numpy())
    np.save(reconstruction, reconstruct(RUN, "bar_single_patch"))
    for name, subject in (("capture", base), ("reconstruction", reconstruction)):
        report = subprocess.run(
            [
                *IMAGE_EVAL,
                "eval",
                "--base-url",
                str(base),
                "--template",
                str(TEMPLATE),
                "--subject-url",
                str(subject),
                "--detail",
                "fits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (directory / f"report_{name}.json").write_text(report)


def load_report(name: str) -> dict:
    return json.loads((OUT / NAME / f"report_{name}.json").read_text())


def mtf_points(report: dict) -> np.ndarray:
    """Every measured MTF point, (cycles/mm, MTF): fundamentals and kept harmonics, X and Y.

    A resolved 3rd harmonic of a bar target is one more sample of the same MTF, at 3f.
    """
    return np.array(
        [
            (point["cycles_per_mm"], point["mtf"])
            for profile in report["mtf"]["fit_profiles"]
            for point in profile["mtf_points"]
        ]
    )


def trend(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median MTF in bins one USAF element (1/6 octave) wide."""
    bins = np.round(np.log2(points[:, 0]) * 6)
    centres = np.unique(bins)
    return 2 ** (centres / 6), np.array(
        [np.median(points[bins == b, 1]) for b in centres]
    )


def nps_curve(report: dict) -> tuple[np.ndarray, np.ndarray]:
    """(cycles/mm, NPS) of the flat black region, in intensity^2 mm^2."""
    rows = report["nps"]["rows"]
    return np.array([r["frequency"] for r in rows]), np.array(
        [np.nan if r["black_nps"] is None else r["black_nps"] for r in rows]
    )


def render() -> None:
    style.apply()
    fig, (mtf_ax, nps_ax) = plt.subplots(
        1, 2, figsize=(style.COLUMN_WIDTH_IN, 2.9), gridspec_kw={"wspace": 0.35}
    )
    # (name, label, colour, label position in data coordinates, alignment)
    series = (
        ("capture", "one capture", style.INK_MUTED, (262, 0.25), "right"),
        ("reconstruction", "reconstruction", style.SIGNAL, (420, 0.47), "left"),
    )
    reference = np.nanmean(nps_curve(load_report("capture"))[1])
    for name, label, color, (x, y), ha in series:
        report = load_report(name)
        points = mtf_points(report)
        mtf_ax.scatter(
            points[:, 0], points[:, 1], s=7, color=color, lw=0, alpha=0.5, zorder=3
        )
        mtf_ax.plot(*trend(points), color=color, lw=1.4, solid_capstyle="round")
        mtf_ax.text(x, y, label, color=color, fontsize=8, ha=ha, va="center")

        f, nps = nps_curve(report)
        nps_ax.plot(f, nps / reference, color=color, lw=1.4, solid_capstyle="round")

    mtf_ax.set_xlabel("spatial frequency (cycles/mm)")
    mtf_ax.set_ylabel("MTF")
    mtf_ax.set_ylim(0, 0.8)
    nps_ax.set_xlabel("spatial frequency (cycles/mm)")
    nps_ax.set_ylabel("noise power (relative to one capture)")
    nps_ax.set_xlim(0, 2000)
    nps_ax.set_ylim(0, None)
    # USAF bar width in µm along the top: one cycle is a bar and a gap.
    for ax, widths in ((mtf_ax, [2, 1.5, 1, 0.75]), (nps_ax, [2, 1, 0.5, 0.25])):
        ax.tick_params(labelfontfamily="Geist Mono")
        top = ax.secondary_xaxis(
            "top",
            functions=(
                lambda f: 500 / np.maximum(f, 1e-9),
                lambda w: 500 / np.maximum(w, 1e-9),
            ),
        )
        top.set_xticks(widths, labels=[f"{w:g}" for w in widths])
        top.set_xlabel("bar width (µm)")
        top.tick_params(labelfontfamily="Geist Mono")
    fig.savefig(figure_path(NAME))
    plt.close(fig)
