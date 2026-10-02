"""Shared runs and data I/O for the essay figures.

Three layers, so restyling never recomputes anything:
  runs     results/figures/runs/<run>/      cached reconstructions (slow)
  data     results/figures/<figure>/        what one figure shows: data.json + arrays.npz
  render   results/figures/<figure>/<figure>.png, from data + figures/style.py
"""

import json
from pathlib import Path

import numpy as np

from imaged import ImageCrop, PtychStudy, SolverLearningRates, solve_study

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"

DATASETS = {
    "malaria": (
        "datasets/malaria-test",
        ImageCrop(top=1000, left=1400, width=616, height=616),
    ),
    "bar": (
        "datasets/20260728-122052-Bar Pattern",
        ImageCrop(top=0, left=128, width=616, height=616),
    ),
}
OBJECT_TO_CAPTURE_RATIO = 4
# reconstruction.py settings.
LEARNING_RATES = SolverLearningRates(
    object=1e-1, pupil=1e-2, illumination_gains=1e-1, backgrounds=1e-1
)


def load_study(dataset: str) -> PtychStudy:
    path, crop = DATASETS[dataset]
    return PtychStudy.load(ROOT / path, crop=crop)


def capture_pixel_um(study: PtychStudy) -> float:
    return study.manifest.sensor_pixel_size / study.manifest.magnification * 1e6


def reconstruct(run: str, dataset: str) -> np.ndarray:
    """Complex object for a cached run, solving it on first use."""
    path = OUT / "runs" / run / "object.npy"
    if not path.exists():
        result = solve_study(
            load_study(dataset),
            patch_size=416,
            object_to_capture_ratio=OBJECT_TO_CAPTURE_RATIO,
            pupil_phase_radial_order=3,
            pupil_amplitude_radial_order=0,
            epochs=200,
            patch_batch_size=1,
            illumination_chunk_size=145,
            learning_rates=LEARNING_RATES,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, result.object.numpy())
        with (path.parent / "metrics.json").open("w") as file:
            json.dump(result.metrics, file)
    return np.load(path)


def save_data(figure: str, data: dict, **arrays: np.ndarray) -> None:
    directory = OUT / figure
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "data.json").open("w") as file:
        json.dump(data, file, indent=1)
    np.savez_compressed(directory / "arrays.npz", allow_pickle=False, **arrays)


def load_data(figure: str) -> tuple[dict, dict[str, np.ndarray]]:
    directory = OUT / figure
    with (directory / "data.json").open() as file:
        data = json.load(file)
    with np.load(directory / "arrays.npz") as arrays:
        return data, dict(arrays)


def figure_path(figure: str) -> Path:
    return OUT / figure / f"{figure}.png"
