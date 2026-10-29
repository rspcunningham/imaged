## Imaging As Gradient Descent (ImAGeD)

A Python/PyTorch library for Fourier ptychographic microscopy: reconstructing a high-resolution sample from low-resolution captures taken under different illumination angles.

It jointly fits the sample, lens aberrations and aperture radius, illumination brightness, and background light through a differentiable microscope model.

![Block diagram of the forward model: the learned sample spectrum is shifted per LED, filtered by a lens built from learned Zernike coefficients and radius, turned into intensity, scaled by a learned per-LED gain and offset by a learned per-capture background, then compared with the measured capture; gradient descent updates every learned block.](figures/output/model-full.png)

Read the accompanying essay: [High resolution imaging is a learning problem](https://www.rspcunningham.com/imaging-is-learning).

## Run a reconstruction

Requires Python 3.13 and `uv`. From the repository root:

```sh
uv sync
```

Provide a dataset with this layout:

```text
datasets/my-study/
├── info.json
└── captures/
    └── *.npy
```

Each capture is a raw 2D Bayer image stored as a NumPy array. The manifest describes the optics, exposure times, illumination positions, and optional dark captures; see the [dataset schema](INFO_JSON_SCHEMA.md) for the full format. Datasets are stored locally under `datasets/` and kept outside version control. `datasets/` mirrors the Hugging Face dataset [rspcunningham/imaged-data](https://huggingface.co/datasets/rspcunningham/imaged-data). Download it, or one study, with:

```sh
hf download rspcunningham/imaged-data --repo-type dataset --local-dir datasets
hf download rspcunningham/imaged-data --repo-type dataset --local-dir datasets --include "20260728-122052-usaf-g9/*"
```

Push local changes back to the Hub (only changed files are uploaded; files removed locally are removed from the Hub):

```sh
hf upload rspcunningham/imaged-data datasets . --repo-type dataset --delete "*"
```

Validate a dataset before reconstruction:

```sh
uv run imaged-dataset validate datasets/my-study
```

Each study contains one wavelength. The colour acquisition is split into independent studies at `datasets/20250821-160234-usaf-g9-colour-red`, `-green`, and `-blue`.

In [reconstruction.py](reconstruction.py), set `dataset` to your dataset directory and choose the crop, patch size, number of epochs, and learning rates. Then run:

```sh
uv run python reconstruction.py
```

The loader demosaics the captures, subtracts dark frames when available, corrects for exposure, and normalizes intensities. The script selects CUDA, then Apple MPS, then CPU according to availability.

Each run writes to `results/<dataset-name>-<timestamp>/`:

- `object.npy`: the reconstructed complex object. Its intensity is `np.abs(object) ** 2`; its phase is `np.angle(object)`.
- `captures/`: the preprocessed captures used for reconstruction.
- `metrics.json` and `reconstruction_metrics.png`: loss histories and optimization summaries.

## Library usage

`PtychStudy.load` accepts a local dataset directory. Relative paths are resolved from the current working directory; missing directories raise a dataset validation error.

```python
from imaged import PtychStudy, solve_study

study = PtychStudy.load("datasets/my-study")
result = solve_study(study, patch_size=416, epochs=200)

intensity = result.object.abs().square().numpy()
phase = result.object.angle().numpy()
```

Choose a patch size that fits the capture dimensions and pupil bandwidth. `solve_study` returns the complex object, a pupil for each patch, and per-batch metrics. See [reconstruction.py](reconstruction.py) for crop settings, learning rates, and output handling.
