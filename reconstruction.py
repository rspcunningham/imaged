import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ptych import ImageCrop, PtychStudy, SolverLearningRates, solve_study
from ptych.core.metric_plots import save_metrics_summary
from ptych.data.utils import get_default_device

# Select dataset
dataset = "datasets/20260728-122052-Bar Pattern"

CROP_SIZE = 616  # 616px window in the 1232x1232 frame
CROP_TOP = 0
CROP_LEFT = 128  # 512px shift in the 4x output
DARK_SUBTRACTION = "average_all"  # "average_all" or "nearest_only"
# Debug: persist every preprocessed capture, not just the first.
SAVE_ALL_CAPTURES = True

# Reconstruction model settings
OBJECT_TO_CAPTURE_RATIO = 4  # prefer 2 or 4
PUPIL_PHASE_RADIAL_ORDER = 3
PUPIL_AMPLITUDE_RADIAL_ORDER = 0

# Memory/scaling settings
PATCH_SIZE = 416  # prefer power of 2 or 384, 416, 448, 480, 512
PATCH_BATCH_SIZE = 1
# Each chunk re-samples the object's oversampled spectrum, so one chunk covering
# every illumination is cheapest unless memory-bound.
ILLUMINATION_CHUNK_SIZE = 145

# Runtime settings
EPOCHS = 200
LEARNING_RATES = SolverLearningRates(
    object=1e-1,
    pupil=1e-2,
    illumination_gains=1e-1,
    backgrounds=1e-1,
    scatter=3e-2,
)

# Output directory
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = Path(f"results/{Path(dataset).name}-{timestamp}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

study = PtychStudy.load(
    dataset,
    crop=ImageCrop(top=CROP_TOP, left=CROP_LEFT, width=CROP_SIZE, height=CROP_SIZE),
    dark_subtraction=DARK_SUBTRACTION,
)

# Persist preprocessed captures under their original dataset filenames.
captures_dir = OUTPUT_DIR / "captures"
captures_dir.mkdir(exist_ok=True)
saved_capture_count = len(study.captures) if SAVE_ALL_CAPTURES else 1
for capture, metadata in zip(
    study.captures[:saved_capture_count],
    study.capture_metadata[:saved_capture_count],
    strict=True,
):
    np.save(captures_dir / metadata.filename, capture.numpy())

# Get GPU
TORCH_DEVICE = get_default_device()
print(f"Using torch device: {TORCH_DEVICE}")

# Run reconstruction
result = solve_study(
    study,
    patch_size=PATCH_SIZE,
    object_to_capture_ratio=OBJECT_TO_CAPTURE_RATIO,
    pupil_phase_radial_order=PUPIL_PHASE_RADIAL_ORDER,
    pupil_amplitude_radial_order=PUPIL_AMPLITUDE_RADIAL_ORDER,
    epochs=EPOCHS,
    device=TORCH_DEVICE,
    patch_batch_size=PATCH_BATCH_SIZE,
    illumination_chunk_size=ILLUMINATION_CHUNK_SIZE,
    learning_rates=LEARNING_RATES,
)

# Save reconstruction artifacts.
metrics_plot_path = OUTPUT_DIR / "reconstruction_metrics.png"
object_path = OUTPUT_DIR / "object.npy"
metrics_json_path = OUTPUT_DIR / "metrics.json"

save_metrics_summary(
    result.metrics,
    path=metrics_plot_path,
)

np.save(object_path, result.object.cpu().numpy())
with metrics_json_path.open("w") as file:
    json.dump(result.metrics, file)

print("Reconstruction complete!")
print(f"Reconstructed object tensor: {result.object.shape}")
print(f"Saved object: {object_path}")
print(f"Saved {saved_capture_count} preprocessed capture(s): {captures_dir}")
print(f"Saved metrics JSON: {metrics_json_path}")
print(f"Saved metrics plot: {metrics_plot_path}")
