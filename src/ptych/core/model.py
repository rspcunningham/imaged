import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from ptych.core.forward import FPMForwardModel, SpectralWindow
from ptych.core.nuisance import Backgrounds, Scatter
from ptych.core.object import Object
from ptych.core.pupil import DEFAULT_EDGE_WIDTH_PX, Pupil


class IlluminationGains(nn.Module):
    def __init__(self, num_illuminations: int) -> None:
        super().__init__()
        self.log_gains = nn.Parameter(torch.zeros(num_illuminations))

    def forward(self) -> Float[Tensor, "illumination"]:
        return torch.exp(self.log_gains)


def _pupil_cutoff_limits(
    pupil_cutoff_cyc_per_px: Tensor | float,
) -> tuple[float, float]:
    pupil_cutoff = torch.as_tensor(pupil_cutoff_cyc_per_px).detach().flatten().cpu()
    if torch.any(pupil_cutoff <= 0):
        raise ValueError(
            f"pupil_cutoff_cyc_per_px must be positive; got {pupil_cutoff.tolist()}"
        )

    return 0.8 * float(pupil_cutoff.min()), 1.2 * float(pupil_cutoff.max())


class PtychographyModel(nn.Module):
    def __init__(
        self,
        measured_intensity_batch: Float[
            Tensor, "patch_batch illumination height width"
        ],
        illumination_kx: Float[Tensor, "illumination"],
        illumination_ky: Float[Tensor, "illumination"],
        *,
        object_to_capture_ratio: int,
        pupil_cutoff_cyc_per_px_init: Tensor | float,
        pupil_phase_radial_order: int = 2,
        pupil_amplitude_radial_order: int = 0,
    ) -> None:
        super().__init__()
        patch_batch_size, num_illuminations, capture_height, _ = (
            measured_intensity_batch.shape
        )
        object_grid_size = capture_height * object_to_capture_ratio
        min_pupil_cutoff, max_pupil_cutoff = _pupil_cutoff_limits(
            pupil_cutoff_cyc_per_px_init
        )
        window = SpectralWindow.from_pupil(
            object_grid_size=object_grid_size,
            capture_grid_size=capture_height,
            max_cutoff_cyc_per_px=max_pupil_cutoff,
            edge_width_px=DEFAULT_EDGE_WIDTH_PX,
        )

        self.object_to_capture_ratio = object_to_capture_ratio
        self.window = window
        self.object = Object(measured_intensity_batch, object_to_capture_ratio)
        self.pupil = Pupil(
            window,
            phase_radial_order=pupil_phase_radial_order,
            amplitude_radial_order=pupil_amplitude_radial_order,
            pupil_cutoff_cyc_per_px=pupil_cutoff_cyc_per_px_init,
            patch_batch_size=patch_batch_size,
            pupil_cutoff_bounds=(min_pupil_cutoff, max_pupil_cutoff),
        )
        self.illumination_gains = IlluminationGains(num_illuminations)
        self.backgrounds = Backgrounds(measured_intensity_batch)
        self.scatter = Scatter(
            measured_intensity_batch,
            illumination_kx,
            illumination_ky,
            object_to_capture_ratio=object_to_capture_ratio,
            pupil_cutoff_cyc_per_px=pupil_cutoff_cyc_per_px_init,
        )
        self.forward_model = FPMForwardModel(
            window,
            illumination_kx / object_to_capture_ratio,
            illumination_ky / object_to_capture_ratio,
        )

    def forward(
        self,
        illumination_slice: slice | None = None,
    ) -> Float[Tensor, "patch_batch illumination height width"]:
        if illumination_slice is None:
            illumination_slice = slice(None)

        spectra = self.forward_model.windowed_spectra(
            self.object(),
            illumination_slice,
        )
        filtered_spectra = self.pupil()[:, None] * spectra
        predicted_low_res = self.forward_model.capture_intensity(filtered_spectra)
        predicted_low_res = (
            predicted_low_res
            * self.illumination_gains()[illumination_slice][None, :, None, None]
        )
        return (
            predicted_low_res
            + self.backgrounds.incoherent_intensity()[illumination_slice][
                None, :, None, None
            ]
            + self.scatter.incoherent_intensity(illumination_slice)
        )
