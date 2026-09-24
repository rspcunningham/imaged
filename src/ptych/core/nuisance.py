import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

_SCATTER_RANK = 2
_BACKGROUND_QUANTILE = 0.01
# Floor on the initial background as a fraction of the capture mean. With
# signed dark subtraction the low quantile of a darkfield capture is at or below
# zero, and a softplus parameter initialised there has no usable gradient.
_BACKGROUND_MEAN_FLOOR = 0.3


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


class Backgrounds(nn.Module):
    """Positive, spatially constant background for every illumination."""

    def __init__(
        self,
        measured_intensity_batch: Float[
            Tensor, "patch_batch illumination height width"
        ],
    ) -> None:
        super().__init__()
        num_illuminations = measured_intensity_batch.shape[1]
        flat = (
            measured_intensity_batch.detach()
            .permute(1, 0, 2, 3)
            .reshape(num_illuminations, -1)
            .cpu()
        )
        kth_index = max(1, int(_BACKGROUND_QUANTILE * flat.shape[1]))
        background = torch.maximum(
            flat.kthvalue(kth_index, dim=1).values,
            _BACKGROUND_MEAN_FLOOR * flat.mean(dim=1),
        ).clamp_min(1e-8)

        self.raw_backgrounds = nn.Parameter(_inverse_softplus(background))

    def incoherent_intensity(self) -> Float[Tensor, "illumination"]:
        return F.softplus(self.raw_backgrounds)


class Scatter(nn.Module):
    """Positive spatial bases with learned weights for every illumination."""

    def __init__(
        self,
        measured_intensity_batch: Float[
            Tensor, "patch_batch illumination height width"
        ],
        illumination_kx: Float[Tensor, "illumination"],
        illumination_ky: Float[Tensor, "illumination"],
        *,
        object_to_capture_ratio: int,
        pupil_cutoff_cyc_per_px: Tensor | float,
    ) -> None:
        super().__init__()
        _, num_illuminations, height, width = measured_intensity_batch.shape
        # Seed from darkfield captures, as in the global-nuisance ablation.
        # This selection is only for initialization; every illumination is fitted.
        illumination_radius = torch.sqrt(
            (illumination_kx.detach().cpu() / object_to_capture_ratio).square()
            + (illumination_ky.detach().cpu() / object_to_capture_ratio).square()
        )
        pupil_cutoff = torch.as_tensor(pupil_cutoff_cyc_per_px).detach().cpu()
        seed_illuminations = illumination_radius > pupil_cutoff

        if seed_illuminations.any():
            seed = (
                measured_intensity_batch.detach()
                .cpu()[:, seed_illuminations]
                .mean(dim=1)
                .clamp_min(1e-8)
            )
        else:
            seed = torch.full(
                (measured_intensity_batch.shape[0], height, width),
                1e-8,
                dtype=measured_intensity_batch.dtype,
            )
        seed = 0.02 * seed[:, None] / _SCATTER_RANK
        seed = seed.expand(-1, _SCATTER_RANK, -1, -1).clone()
        coefficients = torch.ones(num_illuminations, _SCATTER_RANK)

        self.raw_basis = nn.Parameter(_inverse_softplus(seed))
        self.raw_coefficients = nn.Parameter(_inverse_softplus(coefficients))

    def incoherent_intensity(
        self,
        illumination_slice: slice,
    ) -> Float[Tensor, "patch_batch illumination height width"]:
        basis = F.softplus(self.raw_basis)
        coefficients = F.softplus(self.raw_coefficients)[illumination_slice]
        return torch.einsum("brhw,ir->bihw", basis, coefficients)
