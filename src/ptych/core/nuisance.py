import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

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
