import torch
import torch.nn.functional as F
from jaxtyping import Complex, Float

from ptych.core.forward import fft2, ifft2


def synthesize_captures(
    object_tensor: Complex[torch.Tensor, "object_height object_width"],
    pupil_tensor: Complex[torch.Tensor, "object_height object_width"],
    illumination_kx: Float[torch.Tensor, "illumination"],
    illumination_ky: Float[torch.Tensor, "illumination"],
    object_to_capture_ratio: int,
) -> Float[torch.Tensor, "illumination height width"]:
    """
    Generate synthetic captures with the full-resolution reference model: tilt the
    object by each illumination, transform, apply the pupil, transform back, and
    average |field|^2 over each capture pixel.

    Args:
        object_tensor: Complex object tensor [object_height, object_width]
        pupil_tensor: Complex pupil tensor [object_height, object_width], fft order
        illumination_kx: Normalized k-vectors in x direction [illumination]
        illumination_ky: Normalized k-vectors in y direction [illumination]
        object_to_capture_ratio: Linear ratio between object and capture grids

    Returns:
        Synthetic captures [illumination, height, width] as float intensities
    """
    object_grid_size = object_tensor.shape[-1]
    coords = torch.arange(
        object_grid_size,
        dtype=torch.get_default_dtype(),
        device=object_tensor.device,
    )
    shift_x = (illumination_kx / object_to_capture_ratio * object_grid_size).to(coords)
    shift_y = (illumination_ky / object_to_capture_ratio * object_grid_size).to(coords)
    phase = (
        2
        * torch.pi
        * (
            shift_x[:, None, None] * coords[None, None, :]
            + shift_y[:, None, None] * coords[None, :, None]
        )
        / object_grid_size
    )
    tilted_objects = object_tensor[None] * torch.exp(1j * phase.to(object_tensor.dtype))
    fields = ifft2(pupil_tensor[None] * fft2(tilted_objects))
    intensities = fields.abs().square()

    if object_to_capture_ratio > 1:
        intensities = F.avg_pool2d(
            intensities[:, None],
            kernel_size=object_to_capture_ratio,
        )[:, 0]

    return intensities
