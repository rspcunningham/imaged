from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, cast

import torch
import torch.nn as nn
from jaxtyping import Complex, Float
from torch import Tensor

# Use unitary Fourier transforms
fft2 = cast(Callable[..., Tensor], partial(torch.fft.fft2, norm="ortho"))
ifft2 = cast(Callable[..., Tensor], partial(torch.fft.ifft2, norm="ortho"))
rfft2 = cast(Callable[..., Tensor], partial(torch.fft.rfft2, norm="ortho"))
irfft2 = cast(Callable[..., Tensor], partial(torch.fft.irfft2, norm="ortho"))

# The pupil aperture is a sigmoid of width edge_width_px. Past this many
# e-foldings beyond the largest allowed cutoff it is treated as exactly zero.
PUPIL_TAIL_TOLERANCE = 1e-12
# The tilted object spectrum is read from a 2x oversampled FFT with an 8-tap
# Kaiser-Bessel stencil per axis: ~1e-6 relative error, the float32 FFT floor.
OVERSAMPLING = 2
INTERPOLATION_TAPS = 8


@dataclass(frozen=True)
class SpectralWindow:
    """Square region of the object spectrum around DC outside which the pupil is zero.

    Bins are kept in fft order: [0, half_width) then [-half_width, 0).
    """

    object_grid_size: int
    capture_grid_size: int
    half_width: int

    @classmethod
    def from_pupil(
        cls,
        *,
        object_grid_size: int,
        capture_grid_size: int,
        max_cutoff_cyc_per_px: float,
        edge_width_px: float,
    ) -> SpectralWindow:
        tail_px = edge_width_px * math.log(1.0 / PUPIL_TAIL_TOLERANCE)
        half_width = math.ceil(max_cutoff_cyc_per_px * object_grid_size + tail_px)
        # |field|^2 has twice the field's bandwidth. It must sit under the capture
        # Nyquist for the intensity to be formed exactly on the capture grid.
        if 4 * half_width - 2 >= capture_grid_size:
            raise ValueError(
                f"pupil window half-width {half_width} px does not fit a "
                f"{capture_grid_size} px capture grid"
            )
        return cls(object_grid_size, capture_grid_size, half_width)

    @property
    def size(self) -> int:
        return 2 * self.half_width

    def coords(self) -> Float[Tensor, "window"]:
        """Signed spectral pixel coordinate of each window bin."""
        w = self.half_width
        return torch.cat([torch.arange(0, w), torch.arange(-w, 0)]).to(
            torch.get_default_dtype()
        )

    def object_index(self) -> Tensor:
        w, grid = self.half_width, self.object_grid_size
        return torch.cat([torch.arange(0, w), torch.arange(grid - w, grid)])

    def capture_index(self) -> Tensor:
        w, grid = self.half_width, self.capture_grid_size
        return torch.cat([torch.arange(0, w), torch.arange(grid - w, grid)])


class FPMForwardModel(nn.Module):
    """Capture-grid intensities from the object's spectrum inside the pupil window.

    For LED i with tilt d_i in spectral pixels, FFT(O * exp(2 pi i d_i.x / G))[k]
    equals the DTFT of O at k - d_i. Instead of tilting and transforming the
    object once per LED, the DTFT is sampled once per call on a 2x oversampled
    grid and each LED's window is read through a separable Kaiser-Bessel stencil.
    The pupil-limited field's intensity is band-limited below the capture Nyquist,
    so it is formed on the capture grid directly, with the pixel average applied
    as a transfer function. Both steps reproduce the full-grid model to float32
    precision without touching the 99% of the object grid the pupil zeroes.
    """

    def __init__(
        self,
        window: SpectralWindow,
        illumination_kx: Float[Tensor, "illumination"],
        illumination_ky: Float[Tensor, "illumination"],
    ) -> None:
        super().__init__()
        grid = window.object_grid_size
        capture = window.capture_grid_size
        fine = OVERSAMPLING * grid
        taps = INTERPOLATION_TAPS
        ratio = grid // capture
        self.window = window
        self.fine_grid_size = fine
        self.object_to_capture_ratio = ratio
        real_dtype = torch.get_default_dtype()
        complex_dtype = (
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        )

        beta = math.pi * math.sqrt(
            (taps / OVERSAMPLING) ** 2 * (OVERSAMPLING - 0.5) ** 2 - 0.8
        )

        def kernel(t: Tensor) -> Tensor:
            support = 1.0 - (2.0 * t / taps) ** 2
            return torch.where(
                support > 0,
                torch.i0(beta * support.clamp_min(0).sqrt()),
                torch.zeros_like(t),
            )

        # Deapodization: divide the object by the stencil's transfer function,
        # evaluated where the object sits once centred at [-G/2, G/2) on the fine grid.
        t = torch.linspace(-taps / 2, taps / 2, 8001, dtype=torch.float64)
        nu = (torch.arange(grid, dtype=torch.float64) - grid / 2) / fine
        kernel_transfer = (
            kernel(t)[None, :] * torch.cos(2 * math.pi * nu[:, None] * t[None, :])
        ).sum(1) * (t[1] - t[0])
        self.register_buffer("deapodization", (1.0 / kernel_transfer).to(real_dtype))

        coords = window.coords().to(torch.float64)
        block_length = OVERSAMPLING * (window.size - 1) + taps
        local = torch.arange(block_length, dtype=torch.float64)
        for axis, shift_cyc_per_px in (("y", illumination_ky), ("x", illumination_kx)):
            shift = shift_cyc_per_px.detach().cpu().to(torch.float64) * grid
            kappa = coords[None, :] - shift[:, None]
            start = (
                torch.floor(OVERSAMPLING * (-window.half_width - shift)) - taps // 2 + 1
            )
            position = start[:, None] + local[None, :]
            weights = kernel(OVERSAMPLING * kappa[:, :, None] - position[:, None, :])
            # Centring the object at -G/2 on the fine grid adds a linear phase.
            phase = torch.exp(-1j * math.pi * kappa)
            self.register_buffer(f"block_index_{axis}", position.long() % fine)
            self.register_buffer(f"weights_{axis}", weights.to(real_dtype))
            self.register_buffer(f"phase_{axis}", phase.to(complex_dtype))

        # ratio x ratio pixel average, anchored like avg_pool2d, as a transfer
        # function on the capture grid (rfft2 layout).
        k = torch.fft.fftfreq(capture, dtype=torch.float64) * capture
        a = torch.arange(ratio, dtype=torch.float64)
        axis_transfer = (
            torch.exp(2j * math.pi * k[:, None] * a[None, :] / grid).sum(1) / ratio
        )
        pixel_transfer = (
            axis_transfer[:, None] * axis_transfer[None, : capture // 2 + 1]
        )
        self.register_buffer("pixel_transfer", pixel_transfer.to(complex_dtype))
        self.register_buffer("capture_index", window.capture_index())

    def windowed_spectra(
        self,
        object_tensor: Complex[Tensor, "patch_batch object_height object_width"],
        illumination_slice: slice,
    ) -> Complex[Tensor, "patch_batch illumination window window"]:
        """Tilted object spectra on the pupil window: DTFT(O)(k - d_i) for each LED."""
        grid = self.window.object_grid_size
        fine = self.fine_grid_size
        half = grid // 2
        deapodization = cast(Tensor, self.deapodization)

        deapodized = torch.view_as_real(
            object_tensor * deapodization[:, None] * deapodization[None, :]
        )
        padded = deapodized.new_zeros(object_tensor.shape[0], fine, fine, 2)
        padded[:, :grid, :grid] = deapodized
        padded = torch.roll(padded, shifts=(-half, -half), dims=(1, 2))
        spectrum = torch.view_as_real(
            torch.fft.fft2(torch.view_as_complex(padded)) / grid
        )

        rows = cast(Tensor, self.block_index_y)[illumination_slice]
        cols = cast(Tensor, self.block_index_x)[illumination_slice]
        block = spectrum[:, rows[:, :, None], cols[:, None, :]]
        block = torch.einsum(
            "cjy,bcyxr->bcjxr", cast(Tensor, self.weights_y)[illumination_slice], block
        )
        block = torch.einsum(
            "ckx,bcjxr->bcjkr", cast(Tensor, self.weights_x)[illumination_slice], block
        )
        return (
            torch.view_as_complex(block.contiguous())
            * cast(Tensor, self.phase_y)[illumination_slice][None, :, :, None]
            * cast(Tensor, self.phase_x)[illumination_slice][None, :, None, :]
        )

    def capture_intensity(
        self,
        filtered_spectra: Complex[Tensor, "patch_batch illumination window window"],
    ) -> Float[Tensor, "patch_batch illumination height width"]:
        """Pixel-averaged |field|^2 on the capture grid from pupil-filtered spectra."""
        capture = self.window.capture_grid_size
        index = cast(Tensor, self.capture_index)
        patch_batch_size, num_illuminations = filtered_spectra.shape[:2]

        spectrum = torch.zeros(
            patch_batch_size,
            num_illuminations,
            capture,
            capture,
            2,
            dtype=filtered_spectra.real.dtype,
            device=filtered_spectra.device,
        )
        spectrum[:, :, index[:, None], index[None, :]] = torch.view_as_real(
            filtered_spectra
        )
        # The field at every ratio-th object pixel; exact because the window
        # fits under the capture Nyquist.
        field = torch.view_as_real(ifft2(torch.view_as_complex(spectrum)))
        field = field / self.object_to_capture_ratio
        intensity = field.square().sum(-1)
        return irfft2(
            rfft2(intensity) * cast(Tensor, self.pixel_transfer),
            s=(capture, capture),
        )
