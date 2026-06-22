"""WCS and resolution helpers used by cirrus modeling."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS


def downsample_wcs(wcs_input: WCS, scale: float) -> WCS:
    """Return a WCS for an image resampled by ``scale``.

    ``scale`` is the output/input pixel-count ratio. For example, ``0.25``
    describes 4x4 binning. The celestial transform, including rotated PC or CD
    matrices, is retained while the pixel size and reference pixel are updated.
    """
    if not 0 < scale <= 1:
        raise ValueError("scale must be in the interval (0, 1]")

    result = deepcopy(wcs_input)
    result.wcs.crpix = (np.asarray(wcs_input.wcs.crpix) - 0.5) * scale + 0.5

    if result.wcs.has_cd():
        result.wcs.cd = np.asarray(wcs_input.wcs.cd) / scale
    else:
        result.wcs.cdelt = np.asarray(wcs_input.wcs.cdelt) / scale

    if wcs_input.pixel_shape is not None:
        result.pixel_shape = tuple(
            max(1, int(round(length * scale))) for length in wcs_input.pixel_shape
        )
    return result


def gaussian_beam_kernel(
    pixel_scale: float,
    fwhm_image: float,
    fwhm_target: float = 300.0,
    *,
    truncate: float = 4.0,
) -> Gaussian2DKernel:
    """Create a Gaussian kernel that broadens one Gaussian beam to another.

    FWHM values are in arcsec; ``pixel_scale`` is in arcsec/pixel.
    """
    values = {
        "pixel_scale": pixel_scale,
        "fwhm_image": fwhm_image,
        "fwhm_target": fwhm_target,
    }
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if fwhm_target <= fwhm_image:
        raise ValueError("fwhm_target must be broader than fwhm_image")

    fwhm_kernel = np.sqrt(fwhm_target**2 - fwhm_image**2)
    sigma_pixels = gaussian_fwhm_to_sigma * fwhm_kernel / pixel_scale
    size = max(3, 2 * int(np.ceil(truncate * sigma_pixels)) + 1)
    return Gaussian2DKernel(sigma_pixels, x_size=size, y_size=size)
