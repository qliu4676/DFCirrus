"""Starlet morphology backend."""

from __future__ import annotations

import numpy as np
from astropy.stats import mad_std
from scipy import ndimage

from .base import MorphologyResult, restore_grid, to_working_grid


class StarletFilter:
    """Extract selected scales with an undecimated starlet transform."""

    def __init__(self, config):
        self.config = config

    def extract(self, image, mask, reference) -> MorphologyResult:
        """Filter a luminance image."""
        grid = to_working_grid(
            image,
            mask,
            reference,
            self.config.working_pixel_scale,
        )
        coefficients, coarse = starlet_transform(
            grid.image,
            grid.mask,
            self.config.starlet.scales,
        )
        selected = []
        components = {"input": grid.image, "coarse": coarse}
        for scale, coefficient in enumerate(coefficients, start=1):
            component = coefficient.copy()
            if self.config.starlet.threshold_sigma > 0:
                noise = mad_std(component[~grid.mask], ignore_nan=True)
                if np.isfinite(noise) and noise > 0:
                    component[np.abs(component) < self.config.starlet.threshold_sigma * noise] = 0
            components[f"scale_{scale}"] = component
            if scale in self.config.starlet.keep_scales:
                selected.append(component)

        filtered = np.sum(selected, axis=0)
        if self.config.starlet.include_coarse:
            filtered = filtered + coarse
        filtered[grid.mask] = np.nan
        restored = restore_grid(filtered, grid, reference)
        restored[mask] = np.nan
        components["filtered"] = filtered
        return MorphologyResult(
            image=restored,
            backend="starlet",
            components=components,
            metadata={
                "keep_scales": self.config.starlet.keep_scales,
                "working_pixel_scale": self.config.working_pixel_scale,
            },
        )


def starlet_transform(image, mask, scales):
    """Return starlet coefficients and the coarse residual."""
    current = np.asarray(image, dtype=float).copy()
    valid = ~np.asarray(mask, dtype=bool) & np.isfinite(current)
    current[~valid] = 0
    coefficients = []
    for scale in range(scales):
        smooth = _smooth_masked(current, valid, 2**scale)
        coefficient = current - smooth
        coefficient[~valid] = np.nan
        coefficients.append(coefficient)
        smooth[~valid] = 0
        current = smooth
    current[~valid] = np.nan
    return coefficients, current


def _smooth_masked(image, valid, step):
    kernel = np.zeros(4 * step + 1)
    kernel[::step] = np.array([1, 4, 6, 4, 1], dtype=float) / 16
    numerator = ndimage.convolve1d(image, kernel, axis=0, mode="reflect")
    numerator = ndimage.convolve1d(numerator, kernel, axis=1, mode="reflect")
    weights = ndimage.convolve1d(valid.astype(float), kernel, axis=0, mode="reflect")
    weights = ndimage.convolve1d(weights, kernel, axis=1, mode="reflect")
    result = np.zeros_like(image, dtype=float)
    np.divide(numerator, weights, out=result, where=weights > 0)
    return result


__all__ = ["StarletFilter", "starlet_transform"]
