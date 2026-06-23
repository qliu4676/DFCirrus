"""Rolling Hough transform backend."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import MorphologyResult, restore_grid, to_working_grid


class RHTFilter:
    """Extract cirrus-like structure with directional line filters."""

    def __init__(self, config, mask_config=None):
        self.config = config
        self.mask_config = mask_config

    def extract(self, image, mask, reference) -> MorphologyResult:
        """Filter a luminance image."""
        from ..utils import remove_compact_emission

        grid = to_working_grid(
            image,
            mask,
            reference,
            self.config.working_pixel_scale,
        )
        working_image = grid.image
        median_size = self.config.rht.median_filter_size
        if median_size > 1:
            fill_value = np.nanmedian(working_image[~grid.mask])
            filtered_input = working_image.copy()
            filtered_input[grid.mask] = fill_value
            working_image = ndimage.median_filter(filtered_input, size=median_size)
            working_image[grid.mask] = np.nan

        radius = max(
            2,
            int(round(self.config.rht.radius * 60.0 / self.config.working_pixel_scale)),
        )
        fill_mask = self.config.rht.maskfill
        filtered, details = remove_compact_emission(
            working_image,
            mask=grid.mask,
            kernel_type="linear",
            rht_radius=radius,
            n_theta=(
                None
                if self.config.rht.angle_bins == "auto"
                else self.config.rht.angle_bins
            ),
            use_peak=self.config.rht.response == "peak",
            n_threshold=(
                self.config.compact_rejection.threshold_sigma
                if self.config.compact_rejection.method == "segmentation"
                else None
            ),
            quantile=self.config.compact_rejection.quantile_fallback,
            fill_mask=fill_mask,
            kernel_replace_masked=self.config.rht.infill_radius,
            infill_backend=self.config.rht.infill_backend,
        )
        restored = restore_grid(filtered, grid, reference)
        restored[mask] = np.nan

        components = {"input": grid.image, "filtered": filtered}
        for source, target in (
            ("image_smooth", "smoothed"),
            ("image_response", "response"),
            ("image_residual", "residual"),
            ("image_ratio", "ratio"),
            ("isolated", "compact_mask"),
        ):
            if hasattr(details, source):
                components[target] = np.asarray(getattr(details, source))
        return MorphologyResult(
            image=restored,
            backend="rht",
            components=components,
            metadata={
                "radius_pixels": radius,
                "working_pixel_scale": self.config.working_pixel_scale,
                "maskfill": fill_mask,
                "infill_backend": self.config.rht.infill_backend,
                "infill_radius": self.config.rht.infill_radius,
            },
        )


__all__ = ["RHTFilter"]
