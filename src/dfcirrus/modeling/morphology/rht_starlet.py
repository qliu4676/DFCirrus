"""Sequential RHT and starlet backend."""

from __future__ import annotations

import numpy as np

from .base import MorphologyResult, restore_grid, to_working_grid
from .rht import RHTFilter
from .starlet import reconstruct_starlet


class RHTStarletFilter:
    """Apply starlet scale cleanup to an RHT-filtered image."""

    def __init__(self, config, mask_config=None, *, random_state=None):
        self.config = config
        self.rht = RHTFilter(
            config,
            mask_config,
            random_state=random_state,
        )

    def extract(self, image, mask, reference) -> MorphologyResult:
        """Filter a luminance image."""
        rht_result = self.rht.extract(image, mask, reference)
        grid = to_working_grid(
            image,
            mask,
            reference,
            self.config.working_pixel_scale,
        )
        rht_filtered = rht_result.components["filtered"]
        filtered, starlet_components = reconstruct_starlet(
            rht_filtered,
            grid.mask,
            self.config.starlet,
        )
        restored = restore_grid(filtered, grid, reference)
        restored[mask] = np.nan

        components = {
            f"rht_{name}": values
            for name, values in rht_result.components.items()
        }
        components.update(
            {
                f"starlet_{name}": values
                for name, values in starlet_components.items()
                if name != "filtered"
            }
        )
        components["starlet_removed"] = rht_filtered - filtered
        components["filtered"] = filtered
        dropped = tuple(
            scale
            for scale in range(1, self.config.starlet.scales + 1)
            if scale not in self.config.starlet.keep_scales
        )
        return MorphologyResult(
            image=restored,
            backend="rht_starlet",
            components=components,
            metadata={
                **rht_result.metadata,
                "dropped_scales": dropped,
                "keep_scales": self.config.starlet.keep_scales,
                "include_coarse": self.config.starlet.include_coarse,
            },
        )


__all__ = ["RHTStarletFilter"]
