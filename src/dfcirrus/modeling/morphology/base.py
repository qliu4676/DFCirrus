"""Morphology filter interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from reproject import reproject_interp

from ..config import MorphologyConfig
from ..data import BandImage
from ..geometry import downsample_wcs


@dataclass
class MorphologyResult:
    """Morphology-filtered luminance and diagnostic maps."""

    image: np.ndarray
    backend: str
    components: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingGrid:
    image: np.ndarray
    mask: np.ndarray
    wcs: Any
    scale: float


def to_working_grid(
    image: np.ndarray,
    mask: np.ndarray,
    reference: BandImage,
    pixel_scale: float,
) -> WorkingGrid:
    """Reproject an image and mask to a working pixel scale."""
    if pixel_scale < reference.pixel_scale:
        raise ValueError("Working pixel scale cannot be finer than the image grid")
    scale = reference.pixel_scale / pixel_scale
    if np.isclose(scale, 1):
        return WorkingGrid(image.copy(), mask.copy(), reference.wcs, 1.0)

    working_wcs = downsample_wcs(reference.wcs, scale)
    shape = (
        max(1, int(round(reference.shape[0] * scale))),
        max(1, int(round(reference.shape[1] * scale))),
    )
    working_image, footprint = reproject_interp(
        (image, reference.wcs),
        working_wcs,
        shape_out=shape,
        order="bilinear",
    )
    working_mask, _ = reproject_interp(
        (mask.astype(float), reference.wcs),
        working_wcs,
        shape_out=shape,
        order="nearest-neighbor",
    )
    working_mask = (working_mask > 0.5) | (footprint <= 0) | ~np.isfinite(working_image)
    return WorkingGrid(working_image, working_mask, working_wcs, scale)


def restore_grid(image: np.ndarray, grid: WorkingGrid, reference: BandImage) -> np.ndarray:
    """Reproject a working image to its reference grid."""
    if np.isclose(grid.scale, 1):
        return np.asarray(image, dtype=float)
    restored, _ = reproject_interp(
        (image, grid.wcs),
        reference.wcs,
        shape_out=reference.shape,
        order="bilinear",
    )
    return restored


def create_morphology_filter(
    config: MorphologyConfig,
    mask_config=None,
    *,
    random_state: int | np.random.Generator | None = None,
):
    """Create the configured morphology backend."""
    if config.backend == "rht":
        from .rht import RHTFilter

        return RHTFilter(config, mask_config, random_state=random_state)
    if config.backend == "rht_starlet":
        from .rht_starlet import RHTStarletFilter

        return RHTStarletFilter(config, mask_config, random_state=random_state)
    raise ValueError(f"Unknown morphology backend: {config.backend}")


__all__ = [
    "MorphologyResult", "WorkingGrid", "create_morphology_filter",
    "restore_grid", "to_working_grid",
]
