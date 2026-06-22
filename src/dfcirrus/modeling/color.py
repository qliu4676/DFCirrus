"""Planck calibration and multi-band color models."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .data import ImageCollection


@dataclass(frozen=True)
class LinearRelation:
    """Linear relation with residual scatter."""

    slope: float
    intercept: float
    scatter: float
    samples: int

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.slope * values + self.intercept


def fit_linear_relation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    method: str = "linear",
    clip_sigma: float = 4.0,
    max_iterations: int = 5,
) -> LinearRelation:
    """Fit a sigma-clipped linear relation."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool).ravel()
    if valid.sum() < 3:
        raise ValueError("At least three finite samples are required")

    for _ in range(max_iterations):
        slope, intercept = _linear_coefficients(x[valid], y[valid], method)
        residual = y - (slope * x + intercept)
        center = np.nanmedian(residual[valid])
        scatter = 1.4826 * np.nanmedian(np.abs(residual[valid] - center))
        if not np.isfinite(scatter) or scatter == 0:
            break
        updated = valid & (np.abs(residual - center) <= clip_sigma * scatter)
        if np.array_equal(updated, valid) or updated.sum() < 3:
            break
        valid = updated

    slope, intercept = _linear_coefficients(x[valid], y[valid], method)
    residual = y[valid] - (slope * x[valid] + intercept)
    scatter = float(np.sqrt(np.mean(residual**2)))
    if not np.isfinite(slope) or np.isclose(slope, 0):
        raise ValueError("The fitted slope is zero or non-finite")
    return LinearRelation(float(slope), float(intercept), scatter, int(valid.sum()))


def _linear_coefficients(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    if method not in {"linear", "bisector"}:
        raise ValueError("method must be 'linear' or 'bisector'")
    slope_yx, intercept_yx = np.polyfit(x, y, 1)
    if method == "linear":
        return float(slope_yx), float(intercept_yx)

    slope_xy, _ = np.polyfit(y, x, 1)
    if np.isclose(slope_xy, 0):
        return float(slope_yx), float(intercept_yx)
    inverse_slope = 1.0 / slope_xy
    denominator = slope_yx + inverse_slope
    if np.isclose(denominator, 0):
        return float(slope_yx), float(intercept_yx)
    slope = (
        slope_yx * inverse_slope
        - 1
        + np.sqrt((1 + slope_yx**2) * (1 + inverse_slope**2))
    ) / denominator
    intercept = np.mean(y) - slope * np.mean(x)
    return float(slope), float(intercept)


def combine_luminance(maps: dict[str, np.ndarray], method: str) -> np.ndarray:
    """Combine band-derived luminance maps."""
    stack = np.stack(list(maps.values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "mean":
            return np.nanmean(stack, axis=0)
        if method == "median":
            return np.nanmedian(stack, axis=0)
    raise ValueError("method must be 'mean' or 'median'")


@dataclass(frozen=True)
class MultiBandColorModel:
    """Color relations between a shared luminance field and each band."""

    reference_band: str
    planck_relations: dict[str, LinearRelation]
    color_relations: dict[str, LinearRelation]
    combine: str = "median"

    @classmethod
    def fit(
        cls,
        images: ImageCollection,
        planck_images: ImageCollection,
        planck_map: np.ndarray,
        *,
        reference_band: str,
        bands: tuple[str, ...] | None = None,
        combine: str = "median",
        regression: str = "bisector",
        mask: np.ndarray | None = None,
        iterations: int = 3,
    ) -> "MultiBandColorModel":
        """Fit Planck and inter-band color relations."""
        selected = tuple(images) if bands is None else tuple(bands)
        if reference_band not in selected:
            raise ValueError("reference_band must be included in bands")
        if set(selected) - set(images) or set(selected) - set(planck_images):
            raise ValueError("All selected bands must be present in both image collections")
        planck_map = np.asarray(planck_map, dtype=float)
        if any(images[name].shape != planck_map.shape for name in selected):
            raise ValueError("Planck and band images must share a grid")

        common_mask = np.any(np.stack([images[name].mask for name in selected]), axis=0)
        common_mask |= np.any(
            np.stack([planck_images[name].mask for name in selected]),
            axis=0,
        )
        common_mask |= ~np.isfinite(planck_map)
        if mask is not None:
            common_mask |= np.asarray(mask, dtype=bool)

        planck_relations = {
            name: fit_linear_relation(
                planck_map,
                planck_images[name].data,
                mask=common_mask,
                method=regression,
            )
            for name in selected
        }
        initial = {
            name: (images[name].data - planck_relations[name].intercept)
            / planck_relations[name].slope
            for name in selected
        }
        luminance = combine_luminance(initial, combine)
        luminance[common_mask] = np.nan

        relations = {}
        for _ in range(max(1, iterations)):
            relations = {
                name: fit_linear_relation(
                    luminance,
                    images[name].data - planck_relations[name].intercept,
                    mask=common_mask,
                    method=regression,
                )
                for name in selected
            }
            reference_scale = relations[reference_band].slope
            relations = {
                name: LinearRelation(
                    slope=relation.slope / reference_scale,
                    intercept=relation.intercept,
                    scatter=relation.scatter,
                    samples=relation.samples,
                )
                for name, relation in relations.items()
            }
            transformed = {
                name: (
                    images[name].data
                    - planck_relations[name].intercept
                    - relations[name].intercept
                )
                / relations[name].slope
                for name in selected
            }
            luminance = combine_luminance(transformed, combine)
            luminance[common_mask] = np.nan

        return cls(reference_band, planck_relations, relations, combine)

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(self.color_relations)

    def offsets(self) -> dict[str, float]:
        """Return the fitted additive offsets by band."""
        return {
            name: self.planck_relations[name].intercept + relation.intercept
            for name, relation in self.color_relations.items()
        }

    def transform(self, images: ImageCollection) -> dict[str, np.ndarray]:
        """Transform band images into luminance estimates."""
        offsets = self.offsets()
        transformed = {}
        for name, relation in self.color_relations.items():
            values = (images[name].data - offsets[name]) / relation.slope
            values = np.asarray(values, dtype=float)
            values[images[name].mask] = np.nan
            transformed[name] = values
        return transformed

    def luminance(self, images: ImageCollection) -> np.ndarray:
        """Build the combined luminance image."""
        return combine_luminance(self.transform(images), self.combine)

    def predict(self, luminance: np.ndarray) -> dict[str, np.ndarray]:
        """Predict the cirrus intensity in each band."""
        return {
            name: relation.slope * luminance
            for name, relation in self.color_relations.items()
        }


__all__ = [
    "LinearRelation", "MultiBandColorModel", "combine_luminance",
    "fit_linear_relation",
]
