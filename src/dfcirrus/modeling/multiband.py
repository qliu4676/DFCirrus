"""Multi-band cirrus modeling workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.stats import mad_std
from reproject import reproject_interp
from scipy import ndimage

from .color import LinearRelation, MultiBandColorModel
from .config import ModelingConfig, load_config
from .data import ImageCollection
from .geometry import downsample_wcs
from .image import PlanckImage


MorphologyFilter = Callable[[np.ndarray, np.ndarray], np.ndarray | tuple[np.ndarray, object]]


@dataclass
class MultiBandResult:
    """Outputs from a multi-band cirrus model."""

    images: ImageCollection
    planck_map: np.ndarray
    planck_relations: dict[str, LinearRelation]
    color_model: MultiBandColorModel
    band_luminance: dict[str, np.ndarray]
    luminance: np.ndarray
    filtered_luminance: np.ndarray
    models: dict[str, np.ndarray]
    residuals: dict[str, np.ndarray]
    mask: np.ndarray
    morphology_result: object | None = None


class MultiBandModeler:
    """Build cirrus models from aligned multi-band images."""

    def __init__(self, config: ModelingConfig, images: ImageCollection | None = None):
        self.config = config
        self.images = ImageCollection.from_config(config) if images is None else images

    @classmethod
    def from_config(cls, filename: str | Path, *, check_files: bool = True) -> "MultiBandModeler":
        """Create a modeler from YAML configuration."""
        return cls(load_config(filename, check_files=check_files))

    def prepare_images(self) -> ImageCollection:
        """Calibrate, align, and PSF-match the band images."""
        images = self.images.to_surface_brightness().aligned(self.config.reference_band)
        if self.config.psf_match:
            images = images.matched_psfs()
        return images

    def run(
        self,
        *,
        planck_map: np.ndarray | None = None,
        morphology_filter: MorphologyFilter | None = None,
    ) -> MultiBandResult:
        """Fit the color model and build per-band cirrus images."""
        images = self.prepare_images()
        reference = images[self.config.reference_band]
        if planck_map is None:
            planck_map = PlanckImage(self.config.planck_radiance).reproject(
                reference.wcs,
                reference.shape,
                model="radiance",
            )
            planck_map = np.asarray(planck_map, dtype=float) * 1e7
        else:
            planck_map = np.asarray(planck_map, dtype=float)
        if planck_map.shape != reference.shape:
            raise ValueError("Planck and science images must share a grid")

        planck_images = images.convolved_to_fwhm(300.0)
        fit_mask = self._fit_mask(images, planck_map)
        color = MultiBandColorModel.fit(
            images,
            planck_images,
            planck_map,
            reference_band=self.config.color.reference_band,
            bands=self.config.color.bands,
            combine=self.config.color.combine,
            regression=self.config.color.regression,
            mask=fit_mask,
        )

        band_luminance = color.transform(images)
        luminance = color.luminance(images)
        mask = images.combined_mask(self.config.masks.combine) | ~np.isfinite(luminance)
        if morphology_filter is None:
            filtered_luminance, morphology_result = self._filter_morphology(
                luminance,
                mask,
                reference,
            )
        else:
            output = morphology_filter(luminance.copy(), mask.copy())
            if isinstance(output, tuple):
                filtered_luminance, morphology_result = output
            else:
                filtered_luminance, morphology_result = output, None
        filtered_luminance = np.asarray(filtered_luminance, dtype=float)
        if filtered_luminance.shape != luminance.shape:
            raise ValueError("The morphology filter changed the image shape")
        filtered_luminance[mask] = np.nan

        models = color.predict(filtered_luminance)
        offsets = color.offsets()
        residuals = {
            name: images[name].data - offsets[name] - models[name]
            for name in color.bands
        }
        for values in models.values():
            values[mask] = np.nan
        for name, values in residuals.items():
            values[mask | images[name].mask] = np.nan

        return MultiBandResult(
            images=images,
            planck_map=planck_map,
            planck_relations=color.planck_relations,
            color_model=color,
            band_luminance=band_luminance,
            luminance=luminance,
            filtered_luminance=filtered_luminance,
            models=models,
            residuals=residuals,
            mask=mask,
            morphology_result=morphology_result,
        )

    def _fit_mask(self, images: ImageCollection, planck_map: np.ndarray) -> np.ndarray:
        mask = images.combined_mask("union") | ~np.isfinite(planck_map)
        lower, upper = self.config.color.fit_sigma_range
        for image in images.values():
            valid = ~image.mask & np.isfinite(image.data)
            center = np.nanmedian(image.data[valid])
            scatter = mad_std(image.data[valid], ignore_nan=True)
            if np.isfinite(scatter) and scatter > 0:
                mask |= image.data < center + lower * scatter
                mask |= image.data > center + upper * scatter
        return mask

    def _filter_morphology(self, luminance, mask, reference):
        from .utils import remove_compact_emission

        working_scale = self.config.morphology.working_pixel_scale
        if working_scale < reference.pixel_scale:
            raise ValueError("morphology.working_pixel_scale cannot be finer than the image grid")
        scale = reference.pixel_scale / working_scale
        if scale < 1:
            working_wcs = downsample_wcs(reference.wcs, scale)
            shape = (
                max(1, int(round(reference.shape[0] * scale))),
                max(1, int(round(reference.shape[1] * scale))),
            )
            working_image, footprint = reproject_interp(
                (luminance, reference.wcs),
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
        else:
            working_wcs = reference.wcs
            working_image = luminance.copy()
            working_mask = mask.copy()

        median_size = self.config.morphology.rht.median_filter_size
        if median_size > 1:
            finite_fill = np.nanmedian(working_image[~working_mask])
            filtered_input = working_image.copy()
            filtered_input[working_mask] = finite_fill
            working_image = ndimage.median_filter(filtered_input, size=median_size)
            working_image[working_mask] = np.nan

        radius = max(
            2,
            int(round(self.config.morphology.rht.radius * 60.0 / working_scale)),
        )
        fill_radius = max(
            1,
            int(round(self.config.masks.maximum_infill_radius / working_scale)),
        )
        filtered, details = remove_compact_emission(
            working_image,
            mask=working_mask,
            kernel_type="linear",
            rht_radius=radius,
            use_peak=self.config.morphology.rht.response == "peak",
            n_threshold=self.config.morphology.compact_rejection.threshold_sigma,
            quantile=self.config.morphology.compact_rejection.quantile_fallback,
            fill_mask=self.config.masks.infill_small_holes,
            kernel_replace_masked=fill_radius,
        )
        if scale < 1:
            filtered, _ = reproject_interp(
                (filtered, working_wcs),
                reference.wcs,
                shape_out=reference.shape,
                order="bilinear",
            )
        return filtered, details


def run_multiband_modeling(
    config: ModelingConfig | str | Path,
    *,
    planck_map: np.ndarray | None = None,
    morphology_filter: MorphologyFilter | None = None,
) -> MultiBandResult:
    """Run multi-band cirrus modeling."""
    modeler = (
        MultiBandModeler.from_config(config)
        if isinstance(config, (str, Path))
        else MultiBandModeler(config)
    )
    return modeler.run(
        planck_map=planck_map,
        morphology_filter=morphology_filter,
    )


__all__ = ["MultiBandModeler", "MultiBandResult", "run_multiband_modeling"]
