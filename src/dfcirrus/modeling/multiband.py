"""Multi-band cirrus modeling workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from astropy.io import fits
from astropy.stats import mad_std
from .color import ColorMeasurement, LinearRelation, MultiBandColorModel
from .config import ModelingConfig, load_config
from .data import ImageCollection
from .image import PlanckImage
from .morphology import MorphologyResult, create_morphology_filter


MorphologyFilter = Callable[[np.ndarray, np.ndarray], np.ndarray | tuple[np.ndarray, object]]


@dataclass
class MultiBandResult:
    """Outputs from a multi-band cirrus model."""

    images: ImageCollection
    planck_images: ImageCollection
    planck_map: np.ndarray
    planck_relations: dict[str, LinearRelation]
    color_model: MultiBandColorModel
    band_luminance: dict[str, np.ndarray]
    luminance: np.ndarray
    filtered_luminance: np.ndarray
    models: dict[str, np.ndarray]
    residuals: dict[str, np.ndarray]
    colors: dict[str, ColorMeasurement]
    mask: np.ndarray
    fit_mask: np.ndarray
    morphology_result: object | None = None

    def write(self, output_dir: str | Path, *, overwrite: bool = False) -> None:
        """Write model images and colors."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        reference = self.images[self.color_model.reference_band]
        header = reference.wcs.to_header()
        header["BUNIT"] = "kJy/sr"
        fits.writeto(
            output_dir / "luminance.fits",
            self.luminance,
            header,
            overwrite=overwrite,
        )
        fits.writeto(
            output_dir / "luminance_filtered.fits",
            self.filtered_luminance,
            header,
            overwrite=overwrite,
        )
        for name, model in self.models.items():
            fits.writeto(
                output_dir / f"cirrus_model_{name}.fits",
                model,
                header,
                overwrite=overwrite,
            )
            fits.writeto(
                output_dir / f"residual_{name}.fits",
                self.residuals[name],
                header,
                overwrite=overwrite,
            )
        color_data = {
            name: {
                "value": measurement.value,
                "error": measurement.error,
                "unit": "mag",
                "source": measurement.source,
            }
            for name, measurement in self.colors.items()
        }
        with (output_dir / "cirrus_colors.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(color_data, stream, sort_keys=False)


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
            bootstrap_samples=self.config.color.bootstrap_samples,
            random_state=self.config.run.random_seed,
        )

        band_luminance = color.transform(images)
        luminance = color.luminance(images)
        mask = images.combined_mask(self.config.masks.combine) | ~np.isfinite(luminance)
        if morphology_filter is None:
            backend = create_morphology_filter(
                self.config.morphology,
                self.config.masks,
                random_state=self.config.run.random_seed,
            )
            morphology_result = backend.extract(luminance, mask, reference)
            filtered_luminance = morphology_result.image
        else:
            output = morphology_filter(luminance.copy(), mask.copy())
            if isinstance(output, tuple):
                filtered_luminance, details = output
                morphology_result = MorphologyResult(
                    image=np.asarray(filtered_luminance),
                    backend="custom",
                    metadata={"details": details},
                )
            else:
                filtered_luminance = output
                morphology_result = MorphologyResult(
                    image=np.asarray(output),
                    backend="custom",
                )
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
            planck_images=planck_images,
            planck_map=planck_map,
            planck_relations=color.planck_relations,
            color_model=color,
            band_luminance=band_luminance,
            luminance=luminance,
            filtered_luminance=filtered_luminance,
            models=models,
            residuals=residuals,
            colors=color.colors("planck"),
            mask=mask,
            fit_mask=fit_mask,
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
