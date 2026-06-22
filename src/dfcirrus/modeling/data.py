"""Multi-band image containers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy.convolution import convolve_fft
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from .config import BandConfig, ModelingConfig
from .geometry import gaussian_beam_kernel


def adu_to_kjy_sr(data: np.ndarray, zeropoint: float, pixel_scale: float) -> np.ndarray:
    """Convert calibrated image counts to kJy/sr."""
    arcsec_per_radian = np.degrees(1.0) * 3600.0
    return (
        3.631
        * np.asarray(data, dtype=float)
        * arcsec_per_radian**2
        / (10 ** (zeropoint / 2.5) * pixel_scale**2)
    )


@dataclass
class BandImage:
    """Image data and calibration metadata for one band."""

    name: str
    data: np.ndarray
    wcs: WCS
    wavelength: float
    pixel_scale: float
    psf_fwhm: float
    zeropoint: float
    mask: np.ndarray | None = None
    header: fits.Header | None = None
    path: Path | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError(f"Band {self.name!r} must contain a 2D image")
        if self.mask is None:
            self.mask = ~np.isfinite(self.data)
        else:
            self.mask = np.asarray(self.mask, dtype=bool) | ~np.isfinite(self.data)
            if self.mask.shape != self.data.shape:
                raise ValueError(f"Band {self.name!r} mask shape does not match its image")
        for field_name in ("wavelength", "pixel_scale", "psf_fwhm"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"Band {self.name!r} {field_name} must be positive")

    @classmethod
    def from_config(cls, name: str, config: BandConfig) -> "BandImage":
        """Read one band from FITS files."""
        data, header = fits.getdata(config.image, header=True)
        mask = fits.getdata(config.mask).astype(bool) if config.mask else None
        return cls(
            name=name,
            data=data,
            wcs=WCS(header),
            wavelength=config.wavelength,
            pixel_scale=config.pixel_scale,
            psf_fwhm=config.psf_fwhm,
            zeropoint=config.zeropoint,
            mask=mask,
            header=header,
            path=config.image,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def with_data(
        self,
        data: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        wcs: WCS | None = None,
        pixel_scale: float | None = None,
        psf_fwhm: float | None = None,
    ) -> "BandImage":
        """Return a copy with updated image data."""
        return replace(
            self,
            data=data,
            mask=self.mask.copy() if mask is None else mask,
            wcs=self.wcs if wcs is None else wcs,
            pixel_scale=self.pixel_scale if pixel_scale is None else pixel_scale,
            psf_fwhm=self.psf_fwhm if psf_fwhm is None else psf_fwhm,
        )

    def to_surface_brightness(self) -> "BandImage":
        """Return the image in kJy/sr."""
        return self.with_data(adu_to_kjy_sr(self.data, self.zeropoint, self.pixel_scale))

    def convolved_to_fwhm(self, target_fwhm: float) -> "BandImage":
        """Convolve the image to a Gaussian target FWHM in arcsec."""
        if np.isclose(self.psf_fwhm, target_fwhm):
            return self
        if target_fwhm < self.psf_fwhm:
            raise ValueError(f"Target FWHM is narrower than band {self.name!r}")
        kernel = gaussian_beam_kernel(self.pixel_scale, self.psf_fwhm, target_fwhm)
        fill_value = np.nanmedian(self.data[~self.mask])
        data = convolve_fft(
            self.data,
            kernel,
            mask=self.mask,
            boundary="fill",
            fill_value=fill_value,
            preserve_nan=True,
        )
        return self.with_data(data, psf_fwhm=target_fwhm)


class ImageCollection(Mapping[str, BandImage]):
    """Band images keyed by band name."""

    def __init__(self, images: Mapping[str, BandImage]):
        if len(images) < 2:
            raise ValueError("At least two bands are required")
        self._images = dict(images)
        if set(self._images) != {image.name for image in self._images.values()}:
            raise ValueError("Band keys must match BandImage names")

    @classmethod
    def from_config(cls, config: ModelingConfig) -> "ImageCollection":
        """Read configured band images."""
        return cls({name: BandImage.from_config(name, band) for name, band in config.bands.items()})

    def __getitem__(self, name: str) -> BandImage:
        return self._images[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._images)

    def __len__(self) -> int:
        return len(self._images)

    def map(self, function) -> "ImageCollection":
        """Apply a function to every band image."""
        return ImageCollection({name: function(image) for name, image in self.items()})

    def to_surface_brightness(self) -> "ImageCollection":
        """Convert all images to kJy/sr."""
        return self.map(BandImage.to_surface_brightness)

    def combined_mask(self, method: str = "union") -> np.ndarray:
        """Combine band masks by union or intersection."""
        masks = np.stack([image.mask for image in self.values()])
        if method == "union":
            return np.any(masks, axis=0)
        if method == "intersection":
            return np.all(masks, axis=0)
        raise ValueError("method must be 'union' or 'intersection'")

    def aligned(self, reference_band: str) -> "ImageCollection":
        """Reproject all bands onto a reference grid."""
        reference = self[reference_band]
        aligned = {reference_band: reference}
        for name, image in self.items():
            if name == reference_band:
                continue
            if _on_same_grid(image, reference):
                aligned[name] = image
                continue
            data, footprint = reproject_interp(
                (image.data, image.wcs),
                reference.wcs,
                shape_out=reference.shape,
                order="bilinear",
            )
            mask_values, _ = reproject_interp(
                (image.mask.astype(float), image.wcs),
                reference.wcs,
                shape_out=reference.shape,
                order="nearest-neighbor",
            )
            mask = (mask_values > 0.5) | (footprint <= 0) | ~np.isfinite(data)
            aligned[name] = image.with_data(
                data,
                mask=mask,
                wcs=reference.wcs,
                pixel_scale=reference.pixel_scale,
            )
        return ImageCollection({name: aligned[name] for name in self})

    def matched_psfs(self) -> "ImageCollection":
        """Match all bands to the broadest Gaussian PSF."""
        target = max(image.psf_fwhm for image in self.values())
        result = {}
        for name, image in self.items():
            result[name] = image.convolved_to_fwhm(target)
        return ImageCollection(result)

    def convolved_to_fwhm(self, target_fwhm: float) -> "ImageCollection":
        """Convolve all images to a Gaussian target FWHM in arcsec."""
        result = {}
        for name, image in self.items():
            result[name] = image.convolved_to_fwhm(target_fwhm)
        return ImageCollection(result)


def _on_same_grid(image_a: BandImage, image_b: BandImage, tolerance_arcsec: float = 1e-3) -> bool:
    if image_a.shape != image_b.shape:
        return False
    ny, nx = image_a.shape
    pixels = np.array([[0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1]])
    world_a = image_a.wcs.all_pix2world(pixels, 0)
    world_b = image_b.wcs.all_pix2world(pixels, 0)
    coord_a = SkyCoord(world_a[:, 0], world_a[:, 1], unit="deg")
    coord_b = SkyCoord(world_b[:, 0], world_b[:, 1], unit="deg")
    return bool(np.all(coord_a.separation(coord_b).arcsec <= tolerance_arcsec))


__all__ = ["BandImage", "ImageCollection", "adu_to_kjy_sr"]
