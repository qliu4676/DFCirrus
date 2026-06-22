"""Configuration for cirrus color and morphology modeling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when a modeling configuration is invalid."""


def _require_keys(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ConfigurationError(f"Unknown {context} option(s): {names}")


def _positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if result <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


@dataclass(frozen=True)
class BandConfig:
    """One science band; wavelength is Angstrom and angular values are arcsec."""

    image: Path
    wavelength: float
    zeropoint: float
    pixel_scale: float
    psf_fwhm: float
    mask: Path | None = None

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any], base_dir: Path) -> "BandConfig":
        allowed = {"image", "mask", "wavelength", "zeropoint", "pixel_scale", "psf_fwhm"}
        _require_keys(data, allowed, f"bands.{name}")
        required = allowed - {"mask"}
        missing = sorted(key for key in required if data.get(key) is None)
        if missing:
            raise ConfigurationError(f"bands.{name} is missing: {', '.join(missing)}")
        return cls(
            image=_resolve_path(str(data["image"]), base_dir),
            mask=_resolve_path(data.get("mask"), base_dir),
            wavelength=_positive(data["wavelength"], f"bands.{name}.wavelength"),
            zeropoint=float(data["zeropoint"]),
            pixel_scale=_positive(data["pixel_scale"], f"bands.{name}.pixel_scale"),
            psf_fwhm=_positive(data["psf_fwhm"], f"bands.{name}.psf_fwhm"),
        )


@dataclass(frozen=True)
class RunConfig:
    name: str = "cirrus"
    output_dir: Path = Path("outputs")
    overwrite: bool = False
    random_seed: int = 12345


@dataclass(frozen=True)
class MaskConfig:
    combine: str = "union"
    source_threshold_sigma: float = 3.0
    dilation_radius: float = 10.0
    infill_small_holes: bool = True
    maximum_infill_radius: float = 10.0


@dataclass(frozen=True)
class RHTConfig:
    radius: float = 3.0
    response: str = "peak"
    angle_bins: int | str = "auto"
    median_filter_size: int = 3


@dataclass(frozen=True)
class CompactRejectionConfig:
    method: str = "segmentation"
    threshold_sigma: float = 3.0
    quantile_fallback: float = 0.995


@dataclass(frozen=True)
class MorphologyConfig:
    backend: str = "rht"
    working_pixel_scale: float = 10.0
    rht: RHTConfig = field(default_factory=RHTConfig)
    compact_rejection: CompactRejectionConfig = field(default_factory=CompactRejectionConfig)


@dataclass(frozen=True)
class ColorConfig:
    reference_band: str
    bands: tuple[str, ...]
    relation: str = "linear"
    regression: str = "bisector"
    bootstrap_samples: int = 200
    fit_sigma_range: tuple[float, float] = (-15.0, 20.0)


@dataclass(frozen=True)
class DiagnosticsConfig:
    enabled: bool = True
    save_intermediate_fits: bool = True
    save_plots: bool = True
    log_level: str = "INFO"


@dataclass(frozen=True)
class ModelingConfig:
    """Resolved modeling configuration."""

    bands: dict[str, BandConfig]
    planck_radiance: Path
    run: RunConfig
    reference_band: str
    psf_match: bool
    masks: MaskConfig
    morphology: MorphologyConfig
    color: ColorConfig
    diagnostics: DiagnosticsConfig
    preset: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], base_dir: Path | str = ".") -> "ModelingConfig":
        base_dir = Path(base_dir).resolve()
        allowed = {
            "preset", "run", "bands", "planck_radiance", "alignment",
            "masks", "morphology", "color", "diagnostics",
        }
        _require_keys(data, allowed, "top-level")

        bands_data = data.get("bands")
        if not isinstance(bands_data, Mapping) or len(bands_data) < 2:
            raise ConfigurationError("bands must define at least two science bands")
        bands = {
            str(name): BandConfig.from_dict(str(name), values, base_dir)
            for name, values in bands_data.items()
        }

        run_data = dict(data.get("run", {}))
        _require_keys(run_data, {"name", "output_dir", "overwrite", "random_seed"}, "run")
        run = RunConfig(
            name=str(run_data.get("name", "cirrus")),
            output_dir=_resolve_path(run_data.get("output_dir", "outputs"), base_dir),
            overwrite=bool(run_data.get("overwrite", False)),
            random_seed=int(run_data.get("random_seed", 12345)),
        )

        planck_data = data.get("planck_radiance")
        if not isinstance(planck_data, Mapping):
            raise ConfigurationError("planck_radiance must contain a path")
        _require_keys(planck_data, {"path"}, "planck_radiance")
        if not planck_data.get("path"):
            raise ConfigurationError("planck_radiance.path is required")
        planck_path = _resolve_path(str(planck_data["path"]), base_dir)

        alignment = dict(data.get("alignment", {}))
        _require_keys(alignment, {"reference_band", "psf_match"}, "alignment")
        reference_band = str(alignment.get("reference_band", next(iter(bands))))
        if reference_band not in bands:
            raise ConfigurationError("alignment.reference_band must name a configured band")

        masks_data = dict(data.get("masks", {}))
        _require_keys(
            masks_data,
            {"combine", "source_threshold_sigma", "dilation_radius", "infill_small_holes", "maximum_infill_radius"},
            "masks",
        )
        masks = MaskConfig(
            combine=str(masks_data.get("combine", "union")),
            source_threshold_sigma=_positive(masks_data.get("source_threshold_sigma", 3), "masks.source_threshold_sigma"),
            dilation_radius=_positive(masks_data.get("dilation_radius", 10), "masks.dilation_radius"),
            infill_small_holes=bool(masks_data.get("infill_small_holes", True)),
            maximum_infill_radius=_positive(masks_data.get("maximum_infill_radius", 10), "masks.maximum_infill_radius"),
        )
        if masks.combine not in {"union", "intersection"}:
            raise ConfigurationError("masks.combine must be 'union' or 'intersection'")

        morphology_data = dict(data.get("morphology", {}))
        _require_keys(morphology_data, {"backend", "working_pixel_scale", "rht", "compact_rejection"}, "morphology")
        rht_data = dict(morphology_data.get("rht", {}))
        _require_keys(rht_data, {"radius", "response", "angle_bins", "median_filter_size"}, "morphology.rht")
        angle_bins = rht_data.get("angle_bins", "auto")
        if angle_bins != "auto":
            angle_bins = int(angle_bins)
            if angle_bins < 1:
                raise ConfigurationError("morphology.rht.angle_bins must be positive or 'auto'")
        rht = RHTConfig(
            radius=_positive(rht_data.get("radius", 3), "morphology.rht.radius"),
            response=str(rht_data.get("response", "peak")),
            angle_bins=angle_bins,
            median_filter_size=int(rht_data.get("median_filter_size", 3)),
        )
        if rht.response not in {"peak", "percentile"}:
            raise ConfigurationError("morphology.rht.response must be 'peak' or 'percentile'")

        compact_data = dict(morphology_data.get("compact_rejection", {}))
        _require_keys(compact_data, {"method", "threshold_sigma", "quantile_fallback"}, "morphology.compact_rejection")
        quantile = float(compact_data.get("quantile_fallback", 0.995))
        if not 0 < quantile < 1:
            raise ConfigurationError("compact_rejection.quantile_fallback must be between 0 and 1")
        compact = CompactRejectionConfig(
            method=str(compact_data.get("method", "segmentation")),
            threshold_sigma=_positive(compact_data.get("threshold_sigma", 3), "compact_rejection.threshold_sigma"),
            quantile_fallback=quantile,
        )
        morphology = MorphologyConfig(
            backend=str(morphology_data.get("backend", "rht")),
            working_pixel_scale=_positive(morphology_data.get("working_pixel_scale", 10), "morphology.working_pixel_scale"),
            rht=rht,
            compact_rejection=compact,
        )
        if morphology.backend != "rht":
            raise ConfigurationError("morphology.backend must be 'rht'")

        color_data = dict(data.get("color", {}))
        _require_keys(color_data, {"reference_band", "bands", "relation", "regression", "bootstrap_samples", "fit_sigma_range"}, "color")
        color_bands = tuple(str(name) for name in color_data.get("bands", bands))
        missing_color_bands = set(color_bands) - set(bands)
        if missing_color_bands:
            raise ConfigurationError(f"color.bands contains unknown bands: {sorted(missing_color_bands)}")
        color_reference = str(color_data.get("reference_band", reference_band))
        if color_reference not in color_bands:
            raise ConfigurationError("color.reference_band must be included in color.bands")
        sigma_range = tuple(float(value) for value in color_data.get("fit_sigma_range", (-15, 20)))
        if len(sigma_range) != 2 or sigma_range[0] >= sigma_range[1]:
            raise ConfigurationError("color.fit_sigma_range must contain increasing lower and upper values")
        color = ColorConfig(
            reference_band=color_reference,
            bands=color_bands,
            relation=str(color_data.get("relation", "linear")),
            regression=str(color_data.get("regression", "bisector")),
            bootstrap_samples=int(color_data.get("bootstrap_samples", 200)),
            fit_sigma_range=sigma_range,
        )
        if color.bootstrap_samples < 0:
            raise ConfigurationError("color.bootstrap_samples cannot be negative")

        diagnostics_data = dict(data.get("diagnostics", {}))
        _require_keys(diagnostics_data, {"enabled", "save_intermediate_fits", "save_plots", "log_level"}, "diagnostics")
        diagnostics = DiagnosticsConfig(**diagnostics_data)

        return cls(
            bands=bands,
            planck_radiance=planck_path,
            run=run,
            reference_band=reference_band,
            psf_match=bool(alignment.get("psf_match", True)),
            masks=masks,
            morphology=morphology,
            color=color,
            diagnostics=diagnostics,
            preset=data.get("preset"),
        )

    def validate_files(self) -> None:
        """Check all configured input paths without opening the FITS files."""
        paths = {"planck_radiance.path": self.planck_radiance}
        for name, band in self.bands.items():
            paths[f"bands.{name}.image"] = band.image
            if band.mask is not None:
                paths[f"bands.{name}.mask"] = band.mask
        missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Configured input file(s) not found:\n" + "\n".join(missing))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = yaml.safe_load(stream) or {}
    if not isinstance(result, dict):
        raise ConfigurationError(f"{path} must contain a YAML mapping")
    return result


def _preset_data(name: str) -> dict[str, Any]:
    if name != "dragonfly":
        raise ConfigurationError(f"Unknown preset: {name}")
    resource = files("dfcirrus.presets").joinpath("dragonfly.yaml")
    with resource.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_config(filename: str | Path, *, check_files: bool = False) -> ModelingConfig:
    """Load a YAML file, merge its optional preset, and validate it."""
    path = Path(filename).expanduser().resolve()
    user_data = _load_yaml(path)
    preset = user_data.get("preset")
    resolved = _deep_merge(_preset_data(str(preset)), user_data) if preset else user_data
    config = ModelingConfig.from_dict(resolved, base_dir=path.parent)
    if check_files:
        config.validate_files()
    return config


def default_planck_path() -> str:
    """Return the default Planck radiance map path."""
    return str(_preset_data("dragonfly")["planck_radiance"]["path"])


__all__ = [
    "BandConfig", "ColorConfig", "ConfigurationError", "ModelingConfig",
    "MorphologyConfig", "default_planck_path", "load_config",
]
