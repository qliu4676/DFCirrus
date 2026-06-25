"""Configuration for cirrus color and morphology modeling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import warnings

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
    radius: float = 2.0
    response: str = "peak"
    angle_bins: int | str = "auto"
    median_filter_size: int = 3


@dataclass(frozen=True)
class CompactRejectionConfig:
    method: str = "segmentation"
    threshold_sigma: float = 3.0
    quantile_fallback: float = 0.995


@dataclass(frozen=True)
class StarletConfig:
    scales: int = 5
    keep_scales: tuple[int, ...] = (2, 3, 4, 5)
    threshold_sigma: float = 0.0
    include_coarse: bool = True


@dataclass(frozen=True)
class InfillConfig:
    enabled: bool = True
    backend: str = "maskfill"
    maskfill_window_size: int = 9
    patch_size: int = 51
    training_window: int = 129
    conditioning_radius: float = 25.0
    memory_budget_mb: float = 256.0


@dataclass(frozen=True)
class MorphologyConfig:
    backend: str = "rht"
    working_pixel_scale: float = 2.5
    infill: InfillConfig = field(default_factory=InfillConfig)
    rht: RHTConfig = field(default_factory=RHTConfig)
    starlet: StarletConfig = field(default_factory=StarletConfig)
    compact_rejection: CompactRejectionConfig = field(default_factory=CompactRejectionConfig)


@dataclass(frozen=True)
class ColorConfig:
    reference_band: str
    bands: tuple[str, ...]
    relation: str = "linear"
    regression: str = "bisector"
    combine: str = "median"
    bootstrap_samples: int = 200
    fit_sigma_range: tuple[float, float] = (-15.0, 20.0)
    low_dust_quantile: float = 0.0


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
        _require_keys(
            morphology_data,
            {
                "backend", "working_pixel_scale", "infill_enabled",
                "infill_backend", "maskfill_window_size", "patch_size",
                "training_window", "conditioning_radius", "memory_budget_mb",
                "infill", "rht", "starlet", "compact_rejection",
            },
            "morphology",
        )
        infill_data = dict(morphology_data.get("infill", {}))
        _require_keys(
            infill_data,
            {
                "enabled", "backend", "maskfill_window_size", "patch_size",
                "training_window", "conditioning_radius", "memory_budget_mb",
            },
            "morphology.infill",
        )
        rht_data = dict(morphology_data.get("rht", {}))
        _require_keys(
            rht_data,
            {
                "radius", "response", "angle_bins", "median_filter_size",
                "maskfill", "infill_backend", "infill_radius",
            },
            "morphology.rht",
        )
        angle_bins = rht_data.get("angle_bins", "auto")
        if angle_bins != "auto":
            angle_bins = int(angle_bins)
            if angle_bins < 1:
                raise ConfigurationError("morphology.rht.angle_bins must be positive or 'auto'")
        legacy_values = {}
        for old_name, new_name in (
            ("infill_enabled", "enabled"),
            ("infill_backend", "backend"),
            ("maskfill_window_size", "maskfill_window_size"),
            ("patch_size", "patch_size"),
            ("training_window", "training_window"),
            ("conditioning_radius", "conditioning_radius"),
            ("memory_budget_mb", "memory_budget_mb"),
        ):
            if old_name in morphology_data:
                warnings.warn(
                    f"morphology.{old_name} is deprecated; use "
                    f"morphology.infill.{new_name}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                legacy_values[new_name] = morphology_data[old_name]
        for old_name, new_name in (
            ("maskfill", "enabled"),
            ("infill_backend", "backend"),
            ("infill_radius", "maskfill_window_size"),
        ):
            if old_name in rht_data:
                warnings.warn(
                    f"morphology.rht.{old_name} is deprecated; use "
                    f"morphology.infill.{new_name}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                legacy_values[new_name] = rht_data[old_name]

        maskfill_window_size = int(
            legacy_values.get(
                "maskfill_window_size",
                infill_data.get("maskfill_window_size", 9),
            )
        )
        if maskfill_window_size < 1 or maskfill_window_size % 2 == 0:
            raise ConfigurationError(
                "morphology.infill.maskfill_window_size must be a positive odd integer"
            )
        rht = RHTConfig(
            radius=_positive(rht_data.get("radius", 2), "morphology.rht.radius"),
            response=str(rht_data.get("response", "peak")),
            angle_bins=angle_bins,
            median_filter_size=int(rht_data.get("median_filter_size", 3)),
        )
        if rht.response not in {"peak", "percentile"}:
            raise ConfigurationError("morphology.rht.response must be 'peak' or 'percentile'")

        starlet_data = dict(morphology_data.get("starlet", {}))
        _require_keys(
            starlet_data,
            {"scales", "keep_scales", "threshold_sigma", "include_coarse"},
            "morphology.starlet",
        )
        scales = int(starlet_data.get("scales", 5))
        if scales < 1:
            raise ConfigurationError("morphology.starlet.scales must be positive")
        default_keep_scales = tuple(range(2, scales + 1))
        keep_scales = tuple(
            int(value)
            for value in starlet_data.get("keep_scales", default_keep_scales)
        )
        if not keep_scales or min(keep_scales) < 1 or max(keep_scales) > scales:
            raise ConfigurationError("morphology.starlet.keep_scales must be within the decomposition scales")
        threshold_sigma = float(starlet_data.get("threshold_sigma", 0))
        if threshold_sigma < 0:
            raise ConfigurationError("morphology.starlet.threshold_sigma cannot be negative")
        starlet = StarletConfig(
            scales=scales,
            keep_scales=keep_scales,
            threshold_sigma=threshold_sigma,
            include_coarse=bool(starlet_data.get("include_coarse", True)),
        )

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
        if compact.method not in {"segmentation", "quantile"}:
            raise ConfigurationError("compact_rejection.method must be 'segmentation' or 'quantile'")
        infill = InfillConfig(
            enabled=bool(
                legacy_values.get(
                    "enabled",
                    infill_data.get("enabled", masks.infill_small_holes),
                )
            ),
            backend=str(
                legacy_values.get(
                    "backend",
                    infill_data.get("backend", "maskfill"),
                )
            ),
            maskfill_window_size=maskfill_window_size,
            patch_size=int(legacy_values.get("patch_size", infill_data.get("patch_size", 51))),
            training_window=int(
                legacy_values.get("training_window", infill_data.get("training_window", 129))
            ),
            conditioning_radius=_positive(
                legacy_values.get(
                    "conditioning_radius", infill_data.get("conditioning_radius", 25)
                ),
                "morphology.infill.conditioning_radius",
            ),
            memory_budget_mb=_positive(
                legacy_values.get(
                    "memory_budget_mb", infill_data.get("memory_budget_mb", 256)
                ),
                "morphology.infill.memory_budget_mb",
            ),
        )
        morphology = MorphologyConfig(
            backend=str(morphology_data.get("backend", "rht")),
            working_pixel_scale=_positive(morphology_data.get("working_pixel_scale", 2.5), "morphology.working_pixel_scale"),
            infill=infill,
            rht=rht,
            starlet=starlet,
            compact_rejection=compact,
        )
        if morphology.backend not in {"rht", "rht_starlet"}:
            raise ConfigurationError("morphology.backend must be 'rht' or 'rht_starlet'")
        if infill.backend not in {"maskfill", "cloudcovfix"}:
            raise ConfigurationError(
                "morphology.infill.backend must be 'maskfill' or 'cloudcovfix'"
            )
        if infill.patch_size < 3 or infill.patch_size % 2 == 0:
            raise ConfigurationError(
                "morphology.infill.patch_size must be an odd integer of at least 3"
            )
        if infill.training_window < 1 or infill.training_window % 2 == 0:
            raise ConfigurationError(
                "morphology.infill.training_window must be a positive odd integer"
            )
        if infill.training_window < infill.patch_size:
            raise ConfigurationError(
                "morphology.infill.training_window must be at least patch_size"
            )
        if infill.conditioning_radius > infill.patch_size // 2:
            raise ConfigurationError(
                "morphology.infill.conditioning_radius cannot exceed half the patch size"
            )

        color_data = dict(data.get("color", {}))
        _require_keys(
            color_data,
            {"reference_band", "bands", "relation", "regression", "combine", "bootstrap_samples", "fit_sigma_range", "low_dust_quantile"},
            "color",
        )
        color_bands = tuple(str(name) for name in color_data.get("bands", bands))
        missing_color_bands = set(color_bands) - set(bands)
        if missing_color_bands:
            raise ConfigurationError(f"color.bands contains unknown bands: {sorted(missing_color_bands)}")
        if set(color_bands) != set(bands):
            raise ConfigurationError("color.bands must include all configured bands")
        color_reference = str(color_data.get("reference_band", reference_band))
        if color_reference not in color_bands:
            raise ConfigurationError("color.reference_band must be included in color.bands")
        sigma_range = tuple(float(value) for value in color_data.get("fit_sigma_range", (-15, 20)))
        if len(sigma_range) != 2 or sigma_range[0] >= sigma_range[1]:
            raise ConfigurationError("color.fit_sigma_range must contain increasing lower and upper values")
        low_dust_quantile = float(color_data.get("low_dust_quantile", 0.0))
        if not 0.0 <= low_dust_quantile < 1.0:
            raise ConfigurationError("color.low_dust_quantile must be in [0, 1)")
        color = ColorConfig(
            reference_band=color_reference,
            bands=color_bands,
            relation=str(color_data.get("relation", "linear")),
            regression=str(color_data.get("regression", "bisector")),
            combine=str(color_data.get("combine", "median")),
            bootstrap_samples=int(color_data.get("bootstrap_samples", 200)),
            fit_sigma_range=sigma_range,
            low_dust_quantile=low_dust_quantile,
        )
        if color.relation != "linear":
            raise ConfigurationError("color.relation must be 'linear'")
        if color.regression not in {"linear", "bisector"}:
            raise ConfigurationError("color.regression must be 'linear' or 'bisector'")
        if color.combine not in {"mean", "median"}:
            raise ConfigurationError("color.combine must be 'mean' or 'median'")
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
    "BandConfig", "ColorConfig", "ConfigurationError", "InfillConfig", "ModelingConfig",
    "MorphologyConfig", "default_planck_path", "load_config",
]
