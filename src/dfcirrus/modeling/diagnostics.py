"""Diagnostics for multi-band modeling."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np


def plot_planck_fits(result, *, max_points: int = 30000):
    """Plot each band against Planck radiance."""
    bands = tuple(result.planck_relations)
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 4), squeeze=False)
    for axis, name in zip(axes[0], bands):
        x = result.planck_map
        y = result.planck_images[name].data
        valid = ~result.fit_mask & np.isfinite(x) & np.isfinite(y)
        xs, ys = _sample_pair(x[valid], y[valid], max_points)
        axis.scatter(xs, ys, s=2, alpha=0.15, rasterized=True)
        relation = result.planck_relations[name]
        grid = np.linspace(np.nanmin(xs), np.nanmax(xs), 200)
        axis.plot(grid, relation.predict(grid), color="tab:red", lw=2)
        axis.set(
            title=f"{name}: $R^2$={relation.r_squared:.3f}",
            xlabel=r"Planck radiance $\times 10^7$",
            ylabel=f"{name} [kJy/sr]",
        )
    fig.tight_layout()
    return fig, axes


def plot_interband_fits(result, *, max_points: int = 30000):
    """Plot each band against the shared luminance image."""
    bands = result.color_model.bands
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 4), squeeze=False)
    for axis, name in zip(axes[0], bands):
        planck_relation = result.planck_relations[name]
        relation = result.color_model.color_relations[name]
        x = result.luminance
        y = result.images[name].data - planck_relation.intercept
        valid = ~result.fit_mask & np.isfinite(x) & np.isfinite(y)
        xs, ys = _sample_pair(x[valid], y[valid], max_points)
        axis.scatter(xs, ys, s=2, alpha=0.15, rasterized=True)
        grid = np.linspace(np.nanmin(xs), np.nanmax(xs), 200)
        axis.plot(grid, relation.predict(grid), color="tab:green", lw=2)
        axis.set(
            title=f"{name}: $R^2$={relation.r_squared:.3f}",
            xlabel="Luminance [reference-band scale]",
            ylabel=f"Calibrated {name} [kJy/sr]",
        )
    fig.tight_layout()
    return fig, axes


def plot_morphology(result, *, max_panels: int = 6, cmap: str = "gray_r"):
    """Plot morphology backend intermediates."""
    morphology = result.morphology_result
    if morphology is None:
        raise ValueError("No morphology diagnostics are available")
    components = morphology.components or {"filtered": morphology.image}
    preferred = ("input", "smoothed", "response", "residual", "compact_mask", "filtered")
    names = [name for name in preferred if name in components]
    names.extend(name for name in components if name not in names)
    names = names[:max_panels]
    columns = min(3, len(names))
    rows = math.ceil(len(names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for axis, name in zip(axes.ravel(), names):
        image = components[name]
        if image.dtype == bool or name.endswith("mask"):
            axis.imshow(image, origin="lower", cmap="gray")
        else:
            finite = image[np.isfinite(image)]
            limits = np.nanpercentile(finite, (1, 99)) if finite.size else (0, 1)
            axis.imshow(image, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_title(name.replace("_", " ").title())
        axis.set_axis_off()
    for axis in axes.ravel()[len(names):]:
        axis.set_visible(False)
    fig.suptitle(f"Morphology backend: {morphology.backend}")
    fig.tight_layout()
    return fig, axes


def plot_model_images(result, *, cmap: str = "gray_r"):
    """Plot luminance and per-band cirrus models."""
    images = {"luminance": result.luminance, **result.models}
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 4), squeeze=False)
    for axis, (name, image) in zip(axes[0], images.items()):
        finite = image[np.isfinite(image)]
        limits = np.nanpercentile(finite, (1, 99)) if finite.size else (0, 1)
        axis.imshow(image, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_title(name)
        axis.set_axis_off()
    fig.tight_layout()
    return fig, axes


def _sample_pair(x, y, max_points):
    if x.size <= max_points:
        return x, y
    rng = np.random.default_rng(0)
    selected = rng.choice(x.size, max_points, replace=False)
    return x[selected], y[selected]


__all__ = [
    "plot_interband_fits", "plot_model_images", "plot_morphology",
    "plot_planck_fits",
]
