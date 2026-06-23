# Cirrus color and morphology modeling

DFCirrus calibrates each science band against Planck radiance, removes fitted
offsets, and jointly fits a color model on a shared luminance scale. The
band-derived luminance maps are combined before morphology filtering. The
filtered luminance image is then mapped back to every configured band.

## Python API

```python
from dfcirrus.modeling import MultiBandModeler

modeler = MultiBandModeler.from_config("modeling.yaml")
result = modeler.run()
result.write(modeler.config.run.output_dir)
```

Primary result attributes are:

- `luminance` and `filtered_luminance`;
- `models[band]` and `residuals[band]`;
- `planck_relations` and `color_model.color_relations`;
- `colors`, containing magnitude colors and bootstrap uncertainties;
- `morphology_result`, containing backend diagnostics.

## Morphology backends

RHT is the default backend:

```yaml
morphology:
  backend: rht
  working_pixel_scale: 2.5  # arcsec/pixel
  rht:
    radius: 2               # arcmin
    maskfill: true
    infill_radius: 9        # pixels; positive odd integer
```

The sequential backend applies starlet cleanup to the RHT result:

```yaml
morphology:
  backend: rht_starlet
  working_pixel_scale: 2.5  # arcsec/pixel
  rht:
    radius: 2               # arcmin
    maskfill: true
    infill_radius: 9        # pixels; positive odd integer
  starlet:
    scales: 5
    keep_scales: [2, 3, 4, 5]
    threshold_sigma: 0
    include_coarse: true
```

Starlet scale numbers are one-based. The defaults omit the first detail scale
and retain the remaining scales and coarse residual. With `backend: rht`, the
starlet stage is not run. `maskfill` enables masked-pixel infilling and
`infill_radius` sets the odd-sized pixel window passed to the infilling method.

## Command line

```bash
dfcirrus validate modeling.yaml
dfcirrus run modeling.yaml
```

`dfcirrus run` writes luminance images, one model and residual per band, and
`cirrus_colors.yaml`.

## Legacy API

The two-band functions in `dfcirrus.modeling.pipe` remain available. New
workflows should use `MultiBandModeler`, which avoids pairwise band transforms.
