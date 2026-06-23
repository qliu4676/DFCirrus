# DFCirrus
Tools for modeling Galactic cirrus in deep wide-field imaging using morphology
and multi-band color constraints. Dragonfly is supported as a built-in preset,
but the modeling configuration is instrument-independent.

- Sky background modelling using Planck thermal dust models.
- Cirrus decomposition using morphology and colors.

### Dependencies

- Python >= 3.10
- Astropy >= 5
- joblib >= 1.2
- maskfill >= 1.1.1
- Matplotlib >= 3.6
- NumPy >= 1.23
- pandas >= 1.5
- Photutils >= 1.9
- PyYAML >= 6
- reproject >= 0.10
- scikit-image >= 0.20
- scikit-learn >= 1.2
- SciPy >= 1.9
- tqdm >= 4.65

`maskfill` is installed by default and is the default infilling backend.
The Python implementation of CloudCovFix is available as an optional backend
from the [`qliu4676/CloudCovFix`](https://github.com/qliu4676/CloudCovFix)
fork of the original julia package [[`andrew-saydjari/CloudCovFix`]](https://github.com/andrew-saydjari/CloudCovFix.jl).

### Installation

```bash
cd <install directory>
git clone https://github.com/qliu4676/DFCirrus.git
cd DFCirrus
pip install -e .
```

To install DFCirrus with the optional CloudCovFix backend:

```bash
pip install -e ".[cloudcovfix]"
```

### Modeling configuration

Start from [`configs/modeling_example.yaml`](configs/modeling_example.yaml).
Load and validate it with:

```python
from dfcirrus.modeling import load_config

config = load_config("modeling.yaml", check_files=True)
```

Run the multi-band model with:

```python
from dfcirrus.modeling import MultiBandModeler

result = MultiBandModeler.from_config("modeling.yaml").run()
```

Each band is calibrated against Planck radiance and transformed to a shared
luminance scale. The combined luminance image is morphology-filtered and then
mapped back to a cirrus model in every configured band.

See [the modeling guide](docs/modeling.md) and the
[Dragonfly modeling notebook](notebooks/dragonfly_gr_modeling.ipynb).
