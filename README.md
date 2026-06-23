# DFCirrus
Tools for modeling Galactic cirrus in deep wide-field imaging using morphology
and multi-band color constraints. Dragonfly is supported as a built-in preset,
but the modeling configuration is instrument-independent.

- Sky background modelling using Planck thermal dust models.
- Cirrus decomposition using morphology and colors.

### Dependencies

- Python>=3.12.12
- matplotlib
- numpy>=2.4.4
- scipy>=1.6.0
- astropy>=7.2.0
- photutils>=3.0.0
- scikit-learn>=0.24.2
- scikit-image>=0.19.2
- reproject>=0.9
- tqdm
- joblib


### Installation

```bash
cd <install directory>
git clone https://github.com/qliu4676/DFCirrus.git
cd DFCirrus
pip install -e .
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
[Dragonfly g/r notebook](notebooks/dragonfly_gr_modeling.ipynb).
