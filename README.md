# DFCirrus
Tools for modeling Galactic cirrus in deep wide-field imaging using morphology
and multi-band color constraints. Dragonfly is supported as a built-in preset,
but the modeling configuration is instrument-independent.

- Sky background modelling using Planck thermal dust models.
- Cirrus decomposition using morphology and colors.

### Dependencies

- matplotlib
- numpy
- scipy>=1.6.0
- astropy>=4.0
- photutils>=0.7.2
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
