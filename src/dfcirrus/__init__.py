"""Tools for modeling Galactic cirrus in deep wide-field images."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dfcirrus")
except PackageNotFoundError:  # source-tree import
    __version__ = "0.2.0.dev0"

__all__ = ["__version__"]
