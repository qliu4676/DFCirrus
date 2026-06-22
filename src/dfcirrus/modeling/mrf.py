"""Legacy imports for the MRF integration."""

from __future__ import annotations

import warnings


def _warn() -> None:
    warnings.warn(
        "dfcirrus.modeling.mrf is deprecated; use dfcirrus.mrf.",
        DeprecationWarning,
        stacklevel=3,
    )


def run_mrf(*args, **kwargs):
    _warn()
    from dfcirrus.mrf import run_mrf as implementation

    return implementation(*args, **kwargs)


def run_wide_subbright(*args, **kwargs):
    _warn()
    from dfcirrus.mrf import run_wide_subbright as implementation

    return implementation(*args, **kwargs)


class MRF_Result:
    def __new__(cls, *args, **kwargs):
        _warn()
        from dfcirrus.mrf import MRF_Result as implementation

        return implementation(*args, **kwargs)


__all__ = ["MRF_Result", "run_mrf", "run_wide_subbright"]
