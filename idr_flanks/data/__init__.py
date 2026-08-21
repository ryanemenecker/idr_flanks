"""Access to data files shipped with :mod:`idr_flanks`."""

from __future__ import annotations

import os
from typing import List

__all__ = ["data_dir", "structure_path", "available_structures"]

_ROOT = os.path.abspath(os.path.dirname(__file__))


def data_dir() -> str:
    """Absolute path to the packaged data directory."""
    return _ROOT


def structure_path(name: str) -> str:
    """Absolute path to a packaged structure file.

    Parameters
    ----------
    name : str
        File name within ``data/structures``, e.g. ``"1ycr.pdb"``.

    Returns
    -------
    str
    """
    path = os.path.join(_ROOT, "structures", name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No packaged structure named {name!r}. "
            f"Available: {available_structures()}"
        )
    return path


def available_structures() -> List[str]:
    """Names of the structure files shipped with the package."""
    d = os.path.join(_ROOT, "structures")
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith((".pdb", ".cif")))
