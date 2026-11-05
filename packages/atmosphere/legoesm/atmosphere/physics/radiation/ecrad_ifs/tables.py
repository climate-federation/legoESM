"""RRTMG k-distribution tables for the ecRad (OpenIFS 49r1) port.

The tables are the arrays the 49r1 Fortran oracle holds after its own RRTMG
setup (SURRTAB, SURRTPK, SURRTRF, RRTM_INIT_140GP, SRTM_INIT: 256 -> 140 LW
and 224 -> 112 SW g-points), exported by
``scripts/validate/oracle/ecrad_ifs/gen_tables49.py`` and vendored as
``data/rrtmg_tables_49r1.npz`` (vote V35; provenance in the sibling JSON).
Nothing here re-derives a table.

Arrays keep the Fortran shape and the Fortran lower bound of every dimension;
:func:`fortran_index` converts a Fortran index to a zero-based one.
"""

import functools
import json
import os

import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "data")
TABLE_FILE = os.path.join(_DATA, "rrtmg_tables_49r1.npz")
PROVENANCE_FILE = os.path.join(_DATA, "rrtmg_tables_49r1.json")


@functools.lru_cache(maxsize=1)
def load_tables():
    """Return {"module.NAME": (array, lower_bounds)} for every exported table."""
    if not os.path.exists(TABLE_FILE):
        raise FileNotFoundError(f"RRTMG table file missing: {TABLE_FILE}")
    with np.load(TABLE_FILE) as z:
        names = [k for k in z.files if not k.startswith("lb:")]
        return {n: (z[n], tuple(int(b) for b in z["lb:" + n])) for n in names}


def table(name):
    """Array for "module.NAME" (e.g. "yoerrta1.KA"); raises KeyError if absent."""
    return load_tables()[name][0]


def lower_bounds(name):
    """Fortran lower bound of each dimension of "module.NAME"."""
    return load_tables()[name][1]


def fortran_index(name, dim, index):
    """Zero-based offset of Fortran ``index`` along dimension ``dim`` of a table."""
    return index - lower_bounds(name)[dim]


def provenance():
    """Provenance record of the vendored table file."""
    with open(PROVENANCE_FILE) as f:
        return json.load(f)
