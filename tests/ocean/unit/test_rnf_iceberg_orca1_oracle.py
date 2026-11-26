"""Oracle test: OMIP runoff composition vs NEMO 5.0.1 ORCA1 (tier-3 RNF unit, defect P3).

NEMO ORCA1 reads the Dai-Trenberth file's ``sorunoff`` only (``sn_rnf``;
``ln_rnf_icb`` = namelist_ref ``.false.``, so ``sbcrnf.F90:120-131`` never adds
``sn_i_rnf``).  ``sorunoff`` already contains the iceberg melt, so the loader
must not add ``Icb_flux`` again.  Fixture = the file's three components at 70
cells (40 iceberg, 20 river-only, 10 zero) for the two months NEMO's
``fld_read`` bracketed at kt=17281, plus the ``fdta`` NEMO recorded for them;
provenance in ``rnf_iceberg_composition.json``.  NEMO input is a raw file
read, so the comparison is bitwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.run import run_omip_core2 as R

_FIX = Path(__file__).parent / "fixtures" / "orca1_t3" / "rnf_iceberg_composition.npz"


def _fields():
    d = np.load(_FIX)
    return d, {k: d[k] for k in ("sorunoff", "sornfisf", "Icb_flux")}


def test_runoff_composition_matches_nemo_fdta_bitwise():
    # --isf on (the ORCA1 card: ice-shelf melt applied at depth, as NEMO
    # ln_isf=.true. cn_isfpar_mlt='spe'), so the surface runoff is NEMO's rnf.
    d, f = _fields()
    assert (d["Icb_flux"] > 0).sum() >= 40          # fixture exercises icebergs
    got = R._sum_runoff_components(f, exclude_isf=True)
    np.testing.assert_array_equal(got, d["nemo_fdta"])


def test_iceberg_flux_never_summed():
    d, f = _fields()
    for exclude_isf in (True, False):
        assert "Icb_flux" not in R._runoff_component_vars(exclude_isf)
    got = R._sum_runoff_components(f, exclude_isf=False)
    np.testing.assert_array_equal(got, d["sorunoff"] + d["sornfisf"])
