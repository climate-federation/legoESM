"""NEMO 5.0.1 eORCA1 oracle for the SI3 C-grid metric builder (``si3_orca_cgrid_metrics``).

Fixture ``fixtures/orca1_t4/si3_cgrid_metrics_orca1.{npz,json}``: domain_cfg rows global j 1-4 and 328-331 (all 360
columns) and NEMO's in-memory ``dom_hgr`` metrics on the matching bands of the (364, 335) nn_hls=2 extended grid
(ext rows 0-5 and 329-334), assembled from the 32-rank tier-4 ORCA1_T4DYN records.  Scored: owner cells and
recorded halo cells next to a wet T cell (i cyclic, j not), bitwise, no exclusions.  The builder is row-local except
the fold (top inner rows) and the south copy (bottom inner rows), so the stitched 8-row domain reproduces both bands.
eORCA1 has no wet T cell within 4 rows of the south edge, so the south halo never reaches a wet stencil and is
unscored; its mirror copy (jpfillcopy, lbc_lnk_pt2pt_generic.h90:308-345) is pinned on a synthetic field instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from legoesm.ice.dynamics import si3_orca_cgrid_metrics

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).resolve().parent / "fixtures" / "orca1_t4" / "si3_cgrid_metrics_orca1"
_D = dict(np.load(str(_FIX) + ".npz"))
_IN = ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f")
_AREA = {"area_t": "e1e2t", "area_u": "e1e2u", "area_v": "e1e2v", "area_f": "e1e2f"}


def _bands(metrics):
    """Stitched 8-row domain -> ext rows 0..11; ext 0-5 = south band, ext 6-11 = north band."""
    return {k: np.asarray(getattr(metrics, k)) for k in _IN + tuple(_AREA)}


def test_fixture_provenance():
    prov = json.loads(Path(str(_FIX) + ".json").read_text())
    assert prov["oracle"].startswith("NEMO 5.0.1") and int(_D["scored"].sum()) == prov["n_scored"]
    assert _D["in_e1t"].shape == (8, 360) and _D["scored"].shape == (364, 12)


def test_metrics_match_nemo_bitwise():
    got = _bands(si3_orca_cgrid_metrics({k: _D["in_" + k] for k in _IN}))
    s = _D["scored"]
    for k, g in got.items():
        assert g.shape == (364, 12)
        want = _D["out_" + _AREA.get(k, k)]
        np.testing.assert_array_equal(g[s], want[s], err_msg=k)


def test_south_halo_is_nemo_mirror_copy():
    # jpfillcopy, south side: halo row jj (counted outward) <- inner row jj (counted inward).
    rows = np.arange(1.0, 9.0)[:, None] * np.ones((1, 360))
    m = si3_orca_cgrid_metrics({k: rows for k in _IN})
    e1t = np.asarray(m.e1t)
    np.testing.assert_array_equal(e1t[:, 1], 1.0)
    np.testing.assert_array_equal(e1t[:, 0], 2.0)
    np.testing.assert_array_equal(np.asarray(m.area_t)[:, 0], 4.0)


def test_strait_reduced_surfaces_are_refused():
    with pytest.raises(ValueError, match="e1e2u"):
        si3_orca_cgrid_metrics({**{k: _D["in_" + k] for k in _IN}, "e1e2u": _D["in_e1t"]})
