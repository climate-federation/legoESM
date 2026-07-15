"""Phase 3 of the FV3-native cubed-sphere effort: exact duo-grid halo tables.

Pins ``legoesm.grids.fv3_native_halos.compute_fv3_native_k2e`` — the ED
kinked-to-extended Lagrange remap tables for ALL six staggers (A, B, CX, CY,
DX, DY) — record-by-record against the VERBATIM duo-grid Fortran oracle
(``scripts/validate/fv3_native/gen_duogrid_oracle.sh``; reference mirror
``luanfs/FV3_container`` @ 7d06431e — re-pin against the authoritative
``atmos_cubed_sphere-symmetryclean`` tree if they ever diverge).

Also pins that the legacy equiangular duogrid tables
(:mod:`legoesm.grids.duogrid`) genuinely DIFFER from the ED-native tables
(the phase-3 finding: legoESM's duogrid was built on equiangular extension
lines, so its coefficients are not the FV3 duo-grid's on the ED path).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.fv3_native_halos import compute_fv3_native_k2e  # noqa: E402

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "fv3_duogrid_oracle.npz"

_STAGGERS = ("A", "B", "CX", "CY", "DX", "DY")


class TestK2EOracle:
    @pytest.mark.parametrize("res", [12, 24])
    def test_all_stagger_tables_match_fortran(self, res):
        d = np.load(_FIXTURE)
        tab = compute_fv3_native_k2e(res, remap_ng=3, k2e_nord=4)
        assert int(d[f"k2e_nord_c{res}"]) == 4
        for stag in _STAGGERS:
            ij_ref = d[f"k2e_{stag}_ij_c{res}"]
            loc_ref = d[f"k2e_{stag}_loc_c{res}"]
            coef_ref = d[f"k2e_{stag}_coef_c{res}"]
            ij = tab[f"{stag}_ij"]
            loc = tab[f"{stag}_loc"]
            coef = tab[f"{stag}_coef"]
            assert ij.shape == ij_ref.shape, (
                f"{stag}: record set {ij.shape} vs oracle {ij_ref.shape}")
            assert np.array_equal(ij, ij_ref), f"{stag}: record keys differ"
            assert np.array_equal(loc, loc_ref), f"{stag}: klo anchors differ"
            dmax = np.abs(coef - coef_ref).max()
            # 5e-13: the handful of extreme corner-window records whose
            # Lagrange window includes upstream's ±999 sentinel node (weight
            # ~1e-11) accumulate a few e-13 of sentinel-arithmetic roundoff;
            # all regular records agree to ~1e-15.
            assert dmax < 5e-13, f"{stag}: coef max dev {dmax:.3e}"

    def test_partition_of_unity(self):
        tab = compute_fv3_native_k2e(12)
        for stag in _STAGGERS:
            s = tab[f"{stag}_coef"].sum(axis=1)
            assert np.abs(s - 1.0).max() < 1e-12, stag

    def test_legacy_equiangular_tables_differ(self):
        # The phase-3 finding, pinned: the legacy duogrid (equiangular
        # extension lines) is NOT the FV3 duo-grid table set on ED.
        from legoesm.grids.duogrid import create_duogrid_data

        tab = compute_fv3_native_k2e(12)
        dg = create_duogrid_data(12, ng=3, k2e_nord=4)
        legacy = np.asarray(dg.k2e_coef)[0, 0, 0]      # face 0, edge 0, d 0
        # native A-table depth-1 south row, interior window i=1..12, j=0
        ij = tab["A_ij"]
        sel = (ij[:, 1] == 0) & (ij[:, 0] >= 1) & (ij[:, 0] <= 12)
        native = tab["A_coef"][sel]
        assert legacy.shape[0] == native.shape[0]
        assert np.abs(legacy[:, :4] - native).max() > 1e-3
