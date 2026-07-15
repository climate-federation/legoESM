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

    def test_ed_grid_pairs_with_native_duogrid(self, monkeypatch):
        # ED + use_duogrid must take the ED-native tables; equiangular +
        # use_duogrid must keep the legacy builder (spy proves dispatch).
        import legoesm.grids.duogrid as legacy
        from legoesm.grids.cubed_sphere import create_cubed_sphere

        calls = {"legacy": 0}
        orig = legacy.create_duogrid_data

        def _spy(*a, **k):
            calls["legacy"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(legacy, "create_duogrid_data", _spy)
        g_eq = create_cubed_sphere(8, dtype=np.float64, use_duogrid=True)
        assert calls["legacy"] == 1
        assert g_eq.duogrid is not None
        calls["legacy"] = 0
        g_ed = create_cubed_sphere(8, dtype=np.float64, gnomonic="ed",
                                   use_duogrid=True)
        assert calls["legacy"] == 0
        assert g_ed.duogrid is not None
        # native tables genuinely different from what legacy would build
        dg_legacy = orig(8, ng=3, k2e_nord=4)
        assert np.abs(np.asarray(g_ed.duogrid.k2e_coef)
                      - np.asarray(dg_legacy.k2e_coef)).max() > 1e-3

    def test_native_duogrid_fails_loudly_on_unsupported(self):
        from legoesm.grids.fv3_native_halos import (
            create_fv3_native_duogrid_data,
        )

        with pytest.raises(NotImplementedError, match="k2e_nord"):
            create_fv3_native_duogrid_data(12, ng=3, k2e_nord=2)
        with pytest.raises(NotImplementedError, match="ng="):
            create_fv3_native_duogrid_data(12, ng=4)


class TestGhostValues:
    """Functional certification of the applied halos on the ED path."""

    def _pad(self, n, field_fn, halo):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.halo import pad_halo

        g = create_cubed_sphere(n, dtype=np.float64, gnomonic="ed",
                                use_duogrid=True, duogrid_ng=3)
        f = field_fn(np.asarray(g.lon), np.asarray(g.lat))
        padded = np.asarray(pad_halo(jax.numpy.asarray(f), halo=halo,
                                     duogrid=g.duogrid))
        return g, padded

    @pytest.mark.parametrize("halo", [1, 2, 3])
    def test_constant_field_ghosts_exact(self, halo):
        # partition of unity + exact corner fill => constant survives
        g, padded = self._pad(12, lambda lo, la: np.ones_like(lo), halo)
        assert np.abs(padded - 1.0).max() < 1e-12

    @staticmethod
    def _regions(n, h):
        m = n + 2 * h
        corner = np.zeros((m, m), bool)
        corner[:h, :h] = corner[:h, -h:] = True
        corner[-h:, :h] = corner[-h:, -h:] = True
        return corner

    @pytest.mark.parametrize("halo", [1, 2, 3])
    def test_smooth_field_ghosts_match_extension_positions(self, halo):
        # duo-grid ghosts approximate the analytic field AT THE EXTENDED
        # (own-coordinate-continuation) positions — the whole point of the
        # duo grid.  EDGE rings are the certified k2e remap: 4th-order,
        # gated tight (measured <=3.2e-3 at C12 across depths 1-3, an 8x
        # improvement over the legacy equiangular tables on the same test).
        # CORNER blocks are the upstream fill_corner_region algorithm's own
        # one-sided Lagrange extrapolation, whose ED ghost-line spacing
        # stretches toward the tips: at C12 the outer diagonal tip carries
        # O(0.3) error BY CONSTRUCTION (the legacy path shows the same
        # class, 0.17), converging ~4th order (test below) — gate loose.
        def field(lo, la):
            return np.sin(la) * np.cos(2.0 * lo) + 0.5 * np.cos(la) ** 2

        g, padded = self._pad(12, field, halo)
        dgd = g.duogrid
        ext_lon = np.asarray(dgd.ext_lon)
        ext_lat = np.asarray(dgd.ext_lat)
        ng = int(dgd.ng)
        n = int(dgd.n)
        h = halo
        sl = slice(ng - h, ng + n + h)
        ref = field(ext_lon[:, sl, sl], ext_lat[:, sl, sl])
        err = np.abs(padded - ref)
        corner = self._regions(n, h)
        edge_err = err[:, ~corner].max()
        corner_err = err[:, corner].max() if corner.any() else 0.0
        assert edge_err < 5e-3, f"halo={halo}: edge ghost error {edge_err:.3e}"
        assert corner_err < 0.5, (
            f"halo={halo}: corner ghost error {corner_err:.3e}")

    def test_fourth_order_convergence(self):
        def field(lo, la):
            return np.sin(la) * np.cos(2.0 * lo) + 0.5 * np.cos(la) ** 2

        errs_edge, errs_corner = {}, {}
        for n in (12, 24):
            g, padded = self._pad(n, field, 3)
            dgd = g.duogrid
            ext_lon = np.asarray(dgd.ext_lon)
            ext_lat = np.asarray(dgd.ext_lat)
            ng = int(dgd.ng)
            sl = slice(ng - 3, ng + n + 3)
            ref = field(ext_lon[:, sl, sl], ext_lat[:, sl, sl])
            err = np.abs(padded - ref)
            corner = self._regions(n, 3)
            errs_edge[n] = err[:, ~corner].max()
            errs_corner[n] = err[:, corner].max()
        # 4th order => ~16x; slack for metric variation
        r_edge = errs_edge[12] / errs_edge[24]
        r_corner = errs_corner[12] / errs_corner[24]
        assert r_edge > 6.0, f"edge convergence {r_edge:.2f} ({errs_edge})"
        assert r_corner > 6.0, (
            f"corner convergence {r_corner:.2f} ({errs_corner})")


class TestLegacyDistinct:
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
