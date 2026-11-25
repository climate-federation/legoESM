"""DCMIP 1-3 rows of the atmosphere matrix (user decision 2026-11-24): the
lat-lon, MPAS and spectral rows transport in flux form on the steady 1-3 air
mass with the discretely consistent air fluxes (``consistent_mass_flux``)
and carry the 1-3 acceptance audit (air-mass closure, uniform tracer,
global tracer mass with the model's air mass, terrain crossing) in their
notes; a failed criterion is marked with its measured number."""

from __future__ import annotations

import re

import jax
import pytest

jax.config.update("jax_enable_x64", True)


def _row(tmp_path, grid, res, vert="hybrid"):
    from scripts.matrix.run_atmosphere_test_matrix import TestCase, run_dcmip_transport
    tc = TestCase("hydrostatic", "dcmip_transport_13", grid, res, vert, 12, 1,
                  {"test_num": 13})
    return run_dcmip_transport(tc, tmp_path, 0.05)


def test_13_matrix_rows_use_the_kent_hybrid_levels():
    """1-3 on lat-lon / MPAS / spectral runs the Kent et al. (2014, App. B)
    hybrid levels; the cube 1-3 row and the 1-1/1-2 rows stay sigma."""
    from scripts.matrix.run_atmosphere_test_matrix import TEST_MATRIX
    vert = {(tc.case, tc.grid_type): tc.vertical_coord for tc in TEST_MATRIX
            if tc.case.startswith("dcmip_transport_")}
    for g in ("latlon", "icosahedral", "spectral"):
        assert vert[("dcmip_transport_13", g)] == "hybrid"
    assert vert[("dcmip_transport_13", "cubed_sphere")] == "sigma"
    assert all(v == "sigma" for (c, _), v in vert.items() if c != "dcmip_transport_13")


def test_vcoord_probe_override_leaves_dcmip_transport_rows(monkeypatch):
    """The #1029 LEGOESM_MATRIX_VCOORD probe override does not relabel the
    DCMIP transport rows (their coordinate is part of the test)."""
    import scripts.matrix.run_atmosphere_test_matrix as M
    monkeypatch.delenv("LEGOESM_MATRIX_VCOORD", raising=False)
    base = {(tc.case, tc.grid_type): tc.vertical_coord for tc in M._build_test_matrix()}
    for v in ("hybrid", "sigma"):
        monkeypatch.setenv("LEGOESM_MATRIX_VCOORD", v)
        got = {(tc.case, tc.grid_type): tc.vertical_coord for tc in M._build_test_matrix()}
        assert all(got[k] == base[k] for k in got if k[0].startswith("dcmip_transport_"))
        assert any(got[k] == v != base[k] for k in got)          # others still overridden


def test_hybrid_refused_outside_the_13_flux_form_rows(tmp_path):
    from scripts.matrix.run_atmosphere_test_matrix import TestCase, run_dcmip_transport
    for case, tn, grid in (("dcmip_transport_11", 11, "latlon"),
                           ("dcmip_transport_13", 13, "cubed_sphere")):
        tc = TestCase("hydrostatic", case, grid, "C6" if grid == "cubed_sphere" else "18x36",
                      "hybrid", 12, 1, {"test_num": tn})
        with pytest.raises(ValueError, match="hybrid"):
            run_dcmip_transport(tc, tmp_path, 0.05)


def test_skip_rows_stay_out_of_cross_grid_comparisons(tmp_path):
    from scripts.matrix.run_atmosphere_test_matrix import _collect_grid_results_atmosphere
    for g, st in (("icosahedral", "SKIP"), ("latlon", "PASS")):
        leaf = tmp_path / g / "r" / "sigma"
        leaf.mkdir(parents=True)
        (leaf / "mean_timeseries.csv").write_text("step,q1_mean\n0,1.0\n")
        (leaf / "results.txt").write_text(f"status: {st}\n")
    got = _collect_grid_results_atmosphere(tmp_path, "sigma")
    assert set(got) == {"latlon"}


@pytest.mark.parametrize("vert", ["hybrid", "sigma"])
@pytest.mark.parametrize("grid,res", [("latlon", "18x36"), ("spectral", "T10"),
                                      ("icosahedral", "ico2")])
def test_13_rows_pass_the_conservation_criteria(tmp_path, grid, res, vert):
    status, _, notes = _row(tmp_path, grid, res, vert)
    assert status == "PASS"
    assert "UNSUPPORTED" not in notes
    drift = float(re.search(r"air-mass closure drift ([0-9.e+-]+)", notes).group(1))
    unif = float(re.search(r"uniform-tracer max\|q-1\| ([0-9.e+-]+)", notes).group(1))
    assert drift <= 1e-6 and unif <= 1e-12, notes
    # tracer mass checked at every diagnostic sample, not only the end
    assert int(re.search(r"peak over (\d+) samples", notes).group(1)) > 2
    peak = float(re.search(r"peak over \d+ samples q\d ([0-9.e+-]+)", notes).group(1))
    assert peak <= 1e-12, notes
    # 0.05 day: the cloud has not reached the mountain: terrain crossing is
    # reported untested, not failed
    assert "CRITERION FAIL" not in notes, notes
    assert "AUDIT INCOMPLETE: terrain crossing untested" in notes, notes


def test_13_measured_criterion_failure_fails_the_row(tmp_path, monkeypatch):
    """A measured acceptance failure turns the row's status to FAIL."""
    import scripts.matrix.run_atmosphere_test_matrix as M
    monkeypatch.setattr(M, "_DCMIP13_UNIFORM_TOL", 0.0)
    status, _, notes = _row(tmp_path, "spectral", "T10")
    assert status == "FAIL"
    assert "CRITERION FAIL: uniform-tracer preservation" in notes
    assert "status: FAIL" in (tmp_path / "results.txt").read_text()


@pytest.mark.slow
def test_13_spectral_full_row_passes_every_criterion(tmp_path):
    """The full 12-day spectral T21 matrix row (the cheapest core at the
    matrix resolution, ~1 min) meets all four criteria, terrain crossing
    included (the lat-lon 72x144 and MPAS ico5 rows take 5-25 min; measured
    in the commit)."""
    from scripts.matrix.run_atmosphere_test_matrix import TestCase, run_dcmip_transport
    tc = TestCase("hydrostatic", "dcmip_transport_13", "spectral", "T21", "hybrid", 12, 1,
                  {"test_num": 13})
    status, _, notes = run_dcmip_transport(tc, tmp_path, 12.0)
    assert status == "PASS", notes
    assert "CRITERION FAIL" not in notes and "AUDIT INCOMPLETE" not in notes, notes
    ratio = float(re.search(r"\(ratio ([-0-9.e+]+);", notes).group(1))
    assert abs(ratio) <= 0.5, notes


def test_13_full_period_without_terrain_crossing_fails(tmp_path, monkeypatch):
    """Over the full 12 days the cloud must have met the terrain: an
    unmeasured crossing then fails the row (forced by an unreachable
    qualifying shift; spectral T10, 144 steps)."""
    import scripts.matrix.run_atmosphere_test_matrix as M
    monkeypatch.setattr(M, "_DCMIP13_TERRAIN_MIN_M", 1.0e9)
    tc = M.TestCase("hydrostatic", "dcmip_transport_13", "spectral", "T10", "hybrid", 12, 1,
                    {"test_num": 13})
    status, _, notes = M.run_dcmip_transport(tc, tmp_path, 12.0)
    assert status == "FAIL", notes
    assert "terrain crossing untested over the full period" in notes
    assert "q1 L1=" in notes                     # 1-3 norms at the full period
