"""DCMIP 1-3 rows of the atmosphere matrix (codex-voted user decision
2026-11-24): the MPAS row is reported UNSUPPORTED (SKIP) with its run kept
only as the labelled sigma-following variant; the lat-lon / spectral rows
carry the 1-3 acceptance audit (air-mass closure, uniform tracer, global
tracer mass with the 1-3 air mass, terrain crossing) in their notes, and a
failed criterion is marked with its measured number."""

from __future__ import annotations

import re

import jax
import pytest

jax.config.update("jax_enable_x64", True)


def _row(tmp_path, grid, res):
    from scripts.matrix.run_atmosphere_test_matrix import TestCase, run_dcmip_transport
    tc = TestCase("hydrostatic", "dcmip_transport_13", grid, res, "sigma", 12, 1,
                  {"test_num": 13})
    return run_dcmip_transport(tc, tmp_path, 0.05)


def test_skip_rows_stay_out_of_cross_grid_comparisons(tmp_path):
    from scripts.matrix.run_atmosphere_test_matrix import _collect_grid_results_atmosphere
    for g, st in (("icosahedral", "SKIP"), ("latlon", "PASS")):
        leaf = tmp_path / g / "r" / "sigma"
        leaf.mkdir(parents=True)
        (leaf / "mean_timeseries.csv").write_text("step,q1_mean\n0,1.0\n")
        (leaf / "results.txt").write_text(f"status: {st}\n")
    got = _collect_grid_results_atmosphere(tmp_path, "sigma")
    assert set(got) == {"latlon"}


def test_mpas_13_row_is_unsupported_with_labelled_variant(tmp_path):
    status, _, notes = _row(tmp_path, "icosahedral", "ico2")
    assert status == "SKIP"
    assert notes.startswith("UNSUPPORTED as DCMIP 1-3")
    assert "1-3 sigma-following variant" in notes and "not comparable" in notes
    assert "status: SKIP" in (tmp_path / "results.txt").read_text()
    assert "CRITERION FAIL" not in notes          # criteria gate lat-lon/spectral only


@pytest.mark.parametrize("grid,res", [("latlon", "18x36"), ("spectral", "T10")])
def test_latlon_spectral_13_rows_carry_the_audit(tmp_path, grid, res):
    status, _, notes = _row(tmp_path, grid, res)
    assert status == "PASS"
    for key in ("air-mass closure drift", "uniform-tracer max|q-1|",
                "mass[q p_s(z_s) dsigma] rel", "terrain"):
        assert key in notes, (key, notes)
    # tracer mass checked at every diagnostic sample, not only the end
    assert int(re.search(r"peak over (\d+) samples", notes).group(1)) > 2
    drift = float(re.search(r"air-mass closure drift ([0-9.e+-]+)", notes).group(1))
    # the analytic sigma_dot does not close the core's discrete continuity
    # (measured at 0.05 day: 0.106 lat-lon 18x36, 0.076 spectral T10): marked failed
    assert drift > 1e-6 and "CRITERION FAIL: local air-mass closure" in notes
