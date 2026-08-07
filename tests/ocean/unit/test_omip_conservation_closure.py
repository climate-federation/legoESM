"""Direct tests for the canonical OMIP conservation-closure probe.

Per the Phase-0.3 convention in
``docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md`` (#1492) a probe is
only citable if it is committed AND exercised.  These tests pin the two
properties the probe's conclusions rest on:

1. global integrals are AREA-WEIGHTED (an unweighted lat-lon global integral
   over-counts polar cells -- the exact error an earlier inline probe made);
2. the two-cell-sum discriminator returns EXACTLY zero for a conservative
   redistribution and non-zero for a source, because the whole
   "diffusion instability vs source" verdict turns on that.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[3]
           / "scripts" / "validate" / "ocean_fidelity"
           / "omip_conservation_closure.py")


def _load():
    spec = importlib.util.spec_from_file_location("omip_closure", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load()

_DZ = np.array([5.0, 5.0, 10.0, 10.0])


def _grid(nlat=4, nlon=6):
    lat = np.linspace(-75.0, 75.0, nlat)
    return np.broadcast_to(lat[:, None], (nlat, nlon)).copy()


def test_cell_area_carries_the_cos_lat_factor():
    """A polar cell must be smaller than an equatorial one. Without cos(lat)
    every global integral in this probe would over-weight the poles."""
    lat2d = np.array([[0.0, 0.0], [60.0, 60.0]])
    a = probe.cell_area(lat2d)
    assert a[0, 0] > a[1, 0]
    assert a[1, 0] / a[0, 0] == pytest.approx(np.cos(np.deg2rad(60.0)), rel=1e-12)
    assert not np.allclose(a[0, 0], a[1, 0])


def test_global_integral_is_area_weighted_not_a_plain_sum():
    """Same field, two latitudes: the weighted integral must NOT equal the
    unweighted one, or the cos(lat) factor is not reaching the result."""
    lat2d = np.array([[0.0, 0.0], [60.0, 60.0]])
    wet3 = np.ones((2, 2, _DZ.size), dtype=bool)
    field = np.ones((2, 2, _DZ.size))
    area = probe.cell_area(lat2d)
    g = probe.global_integral(field, _DZ, wet3, area)
    plain = float((probe.column_integral(field, _DZ, wet3)).sum())
    assert g != pytest.approx(plain)
    assert g == pytest.approx(float((probe.column_integral(field, _DZ, wet3)
                                     * area).sum()), rel=1e-12)


def test_two_cell_sum_is_exactly_zero_for_a_conservative_swap():
    """Move tracer between two cells conserving q*dz: the discriminator must
    read EXACTLY zero, otherwise it would call a diffusion instability a
    source."""
    T0 = np.zeros((1, 1, _DZ.size))
    T0[0, 0, 2] = 10.0
    T0[0, 0, 3] = 10.0
    T1 = T0.copy()
    # equal thickness (10 m both): move 4 units of q from level 3 to level 2
    T1[0, 0, 2] = 14.0
    T1[0, 0, 3] = 6.0
    r = probe.two_cell_sum_delta(T0, T0, T1, T1, _DZ, 0, 0, 2, 3)
    assert r["T"]["delta"] == pytest.approx(0.0, abs=1e-12)


def test_two_cell_sum_detects_a_source():
    """Add tracer to one cell without removing it elsewhere -> non-zero."""
    T0 = np.zeros((1, 1, _DZ.size))
    T1 = T0.copy()
    T1[0, 0, 2] = 3.0                      # 3 * 10 m = 30
    r = probe.two_cell_sum_delta(T0, T0, T1, T1, _DZ, 0, 0, 2, 3)
    assert r["T"]["delta"] == pytest.approx(30.0, rel=1e-12)


def test_two_cell_sum_respects_unequal_thickness():
    """A swap that conserves the VALUE but not q*dz across cells of different
    thickness is a SOURCE, and must be reported as one."""
    T0 = np.zeros((1, 1, _DZ.size))
    T0[0, 0, 1] = 10.0   # dz = 5
    T1 = T0.copy()
    T1[0, 0, 1] = 0.0
    T1[0, 0, 2] = 10.0   # dz = 10 -> integral doubled
    r = probe.two_cell_sum_delta(T0, T0, T1, T1, _DZ, 0, 0, 1, 2)
    assert r["T"]["delta"] == pytest.approx(50.0, rel=1e-12)


def test_column_integral_ignores_dry_cells():
    wet3 = np.ones((1, 1, _DZ.size), dtype=bool)
    wet3[0, 0, 2:] = False
    field = np.ones((1, 1, _DZ.size))
    assert probe.column_integral(field, _DZ, wet3)[0, 0] == pytest.approx(10.0)


def test_wet_mask_3d_cuts_at_the_local_seafloor():
    land = np.array([[1.0, 0.0]])
    H = np.array([[12.0, 999.0]])
    z_full = -np.array([2.5, 7.5, 15.0, 25.0])
    w = probe.wet_mask_3d(land, H, z_full)
    assert w[0, 0].tolist() == [True, True, False, False]
    assert not w[0, 1].any(), "land column must be dry at every level"


def test_closure_report_flags_a_source_column():
    lat2d = _grid(2, 2)
    wet3 = np.ones((2, 2, _DZ.size), dtype=bool)
    area = probe.cell_area(lat2d)
    T0 = np.full((2, 2, _DZ.size), 5.0)
    T1 = T0.copy()
    T1[1, 1, 0] += 100.0
    # S0 == S1 == T0 (unchanged); only T carries the injected source.
    rep = probe.closure_report(T0, T0, T1, T0, _DZ, wet3, area)
    assert rep["T"]["relative_change"] > 0
    assert rep["T"]["worst_columns"][0]["index"] == [1, 1]
    # S untouched -> exactly conserved, which also proves the two tracers are
    # reported independently rather than sharing a buffer.
    assert rep["S"]["relative_change"] == pytest.approx(0.0, abs=1e-15)
    assert rep["S"]["max_abs_column_delta"] == pytest.approx(0.0, abs=1e-15)


def test_load_dz_ref_rejects_mixed_sign_axis(tmp_path):
    """A stray negative must be REJECTED, not silently abs()'d into a
    plausible positive thickness. The first version of this probe called
    np.abs() before validating, so -5.0 became 5.0 and the positivity guard
    below could never fire — a corrupt axis would have run clean."""
    p = tmp_path / "bad.txt"
    p.write_text("10.0\n-5.0\n10.0\n")
    with pytest.raises(ValueError, match="mixed-sign"):
        probe.load_dz_ref(p)


def test_load_dz_ref_rejects_zero_thickness(tmp_path):
    """All-positive but containing a zero layer: the positivity guard owns
    this one (it is not a mixed-sign case)."""
    p = tmp_path / "zero.txt"
    p.write_text("10.0\n0.0\n10.0\n")
    with pytest.raises(ValueError, match="non-positive"):
        probe.load_dz_ref(p)


def test_load_dz_ref_accepts_all_negative_fesom_style_depths(tmp_path):
    """FESOM writes depths as negatives; those are legitimate and must load."""
    p = tmp_path / "neg.txt"
    p.write_text("\n".join(str(v) for v in [-0.0, -5.0, -10.0, -20.0]))
    np.testing.assert_allclose(probe.load_dz_ref(p), [5.0, 5.0, 10.0])


def test_load_dz_ref_accepts_interfaces_and_thicknesses(tmp_path):
    iface = tmp_path / "iface.txt"
    iface.write_text("\n".join(str(v) for v in [0.0, 5.0, 10.0, 20.0]))
    np.testing.assert_allclose(probe.load_dz_ref(iface), [5.0, 5.0, 10.0])
    thick = tmp_path / "thick.txt"
    thick.write_text("\n".join(str(v) for v in [5.0, 5.0, 10.0]))
    np.testing.assert_allclose(probe.load_dz_ref(thick), [5.0, 5.0, 10.0])
