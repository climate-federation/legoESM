"""#1455 regional_audit: the arithmetic and the refusals the probe itself owns.

Everything this probe reports is reduced by an imported harness; what it owns is
the six-band PARTITION, the per-band floor assembly, the verdict vocabulary and
the controls.  Those are what is tested here.

The probe's ``--self-test`` asserts the same things at run time and is invoked
directly, so a regression fails in CI rather than in the middle of a run.
Follows ``test_dino_1455_verdict360.py``: the sibling probes import the recorded
mesh at module scope and their tests do the same.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "scripts" / "validate" / "ocean_fidelity" / "dino_1226"


@pytest.fixture(scope="module")
def R():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import regional_audit
        return regional_audit
    finally:
        try:
            sys.path.remove(str(PROBE_DIR))
        except ValueError:
            pass


def test_probe_self_test_passes(R, capsys):
    R.self_test()
    assert "SELF-TEST PASSED" in capsys.readouterr().out


# --------------------------------------------------------------- the partition
def test_the_six_bands_are_a_partition_of_every_row(R):
    import acc_thermal_wind as A
    covered = np.zeros(A.NY, dtype=int)
    for _, rows in R.BANDS:
        covered[rows] += 1
    assert np.array_equal(covered, np.ones(A.NY, dtype=int))


def test_bands_p1_p2_are_the_recorded_slices_not_retyped(R):
    """If these ever stop being the recorded ones, this audit is no longer a
    refinement of the verdict run's split and its sums stop meaning anything."""
    import acc_driver_decomp as D
    assert R.BANDS[0][1] == D.LAT_GROUPS[0][1]
    assert R.BANDS[1][1] == D.LAT_GROUPS[1][1]
    north = D.LAT_GROUPS[2][1]
    refined = np.concatenate([np.arange(199)[r] for _, r in R.BANDS[2:]])
    assert np.array_equal(np.sort(refined), np.arange(199)[north])


def test_the_nested_band_is_inside_p4_and_is_never_in_the_partition(R):
    p4 = set(np.arange(199)[R.BANDS[3][1]].tolist())
    e10 = set(np.arange(199)[R.NESTED[0][1]].tolist())
    assert e10 < p4                       # strict subset
    assert R.NESTED[0] not in R.BANDS     # and it is not summed with them


def test_the_structural_zero_row_is_where_the_registration_says(R):
    import acc_thermal_wind as A
    lat = np.asarray(A.gphit)[:, 25]
    assert lat[R.STRUCTURAL_ZERO_ROW] == 0.0
    f = 2.0 * R.OMEGA * np.sin(np.deg2rad(lat))
    assert f[R.STRUCTURAL_ZERO_ROW] == 0.0
    assert f[R.STRUCTURAL_ZERO_ROW - 1] != 0.0


def test_geometry_control_refuses_a_broken_partition(R, monkeypatch):
    """Non-vacuity: K7 must ABORT when the bands stop covering every row."""
    broken = tuple([R.BANDS[0]] + list(R.BANDS[2:]))     # P2 dropped
    monkeypatch.setattr(R, "BANDS", broken)
    monkeypatch.setattr(R, "ALL_BANDS", broken + R.NESTED)
    with pytest.raises(SystemExit, match="not a partition"):
        R.control_geometry()


# ------------------------------------------------------------------ the floors
def test_two_sided_floor_is_the_rss_of_the_two_sample_stds(R):
    a, b = [1.0, 1.1, 0.9, 1.0], [2.0, 2.6, 1.4, 2.0]
    fl, ls, ns = R.two_sided_floor(a, b)
    assert ls == pytest.approx(np.std(a, ddof=1))
    assert ns == pytest.approx(np.std(b, ddof=1))
    assert fl == pytest.approx(np.hypot(ls, ns))
    assert fl > ls and fl > ns


def test_a_nan_member_propagates_into_the_floor_rather_than_vanishing(R):
    """A blown-up member must not yield a plausible finite floor."""
    fl, ls, ns = R.two_sided_floor([1.0, np.nan, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert np.isnan(fl) and np.isnan(ls)


# --------------------------------------------------------- the verdict vocabulary
@pytest.mark.parametrize("gap,floor,want", [
    (1.0, 1.0, "INDISTINGUISHABLE"),
    (2.0, 1.0, "INDISTINGUISHABLE"),          # the bar is inclusive
    (2.0 + 1e-9, 1.0, "gap-at-"),             # and it is a bar
    (5.0, 1.0, "gap-at-"),
])
def test_the_two_x_rule_decides(R, gap, floor, want):
    v, _, _ = R.classify(gap, floor, 1.0, 1.0, 0.0, True)
    assert v.startswith(want)


def test_unmeasurable_is_not_a_pass(R):
    """A zero gap under the quantum margin must WITHHOLD a verdict, not grant
    one -- the failure mode is a dtype-dominated band reading as agreement."""
    v, ratio, flag = R.classify(0.0, 1.0, 1.0, 1.0, 1.0, True)
    assert v == "UNMEASURABLE" and ratio is None and "q" in flag
    ok, _, _ = R.classify(0.0, 1.0, 1.0, 1.0, 0.01, True)
    assert ok == "INDISTINGUISHABLE"


def test_the_one_sided_flag_fires_only_past_a_decade(R):
    _, _, wide = R.classify(1.0, 1.0, 1.0, 1e-3, 0.0, True)
    _, _, even = R.classify(1.0, 1.0, 1.0, 1.0, 0.0, True)
    assert "1s" in wide and "1s" not in even


# -------------------------------------------------------------- the saturation
def test_saturation_needs_two_quarters_and_both_sides(R):
    flat = {(s, d): 1.0 for s in ("lego", "nemo") for d in (180, 270, 360)}
    assert R.saturated(flat)[0]
    grow = {**flat, ("lego", 360): 2.0}
    assert not R.saturated(grow)[0]
    one_quarter = {**flat, ("lego", 270): 2.0, ("lego", 360): 2.0}
    assert not R.saturated(one_quarter)[0]


def test_a_growing_side_refuses_even_when_it_is_tiny_in_the_rss(R):
    """The reason the test is PER SIDE: a combined-floor test would call this
    saturated because the other side dominates the RSS."""
    one_side = {(s, d): 1.0 for s in ("lego", "nemo") for d in (180, 270, 360)}
    one_side[("nemo", 180)] = one_side[("nemo", 270)] = 1e-6
    one_side[("nemo", 360)] = 1e-2
    ok, why = R.saturated(one_side)
    assert not ok and "nemo" in why


def test_the_u_flag_is_attached_only_where_it_could_overturn_the_verdict(R):
    """`u` on every row voids every `no` for free, which is not honest."""
    flat = {(s, d): 1.0 for s in ("lego", "nemo") for d in (270, 360)}
    dead, x_dead, g_dead = R.u_is_material(15.0, flat, 360)
    assert not dead and x_dead == pytest.approx(7.5) and g_dead == 1.0
    doubling = {**flat, ("lego", 360): 2.0}
    live, x_live, g_live = R.u_is_material(2.1, doubling, 360)
    assert live and g_live == 2.0


# ------------------------------------------------------- the known-answer control
def test_k9_reproduces_the_recorded_gap_and_aborts_on_a_planted_wrong_one(R):
    band = R.BANDS[0][0]
    rows = {"lego": {0: {360: {band: 0.0}}},
            "nemo": {0: {360: {band: -R.D360_P1_RECORDED_SV}}}}
    assert R.control_known_answer(rows) == pytest.approx(
        R.D360_P1_RECORDED_SV, abs=R.D360_P1_TOL_SV)
    with pytest.raises(SystemExit, match="not the recorded one"):
        R.control_known_answer(rows, recorded=-0.90000)


def test_k4_treats_a_nan_as_fatal_rather_than_skipping_it(R):
    R.control_finite("clean", [1.0, 2.0, 3.0])
    with pytest.raises(SystemExit, match="non-finite"):
        R.control_finite("dirty", [1.0, np.nan])


# ---------------------------------------------------------------- the reductions
def test_a_depth_uniform_flow_puts_nothing_in_the_shear_leg(R):
    """bt+bc is exact by construction; the leg that can silently break is the
    SPLIT, so both legs are checked on fields with a known answer."""
    import acc_thermal_wind as A
    u = np.ones((A.NY, A.NX, A.NZ), dtype=np.float64)
    for name, rows in R.ALL_BANDS:
        bt, bc = R.band_bt_bc(u, A.umask, rows)
        assert abs(bc) < 1e-12, name
        assert bt + bc == pytest.approx(R.band_transport(u, A.umask, rows),
                                        abs=1e-9)


def test_a_surface_only_jet_puts_nothing_in_the_reference_level_leg(R):
    import acc_thermal_wind as A
    u = np.zeros((A.NY, A.NX, A.NZ), dtype=np.float64)
    u[:, :, 0] = 1.0
    bt, bc = R.band_bt_bc(u, A.umask, R.BANDS[3][1])
    assert abs(bt) < 1e-12 and abs(bc) > 1e-3


def test_the_bands_sum_to_the_full_section_on_a_nonuniform_field(R):
    """A partition that sums on u=1 can still be broken; the additivity is
    checked on a field that varies with row, column AND level."""
    import acc_thermal_wind as A
    j, i, k = np.indices((A.NY, A.NX, A.NZ))
    u = np.sin(0.3 * j) * np.cos(0.7 * i) * (1.0 + 0.1 * k)
    total = R.band_transport(u, A.umask, R.FULL)
    assert sum(R.band_transport(u, A.umask, r) for _, r in R.BANDS) == \
        pytest.approx(total, abs=1e-9)


def test_a_dry_cell_cannot_reach_any_band_transport(R):
    """K5, as a test rather than only as a run-time control.  Both arms are
    C-contiguous: the reduction is not layout-invariant at ~2e-14 Sv, so a view
    against a copy manufactures a leak that is not one."""
    import acc_thermal_wind as A
    rng = np.random.default_rng(0)
    u = np.ascontiguousarray(rng.normal(size=(A.NY, A.NX, A.NZ)))
    base = R.transport_row({"u": u}, A.umask)
    dry = np.argwhere(~A.umask)
    poisoned = u.copy()
    poisoned[tuple(dry[len(dry) // 2])] = 1e6
    after = R.transport_row({"u": poisoned}, A.umask)
    assert max(abs(after[k] - base[k]) for k in base) == 0.0


def test_the_dry_domain_wall_rows_move_nothing(R):
    """K3c: what admits rows 0 and 198 into P1/P6 is this measurement, not an
    argument about which reductions are sums."""
    import acc_thermal_wind as A
    rng = np.random.default_rng(1)
    u = np.ascontiguousarray(rng.normal(size=(A.NY, A.NX, A.NZ)))
    base = R.transport_row({"u": u}, A.umask)
    poisoned = u.copy()
    poisoned[0, :, :] = poisoned[A.NY - 1, :, :] = 1e6
    after = R.transport_row({"u": poisoned}, A.umask)
    assert max(abs(after[k] - base[k]) for k in base) == 0.0


def test_a_wet_plant_moves_its_own_band_and_no_other(R):
    """The non-vacuity half: without this the two tests above pass for a
    reduction that reads nothing at all."""
    import acc_thermal_wind as A
    u = np.zeros((A.NY, A.NX, A.NZ), dtype=np.float64)
    base = R.transport_row({"u": u}, A.umask)
    p4 = np.arange(A.NY)[R.BANDS[3][1]]
    wet = np.argwhere(A.umask[p4.min():p4.max() + 1])
    jw, iw, kw = wet[len(wet) // 2]
    u[jw + p4.min(), iw, kw] = 1e6
    after = R.transport_row({"u": u}, A.umask)
    assert abs(after[R.BANDS[3][0]] - base[R.BANDS[3][0]]) > 1.0
    for other in (R.BANDS[0][0], R.BANDS[1][0], R.BANDS[5][0]):
        assert after[other] == base[other]


def test_an_empty_band_and_depth_selection_aborts_rather_than_returning_nan(R):
    """A mis-specified band must not print a plausible blank into a table."""
    import ts_divergence_atlas as X
    import acc_thermal_wind as A
    wet, w = X.build_weights(np.ones((A.NY, A.NX)))
    d = np.zeros((A.NY, A.NX, A.NZ))
    with pytest.raises(SystemExit, match="zero wet volume"):
        X.wrms(d, w, np.zeros_like(wet, dtype=bool))
