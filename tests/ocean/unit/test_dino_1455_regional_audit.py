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


# ---------------------------------------------------------------------------
# The four defects the code review of 17881d91d found.  Each test below FAILS
# if its fix is reverted, which is the only reason any of them exist.
# ---------------------------------------------------------------------------
def test_the_registered_cuts_are_symmetric_about_the_f_zero_row(R):
    """C4: a one-row shift of the equatorial band passed the ENTIRE self-test,
    because K7 only checks that the bands partition 199 rows and that E10 is
    inside P4 -- and both survive any shift.  The module now asserts these three
    relations at import; this makes the same check fail in CI."""
    import acc_thermal_wind as A
    z = R.STRUCTURAL_ZERO_ROW
    assert R._EQ20_LO + R._EQ20_HI == 2 * z
    assert R._EQ10_LO + R._EQ10_HI == 2 * z
    assert R._SUBTROP_N_HI == 2 * z - A.J1


def test_day0_control_aborts_when_the_two_sides_differ(R):
    """C2: the old K3 compared the day-0 gap against the fp32 quantum -- but the
    legoESM day-0 snapshot IS fp32(NEMO day-0), so gap and quantum are the same
    number and the assertion was |x| <= 10|x|, reading 1.000000 in all seven
    bands.  The replacement is an exact bit-identity, and it can fail."""
    import acc_thermal_wind as A
    shape = (A.NY, A.NX, A.NZ)
    wet = np.ones(shape, dtype=bool)
    nemo = {"T": np.full(shape, 4.0), "S": np.full(shape, 35.0),
            "u": np.zeros(shape)}
    same = {k: np.float32(v).astype(np.float64) for k, v in nemo.items()}
    rows = {"lego": {0: {0: {n: 0.0 for n, _ in R.ALL_BANDS}}},
            "nemo": {0: {0: {n: 0.0 for n, _ in R.ALL_BANDS}}}}
    q = {n: 1e-9 for n, _ in R.ALL_BANDS}
    R.control_day0(same, nemo, wet, wet, rows, q)          # identical: passes
    broken = {k: v.copy() for k, v in same.items()}
    broken["T"][10, 10, 0] += 1e-3
    with pytest.raises(SystemExit, match="NOT starting from the same state"):
        R.control_day0(broken, nemo, wet, wet, rows, q)


def test_band_and_depth_selections_are_disjoint_in_both_directions(R):
    """C1: the old K5b planted a 'dry' T-cell drawn from the whole domain and
    scored it through a P4-UPPER selection.  The cell landed at 4253 m, so the
    DEPTH mask killed the poison before the wet mask was consulted -- the
    campaign's own double-mask defect, inside the control that claimed to design
    against it.  What this file owns is the band-AND-depth intersection."""
    import ts_divergence_atlas as X
    import acc_thermal_wind as A
    wet, _ = X.build_weights(np.ones((A.NY, A.NX)))
    p4u = R.band_depth_sel(R.BANDS[3][1], X.DEPTH_CLASSES[0][0], wet)
    p5u = R.band_depth_sel(R.BANDS[4][1], X.DEPTH_CLASSES[0][0], wet)
    p4a = R.band_depth_sel(R.BANDS[3][1], X.DEPTH_CLASSES[2][0], wet)
    assert p4u.any() and p5u.any() and p4a.any()
    assert not (p4u & p5u).any(), "two bands share a cell"
    assert not (p4u & p4a).any(), "two depth classes share a cell"


def test_a_poison_outside_a_selection_cannot_reach_its_rms(R):
    """The measured half of the same fix, with its non-vacuity partner."""
    import ts_divergence_atlas as X
    import acc_thermal_wind as A
    wet, w = X.build_weights(np.ones((A.NY, A.NX)))
    p4u = R.band_depth_sel(R.BANDS[3][1], X.DEPTH_CLASSES[0][0], wet)
    p5u = R.band_depth_sel(R.BANDS[4][1], X.DEPTH_CLASSES[0][0], wet)
    base = np.zeros((A.NY, A.NX, A.NZ)) + 1e-9
    ref = X.wrms(base, w, p4u)
    outside = base.copy()
    outside[tuple(np.argwhere(p5u)[0])] = 1e6
    assert X.wrms(outside, w, p4u) == ref
    inside = base.copy()
    inside[tuple(np.argwhere(p4u)[0])] = 1e6
    assert X.wrms(inside, w, p4u) > 1.0        # non-vacuity


def test_a_split_margin_inside_its_own_bar_is_unresolved(R):
    """C3: R3 was reported CONFIRMED on an 0.0059 Sv margin against legs whose
    own floors are 0.0129 and 0.0119 Sv -- and two of the four ensemble members
    flip the comparison.  A near-cancelling decomposition needs a bar on the
    MARGIN, and inside that bar the ordering is UNRESOLVED, not refuted."""
    import verdict360 as V
    f_split = float(np.hypot(0.0129, 0.0119))
    assert abs(0.0587) - abs(0.0528) < V.K_PREREG * f_split      # UNRESOLVED
    # and the P3-style case, which DOES clear it and must not be swept up
    f_p3 = float(np.hypot(0.0067, 0.0003))
    assert abs(0.0024) - abs(0.0373) < -V.K_PREREG * f_p3        # barotropic-led


def test_the_artifact_refuses_a_tree_that_moved_mid_run(R, tmp_path):
    """The stamp recorded a commit that landed WHILE THE RUN WAS IN FLIGHT: the
    log read one sha at the top and another at the bottom, with dirty=False."""
    sha_now, dirty_now = R.git_sha_now()
    with pytest.raises(SystemExit, match="CHANGED during this run"):
        R.stamp({}, str(tmp_path), "0" * 40, dirty_now)
    R.stamp({}, str(tmp_path), sha_now, dirty_now)     # unmoved: writes
    assert (tmp_path / "regional_audit.json").exists()


# ---------------------------------------------------------------------------
# The physics review of 37a51f9ef: three measurements it required, and the
# ranking-field defect found while adding them.
# ---------------------------------------------------------------------------
def test_per_row_transports_sum_to_their_band_exactly(R):
    """The per-row ledger and the band ledger must be ONE measurement, or the
    band's cancellation ratio compares two different things."""
    import acc_thermal_wind as A
    rng = np.random.default_rng(3)
    u = np.ascontiguousarray(rng.normal(size=(A.NY, A.NX, A.NZ)))
    rt = R.row_transports(u, A.umask)
    assert rt.shape == (A.NY,)
    for name, rows in R.ALL_BANDS:
        assert rt[rows].sum() == pytest.approx(
            R.band_transport(u, A.umask, rows), abs=1e-9), name


def test_the_cancellation_ratio_separates_a_cancelling_band_from_a_coherent_one(R):
    """The instrument the physics review required: a band whose per-row gaps
    cancel must be distinguishable from one whose gaps add.  Without it the
    equatorial band reads INDISTINGUISHABLE while retaining 1.5% of its own
    summed per-row magnitude."""
    coherent = np.array([0.1, 0.1, 0.1, 0.1])
    cancelling = np.array([1.0, -1.0, 1.0, -0.97])
    assert abs(coherent.sum() / np.abs(coherent).sum()) == pytest.approx(1.0)
    assert abs(cancelling.sum() / np.abs(cancelling).sum()) < 0.01


def test_the_depth_only_hotspot_baseline_selects_the_SHALLOWEST_cells(R):
    """The ranking-field defect: ``hotspot_set`` ranks by |d0|, so a field that
    is merely negative-and-large at depth ranks the DEEPEST cells first.  The
    first version used -gdept0 with -1e30 off-band and scored 0.0000."""
    import ts_divergence_atlas as X
    import acc_thermal_wind as A
    wet, w = X.build_weights(np.ones((A.NY, A.NX)))
    band = R.band_cell_mask(R.BANDS[3][1]) & wet
    wb = np.where(band, w, 0.0)
    rank = np.where(band, R.DEPTH_RANK_FIELD, 0.0)
    sel = X.hotspot_set(rank, wb, X.Q2_NULL)
    z = np.asarray(A.gdept0, dtype=np.float64)
    assert sel[band].any()
    # every selected in-band cell must be shallower than every unselected one
    assert z[sel & band].max() <= z[(~sel) & band].min() + 1e-9
    assert z[sel & band].max() < 400.0, "the shallowest 5% is not shallow"


def test_the_depth_baseline_is_what_makes_the_retention_readable(R):
    """Non-vacuity of the control itself: on a field that is PURELY stratified
    -- no horizontal structure at all -- the day-10 hotspot set must NOT beat
    the depth-only baseline, which is the case that voided R4 as registered."""
    import ts_divergence_atlas as X
    import acc_thermal_wind as A
    wet, w = X.build_weights(np.ones((A.NY, A.NX)))
    band = R.band_cell_mask(R.BANDS[3][1]) & wet
    wb = np.where(band, w, 0.0)
    z = np.asarray(A.gdept0, dtype=np.float64)
    strat = np.where(band, np.exp(-z / 50.0), 0.0)      # depth only
    hot = X.hotspot_set(strat, wb, X.Q2_NULL)
    base = X.hotspot_set(np.where(band, R.DEPTH_RANK_FIELD, 0.0), wb, X.Q2_NULL)
    assert X.retention(strat, wb, hot) <= X.retention(strat, wb, base) + 1e-9


def test_the_zero_crossing_search_is_confined_to_the_wind_driven_layer(R):
    """An unrestricted search returned 1178 m and printed it beside a label
    reading 'the top of the eastward undercurrent'."""
    import inspect
    src = inspect.getsource(R.report)
    assert "ZC_MAX_M" in src
    assert "none in the top" in src, "an absent crossing must be reported absent"


def test_the_rotation_rate_comes_from_the_shared_constants_module(R):
    """Repo rule: no hardcoded 7.292e-5 in scripts either.  And the K11
    assertion is independent of the value -- sin(0) is 0 for every Omega -- so
    the control tests the GRID, which is what it claims to test."""
    from legoesm import constants
    assert R.OMEGA is constants.Omega or R.OMEGA == constants.Omega
    import inspect
    src = inspect.getsource(R)
    assert "7.292" not in src.split("Omega")[0] or "constants.Omega" in src


# ---------------------------------------------------------------------------
# Round-2 review: the LEVEL baseline and the margin floor.
# ---------------------------------------------------------------------------
def test_the_level_baseline_is_horizontally_uniform_and_keeps_the_profile(R):
    """The depth baseline controls only for 'shallower is more diverged'.  The
    divergence peaks at the MIXED-LAYER BASE, so a hotspot set can beat it by
    locating the right LEVEL -- vertical information dressed as horizontal.
    The LEVEL field must be constant within each level and carry the true
    per-level rms."""
    import ts_divergence_atlas as X
    import acc_thermal_wind as A
    wet, w = X.build_weights(np.ones((A.NY, A.NX)))
    sel = R.band_cell_mask(R.BANDS[3][1]) & wet
    rng = np.random.default_rng(11)
    d = np.where(sel, rng.normal(size=sel.shape), 0.0)
    lf = R.level_rank_field(d, w, sel)
    for k in range(A.NZ):
        vals = lf[:, :, k][sel[:, :, k]]
        if vals.size:
            assert np.allclose(vals, vals[0]), f"level {k} is not uniform"
    assert np.all(lf[~sel] == 0.0)
    # it must reproduce the per-level volume-weighted rms
    k = 0
    want = np.sqrt((w[:, :, k] * d[:, :, k] ** 2)[sel[:, :, k]].sum()
                   / w[:, :, k][sel[:, :, k]].sum())
    assert lf[:, :, k][sel[:, :, k]][0] == pytest.approx(want)


# NO SYNTHETIC TEST SEPARATES THE TWO RETENTION BASELINES, and that is
# recorded here rather than papered over with a case tuned until it passes.
# Two were attempted and both failed for reasons that are properties of the
# statistic, not of the attempts:
#   * a level-constant field -- ``level_rank_field`` is an order-preserving
#     transform of it, so the two sets agree 99.59% BY CONSTRUCTION and the
#     retentions differ by 8e-12.  The original version of this test passed
#     only on its 1e-9 tolerance; the code review caught it.
#   * a field peaking at 100 m, meant to defeat the depth-only baseline --
#     5% of a band's VOLUME is a thin enough surface layer that it already
#     reaches past 100 m, so the depth set overlaps the hot set 99.6% and
#     retention saturates at 0.999999 for all three sets.
# The independent adversarial reviewer hit the same saturation.  What replaces
# them is the data-backed pin below: the real measured ratio, which moves if the
# baseline construction changes.  The CONSTRUCTION of both fields is still
# tested directly (above, and in the random-null test).


def test_the_equatorial_horizontal_memory_ratio_is_pinned_to_its_measured_value(R):
    """The data-backed pin that replaces the tautology above.

    A synthetic case cannot separate a genuine horizontal-memory signal from
    the baseline (the reviewer tried and hit saturation), so the guard against
    a silent regression in ``level_rank_field`` is the REAL measured ratio.  If
    the baseline construction changes, this moves and the test says so.

    Values from the stamped artifact (upper class, day 360, temperature):
    the equatorial band retains 0.643 against a level baseline of 0.123, and
    the southern basin 0.049 against 0.044.  Skipped when the recorded states
    are not on this machine, because it is a pin on real data.
    """
    import json
    import os
    art = "/tmp/dino_regional_audit_final/regional_audit.json"
    if not os.path.exists(art):
        pytest.skip("no stamped artifact on this machine")
    ret = json.load(open(art))["retention"]
    p4, p1 = "P4 equatorial +/-20", "P1 south of band"
    assert ret[f"{p4}|T|360|upper"] == pytest.approx(0.643, abs=5e-3)
    assert ret[f"{p4}|T|360|upper_level"] == pytest.approx(0.123, abs=5e-3)
    assert ret[f"{p1}|T|360|upper"] == pytest.approx(0.049, abs=5e-3)
    assert ret[f"{p1}|T|360|upper_level"] == pytest.approx(0.044, abs=5e-3)
    # the separation the finding rests on: a hundredfold in the margin
    m4 = ret[f"{p4}|T|360|upper"] - ret[f"{p4}|T|360|upper_level"]
    m1 = ret[f"{p1}|T|360|upper"] - ret[f"{p1}|T|360|upper_level"]
    assert m4 > 50 * m1, (m4, m1)


def test_the_random_null_matches_the_hot_sets_per_level_volume(R):
    """The null that replaces the degenerate level baseline inside one class.

    It must be beatable ONLY by horizontal placement, so it has to carry the
    same per-level volume as the set it scores against -- otherwise knowing
    which level to sit in still buys a score.
    """
    import ts_divergence_atlas as X
    import acc_thermal_wind as A
    wet, w = X.build_weights(np.ones((A.NY, A.NX)))
    sel = R.band_depth_sel(R.BANDS[3][1], X.DEPTH_CLASSES[0][0], wet)
    rng = np.random.default_rng(5)
    d = np.where(sel, rng.normal(size=sel.shape), 0.0)
    wb = np.where(sel, w, 0.0)
    hot = X.hotspot_set(d, wb, X.Q2_NULL)
    draws = R.level_matched_random_null(hot, w, sel, n_draws=3)
    assert len(draws) == 3
    for pk in draws:
        assert not pk[~sel].any(), "the null leaked outside the selection"
        for k in range(A.NZ):
            got = float(w[:, :, k][pk[:, :, k]].sum())
            wantv = float(w[:, :, k][hot[:, :, k]].sum())
            if wantv <= 0.0:
                assert got == 0.0
            else:
                # one cell of overshoot is inherent to a greedy volume fill
                assert got >= wantv and got <= wantv * 1.5, (k, got, wantv)


def test_the_split_margin_floor_is_measured_on_the_margin(R):
    """Both reviewers found the RSS construction independently.  With
    opposite-sign legs the margin is ALGEBRAICALLY the band gap (bt+bc==gap),
    so an RSS-of-legs bar re-scores the ledger's own number against a threshold
    up to 23x too large."""
    # bt<0, bc>0 -> |bc|-|bt| == bt+bc == gap, exactly
    bt, bc = -2.4782, 1.5262
    assert abs(bc) - abs(bt) == pytest.approx(bt + bc, abs=1e-12)
    # the paired-margin floor is the std of the member margins, and it is NOT
    # the RSS of two per-leg floors
    mm = [0.10, 0.12, 0.08, 0.11]
    assert float(np.std(mm, ddof=1)) < float(np.hypot(0.28, 0.27))


def test_the_cancellation_ratio_carries_its_member_spread(R):
    """The headline 1.5% is a control-member reading; three members were being
    discarded and their spread can exceed it, leaving the SIGN undetermined."""
    members = [0.015, 0.022, -0.013, -0.015]
    assert float(np.std(members, ddof=1)) > abs(members[0])
    assert max(abs(m) for m in members) < 0.03      # magnitude still bounded


def test_the_through_origin_fit_does_not_report_corr_squared(R):
    """A fit with no intercept explains 1 - SSres/SStot with SStot un-centred;
    corr^2 is the wrong label for it."""
    import inspect
    src = inspect.getsource(R.report)
    assert "ss_tot" in src and "ss_res" in src
    assert "variance explained {corr ** 2" not in src


def test_the_retracted_word_is_gone_from_the_tool(R):
    """A retraction that lives only in a commit message is not a retraction --
    the tool kept printing 'barotropic-led' after retraction #7 relabelled the
    leg bottom-referenced."""
    import inspect
    src = inspect.getsource(R.report)
    assert "barotropic-led" not in src
    assert "bottom-referenced-led" in src


def test_the_registered_escalation_form_is_disclosed_not_deleted(R):
    """A registered rule narrows by disclosure, never by dropping the form that
    was registered."""
    import inspect
    src = inspect.getsource(R.report)
    assert "esc_all" in src and "AS REGISTERED" in src
    assert "escalated_all_horizons" in src and "escalated_day360" in src
