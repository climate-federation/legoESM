"""Direct test for #1455's per-cell two-step projection of the barotropic
deposit at the DINO end-wall rows.

Every assertion is written so it FAILS if the logic it covers is removed or
inverted.  Four things are pinned:

(a) the projector's normalisation and its blindness to slow fields -- the
    criteria are stated in amplitude units and would silently move if it
    returned 2A instead of A;
(b) that the controls CAN fail, since two controls on this branch already
    passed while proving nothing;
(c) that a hole in the state series is fatal rather than a shorter series,
    because the two-step operator is indexed by position; and
(d) THE OUTPUT SURFACE.  Adversarial review 2026-08-26 mutated six lines of
    ``project_component``/``magnitude_check`` -- inverting the enrichment,
    swapping the side ratio, pointing a term at the wrong array, changing
    which rows are scored, scaling dt by 10, narrowing the floor window -- and
    the whole suite stayed green.  The tests at the bottom go red under each.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_DIR = os.path.join(_ROOT, "scripts", "validate", "ocean_fidelity",
                    "dino_1226")
_MOD_PATH = os.path.join(_DIR, "baro_deposit_wall_nyquist.py")


def _load():
    if _DIR not in sys.path:
        sys.path.insert(0, _DIR)
    spec = importlib.util.spec_from_file_location("_wall_nyq", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wall_nyq"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()

#: A 9x16 basin with a dry rim.  Wide enough that each end-wall row carries
#: more than ``MIN_ROW_CELLS`` cells, so the per-row path is exercised rather
#: than skipped -- a toy narrower than the threshold would silently test
#: nothing.
_NY, _NX = 9, 16
_ROW_CELLS = 12          # 14 wet faces in the row, minus its two corners


def _toy_geometry():
    wet = np.zeros((_NY, _NX), dtype=bool)
    wet[1:8, 1:15] = True
    weights = np.zeros_like(wet, dtype=np.float64)
    # deliberately NON-uniform, so the load-bearing control has something to
    # find; a constant weight would make that control vacuous.
    weights[wet] = 1.0 + np.arange(wet.sum(), dtype=np.float64)
    return wet, weights


def _write_maps(tmp_path, per_state: dict, wet=None, weights=None):
    """Write a full five-state map series, so no test can pass on a stub.

    ``per_state`` maps an npz key to an array stacked over states.  Everything
    the loader requires and the test does not care about is filled in here, in
    one place, so adding a required key breaks one helper rather than six
    tests.
    """
    if wet is None:
        wet, weights = _toy_geometry()
    zero = np.zeros((len(M.CONSECUTIVE_KTS),) + wet.shape)
    full = {}
    for c in ("U", "V"):
        for stem in (f"d{c}_avg", f"d{c}_sub", f"lego_{c}bar_avg",
                     f"nemo_{c}bar_avg", f"lego_{c}bar_avg_sub"):
            full[stem] = zero
    full.update(per_state)
    for i, kt in enumerate(M.CONSECUTIVE_KTS):
        np.savez(tmp_path / f"deposit_map_kt{kt}.npz",
                 ic_step=np.int64(kt), acc_w=weights, acc_w_v=weights,
                 wetu=wet.astype(np.int8), wetv=wet.astype(np.int8),
                 wgt_primary=np.array([0.0, 0.5, 0.5]),
                 seqdump=np.array("x"), provenance=np.array("y"),
                 **{k: v[i] for k, v in full.items()})


# ---------------------------------------------------------------------------
# (a) the projector
# ---------------------------------------------------------------------------

def test_projector_returns_the_planted_amplitude_not_twice_it():
    wet, weights = _toy_geometry()
    n, amp = 5, 3.7e-4
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    field = np.broadcast_to(sign, (n,) + wet.shape) * amp
    got = M.amp_on(np.ascontiguousarray(field), wet, weights)
    assert got == pytest.approx(amp, rel=1e-12)
    # a projector returning 2A would pass a naive "is it nonzero" test while
    # putting every criterion off by a factor 2.
    assert got != pytest.approx(2.0 * amp, rel=1e-6)


@pytest.mark.parametrize("kind", ["constant", "ramp"])
def test_projector_is_blind_to_slow_fields(kind):
    wet, weights = _toy_geometry()
    n = 5
    if kind == "constant":
        field = np.full((n,) + wet.shape, 5.0)
    else:
        field = (np.arange(n, dtype=np.float64)[:, None, None]
                 * np.ones((n,) + wet.shape))
    assert M.amp_on(field, wet, weights) < 1e-12


def test_cell_helper_reproduces_the_committed_estimator():
    """The bootstrap uses a per-cell helper; if it drifted from
    ``per_cell_nyquist`` the table and its interval would be two estimators."""
    wet, weights = _toy_geometry()
    rng = np.random.default_rng(7)
    series = rng.normal(size=(5,) + wet.shape) * wet
    assert (M.aggregate(M.cell_amplitudes(series, wet), weights[wet])
            == M.amp_on(series, wet, weights))


def test_synthetic_checks_reject_a_broken_projector(monkeypatch):
    """The self-check must FAIL when the projector is wrong -- otherwise it is
    decoration.  Doubling the operator is exactly the normalisation error the
    check exists to catch."""
    wet, weights = _toy_geometry()
    assert M.synthetic_checks(wet, weights, 5)["constant"] < 1e-12

    # doubling the SHARED operator doubles both paths consistently, so the
    # estimator-agreement check still passes and the NORMALISATION check is the
    # one that must fire.
    real_op = M.ewt.two_dt_component
    monkeypatch.setattr(M.ewt, "two_dt_component",
                        lambda f: 2.0 * real_op(f))
    with pytest.raises(SystemExit, match="normalisation"):
        M.synthetic_checks(wet, weights, 5)


def test_synthetic_checks_catch_two_estimators_drifting_apart(monkeypatch):
    """The bootstrap and the table must be one estimator, not two."""
    wet, weights = _toy_geometry()
    real = M.zwb.per_cell_nyquist
    monkeypatch.setattr(M.zwb, "per_cell_nyquist",
                        lambda s, w: 2.0 * real(s, w))
    with pytest.raises(SystemExit, match="two different estimators"):
        M.synthetic_checks(wet, weights, 5)


# ---------------------------------------------------------------------------
# (b) bands and controls
# ---------------------------------------------------------------------------

def test_bands_are_disjoint_and_the_zonal_rows_are_the_end_walls():
    wet, _ = _toy_geometry()
    bands = M.build_bands(wet)
    assert not (bands["zonal_wall"] & bands["interior"]).any()
    assert not (bands["zonal_wall"] & bands["meridional_wall"]).any()
    rows = M.zonal_wall_rows(bands)
    assert sorted(r["j"] for r in rows) == [1, 7]
    # the row's own end columns drop out as corners
    assert all(r["cells"] == _ROW_CELLS for r in rows)


def test_empty_band_is_fatal():
    wet = np.ones((5, 5), dtype=bool)   # no dry cell anywhere
    with pytest.raises(SystemExit):
        M.build_bands(wet)


def test_land_poison_control_fires_when_land_leaks_in():
    """A poison that does not move an UNMASKED statistic proves nothing, and a
    band that admits land must be caught.  Both arms are exercised."""
    wet, weights = _toy_geometry()
    rng = np.random.default_rng(0)
    series = rng.normal(size=(5,) + wet.shape) * wet
    bands = M.build_bands(wet)
    ok = M.land_poison_check(series, bands["zonal_wall"], weights, wet)
    assert ok["production_identical"] and ok["unmasked_moves"]

    leaky = np.where(weights > 0, weights, 1.0)
    with pytest.raises(SystemExit, match="band is not a subset"):
        M.land_poison_check(series, bands["zonal_wall"] | ~wet, leaky, wet)


def test_weights_control_refuses_inert_weights():
    wet, weights = _toy_geometry()
    rng = np.random.default_rng(1)
    series = rng.normal(size=(5,) + wet.shape) * wet
    bands = M.build_bands(wet)
    assert M.weights_load_bearing(series, bands["zonal_wall"],
                                  weights)["load_bearing"]
    flat = np.where(weights > 0, 1.0, 0.0)
    with pytest.raises(SystemExit, match="inert"):
        M.weights_load_bearing(series, bands["zonal_wall"], flat)


# ---------------------------------------------------------------------------
# (c) the state series
# ---------------------------------------------------------------------------

def test_a_missing_state_is_fatal_not_a_shorter_series(tmp_path):
    with pytest.raises(SystemExit, match="re-phases"):
        M.load_maps(str(tmp_path))


def test_a_map_without_the_meridional_side_is_refused(tmp_path):
    """A u-only map predates the two-component instrument; projecting it would
    measure the TANGENTIAL component at a zonal wall and call it a null."""
    wet, weights = _toy_geometry()
    for kt in M.CONSECUTIVE_KTS:
        np.savez(tmp_path / f"deposit_map_kt{kt}.npz",
                 ic_step=np.int64(kt), acc_w=weights,
                 wetu=wet.astype(np.int8), dU_avg=np.zeros(wet.shape),
                 dU_sub=np.zeros(wet.shape),
                 lego_Ubar_avg=np.zeros(wet.shape),
                 nemo_Ubar_avg=np.zeros(wet.shape),
                 lego_Ubar_avg_sub=np.zeros(wet.shape),
                 seqdump=np.array("x"), provenance=np.array("y"))
    with pytest.raises(SystemExit, match="one velocity component"):
        M.load_maps(str(tmp_path))


def test_a_mislabelled_state_is_refused(tmp_path):
    wet, weights = _toy_geometry()
    _write_maps(tmp_path, {})
    bad = M.CONSECUTIVE_KTS[2]
    z = dict(np.load(tmp_path / f"deposit_map_kt{bad}.npz"))
    z["ic_step"] = np.int64(9999)
    np.savez(tmp_path / f"deposit_map_kt{bad}.npz", **z)
    with pytest.raises(SystemExit, match="not be consecutive"):
        M.load_maps(str(tmp_path))


def test_moving_geometry_is_refused(tmp_path):
    wet, weights = _toy_geometry()
    _write_maps(tmp_path, {})
    moved = M.CONSECUTIVE_KTS[3]
    z = dict(np.load(tmp_path / f"deposit_map_kt{moved}.npz"))
    z["acc_w"] = z["acc_w"] * 2.0
    np.savez(tmp_path / f"deposit_map_kt{moved}.npz", **z)
    with pytest.raises(SystemExit, match="different cells"):
        M.load_maps(str(tmp_path))


def test_forcing_term_is_the_difference_of_the_two_shares(tmp_path):
    wet, _ = _toy_geometry()
    rng = np.random.default_rng(2)
    tot = rng.normal(size=(5,) + wet.shape)
    sub = rng.normal(size=(5,) + wet.shape)
    _write_maps(tmp_path, {"dU_avg": tot, "dU_sub": sub,
                           "dV_avg": 2 * tot, "dV_sub": sub})
    maps = M.load_maps(str(tmp_path))
    assert np.allclose(maps["dU_forcing"], tot - sub)
    assert np.allclose(maps["dV_forcing"], 2 * tot - sub)


# ---------------------------------------------------------------------------
# (d) the output surface -- the six surviving mutations
# ---------------------------------------------------------------------------

def _planted(tmp_path, wall_amp=4.0e-3, int_amp=1.0e-3, lego_gain=3.0):
    """Five states whose two-step structure is known by construction.

    Wall cells alternate at ``wall_amp``, everything else at ``int_amp``, so
    the enrichment is exactly ``wall_amp/int_amp``.  legoESM's side is
    ``lego_gain`` times NEMO's EVERYWHERE, so the side ratio is exactly
    ``lego_gain`` while the double ratio -- which divides out any uniform
    gain -- is exactly 1.
    """
    wet, weights = _toy_geometry()
    bands = M.build_bands(wet)
    n = len(M.CONSECUTIVE_KTS)
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    nemo = (sign * int_amp * (bands["interior"] | bands["meridional_wall"])
            + sign * wall_amp * bands["zonal_wall"])
    nemo = np.ascontiguousarray(nemo, dtype=np.float64)
    lego = lego_gain * nemo
    per = {}
    for c in ("U", "V"):
        per[f"nemo_{c}bar_avg"] = nemo
        per[f"lego_{c}bar_avg"] = lego
        per[f"lego_{c}bar_avg_sub"] = nemo      # IN_LOOP reproduces NEMO
        per[f"d{c}_avg"] = lego - nemo
        per[f"d{c}_sub"] = np.zeros_like(nemo)
    _write_maps(tmp_path, per)
    return M.load_maps(str(tmp_path))


def test_project_recovers_planted_enrichment_side_and_double_ratio(tmp_path):
    maps = _planted(tmp_path, wall_amp=4.0e-3, int_amp=1.0e-3, lego_gain=3.0)
    comp = M.project_component(maps, "U", "acc_w", "wetu")
    tot = comp["terms"]["TOTAL"]
    for r in tot["rows"].values():
        # an INVERTED enrichment would read 0.25
        assert r["enrichment"] == pytest.approx(4.0, rel=1e-9)
        # a SWAPPED side ratio would read 1/3
        assert r["side_ratio"] == pytest.approx(3.0, rel=1e-9)
        # a uniform gain must cancel out of the registered comparable
        assert r["double_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert tot["enrichment_band"] == pytest.approx(4.0, rel=1e-9)
    assert tot["double_ratio_band"] == pytest.approx(1.0, rel=1e-9)


def test_double_ratio_detects_a_wall_only_excess(tmp_path):
    """The registered comparable must separate a wall-preferential excess from
    a uniform one; if it could not, the reading would rest on nothing."""
    wet, weights = _toy_geometry()
    bands = M.build_bands(wet)
    n = len(M.CONSECUTIVE_KTS)
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    nemo = np.ascontiguousarray(sign * 1.0e-3 * wet, dtype=np.float64)
    lego = nemo + sign * 1.0e-3 * bands["zonal_wall"]     # 2x at the wall only
    per = {}
    for c in ("U", "V"):
        per[f"nemo_{c}bar_avg"] = nemo
        per[f"lego_{c}bar_avg"] = lego
        per[f"lego_{c}bar_avg_sub"] = nemo
        per[f"d{c}_avg"] = lego - nemo
        per[f"d{c}_sub"] = np.zeros_like(nemo)
    _write_maps(tmp_path, per)
    comp = M.project_component(M.load_maps(str(tmp_path)), "U", "acc_w",
                               "wetu")
    for r in comp["terms"]["TOTAL"]["rows"].values():
        assert r["double_ratio"] == pytest.approx(2.0, rel=1e-9)
    # the machinery term, where lego reproduces NEMO, must read exactly 1
    for r in comp["terms"]["IN_LOOP"]["rows"].values():
        assert r["double_ratio"] == pytest.approx(1.0, rel=1e-9)


def test_in_loop_reads_the_substituted_array_not_the_total(tmp_path):
    """Pointing IN_LOOP at the wrong lego array silently turns it into TOTAL --
    review mutated exactly that and the suite stayed green."""
    maps = _planted(tmp_path, lego_gain=3.0)
    comp = M.project_component(maps, "U", "acc_w", "wetu")
    tot = comp["terms"]["TOTAL"]["rows"]["j=1"]
    inl = comp["terms"]["IN_LOOP"]["rows"]["j=1"]
    assert inl["side_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert tot["side_ratio"] == pytest.approx(3.0, rel=1e-9)
    assert inl["amp_diff"] < 1e-15
    assert tot["amp_diff"] > 1e-6


def test_only_rows_above_the_cell_threshold_are_scored(tmp_path):
    """Which rows get a per-row ratio is a decision the output depends on."""
    maps = _planted(tmp_path)
    comp = M.project_component(maps, "U", "acc_w", "wetu")
    assert comp["scored_rows"] == ["j=1", "j=7"]
    assert {r["j"] for r in comp["rows"]} == {1, 7}
    assert _ROW_CELLS >= M.MIN_ROW_CELLS


def test_a_basin_with_no_scorable_row_is_fatal(tmp_path):
    """A threshold above every row must refuse, not report an empty table."""
    maps = _planted(tmp_path)
    old = M.MIN_ROW_CELLS
    try:
        M.MIN_ROW_CELLS = _ROW_CELLS + 1
        with pytest.raises(SystemExit, match="nothing to resolve"):
            M.project_component(maps, "U", "acc_w", "wetu")
    finally:
        M.MIN_ROW_CELLS = old


def test_forcing_term_carries_no_side_statistics(tmp_path):
    maps = _planted(tmp_path)
    forc = M.project_component(maps, "U", "acc_w", "wetu")["terms"]["FORCING"]
    assert forc["has_nemo_side"] is False
    for r in forc["rows"].values():
        assert "side_ratio" not in r and "double_ratio" not in r


def test_both_components_are_projected_independently(tmp_path):
    """A zonal wall's NORMAL component is v; the two must not be aliased."""
    wet, _ = _toy_geometry()
    maps = _planted(tmp_path)
    maps["dV_avg"] = 5.0 * maps["dU_avg"]
    maps["dV_forcing"] = maps["dV_avg"] - maps["dV_sub"]
    u = M.project_component(maps, "U", "acc_w", "wetu")
    v = M.project_component(maps, "V", "acc_w_v", "wetv")
    assert (v["terms"]["TOTAL"]["rows"]["j=1"]["amp_diff"]
            == pytest.approx(
                5.0 * u["terms"]["TOTAL"]["rows"]["j=1"]["amp_diff"],
                rel=1e-9))


def test_magnitude_check_uses_local_metrics_and_the_named_dt(tmp_path,
                                                             monkeypatch):
    """S5 must be amp*H*dt/dx with the row's OWN dx and H.

    Review caught version 1 pairing a wall-row amplitude with basin-median
    geometry, and a dt scaled by 10 left every test green.
    """
    maps = _planted(tmp_path)
    comp = M.project_component(maps, "U", "acc_w", "wetu")
    wet, weights = _toy_geometry()

    class _Var:
        def __init__(self, a):
            self._a = a

        def __getitem__(self, k):
            return self._a[k]

    class _DS:
        def __init__(self, *a, **k):
            self.variables = {"e1u": _Var(np.full((1,) + wet.shape, 2.0e4)),
                              "e2u": _Var(np.full((1,) + wet.shape, 1.0e4))}

        def close(self):
            pass

    import netCDF4
    monkeypatch.setattr(netCDF4, "Dataset", _DS)
    m5 = M.magnitude_check(comp, maps, "U")
    row = int(m5["row"].split("=")[1])
    depth = weights[row][weights[row] > 0] / 1.0e4
    expect = m5["amp_ms"] * float(np.median(depth)) * M.DT_S / 2.0e4
    assert m5["eta_m"] == pytest.approx(expect, rel=1e-12)
    assert m5["dx_km"] == pytest.approx(20.0, rel=1e-12)
    assert M.DT_S == 2700.0


def test_window_attenuation_is_measured_from_the_weights():
    """The 'what this cannot see' number must come from data, not a literal."""
    uniform45 = np.concatenate([np.zeros(23), np.full(45, 1.0 / 45.0)])
    got = M.window_attenuation(uniform45)
    assert got["n_nonzero"] == 45 and got["first_nonzero_substep"] == 24
    assert got["substep_nyquist_transfer"] == pytest.approx(1.0 / 45.0,
                                                            rel=1e-12)
    assert got["attenuation_factor"] == pytest.approx(45.0, rel=1e-12)
    # a one-substep window passes an alternating sequence untouched
    assert M.window_attenuation(np.array([1.0]))["attenuation_factor"] == 1.0


def test_a_dead_denominator_raises_instead_of_returning_a_huge_ratio():
    """`max(den, 1e-300)` turned a dead band into a 1e300 'enrichment'."""
    assert M.ratio_or_refuse(2.0, 4.0, "x") == pytest.approx(0.5)
    with pytest.raises(SystemExit, match="non-positive denominator"):
        M.ratio_or_refuse(1.0, 0.0, "x")


def test_double_ratio_interval_brackets_its_own_point_estimate(tmp_path):
    maps = _planted(tmp_path, wall_amp=4.0e-3, int_amp=1.0e-3, lego_gain=3.0)
    comp = M.project_component(maps, "U", "acc_w", "wetu")
    r = comp["terms"]["TOTAL"]["rows"]["j=1"]
    iv = r["double_ratio_interval"]
    assert iv["point"] == pytest.approx(r["double_ratio"], rel=1e-12)
    assert iv["subwindow_lo"] <= iv["point"] <= iv["subwindow_hi"]
    assert len(iv["subwindow"]) == len(M.CONSECUTIVE_KTS) - 2


def test_enrichment_is_scale_invariant_which_is_not_a_power_argument():
    """Homogeneity of degree 1, and NOTHING MORE.

    Version 1 of the probe carried a test with this arithmetic and a docstring
    claiming it showed the sample count cancels out of an enrichment.  It does
    not: this is true of any RMS ratio.  The claim is retracted in the probe
    and the test is renamed to say what it actually pins.
    """
    wet, weights = _toy_geometry()
    bands = M.build_bands(wet)
    rng = np.random.default_rng(3)
    series = rng.normal(size=(5,) + wet.shape) * wet
    a = (M.amp_on(series, bands["zonal_wall"], weights)
         / M.amp_on(series, bands["interior"], weights))
    b = (M.amp_on(11.0 * series, bands["zonal_wall"], weights)
         / M.amp_on(11.0 * series, bands["interior"], weights))
    assert a == pytest.approx(b, rel=1e-12)


def test_subwindow_enrichments_vary_so_the_point_estimate_needs_an_interval():
    """The retracted power argument, made falsifiable.

    If three-state sub-windows of the SAME series gave the same enrichment,
    version 1's "the sample count cancels" would have been harmless.  They do
    not, which is why every enrichment now ships a spread.
    """
    wet, weights = _toy_geometry()
    bands = M.build_bands(wet)
    rng = np.random.default_rng(11)
    series = rng.normal(size=(7,) + wet.shape) * wet
    sub = [M.amp_on(series[s:s + 3], bands["zonal_wall"], weights)
           / M.amp_on(series[s:s + 3], bands["interior"], weights)
           for s in range(series.shape[0] - 2)]
    assert max(sub) / min(sub) > 1.05


# ---------------------------------------------------------------------------
# Round-2 review: the FLOOR column was untested (narrowing its window and even
# inverting the ratio left the suite green), the paired bootstrap was untested,
# S3 was registered and never computed, and the diagnostics that overturned the
# v2 headline did not exist.  Each is pinned below.
# ---------------------------------------------------------------------------

def test_floor_ratio_arithmetic_and_both_zero_floor_arms():
    assert M.floor_ratio(6.0, 2.0) == pytest.approx(3.0)
    # inverting to amp*floor would give 12.0
    assert M.floor_ratio(6.0, 2.0) != pytest.approx(12.0)
    # a term reproducing the oracle exactly: absent, not infinite
    assert M.floor_ratio(0.0, 0.0) == 0.0
    # real amplitude with no slow curvature to leak: infinitely above floor
    assert M.floor_ratio(1.0, 0.0) == float("inf")


def test_the_leakage_floor_window_is_honoured(tmp_path):
    """Narrowing the floor window silently changes every floor ratio.

    Review narrowed it to one sample and the whole suite stayed green.  Pin it
    by checking the floor is the one the FULL usable window gives.
    """
    wet, weights = _toy_geometry()
    rng = np.random.default_rng(23)
    series = rng.normal(size=(7,) + wet.shape) * wet
    area = np.where(weights > 0, weights, 0.0)
    full = float(M.ewt.two_dt_leakage_floor(series, wet, area=area, mask=wet,
                                            window=slice(0, None)))
    narrow = float(M.ewt.two_dt_leakage_floor(series, wet, area=area, mask=wet,
                                              window=slice(0, 1)))
    # with 7 states the usable series has 3 samples, so the two DIFFER --
    # which is what makes the window a real choice rather than a no-op.
    assert full != pytest.approx(narrow, rel=1e-9)
    per = M.ewt.two_dt_leakage_floor(series, wet, area=area, mask=wet,
                                     return_series=True)
    assert full == pytest.approx(float(np.mean(per)), rel=1e-12)


def test_the_bootstrap_is_paired_across_the_two_models(tmp_path):
    """Resampling NEMO's cells independently of legoESM's destroys the pairing
    that makes the double-ratio interval mean anything; review found that
    change invisible to the suite."""
    wet, weights = _toy_geometry()
    bands = M.build_bands(wet)
    n = len(M.CONSECUTIVE_KTS)
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    rng = np.random.default_rng(5)
    nemo = np.ascontiguousarray(sign * (1.0 + rng.random(wet.shape)) * wet)
    lego = nemo * (1.0 + 1.0 * bands["zonal_wall"])     # wall-only excess
    paired = M.double_ratio_interval(lego, nemo, bands["zonal_wall"],
                                     bands["interior"], weights, n_boot=400)
    # a perfectly paired resample of a pure multiplicative wall excess has a
    # TIGHT interval; unpaired it must widen.
    width = paired["boot_hi"] - paired["boot_lo"]
    assert width < 0.05 * paired["point"]
    assert paired["point"] == pytest.approx(2.0, rel=1e-9)


def test_state_independence_separates_a_constant_from_a_varying_residual():
    """THE diagnostic that overturned version 2's headline.

    A residual that is the same field at every state is annihilated by the
    two-step operator, so its size cannot be read off the projection.
    """
    wet, weights = _toy_geometry()
    fixed = np.broadcast_to(np.linspace(1.0, 2.0, wet.size).reshape(wet.shape),
                            (5,) + wet.shape) * wet
    got = M.state_independence(np.ascontiguousarray(fixed), wet, weights)
    assert got["state_constant_share"] == pytest.approx(1.0, rel=1e-12)
    assert got["cross_state_corr_min"] == pytest.approx(1.0, rel=1e-12)
    # and the projector must indeed report ~nothing for it
    assert M.amp_on(np.ascontiguousarray(fixed), wet, weights) < 1e-12

    rng = np.random.default_rng(9)
    varying = rng.normal(size=(5,) + wet.shape) * wet
    got2 = M.state_independence(varying, wet, weights)
    assert got2["state_constant_share"] < 0.8
    assert got2["cross_state_corr_max"] < 0.9


def test_reduction_sensitivity_reports_a_range_not_one_number():
    """Version 2 quoted one reduction; the headline moved 2.6x under others."""
    wet, weights = _toy_geometry()
    bands = M.build_bands(wet)
    n = len(M.CONSECUTIVE_KTS)
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    nemo = np.ascontiguousarray(sign * 1.0e-3 * wet)
    lego = nemo * (1.0 + 1.0 * bands["zonal_wall"])
    rs = M.reduction_sensitivity(lego, nemo, bands["zonal_wall"],
                                 bands["interior"], weights)
    # a clean multiplicative wall excess reads 2.0 under EVERY reduction
    for key in ("area_weighted_rms", "median", "trimmed_10_90"):
        assert rs[key] == pytest.approx(2.0, rel=1e-9)
    assert rs["range_lo"] == pytest.approx(rs["range_hi"], rel=1e-9)
    assert rs["interior_cells"] == int(bands["interior"].sum())
    # a uniform interior has a participation ratio near its cell count
    assert rs["interior_participation_ratio"] > 0.5 * rs["interior_cells"]


def test_reduction_sensitivity_exposes_a_hot_cell_denominator():
    """The interior denominator is a power mean; one hot cell can carry it.

    This is the defect review found in the real data (top 1% of cells = 98% of
    the power).  The area-weighted reduction must move while the median does
    not, or the sensitivity report cannot reveal it.
    """
    wet, weights = _toy_geometry()
    bands = M.build_bands(wet)
    n = len(M.CONSECUTIVE_KTS)
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    nemo = np.ascontiguousarray(sign * 1.0e-3 * wet)
    lego = nemo * (1.0 + 1.0 * bands["zonal_wall"])
    hot = np.zeros_like(wet, dtype=float)
    hot[4, 8] = 1.0e3                       # one interior cell, huge
    nemo_hot = nemo + sign * hot
    rs = M.reduction_sensitivity(lego, nemo_hot, bands["zonal_wall"],
                                 bands["interior"], weights)
    assert rs["range_hi"] / rs["range_lo"] > 10.0
    # the hot cell is on NEMO's side, so it is NEMO's participation ratio that
    # collapses -- reporting only one side would have hidden it.
    assert rs["interior_participation_ratio_nemo"] < 5.0
    assert rs["interior_participation_ratio_min"] < 5.0
    assert rs["interior_participation_ratio"] > 5.0


def test_s3_geometry_is_computed_and_fails_when_one_wall_is_below_one(tmp_path):
    """S3 was pre-registered in v1 and never computed until round 2."""
    wet, weights = _toy_geometry()
    bands = M.build_bands(wet)
    n = len(M.CONSECUTIVE_KTS)
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    nemo = np.ascontiguousarray(sign * 1.0e-3 * wet)
    south = np.zeros_like(wet, dtype=float)
    south[bands["zonal_wall"] & (np.arange(wet.shape[0])[:, None] == 1)] = 1.0
    lego = nemo * (1.0 + 2.0 * south)       # excess at the SOUTH wall only
    per = {}
    for c in ("U", "V"):
        per[f"nemo_{c}bar_avg"] = nemo
        per[f"lego_{c}bar_avg"] = lego
        per[f"lego_{c}bar_avg_sub"] = nemo
        per[f"d{c}_avg"] = lego - nemo
        per[f"d{c}_sub"] = np.zeros_like(nemo)
    _write_maps(tmp_path, per)
    comp = M.project_component(M.load_maps(str(tmp_path)), "U", "acc_w",
                               "wetu")
    g = comp["terms"]["TOTAL"]["s3_geometry"]
    assert g["holds_at_both_end_walls"] is False
    assert g["rows_above_one"] == ["j=1"]
    assert g["rows_at_or_below_one"] == ["j=7"]


def test_operand_floors_are_computed_for_both_sides(tmp_path):
    """S6/S4 are built from the two AMPLITUDES; v2 floored only the
    difference, so a ratio of two barely-resolved operands read as a result."""
    maps = _planted(tmp_path, lego_gain=3.0)
    r = M.project_component(maps, "U", "acc_w",
                            "wetu")["terms"]["TOTAL"]["rows"]["j=1"]
    for key in ("lego_floor", "nemo_floor", "lego_floor_ratio",
                "nemo_floor_ratio", "operands_clear_floor"):
        assert key in r
    assert isinstance(r["operands_clear_floor"], bool)


def test_the_double_ratio_decomposition_is_reported(tmp_path):
    """D = (wall side ratio) / (interior side ratio).  The interior factor is a
    basin-wide amplitude fact with nothing to do with walls, and hiding it is
    what let version 2 read a null as an excess."""
    maps = _planted(tmp_path, lego_gain=3.0)
    r = M.project_component(maps, "U", "acc_w",
                            "wetu")["terms"]["TOTAL"]["rows"]["j=1"]
    assert (r["double_ratio"]
            == pytest.approx(r["side_ratio"] / r["interior_side_ratio"],
                             rel=1e-12))


# ---------------------------------------------------------------------------
# Round-3 review: four mutations still survived -- the floor WINDOW at its call
# site, the floor GATE's polarity, the meridional S5 metrics, and the sign of
# the cross-state correlation.  Each is pinned below.
# ---------------------------------------------------------------------------

def test_the_floor_call_site_uses_the_full_usable_window(tmp_path):
    """Narrowing `window=` inside `project_component` changes every floor and
    every `ok` flag, and round 2's test only exercised the helper directly."""
    wet, weights = _toy_geometry()
    bands = M.build_bands(wet)
    rng = np.random.default_rng(31)
    nemo = np.ascontiguousarray(rng.normal(size=(5,) + wet.shape) * wet)
    lego = 2.0 * nemo
    per = {}
    for c in ("U", "V"):
        per[f"nemo_{c}bar_avg"] = nemo
        per[f"lego_{c}bar_avg"] = lego
        per[f"lego_{c}bar_avg_sub"] = nemo
        per[f"d{c}_avg"] = lego - nemo
        per[f"d{c}_sub"] = np.zeros_like(nemo)
    _write_maps(tmp_path, per)
    r = M.project_component(M.load_maps(str(tmp_path)), "U", "acc_w",
                            "wetu")["terms"]["TOTAL"]["rows"]["j=1"]
    rowm = np.zeros_like(bands["zonal_wall"])
    rowm[1] = bands["zonal_wall"][1]
    area = np.where(weights > 0, weights, 0.0)
    expect = M.full_window_floor(lego - nemo, wet, rowm, weights)
    assert r["floor"] == pytest.approx(expect, rel=1e-12)
    # the same for the two OPERAND floors, which gate `operands_clear_floor`
    assert r["lego_floor"] == pytest.approx(
        M.full_window_floor(lego, wet, rowm, weights), rel=1e-12)
    assert r["nemo_floor"] == pytest.approx(
        M.full_window_floor(nemo, wet, rowm, weights), rel=1e-12)
    del area


def test_full_window_floor_averages_every_usable_sample():
    """The window parameter is GONE, so it cannot be mis-set.  What must hold
    is that all usable samples are averaged -- pinned at a state count where
    there is more than one of them (at five states there is exactly one, which
    is why mutating the old parameter was invisible rather than untested)."""
    wet, weights = _toy_geometry()
    rng = np.random.default_rng(41)
    series = rng.normal(size=(9,) + wet.shape) * wet
    per = np.atleast_1d(M.ewt.two_dt_leakage_floor(
        series, wet, area=np.where(weights > 0, weights, 0.0), mask=wet,
        return_series=True))
    assert per.size > 1                      # the average is a real average
    assert M.full_window_floor(series, wet, wet, weights) == pytest.approx(
        float(np.mean(per)), rel=1e-12)
    # taking only the first sample would be a different number
    assert M.full_window_floor(series, wet, wet, weights) != pytest.approx(
        float(per[0]), rel=1e-9)


def test_the_operand_floor_gate_has_the_right_polarity():
    """Inverting `> 2.0` to `< 2.0` flips every `ok` column, including the one
    the report's load-bearing caveat rests on, and stayed green in round 3."""
    wet, weights = _toy_geometry()
    bands = M.build_bands(wet)
    n = len(M.CONSECUTIVE_KTS)
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    # a clean two-step field has NO slow curvature, so its floor is ~0 and its
    # floor ratio is huge -> the gate must read True.
    clean = np.ascontiguousarray(sign * 1.0e-3 * wet)
    per = {}
    for c in ("U", "V"):
        per[f"nemo_{c}bar_avg"] = clean
        per[f"lego_{c}bar_avg"] = 2.0 * clean
        per[f"lego_{c}bar_avg_sub"] = clean
        per[f"d{c}_avg"] = clean
        per[f"d{c}_sub"] = np.zeros_like(clean)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        _write_maps(Path(td), per)
        r = M.project_component(M.load_maps(td), "U", "acc_w",
                                "wetu")["terms"]["TOTAL"]["rows"]["j=1"]
    assert r["lego_floor_ratio"] > 2.0 and r["nemo_floor_ratio"] > 2.0
    assert r["operands_clear_floor"] is True      # inverted gate -> False
    del bands, weights


def test_state_independence_keeps_the_sign_of_the_correlation():
    """An ANTI-correlated pair is the two-step mode itself.  Wrapping the
    correlation in abs() would report it as state-constant -- the exact
    inversion this diagnostic exists to prevent."""
    wet, weights = _toy_geometry()
    n = 4
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    base = np.linspace(1.0, 2.0, wet.size).reshape(wet.shape)
    flipping = np.ascontiguousarray(sign * base * wet)
    got = M.state_independence(flipping, wet, weights)
    assert got["cross_state_corr_min"] == pytest.approx(-1.0, abs=1e-9)
    assert got["cross_state_corr_min"] < 0.0
    # and it must NOT be reported as state-constant
    assert got["state_constant_share"] < 1e-9
    assert got["state_varying_share_rms"] == pytest.approx(1.0, rel=1e-9)


def test_the_varying_share_is_not_one_minus_the_constant_share():
    """The share is a ratio of RMS amplitudes, so 99.9% constant leaves 4.5%
    varying, not 0.1%.  Round 3 caught the report quoting the wrong one."""
    wet, weights = _toy_geometry()
    n = 5
    const = np.broadcast_to(np.ones(wet.shape), (n,) + wet.shape) * wet
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    mixed = np.ascontiguousarray(const + 0.05 * sign * wet)
    got = M.state_independence(mixed, wet, weights)
    assert got["state_constant_share"] > 0.998
    # the naive complement would be < 0.002; the correct one is ~0.05
    assert got["state_varying_share_rms"] > 0.02
    assert got["state_varying_rms"] == pytest.approx(
        got["unprojected_rms"] * got["state_varying_share_rms"], rel=1e-12)


def test_s5_uses_the_meridional_metrics_on_the_meridional_component(tmp_path,
                                                                    monkeypatch):
    """Swapping the two v metrics left round 3's suite green because the test
    stubbed only the zonal names, so the v branch never ran."""
    maps = _planted(tmp_path)
    comp = M.project_component(maps, "V", "acc_w_v", "wetv")
    wet, weights = _toy_geometry()

    class _Var:
        def __init__(self, a):
            self._a = a

        def __getitem__(self, k):
            return self._a[k]

    class _DS:
        def __init__(self, *a, **k):
            # deliberately ANISOTROPIC so a swap is visible
            self.variables = {
                "e1u": _Var(np.full((1,) + wet.shape, 9.9e9)),
                "e2u": _Var(np.full((1,) + wet.shape, 9.9e9)),
                "e1v": _Var(np.full((1,) + wet.shape, 1.0e4)),   # across
                "e2v": _Var(np.full((1,) + wet.shape, 5.0e4)),   # along
            }

        def close(self):
            pass

    import netCDF4
    monkeypatch.setattr(netCDF4, "Dataset", _DS)
    m5 = M.magnitude_check(comp, maps, "V")
    # along = e2v -> dx must be 50 km, not 10 km
    assert m5["dx_km"] == pytest.approx(50.0, rel=1e-12)
    # across = e1v -> depth = acc_w_v / e1v
    row = int(m5["row"].split("=")[1])
    depth = weights[row][weights[row] > 0] / 1.0e4
    assert m5["H_m"] == pytest.approx(float(np.median(depth)), rel=1e-12)


def test_the_side_ratio_has_its_own_subwindow_series(tmp_path):
    """A caveat on S4 has to be made of S4; round 3 found the report quoting
    S6's sub-windows to dismiss an S4 result."""
    maps = _planted(tmp_path, lego_gain=3.0)
    r = M.project_component(maps, "U", "acc_w",
                            "wetu")["terms"]["TOTAL"]["rows"]["j=1"]
    sub = r["side_ratio_subwindows"]
    assert len(sub) == len(M.CONSECUTIVE_KTS) - 2
    # a uniform gain reads exactly that gain in every window
    assert all(x == pytest.approx(3.0, rel=1e-9) for x in sub)
    # and it is NOT the same series as S6's
    assert sub != r["double_ratio_interval"]["subwindow"]
