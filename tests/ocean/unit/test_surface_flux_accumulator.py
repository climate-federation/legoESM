"""``--flux-accumulate``: the applied surface fluxes a heat budget needs.

WHY THIS EXISTS.  A snapshot pair gives the STATE change but not the boundary
fluxes that drove it, so no heat budget can be closed from snapshots alone --
``omip_conservation_closure.py`` says so in its own docstring and refuses to
fake the missing term.  Three independent sources converged on that gap: codex
(endpoint states cannot recover transport correlations), GLM (the state
signature has several sufficient causes), and a direct attempt whose residual
exceeded its own vertical term and voided.

These tests pin the behaviours that would silently corrupt a budget:

  1. the window mean is a MEAN, not a sum -- a sum divided by the wrong count
     is the classic way a flux diagnostic looks plausible and is wrong by the
     number of steps in the window;
  2. ``drain`` RESETS, so consecutive windows are independent rather than
     cumulative;
  3. an empty window writes NO key -- zero is a physically meaningful flux and
     must never be manufactured by the instrument (the "plausible-looking
     value is more dangerous than a NaN" rule);
  4. a ``None`` field is skipped rather than zero-filled, for the same reason;
  5. the flag stays OUT of the restart fingerprint, so turning a read-only
     diagnostic on cannot make a chained leg un-resumable -- a failure this
     driver has already had once, for the profile sampler.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _driver():
    from scripts.run import run_omip_core2
    return run_omip_core2


def _sf(q_net=None, sw_down=None, tau_x=None, tau_y=None):
    return SimpleNamespace(q_net=q_net, sw_down=sw_down,
                           tau_x=tau_x, tau_y=tau_y)


def test_window_mean_is_a_mean_not_a_sum():
    acc = _driver()._SurfaceFluxAccumulator()
    acc.add(_sf(q_net=np.full((2, 2), 10.0)))
    acc.add(_sf(q_net=np.full((2, 2), 20.0)))
    acc.add(_sf(q_net=np.full((2, 2), 30.0)))
    out = acc.drain()
    assert out["flux_mean_n_steps"] == 3
    # 20.0 is the mean; 60.0 would be the sum.
    np.testing.assert_allclose(out["q_net_mean"], 20.0, rtol=1e-12)


def test_drain_resets_so_consecutive_windows_are_independent():
    acc = _driver()._SurfaceFluxAccumulator()
    acc.add(_sf(q_net=np.full((2, 2), 100.0)))
    first = acc.drain()
    acc.add(_sf(q_net=np.full((2, 2), 1.0)))
    second = acc.drain()
    np.testing.assert_allclose(first["q_net_mean"], 100.0, rtol=1e-12)
    # Without the reset the second window would carry the first one's 100.
    np.testing.assert_allclose(second["q_net_mean"], 1.0, rtol=1e-12)
    assert second["flux_mean_n_steps"] == 1


def test_an_empty_window_writes_no_key_rather_than_a_zero():
    acc = _driver()._SurfaceFluxAccumulator()
    assert acc.drain() == {}
    # And a window of steps that carried no forcing object at all is still
    # empty rather than zero-valued.
    acc.add(None)
    assert acc.drain() == {}


def test_a_none_field_is_skipped_not_zero_filled():
    acc = _driver()._SurfaceFluxAccumulator()
    acc.add(_sf(q_net=np.full((2, 2), 5.0), tau_x=None))
    out = acc.drain()
    assert "q_net_mean" in out
    assert "tau_x_mean" not in out


def test_all_four_fields_are_carried():
    acc = _driver()._SurfaceFluxAccumulator()
    acc.add(_sf(q_net=np.ones((2, 2)), sw_down=2 * np.ones((2, 2)),
                tau_x=3 * np.ones((2, 2)), tau_y=4 * np.ones((2, 2))))
    out = acc.drain()
    for k, v in (("q_net_mean", 1.0), ("sw_down_mean", 2.0),
                 ("tau_x_mean", 3.0), ("tau_y_mean", 4.0)):
        np.testing.assert_allclose(out[k], v, rtol=1e-12)


def test_snapshot_extra_returns_none_when_every_diagnostic_is_off():
    d = _driver()
    args = SimpleNamespace(kprofile_snapshots=False)
    assert d._snapshot_extra(args, None, None, None, None, None, None) is None


def test_snapshot_extra_carries_the_flux_means_through():
    d = _driver()
    acc = d._SurfaceFluxAccumulator()
    acc.add(_sf(q_net=np.full((2, 2), 7.0)))
    args = SimpleNamespace(kprofile_snapshots=False)
    extra = d._snapshot_extra(args, None, None, None, None, None, acc)
    np.testing.assert_allclose(extra["q_net_mean"], 7.0, rtol=1e-12)
    assert extra["flux_mean_n_steps"] == 1


def test_window_duration_is_recoverable_from_the_file():
    """A budget needs SECONDS, and the step count alone does not give them.

    codex: flux_mean_n_steps plus the endpoint step identify which steps were
    included but not how long the window was.
    """
    acc = _driver()._SurfaceFluxAccumulator()
    for _ in range(4):
        acc.add(_sf(q_net=np.ones((2, 2))))
    out = acc.drain(dt=150.0)
    assert out["flux_mean_window_s"] == pytest.approx(600.0)
    assert out["flux_mean_n_steps"] == 4


def test_window_duration_is_omitted_when_dt_is_unknown():
    acc = _driver()._SurfaceFluxAccumulator()
    acc.add(_sf(q_net=np.ones((2, 2))))
    assert "flux_mean_window_s" not in acc.drain()


def test_the_scan_block_lane_refuses_the_flag_rather_than_ignoring_it():
    """A flag that is silently inert on a lane is the defect class this repo
    bans outright: the run looks successful and the snapshots carry nothing.

    codex found that the --scan-block path returns before the accumulator is
    constructed. The driver must REFUSE, exactly as it already does for
    --kprofile-snapshots. Scoped to the guard block so an unrelated mention of
    the flag elsewhere cannot satisfy it.
    """
    from pathlib import Path
    import scripts.run.run_omip_core2 as mod

    src = Path(mod.__file__).read_text()
    assert "if args.flux_accumulate and use_scan:" in src
    guard = src.split("if args.flux_accumulate and use_scan:", 1)[1][:600]
    assert "raise SystemExit" in guard
    assert "--scan-block" in guard


def test_flag_defaults_off_and_round_trips():
    p = _driver()._build_arg_parser()
    assert p.parse_args(["--grid", "tripole"]).flux_accumulate is False
    assert p.parse_args(["--grid", "tripole",
                         "--flux-accumulate"]).flux_accumulate is True


def test_flag_is_excluded_from_the_restart_fingerprint():
    """Enabling a READ-ONLY diagnostic must not break a chained leg.

    The driver already lost a two-day diurnal control this way once, when the
    profile sampler's flag was hashed into the restart fingerprint and a leg
    could not resume its own baseline.

    ``_RESTART_FP_EXCLUDE`` is a LOCAL inside ``main()``, so it cannot be
    imported; this reads the source block instead.  A source-inspection test is
    weak by construction, so it is scoped to the exact block rather than the
    whole file -- the flag name appearing anywhere else in the driver (and it
    appears in several places) must NOT satisfy it.
    """
    from pathlib import Path
    import scripts.run.run_omip_core2 as mod

    src = Path(mod.__file__).read_text().splitlines()
    start = next(i for i, ln in enumerate(src)
                 if "_RESTART_FP_EXCLUDE = frozenset({" in ln)
    end = next(i for i in range(start, len(src)) if src[i].strip() == "})")
    # codex: the first version of this test passed on a COMMENT that merely
    # contained the flag name. Strip comments and require the quoted literal,
    # so only a real set member satisfies it.
    code = [ln.split("#", 1)[0] for ln in src[start:end + 1]]
    block = "\n".join(code)
    for flag in ('"flux_accumulate"', '"mld_accumulate"', '"state_accumulate"'):
        assert flag in block, (
            f"the read-only {flag} flag must be excluded from the restart "
            "fingerprint or enabling it makes a chained leg un-resumable")


# ---------------------------------------------------------------------------
# --mld-accumulate: the same window, for the mixed-layer depth.
#
# WHY.  NEMO publishes mldr10_1 as a FIVE-DAY MEAN and we compared a SNAPSHOT
# against it.  That confound has produced retracted numbers in this campaign
# twice (the z0 surface-TKE arm was refuted by it; a "K_M 5%" claim turned out
# to be a wind-lull snapshot).  Averaging our MLD over the same window removes
# it -- but only if the two sides also share a THRESHOLD, which is what the
# last test here pins.
# ---------------------------------------------------------------------------


def test_mld_rides_the_same_window_and_the_same_single_count():
    """One step is one count even when both a flux and an MLD are added."""
    acc = _driver()._SurfaceFluxAccumulator()
    acc.add(_sf(q_net=np.full((2, 2), 10.0)), mld=np.full((2, 2), 40.0))
    acc.add(_sf(q_net=np.full((2, 2), 30.0)), mld=np.full((2, 2), 60.0))
    out = acc.drain()
    assert out["flux_mean_n_steps"] == 2
    np.testing.assert_allclose(out["q_net_mean"], 20.0, rtol=1e-12)
    np.testing.assert_allclose(out["mld_mean"], 50.0, rtol=1e-12)


def test_an_mld_only_window_is_normalised_by_its_own_count():
    """--mld-accumulate without --flux-accumulate must still give a MEAN.

    The count is incremented once per ``add``, not once per field, so a
    caller that supplies only the MLD divides by the number of steps rather
    than by zero or by one.
    """
    acc = _driver()._SurfaceFluxAccumulator()
    for v in (10.0, 20.0, 60.0):
        acc.add(None, mld=np.full((2, 2), v))
    out = acc.drain()
    assert out["flux_mean_n_steps"] == 3
    np.testing.assert_allclose(out["mld_mean"], 30.0, rtol=1e-12)
    assert "q_net_mean" not in out


def test_adding_nothing_does_not_advance_the_count():
    """Both flags off -> the accumulator is not constructed, but if a caller
    ever reaches ``add`` with neither payload the window must not inflate."""
    acc = _driver()._SurfaceFluxAccumulator()
    acc.add(_sf(q_net=np.full((2, 2), 4.0)), mld=None)
    acc.add(None, None)
    acc.add(None, None)
    out = acc.drain()
    assert out["flux_mean_n_steps"] == 1
    np.testing.assert_allclose(out["q_net_mean"], 4.0, rtol=1e-12)


def test_mld_flag_is_excluded_from_the_restart_fingerprint():
    """Same reason as the flux flag: a read-only diagnostic must not make a
    chained leg un-resumable."""
    from pathlib import Path
    import scripts.run.run_omip_core2 as mod

    src = Path(mod.__file__).read_text().splitlines()
    start = next(i for i, ln in enumerate(src)
                 if "_RESTART_FP_EXCLUDE = frozenset({" in ln)
    end = next(i for i in range(start, len(src)) if src[i].strip() == "})")
    block = "\n".join(ln.split("#", 1)[0] for ln in src[start:end + 1])
    assert '"mld_accumulate"' in block


def test_the_scan_block_lane_refuses_the_mld_flag_too():
    from pathlib import Path
    import scripts.run.run_omip_core2 as mod

    src = Path(mod.__file__).read_text()
    assert "if args.mld_accumulate and use_scan:" in src
    guard = src.split("if args.mld_accumulate and use_scan:", 1)[1][:600]
    assert "raise SystemExit" in guard
    assert "--scan-block" in guard


def test_a_lane_without_interface_depths_refuses_rather_than_writing_nothing():
    """The silently-inert-flag class, refused.

    Returning ``None`` here would leave the snapshot with no ``mld_mean``
    while the run reported success -- exactly the failure the seven MPAS flag
    guards were written for.  It must abort on the first step instead.
    """
    import pytest as _pytest
    z_coord = SimpleNamespace()          # no z_half_ref
    with _pytest.raises(SystemExit):
        _driver()._mld_now(SimpleNamespace(), z_coord)


def test_the_online_mld_uses_the_scorer_threshold_not_the_library_default():
    """0.01 wrt 10 m (NEMO mldr10_1), NOT the library's 0.03 default.

    NON-VACUITY: the profile below is chosen so the two thresholds give
    DIFFERENT depths, and the test asserts both -- that ours equals the 0.01
    answer AND that the 0.03 answer differs by more than a level.  A profile
    where the two agree would let a wrong threshold pass.
    """
    import jax.numpy as jnp
    from legoesm.ocean.diagnostics import mixed_layer_depth

    nlev = 10
    z_half = jnp.arange(nlev + 1, dtype=jnp.float64) * 10.0     # 0..100 m
    z_c = jnp.abs(0.5 * (z_half[:-1] + z_half[1:]))             # 5..95 m
    # Weak, uniform stratification: ~0.01 kg/m^3 per level, so the 0.01 and
    # 0.03 crossings are three levels apart.
    T = (20.0 - 0.05 * jnp.arange(nlev, dtype=jnp.float64))[None, None, :]
    T = jnp.broadcast_to(T, (2, 2, nlev))
    S = jnp.full((2, 2, nlev), 35.0)
    Hb = jnp.full((2, 2), 100.0)
    mask = jnp.ones((2, 2))
    state = SimpleNamespace(T=T, S=S, H_bathy=Hb, land_mask=mask)

    got = _driver()._mld_now(state, SimpleNamespace(z_half_ref=z_half))

    wet = ((z_c[None, None, :] < Hb[..., None]) & (mask[..., None] > 0.5)
           ).astype(T.dtype)
    want = mixed_layer_depth(T, S, z_c, delta_sigma=0.01,
                             wet_mask=wet, bottom_depth=Hb)
    deep = mixed_layer_depth(T, S, z_c, delta_sigma=0.03,
                             wet_mask=wet, bottom_depth=Hb)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12)
    assert float(np.asarray(deep).mean() - np.asarray(want).mean()) > 10.0, (
        "the two thresholds must disagree on this profile or the test is "
        "vacuous")


def test_state_mean_is_the_window_mean_of_T_and_S():
    """--state-accumulate: T_mean/S_mean are MEANS over the window of the 3-D
    fields, shared with the flux window; snapshots without it carry no key."""
    acc = _driver()._SurfaceFluxAccumulator()
    st1 = SimpleNamespace(T=SimpleNamespace(data=np.full((2, 2, 3), 10.0)),
                          S=SimpleNamespace(data=np.full((2, 2, 3), 34.0)))
    st2 = SimpleNamespace(T=SimpleNamespace(data=np.full((2, 2, 3), 12.0)),
                          S=SimpleNamespace(data=np.full((2, 2, 3), 36.0)))
    acc.add(None, state=st1)
    acc.add(None, state=st2)
    out = acc.drain()
    np.testing.assert_allclose(out["T_mean"], 11.0, rtol=1e-12)
    np.testing.assert_allclose(out["S_mean"], 35.0, rtol=1e-12)
    assert out["flux_mean_n_steps"] == 2
    assert "q_net_mean" not in out
    assert acc.drain() == {}


def test_state_flag_round_trips_and_scan_lane_refuses_it():
    from pathlib import Path
    import scripts.run.run_omip_core2 as mod
    p = _driver()._build_arg_parser()
    assert p.parse_args(["--grid", "tripole"]).state_accumulate is False
    assert p.parse_args(["--grid", "tripole", "--state-accumulate"]).state_accumulate is True
    src = Path(mod.__file__).read_text()
    # THIS guard's block only (up to the next top-level `if` in main), so a
    # neighbouring guard's raise cannot satisfy it (codex).
    guard = src.split("if args.state_accumulate and use_scan:", 1)[1]
    guard = guard.split("\n    if ", 1)[0]
    assert "raise SystemExit" in guard
    assert "--state-accumulate" in guard


def test_state_sampled_after_the_step_at_both_host_loop_sites():
    """XIOS averages the completed state of each step; the add must follow
    the step call at both host-loop call sites, not precede it."""
    from pathlib import Path
    import scripts.run.run_omip_core2 as mod
    src = Path(mod.__file__).read_text()
    sites = src.split("state=(state if args.state_accumulate else None))")[:-1]
    assert len(sites) == 2, len(sites)
    for site in sites:
        tail = site[-900:]
        assert ("state = model.step(" in tail) or ("state = _ocean_step(" in tail), tail
