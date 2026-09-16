"""The process ledger restricted to a vertical band.

The column ledger integrates over the whole column and is therefore blind to a
vertical-REDISTRIBUTION bias: measured on the production AMIP, convection's
column water row is exactly zero (correct for a scheme that only moves water up
and down) while the tropical free troposphere is twice as moist as observed.
Restricting the integral to a band is what makes the instrument able to say
which process supplies a LAYER, and these tests pin that it does so without
changing the full-column answer when no band is asked for.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.diagnostics.process_ledger import (
    apply_level_weight, column_store_snapshot_column, ledger_entry_column,
    sigma_band_weight,
)

NLEV = 10
SIGMA_HALF = jnp.linspace(0.0, 1.0, NLEV + 1)
DSIGMA = jnp.diff(SIGMA_HALF)
P_S = jnp.asarray([1.0e5, 9.0e4])


def test_band_weight_is_one_inside_zero_outside():
    w = np.asarray(sigma_band_weight(SIGMA_HALF, 0.3, 0.7))
    assert np.allclose(w[3:7], 1.0)
    assert np.allclose(w[:3], 0.0) and np.allclose(w[7:], 0.0)


def test_band_edges_are_fractional_and_adjacent_bands_partition_the_column():
    """A band edge inside a layer must split it, and two abutting bands must
    together weight every layer exactly once — otherwise the rows of two bands
    would not sum to the full-column row."""
    lower = np.asarray(sigma_band_weight(SIGMA_HALF, 0.0, 0.35))
    upper = np.asarray(sigma_band_weight(SIGMA_HALF, 0.35, 1.0))
    assert lower[3] == pytest.approx(0.5)
    assert upper[3] == pytest.approx(0.5)
    np.testing.assert_allclose(lower + upper, np.ones(NLEV))


def test_full_column_band_reproduces_the_unbanded_ledger_exactly():
    """The byte-identical default: asking for the whole column, or asking for
    no band at all, must give the same number."""
    rng = np.random.default_rng(0)
    dq = jnp.asarray(rng.normal(size=(2, NLEV)) * 1e-6)
    dT = jnp.asarray(rng.normal(size=(2, NLEV)) * 1e-4)
    plain = ledger_entry_column(dq, dT, P_S, DSIGMA)
    banded = ledger_entry_column(dq, dT, P_S, DSIGMA,
                                 level_weight=sigma_band_weight(SIGMA_HALF, 0.0, 1.0))
    np.testing.assert_allclose(np.asarray(plain), np.asarray(banded), rtol=0, atol=0)
    none_band = ledger_entry_column(dq, dT, P_S, DSIGMA, level_weight=None)
    np.testing.assert_allclose(np.asarray(plain), np.asarray(none_band), rtol=0, atol=0)


def test_two_bands_sum_to_the_full_column():
    """The property that makes the banded ledger trustworthy: attributing a
    layer cannot change the total."""
    rng = np.random.default_rng(1)
    dq = jnp.asarray(rng.normal(size=(2, NLEV)) * 1e-6)
    full = np.asarray(ledger_entry_column(dq, None, P_S, DSIGMA))
    lo = np.asarray(ledger_entry_column(
        dq, None, P_S, DSIGMA, level_weight=sigma_band_weight(SIGMA_HALF, 0.0, 0.42)))
    hi = np.asarray(ledger_entry_column(
        dq, None, P_S, DSIGMA, level_weight=sigma_band_weight(SIGMA_HALF, 0.42, 1.0)))
    np.testing.assert_allclose(lo + hi, full, rtol=1e-12)


def test_band_isolates_a_planted_layer():
    """A tendency placed in ONE layer must appear only in the band containing
    it — the test fails if the weight is applied after integration."""
    dq = np.zeros((2, NLEV)); dq[:, 5] = 1e-6
    dq = jnp.asarray(dq)
    inside = np.asarray(ledger_entry_column(
        dq, None, P_S, DSIGMA, level_weight=sigma_band_weight(SIGMA_HALF, 0.5, 0.6)))
    outside = np.asarray(ledger_entry_column(
        dq, None, P_S, DSIGMA, level_weight=sigma_band_weight(SIGMA_HALF, 0.0, 0.5)))
    assert np.all(inside[:, 0] > 0.0)
    np.testing.assert_allclose(outside[:, 0], 0.0, atol=1e-30)


def test_snapshot_honours_the_band_too():
    """The dynamics and clips rows come from paired SNAPSHOTS, so if those
    ignored the band those two rows would be full-column while the physics rows
    were banded -- a silently inconsistent table."""
    rng = np.random.default_rng(2)
    q = jnp.asarray(np.abs(rng.normal(size=(2, NLEV))) * 1e-3)
    T = jnp.asarray(250.0 + rng.normal(size=(2, NLEV)))
    w = sigma_band_weight(SIGMA_HALF, 0.0, 0.5)
    full = np.asarray(column_store_snapshot_column(P_S, DSIGMA, T, q))
    band = np.asarray(column_store_snapshot_column(P_S, DSIGMA, T, q, level_weight=w))
    assert np.all(band[:, 0] < full[:, 0])
    assert np.all(band[:, 1] < full[:, 1])


def test_wrong_length_weight_is_refused():
    with pytest.raises(ValueError, match="levels"):
        apply_level_weight(jnp.zeros((2, NLEV)), jnp.ones(NLEV + 3))


@pytest.mark.parametrize("lo,hi", [(0.5, 0.5), (0.7, 0.3), (-0.1, 0.5), (0.5, 1.2)])
def test_invalid_band_is_refused(lo, hi):
    with pytest.raises(ValueError, match="sigma_lo"):
        sigma_band_weight(SIGMA_HALF, lo, hi)


def test_cli_flag_and_config_field_round_trip():
    """The band must be reachable from a run, and asking for it without the
    ledger itself must be refused rather than silently ignored."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run"))
    from run_amip import build_arg_parser, _postprocess_args, build_config_from_args

    parser = build_arg_parser()
    base = ["--dataset", "analytical", "--grid-type", "voronoi",
            "--discretization", "mpas"]
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(base), parser))
    assert cfg.output.budget_ledger_sigma_band is None

    cfg = build_config_from_args(_postprocess_args(parser.parse_args(
        base + ["--budget-ledger", "--budget-ledger-sigma-band", "0.3", "0.7"]), parser))
    assert cfg.output.budget_ledger is True
    assert cfg.output.budget_ledger_sigma_band == (0.3, 0.7)
    cfg.validate_strict()


def test_the_physics_factory_accepts_the_weight():
    """The weight must reach the physics ledger rows, not just the config."""
    import inspect
    from legoesm.atmosphere.physics import combined
    # BOTH the factory and the PUBLIC wrapper the driver actually calls: the
    # first wiring added it only to the inner factory, so every banded run died
    # on an unexpected-keyword TypeError after the band had been computed.
    for fn in (combined._make_hydrostatic_combined, combined.make_physics):
        assert "budget_ledger_level_weight" in inspect.signature(fn).parameters, fn


def test_the_dycore_config_carries_the_weight():
    """The dynamics and clips rows come from the DYCORE's snapshots, so it must
    carry the same weight or the table stops summing to the column change."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
    )
    c = MPASPrimitiveEquationConfig()
    assert c.budget_ledger_level_weight is None
    assert c._replace(budget_ledger_level_weight=jnp.ones(5)
                      ).budget_ledger_level_weight is not None


def test_the_driver_reads_the_sigma_coordinate_it_actually_has():
    """Regression: the first wiring read a sigma_coord attribute the driver does
    not have, so every banded run died at setup with an AttributeError. Pin the
    attribute name against the one the driver builds."""
    import inspect
    from legoesm.driver import model_driver
    src = inspect.getsource(model_driver.ModelDriver)
    assert "self.sigma = create_sigma_coordinate(" in src
    assert "self.sigma_coord.sigma_half" not in src
    assert "sigma_band_weight(\n                self.sigma.sigma_half" in src


def test_the_dycore_actually_receives_the_band_from_a_config():
    """The field existing on the dycore config is not enough — the factory must
    SET it. Measured once when it did not: dynamics read 1.1504 identically in
    a full-column, a free-troposphere and a boundary-layer run, so the table did
    not partition and no physics could be read off it."""
    from legoesm.driver.component_factory import _ledger_level_weight
    from legoesm.grids.vertical import create_sigma_coordinate

    class _Out:
        budget_ledger_sigma_band = None

    class _Cfg:
        output = _Out()

    sigma = create_sigma_coordinate(NLEV)
    assert _ledger_level_weight(_Cfg(), sigma) is None

    _Out.budget_ledger_sigma_band = (0.3, 0.7)
    w = np.asarray(_ledger_level_weight(_Cfg(), sigma))
    assert w.shape == (NLEV,)
    assert w.sum() > 0.0 and w.sum() < NLEV      # a real band, not all-or-nothing
    _Out.budget_ledger_sigma_band = None


def test_every_physics_build_site_receives_the_band():
    """EVERY make_physics call in the driver must pass the weight, not just the
    first. Measured when one did not: the production run subcycles radiation,
    so 47 of every 48 steps ran a SECOND, unbanded physics function and the
    ledger rows came out a 2%-banded, 98%-unbanded blend that looked plausible
    and partitioned nowhere."""
    import inspect, re
    from legoesm.driver import model_driver
    src = inspect.getsource(model_driver)
    calls = [m.start() for m in re.finditer(r"\bmake_physics\(", src)]
    assert calls, "no make_physics call found — test cannot fail"
    for pos in calls:
        tail = src[pos:pos + 1200]
        depth, end = 0, None
        for i, ch in enumerate(tail):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        assert end is not None, "unbalanced make_physics call"
        call = tail[:end]
        if "budget_ledger=" in call:
            assert "budget_ledger_level_weight=" in call, (
                "a make_physics call passes the ledger flag but not the band:\n"
                + call)
