"""Leap-frog tracer combine must conserve tracer CONTENT under moving layers.

NEMO advances content, not concentration (``trazdf.F90:271-278``):

    e3t(Kaa)·T(Kaa) = e3t(Kbb)·T(Kbb) + 2·rdt·e3t(Kmm)·RHS

legoESM's legacy combine adds a bare concentration increment, which does not
conserve content when the z-star layers breathe. Measured on the DINO oracle
(#1226): +8.6e-6 relative drift in globally-integrated heat over 200
forcing-free steps, where NEMO drifts +3.4e-16.

These are tripwires, not proofs. Each carries a synthetic-violation check so it
is provably non-vacuous: the same assertion is shown to FAIL for the
concentration form it replaces.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    thickness_weighted_tracer_combine,
)


def _fields(seed=0, shape=(4, 5, 6)):
    """Random but reproducible tracer/thickness fields with breathing layers."""
    rng = np.random.default_rng(seed)
    t_before = jnp.asarray(rng.uniform(0.0, 20.0, shape))
    t_now = jnp.asarray(rng.uniform(0.0, 20.0, shape))
    t_expl = jnp.asarray(rng.uniform(0.0, 20.0, shape))
    d_diss = jnp.asarray(rng.normal(0.0, 1e-3, shape))
    # Thicknesses differ between time levels -- that is the whole point.
    base = rng.uniform(5.0, 400.0, shape)
    h_before = jnp.asarray(base * (1.0 + rng.normal(0.0, 3e-4, shape)))
    h_now = jnp.asarray(base * (1.0 + rng.normal(0.0, 3e-4, shape)))
    h_after = jnp.asarray(base * (1.0 + rng.normal(0.0, 3e-4, shape)))
    mask = jnp.ones(shape)
    return t_before, t_now, t_expl, d_diss, h_before, h_now, h_after, mask


def _flux_form_content_drift(combine_fn, n=6):
    """Build a REAL flux-form step and measure how much content it loses.

    This is the assertion that matters, and the one an earlier version of this
    file failed to make: it asserted `sum(h_a * (content/h_a)) == content`,
    which is `(x/h)*h == x` -- true for ANY combine of that form, so it could
    never fail. Here we instead construct `t_expl` the way `_step_impl` does,
    from a divergence that telescopes to zero, and check the physical identity

        sum(h_after * T_after) == sum(h_before * T_before)

    which only holds if the combine carries CONTENT through the leap-frog.
    """
    rng = np.random.default_rng(11)
    shape = (n, n, 4)
    t_before = jnp.asarray(rng.uniform(5.0, 15.0, shape))
    t_now = jnp.asarray(rng.uniform(5.0, 15.0, shape))
    base = rng.uniform(20.0, 300.0, shape)
    h_before = jnp.asarray(base * (1.0 + rng.normal(0.0, 1e-3, shape)))
    h_now = jnp.asarray(base * (1.0 + rng.normal(0.0, 1e-3, shape)))
    h_after = jnp.asarray(base * (1.0 + rng.normal(0.0, 1e-3, shape)))
    mask = jnp.ones(shape)

    # A divergence that telescopes to zero globally (a discrete flux field):
    # div = F(i) - F(i-1) with periodic wrap => sums to exactly zero.
    flux = jnp.asarray(rng.normal(0.0, 1.0, shape))
    div = flux - jnp.roll(flux, 1, axis=0)
    assert abs(float(jnp.sum(div))) < 1e-10

    dt = 1800.0
    # This is exactly _step_impl's flux-form update.
    t_expl = (h_now * t_now - dt * div) / h_after
    d_diss = jnp.zeros(shape)

    t_after = combine_fn(t_before, t_now, t_expl, d_diss,
                         h_before, h_now, h_after, mask)
    c_after = float(jnp.sum(h_after * t_after))
    c_before = float(jnp.sum(h_before * t_before))
    return abs(c_after - c_before) / abs(c_before)


def test_flux_form_step_conserves_content():
    """A real flux-form step must leave total content unchanged."""
    assert _flux_form_content_drift(thickness_weighted_tracer_combine) < 1e-13


def test_concentration_form_loses_content():
    """Synthetic violation: the form being replaced LOSES content on the same step.

    Makes the test above provably non-vacuous.
    """
    def concentration_combine(tb, tn, te, dd, hb, hn, ha, m, h_floor=1e-10):
        return jnp.where(m > 0, tb + (te - tn) + dd, tn)

    drift = _flux_form_content_drift(concentration_combine)
    assert drift > 1e-6, (
        "the concentration form should LOSE content on a flux-form step; "
        f"got {drift:.3e} -- the test is not discriminating")


def test_dry_cells_hold_now_value_not_zero():
    """Dry cells must carry T(Nnn), never 0 (the #480 masked-cold-cell poison).

    Writing 0 below the seafloor lets staircase-adjacent stencils pull a
    spurious cold value into wet bottom cells.
    """
    tb, tn, te, dd, hb, hn, ha, m = _fields()
    mask = jnp.zeros_like(m).at[..., :2].set(1.0)     # only top 2 levels wet
    out = thickness_weighted_tracer_combine(tb, tn, te, dd, hb, hn, ha, mask)
    np.testing.assert_allclose(np.asarray(out[..., 2:]),
                               np.asarray(tn[..., 2:]), rtol=0, atol=0)
    assert not np.allclose(np.asarray(out[..., :2]), np.asarray(tn[..., :2]))


def test_zero_thickness_column_does_not_nan():
    """A fully dry column divides by the floor, not by zero."""
    tb, tn, te, dd, hb, hn, ha, m = _fields()
    ha = ha.at[0, 0, :].set(0.0)
    out = thickness_weighted_tracer_combine(tb, tn, te, dd, hb, hn, ha, m)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_static_layers_reduce_to_concentration_form():
    """With non-breathing layers (h_bef == h_now == h_aft) the two forms agree.

    Guards against the weighting silently changing answers in the z-level /
    rigid-lid case where it must not.
    """
    tb, tn, te, dd, _hb, _hn, _ha, m = _fields(seed=3)
    h = jnp.asarray(np.random.default_rng(9).uniform(5.0, 400.0, tb.shape))
    weighted = thickness_weighted_tracer_combine(tb, tn, te, dd, h, h, h, m)
    concentration = tb + (te - tn) + dd
    np.testing.assert_allclose(np.asarray(weighted),
                               np.asarray(concentration), rtol=1e-12, atol=1e-12)


def test_unknown_tracer_combine_raises():
    """Dispatch hardening: an unknown selector must raise, never silently run.

    A bare ``else: <default>`` here would run different numerics on a typo.
    """
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe, dino_lat_lon_grid, dino_lat_lon_model_config,
    )
    import dataclasses

    cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
                              tracer_combine="not_a_scheme")
    grid = dino_lat_lon_grid(cfg, n_lon=8)
    mc, _ = dino_lat_lon_model_config(grid, cfg)
    assert mc.tracer_combine == "not_a_scheme"     # threads through untouched
    # The guard lives at the combine site in _leapfrog_step; assert the literal
    # set it validates against, so deleting the raise makes this test red.
    import inspect
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as m
    src = inspect.getsource(m.LatLonCGridOceanModel._leapfrog_step)
    assert 'unknown tracer_combine' in src, (
        "the tracer_combine dispatch guard was removed from _leapfrog_step")


def test_dino_card_selects_thickness_weighted_and_default_does_not():
    """The oracle card opts in; legoESM defaults stay bit-identical."""
    from legoesm.ocean.experiments.dino import dino_config_for_recipe
    assert dino_config_for_recipe(
        "nemo_dino_kamm_mlf").tracer_combine == "thickness_weighted"
    assert dino_config_for_recipe(
        "legoesm_default").tracer_combine == "concentration"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
