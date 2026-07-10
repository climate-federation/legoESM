"""Unit tests for the per-archetype simulated leaf-delta13C forward
(``legoesm.land.carbon.d13c_forward``) -- including the CORRECTNESS GATE (sign + range +
monotonicity of the leaf carbon-isotope discrimination).

Compute-node scale (JIT-compiles the coupled-Farquhar solve + its reverse-mode); run via the
sbatch/srun wrapper, NOT the login node.  ``JAX_ENABLE_X64=1``; forced CPU here.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import numpy.testing as npt


def _table(pft_id, mat_k, sw, *, t_amp=None, map_yr=None):
    from legoesm.land.carbon.global_init import ArchetypeTable
    pft_id = np.asarray(pft_id, dtype=int)
    n = pft_id.shape[0]
    return ArchetypeTable(
        pft_id=pft_id,
        mat_k=np.asarray(mat_k, dtype=float),
        map_yr=np.asarray(map_yr if map_yr is not None else [1200.0] * n, dtype=float),
        t_seasonal_amp_k=np.asarray(t_amp if t_amp is not None else [6.0] * n, dtype=float),
        aridity=np.asarray([1.0] * n, dtype=float),
        sw_mean_w=np.asarray(sw, dtype=float),
        soil_class=np.asarray(["loam"] * n, dtype=object),
    )


def _lit_c3():
    """A few warm, well-lit C3 (tree) archetypes with active growing-season photosynthesis."""
    return _table([1, 4, 5, 7], [283.0, 298.0, 295.0, 291.0],
                  [200.0, 240.0, 230.0, 210.0], t_amp=[10.0, 2.0, 3.0, 6.0])


# ---------------------------------------------------------------------------
# CORRECTNESS GATE (pure discrimination): sign + range + monotonicity
# ---------------------------------------------------------------------------
def test_pure_discrimination_endpoints_pin_farquhar_constants():
    """leaf delta13C = delta13C_air - Delta with Delta = a + (b-a)*Ci/Ca: the Ci/Ca=0 and =1
    endpoints pin a (diffusion), b (Rubisco) and delta13C_air (Farquhar 1989), so a typo in a
    provenance constant fails LOUDLY.  delta13C(0) = air - a = -8 - 4.4 = -12.4;
    delta13C(1) = air - b = -8 - 27 = -35."""
    from legoesm.land.carbon.d13c_forward import leaf_d13c_from_ci_ca

    npt.assert_allclose(float(leaf_d13c_from_ci_ca(0.0)), -12.4, atol=1e-9)
    npt.assert_allclose(float(leaf_d13c_from_ci_ca(1.0)), -35.0, atol=1e-9)


def test_pure_discrimination_in_c3_range_and_strictly_decreasing():
    """Over the physiological well-watered..moderately-stressed C3 Ci/Ca band (~0.5..0.9) the
    leaf delta13C is in the physical C3 range (~ -22..-34 permil) and MONOTONIC DECREASING
    (higher Ci/Ca -> more discrimination -> MORE NEGATIVE delta13C; d(delta13C)/d(Ci/Ca) =
    -(b-a) = -22.6 permil, exact).  (Ci/Ca ~ 0.4 is the extreme water-stressed edge bordering
    the C4 regime, so the physiological C3 band starts ~0.5.)"""
    import jax.numpy as jnp
    from legoesm.land.carbon.d13c_forward import leaf_d13c_from_ci_ca

    ci_ca = jnp.asarray([0.5, 0.6, 0.7, 0.8, 0.9])
    d13c = np.asarray(leaf_d13c_from_ci_ca(ci_ca))
    # physical C3 leaf range
    assert np.all(d13c <= -22.0) and np.all(d13c >= -34.0), d13c
    # strictly decreasing
    assert np.all(np.diff(d13c) < 0.0), d13c
    # exact slope -(b - a) = -(27 - 4.4) = -22.6 permil per unit Ci/Ca
    npt.assert_allclose(np.diff(d13c), -22.6 * 0.1, rtol=1e-9)


# ---------------------------------------------------------------------------
# CORRECTNESS GATE (full forward): range + physical monotonicity in the trained param
# ---------------------------------------------------------------------------
def test_simulated_d13c_in_physical_c3_range():
    """The full forward (model's own coupled Farquhar Ci/Ca -> C3 discrimination) gives a
    LEAF delta13C in the physical C3 band for warm, well-lit C3 archetypes -- decisively C3
    (< -20) and never below the theoretical max discrimination (> -35).  A C4-magnitude
    (~ -13) or wrong-sign (~ -8 / positive) value FAILS.  Prints the per-archetype summary."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.d13c_forward import simulate_archetype_d13c
    from legoesm.land.carbon.stomata import StomataConfig

    table = _lit_c3()
    d13c = np.asarray(simulate_archetype_d13c(
        table, StomataConfig(enabled=True, stomata_model="ball_berry")))
    print(f"\n[d13c gate] per-archetype simulated leaf delta13C [permil] = "
          f"{np.round(d13c, 2)} (mean {d13c.mean():.2f})")
    assert d13c.shape == (4,)
    assert np.all(np.isfinite(d13c))
    # decisively C3-magnitude + right sign (not C4 ~ -13, not undiscriminated air ~ -8)
    assert np.all(d13c < -20.0), d13c
    assert np.all(d13c > -35.0), d13c
    # the cover of warm well-lit C3 trees sits in the typical C3 leaf band
    assert -34.0 < float(d13c.mean()) < -22.0, d13c


def test_simulated_d13c_more_negative_with_higher_ball_berry_slope():
    """PHYSICAL monotonicity in the TRAINED param: a larger Ball-Berry slope g1_bb -> more
    open stomata -> higher Ci -> higher Ci/Ca -> MORE discrimination -> MORE NEGATIVE leaf
    delta13C.  This is the water-use-efficiency sign the delta13C stream trains."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.d13c_forward import simulate_archetype_d13c
    from legoesm.land.carbon.stomata import StomataConfig

    table = _lit_c3()
    lo = np.asarray(simulate_archetype_d13c(
        table, StomataConfig(enabled=True, stomata_model="ball_berry", g1_bb=5.0)))
    hi = np.asarray(simulate_archetype_d13c(
        table, StomataConfig(enabled=True, stomata_model="ball_berry", g1_bb=14.0)))
    print(f"\n[d13c monotonic] g1_bb=5 -> {np.round(lo, 2)}; g1_bb=14 -> {np.round(hi, 2)}")
    # higher slope -> more negative everywhere (no violation) and strictly on the cover mean
    assert np.all(hi <= lo + 1e-9), np.stack([lo, hi])
    assert float(hi.mean()) < float(lo.mean()) - 0.1, np.stack([lo, hi])


def test_build_d13c_forward_matches_simulate():
    """The trainer's precompute path (frozen forcing, re-solved Farquhar) equals the all-in-one
    forward for any StomataConfig."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.d13c_forward import build_d13c_forward, simulate_archetype_d13c
    from legoesm.land.carbon.stomata import StomataConfig

    table = _lit_c3()
    d13c_fn = build_d13c_forward(table)
    for cfg in (
        StomataConfig(enabled=True, stomata_model="ball_berry"),
        StomataConfig(enabled=True, stomata_model="ball_berry", g1_bb=12.0, g0=0.02),
    ):
        npt.assert_allclose(
            np.asarray(d13c_fn(cfg)), np.asarray(simulate_archetype_d13c(table, cfg)),
            rtol=1e-10)


def test_d13c_forward_differentiable_in_ball_berry_slope():
    """AD de-risk: grad of a delta13C MSE loss w.r.t. the Ball-Berry slope g1_bb is FINITE and
    NON-ZERO -- Ci is the traced stomatal quantity, so the gradient reaches g1 through the
    coupled Farquhar solve (single-step; no spin-up scan)."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.d13c_forward import simulate_archetype_d13c
    from legoesm.land.carbon.stomata import StomataConfig

    table = _lit_c3()
    target = jnp.asarray([-27.0, -29.0, -28.0, -26.5], dtype=jnp.float64)

    def loss(g1_bb):
        cfg = StomataConfig(enabled=True, stomata_model="ball_berry", g1_bb=g1_bb)
        return jnp.mean((simulate_archetype_d13c(table, cfg) - target) ** 2)

    val, grad = jax.value_and_grad(loss)(9.0)
    assert np.isfinite(float(val))
    assert np.isfinite(float(grad))
    assert abs(float(grad)) > 1e-10, grad


def test_d13c_forward_differentiable_through_param_override_path():
    """The trainer's override path is differentiable: grad of a delta13C MSE loss w.r.t. the
    land.stomata trainables (applied via apply_param_overrides INSIDE the loss, production
    defaults untouched) is finite and moves through g1_bb."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.d13c_forward import simulate_archetype_d13c
    from legoesm.land.carbon.stomata import StomataConfig
    from legoesm.training.param_collector import (
        apply_param_overrides,
        build_trainable_params,
    )

    table = _lit_c3()
    params = build_trainable_params(
        active_scheme_keys={"land.stomata"}, tier="extended", dtype=jnp.float64)
    target = jnp.asarray([-27.0, -29.0, -28.0, -26.5], dtype=jnp.float64)

    def loss(p):
        ov = p.to_overrides().get("land.stomata", {})
        cfg = apply_param_overrides(
            StomataConfig(enabled=True, stomata_model="ball_berry"), ov)
        return jnp.mean((simulate_archetype_d13c(table, cfg) - target) ** 2)

    val, grads = eqx.filter_value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    gmax = {c.field: float(jnp.max(jnp.abs(grads.raw_values[c.name])))
            for c in params.constraints}
    assert all(np.isfinite(v) for v in gmax.values()), gmax
    assert gmax["g1_bb"] > 1e-10, gmax   # the Ball-Berry slope moves the delta13C loss
