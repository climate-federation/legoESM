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


# CLM5 C4 PFT ids (see legoesm.land.surface_params.CLM5_PFT_NAMES): 14=c4_grass, 16=crop_c4.
_C4_PFT_IDS = (14, 16)


def _c3_and_c4():
    """Two warm, well-lit C3 archetypes (broadleaf tree ids 4/7) + the two C4 PFTs
    (14=c4_grass, 16=crop_c4) -- exercises BOTH discrimination pathways in one forward."""
    return _table([4, 7, 14, 16], [298.0, 293.0, 300.0, 299.0],
                  [230.0, 210.0, 240.0, 235.0], t_amp=[2.0, 6.0, 3.0, 3.0])


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
    LEAF delta13C in the physical C3 band [-34, -22] permil for warm, well-lit C3 archetypes.
    A C4-magnitude (~ -13) or wrong-sign (~ -8 / positive) value FAILS -- and the band is
    asserted at the STATED C3 edge -22 (not a looser -20), so a C4-ish value leaking into the
    C3 band is caught.  Prints the per-archetype summary."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.d13c_forward import simulate_archetype_d13c
    from legoesm.land.stomata import StomataConfig

    table = _lit_c3()
    d13c = np.asarray(simulate_archetype_d13c(
        table, StomataConfig(enabled=True, stomata_model="ball_berry")))
    print(f"\n[d13c gate] per-archetype simulated leaf delta13C [permil] = "
          f"{np.round(d13c, 2)} (mean {d13c.mean():.2f})")
    assert d13c.shape == (4,)
    assert np.all(np.isfinite(d13c))
    # physical C3 band [-34, -22] at the STATED edges (not C4 ~ -13, not undiscriminated air ~ -8)
    assert np.all(d13c <= -22.0), d13c
    assert np.all(d13c >= -34.0), d13c
    # the cover of warm well-lit C3 trees sits in the typical C3 leaf band
    assert -34.0 < float(d13c.mean()) < -22.0, d13c


def test_simulated_d13c_more_negative_with_higher_ball_berry_slope():
    """PHYSICAL monotonicity in the TRAINED param: a larger Ball-Berry slope g1_bb -> more
    open stomata -> higher Ci -> higher Ci/Ca -> MORE discrimination -> MORE NEGATIVE leaf
    delta13C.  This is the water-use-efficiency sign the delta13C stream trains."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.d13c_forward import simulate_archetype_d13c
    from legoesm.land.stomata import StomataConfig

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
    from legoesm.land.stomata import StomataConfig

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
    from legoesm.land.stomata import StomataConfig

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
    from legoesm.land.stomata import StomataConfig
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


# ---------------------------------------------------------------------------
# CORRECTNESS GATE (C4 pathway): sign + range + the phi lever (Farquhar-Cerling)
# ---------------------------------------------------------------------------
def test_pure_c4_discrimination_endpoints_pin_c4_constants():
    """C4 form Delta_C4 = a + (b4 + (b3-s)*phi - a)*Ci/Ca pinned at the Ci/Ca endpoints, so a
    typo in a C4 provenance constant (b4=-5.7, s=1.8) fails LOUDLY.  At Ci/Ca=0, Delta=a=4.4 ->
    delta13C=air-a=-12.4 (SAME as C3, both diffusion-only at Ci/Ca=0).  At Ci/Ca=1, phi=0.21:
    Delta = b4+(b3-s)*phi = -5.7 + 25.2*0.21 = -0.408 -> delta13C = -8 - (-0.408) = -7.592."""
    from legoesm.land.carbon.d13c_forward import leaf_d13c_c4_from_ci_ca

    npt.assert_allclose(float(leaf_d13c_c4_from_ci_ca(0.0, 0.21)), -12.4, atol=1e-9)
    npt.assert_allclose(float(leaf_d13c_c4_from_ci_ca(1.0, 0.21)), -7.592, atol=1e-6)


def test_pure_c4_discrimination_at_setpoint_is_in_c4_band_and_less_negative_than_c3():
    """At the C4-characteristic setpoint (Ci/Ca=0.4, phi=0.21) the C4 leaf delta13C is in the
    physical C4 band (~ -16..-10 permil) and DISTINCTLY less negative than any C3 value:
    Delta = 4.4 + (-5.7 + 25.2*0.21 - 4.4)*0.4 = 4.4 - 1.9232 = 2.4768 -> delta13C = -10.4768.
    phi is the C4 lever: MORE leaky -> MORE discrimination -> MORE negative delta13C
    (d(delta13C)/d(phi) = -(b3-s)*Ci/Ca = -25.2*0.4 = -10.08 < 0)."""
    from legoesm.land.carbon.d13c_forward import leaf_d13c_c4_from_ci_ca

    d = float(leaf_d13c_c4_from_ci_ca(0.4, 0.21))
    npt.assert_allclose(d, -10.4768, atol=1e-4)
    assert -16.0 <= d <= -10.0, d       # in the C4 band, decisively out of the C3 band
    # phi monotonicity (the C4 water-use-efficiency / delta13C lever)
    assert leaf_d13c_c4_from_ci_ca(0.4, 0.35) < leaf_d13c_c4_from_ci_ca(0.4, 0.15)


def test_simulated_d13c_selects_c3_and_c4_pathways_in_their_bands():
    """The full forward returns a FAITHFUL C4 value for C4 archetypes (Farquhar-Cerling) and
    the C3 value for C3 archetypes, selected per-archetype by is_c4 -- NOT the C3 form for all.
    C3 in ~[-34,-22], C4 distinctly less negative in ~[-16,-10]; a C4 in the C3 band (or a C3
    in the C4 band) FAILS.  Asserts the C4 archetypes are exactly the {c4_grass, crop_c4} ids
    and prints the per-archetype C3-vs-C4 summary."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.d13c_forward import simulate_archetype_d13c
    from legoesm.land.stomata import StomataConfig
    from legoesm.land.surface_params import is_c4_pft_id

    table = _c3_and_c4()
    d13c = np.asarray(simulate_archetype_d13c(
        table, StomataConfig(enabled=True, stomata_model="ball_berry")))
    is_c4 = is_c4_pft_id(table.pft_id)
    # the selector flags EXACTLY the two C4 PFT ids (14 c4_grass, 16 crop_c4)
    npt.assert_array_equal(is_c4, np.isin(np.asarray(table.pft_id), _C4_PFT_IDS))
    c3v, c4v = d13c[~is_c4], d13c[is_c4]
    print(f"\n[d13c C3 vs C4] C3={np.round(c3v, 2)} (mean {c3v.mean():.2f}); "
          f"C4={np.round(c4v, 2)} (mean {c4v.mean():.2f})")
    assert np.all(np.isfinite(d13c))
    # C3 archetypes: in the physical C3 leaf band, asserted at the STATED edge -22 (not -20)
    assert np.all(c3v <= -22.0) and np.all(c3v >= -34.0), c3v
    # C4 archetypes: distinctly LESS negative, in the physical C4 leaf band
    assert np.all(c4v <= -10.0) and np.all(c4v >= -16.0), c4v
    # strict separation: every C4 value is less negative than every C3 value (a swap FAILS)
    assert c4v.min() > c3v.max(), np.stack([c3v.max(), c4v.min()])


def test_d13c_both_levers_reach_the_loss_on_a_mixed_c3_c4_table():
    """Grad of a mixed-table delta13C loss reaches BOTH pathway levers: the C3 Ball-Berry slope
    g1_bb (through the coupled Farquhar Ci of the C3 archetypes) AND the C4 leakiness phi
    (through the C4 branch) -- finite and non-zero, and DECOUPLED (each lever only moves its
    own pathway's archetypes)."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import D13CConfig
    from legoesm.land.carbon.d13c_forward import simulate_archetype_d13c
    from legoesm.land.stomata import StomataConfig

    table = _c3_and_c4()
    target = jnp.asarray([-27.0, -26.0, -12.5, -12.5], dtype=jnp.float64)

    def loss(g1_bb, phi):
        st = StomataConfig(enabled=True, stomata_model="ball_berry", g1_bb=g1_bb)
        d = D13CConfig(phi_c4_leakiness=phi)
        return jnp.mean((simulate_archetype_d13c(table, st, d) - target) ** 2)

    val, (g_g1, g_phi) = jax.value_and_grad(loss, argnums=(0, 1))(9.0, 0.21)
    assert np.isfinite(float(val))
    assert abs(float(g_g1)) > 1e-10, g_g1     # C3 stomatal WUE lever moves the loss
    assert abs(float(g_phi)) > 1e-10, g_phi   # C4 leakiness lever moves the loss


def test_c3_c4_pathways_are_fully_decoupled_by_the_selector():
    """The jnp.where(is_c4,...) selection guarantees BOTH decoupling directions:
    (i) a C4 archetype's delta13C does NOT depend on the C3 Ball-Berry slope g1_bb (the C4
        branch uses a FIXED C4 Ci/Ca), while a C3 archetype's DOES; and
    (ii) a C3 archetype's delta13C does NOT depend on the C4 leakiness phi, while a C4
        archetype's DOES.
    (A one-sided check would miss e.g. an accidental phi->C3 coupling.)"""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import D13CConfig
    from legoesm.land.carbon.d13c_forward import simulate_archetype_d13c
    from legoesm.land.stomata import StomataConfig
    from legoesm.land.surface_params import is_c4_pft_id

    table = _c3_and_c4()
    is_c4 = is_c4_pft_id(table.pft_id)

    # (i) vary g1_bb (C3 lever) at fixed phi -> C4 unchanged, C3 moves
    lo_g1 = np.asarray(simulate_archetype_d13c(
        table, StomataConfig(enabled=True, stomata_model="ball_berry", g1_bb=5.0)))
    hi_g1 = np.asarray(simulate_archetype_d13c(
        table, StomataConfig(enabled=True, stomata_model="ball_berry", g1_bb=14.0)))
    npt.assert_allclose(lo_g1[is_c4], hi_g1[is_c4], rtol=1e-12)         # C4 invariant under g1_bb
    assert np.any(np.abs(hi_g1[~is_c4] - lo_g1[~is_c4]) > 1e-6)         # C3 moves with g1_bb

    # (ii) vary phi (C4 lever) at fixed g1_bb -> C3 unchanged, C4 moves
    st = StomataConfig(enabled=True, stomata_model="ball_berry")
    lo_phi = np.asarray(simulate_archetype_d13c(table, st, D13CConfig(phi_c4_leakiness=0.15)))
    hi_phi = np.asarray(simulate_archetype_d13c(table, st, D13CConfig(phi_c4_leakiness=0.35)))
    npt.assert_allclose(lo_phi[~is_c4], hi_phi[~is_c4], rtol=1e-12)     # C3 invariant under phi
    assert np.any(np.abs(hi_phi[is_c4] - lo_phi[is_c4]) > 1e-6)         # C4 moves with phi
