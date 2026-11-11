"""Direct contract tests for shared NEMO source-statement rounding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.source_rounding import nemo_source_round


@pytest.fixture(autouse=True)
def _enable_x64():
    """Run these binary64 bit-preservation checks in float64 (restored after).

    Without this, the jnp.float64 inputs below are silently truncated to
    float32 before nemo_source_round ever sees them, so the bit-preservation
    assertions would pass vacuously on already-rounded-off float32 bits.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _bits(values: jax.Array) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def test_nemo_source_round_preserves_binary64_bits_and_jit_parity() -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    values_np = np.array(
        [-np.inf, -2.0, -subnormal, -0.0, subnormal, 3.0, np.inf],
        dtype=np.float64,
    )
    values = jnp.asarray(values_np, dtype=jnp.float64)

    eager = nemo_source_round(values)
    compiled = jax.jit(nemo_source_round)(values)

    # Ground truth computed purely in NumPy (never touches JAX/x64), so a
    # float64->float32->float64 round-trip on `values` itself cannot also
    # corrupt this comparison target. The subnormal (~4.94e-324) underflows
    # to exactly 0.0 under float32, so a truncation bug flips its bit
    # pattern away from expected_bits and this assertion catches it.
    expected_bits = values_np.view(np.uint64)
    np.testing.assert_array_equal(_bits(eager), expected_bits)
    np.testing.assert_array_equal(_bits(compiled), expected_bits)
    np.testing.assert_array_equal(_bits(compiled), _bits(eager))


def test_nemo_source_round_has_identity_gradient_for_finite_inputs() -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    values = jnp.asarray([-2.0, -subnormal, -0.0, subnormal, 3.0], dtype=jnp.float64)

    gradient = jax.grad(lambda x: jnp.sum(nemo_source_round(x)))(values)

    np.testing.assert_array_equal(np.asarray(gradient), np.ones(5, dtype=np.float64))


def _polynomial(t: jax.Array, s: jax.Array) -> jax.Array:
    """An elementwise producer XLA would otherwise fuse into every consumer.

    Source-rounded statement by statement, like the two production call sites:
    an unrounded expression may be contracted differently once fused, so the
    store's bit identity is a property of source-rounded producers only.
    """
    R = nemo_source_round
    linear = R(R(1.0 + R(2.0e-4 * t)) + R(8.0e-4 * s))
    quadratic = R(R(7.0e-6 * t) * t)
    cross = R(1.0 - R(R(3.0e-5 * t) * s))
    return R(R(linear - quadratic) * cross)


def _consumers(rho: jax.Array) -> tuple[jax.Array, ...]:
    # Three kernels reading the producer: i+1 and j+1 stencils, vertical difference.
    return (jnp.roll(rho, -1, axis=0) - rho, jnp.roll(rho, -1, axis=1) - rho,
            rho[..., 1:] - rho[..., :-1])


def test_nemo_source_store_keeps_bits_and_gradient_and_compiles_a_stored_branch() -> None:
    from legoesm.core.source_rounding import nemo_source_store

    shape = (6, 8, 5)
    t = jnp.linspace(-1.5, 29.0, np.prod(shape), dtype=jnp.float64).reshape(shape)
    s = jnp.linspace(31.0, 37.5, np.prod(shape), dtype=jnp.float64).reshape(shape)[::-1]

    def plain(t_, s_):
        return _consumers(_polynomial(t_, s_))

    def stored(t_, s_):
        return _consumers(nemo_source_store(_polynomial, t_, s_))

    reference = jax.jit(plain)(t, s)
    for got in (jax.jit(stored)(t, s), stored(t, s)):
        for a, b in zip(got, reference, strict=True):
            np.testing.assert_array_equal(_bits(a), _bits(b))

    def loss(fn):
        return lambda t_, s_: sum(jnp.sum(r * r) for r in fn(t_, s_))

    g_stored = jax.jit(jax.grad(loss(stored), argnums=(0, 1)))(t, s)
    g_plain = jax.jit(jax.grad(loss(plain), argnums=(0, 1)))(t, s)
    for a, b in zip(g_stored, g_plain, strict=True):
        assert float(np.abs(np.asarray(a)).max()) > 0.0
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * float(np.abs(b).max()))

    # The store is a conditional on a runtime predicate; a plain call has none,
    # so this fails if nemo_source_store degrades to fn(*operands).
    assert "conditional" in jax.jit(stored).lower(t, s).compile().as_text()
    assert "conditional" not in jax.jit(plain).lower(t, s).compile().as_text()


def test_geometric_depth_pressure_density_is_stored() -> None:
    """The NEMO EOS-80 geometric-depth density read by the pressure gradient goes
    through nemo_source_store (ocean_pe_latlon_cgrid._bc_geometry_and_density):
    one conditional more than the in-situ arm, which has no such store (the
    land-mask fills store their mask terms in both arms)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import compute_frozen_geom_density
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(12, 24)
    z_coord = create_ocean_z_star(n_levels=6, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(grid, z_coord)

    def conditionals(eos_depth):
        cfg = LatLonCGridOceanConfig.from_flat(eos="nemo_eos80", eos_depth=eos_depth)
        return jax.jit(
            lambda st: compute_frozen_geom_density(st, grid, z_coord, cfg)[2:]
        ).lower(state).compile().as_text().count(" conditional(")

    assert conditionals("geometric") - conditionals("insitu") == 1


def test_bn2_alpha_beta_is_stored() -> None:
    """NEMO rab (alpha/beta) inside bn2 goes through nemo_source_store
    (eos.compute_buoyancy_frequency_nemo_bn2) when source-rounded, and only
    then; without the store this compiled program has no conditional."""
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2

    shape = (5, 7, 6)
    t = jnp.linspace(-1.5, 29.0, np.prod(shape), dtype=jnp.float64).reshape(shape)
    s = jnp.linspace(31.0, 37.5, np.prod(shape), dtype=jnp.float64).reshape(shape)
    gdept = jnp.linspace(5.0, 4000.0, shape[-1], dtype=jnp.float64)
    gdepw = 0.5 * (gdept[1:] + gdept[:-1])

    def text(source_rounded):
        return jax.jit(lambda t_, s_: compute_buoyancy_frequency_nemo_bn2(
            t_, s_, gdept, gdepw, eos_form="eos80", e3w_source="depth_difference",
            source_rounded=source_rounded)).lower(t, s).compile().as_text()

    assert " conditional(" in text(True)
    assert " conditional(" not in text(False)


def test_r3t_stretch_is_stored() -> None:
    """NEMO (1 + r3t) goes through nemo_source_store (eos.nemo_r3t_stretch)
    with the card's source-rounded r3t product, and only then; without the
    store this compiled program has no conditional."""
    from types import SimpleNamespace

    from legoesm.ocean.eos import nemo_r3t_stretch

    eta = jnp.linspace(-0.5, 0.5, 35, dtype=jnp.float64).reshape(5, 7)
    h = jnp.linspace(0.0, 4000.0, 35, dtype=jnp.float64).reshape(5, 7)
    for stored_product in (True, False):
        z_coord = SimpleNamespace(linear_free_surface=False,
                                  nemo_r3t_stored_product=stored_product)
        text = jax.jit(lambda e, b: nemo_r3t_stretch(
            z_coord, e, b, evaluation="nemo_reciprocal")).lower(eta, h).compile().as_text()
        assert (" conditional(" in text) is stored_product


def test_neumann_fill_mask_terms_are_stored_for_a_runtime_mask() -> None:
    """neumann_fill_cgrid stores each pass's mask-only terms (one conditional
    per pass) when the mask is a runtime array; a compile-time-constant mask
    folds them, and the store with them."""
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid

    rng = np.random.default_rng(1)
    mask = jnp.asarray((rng.random((12, 16)) > 0.3).astype(np.float64))
    f = jnp.asarray(rng.random((12, 16, 5)))
    runtime = jax.jit(lambda f_, m_: neumann_fill_cgrid(f_, m_))
    constant = jax.jit(lambda f_: neumann_fill_cgrid(f_, mask))
    assert runtime.lower(f, mask).compile().as_text().count(" conditional(") == 3
    assert " conditional(" not in constant.lower(f).compile().as_text()
    np.testing.assert_array_equal(
        np.asarray(runtime(f, mask))[np.asarray(mask) > 0.5],
        np.asarray(f)[np.asarray(mask) > 0.5])


def test_mle_density_is_stored(monkeypatch) -> None:
    """The MLE density of the through-FCT route goes through nemo_source_store
    in the step (ocean_model_latlon_cgrid ... _step_impl); traced on the GYRE
    lane card with ORCA2's L5 MLE."""
    import sys

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as om
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_gyre_zco_lane_certified_card,
        orca2_l5_mle,
    )

    callers = []
    store = om.nemo_source_store

    def recording_store(fn, *operands):
        callers.append(sys._getframe(1).f_code.co_name)
        return store(fn, *operands)

    monkeypatch.setattr(om, "nemo_source_store", recording_store)
    prev = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        card = build_gyre_zco_lane_certified_card()
        base = card.recipe.model_config
        cfg = base._replace(physics=base.physics._replace(
            mle=orca2_l5_mle()._replace(through_fct=True, nemo_association=True)))
        model = om.LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
        state = model._seed_tke_preclosure_carry(card.recipe.initial_state)
        jax.eval_shape(lambda st: model._step_jitted(st, card.dt_s), state)
    finally:
        set_policy(prev)
    assert "_step_impl" in callers
