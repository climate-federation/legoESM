"""Gradient-only gate at the Morrison entry of the hydrostatic microphysics
wrapper: hydrometeor masses below MG2_QSMALL, and their numbers, are held
constant for the gradient; the forward is bitwise unchanged.

Measured motivation (.physics-validator/mg2_absent_gradient_gate/README.md):
sub-threshold amounts of 1e-220 .. 1e-290 drove per-operation derivative
gains up to 1e106 and a physics-call adjoint of 1e75 against FD 1e3.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import legoesm.atmosphere.physics.microphysics.integration as integ  # noqa: E402
from legoesm.atmosphere.physics.microphysics._warm_rain import (  # noqa: E402
    MG2_QSMALL,
    mg2_absent_gradient_gate,
)
from legoesm.atmosphere.physics.microphysics.config import (  # noqa: E402
    MicrophysicsConfig,
    MorrisonConfig,
)
from legoesm.core.field import Field  # noqa: E402

SPECIES = ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g", "N_c", "N_r", "N_i")


def test_helper_value_identity_gradient_gated_and_no_nan_leak():
    m = jnp.asarray([1e-290, 1e-19, 1e-18, 1e-6])
    x = jnp.asarray([3.0, 2.0, 1.0, 5.0])
    assert bool(jnp.all(mg2_absent_gradient_gate(x, m) == x))
    g = jax.grad(lambda y: jnp.sum(mg2_absent_gradient_gate(y, m) * 7.0))(x)
    np.testing.assert_array_equal(np.asarray(g), [0.0, 0.0, 7.0, 7.0])
    # a non-finite cotangent in a gated cell must not leak into the gradient
    g_inf = jax.grad(lambda y: jnp.sum(
        jnp.sqrt(mg2_absent_gradient_gate(y, m) - jnp.asarray([3.0, 0, 0, 0]))))(x)
    assert bool(jnp.all(jnp.isfinite(g_inf)))
    assert float(g_inf[0]) == 0.0


MASS = ("q_c", "q_r", "q_i", "q_s", "q_g")
NUMBER_MASS = {"N_c": "q_c", "N_r": "q_r", "N_i": "q_i", "N_s": "q_s"}


@pytest.fixture(scope="module")
def setup():
    """Each species gets its OWN presence mask, so a number gate keyed on the
    wrong mass zeroes (or keeps) cells where its own mass says otherwise and
    the present-cell comparison fails."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_cubed_sphere(8)
    sigma = create_sigma_coordinate(10)
    state = held_suarez_init(grid, sigma)
    shp = state.T.data.shape
    rng = np.random.default_rng(0)
    present = {k: jnp.asarray(rng.uniform(size=shp) > 0.5) for k in MASS}
    tr = {"q_v": jnp.asarray(10.0 ** rng.uniform(-5, -2, shp))}
    for k in MASS:
        tr[k] = jnp.where(present[k], 10.0 ** rng.uniform(-8, -4, shp),
                          10.0 ** rng.uniform(-290, -20, shp))
    for k in NUMBER_MASS:
        tr[k] = jnp.asarray(10.0 ** rng.uniform(3, 7, shp))   # numbers always "large"
    return state, grid, sigma, tr, present


def _physics(monkeypatch=None, bypass=False):
    if bypass:
        monkeypatch.setattr(integ, "mg2_absent_gradient_gate", lambda x, m: x)
    cfg = MicrophysicsConfig(scheme="morrison",
                             morrison=MorrisonConfig(warm_rain_scheme="kk2000"))
    return integ.make_microphysics_physics(cfg, "hydrostatic", dt=300.0)


def _run(fn, setup, tr):
    state, grid, sigma, _, _ = setup
    s = state._replace(tracers={k: Field(data=v, name=k, units="")
                                for k, v in tr.items()})
    return fn(s, grid, sigma)


def _loss(fn, setup):
    def f(tr):
        t = _run(fn, setup, tr)
        out = jnp.sum(t.dT_dt.data * 1e3)
        for k, v in sorted(t.tracer_tendencies.items()):
            out = out + jnp.sum(v.data) * 1e6
        return out
    return f


def test_forward_is_bitwise_the_ungated_wrapper(setup, monkeypatch):
    tr = setup[3]
    gated = _run(_physics(), setup, tr)
    plain = _run(_physics(monkeypatch, bypass=True), setup, tr)
    assert bool(jnp.all(gated.dT_dt.data == plain.dT_dt.data))
    for k in gated.tracer_tendencies:
        assert bool(jnp.all(gated.tracer_tendencies[k].data
                            == plain.tracer_tendencies[k].data)), k


def test_gradient_zero_on_absent_and_unchanged_on_present(setup, monkeypatch):
    tr, present = setup[3], setup[4]
    g = jax.grad(_loss(_physics(), setup))(tr)
    g_plain = jax.grad(_loss(_physics(monkeypatch, bypass=True), setup))(tr)
    for k in MASS + tuple(NUMBER_MASS):
        if k not in g:
            continue
        own = present[NUMBER_MASS.get(k, k)]
        assert bool(jnp.all(jnp.isfinite(g[k]))), k
        assert bool(jnp.all(jnp.where(own, 0.0, g[k]) == 0.0)), k
        np.testing.assert_allclose(np.asarray(jnp.where(own, g[k], 0.0)),
                                   np.asarray(jnp.where(own, g_plain[k], 0.0)),
                                   rtol=1e-12, atol=0.0, err_msg=k)
    # non-vacuity, per species: every number species the scheme reads at all
    # in this mode (N_c is the specified-Nc carry here, never read) carries
    # gradient in its mass-absent cells without the gate -- numbers are large
    # there, so a gate keyed on the number itself would not zero them.
    exercised = [k for k in NUMBER_MASS
                 if k in g_plain and bool(jnp.any(g_plain[k] != 0.0))]
    assert {"N_r", "N_i"} <= set(exercised), exercised
    for k in exercised:
        m = NUMBER_MASS[k]
        assert int(jnp.sum(jnp.where(present[m], 0.0, g_plain[k]) != 0.0)) > 0, k
    assert bool(jnp.all(jnp.isfinite(g["q_v"])))


def test_qsmall_is_the_mg2_constant():
    assert MG2_QSMALL == 1.0e-18
