"""Regressions for three spectral-PE dycore audit fixes (2026-07-12).

1. ``soft_clip_lnps`` interior identity — the unscaled softplus clamp
   silently mapped p_s = 1e5 Pa to 95333 Pa (4.8% bias well inside the
   physical range).
2. Flux-form surface-pressure tendency (Hoskins & Simmons 1975):
   d(lnps)/dt integrand must be ``div + v·grad(lnps)``, not ``div``
   alone — the advective form leaves an O(v·grad p_s) spurious global
   mass tendency whenever p_s is nonuniform.
3. ``implicit_hyperdiff=True`` under SSP/RK integrators — previously the
   implicit filter was applied only on the leapfrog path, so SSP runs
   with ``implicit_hyperdiff=True`` had NO vor/div/T diffusion at all.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralPEConfig,
    SpectralPrimitiveEquationModel,
    isothermal_rest_state_spectral,
    soft_clip_lnps,
    spectral_pe_tendencies,
    _LNPS_MIN,
    _LNPS_MAX,
)

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="spectral tests need JAX_ENABLE_X64=1",
)


# ---------------------------------------------------------------------------
# 1. lnps soft clip: identity in the interior, active only near bounds
# ---------------------------------------------------------------------------

class TestSoftClipLnps:
    def test_interior_identity(self):
        """Physical surface pressures must pass through unchanged."""
        for p in (1.0e3, 2.0e4, 5.0e4, 1.0e5, 1.05e5):
            lnp = math.log(p)
            p_mapped = float(jnp.exp(soft_clip_lnps(jnp.float64(lnp))))
            # Regression: unscaled clip mapped 1e5 -> 95333.33 Pa.
            assert abs(p_mapped - p) < 1e-6 * p, (
                f"clip distorted interior: {p} Pa -> {p_mapped} Pa"
            )

    def test_clamps_below_lower_bound(self):
        lnp = math.log(1.0)  # 1 Pa, far below the 100 Pa bound
        out = float(soft_clip_lnps(jnp.float64(lnp)))
        assert out > lnp  # pulled up toward the bound
        assert out <= _LNPS_MIN + 0.5

    def test_clamps_above_upper_bound(self):
        lnp = math.log(1.0e7)  # far above the 2e6 Pa bound
        out = float(soft_clip_lnps(jnp.float64(lnp)))
        assert out < lnp
        assert out >= _LNPS_MAX - 0.5

    def test_monotone_and_differentiable(self):
        xs = jnp.linspace(_LNPS_MIN - 2.0, _LNPS_MAX + 2.0, 201)
        ys = soft_clip_lnps(xs)
        # Non-decreasing everywhere (saturates flat at the far tails in
        # fp64 — that IS the clamp); strictly increasing in the interior.
        assert bool(jnp.all(jnp.diff(ys) >= 0))
        interior = jnp.linspace(_LNPS_MIN + 0.5, _LNPS_MAX - 0.5, 101)
        assert bool(jnp.all(jnp.diff(soft_clip_lnps(interior)) > 0))
        g = jax.vmap(jax.grad(soft_clip_lnps))(xs)
        assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# 2. Flux-form continuity: global mass tendency ~ 0 with nonuniform p_s
# ---------------------------------------------------------------------------

def _hill_state_with_wind(grid, sigma_coord):
    """Isothermal state over a Gaussian hill (nonuniform p_s) + winds."""
    lat2d = grid.lat2d
    lon2d = grid.lon2d
    phis = 2000.0 * constants.g * jnp.exp(
        -((lat2d - 0.5) ** 2 + (lon2d - jnp.pi) ** 2) / 0.3 ** 2
    )
    state = isothermal_rest_state_spectral(
        grid, sigma_coord, phis=phis, perturbation_amplitude=0.0,
    )
    # Inject smooth low-n rotational + divergent flow so v·grad(lnps) != 0.
    ls = np.asarray(grid.ls)
    vor = state.vor_hat.data
    div = state.div_hat.data
    for n_target, amp in ((3, 4e-6), (5, 3e-6)):
        idx = int(np.argmax(ls == n_target))
        vor = vor.at[idx, :].set(amp)
        div = div.at[idx, :].set(0.5 * amp)
    return state._replace(
        vor_hat=state.vor_hat.replace(data=vor),
        div_hat=state.div_hat.replace(data=div),
    )


def test_global_mass_tendency_vanishes_with_nonuniform_ps():
    """∮ p_s · d(lnps)/dt dA ≈ 0: flux form telescopes; the advective
    form leaves an O(1) fraction of the local tendency magnitude."""
    grid = create_gaussian_grid(21)
    sigma_coord = create_sigma_coordinate(8)
    config = SpectralPEConfig(hyperdiff_coeff=0.0, fix_mass=False)
    state = _hill_state_with_wind(grid, sigma_coord)

    tend = spectral_pe_tendencies(state, grid, sigma_coord, config)

    lnps = sh_synthesis(grid, state.lnps_hat.data)
    p_s = jnp.exp(lnps)
    dlnps = sh_synthesis(grid, tend.lnps_hat.data)
    area = grid.grid_area

    # dp_s/dt = p_s * dlnps/dt; global integral must vanish (mass).
    net = float(jnp.sum(p_s * dlnps * area))
    activity = float(jnp.sum(jnp.abs(p_s * dlnps) * area))
    assert activity > 0.0, "test vacuous: dynamics produced no ps tendency"
    rel = abs(net) / activity
    # Advective form gives rel ~ O(0.1-1).  The flux form is exact up to
    # Gaussian-quadrature error on the exp(lnps)-weighted products (p_s is
    # not band-limited), measured 7.7e-8 at T21/L8 over a 2000 m hill —
    # six orders below the advective-form signal.
    assert rel < 5e-7, f"global mass tendency residual too large: rel={rel:.3e}"


# ---------------------------------------------------------------------------
# 3. implicit_hyperdiff must act under SSP/RK integrators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integrator", ["ssp_rk3", "ssp_rk54"])
def test_implicit_hyperdiff_damps_under_ssp(integrator):
    grid = create_gaussian_grid(21)
    sigma_coord = create_sigma_coordinate(5)
    dt = 300.0

    # nu chosen so one step damps the n=n_max mode by ~e^-2.3 ≈ 0.1.
    n_max = grid.n_max
    eig = (n_max * (n_max + 1) / grid.radius ** 2) ** 2
    nu = 2.3 / (dt * eig)

    def _run(nu_run):
        cfg = SpectralPEConfig(
            hyperdiff_coeff=nu_run,
            hyperdiff_order=2,
            implicit_hyperdiff=True,
            spectral_filter_strength=0.0,
            time_integrator=integrator,
            fix_mass=False,
        )
        model = SpectralPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma_coord, config=cfg,
        )
        state = isothermal_rest_state_spectral(
            grid, sigma_coord, perturbation_amplitude=0.0,
        )
        ls = np.asarray(grid.ls)
        idx = int(np.argmax(ls == n_max))
        vor = state.vor_hat.data.at[idx, 2].set(1e-8)
        state = state._replace(vor_hat=state.vor_hat.replace(data=vor))
        for _ in range(3):
            state = model.step(state, dt)
        return float(jnp.abs(state.vor_hat.data[idx, 2]))

    amp_diffused = _run(nu)
    amp_free = _run(0.0)
    assert amp_free > 0.0
    # Three steps of exp(-2.3) each: expect ~1e-3 ratio; require < 0.2 to
    # be robust to the (identical) explicit dynamics in both runs.
    assert amp_diffused < 0.2 * amp_free, (
        f"implicit hyperdiff inert under {integrator}: "
        f"{amp_diffused:.3e} vs {amp_free:.3e}"
    )


# ---------------------------------------------------------------------------
# 4. JIT-cache freshness: dt change retraces; target-mass reset is honored
# ---------------------------------------------------------------------------

def _hyperdiff_model_and_state(nu, dt):
    grid = create_gaussian_grid(21)
    sigma_coord = create_sigma_coordinate(5)
    cfg = SpectralPEConfig(
        hyperdiff_coeff=nu, hyperdiff_order=2, implicit_hyperdiff=True,
        spectral_filter_strength=0.0, time_integrator="ssp_rk3",
        fix_mass=False,
    )
    model = SpectralPrimitiveEquationModel(
        grid=grid, sigma_coord=sigma_coord, config=cfg,
    )
    state = isothermal_rest_state_spectral(
        grid, sigma_coord, perturbation_amplitude=0.0,
    )
    ls = np.asarray(grid.ls)
    idx = int(np.argmax(ls == grid.n_max))
    vor = state.vor_hat.data.at[idx, 2].set(1e-8)
    return model, state._replace(vor_hat=state.vor_hat.replace(data=vor)), idx


def test_dt_change_uses_fresh_filters():
    """codex 2026-07-12: dt was traced while the dt-dependent hyperdiff/
    SI/sponge/tracer filters were closure-captured, so a dt change reused
    the STALE filters of the first trace.  dt is now static: stepping a
    reused model at a new dt must be bit-identical to a fresh model."""
    grid = create_gaussian_grid(21)
    eig = (grid.n_max * (grid.n_max + 1) / grid.radius ** 2) ** 2
    nu = 2.3 / (300.0 * eig)

    reused, state, idx = _hyperdiff_model_and_state(nu, 300.0)
    reused.step(state, 300.0)          # bake the dt=300 trace
    out_reused = reused.step(state, 600.0)

    fresh, state2, _ = _hyperdiff_model_and_state(nu, 600.0)
    out_fresh = fresh.step(state2, 600.0)

    for f in ("vor_hat", "div_hat", "T_hat", "lnps_hat"):
        a = getattr(out_reused, f).data
        b = getattr(out_fresh, f).data
        assert bool(jnp.array_equal(a, b)), (
            f"stale dt-cached program: {f} differs after dt 300->600"
        )


def test_set_target_mass_is_honored_by_compiled_step():
    """codex 2026-07-12: the anchored-mass target was a closure constant
    of the compiled step, so set_target_mass()/reset after the first
    trace was silently ignored.  It is now a traced argument."""
    grid = create_gaussian_grid(21)
    sigma_coord = create_sigma_coordinate(5)
    cfg = SpectralPEConfig(
        hyperdiff_coeff=0.0, fix_mass=True, anchor_mass_to_initial=True,
        time_integrator="ssp_rk3",
    )
    model = SpectralPrimitiveEquationModel(
        grid=grid, sigma_coord=sigma_coord, config=cfg,
    )
    state = isothermal_rest_state_spectral(
        grid, sigma_coord, perturbation_amplitude=0.0,
    )
    out1 = model.step(state, 300.0)          # snapshots + bakes trace
    m0 = float(model._compute_initial_mass(state))
    m1 = float(model._compute_initial_mass(out1))
    assert abs(m1 - m0) / m0 < 1e-12

    model.set_target_mass(jnp.float64(2.0 * m0))
    out2 = model.step(state, 300.0)          # SAME trace, new target
    m2 = float(model._compute_initial_mass(out2))
    assert abs(m2 - 2.0 * m0) / m0 < 1e-9, (
        f"set_target_mass ignored by compiled step: mass {m2} vs {2*m0}"
    )
