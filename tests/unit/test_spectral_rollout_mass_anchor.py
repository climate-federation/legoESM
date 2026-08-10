"""The dry-mass anchor must act in ``spectral_rollout``, not only in the class.

WHY THIS FILE EXISTS. ``SpectralPrimitiveEquationModel`` has carried an
anchored-mass fixer since iter-3, but the AIMIP arms that use the spectral
dycore (classical, column_nn) never touch that class — they integrate through
the functional ``spectral_rollout``, which had no fixer. Both arms lose ~16 hPa
of area-weighted mean sea-level pressure over a 10-day forecast (2017 WB2
scorecards) while the arm with no dycore loses 0.8 hPa. A first attempt at the
fix set the config flags and stopped there; the flags reached a code path the
forecast never executes, so the A/B would have measured nothing.

``spectral_rollout`` has TWO step bodies — rad-gated and ungated — and the two
AIMIP arms take DIFFERENT ones (classical sets aimip_rad_update_interval 36 and
is gated; column_nn supplies no radiation fn and is not). Both are covered here
for that reason.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    SpectralPEConfig,
    global_dry_mass,
)
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.training.neural_gcm_spectral import (  # noqa: E402
    spectral_rollout,
)
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    isothermal_rest_state_spectral,
)

_N_MAX = 10
_N_LEV = 4
_DT = 900.0


@pytest.fixture(scope="module")
def setup():
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(_N_LEV)
    state = isothermal_rest_state_spectral(
        grid, sigma, T_init=280.0, p_s_init=1.0e5)
    return grid, sigma, state


def _perturbed(state, grid):
    """Give lnps a nonzero non-mean structure.

    A rest state whose ``lnps`` is a pure global constant is the ONE case where
    the anchor is trivially satisfied and every filter is a no-op on it — a
    control that perturbs a zero. Perturbing a non-(0,0) coefficient makes the
    spectral filter (which damps ``lnps`` modes and therefore moves
    ``∫exp(lnps)dA``) actually able to change the mass.
    """
    data = state.lnps_hat.data
    idx = int(np.argmax(np.asarray(grid.ls) == 2))
    data = data.at[idx].add(0.02)
    return state._replace(lnps_hat=state.lnps_hat.replace(data=data))


def _zero_physics(state, grid_, sigma_, **_kw):
    """Physics that returns an all-zero tendency of the right structure."""
    return jax.tree.map(jnp.zeros_like, state)


def _cfg(**kw):
    return SpectralPEConfig(
        hyperdiff_coeff=1e14, time_integrator="ssp_rk3", **kw)


def _drift_ppm(grid, before, after):
    m0 = float(global_dry_mass(grid, before.lnps_hat.data))
    m1 = float(global_dry_mass(grid, after.lnps_hat.data))
    return 1.0e6 * (m1 - m0) / m0


@pytest.mark.parametrize("gated", [False, True])
def test_anchor_holds_global_mass_in_the_rollout(setup, gated):
    """With the anchor on, ∫p_s dA returns to the IC value every step.

    Tolerance is 1e-3 ppm of the initial mass: the correction is exact in
    exp/log arithmetic, so what is left is fp64 round-off in one global sum,
    not a physical residual.
    """
    grid, sigma, rest = setup
    state = _perturbed(rest, grid)
    filt = jnp.exp(-1e-2 * (jnp.asarray(grid.ls, jnp.float64) / _N_MAX) ** 8)

    kw = {}
    if gated:
        kw = {"rad_physics_fn": _zero_physics, "rad_update_interval": 3}

    on = spectral_rollout(
        state, _zero_physics, grid, sigma,
        _cfg(fix_mass=True, anchor_mass_to_initial=True),
        _DT, 6, None, filt, **kw)
    assert abs(_drift_ppm(grid, state, on)) < 1.0e-3


@pytest.mark.parametrize("gated", [False, True])
def test_anchor_off_is_byte_identical_to_before(setup, gated):
    """Default-off must not perturb a single existing arm.

    Asserted as EXACT equality, not a tolerance: with the flags off the anchor
    branch is not traced at all, so any difference would mean the change leaked
    into the default path.
    """
    grid, sigma, rest = setup
    state = _perturbed(rest, grid)
    filt = jnp.exp(-1e-2 * (jnp.asarray(grid.ls, jnp.float64) / _N_MAX) ** 8)
    kw = {}
    if gated:
        kw = {"rad_physics_fn": _zero_physics, "rad_update_interval": 3}

    a = spectral_rollout(state, _zero_physics, grid, sigma, _cfg(),
                         _DT, 6, None, filt, **kw)
    b = spectral_rollout(
        state, _zero_physics, grid, sigma,
        _cfg(fix_mass=False, anchor_mass_to_initial=False),
        _DT, 6, None, filt, **kw)
    np.testing.assert_array_equal(
        np.asarray(a.lnps_hat.data), np.asarray(b.lnps_hat.data))


def test_the_unanchored_rollout_really_does_drift(setup):
    """Non-vacuity: the anchored test above must be measuring something.

    If the unanchored rollout conserved mass on its own, the anchor test would
    pass for the wrong reason. The spectral filter damping non-mean ``lnps``
    modes changes ``∫exp(lnps)dA``, which is exactly the leak the anchor closes.
    """
    grid, sigma, rest = setup
    state = _perturbed(rest, grid)
    # Strong filter so the leak is unambiguous at 6 steps.
    filt = jnp.exp(-2.0 * (jnp.asarray(grid.ls, jnp.float64) / _N_MAX) ** 4)
    off = spectral_rollout(state, _zero_physics, grid, sigma, _cfg(),
                           _DT, 6, None, filt)
    assert abs(_drift_ppm(grid, state, off)) > 1.0e-2


def test_anchor_also_acts_in_the_prescribed_sst_amip_rollout(setup):
    """``spectral_amip_rollout`` is a THIRD step body and needs the anchor too.

    It is the path the classical AMIP inference runner and the AMIP fine-tune
    take, i.e. the runs where a secular surface-pressure drift compounds
    longest. Wiring only ``spectral_rollout`` would have left ``fix_mass`` a
    knob that silently does nothing there (codex round 2).
    """
    from legoesm.training.neural_gcm_spectral import spectral_amip_rollout

    grid, sigma, rest = setup
    state = _perturbed(rest, grid)
    filt = jnp.exp(-1e-2 * (jnp.asarray(grid.ls, jnp.float64) / _N_MAX) ** 8)

    def _non_rad(s, g, sc, *, phys_state=None, forcing=None):
        return jax.tree.map(jnp.zeros_like, s)

    def _rad(s, g, sc, *, forcing=None):
        return jax.tree.map(jnp.zeros_like, s)

    class _PhysStub:
        """Minimal stand-in: the rollout only ``_replace``s one field on it."""

        def _replace(self, **_kw):
            return self

    kw = dict(
        sst_col=jnp.zeros(grid.n_lat * grid.n_lon),
        sizing_phys_state=_PhysStub(),
        rad_update_interval=3,
        sponge_factor=None,
        spectral_filter=filt,
    )
    on = spectral_amip_rollout(
        state, _non_rad, _rad, grid, sigma,
        _cfg(fix_mass=True, anchor_mass_to_initial=True), _DT, 6, **kw)
    assert abs(_drift_ppm(grid, state, on)) < 1.0e-3

    off = spectral_amip_rollout(
        state, _non_rad, _rad, grid, sigma, _cfg(), _DT, 6, **kw)
    # Non-vacuity for THIS body: unanchored must not already be conserving.
    assert abs(_drift_ppm(grid, state, off)) > 0.0


def test_anchor_is_jit_and_grad_safe(setup):
    """No object state: the target is traced, so jit and grad both work.

    The class-side fixer refuses to snapshot inside a trace because it would
    store a tracer on the model; the rollout captures the target from its own
    initial state instead, which is what makes it usable in training.
    """
    grid, sigma, rest = setup
    state = _perturbed(rest, grid)
    cfg = _cfg(fix_mass=True, anchor_mass_to_initial=True)

    # Differentiate w.r.t. a REAL scalar amplitude on the lnps perturbation:
    # lnps_hat is complex128 (spectral coefficients), and jax.grad demands a
    # real-valued output, so the scalar knob is the honest formulation rather
    # than a holomorphic=True that would not match how the trainer uses this.
    idx = int(np.argmax(np.asarray(grid.ls) == 2))
    base = state.lnps_hat.data

    def loss(amp):
        s = state._replace(
            lnps_hat=state.lnps_hat.replace(data=base.at[idx].add(amp)))
        out = spectral_rollout(s, _zero_physics, grid, sigma, cfg,
                               _DT, 2, None, None)
        return jnp.sum(jnp.abs(out.lnps_hat.data) ** 2)

    g = float(jax.jit(jax.grad(loss))(0.01))
    assert np.isfinite(g)
    assert abs(g) > 0.0
