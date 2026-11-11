"""Moist microphysics hook for the pseudo-incompressible dycore (composable, reused
spectral-LES adapter). A supersaturated column must condense (q_c↑), release latent heat
(θ↑), conserve total water, and integrate stably operator-split with the dynamics step.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.les import pseudo_incompressible_plane as pip
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig


def _grid(nz=40):
    cfg = pip.PseudoIncompressibleConfig(
        nx=8, ny=4, nz=nz, Lx=800.0, Ly=400.0, Lz=2000.0, theta_ref0=300.0,
        scheme="van_leer", moist=True, n_tracers=3, sgs="none", surface="free")
    return pip.make_grid(cfg)


def test_make_microphysics_unknown_scheme_raises():
    g = _grid()
    with pytest.raises((ValueError, KeyError)):
        pip.make_microphysics(g, MicrophysicsConfig(scheme="bogus"), dt=1.0)


def test_supersaturated_column_condenses_and_heats():
    g = _grid()
    micro = pip.make_microphysics(g, MicrophysicsConfig(scheme="kessler"), dt=1.0)
    sh = (g.cfg.ny, g.cfg.nx, g.cfg.nz)
    theta = jnp.broadcast_to(g.theta0[None, None, :], sh)
    # heavily supersaturated vapour (q_v=20 g/kg warm) → kessler must condense q_c
    tr = jnp.zeros(sh + (3,)).at[..., 0].set(0.020)
    dth, dtr, _precip = micro(theta, tr)
    # latent heating warms θ where condensation occurs
    assert float(jnp.max(dth)) > 0.0, "no latent heating from condensation"
    # vapour decreases, cloud water increases
    assert float(jnp.min(dtr[..., 0])) < 0.0, "q_v did not decrease"
    assert float(jnp.max(dtr[..., 1])) > 0.0, "q_c did not form"
    # total water (qv+qc+qr) tendency ≈ 0 (condensation conserves water)
    dwater = dtr[..., 0] + dtr[..., 1] + dtr[..., 2]
    assert float(jnp.max(jnp.abs(dwater))) < 1e-6, "total water not conserved"


def test_operator_split_step_stable():
    """Dynamics step + operator-split microphysics integrate stably over a few steps."""
    g = _grid(nz=30)
    dt = 1.0
    micro = pip.make_microphysics(g, MicrophysicsConfig(scheme="kessler"), dt=dt)
    sh = (g.cfg.ny, g.cfg.nx, g.cfg.nz)
    theta = jnp.broadcast_to(g.theta0[None, None, :], sh) + 0.01 * jax.random.normal(
        jax.random.PRNGKey(0), sh)
    tr = jnp.zeros(sh + (3,)).at[..., 0].set(0.012)
    s = pip.PseudoIncompressibleState(
        u=jnp.zeros(sh), v=jnp.zeros(sh), w=jnp.zeros((g.cfg.ny, g.cfg.nx, g.cfg.nz + 1)),
        theta=theta, pi_prev=jnp.zeros(sh), tracers=tr)
    for _ in range(5):
        s = pip.step(s, g, dt)
        dth, dtr, _ = micro(s.theta, s.tracers)
        s = s._replace(theta=s.theta + dt * dth, tracers=s.tracers + dt * dtr)
    assert np.all(np.isfinite(np.asarray(s.theta)))
    assert np.all(np.isfinite(np.asarray(s.tracers)))
    assert float(jnp.min(s.tracers)) > -1e-9     # positivity (no large negatives)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
