"""Faithfulness / disclosure canaries for the ML gravity-wave-drag emulator.

``ml_emulator.py`` is a LEARNED surrogate, honestly self-labelled "not a
physical closure" (conserves=["none"], signs LEARNED not enforced). This pins
those honest disclaimers so a future edit cannot silently re-inflate them, and
complements ``test_gwd_physical_realism.py`` (which already covers finiteness /
boundedness / grad-jit-vmap and explicitly does NOT assert sign/conservation):

* ``conserves == ["none"]``;
* du_dt/dv_dt/dT_dt are DIRECT network output channels (residual-scaled) with
  NO sign enforcement (an untrained model's tendencies are unconstrained in
  sign and magnitude);
* ``eps_gwd`` is a mean-flow-KE-removal DIAGNOSTIC decoupled from ``dT_dt`` --
  the physical KE->heat tie-back ``c_pd*sum(rho*dT_dt*dz) == eps_gwd`` is NOT
  ENFORCED by the architecture (a representative untrained init violates it);
* ``eps_gwd`` carries NO sign constraint (no clip/abs/sign rule), so the model
  can ADD mean-flow KE (eps_gwd < 0) -- what the signed form is meant to expose.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants as C
from legoesm.atmosphere.physics.gravity_wave_drag.config import GWDMLEmulatorConfig
from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
    GWDEmulator,
    __physics_contract__,
    ml_gwd,
)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _column(nlev=40, ncol=1, T0=250.0):
    """Isothermal hydrostatic westerly-jet column (nonzero wind so the
    KE-removal diagnostic eps_gwd = -sum(rho*u*du*dz) is non-trivial)."""
    g, rair = float(C.g), float(C.R_d)
    p_top, p_surf = 100.0, 1.0e5
    pint = np.linspace(p_top, p_surf, nlev + 1)
    pmid = 0.5 * (pint[:-1] + pint[1:])
    T = np.full(nlev, T0)
    dp = np.diff(pint)
    dz = rair * T * dp / (g * pmid)
    z_half = np.zeros(nlev + 1)
    for k in range(nlev, 0, -1):
        z_half[k - 1] = z_half[k] + dz[k - 1]
    zm = 0.5 * (z_half[:-1] + z_half[1:])
    u = np.linspace(5.0, 30.0, nlev)

    def rep(a):
        return jnp.broadcast_to(jnp.asarray(a)[None, :], (ncol, len(a)))

    pmid_c = rep(pmid)
    pint_c = jnp.broadcast_to(jnp.asarray(pint)[None, :], (ncol, nlev + 1))
    T_c = rep(T)
    zf_c = rep(zm)
    zh_c = jnp.broadcast_to(jnp.asarray(z_half)[None, :], (ncol, nlev + 1))
    rho_c = pmid_c / (rair * T_c)
    u_c = rep(u)
    v_c = jnp.zeros((ncol, nlev))
    lat = jnp.zeros(ncol)
    return u_c, v_c, T_c, pmid_c, pint_c, zf_c, zh_c, rho_c, lat


def _model(cfg):
    return GWDEmulator(cfg.n_input, cfg.n_hidden, cfg.n_layers, cfg.n_output,
                       key=jax.random.PRNGKey(cfg.seed))


def _run(cfg):
    u, v, T, pf, ph, zf, zh, rho, lat = _column()
    out = ml_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0, cfg, _model(cfg))
    return out, (u, v, T, pf, ph, zf, zh, rho, lat)


# --------------------------------------------------------------------------
# 1. The honest conservation declaration.
# --------------------------------------------------------------------------

def test_conserves_is_none():
    """A learned surrogate guarantees no conservation; ``conserves`` must be the
    audited ["none"]. Re-inflating it fails here."""
    assert __physics_contract__["conserves"] == ["none"]


# --------------------------------------------------------------------------
# 2. Tendencies are DIRECT network outputs, residual-scaled (no sign rule).
# --------------------------------------------------------------------------

def test_tendencies_are_residual_scaled_network_outputs():
    """du/dv/dT are direct MLP output channels; use_residual multiplies all
    three by the fixed _GWD_OUTPUT_SCALE. Regression pin of that ratio (0.01):
    with identical weights + inputs, residual output == 0.01 * non-residual
    output (to rtol 1e-12). Guarded non-vacuous (the map is active, outputs
    nonzero) so the ratio is not tested on all-zeros."""
    base = dict(n_hidden=32, seed=3)
    out_res, _ = _run(GWDMLEmulatorConfig(**base, use_residual=True))
    out_raw, _ = _run(GWDMLEmulatorConfig(**base, use_residual=False))
    # Non-vacuous: all three channels carry a nonzero signal.
    for ch in (out_raw.du_dt, out_raw.dv_dt, out_raw.dT_dt):
        assert float(jnp.max(jnp.abs(ch))) > 0.0
    for a, b in ((out_res.du_dt, out_raw.du_dt),
                 (out_res.dv_dt, out_raw.dv_dt),
                 (out_res.dT_dt, out_raw.dT_dt)):
        np.testing.assert_allclose(np.asarray(a), 0.01 * np.asarray(b),
                                   rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------
# 3. eps_gwd is DECOUPLED from dT_dt: the KE->heat tie-back is NOT ENFORCED.
# --------------------------------------------------------------------------

def test_eps_gwd_decoupled_from_dT_dt():
    """eps_gwd = -sum(rho*(u*du+v*dv)*dz) uses only the wind channels; dT_dt is
    a SEPARATE unconstrained MLP channel. So the KE->heat tie-back
    c_pd*sum(rho*dT_dt*dz) == eps_gwd is NOT ENFORCED by construction: for a
    representative untrained model the two are large and unequal. (A trained
    model COULD approximate the tie-back; this pins that the architecture does
    not impose it, matching the corrected code comment.)"""
    out, col = _run(GWDMLEmulatorConfig(n_hidden=32, seed=3))
    _, _, _, _, _, _, zh, rho, _ = col
    dz = jnp.abs(zh[:, :-1] - zh[:, 1:])
    heat = float(C.c_pd) * jnp.sum(rho * out.dT_dt * dz, axis=1)
    eps = out.eps_gwd
    # Both are physically LARGE (not a tiny-denominator artifact): the seed-3
    # untrained model gives |heat| ~ 5e3 and |eps| ~ 2e2 W/m^2.
    assert float(jnp.min(jnp.abs(eps))) > 1.0
    assert float(jnp.min(jnp.abs(heat))) > 1.0
    # ... and they are unequal by a wide margin (not accidentally tied).
    rel = float(jnp.max(jnp.abs(heat - eps) / jnp.abs(eps)))
    assert rel > 1e-2, (
        f"emulator heat matched the KE removal (rel={rel:.2e}) for this init -> "
        "the tie-back looks enforced, contradicting the learned-surrogate "
        "disclosure that it is not"
    )


# --------------------------------------------------------------------------
# 4. eps_gwd is sign-UNCONSTRAINED (can be < 0: the model can add KE).
# --------------------------------------------------------------------------

def test_eps_gwd_sign_unconstrained_across_seeds():
    """The emulator imposes NO KE-sink sign constraint on eps_gwd (no clip, no
    abs, no sign rule) -- unlike a dissipative drag whose eps_gwd is >= 0 by
    construction. Demonstrated (not proven) by both signs appearing across
    random inits: the signed (not abs) form is what lets an energy-ADDING model
    surface as eps_gwd < 0. (This is an init-dependent demonstration; the point
    is the code path imposes no sign constraint, so a negative value is legal.)"""
    signs = set()
    for seed in range(16):
        out, _ = _run(GWDMLEmulatorConfig(n_hidden=32, seed=seed))
        signs.add(bool(float(out.eps_gwd[0]) > 0.0))
        if len(signs) == 2:
            break
    assert signs == {True, False}, (
        f"eps_gwd stayed one sign across the sampled inits ({signs}); expected "
        "both signs since the emulator imposes no KE-sink constraint"
    )
