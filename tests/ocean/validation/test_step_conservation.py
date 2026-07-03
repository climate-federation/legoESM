"""Closed-domain heat / salt / volume conservation over a long rollout, with
the full production scheme stack active.

The existing ocean suite asserts tracer conservation per-component (each
advection scheme telescopes, the implicit-mixing solve is flux-form, …) and
checks ``compute_tracer_budget`` only *at rest* (test_longterm_diagnostics).
What was missing: that the *combined* operator chain — advection + GM/Redi +
vertical mixing + convection + barotropic coupling, stepped O(100) times —
conserves total heat content, salt mass, and volume in a closed basin. A leak
that only appears when several conservative-in-isolation schemes interact (a
thickness inconsistency at the barotropic split, a tracer applied off the
flux-form path, a convective adjustment that is not heat-neutral) is invisible
to the per-component tests but caught here.

Setup: a zonally-periodic basin walled at +-70 deg latitude (no open
boundary), NO surface forcing / sponge / freshwater, the tracer
conservation-fixer OFF (default) so we measure the SCHEMES' own conservation
rather than a fixer papering over a leak. The initial state has a warm blob, a
cold surface patch (so convection actually fires) and a small random velocity
(so advection / GM/Redi actually transport). ``compute_tracer_budget``
(``legoesm.ocean.budgets``) integrates rho_0*c_p*sum(T*vol) and rho_0*sum(S*vol)
over the wet domain; its own docstring notes closed-basin drift should be at
floating-point precision.

We deliberately disable ``fix_eta_drift`` (the per-step projection that snaps
total volume back to ``vol(eta_old)+dt*F_slow_eta``): with it ON the volume
assertion would be vacuous — the projection overwrites any leak so volume drift
is identically 0 regardless of whether the barotropic continuity actually
conserves. OFF, the volume gate measures the genuine continuity drift. The
tracer conservation-fixer is already off by default (so heat/salt drift is the
schemes' own); these never depend on ``fix_eta_drift``.

Observed drift (bring-up, all three stacks, fix_eta_drift OFF): volume ~6e-11,
heat & salt ~1.05e-10 relative over 100 steps — scheme-independent, i.e. the
barotropic/thickness-coupling roundoff floor, not the flux-form physics. The
gates carry ~10-100x margin over that floor while a real leak (a
non-conservative scheme loses O(1e-3..1e-2); an injected 1e-7 K/s heat source
gives ~5e-4) is caught by many orders of magnitude.

Caveat on the plume stack: ``plume_convection`` subtracts its column-integral
from level 0 by construction, so its heat/salt conservation is self-enforced
rather than emergent — that stack mainly adds "runs in the closed basin without
NaN and doesn't leak elsewhere" coverage; the tvd/centered stacks are the
binding heat/salt gates.

fp64 forced (the model defaults to float32 regardless of JAX_ENABLE_X64; at
float32 these gates would measure precision floors — cf. test_step_fd_vs_ad).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

N_LAT, N_LON, NLEV = 12, 16, 6
H_MAX = 1000.0
DT = 600.0
N_STEPS = 100

VOLUME_TOL = 1e-9    # genuine barotropic-continuity drift (fix_eta_drift OFF);
# ~17x over the observed ~6e-11 floor. With fix_eta_drift ON this would be 0
# and the gate vacuous — see module docstring.
TRACER_TOL = 1e-8    # heat/salt: ~100x over the observed 1.05e-10 floor.


@pytest.fixture(scope="module", autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


# (tracer_advection, vertical_mixing, convection) — span the
# conservation-relevant axes: limiter vs unlimited advection, the three
# vertical-mixing closures, and both convective-adjustment schemes.
_STACKS = {
    "tvd_kpp_enhanced": ("tvd", "kpp", "enhanced_diffusion"),
    "centered_tke_enhanced": ("centered", "tke", "enhanced_diffusion"),
    "superbee_constant_plume": ("superbee", "constant", "plume"),
}


def _config(tracer_adv, vmix, convection):
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import (
        GMRediConfig,
        LateralMixingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig

    phys = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme=vmix),
        convection=OceanConvectionConfig(scheme=convection),
        lateral_mixing=LateralMixingConfig(scheme="none"),
    )
    return LatLonCGridOceanConfig.from_flat(
        A_h=1000.0, K_h=100.0, A_v=1e-3, K_v=1e-4, bottom_drag_r=1e-3,
        tracer_advection=tracer_adv, n_barotropic_substeps=4,
        differentiable_barotropic=True, enable_runtime_checks=False,
        # OFF so the volume gate measures genuine barotropic-continuity
        # conservation rather than the per-step volume projection (which would
        # make the gate vacuous). Tracer fixer is already off by default.
        fix_eta_drift=False,
        gm_redi=GMRediConfig(kappa_GM=300.0, kappa_Redi=300.0),
        physics=phys,
    )


def _closed_basin_state(grid, z_coord):
    """Walled basin (land poleward of +-70 deg) with a warm blob, a cold
    surface patch (drives convection) and a small random velocity (drives
    advection / GM/Redi). No surface forcing → a closed system."""
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=H_MAX, land_lat_threshold=70.0)
    lat = jnp.linspace(-1.0, 1.0, N_LAT)[:, None, None]
    lon = jnp.linspace(-1.0, 1.0, N_LON)[None, :, None]
    lev = jnp.linspace(0.0, 1.0, NLEV)[None, None, :]
    T = (state.T.data
         + 2.0 * jnp.exp(-((lat / 0.5) ** 2 + (lon / 0.5) ** 2) - 3.0 * lev)
         - 5.0 * jnp.exp(
             -((lat / 0.3) ** 2 + ((lon - 0.5) / 0.3) ** 2) - 8.0 * lev))
    S = state.S.data + 0.5 * jnp.exp(
        -((lat / 0.6) ** 2 + (lon / 0.6) ** 2) - 2.0 * lev)
    ku, kv = jax.random.split(jax.random.PRNGKey(0))
    return state._replace(
        T=state.T.replace(data=T),
        S=state.S.replace(data=S),
        u=state.u.replace(data=(state.u.data + 0.03 * jax.random.normal(
            ku, state.u.data.shape)) * state.u_mask.data[..., None]),
        v=state.v.replace(data=(state.v.data + 0.03 * jax.random.normal(
            kv, state.v.data.shape)) * state.v_mask.data[..., None]),
    )


def _rel(a, b):
    return abs(b - a) / max(abs(a), 1e-300)


@pytest.mark.parametrize("name", list(_STACKS))
def test_closed_basin_conserves_heat_salt_volume(name):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.budgets import compute_tracer_budget
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    tracer_adv, vmix, convection = _STACKS[name]
    grid = create_latlon_grid(N_LAT, N_LON)
    z_coord = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
    model = LatLonCGridOceanModel(
        grid, z_coord, _config(tracer_adv, vmix, convection))
    state = _closed_basin_state(grid, z_coord)

    def budget(s):
        return compute_tracer_budget(
            s, z_coord, grid_type="latlon_cgrid", grid=grid)

    b0 = budget(state)
    step = jax.jit(lambda s: model._step_impl(s, DT, surface_forcing=None))
    s = state
    for _ in range(N_STEPS):
        s = step(s)

    assert bool(jnp.isfinite(s.T.data).all() and jnp.isfinite(s.S.data).all()), \
        f"{name}: non-finite state after {N_STEPS} steps"
    # The rollout must actually do something — a state frozen at the IC would
    # conserve trivially and make this gate vacuous. Check BOTH the tracer
    # field (heat/salt path) and eta (the volume/barotropic path) moved.
    assert not bool(jnp.allclose(s.T.data, state.T.data)), \
        f"{name}: tracer state did not evolve — heat/salt gate would be vacuous"
    assert not bool(jnp.allclose(s.eta.data, state.eta.data)), \
        f"{name}: eta did not evolve — volume gate would be vacuous"

    b1 = budget(s)
    assert _rel(b0.volume, b1.volume) < VOLUME_TOL, (
        f"{name}: volume drift {_rel(b0.volume, b1.volume):.2e} "
        f"over {N_STEPS} steps")
    assert _rel(b0.heat_content, b1.heat_content) < TRACER_TOL, (
        f"{name}: heat-content drift "
        f"{_rel(b0.heat_content, b1.heat_content):.2e} over {N_STEPS} steps "
        f"— a scheme is leaking heat in the closed basin")
    assert _rel(b0.salt_mass, b1.salt_mass) < TRACER_TOL, (
        f"{name}: salt-mass drift {_rel(b0.salt_mass, b1.salt_mass):.2e} "
        f"over {N_STEPS} steps — a scheme is leaking salt in the closed basin")
