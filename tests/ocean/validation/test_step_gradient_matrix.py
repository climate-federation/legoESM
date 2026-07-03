"""Step-level gradient health matrix for the lat-lon C-grid ocean.

CI distillation of the 2026-06-11 ocean differentiability audit
(.physics-validator/ocean_diff_audit/, 29-variant campaign): for each
config variant, run a short ``_step_impl`` rollout inside
``jit(value_and_grad)`` and gate

  1. loss finite,
  2. gradient finite (NaN/Inf scan) and flowing (nonzero on >50% of wet
     cells),
  3. reverse-mode == forward-mode AD on a random direction — the
     kink-immune transpose check (AD-vs-FD saturates on the C0
     limiter/where landscape; rev-vs-fwd does not).

Everything runs under an explicitly-forced fp64 compute policy: the
models default to float32 compute REGARDLESS of JAX_ENABLE_X64, and at
float32 these gates measure precision floors instead of correctness
(audit Finding 5). The policy is restored after the module's tests.

Axes covered (one variant each — the audit's previously-untested set):
limiter advection (superbee), FCT (ppm_fct, with a looser gate: its
Zalesak ratio chain is the worst-conditioned operator in the family),
AB2 outer integrator, KPP vertical mixing under wind+heat forcing,
enhanced-diffusion convection on an unstable column, flux_feedback
surface forcing, GM/Redi + prognostic EKE with the GEOMETRIC closure,
and the rigid-lid + AB2 + explicit-Coriolis oracle-match stack (island
cache pre-built eagerly — required for cold models traced under jit).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

N_LAT, N_LON, NLEV = 10, 16, 4
H_MAX = 1000.0
DT = 600.0
N_STEPS = 2

REVFWD_TOL = 1e-9
REVFWD_TOL_FCT = 1e-5  # ppm_fct: Zalesak ratio conditioning (audit S3)


@pytest.fixture(scope="module", autouse=True)
def _fp64_policy():
    """Force fp64 compute for this module; restore the prior policy."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


def _build(config, *, unstable_column=False, land_lat_threshold=85.0):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(N_LAT, N_LON)
    z_coord = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=H_MAX, land_lat_threshold=land_lat_threshold)
    key = jax.random.PRNGKey(0)
    ku, kv, ke = jax.random.split(key, 3)
    lat = jnp.linspace(-1.0, 1.0, N_LAT)[:, None, None]
    lon = jnp.linspace(-1.0, 1.0, N_LON)[None, :, None]
    lev = jnp.linspace(0.0, 1.0, NLEV)[None, None, :]
    T = state.T.data + 1.5 * jnp.exp(
        -((lat / 0.5) ** 2 + (lon / 0.5) ** 2) - 3.0 * lev)
    if unstable_column:
        T = T - 4.0 * jnp.exp(
            -((lat / 0.3) ** 2 + ((lon - 0.5) / 0.3) ** 2) - 8.0 * lev)
    state = state._replace(
        T=state.T.replace(data=T),
        u=state.u.replace(data=(state.u.data + 0.03 * jax.random.normal(
            ku, state.u.data.shape)) * state.u_mask.data[..., None]),
        v=state.v.replace(data=(state.v.data + 0.03 * jax.random.normal(
            kv, state.v.data.shape)) * state.v_mask.data[..., None]),
        eta=state.eta.replace(data=(0.01 * jax.random.normal(
            ke, state.eta.data.shape)) * state.land_mask.data),
    )
    model = LatLonCGridOceanModel(grid, z_coord, config)
    return model, state


def _surface_forcing(kind):
    from legoesm.ocean.state import OceanSurfaceForcing

    shape = (N_LAT, N_LON)
    lat = jnp.linspace(-1.0, 1.0, N_LAT)[:, None]
    tau_x = 0.05 * jnp.cos(jnp.pi * lat) * jnp.ones(shape)
    if kind == "tau_qnet":
        return OceanSurfaceForcing(
            tau_x=tau_x, tau_y=jnp.zeros(shape), q_net=50.0 * jnp.ones(shape))
    if kind == "flux_feedback":
        return OceanSurfaceForcing(
            tau_x=tau_x, tau_y=jnp.zeros(shape),
            q_prescribed=30.0 * jnp.ones(shape),
            q_feedback=40.0 * jnp.ones(shape),
            T_feedback_target=18.0 * jnp.ones(shape),
            S_restore_target=34.5 * jnp.ones(shape))
    return None


def _variants():
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import (
        GMRediConfig,
        LateralMixingConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.eke import (
        EKEConfig,
        GeometricConfig,
    )
    from legoesm.ocean.physics.surface_forcing.config import (
        SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig

    base = LatLonCGridOceanConfig.from_flat(
        A_h=1000.0, K_h=100.0, A_v=1e-3, K_v=1e-4, bottom_drag_r=1e-3,
        tracer_advection="upwind", n_barotropic_substeps=4,
        differentiable_barotropic=True, enable_runtime_checks=False)

    def phys(**kw):
        kw.setdefault("lateral_mixing", LateralMixingConfig(scheme="none"))
        return OceanPhysicsConfig(**kw)

    return {
        "base_upwind": (base, {}),
        "adv_superbee": (base._replace(tracer_advection="superbee"), {}),
        "adv_ppm_fct": (base._replace(tracer_advection="ppm_fct"), {}),
        "outer_ab2": (base._replace(outer_integrator="ab2"), {}),
        "kpp": (base._replace(physics=phys(
            vertical_mixing=VerticalMixingConfig(scheme="kpp"))),
            {"forcing": "tau_qnet"}),
        "conv_enhanced": (base._replace(physics=phys(
            convection=OceanConvectionConfig(scheme="enhanced_diffusion"))),
            {"unstable_column": True}),
        "flux_feedback": (base._replace(physics=phys(
            surface_forcing=SurfaceForcingConfig(scheme="flux_feedback"))),
            {"forcing": "flux_feedback"}),
        "gm_eke_geometric": (base._replace(gm_redi=GMRediConfig(
            kappa_GM=500.0, kappa_Redi=500.0,
            eke=EKEConfig(closure="geometric", geometric=GeometricConfig()))),
            {}),
        "rigid_lid_ab2": (base._replace(
            outer_integrator="ab2", coriolis_scheme="explicit_ab2",
            barotropic_solver="rigid_lid"),
            {"land_lat_threshold": 70.0, "warm_rigid_lid": True}),
    }


@pytest.mark.parametrize("name", [
    "base_upwind", "adv_superbee", "adv_ppm_fct", "outer_ab2", "kpp",
    "conv_enhanced", "flux_feedback", "gm_eke_geometric", "rigid_lid_ab2",
])
def test_step_gradient_health(name):
    config, opts = _variants()[name]
    model, state = _build(
        config,
        unstable_column=opts.get("unstable_column", False),
        land_lat_threshold=opts.get("land_lat_threshold", 85.0))
    if opts.get("warm_rigid_lid"):
        # island flood-fill is host-side; build the cache eagerly so the
        # first traced step does not have to (see test_rigid_lid_cache_jit)
        model._ensure_rigid_lid_data(state)
    sf = _surface_forcing(opts.get("forcing")) if opts.get("forcing") else None
    wet3 = state.land_mask.data[..., None] * jnp.ones((1, 1, NLEV))

    def loss(dT0):
        s = state._replace(T=state.T.replace(data=state.T.data + dT0))
        for _ in range(N_STEPS):
            s = model._step_impl(s, DT, surface_forcing=sf)
        return (jnp.sum((s.T.data * wet3) ** 2)
                + jnp.sum((s.S.data * wet3) ** 2)
                + 1e4 * (jnp.sum(s.u.data ** 2) + jnp.sum(s.v.data ** 2))
                + 1e4 * jnp.sum(s.eta.data ** 2))

    zero = jnp.zeros_like(state.T.data)
    val, g = jax.jit(jax.value_and_grad(loss))(zero)

    assert bool(jnp.isfinite(val)), f"{name}: loss not finite"
    n_bad = int(jnp.sum(~jnp.isfinite(g)))
    assert n_bad == 0, f"{name}: {n_bad} non-finite gradient entries"
    wet_n = float(jnp.sum(wet3 > 0))
    nz = float(jnp.sum((jnp.abs(g) > 1e-300) & (wet3 > 0))) / max(wet_n, 1.0)
    assert nz > 0.5, f"{name}: gradient flows on only {nz:.0%} of wet cells"

    d = jax.random.normal(jax.random.PRNGKey(42), g.shape) * wet3
    d = d / jnp.linalg.norm(d)
    rev = float(jnp.vdot(g, d))
    try:
        _, fwd = jax.jvp(loss, (zero,), (d,))
    except TypeError:
        # custom_vjp solves (rigid lid) define no JVP rule: fall back to a
        # central-difference cross-check with a kink-tolerant gate.
        h = 1e-3
        lj = jax.jit(loss)
        fd = (float(lj(h * d)) - float(lj(-h * d))) / (2.0 * h)
        rel = abs(rev - fd) / max(abs(rev), abs(fd), 1e-300)
        assert rel < 1e-4, f"{name}: AD-vs-FD rel={rel:.2e} (rev={rev:.6e})"
        return
    tol = REVFWD_TOL_FCT if name == "adv_ppm_fct" else REVFWD_TOL
    rel = abs(rev - float(fwd)) / max(abs(rev), abs(float(fwd)), 1e-300)
    assert rel < tol, (
        f"{name}: reverse-vs-forward AD rel={rel:.2e} "
        f"(rev={rev:.9e}, fwd={float(fwd):.9e}) — transpose inconsistency")
    # FD sanity (kink-tolerant): catches a consistently-wrong (rev, fwd)
    # pair. SKIPPED for ppm_fct: the Zalesak limiter flips branches inside
    # any usable FD stencil on this landscape (audit: FD saturates ~1e-1
    # there while rev==fwd holds) — the rev-vs-fwd gate above is binding.
    if name == "adv_ppm_fct":
        return
    h = 1e-3
    lj = jax.jit(loss)
    fd = (float(lj(h * d)) - float(lj(-h * d))) / (2.0 * h)
    rel_fd = abs(rev - fd) / max(abs(rev), abs(fd), 1e-300)
    assert rel_fd < 0.05, f"{name}: AD-vs-FD rel={rel_fd:.2e}"
