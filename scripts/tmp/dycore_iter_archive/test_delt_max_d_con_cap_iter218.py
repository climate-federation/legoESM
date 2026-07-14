"""FV3_3D iter 218: per-step cap on dissipative heating magnitude
``delt_max`` (FV3 ``dyn_core.F90:1774``).

iter-203/207/208/209 ported the FV3 ``d_con`` KE→heat conversion
for the iter-12/169 ``damp_v`` and iter-193 ``damp_w`` post-step
dampings, but NOT the ``delt_max`` limiter that FV3 applies to the
resulting per-step temperature change ``dtmp``:

    pt += sign(min(|bdt*delt_max|, |dtmp|), dtmp) / pkz

Without the cap, an extreme transient (model spinup, locally large
KE perturbation, large ``damp_v_d_con``) can produce a per-step
ΔT >> 1 K and drive a thermodynamic instability.

iter 218 adds the FV3 cap as a config knob ``delt_max`` (K/s,
default 0.0 = disabled, FV3 production = 1.0).  Active when
``> 0`` it clips ``|ΔT|`` (PE) or ``|Δθ_p * Π_ref|`` (NH) to
``dt * delt_max`` per step using ``jnp.clip`` (AD-safe).

Tests
-----

1. ``test_pe_delt_max_off_baseline`` — ``delt_max = 0.0`` is bit-
   for-bit equal to the pre-iter-218 PE damp_v_d_con baseline.
2. ``test_pe_delt_max_caps_per_step_dT`` — with strong damp_v +
   strong d_con + small ``delt_max``, the per-step ΔT magnitude
   is reduced compared to the uncapped run.
3. ``test_nh_delt_max_off_baseline`` — same for NH ``damp_v_d_con``.
4. ``test_nh_delt_max_caps_damp_w_dtheta_p`` — same on the
   iter-203 ``damp_w_d_con`` block.
5. ``test_pe_delt_max_differentiable_at_rest`` — at rest with
   ``delt_max > 0``, ``jax.grad`` w.r.t. wind perturbation is
   finite (no NaN through ``jnp.clip``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
    standard_hybrid_levels,
)


def _pe_state_with_perturbation():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=218)
    n_corners = n + 1
    u_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


def _nh_state_with_perturbation():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=2180)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev + 1))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def test_pe_delt_max_off_baseline():
    """delt_max = 0.0 preserves pre-iter-218 d_con behavior."""
    grid, cdgrid, coord, state = _pe_state_with_perturbation()

    common = dict(
        damp_v=0.030, nord_v=0, damp_v_d_con=1.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_off = CDGridPrimitiveEquationConfig(**common, delt_max=0.0)
    cfg_default = CDGridPrimitiveEquationConfig(**common)

    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_default = CDGridPrimitiveEquationModel(grid, coord, cfg_default)

    s_off = m_off.step(state, 100.0)
    s_default = m_default.step(state, 100.0)

    np.testing.assert_array_equal(s_off.T.data, s_default.T.data)


def test_pe_delt_max_caps_per_step_dT():
    """With strong damp_v + strong d_con + small delt_max, the
    d_con-only contribution to T is bounded by ``dt * delt_max``.

    The cap acts on the d_con heat term ``dT = -damp_v_d_con * ΔKE
    / c_pd`` only.  To isolate it from the rest of the per-step T
    tendency (advection, vertical adjustment, etc.) we subtract a
    no-d_con reference run (``damp_v_d_con=0``)."""
    grid, cdgrid, coord, state = _pe_state_with_perturbation()
    dt = 100.0
    delt_max = 1e-4    # K/s; per-step cap = dt * delt_max = 0.01 K
    common = dict(
        damp_v=0.030, nord_v=0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_no_dcon = CDGridPrimitiveEquationConfig(
        **common, damp_v_d_con=0.0, delt_max=0.0,
    )
    cfg_uncapped = CDGridPrimitiveEquationConfig(
        **common, damp_v_d_con=1.0, delt_max=0.0,
    )
    cfg_capped = CDGridPrimitiveEquationConfig(
        **common, damp_v_d_con=1.0, delt_max=delt_max,
    )
    m_no_dcon = CDGridPrimitiveEquationModel(grid, coord, cfg_no_dcon)
    m_uncapped = CDGridPrimitiveEquationModel(grid, coord, cfg_uncapped)
    m_capped = CDGridPrimitiveEquationModel(grid, coord, cfg_capped)

    s_no_dcon = m_no_dcon.step(state, dt)
    s_uncapped = m_uncapped.step(state, dt)
    s_capped = m_capped.step(state, dt)

    # d_con-only contribution: difference between the run with d_con
    # active and the reference run with d_con off.  The cap acts on
    # exactly this signal.  iter 219 adds sponge-layer awareness:
    # PE k=0,1 are intentionally NOT capped, so we test only the
    # interior layers (k>=2).
    d_con_uncapped = s_uncapped.T.data - s_no_dcon.T.data
    d_con_capped = s_capped.T.data - s_no_dcon.T.data

    cap = dt * delt_max
    # Uncapped d_con contribution must EXCEED the cap somewhere
    # (we evaluate over interior layers since iter-219 leaves the
    # top-2 sponge layers uncapped by design).
    assert float(jnp.max(jnp.abs(d_con_uncapped[..., 2:]))) > cap, (
        f"Uncapped d_con max|ΔT|="
        f"{float(jnp.max(jnp.abs(d_con_uncapped[..., 2:]))):.4e} ≤ "
        f"cap={cap:.4e} — IC too quiet to test the cap."
    )

    # Capped d_con contribution at INTERIOR levels must respect the
    # cap (PE sponge layers k=0,1 are uncapped — see iter 219).
    max_capped_interior = float(
        jnp.max(jnp.abs(d_con_capped[..., 2:])),
    )
    assert max_capped_interior <= cap * 1.0 + 1e-12, (
        f"Capped d_con max|ΔT[k>=2]|={max_capped_interior:.4e} > "
        f"cap={cap:.4e} — jnp.clip is not bounding the interior."
    )


def test_nh_delt_max_off_baseline():
    """NH delt_max = 0.0 preserves pre-iter-218 d_con behavior."""
    grid, hc, tm, state = _nh_state_with_perturbation()

    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=0, damp_v_d_con=1.0,
    )
    cfg_off = CDGridCompressibleEulerConfig(**common, delt_max=0.0)
    cfg_default = CDGridCompressibleEulerConfig(**common)

    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_default = CDGridCompressibleEulerModel(grid, hc, tm, cfg_default)

    s_off = m_off.step(state, 10.0)
    s_default = m_default.step(state, 10.0)

    np.testing.assert_array_equal(
        s_off.theta_prime.data, s_default.theta_prime.data,
    )


def test_nh_delt_max_caps_damp_w_dtheta_p():
    """damp_w_d_con with strong damping + small delt_max keeps the
    per-step Δθ_p smaller than the uncapped run."""
    grid, hc, tm, state = _nh_state_with_perturbation()

    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.30, nord_w=0, damp_w_d_con=1.0,
    )
    cfg_uncapped = CDGridCompressibleEulerConfig(
        **common, delt_max=0.0,
    )
    cfg_capped = CDGridCompressibleEulerConfig(
        **common, delt_max=1e-5,    # dt=10 → cap = 1e-4 K
    )
    m_un = CDGridCompressibleEulerModel(grid, hc, tm, cfg_uncapped)
    m_cap = CDGridCompressibleEulerModel(grid, hc, tm, cfg_capped)

    s_un = m_un.step(state, 10.0)
    s_cap = m_cap.step(state, 10.0)

    dtheta_un = s_un.theta_prime.data - state.theta_prime.data
    dtheta_cap = s_cap.theta_prime.data - state.theta_prime.data

    assert float(jnp.max(jnp.abs(dtheta_cap))) < float(
        jnp.max(jnp.abs(dtheta_un))
    ), (
        f"NH delt_max cap on damp_w_d_con must reduce max|Δθ_p|: "
        f"uncapped={float(jnp.max(jnp.abs(dtheta_un))):.4e}, "
        f"capped={float(jnp.max(jnp.abs(dtheta_cap))):.4e}."
    )


def test_pe_delt_max_differentiable_at_rest():
    """At rest, ``jax.grad`` of T-norm w.r.t. perturbation amplitude
    is finite when delt_max > 0 (no NaN through jnp.clip)."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state_rest = hydrostatic_to_fv3(state_cc, cdgrid)
    state_rest = state_rest._replace(
        u_d=state_rest.u_d.replace(
            data=jnp.zeros_like(state_rest.u_d.data),
        ),
        v_d=state_rest.v_d.replace(
            data=jnp.zeros_like(state_rest.v_d.data),
        ),
    )

    cfg = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=0, damp_v_d_con=1.0,
        delt_max=1.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss(eps):
        u_d_pert = state_rest.u_d.data + eps * jnp.ones_like(
            state_rest.u_d.data,
        )
        s = state_rest._replace(
            u_d=state_rest.u_d.replace(data=u_d_pert),
        )
        s_new = model.step(s, 100.0)
        return jnp.sum(s_new.T.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), (
        f"PE delt_max + jnp.clip must be AD-safe at rest; got grad={g}"
    )
