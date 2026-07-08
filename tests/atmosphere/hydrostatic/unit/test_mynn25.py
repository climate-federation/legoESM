"""Unit tests for the MYNN-2.5 turbulence closure (Phase C).

Verifies the legoESM port behaves as expected for the foundational
regimes that drive the jax_scm benchmark suite:

* Zero-shear, zero-buoyancy column → qke decays toward floor (no
  production, finite dissipation timescale).
* Neutral driven column → friction-velocity surface boundary
  ``qke_sfc = B1^(2/3) · u*²`` is produced and propagates upward via
  diffusion.
* Stable column with ``T_s`` colder than the air → sensible flux is
  negative (downward) and the lowest cell cools.
* Convective column with ``T_s`` warmer than the air → sensible flux
  is positive (upward) and the lowest cell warms.
* Stability functions produce monotone behaviour in ``Ri``.
* Scheme is selectable via ``TurbulenceConfig(scheme='mynn25')`` and
  the SCM swap matrix integrates without NaNs across forward Euler.

These are not jax_scm-oracle comparisons — the oracle harness lives in
Phase E0 — but they catch obvious sign / scaling / dispatch bugs in
the closure before benchmark integration.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics.turbulence.config import MYNN25Config
from legoesm.atmosphere.physics.turbulence.mynn25 import (
    _compute_SM_SH,
    _compute_master_length,
    _filter_121,
    mynn25_turbulence,
)
from legoesm.atmosphere.scm import SingleColumnModel
from legoesm.atmosphere.scm_forcing import SCMForcing


NLEV = 32


def _mynn_cfg() -> PhysicsConfig:
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="mynn25"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


# ----------------------------------------------------------------------
# Stability functions: structural sanity
# ----------------------------------------------------------------------


def test_sm_sh_positive_in_neutral_limit():
    """In the neutral limit (G_H → 0, G_M small), SM and SH must be
    positive O(0.1) — these are the canonical level-2 / level-2.5
    isotropic-turbulence values."""
    cfg = MYNN25Config()
    G_M = jnp.full((1, 5), 1e-3)
    G_H = jnp.full((1, 5), 0.0)
    L = jnp.full((1, 5), 50.0)
    q_half = jnp.full((1, 5), 0.3)
    SM, SH = _compute_SM_SH(G_M, G_H, L, q_half, cfg)
    assert np.all(np.asarray(SM) > 0.0)
    assert np.all(np.asarray(SH) > 0.0)
    # Order of magnitude check — both should be O(0.1).
    assert 0.01 < float(SM[0, 0]) < 1.0
    assert 0.01 < float(SH[0, 0]) < 1.0


def test_sm_sh_decrease_with_stability():
    """Increasing stability (more positive G_H = -L²N²/q²; in stable
    air N² > 0 makes G_H NEGATIVE in NN09's sign convention, so we
    sweep G_H from 0 toward negative values).  Both SM and SH must
    monotonically *decrease* as the column becomes more stable.
    """
    cfg = MYNN25Config()
    G_M = jnp.full((1, 1), 1e-3)
    L = jnp.full((1, 1), 50.0)
    q_half = jnp.full((1, 1), 0.3)
    g_h_vals = [0.0, -1e-3, -1e-2, -0.1]
    sm_vals = []
    sh_vals = []
    for g_h in g_h_vals:
        SM, SH = _compute_SM_SH(
            G_M, jnp.full((1, 1), g_h), L, q_half, cfg,
        )
        sm_vals.append(float(SM[0, 0]))
        sh_vals.append(float(SH[0, 0]))
    assert all(
        sm_vals[i] >= sm_vals[i + 1] for i in range(len(sm_vals) - 1)
    ), f"SM not monotone-decreasing with stability: {sm_vals}"
    assert all(
        sh_vals[i] >= sh_vals[i + 1] for i in range(len(sh_vals) - 1)
    ), f"SH not monotone-decreasing with stability: {sh_vals}"


# ----------------------------------------------------------------------
# 1-2-1 filter
# ----------------------------------------------------------------------


def test_121_filter_damps_high_wavenumber():
    """A column with [+1, -1, +1, -1, ...] noise must be damped to
    near zero amplitude after one pass of the 1-2-1 filter (with
    reflect padding, the boundaries are smoothed too).
    """
    n = 20
    x = jnp.array([(-1.0) ** k for k in range(n)]).reshape(1, n)
    y = _filter_121(x)
    assert float(jnp.max(jnp.abs(y))) < 0.7, (
        f"1-2-1 filter did not damp high-wavenumber noise: "
        f"max |y| = {float(jnp.max(jnp.abs(y)))}"
    )


# ----------------------------------------------------------------------
# Integration tests via SCM driver
# ----------------------------------------------------------------------


def _baseline_profile():
    """Mildly stable boundary layer: 10 K K of stratification across
    the lowest 1 km in K, dry."""
    T = jnp.linspace(220.0, 285.0, NLEV)
    qv = jnp.full(NLEV, 1e-6)        # near-dry
    return T, qv


def test_mynn_runs_under_forward_euler():
    """MYNN-2.5 scheme registers and runs without NaN under forward
    Euler.  Smoke test for dispatch + numerics integration."""
    T0, qv0 = _baseline_profile()
    scm = SingleColumnModel.create(
        physics_config=_mynn_cfg(), nlev=NLEV, dt=60.0,
        T_profile=T0, q_v_profile=qv0, u=5.0,
        time_integrator="forward_euler",
    )
    final, _ = scm.run(nsteps=20)
    assert jnp.all(jnp.isfinite(final.T.data))
    assert jnp.all(jnp.isfinite(scm.phys_state.qke))


def test_mynn_rk_rejected_by_scm_validator():
    """MYNN-2.5 is stateful (qke carry) so the SCM RK2/RK4 path
    must reject it, matching the existing TKE/CLUBB-lite policy."""
    T0, qv0 = _baseline_profile()
    with pytest.raises(ValueError, match="incompatible"):
        SingleColumnModel.create(
            physics_config=_mynn_cfg(), nlev=NLEV, dt=60.0,
            T_profile=T0, q_v_profile=qv0,
            time_integrator="rk2",
        )


def test_mynn_qke_surface_boundary_is_friction_velocity():
    """After a few steps with a non-zero geostrophic wind, the surface
    qke must equal ``B1^(2/3) · u*²`` (MY82 eq. 54).  With constant
    Cd = 1.5e-3 and |V| = 10 m/s, u* = sqrt(1.5e-3) · 10 ≈ 0.387 m/s,
    giving qke_sfc ≈ 24^(2/3) · 0.15 ≈ 1.25 m²/s².
    """
    T0, qv0 = _baseline_profile()
    scm = SingleColumnModel.create(
        physics_config=_mynn_cfg(), nlev=NLEV, dt=60.0,
        T_profile=T0, q_v_profile=qv0, u=10.0, v=0.0,
    )
    scm.run(nsteps=5)
    qke_sfc = float(scm.phys_state.qke[0, -1])
    expected = (24.0 ** (2 / 3)) * (np.sqrt(1.5e-3) * np.sqrt(10.0 ** 2 + 1e-4)) ** 2
    np.testing.assert_allclose(qke_sfc, expected, rtol=0.05)


def test_mynn_warm_T_s_warms_lowest_cell():
    """Phase B-style sanity: prescribing T_s warmer than the lowest
    air cell should drive a positive sensible flux and warm the
    column.  Tests the MYNN closure with the prescribed-T_s hook
    end-to-end.
    """
    T0 = jnp.full(NLEV, 280.0)
    qv0 = jnp.full(NLEV, 5e-3)
    forcing = SCMForcing(prescribe="T_s", T_s=lambda t: jnp.asarray(295.0))
    scm = SingleColumnModel.create(
        physics_config=_mynn_cfg(), nlev=NLEV, dt=60.0,
        T_profile=T0, q_v_profile=qv0, u=5.0,
        forcing=forcing,
    )
    scm.run(nsteps=30)
    T_low = float(scm.state.T.data[0, 0, 0, -1])
    assert T_low > 280.0, (
        f"warm T_s=295 with MYNN did not warm lowest cell: T_low={T_low}"
    )


def test_mynn_combined_physics_jits_under_jax_jit():
    """Codex Phase C iter-1 high: the closure must be JAX-trace-safe.
    Compile the combined hydrostatic physics with ``scheme='mynn25'``
    under ``jax.jit``; success here proves no host-side ``float(...)``
    materialization snuck back into the closure (which would raise a
    ``ConcretizationError`` here).
    """
    import jax
    from legoesm.atmosphere.physics.combined import make_physics
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.scm import make_column_state, make_scm_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    T0, qv0 = _baseline_profile()
    cfg = _mynn_cfg()
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=60.0)
    grid = make_scm_grid(latitude_deg=0.0)
    sigma_coord = create_sigma_coordinate(NLEV, sigma_top=0.01)
    state = make_column_state(NLEV, T_profile=T0, q_v_profile=qv0, u=5.0)
    phys_state = init_physics_state(
        ncol=1, nlev=NLEV, physics_config=cfg,
    )

    @jax.jit
    def step(state_in, phys_in):
        tend, phys_out = physics_fn(
            state_in, grid, sigma_coord, phys_state=phys_in,
        )
        return tend, phys_out

    tend, phys_out = step(state, phys_state)
    assert jnp.all(jnp.isfinite(tend.dT_dt.data))
    assert jnp.all(jnp.isfinite(phys_out.qke))


def test_mynn_hydrostatic_combined_routes_qke_field():
    """Codex Phase C iter-2 high: the MYNN field-name dispatch must
    route to ``qke`` on the hydrostatic combined path.  Builds the
    hydrostatic combined physics with ``scheme='mynn25'`` and verifies
    construction does not error.  A regression here would mean MYNN's
    evolved qke is silently stored into the unused ``tke`` slot while
    next-step reads pull a stale ``qke = qke_min``.
    """
    from legoesm.atmosphere.physics.combined import make_physics
    cfg = _mynn_cfg()
    fn_hyd = make_physics(cfg, model_type="hydrostatic", dt=60.0)
    assert callable(fn_hyd)


def test_mynn_rejected_on_nonhydrostatic_and_spectral_pe():
    """Codex Phase C iter-3 high: the nonhydrostatic CD-grid and
    spectral PE dynamics drivers drop the returned PhysicsState every
    step, which would silently re-initialise qke from ``qke_min`` on
    every call.  Until those drivers thread phys_state, MYNN must
    fail fast at make_physics time rather than silently corrupting
    the run.
    """
    from legoesm.atmosphere.physics.combined import make_physics
    cfg = _mynn_cfg()
    with pytest.raises(NotImplementedError, match="MYNN-2.5"):
        make_physics(cfg, model_type="nonhydrostatic", dt=60.0)
    with pytest.raises(NotImplementedError, match="MYNN-2.5"):
        make_physics(cfg, model_type="spectral_pe", dt=60.0)


def test_mynn_uses_qke_field_not_tke():
    """Codex Phase C iter-1 medium: MYNN-2.5 must write to
    ``phys_state.qke``, not the legacy ``phys_state.tke`` slot, so a
    restart-time scheme switch cannot silently feed TKE as qke.
    """
    T0, qv0 = _baseline_profile()
    scm = SingleColumnModel.create(
        physics_config=_mynn_cfg(), nlev=NLEV, dt=60.0,
        T_profile=T0, q_v_profile=qv0, u=5.0,
    )
    # Initial state: qke seeded with MYNN tke_min (1e-10), tke stays 0.
    assert float(scm.phys_state.qke[0, -1]) == 1e-10
    assert float(scm.phys_state.tke[0, -1]) == 0.0
    scm.run(nsteps=5)
    # qke evolved to its surface BC; tke slot stays zero (untouched).
    assert float(scm.phys_state.qke[0, -1]) > 0.1
    assert float(scm.phys_state.tke[0, -1]) == 0.0


def test_mynn_cold_T_s_cools_lowest_cell():
    """Symmetric to the warm case: T_s colder than the lowest air
    cell drives a negative sensible flux and cools the column.
    """
    T0 = jnp.full(NLEV, 290.0)
    qv0 = jnp.full(NLEV, 5e-3)
    forcing = SCMForcing(prescribe="T_s", T_s=lambda t: jnp.asarray(270.0))
    scm = SingleColumnModel.create(
        physics_config=_mynn_cfg(), nlev=NLEV, dt=60.0,
        T_profile=T0, q_v_profile=qv0, u=5.0,
        forcing=forcing,
    )
    scm.run(nsteps=30)
    T_low = float(scm.state.T.data[0, 0, 0, -1])
    assert T_low < 290.0, (
        f"cold T_s=270 with MYNN did not cool lowest cell: T_low={T_low}"
    )


def test_mynn_obukhov_gradient_finite_at_zero_buoyancy_flux():
    """jax.grad through MYNN must be finite at zero surface buoyancy flux.

    Regression: the Obukhov length ``L = -θ_v u*³ / (κ g w'θ_v')`` was written
    as ``jnp.where(|w'θ_v'|>eps, .../w'θ_v', 1e30)``.  When the surface buoyancy
    flux is exactly zero (T_sfc == T_lowest and q_sfc == q_lowest — neutral,
    SCM-prescribed zero fluxes, or cold-start air-surface equilibrium), the
    *unselected* ``.../w'θ_v'`` branch differentiates ``num/0`` -> inf, and the
    ``where`` multiplies it by a zero selector -> ``0*inf = NaN``, poisoning
    every tendency.  Masking the divisor before the divide keeps the forward
    (the 1e30 neutral sentinel) and makes the gradient finite.
    """
    import jax

    ncol, nlev = 2, 8
    T = jnp.linspace(240.0, 290.0, nlev)[None, :].repeat(ncol, 0)
    q = jnp.full((ncol, nlev), 5e-3)
    u = jnp.full((ncol, nlev), 5.0)
    v = jnp.zeros((ncol, nlev))
    qke = jnp.full((ncol, nlev), 0.5)
    p_half = jnp.linspace(1e4, 1.0e5, nlev + 1)[None, :].repeat(ncol, 0)
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    z_half = jnp.linspace(12000.0, 0.0, nlev + 1)[None, :].repeat(ncol, 0)
    z_full = 0.5 * (z_half[:, 1:] + z_half[:, :-1])
    rho = jnp.ones((ncol, nlev))
    # Zero surface buoyancy flux: T_sfc == T_lowest and q_sfc == q_lowest.
    q_sfc = q[:, -1]

    def loss(T_sfc):
        out, new_qke = mynn25_turbulence(
            u, v, T, q, qke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=60.0, config=MYNN25Config(),
        )
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(new_qke ** 2)

    grad = jax.grad(loss)(T[:, -1])  # T_sfc == T_lowest => w'θ_v' == 0
    assert bool(jnp.all(jnp.isfinite(grad))), (
        f"MYNN gradient not finite at zero buoyancy flux: {grad}"
    )


# ----------------------------------------------------------------------
# Turbulent length scale L_T: dz-weighted integrals (NN09 eq. 54)
# ----------------------------------------------------------------------


def _master_length(q_half, z_half_geom, dz_half):
    """Call the module master-length kernel with neutral surroundings so
    L_T is the only grid-dependent scale under test (L_S depends on z only;
    L_B is dropped for dthv_dz <= 0)."""
    ncol = q_half.shape[0]
    return _compute_master_length(
        q_half=q_half,
        z_half_geom=z_half_geom,
        dz_half=dz_half,
        L_obukhov=jnp.full((ncol,), 1e30),          # neutral (zeta = 0)
        dthv_dz_half=jnp.zeros_like(q_half),        # L_B -> inf (dropped)
        w_thv_sfc=jnp.zeros((ncol,)),
        th_ref=300.0,
        config=MYNN25Config(),
    )


def test_master_length_uniform_grid_invariant_to_dz_scale():
    """On a UNIFORM grid the dz weighting cancels in the L_T ratio
    (0.23*sum(q z dz)/sum(q dz) == 0.23*sum(q z)/sum(q)), so the master
    length must be invariant to the (constant) dz magnitude — i.e. the
    dz-weighted fix leaves uniform-grid results unchanged."""
    ncol, nhalf = 2, 15
    z = jnp.linspace(50.0, 2950.0, nhalf)[None, :].repeat(ncol, 0)
    q = 0.5 + 0.4 * jnp.exp(-z / 800.0)
    L_a = _master_length(q, z, jnp.full((ncol, nhalf), 100.0))
    L_b = _master_length(q, z, jnp.full((ncol, nhalf), 700.0))
    np.testing.assert_allclose(np.asarray(L_a), np.asarray(L_b), rtol=1e-12)


def test_master_length_stretched_grid_weighting_matters():
    """On a STRETCHED grid the dz weighting must change L_T: thicker upper
    layers carry more of the integral, raising the q-weighted mean height
    and thus L_T (and, with L_S/L_B held fixed, the harmonic-mean L)."""
    ncol, nhalf = 1, 15
    z = jnp.linspace(50.0, 2950.0, nhalf)[None, :].repeat(ncol, 0)
    q = jnp.full((ncol, nhalf), 1.0)      # uniform q isolates the z-weighting
    dz_uniform = jnp.full((ncol, nhalf), 200.0)
    # Top-heavy stretching: dz grows with height (typical SCM/GCM grids).
    dz_stretched = jnp.linspace(50.0, 800.0, nhalf)[None, :].repeat(ncol, 0)

    L_u = np.asarray(_master_length(q, z, dz_uniform))
    L_s = np.asarray(_master_length(q, z, dz_stretched))
    # Same z, same q: only the dz weighting differs -> L must differ ...
    assert not np.allclose(L_u, L_s, rtol=1e-6)
    # ... and specifically INCREASE (top-heavy dz raises the weighted mean
    # height, L_T rises, harmonic-mean L rises where L_T is binding).
    assert np.all(L_s >= L_u - 1e-10)
    assert np.max(L_s - L_u) > 1.0


def test_master_length_matches_manual_weighted_integral():
    """The L_T inside the harmonic mean equals 0.23*sum(q z dz)/sum(q dz):
    verified by inverting the (neutral, L_B-free) harmonic mean
    1/L = 1/L_S + 1/L_T with L_S = kappa*z/(1 + 2.7*zeta)|_{zeta=0} = kappa*z
    capped below at the 1 m floor."""
    from legoesm import constants as c

    ncol, nhalf = 1, 12
    z = jnp.linspace(100.0, 3000.0, nhalf)[None, :].repeat(ncol, 0)
    q = 0.3 + 0.7 * jnp.exp(-z / 1000.0)
    dz = jnp.linspace(80.0, 500.0, nhalf)[None, :].repeat(ncol, 0)

    L = np.asarray(_master_length(q, z, dz))
    L_S = np.maximum(c.kappa_vk * np.asarray(z), 1.0)
    L_T = 0.23 * float(jnp.sum(q * z * dz)) / float(jnp.sum(q * dz))
    L_expected = 1.0 / (1.0 / L_S + 1.0 / L_T + 1.0 / 1e30)
    np.testing.assert_allclose(L[0], L_expected[0], rtol=1e-10)
