"""RCEMIP1 plane smoke test (10 steps).

Validates the PR4 RCE harness skeleton — not the full RCEMIP
equilibrium (which requires ~100 days of integration on a 100x100
km / 1 km grid; see ``scripts/run/run_rcemip_plane.py``). The CI test
just confirms the harness composes cleanly: dycore + Smag +
hyperdiff + sponge + tracer transport + bulk-flux + gray
radiation + Kessler microphysics physics_fn → stable,
mass-conserving, no NaN over 10 steps.

iter-247: the radiation backend was originally a 5-day Newtonian
relaxation toward 300 K (the PR4 harness scaffold at commit
04712098); commit 0ec1da4b swapped that for the canonical gray
radiation factory. iter-243 silently dropped radiation entirely
by passing ``radiation_config=None``; iter-247 restored the
canonical gray-radiation + Kessler microphysics coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Add scripts/ to import path so the test can import make_rcemip_physics.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "run"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_rcemip_plane import (  # type: ignore
    _build_rcemip_initial_state,
    make_rcemip_physics,
    _rcemip_qv_profile,
    _rcemip_theta_profile,
)

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    compute_dry_mass_plane,
    make_flat_plane_terrain_metric,
)
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig, MicrophysicsConfig,
)
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig, RadiationConfig,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


# iter-247 (Codex iter-243 round-1 HIGH#2): the original
# scripts/run/run_rcemip_plane.py:_make_rcemip_physics composed bulk
# surface fluxes + 5-day Newtonian relaxation toward 300 K + Kessler
# microphysics. The 0ec1da4b rename to make_rcemip_physics swapped
# Newtonian for the canonical gray-radiation factory. The iter-243
# minimal-rename fix passed radiation_config=None, silently dropping
# radiation coverage — restored here with the canonical replacement.
_RCEMIP_RADIATION_CFG = RadiationConfig(
    scheme="gray", gray=GrayRadiationConfig(),
)
_RCEMIP_MICROPHYSICS_CFG = MicrophysicsConfig(
    scheme="kessler", kessler=KesslerConfig(),
)


jax.config.update("jax_enable_x64", True)


def _setup_rcemip():
    grid = create_plane_grid(
        nx=8, ny=8, nlev=12, dx=4_000.0, dy=4_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(grid.nlev, H=12_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=2_000.0,
        hyperdiff_coeff=1.0e6, hyperdiff_rho_coeff=1.0e6,
        hyperdiff_w_coeff=1.0e6,
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
        fix_mass=True, anchor_mass_to_initial=True,
        use_coriolis=False,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = _build_rcemip_initial_state(grid, hc)
    return model, state, grid, hc, tm


def test_rcemip_profiles_match_wing_2018_values():
    """Wing 2018 Tab A1 expected values at specific altitudes.

    iter-61 (CONV-TRIGGER #83 fix): `_rcemip_theta_profile`/`_rcemip_qv_profile` now
    DELEGATE to the library `make_wing2018_theta_ref_fn`/`wing2018_qv_profile`
    (T-based, hydrostatic — Γ=6.7e-3 is the Wing *temperature* lapse, NOT a θ
    gradient as the old local formula wrongly used it). The old values (θ(15 km)=
    400.5 K const cap) were a too-stable, near-zero-deep-CAPE IC that left RCE
    laminar; the faithful profile DEEP-CONVECTS (max|w|~5 m/s by ~9 sim-hr). New
    values: θ RISES through the isothermal stratosphere; troposphere conditionally
    unstable."""
    z = jnp.array([0.0, 4_000.0, 15_000.0, 20_000.0])
    theta = _rcemip_theta_profile(z, T_sfc=300.0)
    q_v = _rcemip_qv_profile(z, q_sfc=0.018)
    # Anchors RE-MEASURED 2026-08-06 (SLURM job 9331710, x64) after the RCE300
    # constants were calibrated against the gSAM oracle sounding
    # (T_v0 295->299.274 K, Gamma 0.0067->0.0069901, q_sfc 0.01865->0.0142014).
    # The isothermal cap is T_v0-Γ·z_t = 194.42 K, which is gSAM's own cold
    # point exactly — the calibration is endpoint-constrained through it. gSAM's
    # stratosphere then WARMS, a structure this two-piece form cannot carry
    # (use --sounding). Previous anchors under the disproven 295 K profile:
    # 290.47 / 305.87 / 355.34 / 456.69; and under an intermediate,
    # since-retracted least-squares calibration: 296.63 / 308.50 / 346.54 /
    # 448.40 (that one put the cold point 5.03 K below the oracle).
    assert float(theta[0]) == pytest.approx(295.4701, rel=1.0e-3)
    assert float(theta[1]) == pytest.approx(309.0968, rel=1.0e-3)
    assert float(theta[2]) == pytest.approx(353.5882, rel=1.0e-3)
    # stratosphere is ISOTHERMAL ⇒ θ RISES (was the buggy constant cap)
    assert float(theta[3]) == pytest.approx(454.4805, rel=1.0e-3)
    assert float(theta[3]) > float(theta[2])
    # q_v: two-scale Wing decay; surface=0.018, 4 km ≈ 4.98e-3, strat floor 1e-11
    assert float(q_v[0]) == pytest.approx(0.018, rel=1.0e-6)
    assert float(q_v[1]) == pytest.approx(4.982e-3, rel=1.0e-3)
    assert float(q_v[3]) == pytest.approx(1.0e-11, rel=1.0e-6)


def test_nondefault_q_sfc_flows_consistently_to_theta_and_qv():
    """codex iter-61 [MED]: a non-default ``q_sfc`` must reach BOTH θ and q_v.

    The Wing θ is built on a *virtual*-T hydrostatic base
    ``T_v0 = T_sfc·(1+0.608·q_sfc)``, so θ and q_v MUST share the same surface
    humidity or the IC is hydrostatically/moist-inconsistent. This guards that the
    driver wrappers forward ``q_sfc`` to the library (not silently use the default)
    — i.e. one ``q_sfc`` drives both profiles. Without forwarding, θ would be
    invariant to ``q_sfc`` and this test would fail on the `theta_hot != theta_ref`
    assertion."""
    from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
        make_wing2018_theta_ref_fn,
        wing2018_qv_profile,
    )

    z = jnp.array([0.0, 4_000.0, 15_000.0])
    q_hot = 0.024  # RCE305-like, deliberately != WING_Q_SFC_DEFAULT (0.0142014)

    # (a) driver wrappers must equal the library evaluated at the SAME q_sfc
    theta_drv = _rcemip_theta_profile(z, T_sfc=300.0, q_sfc=q_hot)
    theta_lib = make_wing2018_theta_ref_fn(q_sfc=q_hot)(z)  # default T_v0 (oracle-fit)
    qv_drv = _rcemip_qv_profile(z, q_sfc=q_hot)
    qv_lib = wing2018_qv_profile(z, q_sfc=q_hot)
    for k in range(z.shape[0]):
        assert float(theta_drv[k]) == pytest.approx(float(theta_lib[k]), rel=1e-9)
        assert float(qv_drv[k]) == pytest.approx(float(qv_lib[k]), rel=1e-9)

    # (b) q_sfc must ACTUALLY move θ aloft (proves it's threaded into the
    # virtual-T hydrostatic base, not dropped). NB at z=0, T(0)=T_sfc exactly
    # (the 1+0.608q virtual factor cancels), so θ(0) is q_sfc-INVARIANT — the
    # dependence enters only via the hydrostatic p(z) integral aloft. Check the
    # 4 km level, where a wetter base (higher T_v0) shifts p(z) ⇒ shifts θ.
    theta_ref = _rcemip_theta_profile(z, T_sfc=300.0, q_sfc=0.01865)
    assert float(theta_drv[1]) != pytest.approx(float(theta_ref[1]), rel=1e-5)
    # and q_v surface value must equal the requested q_sfc
    assert float(qv_drv[0]) == pytest.approx(q_hot, rel=1e-6)


def test_rcemip_ic_is_conditionally_unstable():
    """codex iter-61 [MED] / CONV-TRIGGER #83: the IC must be conditionally
    UNSTABLE or RCE never convects. This is the direct guard for the #83 bug —
    the old `θ = T_sfc + Γ·z` profile produced an *actual-T* lapse of only
    ~4.3 K/km (LESS than the moist adiabat ⇒ absolutely stable ⇒ ~0 deep CAPE
    ⇒ permanently laminar). The faithful Wing profile gives ~6.1 K/km (between
    the moist ~5 and dry 9.8 adiabats ⇒ conditionally unstable). Assert the
    lower-tropospheric actual-T lapse sits in the conditionally-unstable band;
    the buggy profile would fail the >5.5 K/km lower bound."""
    from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
        wing2018_temperature_profile,
    )
    import numpy as np

    z = jnp.array([0.0, 5_000.0])
    T = np.asarray(wing2018_temperature_profile(z))  # T_v0=295 RCEMIP fixed
    lapse_K_per_km = -(float(T[1]) - float(T[0])) / 5.0  # K per km, 0→5 km
    # conditionally unstable: steeper than moist-adiabat (~5 K/km warm trop),
    # shallower than dry-adiabat (9.8 K/km). Buggy θ-gradient gave ~4.3 (fails).
    assert 5.5 < lapse_K_per_km < 9.0, (
        f"lower-trop actual-T lapse {lapse_K_per_km:.2f} K/km not conditionally "
        f"unstable (expected ~6.1; <5.5 means the #83 too-stable bug regressed)"
    )


def test_deep_column_reference_state_requires_physical_theta_profile():
    """Regression (iter-4): the constant-θ=300 default reference is
    ISENTROPIC, and an isentropic atmosphere reaches exner=0 (p=T=0) at
    z = c_p·θ/g ≈ 30.7 km.  Over the 33 km RCEMIP column the top levels
    would get exner_ref < 0 → rho_ref = NaN → a step-1 NaN.  Per the
    iter-4 codex review, ``compute_reference_state`` now FAILS FAST
    (raises) on non-positive Exner, and the Wing 2018 θ profile (capped
    at ≈400 K aloft) keeps the reference physical — what
    ``scripts/run/run_rcemip_plane.py`` now passes as ``theta_ref_fn``."""
    import numpy as np
    # Constant-θ default over 33 km now RAISES at construction (was a
    # silent NaN reference → step-1 NaN).
    with pytest.raises(ValueError, match="non-physical reference"):
        create_height_coordinate(30, H=33_000.0, p_sfc=101_480.0)

    # RCEMIP θ profile keeps the reference physical (the fix).
    def theta_ref_fn(z):
        return _rcemip_theta_profile(z, T_sfc=300.0)

    hc = create_height_coordinate(
        30, H=33_000.0, p_sfc=101_480.0, theta_ref_fn=theta_ref_fn,
    )
    assert bool(np.all(np.isfinite(np.asarray(hc.exner_ref))))
    assert bool(np.all(np.isfinite(np.asarray(hc.rho_ref))))
    assert float(np.min(np.asarray(hc.exner_ref))) > 0.0
    assert float(np.min(np.asarray(hc.rho_ref))) > 0.0


def test_rcemip_deep_column_integration_finite():
    """The 33 km / 30-level column (the ``run_rcemip_plane.py`` default)
    must integrate without the step-1 NaN once the reference θ is
    physical — guards the iter-4 fix end-to-end (dry dycore, clean IC)."""
    def theta_ref_fn(z):
        return _rcemip_theta_profile(z, T_sfc=300.0)

    grid = create_plane_grid(
        nx=8, ny=8, nlev=30, dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(
        grid.nlev, H=33_000.0, p_sfc=101_480.0, theta_ref_fn=theta_ref_fn,
    )
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        hyperdiff_coeff=1.0e6, hyperdiff_rho_coeff=1.0e6,
        hyperdiff_w_coeff=1.0e6, smagorinsky_cs=0.0,
        fix_mass=True, anchor_mass_to_initial=True, use_coriolis=False,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = _build_rcemip_initial_state(
        grid, hc, theta_noise_amp=0.0, n_tracers=3,
    )
    for _ in range(5):
        state = model.step(state, dt=1.0)
    assert bool(jnp.all(jnp.isfinite(state.w.data)))
    assert bool(jnp.all(jnp.isfinite(state.theta_prime.data)))
    assert bool(jnp.all(jnp.isfinite(state.rho_prime.data)))


def test_rcemip_smoke_10_step_integration():
    """10-step integration must stay finite + mass-conservative."""
    model, state, grid, hc, tm = _setup_rcemip()
    physics_fn = make_rcemip_physics(grid, hc, tm, radiation_config=_RCEMIP_RADIATION_CFG, microphysics_config=_RCEMIP_MICROPHYSICS_CFG, dt=2.0)
    mass0 = float(compute_dry_mass_plane(state, grid, hc, tm))
    for _ in range(10):
        state = model.step(state, dt=2.0, physics_fn=physics_fn)
    # No NaN / Inf.
    for name, field in (
        ("u", state.u.data), ("w", state.w.data),
        ("theta_p", state.theta_prime.data),
        ("rho_p", state.rho_prime.data),
        ("tracers", state.tracers.data),
    ):
        assert bool(jnp.all(jnp.isfinite(field))), name
    # Mass conserved with anchor fixer.
    mass_f = float(compute_dry_mass_plane(state, grid, hc, tm))
    rel = abs(mass_f - mass0) / abs(mass0)
    assert rel < 1.0e-10, f"Dry mass drift = {rel:.3e}"


def test_rcemip_water_budget_positive_q_v_source():
    """Codex iter-1 minor: latent flux must increase domain-mean
    surface-layer q_v over one physics call; documents that water
    mass is NOT conserved (surface acts as a source) while dry mass
    is."""
    model, state, grid, hc, tm = _setup_rcemip()
    physics_fn = make_rcemip_physics(grid, hc, tm, radiation_config=_RCEMIP_RADIATION_CFG, microphysics_config=_RCEMIP_MICROPHYSICS_CFG, dt=2.0)
    tend = physics_fn(state, grid, hc, tm)
    # surface q_v tendency = lhflx / (rho L_v dz_sfc); for q_lo < q_sfc
    # this is positive.
    k_sfc = state.tracers.data.shape[-2] - 1
    surface_dq = tend.dtracers_dt.data[..., k_sfc, 0]
    mean_dq = float(jnp.mean(surface_dq))
    assert mean_dq > 0.0, (
        f"Surface latent flux tendency not positive: mean = {mean_dq:.3e}; "
        "should be > 0 because q_lo (initial) < q_sfc."
    )


def test_rcemip_physics_fn_is_differentiable():
    """Codex iter-1 finding M4: jax.grad through the bulk-flux +
    Newtonian physics must produce finite gradients. The sqrt in
    wind_speed has a 1 m/s floor so the kink at zero wind is avoided
    by construction; the Exner conversion is a constant scalar."""
    model, state, grid, hc, tm = _setup_rcemip()
    physics_fn = make_rcemip_physics(grid, hc, tm, radiation_config=_RCEMIP_RADIATION_CFG, microphysics_config=_RCEMIP_MICROPHYSICS_CFG, dt=2.0)
    # Scalar loss = sum(dtheta_prime_dt² + dtracers_dt²)
    # differentiated w.r.t. surface theta + q_v perturbations.
    k_sfc = state.theta_prime.data.shape[-1] - 1
    theta_sfc0 = state.theta_prime.data[..., k_sfc]
    qv_sfc0 = state.tracers.data[..., k_sfc, 0]

    def loss(theta_sfc, qv_sfc):
        s = state._replace(
            theta_prime=state.theta_prime.replace(
                data=state.theta_prime.data.at[..., k_sfc].set(theta_sfc),
            ),
            tracers=state.tracers.replace(
                data=state.tracers.data.at[..., k_sfc, 0].set(qv_sfc),
            ),
        )
        tend = physics_fn(s, grid, hc, tm)
        return (
            jnp.sum(tend.dtheta_prime_dt.data ** 2)
            + jnp.sum(tend.dtracers_dt.data ** 2)
        )

    g_theta, g_qv = jax.grad(loss, argnums=(0, 1))(theta_sfc0, qv_sfc0)
    assert bool(jnp.all(jnp.isfinite(g_theta)))
    assert bool(jnp.all(jnp.isfinite(g_qv)))
    # Non-trivial (perturbing theta_sfc / qv_sfc changes flux ->
    # tendencies -> loss).
    assert float(jnp.max(jnp.abs(g_theta))) > 0.0
    assert float(jnp.max(jnp.abs(g_qv))) > 0.0


def test_rcemip_physics_fn_returns_correct_tendency_shape():
    """``make_rcemip_physics`` must return a PlaneNonHydrostaticTendencies
    pytree with shapes matching the state."""
    model, state, grid, hc, tm = _setup_rcemip()
    physics_fn = make_rcemip_physics(grid, hc, tm, radiation_config=_RCEMIP_RADIATION_CFG, microphysics_config=_RCEMIP_MICROPHYSICS_CFG, dt=2.0)
    tend = physics_fn(state, grid, hc, tm)
    assert tend.du_dt.data.shape == state.u.data.shape
    assert tend.dv_dt.data.shape == state.v.data.shape
    assert tend.dw_dt.data.shape == state.w.data.shape
    assert tend.dtheta_prime_dt.data.shape == state.theta_prime.data.shape
    assert tend.drho_prime_dt.data.shape == state.rho_prime.data.shape
    assert tend.dphis_dt.data.shape == state.phis.data.shape
    assert tend.dtracers_dt.data.shape == state.tracers.data.shape
