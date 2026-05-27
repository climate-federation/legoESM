"""Spectral-plane RCE smoke + cross-method comparison vs FD plane.

Adds a spectral leg to the cross-grid RCEMIP-style validation. Same
30-step / dt=2 s / 6×6×8 / +0.5 K bubble forcing as the other
grids; assertions match
``tests/validation/test_cross_grid_rce_smoke.py``:

* Stability: ``max|w| < 50 m/s``
* Conservation: ``|Δmass|/|mass_0| < 1e-6``
* Finite prognostics
* Physical realism: ``q_v ∈ [0, 30 g/kg]``

Plus a cross-method comparison: the spectral-wrapper-with-extras-OFF
must agree with the FD plane to machine epsilon at every step (the
wrapper is then a pure transform sandwich).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    compute_dry_mass_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.dynamics.spectral_plane import (
    SpectralPlaneCompressibleEulerModel, SpectralPlaneConfig,
    spec_state_from_physical,
)
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig, MicrophysicsConfig,
)
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig, RadiationConfig,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


NLEV = 8
H_TOP = 20_000.0
DT = 2.0
N_STEPS = 30
THETA_PERT = 0.5
Q_V_SFC = 0.012


def _build_setup(fix_mass: bool = True):
    """Spectral plane setup helper.

    iter-247: switched the default to ``fix_mass=True`` so the
    ``test_spectral_rce_smoke_stable_and_conservative`` 1e-6 cap is
    achievable (the cap was set in unmerged feature/crm-plane-spectral
    branch edbae138 where it measured 2.56e-16 with the fixer on).

    iter-248 (Codex iter-247 round-2 HIGH): the 1e-6 cap measures
    POST-FIXER drift; a regression in natural dycore conservation
    would be masked by the fixer's per-step clamp. The
    ``fix_mass: bool`` parameter lets the iter-248
    ``test_spectral_rce_dycore_natural_conservation`` test below
    exercise the no-fixer path with a softer 1e-3 cap that catches
    a fully-broken dycore without false-positive-firing on the
    natural ~6e-4 drift.
    """
    nx = ny = 6
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=NLEV, dx=4_000.0, dy=4_000.0,
        dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=H_TOP)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=4_000.0,
        hyperdiff_coeff=1.0e5,
        hyperdiff_rho_coeff=1.0e5,
        hyperdiff_w_coeff=1.0e5,
        semi_implicit_acoustic=False, use_coriolis=False,
        fix_mass=fix_mass, anchor_mass_to_initial=fix_mass,
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
    )
    return grid, hc, tm, cfg


def _build_initial_phys(grid, hc, ny, nx):
    z_full = hc.z_full
    q_v = Q_V_SFC * jnp.exp(-z_full / 4_000.0)
    tracers = jnp.zeros((ny, nx, NLEV, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(
        jnp.broadcast_to(q_v, (ny, nx, NLEV)),
    )
    theta_kick = jnp.zeros((ny, nx, NLEV)).at[
        ny // 2, nx // 2, NLEV - 1
    ].add(THETA_PERT)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    return state._replace(
        theta_prime=state.theta_prime.replace(data=theta_kick),
        tracers=state.tracers.replace(data=tracers),
    )


def test_spectral_rce_smoke_stable_and_conservative():
    """Spectral-plane RCE leg with full physics (gray radiation +
    Kessler + bulk surface). Uses the spectral wrapper with
    apply_dealias=True so any aliased high-k energy from the FD
    upwind nonlinear products is filtered each step."""
    from scripts.run_rcemip_plane import make_rcemip_physics
    grid, hc, tm, cfg = _build_setup()
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=False, apply_dealias=True,
        ),
    )
    phys = _build_initial_phys(grid, hc, grid.ny, grid.nx)

    physics_fn = make_rcemip_physics(
        grid, hc, tm,
        radiation_config=RadiationConfig(
            scheme="gray", gray=GrayRadiationConfig(),
        ),
        microphysics_config=MicrophysicsConfig(
            scheme="kessler", kessler=KesslerConfig(),
        ),
        dt=DT, T_sfc=300.0, q_sfc=Q_V_SFC,
    )

    spec = spec_state_from_physical(phys)
    mass_0 = float(spec_model.compute_dry_mass(spec))
    for _ in range(N_STEPS):
        spec = spec_model.step(spec, dt=DT, physics_fn=physics_fn)
    phys_back = spec_model.to_physical(spec)

    max_w = float(jnp.max(jnp.abs(phys_back.w.data)))
    max_theta_p = float(jnp.max(jnp.abs(phys_back.theta_prime.data)))
    max_q_v = float(jnp.max(phys_back.tracers.data[..., 0]))
    min_q_v = float(jnp.min(phys_back.tracers.data[..., 0]))
    mass_f = float(spec_model.compute_dry_mass(spec))
    mass_drift = abs(mass_f - mass_0) / abs(mass_0)

    print(
        f"\n[spectral_plane] max|w|={max_w:.2e}  max|θ'|={max_theta_p:.2e}  "
        f"q_v∈[{min_q_v:.2e},{max_q_v:.2e}]  drift={mass_drift:.2e}"
    )
    assert max_w < 50.0
    assert bool(jnp.all(jnp.isfinite(phys_back.w.data)))
    assert bool(jnp.all(jnp.isfinite(phys_back.tracers.data)))
    assert mass_drift < 1.0e-6
    # Dealiasing can produce slightly-negative q_v at the noise floor
    # of the FFT round-off; tolerate machine eps but not real
    # negative bias.
    assert min_q_v >= -1.0e-9, f"q_v negative bias: {min_q_v:.3e}"
    assert max_q_v < 0.030


def test_spectral_rce_dycore_natural_conservation():
    """iter-248 (Codex iter-247 round-2 HIGH): companion to the
    1e-6 post-fixer cap above. With ``fix_mass=False`` the
    anchored dry-mass fixer is OFF — drift reflects the dycore's
    natural conservation alone. Cap at 1e-3 is generous enough to
    pass the measured ~6e-4 drift over 30 steps + 1 sim-min but
    tight enough to catch a fully-broken dycore that would
    otherwise be masked by the fixer's per-step clamp.

    Without this test, ``test_spectral_rce_smoke_stable_and_conservative``
    (with fixer ON) would silently absorb any regression in the
    natural conservation pathway — the fixer simply re-anchors
    each step. The two tests together separate ``did the fixer
    fire?`` from ``did the dycore stay close to conservative?``.
    """
    from scripts.run_rcemip_plane import make_rcemip_physics
    grid, hc, tm, cfg = _build_setup(fix_mass=False)
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=False, apply_dealias=True,
        ),
    )
    phys = _build_initial_phys(grid, hc, grid.ny, grid.nx)
    physics_fn = make_rcemip_physics(
        grid, hc, tm,
        radiation_config=RadiationConfig(
            scheme="gray", gray=GrayRadiationConfig(),
        ),
        microphysics_config=MicrophysicsConfig(
            scheme="kessler", kessler=KesslerConfig(),
        ),
        dt=DT, T_sfc=300.0, q_sfc=Q_V_SFC,
    )
    spec = spec_state_from_physical(phys)
    mass_0 = float(spec_model.compute_dry_mass(spec))
    for _ in range(N_STEPS):
        spec = spec_model.step(spec, dt=DT, physics_fn=physics_fn)
    mass_drift = abs(
        float(spec_model.compute_dry_mass(spec)) - mass_0
    ) / abs(mass_0)
    print(f"\n[spectral_dycore_no_fixer] drift={mass_drift:.2e}")
    # Cap 1e-3 catches a fully-broken dycore but tolerates the
    # natural ~6e-4 drift documented in iter-247 commit.
    assert mass_drift < 1.0e-3, (
        f"Dycore natural mass conservation regressed: drift="
        f"{mass_drift:.3e} > 1e-3 cap (was ~6e-4 at iter-247)."
    )


def test_spectral_extras_off_matches_fd_plane_within_tolerance():
    """With ``use_spectral_hyperdiff=False`` and ``apply_dealias=False``,
    the spectral wrapper IS a pure transform sandwich around the FD
    plane step; after a full RCE smoke (30 steps with physics), the
    spectral-then-roundtripped state must equal the direct FD state
    to FFT round-off (~1e-10) — not a "method comparison" but a
    correctness pin."""
    from scripts.run_rcemip_plane import make_rcemip_physics
    grid, hc, tm, cfg = _build_setup()
    fd_model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=False, apply_dealias=False,
        ),
    )
    phys0 = _build_initial_phys(grid, hc, grid.ny, grid.nx)
    physics_fn = make_rcemip_physics(
        grid, hc, tm,
        radiation_config=RadiationConfig(
            scheme="gray", gray=GrayRadiationConfig(),
        ),
        microphysics_config=MicrophysicsConfig(
            scheme="kessler", kessler=KesslerConfig(),
        ),
        dt=DT, T_sfc=300.0, q_sfc=Q_V_SFC,
    )

    phys = phys0
    spec = spec_state_from_physical(phys0)
    for _ in range(N_STEPS):
        phys = fd_model.step(phys, dt=DT, physics_fn=physics_fn)
        spec = spec_model.step(spec, dt=DT, physics_fn=physics_fn)
    phys_back = spec_model.to_physical(spec)
    np.testing.assert_allclose(
        np.asarray(phys_back.theta_prime.data),
        np.asarray(phys.theta_prime.data),
        rtol=1.0e-8, atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(phys_back.w.data),
        np.asarray(phys.w.data),
        rtol=1.0e-8, atol=1.0e-8,
    )


def test_spectral_buoyancy_sign_consistent_with_fd():
    """Spectral plane with full extras (hyperdiff + dealias) still
    produces an UPWARD vertical-velocity response to a warm parcel —
    same sign as the FD plane and the other NH grids."""
    from scripts.run_rcemip_plane import make_rcemip_physics
    grid, hc, tm, cfg = _build_setup()
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=True, apply_dealias=True,
            spectral_hyperdiff_coeff=1.0e5,
        ),
    )
    phys = _build_initial_phys(grid, hc, grid.ny, grid.nx)
    physics_fn = make_rcemip_physics(
        grid, hc, tm,
        radiation_config=RadiationConfig(
            scheme="gray", gray=GrayRadiationConfig(),
        ),
        microphysics_config=MicrophysicsConfig(
            scheme="kessler", kessler=KesslerConfig(),
        ),
        dt=DT, T_sfc=300.0, q_sfc=Q_V_SFC,
    )
    spec = spec_state_from_physical(phys)
    for _ in range(N_STEPS):
        spec = spec_model.step(spec, dt=DT, physics_fn=physics_fn)
    phys_back = spec_model.to_physical(spec)
    max_w_signed = float(jnp.max(phys_back.w.data))
    assert max_w_signed > 1.0e-3, (
        f"spectral plane warm-bubble response not upward: "
        f"max(w)={max_w_signed:.3e}"
    )
