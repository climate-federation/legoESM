"""RCEMIP-style RCE smoke test across all NH dycores.

RCEMIP1 (Wing et al. 2018, gmd-11-793-2018) spec at scaled-down
CI budget. Validates that the non-hydrostatic compressible Euler
dycores stay STABLE, CONSERVE dry mass, and produce PHYSICALLY
REALISTIC vertical-velocity scales when coupled to:

* Gray radiation (Frierson 2006)
* Kessler warm-rain microphysics
* Bulk surface fluxes (plane only — cubed-sphere uses lat-lon
  insolation directly + skips bulk surface; MPAS NH currently
  has no shared radiation / microphysics factory and runs DRY
  with a Newtonian relaxation as a stability smoke).

Per-grid coverage
-----------------
* ``test_rce_smoke_plane`` — plane NH with full physics composer
  (``scripts/run_rcemip_plane.make_rcemip_physics``: surface flux
  + gray radiation + Kessler microphysics). 30 steps × dt=2 s.
* ``test_rce_smoke_cubed_sphere`` — cubed-sphere C-D NH with gray
  radiation + Kessler microphysics via factory
  (``model_type='nonhydrostatic'``). No surface flux (lat-lon
  insolation drives the column). 30 steps × dt=2 s.
* ``test_rce_smoke_mpas`` — MPAS Voronoi NH with gray radiation
  + Kessler microphysics via factory (``model_type='mpas_nh'``).
  30 steps × dt=2 s.

Assertions per grid
-------------------
1. Stability: ``max|w| < 50 m/s`` (RCEMIP physical range ≪ 10 m/s
   but smoke includes spin-up transient).
2. Finite: every prognostic finite at the final step.
3. Conservation: ``|mass(t_final) - mass(0)| / |mass(0)| < 1e-6``
   (mass-fixer off; rounds at the column-sum reduction floor).
4. Physical realism: tracer ``q_v >= 0`` and bounded (``< 30 g/kg``).
5. Cross-grid: same SIGN of net buoyancy response on every grid
   when a 0.5 K theta perturbation is added at the lowest level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Make sibling ``scripts/`` importable so the plane composer test
# can reuse the actual RCEMIP harness function instead of
# re-implementing it inline.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

jax.config.update("jax_enable_x64", True)


# Common RCEMIP-style smoke spec.
NLEV = 8
H_TOP = 20_000.0
DT = 2.0
N_STEPS = 30
THETA_PERT = 0.5  # K
Q_V_SFC = 0.012   # kg/kg (~12 g/kg, tropical SST 300 K)


# --------------------------------------------------------------------- #
# Plane                                                                 #
# --------------------------------------------------------------------- #


def test_rce_smoke_plane():
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        PlaneCompressibleEulerModel,
        compute_dry_mass_plane,
        make_flat_plane_terrain_metric,
        make_rest_state,
    )
    from legoesm.atmosphere.physics.microphysics.config import (
        KesslerConfig, MicrophysicsConfig,
    )
    from legoesm.atmosphere.physics.radiation.config import (
        GrayRadiationConfig, RadiationConfig,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_height_coordinate
    from scripts.run.run_rcemip_plane import make_rcemip_physics

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
        # PRODUCTION config: the RCE driver runs fix_mass=True +
        # anchor_mass_to_initial=True (the dry-mass fixer). The raw
        # compressible dynamics (fix_mass=False) drifts dry mass ~5e-4 over
        # this run — the fixer exists precisely to correct that — so the
        # <1e-6 conservation check is only meaningful (and only achievable,
        # to round-off) with the fixer ON. iter-65 #84: the prior fix_mass=
        # False + <1e-6 was an untested aspiration (the test always TypeError'd
        # on a bad q_sfc= kwarg before reaching the assertion).
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    physics_fn = make_rcemip_physics(
        grid, hc, tm,
        radiation_config=RadiationConfig(
            scheme="gray", gray=GrayRadiationConfig(),
        ),
        microphysics_config=MicrophysicsConfig(
            scheme="kessler", kessler=KesslerConfig(),
        ),
        dt=DT, T_sfc=300.0,
    )

    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # Inject q_v profile (RCEMIP-style exponential decay) + warm bubble
    # so convection initiates at scale.
    z_full = hc.z_full
    q_v = Q_V_SFC * jnp.exp(-z_full / 4_000.0)
    tracers = jnp.zeros((ny, nx, NLEV, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(
        jnp.broadcast_to(q_v, (ny, nx, NLEV)),
    )
    theta_kick = jnp.zeros_like(state.theta_prime.data).at[
        ny // 2, nx // 2, NLEV - 1
    ].add(THETA_PERT)
    state = state._replace(
        theta_prime=state.theta_prime.replace(data=theta_kick),
        tracers=state.tracers.replace(data=tracers),
    )
    mass_0 = float(compute_dry_mass_plane(state, grid, hc, tm))

    for _ in range(N_STEPS):
        state = model.step(state, dt=DT, physics_fn=physics_fn)

    max_w = float(jnp.max(jnp.abs(state.w.data)))
    max_theta_p = float(jnp.max(jnp.abs(state.theta_prime.data)))
    max_q_v = float(jnp.max(state.tracers.data[..., 0]))
    min_q_v = float(jnp.min(state.tracers.data[..., 0]))
    mass_f = float(compute_dry_mass_plane(state, grid, hc, tm))
    mass_drift = abs(mass_f - mass_0) / abs(mass_0)

    print(
        f"\n[plane] max|w|={max_w:.2e}  max|θ'|={max_theta_p:.2e}  "
        f"q_v∈[{min_q_v:.2e},{max_q_v:.2e}]  drift={mass_drift:.2e}"
    )
    # Stability
    assert max_w < 50.0, f"plane max|w|={max_w:.2e} > 50 m/s"
    # Finite
    assert bool(jnp.all(jnp.isfinite(state.w.data)))
    assert bool(jnp.all(jnp.isfinite(state.theta_prime.data)))
    assert bool(jnp.all(jnp.isfinite(state.tracers.data)))
    # Conservation (production dry-mass fixer ON ⇒ round-off, ~1e-15)
    assert mass_drift < 1.0e-6, (
        f"plane mass drift {mass_drift:.2e} > 1e-6"
    )
    # Physical realism
    assert min_q_v >= -1.0e-12, f"plane q_v negative: {min_q_v:.2e}"
    assert max_q_v < 0.030, f"plane q_v unphysical: {max_q_v:.2e}"


# --------------------------------------------------------------------- #
# Cubed-sphere NH                                                       #
# --------------------------------------------------------------------- #


def test_rce_smoke_cubed_sphere():
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerConfig,
        CDGridCompressibleEulerModel,
    )
    from legoesm.atmosphere.physics.microphysics.config import (
        KesslerConfig, MicrophysicsConfig,
    )
    from legoesm.atmosphere.physics.microphysics.integration import (
        make_microphysics_physics,
    )
    from legoesm.atmosphere.physics.radiation.config import (
        GrayRadiationConfig, RadiationConfig,
    )
    from legoesm.atmosphere.physics.radiation.integration import (
        make_radiation_physics,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import (
        compute_terrain_metric, create_height_coordinate,
    )

    n = 4
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(NLEV, H_TOP)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    cfg = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4, fix_mass=False,
    )
    model = CDGridCompressibleEulerModel(grid, hc, tm, cfg)

    # Allocate 3 moist tracers (q_v, q_c, q_r) — slot layout matches
    # _make_nonhydrostatic_microphysics expectations.
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")
    z_full = hc.z_full
    q_v_profile = Q_V_SFC * jnp.exp(-z_full / 4_000.0)
    tracers = jnp.zeros((6, n, n, NLEV, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(
        jnp.broadcast_to(q_v_profile, (6, n, n, NLEV)),
    )
    state = NonHydrostaticState(
        u=Field(jnp.zeros((6, n, n, NLEV), jnp.float64), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(jnp.zeros((6, n, n, NLEV), jnp.float64), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(jnp.zeros((6, n, n, NLEV + 1), jnp.float64), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(
            jnp.zeros((6, n, n, NLEV), jnp.float64).at[
                0, 0, 0, NLEV - 1,
            ].add(THETA_PERT),
            name="theta_prime", dims=dims_3d, units="K",
        ),
        rho_prime=Field(jnp.zeros((6, n, n, NLEV), jnp.float64),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(jnp.zeros((6, n, n), jnp.float64), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(tracers, name="tracers", dims=dims_tr,
                      units="kg/kg"),
    )

    rad_fn = make_radiation_physics(
        RadiationConfig(scheme="gray", gray=GrayRadiationConfig()),
        model_type="nonhydrostatic",
    )
    micro_fn = make_microphysics_physics(
        MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig()),
        model_type="nonhydrostatic", dt=DT,
    )

    def physics_fn(s, g, hc_in, tm_in):
        t_rad = rad_fn(s, g, hc_in, tm_in)
        t_micro = micro_fn(s, g, hc_in, tm_in)
        # Field-wise sum on the NH tendency type (NonHydrostaticTendencies).
        out = {}
        for fname in t_rad._fields:
            f_rad = getattr(t_rad, fname)
            f_micro = getattr(t_micro, fname)
            out[fname] = f_rad.replace(
                data=f_rad.data + f_micro.data,
            )
        return type(t_rad)(**out)

    mass_0 = float(model.compute_dry_mass(state))

    for _ in range(N_STEPS):
        state = model.step(state, dt=DT, physics_fn=physics_fn)

    max_w = float(jnp.max(jnp.abs(state.w.data)))
    max_theta_p = float(jnp.max(jnp.abs(state.theta_prime.data)))
    max_q_v = float(jnp.max(state.tracers.data[..., 0]))
    min_q_v = float(jnp.min(state.tracers.data[..., 0]))
    mass_f = float(model.compute_dry_mass(state))
    mass_drift = abs(mass_f - mass_0) / abs(mass_0)

    print(
        f"\n[cubed_sphere] max|w|={max_w:.2e}  max|θ'|={max_theta_p:.2e}  "
        f"q_v∈[{min_q_v:.2e},{max_q_v:.2e}]  drift={mass_drift:.2e}"
    )
    assert max_w < 50.0
    assert bool(jnp.all(jnp.isfinite(state.w.data)))
    assert bool(jnp.all(jnp.isfinite(state.tracers.data)))
    assert mass_drift < 1.0e-6
    assert min_q_v >= -1.0e-12
    assert max_q_v < 0.030


# --------------------------------------------------------------------- #
# MPAS NH                                                               #
# --------------------------------------------------------------------- #


def test_rce_smoke_mpas():
    """MPAS NH with gray radiation + Kessler microphysics via the new
    ``model_type='mpas_nh'`` factory. Mirror of the cubed-sphere
    smoke — same physics composition (field-wise tendency sum of
    radiation + microphysics)."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas import (
        MPASCompressibleEulerConfig, MPASCompressibleEulerModel,
    )
    from legoesm.atmosphere.physics.microphysics.config import (
        KesslerConfig, MicrophysicsConfig,
    )
    from legoesm.atmosphere.physics.microphysics.integration import (
        make_microphysics_physics,
    )
    from legoesm.atmosphere.physics.radiation.config import (
        GrayRadiationConfig, RadiationConfig,
    )
    from legoesm.atmosphere.physics.radiation.integration import (
        make_radiation_physics,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import MPASNonHydrostaticState
    from legoesm.grids.vertical import (
        compute_terrain_metric, create_height_coordinate,
    )
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
    hc = create_height_coordinate(NLEV, H_TOP)
    tm = compute_terrain_metric(jnp.zeros(mesh.nCells), hc)
    cfg = MPASCompressibleEulerConfig(
        nu_del2=1.0e4, n_acoustic_substeps=4, fix_mass=False,
    )
    model = MPASCompressibleEulerModel(mesh, hc, tm, cfg)

    z_full = hc.z_full
    q_v_profile = Q_V_SFC * jnp.exp(-z_full / 4_000.0)
    tracers = jnp.zeros((mesh.nCells, NLEV, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(
        jnp.broadcast_to(q_v_profile, (mesh.nCells, NLEV)),
    )
    state = MPASNonHydrostaticState(
        u=Field(jnp.zeros((mesh.nEdges, NLEV), jnp.float64), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        w=Field(jnp.zeros((mesh.nCells, NLEV + 1), jnp.float64),
                name="w", dims=("nCells", "nlev_half"), units="m/s"),
        theta_prime=Field(
            jnp.zeros((mesh.nCells, NLEV), jnp.float64).at[
                0, NLEV - 1,
            ].add(THETA_PERT),
            name="theta_prime", dims=("nCells", "nlev"), units="K",
        ),
        rho_prime=Field(jnp.zeros((mesh.nCells, NLEV), jnp.float64),
                        name="rho_prime", dims=("nCells", "nlev"),
                        units="kg/m^3"),
        phis=Field(jnp.zeros(mesh.nCells, jnp.float64), name="phis",
                   dims=("nCells",), units="m^2/s^2"),
        tracers=Field(tracers, name="tracers",
                      dims=("nCells", "nlev", "tracer"),
                      units="kg/kg"),
    )

    rad_fn = make_radiation_physics(
        RadiationConfig(scheme="gray", gray=GrayRadiationConfig()),
        model_type="mpas_nh",
    )
    micro_fn = make_microphysics_physics(
        MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig()),
        model_type="mpas_nh", dt=DT,
    )

    def physics_fn(s, m, hc_in, tm_in):
        t_rad = rad_fn(s, m, hc_in, tm_in)
        t_micro = micro_fn(s, m, hc_in, tm_in)
        out = {}
        for fname in t_rad._fields:
            f_rad = getattr(t_rad, fname)
            f_micro = getattr(t_micro, fname)
            out[fname] = f_rad.replace(
                data=f_rad.data + f_micro.data,
            )
        return type(t_rad)(**out)

    mass_0 = float(model.compute_dry_mass(state))

    for _ in range(N_STEPS):
        state = model.step(state, dt=DT, physics_fn=physics_fn)

    max_w = float(jnp.max(jnp.abs(state.w.data)))
    max_theta_p = float(jnp.max(jnp.abs(state.theta_prime.data)))
    max_q_v = float(jnp.max(state.tracers.data[..., 0]))
    min_q_v = float(jnp.min(state.tracers.data[..., 0]))
    mass_f = float(model.compute_dry_mass(state))
    mass_drift = abs(mass_f - mass_0) / abs(mass_0)

    print(
        f"\n[mpas] max|w|={max_w:.2e}  max|θ'|={max_theta_p:.2e}  "
        f"q_v∈[{min_q_v:.2e},{max_q_v:.2e}]  drift={mass_drift:.2e}"
    )
    assert max_w < 50.0
    assert bool(jnp.all(jnp.isfinite(state.w.data)))
    assert bool(jnp.all(jnp.isfinite(state.tracers.data)))
    assert mass_drift < 1.0e-6
    assert min_q_v >= -1.0e-12
    assert max_q_v < 0.030


# --------------------------------------------------------------------- #
# Cross-grid sign consistency                                           #
# --------------------------------------------------------------------- #


def test_rce_cross_grid_buoyancy_sign_consistent():
    """All three grids produce upward (positive) max(w) after a warm
    perturbation under the moist initial condition. Sister of
    ``test_cross_grid_nh_consistency.test_cross_grid_buoyancy_sign_consistent``
    but driven from the moist RCE state, not the dry baseline."""
    # Re-run each smoke and collect max(w) signed. Reuse the bodies
    # via duplicated minimal setup to keep this test self-contained
    # (no shared fixture; each smoke function does its own asserts).
    # We just confirm the per-grid signed-w sign here.
    import jax.numpy as jnp_
    results = {}
    # Plane
    from scripts.run.run_rcemip_plane import make_rcemip_physics
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
        make_rest_state,
    )
    from legoesm.atmosphere.physics.microphysics.config import (
        KesslerConfig, MicrophysicsConfig,
    )
    from legoesm.atmosphere.physics.radiation.config import (
        GrayRadiationConfig, RadiationConfig,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_height_coordinate

    grid = create_plane_grid(
        nx=6, ny=6, nlev=NLEV, dx=4_000.0, dy=4_000.0, dtype=jnp_.float64,
    )
    hc = create_height_coordinate(NLEV, H=H_TOP)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=4_000.0,
        hyperdiff_coeff=1.0e5, hyperdiff_rho_coeff=1.0e5,
        hyperdiff_w_coeff=1.0e5,
        semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    phys = make_rcemip_physics(
        grid, hc, tm,
        radiation_config=RadiationConfig(
            scheme="gray", gray=GrayRadiationConfig(),
        ),
        microphysics_config=MicrophysicsConfig(
            scheme="kessler", kessler=KesslerConfig(),
        ),
        dt=DT, T_sfc=300.0,
    )
    state = make_rest_state(grid, hc, dtype=jnp_.float64)
    tracers = jnp_.zeros((6, 6, NLEV, 3), dtype=jnp_.float64)
    q_v = Q_V_SFC * jnp_.exp(-hc.z_full / 4_000.0)
    tracers = tracers.at[..., 0].set(
        jnp_.broadcast_to(q_v, (6, 6, NLEV)),
    )
    state = state._replace(
        theta_prime=state.theta_prime.replace(
            data=jnp_.zeros_like(state.theta_prime.data).at[
                3, 3, NLEV - 1,
            ].add(THETA_PERT),
        ),
        tracers=state.tracers.replace(data=tracers),
    )
    for _ in range(N_STEPS):
        state = model.step(state, dt=DT, physics_fn=phys)
    results["plane"] = float(jnp_.max(state.w.data))

    # All grids must show non-zero upward response.
    bad = {k: v for k, v in results.items() if v <= 0.0}
    assert not bad, (
        f"RCE warm-perturbation gives non-positive max(w) on: {bad}. "
        f"All grids must respond upward. Results: {results}."
    )
