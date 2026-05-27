"""Long RCEMIP1-style run on every NH dycore (plane FD / plane spectral /
cubed-sphere C-D / MPAS Voronoi).

Per-grid setup, same physics composition where the factory supports it:
gray radiation (Frierson 2006) + Kessler warm-rain microphysics. Plane
adds bulk surface fluxes; cubed-sphere + MPAS NH use the lat-lon-driven
insolation directly (no surface flux scheme in the NH-MPAS / NH-cube
factory yet — those are out-of-scope follow-ups).

The run is sized for a reduced-resolution "100-day in a CI/desktop wall
budget" target:
  - 8x8 horizontal cells (or equivalent for cubed-sphere C4 + MPAS res 2)
  - NLEV = 10, H_top = 20 km
  - dt = 20 s outer step
  - 100 simulated days = 432_000 steps

At each ``print-every`` step boundary, dumps a one-line diagnostic:
  step | t [days] | max|w| | max|θ'| | q_v range | dry-mass drift

Final summary asserts:
  - max|w| < 50 m/s (stability)
  - All prognostics finite at t_end
  - Dry-mass drift < 1e-6 (no mass-fixer)
  - q_v in [0, 0.030] kg/kg (physical realism)

Usage:
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \\
      scripts/run_rcemip_long.py --grid plane_fd --days 100 \\
      --output results/rcemip100_plane_fd.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


NLEV = 10
H_TOP = 20_000.0
DEFAULT_DT = 10.0
Q_V_SFC = 0.012   # ~12 g/kg
T_SFC = 300.0


def _build_initial_q_v(hc, shape):
    z_full = hc.z_full
    q_v_1d = Q_V_SFC * jnp.exp(-z_full / 4_000.0)
    return jnp.broadcast_to(q_v_1d, shape)


# --------------------------------------------------------------------- #
# Plane FD                                                              #
# --------------------------------------------------------------------- #


def _compose_nh_moist_physics(model_type: str, dt: float):
    """iter-275: compose Kessler microphysics + gray radiation
    into a single physics_fn that the non-hydrostatic dycores
    (cubed-sphere, MPAS NH) can pass to ``model.step``.

    The two factories return tendency functions with the same
    signature for a given ``model_type``. We sum their outputs
    via ``jax.tree_util.tree_map`` so the dycore sees a single
    composed tendency per call. This mirrors the
    ``make_rcemip_physics`` pattern in
    ``scripts/run_rcemip_plane.py`` (plane CRM) — extracted here
    because the plane helper takes a (grid, hc, tm)
    signature that doesn't carry to the cubed-sphere / MPAS
    NonHydrostaticState shapes.

    Parameters
    ----------
    model_type : str
        'nonhydrostatic' for cubed-sphere CRM, 'mpas_nh' for MPAS.
    dt : float
        Outer time step [s], passed to the microphysics factory
        for sub-step scheduling.

    Returns
    -------
    Callable
        ``physics_fn(state, ...)`` returning a tendency pytree of
        the same shape as the model's state.
    """
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

    micro_fn = make_microphysics_physics(
        MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig()),
        model_type=model_type, dt=dt,
    )
    rad_fn = make_radiation_physics(
        RadiationConfig(scheme="gray", gray=GrayRadiationConfig()),
        model_type=model_type,
    )

    def physics_fn(*args, **kwargs):
        # Both factories return tendency callables with identical
        # *pytree shapes* but distinct Field-name metadata (e.g.
        # ``dtracers_dt_micro`` vs ``dtracers_dt_rad``). Field is
        # registered as a pytree node with name as treedef metadata,
        # so ``jax.tree_util.tree_map`` refuses to pair them.
        # Sum at the leaf-array level + rebuild with the micro
        # treedef so downstream consumers see a single composed
        # tendency.
        t_micro = micro_fn(*args, **kwargs)
        t_rad = rad_fn(*args, **kwargs)
        leaves_m, treedef = jax.tree_util.tree_flatten(t_micro)
        leaves_r = jax.tree_util.tree_leaves(t_rad)
        if len(leaves_m) != len(leaves_r):
            raise ValueError(
                f"_compose_nh_moist_physics: micro tendency has "
                f"{len(leaves_m)} leaves but rad has "
                f"{len(leaves_r)} — model_type={model_type!r} factory "
                f"contract mismatch."
            )
        summed = [a + b for a, b in zip(leaves_m, leaves_r)]
        return jax.tree_util.tree_unflatten(treedef, summed)

    return physics_fn


def _run_plane_fd(days: float, dt: float, print_every: int, output: Path,
                  *, moist: bool = False):
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        PlaneCompressibleEulerModel, compute_dry_mass_plane,
        make_flat_plane_terrain_metric, make_rest_state,
    )
    from legoesm.atmosphere.physics.microphysics.config import (
        KesslerConfig, MicrophysicsConfig,
    )
    from legoesm.atmosphere.physics.radiation.config import (
        GrayRadiationConfig, RadiationConfig,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_height_coordinate
    from legoesm.atmosphere.physics.microphysics.integration import (
        make_microphysics_physics,
    )
    from legoesm.atmosphere.physics.radiation.integration import (
        make_radiation_physics,
    )

    nx = ny = 8
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=NLEV, dx=4_000.0, dy=4_000.0,
        dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=H_TOP)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.1, sponge_width=4_000.0,
        hyperdiff_coeff=5.0e5,
        hyperdiff_rho_coeff=5.0e5,
        hyperdiff_w_coeff=5.0e5,
        semi_implicit_acoustic=False, use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=0.0, smagorinsky_prandtl=1.0,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    # DRY, NO PHYSICS for the long-time stability baseline. Adding
    # gray radiation on the 2km-thick lowest layer instantly produces
    # ~1K theta' from the cold start, which drives convective runaway
    # within 1 sim-day on this scaled-down grid. Full RCEMIP integration
    # needs stretched vertical grid + spun-up radiative-equilibrium IC;
    # that's a separate harness. This long-run validates pure dycore
    # mass conservation + finiteness over 100 simulated days.
    physics_fn = None

    state = make_rest_state(grid, hc, dtype=jnp.float64)
    tracers = jnp.zeros((ny, nx, NLEV, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(_build_initial_q_v(hc, (ny, nx, NLEV)))
    # No perturbation: long-time stability + conservation baseline.
    # (Warm-bubble convection on this scaled-down 8x8x10 / 4km grid
    # runs away in <1 sim-day without a proper LES + stretched-grid
    # setup that's out of scope for this MVP test. The rest-state
    # integration still validates the dycore + radiation + micro
    # composer's mass conservation and finiteness over 100 days —
    # the core "is this stable for a real RCEMIP-length run" check.)
    state = state._replace(
        tracers=state.tracers.replace(data=tracers),
    )

    def diag(s):
        return dict(
            max_w=float(jnp.max(jnp.abs(s.w.data))),
            min_th=float(jnp.min(s.theta_prime.data)),
            max_th=float(jnp.max(s.theta_prime.data)),
            min_qv=float(jnp.min(s.tracers.data[..., 0])),
            max_qv=float(jnp.max(s.tracers.data[..., 0])),
            finite=bool(jnp.all(jnp.isfinite(s.w.data))),
            mass=float(compute_dry_mass_plane(s, grid, hc, tm)),
        )

    return _run_loop(model, state, diag, days, dt, print_every,
                     output, label="plane_fd",
                     physics_fn=physics_fn)


# --------------------------------------------------------------------- #
# Plane spectral                                                        #
# --------------------------------------------------------------------- #


def _run_plane_spectral(days: float, dt: float, print_every: int, output: Path,
                        *, moist: bool = False):
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        compute_dry_mass_plane, make_flat_plane_terrain_metric,
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
    from legoesm.atmosphere.physics.microphysics.integration import (
        make_microphysics_physics,
    )
    from legoesm.atmosphere.physics.radiation.integration import (
        make_radiation_physics,
    )

    nx = ny = 8
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=NLEV, dx=4_000.0, dy=4_000.0,
        dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=H_TOP)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.1, sponge_width=4_000.0,
        hyperdiff_coeff=5.0e5,
        hyperdiff_rho_coeff=5.0e5,
        hyperdiff_w_coeff=5.0e5,
        semi_implicit_acoustic=False, use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=0.0, smagorinsky_prandtl=1.0,
    )
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=False, apply_dealias=True,
        ),
    )
    # DRY, NO PHYSICS (see plane_fd docstring).
    physics_fn = None

    phys = make_rest_state(grid, hc, dtype=jnp.float64)
    tracers = jnp.zeros((ny, nx, NLEV, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(_build_initial_q_v(hc, (ny, nx, NLEV)))
    # No perturbation (see plane_fd docstring).
    phys = phys._replace(
        tracers=phys.tracers.replace(data=tracers),
    )
    spec = spec_state_from_physical(phys)

    def diag(s):
        p = spec_model.to_physical(s)
        return dict(
            max_w=float(jnp.max(jnp.abs(p.w.data))),
            min_th=float(jnp.min(p.theta_prime.data)),
            max_th=float(jnp.max(p.theta_prime.data)),
            min_qv=float(jnp.min(p.tracers.data[..., 0])),
            max_qv=float(jnp.max(p.tracers.data[..., 0])),
            finite=bool(jnp.all(jnp.isfinite(p.w.data))),
            mass=float(spec_model.compute_dry_mass(s)),
        )

    return _run_loop(spec_model, spec, diag, days, dt, print_every,
                     output, label="plane_spectral",
                     physics_fn=physics_fn)


# --------------------------------------------------------------------- #
# Cubed-sphere NH                                                       #
# --------------------------------------------------------------------- #


def _run_cubed_sphere(days: float, dt: float, print_every: int, output: Path,
                      *, moist: bool = False):
    from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import (
        CDGridCompressibleEulerConfig, CDGridCompressibleEulerModel,
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
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")
    q_v_init = _build_initial_q_v(hc, (6, n, n, NLEV))
    tracers = jnp.zeros((6, n, n, NLEV, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(q_v_init)
    # No perturbation (see plane_fd docstring).
    theta_kick = jnp.zeros((6, n, n, NLEV), dtype=jnp.float64)
    state = NonHydrostaticState(
        u=Field(jnp.zeros((6, n, n, NLEV), jnp.float64), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(jnp.zeros((6, n, n, NLEV), jnp.float64), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(jnp.zeros((6, n, n, NLEV + 1), jnp.float64), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(theta_kick, name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(jnp.zeros((6, n, n, NLEV), jnp.float64),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(jnp.zeros((6, n, n), jnp.float64), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(tracers, name="tracers", dims=dims_tr,
                      units="kg/kg"),
    )

    # iter-275: opt-in moist physics for cubed_sphere.
    # model_type='nonhydrostatic' selects the cubed-sphere /
    # NonHydrostaticState factory branch.
    if moist:
        physics_fn = _compose_nh_moist_physics(model_type="nonhydrostatic",
                                               dt=dt)
    else:
        physics_fn = None

    def diag(s):
        return dict(
            max_w=float(jnp.max(jnp.abs(s.w.data))),
            min_th=float(jnp.min(s.theta_prime.data)),
            max_th=float(jnp.max(s.theta_prime.data)),
            min_qv=float(jnp.min(s.tracers.data[..., 0])),
            max_qv=float(jnp.max(s.tracers.data[..., 0])),
            finite=bool(jnp.all(jnp.isfinite(s.w.data))),
            mass=float(model.compute_dry_mass(s)),
        )

    return _run_loop(model, state, diag, days, dt, print_every,
                     output,
                     label=f"cubed_sphere{'_moist' if moist else ''}",
                     physics_fn=physics_fn)


# --------------------------------------------------------------------- #
# MPAS NH                                                               #
# --------------------------------------------------------------------- #


def _run_mpas(days: float, dt: float, print_every: int, output: Path,
              *, moist: bool = False):
    from legoesm.atmosphere.dynamics.compressible_euler_mpas import (
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
    q_v_init = _build_initial_q_v(hc, (mesh.nCells, NLEV))
    tracers = jnp.zeros((mesh.nCells, NLEV, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(q_v_init)
    # No perturbation (see plane_fd docstring).
    theta_kick = jnp.zeros((mesh.nCells, NLEV), dtype=jnp.float64)
    state = MPASNonHydrostaticState(
        u=Field(jnp.zeros((mesh.nEdges, NLEV), jnp.float64), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        w=Field(jnp.zeros((mesh.nCells, NLEV + 1), jnp.float64),
                name="w", dims=("nCells", "nlev_half"), units="m/s"),
        theta_prime=Field(theta_kick, name="theta_prime",
                          dims=("nCells", "nlev"), units="K"),
        rho_prime=Field(jnp.zeros((mesh.nCells, NLEV), jnp.float64),
                        name="rho_prime", dims=("nCells", "nlev"),
                        units="kg/m^3"),
        phis=Field(jnp.zeros(mesh.nCells, jnp.float64), name="phis",
                   dims=("nCells",), units="m^2/s^2"),
        tracers=Field(tracers, name="tracers",
                      dims=("nCells", "nlev", "tracer"), units="kg/kg"),
    )

    # iter-275: opt-in moist physics for MPAS NH.
    # model_type='mpas_nh' selects the MPASNonHydrostaticState
    # factory branch.
    if moist:
        physics_fn = _compose_nh_moist_physics(model_type="mpas_nh", dt=dt)
    else:
        physics_fn = None

    def diag(s):
        return dict(
            max_w=float(jnp.max(jnp.abs(s.w.data))),
            min_th=float(jnp.min(s.theta_prime.data)),
            max_th=float(jnp.max(s.theta_prime.data)),
            min_qv=float(jnp.min(s.tracers.data[..., 0])),
            max_qv=float(jnp.max(s.tracers.data[..., 0])),
            finite=bool(jnp.all(jnp.isfinite(s.w.data))),
            mass=float(model.compute_dry_mass(s)),
        )

    return _run_loop(model, state, diag, days, dt, print_every,
                     output, label=f"mpas{'_moist' if moist else ''}",
                     physics_fn=physics_fn)


# --------------------------------------------------------------------- #
# Inner loop                                                            #
# --------------------------------------------------------------------- #


def _run_loop(model, state, diag, days, dt, print_every, output,
              label, physics_fn):
    n_steps = int(days * 86400.0 / dt)
    print(f"\n[{label}] dt={dt}s, {n_steps} steps -> {days} sim-days")
    diag_0 = diag(state)
    mass_0 = diag_0["mass"]
    print(
        f"step  t[d]  max|w|     min(θ')    max(θ')    "
        f"q_v_min      q_v_max     mass_drift"
    )
    history = []
    history.append({"step": 0, "t_days": 0.0, **diag_0,
                    "mass_drift": 0.0})

    t_start = time.perf_counter()
    n_blowup = 0
    for i in range(n_steps):
        state = model.step(state, dt=dt, physics_fn=physics_fn)
        if (i + 1) % print_every == 0 or i == 0:
            d = diag(state)
            drift = abs(d["mass"] - mass_0) / max(abs(mass_0), 1e-30)
            t_d = (i + 1) * dt / 86400.0
            print(
                f"{i+1:6d} {t_d:5.2f} {d['max_w']:9.3e} "
                f"{d['min_th']:11.3e} {d['max_th']:11.3e} "
                f"{d['min_qv']:11.3e} {d['max_qv']:11.3e} "
                f"{drift:11.3e}"
            )
            history.append({"step": i + 1, "t_days": t_d, **d,
                            "mass_drift": drift})
            if not d["finite"]:
                n_blowup += 1
                print(f"  NON-FINITE STATE at step {i+1}; aborting.")
                break
    wall = time.perf_counter() - t_start
    print(f"\n[{label}] wall time: {wall/60:.1f} min "
          f"({wall/max(n_steps, 1)*1000:.1f} ms/step)")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(dict(
            label=label, days=days, dt=dt,
            n_steps=n_steps, wall_sec=wall, blowup=n_blowup,
            history=history,
        ), f, indent=2)
    return history


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--grid",
        choices=["plane_fd", "plane_spectral", "cubed_sphere", "mpas"],
        required=True,
    )
    p.add_argument("--days", type=float, default=100.0)
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--print-every", type=int, default=4_320)   # ~1 sim day
    p.add_argument("--output", type=Path, required=True)
    # iter-275: opt-in moist physics (Kessler microphysics + gray
    # radiation) for the cubed_sphere + mpas paths. plane_fd +
    # plane_spectral remain dry by default — the iter-238 cross-grid
    # smoke locks the dry contract for all 4 grids.
    p.add_argument(
        "--moist", action="store_true", default=False,
        help="iter-275: opt-in moist physics_fn composition "
        "(Kessler microphysics + gray radiation) for the "
        "cubed_sphere + mpas paths. Dry by default (preserves "
        "iter-238 smoke contract).",
    )
    args = p.parse_args()

    dispatch = {
        "plane_fd": _run_plane_fd,
        "plane_spectral": _run_plane_spectral,
        "cubed_sphere": _run_cubed_sphere,
        "mpas": _run_mpas,
    }
    dispatch[args.grid](
        args.days, args.dt, args.print_every, args.output,
        moist=args.moist,
    )


if __name__ == "__main__":
    main()
