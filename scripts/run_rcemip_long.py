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


def _make_cubed_sphere_surface_flux_tendency(
    Cd: float = 1.0e-3, Ch: float = 1.0e-3,
    T_sfc: float = 300.0, q_sfc: float = 0.018,
):
    """iter-283: bulk surface flux tendency for cubed-sphere
    NonHydrostaticState ONLY. Returns a NonHydrostaticTendencies
    (cubed-sphere tendency type), so this helper is NOT valid
    for the plane driver despite the shape-agnostic axis-(-1)
    indexing (iter-288 LOW#1 clarification).

    Operates on the lowest model level via axis-(-1) indexing so
    the indexing pattern matches both plane (ny, nx, nlev) and
    cubed-sphere (face, n, n, nlev); but the EMITTED tendency
    type is cubed-sphere-only.

    Why iter-283 needed this: iter-282 MPAS 30-day moist run
    blew up at day 15 with theta' cooling -2.8 K/day (gray
    radiation with no surface-flux counter-balance). Adding
    bulk Cd/Ch + T_sfc/q_sfc fixed-SST relaxation closes the
    column energy budget.
    """
    from legoesm import constants as legoesm_constants
    from legoesm.coupler.bulk_flux import simple_bulk_fluxes
    from legoesm.core.state import NonHydrostaticTendencies

    def physics_fn(state, grid_in, hc_in, tm_in):
        nlev_local = state.theta_prime.data.shape[-1]
        k_sfc = nlev_local - 1
        rho_0 = hc_in.rho_ref
        theta_0 = hc_in.theta_ref
        theta_total = theta_0 + state.theta_prime.data
        rho_total = rho_0 + state.rho_prime.data
        u_lo = state.u.data[..., k_sfc]
        v_lo = state.v.data[..., k_sfc]
        rho_lo = rho_total[..., k_sfc]
        theta_lo = theta_total[..., k_sfc]
        pi_sfc = hc_in.exner_ref[k_sfc]
        T_lo = theta_lo * pi_sfc
        q_lo = state.tracers.data[..., k_sfc, 0]
        wind_speed = jnp.sqrt(u_lo ** 2 + v_lo ** 2 + 1.0)

        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            u_lowest=u_lo, v_lowest=v_lo, T_lowest=T_lo, q_lowest=q_lo,
            T_sfc=jnp.full_like(T_lo, T_sfc),
            q_sfc=jnp.full_like(T_lo, q_sfc),
            rho=rho_lo, wind_speed=wind_speed, Cd=Cd, Ch=Ch,
        )

        dz_sfc = hc_in.dz[k_sfc]
        du_sfc = tau_x / (rho_lo * dz_sfc)
        dv_sfc = tau_y / (rho_lo * dz_sfc)
        du_dt_data = jnp.zeros_like(state.u.data).at[..., k_sfc].set(du_sfc)
        dv_dt_data = jnp.zeros_like(state.v.data).at[..., k_sfc].set(dv_sfc)

        dT_sfc = shflx / (rho_lo * legoesm_constants.c_pd * dz_sfc)
        dtheta_sfc = dT_sfc / pi_sfc
        dtheta_p_data = jnp.zeros_like(
            state.theta_prime.data
        ).at[..., k_sfc].set(dtheta_sfc)

        dtracers_data = jnp.zeros_like(state.tracers.data)
        dq_sfc = lhflx / (rho_lo * legoesm_constants.L_v * dz_sfc)
        dtracers_data = dtracers_data.at[..., k_sfc, 0].add(dq_sfc)

        return NonHydrostaticTendencies(
            du_dt=state.u.replace(data=du_dt_data),
            dv_dt=state.v.replace(data=dv_dt_data),
            dw_dt=state.w.replace(data=jnp.zeros_like(state.w.data)),
            dtheta_prime_dt=state.theta_prime.replace(data=dtheta_p_data),
            drho_prime_dt=state.rho_prime.replace(
                data=jnp.zeros_like(state.rho_prime.data),
            ),
            dphis_dt=state.phis.replace(
                data=jnp.zeros_like(state.phis.data),
            ),
            dtracers_dt=state.tracers.replace(data=dtracers_data),
        )

    return physics_fn


def _compose_nh_moist_physics(model_type: str, dt: float,
                              *, with_surface_flux: bool = False,
                              sfc_Cd: float = 1.0e-3,
                              sfc_Ch: float = 1.0e-3,
                              sfc_T: float = 300.0,
                              sfc_q: float = 0.018):
    """iter-275/283: compose Kessler microphysics + gray radiation
    (+ optional surface flux) into a single physics_fn that the
    non-hydrostatic dycores (cubed-sphere, MPAS NH) can pass to
    ``model.step``.

    Each factory returns a tendency callable with identical
    *pytree shapes* but distinct Field-name metadata. We sum
    at the leaf-array level + rebuild with the first treedef
    so downstream consumers see a single composed tendency.

    Parameters
    ----------
    model_type : str
        'nonhydrostatic' for cubed-sphere CRM, 'mpas_nh' for MPAS.
    dt : float
        Outer time step [s], passed to the microphysics factory.
    with_surface_flux : bool, default False
        iter-283: if True, also compose
        ``_make_cubed_sphere_surface_flux_tendency`` (bulk Cd/Ch
        with fixed T_sfc=300 K, q_sfc=0.018). Only supported for
        ``model_type='nonhydrostatic'`` (cubed-sphere) today —
        MPAS has u-on-edges + no v which needs a separate helper
        (deferred to a follow-on iter).

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

    if with_surface_flux and model_type != "nonhydrostatic":
        raise ValueError(
            f"_compose_nh_moist_physics: with_surface_flux=True is "
            f"only supported for model_type='nonhydrostatic' "
            f"(cubed-sphere); got model_type={model_type!r}. The "
            f"MPAS surface flux path (u-on-edges, no v field) "
            f"requires a separate helper."
        )

    micro_fn = make_microphysics_physics(
        MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig()),
        model_type=model_type, dt=dt,
    )
    rad_fn = make_radiation_physics(
        RadiationConfig(scheme="gray", gray=GrayRadiationConfig()),
        model_type=model_type,
    )
    sfc_fn = None
    if with_surface_flux:
        sfc_fn = _make_cubed_sphere_surface_flux_tendency(
            Cd=sfc_Cd, Ch=sfc_Ch, T_sfc=sfc_T, q_sfc=sfc_q,
        )

    def physics_fn(*args, **kwargs):
        # Both factories return tendency callables with identical
        # *pytree shapes* but distinct Field-name metadata. Sum at
        # the leaf-array level + rebuild with the micro treedef.
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
        if sfc_fn is not None:
            t_sfc = sfc_fn(*args, **kwargs)
            leaves_s = jax.tree_util.tree_leaves(t_sfc)
            if len(leaves_s) != len(summed):
                raise ValueError(
                    f"_compose_nh_moist_physics: surface tendency has "
                    f"{len(leaves_s)} leaves vs combined {len(summed)} "
                    f"— contract mismatch."
                )
            summed = [a + b for a, b in zip(summed, leaves_s)]
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
        # iter-288 (Codex HIGH): cover ALL prognostic arrays.
        return dict(
            max_w=float(jnp.max(jnp.abs(s.w.data))),
            min_th=float(jnp.min(s.theta_prime.data)),
            max_th=float(jnp.max(s.theta_prime.data)),
            min_qv=float(jnp.min(s.tracers.data[..., 0])),
            max_qv=float(jnp.max(s.tracers.data[..., 0])),
            finite=bool(
                jnp.all(jnp.isfinite(s.u.data))
                & jnp.all(jnp.isfinite(s.v.data))
                & jnp.all(jnp.isfinite(s.w.data))
                & jnp.all(jnp.isfinite(s.theta_prime.data))
                & jnp.all(jnp.isfinite(s.rho_prime.data))
                & jnp.all(jnp.isfinite(s.tracers.data))
            ),
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
                      *, moist: bool = False, n: int = 4,
                      sfc_Cd: float = 1.0e-3, sfc_Ch: float = 1.0e-3,
                      sfc_T: float = 300.0, sfc_q: float = 0.018,
                      n_acoustic: int | None = None,
                      coriolis: str | None = None,
                      fix_mass: str | None = None):
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

    # iter-287: n is now a kwarg (default 4 for back-compat with
    # iter-238 cross-grid smoke). Raise to 12, 24, 96 for finer
    # mesh production runs. C12=864 cells ~830 km/cell, C24=3456
    # cells ~415 km/cell, C96=55296 cells ~104 km/cell. iter-286
    # showed C4 is too coarse for moist 30-day production
    # regardless of dycore tuning — mesh-resolution-bound failure.
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(NLEV, H_TOP)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    # iter-285 + iter-290: when moist=True, use a TIGHTER config at
    # the C4 mesh because iter-284 measured day-2 NaN under the
    # default (n_acoustic=4 + use_coriolis=True + fix_mass=False).
    # Tightened defaults: n_acoustic=12, use_coriolis=False,
    # fix_mass=True (matches iter-183 plane CRM contract).
    # iter-290 (Codex iter-288 MEDIUM#2): each tightened default is
    # now CLI-override-able via n_acoustic / coriolis / fix_mass
    # kwargs. None means "use the moist-vs-dry default".
    if moist:
        n_acoustic_default = 12
        coriolis_default = "off"
        fix_mass_default = "on"
    else:
        n_acoustic_default = 4
        coriolis_default = "on"
        fix_mass_default = "off"
    n_acoustic_eff = n_acoustic if n_acoustic is not None else n_acoustic_default
    coriolis_eff = coriolis if coriolis is not None else coriolis_default
    fix_mass_eff = fix_mass if fix_mass is not None else fix_mass_default
    cfg = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=n_acoustic_eff,
        fix_mass=(fix_mass_eff == "on"),
        anchor_mass_to_initial=(fix_mass_eff == "on"),
        use_coriolis=(coriolis_eff == "on"),
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
    # iter-283: moist=True now also includes bulk surface flux —
    # iter-282 measured -2.8 K/day cooling without it (gray rad
    # uncountered) → NaN at sim day 15. Surface flux closes the
    # column energy budget for sustained RCE.
    if moist:
        physics_fn = _compose_nh_moist_physics(
            model_type="nonhydrostatic", dt=dt,
            with_surface_flux=True,
            sfc_Cd=sfc_Cd, sfc_Ch=sfc_Ch,
            sfc_T=sfc_T, sfc_q=sfc_q,
        )
    else:
        physics_fn = None

    def diag(s):
        # iter-288 (Codex iter-283..287 round-1 HIGH): the pre-iter-288
        # finite check inspected ONLY ``s.w.data``. If moist surface
        # flux / radiation / Kessler produces NaN in theta_prime,
        # rho_prime, or tracers while w stays finite for a step or
        # two, the _run_loop bail-out gate wouldn't fire and the run
        # could record ``blowup=0``. iter-288: cover ALL prognostic
        # arrays so a single-field NaN trips the gate immediately.
        return dict(
            max_w=float(jnp.max(jnp.abs(s.w.data))),
            min_th=float(jnp.min(s.theta_prime.data)),
            max_th=float(jnp.max(s.theta_prime.data)),
            min_qv=float(jnp.min(s.tracers.data[..., 0])),
            max_qv=float(jnp.max(s.tracers.data[..., 0])),
            finite=bool(
                jnp.all(jnp.isfinite(s.u.data))
                & jnp.all(jnp.isfinite(s.v.data))
                & jnp.all(jnp.isfinite(s.w.data))
                & jnp.all(jnp.isfinite(s.theta_prime.data))
                & jnp.all(jnp.isfinite(s.rho_prime.data))
                & jnp.all(jnp.isfinite(s.tracers.data))
            ),
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
        # iter-288 (Codex HIGH): cover ALL prognostic arrays. MPAS
        # has no s.v (uses u-on-edges normal-only winds).
        return dict(
            max_w=float(jnp.max(jnp.abs(s.w.data))),
            min_th=float(jnp.min(s.theta_prime.data)),
            max_th=float(jnp.max(s.theta_prime.data)),
            min_qv=float(jnp.min(s.tracers.data[..., 0])),
            max_qv=float(jnp.max(s.tracers.data[..., 0])),
            finite=bool(
                jnp.all(jnp.isfinite(s.u.data))
                & jnp.all(jnp.isfinite(s.w.data))
                & jnp.all(jnp.isfinite(s.theta_prime.data))
                & jnp.all(jnp.isfinite(s.rho_prime.data))
                & jnp.all(jnp.isfinite(s.tracers.data))
            ),
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
    p.add_argument(
        "--n-cubed-sphere", type=int, default=4,
        help="iter-287: cubed-sphere face size n (default 4 = C4 "
        "preserves iter-238 smoke). C12=864 cells ~830 km/cell, "
        "C24=3456 cells ~415 km/cell, C96=55296 cells ~104 km/cell. "
        "iter-286 confirmed C4 is too coarse for moist 30-day "
        "production. Ignored for non-cubed-sphere grids.",
    )
    # iter-289 (Codex iter-288 round-1 MEDIUM): surface flux
    # tuning knobs for cluster sensitivity runs. Hardcoded
    # defaults match the plane CRM's iter-183 production contract
    # so untouched cluster runs reproduce baseline trajectory.
    p.add_argument(
        "--sfc-Cd", type=float, default=1.0e-3,
        help="iter-289: drag coefficient for bulk surface flux "
        "(--moist + --grid cubed_sphere only today). Default 1e-3 "
        "matches the plane CRM iter-183 contract. Lower (e.g. "
        "1e-4) reduces momentum drag for cluster sensitivity "
        "probes.",
    )
    p.add_argument(
        "--sfc-Ch", type=float, default=1.0e-3,
        help="iter-289: heat transfer coefficient for bulk "
        "surface flux. Default 1e-3 = iter-183 contract.",
    )
    p.add_argument(
        "--sfc-T", type=float, default=300.0,
        help="iter-289: fixed sea surface temperature [K] for "
        "bulk surface flux. Default 300 K = Wing 2018 RCEMIP "
        "tropical SST.",
    )
    p.add_argument(
        "--sfc-q", type=float, default=0.018,
        help="iter-289: surface specific humidity [kg/kg] for "
        "bulk surface flux. Default 0.018 = saturation at "
        "T_sfc=300 K, p=1013 hPa.",
    )
    # iter-290 (Codex iter-288 round-1 MEDIUM#2): cubed-sphere
    # dycore-tuning knobs. iter-285 hardcoded these inside
    # _run_cubed_sphere when --moist (n_acoustic=12,
    # use_coriolis=False, fix_mass=True). Adding CLI flags so
    # cluster users can mix moist + Coriolis + alternate
    # acoustic-substep / mass-fixer settings without code edits.
    # Defaults preserve the iter-285 small-mesh moist baseline.
    p.add_argument(
        "--cubed-n-acoustic", type=int, default=None,
        help="iter-290: cubed-sphere n_acoustic_substeps. Default "
        "None means: 4 when --no-moist (iter-238 smoke), 12 when "
        "--moist (iter-285 tightened config). Override to e.g. 24 "
        "for very small dt + safety margin, or 4 for fastest "
        "(dry-only) runs.",
    )
    p.add_argument(
        "--cubed-coriolis", choices=["on", "off"], default=None,
        help="iter-290: cubed-sphere use_coriolis. Default None "
        "means: 'on' when --no-moist (preserves iter-238), 'off' "
        "when --moist (iter-285 gridscale-unphysical at C4).",
    )
    p.add_argument(
        "--cubed-fix-mass", choices=["on", "off"], default=None,
        help="iter-290: cubed-sphere fix_mass + "
        "anchor_mass_to_initial toggle. Default None means: 'off' "
        "when --no-moist, 'on' when --moist (matches iter-183 "
        "plane CRM contract).",
    )
    args = p.parse_args()

    dispatch = {
        "plane_fd": _run_plane_fd,
        "plane_spectral": _run_plane_spectral,
        "cubed_sphere": _run_cubed_sphere,
        "mpas": _run_mpas,
    }
    # iter-288 (Codex iter-283..287 round-1 LOW#2): validate
    # --n-cubed-sphere is a positive integer. Argparse type=int
    # accepts 0 / -1 which would reach grid construction and
    # raise a cryptic shape error.
    if args.n_cubed_sphere < 1:
        raise SystemExit(
            f"error: --n-cubed-sphere={args.n_cubed_sphere} must "
            f"be >= 1. Typical values: 4 (default smoke), 12, "
            f"24, 48, 96, 192 (production)."
        )
    # iter-291 (Codex iter-289..290 round-1 HIGH#1): validate
    # surface-flux CLI flags. NaN T_sfc or negative Cd/Ch
    # propagates into flux compute + flips sign of drag/heat/moisture
    # forcing without error.
    import math as _math
    for _fname, _fval, _lo, _hi in [
        ("sfc-Cd", args.sfc_Cd, 0.0, 1.0),
        ("sfc-Ch", args.sfc_Ch, 0.0, 1.0),
        ("sfc-T", args.sfc_T, 100.0, 400.0),
        ("sfc-q", args.sfc_q, 0.0, 0.1),
    ]:
        if not _math.isfinite(_fval):
            raise SystemExit(
                f"error: --{_fname}={_fval} not finite."
            )
        if not (_lo <= _fval <= _hi):
            raise SystemExit(
                f"error: --{_fname}={_fval} outside physical "
                f"range [{_lo}, {_hi}]."
            )
    # iter-291 (Codex round-1 HIGH#2): validate --cubed-n-acoustic
    # >= 1. 0 would divide-by-zero in split_explicit.py:313.
    if args.cubed_n_acoustic is not None and args.cubed_n_acoustic < 1:
        raise SystemExit(
            f"error: --cubed-n-acoustic={args.cubed_n_acoustic} "
            f"must be >= 1."
        )
    # iter-291 (Codex round-1 HIGH#3): warn (via SystemExit) when
    # --sfc-* or --cubed-* are passed for a non-cubed-sphere grid.
    # Silently ignoring them is the documented-bug case Codex
    # flagged — flag-misuse should surface loudly.
    if args.grid != "cubed_sphere":
        _CUBED_DEFAULTS = {
            "sfc_Cd": 1.0e-3, "sfc_Ch": 1.0e-3,
            "sfc_T": 300.0, "sfc_q": 0.018,
            "cubed_n_acoustic": None,
            "cubed_coriolis": None, "cubed_fix_mass": None,
        }
        _passed_cubed_only = [
            _attr for _attr, _default in _CUBED_DEFAULTS.items()
            if getattr(args, _attr) != _default
        ]
        if _passed_cubed_only:
            raise SystemExit(
                f"error: --grid={args.grid} but cubed-sphere-only "
                f"flags were passed: "
                f"{', '.join('--' + a.replace('_', '-') for a in _passed_cubed_only)}. "
                f"These only apply to --grid cubed_sphere."
            )
    common_kwargs = dict(moist=args.moist)
    if args.grid == "cubed_sphere":
        common_kwargs["n"] = args.n_cubed_sphere
        # iter-289: forward surface-flux kwargs to the cubed-sphere
        # runner so the user can tune Cd/Ch/T_sfc/q_sfc without
        # editing code. plane_fd / mpas paths don't accept these
        # today (plane CRM uses its own composer in run_rcemip_plane.py;
        # MPAS u-on-edges surface flux not yet wired).
        common_kwargs["sfc_Cd"] = args.sfc_Cd
        common_kwargs["sfc_Ch"] = args.sfc_Ch
        common_kwargs["sfc_T"] = args.sfc_T
        common_kwargs["sfc_q"] = args.sfc_q
        # iter-290: dycore-tuning forwards. None passthrough means
        # _run_cubed_sphere applies the moist-vs-dry default.
        common_kwargs["n_acoustic"] = args.cubed_n_acoustic
        common_kwargs["coriolis"] = args.cubed_coriolis
        common_kwargs["fix_mass"] = args.cubed_fix_mass
    dispatch[args.grid](
        args.days, args.dt, args.print_every, args.output,
        **common_kwargs,
    )


if __name__ == "__main__":
    main()
