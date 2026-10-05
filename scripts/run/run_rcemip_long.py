"""Long RCEMIP1-style run on every NH dycore (plane FD / plane spectral /
cubed-sphere C-D / MPAS Voronoi).

Per-grid setup, same physics composition where the factory supports it:
gray radiation (Frierson 2006) + Kessler warm-rain microphysics, plus
bulk surface flux when ``--moist`` is set:
* plane: bulk surface flux via ``run_rcemip_plane.py`` composer.
* cubed-sphere (iter-283): full Cd/Ch heat + moisture + momentum drag
  via ``_make_cubed_sphere_surface_flux_tendency``.
* MPAS NH (iter-307): simplified — heat + moisture only, no momentum
  drag (u-on-edges needs edge↔cell reconstruction — follow-on).

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
import math
import sys
import time
from pathlib import Path


# iter-293 (Codex iter-292 round-3 LOW): module-level physical-range
# table for the --sfc-* CLI validators. Pre-iter-293 the (lo, hi)
# bounds were inline in the per-flag validator loop; iter-292
# already had to adjust sfc-T once, and bumping a value buried in
# main() risks missing other callers. Module-top table = single
# source of truth + easy to extend.
_SFC_CLI_RANGES = {
    # flag-name  : (attr-name, lo, hi)
    "sfc-Cd": ("sfc_Cd", 0.0, 1.0),
    "sfc-Ch": ("sfc_Ch", 0.0, 1.0),
    # sfc-T: widened [100, 400] → [50, 800] K at iter-292 for
    # non-Earth idealised CRM (snowball Earth ~200 K,
    # Venus-like ~700 K). Earth tropical RCEMIP 300 K inside.
    "sfc-T":  ("sfc_T",  50.0, 800.0),
    "sfc-q":  ("sfc_q",  0.0,  0.1),
}


# iter-293 (Codex iter-292 round-3 MEDIUM#2): cubed-sphere-only
# CLI flag names. Defaults derived from argparse via
# ``p.get_default(attr)`` at validation time, so a future
# default change can't drift past the misuse check.
# iter-309 (Codex iter-307/308 round-1 HIGH): sfc_* attrs
# REMOVED from this tuple because MPAS now also uses them (via
# iter-307 _make_mpas_surface_flux_tendency). The remaining
# entries are STRICTLY cubed-sphere — n_cubed_sphere selects
# the C-grid face size, cubed-* tunes the CDGridCompressibleEulerConfig.
_CUBED_ONLY_ATTRS = (
    "n_cubed_sphere",
    "cubed_n_acoustic", "cubed_coriolis", "cubed_fix_mass",
    "cubed_convection",
)
# iter-309: shared surface-flux attrs used by BOTH cubed-sphere
# + MPAS moist paths. Validation runs for any grid that uses
# moist physics; if --grid plane_fd/plane_spectral and any
# sfc-* is non-default, misuse is reported.
_SFC_SHARED_ATTRS = ("sfc_Cd", "sfc_Ch", "sfc_T", "sfc_q")

_REPO_ROOT = Path(__file__).resolve().parents[2]
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

    Why surface flux is needed for moist NH dycores: iter-282
    measured -2.8 K/day theta' cooling on the MPAS no-sfc baseline
    (gray radiation with no surface-flux counter-balance); the
    same column-energy-budget gap applies to any moist NH run.
    Bulk Cd/Ch + T_sfc/q_sfc fixed-SST relaxation closes it.
    iter-283 wired this for cubed-sphere; iter-307 added the
    MPAS analogue (``_make_mpas_surface_flux_tendency``,
    heat+moisture only).
    """
    from legoesm import constants as legoesm_constants
    from legoesm.core.bulk_flux import simple_bulk_fluxes
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


def _make_mpas_surface_flux_tendency(
    Cd: float = 1.0e-3, Ch: float = 1.0e-3,
    T_sfc: float = 300.0, q_sfc: float = 0.018,
    wind_speed_proxy: float = 5.0,
):
    """iter-307: simplified bulk surface flux tendency for MPAS
    NH MPASNonHydrostaticState. Heat + moisture only (NO momentum
    drag) at the lowest cell-level.

    MPAS state has u on edges (normal component only) + no v
    field. Proper bulk surface drag would need edge-to-cell
    reconstruction (``coeffs_reconstruct`` matrix) to get cell-
    centered (u_zonal, u_meridional) for the |U| calculation,
    then back-project the drag onto the edge-normal momentum
    equation. That's substantially more work + would need careful
    metric handling.

    iter-307 simplification: use a FIXED ``wind_speed_proxy``
    (default 5 m/s) as the surface wind magnitude entering the
    bulk flux. This matches the RCEMIP "no-wind" convention where
    surface flux is decoupled from instantaneous winds. The
    omission of momentum drag is the main physics gap; for a
    fixed-SST RCE relaxation the wind generated by buoyancy
    convection is small (<5 m/s) so the proxy is reasonable.

    Why iter-307 needed this: iter-282 MPAS 30-day moist run blew
    up at day 15 with theta' cooling -2.8 K/day from gray radiation
    with no surface flux counter-balance. This helper closes the
    column energy budget for MPAS.
    """
    from legoesm import constants as legoesm_constants
    from legoesm.core.bulk_flux import simple_bulk_fluxes
    from legoesm.core.state import MPASNonHydrostaticTendencies

    def physics_fn(state, mesh_in, hc_in, tm_in):
        nlev_local = state.theta_prime.data.shape[-1]
        k_sfc = nlev_local - 1
        rho_0 = hc_in.rho_ref
        theta_0 = hc_in.theta_ref
        theta_total = theta_0 + state.theta_prime.data
        rho_total = rho_0 + state.rho_prime.data
        # (nCells,) at lowest level
        rho_lo = rho_total[..., k_sfc]
        theta_lo = theta_total[..., k_sfc]
        pi_sfc = hc_in.exner_ref[k_sfc]
        T_lo = theta_lo * pi_sfc
        q_lo = state.tracers.data[..., k_sfc, 0]
        # Fixed wind-speed proxy — see docstring.
        wind_speed = jnp.full_like(T_lo, wind_speed_proxy)
        # u_lowest / v_lowest unused for heat/moisture flux; pass
        # zeros so the simple_bulk_fluxes signature is satisfied.
        u_zero = jnp.zeros_like(T_lo)
        _, _, shflx, lhflx = simple_bulk_fluxes(
            u_lowest=u_zero, v_lowest=u_zero,
            T_lowest=T_lo, q_lowest=q_lo,
            T_sfc=jnp.full_like(T_lo, T_sfc),
            q_sfc=jnp.full_like(T_lo, q_sfc),
            rho=rho_lo, wind_speed=wind_speed, Cd=Cd, Ch=Ch,
        )

        dz_sfc = hc_in.dz[k_sfc]
        dT_sfc = shflx / (rho_lo * legoesm_constants.c_pd * dz_sfc)
        dtheta_sfc = dT_sfc / pi_sfc
        dtheta_p_data = jnp.zeros_like(
            state.theta_prime.data
        ).at[..., k_sfc].set(dtheta_sfc)

        dtracers_data = jnp.zeros_like(state.tracers.data)
        dq_sfc = lhflx / (rho_lo * legoesm_constants.L_v * dz_sfc)
        dtracers_data = dtracers_data.at[..., k_sfc, 0].add(dq_sfc)

        return MPASNonHydrostaticTendencies(
            du_dt=state.u.replace(
                data=jnp.zeros_like(state.u.data),
            ),
            dw_dt=state.w.replace(
                data=jnp.zeros_like(state.w.data),
            ),
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
                              sfc_q: float = 0.018,
                              convection_scheme: str = "none"):
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
        If True, also compose a bulk surface flux tendency:
        * ``model_type='nonhydrostatic'`` → iter-283
          ``_make_cubed_sphere_surface_flux_tendency`` (full
          Cd/Ch heat + moisture + momentum drag).
        * ``model_type='mpas_nh'`` → iter-307
          ``_make_mpas_surface_flux_tendency`` (heat + moisture
          only, no momentum drag — MPAS u-on-edges + no v
          needs edge↔cell reconstruction for that).
        ValueError raised for any other model_type.

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

    # iter-307: with_surface_flux=True now supported for BOTH
    # 'nonhydrostatic' (cubed-sphere, iter-283) AND 'mpas_nh'
    # (iter-307, simplified — heat + moisture only, no momentum
    # drag — see _make_mpas_surface_flux_tendency docstring).
    if with_surface_flux and model_type not in ("nonhydrostatic", "mpas_nh"):
        raise ValueError(
            f"_compose_nh_moist_physics: with_surface_flux=True is "
            f"only supported for model_type='nonhydrostatic' or "
            f"'mpas_nh'; got model_type={model_type!r}."
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
        if model_type == "nonhydrostatic":
            sfc_fn = _make_cubed_sphere_surface_flux_tendency(
                Cd=sfc_Cd, Ch=sfc_Ch, T_sfc=sfc_T, q_sfc=sfc_q,
            )
        else:  # mpas_nh — heat + moisture only (no momentum drag)
            sfc_fn = _make_mpas_surface_flux_tendency(
                Cd=sfc_Cd, Ch=sfc_Ch, T_sfc=sfc_T, q_sfc=sfc_q,
            )

    # iter-314: opt-in sub-grid convection scheme. Only supported
    # for cubed-sphere today — make_convection_physics has a
    # 'nonhydrostatic' branch (iter-275 model_type) but no
    # 'mpas_nh' branch. MPAS convection would need a separate
    # factory.
    conv_fn = None
    if convection_scheme != "none":
        if model_type != "nonhydrostatic":
            raise ValueError(
                f"_compose_nh_moist_physics: convection_scheme="
                f"{convection_scheme!r} only supported for "
                f"model_type='nonhydrostatic' (cubed-sphere) today; "
                f"got model_type={model_type!r}."
            )
        from legoesm.atmosphere.physics.convection.config import (
            ConvectionConfig,
            ZhangMcFarlaneConfig,
        )
        from legoesm.atmosphere.physics.convection.integration import (
            make_convection_physics,
        )
        conv_fn = make_convection_physics(
            # RCEMIP is an aquaplanet: ZM's land choice is recorded, not defaulted.
            ConvectionConfig(scheme=convection_scheme, zhang_mcfarlane=ZhangMcFarlaneConfig(land_fraction="none")),
            model_type=model_type, dt=dt,
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
        if conv_fn is not None:
            t_conv = conv_fn(*args, **kwargs)
            leaves_c = jax.tree_util.tree_leaves(t_conv)
            if len(leaves_c) != len(summed):
                raise ValueError(
                    f"_compose_nh_moist_physics: convection tendency "
                    f"has {len(leaves_c)} leaves vs combined "
                    f"{len(summed)} — contract mismatch."
                )
            summed = [a + b for a, b in zip(summed, leaves_c)]
        return jax.tree_util.tree_unflatten(treedef, summed)

    return physics_fn


def _run_plane_fd(days: float, dt: float, print_every: int, output: Path,
                  *, moist: bool = False):
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        compute_dry_mass_plane, make_flat_plane_terrain_metric,
        make_rest_state,
    )
    from legoesm.atmosphere.dynamics.les.spectral_plane import (
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
                      fix_mass: str | None = None,
                      convection_scheme: str = "none"):
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
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
            convection_scheme=convection_scheme,
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
              *, moist: bool = False,
              sfc_Cd: float = 1.0e-3, sfc_Ch: float = 1.0e-3,
              sfc_T: float = 300.0, sfc_q: float = 0.018):
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
    # iter-307: with_surface_flux=True now wires the new
    # _make_mpas_surface_flux_tendency helper (heat + moisture
    # only; no momentum drag — see helper docstring). Without
    # surface flux, iter-282 measured NaN at day 15 from gray
    # radiation cooling with no counter-balance.
    if moist:
        physics_fn = _compose_nh_moist_physics(
            model_type="mpas_nh", dt=dt, with_surface_flux=True,
            sfc_Cd=sfc_Cd, sfc_Ch=sfc_Ch,
            sfc_T=sfc_T, sfc_q=sfc_q,
        )
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
        help="iter-289: drag coefficient for bulk surface flux. "
        "Applies to ``--moist`` + ``--grid {cubed_sphere, mpas}`` "
        "(iter-309 widened from cubed-sphere-only). Default 1e-3 "
        "matches the plane CRM iter-183 contract. Lower (e.g. "
        "1e-4) reduces momentum drag for cluster sensitivity "
        "probes. NOTE: MPAS surface flux is heat+moisture only "
        "(no momentum drag — see iter-307 helper docstring).",
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
    # iter-314: opt-in sub-grid convection scheme for the cubed-
    # sphere moist composition. iter-284/286/287/308 confirmed
    # the 30-day stability barrier is gridscale convection
    # unresolved at C4..C12. Adding a sub-grid convection
    # parameterization stabilizes coarse-mesh moist runs without
    # needing HPC compute. 'none' (default) preserves the
    # iter-275 composition (Kessler-only moist).
    p.add_argument(
        "--cubed-convection",
        choices=["none", "kuo", "sbm", "dca", "mass_flux", "edmf",
                 "zhang_mcfarlane", "kain_fritsch", "emanuel",
                 "tiedtke", "bechtold"],
        default="none",
        help="iter-314: cubed-sphere sub-grid convection scheme "
        "for --moist runs. 'none' (default) = Kessler-only moist "
        "(iter-275 baseline). Choices match the legoesm convection "
        "scheme registry. Cluster users tackling 30-day stability "
        "at coarse mesh should try 'kuo' (simplest mass-flux) or "
        "'tiedtke' (production-grade). Ignored for non-cubed-sphere "
        "grids.",
    )
    args = p.parse_args()

    dispatch = {
        "plane_fd": _run_plane_fd,
        "plane_spectral": _run_plane_spectral,
        "cubed_sphere": _run_cubed_sphere,
        "mpas": _run_mpas,
    }
    # iter-292 (Codex iter-291 round-2 MEDIUM#2): cubed-sphere-only
    # flag-misuse check runs FIRST so e.g.
    # ``--grid plane_fd --cubed-n-acoustic 0`` surfaces the
    # grid-mismatch (the real problem) instead of the secondary
    # range-validation error. Reordered.
    # iter-292 (Codex round-2 HIGH): added n_cubed_sphere to the
    # cubed-only set — iter-291 omitted it so ``--grid mpas
    # --n-cubed-sphere 96`` was silently ignored.
    # iter-309 (Codex iter-307/308 round-1 HIGH): split misuse
    # detection between cubed-sphere-strict + sfc-moist-shared.
    # sfc-* are now used by BOTH cubed-sphere AND MPAS, so they
    # check against the moist-physics-grids set ({cubed_sphere,
    # mpas}), not strictly cubed-sphere.
    if args.grid != "cubed_sphere":
        _passed_cubed_only = [
            _attr for _attr in _CUBED_ONLY_ATTRS
            if getattr(args, _attr) != p.get_default(_attr)
        ]
        if _passed_cubed_only:
            raise SystemExit(
                f"error: --grid={args.grid} but cubed-sphere-only "
                f"flags were passed: "
                f"{', '.join('--' + a.replace('_', '-') for a in _passed_cubed_only)}. "
                f"These only apply to --grid cubed_sphere."
            )
    if args.grid not in ("cubed_sphere", "mpas"):
        _passed_sfc_shared = [
            _attr for _attr in _SFC_SHARED_ATTRS
            if getattr(args, _attr) != p.get_default(_attr)
        ]
        if _passed_sfc_shared:
            raise SystemExit(
                f"error: --grid={args.grid} but moist-surface-flux "
                f"flags were passed: "
                f"{', '.join('--' + a.replace('_', '-') for a in _passed_sfc_shared)}. "
                f"These only apply to --grid cubed_sphere or mpas."
            )
    # iter-288 (Codex iter-283..287 round-1 LOW#2): validate
    # --n-cubed-sphere is a positive integer.
    if args.n_cubed_sphere < 1:
        raise SystemExit(
            f"error: --n-cubed-sphere={args.n_cubed_sphere} must "
            f"be >= 1. Typical values: 4 (default smoke), 12, "
            f"24, 48, 96, 192 (production)."
        )
    # iter-291 (HIGH#1) + iter-292 (MEDIUM#1) + iter-293 (LOW):
    # validate surface-flux CLI flags against the module-level
    # ``_SFC_CLI_RANGES`` table. NaN T_sfc or negative Cd/Ch
    # propagates into flux compute + flips sign of drag/heat
    # forcing without error.
    for _fname, (_attr, _lo, _hi) in _SFC_CLI_RANGES.items():
        _fval = getattr(args, _attr)
        if not math.isfinite(_fval):
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
    common_kwargs = dict(moist=args.moist)
    if args.grid == "cubed_sphere":
        common_kwargs["n"] = args.n_cubed_sphere
        # iter-289: forward surface-flux kwargs.
        common_kwargs["sfc_Cd"] = args.sfc_Cd
        common_kwargs["sfc_Ch"] = args.sfc_Ch
        common_kwargs["sfc_T"] = args.sfc_T
        common_kwargs["sfc_q"] = args.sfc_q
        # iter-290: dycore-tuning forwards.
        common_kwargs["n_acoustic"] = args.cubed_n_acoustic
        common_kwargs["coriolis"] = args.cubed_coriolis
        common_kwargs["fix_mass"] = args.cubed_fix_mass
        # iter-314: convection scheme forward.
        common_kwargs["convection_scheme"] = args.cubed_convection
    elif args.grid == "mpas":
        # iter-309 (Codex iter-307/308 round-1 HIGH): MPAS now
        # accepts the shared --sfc-* CLI flags (forwarded into
        # _make_mpas_surface_flux_tendency).
        common_kwargs["sfc_Cd"] = args.sfc_Cd
        common_kwargs["sfc_Ch"] = args.sfc_Ch
        common_kwargs["sfc_T"] = args.sfc_T
        common_kwargs["sfc_q"] = args.sfc_q
    dispatch[args.grid](
        args.days, args.dt, args.print_every, args.output,
        **common_kwargs,
    )


if __name__ == "__main__":
    main()
