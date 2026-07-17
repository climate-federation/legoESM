#!/usr/bin/env python
"""Moist Radiative-Convective Equilibrium (RCE) with slab ocean or slab land.

Aquaplanet or land-planet experiment:
- Gray radiation (Frierson 2006, moisture-dependent LW optical depth)
- SBM convection + large-scale condensation
- Bulk aerodynamic boundary-layer coupling
- Slab ocean (50 m mixed layer) or slab land (1 m soil + bucket hydrology)
- Rayleigh friction (BL drag + weak free-atmosphere drag)

Operator-split coupling: dynamics step, then physics step each timestep.

Usage:
    JAX_ENABLE_X64=1 python scripts/run_rce.py --days 200
    JAX_ENABLE_X64=1 python scripts/run_rce.py --mode land --days 100
    JAX_ENABLE_X64=1 python scripts/run_rce.py --resolution 24 --dt 300 --days 500
    JAX_ENABLE_X64=1 python scripts/run_rce.py --grid-type gaussian --truncation 21
    JAX_ENABLE_X64=1 python scripts/run_rce.py --grid-type latlon --resolution 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import jax


def _spectral_selected(argv):
    """True when the CLI selects a spectral/Gaussian grid (``--grid-type
    gaussian``, ``--discretization spectral``, or any ``--truncation``).

    The spherical-harmonic transforms REQUIRE float64 — ``create_gaussian_grid``
    raises otherwise — so the driver must enable x64 for these before JAX
    initializes. Scans argv by exact option/value (not substring) so an output
    path containing "spectral" cannot trip it."""
    for i, a in enumerate(argv):
        if a == "--truncation" or a.startswith("--truncation="):
            return True
        if a in ("--grid-type=gaussian", "--discretization=spectral"):
            return True
        if a in ("--grid-type", "--discretization") and i + 1 < len(argv) \
                and argv[i + 1] in ("gaussian", "spectral"):
            return True
    return False


# Auto-enable float64 for spectral grids so a naive ``--grid-type gaussian``
# does not hard-crash (the transforms reject x32); other grids keep the caller's
# precision (float32 default). Must run before any array op — jnp is imported
# below, after this. A one-line notice keeps the coercion non-silent.
# Gated on ``__name__ == "__main__"`` (true from the top when run as a script,
# ``python run_rce.py …``) so a bare ``import run_rce`` NEVER flips the global
# x64 flag off the importer's argv (parser is ``allow_abbrev=False``, so only the
# full option spellings the sniff checks are valid). The argparse layer still
# validates the actual flags; this is only the precisely-needed early bootstrap.
if __name__ == "__main__" and _spectral_selected(sys.argv[1:]) \
        and not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)
    print("[run_rce] spectral grid selected -> auto-enabled JAX_ENABLE_X64 "
          "(spherical-harmonic transforms require float64)", flush=True)

import jax.numpy as jnp
import numpy as np


def main():
    # ``allow_abbrev=False`` ensures the legacy ``latlon_fv`` alias
    # logic below (which inspects ``sys.argv`` to decide whether
    # ``--grid-type`` was explicit) cannot be fooled by argparse-style
    # long-option abbreviations like ``--grid`` for ``--grid-type``.
    parser = argparse.ArgumentParser(
        description="Moist RCE experiment", allow_abbrev=False,
    )
    parser.add_argument("--mode", choices=["ocean", "land"], default="ocean",
                        help="Surface type: slab ocean or slab land")
    parser.add_argument("--ocean-mode", choices=["prescribed", "slab", "two_layer"],
                        default="slab",
                        help="Ocean coupling: prescribed (fixed SST), slab (50m), "
                             "two_layer (50m mixed + 200m deep)")
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=16,
                        help="Grid resolution (N for cubed-sphere, n_max for spectral, etc.)")
    parser.add_argument("--nlev", type=int, default=20)
    parser.add_argument("--dt", type=float, default=None,
                        help="Timestep [s]. Auto: 600 for N<=24, 300 "
                             "for N>24 on cubed_sphere/latlon/gaussian; "
                             "300 unconditionally on voronoi/MPAS (V4 "
                             "blows up at dt>=450 in the cross-grid "
                             "smoke).")
    parser.add_argument("--diag-days", type=int, default=5)
    parser.add_argument("--sst-init", type=float, default=300.0,
                        help="Initial SST [K] (ocean) or soil T [K] (land)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--grid-type", type=str, default="cubed_sphere",
                        choices=["cubed_sphere", "gaussian", "latlon", "voronoi"],
                        help="Horizontal grid type")
    parser.add_argument("--discretization", type=str, default="cdgrid",
                        choices=["cdgrid", "centered", "spectral",
                                 "latlon_fv", "finite_volume",
                                 "latlon_cgrid", "mpas"],
                        help="Discretization method.  ``latlon_fv`` is a "
                             "legacy shorthand for ``--grid-type latlon "
                             "--discretization finite_volume`` (lat-lon "
                             "C-grid solver); using it implies the "
                             "lat-lon grid.  ``finite_volume`` on its own "
                             "stays on whatever ``--grid-type`` is "
                             "selected (e.g. cubed_sphere → CDGrid FV). "
                             "``centered`` and ``latlon_cgrid`` are "
                             "supported aliases on cubed_sphere/latlon "
                             "and latlon respectively.")
    parser.add_argument("--truncation", type=int, default=None,
                        help="Spectral truncation (T21, T42, etc.). Sets grid_type=gaussian.")
    # Detect whether ``--grid-type`` was passed explicitly so the
    # legacy-alias handler below can distinguish "user accepted the
    # default" from "user explicitly asked for X".
    _grid_type_explicit = any(
        a == "--grid-type" or a.startswith("--grid-type=")
        for a in sys.argv[1:]
    )

    args = parser.parse_args()

    # iter-71 (mirrors iter-67/70 on run_rce_mpi_long.py): validate
    # numeric CLI args reject NaN/inf + out-of-range values with
    # concise SystemExit. Without these, ``--days -1`` silently
    # produces a 0-day run with empty diag_log; ``--resolution 0``
    # crashes deep in grid creation; ``--sst-init nan`` propagates
    # to all column ICs.
    import math as _math
    if args.dt is not None and not _math.isfinite(args.dt):
        raise SystemExit(
            f"error: --dt rejected: must be finite, got {args.dt!r}"
        )
    if args.dt is not None and args.dt <= 0.0:
        raise SystemExit(
            f"error: --dt rejected: must be positive, got {args.dt!r}"
        )
    if not _math.isfinite(args.sst_init):
        raise SystemExit(
            f"error: --sst-init rejected: must be finite, got "
            f"{args.sst_init!r}"
        )
    # iter-74 Codex HIGH: --sst-init is a Kelvin temperature; negative
    # values are physically invalid. iter-71 only checked finiteness.
    if args.sst_init <= 0.0:
        raise SystemExit(
            f"error: --sst-init rejected: must be positive Kelvin "
            f"temperature, got {args.sst_init!r}"
        )
    _POSITIVE_INTS = {
        "--days": args.days,
        "--resolution": args.resolution,
        "--nlev": args.nlev,
        "--diag-days": args.diag_days,
    }
    for _flag, _val in _POSITIVE_INTS.items():
        if _val <= 0:
            raise SystemExit(
                f"error: {_flag} rejected: must be positive integer, "
                f"got {_val!r}"
            )
    if args.truncation is not None and args.truncation <= 0:
        raise SystemExit(
            f"error: --truncation rejected: must be positive integer "
            f"when set, got {args.truncation!r}"
        )

    # Translate the legacy ``latlon_fv`` alias to the canonical
    # ``finite_volume`` name registered in
    # ``component_factory._DRIVER_SUPPORTED`` for the lat-lon C-grid
    # solver.  The alias is meaningful only on the lat-lon grid; on
    # any other grid_type it would silently re-route to a different
    # solver (e.g. cubed_sphere/finite_volume → CDGrid), which is not
    # what a user typing ``--discretization latlon_fv`` is asking for.
    # When the user did NOT explicitly choose ``--grid-type``, infer
    # ``latlon``; when they did, fail fast on a mismatch instead of
    # silently overriding the explicit grid choice.
    if args.discretization == "latlon_fv":
        if args.grid_type != "latlon":
            if _grid_type_explicit:
                parser.error(
                    f"--discretization latlon_fv requires --grid-type "
                    f"latlon (got {args.grid_type!r}). Pass "
                    f"--discretization finite_volume / latlon_cgrid for "
                    f"another grid, or remove --grid-type."
                )
            args.grid_type = "latlon"
        args.discretization = "finite_volume"

    # Auto-configure for spectral discretization
    if args.discretization == "spectral" or args.truncation is not None:
        args.discretization = "spectral"
        args.grid_type = "gaussian"
        if args.truncation is not None:
            args.resolution = args.truncation

    # The grid-specific adapters (init function, to_grid_arrays,
    # apply_T_update, apply_friction) below are tied to the unique
    # model class supported on each grid (cubed_sphere → CDGrid,
    # gaussian → Spectral, latlon → CGridLatLon, voronoi → MPAS).
    # Reject mismatched (grid_type, discretization) combinations up
    # front so the spectral adapter is never paired with a non-spectral
    # state, etc.  Driver factory would catch this too, but the message
    # here is more direct for a user who wired the script wrong.
    _expected_disc = {
        "cubed_sphere": {"cdgrid", "centered", "finite_volume"},
        "gaussian": {"spectral"},
        "latlon": {"finite_volume", "centered", "latlon_cgrid"},
        "voronoi": {"mpas"},
    }
    _allowed = _expected_disc[args.grid_type]
    if args.discretization not in _allowed:
        parser.error(
            f"Unsupported (--grid-type {args.grid_type!r}, "
            f"--discretization {args.discretization!r}) combination. "
            f"Allowed for {args.grid_type!r}: {sorted(_allowed)}"
        )

    N = args.resolution
    NLEV = args.nlev
    # Auto-dt ladder lives in legoesm.driver.rce_dt (iter-24: was
    # inline here through iter-21 but Codex iter-19 LOW asked for an
    # importable function so the test mirror in
    # test_rce_cross_grid_dt_defaults.py can call the SAME logic
    # instead of hand-copying it). See CRM_implementation.md
    # iter-12..23 for the per-tier empirical measurements; the
    # function raises for N>96 to refuse silent extrapolation.
    from legoesm.driver.rce_dt import auto_dt_rce
    if args.dt is not None:
        DT = args.dt
    else:
        DT = auto_dt_rce(args.grid_type, N)
    OUTPUT_DIR = Path(args.output or f"results/rce_{args.mode}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Grid, vertical coordinate, dynamical core
    # ---------------------------------------------------------------
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.driver.component_factory import create_atmosphere_dycore, compute_diffusion
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig

    grid_type = args.grid_type
    discretization = args.discretization

    if grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        grid = create_cubed_sphere(N)
    elif grid_type == "gaussian":
        from legoesm.grids.gaussian import create_gaussian_grid
        grid = create_gaussian_grid(N)
    elif grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(N)
    elif grid_type == "voronoi":
        from legoesm.grids.voronoi import create_voronoi_mesh
        grid = create_voronoi_mesh(N, lloyd_iterations=50)
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")

    sigma = create_sigma_coordinate(NLEV)
    shape_2d = grid.grid_shape_2d

    # Pre-clamp DT to the pole-cell advective CFL limit on lat-lon
    # grids before constructing ``DycoreConfig`` so that
    # ``config.dycore.dt``, ``HYPERDIFF`` (via ``compute_diffusion``),
    # the model's internal ``A_h``, and the script's stepping ``DT``
    # all agree on a single value.  The factory would clamp internally
    # too, but doing it here keeps every dt-derived quantity consistent.
    if grid_type == "latlon":
        from legoesm.core.cfl import pole_cell_dx, cfl_max_dt
        _dt_max_pole = float(cfl_max_dt(
            pole_cell_dx(grid), 300.0, cfl_number=0.8, ndim=1,
        ))
        if DT > _dt_max_pole:
            DT = _dt_max_pole

    # CFL-formula advisory (Codex iter-21 MEDIUM #2 — does not
    # override the ladder, just surfaces the formula bound so
    # operators see the headroom they're running on).
    #
    # Codex iter-24 HIGH: previous broad `except Exception` would
    # swallow TypeErrors from signature drift in cfl_max_dt and
    # AttributeErrors from renamed estimators, masking real bugs.
    # Now we only swallow ImportError (e.g. mid-refactor missing
    # module) and re-raise everything else.
    try:
        from legoesm.core.cfl import (
            cfl_max_dt, estimate_min_dx_cubed_sphere,
            estimate_min_dx_gaussian, estimate_min_dx_icosahedral,
            estimate_min_dx_latlon,
        )
    except ImportError as _exc:
        print(f"  CFL advisory unavailable ({_exc!r}); skipping.")
    else:
        if grid_type == "cubed_sphere":
            _dx_min = estimate_min_dx_cubed_sphere(N)
        elif grid_type == "latlon":
            _dx_min = estimate_min_dx_latlon(N)
        elif grid_type == "gaussian":
            _dx_min = estimate_min_dx_gaussian(N)
        elif grid_type == "voronoi":
            _dx_min = estimate_min_dx_icosahedral(N)
        else:
            _dx_min = None
        if _dx_min is not None:
            # Two reference bounds (both informational, do NOT
            # override the ladder):
            #   * gravity-wave CFL (iter-23): dt ∝ dx with safety 0.8.
            #     Loose upper bound; the destabilising mode isn't
            #     gravity-wave CFL.
            #   * empirical dx² fit (iter-29): α≈2 fit of the
            #     iter-12..26 ladder. Tight reference — the ladder
            #     should agree with this within 30 %.
            _dt_cfl_gravity = float(cfl_max_dt(_dx_min, 300.0, 0.8, ndim=2))
            try:
                from legoesm.driver.rce_dt import empirical_dt_dx2
                _dt_fit = float(empirical_dt_dx2(_dx_min))
                _fit_ratio = DT / _dt_fit
                _fit_part = (
                    f", dx² fit dt={_dt_fit:.0f} s "
                    f"({_fit_ratio:.2f}× fit)"
                )
            except ImportError:
                _fit_part = ""
            print(
                f"  CFL advisory: dx_min={_dx_min:.0f} m, "
                f"gravity-wave dt_max={_dt_cfl_gravity:.0f} s "
                f"({DT/_dt_cfl_gravity:.2f}× formula)"
                f"{_fit_part}, using DT={DT:.0f} s."
            )

    config = ExperimentConfig(
        grid=GridConfig(grid_type=grid_type, resolution=N, nlev=NLEV),
        dycore=DycoreConfig(discretization=discretization, dt=DT),
    )
    model = create_atmosphere_dycore(config, grid, sigma)
    HYPERDIFF = compute_diffusion(grid, config.dycore).hyperdiff

    # Sanity check: assert script DT matches the model's effective dt
    # after factory clamping.  Triggers only if the factory's internal
    # CFL formula diverges from the pre-clamp above (e.g. a future
    # tightening); fail loudly rather than running with mismatched
    # config/loop timesteps.
    if hasattr(model, "effective_dt") and model.effective_dt is not None:
        _eff = float(model.effective_dt)
        if abs(_eff - DT) > 1e-6 * max(_eff, DT):
            raise RuntimeError(
                f"Script DT={DT:.6f} disagrees with model.effective_dt="
                f"{_eff:.6f}; pre-clamp logic in run_rce.py is stale."
            )

    # ---------------------------------------------------------------
    # Initial atmospheric state (isothermal 280 K, at rest)
    # ---------------------------------------------------------------
    if grid_type == "cubed_sphere":
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
        state = held_suarez_init(grid, sigma, T_init=280.0)
    elif grid_type == "voronoi":
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
        state = held_suarez_init_mpas(grid, sigma, T_init=280.0)
    elif grid_type == "gaussian":
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            isothermal_rest_state_spectral,
        )
        state = isothermal_rest_state_spectral(
            grid, sigma, T_init=280.0, perturbation_amplitude=0.5,
        )
    else:
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon
        state = held_suarez_init_latlon(grid, sigma, T_init=280.0)

    # Grid-specific adapters that bridge the differing state layouts
    # (HydrostaticState / MPAS / SpectralHydrostaticState) to the
    # grid-space arrays the physics step needs.  ``to_grid_arrays``
    # returns ``(T, p_s, u, v)`` at cell centres; ``apply_T_update``
    # writes a new grid-space temperature back; ``apply_friction``
    # applies the per-level Rayleigh decay; ``is_finite`` powers the
    # blowup detector.
    if grid_type == "gaussian":
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            spectral_pe_to_grid,
        )
        from legoesm.grids.gaussian import sh_analysis_3d

        def to_grid_arrays(s):
            g = spectral_pe_to_grid(s, grid, sigma)
            return g["T"], g["p_s"], g["u"], g["v"]

        def apply_T_update(s, new_T_grid):
            new_T_hat = sh_analysis_3d(grid, new_T_grid)
            return s._replace(T_hat=s.T_hat.replace(data=new_T_hat))

        def apply_friction(s, decay):
            # k_f varies only with sigma → applying the per-level decay
            # to (vor_hat, div_hat) is equivalent to scaling u, v by
            # the same factor in grid-space (linear SH transform).
            nv = s.vor_hat.data * decay[None, :]
            nd = s.div_hat.data * decay[None, :]
            return s._replace(
                vor_hat=s.vor_hat.replace(data=nv),
                div_hat=s.div_hat.replace(data=nd),
            )

        def is_finite_state(s):
            # Cover every prognostic spectral field: a NaN in any of
            # vor_hat, div_hat, T_hat, or lnps_hat will silently
            # propagate to grid-space diagnostics on the next step.
            return (
                jnp.all(jnp.isfinite(s.T_hat.data))
                & jnp.all(jnp.isfinite(s.vor_hat.data))
                & jnp.all(jnp.isfinite(s.div_hat.data))
                & jnp.all(jnp.isfinite(s.lnps_hat.data))
            )
    elif grid_type == "voronoi":
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity

        def to_grid_arrays(s):
            u_cell, v_cell = reconstruct_cell_velocity(s.u.data, grid)
            return s.T.data, s.p_s.data, u_cell, v_cell

        def apply_T_update(s, new_T_grid):
            return s._replace(T=s.T.replace(data=new_T_grid))

        def apply_friction(s, decay):
            # u is edge-normal; per-level decay applies uniformly.
            return s._replace(u=s.u.replace(data=s.u.data * decay))

        def is_finite_state(s):
            return (
                jnp.all(jnp.isfinite(s.T.data))
                & jnp.all(jnp.isfinite(s.u.data))
                & jnp.all(jnp.isfinite(s.p_s.data))
            )
    else:
        # cubed_sphere & latlon both expose HydrostaticState with u, v.
        def to_grid_arrays(s):
            return s.T.data, s.p_s.data, s.u.data, s.v.data

        def apply_T_update(s, new_T_grid):
            return s._replace(T=s.T.replace(data=new_T_grid))

        def apply_friction(s, decay):
            return s._replace(
                u=s.u.replace(data=s.u.data * decay),
                v=s.v.replace(data=s.v.data * decay),
            )

        def is_finite_state(s):
            return (
                jnp.all(jnp.isfinite(s.T.data))
                & jnp.all(jnp.isfinite(s.u.data))
                & jnp.all(jnp.isfinite(s.v.data))
                & jnp.all(jnp.isfinite(s.p_s.data))
            )

    # Moisture: 60% RH with sigma^2 vertical decay
    from legoesm import constants
    from legoesm.thermo import saturation_mixing_ratio

    T_grid_init, p_s_grid_init, _, _ = to_grid_arrays(state)
    p_full_init = p_s_grid_init[..., None] * sigma.sigma_full
    q_sat_init = saturation_mixing_ratio(T_grid_init, p_full_init)
    q_v = 0.6 * q_sat_init * sigma.sigma_full ** 2
    q_v = jnp.minimum(q_v, q_sat_init)

    # ---------------------------------------------------------------
    # Surface configuration
    # ---------------------------------------------------------------
    IS_LAND = args.mode == "land"
    OCEAN_MODE = args.ocean_mode if not IS_LAND else "slab"  # land uses slab soil
    IS_TWO_LAYER = OCEAN_MODE == "two_layer"
    IS_PRESCRIBED = OCEAN_MODE == "prescribed"
    if not IS_LAND:
        from legoesm.ocean.simple_ocean import SimpleOceanConfig
        sfc_config = SimpleOceanConfig(mode=OCEAN_MODE, h_mix=50.0, albedo_ocean=0.06,
                                       h_deep=200.0, k_mix=1.0e-4)
        C_sfc = sfc_config.rho_ocean * sfc_config.c_ocean * sfc_config.h_mix
        T_sfc = jnp.full(shape_2d, args.sst_init)
        T_deep = jnp.full(shape_2d, 278.0) if IS_TWO_LAYER else None
        sfc_albedo = 0.06
        T_freeze = sfc_config.T_freeze
        W_bucket = jnp.zeros(shape_2d)  # dummy, unused
        W_max = 1.0                     # dummy
        beta_min = 1.0                  # dummy
    else:
        # Slab land: thin soil (1 m), fast thermal response
        C_soil = 2.0e6     # volumetric heat capacity [J/m3/K]
        h_soil = 1.0       # soil depth [m]
        C_sfc = C_soil * h_soil
        T_sfc = jnp.full(shape_2d, args.sst_init)
        T_deep = None
        sfc_albedo = 0.25
        T_freeze = constants.T_freeze  # 273.15 K (freshwater)
        # Bucket hydrology
        W_max = 150.0      # kg/m2
        beta_min = 0.1     # minimum evaporation efficiency
        W_bucket = jnp.full(shape_2d, 0.75 * W_max)  # 75% saturated

    C_H = 4.4e-3  # bulk transfer coefficient (heat and moisture; Frierson 2006)

    # ---------------------------------------------------------------
    # Physics configuration
    # ---------------------------------------------------------------
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    from legoesm.atmosphere.physics.convection.config import SBMConfig
    from legoesm.atmosphere.physics.radiation.gray import gray_radiation
    from legoesm.atmosphere.physics.radiation.solar import perpetual_equinox_insolation
    from legoesm.atmosphere.physics.convection.sbm import sbm_convection
    from legoesm.diagnostics.column_integrals import column_water_vapor

    # Grid-specific hyperdiffusion (spectral/voronoi handle diffusion in dycore)
    _apply_hyperdiff = None
    if grid_type == "cubed_sphere":
        from legoesm.core.operators_3d import hyperdiffusion_3d
        _apply_hyperdiff = lambda q, coeff: hyperdiffusion_3d(q, grid, coeff)
    elif grid_type == "latlon":
        from legoesm.core.operators_latlon_3d import hyperdiffusion_3d as hyperdiffusion_3d_ll
        _apply_hyperdiff = lambda q, coeff: hyperdiffusion_3d_ll(q, grid, coeff)

    gray_config = GrayRadiationConfig(
        tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
        sfc_albedo=sfc_albedo, perpetual_equinox=True,
    )
    sbm_config = SBMConfig(tau_c=7200.0, rh_ref=0.7)

    # Rayleigh friction profile (Frierson 2006: BL only, no free-atmosphere drag)
    sigma_b = 0.7
    k_f = ((1.0 / 86400.0) * jnp.maximum(0.0,
               (sigma.sigma_full - sigma_b) / (1.0 - sigma_b)))
    fric_decay = jnp.exp(-k_f * DT)

    S_0 = gray_config.S_0
    dsigma = sigma.dsigma

    # ---------------------------------------------------------------
    # JIT-compiled physics step
    # ---------------------------------------------------------------
    @jax.jit
    def physics_step(T, p_s, q_v, u, v, T_sfc, T_deep, W_bkt, lat, dt):
        """Operator-split physics: radiation + convection + BL + surface."""
        nlev = T.shape[-1]
        ncol = T[..., 0].size
        p_full = p_s[..., None] * sigma.sigma_full
        p_half = p_s[..., None] * sigma.sigma_half

        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        q_v_col = q_v.reshape(ncol, nlev)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)

        insol = perpetual_equinox_insolation(lat_col, S_0)

        # (a) Gray radiation
        rad = gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=gray_config,
        )
        dT_rad = rad.heating_rate.reshape(T.shape)

        # (b) SBM convection
        conv = sbm_convection(
            T=T_col, q_v=q_v_col,
            p_full=p_full_col, p_half=p_half_col,
            dt=dt, config=sbm_config,
        )
        dT_conv = conv.dT_dt.reshape(T.shape)
        dq_conv = conv.dq_v_dt.reshape(T.shape)
        # Surface precipitation diagnostic — preserves the exact
        # water-budget definition that ``ConvectionOutput.precipitation``
        # used before the Option-C refactor: the net column moisture
        # sink, clipped at zero. Computing it from ``dq_v_dt`` (sum
        # then clip) is *not* equivalent to integrating
        # ``dq_c_conv_dt`` (clip-per-level then sum): the latter
        # over-counts in columns that mix drying lower levels with
        # moistening upper levels because per-level clipping drops the
        # cancelling moistening contribution.
        # Under the new architecture microphysics would own this
        # diagnostic, but the RCE script intentionally runs without a
        # microphysics chain — so we recover the legacy definition
        # locally to keep the column water budget reproducible.
        dp_col = p_half_col[:, 1:] - p_half_col[:, :-1]
        precip_col = jnp.clip(
            -jnp.sum(conv.dq_v_dt * dp_col / constants.g, axis=1),
            0.0,
            None,
        )
        precip = precip_col.reshape(p_s.shape)

        # (c) Bulk aerodynamic BL coupling
        rho_low = (p_s * sigma.sigma_full[-1]) / (constants.R_d * T[..., -1])
        wind = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + 1.0)
        dp_low = p_s * (sigma.sigma_half[-1] - sigma.sigma_half[-2])

        # Sensible heat flux (positive upward)
        shflx = rho_low * constants.c_pd * C_H * wind * (T_sfc - T[..., -1])

        # Latent heat flux — ocean: beta=1; land: bucket moisture
        q_sat_sfc = saturation_mixing_ratio(T_sfc, p_s)
        if IS_LAND:
            beta = jnp.maximum(beta_min, W_bkt / W_max)
        else:
            beta = 1.0
        lhflx = rho_low * constants.L_v * C_H * wind * beta * (q_sat_sfc - q_v[..., -1])
        lhflx = jnp.maximum(lhflx, 0.0)  # no dew in simple scheme
        evap = lhflx / constants.L_v

        # Atmospheric tendencies (lowest level)
        dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
        dq_BL = constants.g * evap / dp_low

        # (d) Surface energy balance
        sw_net = (rad.sw_flux_down[:, -1] - rad.sw_flux_up[:, -1]).reshape(p_s.shape)
        lw_net = (rad.lw_flux_down[:, -1] - rad.lw_flux_up[:, -1]).reshape(p_s.shape)

        if IS_PRESCRIBED:
            T_sfc_new = T_sfc  # fixed SST
        elif IS_TWO_LAYER:
            # Mixed layer + deep layer with vertical diffusion
            rho_o, c_o = sfc_config.rho_ocean, sfc_config.c_ocean
            d_mid = 0.5 * (sfc_config.h_mix + sfc_config.h_deep)
            F_mix = rho_o * c_o * sfc_config.k_mix * (T_sfc - T_deep) / d_mid
            C_mix = rho_o * c_o * sfc_config.h_mix
            C_deep_v = rho_o * c_o * sfc_config.h_deep
            dT_sfc_dt = (sw_net + lw_net - shflx - lhflx - F_mix) / C_mix
            dT_deep_dt = F_mix / C_deep_v
            T_sfc_new = jnp.maximum(T_sfc + dt * dT_sfc_dt, T_freeze)
            T_deep = T_deep + dt * dT_deep_dt
        else:
            dT_sfc_dt = (sw_net + lw_net - shflx - lhflx) / C_sfc
            T_sfc_new = jnp.maximum(T_sfc + dt * dT_sfc_dt, T_freeze)

        # (e) Bucket hydrology update (land only — traced at JIT time)
        if IS_LAND:
            W_new = jnp.clip(W_bkt + dt * (precip - evap), 0.0, W_max)
        else:
            W_new = W_bkt

        # (f) Total atmospheric tendencies
        dT_dt = dT_rad + dT_conv
        dT_dt = dT_dt.at[..., -1].add(dT_BL)
        dq_dt = dq_conv
        dq_dt = dq_dt.at[..., -1].add(dq_BL)

        # Hyperdiffusion on q_v (dampen 2Δx noise; spectral/voronoi handle in dycore)
        if _apply_hyperdiff is not None:
            dq_dt = dq_dt + _apply_hyperdiff(q_v, HYPERDIFF)

        return dT_dt, dq_dt, T_sfc_new, T_deep, W_new, precip

    # ---------------------------------------------------------------
    # Time integration
    # ---------------------------------------------------------------
    n_steps = int(args.days * 86400 / DT)
    diag_interval = int(args.diag_days * 86400 / DT)
    # iter-75 (mirror of plane CRM iter-75 fix): reject huge --dt that
    # truncates n_steps to 0. iter-71 caught --days <= 0 at parse time
    # but a finite positive --dt > days*86400 silently produces a
    # zero-step run.
    if n_steps < 1:
        raise SystemExit(
            f"error: --dt rejected: n_steps={n_steps} (computed "
            f"from --days={args.days} × 86400 s / dt={DT}); must "
            f"be >= 1. Either --dt is too large or --days is too "
            f"small."
        )

    _ocean_label = OCEAN_MODE if not IS_LAND else "slab_soil"
    print("=" * 70)
    print(f"  Moist RCE: {_ocean_label} {args.mode} + gray radiation + SBM convection")
    print("=" * 70)
    _grid_labels = {
        "cubed_sphere": f"C{N}", "gaussian": f"T{N}",
        "latlon": f"LL{N}", "voronoi": f"V{N}",
    }
    print(f"  Grid:     {_grid_labels[grid_type]}/L{NLEV} ({grid_type}/{discretization})"
          f",  dt={DT:.0f}s,  {args.days} days")
    print(f"  Surface:  T_init={args.sst_init:.0f}K, C_sfc={C_sfc:.2e} J/m2/K")
    print()
    print(f"  {'Day':>6s}  {'<T_sfc>':>8s}  {'<T_atm>':>8s}  {'<Precip>':>8s}"
          f"  {'<CWV>':>6s}  {'max|v|':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}")

    t_start = time.time()

    # iter-24: collect diagnostics per snapshot for matrix-compatible
    # CSV output.  Each diag dict slot records the same scalars that
    # the run loop already prints, so we can emit ``mean_timeseries.csv``
    # for the cross-grid plotter without changing the loop body.
    diag_log: list[dict[str, float]] = []
    blowup = False

    for step in range(n_steps):
        # (1) Dynamics only (no inline physics)
        state = model.step(state, DT)

        # Project the dynamics state into grid-space (T, p_s, u, v) for
        # the physics step.  For HydrostaticState this is a no-op view;
        # for spectral/MPAS states this performs the spectral synthesis
        # / Perot reconstruction.
        T_grid, p_s_grid, u_grid, v_grid = to_grid_arrays(state)

        # (2) Operator-split physics
        _T_deep_in = T_deep if IS_TWO_LAYER else jnp.zeros(shape_2d)
        dT_dt, dq_dt, T_sfc, _T_deep_out, W_bucket, precip = physics_step(
            T_grid, p_s_grid, q_v,
            u_grid, v_grid,
            T_sfc, _T_deep_in, W_bucket, grid.grid_lat, DT,
        )
        if IS_TWO_LAYER:
            T_deep = _T_deep_out
        new_T = T_grid + DT * dT_dt
        q_v = jnp.maximum(q_v + DT * dq_dt, 0.0)

        # (3) Large-scale condensation (saturation adjustment)
        q_sat = saturation_mixing_ratio(
            new_T, p_s_grid[..., None] * sigma.sigma_full,
        )
        excess = jnp.maximum(q_v - q_sat, 0.0)
        q_v = q_v - excess
        new_T = new_T + constants.L_v * excess / constants.c_pd

        # Push the saturation-adjusted temperature back into the
        # grid-native state representation.
        state = apply_T_update(state, new_T)

        # Large-scale precipitation: column-integrated condensation [kg/m2/s]
        ls_precip = jnp.sum(excess * p_s_grid[..., None] * dsigma,
                            axis=-1) / constants.g / DT
        precip = precip + ls_precip  # total = convective + large-scale
        if IS_LAND:
            W_bucket = jnp.clip(W_bucket + DT * ls_precip, 0.0, W_max)

        # (4) Rayleigh friction
        state = apply_friction(state, fric_decay)

        # Diagnostics
        if (step + 1) % diag_interval == 0:
            T_diag, p_s_diag, u_diag, v_diag = to_grid_arrays(state)
            jax.block_until_ready(u_diag)
            day = (step + 1) * DT / 86400.0
            mean_sfc = float(jnp.mean(T_sfc))
            mean_T = float(jnp.mean(T_diag))
            max_v = float(jnp.max(jnp.sqrt(u_diag ** 2 + v_diag ** 2)))
            mean_precip = float(jnp.mean(precip)) * 86400.0
            cwv = column_water_vapor(q_v, p_s_diag, dsigma)
            mean_cwv = float(jnp.mean(cwv))

            print(f"  {day:6.0f}  {mean_sfc:8.2f}  {mean_T:8.2f}"
                  f"  {mean_precip:8.2f}  {mean_cwv:6.1f}  {max_v:8.2f}")

            diag_log.append({
                "step": int(step + 1),
                "time_days": float(day),
                "mean_T_sfc": mean_sfc,
                "mean_T": mean_T,
                "mean_precip": mean_precip,
                "mean_cwv": mean_cwv,
                "max_wind": max_v,
            })

            # BLOWUP threshold lowered from 500 -> 200 m/s in iter-13
            # (2026-05). At 500 m/s the iter-13 C48 run cleared the
            # gate but was clearly unphysical (max|v|=236 m/s after a
            # cooling crash that pulled mean_T_sfc to 289 K — slab
            # ocean dropped 11 K from IC in 30 days). Jet streams cap
            # at ~100 m/s and the sound speed is ~330 m/s; anything
            # over 200 m/s in a hydrostatic RCE is either a CFL crash
            # in progress or a numerical instability that is about to
            # NaN. Catching it earlier surfaces the right config
            # change (lower dt) instead of saving a corrupted file.
            if not bool(is_finite_state(state)) or max_v > 200:
                print(f"  BLOWUP at day {day:.0f}")
                blowup = True
                break

    _, _, u_final, _ = to_grid_arrays(state)
    jax.block_until_ready(u_final)
    total = time.time() - t_start
    print(f"\n  Complete: {total:.1f}s wall time")
    print(f"  Output: {OUTPUT_DIR}")

    # iter-24: persist diagnostics in matrix-compatible format so the
    # cross-grid plotter from ``run_atmosphere_test_matrix.py``
    # (``--cross-grid-plots-only``) can pick this run up.  Emits
    # ``mean_timeseries.csv`` (one row per diag interval) and
    # ``results.txt`` (the matrix's per-test metadata file).
    if diag_log:
        import csv
        csv_path = OUTPUT_DIR / "mean_timeseries.csv"
        cols = list(diag_log[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for row in diag_log:
                writer.writerow(row)

        last = diag_log[-1]
        notes = (
            f"BLOWUP at day {last['time_days']:.0f}"
            if blowup
            else f"mean_T_sfc={last['mean_T_sfc']:.2f}, "
                 f"mean_T={last['mean_T']:.2f}, "
                 f"max|v|={last['max_wind']:.2f}"
        )
        with open(OUTPUT_DIR / "results.txt", "w") as fh:
            fh.write(f"test: rce\n")
            fh.write(f"grid: {grid_type}\n")
            fh.write(f"resolution: {N}\n")
            fh.write(f"days: {args.days}\n")
            fh.write(f"dt: {DT}\n")
            fh.write(f"levels: {NLEV}\n")
            fh.write(f"mode: {args.mode}\n")
            fh.write(f"ocean_mode: {args.ocean_mode}\n")
            fh.write(f"status: {'FAIL' if blowup else 'PASS'}\n")
            fh.write(f"notes: {notes}\n")
            fh.write(f"wall_time: {total:.1f}s\n")

    # iter-101: BLOWUP must surface as a non-zero exit code so
    # wrapper scripts (``run_rce_cross_grid.sh``) and direct
    # invocations can detect failure via ``$?``.  Pre-iter-101,
    # ``run_rce.py`` wrote ``status: FAIL`` to results.txt but
    # exited 0 (the default), which the iter-73 wrapper comment
    # explicitly flagged as a "defensive guard for future
    # regressions" (``run_rce.py`` did not currently exit
    # non-zero the way run_omip.py did).  iter-101 closes that
    # gap, mirroring the iter-100 ``run_amip.py`` and iter-97
    # ``run_omip.py`` exit-code conventions.
    if blowup:
        sys.exit(1)


if __name__ == "__main__":
    main()
