"""BENCH — a performance-benchmark ocean configuration inspired by NEMO BENCH.

Ports the NEMO ``tests/BENCH`` configuration (Irrmann et al. 2022, GMD 15,
1567-1582, Sect. 2.2; NEMO sources at ``tests/BENCH/{MY_SRC,EXPREF}`` on
``nemo-ocean.eu`` main) to legoESM's lat-lon / synthetic-tripole C-grid.

Design principles (verbatim from the paper, Sect. 2.2.1):

1. **Zero input files.**  Grid, bathymetry, initial conditions and forcing
   are all analytic; a full simulation requires only the source checkout.
   The absence of input/output files also removes disk-access noise from
   performance measurements (output can be enabled like any other run).
2. **Unique value at every grid point.**  T, S, u, v and ssh each carry a
   per-point linear ramp perturbation, so any halo-exchange / sharding bug
   that duplicates or drops a point is detectable in the fields
   themselves.  This is NEMO's ``z2d`` device, ported exactly.
3. **Full production complexity.**  The benchmark steps the SAME dycore
   recipe as production OMIP-style runs (``legoesm_nemo_like_v1``), not a
   stripped-down dwarf — the measured step cost is the production cost.
4. **Results are physically meaningless.**  BENCH output must only be used
   for benchmarking; the validation gates below check STABILITY, not
   science.

NEMO BENCH reference formulas (sources quoted as file:line, NEMO main):

* ``usrdef_istate.F90:44-77`` — with the global 1-D point index
  ``p = (2*i_g - 1 + 2*(j_g - 1)*Ni0glo) / (Ni0glo * Nj0glo)``
  (doubled numerator — NEMO's own device so u/v face points also
  get unique values) and the hemisphere-mirrored ramp::

      z2d = 0.1 * (p - 0.5)                (southern hemisphere)
      z2d = 0.1 * (1.5 - p')              (northern hemisphere, p' the
                                           same index counted so the ramp
                                           mirrors at the equator row)

  then, with depth factor ``f = (k) / (jpk - 1)``::

      T = 20 * z2d - 1 - 0.5 * f           [-1 .. -1.5 degC, +/- 1.0]
      S = 30 + 1 * f + z2d                 [30 .. 31 psu, +/- 0.05]
      u = 0.1 * z2d  [m/s]                 (NEMO multiplies by umask)
      v = 0.01 * z2d [m/s]
      ssh = 0.1 * (0.5 - p)               [+/- 0.05 m]

  (amplitudes are NEMO's coefficients times the z2d range — see
  ``build_z2d_ramp`` for the exact range statement.)

* ``usrdef_zgr.F90:139-165`` — flat bottom, UNIFORM vertical grid with
  ``zd = 5000/jpkm1``: NEMO's ``nn_ksize=75`` gives ``jpk=75`` hence
  **74 wet levels** of ``dz = 5000/74 ≈ 67.57 m`` (k_bot = jpkm1
  everywhere, no land, no cavities); ``n_levels`` here is that wet
  count.

* ``usrdef_sbc.F90:56-92`` — surface forcing ALL ZERO for the ocean
  (utau = vtau = qns = qsr = emp = sfx = 0).

* ``namusr_def`` (usrdef_nam.F90:103-114) — domain size set by
  parameters (``nn_isize/jsize/ksize``); a NEGATIVE size means
  per-subdomain size (weak-scaling mode), which requires actual SPMD
  sharding — a documented follow-up in the runners-repo harness, NOT
  wired here (see ``bench_plan.md`` divergences).

legoESM divergences (each deliberate, recorded in
``docs/ocean/experiments/bench_plan.md``):

* **Spherical grids instead of the flat 100-km beta-plane.**  NEMO BENCH
  fixes dx = 100 km regardless of domain size so one dt fits all; legoESM
  runs on the sphere, so dt scales with resolution (per-preset values).
  The synthetic tripole (``create_synthetic_tripole``) attaches an active
  north fold with NO mesh file, reproducing NEMO BENCH's ORCA-like
  fold-communication pattern (``ln_NFold = .true.`` in every BENCH
  preset namelist).
* **The z2d ramp is mapped by global (j, i) index, not by a
  decomposition-dependent MPI rank formula** — NEMO computes it from
  global indices too (``mig``/``mjg``), so the port is exact on the
  serial path.
* **No SI3/PISCES lanes** (legoESM sea-ice/BGC benchmarking is a
  follow-up); **no ``ln_perpetual_ts``** (an Asselin-filter-specific
  NEMO trick, inapplicable to legoESM integrators).
* **ICs target a stably-stratified, lightly-perturbed ocean** rather
  than NEMO's near-freezing BENCH profile (designed so ~1/5 of the domain
  grows sea ice under SI3).  The legoESM BENCH runs TKE vertical mixing
  without SI3, so the T/S background is a light stratification (the
  paper: "almost constant everywhere with a light stratification which
  keeps the vertical stability") around temperate values, with the
  per-point ramp amplitudes and depth trends preserved verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass
class BenchConfig:
    """Configuration for the BENCH performance-benchmark experiment.

    Defaults mirror NEMO BENCH's ORCA1-like preset geometry where the
    concepts transfer (74 wet levels = NEMO jpkm1 for nn_ksize=75,
    5000 m flat bottom) and legoESM's OMIP conventions elsewhere.
    ``n_lat``/``n_lon`` are deliberately NOT defaulted to a production
    size: the test matrix and run harness set them per preset (see
    ``BENCH_PRESETS``).
    """

    # --- Domain / grid ---
    n_lat: int = 278  # Latitude rows (derived: 2K for the mercator preset)
    n_lon: int = 360  # Longitude cells
    n_levels: int = 74  # WET levels (NEMO nn_ksize=75 -> jpk=75 -> jpkm1=74)
    H_max: float = 5000.0  # Flat-bottom depth [m] (NEMO: 5000 m)
    # Mercator truncation latitude [deg] for the latlon lane (see
    # BENCH_PRESETS): NEMO BENCH never runs a full-sphere equirectangular
    # grid (its 100-km box is closed / the ORCA-like presets fold); a
    # full-sphere equirectangular latlon grid's pole-row dx (~110 m at
    # 89.75 deg for 1 deg) is barotropic-CFL-unstable at ANY preset dt
    # (measured, scripts/validate/ocean_bench/stability_probes.py), so the
    # legoESM latlon analog truncates at 80 deg with isotropic (DINO-style
    # Mercator) row placement.
    lat_max_deg: float = 80.0

    # --- NEMO usrdef_istate.F90 amplitudes (ported verbatim) ---
    # z2d is O(0.1) by construction (the 0.1 prefactor is IN the ramp
    # builder; the coefficients below are NEMO's multipliers on z2d).
    T_z2d_coeff: float = 20.0  # T = 20*z2d - T_base - T_strat*f
    T_base_C: float = 10.0  # Background T [degC] (see divergence note)
    T_strat_C: float = 4.0  # T decreases by this over the column [degC]
    S_base: float = 34.0  # Background S [psu] (see divergence note)
    S_strat: float = 1.0  # S increases by this over the column [psu]
    u_z2d_coeff: float = 0.1  # u = 0.1 * z2d [m/s] (NEMO usrdef_istate:74)
    v_z2d_coeff: float = 0.01  # v = 0.01 * z2d [m/s] (NEMO usrdef_istate:75)
    ssh_amplitude: float = 0.1  # ssh = 0.1 * (0.5 - p) [m] (usrdef_istate:104)

    # --- Stability / validation gates (pre-registered; stability only) ---
    max_speed_limit: float = 1.0  # |u|,|v| must stay below [m/s]
    max_eta_limit: float = 1.0  # |eta| must stay below [m]

    # --- Resolution presets (NEMO BENCH namelist_cfg_orca{1,025,12}_like) ---
    # NEMO sizes: orca1_like 360x331, orca025_like 1440x1206,
    # orca12_like 4320x3146 (nn_isize x nn_jsize), nn_ksize = 75 (74 wet
    # levels).  The latlon presets are the equirectangular full-sphere
    # analog (360x180 etc.); the tripole presets keep NEMO's EXACT
    # j-count (331/1206/3146) for fold-row-count parity.  Note: NEMO's
    # orca1_like uses an F-point fold (cn_NFtype='F') while the
    # synthetic tripole is a T-fold — a documented divergence.
    A_h: float = 1.0e4  # Laplacian viscosity [m2/s] (1-deg OMIP)
    A_v: float = 1.0e-3  # Vertical viscosity [m2/s]
    K_v: float = 1.0e-4  # Background vertical diffusivity [m2/s]


# Resolution presets for the latlon (Mercator-truncated) lane.  NEMO
# sizes: orca1_like 360x331, orca025_like 1440x1206, orca12_like
# 4320x3146 (nn_isize x nn_jsize), nn_ksize = 75 (74 wet levels).
# The legoESM latlon analog keeps NEMO's n_lon and truncates at
# lat_max_deg=80 with isotropic Mercator rows (n_lat = 2K is DERIVED:
# 278/1116/3350); the tripole presets keep NEMO's EXACT j-count
# (331/1206/3146) for fold-row-count parity.  n_levels is the WET count
# (NEMO jpkm1 = nn_ksize - 1 = 74).
#
# dt values are MEASURED first-stable-guess boundaries from the
# committed probe (scripts/validate/ocean_bench/stability_probes.py,
# 2026-10-10) halved for margin: the 1-deg analog (n_lon=360) is stable
# 60 steps at dt<=1500 s and NaN by dt=1650 s; the 0.5-deg analog
# (n_lon=720) is stable at dt<=600 s, NaN by dt=750 s (the dt ceiling is
# the Mercator-wall row's CFL, scaling with n_lon).  The finer presets
# follow the measured dx-scaling; re-measure before long runs.
# NEMO BENCH's dt is likewise a per-preset namelist value (5400/1440/480
# s for orca{1,025,12}_like — its fixed 100-km box allows one dt family,
# our spherical grids do not).
BENCH_PRESETS: dict[str, dict[str, Any]] = {
    "orca1_like": {
        "n_lon": 360,
        "n_lat": 278,
        "n_lat_tripole": 331,
        "n_levels": 74,
        "dt_seconds": 1200.0,
        "lat_max_deg": 80.0,
    },
    "orca025_like": {
        "n_lon": 1440,
        "n_lat": 1116,
        "n_lat_tripole": 1206,
        "n_levels": 74,
        "dt_seconds": 300.0,
        "lat_max_deg": 80.0,
    },
    "orca12_like": {
        "n_lon": 4320,
        "n_lat": 3350,
        "n_lat_tripole": 3146,
        "n_levels": 74,
        "dt_seconds": 120.0,
        "lat_max_deg": 80.0,
    },
}


def create_latlon_grid_for_preset(n_lon: int, lat_max_deg: float):
    """The BENCH latlon grid builder (Mercator-truncated, isotropic rows).

    n_lat = 2K is DERIVED from (n_lon, lat_max_deg) — the same placement
    as NEMO's DINO test case (K = floor(arctanh(sin(lat_max))/dlon));
    callers must read the row count off the returned grid, not assume it.
    """
    from legoesm.grids.latlon import create_mercator_grid

    return create_mercator_grid(n_lon=n_lon, lat_max_deg=lat_max_deg)


def build_z2d_ramp(n_lat: int, n_lon: int, dtype=None) -> jnp.ndarray:
    """The NEMO BENCH per-point ramp, ported verbatim.

    NEMO ``usrdef_istate.F90:44-54``: with 1-based global indices
    ``i_g in [1, Ni0glo]``, ``j_g in [1, Nj0glo]``::

        p = (2*i_g - 1 + 2*(j_g - 1)*Ni0glo) / (Ni0glo*Nj0glo)
        z2d = 0.1 * (p - 0.5)                    (southern hemisphere)
        z2d = 0.1 * (1.5 - p')                    (northern hemisphere)

    where the northern-hemisphere index ``p'`` counts so the ramp mirrors
    across the equator: NEMO writes ``(2*i_g + 2*(j_g-1)*Ni0glo)`` over the
    northern rows, i.e. the same 0 -> 1 sweep traversed in reverse.  The
    doubled numerator (NEMO ``mig(ji)*2 - 1 + 2*(mjg(jj)-1)*Ni0glo``) is
    what makes the ramp unique per point; it is reproduced exactly here.
    NEMO's own comment says "from -0.05 to 0.05 (Southern Hemisphere) and
    from 0.05 to -0.05 (Northern Hemisphere)"; with the hemisphere
    split at ``mjg < Nj0glo/2`` (REAL division) the first north row
    overshoots by one ramp step: the max is ``0.1*(1.5 - (2 + 2*(j_g-1)
    *Ni0glo)/(Ni0glo*Nj0glo))`` at the first north row (e.g. +0.0621 at
    Nj=16, +0.0503 at Nj=331 — NEMO's construction, reproduced).  A
    unique value per (i, j) — the MPI-bug detector.

    The returned array has shape ``(n_lat, n_lon)``.  The equator split
    follows NEMO's REAL division ``mjg(jj,0) < Nj0glo/2._wp``
    (usrdef_istate.F90:49): for even ``n_lat`` the southern half has
    ``n_lat/2 - 1`` rows, for odd ``n_lat`` it has ``(n_lat-1)/2`` rows —
    expressed branch-free as ``2*j_g < n_lat`` on the 1-based row index.
    """
    i_g = jnp.arange(1, n_lon + 1, dtype=jnp.result_type(jnp.float32) if dtype is None else dtype)
    j_g = jnp.arange(1, n_lat + 1, dtype=i_g.dtype)

    # Southern-hemisphere index (continuous 0 -> ~2 sweep across the globe;
    # NEMO's numerator omits the -1 offset in the north so the two halves
    # meet mirrored — reproduced exactly here).
    num_s = (2.0 * i_g[None, :] - 1.0) + 2.0 * (j_g[:, None] - 1.0) * n_lon
    num_n = (2.0 * i_g[None, :]) + 2.0 * (j_g[:, None] - 1.0) * n_lon
    denom = float(n_lon * n_lat)

    south = 0.1 * (num_s / denom - 0.5)
    north = 0.1 * (1.5 - num_n / denom)

    is_south = 2.0 * j_g[:, None] < n_lat  # NEMO: mjg < Nj0glo/2._wp (REAL div)
    z2d = jnp.where(is_south, south, north)
    return z2d.astype(dtype) if dtype is not None else z2d


def bench_uniform_z_star(n_levels: int, H_max: float):
    """Uniform-level z* coordinate — NEMO BENCH ``usrdef_zgr.F90:139-165``.

    NEMO: ``zd = 5000./REAL(jpkm1, wp)`` with ``jpk = nn_ksize = 75``,
    i.e. **74 wet levels** of ``dz = 5000/74 ≈ 67.57 m`` (interfaces at
    ``k*zd``, k_bot = jpkm1 everywhere).  ``n_levels`` here is the WET
    count, so the exact port is the constant ``dz = H_max / n_levels``
    with ``n_levels = 74``.  The legoESM ``create_ocean_z_star`` cannot
    produce uniform levels (its linear ramp is always normalized), so
    the explicit-thickness builder is used.

    Returns an ``OceanZStarCoordinate`` with n_levels uniform layers.
    """
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    dz = H_max / n_levels
    return create_z_star_from_thicknesses(jnp.full((n_levels,), dz, dtype=jnp.float64))


def create_initial_conditions(grid_type: str, grid, z_coord, config: BenchConfig = None):
    """Create BENCH initial conditions (unique value per point).

    Ports ``usrdef_istate.F90``: every field is the sum of a small
    stratification trend over depth and the per-point ``z2d`` ramp, so
    every horizontal point carries a unique value.  ssh carries the
    point-index ramp directly (``usrdef_istate_ssh``).
    """
    if config is None:
        config = BenchConfig()

    if grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )

        # No land (NEMO BENCH default: closed flat box, k_bot=jpkm1
        # everywhere).  land_lat_threshold=90 => all ocean.
        state = rest_state_latlon_cgrid_ocean(
            grid,
            z_coord,
            T_water_init_C=config.T_base_C,
            T_deep=config.T_base_C,
            S_uniform=config.S_base,
            H_max=config.H_max,
            land_lat_threshold=90.0,
        )
    elif grid_type == "tripole":
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )

        # Synthetic tripole geometry is a LatLonCGridGeometry; the same
        # rest-state initializer applies (fold halo is exercised by the
        # operators, not by the ICs).
        state = rest_state_latlon_cgrid_ocean(
            grid,
            z_coord,
            T_water_init_C=config.T_base_C,
            T_deep=config.T_base_C,
            S_uniform=config.S_base,
            H_max=config.H_max,
            land_lat_threshold=90.0,
        )
    else:
        raise NotImplementedError(
            f"BENCH is implemented for 'latlon' and 'tripole' grids, not {grid_type!r}"
        )

    n_lat = int(state.T.data.shape[0])
    n_lon = int(state.T.data.shape[1])
    nlev = int(state.T.data.shape[2])

    # Per-point ramp z2d (usrdef_istate.F90:44-54): [-0.05, +0.05] for
    # odd n_lat, up to +0.05 + 0.1/n_lat for even n_lat (NEMO's
    # hemisphere-split construction — see build_z2d_ramp).
    z2d = build_z2d_ramp(n_lat, n_lon, dtype=state.T.data.dtype)

    # Depth factor f = (jk-1)/(jpk-1) over the wet levels: 0 at the
    # surface, (nlev-1)/nlev at the last wet level (NEMO: zfact =
    # REAL(jk-1)/REAL(jpk-1) with jpk = wet count + 1).
    k = jnp.arange(nlev, dtype=z2d.dtype)
    f = k / jnp.maximum(nlev, 1.0)

    # usrdef_istate.F90:70-76 (amplitudes from BenchConfig; backgrounds
    # per the divergence note).  C-grid staggering: u lives on the
    # (n_lat, n_lon+1) lon-faces, v on the (n_lat+1, n_lon) lat-faces.
    # DIVERGENCE (recorded in bench_plan.md): NEMO evaluates z2d at the
    # T-grid point and multiplies by umask/vmask; here each FACE axis
    # gets its own ramp built with the face count, so every face value
    # stays unique — preserving the MPI-bug-detection property.
    T_ic = (
        config.T_z2d_coeff * z2d[:, :, None] + config.T_base_C - config.T_strat_C * f[None, None, :]
    )
    S_ic = config.S_base + config.S_strat * f[None, None, :] + z2d[:, :, None]
    z2d_u = build_z2d_ramp(n_lat, n_lon + 1, dtype=z2d.dtype)
    z2d_v = build_z2d_ramp(n_lat + 1, n_lon, dtype=z2d.dtype)
    u_ic = jnp.broadcast_to(config.u_z2d_coeff * z2d_u[:, :, None], (n_lat, n_lon + 1, nlev))
    v_ic = jnp.broadcast_to(config.v_z2d_coeff * z2d_v[:, :, None], (n_lat + 1, n_lon, nlev))

    # usrdef_istate_ssh: pssh = 0.1 * (0.5 - p) with p the plain
    # (i + (j-1)*Ni0glo)/(Ni*Nj) point index — NOT the doubled NEMO
    # numerator used for z2d (that ramp exists so u/v faces get unique
    # values too; the ssh ramp only needs per-cell uniqueness).
    i_g = jnp.arange(1, n_lon + 1, dtype=z2d.dtype)
    j_g = jnp.arange(1, n_lat + 1, dtype=z2d.dtype)
    p = (i_g[None, :] + (j_g[:, None] - 1.0) * n_lon) / (n_lat * n_lon)
    eta_ic = config.ssh_amplitude * (0.5 - p)

    from legoesm.core.field import Field

    return state._replace(
        T=Field(T_ic.astype(state.T.data.dtype), name="T", dims=state.T.dims, units=state.T.units),
        S=Field(S_ic.astype(state.S.data.dtype), name="S", dims=state.S.dims, units=state.S.units),
        u=Field(u_ic.astype(state.u.data.dtype), name="u", dims=state.u.dims, units=state.u.units),
        v=Field(v_ic.astype(state.v.data.dtype), name="v", dims=state.v.dims, units=state.v.units),
        eta=Field(
            eta_ic.astype(state.eta.data.dtype),
            name="eta",
            dims=state.eta.dims,
            units=state.eta.units,
        ),
    )


def create_forcings(grid_type: str, grid, config: BenchConfig = None):
    """BENCH forcing: ALL ZERO (NEMO usrdef_sbc.F90:56-92).

    Returns an ``OceanPhysicsConfig`` whose surface forcing is disabled
    and whose vertical mixing is the production TKE closure (NEMO BENCH
    runs ``ln_zdftke=.true.`` in every orca*_like preset), so the
    benchmarked step cost includes the production mixing physics.
    """
    if config is None:
        config = BenchConfig()

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import (
        EnhancedDiffusionConfig,
        OceanConvectionConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.config import (
        LateralMixingConfig,
    )
    from legoesm.ocean.physics.surface_forcing.config import (
        SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        TKEConfig,
        VerticalMixingConfig,
    )

    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        vertical_mixing=VerticalMixingConfig(
            scheme="tke",
            tke=TKEConfig(),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(
                K_conv=1.0,  # 1 m2/s where statically unstable
                K_bg=config.K_v,
            ),
        ),
        shortwave_penetration=None,
        mle=None,
    )


def bench_model_config(
    config: BenchConfig = None,
    *,
    physics=None,
    eos_config=None,
    gm_redi_cfg=None,
    recipe: str = "legoesm_nemo_like_v1",
    **overrides,
):
    """Assemble the BENCH ``LatLonCGridOceanConfig``.

    The dycore identity is the production-like NEMO-style recipe
    (``legoesm_nemo_like_v1``: EEN-style vector-invariant momentum +
    Hollingsworth KE gradient, PPM/FCT tracers, TEOS-10-like EOS, smc03
    PGF, RK3 momentum — see ``recipes.py``).  Setup parameters
    (viscosity, background mixing) come from ``config``; any
    ``LatLonCGridOceanConfig`` field can be forced through
    ``overrides`` (applied last).

    NEMO BENCH physics mapping (per-preset namelist choices, translated):

    * ``ln_zdftke`` → ``VerticalMixingConfig(scheme="tke")`` (via
      ``create_forcings``; pass ``physics=`` to override) — APPLIED
    * ``ln_zdfevd`` → ``convection.scheme="enhanced_diffusion"`` — APPLIED
    * ``ln_dynspg_ts`` (split-explicit) → recipe barotropic
      ``explicit_substep`` — APPLIED (recipe card)
    * ``ln_teos10`` → recipe ``eos="veros_gsw"`` — APPLIED (recipe card)
    * ``ln_non_lin`` (quadratic bottom drag): NEMO's presets enable it,
      but BENCH here keeps the recipe default (linear drag,
      ``bottom_drag_r=0``) — a documented divergence; pass
      ``bottom_drag_scheme="nemo_quadratic"`` via ``overrides`` to
      enable it.
    * NOT mapped (documented in ``bench_plan.md``): MLE/EIV (orca1
      namelist), bilaplacian tracer diffusion (orca025/12), UBS
      momentum + no lateral dyn. diffusion (orca12), double-diffusion
      and internal-wave mixing (all presets).

    Decision-90 statements (required by the NEMO RK3 momentum program —
    there is no default): the BENCH box is a full-step flat-bottom
    z-star mesh, so NEMO's own statement (``stp2d.F90:177-186``,
    reference face thickness / stored reciprocal, no stretching) and
    the per-level minimum rule are algebraically identical; we state
    NEMO's, matching ``nemo_recipe.nemo_lat_lon_model_config``.

    ``eos_config``/``gm_redi_cfg`` are accepted for signature parity
    with the registry runner's factory path (``_run_experiment_via_
    registry`` passes them); BENCH runs the recipe's non-linear EOS
    (``veros_gsw``) and no GM/Redi, so a non-None value is REFUSED
    loudly rather than silently ignored (a linear-EOS or GM/Redi
    request means the caller wants a different experiment).
    """
    from legoesm.ocean.recipes import assemble_ocean_config, get_recipe
    from legoesm.ocean.state import LatLonCGridOceanConfig

    if config is None:
        config = BenchConfig()
    if physics is None:
        physics = create_forcings("latlon", None, config)
    if eos_config is not None:
        raise ValueError(
            "bench_model_config: eos_config is not accepted — BENCH runs "
            "the recipe's non-linear EOS (veros_gsw for legoesm_nemo_like_"
            "v1). Pass eos_config=None."
        )
    if gm_redi_cfg is not None:
        raise ValueError(
            "bench_model_config: gm_redi_cfg is not accepted — BENCH runs "
            "no GM/Redi (NEMO BENCH presets enable no EIV). Pass "
            "gm_redi_cfg=None."
        )

    bundle = get_recipe(recipe, "latlon")
    # Decision-90 statements the NEMO recipe card requires (no defaults).
    defaults = dict(
        barotropic_slow_forcing_depth_evaluation="nemo_literal",
    )
    params = dict(defaults)
    params.update(bundle)
    return assemble_ocean_config(
        params,
        LatLonCGridOceanConfig,
        overrides=overrides,
        physics=physics,
        A_h=config.A_h,
        A_v=config.A_v,
        K_v=config.K_v,
        bottom_drag_r=0.0,
    )


def create_z_coord(config: BenchConfig = None):
    """Uniform z* coordinate for BENCH (the matrix registry hook).

    ``EXPERIMENT_CONFIG["create_z_coord"]`` lets ``_run_experiment_via_
    registry`` build the experiment's OWN vertical grid instead of the
    default stretched ``create_ocean_z_star`` — without it the matrix
    lane would silently run a different vertical grid than the
    standalone ``run_bench.py`` harness.
    """
    if config is None:
        config = BenchConfig()
    return bench_uniform_z_star(config.n_levels, config.H_max)


def create_domain_config(config: BenchConfig = None) -> dict[str, Any]:
    if config is None:
        config = BenchConfig()
    return {
        "n_lat": config.n_lat,
        "n_lon": config.n_lon,
        "H_max": config.H_max,
        "n_levels": config.n_levels,
        "description": (
            "Flat-bottom all-ocean benchmark box (no input "
            "files); per-point unique ICs; zero forcing"
        ),
        "forcing_type": "none",
        "stratification": "light linear T/S trend + per-point ramp",
        "vertical_grid": "uniform z* (NEMO BENCH usrdef_zgr)",
        "presets": {k: dict(v) for k, v in BENCH_PRESETS.items()},
    }


def validate_results(
    final_state, diagnostics: dict[str, list], config: BenchConfig = None
) -> tuple[bool, str]:
    """BENCH validation: STABILITY ONLY (pre-registered gates).

    The configuration is physically meaningless by design (paper Sect.
    2.2.1); the only meaningful checks are that the integration stays
    finite and bounded.  These gates are written down BEFORE any long
    run and are not to be relaxed to make a run pass:

    * every field finite at the final step;
    * max |u|, |v| < ``max_speed_limit`` (1.0 m/s; ICs are O(0.01));
    * max |eta| < ``max_eta_limit`` (1.0 m; ICs are O(0.05)).

    Planted-violation control: ``tests/ocean/unit/test_bench.py``
    injects NaN / overspeed / overshot-eta states and asserts each gate
    FIRES (a gate that cannot fail is a gate that does not check).
    """
    if config is None:
        config = BenchConfig()

    notes_parts = []
    success = True

    # Finiteness (planted-violation: NaN arrays in the unit test).
    for field_name in ("eta", "T", "S", "u", "v"):
        if hasattr(final_state, field_name):
            data = getattr(final_state, field_name).data
            if not jnp.all(jnp.isfinite(data)):
                return False, f"NaN/Inf in final {field_name}"

    # Speed gate (planted-violation: u > max_speed_limit in the unit test).
    max_speed_list = diagnostics.get("max_speed", [])
    if max_speed_list:
        speed = float(max_speed_list[-1])
        notes_parts.append(f"max_speed={speed:.4f} m/s")
        if speed > config.max_speed_limit:
            success = False
            notes_parts.append(f"FAIL: max_speed {speed:.4f} exceeds {config.max_speed_limit} m/s")

    # SSH gate (planted-violation: |eta| > max_eta_limit in the unit test).
    # The matrix scalar function emits ``max_abs_eta`` (its canonical key)
    # and the standalone harness records ``max_eta``; both are accepted so
    # the gate is LIVE in every consumer (a gate that reads a key nobody
    # writes does not check).
    max_eta_list = diagnostics.get("max_eta") or diagnostics.get("max_abs_eta") or []
    if max_eta_list:
        max_eta = float(max_eta_list[-1])
        notes_parts.append(f"max_eta={max_eta:.4f} m")
        if max_eta > config.max_eta_limit:
            success = False
            notes_parts.append(f"FAIL: max |eta| {max_eta:.4f} exceeds {config.max_eta_limit} m")

    return success, ", ".join(notes_parts)


def get_diagnostic_field_specs() -> list:
    return [
        ("eta", "SSH (m)", "RdBu_r"),
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("speed_sfc", "Surface speed (m/s)", "magma"),
    ]


def get_scalar_units() -> dict[str, str]:
    return {
        "mean_eta": "m",
        "max_speed": "m/s",
        "max_eta": "m",
        "mean_T": "degC",
        "mean_S": "PSU",
    }


EXPERIMENT_CONFIG = {
    "name": "bench",
    "description": (
        "NEMO-BENCH-inspired performance benchmark: flat-bottom "
        "all-ocean box, zero forcing, per-point unique ICs"
    ),
    "scientific_purpose": (
        "Benchmarking only — results are physically meaningless by design (Irrmann et al. 2022)"
    ),
    "reference": (
        "Irrmann et al. (2022), GMD 15, 1567-1582, Sect. 2.2; "
        "NEMO tests/BENCH (MY_SRC/usrdef_{istate,zgr,sbc}.F90)"
    ),
    "config_class": BenchConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_model_config": bench_model_config,
    "create_z_coord": create_z_coord,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 5.0,  # days — arbitrary; benchmarks count STEPS
    "quick_duration": 0.25,  # days
    "expected_metrics": {
        "finiteness": "all fields finite",
        "max_speed": "< 1.0 m/s",
        "max_eta": "< 1.0 m",
    },
    "grid_support": {
        "cubed_sphere": False,
        "latlon": True,
        "mpas": False,  # follow-up: MPAS Voronoi lane
        "spectral": False,
        "tripole": True,  # synthetic tripole (active fold, no mesh file)
    },
}
