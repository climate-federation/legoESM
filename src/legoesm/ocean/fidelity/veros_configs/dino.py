"""Veros DINO (Diabatic Neverworld Ocean) setup — Kamm et al. 2025.

Reference run for fidelity comparison against the legoESM ``dino``
experiment (:mod:`legoesm.ocean.experiments.dino`). The geometry follows
Kamm et al. (2025, GMD 18, 8091-8107) and the upstream NEMO source
``vopikamm/DINO@v0.2.0``:

* Pole-to-pole 50 deg-wide sector basin, lon in [-50, 0] deg, lat in
  [-70, 70] deg.
* Re-entrant zonal channel between 65 deg S and 45 deg S (the
  "Drake-passage" band).
* Idealised bathymetry tapering from H_deep = 4000 m in the interior to
  H_shallow = 2000 m at the coasts (smooth ramp; no Drake-passage sill
  on the Veros side -- the sill is in the legoESM port but is a
  second-order detail relative to the basin shape).
* Cubic-Hermite zonal wind stress through the seven (lat, tau) knots of
  the paper, surface T* and S* restoring profiles (cos in lat plus an
  equatorial Gaussian dip in S*).

Solar / Jerlov shortwave penetration is intentionally omitted on the
Veros side: Veros does not expose a Jerlov module out of the box and
the bulk-restoring view of the surface buoyancy forcing already covers
the dominant signal we want to compare (ACC transport, MOC topology,
deep T/S). The legoESM port applies Q_sr separately, so the comparison
script flags this divergence rather than hiding it.

Equation of state is ``eq_of_state_type=3`` (nonlinear, like the paper
which uses TEOS-10 / EOS-80) and the canonical Veros TKE / GM closures
are turned on so the reference run integrates without crashing.

Defaults are paper R1 (1 deg, nx=50, ny=140, nz=36, dt=2700 s). The
factory accepts ``nx``, ``ny``, ``nz``, ``dt_mom``, ``dt_tracer`` and
``runlen_s`` overrides so the smoke-test path can use a coarser grid
and a shorter run length.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Domain (Kamm et al. 2025 Sect 2.2, Table 2)
# ---------------------------------------------------------------------------
LON_WEST_DEG: float = -50.0
LON_EAST_DEG: float = 0.0
LAT_MAX_DEG: float = 70.0                       # symmetric N/S truncation
CHANNEL_LAT_SOUTH_DEG: float = -65.0
CHANNEL_LAT_NORTH_DEG: float = -45.0

H_DEEP_M: float = 4000.0
H_SHALLOW_M: float = 2000.0
COAST_TAPER_DEG: float = 5.0                    # half-width of the H taper near walls

# Wind-stress knots (paper eq 7)
WIND_TAU_LATS_DEG: tuple[float, ...] = (
    -70.0, -45.0, -15.0, 0.0, 15.0, 45.0, 70.0,
)
WIND_TAU_VALUES: tuple[float, ...] = (
    0.0, 0.2, -0.1, -0.02, -0.1, 0.1, 0.0,
)  # N/m^2

# Surface restoring targets (paper eqs B1-B2; annual mean)
T_STAR_EQ: float = 27.0
T_STAR_N: float = 5.0
T_STAR_S: float = -0.5
S_STAR_EQ: float = 37.25
S_STAR_N: float = 35.0
S_STAR_S: float = 35.1
S_STAR_EQ_DIP_AMP: float = 1.25
S_STAR_EQ_DIP_SIGMA_DEG: float = 7.5
L_PHI_DEG: float = 140.0

# Restoring timescale (days): chosen to match the paper coefficients
# A_theta = 40 W/m^2/K with dz_top = 10 m, rho_0=1026, c_p=3991.86
# tau_T = rho_0 * c_p * dz_top / A_theta ~ 11.85 d.
T_RESTORING_DAYS: float = 11.85
S_RESTORING_DAYS: float = 30.8

# Default resolution (paper R1)
NX_DEFAULT: int = 50
NY_DEFAULT: int = 140
NZ_DEFAULT: int = 36
DT_MOM_DEFAULT: float = 2700.0
DT_TRACER_DEFAULT: float = 2700.0

DEFAULT_RUNLEN_S: float = 30.0 * 86400.0   # 30 days shake-down


# ---------------------------------------------------------------------------
# Pure-NumPy helpers (no Veros / JAX imports). Used both inside and outside
# the setup class so the smoke-test code can re-derive the same forcing.
# ---------------------------------------------------------------------------

def _wind_stress(lat_deg, npx):
    """Cubic-Hermite smooth-step interpolation of tau_u through the knots."""
    lats = npx.asarray(WIND_TAU_LATS_DEG)
    taus = npx.asarray(WIND_TAU_VALUES)
    out = npx.zeros_like(lat_deg, dtype=lats.dtype)
    for i in range(len(WIND_TAU_LATS_DEG) - 1):
        lat_lo = WIND_TAU_LATS_DEG[i]
        lat_hi = WIND_TAU_LATS_DEG[i + 1]
        tau_lo = WIND_TAU_VALUES[i]
        tau_hi = WIND_TAU_VALUES[i + 1]
        s = npx.clip((lat_deg - lat_lo) / (lat_hi - lat_lo), 0.0, 1.0)
        weight = (3.0 - 2.0 * s) * s ** 2
        seg_val = tau_lo + (tau_hi - tau_lo) * weight
        in_seg = (lat_deg >= lat_lo) & (lat_deg <= lat_hi)
        out = npx.where(in_seg, seg_val, out)
    return out


def _T_star(lat_deg, npx):
    """Annual-mean T*(lat) per paper eq B1."""
    T_star_ns = npx.where(lat_deg <= 0.0, T_STAR_S, T_STAR_N)
    return T_star_ns + (T_STAR_EQ - T_star_ns) * npx.cos(
        math.pi * lat_deg / L_PHI_DEG,
    )


def _S_star(lat_deg, npx):
    """Annual-mean S*(lat) with equatorial Gaussian dip (paper eq B2)."""
    S_star_ns = npx.where(lat_deg <= 0.0, S_STAR_S, S_STAR_N)
    cos_factor = (1.0 + npx.cos(2.0 * math.pi * lat_deg / L_PHI_DEG)) / 2.0
    dip = S_STAR_EQ_DIP_AMP * npx.exp(
        -(lat_deg ** 2) / (S_STAR_EQ_DIP_SIGMA_DEG ** 2),
    )
    return S_star_ns + (S_STAR_EQ - S_star_ns) * cos_factor - dip


def _T_profile_1d(z_pos, npx):
    """Equatorial T(z) profile (paper eq D2; z_pos = depth, positive down)."""
    deep = 16.0 - 12.0 * npx.tanh((z_pos - 400.0) / 700.0)
    shallow = (
        15.0 * (1.0 - npx.tanh((z_pos - 50.0) / 1500.0))
        - 1.4 * npx.tanh((z_pos - 100.0) / 100.0)
        + 7.0 * (1500.0 - z_pos) / 1500.0
    )
    w_deep = (1.0 - npx.tanh((500.0 - z_pos) / 150.0)) / 2.0
    w_shallow = (1.0 - npx.tanh((z_pos - 500.0) / 150.0)) / 2.0
    return deep * w_deep + shallow * w_shallow


def _S_profile_1d(z_pos, npx):
    """Equatorial S(z) profile (paper eq D3)."""
    deep = 36.25 - 1.13 * npx.tanh((z_pos - 305.0) / 460.0)
    shallow = (
        35.55 + 1.25 * (5000.0 - z_pos) / 5000.0
        - 1.62 * npx.tanh((z_pos - 60.0) / 650.0)
        + 0.2 * npx.tanh((z_pos - 35.0) / 100.0)
        + 0.2 * npx.tanh((z_pos - 1000.0) / 5000.0)
    )
    w_deep = (1.0 - npx.tanh((500.0 - z_pos) / 150.0)) / 2.0
    w_shallow = (1.0 - npx.tanh((z_pos - 500.0) / 150.0)) / 2.0
    return deep * w_deep + shallow * w_shallow


def _bathy_taper(coord_deg, lo_deg, hi_deg, npx):
    """Smooth ramp from 0 (within COAST_TAPER_DEG of the boundary) to 1
    (interior). Quintic smooth-step on each side, multiplied together."""
    width = COAST_TAPER_DEG
    s_lo = npx.clip((coord_deg - lo_deg) / width, 0.0, 1.0)
    s_hi = npx.clip((hi_deg - coord_deg) / width, 0.0, 1.0)
    f_lo = 6.0 * s_lo ** 5 - 15.0 * s_lo ** 4 + 10.0 * s_lo ** 3
    f_hi = 6.0 * s_hi ** 5 - 15.0 * s_hi ** 4 + 10.0 * s_hi ** 3
    return f_lo * f_hi


def _build_dino_setup_class(*,
                             nx: int,
                             ny: int,
                             nz: int,
                             dt_mom: float,
                             dt_tracer: float,
                             uniform_z: bool = False):
    """Build the DINOSetup class with the requested grid parameters."""
    from veros import VerosSetup, veros_routine
    from veros.variables import Variable
    from veros.core.operators import numpy as npx, update, at

    class DINOSetup(VerosSetup):
        """Kamm et al. 2025 DINO reference setup."""

        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "dino"
            settings.description = (
                "DINO (Diabatic Neverworld Ocean) reference run, "
                "Kamm et al. 2025 GMD."
            )

            settings.nx, settings.ny, settings.nz = nx, ny, nz
            settings.dt_mom = dt_mom
            settings.dt_tracer = dt_tracer
            settings.runlen = DEFAULT_RUNLEN_S

            settings.x_origin = LON_WEST_DEG
            settings.y_origin = -LAT_MAX_DEG

            settings.coord_degree = True
            # Cyclic-X so the channel band is naturally re-entrant. The
            # closed walls outside the channel are imposed via ``kbot``.
            settings.enable_cyclic_x = True

            # GM/Redi + neutral diffusion to absorb the eddy work that the
            # 1 deg grid cannot resolve (paper Sect 2.3).
            settings.enable_neutral_diffusion = True
            settings.K_iso_0 = 1000.0
            settings.K_iso_steep = 500.0
            settings.iso_dslope = 0.005
            settings.iso_slopec = 0.01
            settings.enable_skew_diffusion = True
            settings.K_gm_0 = 1000.0

            # Lateral momentum mixing -- harmonic with cos(lat) scaling to
            # mimic the paper's A_h(j) = 0.5 * U_M * dx(j).
            settings.enable_hor_friction = True
            settings.A_h = (2 * settings.degtom) ** 3 * 2e-11
            settings.enable_hor_friction_cos_scaling = True
            settings.hor_friction_cosPower = 1

            # Bottom drag (paper rn_drag = 1e-3, quadratic, but Veros only
            # has the linear form built in; the magnitude is comparable
            # for the bulk circulation diagnostics we compare).
            settings.enable_bottom_friction = True
            settings.r_bot = 1.0e-5

            settings.enable_implicit_vert_friction = True

            # TKE for vertical mixing (Veros canonical) -- the paper uses
            # TKE in NEMO so this is the closest match.
            settings.enable_tke = True
            settings.c_k = 0.1
            settings.c_eps = 0.7
            settings.alpha_tke = 30.0
            settings.mxl_min = 1e-8
            settings.tke_mxl_choice = 2
            settings.kappaM_min = 2e-4
            settings.kappaH_min = 2e-5
            settings.enable_kappaH_profile = True

            settings.enable_eke = False
            settings.enable_idemix = False

            # Nonlinear EOS to match the paper.
            settings.eq_of_state_type = 3

            # Allow temperature/salinity restoring fields.
            settings.enable_tempsalt_sources = False  # use surface forcing only

            var_meta = state.var_meta
            var_meta.update(
                t_star=Variable(
                    "t_star", ("yt",), "deg C",
                    "Reference surface temperature",
                ),
                s_star=Variable(
                    "s_star", ("yt",), "g/kg",
                    "Reference surface salinity",
                ),
                t_rest=Variable(
                    "t_rest", ("xt", "yt"), "1/s",
                    "Surface temperature restoring rate",
                ),
                s_rest=Variable(
                    "s_rest", ("xt", "yt"), "1/s",
                    "Surface salinity restoring rate",
                ),
            )

        @veros_routine
        def set_grid(self, state):
            vs = state.variables
            dx_deg = (LON_EAST_DEG - LON_WEST_DEG) / nx
            dy_deg = (2.0 * LAT_MAX_DEG) / ny
            vs.dxt = update(vs.dxt, at[...], dx_deg)
            vs.dyt = update(vs.dyt, at[...], dy_deg)

            # Two paths:
            # * ``uniform_z=True`` (used by the cross-model fidelity
            #   comparison): a single uniform thickness ``H_DEEP / nz``
            #   so the Veros and legoESM vertical grids match exactly.
            # * Default: cosh-stretched grid (modest top→bottom thinning
            #   matched to Veros's ACC stretched setup) — preserved for
            #   future production runs even though it is not Madec's
            #   thin-top Lévy stretch.
            if uniform_z:
                dz_uniform = H_DEEP_M / nz
                vs.dzt = update(vs.dzt, at[...], dz_uniform)
            else:
                k = npx.arange(nz, dtype="float64")
                a_cr = 10.5
                k_th = float(nz) - 1.0
                raw = npx.cosh((k - k_th) / a_cr)
                dz = raw / npx.sum(raw) * H_DEEP_M
                vs.dzt = update(vs.dzt, at[...], dz)

        @veros_routine
        def set_coriolis(self, state):
            vs = state.variables
            settings = state.settings
            vs.coriolis_t = update(
                vs.coriolis_t, at[...],
                2.0 * settings.omega
                * npx.sin(vs.yt[npx.newaxis, :] / 180.0 * settings.pi),
            )

        @veros_routine
        def set_topography(self, state):
            vs = state.variables
            x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")

            # Basin bathymetry -- smooth taper near the lat/lon walls.
            taper_x = _bathy_taper(x, LON_WEST_DEG, LON_EAST_DEG, npx)
            taper_y = _bathy_taper(y, -LAT_MAX_DEG, LAT_MAX_DEG, npx)
            depth = H_SHALLOW_M + (H_DEEP_M - H_SHALLOW_M) * taper_x * taper_y

            # Re-entrant channel band: drop the western-wall taper inside
            # the channel so the column is open right at the wall.
            in_channel = (y >= CHANNEL_LAT_SOUTH_DEG) & (
                y <= CHANNEL_LAT_NORTH_DEG
            )
            depth = npx.where(in_channel, H_SHALLOW_M
                              + (H_DEEP_M - H_SHALLOW_M) * taper_y, depth)

            # Translate depth -> kbot (first wet level from below; cells
            # below `depth` are land).
            zw = vs.zw  # interface depths, negative below surface, length nz
            # zw[k] is the bottom of cell k. A column of depth D is wet
            # for levels k where -zw[k-1] <= D, i.e. cells whose top is
            # shallower than D.
            depth_3d = depth[:, :, npx.newaxis]
            zw_3d = -zw[npx.newaxis, npx.newaxis, :]  # positive depth
            wet = (zw_3d <= depth_3d).astype("int32")
            kbot = npx.maximum(nz - wet.sum(axis=2), 0).astype("int32") + 1

            # Cells with depth less than the top cell thickness become land.
            kbot = npx.where(depth < float(vs.dzt[-1]), 0, kbot)
            # Outside the lat band, mark land (the Veros y-axis already
            # spans only [-LAT_MAX, +LAT_MAX], so this is a no-op for the
            # default grid; kept for safety if a caller widens y).
            in_lat_band = (y >= -LAT_MAX_DEG) & (y <= LAT_MAX_DEG)
            kbot = npx.where(in_lat_band, kbot, 0)

            # Close the western wall everywhere except in the channel band
            # by marking column 0 as land outside the channel. The cyclic
            # halo will then see land on one side and ocean on the other
            # everywhere except at channel latitudes, which is the desired
            # partial-periodic topology.
            i_seam = 0
            kbot_col = kbot[i_seam, :]
            kbot_col_blocked = npx.where(in_channel[i_seam, :], kbot_col, 0)
            kbot = update(kbot, at[i_seam, :], kbot_col_blocked)

            vs.kbot = update(vs.kbot, at[...], kbot)

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables

            # Vertical profiles (depth positive downward in the formulas;
            # vs.zt is negative below surface).
            z_pos = -vs.zt                              # (nz,) positive
            T_1d = _T_profile_1d(z_pos, npx)            # (nz,)
            S_1d = _S_profile_1d(z_pos, npx)            # (nz,)
            T_bot = T_1d[0]                              # deepest level
            S_bot = S_1d[0]

            # Meridional taper (paper eq D4-D5 / case 4):
            #   T(lat, z) = (T_1d(z) - T_bot) * (phi_max - |lat|) / phi_max + T_bot
            lat = vs.yt                                  # (ny,)
            factor = (LAT_MAX_DEG - npx.abs(lat)) / LAT_MAX_DEG  # (ny,)
            factor_3d = factor[npx.newaxis, :, npx.newaxis]

            T_field = (
                (T_1d - T_bot)[npx.newaxis, npx.newaxis, :] * factor_3d
                + T_bot
            ) * vs.maskT
            S_field = (
                (S_1d - S_bot)[npx.newaxis, npx.newaxis, :] * factor_3d
                + S_bot
            ) * vs.maskT
            for tau in range(vs.temp.shape[-1]):
                vs.temp = update(vs.temp, at[..., tau], T_field)
                vs.salt = update(vs.salt, at[..., tau], S_field)

            # ---------- Surface forcing fields ----------
            # T*(lat), S*(lat) restoring targets and the corresponding
            # rate constants (1/s).
            vs.t_star = update(vs.t_star, at[...], _T_star(lat, npx))
            vs.s_star = update(vs.s_star, at[...], _S_star(lat, npx))

            tau_T = T_RESTORING_DAYS * 86400.0
            tau_S = S_RESTORING_DAYS * 86400.0
            # rest = dzt[-1] / tau_R -> matches the Veros ACC convention
            # of t_rest = dz_top / tau, applied at the top T-cell.
            vs.t_rest = update(
                vs.t_rest, at[...],
                vs.dzt[-1] / tau_T * vs.maskT[:, :, -1],
            )
            vs.s_rest = update(
                vs.s_rest, at[...],
                vs.dzt[-1] / tau_S * vs.maskT[:, :, -1],
            )

            # Zonal wind stress on the U-grid at the top level.
            taux_1d = _wind_stress(vs.yt, npx)           # (ny,)
            taux_2d = (
                taux_1d[npx.newaxis, :]
                * npx.ones((nx + 4, 1))                 # broadcast through halos
            )
            vs.surface_taux = update(
                vs.surface_taux, at[...],
                taux_2d * vs.maskU[:, :, -1],
            )

            if state.settings.enable_tke:
                vs.forc_tke_surface = update(
                    vs.forc_tke_surface,
                    at[2:-2, 2:-2],
                    npx.sqrt(
                        (
                            0.5 * (vs.surface_taux[2:-2, 2:-2]
                                   + vs.surface_taux[1:-3, 2:-2])
                            / state.settings.rho_0
                        ) ** 2
                        + (
                            0.5 * (vs.surface_tauy[2:-2, 2:-2]
                                   + vs.surface_tauy[2:-2, 1:-3])
                            / state.settings.rho_0
                        ) ** 2,
                    ) ** 1.5,
                )

        @veros_routine
        def set_forcing(self, state):
            vs = state.variables
            # Bryan-Cox style restoring: heat-flux units (K * m / s).
            vs.forc_temp_surface = (
                vs.t_rest * (vs.t_star[npx.newaxis, :] - vs.temp[:, :, -1, vs.tau])
            )
            vs.forc_salt_surface = (
                vs.s_rest * (vs.s_star[npx.newaxis, :] - vs.salt[:, :, -1, vs.tau])
            )

        @veros_routine
        def set_diagnostics(self, state):
            settings = state.settings
            diagnostics = state.diagnostics
            diagnostics["snapshot"].output_frequency = 86400.0 * 10
            diagnostics["averages"].output_variables = (
                "salt", "temp", "u", "v", "w", "rho", "psi",
                "surface_taux", "surface_tauy",
            )
            diagnostics["averages"].output_frequency = DEFAULT_RUNLEN_S
            diagnostics["averages"].sampling_frequency = settings.dt_tracer * 10
            diagnostics["overturning"].output_frequency = (
                DEFAULT_RUNLEN_S / 4.0
            )
            diagnostics["overturning"].sampling_frequency = settings.dt_tracer * 10
            diagnostics["tracer_monitor"].output_frequency = (
                DEFAULT_RUNLEN_S / 4.0
            )

        @veros_routine
        def after_timestep(self, state):
            return

    return DINOSetup


def make_setup(*,
               nx: int = NX_DEFAULT,
               ny: int = NY_DEFAULT,
               nz: int = NZ_DEFAULT,
               dt_mom: float = DT_MOM_DEFAULT,
               dt_tracer: float = DT_TRACER_DEFAULT,
               runlen_s: float = DEFAULT_RUNLEN_S,
               uniform_z: bool = False):
    """Return a fresh DINOSetup instance.

    Defaults reproduce the Kamm et al. 2025 R1 (1 deg) configuration.
    Override ``nx``/``ny``/``nz``/``dt_*`` to coarsen for a smoke test.

    ``uniform_z=True`` switches the vertical grid to ``H_DEEP / nz``
    uniform thickness, which is what the cross-model fidelity comparison
    harness wants so the Veros and legoESM dz axes match bit-for-bit.
    """
    cls = _build_dino_setup_class(
        nx=nx, ny=ny, nz=nz, dt_mom=dt_mom, dt_tracer=dt_tracer,
        uniform_z=uniform_z,
    )
    setup = cls()
    setup._legoesm_target_runlen_s = float(runlen_s)
    return setup
