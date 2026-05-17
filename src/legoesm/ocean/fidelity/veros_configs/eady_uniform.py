"""Veros eady_uniform setup — classical Eady baroclinic instability.

Mirrors the legoESM ``eady_uniform`` experiment
(:mod:`legoesm.ocean.experiments.eady_uniform`):

* Re-entrant zonal channel (cyclic-X) centred on 25 deg N
* Uniform stratification (linear T(z), constant N)
* Linear vertical shear u = Lambda * (z - z_dm), depth-mean removed
* Gaussian-in-latitude jet envelope centred on ``lat_center``
* Single-wavenumber zonal T perturbation seeded at the surface to excite
  the most-unstable Eady mode
* No surface forcing, no restoring — the instability comes purely from
  the initial baroclinic shear plus boundary T-gradient.

The Veros side uses the linear equation of state (``eq_of_state_type=1``)
to match the legoESM linear-EOS setup. Veros's hard-coded ``betaT`` (1.67e-4
1/K) is used for the thermal-wind / N^2 derivation so the initial T field
gives the requested N exactly under that EOS — we do not try to override
``betaT`` because it lives at module scope in ``veros.core.density.linear_eq``.

References
----------
- Eady (1949), "Long waves and cyclone waves".
- Vallis (2017), "Atmospheric and Oceanic Fluid Dynamics", Ch. 9.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Domain (matches EadyUniformConfig defaults from legoESM)
# ---------------------------------------------------------------------------
LON_WEST_DEG: float = 0.0
LON_EAST_DEG: float = 10.0
LAT_SOUTH_DEG: float = 16.0
LAT_NORTH_DEG: float = 34.0
LAT_CENTER_DEG: float = 25.0

H_MAX_M: float = 5500.0

# Resolution — coarser than legoESM's 10 km production setup so the smoke
# test is tractable on a laptop. The fidelity layer can pass a finer grid
# via ``factory_kwargs``.
NX: int = 30   # 10 deg / 30 = 1/3 deg in lon
NY: int = 30   # 18 deg / 30 = 0.6 deg in lat
NZ: int = 20   # 275 m uniform vertical resolution

# Physics — match legoESM choices
N_BV: float = 1.2e-3            # buoyancy frequency [1/s]
U_SURFACE: float = 0.8          # surface zonal velocity [m/s]
JET_WIDTH_DEG: float = 5.0      # Gaussian half-width [deg]
T_REF_C: float = 10.0           # mean temperature [deg C]
S_REF_PSU: float = 35.0         # uniform salinity [PSU]

T_PERTURBATION_K: float = 0.1
PERTURBATION_WAVENUMBER: int = 3

# Veros linear-EOS coefficients (veros.core.density.linear_eq) — fixed.
_VEROS_BETA_T: float = 1.67e-4  # 1/K
_VEROS_GRAV: float = 9.81       # m/s^2
_OMEGA: float = 7.292115e-5     # rad/s

# Run length — long enough for the unstable mode to e-fold a few times
# (sigma ~ 0.31 * f0 * Lambda / N implies tau ~ 5 d for the default knobs;
# 30 d gives ~6 e-foldings).
DEFAULT_RUNLEN_S: float = 30.0 * 86400.0


def _lambda_shear() -> float:
    """U_surface / H_max — shear rate of the linear u(z) profile [1/s]."""
    return U_SURFACE / H_MAX_M


def _dT_dz() -> float:
    """dT/dz from N^2 under Veros's linear EOS (alpha_T = betaT).

    N^2 = g * alpha_T * dT/dz   =>   dT/dz = N^2 / (g * alpha_T).
    """
    return N_BV * N_BV / (_VEROS_GRAV * _VEROS_BETA_T)


def _dT_dy() -> float:
    """Thermal wind: f0 * Lambda = -g * alpha_T * dT/dy."""
    f0 = 2.0 * _OMEGA * math.sin(math.radians(LAT_CENTER_DEG))
    return -f0 * _lambda_shear() / (_VEROS_GRAV * _VEROS_BETA_T)


def _build_eady_uniform_setup_class():
    """Build the EadyUniformSetup class lazily so this module imports
    even when Veros itself is not installed."""
    from veros import VerosSetup, veros_routine
    from veros.core.operators import numpy as npx, update, at

    class EadyUniformSetup(VerosSetup):
        """Classical Eady baroclinic instability in a re-entrant channel."""

        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "eady_uniform"
            settings.description = (
                "Eady baroclinic instability: uniform N^2, linear shear, "
                "linear EOS, re-entrant channel."
            )

            settings.nx, settings.ny, settings.nz = NX, NY, NZ
            # Conservative timesteps: dx ~ 30 km, U_max ~ 0.8 m/s => CFL OK
            # at dt = 1800 s. Tracer can step at the same rate (we want the
            # baroclinic adjustment to remain coherent on short timescales).
            settings.dt_mom = 1800.0
            settings.dt_tracer = 1800.0
            settings.runlen = DEFAULT_RUNLEN_S

            settings.coord_degree = True
            settings.enable_cyclic_x = True

            settings.x_origin = LON_WEST_DEG
            settings.y_origin = LAT_SOUTH_DEG

            # Linear EOS to match the legoESM Eady setup.
            settings.eq_of_state_type = 1

            # No surface forcing, no restoring (Eady draws energy from
            # the IC shear + boundary T-gradient).
            settings.enable_tempsalt_sources = False

            # Modest harmonic horizontal friction to control grid-scale
            # noise; biharmonic is not standard in Veros so we use the
            # Laplacian form (default).
            settings.enable_hor_friction = True
            settings.A_h = 5.0e3                       # m^2/s
            settings.enable_hor_friction_cos_scaling = False

            settings.enable_bottom_friction = True
            settings.r_bot = 1.0e-5                    # linear bottom drag

            settings.enable_implicit_vert_friction = True

            # Skip the higher-order turbulence closures; Eady is an inviscid
            # linear instability and adding TKE / IDEMIX / GM here just
            # buries the growth-rate signal.
            settings.enable_tke = False
            settings.enable_eke = False
            settings.enable_idemix = False
            settings.enable_neutral_diffusion = False
            settings.enable_skew_diffusion = False

            # Monotone tracer advection keeps the seeded perturbation from
            # ringing — same choice as the lock-exchange and overflow
            # adapters.
            settings.enable_superbee_advection = True

        @veros_routine
        def set_grid(self, state):
            vs = state.variables
            dx_deg = (LON_EAST_DEG - LON_WEST_DEG) / NX
            dy_deg = (LAT_NORTH_DEG - LAT_SOUTH_DEG) / NY
            dz_m = H_MAX_M / NZ

            vs.dxt = update(vs.dxt, at[...], dx_deg)
            vs.dyt = update(vs.dyt, at[...], dy_deg)
            vs.dzt = update(vs.dzt, at[...], dz_m)

        @veros_routine
        def set_coriolis(self, state):
            vs = state.variables
            settings = state.settings
            vs.coriolis_t = update(
                vs.coriolis_t,
                at[...],
                2.0 * settings.omega
                * npx.sin(vs.yt[npx.newaxis, :] / 180.0 * settings.pi),
            )

        @veros_routine
        def set_topography(self, state):
            vs = state.variables
            # Closed N/S walls (Eady channel): mark the outermost two T-rows
            # as land so the meridional velocity vanishes at the walls.
            # Open zonal boundaries are handled via ``enable_cyclic_x``.
            x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")
            in_channel = (y >= LAT_SOUTH_DEG) & (y <= LAT_NORTH_DEG)
            vs.kbot = update(vs.kbot, at[...], in_channel.astype("int"))

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables

            # zt is negative below surface (Veros convention) — use it
            # directly for the linear T(z) and u(z) profiles.
            z_full = vs.zt                                # shape (nz,)

            # Gaussian jet envelope (same array used for T and u).
            env_t = npx.exp(
                -((vs.yt - LAT_CENTER_DEG) / JET_WIDTH_DEG) ** 2
            )                                              # (ny,)

            # ---------- Temperature ----------
            dTdz = _dT_dz()
            dTdy = _dT_dy()

            # Meridional integral of dTdy * envelope using the cell
            # latitudes converted to metres. Use Veros's default Earth
            # radius (``radius=6370e3`` in ``veros/settings.py``) so the
            # numerator of the thermal-wind integral matches Veros's
            # internal metric exactly.
            R_earth = 6.370e6
            y_m = npx.radians(vs.yt - LAT_CENTER_DEG) * R_earth   # (ny,)
            # Trapezoid: T_anom(j) = sum_{i<j} 0.5*(env[i]+env[i+1])
            #                      * (y[i+1]-y[i]) * dTdy
            dy_m = npx.diff(y_m)
            env_mid = 0.5 * (env_t[1:] + env_t[:-1])
            seg = dy_m * env_mid * dTdy
            T_anom = npx.concatenate([npx.zeros(1), npx.cumsum(seg)])  # (ny,)

            T_zk = T_REF_C + dTdz * z_full                            # (nz,)
            T_field = (
                T_zk[npx.newaxis, npx.newaxis, :]
                + T_anom[npx.newaxis, :, npx.newaxis]
            )                                                          # (nx, ny, nz)
            T_field = T_field * vs.maskT

            # Add a single-wavenumber zonal perturbation at the top level
            # to seed the most-unstable mode.
            lon_west_rad = math.radians(LON_WEST_DEG)
            lon_east_rad = math.radians(LON_EAST_DEG)
            lon_frac = (
                (npx.radians(vs.xt) - lon_west_rad)
                / (lon_east_rad - lon_west_rad)
            )                                                          # (nx,)
            zonal = npx.sin(2.0 * math.pi * PERTURBATION_WAVENUMBER * lon_frac)
            pert_xy = (
                T_PERTURBATION_K
                * env_t[npx.newaxis, :]
                * zonal[:, npx.newaxis]
            )                                                          # (nx, ny)
            pert = npx.zeros_like(T_field)
            pert = update(pert, at[:, :, -1], pert_xy * vs.maskT[:, :, -1])
            T_field = T_field + pert

            # ---------- Salinity ----------
            S_field = S_REF_PSU * vs.maskT

            # Broadcast to (nx, ny, nz, n_tau).
            for tau in range(vs.temp.shape[-1]):
                vs.temp = update(vs.temp, at[..., tau], T_field)
                vs.salt = update(vs.salt, at[..., tau], S_field)

            # ---------- Velocity ----------
            # Purely baroclinic u(z) on the jet centre: u = Lambda*(z - z_dm)
            # so the depth-mean vanishes (eta stays 0, no barotropic Kelvin
            # shock at t=0).
            Lambda = _lambda_shear()
            dz = vs.dzt                                  # (nz,)
            H_col = npx.sum(dz)
            U_prof = Lambda * z_full
            U_bar = npx.sum(U_prof * dz) / H_col
            U_baroclinic = U_prof - U_bar                # (nz,)

            # Same Gaussian-in-latitude envelope is used for u as for T
            # (yt and yu both index latitude in Veros's lon-lat C-grid).
            u_field = (
                U_baroclinic[npx.newaxis, npx.newaxis, :]
                * env_t[npx.newaxis, :, npx.newaxis]
            ) * vs.maskU                                  # (nx, ny, nz)
            for tau in range(vs.u.shape[-1]):
                vs.u = update(vs.u, at[..., tau], u_field)

        @veros_routine
        def set_forcing(self, state):
            # No surface forcing — Eady is an IC adjustment problem.
            return

        @veros_routine
        def set_diagnostics(self, state):
            settings = state.settings
            diagnostics = state.diagnostics
            # Snapshot daily so the growth-rate diagnostics have enough
            # temporal resolution to fit a single exponential.
            diagnostics["snapshot"].output_frequency = 86400.0
            diagnostics["averages"].output_variables = (
                "temp", "salt", "u", "v", "w", "rho", "psi",
            )
            diagnostics["averages"].output_frequency = DEFAULT_RUNLEN_S
            diagnostics["averages"].sampling_frequency = settings.dt_tracer * 4

        @veros_routine
        def after_timestep(self, state):
            return

    return EadyUniformSetup


def make_setup(*, runlen_s: float = DEFAULT_RUNLEN_S):
    """Return a fresh EadyUniformSetup instance.

    Veros is imported lazily so this module is safe to import when
    Veros is absent.
    """
    cls = _build_eady_uniform_setup_class()
    setup = cls()
    setup._legoesm_target_runlen_s = float(runlen_s)
    return setup
