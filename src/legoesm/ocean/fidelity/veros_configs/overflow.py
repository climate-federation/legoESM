"""Veros overflow setup (Petersen et al. 2015 Fig. 7).

Dense cold water sits on a continental shelf and descends a sloping bottom
into a deep abyssal basin. Standard z-coordinate ocean models suffer from
spurious mixing along the slope; the test diagnoses that mixing.

Geometry: 200 km (x) x 4 km (y) x 2000 m depth, 2 km horizontal x 200 m
vertical resolution. A linear slope drops the bottom from 400 m at the
shelf edge to 2000 m over the central 100 km. Cold water (T=5 deg C) sits
above the shelf; the rest of the basin is warm (T=20 deg C).

Built on top of :mod:`legoesm.ocean.fidelity.veros_configs.lock_exchange`
to keep the "minimal physics, linear EOS, no rotation" setup consistent.
"""

from __future__ import annotations

DEFAULT_RUNLEN_S: float = 24.0 * 3600.0  # 24 hours — Petersen 2015 Fig. 7 window

DOMAIN_LX_M: float = 200_000.0   # 200 km
DOMAIN_LY_M: float = 4_000.0     # narrow channel
DOMAIN_LZ_M: float = 2_000.0
NX: int = 100
NY: int = 4
NZ: int = 10

SHELF_DEPTH_M: float = 400.0
ABYSS_DEPTH_M: float = 2_000.0
SHELF_X_END_M: float = 50_000.0     # shelf occupies x < 50 km
SLOPE_X_END_M: float = 150_000.0    # slope from 50 km to 150 km

T_COLD_C: float = 5.0
T_WARM_C: float = 20.0
S_REF_PSU: float = 35.0


def _build_overflow_setup_class():
    from veros import VerosSetup, veros_routine
    from veros.core.operators import numpy as npx, update, at

    class OverflowSetup(VerosSetup):
        """Petersen 2015 Fig. 7 dense-water overflow test."""

        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "overflow"
            settings.description = "Petersen 2015 overflow (shelf -> slope -> abyss)"

            settings.nx, settings.ny, settings.nz = NX, NY, NZ
            settings.dt_mom = 60.0
            settings.dt_tracer = 60.0
            settings.runlen = DEFAULT_RUNLEN_S

            settings.coord_degree = False
            settings.enable_cyclic_x = False

            settings.enable_neutral_diffusion = False
            settings.enable_skew_diffusion = False
            settings.enable_hor_friction = False
            settings.enable_bottom_friction = False
            settings.enable_implicit_vert_friction = False
            settings.enable_explicit_vert_friction = False
            settings.enable_tke = False
            settings.enable_eke = False
            settings.enable_idemix = False
            settings.enable_hor_diffusion = False

            settings.eq_of_state_type = 1
            settings.enable_tempsalt_sources = False
            # Monotone tracer advection (see lock_exchange setup docstring).
            settings.enable_superbee_advection = True

        @veros_routine
        def set_grid(self, state):
            vs = state.variables
            vs.dxt = update(vs.dxt, at[...], DOMAIN_LX_M / NX)
            vs.dyt = update(vs.dyt, at[...], DOMAIN_LY_M / NY)
            vs.dzt = update(vs.dzt, at[...], DOMAIN_LZ_M / NZ)

        @veros_routine
        def set_coriolis(self, state):
            vs = state.variables
            vs.coriolis_t = update(vs.coriolis_t, at[...], 0.0)

        @veros_routine
        def set_topography(self, state):
            vs = state.variables
            # Bathymetry depth(x) ramps from SHELF_DEPTH_M (x<SHELF_X_END)
            # to ABYSS_DEPTH_M (x>SLOPE_X_END_M) linearly.
            x = vs.xt
            depth = npx.where(
                x < SHELF_X_END_M, SHELF_DEPTH_M,
                npx.where(
                    x > SLOPE_X_END_M, ABYSS_DEPTH_M,
                    SHELF_DEPTH_M
                    + (ABYSS_DEPTH_M - SHELF_DEPTH_M)
                    * (x - SHELF_X_END_M) / (SLOPE_X_END_M - SHELF_X_END_M),
                ),
            )
            # Translate to ``kbot``: first wet level (1-indexed). Cells
            # below depth(x) are land.
            dz = DOMAIN_LZ_M / NZ
            n_dry = npx.maximum(
                0,
                (NZ - npx.round(depth / dz)).astype("int32"),
            )
            # kbot is the first wet level from below; for a column with
            # ``n_dry`` dry cells at the top we get the first wet cell at
            # ``n_dry + 1``.
            kbot_2d = (n_dry + 1).astype("int32")
            vs.kbot = update(vs.kbot, at[:], kbot_2d[:, None])

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables

            # Cold above the shelf only; warm everywhere else.
            cold_mask = (vs.xt[:, None, None] < SHELF_X_END_M).astype(vs.temp.dtype)
            init_temp = (cold_mask * T_COLD_C
                         + (1.0 - cold_mask) * T_WARM_C) * vs.maskT
            init_salt = S_REF_PSU * vs.maskT
            for tau in range(vs.temp.shape[-1]):
                vs.temp = update(vs.temp, at[..., tau], init_temp)
                vs.salt = update(vs.salt, at[..., tau], init_salt)

        @veros_routine
        def set_forcing(self, state):
            return

        @veros_routine
        def set_diagnostics(self, state):
            settings = state.settings
            diagnostics = state.diagnostics
            diagnostics["snapshot"].output_frequency = 3600.0
            diagnostics["averages"].output_variables = (
                "temp", "salt", "u", "v", "w", "rho",
            )
            diagnostics["averages"].output_frequency = DEFAULT_RUNLEN_S
            diagnostics["averages"].sampling_frequency = settings.dt_tracer

        @veros_routine
        def after_timestep(self, state):
            return

    return OverflowSetup


def make_setup(*, runlen_s: float = DEFAULT_RUNLEN_S):
    cls = _build_overflow_setup_class()
    setup = cls()
    setup._legoesm_target_runlen_s = float(runlen_s)
    return setup
