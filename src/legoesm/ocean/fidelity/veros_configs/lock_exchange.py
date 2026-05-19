"""Veros lock-exchange setup (Petersen et al. 2015 Fig. 5).

Two-dimensional channel with a vertical density front at the basin centre.
Tracks the descent of dense water and the resulting reference potential
energy (RPE) evolution — the canonical test for spurious numerical mixing
in z-coordinate ocean models.

Geometry: 64 km (x) x 4 km (y) x 20 m depth, 1 km horizontal x 1 m vertical
resolution, no rotation, linear EOS, no surface forcing, free-slip walls.

The setup intentionally turns off every optional physics module
(neutral diffusion, GM, TKE, EKE, IDEMIX, bottom friction) so the only
mixing left is the implicit one from the chosen advection scheme — which
is exactly what Petersen 2015 measures.
"""

from __future__ import annotations

import os

DEFAULT_RUNLEN_S: float = 17.0 * 3600.0  # 17 hours — Petersen 2015 Fig. 5 window

# Petersen 2015 Fig. 5 geometry.
DOMAIN_LX_M: float = 64_000.0    # 64 km
DOMAIN_LY_M: float = 4_000.0     # 4 km (narrow channel; we make ny small)
DOMAIN_LZ_M: float = 20.0        # 20 m total depth
NX: int = 64
NY: int = 4
NZ: int = 20

T_COLD_C: float = 5.0
T_WARM_C: float = 30.0
S_REF_PSU: float = 35.0


def _build_lock_exchange_setup_class():
    """Build the LockExchangeSetup class lazily so this module imports
    even when Veros itself is not installed."""
    from veros import VerosSetup, veros_routine
    from veros.core.operators import numpy as npx, update, at

    class LockExchangeSetup(VerosSetup):
        """Petersen 2015 Fig. 5 lock-exchange test."""

        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "lock_exchange"
            settings.description = "Petersen 2015 lock exchange (2D channel, no rotation)"

            settings.nx, settings.ny, settings.nz = NX, NY, NZ
            # Short tracer/momentum timesteps for the small geometry.
            settings.dt_mom = 30.0
            settings.dt_tracer = 30.0
            settings.runlen = DEFAULT_RUNLEN_S

            settings.coord_degree = False   # Cartesian metres
            settings.enable_cyclic_x = False

            # Turn off optional physics so the only mixing is from advection.
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

            # Linear EOS so the dense-vs-light contrast is exactly
            # proportional to the temperature contrast.
            settings.eq_of_state_type = 1

            # No surface buoyancy fluxes; we drive the flow entirely from
            # the initial T contrast.
            settings.enable_tempsalt_sources = False

            # Veros's default tracer advection is centered, which rings on
            # a sharp front with no explicit diffusion (T blows past the
            # initial bounds by ~50 K on this geometry). The superbee
            # flux-limiter is monotone and is what Petersen 2015 / MOM6
            # use as the baseline scheme for this test.
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
            # kbot = 1 everywhere => full ocean column, no land cells
            vs.kbot = update(vs.kbot, at[...], 1)

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            settings = state.settings

            # x_mid is the basin centre in metres; vs.xt includes 2-cell
            # halos on each side but stores monotone coordinates, so this
            # works without any halo arithmetic.
            x_mid = 0.5 * (float(vs.xt.max()) + float(vs.xt.min()))
            cold_mask = (vs.xt[:, None, None] < x_mid).astype(vs.temp.dtype)
            init_temp = (cold_mask * T_COLD_C
                         + (1.0 - cold_mask) * T_WARM_C) * vs.maskT
            init_salt = S_REF_PSU * vs.maskT
            for tau in range(vs.temp.shape[-1]):
                vs.temp = update(vs.temp, at[..., tau], init_temp)
                vs.salt = update(vs.salt, at[..., tau], init_salt)

        @veros_routine
        def set_forcing(self, state):
            # No wind, no buoyancy, no fresh water flux — purely an IC
            # adjustment problem.
            return

        @veros_routine
        def set_diagnostics(self, state):
            settings = state.settings
            diagnostics = state.diagnostics
            # Snapshot every 30 minutes of model time so the RPE/nose
            # descent post-processor has enough resolution.
            diagnostics["snapshot"].output_frequency = 1800.0
            diagnostics["averages"].output_variables = (
                "temp", "salt", "u", "v", "w", "rho",
            )
            diagnostics["averages"].output_frequency = DEFAULT_RUNLEN_S
            diagnostics["averages"].sampling_frequency = settings.dt_tracer

        @veros_routine
        def after_timestep(self, state):
            return

    return LockExchangeSetup


def make_setup(*, runlen_s: float = DEFAULT_RUNLEN_S):
    """Return a fresh LockExchangeSetup instance.

    Veros is imported lazily so this module is safe to import when
    Veros is absent.
    """
    cls = _build_lock_exchange_setup_class()
    setup = cls()
    setup._legoesm_target_runlen_s = float(runlen_s)
    return setup
