"""Reconstruct the DifferLand carbon-flux breakdown for one coupled land step.

The coupled multi-layer land step (:func:`step_multilayer_land`) returns only
the net CO2 flux, not the GPP/NPP/respiration/allocation/turnover breakdown.
This helper re-derives that breakdown from the END-OF-STEP soil state — the
model's own end-of-step surface temperature and root-zone ``beta_soil`` — by
re-evaluating the same carbon step with diagnostics on.  It is the single
source of truth used by BOTH the land-carbon equilibrium validator and the
``run_lmip`` semi-analytic spin-up (so the two never re-derive this glue).

Leaf module: it depends on ``multilayer_land`` (beta helper), ``stomata_utils``
(effective-beta / GPP), and ``carbon.carbon_cycle`` (the diagnostic carbon
step); nothing in those imports this module, so there is no import cycle.

The reconstruction is faithful to a few percent — the coupled model computes
GPP inside its surface-energy solve (``surface_out.gpp``, at its own iterated
surface temperature), whereas this re-derives GPP post-hoc via
``compute_effective_beta`` (the SimpleSEB + Farquhar/Ball-Berry stomata path).
It is exact enough for the mean-annual slow-pool fluxes the semi-analytic
spin-up needs (Xia et al. 2012); the model's pools and net CO2 flux remain the
authoritative quantities.  NOTE: for the ``TwoLeafCanopy`` / CLM-ML canopy
surface schemes the model's GPP comes from the canopy solver, so this SimpleSEB
re-derivation is only an approximation there — use it with the SimpleSEB
surface scheme (the ``run_lmip`` / validator default) for the tightest match.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.land.carbon.carbon_cycle import step_carbon_differland
from legoesm.land.multilayer_land import root_zone_beta_soil
from legoesm.land.stomata_utils import compute_effective_beta


def reconstruct_carbon_diagnostics(
    new_state,
    forcing,
    carbon_state,
    config,
    root_frac,
    theta_wp,
    theta_fc,
    beta_min: float,
    lat: jnp.ndarray,
    doy,
    dt: float,
    spatial: bool = False,
    land_params=None,
    soil_frozen_fraction: jnp.ndarray | None = None,
    gpp_override: jnp.ndarray | None = None,
):
    """Return the :class:`CarbonDiagnostics` for one coupled land step.

    Parameters
    ----------
    new_state : MultiLayerLandState
        The END-OF-STEP land state (provides surface T and soil moisture).
    forcing : AtmToSurface
        The atmospheric forcing used for this step.
    carbon_state : CarbonState
        The carbon pools at the START of the step (as the model's carbon step
        consumes them).
    config : MultiLayerLandConfig
        Land config (``.stomata`` / ``.carbon`` sub-configs).
    root_frac, theta_wp, theta_fc, beta_min :
        Root-zone parameters for :func:`root_zone_beta_soil` (same values the
        model used).
    lat, doy, dt :
        Latitude [rad], day-of-year, timestep [s].
    spatial : bool
        Whether ``theta_wp``/``theta_fc`` are per-column arrays.
    land_params : LandSurfaceParams or None
        Per-column surface/physiology parameters.  Threaded to
        :func:`compute_effective_beta` so the reconstructed GPP override sees
        the SAME per-column ``Vc_max25``/``g1``/``LCMA`` the coupled model step
        used (batched-archetype spin-up); ``None`` reproduces the config-scalar
        reconstruction of the single-column ``run_lmip`` / validator path.  The
        in-step carbon config (``config.carbon``) is passed unchanged, matching
        the model's own ``step_carbon`` call (which reads ``config.carbon``, not
        ``land_params.LCMA``).
    soil_frozen_fraction : jnp.ndarray or None
        Per-column annual frozen fraction (perennial-frost index).  MUST match
        the value the coupled step used so the reconstructed SOM decomposition
        losses (``som_active_loss``/``som_slow_loss``/``som_passive_loss``)
        carry the SAME ``f_perma`` protection -- the semi-analytic slow-pool
        reset infers each pool's turnover ``k_X = D_X/C_X`` from these diagnostic
        losses, so an UNPROTECTED loss here would reset the protected spun-up
        SOM back down (and the verification segment would then drift).  ``None``
        (default) -> no protection, matching an unprotected coupled step.
    gpp_override : jnp.ndarray or None
        The GPP the coupled step ACTUALLY fed to its carbon update (the
        surface scheme's ``SurfaceFluxOutput.gpp`` -- two-leaf canopy, CLM-ML
        or SimpleSEB+stomata, with any P-model capacity active).  Pass it
        whenever the caller has the step's surface output; the diagnostics
        then use byte-identical GPP.  ``None`` re-derives GPP through the
        big-leaf :func:`compute_effective_beta`, which is only consistent with
        the coupled step when the surface scheme is SimpleSEB+stomata.
    """
    T_sfc_new = new_state.T_soil[:, 0]
    beta_soil_new, _ = root_zone_beta_soil(
        new_state.theta_soil, root_frac, theta_wp, theta_fc, beta_min,
        spatial=spatial)
    if gpp_override is None:
        _, gpp_override, _ = compute_effective_beta(
            T_sfc_new, forcing, beta_soil_new, config, carbon_state, dt,
            land_params=land_params)
    _, _, diag = step_carbon_differland(
        carbon_state, forcing.sw_down, T_sfc_new, forcing.co2_ppmv,
        beta_soil_new, lat, doy, forcing.precip_total, config.carbon, dt,
        gpp_override=gpp_override, return_diagnostics=True,
        soil_frozen_fraction=soil_frozen_fraction)
    return diag
