"""Shared per-archetype REPRESENTATIVE growing-season leaf forcing for Stage-B calibration.

The single-step photosynthesis forwards used by the Stage-B carbon calibration
(:mod:`legoesm.land.carbon.sif_forward` -- SIF; :mod:`legoesm.land.carbon.d13c_forward` --
leaf carbon-isotope discrimination) both need the SAME per-archetype canopy forcing: the
model's OWN single-column growing-season climate + the archetype's PFT photosynthetic
capacity, evaluated at a representative daytime sampling point.  This module owns that
ONE forcing-construction (no copy-paste across the two forwards) and returns the concrete
per-archetype arrays :func:`legoesm.land.stomata.solve_coupled_farquhar_ci`
consumes.

For each archetype the canopy is evaluated at a REPRESENTATIVE growing-season climate,
built from the :class:`~legoesm.land.carbon.global_init.ArchetypeTable` climate features
(``mat_k``, ``t_seasonal_amp_k``, ``sw_mean_w``) with the model builder
(:func:`legoesm.land.climate_forcing.make_climatological_forcing`), so the reference CO2 /
humidity / pressure come from the model's own forcing convention (not new constants).  The
archetype's PFT ``Vc_max25`` comes from the SAME CLM5 physiology table
(:func:`legoesm.land.surface_params.clm5_pft_table`) that
:func:`~legoesm.land.carbon.global_init.iter_archetype_batches` uses.

Assumptions (documented; refinements deferred like the SIF/biomass-obs fetchers):

* Leaf temperature ~ the representative near-surface air temperature (growing-season
  peak day, local solar noon); a full leaf-energy balance is out of scope for this
  archetype diagnostic.
* A single representative growing-season canopy LAI (:data:`REF_LAI`) sets fAPAR; the
  archetype-to-archetype contrast is then carried by climate + ``Vc_max25``.  A
  per-archetype LAI (from the CLM MONTHLY_LAI climatology) is a documented refinement.
* Well-watered growing season (:data:`REF_BETA` = 1); soil-moisture down-regulation is a
  documented refinement (the ArchetypeTable carries ``aridity``).

Pure JAX; the returned forcing is independent of any TRAINED parameter, so a forward may
build it ONCE and re-solve the coupled Farquhar system per optimizer step (SIF freezes the
whole leaf state; the isotope forward re-solves for the trained-stomata-dependent ``Ci``).
"""

from __future__ import annotations

from typing import NamedTuple

# --- representative growing-season sampling point (NH-phased, like the model forcing) ---
# Peak day of the annual temperature/insolation cycle used by
# ``climate_forcing.make_climatological_forcing`` (NH summer); local solar noon for the
# peak-insolation daytime sample.  Sampling parameters (WHEN/HOW the representative canopy
# is probed), not empirical coefficients.
REF_DOY = 200.0        # day-of-year of the representative growing-season sample [-]
REF_HOUR = 12.0        # local solar hour of the representative sample [h]

# --- representative canopy structure / water status for the standalone diagnostic ---
# A canonical growing-season canopy LAI (fAPAR = 1 - exp(-k_ext*LAI)); the per-archetype
# LAI climatology is a documented refinement.  Well-watered growing season (no soil-
# moisture stress) -> beta = 1.  Reference values, not tuned coefficients.  Exported so the
# SIF and isotope forwards feed the SAME canopy into ``solve_coupled_farquhar_ci``.
REF_LAI = 3.0          # representative growing-season canopy LAI [m2/m2]
REF_BETA = 1.0         # well-watered growing-season soil-moisture factor [-]


class ArchetypeForcing(NamedTuple):
    """Per-archetype representative growing-season leaf forcing (concrete ``(n_arch,)``).

    The exact inputs :func:`legoesm.land.stomata.solve_coupled_farquhar_ci` takes
    (``T_leaf, sw_down, co2_ppmv, q_air, p_surface``) plus the per-archetype PFT
    photosynthetic capacity ``Vc_max25`` [umol/m2/s].  Independent of every trained
    parameter, so a forward precomputes it once.
    """

    T_leaf: object       # representative leaf ~ air temperature [K]
    sw_down: object      # downward shortwave [W/m2]
    co2_ppmv: object     # ambient CO2 Ca [umol/mol]
    q_air: object        # near-surface specific humidity [kg/kg]
    p_surface: object    # surface pressure [Pa]
    Vc_max25: object     # per-archetype PFT max carboxylation at 25 C [umol/m2/s]


def build_archetype_forcing(table) -> ArchetypeForcing:
    """Per-archetype representative growing-season forcing + PFT ``Vc_max25`` (``(n_arch,)``).

    Vmaps the model's single-column forcing builder
    (:func:`legoesm.land.climate_forcing.make_climatological_forcing`) over the archetype
    axis at the representative sampling point (:data:`REF_DOY` / :data:`REF_HOUR`), drops
    the trailing length-1 column axis (matches ``iter_archetype_batches._build_forcing_fn``),
    and reads each archetype's ``Vc_max25`` from the SAME CLM5 table + column order
    (:func:`legoesm.land.surface_params.clm5_pft_table`) the archetype IC builder uses -- no
    re-derived forcing or physiology.  The returned arrays are the model's own, never
    re-derived here, and feed :func:`legoesm.land.stomata.solve_coupled_farquhar_ci`.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from legoesm.land.climate_forcing import make_climatological_forcing
    from legoesm.land.surface_params import (
        PARAM_NAMES,
        array_to_params,
        clm5_pft_table,
    )

    pft_id = jnp.asarray(np.asarray(table.pft_id, dtype=int))
    mat_k = jnp.asarray(np.asarray(table.mat_k, dtype=float))
    t_seas = jnp.asarray(np.asarray(table.t_seasonal_amp_k, dtype=float))
    sw_mean = jnp.asarray(np.asarray(table.sw_mean_w, dtype=float))
    n_arch = int(pft_id.shape[0])
    # Precipitation does not enter the Farquhar inputs (only precip_total/snow), so a
    # zero rate keeps the growing-season forcing minimal without a spurious constant.
    precip = jnp.zeros((n_arch,), dtype=mat_k.dtype)

    forcing = jax.vmap(
        make_climatological_forcing, in_axes=(0, 0, 0, 0, None, None),
    )(mat_k, t_seas, sw_mean, precip, REF_DOY, REF_HOUR)
    forcing = jax.tree_util.tree_map(
        lambda x: jnp.squeeze(x, axis=1) if x.ndim >= 2 else x, forcing)

    # Per-archetype PFT physiology (Vc_max25) via the SAME CLM5 table + column order
    # ``iter_archetype_batches`` uses -- no re-derived physiology.
    rows = clm5_pft_table()[pft_id]                    # (n_arch, 12)
    land_params = array_to_params(rows, PARAM_NAMES)
    return ArchetypeForcing(
        T_leaf=forcing.T_lowest,
        sw_down=forcing.sw_down,
        co2_ppmv=forcing.co2_ppmv,
        q_air=forcing.q_lowest,
        p_surface=forcing.p_surface,
        Vc_max25=land_params.Vc_max25,
    )
