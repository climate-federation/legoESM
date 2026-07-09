"""Per-archetype SIMULATED top-of-canopy SIF forward for Stage-B carbon calibration.

The MODELLED counterpart of :mod:`legoesm.land.carbon.sif_observations` (the OBSERVED
side): a per-archetype simulated solar-induced-fluorescence (SIF) value, differentiable
in the :class:`legoesm.land.canopy.sif.SIFConfig` fluorescence parameters.

Unlike the SOM-SOC forward
(:func:`legoesm.land.carbon.global_init.equilibrate_archetypes_traced`), SIF is a
**single-step** photosynthesis diagnostic -- there is NO multi-year spin-up scan.
It reuses the model's OWN coupled-Farquhar solve
(:func:`legoesm.land.carbon.stomata.solve_coupled_farquhar_ci`) to obtain the electron-
transport inputs (net assimilation ``An``, intercellular CO2 ``Ci``, CO2 compensation
point ``Gamma*``, absorbed PAR) and the BEPS-SIF / van der Tol (2014) leaf fluorescence
kernel (:func:`legoesm.land.canopy.sif.leaf_sif`) -- exactly the SIF path in
:func:`legoesm.land.stomata_utils.compute_effective_beta` -- so the ``je``/APAR fed to
``leaf_sif`` are the model's own, never re-derived here.

For each archetype the canopy is evaluated at a REPRESENTATIVE growing-season climate,
built from the :class:`~legoesm.land.carbon.global_init.ArchetypeTable` climate features
(``mat_k``, ``t_seasonal_amp_k``, ``sw_mean_w``) with the SAME model builder
(:func:`legoesm.land.climate_forcing.make_climatological_forcing`), so the reference CO2 /
humidity / pressure come from the model's own forcing convention (not new constants).
The archetype's PFT ``Vc_max25`` comes from the SAME CLM5 physiology table
(:func:`legoesm.land.surface_params.clm5_pft_table`) that
:func:`~legoesm.land.carbon.global_init.iter_archetype_batches` uses.

Assumptions (documented, refinements deferred like the SIF-obs fetcher):

* Leaf temperature ~ the representative near-surface air temperature (growing-season
  peak day, local solar noon); a full leaf-energy balance is out of scope for this
  archetype diagnostic.
* A single representative growing-season canopy LAI (``_SIF_REF_LAI``) sets fAPAR; the
  archetype-to-archetype SIF contrast is then carried by climate + ``Vc_max25``.  A
  per-archetype LAI (from the CLM MONTHLY_LAI climatology) is a documented refinement.
* Well-watered growing season (``_SIF_REF_BETA = 1``); soil-moisture down-regulation is
  a documented refinement (the ArchetypeTable carries ``aridity``).

UNITS: the returned SIF is the model's NATIVE emitted/observed-top-of-canopy fluorescence
PHOTON FLUX [umol m-2 s-1] (the :func:`leaf_sif` unit), NOT a satellite spectral radiance
[mW m-2 nm-1 sr-1].  The observed target (:mod:`sif_observations`) MUST be provided in the
SAME photon-flux units; the absolute scale between the two is partly absorbed by the
tier-1 calibration knobs ``max_electron_yield`` and the escape probability ``fesc`` during
training (see :func:`legoesm.land.canopy.sif.multilayer_canopy_sif`).  Converting a
satellite radiance product to this photon-flux convention is a data-prep follow-up
(documented in :func:`sif_observations.load_gridded_sif`).

SIF >= 0 and increases with absorbed PAR (``leaf_sif`` returns 0 when APAR = 0); all
functions are pure JAX and differentiable in the SIFConfig fluorescence params.
"""

from __future__ import annotations

import numpy as np

# --- representative growing-season sampling point (NH-phased, like the model forcing) ---
# Peak day of the annual temperature/insolation cycle used by
# ``climate_forcing.make_climatological_forcing`` (NH summer); local solar noon for the
# peak-insolation daytime SIF.  Sampling parameters (WHEN/HOW the representative canopy is
# probed), not empirical coefficients.
_SIF_REF_DOY = 200.0        # day-of-year of the representative growing-season sample [-]
_SIF_REF_HOUR = 12.0        # local solar hour of the representative sample [h]

# --- representative canopy structure / water status for the standalone diagnostic ---
# A canonical growing-season canopy LAI (fAPAR = 1 - exp(-k_ext*LAI)); the per-archetype
# LAI climatology is a documented refinement.  Well-watered growing season (no soil-
# moisture stress) -> beta = 1.  Reference values, not tuned coefficients.
_SIF_REF_LAI = 3.0          # representative growing-season canopy LAI [m2/m2]
_SIF_REF_BETA = 1.0         # well-watered growing-season soil-moisture factor [-]


def _archetype_leaf_state(table):
    """Per-archetype coupled-Farquhar leaf state at the representative growing-season
    climate: ``(An, Ci, gamma_star, apar)`` each ``(n_arch,)``.

    Runs the model's OWN :func:`solve_coupled_farquhar_ci` (no re-implemented
    photosynthesis) at the ArchetypeTable climate + PFT ``Vc_max25``; these are the
    electron-transport inputs :func:`leaf_sif` inverts ``je`` from.  Independent of any
    trained SIF parameter, so the trainer precomputes it ONCE (see
    :func:`build_sif_forward`) and the SIF-parameter gradient then flows only through
    :func:`leaf_sif`.
    """
    import jax
    import jax.numpy as jnp
    from legoesm.land.carbon.stomata import StomataConfig, solve_coupled_farquhar_ci
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

    # Representative growing-season forcing per archetype: vmap the single-column model
    # builder over the archetype axis, then drop the trailing length-1 column axis
    # (matches ``iter_archetype_batches._build_forcing_fn``).
    forcing = jax.vmap(
        make_climatological_forcing, in_axes=(0, 0, 0, 0, None, None),
    )(mat_k, t_seas, sw_mean, precip, _SIF_REF_DOY, _SIF_REF_HOUR)
    forcing = jax.tree_util.tree_map(
        lambda x: jnp.squeeze(x, axis=1) if x.ndim >= 2 else x, forcing)

    # Per-archetype PFT physiology (Vc_max25) via the SAME CLM5 table + column order
    # ``iter_archetype_batches`` uses -- no re-derived physiology.
    rows = clm5_pft_table()[pft_id]                    # (n_arch, 12)
    land_params = array_to_params(rows, PARAM_NAMES)
    stomata = StomataConfig(
        enabled=True, stomata_model="ball_berry",
        Vc_max25=land_params.Vc_max25)                 # per-archetype (n_arch,) capacity

    leaf = solve_coupled_farquhar_ci(
        forcing.T_lowest, forcing.sw_down, forcing.co2_ppmv,
        forcing.q_lowest, forcing.p_surface,
        _SIF_REF_LAI, _SIF_REF_BETA, stomata)
    return leaf.A_net, leaf.Ci, leaf.gamma_star, leaf.APAR_umol


def _apply_leaf_sif(An, Ci, gamma_star, apar, sif_config):
    """Observed top-of-canopy SIF ``leaf_sif(An,Ci,Gamma*,APAR) * fesc`` [umol m-2 s-1].

    Mirrors the SIF aggregation in
    :func:`legoesm.land.stomata_utils.compute_effective_beta`: the big-leaf
    :func:`leaf_sif` emits the leaf/canopy fluorescence and the escape probability
    ``fesc`` (clamped to ``[0, 1]`` -- a probability) converts emitted to observed
    top-of-canopy.  Differentiable in ``sif_config``.
    """
    import jax.numpy as jnp
    from legoesm.land.canopy.sif import leaf_sif

    fesc = jnp.clip(sif_config.escape_probability, 0.0, 1.0)
    return leaf_sif(An, Ci, gamma_star, apar, sif_config) * fesc


def simulate_archetype_sif(table, sif_config):
    """Per-archetype simulated top-of-canopy SIF ``(n_arch,)`` [umol m-2 s-1].

    All-in-one forward: solve the model's coupled Farquhar at each archetype's
    representative growing-season climate (:func:`_archetype_leaf_state`) then apply the
    BEPS-SIF leaf fluorescence kernel (:func:`_apply_leaf_sif`).  Differentiable in
    ``sif_config`` (the Farquhar solve does not depend on the SIF params, so the
    SIF-parameter gradient flows through :func:`leaf_sif` only -- a simple single-step
    AD graph, no spin-up scan).  Used directly by the unit test / AD de-risk; the trainer
    uses :func:`build_sif_forward` to avoid re-solving the (static) Farquhar each step.

    Parameters
    ----------
    table : ArchetypeTable
        Stage-A archetypes (static; the SIF forward differentiates only the SIF params).
    sif_config : SIFConfig
        Fluorescence configuration (its tier-1/2 params may be TRACED overrides spliced
        via ``apply_param_overrides`` inside a loss; production defaults are untouched).
    """
    An, Ci, gamma_star, apar = _archetype_leaf_state(table)
    return _apply_leaf_sif(An, Ci, gamma_star, apar, sif_config)


def build_sif_forward(table):
    """Precompute the static per-archetype leaf state ONCE and return
    ``sif_fn(sif_config) -> (n_arch,)`` SIF [umol m-2 s-1].

    The coupled-Farquhar solve (:func:`_archetype_leaf_state`) is independent of every
    trained SIF parameter, so it is evaluated ONCE here (concrete arrays) and the returned
    closure applies only the differentiable :func:`leaf_sif` kernel.  This makes the
    SIF-parameter gradient a pure single-step ``leaf_sif`` graph (fast + trivial to
    compile) and avoids re-solving Farquhar on every optimizer step -- the trainer's
    ``sif`` loss term calls this.  Numerically identical to
    :func:`simulate_archetype_sif` for any ``sif_config``.
    """
    An, Ci, gamma_star, apar = _archetype_leaf_state(table)

    def sif_fn(sif_config):
        return _apply_leaf_sif(An, Ci, gamma_star, apar, sif_config)

    return sif_fn
