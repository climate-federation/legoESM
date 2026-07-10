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
* A single representative growing-season canopy LAI (``archetype_forcing.REF_LAI``) sets
  fAPAR; the archetype-to-archetype SIF contrast is then carried by climate + ``Vc_max25``.
  A per-archetype LAI (from the CLM MONTHLY_LAI climatology) is a documented refinement.
* Well-watered growing season (``archetype_forcing.REF_BETA = 1``); soil-moisture
  down-regulation is a documented refinement (the ArchetypeTable carries ``aridity``).

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


def _archetype_leaf_state(table):
    """Per-archetype coupled-Farquhar leaf state at the representative growing-season
    climate: ``(An, Ci, gamma_star, apar)`` each ``(n_arch,)``.

    Runs the model's OWN :func:`solve_coupled_farquhar_ci` (no re-implemented
    photosynthesis) at the SHARED per-archetype growing-season forcing + PFT ``Vc_max25``
    (:func:`legoesm.land.carbon.archetype_forcing.build_archetype_forcing`, also used by the
    isotope forward -- one forcing definition, no copy-paste); these are the electron-
    transport inputs :func:`leaf_sif` inverts ``je`` from.  Independent of any trained SIF
    parameter, so the trainer precomputes it ONCE (see :func:`build_sif_forward`) and the
    SIF-parameter gradient then flows only through :func:`leaf_sif`.
    """
    from legoesm.land.carbon.archetype_forcing import (
        REF_BETA,
        REF_LAI,
        build_archetype_forcing,
    )
    from legoesm.land.carbon.stomata import StomataConfig, solve_coupled_farquhar_ci

    forcing = build_archetype_forcing(table)
    stomata = StomataConfig(
        enabled=True, stomata_model="ball_berry",
        Vc_max25=forcing.Vc_max25)                     # per-archetype (n_arch,) capacity
    leaf = solve_coupled_farquhar_ci(
        forcing.T_leaf, forcing.sw_down, forcing.co2_ppmv,
        forcing.q_air, forcing.p_surface,
        REF_LAI, REF_BETA, stomata)
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
