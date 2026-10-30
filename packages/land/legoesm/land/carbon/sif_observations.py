"""Per-archetype OBSERVED solar-induced-fluorescence (SIF) target for Stage-B carbon
calibration.

The MODELLED side (:func:`legoesm.land.carbon.sif_forward.simulate_archetype_sif`)
predicts a per-archetype simulated top-of-canopy SIF; the calibration ``sif`` loss term
compares it against an OBSERVED per-archetype SIF derived from a GRIDDED satellite SIF
product, cover-weighted per archetype (mirroring
:mod:`legoesm.land.carbon.soc_observations`):

* per-cell observed SIF from a gridded NetCDF (a growing-season / annual mean), reshaped
  to the SAME row-major ``(ncell = nlat*nlon)`` order the cover loader uses so cell indices
  align with the archetype membership;
* the per-archetype target = the COVER-WEIGHTED mean of that per-cell SIF over the grid
  cells assigned to the archetype, via the SHARED
  :func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weighted_mean` (one
  cover-weighted-mean definition, no duplicated numerics).

Pure NumPy: a FIXED target (built once, never inside a JAX-traced model step), like
``soc_observations``.

UNITS / real source: the target MUST be in the SAME units as the simulated forward -- the
model's native emitted/observed-top-of-canopy fluorescence PHOTON FLUX [umol m-2 s-1] (see
``sif_forward``), NOT a satellite spectral radiance [mW m-2 nm-1 sr-1].  A real product
(TROPOMI, OCO-2, or GOME-2 gridded SIF -- e.g. the Caltech global SIF products) reports a
narrowband radiance at the retrieval wavelength (740/757 nm); converting it to the model's
broadband photon-flux convention (integrate the fluorescence emission spectrum, apply the
retrieval/escape geometry) AND regridding it onto the surfdata grid is a DATA-PREP
FOLLOW-UP -- a fetcher/regridder, exactly like the ERA5 climatology builder that feeds the
SOC path.  The absolute scale between model and product is partly absorbed by the tier-1
knobs ``max_electron_yield`` / ``fesc`` during calibration.  For a self-contained test
without the real product, :func:`synthetic_observed_sif` fabricates a physically-oriented
per-archetype target in the model's units.
"""

from __future__ import annotations

import numpy as np
from legoesm.land.carbon.soc_observations import per_archetype_cover_weighted_mean

from legoesm import constants

# --- synthetic (test-only) observed-SIF field [umol m-2 s-1]; NOT a real product ---
# Physically-oriented offsets so brighter/warmer archetypes fluoresce more (the
# fluorescence params then have signal to fit); magnitudes are in the model's native
# emitted-SIF photon-flux scale (O(10) umol/m2/s) and deliberately offset from the
# default-parameter simulated SIF so the synthetic dry-run has a clear residual.
_SIF_SYNTH_BASE_UMOL = 4.0        # baseline emitted SIF [umol m-2 s-1]
_SIF_SYNTH_PER_W = 0.04           # per growing-season mean shortwave [umol m-2 s-1 / (W m-2)]
_SIF_SYNTH_PER_K = 0.05           # per degree above freezing [umol m-2 s-1 / K]


def per_archetype_observed_sif(
    sif_cell,
    cell_archetype_id,
    cell_archetype_weight,
    *,
    n_arch=None,
):
    """Cover-weighted per-archetype observed SIF [umol m-2 s-1].

    For archetype ``a`` the target is the cover-weighted mean of the per-cell observed
    SIF of every grid cell assigned to it::

        SIF_obs[a] = ( sum_{(c,p): id[c,p]==a} w[c,p] * sif_cell[c] )
                     / ( sum_{(c,p): id[c,p]==a} w[c,p] )

    SIF is a per-cell quantity (independent of PFT), so a cell's SIF is shared by every
    (cell, PFT) pair that maps to an archetype there.  Delegates to the SHARED
    :func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weighted_mean` (the
    same cover-weighting the observed-SOC target uses -- no duplicated numerics).

    Parameters
    ----------
    sif_cell : array (ncell,)
        Per-cell observed SIF [umol m-2 s-1, model photon-flux units].
    cell_archetype_id : array (ncell, npft) int
        Archetype index per (cell, PFT); ``-1`` where absent / below threshold / bare.
    cell_archetype_weight : array (ncell, npft) float
        PFT cover weight per (cell, PFT); ``0`` where the id is ``-1``.
    n_arch : int, optional
        Number of archetypes.  Defaults to ``cell_archetype_id.max() + 1``.

    Returns
    -------
    array (n_arch,)
        Per-archetype observed SIF [umol m-2 s-1].  ``NaN`` for an archetype with zero
        total assigned cover (no member cells) -- surfaced, never silently 0.
    """
    return per_archetype_cover_weighted_mean(
        sif_cell, cell_archetype_id, cell_archetype_weight, n_arch=n_arch)


# Variable names :func:`soc_observations.load_gridded_obs` auto-detects in a gridded SIF
# NetCDF (``--sif-obs``).  The product must be on the surfdata grid and in the model's
# photon-flux units [umol m-2 s-1]; regridding + the radiance->photon-flux conversion is
# the data-prep follow-up (see the module docstring).
SIF_VAR_CANDIDATES = ("sif", "SIF", "sif_dc", "SIF_740", "sif_740", "sif_ann", "SIF_Corr_740")


def synthetic_observed_sif(table):
    """Deterministic per-archetype observed SIF [umol m-2 s-1, model units] for
    ``--dry-run-synthetic`` (no data files).

    SIF rises with growing-season shortwave and warmth (brighter/warmer archetypes
    photosynthesise + fluoresce more), so the fluorescence parameters have signal to fit.
    NOT a real product -- see ``SIF_VAR_CANDIDATES`` for the satellite source + the
    radiance->photon-flux conversion follow-up.
    """
    sw = np.asarray(table.sw_mean_w, dtype=float)
    mat_c = np.asarray(table.mat_k, dtype=float) - constants.T_freeze
    return (_SIF_SYNTH_BASE_UMOL
            + _SIF_SYNTH_PER_W * np.maximum(sw, 0.0)
            + _SIF_SYNTH_PER_K * np.maximum(mat_c, 0.0))
