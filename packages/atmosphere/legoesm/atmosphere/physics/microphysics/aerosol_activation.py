"""Aerosol → cloud-droplet-number coupling from aerosol optical depth.

The AMIP forcing pipeline carries a column aerosol optical depth (Kinne
climatology + volcanic) that, until this module, only entered the
radiation direct effect.  This module closes the aerosol → microphysics
link with the published global AOT–CCN correlation of Andreae (2009):

    AOT500 = 0.0027 · (N_CCN,0.4%)^0.640        (r² = 0.88)

inverted to diagnose the CCN number concentration at 0.4 %
supersaturation from the column AOD:

    N_CCN [cm⁻³] = (AOT / 0.0027)^(1 / 0.640)

The diagnosed N_CCN is used as the SPECIFIED cloud-droplet number for
double-moment microphysics running in specified-Nc mode (SAM M2005
``dopredictNc=.false.``) and for the radiation effective-radius PSD —
i.e. the first (Twomey, via r_eff) and second (lifetime, via KK2000
autoconversion ``Nc^-1.79``) indirect effects respond to the prescribed
aerosol climatology.  It is a climatological proxy, not a prognostic
activation scheme: no supersaturation balance, no size distribution,
no updraft dependence (Abdul-Razzak & Ghan-style activation needs
aerosol size/composition information the AOD forcing does not carry).

Reference
---------
Andreae, M. O. (2009): Correlation between cloud condensation nuclei
concentration and aerosol optical thickness in remote and polluted
regions, Atmos. Chem. Phys., 9, 543–556, doi:10.5194/acp-9-543-2009.
Fit from Fig. 1 / Sect. 8; observed range spans remote marine
(~tens cm⁻³, AOT ≈ 0.05) to polluted continental (Beijing ~7200 cm⁻³,
AOT ≈ 0.77).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

__physics_contract__ = {
    "summary": (
        "Aerosol -> cloud-droplet (CCN) number concentration from column "
        "aerosol optical depth (Andreae 2009 AOT-CCN relation, inverted)."
    ),
    "inputs": {"aod": "1 (column aerosol optical depth at ~500 nm)"},
    "outputs": {"N_ccn": "1/m^3 (cloud droplet / CCN number concentration)"},
    "sign_convention": (
        "aod >= 0; returned CCN number > 0 and monotonically increasing in aod."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Andreae (2009), ACP 9, 543-556, doi:10.5194/acp-9-543-2009: "
        "AOT500 = 0.0027 * CCN0.4^0.640 (r^2=0.88), inverted"
    ),
    "idealized_test": (
        "AOT=0.075 (paper clean-continental average) -> "
        "~200 cm^-3 (paper Table 2 average 200+-90); monotone in AOD; "
        "floor/cap respected"
    ),
}


__param_spec__ = {
    "CCNFromAODConfig": {
        "scheme_key": "atm.aerosol.CCNFromAODConfig",
        "excluded": {
        },
        "params": {
            # CCN = (AOT / c)^(1/b) inversion of the Andreae (2009) global fit AOT = 0.0027*CCN^0.640
            "aot_coeff": {"units": "1", "bounds": (0.000891, 0.0081), "tunable_tier": 1, "transform": "sigmoid", "category": "ccn_activation", "reference": "Andreae (2009) AOD-CCN power law", "shape": None},
            "aot_exponent": {"units": "1", "bounds": (0.2112, 1.92), "tunable_tier": 2, "transform": "sigmoid", "category": "ccn_activation", "reference": "Andreae (2009) AOD-CCN power law", "shape": None},
            "n_ccn_max_cm3": {"units": "1/cm^3", "bounds": (3300.0, 30000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "ccn_bounds", "reference": "polluted-continental CCN upper clip", "shape": None},
            "n_ccn_min_cm3": {"units": "1/cm^3", "bounds": (3.3, 30.0), "tunable_tier": 3, "transform": "sigmoid", "category": "ccn_bounds", "reference": "pristine-maritime CCN lower clip", "shape": None},
        },
    },
}


class CCNFromAODConfig(NamedTuple):
    """Configuration for the Andreae (2009) AOD → CCN inversion.

    Fields
    ------
    aot_coeff, aot_exponent : float
        The published fit ``AOT = aot_coeff · CCN^aot_exponent``
        (0.0027, 0.640 from Andreae 2009 Fig. 1).  Override only to
        re-fit against a different AOT wavelength/product.
    n_ccn_min_cm3 : float
        Floor on the diagnosed CCN [cm⁻³].  10 cm⁻³ matches the
        cleanest wintertime Southern-Ocean values in the paper ("a few
        tens per cm³") and guards the power law's blow-down at AOD→0.
    n_ccn_max_cm3 : float
        Cap [cm⁻³].  1e4 cm⁻³ sits just above the most polluted study
        average in the paper (Guangdong 9100±4800 cm⁻³) — beyond it the
        fit is unconstrained by data.
    """

    aot_coeff: float = 0.0027
    aot_exponent: float = 0.640
    n_ccn_min_cm3: float = 10.0
    n_ccn_max_cm3: float = 1.0e4


def ccn_from_aod(
    aod: jnp.ndarray,
    config: CCNFromAODConfig = CCNFromAODConfig(),
) -> jnp.ndarray:
    """Diagnose CCN number concentration [1/m³] from column AOD.

    Inverts the Andreae (2009) global power-law fit
    ``AOT = 0.0027 · CCN^0.640`` (CCN at 0.4 % supersaturation in
    cm⁻³), then converts to per-m³.  The result is clipped to the
    observationally constrained range of the fit.

    Parameters
    ----------
    aod : jnp.ndarray
        Column aerosol optical depth (any shape, ≥ 0).
    config : CCNFromAODConfig

    Returns
    -------
    n_ccn : jnp.ndarray
        CCN number concentration [1/m³], same shape as ``aod``.
    """
    # Invert AOT = c · N^b  →  N = (AOT / c)^(1/b).  Clip AOD at the
    # value the floor maps to so the fractional power never sees 0
    # (gradient-safe at aod=0; the floor dominates there anyway).
    aod_floor = config.aot_coeff * config.n_ccn_min_cm3 ** config.aot_exponent
    aod_safe = jnp.maximum(aod, aod_floor)
    n_cm3 = (aod_safe / config.aot_coeff) ** (1.0 / config.aot_exponent)
    n_cm3 = jnp.clip(n_cm3, config.n_ccn_min_cm3, config.n_ccn_max_cm3)
    return n_cm3 * 1.0e6  # cm⁻³ → m⁻³


def specified_nc_field(
    aerosol_od: jnp.ndarray,
    target_shape,
    config: CCNFromAODConfig = CCNFromAODConfig(),
    ccn_aod: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-column SPECIFIED cloud-droplet number [1/m³] from per-layer AOD.

    Column AOD is the sum of the per-layer optical depths the forcing
    pipeline distributed from the Kinne climatology, inverted to a CCN
    number via :func:`ccn_from_aod` (Andreae 2009) and broadcast across all
    levels of each column.  This is the glue used by BOTH the coupled
    (cube / lat-lon) ``physics_pipeline`` AND the combined-physics
    (MPAS / hydrostatic) microphysics + radiation factories so the
    AOD → specified-Nc mapping is written once (no duplicated numerics).

    Parameters
    ----------
    aerosol_od : jnp.ndarray
        Per-layer aerosol optical depth, shape ``(ncol, nlev)`` (column AOD
        is the ``axis=-1`` sum).
    target_shape : tuple of int
        Output shape — the per-column ``(ncol, nlev)`` cloud-droplet field
        the microphysics / radiation kernel expects.  The diagnosed
        per-column value is broadcast to it.
    config : CCNFromAODConfig
    ccn_aod : jnp.ndarray, optional
        Column AOD ``(ncol,)`` to use instead of the ``aerosol_od`` sum (the
        driver's tropospheric visible-band ``forcing["aerosol_ccn_aod"]``).

    Returns
    -------
    n_c : jnp.ndarray
        Specified cloud-droplet number [1/m³] of shape ``target_shape``.
    """
    aod_col = jnp.sum(aerosol_od, axis=-1) if ccn_aod is None else ccn_aod
    n_ccn = ccn_from_aod(aod_col, config)
    return jnp.broadcast_to(n_ccn[..., None], target_shape)
