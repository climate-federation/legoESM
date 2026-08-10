"""Stevens et al. (2005) simple interactive longwave for stratocumulus.

The DYCOMS-II RF01 intercomparison replaced a full longwave solver with a
two-exponential fit to the net upward LW flux plus a clear-sky correction
above the inversion. gSAM implements it as ``SRC/rad_simple.f90``
(``doradsimple``), and every deck in ``data/les_cases`` that sets
``dolongwave = .true., doradsimple = .true.`` -- DYCOMS_RF01 and ASTEX209 --
is driven by exactly that routine with exactly these constants.

This module is the ONE implementation of that parameterization. It previously
lived inline in ``scripts/run/run_dycoms_les.py`` as ``make_stevens_lw``,
which meant the LES had cloud-top radiative cooling and the single-column
model had none: the SCM-vs-LES turbulence tuner had to REFUSE both
stratocumulus cases, because scoring an SCM with no cloud-top cooling against
an LES that has it compares two different problems. Both sides now call the
same kernel.

Physics, in the oracle's own terms (``rad_simple.f90`` lines 29-99)::

    dq(k)   = kappa * rho(k) * q_cond(k) * dz(k)      optical depth increment
    Q_up(k) = sum of dq ABOVE face k                  (gSAM ``qzinf``)
    Q_dn(k) = sum of dq BELOW face k                  (gSAM ``qzeroz``)
    F(k)    = F0*exp(-Q_up) + F1*exp(-Q_dn) + F_cs(k)
    dT/dt   = -(F(k+1) - F(k)) / (cp * rho * dz)

``F`` is the NET UPWARD longwave flux [W/m^2] and ``z`` is positive up, so a
flux that INCREASES upward is a cooling: at cloud top the ``F0`` term emerges
from under the cloud's optical depth over a few tens of metres, which is the
strong cloud-top radiative cooling this scheme exists to produce. The clear-sky
correction ``F_cs`` applies only ABOVE the inversion, where it roughly balances
the prescribed subsidence warming.

Only CLOUD condensate enters the optical depth. Rain does not (gSAM line 54
sums ``qcl + qci``, never ``qpl``), and RF01 is non-drizzling in any case.

FAITHFULNESS -- three places this repo differs from ``rad_simple.f90``, all
deliberate, none silent:

1. ``cp``. gSAM hardcodes ``cp_spec`` "from DycomsII LES intercomparison
   specs" (line 24), which sits about 1 % above this repo's
   ``constants.c_pd``. ``SimpleLWConfig.cp_j_kg_k`` defaults to ``c_pd`` --
   the value the existing DYCOMS LES reference was generated with -- and the
   spec value is selectable as ``DYCOMS_SPEC_CP_J_KG_K``. It scales the
   heating rate directly.
2. Density in the clear-sky term. gSAM uses the LOCAL face density
   ``rhow(k)``, and says so (lines 86-90: "our flux differs from the
   specification in that it uses the local density ... Formulating things in
   this way ensures a rough balance between the subsidence heating and
   radiative cooling above the inversion"). The specification, and this
   module's default, use the density AT THE INVERSION.
   ``density_at_inversion=False`` selects gSAM's form.
3. Cloud ice. gSAM's optical depth is ``qcl + qci``; the caller here passes
   whatever it calls cloud condensate. Both stratocumulus decks are warm, so
   the two agree for every case currently driven by this module.

Constants are case-INDEPENDENT in the oracle: ``rad_simple`` hardcodes
``coef``, ``coef1``, ``f0`` and ``xk`` and applies them to every deck that
selects it, so ASTEX209 legitimately runs the RF01 numbers. They are exposed
as configuration because they are the scheme's empirical fit, not because any
shipped case overrides them.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants

__all__ = [
    "SimpleLWConfig",
    "simple_lw_net_upward_flux",
    "simple_lw_temperature_tendency",
    "simple_lw_inversion_height",
    "DYCOMS_SPEC_CP_J_KG_K",
]

__physics_contract__ = {
    "summary": (
        "Stevens et al. (2005) DYCOMS-II RF01 'simple interactive radiation': "
        "net upward longwave as two exponentials in cloud-condensate optical "
        "depth plus a clear-sky correction above the inversion, differenced "
        "to a temperature tendency. Port of gSAM SRC/rad_simple.f90; "
        "longwave only, no shortwave."
    ),
    "inputs": {
        "q_cond": "kg/kg (cloud condensate; rain excluded)",
        "q_total": "kg/kg (vapour + cloud condensate, for the inversion)",
        "rho": "kg/m^3", "dz": "m", "z_half": "m (heights, surface = 0)",
    },
    "outputs": {
        "net_upward_flux": "W/m^2 (positive UP, at half levels)",
        "temperature_tendency": "K/s (at full levels)",
    },
    "signs": (
        "z is positive UP and the flux is NET UPWARD, so dT/dt = "
        "-(1/(rho cp)) dF/dz: a flux increasing with height is a COOLING. "
        "Cloud-top cooling is therefore negative dT/dt just below the "
        "inversion, which is the scheme's entire purpose."
    ),
    "conserves": (
        "Nothing by construction -- this is a prescribed-flux parameterization, "
        "not a solver. The column-integrated heating equals the net flux "
        "difference across the column over cp, exactly, because the tendency "
        "is a telescoping flux difference."
    ),
    "differentiable": (
        "Yes in the empirical coefficients and in the state through the "
        "optical depth. The inversion index is an argmax-like integer "
        "selection and is NOT differentiable; it is a threshold crossing, and "
        "the oracle defines it that way."
    ),
    "reference": (
        "Stevens et al. (2005), Mon. Wea. Rev. 133, 1443-1462 (DYCOMS-II RF01 "
        "intercomparison); gSAM 1.8.8 SRC/rad_simple.f90."
    ),
    "idealized_test": (
        "tests/atmosphere/physics/test_simple_lw.py -- a clear column reduces "
        "to F0+F1 with zero heating; an optically thick cloud produces "
        "cloud-top cooling and cloud-base warming; the column integral of the "
        "heating equals the flux difference across the column."
    ),
}

__param_spec__ = {
    "SimpleLWConfig": {
        "scheme_key": "atm.rad.SimpleLWConfig",
        "fields": {
            "f0_w_m2": {
                "units": "W/m^2", "bounds": (0.0, 200.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "longwave",
                "reference": "Stevens et al. 2005 RF01; gSAM rad_simple coef",
                "shape": None,
            },
            "f1_w_m2": {
                "units": "W/m^2", "bounds": (0.0, 100.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "longwave",
                "reference": "Stevens et al. 2005 RF01; gSAM rad_simple coef1",
                "shape": None,
            },
            "kappa_m2_kg": {
                "units": "m^2/kg", "bounds": (10.0, 200.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "longwave",
                "reference": "Stevens et al. 2005 RF01; gSAM rad_simple xk",
                "shape": None,
            },
            "divergence_s": {
                "units": "1/s", "bounds": (0.0, 2.0e-5), "tunable_tier": 2,
                "transform": "sigmoid", "category": "longwave",
                "reference": "Stevens et al. 2005 RF01 large-scale divergence D",
                "shape": None,
            },
            "qt_inversion_kg_kg": {
                "units": "kg/kg", "bounds": (1.0e-3, 2.0e-2),
                "tunable_tier": 0, "transform": "none", "category": "longwave",
                "reference": (
                    "RF01 spec inversion definition; a measurement convention "
                    "that selects a level, not a calibratable closure."
                ),
                "shape": None,
            },
            "cp_j_kg_k": {
                "units": "J/kg/K", "bounds": (900.0, 1100.0),
                "tunable_tier": 0, "transform": "none", "category": "longwave",
                "reference": (
                    "thermodynamic constant, not a closure: constants.c_pd "
                    "here, the DycomsII intercomparison value in the spec."
                ),
                "shape": None,
            },
        },
    },
}

# --- Stevens et al. (2005) RF01 fit, as hardcoded in gSAM rad_simple.f90 ---
# coef, coef1, f0 and xk on lines 29-32; the 8 g/kg inversion isoline on line 57.
_F0_W_M2 = 70.0             # cloud-top flux jump [W/m^2]      (gSAM coef)
_F1_W_M2 = 22.0             # cloud-base flux jump [W/m^2]     (gSAM coef1)
_KAPPA_M2_KG = 85.0         # LW mass absorption [m^2/kg]      (gSAM xk)
_DIVERGENCE_S = 3.75e-6     # large-scale divergence D [1/s]   (gSAM f0)
_QT_INVERSION_KG_KG = 8.0e-3   # q_t isoline defining z_i [kg/kg]
# The DycomsII intercomparison's own c_p (gSAM ``cp_spec``), kept for callers
# that want to match the spec rather than this repo's thermodynamic constant.
DYCOMS_SPEC_CP_J_KG_K = 1015.0   # const-ok: DycomsII intercomparison spec c_p,
# deliberately NOT constants.c_pd -- see the module docstring's Faithfulness
# point 1. Selecting it is how a caller reproduces gSAM's heating rate.


class SimpleLWConfig(NamedTuple):
    """Coefficients of the Stevens (2005) simple longwave.

    Defaults are the oracle's hardcoded values, which it applies to every deck
    that selects ``doradsimple`` regardless of case.
    """
    f0_w_m2: float = _F0_W_M2
    f1_w_m2: float = _F1_W_M2
    kappa_m2_kg: float = _KAPPA_M2_KG
    divergence_s: float = _DIVERGENCE_S
    qt_inversion_kg_kg: float = _QT_INVERSION_KG_KG
    cp_j_kg_k: float = constants.c_pd
    # True  = density at the inversion (the RF01 SPECIFICATION, and the value
    #         the existing DYCOMS LES reference was generated with).
    # False = local layer density (what gSAM actually does; see the module
    #         docstring, Faithfulness point 2).
    density_at_inversion: bool = True


def simple_lw_inversion_height(q_total, z_half, config: SimpleLWConfig):
    """Height of the face above the highest cell with ``q_t >= threshold``.

    gSAM (line 57) tracks ``itop = max(itop, k+1)`` over cells exceeding the
    isoline and uses ``zi(itop)`` -- the face ABOVE the highest such cell. A
    column that never exceeds the threshold keeps ``itop = 1``, i.e. z = 0, and
    the clear-sky term then applies its weak divergence from the surface.

    Arrays are ordered BOTTOM-UP along the last axis; ``z_half`` has one more
    entry than ``q_total``. Returns a trailing singleton axis so the result
    broadcasts against half-level arrays.
    """
    idx = jnp.arange(q_total.shape[-1])
    below = q_total >= config.qt_inversion_kg_kg
    k_top = jnp.max(jnp.where(below, idx, -1), axis=-1)
    return jnp.take(z_half, k_top + 1, axis=-1)[..., None]


def simple_lw_net_upward_flux(q_cond, q_total, rho, dz, z_half,
                              config: SimpleLWConfig = SimpleLWConfig()):
    """Net UPWARD longwave flux at half levels [W/m^2], positive up.

    Parameters
    ----------
    q_cond : array (..., nlev)
        Cloud condensate mixing ratio [kg/kg]. Rain is excluded by the
        oracle and must be excluded by the caller.
    q_total : array (..., nlev)
        Vapour + cloud condensate [kg/kg], for the inversion isoline.
    rho, dz : array (nlev,)
        Layer density [kg/m^3] and thickness [m]. These are reference-state
        profiles in the LES and column profiles in the SCM; both are
        one-dimensional in the vertical and broadcast against ``q_cond``.
    z_half : array (nlev+1,)
        Half-level heights [m], surface = 0.

    All profile axes are ordered BOTTOM-UP (index 0 nearest the surface),
    which is the orientation the cumulative optical depths are defined in.
    """
    # Optical-depth increment per layer, cloud condensate only (gSAM line 54).
    dq = config.kappa_m2_kg * rho * q_cond * dz
    # Cumulative optical depth at the TOP face of each layer, padded with the
    # surface face, where nothing lies below.
    pad = [(0, 0)] * (dq.ndim - 1) + [(1, 0)]
    q_below = jnp.pad(jnp.cumsum(dq, axis=-1), pad)
    q_above = q_below[..., -1:] - q_below

    z_i = simple_lw_inversion_height(q_total, z_half, config)
    # Zero below the inversion, so the clear-sky term switches itself off
    # there exactly as gSAM's `k = itop+1, nzm` loop bound does.
    dz_i = jnp.clip(z_half - z_i, 0.0, None)
    if config.density_at_inversion:
        z_full = 0.5 * (z_half[:-1] + z_half[1:])
        rho_ref = jnp.interp(z_i[..., 0], z_full, rho)[..., None]
    else:
        # gSAM's local density, held at the layer below each face; the top
        # face reuses the topmost layer (gSAM's flux(nz) uses rhow(nz)).
        rho_ref = jnp.pad(rho, (0, 1), mode="edge")
    clear_sky = (rho_ref * config.cp_j_kg_k * config.divergence_s
                 * (0.25 * dz_i ** (4.0 / 3.0) + z_i * dz_i ** (1.0 / 3.0)))
    return (config.f0_w_m2 * jnp.exp(-q_above)
            + config.f1_w_m2 * jnp.exp(-q_below)
            + clear_sky)


def simple_lw_temperature_tendency(q_cond, q_total, rho, dz, z_half,
                                   config: SimpleLWConfig = SimpleLWConfig()):
    """Radiative TEMPERATURE tendency [K/s] at full levels, bottom-up.

    ``dT/dt = -(1/(rho cp)) dF/dz`` with ``F`` the net UPWARD flux and ``z``
    positive up, so a flux increasing with height cools -- the cloud-top
    cooling this scheme exists for.

    This is a TEMPERATURE tendency, not a potential-temperature one. A caller
    that prognoses theta must divide by the Exner function itself; doing that
    here would silently apply an Exner factor to the single-column model,
    which prognoses T.
    """
    flux = simple_lw_net_upward_flux(q_cond, q_total, rho, dz, z_half, config)
    d_flux = flux[..., 1:] - flux[..., :-1]
    return -d_flux / (rho * config.cp_j_kg_k * dz)
