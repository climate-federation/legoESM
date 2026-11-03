"""ecRad 49r1 cloud optics, IFS-RRTMG configuration (JAX port).

Port of ``radiation_cloud_optics.F90:cloud_optics`` (OpenIFS 49r1, 8f6f722) for
the configuration of ``config49r1_rrtmg.nam``: ``use_general_cloud_optics =
false``, liquid ``SOCRATES`` (``radiation_liquid_optics_socrates.F90``), ice
``Fu-IFS`` (``radiation_ice_optics_fu.F90``), ``do_fu_lw_ice_optics_bug =
false``, ``do_lw_cloud_scattering = true``, ``do_sw_delta_scaling_with_gases =
false``, McICA solvers (``is_homogeneous = false``: in-cloud water path is
divided by cloud fraction).  Also ports ``radiation_cloud.F90:
crop_cloud_fraction`` (called just before ``cloud_optics`` in
``radiation_interface.F90:346``).  No other configuration is implemented.

Layout is column first, like the saved radiative-properties file:
inputs ``(..., nlev)`` (``pressure_hl`` ``(..., nlev+1)``), outputs
``(..., nlev, nband)`` with 16 LW and 14 SW RRTMG bands.  Outputs are
in-cloud properties, exactly the arrays ``radiation_save.F90`` writes as
``od/ssa/asymmetry_{lw,sw}_cloud`` (after delta-Eddington scaling).

Smoothing (votes CO2-CO5): with ``sharpness = math.inf`` every switch is the
oracle's hard form.  Finite sharpness: crop gates are sigmoids of
``(x - thr)/thr``; radius clamps are logsumexp min/max of ``x/bound`` (width
1 % of the bound); the Fu asymmetry cap uses width 0.1 % (CO2); the
``wp > 0`` presence branches act on a C1 positive part of the mixing ratio
with width 1 % of the crop threshold (CO4); the smooth crop weight multiplies
the outputs and the cropped fraction while in-cloud paths divide by the raw
fraction (CO3).  Division guards are where-before-divide only (CO5); the
smooth form treats phase water paths and ratio denominators below
sqrt(finfo.tiny) as zero so reverse-mode AD cannot underflow (CO8).
Known limit (vote CO9): Fu ice optics have no lower radius bound in the
oracle, so ``re_ice <= 0`` with ice present gives inf/NaN (oracle too, in
cloudy cells); the smooth form also evaluates crop-failed cells (w > 0 in the
sigmoid tail), so there it is NaN where the hard form returns 0.
"""

import functools
import json
import math
import os
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.smooth import positive_part_c1, sigmoid_switch, smooth_max, smooth_min

__physics_contract__ = {
    "summary": (
        "ecRad 49r1 (IFS) cloud optics: SOCRATES liquid and Fu-IFS ice optical "
        "depth, single-scattering albedo and asymmetry factor per RRTMG band "
        "(16 LW, 14 SW), delta-Eddington scaled (liquid SW, ice SW and LW), "
        "combined by scattering-weighted averaging; plus ecRad's cloud-fraction "
        "crop."
    ),
    "inputs": {
        "pressure_hl": "Pa (half-level pressure, index 0 = model top)",
        "cloud_fraction": "1 (layer cloud fraction)",
        "q_liq": "kg/kg (grid-mean liquid mixing ratio)",
        "q_ice": "kg/kg (grid-mean ice mixing ratio)",
        "re_liq": "m (liquid effective radius)",
        "re_ice": "m (ice effective radius)",
    },
    "outputs": {
        "od_lw, od_sw": "1 (in-cloud layer optical depth per band)",
        "ssa_lw, ssa_sw": "1 (single-scattering albedo)",
        "g_lw, g_sw": "1 (asymmetry factor, delta-scaled)",
        "cloud_fraction": "1 (cropped cloud fraction)",
    },
    "sign_convention": (
        "od >= 0, 0 <= ssa <= 1, g < 1; zero where the cropped cloud "
        "fraction is zero. Optical properties only (no budget)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "OpenIFS 49r1 (8f6f722) ifs-source/radiation/radiation_cloud_optics.F90, "
        "radiation_liquid_optics_socrates.F90, radiation_ice_optics_fu.F90, "
        "radiation_delta_eddington.h, radiation_cloud.F90 crop_cloud_fraction; "
        "Edwards & Slingo (1996), Fu (1996), Fu et al. (1998)."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_cloud_optics_oracle.py: matches the "
        "running 49r1 ecRad saved radiative properties to 1e-12 (fp64)."
    ),
}

# --- configuration (config49r1_rrtmg.nam; radiation_config.F90:185-187) ---
_CLOUD_FRACTION_THRESHOLD = 1.0e-6          # namelist cloud_fraction_threshold
_CLOUD_MIXING_RATIO_THRESHOLD = 1.0e-9      # radiation_config.F90:187 default

# --- ecRad constants (radiation_constants.F90:26; vote CO7) ---
_ACCEL_DUE_TO_GRAVITY = constants.g_ecrad   # AccelDueToGravity (radiation_constants.F90:26)

# --- SOCRATES radius bounds (radiation_liquid_optics_socrates.F90:31-32) ---
# Default-real (single-precision) literals in the oracle, promoted to jprb.
_SOCRATES_MIN_RE = float(np.float32(1.2e-6))
_SOCRATES_MAX_RE = float(np.float32(50.0e-6))

# --- Fu-IFS limits and conversions (radiation_ice_optics_fu.F90:29-30,71-73) ---
_FU_MAX_G = 1.0 - 10.0 * float(np.finfo(np.float64).eps)   # MaxAsymmetryFactor
_FU_MAX_RE = 100.0e-6                       # MaxEffectiveRadius, m
_FU_RE_TO_DE_UM = 1.0e6 / 0.64952           # re (m) -> Fu effective diameter (um)
_KG_TO_G = 1000.0                           # iwp kg m-2 -> g m-2

# --- smoothing widths relative to sharpness (votes CO2, CO4) ---
_G_CAP_SHARPNESS_FACTOR = 10.0              # Fu g cap width 0.1 % (CO2 b)

_DATA = os.path.join(os.path.dirname(__file__), "data")
TABLE_FILE = os.path.join(_DATA, "cloud_optics_48r1.npz")
PROVENANCE_FILE = os.path.join(_DATA, "cloud_optics_48r1.json")


class CloudOptics(NamedTuple):
    od_lw: jnp.ndarray
    ssa_lw: jnp.ndarray
    g_lw: jnp.ndarray
    od_sw: jnp.ndarray
    ssa_sw: jnp.ndarray
    g_sw: jnp.ndarray
    cloud_fraction: jnp.ndarray


@functools.lru_cache(maxsize=1)
def load_coefficients():
    """{"liq.coeff_lw", "liq.coeff_sw", "ice.coeff_lw", "ice.coeff_sw"}: fp64 (band, coeff)."""
    if not os.path.exists(TABLE_FILE):
        raise FileNotFoundError(f"cloud optics table file missing: {TABLE_FILE}")
    with np.load(TABLE_FILE) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def provenance():
    with open(PROVENANCE_FILE) as f:
        return json.load(f)


def _hard(sharpness):
    return sharpness == math.inf


def _smin(x, bound, sharpness):
    """min(x, bound); smooth width bound/sharpness."""
    if _hard(sharpness):
        return jnp.minimum(x, bound)
    return bound * smooth_min(x / bound, 1.0, sharpness)


def _smax(x, bound, sharpness):
    """max(bound, x); smooth width bound/sharpness."""
    if _hard(sharpness):
        return jnp.maximum(bound, x)
    return bound * smooth_max(x / bound, 1.0, sharpness)


def _gate(x, threshold, sharpness):
    """1 where x >= threshold (oracle keeps the cloud), else 0."""
    return sigmoid_switch((x - threshold) / threshold, sharpness)


def crop_cloud_fraction(cloud_fraction, q_liq, q_ice, sharpness=100.0):
    """radiation_cloud.F90:609-649; returns (cropped fraction, crop weight)."""
    sum_mixing_ratio = (0.0 + q_liq) + q_ice
    w = (_gate(cloud_fraction, _CLOUD_FRACTION_THRESHOLD, sharpness)
         * _gate(sum_mixing_ratio, _CLOUD_MIXING_RATIO_THRESHOLD, sharpness))
    return w * cloud_fraction, w


def _delta_eddington_scat_od(od, scat_od, g):
    """radiation_delta_eddington.h: delta_eddington_scat_od."""
    f = g * g
    od = od - scat_od * f
    scat_od = scat_od * (1.0 - f)
    g = g / (1.0 + g)
    return od, scat_od, g


def _liq_socrates(coeff, lwp, re_in, sharpness):
    """radiation_liquid_optics_socrates.F90:calc_liq_optics_socrates."""
    re = _smax(_smin(re_in, _SOCRATES_MAX_RE, sharpness), _SOCRATES_MIN_RE, sharpness)
    re = re[..., None]
    lwp = lwp[..., None]
    c = [coeff[:, i] for i in range(coeff.shape[1])]
    od = lwp * (c[0] + re * (c[1] + re * c[2])) \
        / (1.0 + re * (c[3] + re * (c[4] + re * c[5])))
    scat_od = od * (1.0 - (c[6] + re * (c[7] + re * c[8]))
                    / (1.0 + re * (c[9] + re * c[10])))
    g = (c[11] + re * (c[12] + re * c[13])) / (1.0 + re * (c[14] + re * c[15]))
    return od, scat_od, g


def _fu_de_um(re, sharpness):
    de_um = _smin(re, _FU_MAX_RE, sharpness) * _FU_RE_TO_DE_UM
    return de_um[..., None], (1.0 / de_um)[..., None]


def _g_cap(g, sharpness):
    return _smin(g, _FU_MAX_G, _G_CAP_SHARPNESS_FACTOR * sharpness)


def _ice_fu_sw(coeff, ice_wp, re, sharpness):
    """radiation_ice_optics_fu.F90:calc_ice_optics_fu_sw."""
    de_um, inv_de_um = _fu_de_um(re, sharpness)
    iwp_gm_2 = (ice_wp * _KG_TO_G)[..., None]
    c = [coeff[:, i] for i in range(coeff.shape[1])]
    od = iwp_gm_2 * (c[0] + c[1] * inv_de_um)
    scat_od = od * (1.0 - (c[2] + de_um * (c[3] + de_um * (c[4] + de_um * c[5]))))
    g = _g_cap(c[6] + de_um * (c[7] + de_um * (c[8] + de_um * c[9])), sharpness)
    return od, scat_od, g


def _ice_fu_lw(coeff, ice_wp, re, sharpness):
    """radiation_ice_optics_fu.F90:calc_ice_optics_fu_lw."""
    de_um, inv_de_um = _fu_de_um(re, sharpness)
    iwp_gm_2 = (ice_wp * _KG_TO_G)[..., None]
    c = [coeff[:, i] for i in range(coeff.shape[1])]
    od = iwp_gm_2 * (c[0] + inv_de_um * (c[1] + inv_de_um * c[2]))
    scat_od = od - iwp_gm_2 * inv_de_um * (c[3] + de_um * (c[4] + de_um * (c[5] + de_um * c[6])))
    g = _g_cap(c[7] + de_um * (c[8] + de_um * (c[9] + de_um * c[10])), sharpness)
    return od, scat_od, g


def _where_present(present, props):
    p = present[..., None]
    return tuple(jnp.where(p, x, 0.0) for x in props)


def _tiny_floor(x, sharpness):
    """0 in the hard form; sqrt(finfo.tiny) when smoothed (vote CO8: keeps den**2 normal in AD)."""
    return 0.0 if _hard(sharpness) else float(np.sqrt(np.finfo(np.dtype(x.dtype)).tiny))


def _safe_ratio(num, den, ok):
    """num/den where ok, else 0 (where-before-divide; vote CO5)."""
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def cloud_optics(pressure_hl, cloud_fraction, q_liq, q_ice, re_liq, re_ice,
                 sharpness=100.0, coeffs=None):
    """In-cloud optical properties per band; see module docstring.

    ``sharpness = math.inf`` is the exact 49r1 hard form (oracle test);
    production uses the default 100.
    """
    co = load_coefficients() if coeffs is None else coeffs
    pressure_hl = jnp.asarray(pressure_hl)
    fraction_cropped, w = crop_cloud_fraction(cloud_fraction, q_liq, q_ice, sharpness)

    # In-cloud water path (cloud_optics, is_homogeneous = false), divided by
    # the raw fraction (CO3; equal to the cropped one wherever the oracle uses it).
    floor = _tiny_floor(pressure_hl, sharpness)
    ok_f = cloud_fraction > floor   # CO8 floor on this denominator too
    f_safe = jnp.where(ok_f, cloud_fraction, 1.0)
    factor = (pressure_hl[..., 1:] - pressure_hl[..., :-1]) / (_ACCEL_DUE_TO_GRAVITY * f_safe)
    q_thr_sharp = sharpness / _CLOUD_MIXING_RATIO_THRESHOLD
    lwp = factor * positive_part_c1(q_liq, q_thr_sharp)
    iwp = factor * positive_part_c1(q_ice, q_thr_sharp)

    # "if (lwp_in_cloud > 0)" / "if (iwp_in_cloud > 0)": formulas on safe
    # inputs, zero otherwise (where-before-compute).
    cloud = w > 0.0                 # oracle skips cropped cells entirely
    liq = (lwp > floor) & cloud
    ice = (iwp > floor) & cloud
    lwp_s = jnp.where(liq, lwp, 0.0)
    iwp_s = jnp.where(ice, iwp, 0.0)
    re_liq_s = jnp.where(liq, re_liq, _SOCRATES_MAX_RE)
    re_ice_s = jnp.where(ice, re_ice, _FU_MAX_RE)

    od_lw_l, sc_lw_l, g_lw_l = _where_present(
        liq, _liq_socrates(co["liq.coeff_lw"], lwp_s, re_liq_s, sharpness))
    od_sw_l, sc_sw_l, g_sw_l = _where_present(
        liq, _delta_eddington_scat_od(*_liq_socrates(co["liq.coeff_sw"], lwp_s, re_liq_s, sharpness)))
    od_sw_i, sc_sw_i, g_sw_i = _where_present(
        ice, _delta_eddington_scat_od(*_ice_fu_sw(co["ice.coeff_sw"], iwp_s, re_ice_s, sharpness)))
    od_lw_i, sc_lw_i, g_lw_i = _where_present(
        ice, _delta_eddington_scat_od(*_ice_fu_lw(co["ice.coeff_lw"], iwp_s, re_ice_s, sharpness)))

    # Combine liquid and ice (do_lw_cloud_scattering = true).
    # Oracle guards only the LW g division (sum(scat) > 0); the other three
    # divide unguarded, so the hard form guards them only against den == 0
    # (cloud-free cells, which the crop weight then zeroes).
    nonzero = (lambda d: d != 0.0) if floor == 0.0 else (lambda d: jnp.abs(d) > floor)
    od_lw = od_lw_l + od_lw_i
    sc_lw = sc_lw_l + sc_lw_i
    g_lw = _safe_ratio(g_lw_l * sc_lw_l + g_lw_i * sc_lw_i, sc_lw, sc_lw > floor)
    ssa_lw = _safe_ratio(sc_lw, od_lw, nonzero(od_lw))
    od_sw = od_sw_l + od_sw_i
    sc_sw = sc_sw_l + sc_sw_i
    g_sw = _safe_ratio(g_sw_l * sc_sw_l + g_sw_i * sc_sw_i, sc_sw, nonzero(sc_sw))
    ssa_sw = _safe_ratio(sc_sw, od_sw, nonzero(od_sw))

    # Cloud-present gate (fraction > 0 after the crop): weight w (CO3).
    wb = w[..., None]
    return CloudOptics(od_lw * wb, ssa_lw * wb, g_lw * wb,
                       od_sw * wb, ssa_sw * wb, g_sw * wb, fraction_cropped)
