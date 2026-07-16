"""Collision kernels and terminal velocities for the Super-Droplet Method.

Faithful JAX port of ERF ``Source/Particles/ERF_SuperDropletPCCoalescence.H``
(``CollisionKernel``) and ``Source/MaterialProperties/ERF_TerminalVelocity.H``
(``TerminalVelocity``). Pure, ``vmap``/``jit``-friendly, elementwise functions.

**Collision kernel** ``K(R_i, R_j[, Δv])`` [m³/s] — the rate coefficient in the
stochastic coalescence equation; combined with droplet number density it gives
a collision probability per unit time (see ``coalescence.py``):

* ``golovin``        — analytic test kernel ``K = b(X_i + X_j)``, ``X=(4/3)πR³``
  (Golovin 1963; the canonical SDM validation kernel, Shima 2009).
* ``sedimentation``  — geometric gravitational sweep-out
  ``K = E·π(R_i+R_j)²·|Δv|`` with ``E = ½p²/(1+p)²``, ``p = R_min/R_max``.
* ``long``           — Long (1974) polynomial collision efficiency.
* ``hall``           — Hall (1980) tabulated collision efficiency
  ``E(r_large, r_small/r_large)`` (21×15 table, bilinear interpolation;
  E > 1 at large ratio = wake-capture enhancement, faithful to the table).

The hydrodynamic kernels take ``dv``, the **relative speed** (a non-negative
magnitude ``|v_i − v_j| = √Σ(v_i−v_j)²``). The oracle computes this Euclidean
norm from the two velocity vectors *inside* the kernel; here the caller (the
coalescence driver) computes the norm — including the terminal-velocity
difference along the vertical, exactly as ERF — and passes the resulting scalar
speed. The kernel value depends only on that magnitude, so this is equivalent;
``dv`` must NOT be a per-component velocity vector.

**Terminal velocity** ``v_t(R[, ρ, p, T])`` [m/s] (fall-speed magnitude, > 0):

* ``rogers_yau``       — Stokes regime ``v = k₁R²`` (Rogers & Yau 1989).
* ``atlas_ulbrich``    — rain power law ``v = a·Dᵇ`` (Atlas & Ulbrich 1977).
* ``cloud_rain_shima`` — SCALE-SDM piecewise Stokes/Beard/Bond (Shima 2009),
  ported in CGS exactly as the oracle, returned in SI.

Empirical-fit coefficients (Beard Reynolds polynomials, Pruppacher-Klett
viscosity, Atlas-Ulbrich a/b, etc.) are module-level named constants with
references — they are fixed physical-formula coefficients, not tunables.
Gravity and the freezing point come from :mod:`legoesm.constants`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig

__physics_contract__ = {
    "summary": (
        "Collision kernels (Golovin/sedimentation/Long/Hall) and droplet "
        "terminal velocities (Rogers-Yau/Atlas-Ulbrich/SCALE-SDM) for SDM "
        "coalescence and sedimentation."
    ),
    "inputs": {
        "r_i": "m",
        "r_j": "m",
        "dv": "m/s (relative speed |v_i - v_j|, a non-negative magnitude)",
        "r": "m",
        "rho": "kg/m^3",
        "p": "Pa",
        "T": "K",
    },
    "outputs": {"kernel": "m^3/s", "v_terminal": "m/s"},
    "sign_convention": (
        "Collision kernels are non-negative. Terminal velocity is the positive "
        "fall-speed magnitude (downward); the caller applies the downward sign."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Shima et al. (2009) QJRMS 135:1307; Golovin (1963); Long (1974); Rogers & Yau (1989); Atlas & Ulbrich (1977); SCALE-SDM",
    "idealized_test": (
        "Golovin K(r,r) = b·(8/3)πr³; equal-radius sedimentation kernel = "
        "π(2r)²·(1/8)·dv; every terminal velocity is positive and monotonically "
        "increasing in r; v_RogersYau(10um) ≈ 1.2e-2 m/s."
    ),
}

# (4/3)π — sphere volume prefactor (pure geometry).
_FOUR_THIRDS_PI = 4.0 / 3.0 * jnp.pi

# --- Rogers & Yau (1989) Stokes cloud-droplet fall speed ---
_ROGERS_YAU_K1 = 1.233e8        # [1/(m·s)]  v = k1 R²

# --- Atlas & Ulbrich (1977) rain power law  v = a D^b, D in mm ---
_ATLAS_ULBRICH_A = 3.778
_ATLAS_ULBRICH_B = 0.67

# --- Pruppacher & Klett (1997) dynamic viscosity of air, CGS [g/(cm·s)] ---
# μ = (1.718 + 4.9e-3·Tc [+ -1.2e-5·Tc² for Tc<0]) · 1e-4
_VISC_MU0 = 1.718
_VISC_SLOPE = 4.9e-3
_VISC_CURV = 1.2e-5
_VISC_SCALE = 1.0e-4

# --- SCALE-SDM CloudRainShima CGS base values (mean-free-path scaling) ---
_SD_LB4L = 6.62e-6       # base mean free path [cm]
_SD_VISB4L = 1.818e-4    # base viscosity [g/(cm·s)]
_SD_PB4L = 1013.25       # base pressure [hPa]
_SD_TB4L = 293.15        # base temperature [K]
_SD_SLIP = 2.510         # Cunningham slip coefficient
_SD_D_SMALL_CM = 1.9e-3  # small-cloud upper diameter [cm]
_SD_D_MED_CM = 1.07e-1   # large-cloud/small-rain upper diameter [cm]
# Safety floor on the CGS diameter so the 1/d denominators stay finite for a
# (non-physical) r -> 0 input; far below any real droplet so it never binds
# for r > 0. The final velocity is forced to 0 for r <= 0.
_DIAM_FLOOR_CM = 1.0e-20

# Water surface-tension fit + critical temperature (SCALE-SDM large-rain regime)
_SD_TCRIT = 647.096      # critical temperature of water [K]
_SD_SIG_A = 0.2358
_SD_SIG_B = 1.256
_SD_SIG_C = 0.625
_SD_LOWT_THRESH = 267.5
_SD_LOWT_A = 243.9
_SD_LOWT_B = 35.35
_SD_LOWT_C = 2.854e-3
_SD_LOWT_D = 1.666e-3

# Beard Reynolds-number polynomials (SCALE-SDM vz_b, vz_c)
_VZ_B = (-3.18657, 0.9926960, -1.53193e-3, -0.987059e-3,
         -0.578878e-3, 0.855176e-4, -0.327815e-5)
_VZ_C = (-5.00015, 5.23778, -2.04914, 0.475294, -0.542819e-1, 0.238449e-2)

# --- Hall (1980) collision-efficiency table (SCALE-SDM / ERF transcription) ---
# Stored as NumPy arrays, NOT jnp: a module-top jnp.array allocates a JAX device
# array at import time (forbidden by test_no_module_top_jax_alloc — crashes the
# import chain on the experimental Metal backend).  Each is consumed below via
# ``.astype(r_l.dtype)`` (NumPy accepts the JAX dtype) and device-put lazily when
# it first enters a jnp computation.
# Collector (larger) radii [um]:
_HALL_R0_UM = np.array(
    [6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0,
     100.0, 150.0, 200.0, 300.0])
# Radius ratios r_small/r_large [-]:
_HALL_RAT = np.array(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
# Efficiency E[ratio, r_large] (21 x 15). Values > 1 at large ratio/radius are
# the wake-capture enhancement in the original table.
_HALL_ECOLL = np.array([
    [0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010],
    [0.0030, 0.0030, 0.0030, 0.0040, 0.0050, 0.0050, 0.0050, 0.0100, 0.1000, 0.0500, 0.2000, 0.5000, 0.7700, 0.8700, 0.9700],
    [0.0070, 0.0070, 0.0070, 0.0080, 0.0090, 0.0100, 0.0100, 0.0700, 0.4000, 0.4300, 0.5800, 0.7900, 0.9300, 0.9600, 1.0000],
    [0.0090, 0.0090, 0.0090, 0.0120, 0.0150, 0.0100, 0.0200, 0.2800, 0.6000, 0.6400, 0.7500, 0.9100, 0.9700, 0.9800, 1.0000],
    [0.0140, 0.0140, 0.0140, 0.0150, 0.0160, 0.0300, 0.0600, 0.5000, 0.7000, 0.7700, 0.8400, 0.9500, 0.9700, 1.0000, 1.0000],
    [0.0170, 0.0170, 0.0170, 0.0200, 0.0220, 0.0600, 0.1000, 0.6200, 0.7800, 0.8400, 0.8800, 0.9500, 1.0000, 1.0000, 1.0000],
    [0.0300, 0.0300, 0.0240, 0.0220, 0.0320, 0.0620, 0.2000, 0.6800, 0.8300, 0.8700, 0.9000, 0.9500, 1.0000, 1.0000, 1.0000],
    [0.0250, 0.0250, 0.0250, 0.0360, 0.0430, 0.1300, 0.2700, 0.7400, 0.8600, 0.8900, 0.9200, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0270, 0.0270, 0.0270, 0.0400, 0.0520, 0.2000, 0.4000, 0.7800, 0.8800, 0.9000, 0.9400, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0300, 0.0300, 0.0300, 0.0470, 0.0640, 0.2500, 0.5000, 0.8000, 0.9000, 0.9100, 0.9500, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0400, 0.0400, 0.0330, 0.0370, 0.0680, 0.2400, 0.5500, 0.8000, 0.9000, 0.9100, 0.9500, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0350, 0.0350, 0.0350, 0.0550, 0.0790, 0.2900, 0.5800, 0.8000, 0.9000, 0.9100, 0.9500, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0370, 0.0370, 0.0370, 0.0620, 0.0820, 0.2900, 0.5900, 0.7800, 0.9000, 0.9100, 0.9500, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0370, 0.0370, 0.0370, 0.0600, 0.0800, 0.2900, 0.5800, 0.7700, 0.8900, 0.9100, 0.9500, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0370, 0.0370, 0.0370, 0.0410, 0.0750, 0.2500, 0.5400, 0.7600, 0.8800, 0.9200, 0.9500, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0370, 0.0370, 0.0370, 0.0520, 0.0670, 0.2500, 0.5100, 0.7700, 0.8800, 0.9300, 0.9700, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0370, 0.0370, 0.0370, 0.0470, 0.0570, 0.2500, 0.4900, 0.7700, 0.8900, 0.9500, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0360, 0.0360, 0.0360, 0.0420, 0.0480, 0.2300, 0.4700, 0.7800, 0.9200, 1.0000, 1.0200, 1.0200, 1.0200, 1.0200, 1.0200],
    [0.0400, 0.0400, 0.0350, 0.0330, 0.0400, 0.1120, 0.4500, 0.7900, 1.0100, 1.0300, 1.0400, 1.0400, 1.0400, 1.0400, 1.0400],
    [0.0330, 0.0330, 0.0330, 0.0330, 0.0330, 0.1190, 0.4700, 0.9500, 1.3000, 1.7000, 2.3000, 2.3000, 2.3000, 2.3000, 2.3000],
    [0.0270, 0.0270, 0.0270, 0.0270, 0.0270, 0.1250, 0.5200, 1.4000, 2.3000, 3.0000, 4.0000, 4.0000, 4.0000, 4.0000, 4.0000],
])


# ==========================================================================
# Collision kernels  K(R_i, R_j[, Δv])  [m^3/s]
# ==========================================================================
# Fixed collision-coalescence / kinetic-theory constants (Long 1974, Cunningham
# slip, Beard 1976, Stokes settling).
_LONG_KERNEL_PREFACTOR = 4.5e8
_LONG_R0 = 3.0e-6
_LONG_R0_FLOOR = 3.01e-6
_LONG_RADIUS_THRESHOLD = 5.0e-5
_VISC_SCALE_SI = 1.0e-5
_KINETIC_8 = 8.0
_CUNNINGHAM_A = 1.2570
_CUNNINGHAM_B = 0.40
_CUNNINGHAM_C = 0.550
_BEARD_32 = 1.5
_BEARD_COLLISION_8 = 8.0
_STOKES_18 = 18.0

def golovin_kernel(r_i: jax.Array, r_j: jax.Array, b: float) -> jax.Array:
    """Golovin (1963) additive kernel ``K = b(X_i + X_j)``, ``X = (4/3)πR³``."""
    X_i = _FOUR_THIRDS_PI * r_i**3
    X_j = _FOUR_THIRDS_PI * r_j**3
    return b * (X_i + X_j)


def sedimentation_kernel(r_i: jax.Array, r_j: jax.Array, dv: jax.Array) -> jax.Array:
    """Geometric gravitational-settling kernel ``E·π(R_i+R_j)²·|Δv|``.

    ``E = ½ p² / (1+p)²`` with ``p = R_min/R_max`` (SCALE-SDM ``sedimentation``).
    ``dv`` is the relative *speed* ``|v_i − v_j|`` (a non-negative magnitude the
    caller computes); ``jnp.abs`` is a defensive guard, not a vector reduction.
    """
    r_min = jnp.minimum(r_i, r_j)
    r_max = jnp.maximum(r_i, r_j)
    # AD-safe 0/0 guard (zero-radius pair has zero cross-section -> K = 0):
    # where-before-divide so the VJP never differentiates 1/r at r=0.
    p = safe_divide(r_min, r_max, 1.0e-30)
    E = 0.5 * p * p / ((1.0 + p) * (1.0 + p))
    return jnp.pi * (r_i + r_j) ** 2 * E * jnp.abs(dv)


def long_kernel(r_i: jax.Array, r_j: jax.Array, dv: jax.Array) -> jax.Array:
    """Long (1974) polynomial collision-efficiency kernel.

    For the larger radius ``r_l <= 50 um`` the efficiency follows the cloud
    polynomial ``4.5e8·r_l²·(1 - 3e-6/max(3.01e-6, r_l))``; above that it is 1
    (geometric). ``K = c_rate·π(r_l+r_s)²·|Δv|``. ``dv`` is the relative *speed*
    ``|v_i − v_j|`` (non-negative magnitude the caller computes).
    """
    r_l = jnp.maximum(r_i, r_j)
    r_s = jnp.minimum(r_i, r_j)
    sumr = r_l + r_s
    c_cloud = _LONG_KERNEL_PREFACTOR * (r_l * r_l) * (1.0 - _LONG_R0 / jnp.maximum(_LONG_R0_FLOOR, r_l))
    c_rate = jnp.where(r_l <= _LONG_RADIUS_THRESHOLD, c_cloud, 1.0)
    return c_rate * (jnp.pi * sumr * sumr) * jnp.abs(dv)


def hall_kernel(r_i: jax.Array, r_j: jax.Array, dv: jax.Array) -> jax.Array:
    """Hall (1980) tabulated collision-efficiency kernel [m³/s].

    ``K = E(r_l, r_s/r_l)·π(r_l+r_s)²·|Δv|`` with the efficiency bilinearly
    interpolated from the 21(ratio) × 15(collector-radius) table, exactly as
    the SCALE-SDM/ERF oracle:

    * ``r_l > 300 µm``: 1-D interpolation in ratio at the last radius column,
      **capped at E = 1** (the oracle caps only this branch);
    * ``6 µm < r_l <= 300 µm``: bilinear in (radius, ratio), uncapped — table
      values > 1 (wake capture) are faithful;
    * ``r_l <= 6 µm``: 1-D interpolation in ratio at the first radius column.

    ``dv`` is the relative *speed* (non-negative magnitude the caller computes).
    """
    r_l = jnp.maximum(r_i, r_j)
    r_s = jnp.minimum(r_i, r_j)
    sumr = r_l + r_s
    r_um = r_l * 1.0e6
    # Two zero-radius droplets: zero cross-section -> ratio irrelevant, K = 0.
    # AD-safe where-before-divide (a plain where still NaNs the gradient).
    ratio = safe_divide(r_s, r_l, 1.0e-30)

    # Host tables (module-top np — no import-time device alloc) → device arrays
    # here so the traced index gathers (rat[iqq], ecoll[iqq, irr_b], …) run under
    # jit/vmap. `.astype` on a numpy array stays numpy, which a traced index
    # cannot gather (TracerArrayConversionError under vmap); jnp.asarray fixes it
    # and constant-folds.
    r0 = jnp.asarray(_HALL_R0_UM, dtype=r_l.dtype)
    rat = jnp.asarray(_HALL_RAT, dtype=r_l.dtype)
    ecoll = jnp.asarray(_HALL_ECOLL, dtype=r_l.dtype)

    # Oracle index search: irr = first i with r_um <= r0[i] (15 if beyond);
    # iqq = first i in 1..20 with ratio <= rat[i].
    irr = jnp.searchsorted(r0, r_um, side="left")          # 0..15
    iqq = jnp.clip(jnp.searchsorted(rat, ratio, side="left"), 1, 20)  # coeff-ok: collision-table index bound

    q = (ratio - rat[iqq - 1]) / (rat[iqq] - rat[iqq - 1])

    # Branch: large collector (irr >= 15) — last column, capped at 1.
    e_large = (1.0 - q) * ecoll[iqq - 1, 14] + q * ecoll[iqq, 14]
    e_large = jnp.minimum(e_large, 1.0)

    # Branch: interior (1 <= irr < 15) — bilinear; clip indices for safe gather
    # (the mask below selects which branch's value is used).
    irr_b = jnp.clip(irr, 1, 14)
    p_w = (r_um - r0[irr_b - 1]) / (r0[irr_b] - r0[irr_b - 1])
    e_bilin = ((1.0 - p_w) * (1.0 - q) * ecoll[iqq - 1, irr_b - 1]
               + p_w * (1.0 - q) * ecoll[iqq - 1, irr_b]
               + (1.0 - p_w) * q * ecoll[iqq, irr_b - 1]
               + p_w * q * ecoll[iqq, irr_b])

    # Branch: small collector (irr == 0) — first column.
    e_small = (1.0 - q) * ecoll[iqq - 1, 0] + q * ecoll[iqq, 0]

    E = jnp.where(irr >= 15, e_large, jnp.where(irr >= 1, e_bilin, e_small))
    return E * (jnp.pi * sumr * sumr) * jnp.abs(dv)


def brownian_kernel(
    r_i: jax.Array,
    r_j: jax.Array,
    m_i: jax.Array,
    m_j: jax.Array,
    p: jax.Array,
    T: jax.Array,
) -> jax.Array:
    """Brownian coagulation coefficient K₁₂ [m³/s] (Seinfeld & Pandis).

    Faithful transcription of ERF ``CollisionKernel::Brownian_SeinfeldPandis``
    (Fuchs transition-regime form): per-droplet diffusivity with Cunningham
    slip, thermal speed from the droplet mass, droplet mean free path, and the
    Fuchs ``g`` length correction::

        K12 = 2π·(d₁+d₂)·(D₁+D₂) / [ (d₁+d₂)/((d₁+d₂)+2√(2g₁²+2g₂²))
                                      + 8(D₁+D₂)/((d₁+d₂)·√(c̄₁²+c̄₂²)) ]

    In ERF this is **additive**: when enabled it is added on top of the chosen
    collision kernel (``k_val += k_brown``), exactly as ported in
    ``coalescence_step`` via ``cfg.include_brownian``.

    Note: the air mean-free-path expression is transcribed as the oracle wrote
    it (``λ = 2μ/(p·√(8·M_air/(π·R_d·T)))``, SCALE-SDM lineage); it gives a
    few× the textbook λ_air, affecting only the slip correction of sub-micron
    droplets. Continuum-regime behaviour (slip→1) matches the classical
    ``8k_BT/3μ`` equal-size limit.

    Parameters: radii [m], total droplet masses [kg], pressure [Pa],
    temperature [K]. Elementwise. A pair with any zero radius/mass returns
    K = 0 with a finite gradient (where-substituted dummy inputs on the
    inactive branch — the where-in/where-out AD-safe pattern).
    """
    dtype = jnp.result_type(r_i, r_j)
    # A zero-radius/zero-mass pair has no Brownian cross-section: gate it out
    # with where-substituted PHYSICAL dummies (1 um, its water mass) so every
    # 1/d, 1/m, and the VJP stay finite on the inactive branch, then return 0
    # there (a plain max-floor at a tiny d makes D ~ 1/d explode instead).
    valid = (r_i > 0.0) & (r_j > 0.0) & (m_i > 0.0) & (m_j > 0.0)
    r_dummy = jnp.asarray(1.0e-6, dtype=dtype)
    m_dummy = 4.0 / 3.0 * jnp.pi * constants.rho_water * r_dummy**3
    d1 = 2.0 * jnp.where(valid, r_i, r_dummy)
    d2 = 2.0 * jnp.where(valid, r_j, r_dummy)
    m1 = jnp.where(valid, m_i, m_dummy)
    m2 = jnp.where(valid, m_j, m_dummy)
    kB = constants.k_B

    # Dynamic viscosity [Pa·s] (Pruppacher & Klett; SI twin of _visc_air_cgs).
    Tc = T - constants.T_freeze
    visc = jnp.where(
        Tc >= 0.0,
        (_VISC_MU0 + _VISC_SLOPE * Tc) * _VISC_SCALE_SI,
        (_VISC_MU0 + _VISC_SLOPE * Tc - _VISC_CURV * Tc * Tc) * _VISC_SCALE_SI,
    )

    # Air mean free path [m] — oracle expression (see docstring note).
    M_air_kg = constants.M_air * 1.0e-3  # coeff-ok: g/mol -> kg/mol
    lam_air = (2.0 * visc) / (p * jnp.sqrt(_KINETIC_8 * M_air_kg / (jnp.pi * constants.R_d * T)))

    # Cunningham slip corrections.
    c1 = _CUNNINGHAM_A + _CUNNINGHAM_B * jnp.exp(-_CUNNINGHAM_C * d1 / lam_air)
    c2 = _CUNNINGHAM_A + _CUNNINGHAM_B * jnp.exp(-_CUNNINGHAM_C * d2 / lam_air)
    slip1 = 1.0 + 2.0 * lam_air * c1 / d1
    slip2 = 1.0 + 2.0 * lam_air * c2 / d2

    # Diffusivities [m²/s] and thermal speeds [m/s].
    dcoef = kB * T / (3.0 * jnp.pi * visc)
    D1 = dcoef * slip1 / d1
    D2 = dcoef * slip2 / d2
    vcoef = _KINETIC_8 * kB * T / jnp.pi
    cb1 = jnp.sqrt(vcoef / m1)
    cb2 = jnp.sqrt(vcoef / m2)

    # Droplet mean free paths and Fuchs g length terms.
    lam1 = (_KINETIC_8 / jnp.pi) * D1 / cb1
    lam2 = (_KINETIC_8 / jnp.pi) * D2 / cb2
    g1 = ((d1 + lam1) ** 3
          - jnp.exp(_BEARD_32 * jnp.log(d1 * d1 + lam1 * lam1))) / (3.0 * d1 * lam1) - d1
    g2 = ((d2 + lam2) ** 3
          - jnp.exp(_BEARD_32 * jnp.log(d2 * d2 + lam2 * lam2))) / (3.0 * d2 * lam2) - d2

    sumdia = d1 + d2
    sumd = D1 + D2
    sumc = jnp.sqrt(cb1 * cb1 + cb2 * cb2)
    sumg = jnp.sqrt(2.0 * g1 * g1 + 2.0 * g2 * g2)

    denom = sumdia / (sumdia + 2.0 * sumg) + (_BEARD_COLLISION_8 * sumd) / (sumdia * sumc)
    K12 = 2.0 * jnp.pi * sumdia * sumd / denom
    return jnp.where(valid, K12, 0.0).astype(dtype)


def collision_kernel(
    r_i: jax.Array,
    r_j: jax.Array,
    dv: jax.Array,
    cfg: SDMConfig,
) -> jax.Array:
    """Dispatch to the configured collision kernel [m³/s].

    ``dv`` is the relative-speed magnitude |v_i - v_j| (ignored by Golovin).
    Unknown ``cfg.collision_kernel`` raises (no silent default).
    """
    if cfg.collision_kernel == "golovin":
        return golovin_kernel(r_i, r_j, cfg.golovin_b)
    elif cfg.collision_kernel == "sedimentation":
        return sedimentation_kernel(r_i, r_j, dv)
    elif cfg.collision_kernel == "long":
        return long_kernel(r_i, r_j, dv)
    elif cfg.collision_kernel == "hall":
        return hall_kernel(r_i, r_j, dv)
    else:
        raise ValueError(
            f"Unknown SDM collision_kernel: {cfg.collision_kernel!r} "
            "(expected 'golovin', 'sedimentation', 'long', or 'hall')"
        )


# ==========================================================================
# Terminal velocities  v_t(R[, ρ, p, T])  [m/s]
# ==========================================================================
def _visc_air_cgs(T: jax.Array) -> jax.Array:
    """Dynamic viscosity of air [g/(cm·s)] (Pruppacher & Klett 1997, CGS)."""
    Tc = T - constants.T_freeze
    visc_warm = (_VISC_MU0 + _VISC_SLOPE * Tc) * _VISC_SCALE
    visc_cold = (_VISC_MU0 + _VISC_SLOPE * Tc - _VISC_CURV * Tc * Tc) * _VISC_SCALE
    return jnp.where(Tc >= 0.0, visc_warm, visc_cold)


def terminal_velocity_rogers_yau(r: jax.Array) -> jax.Array:
    """Stokes-regime cloud-droplet terminal velocity ``v = k₁R²`` [m/s]."""
    return _ROGERS_YAU_K1 * r * r


def terminal_velocity_atlas_ulbrich(r: jax.Array) -> jax.Array:
    """Atlas & Ulbrich (1977) rain terminal velocity ``v = a·Dᵇ`` [m/s]."""
    D_mm = 2.0 * r * 1000.0
    # exp(b·ln D) = D^b; floor D so log stays finite for vanishing droplets.
    return _ATLAS_ULBRICH_A * jnp.exp(_ATLAS_ULBRICH_B * jnp.log(jnp.maximum(D_mm, 1.0e-30)))


def _poly(coeffs, x):
    """Horner evaluation of a polynomial with ascending-power ``coeffs``."""
    y = jnp.zeros_like(x) + coeffs[-1]
    for c in coeffs[-2::-1]:
        y = y * x + c
    return y


def terminal_velocity_cloud_rain_shima(
    r: jax.Array, rho: jax.Array, p: jax.Array, T: jax.Array,
) -> jax.Array:
    """SCALE-SDM piecewise cloud/rain terminal velocity [m/s] (Shima 2009).

    Ported in CGS exactly as the oracle (Stokes+slip for small cloud droplets,
    Beard Reynolds-number fit for large cloud/small rain, Bond-number fit for
    large rain), returned in SI. All three regime velocities are computed and
    selected by droplet diameter (data-dependent ``where``).
    """
    # Floor the diameter for every 1/d denominator so r -> 0 stays finite (and
    # grad-safe through the eager jnp.where branches); zeroed at the end for r<=0.
    diameter_cm = jnp.maximum(2.0 * r * 100.0, _DIAM_FLOOR_CM)
    P_hPa = p / 100.0
    rho_mat_cgs = constants.rho_water / 1000.0
    rho_air_cgs = rho / 1000.0
    grav_cgs = constants.g * 100.0
    gxdrow = grav_cgs * (rho_mat_cgs - rho_air_cgs)
    visc = _visc_air_cgs(T)

    sd_l = _SD_LB4L * (visc / _SD_VISB4L) * (_SD_PB4L / P_hPa) * jnp.sqrt(T / _SD_TB4L)
    csc = 1.0 + _SD_SLIP * (sd_l / diameter_cm)

    # regime 1: small cloud droplets (Stokes + slip)
    c1 = gxdrow / (_STOKES_18 * visc)
    v1 = c1 * csc * diameter_cm * diameter_cm

    # regime 2: large cloud droplets / small raindrops (Beard via vz_b)
    c2 = rho_air_cgs * (4.0 * gxdrow) / (3.0 * visc * visc)
    nda = c2 * diameter_cm**3
    nre2 = csc * jnp.exp(_poly(_VZ_B, jnp.log(jnp.maximum(nda, 1.0e-300))))
    v2 = visc * nre2 / (rho_air_cgs * diameter_cm)

    # regime 3: large raindrops (Bond number via vz_c)
    tau = 1.0 - T / _SD_TCRIT
    sigma = _SD_SIG_A * jnp.exp(_SD_SIG_B * jnp.log(jnp.maximum(tau, 1.0e-30))) \
        * (1.0 - _SD_SIG_C * tau)
    tau_lowT = jnp.tanh((T - _SD_LOWT_A) / _SD_LOWT_B)
    sigma_lowT = sigma - _SD_LOWT_C * tau_lowT + _SD_LOWT_D
    sigma = jnp.where(T < _SD_LOWT_THRESH, sigma_lowT, sigma)
    sigma_cgs = sigma * 1.0e3  # N/m -> g/s^2
    c3 = (4.0 * gxdrow) / (3.0 * sigma_cgs)
    bond = c3 * diameter_cm * diameter_cm
    npp = (rho_air_cgs * rho_air_cgs) * (sigma_cgs**3) / (gxdrow * visc**4)
    npp = jnp.exp(jnp.log(jnp.maximum(npp, 1.0e-300)) / 6.0)
    nre3 = npp * jnp.exp(_poly(_VZ_C, jnp.log(jnp.maximum(bond * npp, 1.0e-300))))
    v3 = visc * nre3 / (rho_air_cgs * diameter_cm)

    v_cgs = jnp.where(
        diameter_cm < _SD_D_SMALL_CM, v1,
        jnp.where(diameter_cm < _SD_D_MED_CM, v2, v3),
    )
    # cm/s -> m/s; a non-physical r <= 0 droplet has zero fall speed.
    return jnp.where(r > 0.0, v_cgs / 100.0, 0.0)


def terminal_velocity(
    r: jax.Array,
    rho: jax.Array,
    p: jax.Array,
    T: jax.Array,
    cfg: SDMConfig,
) -> jax.Array:
    """Dispatch to the configured terminal-velocity law [m/s].

    ``rho, p, T`` are used only by ``cloud_rain_shima``. Unknown
    ``cfg.terminal_velocity`` raises (no silent default).
    """
    if cfg.terminal_velocity == "rogers_yau":
        return terminal_velocity_rogers_yau(r)
    elif cfg.terminal_velocity == "atlas_ulbrich":
        return terminal_velocity_atlas_ulbrich(r)
    elif cfg.terminal_velocity == "cloud_rain_shima":
        return terminal_velocity_cloud_rain_shima(r, rho, p, T)
    else:
        raise ValueError(
            f"Unknown SDM terminal_velocity: {cfg.terminal_velocity!r} "
            "(expected 'rogers_yau', 'atlas_ulbrich', or 'cloud_rain_shima')"
        )
