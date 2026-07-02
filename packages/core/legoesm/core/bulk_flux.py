"""Monin-Obukhov bulk air-sea flux algorithms.

Provides stability-dependent transfer coefficients using iterative
Monin-Obukhov similarity theory (MOST). Three schemes:

1. ``constant`` — Fixed neutral transfer coefficients (no iteration)
2. ``coare3`` — COARE 3.0 (Fairall et al. 2003): Charnock + smooth-flow
   roughness, Businger-Dyer stability functions
3. ``large_yeager`` — Large & Yeager 2009 (CORE/OMIP): empirical C_DN(U_10N)
   with high-wind quintic correction, stability-dependent Stanton/Dalton
   numbers

The iterative Obukhov length loop uses ``jax.lax.fori_loop`` for
full JAX differentiability (compatible with jax.grad, jax.jit).

The ``large_yeager`` path is the OMIP-2 protocol bulk formula (Griffies
2016 §2.2 mandates Large & Yeager 2009).

References
----------
- Fairall, C. W., et al. (2003). Bulk parameterization of air-sea fluxes:
  Updates and verification for the COARE algorithm. J. Climate, 16, 571-591.
- Large, W. G., & Yeager, S. G. (2009). The global climatology of an
  interannually varying air-sea flux data set. Climate Dynamics, 33,
  341-364. doi:10.1007/s00382-008-0441-3.
- Businger, J. A., et al. (1971). Flux-profile relationships in the
  atmospheric surface layer. J. Atmos. Sci., 28, 181-189.
"""

from __future__ import annotations

import math
import numbers

import jax
import jax.numpy as jnp

from legoesm import constants

# Physical constants
KAPPA = constants.kappa_vk  # von Kármán constant (0.4)
G = constants.g
NU_AIR = constants.nu_air  # kinematic viscosity of air [m²/s]

# Known bulk-flux scheme names (union across all surface-flux dispatchers):
#   "constant"     — fixed neutral transfer coefficients (no iteration)
#   "most"         — iterative MOST with fixed roughness (constant-z0 stability)
#   "coare3"       — COARE 3.0
#   "large_yeager" — Large & Yeager 2009 (OMIP)
_VALID_BULK_SCHEMES = ("constant", "most", "coare3", "large_yeager")


def apply_gustiness(u: jax.Array, v: jax.Array, gustiness: float) -> jax.Array:
    """Effective surface wind with a sub-grid convective gustiness floor.

    ``|U|_eff = sqrt(u^2 + v^2 + u_gust^2)``.  The resolved grid-mean wind misses
    sub-grid wind variability — boundary-layer convective gustiness — which in
    calm/convective regions (the tropics, where the mean wind is light but deep
    convection drives gusts) is the dominant contributor to the air-sea latent
    and sensible flux.  Omitting it (or using a ~1 m/s numerical floor) yields
    ~1/3 of the realistic surface evaporation at light winds, starving the
    hydrological cycle (measured: coupled hfls ~35 vs ~80 W/m^2 at a 1.9 m/s
    mean surface wind).  Same algebraic form as the numerical wind floor
    ``sqrt(u^2+v^2+U_min^2)`` — a larger, physically-motivated ``u_gust``
    subsumes it.

    The floor lives INSIDE the single ``sqrt`` (not a two-stage
    ``sqrt(sqrt(u^2+v^2)^2 + g^2)``): with ``gustiness > 0`` the argument is
    strictly positive everywhere, so the gradient is finite even at exact calm
    wind ``u = v = 0`` (a bare ``sqrt(u^2+v^2)`` would give a NaN gradient
    there).  AD-safe for the differentiable coupler.

    Parameters
    ----------
    u, v : array
        Resolved grid-mean surface wind components [m/s].  A caller holding only
        the wind SPEED magnitude ``s`` passes ``apply_gustiness(s, 0.0, gust)``
        (``sqrt(s^2 + gust^2)``), which is equally AD-safe.
    gustiness : float
        Gustiness floor ``u_gust`` [m/s] (~5 m/s, Wing 2018 RCEMIP1).

    Returns
    -------
    array
        Effective wind speed for the bulk flux [m/s].

    References
    ----------
    - Beljaars (1995), QJRMS 121, 255-270 — convective gustiness in bulk fluxes.
    - Wing et al. (2018), GMD 11, 793-813 — RCEMIP1 5 m/s gustiness floor.
    """
    return jnp.sqrt(u ** 2 + v ** 2 + gustiness ** 2)


def validate_bulk_scheme(scheme: str) -> None:
    """Raise ``ValueError`` on an unknown bulk-flux scheme name.

    Every surface-flux dispatcher gates on ``bulk_scheme`` with
    ``if scheme in (<MOST schemes>): ... else: <constant>``.  Without this guard
    a *typo'd* scheme silently falls through to the constant-coefficient branch
    and runs the wrong air-sea physics (CLAUDE.md dispatch rule: factories must
    raise on unknown schemes).  ``scheme`` is a static Python string resolved at
    trace time, so this validates at function entry — never inside a traced /
    ``jit`` body.
    """
    if scheme not in _VALID_BULK_SCHEMES:
        raise ValueError(
            f"Unknown bulk_scheme {scheme!r}; expected one of "
            f"{_VALID_BULK_SCHEMES}."
        )


def large_yeager_neutral_cd(wind, *, nemo_parity: bool = False):
    """Large & Yeager (2009) Eq. 6 neutral 10-m drag coefficient ``C_DN``.

    .. math::
        C_{DN} = \\left(\\frac{2.7}{U} + 0.142 + \\frac{U}{13.09}
                  - 3.14807\\times10^{-10}\\,U^{6}\\right)\\times10^{-3}

    Default (``nemo_parity=False``, the MOST-solver convention): clipped to
    ``[0.5e-3, 3.0e-3]``.  The ``-3.14807e-10·U⁶`` high-wind term is LY09's
    correction over LY04 (which over-estimated drag at ``U > 30 m/s``) and is
    REQUIRED by the OMIP-2 protocol (Griffies 2016 §2.2).  ``wind`` is the
    10-m wind speed [m/s]; floored to 0.5 to avoid the ``1/U`` blow-up at calm
    winds.  The canonical drag law shared by the MOST flux solver and the OMIP-2
    air-sea bulk formulas (no per-component re-derivation).

    ``nemo_parity=True`` reproduces NEMO/aerobulk ``cd_n10_ncar`` EXACTLY
    (sbcblk_algo_ncar.F90): the same polynomial, but (a) a constant cyclone
    plateau ``2.34e-3`` for ``U >= 33 m/s`` instead of letting the ``U⁶`` term
    pull the polynomial down, and (b) ONLY the NEMO floor ``Cx_min = 1e-4`` —
    no upper clip, so the calm-wind ``2.7/U`` enhancement (up to ``5.54e-3``
    at the 0.5 m/s floor) is kept.  The two conventions differ only for
    ``U < ~0.97 m/s`` (where the 3.0e-3 clip binds) and ``U > 33 m/s``.
    """
    U = jnp.maximum(jnp.asarray(wind), 0.5)
    C_DN = (
        2.7 / U + 0.142 + U / 13.09 - 3.14807e-10 * U ** 6
    ) * 1e-3
    if nemo_parity:
        C_DN = jnp.where(U >= 33.0, 2.34e-3, C_DN)
        return jnp.maximum(C_DN, 0.1e-3)
    return jnp.clip(C_DN, 0.5e-3, 3.0e-3)


# ============================================================================
# Stability functions (Businger-Dyer)
# ============================================================================

def psi_m(zeta):
    """MOST momentum stability function.

    Unstable (ζ < 0): Businger-Dyer
        ψ_m = 2 ln((1+x)/2) + ln((1+x²)/2) − 2 arctan(x) + π/2
        where x = (1 − 16ζ)^{1/4}
    Stable (ζ > 0): Dyer (1974)
        ψ_m = −5ζ

    Uses safe branching (min/max on inputs) to avoid NaN gradients
    in the inactive branch.
    """
    zeta_c = jnp.clip(zeta, -10.0, 10.0)
    # Safe inputs: each branch only sees valid arguments
    zeta_neg = jnp.minimum(zeta_c, -1e-10)
    zeta_pos = jnp.maximum(zeta_c, 1e-10)

    x = jnp.power(1.0 - 16.0 * zeta_neg, 0.25)
    unstable = (
        2.0 * jnp.log((1.0 + x) / 2.0)
        + jnp.log((1.0 + x ** 2) / 2.0)
        - 2.0 * jnp.arctan(x)
        + jnp.pi / 2.0
    )
    stable = -5.0 * zeta_pos

    return jnp.where(zeta_c < 0.0, unstable, stable)


def psi_h(zeta):
    """MOST heat/moisture stability function.

    Unstable (ζ < 0): Businger-Dyer
        ψ_h = 2 ln((1+y)/2)  where y = (1 − 16ζ)^{1/2}
    Stable (ζ > 0): Dyer (1974)
        ψ_h = −5ζ
    """
    zeta_c = jnp.clip(zeta, -10.0, 10.0)
    zeta_neg = jnp.minimum(zeta_c, -1e-10)
    zeta_pos = jnp.maximum(zeta_c, 1e-10)

    y = jnp.sqrt(1.0 - 16.0 * zeta_neg)
    unstable = 2.0 * jnp.log((1.0 + y) / 2.0)
    stable = -5.0 * zeta_pos

    return jnp.where(zeta_c < 0.0, unstable, stable)


# ============================================================================
# Main MOST flux computation
# ============================================================================

def compute_most_fluxes(
    u_rel,
    v_rel,
    T_atm,
    q_atm,
    T_sfc,
    q_sfc,
    rho,
    z_ref=10.0,
    z_t=None,
    z_q=None,
    z0_init=1e-4,
    scheme="coare3",
    n_iter=5,
    charnock=0.011,
    L_latent=None,
    gustiness_w_zi=0.0,
    gustiness_beta=1.25,
    return_2m=False,
    z_diag=2.0,
    max_exchange_coeff=None,
):
    """Compute stability-dependent bulk fluxes via iterative MOST.

    Uses ``jax.lax.fori_loop`` for the Obukhov length iteration,
    ensuring full JAX differentiability (jax.grad, jax.jit).

    Parameters
    ----------
    u_rel, v_rel : array
        Wind components relative to z_ref [m/s].
    T_atm : array
        Atmospheric temperature at z_t [K].
    q_atm : array
        Atmospheric specific humidity at z_q [kg/kg].
    T_sfc : array
        Surface temperature [K].
    q_sfc : array
        Surface saturation specific humidity [kg/kg].
    rho : array
        Air density at z_t [kg/m³].
    z_ref : float
        Reference height for the wind / momentum [m] (default 10).
    z_t : float or None
        Reference height for atmospheric temperature [m]. Defaults to
        ``z_ref`` (single-height mode). For OMIP / JRA55-do, set to 2.0
        — JRA55-do delivers ``tas`` at 2 m while ``uas, vas`` are at 10 m.
    z_q : float or None
        Reference height for atmospheric specific humidity [m]. Defaults
        to ``z_ref`` (single-height mode). Typically 2.0 for OMIP.
    z0_init : float
        Initial momentum roughness length [m] (default 1e-4).
    scheme : str
        ``"coare3"`` or ``"large_yeager"``.
    n_iter : int
        Number of MOST iterations (default 5).
    charnock : float
        Charnock coefficient (COARE only, default 0.011).
    max_exchange_coeff : float or None
        Optional physical ceiling on the neutral-equivalent bulk transfer
        coefficient ``C = κ²/(denom_m·denom_h)``.  ``None`` (default) leaves
        the MOST log-law denominators floored at the standard ``0.5``
        (``C ≤ κ²/0.25 ≈ 0.64``), i.e. BYTE-IDENTICAL to the prior behaviour
        for every existing (ocean/atmosphere) caller.  When set to ``C_max``,
        each denominator is instead floored at ``max(0.5, κ/√C_max)`` so the
        implied ``C_d``, ``C_h`` and ``C_e`` cannot exceed ``C_max``.  This
        caps ``u*``, ``θ*`` and ``q*`` CONSISTENTLY (the fluxes stay linear in
        the T/q gradients, so a surface-energy-balance solve retains its
        self-limiting feedback) and removes the extreme-instability cold-start
        singularity where the floored ``0.5`` denominator lets a trivial
        ~0.6 g/kg humidity gradient generate a spurious ~4900 W/m² latent
        shock (``C_e≈0.64`` vs a physical ~3.4e-3).  Intended for the land
        surface tile, whose stiff thin top layer at ``dt_rad`` is unstable to
        that shock; ocean/atmosphere callers never approach the floor.

    Returns
    -------
    tau_x, tau_y : array
        Surface stress [Pa] (opposes wind direction).
    shflx : array
        Sensible heat flux [W/m²] (positive upward = surface warmer).
    lhflx : array
        Latent heat flux [W/m²] (positive upward = surface moister).
    ustar : array
        Friction velocity [m/s].
    """
    # Dispatch hardening (CLAUDE.md): ``scheme`` is a static Python string
    # resolved at trace time. Validate it at function entry so a typo'd name
    # (e.g. ``"coar3"``) fails LOUDLY instead of silently falling through the
    # ``else`` branch below to the constant-roughness MOST path and running the
    # wrong air-sea physics. ``coare3``/``large_yeager`` take dedicated
    # branches; ``constant``/``most`` are the (valid) fixed-roughness else path.
    validate_bulk_scheme(scheme)

    # Resolve scalar reference heights. ``z_ref`` is the wind/momentum
    # height (always = z_u in the formulas below); z_t, z_q default to
    # z_ref so that single-height callers (lake, idealized adapter,
    # legacy tests) keep their existing behaviour bit-identically.
    z_u = z_ref
    if z_t is None:
        z_t = z_ref
    if z_q is None:
        z_q = z_ref

    wind_speed = jnp.sqrt(u_rel ** 2 + v_rel ** 2 + 1e-4)
    dT = T_sfc - T_atm
    dq = q_sfc - q_atm
    # Virtual-T moisture coefficient = 1/ε − 1 ≈ 0.6078 (canonical, not 0.61).
    _vT_coef = 1.0 / constants.epsilon - 1.0
    T_v = T_atm * (1.0 + _vT_coef * q_atm)

    # Log-law denominator floor. Default 0.5 (=> C ≤ κ²/0.25 ≈ 0.64, the
    # historical behaviour, byte-identical for every existing caller). When a
    # physical ceiling ``max_exchange_coeff`` is requested, floor each
    # denominator at κ/√C_max so the implied C_d/C_h/C_e ≤ C_max — a static
    # Python float, constant-folded, no retrace (feature-gating, not traced
    # selection). See the ``max_exchange_coeff`` docstring entry.
    if max_exchange_coeff is None:
        _denom_floor = 0.5
        _coeff_cap = None
    else:
        # Static feature-gate value (constant-folded). A traced/array scalar
        # would break the Python ``max``/``if`` below, so require a real Python
        # or numpy scalar and reject it loudly rather than silently mis-tracing.
        # ``bool`` is a numbers.Real subclass — exclude it explicitly. Then
        # guard finite > 0: C_max = 0 gives an infinite floor (silent zero
        # fluxes) and C_max < 0 gives ``sqrt`` of a negative (NaN).
        if isinstance(max_exchange_coeff, bool) or not isinstance(
            max_exchange_coeff, numbers.Real
        ):
            raise TypeError(
                "max_exchange_coeff must be a static real scalar or None, "
                f"got {type(max_exchange_coeff).__name__}"
            )
        max_exchange_coeff = float(max_exchange_coeff)
        if not (math.isfinite(max_exchange_coeff) and max_exchange_coeff > 0.0):
            raise ValueError(
                "max_exchange_coeff must be finite and > 0, "
                f"got {max_exchange_coeff}"
            )
        _denom_floor = max(0.5, KAPPA / (max_exchange_coeff ** 0.5))
        # Coefficient-space equivalent ceiling for the large_yeager branch,
        # which forms rd/rh/re directly (u*=rd·U, θ*=rh·dT, q*=re·dq) and never
        # touches the log-law denominators.  The log-law path caps the
        # "rd-equivalent" κ/denom at κ/_denom_floor, so bounding rd, rh, re at
        # the SAME value gives C_d=rd² , C_h=rd·rh , C_e=rd·re ≤ C_max
        # identically.  = min(√C_max, 0.8) so it never tightens below the
        # historical 0.5-floor behaviour.
        _coeff_cap = KAPPA / _denom_floor

    # Initialize with neutral log-law profile
    z0 = jnp.full_like(wind_speed, z0_init)
    z0_t = z0 * 0.1
    z0_q = z0_t

    ln_zu_z0 = jnp.log(z_u / jnp.maximum(z0, 1e-12))
    ln_zt_z0t = jnp.log(z_t / jnp.maximum(z0_t, 1e-12))
    ln_zq_z0q = jnp.log(z_q / jnp.maximum(z0_q, 1e-12))
    u_star = KAPPA * wind_speed / jnp.maximum(ln_zu_z0, _denom_floor)
    theta_star = KAPPA * dT / jnp.maximum(ln_zt_z0t, _denom_floor)
    q_star_val = KAPPA * dq / jnp.maximum(ln_zq_z0q, _denom_floor)

    carry = (u_star, z0, z0_t, z0_q, theta_star, q_star_val)
    # The MOST iteration mixes the (possibly float32) input state with float64
    # physical constants (G, NU_AIR, c_pd via the virtual-T coefficient), so a
    # carry leaf would silently promote float32 -> float64 mid-loop and trip
    # ``fori_loop``'s equal-types invariant.  This only bites the float32
    # atmosphere coupled path; the OMIP ocean path runs float64 so the re-casts
    # below are no-ops (byte-identical).  Pin each leaf back to its input dtype.
    _carry_dtypes = tuple(c.dtype for c in carry)

    def body_fn(i, carry):
        u_star, z0, z0_t, z0_q, theta_star, q_star_val = carry
        u_star_safe = jnp.maximum(u_star, 1e-6)

        # Virtual potential temperature scale (1/ε − 1 ≈ 0.6078)
        theta_v_star = theta_star + _vT_coef * T_atm * q_star_val

        # COARE 3.0 convective gustiness (opt-in; gustiness_w_zi=0 => off =>
        # byte-identical, so the OMIP/forward-default paths are unchanged).  Over
        # a calm but convectively-unstable warm ocean the mean wind alone gives
        # an anemic flux (the tropical hfls ~45 vs ~120 W/m² bias); the
        # free-convection velocity scale w* = (g·z_i·<w'θv'>/θv)^(1/3) adds a
        # sub-grid gust U_eff = sqrt(|U|² + (β·w*)²) (Fairall et al. 2003,
        # β~1.25, z_i = BL depth ~600 m).  <w'θv'> = u*·θv* (kinematic, upward
        # +; unstable only).
        if gustiness_w_zi > 0.0:
            wpthvp = jnp.maximum(u_star_safe * theta_v_star, 0.0)
            wstar = jnp.cbrt(G * gustiness_w_zi * wpthvp / T_v)
            U_eff = jnp.sqrt(wind_speed ** 2 + (gustiness_beta * wstar) ** 2)
        else:
            U_eff = wind_speed

        # Inverse Obukhov length: 1/L = −κ g θ_v* / (u*² T_v).
        #
        # The stability parameter ζ = z/L is the *only* way L enters this
        # solver, so we carry the reciprocal directly.  The reciprocal form
        # is singularity-free: the denominator u*² T_v is strictly positive
        # (u_star_safe ≥ 1e-6, T_v > 0), so there is no divide-by-zero and
        # no need for a sentinel or ``safe_divide``.  Crucially it gives the
        # correct neutral limit *and* its derivative: at exact neutral
        # (θ_v* = 0) ⇒ 1/L = 0 ⇒ ζ = 0 with the true finite sensitivity
        # dζ/dθ_v* = −z κ g /(u*² T_v).  Forming L = −u*²T_v/(κgθ_v*) first
        # and then z/L would instead either inject a NaN gradient (0·∞ from
        # the dead branch of a neutral-limit ``where``) or, if masked with a
        # constant ±1e6 sentinel, flatten dζ/dθ_v* to zero across the
        # neutral band and jump at the mask threshold.  Away from neutral ζ
        # equals the textbook z/L to round-off.
        inv_L = -KAPPA * G * theta_v_star / (u_star_safe ** 2 * T_v)

        # Stability parameters and ψ functions evaluated at each
        # measurement height. When z_t == z_q == z_u (single-height),
        # zeta_t == zeta_q == zeta_u and psi_h_t == psi_h_q.
        zeta_u = jnp.clip(z_u * inv_L, -10.0, 10.0)
        zeta_t = jnp.clip(z_t * inv_L, -10.0, 10.0)
        zeta_q = jnp.clip(z_q * inv_L, -10.0, 10.0)
        psi_m_u = psi_m(zeta_u)
        psi_h_t = psi_h(zeta_t)
        psi_h_q = psi_h(zeta_q)

        # --- Roughness update (Python if resolved at trace time) ---
        if scheme == "coare3":
            # COARE 3.0: Charnock + smooth-flow regime
            z0_new = (
                charnock * u_star_safe ** 2 / G
                + 0.11 * NU_AIR / u_star_safe
            )
            Re = u_star_safe * z0_new / NU_AIR
            z0_t_new = jnp.minimum(
                1.15e-4,
                5.5e-5 * jnp.power(jnp.maximum(Re, 0.01), -0.6),
            )
            z0_q_new = z0_t_new

        elif scheme == "large_yeager":
            # Large & Yeager 2009 (CORE/OMIP): iterate in coefficient space.
            # 1) Neutral 10-m wind from current u_star and z0
            U_10N = u_star_safe / KAPPA * jnp.log(
                10.0 / jnp.maximum(z0, 1e-12)
            )
            U_10N = jnp.clip(U_10N, 0.5, 50.0)

            # 2) LY09 neutral 10-m drag coefficient — the canonical shared drag
            # law (Large & Yeager 2009 Eq. 6, incl. the −3.14807e-10·U⁶ high-wind
            # correction + clip), NOT a per-component re-derivation.  U_10N is
            # already clipped to [0.5, 50], so the helper's 0.5 floor is a no-op.
            C_DN = large_yeager_neutral_cd(U_10N)

            # 3) Neutral exchange coefficients at 10 m
            rdn = jnp.sqrt(C_DN)
            # Stability-dependent 10-m Stanton/Dalton number (LY09 Table 4).
            # Use the wind-height stability for the unstable/stable branch.
            CHN10 = jnp.where(zeta_u < 0.0, 32.7e-3, 18.0e-3) * rdn
            CEN10 = 34.6e-3 * rdn
            rhn = CHN10 / rdn  # = ch_coeff
            ren = CEN10 / rdn  # = ce_coeff

            # 4) Shift coefficients from 10 m to the appropriate
            #    measurement height per variable (LY09 §3 / LY04 Eq. 9-11):
            #    rd = rdn / (1 + rdn/κ · (ln(z_u/10) − Δψ_m))
            #    rh = rhn / (1 + rhn/κ · (ln(z_t/10) − Δψ_h_t))
            #    re = ren / (1 + ren/κ · (ln(z_q/10) − Δψ_h_q))
            ln_zr_u = jnp.log(z_u / 10.0)
            ln_zr_t = jnp.log(z_t / 10.0)
            ln_zr_q = jnp.log(z_q / 10.0)
            zeta_10 = jnp.clip(10.0 * inv_L, -10.0, 10.0)
            psi_m_10 = psi_m(zeta_10)
            psi_h_10 = psi_h(zeta_10)
            dpsi_m = psi_m_u - psi_m_10
            dpsi_h_t = psi_h_t - psi_h_10
            dpsi_h_q = psi_h_q - psi_h_10

            rd = rdn / jnp.maximum(
                1.0 + rdn / KAPPA * (ln_zr_u - dpsi_m), 0.2
            )
            rh = rhn / jnp.maximum(
                1.0 + rhn / KAPPA * (ln_zr_t - dpsi_h_t), 0.2
            )
            re = ren / jnp.maximum(
                1.0 + ren / KAPPA * (ln_zr_q - dpsi_h_q), 0.2
            )

            # Optional transfer-coefficient ceiling (max_exchange_coeff). The LY
            # coefficient-space path bypasses the log-law denominator floor, so
            # apply the equivalent cap directly: rd, rh, re ≤ κ/_denom_floor
            # bounds C_d=rd², C_h=rd·rh, C_e=rd·re ≤ C_max — the SAME guarantee
            # the else-branch gets from _denom_floor. _coeff_cap is None (skip,
            # byte-identical) unless a ceiling was requested. z0_new below reads
            # the capped rd, so the roughness estimate stays consistent.
            if _coeff_cap is not None:
                rd = jnp.minimum(rd, _coeff_cap)
                rh = jnp.minimum(rh, _coeff_cap)
                re = jnp.minimum(re, _coeff_cap)

            # 5) Update scaling parameters directly from coefficients
            u_star_new = rd * U_eff
            theta_star_new = rh * dT
            q_star_new = re * dq

            # Still need z0 for the next iteration's U_10N estimate
            z0_new = z_u / jnp.exp(KAPPA / rd + psi_m_u)
            z0_new = jnp.clip(z0_new, 1e-12, 1.0)
            z0_t_new = z0_t  # not used in coefficient path
            z0_q_new = z0_q  # not used in coefficient path

            return (u_star_new, z0_new, z0_t_new, z0_q_new,
                    theta_star_new, q_star_new)

        else:
            z0_new = z0
            z0_t_new = z0_t
            z0_q_new = z0_q

        # For COARE and constant: transfer coefficients via log-law + stability,
        # with each variable referenced to its own measurement height.
        ln_zu_z0 = jnp.log(z_u / jnp.maximum(z0_new, 1e-12))
        ln_zt_z0t = jnp.log(z_t / jnp.maximum(z0_t_new, 1e-12))
        ln_zq_z0q = jnp.log(z_q / jnp.maximum(z0_q_new, 1e-12))

        denom_m = jnp.maximum(ln_zu_z0 - psi_m_u, _denom_floor)
        denom_h = jnp.maximum(ln_zt_z0t - psi_h_t, _denom_floor)
        denom_q = jnp.maximum(ln_zq_z0q - psi_h_q, _denom_floor)

        u_star_new = KAPPA * U_eff / denom_m
        theta_star_new = KAPPA * dT / denom_h
        q_star_new = KAPPA * dq / denom_q

        return (u_star_new, z0_new, z0_t_new, z0_q_new,
                theta_star_new, q_star_new)

    def _body_fn_dtype_stable(i, carry):
        out = body_fn(i, carry)
        return tuple(jnp.asarray(o).astype(d)
                     for o, d in zip(out, _carry_dtypes))

    carry = jax.lax.fori_loop(0, n_iter, _body_fn_dtype_stable, carry)
    u_star, z0, z0_t, z0_q, theta_star, q_star_val = carry

    # Fluxes from scaling parameters
    _L = constants.L_v if L_latent is None else L_latent
    tau_x = -rho * u_star ** 2 * u_rel / wind_speed
    tau_y = -rho * u_star ** 2 * v_rel / wind_speed
    shflx = rho * constants.c_pd * u_star * theta_star
    lhflx = rho * _L * u_star * q_star_val

    if return_2m:
        # Air temperature at the diagnostic height (default 2 m) from the
        # converged MOST similarity profile: T(z) = T_sfc − (θ*/κ)·[ln(z/z0t) −
        # ψ_h(z/L)], which reduces to T_atm at z=z_t.  Over a warm ocean the
        # lowest model level (~100 m at nlev=20) reads colder than 2 m, so the
        # raw lowest-level "tas" exaggerates the cold/air-sea-gap bias — this
        # gives the physically correct CMIP 2 m value.  Recompute 1/L from the
        # converged scales (the iteration carries scales, not L).
        u_star_safe = jnp.maximum(u_star, 1e-6)
        theta_v_star = theta_star + _vT_coef * T_atm * q_star_val
        inv_L = -KAPPA * G * theta_v_star / (u_star_safe ** 2 * T_v)
        zeta_d = jnp.clip(z_diag * inv_L, -10.0, 10.0)
        denom_d = jnp.log(z_diag / jnp.maximum(z0_t, 1e-12)) - psi_h(zeta_d)
        T_2m = T_sfc - (theta_star / KAPPA) * denom_d
        # Guard against profile extrapolation outside [T_atm, T_sfc].
        lo = jnp.minimum(T_atm, T_sfc)
        hi = jnp.maximum(T_atm, T_sfc)
        T_2m = jnp.clip(T_2m, lo, hi)
        return tau_x, tau_y, shflx, lhflx, u_star, T_2m

    return tau_x, tau_y, shflx, lhflx, u_star


def sam_ocean_surface_q(
    T_sfc: jnp.ndarray,
    p_sfc: jnp.ndarray | float,
    salt_factor: float = 0.981,
) -> jnp.ndarray:
    """SAM ocean surface saturation specific humidity (``oceflx`` ``qs``).

    SAM sets the ocean surface humidity to the SATURATION value at the SST,
    reduced by a salinity factor (``qs = salt_factor · qsat(SST, p_sfc)``,
    ``salt_factor = 0.981`` ≈ the 2 % saturation reduction over saline
    ocean water).  At SST = 300 K, p_sfc = 1014.8 hPa this is ≈ 0.0217
    kg/kg — about 20 % larger than the hardcoded ``q_sfc = 0.018`` the
    RCEMIP plane harness used, so the latent-heat flux (and the whole
    WISHE feedback / RCE moisture budget) was biased low.

    Uses the shared :func:`legoesm.thermo.saturation_specific_humidity`
    (CLAUDE.md — no re-implemented Clausius–Clapeyron).

    Parameters
    ----------
    T_sfc : array
        Sea-surface temperature [K].
    p_sfc : array or float
        Surface pressure [Pa].
    salt_factor : float
        Salinity reduction of saturation humidity (SAM default 0.981).

    Returns
    -------
    array
        Surface saturation specific humidity [kg/kg].
    """
    from legoesm.thermo import saturation_specific_humidity

    return salt_factor * saturation_specific_humidity(T_sfc, p_sfc)


def _sam_cdn(u10: jnp.ndarray) -> jnp.ndarray:
    """SAM/CESM neutral 10 m drag coefficient (``oceflx.f90`` ``cdn``).

    ``cdn(U) = 0.0027/U + 0.000142 + 0.0000764·U``.  ``U`` is floored at
    the SAM minimum wind ``umin = 1 m/s`` so the ``1/U`` term stays bounded
    (and AD-safe).
    """
    u = jnp.maximum(u10, 1.0)
    return 0.0027 / u + 0.000142 + 0.0000764 * u


def compute_sam_oceflx_fluxes(
    u_atm: jnp.ndarray,
    v_atm: jnp.ndarray,
    theta_atm: jnp.ndarray,
    q_atm: jnp.ndarray,
    T_sfc: jnp.ndarray,
    q_sfc: jnp.ndarray,
    rho: jnp.ndarray,
    z_bot: jnp.ndarray | float,
    exner_sfc: jnp.ndarray | float = 1.0,
    wd: jnp.ndarray | float = 0.0,
    n_iter: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """SAM ocean surface fluxes — faithful port of ``oceflx.f90`` (CESM1).

    Iterative Monin–Obukhov bulk scheme with SAM's exact neutral transfer
    coefficients and Businger–Dyer stability functions:

    - ``vmag = max(1.0, sqrt(|U|² + wd²))`` — 1 m/s minimum, optional gust
      enhancement ``wd`` (SAM default 0; NOT the Wing-2018 5 m/s floor).
    - Neutral drag ``rdn² = cdn(U10)`` (:func:`_sam_cdn`).
    - Neutral Stanton ``rhn = 0.0327`` (unstable) / ``0.018`` (stable);
      neutral Dalton ``ren = 0.0346``.
    - ``n_iter`` (SAM uses 2) stability iterations: Obukhov ratio
      ``hol = κ g z (t*/θ + q*/(1/ε+q)) / u*²`` (clipped |hol|≤10), then
      the coefficients are shifted to the model level via
      ``r = rn / (1 + rn/κ·(ln(z/z_ref) − ψ))``.

    Fluxes are returned in legoESM's UPWARD-POSITIVE W/m² convention
    (surface warmer/moister ⇒ positive), matching
    :func:`simple_bulk_fluxes`:

        SHF = −ρ c_pd u* t*,   LHF = −ρ L_v u* q*,   τ = ρ u*²

    Parameters
    ----------
    u_atm, v_atm : array
        Lowest-level wind components [m/s].
    theta_atm : array
        Lowest-level POTENTIAL temperature [K] (SAM ``thbot``).
    q_atm : array
        Lowest-level specific humidity [kg/kg].
    T_sfc : array
        Surface temperature [K].  The surface POTENTIAL temperature
        ``θ_sfc = T_sfc / exner_sfc`` is formed for the sensible-heat
        gradient (Codex iter-5 HIGH: ``exner_sfc`` differs from 1 by
        ~0.4 % over the ocean ⇒ ~1 K offset in ``θ_sfc``, comparable to
        the air–sea disequilibrium and able to flip the stable/unstable
        classification near neutral — so do NOT use ``T_sfc`` as
        ``θ_sfc`` directly).
    q_sfc : array
        Surface saturation specific humidity [kg/kg] (saturation at the
        absolute SST; see :func:`sam_ocean_surface_q`).
    rho : array
        Air density at the lowest level [kg/m³].
    z_bot : array or float
        Lowest model-level height [m] (SAM ``zbot``).
    exner_sfc : array or float
        Surface Exner function ``(p_sfc/p_ref)^κ`` used to convert
        ``T_sfc`` to the surface potential temperature.  Default 1.0
        reproduces SAM's ``ts``-as-``θ`` approximation.
    wd : array or float
        Gust-enhancement wind [m/s] added in quadrature to ``vmag``
        (SAM ``wd``, default 0).
    n_iter : int
        Number of MO stability iterations (SAM uses 2).

    Returns
    -------
    tau_x, tau_y : array
        Surface stress [Pa] (opposes wind).
    shflx : array
        Sensible heat flux [W/m²] (positive upward = surface warmer).
    lhflx : array
        Latent heat flux [W/m²] (positive upward = surface moister).
    ustar : array
        Friction velocity [m/s].
    """
    karman = KAPPA
    eps_v = 1.0 / constants.epsilon - 1.0
    z_ref = 10.0
    umin = 1.0

    # Safe sqrt (+eps) so the calm-wind (u=v=wd=0) reverse-mode gradient
    # stays finite — bare sqrt(0) gives a 0·inf NaN even though the value
    # is floored to umin (Codex iter-5). eps ≪ umin² so the primal is
    # unchanged to round-off.
    vmag = jnp.maximum(
        umin, jnp.sqrt(u_atm ** 2 + v_atm ** 2 + wd ** 2 + 1e-12)
    )
    theta_sfc = T_sfc / exner_sfc             # surface potential temperature
    delt = theta_atm - theta_sfc             # SAM thbot - ts (pot-T diff)
    delq = q_atm - q_sfc                      # spec-humidity diff
    alz = jnp.log(z_bot / z_ref)

    # Businger–Dyer unstable stability integrals (SAM psimhu / psixhu).
    def psimhu(xd):
        return (
            jnp.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)
            - 2.0 * jnp.arctan(xd) + 1.571
        )

    def psixhu(xd):
        return 2.0 * jnp.log((1.0 + xd * xd) / 2.0)

    # --- neutral first guess ---
    # Fortran ``sign(0.5,x)`` treats x==0 as POSITIVE ⇒ stable=1 at exact
    # neutral; ``where(x>=0,1,0)`` matches that (0.5+0.5*sign would give
    # 0.5 — a non-SAM averaged branch).
    stable = jnp.where(delt >= 0.0, 1.0, 0.0)
    rdn = jnp.sqrt(_sam_cdn(vmag))
    rhn = (1.0 - stable) * 0.0327 + stable * 0.018
    ren = jnp.full_like(jnp.asarray(rdn), 0.0346)
    u_star = rdn * vmag
    t_star = rhn * delt
    q_star = ren * delq

    # --- MO stability iterations (SAM does 2) ---
    for _ in range(n_iter):
        hol = (
            karman * G * z_bot
            * (t_star / theta_atm + q_star / (1.0 / eps_v + q_atm))
            / jnp.maximum(u_star ** 2, 1e-12)
        )
        # jnp.clip (not sign*min(abs)) clamps hol to [-10, 10] with the SAME
        # forward values but a SMOOTH, nonzero gradient in (-10, 10): the old
        # sign*min(abs) form has a zero-gradient flat spot at hol=0 (neutral)
        # that severs d(flux)/d(state) sensitivity there under jax.grad.
        hol = jnp.clip(hol, -10.0, 10.0)
        stable = jnp.where(hol >= 0.0, 1.0, 0.0)
        xsq = jnp.maximum(jnp.sqrt(jnp.abs(1.0 - 16.0 * hol)), 1.0)
        xqq = jnp.sqrt(xsq)
        psimh = -5.0 * hol * stable + (1.0 - stable) * psimhu(xqq)
        psixh = -5.0 * hol * stable + (1.0 - stable) * psixhu(xqq)
        # Shift wind, recompute neutral coeffs at u10n, then shift all.
        rd = rdn / (1.0 + rdn / karman * (alz - psimh))
        u10n = vmag * rd / rdn
        rdn = jnp.sqrt(_sam_cdn(u10n))
        rhn = (1.0 - stable) * 0.0327 + stable * 0.018
        ren = jnp.full_like(jnp.asarray(rdn), 0.0346)
        rd = rdn / (1.0 + rdn / karman * (alz - psimh))
        rh = rhn / (1.0 + rhn / karman * (alz - psixh))
        re = ren / (1.0 + ren / karman * (alz - psixh))
        u_star = rd * vmag
        t_star = rh * delt
        q_star = re * delq

    # --- fluxes (legoESM upward-positive W/m²) ---
    tau = rho * u_star ** 2
    tau_x = -tau * u_atm / vmag
    tau_y = -tau * v_atm / vmag
    shflx = -rho * constants.c_pd * u_star * t_star
    lhflx = -rho * constants.L_v * u_star * q_star
    return tau_x, tau_y, shflx, lhflx, u_star


def simple_bulk_fluxes(
    u_lowest: jnp.ndarray,
    v_lowest: jnp.ndarray,
    T_lowest: jnp.ndarray,
    q_lowest: jnp.ndarray,
    T_sfc: jnp.ndarray,
    q_sfc: jnp.ndarray,
    rho: jnp.ndarray,
    wind_speed: jnp.ndarray,
    Cd: float,
    Ch: float,
    L_latent: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simple bulk-aerodynamic surface fluxes with constant coefficients.

    Parameters
    ----------
    u_lowest, v_lowest : array
        Lowest-level wind components [m/s].
    T_lowest : array
        Lowest-level air temperature [K].
    q_lowest : array
        Lowest-level specific humidity [kg/kg].
    T_sfc : array
        Surface temperature [K].
    q_sfc : array
        Surface specific humidity [kg/kg].
    rho : array
        Air density at lowest level [kg/m3].
    wind_speed : array
        Wind speed (with minimum floor applied) [m/s].
    Cd : float
        Drag coefficient for momentum.
    Ch : float
        Transfer coefficient for heat and moisture.

    Returns
    -------
    tau_x, tau_y : array
        Surface stress [Pa] (opposes wind).
    shflx : array
        Sensible heat flux [W/m2] (positive upward = surface warmer).
    lhflx : array
        Latent heat flux [W/m2] (positive upward = surface moister).
    """
    _L = constants.L_v if L_latent is None else L_latent
    tau_x = -rho * Cd * wind_speed * u_lowest
    tau_y = -rho * Cd * wind_speed * v_lowest
    shflx = rho * constants.c_pd * Ch * wind_speed * (T_sfc - T_lowest)
    lhflx = rho * _L * Ch * wind_speed * (q_sfc - q_lowest)
    return tau_x, tau_y, shflx, lhflx
