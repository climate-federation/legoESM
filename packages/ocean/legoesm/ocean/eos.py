"""Equation of state for seawater: Wright (1997) and linear.

Provides:
- ``wright_eos`` — nonlinear Wright (1997) EOS (MOM6 implementation)
- ``linear_eos`` — configurable linear EOS: ρ = ρ₀[1 - αT(T-Tref) + βS(S-Sref)]
- ``make_eos_fn`` — dispatcher returning an EOS callable based on config

Pure JAX functions, compatible with jit/grad/vmap.

Faithfulness
------------
``tests/ocean/unit/test_eos_wright_faithful.py`` pins the DEFAULT ``wright_eos``
(previously only qualitatively checked, while every alternate EOS was value-pinned):
(1) the density rational polynomial to round-off (rel 1e-12) across a T×S×p grid
vs an independent scalar reimplementation; (2) an INDEPENDENT physical cross-check
that ρ_Wright agrees with the separately-pinned UNESCO-80 EOS to < 0.01 kg/m^3 at
the surface (catches any material coefficient error affecting these points — four
surface values cannot pin down all 15 coefficients, but a gross error diverges);
(3) ``thermal_expansion_coeff`` α and ``haline_contraction_coeff`` β (which use
``jax.grad``) against independent ANALYTIC derivatives of the closed form (rel
1e-9); (4) the MOM6/Wright coefficients as a canary; (5) physical monotonicity and
x64/float32 AD-finiteness.  The EOS polynomial runs in the precision policy's
compute dtype (float32 by default), so the round-off pins set the fp64 policy.

Reference
---------
Wright, D. G. (1997): An Equation of State for Use in Ocean Models:
Ockham's Razor Revisited. J. Atmos. Oceanic Tech., 14(3), 735-740.
Coefficients from MOM6 ``MOM_EOS_Wright.F90`` (reduced-range fit).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
from jax import lax
from jax.experimental import checkify
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.precision import resolve_dtype
from legoesm.core.source_rounding import nemo_source_round

# ==============================================================================
# Ocean constants
# ==============================================================================
# Re-export central physical constants so existing call sites
# (``from legoesm.ocean.eos import rho_0, c_sw, ...``) keep working.
# Per CLAUDE.md the canonical values live in ``legoesm.constants``;
# the prior literal definitions here violated the
# constant-discipline rule and could silently drift from the central
# values.  Module-level binding to ``constants.X`` keeps a single
# source of truth.
rho_0 = constants.rho_ocean         # Reference seawater density [kg/m^3]
c_sw = constants.c_sw               # Specific heat of seawater [J/(kg*K)]


def nemo_potential_temperature_from_conservative(ct, salinity):
    """NEMO TEOS-10 surface CT-to-potential-temperature approximation.

    Literal 5/3 rational recurrence from NEMO 5.0.2
    ``eosbn2.F90:eos_pt_from_ct_t`` (lines 1519-1542).  NEMO's ``sbc_ssm``
    applies this conversion to ``sst_m`` when ``ln_teos10`` is true; GYRE's
    freshwater heat-content correction then consumes that potential
    temperature even though its Haney restoring term consumes ``ts(Kbb)``.
    """
    ct = jnp.asarray(ct)
    salinity = jnp.asarray(salinity, dtype=ct.dtype)
    zt = ct * jnp.asarray(1.0 / 40.0, dtype=ct.dtype)
    zs = jnp.sqrt(
        jnp.abs(salinity + jnp.asarray(5.0, dtype=ct.dtype))
        * jnp.asarray(0.875 / 35.16504, dtype=ct.dtype))
    zn = -2.1385727895e-01 * zt - 2.7674419971e-01 * zs + 1.0728094330
    zn = zn * zt + (2.6366564313 * zs + 3.3546960647) * zs - 7.8012209473
    zn = (zn * zt + ((1.8835586562 * zs + 7.3949191679) * zs
                     - 3.3937395875) * zs - 5.6414948432)
    zn = (zn * zt + (((3.5737370589 * zs - 1.5512427389e+01) * zs
                      + 2.4625741105e+01) * zs
                     + 1.9912291000e+01) * zs - 3.2191146312e+01)
    zn = (zn * zt + ((((5.7153204649e-01 * zs - 3.0943149543) * zs
                       + 9.3052495181) * zs - 9.4528934807) * zs
                     + 3.1066408996) * zs - 4.3504021262e-01)
    zd = ((2.0035003456 * zt - 3.4570358592e-01 * zs
           + 5.6471810638) * zt
          + (1.5393993508 * zs - 6.9394762624) * zs
          + 1.2750522650e+01)
    return ct + zn / zd
T_freeze_ocean = constants.T_freeze_ocean  # Freezing point of seawater [K]
scale_depth = 1000.0     # Reference e-folding depth for stratification [m]

# ==============================================================================
# Wright (1997) EOS coefficients — from MOM6 (MOM_EOS_Wright.F90)
# Pressure units: Pa. Temperature: degC. Salinity: PSU.
#
# Formula: rho = (p + p0) / (lambda + al0 * (p + p0))
#   al0(T, S) = a0 + a1*T + a2*S
#   p0(T, S)  = (b0 + b4*S) + T*(b1 + T*(b2 + b3*T) + b5*S)
#   lambda(T, S) = (c0 + c4*S) + T*(c1 + T*(c2 + c3*T) + c5*S)
# ==============================================================================

# Specific volume coefficients al0(T, S)
WRIGHT_A0 = 7.057924e-4
WRIGHT_A1 = 3.480336e-7
WRIGHT_A2 = -1.112733e-7

# Pressure offset p0(T, S) [Pa]
WRIGHT_B0 = 5.790749e8
WRIGHT_B1 = 3.516535e6
WRIGHT_B2 = -4.002714e4
WRIGHT_B3 = 2.084372e2
WRIGHT_B4 = 5.944068e5
WRIGHT_B5 = -9.643486e3

# Lambda(T, S) [m^2/s^2]
WRIGHT_C0 = 1.704853e5
WRIGHT_C1 = 7.904722e2
WRIGHT_C2 = -7.984422
WRIGHT_C3 = 5.140652e-2
WRIGHT_C4 = -2.302158e2
WRIGHT_C5 = -3.079464


def wright_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    compute_dtype: "jnp.dtype | None" = None,
) -> jnp.ndarray:
    """Compute in-situ density from Wright (1997) EOS.

    Parameters
    ----------
    T : array
        Potential temperature [degC].
    S : array
        Salinity [PSU].
    p : array
        Pressure [Pa]. Use 0 for surface.

    Returns
    -------
    array : In-situ density [kg/m^3].

    Notes
    -----
    Intermediate computation is promoted to float64 to avoid precision
    loss from large polynomial coefficients (e.g., WRIGHT_B0 ~ 5.79e8).
    If ``JAX_ENABLE_X64=1`` is not set, the astype calls are no-ops
    (safe but no precision improvement).  ``jnp.astype`` is
    differentiable in JAX.

    The Wright (1997) polynomial is nominally valid for T in [-2, 40] degC
    and S in [0, 42] PSU, but extrapolates smoothly outside that box.
    Inputs are not clipped: silent clipping would zero gradients at the
    boundary and mask unphysical state from advection overshoots or
    coupler bugs. See issue #165.
    """
    orig_dtype = T.dtype

    # Promote to the EOS compute dtype (float64 in mixed mode) for
    # intermediate polynomial evaluation.  On backends that lack float64
    # (e.g. Metal), resolve_dtype silently returns float32.
    # ``compute_dtype`` (opt-in mixed-precision baroclinic lever) overrides the
    # policy compute dtype — e.g. float32 for the f32-EOS path — and is honoured
    # over the policy; the density ANOMALY rho'~O(1) formed downstream keeps this
    # precision-safe (offline experiment 8520588: PGF relRMS ~8e-5).  ``orig_dtype``
    # is still restored on return, so the STATE stays f64 (only the polynomial
    # runs in compute_dtype).
    hi = compute_dtype if compute_dtype is not None else resolve_dtype(
        "equation_of_state", "compute")
    T = T.astype(hi)
    S = S.astype(hi)
    p = p.astype(hi)

    # Specific volume parameter
    al0 = WRIGHT_A0 + WRIGHT_A1 * T + WRIGHT_A2 * S

    # Pressure offset
    p0 = (WRIGHT_B0 + WRIGHT_B4 * S) + T * (WRIGHT_B1 + T * (WRIGHT_B2 + WRIGHT_B3 * T) + WRIGHT_B5 * S)

    # Lambda
    lam = (WRIGHT_C0 + WRIGHT_C4 * S) + T * (WRIGHT_C1 + T * (WRIGHT_C2 + WRIGHT_C3 * T) + WRIGHT_C5 * S)

    # Density: rho = (p + p0) / (lambda + al0 * (p + p0))
    p_plus_p0 = p + p0
    rho = p_plus_p0 / (lam + al0 * p_plus_p0)

    return rho.astype(orig_dtype)


def _wright_eos_scalar(T: float, S: float, p: float) -> float:
    """Scalar Wright EOS for JAX grad (no dtype promotion).

    Used internally by ``thermal_expansion_coeff`` and
    ``haline_contraction_coeff`` via ``jax.grad``.
    """
    al0 = WRIGHT_A0 + WRIGHT_A1 * T + WRIGHT_A2 * S
    p0 = (WRIGHT_B0 + WRIGHT_B4 * S) + T * (WRIGHT_B1 + T * (WRIGHT_B2 + WRIGHT_B3 * T) + WRIGHT_B5 * S)
    lam = (WRIGHT_C0 + WRIGHT_C4 * S) + T * (WRIGHT_C1 + T * (WRIGHT_C2 + WRIGHT_C3 * T) + WRIGHT_C5 * S)
    p_plus_p0 = p + p0
    return p_plus_p0 / (lam + al0 * p_plus_p0)


# Partial derivatives via JAX autodiff (scalar → vmap for arrays).
import jax
_drho_dT_scalar = jax.grad(_wright_eos_scalar, argnums=0)
_drho_dS_scalar = jax.grad(_wright_eos_scalar, argnums=1)


def thermal_expansion_coeff(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    r"""Thermal expansion coefficient α = -(1/ρ) ∂ρ/∂T.

    Parameters
    ----------
    T : array — Potential temperature [degC].
    S : array — Salinity [PSU].
    p : array — Pressure [Pa].

    Returns
    -------
    array : α [1/K], same shape as inputs.
    """
    hi = resolve_dtype("equation_of_state", "compute")
    T64 = T.astype(hi)
    S64 = S.astype(hi)
    p64 = p.astype(hi)
    flat_T = T64.ravel()
    flat_S = S64.ravel()
    flat_p = p64.ravel()
    drho_dT = jax.vmap(_drho_dT_scalar)(flat_T, flat_S, flat_p).reshape(T.shape)
    rho = wright_eos(T, S, p)
    return (-drho_dT / rho).astype(T.dtype)


def haline_contraction_coeff(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    r"""Haline contraction coefficient β = (1/ρ) ∂ρ/∂S.

    Parameters
    ----------
    T : array — Potential temperature [degC].
    S : array — Salinity [PSU].
    p : array — Pressure [Pa].

    Returns
    -------
    array : β [1/PSU], same shape as inputs.
    """
    hi = resolve_dtype("equation_of_state", "compute")
    T64 = T.astype(hi)
    S64 = S.astype(hi)
    p64 = p.astype(hi)
    flat_T = T64.ravel()
    flat_S = S64.ravel()
    flat_p = p64.ravel()
    drho_dS = jax.vmap(_drho_dS_scalar)(flat_T, flat_S, flat_p).reshape(T.shape)
    rho = wright_eos(T, S, p)
    return (drho_dS / rho).astype(T.dtype)


def eos_density_derivatives(
    eos_fn,
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Locally-referenced EOS partial derivatives ``(∂ρ/∂T, ∂ρ/∂S)``.

    Returns the two partial derivatives of an arbitrary EOS callable
    ``eos_fn(T, S, p) -> ρ`` evaluated at the LOCAL pressure ``p`` — the
    ingredients of the locally-referenced *neutral* density gradient
    ``∇_neutral ρ = (∂ρ/∂T)·∇T + (∂ρ/∂S)·∇S`` used by isoneutral mixing
    (Veros ``get_drhodT`` / ``get_drhodS`` at ``abs(zt)``;
    ``veros/core/isoneutral/isoneutral.py:40-41``).

    Relationship to the thermodynamic coefficients already in this module:
    ``∂ρ/∂T = −ρ·α`` (``thermal_expansion_coeff``) and
    ``∂ρ/∂S = +ρ·β`` (``haline_contraction_coeff``).  Rather than restrict
    to those wright-only helpers, this differentiates the *selected*
    ``eos_fn`` directly via :func:`jax.grad` so EVERY dispatchable EOS
    (wright, linear, unesco80, veros_nonlin2/3) is supported with one code
    path — exactly the autodiff α/β already use internally for wright.

    Holding ``p`` fixed during the differentiation is deliberate and
    faithful: the neutral gradient is the density change at constant
    (local) reference pressure, so the adiabatic compressibility term
    ``∂ρ/∂p·∂p/∂z`` — which makes the in-situ ``∂ρ/∂z`` ~4× too steep — is
    excluded by construction.

    Parameters
    ----------
    eos_fn : Callable[[array, array, array], array]
        Equation of state (e.g. from :func:`make_eos_fn`).
    T, S, p : array
        Potential temperature [°C], salinity [PSU], pressure [Pa] at the
        SAME points; identical shapes.

    Returns
    -------
    drho_dT : array — ``∂ρ/∂T`` [kg/m³/K], same shape as inputs.
    drho_dS : array — ``∂ρ/∂S`` [kg/m³/(g/kg)], same shape as inputs.

    Notes
    -----
    Fully ``jax.grad``-safe: only EOS evaluations + autodiff, no
    ``where``/``cond`` on traced values.  Promotes to the EOS compute dtype
    (float64 in mixed mode) for the polynomial evaluation, matching α/β.
    """
    hi = resolve_dtype("equation_of_state", "compute")
    T64 = T.astype(hi)
    S64 = S.astype(hi)
    p64 = p.astype(hi)
    flat_T = T64.ravel()
    flat_S = S64.ravel()
    flat_p = p64.ravel()

    def _scalar(t, s, pp):
        # eos_fn is array-shaped; wrap scalars in length-1 arrays so the
        # promotion-to-float64 astype inside the EOS sees an ndarray.
        return eos_fn(t[None], s[None], pp[None])[0]

    drho_dT = jax.vmap(jax.grad(_scalar, argnums=0))(
        flat_T, flat_S, flat_p
    ).reshape(T.shape)
    drho_dS = jax.vmap(jax.grad(_scalar, argnums=1))(
        flat_T, flat_S, flat_p
    ).reshape(T.shape)
    return drho_dT.astype(T.dtype), drho_dS.astype(T.dtype)


def int_drhodTS_dynamic_enthalpy(
    eos_fn,
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_full: jnp.ndarray,
    rho_0: float,
    g: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Dynamic-enthalpy integrands ``(int_drhodT, int_drhodS)`` = ``∫_z^0 ∂ρ/∂X dz'``.

    These are exactly Veros's ``int_drhodT`` / ``int_drhodS`` (``get_int_drhodT`` /
    ``get_int_drhodS``, ``veros/core/density/get_rho.py:157-195``) — the
    vertically-integrated EOS partial derivatives that drive the isoneutral /
    skew APE-dissipation diagnostics ``P_diss_skew`` / ``P_diss_iso``
    (``veros/core/isoneutral/diffusion.py``).  The gradient of these fields
    contracted with the GM/Redi flux is the energy the parameterization extracts
    from (skew) / dissipates into (iso) the mean APE — the quantity the EKE
    source needs (``-P_diss_skew`` feeds EKE; ``-P_diss_iso`` sinks it).

    Leading-order form (EOS-agnostic, used here):

        int_drhodX(z) ≈ -z · (∂ρ/∂X)(T, S, p_local),  z < 0 (depth ≈ -z),

    i.e. the cell-local EOS partial derivative (from
    :func:`eos_density_derivatives` at the LOCAL hydrostatic pressure
    ``p ≈ rho_0·g·|z|``) times the depth ``|z| = -z``.  This is the exact integral
    when ``∂ρ/∂X`` is pressure-independent (linear EOS) and captures the dominant
    term for the nonlinear EOSs; the residual is the thermobaric correction
    ``∫(∂²ρ/∂X∂p)·∂p/∂z·z' dz'`` — measured ≈5 % on the ACC column means for
    veros_nonlin2 but up to ≈16 % pointwise at ~2000 m for the Wright EOS (the
    leading form evaluates ∂ρ/∂X at the LOCAL bottom-of-column pressure × full
    depth instead of integrating the depth-varying derivative; adversarial
    review 2026-06-10). A documented approximation that keeps this helper a
    single EOS-agnostic code path (no per-EOS analytic antiderivative); a
    faithful per-EOS analytic dynamic enthalpy is the refinement if the signed
    conversions ever need the last few percent.

    Sign convention: ``z_full < 0`` below the surface, so ``-z_full = |z| ≥ 0``
    and ``int_drhodX`` carries the sign of ``∂ρ/∂X``.  Fully ``jax.grad``-safe
    (only :func:`eos_density_derivatives` + a multiply).

    Parameters
    ----------
    eos_fn : Callable[[array, array, array], array] — equation of state.
    T, S : array — potential temperature [°C], salinity [g/kg] at cell centres.
    z_full : array broadcastable to ``T`` — cell-centre height [m] (negative
        below the surface; the ``z_full_ref`` of the vertical coordinate).
    rho_0, g : reference density [kg/m³] and gravity [m/s²] for ``p = rho_0·g·|z|``.

    Returns
    -------
    int_drhodT, int_drhodS : arrays, same shape as ``T`` —
        ``∫_z^0 ∂ρ/∂T dz'`` [kg/m³·m / K] and ``∫_z^0 ∂ρ/∂S dz'`` [kg/m³·m / (g/kg)].
    """
    z = jnp.broadcast_to(jnp.asarray(z_full), T.shape)
    p_local = jnp.asarray(rho_0 * g) * jnp.abs(z)
    drho_dT, drho_dS = eos_density_derivatives(eos_fn, T, S, p_local)
    int_drhodT = -z * drho_dT
    int_drhodS = -z * drho_dS
    return int_drhodT, int_drhodS


# ==============================================================================
# Linear equation of state
# ==============================================================================

class LinearEOSConfig(NamedTuple):
    """Configuration for the linear equation of state.

    ρ = rho_ref * [1 - alpha_T * (T - T_ref) + beta_S * (S - S_ref)]
    """
    rho_ref: float = constants.rho_ocean
    alpha_T: float = 2.0e-4    # Thermal expansion coefficient [1/K]
    beta_S: float = 7.4e-4     # Haline contraction coefficient [1/PSU]
    T_ref: float = 10.0        # Reference temperature [°C]
    S_ref: float = 35.0        # Reference salinity [PSU]


def linear_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    rho_ref: float = rho_0,
    alpha_T: float = 2.0e-4,
    beta_S: float = 7.4e-4,
    T_ref: float = 10.0,
    S_ref: float = 35.0,
) -> jnp.ndarray:
    """Compute density from a linear equation of state.

    ρ = rho_ref * [1 - alpha_T * (T - T_ref) + beta_S * (S - S_ref)]

    Parameters
    ----------
    T : array — Potential temperature [°C].
    S : array — Salinity [PSU].
    p : array — Pressure [Pa] (unused, accepted for API compatibility).
    rho_ref : float — Reference density [kg/m³].
    alpha_T : float — Thermal expansion coefficient [1/K].
    beta_S : float — Haline contraction coefficient [1/PSU].
    T_ref : float — Reference temperature [°C].
    S_ref : float — Reference salinity [PSU].

    Returns
    -------
    array : In-situ density [kg/m³].
    """
    return rho_ref * (1.0 - alpha_T * (T - T_ref) + beta_S * (S - S_ref))


# ==============================================================================
# NEMO "simplified" EOS (S-EOS / np_seos), Roquet et al. (2015), Ocean
# Modelling 90, 29-43; NEMO ``src/OCE/TRA/eosbn2.F90`` (np_seos branch).
#
# A polynomial in (potential temperature, practical salinity, depth) that
# retains the leading thermobaric + cabbeling nonlinearities of the full
# TEOS-10/EOS-80 forms while staying cheap and tunable.  Density anomaly:
#
#   zt = T - T0 ;  zs = S - S0 ;  zh = depth [m, positive down]
#   zn = - a0 (1 + ½ λ1 zt + μ1 zh) zt        (temperature, thermobaric μ1)
#        + b0 (1 - ½ λ2 zs - μ2 zh) zs        (salinity,    thermobaric μ2)
#        - nu  zt zs                          (cabbeling)
#   ρ  = ρ0 + zn
#
# Defaults are the Kamm et al. (2025) DINO coefficients (their Table /
# DINO ``namelist_cfg`` &nameos): a0=0.165, b0=0.76554, λ1=0.06,
# λ2=μ2=ν=0, μ1=1.4970e-4, T0=10 °C, S0=35 PSU, ρ0=1026 kg/m³ — i.e. a
# linear-in-S, weakly-nonlinear-in-T state with a thermobaric term, used
# as the NEMO oracle for the DINO ACC thermocline comparison.
# ==============================================================================

class NemoSEOSConfig(NamedTuple):
    """Coefficients for the NEMO simplified EOS (Roquet et al. 2015).

    ρ = ρ0 - a0(1 + ½λ1·zt + μ1·zh)·zt + b0(1 - ½λ2·zs - μ2·zh)·zs - ν·zt·zs
    with zt = T - T0, zs = S - S0, zh = depth [m].

    Defaults are the Kamm et al. (2025) DINO configuration.
    """
    rho0: float = 1026.0       # Reference (Boussinesq) density [kg/m³]
    a0: float = 0.165          # Thermal contraction coefficient [kg/m³/K]
    b0: float = 0.76554        # Haline contraction coefficient [kg/m³/PSU]
    lambda1: float = 0.06      # T·T (cabbeling-in-T) coefficient [1/K]
    lambda2: float = 0.0       # S·S (cabbeling-in-S) coefficient [1/PSU]
    mu1: float = 1.4970e-4     # T thermobaric coefficient [1/m]
    mu2: float = 0.0           # S thermobaric coefficient [1/m]
    nu: float = 0.0            # T·S cabbeling coefficient [kg/m³/K/PSU]
    T0: float = 10.0           # Reference temperature [°C]
    S0: float = 35.0           # Reference salinity [PSU]


def nemo_seos_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    cfg: NemoSEOSConfig | None = None,
) -> jnp.ndarray:
    """In-situ density from the NEMO simplified EOS (Roquet et al. 2015).

    Mirrors NEMO ``eosbn2.F90`` (``np_seos``): the density anomaly ``zn``
    is added to ``ρ0`` (NEMO stores ``prd = zn / ρ0``; ``ρ = ρ0(1+prd) =
    ρ0 + zn``).  Depth enters via the Boussinesq hydrostatic relation
    ``zh = p / (ρ0·g)`` (NEMO uses ``gdept`` in metres).

    Pressure-vs-depth note: ``p`` is the SAME 3rd argument every legoESM EOS
    takes — the dycore's hydrostatic pressure ``p = ∫ρg dz`` (positive,
    increasing downward, eta-free in the baroclinic PGF path).  ``zh =
    p/(ρ0·g)`` is therefore the Boussinesq reconstruction of geometric depth;
    it equals NEMO's ``gdept`` to ``O(ρ'/ρ0) ≈ 0.3 %`` (the in-situ vs
    reference-density difference).  For the DINO thermobaric term that ~12 m
    depth error at 4000 m perturbs ``zn`` by ``~3e-3 kg/m³`` — negligible vs
    the ~1 kg/m³ density signal, and in the physically correct direction (real
    pressure, not geometric depth, sets compressibility).  This matches how
    ``wright``/``unesco80``/``veros_*`` already consume ``p``.

    Parameters
    ----------
    T : array — Potential temperature [°C].
    S : array — Practical salinity [PSU].
    p : array — Pressure [Pa]; depth recovered as ``p/(ρ0·g)``.
    cfg : NemoSEOSConfig — Coefficients (defaults to the DINO set).

    Returns
    -------
    array : In-situ density [kg/m³].
    """
    if cfg is None:
        cfg = NemoSEOSConfig()
    zt = T - cfg.T0
    zs = S - cfg.S0
    zh = p / (cfg.rho0 * constants.g)   # Boussinesq depth [m, positive down]
    zn = (
        -cfg.a0 * (1.0 + 0.5 * cfg.lambda1 * zt + cfg.mu1 * zh) * zt
        + cfg.b0 * (1.0 - 0.5 * cfg.lambda2 * zs - cfg.mu2 * zh) * zs
        - cfg.nu * zt * zs
    )
    return cfg.rho0 + zn


def nemo_seos_prd_literal(
    T: jnp.ndarray,
    S: jnp.ndarray,
    depth_m: jnp.ndarray,
    cfg: NemoSEOSConfig | None = None,
) -> jnp.ndarray:
    """NEMO simplified-EOS density anomaly ``prd`` in literal association.

    This is the quantity written by ``eosbn2.F90:301-305``: NEMO forms
    ``zn`` directly from conservative temperature, salinity, and geometric
    ``gdept``, then stores ``zn * r1_rho0``.  It deliberately does *not* call
    :func:`nemo_seos_eos` and recover the anomaly through
    ``(rho0 + zn) / rho0 - 1``; that non-oracle round trip loses bits to the
    large reference-density offset before the anomaly is consumed by
    ``ldf_slp``.

    The helper is pure JAX and retains the source expression's written order,
    so it is safe under JIT and reverse-mode AD.
    """
    if cfg is None:
        cfg = NemoSEOSConfig()
    zt = lax.optimization_barrier(T - cfg.T0)
    zs = lax.optimization_barrier(S - cfg.S0)
    t_linear = lax.optimization_barrier(0.5 * cfg.lambda1 * zt)
    t_depth = lax.optimization_barrier(cfg.mu1 * depth_m)
    t_factor = lax.optimization_barrier(1.0 + t_linear + t_depth)
    t_term = lax.optimization_barrier(-cfg.a0 * t_factor * zt)
    s_linear = lax.optimization_barrier(0.5 * cfg.lambda2 * zs)
    s_depth = lax.optimization_barrier(cfg.mu2 * depth_m)
    s_factor = lax.optimization_barrier(1.0 - s_linear - s_depth)
    s_term = lax.optimization_barrier(cfg.b0 * s_factor * zs)
    cross = lax.optimization_barrier(cfg.nu * zt * zs)
    zn = lax.optimization_barrier(t_term + s_term - cross)
    return lax.optimization_barrier(zn * (1.0 / cfg.rho0))


def nemo_seos_alpha_beta(
    T: jnp.ndarray,
    S: jnp.ndarray,
    gdept: jnp.ndarray,
    cfg: NemoSEOSConfig | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Thermal/haline expansion coefficients ``(alpha, beta)`` — NEMO ``rab``.

    Transcribes NEMO ``eosbn2.F90`` ``rab_3d_t`` (``np_seos`` branch,
    lines 1161-1173): the analytic ``(T, S)`` derivatives of the S-EOS
    density anomaly :func:`nemo_seos_eos`, normalised by the constant
    reference density ``rn_rho0`` (NEMO's ``r1_rho0``, NOT ``1/rho``)::

        alpha = [ a0 (1 + lambda1 zt + mu1 zh) + nu zs ] / rho0     (jp_tem)
        beta  = [ b0 (1 - lambda2 zs - mu2 zh) - nu zt ] / rho0     (jp_sal)

    with ``zt = T - T0``, ``zs = S - S0`` and ``zh = gdept`` the geometric
    T-point depth [m, positive down] — each cell's OWN depth (the thermobaric
    ``mu`` term). Note the ``½ lambda1`` prefactor in the DENSITY polynomial
    (:func:`nemo_seos_eos`) becomes the FULL ``lambda1`` here because
    ``d(½ lambda1 zt²)/dzt = lambda1 zt`` — the sign of ``alpha`` follows
    ``alpha = -(1/rho0) dρ/dT`` so denser-when-warmer never occurs for the
    DINO set.

    Fully differentiable (arithmetic + the passed ``gdept``); shares the
    canonical :class:`NemoSEOSConfig` with the density EOS — no re-derivation.

    Parameters
    ----------
    T, S : array — Potential temperature [°C] / salinity [PSU] at T-points.
    gdept : array — Geometric T-point depth [m, positive down], broadcastable
        against ``T`` on the last (vertical) axis.
    cfg : NemoSEOSConfig — Coefficients (defaults = the DINO/Kamm set).

    Returns
    -------
    (alpha, beta) : arrays [1/°C], [1/PSU], same shape as ``T``.
    """
    if cfg is None:
        cfg = NemoSEOSConfig()
    zt = T - cfg.T0
    zs = S - cfg.S0
    inv_rho0 = 1.0 / cfg.rho0
    alpha = (cfg.a0 * (1.0 + cfg.lambda1 * zt + cfg.mu1 * gdept)
             + cfg.nu * zs) * inv_rho0
    beta = (cfg.b0 * (1.0 - cfg.lambda2 * zs - cfg.mu2 * gdept)
            - cfg.nu * zt) * inv_rho0
    return alpha, beta


def _nemo_bn2_zrw(
    gdept: jnp.ndarray,
    gdepw_int: jnp.ndarray,
    *,
    evaluation: str,
    gdept_0: jnp.ndarray | None = None,
    gdepw_0: jnp.ndarray | None = None,
    stretch: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Evaluate eosbn2.F90:1459-1460 with a selectable rounding boundary."""
    gd = jnp.asarray(gdept)
    if evaluation == "preassembled_live":
        if any(value is not None for value in (gdept_0, gdepw_0, stretch)):
            raise ValueError(
                "raw zrw operands require evaluation='nemo_literal'")
        return ((jnp.asarray(gdepw_int) - gd[..., 1:])
                / (gd[..., :-1] - gd[..., 1:]))
    if evaluation != "nemo_literal":
        raise ValueError(
            f"unknown zrw_evaluation {evaluation!r}; expected "
            "'preassembled_live' or 'nemo_literal'")
    if any(value is None for value in (gdept_0, gdepw_0, stretch)):
        raise ValueError(
            "zrw_evaluation='nemo_literal' requires zrw_gdept_0, "
            "zrw_gdepw_0, and zrw_stretch")
    stretch3 = jnp.asarray(stretch)[..., jnp.newaxis]
    gd0 = jnp.asarray(gdept_0)
    gw0 = jnp.asarray(gdepw_0)
    gd_up = lax.optimization_barrier(gd0[..., :-1] * stretch3)
    gd_lo = lax.optimization_barrier(gd0[..., 1:] * stretch3)
    gw = lax.optimization_barrier(gw0 * stretch3)
    return (gw - gd_lo) / (gd_up - gd_lo)


def compute_buoyancy_frequency_nemo_bn2(
    T: jnp.ndarray,
    S: jnp.ndarray,
    gdept: jnp.ndarray,
    gdepw_int: jnp.ndarray,
    cfg: NemoSEOSConfig | None = None,
    g: float = constants.g,
    eos_form: str = "seos",
    *,
    e3w_int: jnp.ndarray | None = None,
    e3w_source: str = "mesh_reference",
    zrw_evaluation: str = "preassembled_live",
    zrw_gdept_0: jnp.ndarray | None = None,
    zrw_gdepw_0: jnp.ndarray | None = None,
    zrw_stretch: jnp.ndarray | None = None,
    _alpha_beta_override: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    _return_intermediate: str = "",
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    r"""Brunt-Väisälä ``N²`` by NEMO's exact ``bn2``.

    ``eos_form`` selects where alpha/beta come from. ``"seos"`` (default,
    BIT-IDENTICAL legacy) uses the 3-term simplified EOS. ``"teos10"`` uses
    NEMO's Roquet polynomial with the TEOS-10 coefficient set
    (:func:`nemo_roquet_alpha_beta`) -- what ORCA1 actually runs
    (``ln_teos10 = .true.``). The bn2 ASSEMBLY is identical either way; NEMO
    shares ``bn2_t`` across EOS branches and only ``pab`` differs.

    Transcribes NEMO ``eosbn2.F90`` ``bn2_t`` (lines 1459-1467; the last line
    is the ``/ e3w(...,Kmm) * wmask`` continuation)::

        zrw = (gdepw_k − gdept_k) / (gdept_{k-1} − gdept_k)
        alpha_w = alpha_k (1 − zrw) + alpha_{k-1} zrw
        beta_w  = beta_k  (1 − zrw) + beta_{k-1}  zrw
        N²_k = g ( alpha_w ΔT − beta_w ΔS ) / e3w_k

    where ``alpha``, ``beta`` (:func:`nemo_seos_alpha_beta`) are evaluated at
    EACH T-cell's OWN geometric depth ``gdept`` and interpolated to the
    w-interface by the geometric weight ``zrw``; ``ΔT = T_upper − T_lower``.
    This differs from :func:`compute_buoyancy_frequency_adiabatic`, which
    displaces both parcels to a single reference pressure and differences the
    full nonlinear density — near-neutral marginal cells flip between the two.

    Index convention: legoESM interior interface ``i`` sits between the UPPER
    cell ``i`` (shallower, NEMO ``jk-1``) and the LOWER cell ``i+1`` (NEMO
    ``jk``); the w-point is NEMO ``jk``. So ``e3w = gdept[i+1] − gdept[i]`` is
    the (positive) centre-to-centre spacing and ``gdepw_int[i]`` is the
    interface depth. **Signed** — ``N² < 0`` marks a statically unstable
    interface (the convection / TKE trigger); NOT clipped.

    Fully ``jax.grad``-safe (arithmetic through the EOS coefficients + T/S).

    Parameters
    ----------
    T, S : array — T-point potential temperature [°C] / salinity [PSU],
        shape ``(..., nlev)``.
    gdept : array — Geometric T-point depths [m, positive down], shape
        ``(nlev,)`` or ``(..., nlev)``.
    gdepw_int : array — Interior w-interface depths [m, positive down], shape
        ``(nlev-1,)`` or ``(..., nlev-1)`` (== ``|z_half_ref[1:-1]|``).
    cfg : NemoSEOSConfig — S-EOS coefficients (defaults = the DINO/Kamm set).
    g : float — Gravitational acceleration [m/s²].

    Returns
    -------
    array : signed ``N²`` at interior interfaces [1/s²], shape ``(..., nlev-1)``.
    """
    if e3w_source not in ("mesh_reference", "depth_difference"):
        raise ValueError(
            f"unknown e3w_source {e3w_source!r}; expected 'mesh_reference' "
            "or 'depth_difference'")
    if e3w_source == "mesh_reference":
        if e3w_int is None:
            raise ValueError(
                "e3w_source='mesh_reference' requires raw-mesh e3w_int; "
                "the depth-difference construction is legacy opt-in")
        e3w = jnp.asarray(e3w_int)
        if e3w.shape[-1] != T.shape[-1] - 1:
            raise ValueError(
                f"e3w_int trailing size {e3w.shape[-1]} != nlev-1="
                f"{T.shape[-1] - 1}")
        e3w = eqx.error_if(
            e3w,
            ~jnp.all(jnp.isfinite(e3w) & (e3w > 0.0)),
            "raw-mesh e3w_int must contain only finite values > 0",
        )
    else:
        if e3w_int is not None:
            raise ValueError(
                "e3w_int must be omitted when e3w_source='depth_difference'")
        e3w = None
    if eos_form not in ("seos", "teos10"):
        raise ValueError(
            f"compute_buoyancy_frequency_nemo_bn2 eos_form={eos_form!r} "
            "invalid; expected 'seos' (the 3-term simplified EOS) or "
            "'teos10' (NEMO's Roquet polynomial with the TEOS-10 coefficient "
            "set, which is what ORCA1 runs: ln_teos10=.true.).")
    eos_gdept = jnp.asarray(gdept)
    if zrw_evaluation == "nemo_literal":
        if zrw_gdept_0 is None or zrw_stretch is None:
            raise ValueError(
                "zrw_evaluation='nemo_literal' requires zrw_gdept_0 and "
                "zrw_stretch for the literal eos_rab depth")
        eos_gdept = lax.optimization_barrier(
            jnp.asarray(zrw_gdept_0)
            * jnp.asarray(zrw_stretch)[..., jnp.newaxis])
    if _alpha_beta_override is not None:
        alpha, beta = (
            jnp.asarray(_alpha_beta_override[0]),
            jnp.asarray(_alpha_beta_override[1]),
        )
        if alpha.shape != T.shape or beta.shape != T.shape:
            raise ValueError(
                "the private bn2 alpha/beta override must match T/S shape; "
                f"got alpha={alpha.shape}, beta={beta.shape}, T={T.shape}")
    elif eos_form == "teos10":
        # NEMO's rab_3d takes the GEOMETRIC depth, and both alpha and beta come
        # from the polynomial rather than the 3-term fit. Everything below this
        # line -- the zrw interpolation, the /e3w, the sign convention -- is
        # unchanged, because NEMO's bn2_t is shared across EOS branches.
        alpha, beta = nemo_roquet_alpha_beta(T, S, eos_gdept)
    else:
        if cfg is None:
            cfg = NemoSEOSConfig()
        alpha, beta = nemo_seos_alpha_beta(
            T, S, eos_gdept, cfg)  # (..., nlev)
    gd = jnp.asarray(gdept)
    gd_up = gd[..., :-1]                                       # cell i  (upper)
    gd_lo = gd[..., 1:]                                        # cell i+1 (lower)
    zrw = _nemo_bn2_zrw(
        gd, gdepw_int, evaluation=zrw_evaluation,
        gdept_0=zrw_gdept_0, gdepw_0=zrw_gdepw_0,
        stretch=zrw_stretch)
    a_w = alpha[..., 1:] * (1.0 - zrw) + alpha[..., :-1] * zrw
    b_w = beta[..., 1:] * (1.0 - zrw) + beta[..., :-1] * zrw
    if e3w is None:
        e3w = gd_lo - gd_up                                    # explicit legacy
    dT = T[..., :-1] - T[..., 1:]                              # T_upper - T_lower
    dS = S[..., :-1] - S[..., 1:]
    temperature_contribution = a_w * dT
    salinity_contribution = b_w * dS
    contribution_difference = temperature_contribution - salinity_contribution
    gravity_product = g * contribution_difference
    thickness_division = gravity_product / e3w
    if not _return_intermediate:
        return thickness_division
    intermediates = {
        "zrw": zrw,
        "zaw": a_w,
        "zbw": b_w,
        "temperature_contribution": temperature_contribution,
        "salinity_contribution": salinity_contribution,
        "contribution_difference": contribution_difference,
        "gravity_product": gravity_product,
        "thickness_division": thickness_division,
        # The shared implementation does not yet apply NEMO's trailing wmask;
        # this selector intentionally exposes the value production returns so
        # the gate can score that missing source boundary, including -0/+0.
        "masked_rn2": thickness_division,
    }
    if _return_intermediate not in intermediates:
        raise ValueError(
            f"unknown private bn2 intermediate {_return_intermediate!r}; "
            f"expected one of {tuple(intermediates)}")
    return thickness_division, intermediates[_return_intermediate]


def nemo_bn2_live_geometry(
    z_coord, eta: jnp.ndarray, H_bathy: jnp.ndarray,
    *, r3t_evaluation: str = "quotient",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return canonical live ``(gdept, gdepw_int, e3w_int)`` for NEMO bn2.

    The faithful/default divisor is the raw mesh ``e3w_0(:,:,jk)`` multiplied
    by NEMO's live ``1+r3t``.  ``depth_difference`` is an explicit legacy
    coordinate option and reproduces the former ``diff(gdept)`` construction.
    """
    gdept, gdepw_int = nemo_bn2_live_ladders(
        z_coord, eta, H_bathy, r3t_evaluation=r3t_evaluation)
    stretch = (None if getattr(z_coord, "linear_free_surface", False)
               else nemo_r3t_stretch(
                   z_coord, eta, H_bathy,
                   evaluation=r3t_evaluation))
    e3w = nemo_e3w_from_live_gdept(
        z_coord, gdept, stretch=stretch, interior=True)
    return gdept, gdepw_int, e3w


def nemo_e3w_from_live_gdept(
    z_coord, live_gdept: jnp.ndarray, *, stretch: jnp.ndarray | None,
    interior: bool = True,
) -> jnp.ndarray:
    """Select NEMO W spacing once for bn2 and every paired consumer."""
    mesh_reference = getattr(z_coord, "nemo_e3w_mesh_reference", False)
    if not mesh_reference:
        if interior:
            return jnp.diff(live_gdept, axis=-1)
        return jnp.concatenate(
            [2.0 * live_gdept[..., :1], jnp.diff(live_gdept, axis=-1)],
            axis=-1)
    raw = getattr(z_coord, "nemo_e3w_0", None)
    if raw is None:
        raise ValueError(
            "nemo_e3w_source='mesh_reference' requires z_coord.nemo_e3w_0; "
            "use the legacy option explicitly only when no raw mesh exists")
    raw = jnp.asarray(raw)
    if raw.shape[-1] != z_coord.n_levels:
        raise ValueError(
            f"z_coord.nemo_e3w_0 trailing size {raw.shape[-1]} != n_levels="
            f"{z_coord.n_levels}")
    # NEMO jk=2..jpk maps to Python raw[...,1:] for the interior bn2 rows;
    # ldf_slp also needs jk=1 and requests the complete raw field.
    e3w = raw[..., 1:] if interior else raw
    if stretch is not None:
        e3w = e3w * jnp.asarray(stretch)[..., None]
    return e3w


def nemo_bn2_depth_ladders(z_coord) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(gdept, gdepw_int)`` geometric depth ladders for the NEMO ``bn2`` N².

    ``gdept`` = ``z_coord.t_depth_ref`` (the exact NEMO ``gdept_1d``) when a
    fidelity coordinate supplies it, else the interface-midpoint
    ``|z_full_ref|``; ``gdepw_int`` = ``|z_half_ref[1:-1]|`` (interior
    w-interface depths). Both positive-down [m], static ``(nlev,)`` /
    ``(nlev-1,)`` grid quantities.

    These are the STATIC REFERENCE ladders (NEMO ``gdept_1d``/``gdepw_1d``).
    NEMO's ``bn2``/``rab_3d_t`` evaluate at the LIVE ``gdept(Kmm)``; under
    ``key_linssh`` that equals ``gdept_1d`` exactly, but under z* (``key_qco``)
    it is the stretched ladder — use :func:`nemo_bn2_live_ladders` there.
    Measured against NEMO's own ``kt==nit000`` ``gdept(Kmm)`` dump on DINO y5,
    the static ladder is off by median ``1.5e-4`` relative vs ``2.5e-8`` for
    the stretched one (#1226).
    """
    z_full = jnp.abs(z_coord.z_full_ref)
    t_depth = getattr(z_coord, "t_depth_ref", None)
    gdept = z_full if t_depth is None else jnp.asarray(t_depth)
    gdepw_int = jnp.abs(z_coord.z_half_ref[1:-1])
    return gdept, gdepw_int


def nemo_r3t_stretch(
    z_coord, eta: jnp.ndarray, H_bathy: jnp.ndarray, *,
    evaluation: str = "quotient",
) -> jnp.ndarray:
    """NEMO ``key_qco`` T-point z* stretch factor ``(1 + r3t)``.

    ``r3t = ssh/ht_0`` (``domqco.F90:160``, ``dom_qco_r3c``), used throughout
    ``domzgr_substitute.h90`` macros as ``e3t(i,j,k,t) = e3t_0(i,j,k)*(1+r3t
    (i,j,t))`` (``:139``) and ``gdept(i,j,k,t) = gdept_0(i,j,k)*(1+r3t(i,j,t))``
    (``:145``/``:56``).  Standalone extraction of the stretch factor
    previously embedded in :func:`nemo_bn2_live_ladders` (#1226 round-2)
    so OTHER live-thickness/live-depth sites (e.g. the DINO ``trasbc.F90``
    surface-flux divisor, #1226) reuse the identical formula instead of
    re-deriving ``r3t``.

    ``r3t`` uses the LOCAL column depth ``H_bathy`` (NEMO ``ht_0``) — NOT the
    z* Jacobian from :func:`~legoesm.ocean.vertical.compute_ocean_jacobian`,
    which is ``(eta + H_bathy)/H_max`` (normalised by the GLOBAL maximum
    depth) and is a different quantity: using it here is off by median 1.1e-1
    relative vs 2.5e-8 for this form, measured against NEMO's own
    ``gdept(Kmm)`` dump (#1226).

    Parameters
    ----------
    z_coord : vertical coordinate — only ``linear_free_surface`` is read.
    eta : array ``(...)`` — sea-surface height [m] at the SAME time level
        NEMO evaluates.
    H_bathy : array ``(...)`` — local column depth [m], positive.

    Returns
    -------
    array ``(...)`` — the stretch factor ``(1 + r3t)``, floored at 1e-6.
    """
    if evaluation not in ("quotient", "nemo_reciprocal"):
        raise ValueError(
            f"unknown r3t evaluation {evaluation!r}; expected 'quotient' "
            "or 'nemo_reciprocal'")
    if getattr(z_coord, "linear_free_surface", False):
        # NEMO key_linssh: domqco is NOT active, so r3t == 0 -- the column
        # never stretches.  Mirrors the same special case in
        # vertical.compute_ocean_jacobian (:813-815).
        return jnp.ones_like(jnp.asarray(eta))
    # Dry columns (H_bathy == 0) -> r3t = 0 (inert; all their cells are masked)
    # rather than eta/0 -> inf/NaN poisoning the downstream chain.
    H = jnp.asarray(H_bathy)
    wet_H = H > 0.0
    safe_H = jnp.where(wet_H, H, 1.0)
    if evaluation == "nemo_reciprocal":
        # domain.F90:158 stores r1_ht_0 before domqco.F90:160 multiplies it
        # by ssh.  The barrier preserves that fp64 boundary under JIT.
        r1_H = lax.optimization_barrier(1.0 / safe_H)
        r3t = jnp.where(wet_H, jnp.asarray(eta) * r1_H, 0.0)
    else:
        r3t = jnp.where(wet_H, jnp.asarray(eta) / safe_H, 0.0)
    # Safety floor on the stretch, NOT on r3t: a column driven to
    # eta + H_bathy <= 0 (unclamped restart/IC, wetting-drying) would give a
    # NON-POSITIVE geometric depth/thickness, which silently flips the sign
    # of downstream terms.  Callers that already clamp eta (the PGF passes
    # eta_safe) never reach this; it exists so an unclamped caller degrades
    # loudly-wrong rather than silently-plausible.
    return jnp.maximum(1.0 + r3t, 1.0e-6)


def nemo_bn2_live_ladders(
    z_coord, eta: jnp.ndarray, H_bathy: jnp.ndarray,
    *, r3t_evaluation: str = "quotient",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(gdept, gdepw_int)`` at NEMO's LIVE ``gdept(Kmm)`` under z*.

    NEMO ``key_qco`` (no ``key_isf``) expands ``gdept(i,j,k,t)`` to the PURE
    multiplicative stretch ``gdept_0*(1 + r3t)`` with ``r3t = ssh/ht_0``
    (``domzgr_substitute.h90:139`` with ``Tisf -> Time()`` at ``:56``,
    ``domqco.F90:160``).  There is NO ``-ssh`` shift: that is the SEPARATE
    ``gdept_z0`` macro (``:145``), a z=0-referenced depth whose only consumer
    is ``dynhpg.F90``.  ``gdept`` is depth below the instantaneous free
    surface, which is the pressure proxy the S-EOS wants
    (``eosbn2.F90:1166`` uses the plain ``gdept`` macro).

    Parameters
    ----------
    z_coord : vertical coordinate — supplies the static reference ladders.
    eta : array ``(...)`` — sea-surface height [m] at the SAME time level
        NEMO evaluates (``Kmm``; ``rab_b``/``rn2b`` are BEFORE-level).
    H_bathy : array ``(...)`` — local column depth [m], positive.

    Returns
    -------
    tuple of array — ``(gdept, gdepw_int)``, shapes ``(..., nlev)`` /
    ``(..., nlev-1)``, positive-down [m].
    """
    gdept, gdepw_int = nemo_bn2_depth_ladders(z_coord)
    if getattr(z_coord, "linear_free_surface", False):
        # NEMO key_linssh: domqco is NOT active, so r3t == 0 and
        # gdept(Kmm) == gdept_0 EXACTLY -- the column never stretches.
        # Short-circuit BEFORE calling nemo_r3t_stretch: that helper
        # broadcasts against eta's shape even when returning all-ones,
        # which would change the STATIC ladders' shape here (adversarial
        # review finding F1 regression guard,
        # test_linear_free_surface_column_never_stretches).
        return gdept, gdepw_int
    stretch = nemo_r3t_stretch(
        z_coord, eta, H_bathy,
        evaluation=r3t_evaluation)[..., jnp.newaxis]
    return gdept * stretch, gdepw_int * stretch


# ==============================================================================
# NEMO polynomial EOS — Roquet et al. (2015, Ocean Modelling 90:29-43) 55-term
# seawater polynomial, EOS-80 coefficient set.  This is the FULL polynomial NEMO
# runs under ``ln_eos80`` (potential temperature + practical salinity,
# ``l_useCT=.FALSE.``) — distinct from the 3-term ``nemo_seos`` simplified EOS
# above, and from ``veros_gsw`` (a DIFFERENT TEOS-10 fit, the GSW 48-term
# rational polynomial).  Transcribed verbatim from NEMO 5.0.2
# ``src/OCE/TRA/eosbn2.F90``: normalization :2117-2120, coefficients :2122-2173,
# Horner evaluation :260-288.  The selectable ``nemo_teos10`` branch below uses
# the companion TEOS-10 coefficient set (eosbn2.F90:1920-2108, Conservative
# Temperature + Absolute Salinity) with this same evaluator.
# ==============================================================================
_ROQUET_EOS80 = {
    # normalization
    "r1_S0": 1.0 / 40.0, "r1_T0": 1.0 / 40.0, "r1_Z0": 1.0e-4, "rdeltaS": 20.0,
    # zn0 (depth^0)
    "EOS000": 9.5356891948e+02, "EOS100": 1.7136499189e+02, "EOS200": -3.7501039454e+02,
    "EOS300": 5.1856810420e+02, "EOS400": -3.7264470465e+02, "EOS500": 1.4302533998e+02,
    "EOS600": -2.2856621162e+01,
    "EOS010": 1.0087518651e+01, "EOS110": -1.3647741861e+01, "EOS210": 8.8478359933,
    "EOS310": -7.2329388377, "EOS410": 1.4774410611, "EOS510": 2.0036720553e-01,
    "EOS020": -2.5579830599e+01, "EOS120": 2.4043512327e+01, "EOS220": -1.6807503990e+01,
    "EOS320": 8.3811577084, "EOS420": -1.9771060192,
    "EOS030": 1.6846451198e+01, "EOS130": -2.1482926901e+01, "EOS230": 1.0108954054e+01,
    "EOS330": -6.2675951440e-01,
    "EOS040": -8.0812310102, "EOS140": 1.0102374985e+01, "EOS240": -4.8340368631,
    "EOS050": 1.2079167803, "EOS150": 1.1515380987e-01, "EOS060": -2.4520288837e-01,
    # zn1 (depth^1)
    "EOS001": 1.0748601068e+01, "EOS101": -1.7817043500e+01, "EOS201": 2.2181366768e+01,
    "EOS301": -1.6750916338e+01, "EOS401": 4.1202230403,
    "EOS011": -1.5852644587e+01, "EOS111": -7.6639383522e-01, "EOS211": 4.1144627302,
    "EOS311": -6.6955877448e-01,
    "EOS021": 9.9994861860, "EOS121": -1.9467067787e-01, "EOS221": -1.2177554330,
    "EOS031": -3.4866102017, "EOS131": 2.2229155620e-01, "EOS041": 5.9503008642e-01,
    # zn2 (depth^2)
    "EOS002": 1.0375676547, "EOS102": -3.4249470629, "EOS202": 2.0542026429,
    "EOS012": 2.1836324814, "EOS112": -3.4453674320e-01, "EOS022": -1.2548163097,
    # zn3 (depth^3)
    "EOS003": 1.8729078427e-02, "EOS103": -5.7238495240e-02, "EOS013": 3.8306136687e-01,
    # thermal expansion / haline contraction, eosbn2.F90:2175-2246
    "ALP000": -2.5218796628e-01, "ALP100": 3.4119354654e-01,
    "ALP200": -2.2119589983e-01, "ALP300": 1.8082347094e-01,
    "ALP400": -3.6936026529e-02, "ALP500": -5.0091801383e-03,
    "ALP010": 1.2789915300, "ALP110": -1.2021756164,
    "ALP210": 8.4037519952e-01, "ALP310": -4.1905788542e-01,
    "ALP410": 9.8855300959e-02, "ALP020": -1.2634838399,
    "ALP120": 1.6112195176, "ALP220": -7.5817155402e-01,
    "ALP320": 4.7006963580e-02, "ALP030": 8.0812310102e-01,
    "ALP130": -1.0102374985, "ALP230": 4.8340368631e-01,
    "ALP040": -1.5098959754e-01, "ALP140": -1.4394226233e-02,
    "ALP050": 3.6780433255e-02, "ALP001": 3.9631611467e-01,
    "ALP101": 1.9159845880e-02, "ALP201": -1.0286156825e-01,
    "ALP301": 1.6738969362e-02, "ALP011": -4.9997430930e-01,
    "ALP111": 9.7335338937e-03, "ALP211": 6.0887771651e-02,
    "ALP021": 2.6149576513e-01, "ALP121": -1.6671866715e-02,
    "ALP031": -5.9503008642e-02, "ALP002": -5.4590812035e-02,
    "ALP102": 8.6134185799e-03, "ALP012": 6.2740815484e-02,
    "ALP003": -9.5765341718e-03,
    "BET000": 2.1420623987, "BET100": -9.3752598635,
    "BET200": 1.9446303907e+01, "BET300": -1.8632235232e+01,
    "BET400": 8.9390837485, "BET500": -1.7142465871,
    "BET010": -1.7059677327e-01, "BET110": 2.2119589983e-01,
    "BET210": -2.7123520642e-01, "BET310": 7.3872053057e-02,
    "BET410": 1.2522950346e-02, "BET020": 3.0054390409e-01,
    "BET120": -4.2018759976e-01, "BET220": 3.1429341406e-01,
    "BET320": -9.8855300959e-02, "BET030": -2.6853658626e-01,
    "BET130": 2.5272385134e-01, "BET230": -2.3503481790e-02,
    "BET040": 1.2627968731e-01, "BET140": -1.2085092158e-01,
    "BET050": 1.4394226233e-03, "BET001": -2.2271304375e-01,
    "BET101": 5.5453416919e-01, "BET201": -6.2815936268e-01,
    "BET301": 2.0601115202e-01, "BET011": -9.5799229402e-03,
    "BET111": 1.0286156825e-01, "BET211": -2.5108454043e-02,
    "BET021": -2.4333834734e-03, "BET121": -3.0443885826e-02,
    "BET031": 2.7786444526e-03, "BET002": -4.2811838287e-02,
    "BET102": 5.1355066072e-02, "BET012": -4.3067092900e-03,
    "BET003": -7.1548119050e-04,
}


def _nemo_roquet_eos_impl(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    *,
    coeffs: dict | None = None,
    rho0: float = rho_0,
    geometric_depth_m: jnp.ndarray | None = None,
    return_intermediates: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
    """In-situ density [kg/m³] from the NEMO Roquet-55 polynomial EOS (EOS-80).

    Reproduces NEMO ``eos_insitu`` (``eosbn2.F90``): the Horner sum ``zn`` in
    ``(zt, zs, zh)`` IS the in-situ density in kg/m³ (NEMO stores the anomaly
    ``prd = zn/ρ0 - 1``; we return the density itself).  Inputs are potential
    temperature and practical salinity, matching the EOS-80 set
    (``l_useCT=.FALSE.``) — no CT/SA conversion.

    Depth enters as ``zh = (p/(ρ0·g))·r1_Z0`` — the same Boussinesq depth
    reconstruction ``nemo_seos_eos`` uses (NEMO uses geometric ``gdept``; the two
    differ by ``O(ρ'/ρ0) ≈ 0.3 %``).  For a tendency certificate that needs
    NEMO's exact ``gdept``, pass ``p = ρ0·g·gdept`` so ``zh`` recovers ``gdept``
    to round-off.

    Parameters
    ----------
    T : array — Potential temperature [°C].
    S : array — Practical salinity [PSU].
    p : array — Pressure [Pa]; depth recovered as ``p/(ρ0·g)``.
    coeffs : dict — Roquet coefficient set; defaults to :data:`_ROQUET_EOS80`.
    rho0 : float — Boussinesq reference density [kg/m³] for the depth
        reconstruction.  Defaults to legoESM ``rho_0`` (1025); NEMO's
        ``rn_rho0`` default is 1026 — pass ``rho0=1026.0`` for exact NEMO/DINO
        parity.  Enters ONLY the ``zh`` depth term (a ~0.1 % effect on the
        thermobaric correction), never the polynomial coefficients.

    Notes
    -----
    Like NEMO's ``SQRT(ABS(S+rdeltaS))``, the salinity term is not clipped: for
    all physical salinity ``S + 20 > 0`` so the gradient is finite, but at the
    unphysical ``S = -20`` the ``abs`` kink meets the ``sqrt`` singularity and
    ``drho/dS`` diverges.  Kept unclipped for bit-faithfulness to NEMO.

    Returns
    -------
    array : In-situ density [kg/m³].
    """
    c = _ROQUET_EOS80 if coeffs is None else coeffs
    depth_m = (
        p / (rho0 * constants.g)
        if geometric_depth_m is None
        else jnp.asarray(geometric_depth_m, dtype=T.dtype)
    )
    zh = lax.optimization_barrier(depth_m * c["r1_Z0"])
    zt = lax.optimization_barrier(T * c["r1_T0"])
    # S + rdeltaS > 0 for all physical S (rdeltaS = 20), so abs() never kinks
    # and sqrt of a strictly-positive argument keeps the gradient finite.
    zs = lax.optimization_barrier(
        jnp.sqrt(jnp.abs(S + c["rdeltaS"]) * c["r1_S0"]))
    # Horner form (NEMO eosbn2.F90:265-286).  The oracle is compiled without
    # FMA/fast-math and rounds every source multiply/add.  A barrier only after
    # each completed znN is too late: XLA may contract/reassociate *inside* the
    # long statement (the GYRE operand dump first saw 4 ulp at zn0).  These two
    # tiny helpers retain every source operation while keeping one evaluator.
    def _fadd(first, *rest):
        value = first
        for term in rest:
            value = nemo_source_round(value + term)
        return value

    def _fmul(left, right):
        return nemo_source_round(left * right)

    def _horner(variable, *coefficients):
        value = coefficients[0]
        for coefficient in coefficients[1:]:
            value = _fadd(_fmul(value, variable), coefficient)
        return value

    zn3 = _fadd(_fmul(c["EOS013"], zt), _fmul(c["EOS103"], zs), c["EOS003"])
    zn2 = _fadd(
        _fmul(_fadd(_fmul(c["EOS022"], zt), _fmul(c["EOS112"], zs),
                    c["EOS012"]), zt),
        _fmul(_fadd(_fmul(c["EOS202"], zs), c["EOS102"]), zs),
        c["EOS002"],
    )
    zn1_1 = _fadd(
        _fmul(c["EOS041"], zt), _fmul(c["EOS131"], zs), c["EOS031"])
    zn1_2 = _fadd(
        _fmul(zn1_1, zt),
        _fmul(_fadd(_fmul(c["EOS221"], zs), c["EOS121"]), zs),
        c["EOS021"],
    )
    zn1_3 = _fadd(
        _fmul(zn1_2, zt),
        _fmul(_horner(zs, c["EOS311"], c["EOS211"], c["EOS111"]), zs),
        c["EOS011"],
    )
    zn1 = _fadd(
        _fmul(zn1_3, zt),
        _fmul(_horner(zs, c["EOS401"], c["EOS301"], c["EOS201"],
                      c["EOS101"]), zs),
        c["EOS001"],
    )
    zn0_1 = _fadd(
        _fmul(c["EOS060"], zt), _fmul(c["EOS150"], zs), c["EOS050"])
    zn0_2 = _fadd(
        _fmul(zn0_1, zt),
        _fmul(_fadd(_fmul(c["EOS240"], zs), c["EOS140"]), zs),
        c["EOS040"],
    )
    zn0_3 = _fadd(
        _fmul(zn0_2, zt),
        _fmul(_horner(zs, c["EOS330"], c["EOS230"], c["EOS130"]), zs),
        c["EOS030"],
    )
    zn0_4 = _fadd(
        _fmul(zn0_3, zt),
        _fmul(_horner(zs, c["EOS420"], c["EOS320"], c["EOS220"],
                      c["EOS120"]), zs),
        c["EOS020"],
    )
    zn0_5 = _fadd(
        _fmul(zn0_4, zt),
        _fmul(_horner(zs, c["EOS510"], c["EOS410"], c["EOS310"],
                      c["EOS210"], c["EOS110"]), zs),
        c["EOS010"],
    )
    zn0 = _fadd(
        _fmul(zn0_5, zt),
        _fmul(_horner(zs, c["EOS600"], c["EOS500"], c["EOS400"],
                      c["EOS300"], c["EOS200"], c["EOS100"]), zs),
        c["EOS000"],
    )
    zn = _horner(zh, zn3, zn2, zn1, zn0)
    if return_intermediates:
        # WRITE-only diagnostic seam in the source order of NEMO 5.0.2
        # eosbn2.F90:260-287.  The production result is still the same final
        # ``zn`` object; callers cannot replace any operand through this seam.
        return depth_m, zh, zt, zs, zn0, zn1, zn2, zn3, zn
    return zn   # in-situ density [kg/m³]


_nemo_roquet_eos_compiled = jax.jit(
    _nemo_roquet_eos_impl, static_argnames=("return_intermediates",))


def nemo_roquet_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    *,
    coeffs: dict | None = None,
    rho0: float = rho_0,
    geometric_depth_m: jnp.ndarray | None = None,
    return_intermediates: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
    """Evaluate the shared NEMO Roquet polynomial with one arithmetic regime.

    ``lax.optimization_barrier`` is intentionally a no-op under an outer
    :func:`jax.disable_jit` context.  Running the literal polynomial eagerly
    consequently changed its multiply/add contraction and, after hydrostatic
    integration, produced a false 1e-9--1e-7 trajectory split.  Re-enable the
    cached literal kernel locally so eager diagnostics and production JIT call
    the same compiled arithmetic.  This remains a single differentiable JAX
    implementation; it is neither a card selector nor a host callback.
    """
    if isinstance(T, jax.core.Tracer):
        return _nemo_roquet_eos_impl(
            T, S, p, coeffs=coeffs, rho0=rho0,
            geometric_depth_m=geometric_depth_m,
            return_intermediates=return_intermediates)
    with jax.disable_jit(False):
        return _nemo_roquet_eos_compiled(
            T, S, p, coeffs=coeffs, rho0=rho0,
            geometric_depth_m=geometric_depth_m,
            return_intermediates=return_intermediates)


def _nemo_roquet_density_anomaly_ratio_impl(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    *,
    coeffs: dict | None = None,
    rho0: float = rho_0,
    geometric_depth_m: jnp.ndarray | None = None,
    tmask: jnp.ndarray | None = None,
    return_intermediates: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
    """Return NEMO's literal ``prd = (zn * r1_rho0 - 1) * tmask``.

    ``prd`` is the dimensionless density anomaly consumed by ``dynhpg``.
    Dividing before subtracting reproduces NEMO 5.0.2
    ``eosbn2.F90:288``; ``(zn-rho0)/rho0`` is algebraically equivalent but
    loses different fp64 bits.  The Roquet polynomial remains implemented
    once, in :func:`nemo_roquet_eos`.
    """
    # eosbn2.F90:288 has two source-level rounded operations.  Preserve that
    # boundary in the compiled path instead of permitting contraction across
    # the multiply and subtract.  ``tmask`` is optional only for standalone
    # EOS evaluation; every NEMO dynamics caller supplies the real 3-D mask.
    eos_result = nemo_roquet_eos(
        T, S, p, coeffs=coeffs, rho0=rho0,
        geometric_depth_m=geometric_depth_m,
        return_intermediates=return_intermediates)
    if return_intermediates:
        depth_m, zh, zt, zs, zn0, zn1, zn2, zn3, zn = eos_result
    else:
        zn = eos_result
    zn_rho = nemo_source_round(zn * (1.0 / rho0))
    prd = nemo_source_round(zn_rho - 1.0)
    ztm = jnp.asarray(1.0 if tmask is None else tmask, dtype=prd.dtype)
    prd = nemo_source_round(prd * ztm)
    if return_intermediates:
        return (T, S, depth_m, zh, zt, zs, ztm,
                zn0, zn1, zn2, zn3, zn, prd)
    return prd


_nemo_roquet_density_anomaly_ratio_compiled = jax.jit(
    _nemo_roquet_density_anomaly_ratio_impl,
    static_argnames=("return_intermediates",))


def nemo_roquet_density_anomaly_ratio(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    *,
    coeffs: dict | None = None,
    rho0: float = rho_0,
    geometric_depth_m: jnp.ndarray | None = None,
    tmask: jnp.ndarray | None = None,
    return_intermediates: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
    """Return NEMO's literal masked ``rho/rho0 - 1`` arithmetic."""
    if isinstance(T, jax.core.Tracer):
        return _nemo_roquet_density_anomaly_ratio_impl(
            T, S, p, coeffs=coeffs, rho0=rho0,
            geometric_depth_m=geometric_depth_m, tmask=tmask,
            return_intermediates=return_intermediates)
    with jax.disable_jit(False):
        return _nemo_roquet_density_anomaly_ratio_compiled(
            T, S, p, coeffs=coeffs, rho0=rho0,
            geometric_depth_m=geometric_depth_m, tmask=tmask,
            return_intermediates=return_intermediates)


# ==============================================================================
# NEMO polynomial EOS — TEOS-10 coefficient set (Roquet et al. 2015, 55-term
# seawater polynomial, Conservative Temperature + Absolute Salinity branch).
# This is what ORCA1 runs: namelist_cfg sets ``ln_teos10 = .true.`` with
# ``ln_eos80`` commented out, and NEMO then sets ``l_useCT = .TRUE.``
# ("model temperature is Conservative temperature", eosbn2.F90:1924).
#
# Transcribed MECHANICALLY from NEMO 5.0.2 ``src/OCE/TRA/eosbn2.F90`` lines
# 1926-2110, not by eye: a parser read the Fortran assignments and emitted this
# dict, and the SAME parser was first run over the EOS-80 block (:2111-2300)
# and required to reproduce every one of the 52 coefficients in
# ``_ROQUET_EOS80`` exactly.  That is the non-vacuity check on the extraction —
# 122 numbers copied by hand is precisely where a silent wrong-number defect
# enters, and this repo has been bitten by that class before.
#
# The normalization differs from EOS-80 in TWO places, both easy to miss:
# ``rdeltaS`` is 32 (not 20) and ``r1_S0`` is 0.875/35.16504 (the Absolute
# Salinity scaling) rather than 1/40.
# ==============================================================================
_ROQUET_TEOS10 = {
    # normalization (eosbn2.F90:1932-1935). NOTE r1_S0 differs from EOS-80:
    # 0.875/35.16504 is the Absolute-Salinity scaling, and rdeltaS is 32 not 20.
    "r1_S0": 0.875 / 35.16504, "r1_T0": 1.0 / 40.0,
    "r1_Z0": 1.0e-4, "rdeltaS": 32.0,
    # --- density, 52 terms (EOS###) ---
    "EOS000": 801.89615746, "EOS100": 866.72408165, "EOS200": -1786.4682637,
    "EOS300": 2037.5295546, "EOS400": -1284.9161071, "EOS500": 432.27585684,
    "EOS600": -60.579916612, "EOS010": 26.010145068, "EOS110": -65.281885265,
    "EOS210": 81.770425108, "EOS310": -56.888046321, "EOS410": 17.681814114,
    "EOS510": -1.9193502195, "EOS020": -37.074170417, "EOS120": 61.548258127,
    "EOS220": -60.362551501, "EOS320": 29.130021253, "EOS420": -5.4723692739,
    "EOS030": 21.661789529, "EOS130": -33.449108469, "EOS230": 19.717078466,
    "EOS330": -3.1742946532, "EOS040": -8.3627885467, "EOS140": 11.311538584,
    "EOS240": -5.3563304045, "EOS050": 0.54048723791, "EOS150": 0.48169980163,
    "EOS060": -0.19083568888, "EOS001": 19.681925209, "EOS101": -42.549998214,
    "EOS201": 50.774768218, "EOS301": -30.938076334, "EOS401": 6.6051753097,
    "EOS011": -13.336301113, "EOS111": -4.4870114575, "EOS211": 5.0042598061,
    "EOS311": -0.65399043664, "EOS021": 6.7080479603, "EOS121": 3.5063081279,
    "EOS221": -1.8795372996, "EOS031": -2.4649669534, "EOS131": -0.55077101279,
    "EOS041": 0.5592793597, "EOS002": 2.0660924175, "EOS102": -4.9527603989,
    "EOS202": 2.5019633244, "EOS012": 2.0564311499, "EOS112": -0.21311365518,
    "EOS022": -1.2419983026, "EOS003": -0.023342758797, "EOS103": -0.018507636718,
    "EOS013": 0.37969820455,
    # --- thermal expansion alpha, 35 terms (ALP###) ---
    "ALP000": -0.6502536267, "ALP100": 1.6320471316, "ALP200": -2.0442606277,
    "ALP300": 1.422201158, "ALP400": -0.44204535284, "ALP500": 0.047983755487,
    "ALP010": 1.8537085209, "ALP110": -3.0774129064, "ALP210": 3.0181275751,
    "ALP310": -1.4565010626, "ALP410": 0.2736184637, "ALP020": -1.6246342147,
    "ALP120": 2.5086831352, "ALP220": -1.4787808849, "ALP320": 0.23807209899,
    "ALP030": 0.83627885467, "ALP130": -1.1311538584, "ALP230": 0.53563304045,
    "ALP040": -0.067560904739, "ALP140": -0.060212475204, "ALP050": 0.028625353333,
    "ALP001": 0.33340752782, "ALP101": 0.11217528644, "ALP201": -0.12510649515,
    "ALP301": 0.016349760916, "ALP011": -0.33540239802, "ALP111": -0.1753154064,
    "ALP211": 0.093976864981, "ALP021": 0.1848725215, "ALP121": 0.041307825959,
    "ALP031": -0.05592793597, "ALP002": -0.051410778748, "ALP102": 0.0053278413794,
    "ALP012": 0.062099915132, "ALP003": -0.0094924551138,
    # --- haline contraction beta, 35 terms (BET###) ---
    "BET000": 10.783203594, "BET100": -44.452095908, "BET200": 76.04875582,
    "BET300": -63.944280668, "BET400": 26.890441098, "BET500": -4.5221697773,
    "BET010": -0.81219372432, "BET110": 2.0346663041, "BET210": -2.123289517,
    "BET310": 0.87994140485, "BET410": -0.1193963836, "BET020": 0.76574242289,
    "BET120": -1.501981302, "BET220": 1.0872489522, "BET320": -0.2723342908,
    "BET030": -0.41615152308, "BET130": 0.49061350869, "BET230": -0.11847737788,
    "BET040": 0.14073062708, "BET140": -0.13327978879, "BET050": 0.0059929880134,
    "BET001": -0.52937873009, "BET101": 1.2634116779, "BET201": -1.1547328025,
    "BET301": 0.32870876279, "BET011": -0.055824407214, "BET111": 0.12451933313,
    "BET211": -0.024409539932, "BET021": 0.043623149752, "BET121": -0.04676790179,
    "BET031": -0.006852326006, "BET002": -0.061618945251, "BET102": 0.062255521644,
    "BET012": -0.0026514181169, "BET003": -0.00023025968587,
}


def nemo_teos10_density_anomaly_ratio(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    *,
    rho0: float = rho_0,
    geometric_depth_m: jnp.ndarray | None = None,
    tmask: jnp.ndarray | None = None,
    return_intermediates: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
    """TEOS-10 specialization of NEMO's source-associated ``rhd`` value."""
    return nemo_roquet_density_anomaly_ratio(
        T, S, p, coeffs=_ROQUET_TEOS10, rho0=rho0,
        geometric_depth_m=geometric_depth_m, tmask=tmask,
        return_intermediates=return_intermediates)


_NEMO_RHO0 = 1026.0   # NEMO rho0 (eosbn2.F90:1898); legoESM's
# constants.rho_ocean is 1025, and using it here biases alpha, beta and
# hence N2 high by 1026/1025 = +0.0976% (codex 9408213 #6).


def nemo_roquet_alpha_beta(
    T, S, depth_m, rho0: float = _NEMO_RHO0, *, eos_form: str = "teos10"
):
    r"""NEMO ``rab_3d`` thermal expansion / haline contraction (polynomial EOS).

    Transcribes ``eosbn2.F90:1108-1143`` verbatim for the
    ``CASE( np_teos10, np_eos80 )`` branch::

        zh = gdept * r1_Z0 ;  zt = T * r1_T0 ;  zs = SQRT(ABS(S + rdeltaS)*r1_S0)
        alpha = zn_ALP                * r1_rho0
        beta  = zn_BET / zs           * r1_rho0

    Note the ``/ zs`` on beta — it is inside NEMO's expression, not a
    normalization, and dropping it is a silent factor-of-~5 error.

    Depth enters as the GEOMETRIC ``gdept`` (metres, positive down), not a
    pressure: NEMO's ``zh`` is ``gdept * r1_Z0``.  Callers holding pressure
    must convert, unlike :func:`nemo_roquet_eos` which takes Pa.

    Returns ``(alpha, beta)`` with NEMO's sign convention: both POSITIVE for
    normal seawater, and ``bn2 = g(alpha dT - beta dS)/dz`` with
    ``dT = T_upper - T_lower`` (:func:`compute_buoyancy_frequency_nemo_bn2`).

    Differentiable: pure arithmetic in T, S and depth.  ``ABS`` is applied to
    ``S + rdeltaS``, which is strictly positive for physical salinity, so the
    kink is unreachable and the gradient stays finite.
    """
    if eos_form in ("teos10", "nemo_teos10"):
        c = _ROQUET_TEOS10
    elif eos_form in ("eos80", "nemo_eos80"):
        c = _ROQUET_EOS80
    else:
        raise ValueError(
            "eos_form must name NEMO's 'teos10' or 'eos80' coefficient set")
    zh = jnp.asarray(depth_m) * c["r1_Z0"]
    zt = jnp.asarray(T) * c["r1_T0"]
    zs = jnp.sqrt(jnp.abs(jnp.asarray(S) + c["rdeltaS"]) * c["r1_S0"])
    r1_rho0 = 1.0 / rho0

    def _horner(P):
        """Shared ALP/BET nesting — identical in NEMO for both families."""
        zn3 = c[P + "003"]
        zn2 = c[P + "012"] * zt + c[P + "102"] * zs + c[P + "002"]
        zn1 = ((c[P + "031"] * zt
                + c[P + "121"] * zs + c[P + "021"]) * zt
               + (c[P + "211"] * zs + c[P + "111"]) * zs + c[P + "011"]) * zt \
            + ((c[P + "301"] * zs + c[P + "201"]) * zs + c[P + "101"]) * zs \
            + c[P + "001"]
        zn0 = ((((c[P + "050"] * zt
                  + c[P + "140"] * zs + c[P + "040"]) * zt
                 + (c[P + "230"] * zs + c[P + "130"]) * zs + c[P + "030"]) * zt
                + ((c[P + "320"] * zs + c[P + "220"]) * zs
                   + c[P + "120"]) * zs + c[P + "020"]) * zt
               + (((c[P + "410"] * zs + c[P + "310"]) * zs
                   + c[P + "210"]) * zs + c[P + "110"]) * zs + c[P + "010"]) * zt \
            + ((((c[P + "500"] * zs + c[P + "400"]) * zs + c[P + "300"]) * zs
                + c[P + "200"]) * zs + c[P + "100"]) * zs + c[P + "000"]
        return ((zn3 * zh + zn2) * zh + zn1) * zh + zn0

    alpha = _horner("ALP") * r1_rho0
    beta = _horner("BET") / zs * r1_rho0
    return alpha, beta



# ==============================================================================
# UNESCO 1980 EOS coefficients (international one-atmosphere standard;
# Fofonoff & Millard 1983 UNESCO Tech. Papers in Marine Science No. 44).
# Veros's ``eq_of_state_type=3`` is the closely-related Jackett &
# McDougall 1995 polynomial, which modifies a small subset of these
# coefficients to better match measurements but agrees to within
# ~0.001 kg/m³ at typical ocean T/S. For bit-exact Veros parity, read
# Veros's polynomial directly; for tier-2 PGF-tendency comparison the
# UNESCO 1980 form should land inside the discretisation noise floor.
# ==============================================================================

# rho_w(T) — density of pure water [kg/m³]
_UN80_A0 = 999.842594
_UN80_A1 = 6.793952e-2
_UN80_A2 = -9.095290e-3
_UN80_A3 = 1.001685e-4
_UN80_A4 = -1.120083e-6
_UN80_A5 = 6.536332e-9

# A(T)·S coefficients
_UN80_B0 = 8.24493e-1
_UN80_B1 = -4.0899e-3
_UN80_B2 = 7.6438e-5
_UN80_B3 = -8.2467e-7
_UN80_B4 = 5.3875e-9

# B(T)·S^(3/2) coefficients
_UN80_C0 = -5.72466e-3
_UN80_C1 = 1.0227e-4
_UN80_C2 = -1.6546e-6

# C·S^2 coefficient
_UN80_D0 = 4.8314e-4

# K0(T) — secant bulk modulus of pure water at p=0 [bar]
_UN80_E0 = 19652.21
_UN80_E1 = 148.4206
_UN80_E2 = -2.327105
_UN80_E3 = 1.360477e-2
_UN80_E4 = -5.155288e-5

# KS(T) — salinity correction to K at p=0 [bar / PSU]
_UN80_F0 = 54.6746
_UN80_F1 = -0.603459
_UN80_F2 = 1.09987e-2
_UN80_F3 = -6.1670e-5

# S^(3/2) correction to K at p=0 [bar / PSU^(3/2)]
_UN80_G0 = 7.944e-2
_UN80_G1 = 1.6483e-2
_UN80_G2 = -5.3009e-4

# K_p0(T) — pressure correction (coefficient of p) [bar / bar = dimensionless]
_UN80_H0 = 3.239908
_UN80_H1 = 1.43713e-3
_UN80_H2 = 1.16092e-4
_UN80_H3 = -5.77905e-7

# K_pS(T)·S correction (coefficient of p·S)
_UN80_I0 = 2.2838e-3
_UN80_I1 = -1.0981e-5
_UN80_I2 = -1.6078e-6

# K_pS^(3/2) correction (coefficient of p·S^(3/2))
_UN80_J0 = 1.91075e-4

# K_pp(T) — p² coefficient
_UN80_K0 = 8.50935e-5
_UN80_K1 = -6.12293e-6
_UN80_K2 = 5.2787e-8

# K_ppS·S — p²·S coefficient
_UN80_M0 = -9.9348e-7
_UN80_M1 = 2.0816e-8
_UN80_M2 = 9.1697e-10


def unesco80_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    """UNESCO 1980 international one-atmosphere equation of state.

    Parameters
    ----------
    T : array
        IN-SITU temperature [°C] (IPTS-68). Valid range: -2 to 40 °C.

        This said "Potential temperature" until 2026-09-11 and that was
        WRONG. UNESCO 1980 is the in-situ standard; the potential-temperature
        refit is Jackett & McDougall 1995, a different polynomial. Measured
        against the published in-situ check value
        ``rho(S=35, T=25, p=10000 dbar) = 1062.538``, this function returns
        1062.5382 -- agreement to 2e-4 kg/m³, which it could not achieve if
        it were a θ-form. Every prognostic tracer in this package is
        POTENTIAL temperature, so callers must convert with
        :func:`potential_temperature`'s inverse before using this EOS;
        feeding θ straight in leaves the deep ocean too dense by roughly the
        adiabatic compression term. The OMIP runner refuses to select it for
        exactly this reason (``_OMIP_EOS_FORMS``).
    S : array
        Practical salinity [PSU]. Valid range: 0 to 42 PSU.
    p : array
        Sea pressure (gauge; 0 at surface) [Pa]. Valid range: 0 to
        10000 dbar = 1e8 Pa.

    Returns
    -------
    array
        In-situ density [kg/m³]. Reference values:

        - ``rho(0, 0, 0)   ≈ 999.842594``
        - ``rho(0, 35, 0)  ≈ 1028.106331``
        - ``rho(20, 35, 0) ≈ 1024.78``

    Notes
    -----
    All intermediate computations are promoted to float64 to avoid
    precision loss from the large polynomial coefficients (e.g.
    ``E0 ≈ 1.97e4``); the function is differentiable end-to-end.

    The UNESCO 1980 polynomial uses pressure in **bar** internally; the
    input ``p`` (in Pa) is converted via ``p_bar = p · 1e-5``.

    For Veros's ``eq_of_state_type=3`` (JM95) parity at the per-tendency
    level, this implementation should land inside the
    discretisation-truncation noise budget for typical ocean conditions.
    Bit-exact parity requires the exact Veros / JM95 coefficient table
    and is tracked as a follow-up under Phase G.1b in the audit doc.
    """
    T = T.astype(jnp.float64)
    S = S.astype(jnp.float64)
    # Convert pressure Pa → bar (UNESCO convention).
    p_bar = p.astype(jnp.float64) * 1e-5
    # Ensure non-negative salinity in the polynomial (clip floor at 0).
    S_safe = jnp.maximum(S, 0.0)
    S_sqrt = jnp.sqrt(S_safe)

    # Density of pure water at p = 0
    rho_w = (
        _UN80_A0
        + T * (_UN80_A1
        + T * (_UN80_A2
        + T * (_UN80_A3
        + T * (_UN80_A4
        + T * _UN80_A5))))
    )

    # Salinity correction at p = 0
    A_T = (
        _UN80_B0
        + T * (_UN80_B1
        + T * (_UN80_B2
        + T * (_UN80_B3
        + T * _UN80_B4)))
    )
    B_T = (
        _UN80_C0
        + T * (_UN80_C1
        + T * _UN80_C2)
    )
    rho_0 = rho_w + A_T * S + B_T * S_safe * S_sqrt + _UN80_D0 * S * S

    # Secant bulk modulus K(T, S, p)
    K0_T = (
        _UN80_E0
        + T * (_UN80_E1
        + T * (_UN80_E2
        + T * (_UN80_E3
        + T * _UN80_E4)))
    )
    KS_T = (
        _UN80_F0
        + T * (_UN80_F1
        + T * (_UN80_F2
        + T * _UN80_F3))
    )
    KS32_T = (
        _UN80_G0
        + T * (_UN80_G1
        + T * _UN80_G2)
    )
    K_p0 = K0_T + KS_T * S + KS32_T * S_safe * S_sqrt

    Kp_T = (
        _UN80_H0
        + T * (_UN80_H1
        + T * (_UN80_H2
        + T * _UN80_H3))
    )
    KpS_T = (
        _UN80_I0
        + T * (_UN80_I1
        + T * _UN80_I2)
    )
    K_p1 = Kp_T + KpS_T * S + _UN80_J0 * S_safe * S_sqrt

    Kpp_T = (
        _UN80_K0
        + T * (_UN80_K1
        + T * _UN80_K2)
    )
    KppS_T = (
        _UN80_M0
        + T * (_UN80_M1
        + T * _UN80_M2)
    )
    K_p2 = Kpp_T + KppS_T * S

    K = K_p0 + K_p1 * p_bar + K_p2 * p_bar * p_bar

    rho = rho_0 / (1.0 - p_bar / K)
    return rho


# ==============================================================================
# Adiabatic lapse rate and potential temperature (Bryden 1973 / Fofonoff 1977;
# the ``ATG`` / ``THETA`` pair of Fofonoff & Millard 1983, UNESCO Tech. Papers
# in Marine Science No. 44 — the same reference as the UNESCO-80 EOS above).
#
# Why this lives here: observational hydrographies (WOA, PHC) archive IN-SITU
# temperature, while every EOS and prognostic tracer in this package is
# POTENTIAL temperature.  Initialising from an archive without converting
# leaves the deep ocean warm by the adiabatic compression term — about
# 0.1 degC at 1000 m and 0.5 degC at 5000 m — a systematic stratification
# error, not noise.  FESOM2 applies exactly this conversion when
# ``namelist.oce`` sets ``t_insitu = .true.`` (``gen_ic3d.F90`` ->
# ``insitu2pot`` -> ``ptheta``/``atg`` in ``oce_ale_pressure_bv.F90``).
#
# Units are the oceanographic ones the polynomial was fitted in, NOT this
# module's SI pressure: pressure in DECIBARS, temperature in degC (IPTS-68),
# salinity on the practical scale (IPSS-78).  Hence the ``_dbar`` suffixes.
# ==============================================================================

# ATG polynomial coefficients, grouped as in the published Horner form.
# Check value: ATG = 3.255976e-4 degC/dbar at S=40, T=40 degC, p=10000 dbar.
_ATG_T0 = 3.5803e-5
_ATG_T1 = 8.5258e-6
_ATG_T2 = -6.836e-8
_ATG_T3 = 6.6228e-10
_ATG_DS0 = 1.8932e-6
_ATG_DS1 = -4.2393e-8
_ATG_P0 = 1.8741e-8
_ATG_P1 = -6.7795e-10
_ATG_P2 = 8.733e-12
_ATG_P3 = -5.4481e-14
_ATG_PDS0 = -1.1351e-10
_ATG_PDS1 = 2.7759e-12
_ATG_PP0 = -4.6206e-13
_ATG_PP1 = 1.8676e-14
_ATG_PP2 = -2.1687e-16

# Runge-Kutta 4 weights in the Bryden (1973) / Fofonoff (1977) arrangement
# (Gill's method: the 0.29289322 = 1 - 1/sqrt(2) family), transcribed from the
# published ``THETA`` routine rather than re-derived, so the check value below
# pins them.  coeff-ok: fixed integration weights of a cited algorithm, not a
# tunable closure.
_THETA_RK_A1 = 0.29289322
_THETA_RK_B1 = 0.58578644
_THETA_RK_C1 = 0.121320344
_THETA_RK_A2 = 1.707106781
_THETA_RK_B2 = 3.414213562
_THETA_RK_C2 = 4.121320344


def adiabatic_temperature_gradient(
    S: jnp.ndarray,
    T_C: jnp.ndarray,
    p_dbar: jnp.ndarray,
) -> jnp.ndarray:
    """Adiabatic temperature gradient dT/dp [degC/dbar] (Bryden 1973).

    Parameters
    ----------
    S : practical salinity (IPSS-78).
    T_C : in-situ temperature [degC] (IPTS-68).
    p_dbar : pressure [dbar].  In the oceanographic approximation this is
        numerically the depth in metres, which is how both FESOM2's
        ``insitu2pot`` and :func:`potential_temperature`'s callers use it.

    Returns
    -------
    dT/dp [degC/dbar], same shape as the broadcast inputs.
    """
    ds = S - 35.0
    t = T_C
    p = p_dbar
    return (
        (
            ((_ATG_PP2 * t + _ATG_PP1) * t + _ATG_PP0) * p
            + (
                (_ATG_PDS1 * t + _ATG_PDS0) * ds
                + ((_ATG_P3 * t + _ATG_P2) * t + _ATG_P1) * t
                + _ATG_P0
            )
        ) * p
        + (_ATG_DS1 * t + _ATG_DS0) * ds
        + ((_ATG_T3 * t + _ATG_T2) * t + _ATG_T1) * t
        + _ATG_T0
    )


def potential_temperature(
    S: jnp.ndarray,
    T_C: jnp.ndarray,
    p_dbar: jnp.ndarray,
    p_ref_dbar: float = 0.0,
) -> jnp.ndarray:
    """In-situ -> potential temperature [degC] at ``p_ref_dbar``.

    Fourth-order Runge-Kutta integration of
    :func:`adiabatic_temperature_gradient` from ``p_dbar`` to
    ``p_ref_dbar``, i.e. the ``THETA`` routine of Fofonoff & Millard (1983).
    Pure and elementwise, so it is jit/grad/vmap-safe; the fixed four-stage
    integration has no data-dependent control flow.

    Check value (pinned in ``tests/ocean/unit/test_eos_potential_temperature.py``):
    ``theta = 36.89073 degC`` for ``S=40``, ``T_C=40``, ``p_dbar=10000``,
    ``p_ref_dbar=0``.

    Parameters
    ----------
    S : practical salinity (IPSS-78).
    T_C : in-situ temperature [degC] (IPTS-68).
    p_dbar : in-situ pressure [dbar] (~ depth in metres).
    p_ref_dbar : reference pressure [dbar], default 0 (the surface), which is
        the convention for the model's prognostic potential temperature.

    Returns
    -------
    Potential temperature [degC], same shape as the broadcast inputs.
    """
    h = p_ref_dbar - p_dbar
    t = T_C
    p = p_dbar

    xk = h * adiabatic_temperature_gradient(S, t, p)
    t = t + 0.5 * xk
    q = xk
    p = p + 0.5 * h

    xk = h * adiabatic_temperature_gradient(S, t, p)
    t = t + _THETA_RK_A1 * (xk - q)
    q = _THETA_RK_B1 * xk + _THETA_RK_C1 * q

    xk = h * adiabatic_temperature_gradient(S, t, p)
    t = t + _THETA_RK_A2 * (xk - q)
    q = _THETA_RK_B2 * xk - _THETA_RK_C2 * q
    p = p + 0.5 * h

    xk = h * adiabatic_temperature_gradient(S, t, p)
    return t + (xk - 2.0 * q) / 6.0


# Fixed numerical inverse iterations, not a physical closure parameter.
_INSITU_ITERATIONS = 8


def in_situ_temperature(S, theta_C, p_dbar):
    """Invert surface-referenced :func:`potential_temperature`, in Celsius.

    Reverse pressure integration supplies the initial estimate. Fixed residual
    corrections invert the discrete forward conversion (reverse integration
    alone has truncation error). JIT/grad-safe for seawater in the EOS regime.
    Pressure is sea pressure in dbar, as in the forward conversion.
    The final forward residual must be <= 1e-12 K (float64) or 1e-5 K
    (float32). This guards the fixed iteration count's convergence assumption
    for this EOS. Value checks are inert in eager/plain JIT execution: compile
    ``jax.jit(checkify.checkify(fn))`` with user checks enabled (the default)
    and call ``err.throw()`` outside JIT to enforce them.
    """
    initial = potential_temperature(S, theta_C, 0.0, p_dbar)

    def correct(_, t):
        return t + (theta_C - potential_temperature(S, t, p_dbar))

    final = lax.fori_loop(0, _INSITU_ITERATIONS, correct, initial)
    residual = jnp.abs(theta_C - potential_temperature(S, final, p_dbar))
    tolerance = 1e-12 if final.dtype == jnp.float64 else 1e-5
    checkify.debug_check(jnp.all(jnp.isfinite(residual) & (residual <= tolerance)),
                         "EOS: in-situ temperature inverse residual exceeds tolerance")
    return final


# ==============================================================================
# Veros "nonlinear EOS variant 2" (Vallis 2008) — eq_of_state_type=3 in
# Veros (despite the "_eq2" file name, this is what
# ``veros/core/density/get_rho.py`` dispatches to for type=3).
#
# Quadratic in T with a small T² nonlinearity, linear salinity term, and
# pressure dependence with a coupled T·z correction:
#
#     rho_anom = -(grav·z/cs0² + βT·(1 - γs·grav·z·ρ0)·θ + βTs·θ²/2
#                  - βS·(S - S0)) · ρ0
#
# with z = depth (positive downward), θ = T - theta0. Source:
# ``veros/core/density/nonlinear_eq2.py``.
# ==============================================================================


class VerosNonlin2Config(NamedTuple):
    """Veros eq_of_state_type=3 (Vallis 2008 nonlin2) coefficients.

    Defaults taken verbatim from
    ``veros/core/density/nonlinear_eq2.py``.
    """
    rho_0: float = 1024.0
    theta0_C: float = 283.0 - 273.15   # 9.85 °C
    S0: float = 35.0
    grav: float = 9.81
    cs0: float = 1490.0                  # speed of sound [m/s]
    betaT: float = 1.67e-4
    betaTs: float = 1.0e-5
    betaS: float = 0.78e-3
    gammas: float = 1.1e-8
    z0: float = 0.0


def veros_nonlin2_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    cfg: VerosNonlin2Config | None = None,
) -> jnp.ndarray:
    """Veros ``eq_of_state_type=3`` density (Vallis 2008 nonlin2).

    Parameters
    ----------
    T : array — potential temperature [°C]
    S : array — salinity [g/kg or PSU]
    p : array — **pressure in Pa** (the legoESM EOS API convention).
        Internally converted to Veros's "depth in m" convention via
        ``depth_m ≈ p / (rho_0 * g)``. This matches Veros's actual
        usage which passes ``abs(vs.zt)`` (depth in meters) as the
        ``press`` argument to ``get_rho``; that is hydrostatically
        equivalent to ``p_Pa / (rho_0 g)``.

    Returns
    -------
    array
        In-situ density [kg/m³]. (Veros's
        ``nonlin2_eq_of_state_rho`` returns the anomaly; this function
        adds ``rho_0`` so the legoESM EOS API contract stays consistent.)
    """
    cfg = cfg if cfg is not None else VerosNonlin2Config()
    # Convert pressure (Pa) → depth (m) using ``depth ≈ p / (rho_0 g)``.
    # For a hydrostatic ocean column this is exact at the cell centre
    # where ``p_hydro = rho_0 g z``. At depths where rho deviates from
    # rho_0 the equivalence introduces a sub-percent error well below
    # the discretisation noise.
    depth_m = p / (cfg.rho_0 * cfg.grav)
    zz = -depth_m - cfg.z0   # Veros sign convention (negative below surface)
    thetas = T - cfg.theta0_C
    rho_anom = -(
        cfg.grav * zz / (cfg.cs0 * cfg.cs0)
        + cfg.betaT * (1.0 - cfg.gammas * cfg.grav * zz * cfg.rho_0) * thetas
        + 0.5 * cfg.betaTs * thetas * thetas
        - cfg.betaS * (S - cfg.S0)
    ) * cfg.rho_0
    return rho_anom + cfg.rho_0


# ==============================================================================
# Veros "nonlinear EOS variant 3" — eq_of_state_type=4 in Veros.
# Quadratic-in-T (with optional T^2 nonlinearity), zero salinity contribution,
# zero pressure dependency. Despite Veros calling this "nonlinear", it's a
# simple polynomial that captures the leading T-dependence of seawater
# density without invoking UNESCO 1980 / JM95.
#
# Source: veros/core/density/nonlinear_eq3.py:
#   rho = -(betaT·(T - theta0) + betaTs·(T - theta0)^2
#           - betaS·(S - S0)) · rho_0
# Defaults from the Veros source (used by veros.setups.acc.ACCSetup):
#   rho_0 = 1024, theta0 = 9.85 [°C] (= 283 K - T_freeze),
#   S0 = 35, betaT = 1.67e-4, betaTs = 5e-6, betaS = 0
#
# Because betaS = 0 by default, salinity is effectively a passive tracer in
# this EOS — exactly the convention Veros ACC relies on (uniform 35 PSU init).
# ==============================================================================


class VerosNonlin3Config(NamedTuple):
    """Veros eq_of_state_type=3 (nonlinear, T-only) coefficients."""
    rho_0: float = 1024.0
    theta0_C: float = 283.0 - 273.15      # 9.85 °C
    S0: float = 35.0
    betaT: float = 1.67e-4
    betaTs: float = 5.0e-6                # = 1e-5 / 2 (Veros source convention)
    betaS: float = 0.0                    # NOTE: zero by default in Veros nonlin3


def veros_nonlin3_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    cfg: VerosNonlin3Config | None = None,
) -> jnp.ndarray:
    """Veros ``eq_of_state_type=4`` density (quadratic in T, no S, no p).

    Returns
    -------
    array
        In-situ density [kg/m³]. Pressure ``p`` is accepted for API
        symmetry but ignored — this EOS has no pressure dependence.

    Notes
    -----
    Mirrors ``veros/core/density/nonlinear_eq3.py`` exactly.
    NOTE: Veros's ``eq_of_state_type=3`` dispatches to ``nonlin2`` (see
    ``veros/core/density/get_rho.py``), so this function — despite
    being a port of ``nonlinear_eq3.py`` — corresponds to ``type=4``.
    For ``ACCSetup`` (which uses ``type=3``) see :func:`veros_nonlin2_eos`.
    """
    cfg = cfg if cfg is not None else VerosNonlin3Config()
    thetas = T - cfg.theta0_C
    rho_anom = -(
        cfg.betaT * thetas
        + cfg.betaTs * thetas * thetas
        - cfg.betaS * (S - cfg.S0)
    ) * cfg.rho_0
    # Veros returns the density anomaly (rho - rho_0); to match the legoESM
    # EOS API (in-situ rho) we add rho_0 back here.
    return rho_anom + cfg.rho_0



# ==============================================================================
# Veros "gsw" TEOS-10 polynomial EOS — eq_of_state_type=5 in Veros (used by
# the global_4deg setup, among others).
#
# 48-term computationally-efficient rational-polynomial expression for in-situ
# density rho(SA, CT, p) from Absolute Salinity, Conservative Temperature and
# sea pressure (IOC, SCOR and IAPSO 2010 — TEOS-10 manual), ported
# coefficient-for-coefficient from ``veros/core/density/gsw.py``:
#
#     rho = v_hat_denominator(SA, CT, p) / v_hat_numerator(SA, CT, p)
#
# plus the analytic derivatives drho/dT, drho/dS and the Boussinesq dynamic
# enthalpy Hd (with its analytic T/S derivatives, which Veros's
# ``get_int_drhodT/S`` turn into the int_drhodT/S dynamic-enthalpy integrands
# used by the isoneutral skew/iso APE-dissipation diagnostics).
#
# Conventions (IMPORTANT):
# - The Veros kernels nominally take sea pressure in **dbar**, but Veros
#   actually passes ``press = abs(zt)`` — geometric depth in METERS — relying
#   on the standard 1 m ~= 1 dbar oceanographic approximation
#   (``veros/core/thermodynamics.py:76``, ``numerics.py:265``). The legoESM
#   wrappers below take pressure in **Pa** (the eos.py API contract) and
#   convert via ``depth_m = p / (rho_0 * grav)`` with Veros's rho_0 = 1024,
#   grav = 9.81 — the same hydrostatic conversion as :func:`veros_nonlin2_eos`
#   — so a parcel at the same model level sees the bit-same polynomial input
#   as Veros (NOT the exact ``p * 1e-4`` Pa->dbar conversion).
# - Veros's ``gsw_rho`` returns the density ANOMALY (rho - rho0 with
#   rho0 = 1024); :func:`veros_gsw_eos` returns the full in-situ density
#   (= v_hat_denominator / v_hat_numerator directly, no subtract/re-add) to
#   honour the legoESM EOS API, exactly like veros_nonlin2/3.
# - Argument order: Veros kernels are ``(sa, ct, p)`` (salinity FIRST); the
#   legoESM wrappers are ``(T, S, p)`` (the eos.py contract). The private
#   ``_gsw_*_kernel`` functions keep Veros's (sa, ct, p) order and dbar
#   pressure so the bodies stay verbatim-comparable to the source.
# - T is Conservative Temperature [deg C] and S Absolute Salinity [g/kg] in
#   TEOS-10 terms; Veros (and the oracle recipes) feed model potential
#   temperature / practical salinity straight in, and so does legoESM.
# ==============================================================================

# 48-term polynomial coefficients, verbatim from ``veros/core/density/gsw.py``
# (IOC et al. 2010 48-term expression).
_GSW_V01 = 9.998420897506056e2
_GSW_V02 = 2.839940833161907e0
_GSW_V03 = -3.147759265588511e-2
_GSW_V04 = 1.181805545074306e-3
_GSW_V05 = -6.698001071123802e0
_GSW_V06 = -2.986498947203215e-2
_GSW_V07 = 2.327859407479162e-4
_GSW_V08 = -3.988822378968490e-2
_GSW_V09 = 5.095422573880500e-4
_GSW_V10 = -1.426984671633621e-5
_GSW_V11 = 1.645039373682922e-7
_GSW_V12 = -2.233269627352527e-2
_GSW_V13 = -3.436090079851880e-4
_GSW_V14 = 3.726050720345733e-6
_GSW_V15 = -1.806789763745328e-4
_GSW_V16 = 6.876837219536232e-7
_GSW_V17 = -3.087032500374211e-7
_GSW_V18 = -1.988366587925593e-8
_GSW_V19 = -1.061519070296458e-11
_GSW_V20 = 1.550932729220080e-10
_GSW_V21 = 1.0e0
_GSW_V22 = 2.775927747785646e-3
_GSW_V23 = -2.349607444135925e-5
_GSW_V24 = 1.119513357486743e-6
_GSW_V25 = 6.743689325042773e-10
_GSW_V26 = -7.521448093615448e-3
_GSW_V27 = -2.764306979894411e-5
_GSW_V28 = 1.262937315098546e-7
_GSW_V29 = 9.527875081696435e-10
_GSW_V30 = -1.811147201949891e-11
_GSW_V31 = -3.303308871386421e-5
_GSW_V32 = 3.801564588876298e-7
_GSW_V33 = -7.672876869259043e-9
_GSW_V34 = -4.634182341116144e-11
_GSW_V35 = 2.681097235569143e-12
_GSW_V36 = 5.419326551148740e-6
_GSW_V37 = -2.742185394906099e-5
_GSW_V38 = -3.212746477974189e-7
_GSW_V39 = 3.191413910561627e-9
_GSW_V40 = -1.931012931541776e-12
_GSW_V41 = -1.105097577149576e-7
_GSW_V42 = 6.211426728363857e-10
_GSW_V43 = -1.119011592875110e-10
_GSW_V44 = -1.941660213148725e-11
_GSW_V45 = -1.864826425365600e-14
_GSW_V46 = 1.119522344879478e-14
_GSW_V47 = -1.200507748551599e-15
_GSW_V48 = 6.057902487546866e-17

# Unit conversion 1 dbar = 1e4 Pa (only used inside the dynamic-enthalpy
# kernel, where Veros converts its dbar pressure to Pa; NOT used for the
# wrapper Pa->dbar conversion, which goes through rho_0*grav — see above).
_GSW_DB2PA = 1e4


class VerosGswConfig(NamedTuple):
    """Veros ``eq_of_state_type=5`` ("gsw" TEOS-10) conversion constants.

    ``rho_0`` is the Boussinesq reference density removed from the dynamic
    enthalpy (``rho0 = 1024.0`` in ``veros/core/density/gsw.py``); together
    with ``grav`` it also forms the hardcoded ``-(1024.0 / 9.81)`` prefactor
    in Veros's ``get_int_drhodT/S`` (``veros/core/density/get_rho.py:172``)
    and the Pa->depth conversion of the legoESM wrappers. The values are
    intentionally Veros's literals — NOT ``legoesm.constants`` (g = 9.80616)
    — for bit parity with the oracle, matching :class:`VerosNonlin2Config`.
    """
    rho_0: float = 1024.0
    grav: float = 9.81


def _gsw_press_dbar(p, cfg: VerosGswConfig):
    """legoESM pressure [Pa] -> Veros gsw ``press`` argument [dbar~=m].

    See the convention block above: Veros passes depth-in-meters as dbar, so
    the hydrostatically equivalent input is ``p / (rho_0 * grav)``.
    """
    return p / (cfg.rho_0 * cfg.grav)


def _gsw_v_hat(sa, ct, p):
    """Numerator and denominator of the 48-term rational polynomial.

    Veros repeats these two blocks verbatim inside each of gsw_rho /
    gsw_drhodT / gsw_drhodS; factored once here (no-duplicate-numerics),
    bit-identical expression. ``p`` in dbar(~=m), ``sa`` [g/kg], ``ct`` [degC].
    """
    sqrtsa = jnp.sqrt(sa)
    v_hat_denominator = (
        _GSW_V01
        + ct * (_GSW_V02 + ct * (_GSW_V03 + _GSW_V04 * ct))
        + sa * (_GSW_V05 + ct * (_GSW_V06 + _GSW_V07 * ct) + sqrtsa * (_GSW_V08 + ct * (_GSW_V09
            + ct * (_GSW_V10 + _GSW_V11 * ct))))
        + p * (_GSW_V12 + ct * (_GSW_V13 + _GSW_V14 * ct) + sa * (_GSW_V15 + _GSW_V16 * ct)
            + p * (_GSW_V17 + ct * (_GSW_V18 + _GSW_V19 * ct) + _GSW_V20 * sa))
    )
    v_hat_numerator = (
        _GSW_V21
        + ct * (_GSW_V22 + ct * (_GSW_V23 + ct * (_GSW_V24 + _GSW_V25 * ct)))
        + sa
        * (
            _GSW_V26
            + ct * (_GSW_V27 + ct * (_GSW_V28 + ct * (_GSW_V29 + _GSW_V30 * ct)))
            + _GSW_V36 * sa
            + sqrtsa * (_GSW_V31 + ct * (_GSW_V32 + ct * (_GSW_V33 + ct * (_GSW_V34
                + _GSW_V35 * ct))))
        )
        + p
        * (
            _GSW_V37
            + ct * (_GSW_V38 + ct * (_GSW_V39 + _GSW_V40 * ct))
            + sa * (_GSW_V41 + _GSW_V42 * ct)
            + p * (_GSW_V43 + ct * (_GSW_V44 + _GSW_V45 * ct + _GSW_V46 * sa) + p * (_GSW_V47
                + _GSW_V48 * ct))
        )
    )
    return v_hat_denominator, v_hat_numerator


def _gsw_rho_kernel(sa, ct, p):
    """In-situ density [kg/m3] (Veros ``gsw_rho`` WITHOUT the -rho0 shift)."""
    v_hat_denominator, v_hat_numerator = _gsw_v_hat(sa, ct, p)
    return v_hat_denominator / v_hat_numerator


def _gsw_drhodT_kernel(sa, ct, p):
    """drho/dT [kg/m3/K] — verbatim port of Veros ``gsw_drhodT``."""
    a01 = 2.839940833161907e0
    a02 = -6.295518531177023e-2
    a03 = 3.545416635222918e-3
    a04 = -2.986498947203215e-2
    a05 = 4.655718814958324e-4
    a06 = 5.095422573880500e-4
    a07 = -2.853969343267241e-5
    a08 = 4.935118121048767e-7
    a09 = -3.436090079851880e-4
    a10 = 7.452101440691467e-6
    a11 = 6.876837219536232e-7
    a12 = -1.988366587925593e-8
    a13 = -2.123038140592916e-11
    a14 = 2.775927747785646e-3
    a15 = -4.699214888271850e-5
    a16 = 3.358540072460230e-6
    a17 = 2.697475730017109e-9
    a18 = -2.764306979894411e-5
    a19 = 2.525874630197091e-7
    a20 = 2.858362524508931e-9
    a21 = -7.244588807799565e-11
    a22 = 3.801564588876298e-7
    a23 = -1.534575373851809e-8
    a24 = -1.390254702334843e-10
    a25 = 1.072438894227657e-11
    a26 = -3.212746477974189e-7
    a27 = 6.382827821123254e-9
    a28 = -5.793038794625329e-12
    a29 = 6.211426728363857e-10
    a30 = -1.941660213148725e-11
    a31 = -3.729652850731201e-14
    a32 = 1.119522344879478e-14
    a33 = 6.057902487546866e-17

    sqrtsa = jnp.sqrt(sa)
    v_hat_denominator, v_hat_numerator = _gsw_v_hat(sa, ct, p)

    dvhatden_dct = (
        a01
        + ct * (a02 + a03 * ct)
        + sa * (a04 + a05 * ct + sqrtsa * (a06 + ct * (a07 + a08 * ct)))
        + p * (a09 + a10 * ct + a11 * sa + p * (a12 + a13 * ct))
    )

    dvhatnum_dct = (
        a14
        + ct * (a15 + ct * (a16 + a17 * ct))
        + sa * (a18 + ct * (a19 + ct * (a20 + a21 * ct)) + sqrtsa * (a22 + ct * (a23 + ct * (a24
            + a25 * ct))))
        + p * (a26 + ct * (a27 + a28 * ct) + a29 * sa + p * (a30 + a31 * ct + a32 * sa + a33 * p))
    )

    rec_num = 1.0 / v_hat_numerator
    rho = rec_num * v_hat_denominator
    return (dvhatden_dct - dvhatnum_dct * rho) * rec_num


def _gsw_drhodS_kernel(sa, ct, p):
    """drho/dS [kg/m3/(g/kg)] — verbatim port of Veros ``gsw_drhodS``."""
    b01 = -6.698001071123802e0
    b02 = -2.986498947203215e-2
    b03 = 2.327859407479162e-4
    b04 = -5.983233568452735e-2
    b05 = 7.643133860820750e-4
    b06 = -2.140477007450431e-5
    b07 = 2.467559060524383e-7
    b08 = -1.806789763745328e-4
    b09 = 6.876837219536232e-7
    b10 = 1.550932729220080e-10
    b11 = -7.521448093615448e-3
    b12 = -2.764306979894411e-5
    b13 = 1.262937315098546e-7
    b14 = 9.527875081696435e-10
    b15 = -1.811147201949891e-11
    b16 = -4.954963307079632e-5
    b17 = 5.702346883314446e-7
    b18 = -1.150931530388857e-8
    b19 = -6.951273511674217e-11
    b20 = 4.021645853353715e-12
    b21 = 1.083865310229748e-5
    b22 = -1.105097577149576e-7
    b23 = 6.211426728363857e-10
    b24 = 1.119522344879478e-14

    sqrtsa = jnp.sqrt(sa)
    v_hat_denominator, v_hat_numerator = _gsw_v_hat(sa, ct, p)

    dvhatden_dsa = (
        b01
        + ct * (b02 + b03 * ct)
        + sqrtsa * (b04 + ct * (b05 + ct * (b06 + b07 * ct)))
        + p * (b08 + b09 * ct + b10 * p)
    )

    dvhatnum_dsa = (
        b11
        + ct * (b12 + ct * (b13 + ct * (b14 + b15 * ct)))
        + sqrtsa * (b16 + ct * (b17 + ct * (b18 + ct * (b19 + b20 * ct))))
        + b21 * sa
        + p * (b22 + ct * (b23 + b24 * p))
    )

    rec_num = 1.0 / v_hat_numerator
    rho = rec_num * v_hat_denominator
    return (dvhatden_dsa - dvhatnum_dsa * rho) * rec_num


def _gsw_dyn_enthalpy_kernel(sa_in, ct_in, p, rho_0):
    """Boussinesq dynamic enthalpy Hd [m2/s2] — Veros ``gsw_dyn_enthalpy``.

    Verbatim port of the default (non-``pyom_compatibility_mode``) branch:
    the input guards ``sa >= 0.1``, ``ct >= -12`` are Veros's own safety
    floors (division-by-zero / log-blowup protection), kept bit-identical.
    """
    p = jnp.asarray(p)
    sa = jnp.maximum(1e-1, sa_in)  # prevent division by zero (Veros guard)
    ct = jnp.maximum(-12, ct_in)  # prevent blowing up for values < -15 degC (Veros guard)

    sqrtsa = jnp.sqrt(sa)
    a0 = (
        _GSW_V21
        + ct * (_GSW_V22 + ct * (_GSW_V23 + ct * (_GSW_V24 + _GSW_V25 * ct)))
        + sa
        * (
            _GSW_V26
            + ct * (_GSW_V27 + ct * (_GSW_V28 + ct * (_GSW_V29 + _GSW_V30 * ct)))
            + _GSW_V36 * sa
            + sqrtsa * (_GSW_V31 + ct * (_GSW_V32 + ct * (_GSW_V33 + ct * (_GSW_V34
                + _GSW_V35 * ct))))
        )
    )
    a1 = _GSW_V37 + ct * (_GSW_V38 + ct * (_GSW_V39 + _GSW_V40 * ct)) + sa * (_GSW_V41
        + _GSW_V42 * ct)
    a2 = _GSW_V43 + ct * (_GSW_V44 + _GSW_V45 * ct + _GSW_V46 * sa)
    a3 = _GSW_V47 + _GSW_V48 * ct
    b0 = (
        _GSW_V01
        + ct * (_GSW_V02 + ct * (_GSW_V03 + _GSW_V04 * ct))
        + sa * (_GSW_V05 + ct * (_GSW_V06 + _GSW_V07 * ct) + sqrtsa * (_GSW_V08 + ct * (_GSW_V09
            + ct * (_GSW_V10 + _GSW_V11 * ct))))
    )
    b1 = 0.5 * (_GSW_V12 + ct * (_GSW_V13 + _GSW_V14 * ct) + sa * (_GSW_V15 + _GSW_V16 * ct))
    b2 = _GSW_V17 + ct * (_GSW_V18 + _GSW_V19 * ct) + _GSW_V20 * sa
    b1sq = b1 * b1
    sqrt_disc = jnp.sqrt(b1sq - b0 * b2)
    cn = a0 + (2 * a3 * b0 * b1 / b2 - a2 * b0) / b2
    cm = a1 + (4 * a3 * b1sq / b2 - a3 * b0 - 2 * a2 * b1) / b2
    ca = b1 - sqrt_disc
    cb = b1 + sqrt_disc
    part = (cn * b2 - cm * b1) / (b2 * (cb - ca))
    Hd = _GSW_DB2PA * (
        p * (a2 - 2.0 * a3 * b1 / b2 + 0.5 * a3 * p) / b2
        + (cm / (2.0 * b2)) * jnp.log(1.0 + p * (2.0 * b1 + b2 * p) / b0)
        + part * jnp.log(1.0 + (b2 * p * (cb - ca)) / (ca * (cb + b2 * p)))
    )
    return Hd - p * _GSW_DB2PA / rho_0


def _gsw_dHdT_kernel(sa_in, ct_in, p):
    """d(Hd)/dT — verbatim port of Veros ``gsw_dHdT`` (Maple-generated).

    Same Veros input guards as the dynamic enthalpy itself.
    """
    p = jnp.asarray(p)  # convert scalar value if necessary
    sa = jnp.maximum(1e-1, sa_in)  # prevent division by zero
    ct = jnp.maximum(-12, ct_in)  # prevent blowing up for values smaller than -15 degC
    t1 = _GSW_V45 * ct
    t2 = 0.2e1 * t1
    t3 = _GSW_V46 * sa
    t4 = 0.5 * _GSW_V12
    t5 = _GSW_V14 * ct
    t7 = ct * (_GSW_V13 + t5)
    t8 = 0.5 * t7
    t11 = sa * (_GSW_V15 + _GSW_V16 * ct)
    t12 = 0.5 * t11
    t13 = t4 + t8 + t12
    t15 = _GSW_V19 * ct
    t19 = _GSW_V17 + ct * (_GSW_V18 + t15) + _GSW_V20 * sa
    t20 = 1.0 / t19
    t24 = _GSW_V47 + _GSW_V48 * ct
    t25 = 0.5 * _GSW_V13
    t26 = 1.0 * t5
    t27 = sa * _GSW_V16
    t28 = 0.5 * t27
    t29 = t25 + t26 + t28
    t33 = t24 * t13
    t34 = t19**2
    t35 = 1.0 / t34
    t37 = _GSW_V18 + 2.0 * t15
    t38 = t35 * t37
    t48 = ct * (_GSW_V44 + t1 + t3)
    t57 = _GSW_V40 * ct
    t59 = ct * (_GSW_V39 + t57)
    t64 = t13**2
    t68 = t20 * t29
    t71 = t24 * t64
    t74 = _GSW_V04 * ct
    t76 = ct * (_GSW_V03 + t74)
    t79 = _GSW_V07 * ct
    t82 = jnp.sqrt(sa)
    t83 = _GSW_V11 * ct
    t85 = ct * (_GSW_V10 + t83)
    t92 = _GSW_V01 + ct * (_GSW_V02 + t76) + sa * (_GSW_V05 + ct * (_GSW_V06 + t79)
        + t82 * (_GSW_V08 + ct * (_GSW_V09 + t85)))
    t93 = _GSW_V48 * t92
    t105 = _GSW_V02 + t76 + ct * (_GSW_V03 + 2.0 * t74) + sa * (_GSW_V06 + 2.0 * t79
        + t82 * (_GSW_V09 + t85 + ct * (_GSW_V10 + 2.0 * t83)))
    t106 = t24 * t105
    t107 = _GSW_V44 + t2 + t3
    t110 = _GSW_V43 + t48
    t117 = t24 * t92
    t120 = 4.0 * t71 * t20 - t117 - 2.0 * t110 * t13
    t123 = (
        _GSW_V38
        + t59
        + ct * (_GSW_V39 + 2.0 * t57)
        + sa * _GSW_V42
        + (4.0 * _GSW_V48 * t64 * t20 + 8.0 * t33 * t68 - 4.0 * t71 * t38 - t93 - t106
            - 2.0 * t107 * t13 - 2.0 * t110 * t29)
        * t20
        - t120 * t35 * t37
    )
    t128 = t19 * p
    t130 = p * (1.0 * _GSW_V12 + 1.0 * t7 + 1.0 * t11 + t128)
    t131 = 1.0 / t92
    t133 = 1.0 + t130 * t131
    t134 = jnp.log(t133)
    t143 = _GSW_V37 + ct * (_GSW_V38 + t59) + sa * (_GSW_V41 + _GSW_V42 * ct) + t120 * t20
    t152 = t37 * p
    t156 = t92**2
    t165 = _GSW_V25 * ct
    t167 = ct * (_GSW_V24 + t165)
    t169 = ct * (_GSW_V23 + t167)
    t175 = _GSW_V30 * ct
    t177 = ct * (_GSW_V29 + t175)
    t179 = ct * (_GSW_V28 + t177)
    t185 = _GSW_V35 * ct
    t187 = ct * (_GSW_V34 + t185)
    t189 = ct * (_GSW_V33 + t187)
    t199 = t13 * t20
    t217 = 2.0 * t117 * t199 - t110 * t92
    t234 = (
        _GSW_V21
        + ct * (_GSW_V22 + t169)
        + sa * (_GSW_V26 + ct * (_GSW_V27 + t179) + _GSW_V36 * sa + t82 * (_GSW_V31
            + ct * (_GSW_V32 + t189)))
        + t217 * t20
    )
    t241 = t64 - t92 * t19
    t242 = jnp.sqrt(t241)
    t243 = 1.0 / t242
    t244 = t4 + t8 + t12 - t242
    t245 = 1.0 / t244
    t247 = t4 + t8 + t12 + t242 + t128
    t248 = 1.0 / t247
    t249 = t242 * t245 * t248
    t252 = 1.0 + 2.0 * t128 * t249
    t253 = jnp.log(t252)
    t254 = t243 * t253
    t259 = t234 * t19 - t143 * t13
    t264 = t259 * t20
    t272 = 2.0 * t13 * t29 - t105 * t19 - t92 * t37
    t282 = t128 * t242
    t283 = t244**2
    t287 = t243 * t272 / 2.0
    t292 = t247**2
    t305 = (
        0.1e5
        * p
        * (_GSW_V44 + t2 + t3 - 2.0 * _GSW_V48 * t13 * t20 - 2.0 * t24 * t29 * t20
            + 2.0 * t33 * t38 + 0.5 * _GSW_V48 * p)
        * t20
        - 0.1e5 * p * (_GSW_V43 + t48 - 2.0 * t33 * t20 + 0.5 * t24 * p) * t38
        + 0.5e4 * t123 * t20 * t134
        - 0.5e4 * t143 * t35 * t134 * t37
        + 0.5e4 * t143 * t20 * (p * (1.0 * _GSW_V13 + 2.0 * t5 + 1.0 * t27 + t152) * t131
            - t130 / t156 * t105) / t133
        + 0.5e4
        * (
            (
                _GSW_V22
                + t169
                + ct * (_GSW_V23 + t167 + ct * (_GSW_V24 + 2.0 * t165))
                + sa
                * (
                    _GSW_V27
                    + t179
                    + ct * (_GSW_V28 + t177 + ct * (_GSW_V29 + 2.0 * t175))
                    + t82 * (_GSW_V32 + t189 + ct * (_GSW_V33 + t187 + ct * (_GSW_V34
                        + 2.0 * t185)))
                )
                + (
                    2.0 * t93 * t199
                    + 2.0 * t106 * t199
                    + 2.0 * t117 * t68
                    - 2.0 * t117 * t13 * t35 * t37
                    - t107 * t92
                    - t110 * t105
                )
                * t20
                - t217 * t35 * t37
            )
            * t19
            + t234 * t37
            - t123 * t13
            - t143 * t29
        )
        * t20
        * t254
        - 0.5e4 * t259 * t35 * t254 * t37
        - 0.25e4 * t264 / t242 / t241 * t253 * t272
        + 0.5e4
        * t264
        * t243
        * (
            2.0 * t152 * t249
            + t128 * t243 * t245 * t248 * t272
            - 2.0 * t282 / t283 * t248 * (t25 + t26 + t28 - t287)
            - 2.0 * t282 * t245 / t292 * (t25 + t26 + t28 + t287 + t152)
        )
        / t252
    )

    return t305


def _gsw_dHdS_kernel(sa_in, ct_in, p):
    """d(Hd)/dS — verbatim port of Veros ``gsw_dHdS`` (Maple-generated).

    Same Veros input guards as the dynamic enthalpy itself.
    """
    p = jnp.asarray(p)  # convert scalar value if necessary
    sa = jnp.maximum(1e-1, sa_in)  # prevent division by zero
    ct = jnp.maximum(-12.0, ct_in)  # prevent blowing up for values smaller than -15 degC
    t1 = ct * _GSW_V46
    t3 = _GSW_V47 + _GSW_V48 * ct
    t4 = 0.5 * _GSW_V15
    t5 = _GSW_V16 * ct
    t6 = 0.5 * t5
    t7 = t4 + t6
    t13 = _GSW_V17 + ct * (_GSW_V18 + _GSW_V19 * ct) + _GSW_V20 * sa
    t14 = 1.0 / t13
    t17 = 0.5 * _GSW_V12
    t20 = ct * (_GSW_V13 + _GSW_V14 * ct)
    t21 = 0.5 * t20
    t23 = sa * (_GSW_V15 + t5)
    t24 = 0.5 * t23
    t25 = t17 + t21 + t24
    t26 = t3 * t25
    t27 = t13**2
    t28 = 1.0 / t27
    t29 = t28 * _GSW_V20
    t39 = ct * (_GSW_V44 + _GSW_V45 * ct + _GSW_V46 * sa)
    t48 = _GSW_V42 * ct
    t49 = t14 * t7
    t52 = t25**2
    t53 = t3 * t52
    t58 = ct * (_GSW_V06 + _GSW_V07 * ct)
    t59 = jnp.sqrt(sa)
    t66 = t59 * (_GSW_V08 + ct * (_GSW_V09 + ct * (_GSW_V10 + _GSW_V11 * ct)))
    t68 = _GSW_V05 + t58 + 3.0 / 2.0 * t66
    t69 = t3 * t68
    t72 = _GSW_V43 + t39
    t86 = _GSW_V01 + ct * (_GSW_V02 + ct * (_GSW_V03 + _GSW_V04 * ct)) + sa * (_GSW_V05 + t58 + t66)
    t87 = t3 * t86
    t90 = 4.0 * t53 * t14 - t87 - 2.0 * t72 * t25
    t93 = (
        _GSW_V41 + t48 + (8.0 * t26 * t49 - 4.0 * t53 * t29 - t69 - 2.0 * t1 * t25
            - 2.0 * t72 * t7) * t14 - t90 * t28 * _GSW_V20
    )
    t98 = t13 * p
    t100 = p * (1.0 * _GSW_V12 + 1.0 * t20 + 1.0 * t23 + t98)
    t101 = 1.0 / t86
    t103 = 1.0 + t100 * t101
    t104 = jnp.log(t103)
    t115 = (_GSW_V37 + ct * (_GSW_V38 + ct * (_GSW_V39 + _GSW_V40 * ct)) + sa * (_GSW_V41 + t48)
        + t90 * t14)
    t123 = _GSW_V20 * p
    t127 = t86**2
    t142 = ct * (_GSW_V27 + ct * (_GSW_V28 + ct * (_GSW_V29 + _GSW_V30 * ct)))
    t143 = _GSW_V36 * sa
    t151 = _GSW_V31 + ct * (_GSW_V32 + ct * (_GSW_V33 + ct * (_GSW_V34 + _GSW_V35 * ct)))
    t152 = t59 * t151
    t158 = t25 * t14
    t174 = 2.0 * t87 * t158 - t72 * t86
    t189 = (_GSW_V21 + ct * (_GSW_V22 + ct * (_GSW_V23 + ct * (_GSW_V24 + _GSW_V25 * ct)))
        + sa * (_GSW_V26 + t142 + t143 + t152) + t174 * t14)
    t196 = t52 - t86 * t13
    t197 = jnp.sqrt(t196)
    t198 = 1.0 / t197
    t199 = t17 + t21 + t24 - t197
    t200 = 1.0 / t199
    t202 = t17 + t21 + t24 + t197 + t98
    t203 = 1.0 / t202
    t204 = t197 * t200 * t203
    t207 = 1.0 + 2.0 * t98 * t204
    t208 = jnp.log(t207)
    t209 = t198 * t208
    t214 = t189 * t13 - t115 * t25
    t219 = t214 * t14
    t227 = 2.0 * t25 * t7 - t68 * t13 - t86 * _GSW_V20
    t237 = t98 * t197
    t238 = t199**2
    t242 = t198 * t227 / 2.0
    t247 = t202**2
    t260 = (
        0.1e5 * p * (t1 - 2.0 * t3 * t7 * t14 + 2.0 * t26 * t29) * t14
        - 0.1e5 * p * (_GSW_V43 + t39 - 2.0 * t26 * t14 + 0.5 * t3 * p) * t29
        + 0.5e4 * t93 * t14 * t104
        - 0.5e4 * t115 * t28 * t104 * _GSW_V20
        + 0.5e4 * t115 * t14 * (p * (1.0 * _GSW_V15 + 1.0 * t5 + t123) * t101
            - t100 / t127 * t68) / t103
        + 0.5e4
        * (
            (
                _GSW_V26
                + t142
                + t143
                + t152
                + sa * (_GSW_V36 + 1.0 / t59 * t151 / 2.0)
                + (2.0 * t69 * t158 + 2.0 * t87 * t49 - 2.0 * t87 * t25 * t28 * _GSW_V20
                    - t1 * t86 - t72 * t68) * t14
                - t174 * t28 * _GSW_V20
            )
            * t13
            + t189 * _GSW_V20
            - t93 * t25
            - t115 * t7
        )
        * t14
        * t209
        - 0.5e4 * t214 * t28 * t209 * _GSW_V20
        - 0.25e4 * t219 / t197 / t196 * t208 * t227
        + 0.5e4
        * t219
        * t198
        * (
            2.0 * t123 * t204
            + t98 * t198 * t200 * t203 * t227
            - 2.0 * t237 / t238 * t203 * (t4 + t6 - t242)
            - 2.0 * t237 * t200 / t247 * (t4 + t6 + t242 + t123)
        )
        / t207
    )
    return t260


def veros_gsw_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    cfg: VerosGswConfig | None = None,
) -> jnp.ndarray:
    """Veros ``eq_of_state_type=5`` density (TEOS-10 48-term polynomial).

    Parameters
    ----------
    T : array — Conservative Temperature [degC] (Veros feeds model temp).
    S : array — Absolute Salinity [g/kg] (Veros feeds model salt). Must be
        >= 0 (the polynomial contains sqrt(S); not clipped here — see the
        wright_eos no-silent-clipping note).
    p : array — **pressure in Pa** (legoESM EOS API). Converted internally
        to Veros's depth-in-meters-as-dbar convention via
        ``press = p / (rho_0 * grav)`` — see the section header.

    Returns
    -------
    array
        In-situ density [kg/m3]. Veros's ``gsw_rho`` equals this MINUS
        ``rho_0`` (it returns the Boussinesq anomaly); returning the full
        density keeps the legoESM EOS API contract (cf. veros_nonlin2/3).
    """
    cfg = cfg if cfg is not None else VerosGswConfig()
    return _gsw_rho_kernel(S, T, _gsw_press_dbar(p, cfg))


def veros_gsw_drhodT(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    cfg: VerosGswConfig | None = None,
) -> jnp.ndarray:
    """Analytic d(rho)/dT [kg/m3/K] of :func:`veros_gsw_eos` (Veros
    ``gsw_drhodT``; what ``get_drhodT`` returns for ``eq_of_state_type=5``,
    consumed by isoneutral slopes and surface buoyancy forcing).

    Same (T, S, p[Pa]) convention as :func:`veros_gsw_eos`: the derivative
    is taken wrt T at fixed pressure, with the pressure argument converted
    to Veros's meters-as-dbar exactly as in the density itself.
    """
    cfg = cfg if cfg is not None else VerosGswConfig()
    return _gsw_drhodT_kernel(S, T, _gsw_press_dbar(p, cfg))


def veros_gsw_drhodS(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    cfg: VerosGswConfig | None = None,
) -> jnp.ndarray:
    """Analytic d(rho)/dS [kg/m3/(g/kg)] of :func:`veros_gsw_eos` (Veros
    ``gsw_drhodS``). Same conventions as :func:`veros_gsw_drhodT`."""
    cfg = cfg if cfg is not None else VerosGswConfig()
    return _gsw_drhodS_kernel(S, T, _gsw_press_dbar(p, cfg))


def veros_gsw_dyn_enthalpy(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    cfg: VerosGswConfig | None = None,
) -> jnp.ndarray:
    """Boussinesq dynamic enthalpy Hd [m2/s2] (Veros ``gsw_dyn_enthalpy``;
    what ``get_dyn_enthalpy`` returns for ``eq_of_state_type=5`` when
    ``enable_conserve_energy``). Same (T, S, p[Pa]) convention as
    :func:`veros_gsw_eos`."""
    cfg = cfg if cfg is not None else VerosGswConfig()
    return _gsw_dyn_enthalpy_kernel(S, T, _gsw_press_dbar(p, cfg), cfg.rho_0)


def veros_gsw_int_drhodTS_dynamic_enthalpy(
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_full: jnp.ndarray,
    cfg: VerosGswConfig | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """EXACT gsw dynamic-enthalpy integrands ``(int_drhodT, int_drhodS)``.

    The gsw-analytic analogue of the EOS-agnostic leading-order
    :func:`int_drhodTS_dynamic_enthalpy` (same return interface, minus the
    ``eos_fn``/``rho_0``/``g`` arguments which are fixed by the gsw kernels
    and :class:`VerosGswConfig`): Veros computes

        int_drhodT = -(rho_0 / grav) * gsw_dHdT(salt, temp, press)
        int_drhodS = -(rho_0 / grav) * gsw_dHdS(salt, temp, press)

    (``veros/core/density/get_rho.py:157-195``, type-5 branch, with the
    hardcoded ``1024.0 / 9.81`` prefactor lifted into the config) — the
    thermobarically EXACT vertical integrals ``int_z^0 drho/dX dz'``, unlike
    the generic helper's local-derivative-times-depth approximation. Use
    this for the signed-skew / P_diss_skew conversion path when the EOS is
    ``veros_gsw``.

    Parameters
    ----------
    T, S : array — temperature [degC] / salinity [g/kg] at cell centres.
    z_full : array broadcastable to ``T`` — cell-centre height [m], negative
        below the surface (same convention as the generic helper). Veros's
        ``press = abs(zt)`` is ``|z_full|`` directly (meters-as-dbar).
    cfg : VerosGswConfig or None.

    Returns
    -------
    int_drhodT, int_drhodS : arrays, same shape as ``T``.
    """
    cfg = cfg if cfg is not None else VerosGswConfig()
    press = jnp.abs(jnp.broadcast_to(jnp.asarray(z_full), jnp.shape(T)))
    prefactor = -(cfg.rho_0 / cfg.grav)
    int_drhodT = prefactor * _gsw_dHdT_kernel(S, T, press)
    int_drhodS = prefactor * _gsw_dHdS_kernel(S, T, press)
    return int_drhodT, int_drhodS


# Single source of truth for the dispatchable EOS scheme literals. Referenced
# by both make_eos_fn (unknown-scheme ValueError) and config validators
# (fail-fast at construction) so the valid set is never duplicated.
VALID_EOS_SCHEMES = frozenset(
    {"wright", "linear", "nemo_seos", "nemo_eos80", "nemo_teos10", "unesco80",
     "veros_nonlin2", "veros_nonlin3", "veros_gsw"}
)


def _eos_compute_dtype_adapter(base_fn):
    """Wrap an EOS ``(T,S,p)->rho`` that computes in its INPUT dtype so it also
    accepts the ``compute_dtype`` kwarg (the opt-in f32-EOS lever): cast inputs to
    ``compute_dtype``, run the polynomial, then restore the input dtype so the
    STATE stays f64.  ``wright_eos`` instead honours ``compute_dtype`` natively
    (it force-promotes to the precision-policy dtype, which input-casting alone
    cannot override).  ``compute_dtype=None`` -> byte-identical to the bare EOS."""
    def _wrapped(T, S, p, compute_dtype=None):
        if compute_dtype is None:
            return base_fn(T, S, p)
        orig = jnp.result_type(T)
        return base_fn(
            T.astype(compute_dtype), S.astype(compute_dtype),
            p.astype(compute_dtype),
        ).astype(orig)
    return _wrapped


def make_eos_fn(eos="wright", eos_linear=None,
                eos_nemo_seos: NemoSEOSConfig | None = None,
                eos_veros_nonlin2: VerosNonlin2Config | None = None,
                eos_veros_nonlin3: VerosNonlin3Config | None = None,
                eos_veros_gsw: VerosGswConfig | None = None,
                rho0: float = rho_0):
    """Return an EOS callable ``fn(T, S, p) -> rho``.

    Parameters
    ----------
    eos : str
        ``"wright"`` (default, Wright 1997), ``"linear"``,
        ``"nemo_seos"`` (NEMO simplified EOS, Roquet et al. 2015 —
        the DINO oracle EOS, defaults to the Kamm et al. 2025
        coefficients), ``"nemo_eos80"`` or ``"nemo_teos10"`` (NEMO's full
        Roquet polynomial with the corresponding source coefficient set), or
        ``"unesco80"`` (UNESCO 1980 polynomial — close approximation
        to Veros's ``eq_of_state_type=3`` JM95 form, within ~0.001 kg/m³
        at typical ocean T/S; bit-exact Veros parity requires reading
        Veros's polynomial coefficients directly), ``"veros_nonlin2"``
        (Veros type=3), ``"veros_nonlin3"`` (Veros type=4), or
        ``"veros_gsw"`` (Veros type=5, TEOS-10 48-term polynomial —
        the ``global_4deg`` oracle EOS).
    eos_linear : LinearEOSConfig or None
        Parameters for linear EOS.  Ignored unless *eos* is ``"linear"``.
        If ``None`` and *eos* is ``"linear"``, default parameters are used.
    eos_nemo_seos : NemoSEOSConfig or None
        Coefficients for the NEMO simplified EOS.  Ignored unless *eos*
        is ``"nemo_seos"``.  If ``None``, the DINO defaults are used.
    rho0 : float
        Boussinesq reference density [kg/m^3] for the depth reconstruction
        ``zh = (p/(rho0*g))*r1_Z0`` in the ``"nemo_eos80"`` and
        ``"nemo_teos10"`` polynomials.
        Defaults to the module ``rho_0`` (1025) so the default call is
        BYTE-IDENTICAL; pass the config ``rho_0`` (e.g. NEMO's 1026) so it
        stays consistent with the pressure fed to the EOS — required for the
        geometric-depth NEMO-fidelity path where ``zh`` must recover ``gdept``
        exactly (the value itself cancels when it matches the pressure's
        rho0; a MISMATCH stretches the recovered depth).  Ignored by every
        other EOS branch (their depth conversion carries its own rho0).

    Returns
    -------
    Callable[[array, array, array], array]
    """
    # All returned callables accept ``fn(T, S, p, compute_dtype=None)`` — wright
    # honours compute_dtype natively (it force-promotes via the policy); the
    # input-dtype variants are wrapped by _eos_compute_dtype_adapter.  Default
    # (compute_dtype=None) is byte-identical to the bare EOS.  NOTE: variants that
    # internally re-promote to the policy dtype (``unesco80``, ``veros_*``) accept
    # compute_dtype but it is a NO-OP for them (they keep policy precision — safe,
    # just no f32 speedup); the validated f32 lever is the production default
    # ``wright`` (and the genuinely input-dtype ``linear``).
    if eos == "wright":
        return wright_eos
    elif eos == "linear":
        cfg = eos_linear if eos_linear is not None else LinearEOSConfig()
        def _linear(T, S, p):
            return linear_eos(
                T, S, p,
                rho_ref=cfg.rho_ref, alpha_T=cfg.alpha_T,
                beta_S=cfg.beta_S, T_ref=cfg.T_ref, S_ref=cfg.S_ref,
            )
        return _eos_compute_dtype_adapter(_linear)
    elif eos == "nemo_seos":
        cfg = eos_nemo_seos if eos_nemo_seos is not None else NemoSEOSConfig()
        def _nemo_seos(T, S, p):
            return nemo_seos_eos(T, S, p, cfg=cfg)
        return _eos_compute_dtype_adapter(_nemo_seos)
    elif eos == "nemo_eos80":
        # NEMO Roquet-55 EOS-80 polynomial (the full ln_eos80 EOS, distinct from
        # the 3-term nemo_seos). Fixed published coefficients; no config.
        def _nemo_eos80(T, S, p):
            return nemo_roquet_eos(T, S, p, coeffs=_ROQUET_EOS80, rho0=rho0)
        return _eos_compute_dtype_adapter(_nemo_eos80)
    elif eos == "nemo_teos10":
        # NEMO 5.0.2 eosbn2.F90:1920-2108 coefficient selection and :260-288
        # Horner association. Fixed source coefficients; no bespoke config.
        def _nemo_teos10(T, S, p):
            return nemo_roquet_eos(T, S, p, coeffs=_ROQUET_TEOS10, rho0=rho0)
        return _eos_compute_dtype_adapter(_nemo_teos10)
    elif eos == "unesco80":
        return _eos_compute_dtype_adapter(unesco80_eos)
    elif eos == "veros_nonlin2":
        cfg = eos_veros_nonlin2 if eos_veros_nonlin2 is not None else VerosNonlin2Config()
        def _veros_nl2(T, S, p):
            return veros_nonlin2_eos(T, S, p, cfg=cfg)
        return _eos_compute_dtype_adapter(_veros_nl2)
    elif eos == "veros_nonlin3":
        cfg = eos_veros_nonlin3 if eos_veros_nonlin3 is not None else VerosNonlin3Config()
        def _veros_nl3(T, S, p):
            return veros_nonlin3_eos(T, S, p, cfg=cfg)
        return _eos_compute_dtype_adapter(_veros_nl3)
    elif eos == "veros_gsw":
        cfg = eos_veros_gsw if eos_veros_gsw is not None else VerosGswConfig()
        def _veros_gsw(T, S, p):
            return veros_gsw_eos(T, S, p, cfg=cfg)
        return _eos_compute_dtype_adapter(_veros_gsw)
    else:
        raise ValueError(
            f"Unknown EOS scheme: {eos!r}. Valid schemes: "
            f"{sorted(VALID_EOS_SCHEMES)}."
        )


def density_perturbation(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    rho_ref: float = rho_0,
) -> jnp.ndarray:
    """Compute density perturbation rho' = rho(T,S,p) - rho_ref.

    Parameters
    ----------
    T, S, p : array
        Temperature [degC], salinity [PSU], pressure [Pa].
    rho_ref : float
        Reference density [kg/m^3].

    Returns
    -------
    array : Density perturbation [kg/m^3].
    """
    return wright_eos(T, S, p) - rho_ref


def compute_hydrostatic_pressure(
    rho: jnp.ndarray,
    eta: jnp.ndarray,
    dz: jnp.ndarray,
    jacobian: jnp.ndarray,
    rho_ref: float = rho_0,
    g: float = constants.g,
    h_actual: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute hydrostatic pressure at full levels.

    p(z) = rho_ref * g * eta + integral_{z}^{0} rho * g dz'

    Integrated top-to-bottom (k=0 is surface, k=nlev-1 is deepest).
    Pressure at cell center is the cumulative integral from surface
    down to the midpoint of each layer.

    Parameters
    ----------
    rho : array
        In-situ density, shape (..., nlev).
    eta : array
        Sea surface height [m], shape (...).
    dz : array
        Reference layer thickness [m], shape (nlev,).  Ignored when
        ``h_actual`` is provided.
    jacobian : array
        Dynamic Jacobian, shape (...).  Ignored when ``h_actual`` is
        provided.
    rho_ref : float
        Reference density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].
    h_actual : array or None
        Optional pre-computed per-cell layer thickness, shape
        (..., nlev).  When provided, used directly; when None, the
        legacy formula ``dz * jacobian[..., None]`` is used.

        This is the partial-cells extension point: callers using an
        ``OceanPartialCellCoordinate`` should pass
        ``h_actual = compute_layer_thickness(eta, H_bathy, coord)``
        to integrate pressure with the correct partial bottom-cell
        thickness.  Cells below the seafloor have h_actual=0, so they
        contribute zero pressure increment automatically.

    Returns
    -------
    array : Hydrostatic pressure at full levels [Pa], shape (..., nlev).
    """
    # Surface pressure from free surface
    p_surface = rho_ref * g * eta  # (...,)

    # Actual layer thickness — pre-computed (partial cells) or
    # dz * jacobian (legacy z*).
    if h_actual is None:
        h_actual = dz * jacobian[..., jnp.newaxis]  # (..., nlev)

    # Pressure increment per layer: rho * g * h
    dp = rho * g * h_actual  # (..., nlev)

    # Pressure at layer top = cumulative sum from surface
    # p_top[k] = p_surface + sum(dp[0:k])
    p_top = p_surface[..., jnp.newaxis] + jnp.cumsum(dp, axis=-1) - dp

    # Pressure at cell center = p_top + 0.5 * dp
    return p_top + 0.5 * dp


def compute_buoyancy_frequency(
    rho: jnp.ndarray,
    dz: jnp.ndarray,
    jacobian: jnp.ndarray,
    rho_ref: float = rho_0,
    g: float = constants.g,
) -> jnp.ndarray:
    """Compute Brunt-Vaisala frequency N^2.

    N^2 = -(g / rho_ref) * d(rho) / dz

    Computed at interior interfaces (nlev-1 values).

    Parameters
    ----------
    rho : array
        In-situ density, shape (..., nlev).
    dz : array
        Reference layer thickness [m], shape (nlev,).
    jacobian : array
        Dynamic Jacobian, shape (...).
    rho_ref : float
        Reference density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    array : N^2 at interior interfaces [1/s^2], shape (..., nlev-1).
    """
    dz_actual = dz * jacobian[..., jnp.newaxis]
    dz_interface = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])

    # drho/dz: rho[k] is shallower than rho[k+1]
    # N^2 = -(g/rho_0) * (rho[k] - rho[k+1]) / dz_interface
    drho_dz = (rho[..., :-1] - rho[..., 1:]) / dz_interface

    return -(g / rho_ref) * drho_dz


def compute_buoyancy_frequency_adiabatic(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p_cell: jnp.ndarray,
    dz: jnp.ndarray,
    jacobian: jnp.ndarray,
    eos_fn=None,
    rho_ref: float = rho_0,
    g: float = constants.g,
    *,
    dz_half: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Static-stability ``N^2`` via adiabatic parcel displacement.

    This is the **true static stability** used by Veros
    (``veros/core/thermodynamics.py:99-103``) and the cleaner form
    requested for oracle fidelity: instead of differencing the *in-situ*
    densities ``rho(T[k], S[k], p[k]) - rho(T[k+1], S[k+1], p[k+1])``
    (which carries the compressibility difference between two different
    reference pressures and is therefore biased ~6x too stable), both
    parcels are evaluated at the **upper cell's reference pressure**
    ``p_cell[k]`` so only the (potential-) density contrast remains:

    .. math::

        N^2[k] = -\frac{g}{\rho_0}\,
            \frac{\rho(T_{k+1}, S_{k+1}, p_k) - \rho(T_k, S_k, p_k)}
                 {\Delta z_{int}[k]}

    With ``z`` increasing upward and legoESM's ``k=0`` at the surface,
    a statically *unstable* column (denser water displaced over lighter)
    gives ``N^2 < 0``. **Unlike** :func:`compute_buoyancy_frequency` and
    the legacy ``tke._compute_N2``, this is **not** clipped at zero — the
    sign is the convection trigger, so it must be allowed to go negative.

    The choice of the *upper cell-centre* pressure ``p_cell[k]`` (rather
    than the interface pressure) matches Veros exactly: Veros passes
    ``press = abs(zt)`` (the cell-centre geometric depth of the upper
    cell) and compares ``get_rho(T[k+1], S[k+1], press[k])`` against the
    upper cell's own in-situ density ``rho[k] = get_rho(T[k], S[k],
    press[k])`` — i.e. both at ``press[k]``.

    Differentiability: the only operations are the (differentiable) EOS
    evaluations and arithmetic — no ``where``/``cond`` on traced values —
    so this is fully ``jax.grad``-safe (``d N^2 / dT`` etc. flow through
    the EOS at the displaced pressure).

    Parameters
    ----------
    T, S : array
        Potential temperature [degC] / salinity [PSU] at cell centres,
        shape ``(..., nlev)``.
    p_cell : array
        Hydrostatic pressure [Pa] at cell centres, shape ``(..., nlev)``
        (e.g. from :func:`compute_ocean_rho_and_pressure`). The pressure
        of the *upper* cell of each interface, ``p_cell[..., :-1]``, is
        used as the common reference pressure for both displaced parcels.
    dz : array
        Reference layer thickness [m], shape ``(nlev,)``.
    jacobian : array
        Dynamic Jacobian, shape ``(...)``.
    eos_fn : callable or None
        EOS ``fn(T, S, p) -> rho``. If None, uses :func:`wright_eos`.
    rho_ref : float
        Reference density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].
    dz_half : array or None (keyword-only)
        ACTUAL distance between adjacent cell centres [m], shape
        ``(..., nlev-1)`` (already Jacobian-scaled). This is the slot Veros
        divides by — ``dzw`` (thermodynamics.py:99) — which on a stretched
        u_centered grid is NOT the midpoint interface spacing. ``None``
        (default, BIT-IDENTICAL legacy): the midpoint reconstruction
        ``0.5*(dz·J)[k] + 0.5*(dz·J)[k+1]`` is used, preserving every
        existing caller byte-for-byte (the EKE/Visbeck adiabatic path in
        ``lateral_mixing/_gm_redi_common.py`` and legacy TKE configs).
        Callers on a Veros-faithful coordinate (``TKEConfig.veros_dz_slots``)
        pass their ``dz_half`` so N² is taken over the true centre spacing.

    Returns
    -------
    array : ``N^2`` at interior interfaces [1/s^2], shape ``(..., nlev-1)``.
        **Signed** (negative where statically unstable).
    """
    if eos_fn is None:
        eos_fn = wright_eos

    if dz_half is None:
        # Legacy midpoint reconstruction (bit-identical default).
        dz_actual = dz * jacobian[..., jnp.newaxis]
        dz_interface = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    else:
        # Veros dzw slot: the caller's actual centre-to-centre spacing.
        dz_interface = dz_half

    # Common reference pressure = the UPPER cell's centre pressure (Veros
    # press[k] = abs(zt[k])).
    p_ref_int = p_cell[..., :-1]

    # Upper parcel (k) and lower parcel (k+1), BOTH at the upper pressure.
    rho_upper = eos_fn(T[..., :-1], S[..., :-1], p_ref_int)
    rho_lower = eos_fn(T[..., 1:], S[..., 1:], p_ref_int)

    # drho/dz with z positive upward; (rho_upper - rho_lower)/dz. For a
    # stable column rho_upper < rho_lower -> drho/dz < 0 -> N^2 > 0.
    # Land/dry-column guard: with a partial-cell coordinate the Jacobian is 0
    # over land (H_bathy = 0), so dz_interface = 0 there and the unguarded
    # division returned 0/0 = NaN (uniform T=S=0 land columns) — which then
    # poisoned the TKE/EKE chains through every downstream `× mask`
    # (0·NaN = NaN). With the floor, land columns give exactly
    # 0/eps = 0 ⇒ N² = 0 (neutral) — masked downstream as before. Wet
    # interfaces (dz_interface >= O(10 m)) are bit-identical.
    drho_dz = (rho_upper - rho_lower) / jnp.maximum(dz_interface, 1.0e-12)
    return -(g / rho_ref) * drho_dz


# ==============================================================================
# Shared helpers for ocean physics integration modules
# ==============================================================================

def maybe_partial_h_actual(state, z_coord):
    """Return per-cell h_actual when z_coord is a partial-cell coord,
    else None (caller falls back to dz * jacobian).

    Routed through the local import to avoid a circular dependency:
    eos.py imports vertical.py would create a cycle through state.py.
    """
    from legoesm.ocean.vertical import (
        OceanPartialCellCoordinate, compute_layer_thickness,
    )
    if isinstance(z_coord, OceanPartialCellCoordinate):
        return compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord,
        )
    return None


def compute_ocean_rho(state, z_coord, jacobian, eos_fn=None,
                      *, eos_depth="insitu", rho0=None, g=None):
    """Compute in-situ density from ocean state.

    Used by vertical mixing, lateral mixing, and convection integration
    bridges. Avoids triplicating the same hydrostatic pressure + EOS call.

    Dispatches on coord type:

    - ``OceanZStarCoordinate``: legacy path, uses ``dz_ref * jacobian``
      for layer thickness.  Bit-exact unchanged.
    - ``OceanPartialCellCoordinate``: passes per-cell ``h_partial *
      (eta+H_bathy)/H_bathy`` to the hydrostatic integrator so the
      partial bottom cell's contribution is correct.

    Parameters
    ----------
    state : OceanState
        Must have .T, .S, .eta, .H_bathy fields.
    z_coord : OceanZStarCoordinate or OceanPartialCellCoordinate
        Vertical coordinate.
    jacobian : array
        Dynamic Jacobian.  For pure z*: (eta + H) / H_max.  For
        partial cells: (eta + H_bathy) / H_bathy.  Caller is expected
        to use ``compute_ocean_jacobian`` which dispatches.
    eos_fn : callable or None
        EOS function ``fn(T, S, p) -> rho``.  If None, uses ``wright_eos``.
    eos_depth : str, default ``"insitu"``
        Depth the EOS pressure term sees.  ``"insitu"`` (default,
        BYTE-IDENTICAL): the 2-pass in-situ hydrostatic pressure integral
        ``p = g*Sum(rho*dz)`` — recovers depth ~(rho_bar/rho0)*gdept.
        ``"geometric"``: feed ``p = rho0*g*gdept`` from the coordinate's
        geometric T-depth ladder (``z_coord.t_depth_ref``, NEMO ``gdept_1d``)
        so the EOS reconstructs geometric depth exactly — matches NEMO's
        ``eos_insitu`` which uses ``gdept`` directly.  The *eos_fn* MUST have
        been built with the SAME ``rho0`` (``make_eos_fn(rho0=...)``) so the
        value cancels; otherwise the recovered depth is stretched.
    rho0 : float or None
        Boussinesq reference density.  ``None`` (default) uses the module
        ``rho_0``.  Used on BOTH depth paths, not only the geometric one: the
        in-situ integral's surface term is ``rho_ref*g*eta``, so a run pinning
        its own reference density and not passing it here gets the module's in
        that term while everything downstream uses its own.  (The line this
        replaces said "ignored for insitu"; it was read as licence to pass
        gravity alone, and a reviewer caught that the code does no such thing.)

    Returns
    -------
    array : In-situ density [kg/m^3].
    """
    if eos_fn is None:
        eos_fn = wright_eos
    if eos_depth not in ("insitu", "geometric"):
        raise ValueError(
            f"Unknown eos_depth {eos_depth!r}; expected 'insitu' or 'geometric'")
    if eos_depth == "geometric":
        # NEMO eos_insitu: density from the GEOMETRIC gdept, not the in-situ
        # hydrostatic integral.  p = rho0*g*gdept -> zh recovers gdept exactly.
        # LIVE gdept(Kmm) = gdept_0*(1+r3t), matching eosbn2.F90:541 and the
        # production GM/Redi path (gm_redi_density_and_jacobian).  Feeding the
        # STATIC ladder here left this consumer -- reached via
        # fidelity/tendency_probe.py, i.e. the ORACLE TENDENCY COMPARISON --
        # on a different density convention from the model it is measuring
        # (#1226; the same defect cost prd a depth-structured 2.559e-6).
        # Gated exactly like its siblings: bit-identical when no fidelity
        # ladder is carried, and nemo_bn2_live_ladders honours
        # linear_free_surface (key_linssh: the column never stretches).
        _td = getattr(z_coord, "t_depth_ref", None)
        _H = getattr(getattr(state, "H_bathy", None), "data", None)
        _eta = getattr(getattr(state, "eta", None), "data", None)
        if _td is None:
            depth = jnp.abs(z_coord.z_full_ref)          # unchanged
        elif _H is None or _eta is None:
            # No free-surface information on this state (e.g. an analytic
            # column in a unit test): keep the STATIC ladder, bit-identical to
            # the pre-change behaviour rather than silently switching ladders.
            depth = jnp.asarray(_td)
        else:
            depth = nemo_bn2_live_ladders(z_coord, _eta, _H)[0]
        r0 = rho_0 if rho0 is None else rho0
        _g = constants.g if g is None else g
        p_eos = (r0 * _g) * jnp.asarray(depth, dtype=state.T.data.dtype)
        return eos_fn(state.T.data, state.S.data, p_eos)
    h_actual = maybe_partial_h_actual(state, z_coord)
    # Two EOS iterations for density-pressure consistency, matching the
    # dynamical core (ocean_pe_cdgrid.py).
    rho = eos_fn(state.T.data, state.S.data, jnp.zeros_like(state.T.data))
    _r0 = rho_0 if rho0 is None else rho0
    _g = constants.g if g is None else g
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho, state.eta.data, z_coord.dz_ref, jacobian, _r0, _g,
            h_actual=h_actual,
        )
        rho = eos_fn(state.T.data, state.S.data, p_hydro)
    return rho


def compute_ocean_rho_and_pressure(state, z_coord, jacobian, eos_fn=None,
                                   *, rho0=None, g=None):
    """Compute in-situ density and hydrostatic pressure from ocean state.

    Dispatches on coord type — see ``compute_ocean_rho``.

    Parameters
    ----------
    state, z_coord, jacobian : same as ``compute_ocean_rho``.
    eos_fn : callable or None
        EOS function. If None, uses ``wright_eos``.

    Returns
    -------
    rho : array — in-situ density [kg/m^3].
    p_hydro : array — hydrostatic pressure [Pa].
    """
    rho = compute_ocean_rho(state, z_coord, jacobian, eos_fn=eos_fn,
                            rho0=rho0, g=g)
    h_actual = maybe_partial_h_actual(state, z_coord)
    p_hydro = compute_hydrostatic_pressure(
        rho, state.eta.data, z_coord.dz_ref, jacobian,
        rho_0 if rho0 is None else rho0, constants.g if g is None else g,
        h_actual=h_actual,
    )
    return rho, p_hydro


# --- NEMO eos_fzp freezing point (eosbn2.F90; TEOS-10 branch) ---------------
# Polynomial fit of the conservative-temperature freezing point (Roquet et
# al. 2015 TEOS-10 polynomial EOS, as hard-coded in NEMO eos_fzp) plus the
# NEMO 7.53e-4 K/m pressure lowering.  ORCA1 runs ln_teos10=.true., so this
# is the oracle's ISF/sea-ice freezing-point function.
_NEMO_FZP_S0 = 35.16504          # TEOS-10 reference salinity SA0 [g/kg]
_NEMO_FZP_C0 = -5.87701e-2       # eos_fzp polynomial coefficients
_NEMO_FZP_C1 = 2.07679e-2
_NEMO_FZP_C2 = -3.12775e-2
_NEMO_FZP_C3 = 2.28348e-2
_NEMO_FZP_C4 = -9.64972e-3
_NEMO_FZP_C5 = 1.46873e-3
_NEMO_FZP_DEP = -7.53e-4         # [degC/m] freezing-point pressure lowering


def nemo_eos_fzp(S_psu, depth_m=None):
    """Seawater freezing point [°C] — NEMO ``eos_fzp`` (TEOS-10 branch).

    ``T_f(S, z) = S · P(√(S/S0)) − 7.53e-4 · z`` with the eosbn2.F90
    polynomial ``P``; ``depth_m`` positive down (``None`` = surface).
    """
    # At S=0 the full S*P(sqrt(abs(S)/S0)) has derivative P(0), but
    # differentiating the unguarded square root produces 0*inf -> NaN.
    # Guard inside sqrt as well as outside; retain the exact forward value.
    sal_abs = jnp.abs(jnp.asarray(S_psu))
    nonzero = sal_abs > 0.0
    zs = jnp.where(nonzero, jnp.sqrt(jnp.where(
        nonzero, sal_abs / _NEMO_FZP_S0, 1.0)), 0.0)
    poly = ((((_NEMO_FZP_C5 * zs + _NEMO_FZP_C4) * zs + _NEMO_FZP_C3) * zs
             + _NEMO_FZP_C2) * zs + _NEMO_FZP_C1) * zs + _NEMO_FZP_C0
    tf = poly * jnp.asarray(S_psu)
    if depth_m is not None:
        tf = tf + _NEMO_FZP_DEP * jnp.asarray(depth_m)
    return tf


# ==============================================================================
# Seawater freezing point (liquidus)  T_f(S, p)  ->  KELVIN   (MED-1)
# ==============================================================================
# Gap this closes: the freeze checks that cap SST / trigger ice formation
# otherwise use a single FIXED constant ``constants.T_freeze_ocean`` (271.35 K,
# ~-1.8 C).  Real seawater freezes along a LIQUIDUS that DECREASES with salinity
# (and, weakly, with pressure): at S = 35 PSU, p = 0 the true value is ~-1.92 C
# (271.23 K), NOT -1.8 C.  NEMO (``eos_fzp``) and MOM6 (``TFREEZE_FORM``) both
# carry such a liquidus; this exposes it as a selectable scheme.
#
# CONVENTION (stated explicitly per the sign-convention gate):
#   * S in PSU / (g/kg); p is SEA PRESSURE in Pa (the eos.py API convention),
#     converted to dbar via ``p_dbar = p / 1e4`` (dbar >= 0, increasing downward).
#   * The polynomials below give the freezing-point DEPRESSION in degC (a
#     temperature DIFFERENCE, so its numeric value is the same in K); it is ADDED
#     to the pure-water freezing point ``T0 = constants.T_freeze`` to return an
#     ABSOLUTE freezing temperature in KELVIN.  Pure water (S=0, p=0) -> T0 (0 C).
#   * SIGN: the p-term is negative; the S-terms are MIXED-sign (the S^{3/2}
#     coefficient +1.710523e-3 is POSITIVE), but the NET slope
#         dT_f/dS = -0.0575 + 1.5*1.710523e-3*sqrt(S) - 2*2.154996e-4*S
#     is strictly negative for ALL S >= 0: the two curvature terms together
#     peak at +3.82e-3 degC/PSU (at S ~ 8.9 PSU), so
#     dT_f/dS <= -0.0575 + 0.0038 = -0.0537 degC/PSU everywhere (at S=35:
#     -0.0575 + 0.01518 - 0.01508 = -0.0574).  T_f therefore DECREASES with
#     both salinity and depth (more saline / deeper water freezes colder).
#     Enforced by tests/ocean/unit/test_freezing_point.py (monotone-in-S +
#     pressure-lowers).
#
# Distinct from ``nemo_eos_fzp`` above: that is NEMO's TEOS-10 branch
# (``ln_teos10=.true.``, a polynomial in sqrt(S/S0), returns degC).  The
# ``"unesco"`` scheme here is NEMO's EOS-80 branch (``ln_teos10=.false.``), which
# equals MOM6 ``TFREEZE_FORM="MILLERO_78"``; both branches are kept because
# different oracle recipes select different ones.
#
# Provenance of the coefficients: UNESCO 1983 / Millero (1978) freezing-point-of-
# seawater fit, as coded in NEMO ``eosbn2.F90`` (``eos_fzp``, EOS-80 branch) and
# MOM6 ``MOM_EOS.F90`` (``calculate_TFreeze_Millero``):
#       T_f[degC] = a*S + b*S^1.5 + c*S^2 + d*p_dbar
# The leading slope a = -0.0575 degC/PSU is ALSO the MOM6 linear-liquidus slope
# (``TFREEZE_FORM="LINEAR"``), so the ``"linear_S"`` scheme REUSES it rather than
# re-declaring the literal.
# --- Millero / UNESCO 1983 liquidus coefficients ---
_TFRZ_S_LINEAR = -0.0575        # [degC/PSU]     linear salinity term (= MOM6 linear slope)
_TFRZ_S_ONEHALF = 1.710523e-3   # [degC/PSU^1.5] S^{3/2} term
_TFRZ_S_SQUARE = -2.154996e-4   # [degC/PSU^2]   S^2 term
_TFRZ_P_DBAR = -7.53e-4         # [degC/dbar]    sea-pressure (depth) lowering
_PA_PER_DBAR = 1.0e4            # exact unit conversion (1 dbar = 1e4 Pa)

# Single source of truth for the dispatchable liquidus schemes; referenced by
# both ``freezing_point`` (unknown-scheme ValueError) and the consumer configs.
VALID_FREEZE_SCHEMES = frozenset({"constant", "linear_S", "unesco"})


class FreezingPointConfig(NamedTuple):
    """Selects the seawater freezing-point (liquidus) scheme for freeze checks.

    ``scheme``
        * ``"constant"`` (default) — fixed ``constants.T_freeze_ocean``
          (271.35 K); byte-identical to the historical behaviour, NO S/p
          dependence.
        * ``"linear_S"`` — MOM6 linear liquidus ``T_f = T0 - 0.0575*S`` [K]
          (``T0 = constants.T_freeze``); pressure-independent.
        * ``"unesco"`` — UNESCO 1983 / Millero 1978 fit (NEMO ``eos_fzp`` EOS-80
          branch): salinity- AND pressure-dependent.

    This config carries NO float fields — ``scheme`` is a discrete selector, not
    a tunable coefficient — so it is intentionally not a ``__param_spec__``
    module and is not registered in ``param_collector.SPEC_MODULES``.
    """
    scheme: str = "constant"


def freezing_point(S, p=0.0, *, scheme="constant"):
    """Seawater freezing point ``T_f`` [KELVIN] vs salinity (and pressure).

    Parameters
    ----------
    S : array or float
        Practical salinity [PSU] / (g/kg).  Floored at 0 (a negative-salinity
        advection overshoot would otherwise put ``sqrt(S)`` in the complex
        plane); the floor sits at the physical wet-domain edge and leaves
        gradients intact for the physical ``S > 0`` range.
    p : array or float, optional
        Sea pressure [Pa] (default ``0.0`` = surface), converted to dbar via
        ``p / 1e4``.  Used ONLY by ``"unesco"``; ``"constant"``/``"linear_S"``
        ignore it (the surface liquidus, matching MOM6's linear form).
    scheme : {"constant", "linear_S", "unesco"}, keyword-only
        Liquidus scheme (see :class:`FreezingPointConfig`).

    Returns
    -------
    array
        Freezing temperature [K].  Pure JAX — differentiable and jit/vmap-safe.

    Raises
    ------
    ValueError
        Unknown ``scheme`` (fail-fast dispatch hardening; validated at function
        entry on the static Python selector, never inside a traced branch).
    """
    # Dispatch hardening: reject typos loudly on the static selector rather than
    # silently running the wrong (or default) physics.
    if scheme not in VALID_FREEZE_SCHEMES:
        raise ValueError(
            f"Unknown freezing-point scheme: {scheme!r}. "
            f"Valid schemes: {sorted(VALID_FREEZE_SCHEMES)}."
        )

    S_arr = jnp.asarray(S)
    if scheme == "constant":
        # Byte-identical to the historical fixed constant; broadcast so array
        # callers get a per-cell field and scalar callers get a scalar.
        return jnp.broadcast_to(jnp.asarray(constants.T_freeze_ocean), S_arr.shape)

    # S >= 0 guard (shared by linear_S and unesco).
    S_safe = jnp.maximum(S_arr, 0.0)

    if scheme == "linear_S":
        # MOM6 linear liquidus: T_f[K] = T0 + a*S, a < 0 => T_f decreases with S.
        depression_c = _TFRZ_S_LINEAR * S_safe
        return constants.T_freeze + depression_c

    # scheme == "unesco": UNESCO 1983 / Millero 1978 (NEMO eos_fzp EOS-80 branch).
    # ``S_safe ** 1.5`` (analytic pow-JVP 1.5*S**0.5) is grad-safe at S=0, unlike
    # ``S_safe * sqrt(S_safe)`` whose sqrt node yields a 0*inf NaN gradient there.
    p_dbar = jnp.asarray(p) / _PA_PER_DBAR
    depression_c = (
        _TFRZ_S_LINEAR * S_safe
        + _TFRZ_S_ONEHALF * (S_safe ** 1.5)     # S^{3/2}
        + _TFRZ_S_SQUARE * (S_safe * S_safe)    # S^2
        + _TFRZ_P_DBAR * p_dbar                 # pressure lowering (<= 0)
    )
    return constants.T_freeze + depression_c


def slab_freeze_point_K(T_freeze_const, scheme: str = "constant"):
    """Effective seawater freezing point [K] for slab-ocean freeze clamps.

    The slab / two-layer mixed-layer oceans (``simple_ocean.py`` on the
    structured grid, ``simple_ocean_mpas.py`` on the Voronoi mesh) carry no
    prognostic salinity, so a liquidus scheme is evaluated at the fixed
    reference ocean salinity ``constants.S_ocean_ref`` (an environmental
    reference, NOT a tunable).  ``"constant"`` (default) returns the caller's
    ``T_freeze_const`` unchanged -- byte-identical to the historical clamp.
    A typo'd scheme can never silently fall back to the constant: anything
    other than ``"constant"`` is dispatched to :func:`freezing_point`, which
    raises ``ValueError`` on an unknown scheme.

    SINGLE OWNER (MED-1 follow-up): both slab modules call THIS helper for
    both the ``jnp.maximum`` clamp and the ``Q_freeze`` diagnostic -- do not
    re-derive the constant-vs-liquidus branch in a consumer.  ``scheme`` is a
    static config field in every caller, so the Python ``if`` is
    feature-gating, not a data-dependent traced select.
    """
    if scheme == "constant":
        return T_freeze_const
    return freezing_point(constants.S_ocean_ref, 0.0, scheme=scheme)
