"""Prognostic TKE vertical mixing (Gaspar 1990 / Burchard 2002).

Veros's canonical vertical-mixing closure
(``settings.enable_tke = True``). Solves a prognostic budget for the
turbulent kinetic energy per unit mass ``TKE [m^2/s^2]`` at cell
interfaces, then derives ``K_M`` (momentum) and ``K_H`` (tracer)
eddy diffusivities from the TKE field and a mixing-length closure.

Closure equations
-----------------

Per column at interfaces (between cells ``k`` and ``k+1``):

.. math::

   \\frac{\\partial e}{\\partial t}
       = P_s + P_b - \\varepsilon
       + \\frac{\\partial}{\\partial z}
         \\left( \\alpha_{\\text{tke}} \\, K_M
                 \\, \\frac{\\partial e}{\\partial z} \\right)

with

.. math::

   P_s &= K_M \\left[
             \\left(\\frac{\\partial u}{\\partial z}\\right)^2
           + \\left(\\frac{\\partial v}{\\partial z}\\right)^2
         \\right]                 \\;\\text{(shear production)}\\\\

   P_b &= -K_H \\, N^2             \\;\\text{(buoyancy work)}\\\\

   \\varepsilon
       &= c_\\varepsilon \\frac{e^{3/2}}{l_\\varepsilon}
                                  \\;\\text{(dissipation)}\\\\

   K_M &= c_k \\, l_k \\, \\sqrt{2 e}\\\\

   K_H &= K_M \\quad
              (\\text{turbulent Prandtl number} = 1
              \\text{ in the Gaspar / Burchard / Veros canonical form}).

Mixing lengths
--------------

The Bougeault-Lacarrere (1989) asymmetric construction
(``tke_mxl_choice = 2``, Veros default):

.. math::

   l_{up}(z) \\;=\\; \\text{vertical distance a parcel of TKE }e\\,
       \\text{rises before }\\int_z^{z+l_{up}} N^2\\,dz'\\;=\\;e\\\\

   l_{dn}(z) \\;=\\; \\text{symmetric, integrated downward}\\\\

   l_k = \\sqrt{l_{up}\\,l_{dn}}, \\qquad
   l_\\varepsilon = \\max(l_{up},\\,l_{dn})

with a floor ``mxl_min`` enforced everywhere.

Surface flux
------------

::

    forc_tke_surface = (|tau| / rho_0)^{3/2}     [m^3 / s^3]

Applied as a flux boundary condition at the top interface (the
top-of-column TKE diffusion flux).

Implementation notes
--------------------

This implementation expects velocities and tracers at **cell centres**
(shape ``(..., nlev)``). On lat-lon C-grid / MPAS / cubed-sphere the
caller must interpolate before invoking
:func:`tke_vertical_mixing`. Density at cell centres is derived from
T and S via the supplied EOS function.

Vertical positions, interface levels:

- ``e[..., k]`` (TKE) lives at interface ``k`` for
  ``k = 0, ..., nlev - 2`` between cell centres ``k`` and ``k + 1``.
  Total length ``nlev - 1``.
- Cell-centre fields (T, S, u, v) have length ``nlev``.

The prognostic step is solved backward-Euler (linear-in-TKE forms of
the source terms with the dissipation linearised about the previous
TKE) → one tridiagonal solve per column. Stable for the long ocean
``dt`` used in production.

References
----------
- Gaspar, P., Y. Gregoris, and J.-M. Lefevre (1990): A simple eddy
  kinetic energy model for simulations of the oceanic vertical mixing.
  *J. Geophys. Res.*, 95, 16179-16193.
- Bougeault, P. and P. Lacarrere (1989): Parameterization of orography-
  induced turbulence in a mesobeta-scale model. *Mon. Weather Rev.*,
  117, 1872-1890.
- Burchard, H. (2002): Energy-conserving discretisation of turbulent
  shear and buoyancy production. *Ocean Modelling*, 4, 347-361.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.source_rounding import nemo_source_round
from legoesm.core.transcendentals import tanh as precision_tanh
from legoesm.ocean.physics.vertical_mixing._glibc234_exp_table import (
    GLIBC234_EXP_TABLE_BITS,
)
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

__physics_contract__ = {
    "summary": (
        "Prognostic TKE vertical mixing (Gaspar 1990 / Burchard 2002, Veros "
        "enable_tke): advance a turbulent-kinetic-energy budget (shear + "
        "buoyancy production, dissipation, TKE diffusion) with "
        "Bougeault-Lacarrere mixing lengths, then set K_M = c_k*l_k*sqrt(2e) "
        "and K_H = K_M/Pr."
    ),
    "inputs": {
        "u_cell": "m/s", "v_cell": "m/s", "T_cell": "degC", "S_cell": "psu",
        "rho_cell": "kg/m^3", "dz_half": "m", "tke_old": "m^2/s^2",
        "tau_x_surface": "N/m^2", "tau_y_surface": "N/m^2", "dt": "s",
    },
    "outputs": {
        "K_M": "m^2/s", "K_H": "m^2/s", "tke_new": "m^2/s^2",
    },
    "sign_convention": (
        "TKE e >= tke_background >= 0; K_M, K_H >= 0; shear production P_s >= 0, "
        "dissipation eps >= 0 (a sink), buoyancy work P_b = -K_H*N^2 (a source "
        "when N^2<0); surface TKE flux (|tau|/rho_0)^{3/2} injected as a flux BC "
        "at the top interface; z positive up. The TKE budget has genuine "
        "sources/sinks so nothing is conserved; K_M/K_H close the momentum and "
        "tracer budgets in the solver."
    ),
    # Prognostic-TKE diffusivity producer (like CATKE); dissipative budget, so
    # nothing is conserved by the closure itself.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Gaspar, P. et al. (1990), JGR 95, 16179-16193; Burchard, H. (2002); "
        "Bougeault & Lacarrere (1989), MWR 117, 1872-1890"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_tke_closure.py + "
        "tests/ocean/unit/test_tke_prognostic.py — wind-forced surface layer "
        "builds TKE and K_M; a stratified quiescent column decays toward "
        "background TKE; Kato-Phillips mixed-layer deepening."
    ),
}

_EPS = float(jnp.finfo(jnp.float32).eps)

# --- NEMO zdftke surface-term constants (NEMO 5.0.1 src/OCE/ZDF/zdftke.F90) ---
# Fixed published scheme constants, NOT tunables (the tunables are
# TKEConfig.lc_coeff / etau_frac — NEMO rn_lc / rn_efr).
_NEMO_TKE_RHO_AIR = 1.22       # zrhoa  [kg/m³] air density         (zdftke.F90:221)
_NEMO_TKE_CDRAG = 1.5e-3       # zcdrag [-] surface drag coeff      (zdftke.F90:222)
# ½·0.016² / (ρ_air·C_d): surface stress → ½W_lc² (Axell 2002 Eq. 44, via
# |τ| = ρ_air·C_d·U₁₀² and the Stokes drift u_s = 0.016·U₁₀) (zdftke.F90:243)
_NEMO_TKE_LC_CSD = 0.5 * 0.016 * 0.016 / (_NEMO_TKE_RHO_AIR * _NEMO_TKE_CDRAG)
_NEMO_MXL0_VKARMN = 0.4        # vkarmn (phycst) — the ln_mxl0 anchor prefactor
_NEMO_MXL0_LENGTH_SCALE = 2.0e5  # zraug numerator; shipped zdftke.F90:575
_NEMO_MOLECULAR_VISCOSITY = 1.0e-6


def _mixing_length_floor(cfg: "TKEConfig"):
    """Return the active scheme's mixing-length floor.

    ``zdf_tke_init`` chooses ``rmxl_min`` in two arms
    (shipped ``zdftke.F90:841-848``):

    * ``ln_zdfiwm = .TRUE.`` FORCES ``rn_emin = 1.e-10_wp`` and
      ``rmxl_min = 1.e-03_wp`` (``:842-843``) and never evaluates the
      derivation below.  A card on that arm therefore carries ``1.0e-3`` in
      ``cfg.mxl_min`` and leaves ``nemo_derived_mxl_min`` False (ORCA1,
      ORCA2).
    * ``ln_zdfiwm = .FALSE.`` derives
      ``rmxl_min = 1.e-6_wp / (rn_ediff*SQRT(rn_emin))`` (``:846``; GYRE
      preprocessed ``zdftke.f90:815-817``) in binary64.  legoESM's
      corresponding card fields are ``c_k`` (``rn_ediff``) and
      ``tke_background`` (``rn_emin``).  Keep the source association exactly;
      in particular, do not replace division by a reciprocal.

    ``nemo_derived_mxl_min`` selects the second arm.  It is False by default,
    so every card that does not ask for the derivation — Veros choices, FESOM,
    and every ln_zdfiwm card — keeps its own configured ``mxl_min``.
    """
    if not cfg.nemo_derived_mxl_min:
        return cfg.mxl_min
    if not jax.config.x64_enabled:
        raise ValueError(
            "TKEConfig.nemo_derived_mxl_min=True (NEMO-derived rmxl_min) "
            "requires JAX binary64 enabled")
    rn_ediff = jnp.asarray(cfg.c_k, dtype=jnp.float64)
    rn_emin = jnp.asarray(cfg.tke_background, dtype=jnp.float64)
    return (jnp.asarray(_NEMO_MOLECULAR_VISCOSITY, dtype=jnp.float64)
            / (rn_ediff * jnp.sqrt(rn_emin)))


def _mxl0_surface_anchor(
    cfg: "TKEConfig", taum, rho_0: float, g: float, surface_tmask=None,
):
    """ln_mxl0 surface anchor (shipped zdftke.F90:575,602,640-642).

    zdf_tke_init first overwrites rn_mxl0 with the derived rmxl_min when
    ln_mxl0 is true (shipped zdftke.F90:859-862; GYRE ppsrc:829-832), then
    tke_avn evaluates
    l_sfc=max(rn_mxl0,vkarmn*2e5/(rho0*g)*taum). None unless the choice is a
    NEMO nn_mxl scheme (3 = nn_mxl=3, 4 = nn_mxl=2); ORCA1 sets ln_mxl0=.true.
    independently of nn_mxl, so BOTH need the anchor."""
    if cfg.tke_mxl_choice not in (3, 4):
        return None
    taum = jnp.asarray(taum)
    if not cfg.nemo_mxl0_surface_tmask:
        # Default (main's behaviour): unmasked stress.  Callers with no
        # surface T-mask (FESOM) stay supported; NEMO-literal cards opt into
        # the compiled masked statement with nemo_mxl0_surface_tmask=True.
        masked_taum = jnp.maximum(taum, 0.0)
    else:
        if surface_tmask is None:
            raise ValueError(
                "TKEConfig.nemo_mxl0_surface_tmask=True requires "
                "surface_tmask for the compiled `taum*tmask(:,:,1)` ln_mxl0 "
                "statement (zdftke.F90:602).")
        surface_tmask = jnp.asarray(surface_tmask, dtype=taum.dtype)
        if surface_tmask.shape != taum.shape:
            raise ValueError(
                "surface_tmask must match taum; got "
                f"{surface_tmask.shape} vs {taum.shape}.")
        masked_taum = jnp.maximum(taum, 0.0) * surface_tmask
    return jnp.maximum(
        jnp.asarray(_mixing_length_floor(cfg), dtype=taum.dtype),
        _NEMO_MXL0_VKARMN * _NEMO_MXL0_LENGTH_SCALE / (rho_0 * g)
        * masked_taum)
_NEMO_TKE_EBB = 67.83          # rn_ebb  namelist_ref default — surface TKE input coef
_NEMO_TKE_EMIN0 = 1.0e-4       # rn_emin0 [m²/s²] surface TKE minimum


def nemo_surface_avm(
    cfg: "TKEConfig", e_sfc: jnp.ndarray, l_sfc: jnp.ndarray | None,
) -> jnp.ndarray:
    r"""NEMO's TRUE surface-w-level viscosity ``avm(jk=1)`` (T3-exact).

    ``tke_avn`` (zdftke.F90:713-715) evaluates avm at EVERY w-level
    ``jk=1..jpkm1`` from the SAME formula used in the interior:

    .. math::

        avm(1) = \max(rn\_ediff \cdot zmxlm(1) \cdot \sqrt{en(1)},\ avmb(1))

    with ``en(1)`` the surface Dirichlet TKE value (``e_sfc``,
    :func:`_surface_tke_dirichlet`) and ``zmxlm(1)`` the ``ln_mxl0`` surface
    mixing-length anchor (:func:`_mxl0_surface_anchor`, already computed by
    the orchestrator for the ``nn_mxl=3`` sweeps — reused here, not
    re-derived). ``rn_ediff`` is legoESM's ``c_k``; the ``sqrt(en)`` (not
    ``sqrt(2*en)``) form matches ``kappa_convention="veros_sqrte"`` (the
    kamm-card selection) exactly — see :func:`compute_K_from_tke`.

    This is the value NEMO's ``jk=2`` row actually uses in
    ``0.5*(avm(2)+avm(1))`` (zdftke.F90:407-410); the interior floor
    ``avmb`` is legoESM's ``kappaM_min``.

    Parameters
    ----------
    e_sfc : (...,) — surface Dirichlet TKE value (``en(1)``).
    l_sfc : (...,) or None — ``ln_mxl0`` surface mixing length
        (:func:`_mxl0_surface_anchor`; None only when the choice is not a
        NEMO nn_mxl scheme (3 or 4),
        in which case NEMO's ``zmxlm(1)`` formula does not apply — the caller
        must not invoke this function in that regime).

    Returns
    -------
    (...,) avm(1), NEMO's true surface-w-level viscosity, floored at
    ``kappaM_min``.
    """
    if l_sfc is None:
        raise ValueError(
            "nemo_surface_avm requires l_sfc (the ln_mxl0 surface mixing "
            "length, tke_mxl_choice=3) — NEMO's zmxlm(1) has no meaning "
            "under a different mixing-length choice.")
    e_safe = jnp.maximum(jnp.asarray(e_sfc), 0.0)
    zav = cfg.c_k * jnp.asarray(l_sfc) * jnp.sqrt(e_safe)
    return jnp.maximum(zav, cfg.kappaM_min)


def _surface_tke_dirichlet(cfg: "TKEConfig", taum, rho_0: float):
    """NEMO nn_bc_surf=1 Dirichlet surface-TKE value, or None for the Veros
    flux BC — the single owner of the ``TKEConfig.surface_bc`` dispatch shared
    by BOTH TKE entry points (``tke_vertical_mixing`` and the post-mixing
    ``tke_set_diffusivities`` path).  Raises on an unknown value.

    NEMO zdftke.F90:264-269: ``en(1) = MAX(rn_emin0, rn_ebb*|tau|/rho0)`` held
    as the top boundary value of the implicit solve.
    """
    _sbc = getattr(cfg, "surface_bc", "veros_flux")
    if _sbc == "nemo_dirichlet":
        return jnp.maximum(
            jnp.asarray(cfg.tke_surface_min, dtype=taum.dtype),
            _NEMO_TKE_EBB / rho_0 * taum)
    if _sbc == "veros_flux":
        return None
    raise ValueError(
        "Unknown surface-TKE boundary scheme TKEConfig.surface_bc: must be "
        f'one of ("veros_flux", "nemo_dirichlet"), got {_sbc!r}')
# 0.001875 = (rn_ebb0/rho0)*0.5 = 3.75*0.5/1000 (zdftke.F90:284, CAUTION the
# NEMO rCdU_bot it multiplies is <= 0 there — legoESM's r is the POSITIVE
# +Cd*|U| convention, so no sign flip is needed here; see
# nemo_bottom_tke_dirichlet).
_NEMO_TKE_BOTTOM_EBB0_HALF = 0.001875


def nemo_bottom_tke_dirichlet(
    r_bottom_drag: jnp.ndarray, u_bot: jnp.ndarray, v_bot: jnp.ndarray,
    cfg: "TKEConfig",
) -> jnp.ndarray:
    r"""NEMO bottom-friction TKE BC (T15, zdftke.F90:279-288).

    .. math::

        en(m_{bkt}+1) = \max(0.001875\cdot r_{bot}\cdot|u_{bot}|,\;
                             rn\_emin)

    with ``r_bot`` the NEMO non-linear/log-layer bottom-drag rate at the
    TRACER point (``+Cd·|U|``, legoESM's positive convention — see
    :func:`legoesm.ocean.dynamics.ocean_tendency_common.nemo_effective_bottom_drag_r`;
    single-owner doctrine, no re-derived drag coefficient). ``rn_emin`` is
    ``cfg.tke_background`` (legoESM's interior TKE floor — same value as
    NEMO's namelist default 1e-6 m²/s²).

    CAUTION — ``u_bot``/``v_bot`` are NOT the plain T-point average that
    ``nemo_effective_bottom_drag_r`` takes. zdftke uses its own velocity
    convention: the WET-ONLY SUM ``zmsku*( uu(ji) + uu(ji-1) )`` with
    ``zmsku = 2 - umask(ji-1)*umask(ji)`` and NO ``0.5``
    (zdftke.F90:282-287; contrast zdfgls.F90:196-197, which writes the same
    mask expression WITH the ``0.5``). That missing ``0.5`` cancels the one
    inside the ``0.001875`` prefactor (``= (rn_ebb0/rho0)*0.5``,
    zdftke.F90:284), leaving ``en_bot = (rn_ebb0/rho0)*Cd|U|^2``, i.e.
    proportional to ``u_*^2`` exactly like the surface BC at :266 — so the
    factor 2 is structural, not a NEMO slip. The caller
    (``_tke_bottom_dirichlet``) masks the faces explicitly and supplies that
    form; do not pass a plain average here.

    Parameters
    ----------
    r_bottom_drag : (...,) — bottom-drag rate at T-points [m/s], >= 0.
    u_bot, v_bot : (...,) — ``zmsk``-weighted wet-only velocity SUM at the
        bottom T-point [m/s] (see CAUTION above), not the plain average.
    cfg : TKEConfig (uses ``tke_background`` as the ``rn_emin`` floor).

    Returns
    -------
    (...,) the Dirichlet TKE value for the deepest carried interface.
    """
    speed = _safe_stress_modulus(u_bot, v_bot)   # AD-safe |u_bot| at rest
    return jnp.maximum(
        _NEMO_TKE_BOTTOM_EBB0_HALF * r_bottom_drag * speed,
        jnp.asarray(cfg.tke_background, dtype=speed.dtype))
# nn_htau=1 latitude profile: h_tau = max(0.5, min(30, 45·|sin φ|)) m
_NEMO_TKE_HTAU_CONST_M = 10.0  # nn_htau=0 constant penetration depth [m]
_NEMO_TKE_HTAU_MIN_M = 0.5
_NEMO_TKE_HTAU_MAX_M = 30.0
_NEMO_TKE_HTAU_SLOPE_M = 45.0


def _safe_stress_modulus(tx: jnp.ndarray, ty: jnp.ndarray) -> jnp.ndarray:
    """AD-safe |τ| = sqrt(τx² + τy²) with a finite (zero) gradient at τ=0.

    The plain ``jnp.sqrt(tx*tx + ty*ty)`` has a NaN reverse-mode gradient at
    ``tx = ty = 0`` (``d/dx sqrt(x) = 1/(2 sqrt(x))`` → ``0 * inf`` in the VJP),
    though the analytic limit of the stress modulus and its contribution to the
    surface TKE flux ``(|τ|/ρ₀)^{3/2}`` is 0 there. The double-``where`` keeps
    the primal BIT-IDENTICAL to the plain form where ``|τ|² > 0`` and yields a
    finite 0 (with 0 gradient) at zero stress — the same pattern used for the
    AD-safe sqrt elsewhere in this module (see ``_veros_buoyancy_length``).
    """
    t2 = tx * tx + ty * ty
    return jnp.where(t2 > 0.0, jnp.sqrt(jnp.where(t2 > 0.0, t2, 1.0)), 0.0)


class TKEOutput(NamedTuple):
    """Output of :func:`tke_vertical_mixing`."""
    K_M: jnp.ndarray       # (..., nlev-1) momentum eddy viscosity at interfaces
    K_H: jnp.ndarray       # (..., nlev-1) tracer eddy diffusivity at interfaces
    tke_new: jnp.ndarray   # (..., nlev-1) updated TKE at interfaces
    l_eps: jnp.ndarray     # (..., nlev-1) dissipation mixing length (diagnostic)
    K_M_surface: jnp.ndarray | None = None  # (...) post-tke_avn surface avm_k
    dissl: jnp.ndarray | None = None  # (...) carried post-tke_avn sqrt(en)/zmxld
    statement_trace: "TKEStatementTrace | None" = None


class TKEStatementTrace(NamedTuple):
    """WRITE-only production boundaries matching compiled ``tke_tke``.

    ``en_entry`` is the represented prognostic field (NEMO levels
    2:jpkm1).  Every later value prepends the separately held z=0 surface
    row and therefore spans NEMO levels 1:jpkm1.  This trace is built only
    for the private stage-twin path; ordinary model steps request no trace.

    ``matrix_upper``/``matrix_lower``/``matrix_diag`` are the model's
    zd_up/zd_lw/zdiag over NEMO levels 2:jpkm1 exactly as the compiled
    assignments zdftke.f90:434/435/436 leave them, captured before the
    extended-system concatenation (which zeroes the deepest super-diagonal
    for the back-substitution and would not match NEMO's recorded value).
    ``rhs_shear`` is the p_sh2 operand the RHS assignment zdftke.f90:439
    consumes, over the same 2:jpkm1 domain.
    """

    en_entry: jnp.ndarray
    taum_surface: jnp.ndarray
    surface_dirichlet: jnp.ndarray
    en_after_boundaries: jnp.ndarray
    en_after_langmuir: jnp.ndarray
    rhs_pre_sweep: jnp.ndarray
    en_post_sweep: jnp.ndarray
    matrix_upper: jnp.ndarray
    matrix_lower: jnp.ndarray
    matrix_diag: jnp.ndarray
    rhs_shear: jnp.ndarray
    shear_face_metrics: object = None
    rhs_intermediate: object = None
    bn2_intermediate: object = None
    bn2_output: object = None


class TKEEntryN2Bundle(NamedTuple):
    """Step-entry NEMO stability and live W-grid operands.

    NEMO evaluates ``rn2``/``rn2b`` before ``zdf_phy`` and then keeps the
    effective ``Kmm`` geometry unchanged while ``zdf_mxl``/``zdf_tke`` run.
    This trace-local bundle gives legoESM the same operand lifetime without
    adding prognostic state.
    """
    rn2: jnp.ndarray
    rn2b: jnp.ndarray
    gdepw_Kmm: jnp.ndarray
    e3w_Kmm: jnp.ndarray
    e3t_Kmm: jnp.ndarray
    # Full-grid surface W thickness from raw nemo_e3w_0*(1+r3t). The interior
    # e3w_Kmm field above intentionally has nlev-1 eosbn2 interfaces.
    e3w_surface_Kmm: jnp.ndarray | None = None
    bn2_intermediate: object = None


class TKECarryOutput(NamedTuple):
    """Prognostic TKE plus NEMO's post-``tke_avn`` closure memory."""
    tke_new: jnp.ndarray
    tke_entry: jnp.ndarray
    K_M: jnp.ndarray
    K_H: jnp.ndarray
    K_M_surface: jnp.ndarray | None
    dissl: jnp.ndarray | None = None
    statement_trace: TKEStatementTrace | None = None


class TKEPostMixingContext(NamedTuple):
    """Phase-1 closure ingredients for the POST-MIXING TKE solve.

    Produced by :func:`tke_set_diffusivities` (the legoESM analogue of Veros
    ``set_tke_diffusivities``, tke.py:20-113, evaluated from the CARRIED
    ``tke_old`` = Veros ``tke[tau]``) and consumed by
    :func:`tke_integrate_post_mixing` AFTER the implicit tracer solve
    (``TKEConfig.buoyancy_timing="post_mixing_veros"``). A plain trace-local
    container — never crosses a JIT boundary as an argument.
    """
    K_M_old: jnp.ndarray      # (..., nlev-1) kappaM from tke[tau] (feeds friction)
    K_H_old: jnp.ndarray      # (..., nlev-1) kappaH from tke[tau] (feeds tracers + P_diss_v)
    mxl: jnp.ndarray          # (..., nlev-1) buoyancy mixing length (tau)
    sqrttke: jnp.ndarray      # (..., nlev-1) sqrt(max(0, tke[tau]))
    shear_sq: jnp.ndarray     # (..., nlev-1) |du/dz|²+|dv/dz|² (pre-solve)
    tke_old: jnp.ndarray      # (..., nlev-1) the carried TKE
    surface_flux: jnp.ndarray  # (...) wind-work TKE flux (|tau|/rho_0)^{3/2}
    dz_half: jnp.ndarray      # (..., nlev-1) Veros dzw (u_centered centre spacing · J)
    dz_cell: jnp.ndarray      # (..., nlev) cell thicknesses dzt·J
    dz_surface: jnp.ndarray   # (...) surface W half-volume 0.5·dzw_top
    p_cell: jnp.ndarray       # (..., nlev) cell-centre hydrostatic pressure [Pa]
    eos_fn: object            # EOS callable (T, S, p) -> rho (static)
    rho_0: float
    g: float
    # NEMO nn_bc_surf=1 Dirichlet surface-TKE value (None => Veros flux BC).
    surface_dirichlet: jnp.ndarray | None = None
    # NEMO ln_lc Langmuir TKE source on the interior interfaces (None => off).
    langmuir_source: jnp.ndarray | None = None
    # DISSIPATION mixing length l_eps (nn_mxl=3: sqrt(lup*ldown), zdftke:672
    # zmxld feeding dissl=sqrt(en)/zmxld :735). Distinct from ``mxl`` (=l_k=
    # min(lup,ldown), the eddy-coefficient length :730) only for choice 3;
    # None => fall back to ``mxl`` (choices 1/2, where the two coincide, and
    # older ctx constructions).
    l_eps: jnp.ndarray | None = None


# ---------------------------------------------------------------------------
# Mixing length: Bougeault-Lacarrere (1989) asymmetric construction.
# ---------------------------------------------------------------------------


def _bougeault_lacarrere_lengths(
    e: jnp.ndarray,
    N2: jnp.ndarray,
    dz_half: jnp.ndarray,
    mxl_min: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute upward and downward mixing lengths l_up, l_dn.

    Parameters
    ----------
    e : (..., nlev-1)
        TKE at interfaces.
    N2 : (..., nlev-1)
        Squared Brunt-Vaisala frequency at interfaces (>= 0 for stable
        stratification; this function expects ``N2 >= 0``).
    dz_half : (..., nlev-1) or (..., nlev-2)
        Distance between adjacent interface levels. Both length forms
        appear in different ocean modules; this function uses
        ``dz_half[..., k]`` as the distance between interface ``k``
        and interface ``k+1``. Pass either shape; we slice to
        ``(..., nlev-2)`` internally.
    mxl_min : float
        Floor applied to both lengths to prevent zero division and
        to model the no-slip-boundary residual length.

    Returns
    -------
    l_up, l_dn : (..., nlev-1)
    """
    # Distance between adjacent interfaces; length nlev-2 derived from the
    # nlev-1 e-array.
    dz_int = dz_half[..., :e.shape[-1] - 1]   # (..., nlev-2)

    # TKE budget along the asymmetric path:
    #     Σ_k N2[k] · l_up[k] · dz_int[k] = e at the starting interface
    # Discretely this is a running cumulative sum that we stop when the
    # accumulated PE exceeds the available TKE. The continuous form
    # gives the inverse:  l_up · sqrt(N2) ≈ sqrt(2e) so
    # l_up ≈ sqrt(2e / max(N2, eps)).  This is the canonical
    # closed-form approximation used in Gaspar 1990 / Veros — full
    # integral is more expensive and only marginally improves K profiles.
    N2_safe = jnp.maximum(N2, _EPS)
    # AD-safe sqrt: the closed form ``sqrt(2e/N²_safe + eps)`` is NaN in BOTH
    # value and gradient once the argument goes negative (e < 0, the energy
    # debt). The double-``where`` keeps the primal BIT-IDENTICAL where the
    # argument is positive (the only regime any shipped recipe reaches — this
    # legacy in-situ/non-signed-N² branch clips N² ≥ 0 and runs with
    # positivity='floor' so e ≥ tke_background > 0) and yields a finite 0
    # (with 0 gradient) otherwise. Mirrors the choice=1 / ``veros_sqrte``
    # fix; closes the same NaN class on this branch.
    _bl_arg = 2.0 * e / N2_safe + _EPS
    l_up_raw = jnp.where(
        _bl_arg > 0.0, jnp.sqrt(jnp.where(_bl_arg > 0.0, _bl_arg, 1.0)), 0.0)
    l_dn_raw = l_up_raw   # symmetric under the closed-form approximation

    # Apply mxl_min floor. ``dz_int`` is also used as a per-cell upper
    # bound so the length cannot exceed one cell — prevents the
    # closed-form expression from blowing up in nearly-neutral layers
    # (N2 → 0). ``2 * dz_int`` is a typical upper bound used by Veros's
    # implementation; we apply it conservatively.
    upper = 2.0 * jnp.maximum(dz_int, mxl_min)
    # broadcast upper to (..., nlev-1) by repeating the last value if needed
    if upper.shape[-1] < l_up_raw.shape[-1]:
        upper = jnp.concatenate([upper, upper[..., -1:]], axis=-1)
    l_up = jnp.clip(l_up_raw, mxl_min, upper)
    l_dn = jnp.clip(l_dn_raw, mxl_min, upper)
    return l_up, l_dn


def _veros_buoyancy_length(
    e: jnp.ndarray,
    N2: jnp.ndarray,
    dz_int: jnp.ndarray,
    mxl_min: float,
    *,
    dz_cell: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Veros buoyancy mixing length (``tke_mxl_choice=2``), SIGNED-N^2 aware.

    Faithful port of ``veros/core/tke.py:34,48-65``. The raw buoyancy
    length is

    .. math::

        l = \sqrt{2}\,\sqrt{e} \,/\, \sqrt{\max(10^{-12},\,N^2)}

    which **blows up** where ``N^2 <= 0`` (a statically unstable interface):
    the floor ``1e-12`` makes ``l`` enormous, so the parcel mixes across
    the whole unstable column — this is how Veros's TKE convects. The
    growth is then bounded by the MITgcm/OPA two-pass limiter (``l`` may
    increase by at most one cell thickness ``dz`` per level going up and
    going down) and floored at ``mxl_min``. Crucially this is **not** the
    legacy 2-cell hard cap (:func:`_bougeault_lacarrere_lengths`), which
    would suppress the convective blow-up.

    legoESM grid note: legoESM carries TKE at the ``nlev-1`` interior
    interfaces (Veros's W-grid), and ``dz_int[k]`` is the spacing between
    interface ``k`` and ``k+1``. NOTE (metric slot): Veros's per-step growth
    allowance is the CELL thickness ``dzt`` (tke.py:56,62), which on a
    stretched u_centered grid is NOT equal to the interface spacing
    ``dz_int`` (= ``dzw``). The legacy branch below (``dz_cell=None``,
    BIT-IDENTICAL) keeps the historical ``dz_int`` allowance; the
    Veros-faithful branch takes ``dz_cell`` (see below).

    Parameters (faithful branch)
    ----------------------------
    dz_cell : (..., nlev) or None (keyword-only)
        ACTUAL cell thicknesses ``dzt·J`` at the ``nlev`` cell centres.
        When given (``TKEConfig.veros_dz_slots=True``), the limiter is the
        faithful port of Veros tke.py:54-65 in BOTH slot and pass order:

        1. downward sweep from the surface — interface ``k`` is limited by
           the interface ABOVE plus the intervening cell: ``mxl[k] <=
           mxl[k-1] + dzt[k]`` (Veros backwards_pass, bottom-up
           ``mxl[k] <= mxl[k+1] + dzt[k+1]``);
        2. upward sweep from the bottom — ``mxl[k] <= mxl[k+1] +
           dzt[k+1]`` (Veros forwards_pass ``mxl[k] <= mxl[k-1] + dzt[k]``).

        Veros's surface anchor between the sweeps (``mxl[srf] <= mxl_min +
        dzt_top``, tke.py:59) constrains only the SURFACE W point at z=0,
        which legoESM does not carry (TKE lives on the nlev-1 interior
        interfaces); it never propagates into the interior because the
        downward sweep has already passed the top row — dropping it is
        exact for the interior values.

        The legacy branch instead runs (upward, then downward) with the
        ``dz_int`` allowance and no anchor — preserved byte-for-byte.

    Pure ``jax`` (``lax.fori_loop`` for the two sweeps); differentiable
    (the ``maximum``/``minimum`` floors are sub-gradient-safe).
    """
    n_int = e.shape[-1]
    # AD-safe sqrt at the negative-TKE energy debt: ``sqrt(max(0, e))`` has a
    # NaN derivative wherever e <= 0 (``d sqrt`` at 0 is inf; the ``max``
    # tangent there is 0; 0·inf = NaN — it poisons BOTH jvp and vjp through
    # any state carrying Veros's negative interior TKE, e.g. every parameter
    # gradient across >= 2 steps of the ACC recipe).  The double-``where``
    # keeps the primal BIT-IDENTICAL (sqrt is only evaluated where e > 0)
    # and makes the debt-branch derivative exactly 0 (the correct one-sided
    # derivative of the clamped primal).  Same pattern as the documented
    # N²-floor note in ``_gm_redi_common._eady_growth_and_length``.
    sqrttke = jnp.where(e > 0.0, jnp.sqrt(jnp.where(e > 0.0, e, 1.0)), 0.0)
    # Raw length: huge where N2 <= 0 (denominator -> sqrt(1e-12)).
    mxl = jnp.sqrt(2.0) * sqrttke / jnp.sqrt(jnp.maximum(1e-12, N2))

    if dz_cell is not None:
        # ---- Veros-faithful slots + pass order (tke.py:54-65) ----
        # (1) downward from the surface: mxl[k] <= mxl[k-1] + dzt[k]
        # (cell k separates interfaces k-1 and k, top-down).
        def _down(k, m):
            allow = m[..., k - 1] + dz_cell[..., k]
            return m.at[..., k].set(jnp.minimum(m[..., k], allow))

        mxl = jax.lax.fori_loop(1, max(n_int, 1), _down, mxl)

        # (2) upward from the bottom: mxl[k] <= mxl[k+1] + dzt[k+1]
        # (cell k+1 separates interfaces k and k+1).
        def _up(i, m):
            k = n_int - 2 - i
            allow = m[..., k + 1] + dz_cell[..., k + 1]
            return m.at[..., k].set(jnp.minimum(m[..., k], allow))

        mxl = jax.lax.fori_loop(0, max(n_int - 1, 0), _up, mxl)

        return jnp.maximum(mxl, mxl_min)

    # ---- Legacy branch (BIT-IDENTICAL): dz_int allowance, up-then-down ----
    # dz between adjacent interfaces (length n_int - 1); used as the
    # per-step growth allowance in both sweeps.
    dz_step = dz_int[..., :n_int - 1] if dz_int.shape[-1] >= n_int else dz_int

    # Static Python loop bounds (n_int is a compile-time shape) so the
    # fori_loop start/stop are concrete -> reverse-mode AD safe (dynamic
    # bounds break jax.grad). The index clamps below keep dz_step in range.
    _dz_last = dz_step.shape[-1] - 1

    # Backward pass (deep -> shallow): limit growth going up. Index k runs
    # from n_int-2 down to 0; mxl[k] <= mxl[k+1] + dz_step[k].
    def _backward(i, m):
        k = n_int - 2 - i
        allow = m[..., k + 1] + dz_step[..., jnp.minimum(k, _dz_last)]
        return m.at[..., k].set(jnp.minimum(m[..., k], allow))

    mxl = jax.lax.fori_loop(0, max(n_int - 1, 0), _backward, mxl)

    # Forward pass (shallow -> deep): limit growth going down.
    def _forward(k, m):
        allow = m[..., k - 1] + dz_step[..., jnp.minimum(k - 1, _dz_last)]
        return m.at[..., k].set(jnp.minimum(m[..., k], allow))

    mxl = jax.lax.fori_loop(1, max(n_int, 1), _forward, mxl)

    return jnp.maximum(mxl, mxl_min)


def veros_mxl_choice1_boundary_cap(
    z_interface: jnp.ndarray,
    dz_half: jnp.ndarray,
    column_depth: jnp.ndarray,
) -> jnp.ndarray:
    r"""Veros ``tke_mxl_choice=1`` distance-to-boundary cap (tke.py:39-47).

    Veros bounds the buoyancy mixing length by the geometric distance to the
    nearest boundary so a parcel can never mix across more than its distance
    to the surface or the seafloor:

    .. math::

        l \leftarrow \min\!\big(l,\; -zw + \tfrac12 dzw,\; ht + zw\big),
        \qquad l \leftarrow \max(l, mxl\_min)

    (``veros/core/tke.py:43-47``). This is the ``mxl_choice=1`` limiter; the
    ``mxl_choice=2`` MITgcm/OPA two-pass recursion lives in
    :func:`_veros_buoyancy_length`. Without it the raw length
    ``\sqrt{2e}/\sqrt{\max(10^{-12},N^2)}`` is UNBOUNDED where ``N^2 -> 0``
    (a statically-unstable surface interface): it overflows to ``+inf``,
    driving ``K_M``/``l_eps`` non-finite and the implicit momentum solve to
    NaN the first step the near-surface column convects (the measured
    global_1deg south-Pacific step-2 blowup). Only global_1deg selects
    ``mxl_choice=1``; the 4deg/flexible/acc setups use ``mxl_choice=2`` and
    never exercised this path.

    legoESM grid mapping (interior W-grid). legoESM carries TKE at the
    ``nlev-1`` INTERIOR interfaces, so Veros's per-W-point geometry maps as:

    - ``-zw`` -> ``-z_interface`` (depth below surface, ``z_interface`` is the
      static reference interface height ``z_half_ref[1:-1] < 0``);
    - ``dzw`` -> ``dz_half`` (static centre-to-centre spacing ``dz_half_ref``);
    - ``ht`` -> ``column_depth`` (per-column ocean depth ``H_bathy > 0``).

    The cap uses the STATIC reference geometry (as Veros does — the limiter
    reads the fixed ``zw``/``dzw``/``ht``, not the free-surface-perturbed
    layer thicknesses). Sub-seafloor interfaces get ``ht + zw < 0`` -> the
    ``min`` goes negative -> floored to ``mxl_min`` (matching Veros's
    ``maskW`` zeroing; those interfaces are masked downstream anyway).

    Parameters
    ----------
    z_interface : (nlev-1,)
        Static interior interface heights (negative, ``= z_half_ref[1:-1]``).
    dz_half : (nlev-1,)
        Static centre-to-centre spacing (positive, ``= dz_half_ref``).
    column_depth : (...,)
        Per-column ocean depth ``ht`` (positive metres, ``= H_bathy``).

    Returns
    -------
    bound : (..., nlev-1)
        ``min(-zw + dzw/2, ht + zw)`` — the per-interface upper bound for the
        mixing length BEFORE the ``mxl_min`` floor (applied by the caller).
        Pure arithmetic -> differentiable.
    """
    depth = -z_interface                          # (nlev-1,), >0   (= -zw)
    dist_surf = depth + 0.5 * dz_half             # (nlev-1,)       (-zw + dzw/2)
    dist_bot = column_depth[..., jnp.newaxis] + z_interface   # (...,nlev-1) ht+zw
    return jnp.minimum(dist_surf, dist_bot)


def compute_mixing_lengths(
    e: jnp.ndarray,
    N2: jnp.ndarray,
    dz_half: jnp.ndarray,
    cfg: TKEConfig,
    *,
    signed_n2: bool = False,
    dz_cell: jnp.ndarray | None = None,
    boundary_cap: jnp.ndarray | None = None,
    l_surface_anchor: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (l_k, l_eps) for the chosen ``tke_mxl_choice``.

    ``tke_mxl_choice=3`` is NEMO ``nn_mxl=3`` (zdftke.F90:658-672): the
    buoyancy length ``sqrt(2e)/N`` with the ``ln_mxl0`` wind-stress surface
    anchor (``l_surface_anchor``, computed by the caller from taum/rho_0/g),
    bounded by the lup/ldown |dl/dz|<=e3t sweeps;
    ``l_k = min(lup, ldown)``, ``l_eps = sqrt(lup*ldown)``.

    When ``signed_n2`` is True (the ``n2_mode="adiabatic"`` convective
    path), ``N2`` may be negative and the **Veros buoyancy length**
    (:func:`_veros_buoyancy_length`) is used so the length blows up across
    statically-unstable columns (convection). When False (legacy /
    in-situ), the bit-identical Bougeault-Lacarrere closed form with the
    2-cell cap is used.

    ``dz_cell`` (keyword-only, (..., nlev) actual cell thicknesses or
    None) selects the Veros-faithful ``dzt`` growth allowance + pass order
    inside :func:`_veros_buoyancy_length` (``TKEConfig.veros_dz_slots``).
    Only consulted on the ``signed_n2`` buoyancy-length branch; ``None``
    (default) is BIT-IDENTICAL legacy.

    NOTE on sub-seafloor rows: NEMO gets ``zmxlm == rmxl_min`` there for free
    because ``en`` is EXACTLY 0 below the seafloor
    (``en = MAX(en,rn_emin)*wmask``, DINO ``MY_SRC/zdftke.F90:565`` = upstream
    ``src/OCE/ZDF/zdftke.F90:469``) while the buoyancy-length line
    (``:759`` / ``:651``) carries no wmask. legoESM reproduces that by masking
    ``e`` itself in :func:`_solve_tke_backward_euler`
    (``TKEConfig.tke_dry_wmask``) — NOT by special-casing the length here.

    Returns
    -------
    l_k, l_eps : (..., nlev-1) — for use in K = c_k·l_k·sqrt(2e) and
        eps = c_eps·e^{3/2} / l_eps respectively.
    """
    mxl_min = _mixing_length_floor(cfg)
    if cfg.tke_mxl_choice == 2:
        if signed_n2:
            # Veros mxl_choice=2: a single length used for BOTH K_M
            # (l_k) and dissipation (l_eps), as in veros/core/tke.py.
            l_buoy = _veros_buoyancy_length(
                e, N2, dz_half, mxl_min, dz_cell=dz_cell)
            return l_buoy, l_buoy
        l_up, l_dn = _bougeault_lacarrere_lengths(
            e, N2, dz_half, mxl_min,
        )
        l_k = jnp.sqrt(jnp.maximum(l_up * l_dn, mxl_min ** 2))
        l_eps = jnp.maximum(l_up, l_dn)
    elif cfg.tke_mxl_choice in (3, 4):
        # --- NEMO nn_mxl=3 (choice 3) / nn_mxl=2 (choice 4) + ln_mxl0 ---
        # (GYRE ppsrc zdftke.f90:593-675). Both share the lup/ldown sweeps;
        # they differ ONLY in the final l_eps (see below).
        if dz_cell is None:
            raise ValueError(
                f"tke_mxl_choice={cfg.tke_mxl_choice} (NEMO nn_mxl) requires "
                "dz_cell (the e3t cell thicknesses) for the |dl/dz|<=e3t "
                "bounding sweeps.")
        # buoyancy length sqrt(2e)/N at interior interfaces, AD-safe at the
        # negative-TKE debt (same double-where idiom as choice 1/2).
        raw_evaluation = getattr(cfg, "tke_mxl_raw_evaluation", "factored")
        l_int = _tke_raw_mixing_length(e, N2, cfg)
        # ln_mxl0 surface anchor l_sfc = max(rn_mxl0, vkarmn*2e5/(rho0*g)*taum)
        # (shipped zdftke.F90:575,602,640-642), computed by the CALLER
        # (which owns taum/rho_0/g and the surface tmask)
        # and passed via l_surface_anchor. The no-anchor fallback used to claim
        # NEMO's ln_mxl0=F branch, but that branch uses raw rn_mxl0
        # (GYRE ppsrc zdftke.f90:614-615), not rmxl_min; fail closed because
        # legoESM exposes only the ln_mxl0=T NEMO path.
        # With ln_mxl0, NEMO overwrites the namelist rn_mxl0 with rmxl_min at
        # initialization (shipped zdftke.F90:859-862; GYRE ppsrc:829-832).
        if l_surface_anchor is not None:
            l_sfc = jnp.asarray(l_surface_anchor, dtype=l_int.dtype)
        else:
            raise ValueError(
                "NEMO tke_mxl_choice 3/4 requires l_surface_anchor; "
                "ln_mxl0=False would require a separate raw rn_mxl0 path.")
        # W-row stack: surface anchor + interior interfaces
        l_w = jnp.concatenate([l_sfc[..., None], l_int], axis=-1)  # (..., nlev)
        e3t = dz_cell                                              # (..., nlev) or (..., nlev+1)
        # lup: downward scan  l(k) = min(l(k-1) + e3t(k-1), l(k))
        def _down(carry, xs):
            l_km1 = carry
            l_k_, e3_km1 = xs
            out = jnp.minimum(l_km1 + e3_km1, l_k_)
            return out, out
        lT = jnp.moveaxis(l_w, -1, 0)                              # (nlev, ...)
        e3T_raw = jnp.moveaxis(e3t, -1, 0)
        # ``e3T_raw`` normally has the SAME row count as ``lT`` (the legacy,
        # documented ``dz_cell`` contract: ``e.shape[-1]+1`` rows, ending at
        # NEMO's ``e3t(jpkm1)``). #1226 zdftke_chain_walk (STAGE 4 finding):
        # the ldown sweep's seed step (below) actually needs NEMO's
        # ``e3t(jpk)`` — ONE ROW DEEPER than ``e3t(jpkm1)`` — which legoESM's
        # own (nlev T-cell -> nlev-1 interior-interface) grid has no analog
        # of and the legacy contract cannot supply. A caller that DOES have
        # that true extra row (e.g. an oracle-fidelity harness reading
        # NEMO's own mesh_mask e3t through jpk) may pass ONE more row;
        # detected here by STATIC shape (not traced), so every EXISTING
        # caller (``e.shape[-1]+1`` rows) is completely unaffected —
        # ``e3T``/``e3_bottom`` below are identical to the pre-fix arrays in
        # that case.
        n_e3 = e3T_raw.shape[0]
        n_l = lT.shape[0]
        if n_e3 == n_l + 1:
            e3T = e3T_raw[:-1]        # (nlev,), UNCHANGED vs the legacy contract
            e3_bottom = e3T_raw[-1]   # true e3t(jpk), the extra row
        elif n_e3 == n_l:
            e3T = e3T_raw             # legacy contract, bit-identical to before
            e3_bottom = e3T_raw[-1]   # proxy: e3t(jpkm1) (documented residual gap)
        else:
            raise ValueError(
                f"tke_mxl_choice=3: dz_cell has {n_e3} rows, expected "
                f"{n_l} (legacy, e.shape[-1]+1) or {n_l + 1} (with the "
                "extra true e3t(jpk) bottom row)."
            )
        _, lup_rest = jax.lax.scan(_down, lT[0], (lT[1:], e3T[:-1]))
        lup = jnp.concatenate([lT[:1], lup_rest], axis=0)
        # ldown: upward scan  l(k) = min(l(k+1) + e3t(k+1), l(k)),
        # jk = jpkm1 downto 2 (zdftke.F90:786-789, DINO MY_SRC copy).
        #
        # The carry MUST be seeded from NEMO's derived ``rmxl_min``,
        # not from ``lT[-1]`` (the raw buoyancy length at the deepest
        # carried row, NEMO jk=jpkm1). NEMO's ``zmxlm(:,:)`` is initialised
        # to ``rmxl_min`` for ALL jk (GYRE ppsrc zdftke.f90:593-595) BEFORE
        # the raw-fill at :619-621
        # loop, which only runs jk=2..jpkm1 (:739-742) — so ``zmxlm(jpk)``
        # is NEVER overwritten and stays at ``rmxl_min``. The ldown sweep's
        # FIRST iteration (jk=jpkm1) reads exactly that untouched
        # ``zmxlm(jpk)`` as ``zmxlm(jk+1)`` (:786-789), together with the
        # real ``e3t(jpk)`` (``e3_bottom`` above). The OLD code instead
        # seeded the carry from the RAW (unbounded) buoyancy length at
        # ``jk=jpkm1``, which can be arbitrarily large in a near-neutral
        # deep column (N²→0), leaking an unbounded length into the whole
        # ldown chain (the measured bug: legoESM l_k=3454.8m vs NEMO's
        # dumped 617.5m). Verified against tke_dump_zmxlm.bin (pathological
        # near-neutral column, #1226 walk, ``dz_cell`` widened by the extra
        # true e3t(jpk) row): reproduces NEMO's dumped value at jk=jpkm1 to
        # full float precision (max|diff|=0, standalone single-column
        # transcription). Without the extra row (legacy ``+1``-row
        # callers, e.g. production, whose grid has no jpk-th cell to
        # supply), ``e3_bottom`` falls back to ``e3t(jpkm1)`` — a
        # documented, BOUNDED proxy (was UNBOUNDED before this fix).
        _seed = jnp.broadcast_to(
            jnp.asarray(mxl_min, dtype=lT.dtype), lT.shape[1:])
        if raw_evaluation == "nemo_literal":
            # NEMO leaves zmxlm(jpk) at rmxl_min and uses that UNMODIFIED
            # terminal pad as the carry for the first jk=jpkm1 iteration.
            # Do not apply a fictitious update to jpk itself from raw en(jpk).
            _, ldn_rest = jax.lax.scan(
                _down, _seed, (lT[1:-1][::-1], e3T[2:][::-1]))
            ldn = jnp.concatenate(
                [lT[:1], ldn_rest[::-1], _seed[None]], axis=0)
        else:
            # Historical shared recurrence, retained bit-for-bit outside the
            # two literal DINO cards.
            first_ldn = jnp.minimum(_seed + e3_bottom, lT[-1])
            _, ldn_rest = jax.lax.scan(
                _down, first_ldn, (lT[1:-1][::-1], e3T[2:][::-1]))
            ldn = jnp.concatenate(
                [lT[:1], ldn_rest[::-1], first_ldn[None]], axis=0)
        lup = jnp.moveaxis(lup, 0, -1)[..., 1:]                    # interior
        ldn = jnp.moveaxis(ldn, 0, -1)[..., 1:]                    # interior
        l_k = jnp.maximum(jnp.minimum(lup, ldn), mxl_min)
        if cfg.tke_mxl_choice == 4:
            # --- NEMO nn_mxl=2 (zdftke.F90:680-688) ---
            # CASE(2) applies BOTH slope sweeps sequentially IN PLACE to one
            # array and sets zmxld = zmxlm, i.e. a SINGLE length for both the
            # eddy coefficient and the dissipation.  That sequential result
            # equals min(lup, ldown) exactly: the up-sweep of lup is the
            # maximal function <= lup obeying the upward slope bound;
            # min(lup,ldown) is <= lup and obeys both bounds, so up(lup) >=
            # min; and lup <= l_int gives up(lup) <= ldown, so up(lup) <= min.
            # Hence the only difference from nn_mxl=3 is l_eps.
            #
            # Because min(lup,ldown) <= sqrt(lup*ldown), nn_mxl=2 has the
            # SMALLER dissipation length, hence LARGER eps = c_eps*e^{3/2}/l_eps
            # and less retained TKE.  The two coincide where lup ~ ldown
            # (strong stratification, both branches locally limited) and differ
            # most where they diverge (weakly stratified deep columns far from
            # both boundaries) -- which is why this is a HIGH-LATITUDE-selective
            # lever, not a global one.  ORCA1's namelist runs nn_mxl=2.
            l_eps = l_k
        else:
            l_eps = jnp.maximum(jnp.sqrt(lup * ldn), mxl_min)
    elif cfg.tke_mxl_choice == 1:
        # Veros buoyancy length, ``tke_mxl_choice=1`` (veros/core/tke.py:30-47):
        #   sqrttke = sqrt(max(0, e));  mxl = sqrt(2)·sqrttke / sqrt(max(1e-12, N²))
        #   mxl = min(mxl, -zw + dzw/2, ht + zw);  mxl = max(mxl, mxl_min)
        # Two clamps, BOTH required to be debt-safe:
        #  (1) AD-safe ``sqrt(max(0, e))`` — the carried TKE holds Veros's
        #      interior NEGATIVE-energy debt; the clamp goes on ``sqrttke``
        #      BEFORE the division (the old ``sqrt(2·e/N²_safe)`` put raw e
        #      inside the sqrt ⇒ NaN derivative / NaN at e<0). Double-``where``
        #      keeps the primal == sqrt(max(0,e)) with a 0 debt-branch
        #      derivative (same idiom as line 287 / ``veros_sqrte``).
        #  (2) The Veros DISTANCE-TO-BOUNDARY cap (``boundary_cap``,
        #      :func:`veros_mxl_choice1_boundary_cap`). Without it the raw
        #      length is UNBOUNDED where N²→0 at the surface (denominator →
        #      sqrt(1e-12)) and overflows to +inf ⇒ K_M/l_eps non-finite ⇒
        #      momentum-solve NaN at the first convecting step (the measured
        #      global_1deg south-Pacific step-2 blowup). choice=2 is bounded
        #      by ``_veros_buoyancy_length``'s recursion, so 4deg/flexible/ACC
        #      never hit this. ``boundary_cap`` is computed once by the
        #      orchestrator (static zw/dzw + per-column H_bathy); None only on
        #      bare-call test paths (then the floor alone is kept, matching the
        #      pre-cap primal for already-bounded synthetic columns).
        sqrttke = jnp.where(
            e > 0.0, jnp.sqrt(jnp.where(e > 0.0, e, 1.0)), 0.0)
        N2_safe = jnp.maximum(N2, 1.0e-12)
        l_k = jnp.sqrt(2.0) * sqrttke / jnp.sqrt(N2_safe)
        if boundary_cap is not None:
            l_k = jnp.minimum(l_k, boundary_cap)
        l_k = jnp.maximum(l_k, mxl_min)
        l_eps = l_k
    else:
        raise ValueError(
            f"Unknown tke_mxl_choice={cfg.tke_mxl_choice!r}; expected 1 or 2 "
            f"(Veros), 3 (NEMO nn_mxl=3) or 4 (NEMO nn_mxl=2)."
        )
    return l_k, l_eps


def _tke_raw_mixing_length(
    e: jnp.ndarray,
    n2: jnp.ndarray,
    cfg: TKEConfig,
) -> jnp.ndarray:
    """Production selector for the pre-scan TKE buoyancy mixing length."""
    raw_evaluation = getattr(cfg, "tke_mxl_raw_evaluation", "factored")
    mxl_min = _mixing_length_floor(cfg)
    if raw_evaluation == "factored":
        # Historical shared expression: keep byte-identical for every
        # non-DINO consumer.
        sqrt2e = jnp.sqrt(2.0) * jnp.where(
            e > 0.0, jnp.sqrt(jnp.where(e > 0.0, e, 1.0)), 0.0)
        n_safe = jnp.sqrt(jnp.maximum(n2, 1.0e-12))
        return jnp.maximum(sqrt2e / n_safe, mxl_min)
    if raw_evaluation == "nemo_literal":
        # GYRE ppsrc zdftke.f90:619-621, compiled in binary64:
        # rsmall=0.5*EPSILON(1.e0), then SQRT((2*en)/zrn2).
        rsmall = 0.5 * jnp.finfo(e.dtype).eps
        zrn2 = jnp.maximum(n2, rsmall)
        return jnp.maximum(
            jnp.sqrt((jnp.asarray(2.0, e.dtype) * e) / zrn2),
            mxl_min)
    raise ValueError(
        "Unknown TKEConfig.tke_mxl_raw_evaluation: expected "
        f"'factored' or 'nemo_literal', got {raw_evaluation!r}.")


# ---------------------------------------------------------------------------
# Shear, buoyancy, dissipation at interfaces
# ---------------------------------------------------------------------------


# N²/shear/tridiagonal primitives are shared with CATKE — promoted to
# ``_shared`` (public) and imported here under the historical private names so
# the TKE call sites + behaviour stay bit-identical (no cross-module private
# import; one implementation of the column kernels).
from legoesm.ocean.physics.vertical_mixing._shared import (  # noqa: E402
    compute_N2 as _compute_N2,
    tridiag_thomas as _tridiag_thomas,
    vertical_shear_squared as _vertical_shear_squared,
)


# ---------------------------------------------------------------------------
# Prognostic TKE backward-Euler step
# (``_compute_N2`` / ``_tridiag_thomas`` now imported from ``_shared``.)
# ---------------------------------------------------------------------------


def _nemo_literal_tke_solve(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    rhs: jnp.ndarray,
    surface_en: jnp.ndarray,
    w_active: jnp.ndarray,
    floor: float,
) -> jnp.ndarray:
    """Solve the NEMO ``zdftke`` system in literal source order.

    ``a/b/c/rhs`` include the virtual z=0 row followed by NEMO's W rows.
    Unlike the shared Thomas solver, ``zdftke.F90:547-565`` first eliminates
    all diagonal coefficients, then eliminates the RHS in a separate loop,
    seeds the solution at ``jpkm1`` (``zdftke.f90:468``,
    ``en(jpkm1)=zd_lw(jpkm1)/zdiag(jpkm1)`` — i.e. WITHOUT the
    ``zd_up(jpkm1)*en(jpk)`` term, which is how NEMO's held ``jpk`` row leaves
    the recurrence), and only then reverse-substitutes. The final floor and W
    mask are part of the same source-ordered operation.

    The supplied array runs from the z=0 row to NEMO's ``jpkm1`` INCLUSIVE and
    carries NO ``jpk`` row: legoESM holds ``n_levels-1`` interior W-interfaces
    (``z_half_ref[1:-1]``) plus the one prepended surface row, so its last
    index IS ``jpkm1``, the deepest row NEMO SOLVES. ``en(jpk)`` is read by
    nothing — the back-substitution above drops it and ``tke_avn`` loops
    ``jk = 1, jpkm1`` (``zdftke.f90:681-687``) — so no slot is needed for it.
    """
    if not (a.shape == b.shape == c.shape == rhs.shape):
        raise ValueError("literal TKE tridiagonal operands must share a shape")
    if a.ndim < 1 or a.shape[-1] < 2:
        raise ValueError("literal TKE solve requires a surface and W row")
    expected_mask_shape = a.shape[:-1] + (a.shape[-1] - 1,)
    if w_active.shape != expected_mask_shape:
        raise ValueError(
            "w_active must match the non-surface TKE rows; got "
            f"{w_active.shape} vs {expected_mask_shape}")

    # The array is NEMO's jk = 1..jpkm1, so its LAST index is Fortran jpkm1
    # and every row of it is solved.  (It was ``- 2`` until 2026-09-11, which
    # treated the deepest carried row as the held Fortran ``jpk`` row and
    # returned it unsolved, i.e. as its raw right-hand side -- the right-hand
    # side that carries zdftke.f90:422-425's EXPLICIT half of the dissipation
    # split with no zdftke.f90:419 diagonal against it.  See
    # docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_tke_runaway_receipt.md)
    jpkm1 = a.shape[-1] - 1
    diag_seed = 1.0 / jnp.asarray(surface_en, dtype=b.dtype)
    work_seed = jnp.ones_like(diag_seed)

    if jpkm1:
        diag_inputs = tuple(jnp.moveaxis(x, -1, 0) for x in (
            a[..., 1:jpkm1 + 1], b[..., 1:jpkm1 + 1], c[..., :jpkm1]))

        def diagonal_step(previous, operands):
            lower, diagonal, previous_upper = operands
            current = diagonal - lower * previous_upper / previous
            return current, current

        _, diagonal_rows = jax.lax.scan(
            diagonal_step, diag_seed, diag_inputs)
        diagonal = jnp.concatenate(
            [diag_seed[..., None], jnp.moveaxis(diagonal_rows, 0, -1)],
            axis=-1)

        rhs_inputs = tuple(jnp.moveaxis(x, -1, 0) for x in (
            rhs[..., 1:jpkm1 + 1], a[..., 1:jpkm1 + 1],
            diagonal[..., :jpkm1]))

        def rhs_step(previous, operands):
            source, lower, previous_diagonal = operands
            # Preserve NEMO's division-before-multiply association (:556).
            current = source - lower / previous_diagonal * previous
            return current, current

        _, work_rows = jax.lax.scan(rhs_step, work_seed, rhs_inputs)
        work = jnp.concatenate(
            [work_seed[..., None], jnp.moveaxis(work_rows, 0, -1)], axis=-1)

        terminal = work[..., jpkm1] / diagonal[..., jpkm1]
        if jpkm1 > 1:
            reverse_inputs = tuple(
                jnp.moveaxis(x[..., 1:jpkm1][..., ::-1], -1, 0)
                for x in (work, c, diagonal))

            def reverse_step(next_value, operands):
                source, upper, current_diagonal = operands
                current = (source - upper * next_value) / current_diagonal
                return current, current

            _, reverse_rows = jax.lax.scan(
                reverse_step, terminal, reverse_inputs)
            solved_prefix = jnp.concatenate(
                [jnp.moveaxis(reverse_rows, 0, -1)[..., ::-1],
                 terminal[..., None]], axis=-1)
        else:
            solved_prefix = terminal[..., None]
    else:
        solved_prefix = jnp.zeros(a.shape[:-1] + (0,), dtype=rhs.dtype)

    # Every supplied row is solved: the array ends at NEMO's jpkm1 and the
    # held jpk row is not carried, so there is no uneliminated tail here.
    # ``jpkm1`` indexes the LAST row by construction, and a shorter slice
    # would silently drop a solved row rather than raise, so assert it.
    if jpkm1 + 1 != a.shape[-1]:                       # pragma: no cover
        raise AssertionError(
            "literal TKE solve left rows beyond jpkm1 unsolved; the array "
            "must run from the z=0 row to NEMO's jpkm1 inclusive")
    return (jnp.maximum(solved_prefix,
                        jnp.asarray(floor, dtype=solved_prefix.dtype))
            * jnp.asarray(w_active, dtype=solved_prefix.dtype))


def _solve_tke_backward_euler(
    e_old: jnp.ndarray,
    K_M_old: jnp.ndarray,
    K_H_old: jnp.ndarray,
    P_s: jnp.ndarray,
    N2: jnp.ndarray,
    l_eps: jnp.ndarray,
    dz_half: jnp.ndarray,
    surface_flux: jnp.ndarray,
    dt: float,
    cfg: TKEConfig,
    external_source: jnp.ndarray | None = None,
    literal_external_rhs: jnp.ndarray | None = None,
    dz_cell: jnp.ndarray | None = None,
    dz_surface: jnp.ndarray | None = None,
    dz_face_surface: jnp.ndarray | None = None,
    surface_dirichlet: jnp.ndarray | None = None,
    surface_bc_level: str = "interior_pinned",
    bottom_dirichlet: jnp.ndarray | None = None,
    K_M_surface: jnp.ndarray | None = None,
    bottom_level: jnp.ndarray | None = None,
    w_active: jnp.ndarray | None = None,
    nemo_e3t: jnp.ndarray | None = None,
    dissl_old: jnp.ndarray | None = None,
    return_statement_trace: bool = False,
    rhs_materialization: str = "",
    rhs_intermediate: str = "",
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward-Euler tridiagonal solve for one TKE time step.

    Linearises ``-c_eps * e^{3/2} / l_eps`` as ``-c_eps * sqrt(e_old) / l_eps · e_new``
    so the resulting system is linear in e_new. ``P_b = -K_H * N^2`` is
    treated explicitly with sign-aware splitting: when stable
    (``N2 > 0``), ``-K_H · N^2`` is a sink and is linearised in e_new
    via ``K_H = c_k · l_k · sqrt(2 e_old)`` so the implicit step does
    not overshoot to negative TKE.

    Parameters
    ----------
    e_old : (..., nlev-1)
    K_M_old, K_H_old : (..., nlev-1) — used in the linearised P_b sink term.
    P_s : (..., nlev-1) — shear production (explicit source).
    N2 : (..., nlev-1) — squared buoyancy frequency.
    l_eps : (..., nlev-1) — dissipation length.
    dz_half : (..., nlev-1) — cell-centre spacing.
    surface_flux : (...) — flux boundary condition at top interface
        ``forc_tke_surface = (|tau|/rho_0)^{3/2}``.
    dt : float
    cfg : TKEConfig
    external_source : (..., nlev-1) or None — additive energy-recycling source
        ``forc`` [m²/s³] at the interior interfaces (Veros integrate_tke
        ``forc = ... + eke_diss_iw + K_diss_bot``). Enters the RHS explicitly
        (already-dissipated mechanical energy, ≥ 0). ``None`` ⇒ no source ⇒
        BIT-IDENTICAL to the prior form.
    literal_external_rhs : (..., nlev-1) or None
        Pre-updated ``en`` after NEMO's literal Langmuir statement. This is
        accepted only with the literal matrix and a matching diagnostic
        ``external_source``; it preserves line 463's multiply/divide order.
    dz_cell : (..., nlev) or None — ACTUAL cell thicknesses ``dzt·J``. When
        given (with ``dz_surface``; ``TKEConfig.veros_dz_slots=True``) the
        tridiagonal assembly uses the Veros metric slots
        (veros/core/tke.py:185-225), top-down:

        - face gradient between interfaces k and k+1 over the INTERVENING
          CELL thickness ``dzt[k+1]`` (Veros ``delta = dt/dzt[k+1]·α/2·
          (κ[k]+κ[k+1])``, tke.py:193) — NOT the centre spacing ``dz_half``;
        - per-interface control volume = ``dz_half[k]`` directly (Veros
          ``dzw``, the W-cell volume, tke.py:199-222) — NOT the legacy
          average of adjacent face spacings (``dz_int_eff``).
    dz_surface : (...) or None — the surface W control volume
        ``0.5·dzw_top`` (Veros tke.py:225) = the distance from z=0 down to
        the top cell centre, ``-z_full_ref[0]·J``. Used as the surface-flux
        injection denominator (legoESM collapses Veros's surface W point
        onto the topmost interior interface; the injected ENERGY then
        matches Veros's surface deposit). Must be passed together with
        ``dz_cell``; both ``None`` (default) ⇒ BIT-IDENTICAL legacy
        (face gradients over ``dz_half``, avg-of-faces volumes, injection
        over ``dz_half[0]``).
    surface_bc_level : "interior_pinned" (default, BIT-IDENTICAL) or
        "nemo_z0" (``TKEConfig.tke_surface_bc_level``). NEMO holds ``en(1)``
        at the z=0 W-POINT (zdftke.F90:264-269) and SOLVES the tridiagonal
        from ``jk=2`` (zdftke.F90:403: ``DO_2Dik(...,2,jpkm1,1)``) — the
        first INTERIOR w-level (legoESM's interior interface 0). The
        "interior_pinned" mode instead pins interface 0 ITSELF Dirichlet
        (one w-level too deep — the #1 T3 suspect from the Phase-1 audit:
        the whole en/avt profile displaced one cell down). "nemo_z0" fixes
        this by prepending a VIRTUAL surface row (z=0, Dirichlet =
        ``surface_dirichlet``) to the tridiagonal system so interior
        interface 0 becomes a genuinely SOLVED row coupled to that virtual
        surface value — exactly NEMO's ``jk=2`` row (zdftke.F90:403-410):
        ``zzd_lw(jk=2) = -0.5·dt·max(avm(2)+avm(1),2e-5) / (e3t(1)·e3w(2))``.
        The ``(N+1)``-row system is solved once and only the interior
        ``N`` rows are returned (``e_new[..., 1:]``) — the state pytree
        shape is unchanged. Requires ``surface_dirichlet`` (raises
        otherwise: "nemo_z0" has no meaning without a held surface value)
        and ``dz_face_surface`` (NEMO's ``e3t(1)``, the LIVE top-cell
        thickness ``dz_ref[0]·J`` — the z=0-W-point-to-first-interior-
        W-point distance, i.e. the ``e3t(1)`` factor in the quoted
        ``zzd_lw(jk=2)`` denominator).  This is NOT ``dz_surface``
        (``-z_full_ref[0]·J`` = the top cell's MIDPOINT depth, half of
        ``e3t(1)`` on a midpoint grid); feeding the midpoint metric here
        DOUBLES the virtual-surface coupling (#1690).  The surface face is
        ``0.5*(avm(jk=2)+avm(jk=1))`` (zdftke.F90:407-410): ``avm(jk=2)`` is
        ``K_M_old[..., 0]`` (the topmost carried interface's own carried
        diffusivity — exact, same quantity NEMO's ``p_avm(jk=2)`` is) and
        ``avm(jk=1)`` is the TRUE surface-w-level viscosity, passed as
        ``K_M_surface`` (:func:`nemo_surface_avm` — T3-exact). When
        ``K_M_surface`` is None (bit-identical legacy / no ``tke_mxl_choice=3``
        anchor available), the face falls back to the documented
        APPROXIMATION ``avm(1) ~= avm(2) = K_M_old[..., 0]`` (i.e.
        ``avm(1)+avm(1)`` rather than the true ``avm(2)+avm(1)`` — exact only
        where the surface and first-interior viscosities coincide).
    dz_face_surface : (...) or None — NEMO's ``e3t(1)``, the live top-cell
        thickness ``dz_ref[0]·J``.  Required by (and used only by)
        ``surface_bc_level="nemo_z0"`` as the virtual-surface-row face
        distance; must be None otherwise.  Distinct from ``dz_surface``,
        which stays the Veros ``0.5·dzw_top`` injection volume.
    K_M_surface : (...,) or None — NEMO's true surface-w-level viscosity
        ``avm(jk=1)`` (:func:`nemo_surface_avm`), consulted ONLY by the
        ``surface_bc_level="nemo_z0"`` face-coefficient assembly above. None
        ⇒ the documented approximation (BIT-IDENTICAL to the prior
        behaviour); ignored (and must be None) when ``surface_bc_level !=
        "nemo_z0"`` (silent-no-op guard, matching ``bottom_dirichlet``).
    bottom_dirichlet : (...,) or None — NEMO bottom TKE BC (T15,
        zdftke.F90:279-288): ``en(mbkt+1) = max(0.001875·CdU_bot·|u_bot|,
        rn_emin)``. Held Dirichlet at the interior interface the per-column
        seafloor sits at. ``bottom_level`` is None (default, BIT-IDENTICAL):
        the pin lands at the array's LAST row unconditionally — exact only
        on a FLAT-BOTTOM column (every DINO column reaches max depth) and
        one level too deep on a SHALLOWER column (a masked/dry level there;
        downstream wet-interface masking prevents any leak into wet cells,
        so this was never a correctness bug, just a BC that didn't fire at
        the physically correct row). ``bottom_level`` supplied (T15-exact):
        the Dirichlet row is scattered to interior interface
        ``bottom_level`` PER COLUMN (NEMO's ``mbkt+1`` — interface
        ``bottom_level`` sits between T-cells ``bottom_level`` and
        ``bottom_level+1`` in legoESM's 0-based indexing, exactly ``mbkt``'s
        seafloor cell), clamped to ``[0, N-1]`` (N = number of carried
        interfaces) so a dry/degenerate column cannot index out of bounds.
        ``None`` (default) ⇒ BIT-IDENTICAL (the ``[..., -1]`` pin above).

    w_active : (..., nlev-1) or None — NEMO's ``wmask`` at the interior
        w-interfaces (1 = wet, 0 = below the seafloor), selecting
        ``TKEConfig.tke_dry_wmask``. NEMO closes ``tke_tke`` with
        ``en = MAX( en, rn_emin ) * wmask`` (DINO
        ``cfgs/DINO/MY_SRC/zdftke.F90:565`` = upstream
        ``src/OCE/ZDF/zdftke.F90:469``); legoESM transcribed the ``MAX`` and
        DROPPED the ``* wmask``, so the solve re-inflates every dry
        sub-seafloor row to ``tke_background`` instead of leaving it at 0.
        That matters because ``tke_avn``'s buoyancy length
        (``:759`` / ``:651``) carries NO wmask — with ``en == 0`` the dry rows
        are EXACTLY ``rmxl_min``, which is what makes the ``nn_mxl=3`` ldown
        sweep (``:799-812`` / ``:691-704``), running THROUGH them, deliver
        ``ldn(mbkt) = MIN(rmxl_min + e3t(mbkt+1,Kmm), l_int(mbkt))`` at each
        column's own seafloor. ``None`` (default) ⇒ BIT-IDENTICAL legacy.

    Returns
    -------
    e_new : (..., nlev-1)
    """
    if dz_cell is not None and dz_surface is None:
        raise ValueError(
            "_solve_tke_backward_euler: dz_cell requires dz_surface too "
            "(TKEConfig.veros_dz_slots); got dz_cell=set, dz_surface=None."
        )
    # dz_face_surface (NEMO's e3t(1)) is the nemo_z0 virtual-surface face
    # distance and is INDEPENDENT of the Veros metric slots: it may be
    # passed with dz_cell/dz_surface both None. veros_slots (the metric-
    # slot feature) is keyed off dz_cell alone, unaffected.
    if dz_face_surface is not None and surface_bc_level != "nemo_z0":
        # Silent-no-op guard, matching K_M_surface: the e3t(1) face slot is
        # read ONLY by the nemo_z0 assembly, so accepting it under any other
        # surface_bc_level would quietly discard a caller's metric.
        raise ValueError(
            "_solve_tke_backward_euler: dz_face_surface is consumed only by "
            "surface_bc_level='nemo_z0'; got "
            f"surface_bc_level={surface_bc_level!r}.")
    veros_slots = dz_cell is not None
    N = e_old.shape[-1]   # number of interfaces
    matrix_evaluation = getattr(cfg, "tke_matrix_evaluation", "factored")
    if matrix_evaluation not in ("factored", "nemo_literal"):
        raise ValueError(
            "Unknown TKEConfig.tke_matrix_evaluation: expected 'factored' "
            f"or 'nemo_literal', got {matrix_evaluation!r}.")
    literal_matrix = matrix_evaluation == "nemo_literal"
    solver_evaluation = getattr(cfg, "tke_solver_evaluation", "shared_thomas")
    if solver_evaluation not in ("shared_thomas", "nemo_literal"):
        raise ValueError(
            "Unknown TKEConfig.tke_solver_evaluation: expected "
            f"'shared_thomas' or 'nemo_literal', got {solver_evaluation!r}.")
    literal_solver = solver_evaluation == "nemo_literal"
    if literal_solver and not literal_matrix:
        raise ValueError(
            "tke_solver_evaluation='nemo_literal' requires "
            "tke_matrix_evaluation='nemo_literal'.")
    if return_statement_trace and (
            not literal_solver or surface_bc_level != "nemo_z0"):
        raise ValueError(
            "return_statement_trace requires the literal NEMO solver and "
            "surface_bc_level='nemo_z0' so rhs levels 1:jpkm1 exist.")
    if return_statement_trace and N < 2:
        # The traced zd_up/zd_lw/zdiag only exist on the literal assembly
        # branch, which is itself guarded by N >= 2.  Refuse loudly rather
        # than fall through to the generic assembly and raise NameError.
        raise ValueError(
            "return_statement_trace requires at least two interfaces so the "
            f"literal NEMO matrix assembly runs; got N={N}.")
    if rhs_materialization and rhs_intermediate:
        raise ValueError(
            "TKE RHS materialization and intermediate selectors are mutually "
            "exclusive private measurement arms")
    # ``LatLonCGridOceanModel.step`` pairs the private traced production
    # closure with an ordinary state-returning reference closure.  The latter
    # deliberately discards diagnostics, but it must still evaluate the same
    # selected RHS association so the returned prognostic state belongs to
    # the measured arm.  Keep the selected value private and simply discard
    # it below when ``return_statement_trace`` is false.
    rhs_intermediate_value = None
    if literal_matrix:
        if nemo_e3t is None or dissl_old is None or w_active is None:
            raise ValueError(
                "tke_matrix_evaluation='nemo_literal' requires nemo_e3t, "
                "dissl_old, and w_active operands.")
        expected_e3t_shape = e_old.shape[:-1] + (N + 1,)
        if nemo_e3t.shape != expected_e3t_shape:
            raise ValueError(
                "nemo_e3t must contain the full live T-cell ladder; got "
                f"{nemo_e3t.shape} vs {expected_e3t_shape}.")
        if dissl_old.shape != e_old.shape:
            raise ValueError(
                "dissl_old must match e_old shape; got "
                f"{dissl_old.shape} vs {e_old.shape}.")
        # The literal assembly reads N rows of e3w and of the W mask -- one
        # more than it used to, now that NEMO's jpkm1 row is built.  A SHORT
        # operand does not raise in JAX, it broadcasts or truncates, so the
        # deepest row would silently take the wrong metric (diff-review
        # finding, 2026-09-11).
        if dz_half.shape[-1] < N:
            raise ValueError(
                "the literal TKE matrix needs one e3w row per solved W row "
                f"(NEMO jk = 2..jpkm1); got dz_half with {dz_half.shape[-1]} "
                f"rows for {N} W rows.")
        if w_active.shape != e_old.shape:
            raise ValueError(
                "w_active must match e_old shape; got "
                f"{w_active.shape} vs {e_old.shape}.")
        if surface_bc_level != "nemo_z0" or K_M_surface is None:
            raise ValueError(
                "tke_matrix_evaluation='nemo_literal' requires the NEMO "
                "z=0 row and carried surface avm.")
        # Dispatch hardening (claim-review finding, 2026-09-11): the literal
        # diagonal is built WITHOUT ``buoy_sink_rate``, because zdftke.f90:419
        # has no stratification term on zdiag -- the whole `- p_avt*rn2` is
        # explicit on the RHS (:422-425).  So an implicit-linearised buoyancy
        # selection would have its sink silently DELETED rather than moved:
        # ``buoy_sink_rate`` would be computed and then never read.  Every
        # shipped literal card already selects 'nemo_explicit'
        # (nemo_testcase_recipe.py:129; experiments/dino.py:1234), so this
        # raises on a combination nothing selects instead of running it wrong.
        if getattr(cfg, "tke_buoyancy_sink",
                   "implicit_linearized") != "nemo_explicit":
            raise ValueError(
                "tke_matrix_evaluation='nemo_literal' requires "
                "tke_buoyancy_sink='nemo_explicit' (zdftke.f90:419 carries no "
                "stratification term on the diagonal; any implicit split "
                "would be silently dropped by the literal matrix).")
    # Dtype hygiene: surface_flux / external_source can promote to f64 (tau or
    # the EKE-diss source built at default precision) while e_old runs at the
    # storage policy's f32 — cast them down so the tridiagonal RHS scatter does
    # not raise the JAX implicit-downcast FutureWarning. No numeric change when
    # dtypes already match (the default path).
    surface_flux = surface_flux.astype(e_old.dtype)
    if external_source is not None:
        external_source = external_source.astype(e_old.dtype)
    if literal_external_rhs is not None:
        if not literal_matrix or external_source is None:
            raise ValueError(
                "literal_external_rhs requires the literal TKE matrix and "
                "a diagnostic external_source")
        if literal_external_rhs.shape != e_old.shape:
            raise ValueError(
                "literal_external_rhs must match e_old shape; got "
                f"{literal_external_rhs.shape} vs {e_old.shape}.")
        literal_external_rhs = literal_external_rhs.astype(e_old.dtype)
    positivity = getattr(cfg, "positivity", "floor")
    if positivity not in ("floor", "veros_surface_correction"):
        raise ValueError(
            f"Unknown TKEConfig.positivity={positivity!r}; expected 'floor' "
            f"or 'veros_surface_correction'."
        )
    veros_positivity = positivity == "veros_surface_correction"
    if w_active is not None and veros_positivity:
        # Dispatch hardening: the Veros positivity branch RETURNS before the
        # `MAX(en, rn_emin)` floor this mask rides on, so accepting the mask
        # here would be a silent no-op. Veros has no wmask'd `en` anyway.
        raise ValueError(
            "_solve_tke_backward_euler: w_active (TKEConfig.tke_dry_wmask) "
            "transcribes NEMO's `en = MAX(en,rn_emin)*wmask` "
            "(MY_SRC/zdftke.F90:565) and requires TKEConfig.positivity="
            "'floor'; got positivity='veros_surface_correction' (the Veros "
            "branch has no such floor, so the mask would silently no-op).")
    if literal_solver:
        if surface_bc_level != "nemo_z0" or w_active is None:
            raise ValueError(
                "tke_solver_evaluation='nemo_literal' requires the NEMO "
                "z=0 row and w_active.")
        if veros_positivity:
            raise ValueError(
                "tke_solver_evaluation='nemo_literal' requires "
                "positivity='floor'.")
    if veros_positivity:
        # Veros linearisation point: sqrttke = sqrt(max(0, e)) (tke.py:30)
        # — zero where the carried TKE is negative (energy debt), so the
        # dissipation shuts off there exactly as in Veros.  Double-``where``
        # for an AD-safe sqrt at the debt branch (primal BIT-IDENTICAL;
        # ``sqrt(max(0,e))`` itself has a NaN derivative at e <= 0 — see
        # the note in ``_veros_buoyancy_length``).
        e_sqrt = jnp.where(
            e_old > 0.0, jnp.sqrt(jnp.where(e_old > 0.0, e_old, 1.0)), 0.0)
    else:
        e_sqrt = jnp.sqrt(jnp.maximum(e_old, cfg.tke_background))
    # Linearised dissipation rate (per unit e_new).  NEMO's literal matrix
    # consumes the SAVE'd preceding-step ``dissl`` and only overwrites it in
    # post-solve tke_avn (zdftke.F90:499-510,832-837).
    if literal_matrix:
        diss_rate = cfg.c_eps * jnp.asarray(dissl_old, dtype=e_old.dtype)
    else:
        diss_rate = cfg.c_eps * e_sqrt / jnp.maximum(
            l_eps, _mixing_length_floor(cfg))
    _buoy_disc = getattr(cfg, "tke_buoyancy_sink", "implicit_linearized")
    if _buoy_disc not in ("implicit_linearized", "nemo_explicit"):
        raise ValueError(
            "Unknown TKEConfig.tke_buoyancy_sink: must be one of "
            f"('implicit_linearized', 'nemo_explicit'), got {_buoy_disc!r}.")
    if veros_positivity:
        # ---- Veros buoyancy treatment (positivity="veros_surface_correction")
        # P_b = -K_H·N² enters the RHS fully EXPLICITLY, both signs (Veros
        # forc = K_diss_v − P_diss_v, integrate_tke kernel: P_diss_v =
        # kappaH·N² is computed explicitly outside the tridiagonal). The
        # stratification sink may legitimately drive the solved TKE
        # NEGATIVE — the Veros energy debt; no implicit sink on the
        # diagonal.
        buoy_sink_rate = jnp.zeros_like(e_old)
        buoy_source = -K_H_old * N2       # signed; full explicit P_b
    elif _buoy_disc == "nemo_explicit":
        # ---- NEMO explicit buoyancy sink (zdftke.F90:417-418) ----
        # en(jk) += dt*(sh2 - avt(jk)*rn2(jk) + ...); avt is the PREVIOUS-
        # step (before-solve) diffusivity p_avt — exactly legoESM's carried
        # ``K_H_old`` — and rn2 may be either sign (stable OR unstable): the
        # WHOLE term enters the RHS explicitly, no implicit sink on the
        # diagonal. NEMO relies on the POST-solve floor `en=MAX(en,rn_emin)`
        # (zdftke.F90:468-470, already applied below by this function's
        # tail) to catch any negative overshoot from a large stable sink —
        # it does NOT split by sign the way the default linearised form
        # does. Sign check: N2>0 (stable) -> -K_H*N2<0, a genuine TKE SINK;
        # N2<0 (unstable) -> -K_H*N2>0, a genuine TKE SOURCE (convection)
        # -- matches the "P_b = -K_H*N^2" sign convention documented on
        # TKEConfig/the module docstring. NOTE: under n2_mode="insitu" N2 is
        # clipped >= 0 (documented pre-existing "insitu" limitation --
        # convection never fires through the TKE, orthogonal to this
        # branch), so the SOURCE half is inert there -- exactly as it
        # already is for the default "implicit_linearized" split below
        # (its N2_neg term is identically 0 too); nemo_explicit does not
        # introduce a NEW gap relative to insitu's existing behaviour.
        buoy_sink_rate = jnp.zeros_like(e_old)
        buoy_source = -K_H_old * N2       # signed; full explicit P_b
    else:
        # Buoyancy work ``P_b = -K_H · N^2`` split sign-aware so the implicit
        # diagonal stays >= 1 even when N^2 < 0 (statically unstable / signed
        # adiabatic mode):
        #   - N^2 >= 0 (stable): P_b is a SINK, linearised implicitly per unit
        #     e_new via the rate ``K_H · N^2 / e``.
        #   - N^2 <  0 (unstable): P_b is a SOURCE (convective PE -> TKE);
        #     added EXPLICITLY to the RHS (``-K_H · N^2 > 0``) so it does not
        #     drive the diagonal non-positive.
        # For the default in-situ N^2 (clipped >= 0) ``N2_neg`` is identically
        # 0, so this is BIT-IDENTICAL to the prior single-sink form.
        e_safe = jnp.maximum(e_old, cfg.tke_background)
        N2_pos = jnp.maximum(N2, 0.0)
        N2_neg = jnp.minimum(N2, 0.0)
        buoy_sink_rate = K_H_old * N2_pos / e_safe
        buoy_source = -K_H_old * N2_neg   # >= 0; explicit TKE production

    # Diffusion coefficients on the *flux faces* between interface k and
    # interface k+1 (one less than the number of interfaces).
    K_tke_face = cfg.alpha_tke * 0.5 * (K_M_old[..., :-1] + K_M_old[..., 1:])
    # Face-gradient metric: legacy uses the centre spacing dz_half; the
    # Veros slot is the INTERVENING CELL thickness dzt[k+1] (tke.py:193).
    if veros_slots:
        dz_face = jnp.maximum(dz_cell[..., 1:N], _EPS)
    else:
        dz_face = jnp.maximum(dz_half[..., :N - 1], _EPS)
    # Tridiagonal coefficients. Interior interfaces k = 1 ... N-2.
    a_diff = jnp.zeros_like(e_old)
    b_diff = jnp.zeros_like(e_old)
    c_diff = jnp.zeros_like(e_old)

    literal_lw = None
    if literal_matrix and N >= 2:
        # Literal transcription of DINO MY_SRC/zdftke.F90:499-510.  The
        # source carries signed zzd_up/zzd_lw (both <= 0), places e3t(jk,Kmm)
        # in the upper denominator and e3t(jk-1,Kmm) in the lower, and forms
        # zdiag in this exact source association before the Thomas solve.
        # Rows 0..N-1 are NEMO jk = 2..jpkm1, i.e. EVERY row zdftke solves
        # (`DO jk = 2, jpkm1`, zdftke.f90:407).  The deepest of them was
        # omitted until 2026-09-11.
        zcof = (-0.5 * dt) * jnp.asarray(
            w_active[..., :N], dtype=e_old.dtype)
        avm_min = jnp.asarray(2.0e-5, dtype=e_old.dtype)  # coeff-ok: zdftke:503,505
        e3w_rows = jnp.asarray(dz_half[..., :N], dtype=e_old.dtype)
        e3t = jnp.asarray(nemo_e3t, dtype=e_old.dtype)
        # zzd_up at the deepest row needs p_avm(jk+1) = p_avm(jpk), which NEMO
        # never writes: zdfphy.f90:226-228 sets avm_k(:,:,jk)=avmb(jk)*wmask
        # and wmask(:,:,jpk)=0, and tke_avn only loops jk=1,jpkm1
        # (zdftke.f90:681-687).  So the upper neighbour there is exactly zero.
        upper_neighbour = jnp.concatenate(
            [K_M_old[..., 1:N], jnp.zeros_like(K_M_old[..., :1])], axis=-1)
        upper_sum = jnp.maximum(
            upper_neighbour + K_M_old[..., :N], avm_min)
        lower_neighbour = jnp.concatenate(
            [jnp.asarray(K_M_surface, dtype=e_old.dtype)[..., None],
             K_M_old[..., :N - 1]], axis=-1)
        lower_sum = jnp.maximum(
            K_M_old[..., :N] + lower_neighbour, avm_min)
        literal_up = (zcof * upper_sum
                      / (e3t[..., 1:N + 1] * e3w_rows))
        literal_lw = (zcof * lower_sum
                      / (e3t[..., :N] * e3w_rows))
        a_diff = jnp.concatenate(
            [jnp.zeros_like(literal_lw[..., :1]),
             literal_lw[..., 1:]], axis=-1)
        # zd_up(jpkm1) enters zdiag(jpkm1) (zdftke.f90:419) but NOT the
        # back-substitution: zdftke.f90:468 seeds en(jpkm1) without the
        # zd_up(jpkm1)*en(jpk) term.  Hence the trailing zero HERE and the
        # full -literal_up in the diagonal BELOW.
        c_diff = jnp.concatenate(
            [literal_up[..., :N - 1],
             jnp.zeros_like(literal_up[..., :1])], axis=-1)
        b_diff = jnp.zeros_like(e_old)
    elif veros_slots and N >= 2:
        # ---- Veros-faithful assembly (tke.py:199-222), top-down ----
        # Control volume of interface k = dz_half[k] (Veros dzw, the W-cell
        # volume). Flux through the face between interfaces k and k+1 is
        # delta[k] = dt·K_tke_face[k]/dz_face[k]; row k gets
        #   a[k] = -delta[k-1]/vol[k]   (coupling up;   k >= 1)
        #   c[k] = -delta[k]  /vol[k]   (coupling down; k <= N-2)
        #   b_diff = -(a+c)
        # The topmost interface (k=0) has no upward coupling: legoESM does
        # not carry Veros's surface W point — zero-flux top + the explicit
        # surface injection below (see dz_surface in the docstring). The
        # deepest interface keeps only its upper-face term, matching
        # Veros's bottom edge row (b_tri_edge / a_tri[:, :, -1] structure
        # for a flat-bottom column).
        vol = jnp.maximum(dz_half[..., :N], _EPS)
        delta_face = dt * K_tke_face / dz_face          # (..., N-1)
        a_diff = jnp.concatenate(
            [jnp.zeros_like(delta_face[..., :1]),
             -delta_face / vol[..., 1:]], axis=-1,
        )
        c_diff = jnp.concatenate(
            [-delta_face / vol[..., :N - 1],
             jnp.zeros_like(delta_face[..., :1])], axis=-1,
        )
        b_diff = -(a_diff + c_diff)
    # Interior contributions to a/b/c from diffusion (legacy):
    elif N >= 3:
        # For interior interface k, the flux into face (k-1, k) uses
        # K_tke_face[..., k-1] / dz_face[..., k-1], and into face (k, k+1)
        # uses K_tke_face[..., k] / dz_face[..., k].
        coef_lower = K_tke_face / dz_face                # (..., N-1)
        # Build full-length per-interface aggregates:
        #   a[k] = -dt * coef_lower[k-1] / dz_int_eff[k]   (sub-diagonal)
        #   c[k] = -dt * coef_lower[k]   / dz_int_eff[k]   (super-diagonal)
        # Approximate cell-thickness around interface k as average of
        # adjacent face thicknesses.
        dz_int_eff = jnp.where(
            jnp.arange(N) == 0,
            dz_face[..., :1],
            jnp.concatenate([dz_face[..., :1], dz_face], axis=-1)[..., :N],
        )
        # Sub-diagonal at interior interfaces (skip k=0):
        a_inter = dt * coef_lower / jnp.maximum(dz_int_eff[..., 1:N], _EPS)
        a_diff = jnp.concatenate(
            [jnp.zeros_like(a_inter[..., :1]), -a_inter], axis=-1,
        )
        # Super-diagonal at interior interfaces (skip k=N-1):
        c_inter = dt * coef_lower / jnp.maximum(dz_int_eff[..., :N - 1], _EPS)
        c_diff = jnp.concatenate(
            [-c_inter, jnp.zeros_like(c_inter[..., :1])], axis=-1,
        )
        b_diff = -(a_diff + c_diff)

    # Total tridiagonal matrix entries:
    #   (1 + dt * (diss_mult*diss_rate + buoy_sink_rate)) * e_new
    #     + diffusion contribution = e_old + dt * P_s + flux BC
    # Dissipation time-discretization (TKEConfig.dissipation_discretization):
    #   "backward_euler" (default, BIT-IDENTICAL): fully-implicit, diss on the
    #     diagonal at the linearized rate diss_rate.
    #   "nemo_1p5_split": NEMO zdftke semi-implicit split (zdftke.F90:241-242,
    #     414,419): zfact2=1.5·rn_Dt·rn_ediss on the diagonal + zfact3=0.5·
    #     rn_ediss·dissl·en added back EXPLICITLY to the RHS, both linearized at
    #     the CARRIED sqrt(e)/l_eps. Same first-order dissipation; the discrete
    #     decay factor differs from plain backward-Euler at large dt·diss. The
    #     buoyancy sink keeps its own (implicit-split) treatment — only the
    #     Kolmogoroff dissipation is split, matching NEMO. Mirrors the identical
    #     split already used in tke_integrate_post_mixing (the Veros step order).
    _disc = getattr(cfg, "dissipation_discretization", "backward_euler")
    if literal_matrix:
        if _disc != "nemo_1p5_split":
            raise ValueError(
                "tke_matrix_evaluation='nemo_literal' requires "
                "dissipation_discretization='nemo_1p5_split'.")
        dissl = jnp.asarray(dissl_old, dtype=e_old.dtype)
        diag = (1.0 - literal_lw - literal_up
                # NEMO's literal zfact2 = 1.5 * rn_Dt * rn_ediss.
                + ((1.5 * dt) * cfg.c_eps)  # coeff-ok: NEMO zfact2 split weight
                * dissl[..., :N]
                * jnp.asarray(w_active[..., :N], dtype=e_old.dtype))
    elif _disc == "nemo_1p5_split":
        diag = 1.0 + dt * (1.5 * diss_rate + buoy_sink_rate) + b_diff  # coeff-ok: NEMO zdftke semi-implicit dissipation split weight (zfact2=1.5·rn_ediss, zdftke.F90:241)
    elif _disc == "backward_euler":
        diag = 1.0 + dt * (diss_rate + buoy_sink_rate) + b_diff
    else:
        raise ValueError(
            "Unknown TKEConfig.dissipation_discretization: must be one of "
            f"('backward_euler', 'nemo_1p5_split'), got {_disc!r}")

    # RHS: explicit shear-production source + explicit convective buoyancy
    # production (zero in the default in-situ mode) + previous-step e
    # + the external energy-recycling source ``forc`` (eke_diss_iw + K_diss_bot,
    # Veros integrate_tke; zero / None ⇒ bit-identical).
    if literal_matrix:
        # Literal zdftke.F90 RHS association; the private selector walks it.
        rhs_base = e_old
        if literal_external_rhs is not None:
            rhs_base = literal_external_rhs
        elif external_source is not None:
            rhs_base = rhs_base + dt * external_source
        if rhs_intermediate:
            rhs, rhs_intermediate_value = _nemo_literal_rhs_materialized(
                rhs_base, dt, P_s, K_H_old, N2, diss_rate, dissl_old,
                cfg.c_eps, w_active, "", intermediate=rhs_intermediate)
        elif rhs_materialization:
            rhs = _nemo_literal_rhs_materialized(
                rhs_base, dt, P_s, K_H_old, N2, diss_rate, dissl_old,
                cfg.c_eps, w_active, rhs_materialization)
        else:
            rhs = rhs_base + dt * (
                P_s + buoy_source + 0.5 * diss_rate * rhs_base
            ) * jnp.asarray(w_active, dtype=e_old.dtype)
    else:
        rhs = e_old + dt * (P_s + buoy_source)
    if _disc == "nemo_1p5_split" and not literal_matrix:
        # zfact3·dissl·en explicit add-back (NEMO zdftke.F90:419).
        rhs = rhs + dt * 0.5 * diss_rate * e_old
    if external_source is not None and not literal_matrix:
        rhs = rhs + dt * external_source

    if bottom_dirichlet is not None:
        # NEMO bottom TKE BC (zdftke.F90:279-288): en(mbkt+1) =
        # MAX(0.001875·CdU_bot·|u_bot|, rn_emin)*ssmask, set the SAME way as
        # the surface value (a plain identity row — NEMO solves the
        # tridiagonal only from jk=2 to jpkm1, so the bottom row jpk is
        # ALSO held, not coupled). Held AFTER the diffusion assembly so the
        # pinned row's super/sub-diagonal entries are zeroed too (no
        # leftover coupling into a row now pinned).
        _e_bd = jnp.asarray(bottom_dirichlet, dtype=e_old.dtype)
        if bottom_level is None:
            # BIT-IDENTICAL legacy: unconditional pin at the array's last
            # row (exact on a flat-bottom column; see the docstring).
            a_diff = a_diff.at[..., -1].set(0.0)
            diag = diag.at[..., -1].set(1.0)
            rhs = rhs.at[..., -1].set(_e_bd)
        else:
            # T15-exact: scatter the pin to the PER-COLUMN seafloor
            # interface (NEMO's mbkt), clamped into range so a dry column
            # cannot index out of bounds. take/put_along_axis over the last
            # (interface) axis — differentiable (pure gather/scatter, no
            # data-dependent control flow).
            N = e_old.shape[-1]
            idx = jnp.clip(
                jnp.asarray(bottom_level), 0, N - 1)[..., jnp.newaxis]
            is_bottom = jnp.arange(N) == idx    # (..., N) one-hot per column
            a_diff = jnp.where(is_bottom, 0.0, a_diff)
            c_diff = jnp.where(is_bottom, 0.0, c_diff)
            diag = jnp.where(is_bottom, 1.0, diag)
            rhs = jnp.where(is_bottom, _e_bd[..., jnp.newaxis], rhs)

    # Surface flux BC at interface k=0: add the flux divergence with
    # ``forc_tke_surface``-style energy input. Veros injects over the
    # surface W half-volume 0.5·dzw_top (tke.py:225) — the distance from
    # z=0 to the top cell centre — NOT the first interior centre spacing
    # dz_half[0] (the legacy denominator, kept bit-identical by default).
    if veros_slots:
        inj_vol = jnp.maximum(jnp.asarray(dz_surface, dtype=e_old.dtype), _EPS)
    else:
        inj_vol = jnp.maximum(dz_half[..., 0], _EPS)
    if surface_bc_level not in ("interior_pinned", "nemo_z0"):
        raise ValueError(
            "Unknown TKEConfig.tke_surface_bc_level: must be one of "
            f"('interior_pinned', 'nemo_z0'), got {surface_bc_level!r}.")
    if surface_bc_level == "nemo_z0" and surface_dirichlet is None:
        raise ValueError(
            "TKEConfig.tke_surface_bc_level='nemo_z0' requires "
            "surface_bc='nemo_dirichlet' (a held surface TKE value) — "
            "'nemo_z0' only changes WHERE the Dirichlet value is held, it "
            "does not supply one.")

    if surface_bc_level == "nemo_z0":
        # ---- NEMO z=0 surface row (zdftke.F90:264-269,403-410) ----
        # Prepend ONE virtual surface row (z=0, Dirichlet) so interior
        # interface 0 becomes a genuinely SOLVED row coupled to it — NEMO's
        # jk=2, not a pinned Dirichlet row. See the docstring above.
        if dz_face_surface is None and not literal_matrix:
            # literal_matrix forms zzd_lw(jk=2) from the nemo_e3t ladder
            # itself (row0_coupling = -literal_lw[..., 0] below), so the
            # analytic face slot is genuinely unused there -- demanding it
            # would be a false requirement (codex review, #1690).
            raise ValueError(
                "TKEConfig.tke_surface_bc_level='nemo_z0' requires "
                "dz_face_surface = NEMO's e3t(1) (the live top-cell "
                "thickness dz_ref[0]*J), the z=0-W-point-to-first-interior-"
                "W-point distance in the zzd_lw(jk=2) denominator. It is "
                "NOT dz_surface (-z_full_ref[0]*J), which is the top cell's "
                "MIDPOINT depth = half e3t(1) on a midpoint grid and would "
                "double the surface coupling (#1690).")
        e_sfc = jnp.asarray(surface_dirichlet, dtype=e_old.dtype)
        dz_face_sfc = (None if dz_face_surface is None else jnp.maximum(
            jnp.asarray(dz_face_surface, dtype=e_old.dtype), _EPS))
        # Row 0's OWN control volume: dz_half[...,0] in BOTH the legacy and
        # veros_slots assemblies (legacy: dz_int_eff[0] = dz_face[...,0] =
        # dz_half[...,0]; veros_slots: vol[0] = dz_half[...,0]) — the same
        # slot every other row's face-flux-over-volume ratio already uses.
        vol0 = jnp.maximum(dz_half[..., 0], _EPS)
        # Face flux coefficient between the virtual surface row and interior
        # interface 0: NEMO's true face is 0.5*(avm(jk=2)+avm(jk=1))
        # (zdftke.F90:407-410). avm(jk=2) is K_M_old[...,0] (the topmost
        # carried interface's own diffusivity — exact, the same quantity
        # NEMO's p_avm(jk=2) is). avm(jk=1) is the TRUE surface-w-level
        # viscosity: when the caller supplies K_M_surface (T3-exact,
        # :func:`nemo_surface_avm`) the face is bit-exact; None falls back
        # to the documented APPROXIMATION avm(1) ~= avm(2) = K_M_old[...,0]
        # (i.e. avm(1)+avm(1) rather than the true avm(2)+avm(1) —  exact
        # only where the surface and first-interior viscosities coincide).
        _avm1 = K_M_old[..., 0] if K_M_surface is None else jnp.asarray(
            K_M_surface, dtype=e_old.dtype)
        if literal_matrix:
            # zzd_lw(jk=2) was already formed above with the literal e3t(1)
            # and e3w(2) slots.  Its sign is the tridiagonal subdiagonal;
            # the positive coupling magnitude is only needed by the virtual
            # surface-row representation below.
            row0_coupling = -literal_lw[..., 0]
        else:
            K_face_sfc = cfg.alpha_tke * 0.5 * (K_M_old[..., 0] + _avm1)
            delta_sfc = dt * K_face_sfc / dz_face_sfc  # (...,) flux coeff.
            row0_coupling = delta_sfc / vol0           # a[row 0] magnitude
        if bottom_dirichlet is not None and bottom_level is not None:
            # Degenerate-column guard: when the T15 bottom Dirichlet pin
            # ALSO lands on interior interface 0 (a 1-interior-interface
            # column), row 0 must stay a PURE identity row (diag=1, no
            # coupling either direction) — the surface face coupling built
            # above would otherwise silently override the bottom pin's
            # diag=1/rhs=bottom_dirichlet with diag=1+row0_coupling and
            # couple it to the virtual surface row, corrupting the held
            # value. Zero the surface coupling on any column where row 0
            # is the bottom-pinned row (rare on real DINO bathymetry but a
            # genuine correctness edge case once bottom_level can be < the
            # array length).
            row0_coupling = jnp.where(is_bottom[..., 0], 0.0, row0_coupling)

        # Row 0 (interior interface 0, NEMO jk=2): add the upward coupling
        # to the virtual surface row (a_diff[...,0] was 0 — no upward
        # neighbour existed before this row was added). The DIAGONAL gets
        # the new face's magnitude (a genuinely new contribution — the
        # matrix previously had no upward coupling here at all); the RHS is
        # NOT separately incremented by ``row0_coupling*e_sfc`` — that
        # source enters through the tridiagonal system ITSELF via the
        # matrix coupling ``a_ext[row=1] = -row0_coupling`` against the
        # virtual row's identity value ``rhs_ext[row=0] = e_sfc`` (the
        # standard Thomas-elimination first step,
        # ``rhs[1] -= (a[1]/b[0])*rhs[0]``, already reproduces exactly this
        # term). Adding it here TOO double-counted the surface source by 2x
        # (physics-validator review, T3-exact gap-closure 2026-07-24) —
        # bug predates this fix (commit e1744a0530), caught once the
        # approximation-mode ``avm(1)~=avm(2)`` stopped floor-masking the
        # symptom (the exact avm(1) is large enough to push the solved
        # profile above ``tke_surface_min``/``tke_background``, where the
        # prior 2x error becomes visible instead of hidden by the floor).
        b0 = (diag[..., 0] if literal_matrix
              else diag[..., 0] + row0_coupling)

        # Assemble the (N+1)-row system: row 0 = virtual surface (identity,
        # Dirichlet: a=0, b=1, c=0); rows 1..N = the original N rows, with
        # row 1 (= original row 0)'s sub-diagonal now coupling UP into the
        # surface row.
        a_ext = jnp.concatenate(
            [jnp.zeros_like(a_diff[..., :1]),
             -row0_coupling[..., None], a_diff[..., 1:]], axis=-1)
        b_ext = jnp.concatenate(
            [jnp.ones_like(diag[..., :1]), b0[..., None], diag[..., 1:]],
            axis=-1)
        c_ext = jnp.concatenate(
            [jnp.zeros_like(c_diff[..., :1]), c_diff], axis=-1)
        rhs_ext = jnp.concatenate([e_sfc[..., None], rhs], axis=-1)

        if literal_solver:
            e_new = _nemo_literal_tke_solve(
                a_ext, b_ext, c_ext, rhs_ext, e_sfc, w_active,
                cfg.tke_background)
        else:
            e_new_ext = _tridiag_thomas(a_ext, b_ext, c_ext, rhs_ext)
            e_new = e_new_ext[..., 1:]
    elif surface_dirichlet is not None:
        # NEMO nn_bc_surf=1 Dirichlet surface TKE (zdftke.F90:264-269): hold
        # e_new[...,0] = e_sfc exactly by making row 0 an identity row (the
        # k=1 row's sub-diagonal still couples to the held value — the
        # standard Dirichlet-boundary tridiagonal). NO Neumann flux injection
        # in this mode (it would double-count the surface input).
        diag = diag.at[..., 0].set(1.0)
        c_diff = c_diff.at[..., 0].set(0.0)
        rhs = rhs.at[..., 0].set(
            jnp.asarray(surface_dirichlet, dtype=e_old.dtype))
        e_new = _tridiag_thomas(a_diff, diag, c_diff, rhs)
    else:
        rhs = rhs.at[..., 0].add(dt * surface_flux / inj_vol)
        e_new = _tridiag_thomas(a_diff, diag, c_diff, rhs)

    if literal_solver:
        if return_statement_trace:
            # Exact model program boundaries corresponding to NEMO's
            # post-Langmuir and r101_rhs_row calls. ``rhs_base`` is captured
            # from the selected production association (literal or
            # vectorized), not reconstructed by the diagnostic caller.
            # ``literal_up``/``literal_lw``/``diag`` are the model's
            # zd_up/zd_lw/zdiag at NEMO jk = 2..jpkm1 (zdftke.f90:434-436),
            # captured BEFORE the extended-system concatenation so the
            # deepest row keeps the value NEMO records rather than the
            # back-substitution's structural zero.
            return (e_new, rhs_base, rhs_ext,
                    literal_up, literal_lw, diag, rhs_intermediate_value)
        return e_new

    if veros_positivity:
        # Veros tke.py:238-245: interior TKE MAY GO NEGATIVE (the debt is
        # carried; sqrt(max(0,e)) downstream shuts the closure off there).
        # Only the SURFACE level is clamped at zero — Veros records the
        # clamped deficit as ``tke_surf_corr`` (fed back to the surface
        # heat budget, not modelled here); legoESM's collapsed surface
        # point is the topmost interior interface k=0. No
        # ``tke_background`` / ``tke_surface_min`` floors.
        e_new = e_new.at[..., 0].set(jnp.maximum(e_new[..., 0], 0.0))
        return e_new

    # Floor at background; clamp away from negative.
    # NEMO: `en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)`
    # (DINO cfgs/DINO/MY_SRC/zdftke.F90:565 = upstream
    # src/OCE/ZDF/zdftke.F90:469). legoESM historically kept the MAX and
    # DROPPED the `* wmask`; ``w_active`` (TKEConfig.tke_dry_wmask) restores
    # it. None ⇒ BIT-IDENTICAL legacy.
    e_new = jnp.maximum(e_new, cfg.tke_background)
    # The legoESM surface slot is interface 0 only for the historical
    # ``interior_pinned`` layout.  Under ``nemo_z0`` the true surface is the
    # separate virtual Dirichlet row assembled above: NEMO applies rn_emin0
    # there (MY_SRC/zdftke.F90:361), then applies only rn_emin to the solved
    # interior jk=2..jpkm1 (:564-565).  Reapplying tke_surface_min here would
    # incorrectly pin NEMO jk=2 to the surface floor.
    if surface_bc_level == "interior_pinned":
        e_new = e_new.at[..., 0].set(
            jnp.maximum(e_new[..., 0], cfg.tke_surface_min),
        )
    if w_active is not None:
        # LAST statement, exactly as at :565 (the `* wmask` closes tke_tke);
        # this removes either post-solve floor on dry interfaces. With masking
        # disabled, ``interior_pinned`` retains the historical unmasked
        # surface floor, while ``nemo_z0`` now retains only tke_background at
        # dry interface 0 instead of the formerly misplaced surface floor.
        # Both shipped nemo_z0 DINO cards enable tke_dry_wmask.
        e_new = e_new * jnp.asarray(w_active, dtype=e_new.dtype)
    return e_new


# ---------------------------------------------------------------------------
# K_M, K_H from TKE
# ---------------------------------------------------------------------------


def _prandtl_number(
    N2: jnp.ndarray,
    shear_sq: jnp.ndarray,
    kappaM: jnp.ndarray,
    cfg: TKEConfig,
    p_sh2_override: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Turbulent Prandtl number for the K_H = K_M / Pr relation.

    ``p_sh2_override``, when given (``tke_shear_avm_weighting="nemo_face"``),
    replaces the internally-formed ``p_sh2 = kappaM * shear_sq`` with the
    caller's avm-face-weighted ``p_sh2``
    (:func:`_shared.avm_weighted_shear_production`) — the #1455 sh2
    chain-walk avm-weighting fix. ``None`` (default) is BIT-IDENTICAL to the
    prior behaviour.

    Mirrors ``veros/core/tke.py:74-90``:

    - ``prandtl_mode="richardson"`` (Veros ``enable_Prandtl_tke=True``):
      the gradient Richardson number ``Ri = N^2 / max(shear^2, eps)``
      (Veros forms ``Ri = N^2 / max(K_diss_v / kappaM, eps)`` with
      ``K_diss_v / kappaM = shear^2``), then
      ``Pr = max(1, min(10, 6.6*Ri))``. In a convecting column
      ``N^2 < 0 -> Ri < 0 -> Pr = 1`` (K_H tracks the large convective
      K_M); in the stratified interior ``Pr -> 10`` (small abyssal K_H).
    - ``prandtl_mode="constant"``: ``Pr = Prandtl_tke0`` (Veros
      ``enable_Prandtl_tke=False`` fallback, default 10).
    - ``prandtl_mode="nemo_ri"`` (Phase-2 #1317 T8, fixed #1226 item 11;
      sign-condition transcription fixed #1226 zdftke_chain_walk STAGE 2):
      NEMO's EXACT nn_pdl=1 form (cfgs/DINO/MY_SRC/zdftke.F90:477-497
      — the copy DINO actually builds, ``IF(nn_pdl==1)`` at :477 through
      its ``ENDIF`` at :497 inclusive, verified by reading the file;
      upstream src/OCE/ZDF/zdftke.F90 is the same block at :381-401 —
      see the inline
      comment below for the full 3-way branch — ``rn2b<=0 -> zri=0``;
      ``zdiv==0`` exact-zero guard; else ``zri = rn2b·p_avm / zdiv`` taken
      AS-IS including its sign): ``pdlr = max(0.1, ri_cri/max(ri_cri,
      zri))``, ``Pr = 1/pdlr`` — i.e.
      ``Pr = max(1, min(10, (1/ri_cri)*zri))``, the SAME clamp/scaling as
      "richardson" but ``zri``'s denominator is the ``p_sh2`` AVM-WEIGHTED
      shear-production term [m²/s³] (``zdfsh2.F90:80-94``: face-averaged
      OLD ``avm`` times the Burchard now·before velocity-gradient product),
      NOT the plain ``shear_sq`` [1/s²] "richardson" divides by — a bulk/
      flux-Richardson-like ratio (eddy-viscosity-weighted stability), not
      the plain gradient Ri. ``p_avm`` in NEMO is the SAME pre-step
      (OLD/previous-timestep) viscosity that feeds ``zdf_sh2`` — here,
      ``kappaM`` (the caller's ``K_M_old``/prior-iteration K_M, matching
      the ``P_s = K_M_old·shear_sq`` shear-production term already computed
      at the call site). legoESM's single per-interface ``K_M`` (vs NEMO's
      separate u-/v-point avm face-averaged onto the T-point, zdfsh2.F90:
      80-94) is bridged EITHER by the ``p_sh2_override`` argument (the
      exact face-averaged transcription, ``_shared.
      avm_weighted_shear_production``, selected by
      ``tke_shear_avm_weighting="nemo_face"``) OR, when no override is
      given, by the T-collapsed ``p_sh2 ≈ kappaM·shear_sq`` (the
      AVM-WEIGHTED shear, matching units [m²/s³]) — correctly normalised
      since the 2026-08 ``vertical_shear_face_native`` prefactor fix, and
      the same documented simplification as :func:`legoesm.ocean.physics.
      vertical_mixing._shared.vertical_shear_burchard`. Either way,
      ``bshear_floor`` is added (now in the SAME m²/s³ units
      as NEMO's ``rn_bshear``, not ``shear_sq``'s 1/s²). Caller passes
      ``cfg.prandtl_ri_coeff = 1/ri_cri`` (unchanged meaning).

    Differentiable; the ``min``/``max`` clamps are sub-gradient-safe and
    the ``6.6`` / ``1`` / ``10`` are Veros's fixed scheme constants.
    """
    if cfg.prandtl_mode == "constant":
        return jnp.full_like(kappaM, cfg.Prandtl_tke0)
    if cfg.prandtl_mode == "richardson":
        Ri = N2 / jnp.maximum(shear_sq, 1e-12)
        return jnp.maximum(1.0, jnp.minimum(10.0, cfg.prandtl_ri_coeff * Ri))
    if cfg.prandtl_mode == "nemo_ri":
        bshear = jnp.asarray(getattr(cfg, "bshear_floor", 1.0e-20),
                             dtype=N2.dtype)
        # p_sh2 (avm-weighted shear production, [m^2/s^3]) — zdfsh2.F90:80-94
        # face-averages OLD avm onto the shear product. Default (p_sh2_
        # override=None, BIT-IDENTICAL): legoESM's single per-interface
        # K_M has no face-avg analog, so kappaM*shear_sq is the T-collapsed
        # approximation (== P_s_curr at the call site, tke.py:1932).
        # tke_shear_avm_weighting="nemo_face" (#1455): the caller supplies
        # the exact face-averaged p_sh2 instead (_shared.
        # avm_weighted_shear_production).
        p_sh2 = kappaM * shear_sq if p_sh2_override is None else p_sh2_override
        # NB with NEMO's default rn_bshear = 1e-20 the kappaM factors cancel
        # almost everywhere (bshear is ~9 decades below kappaM*shear_sq in any
        # realistic regime), so zri ~= N2/shear_sq: nemo_ri is then
        # NEAR-DEGENERATE with "richardson" up to the ri_cri-vs-6.6 scaling.
        # The weighted form matters only where the floor competes (kappaM or
        # shear ~ 0) — keep it for faithfulness, but do not expect materially
        # different production behaviour (review of 4aeeb867d, #1226).
        #
        # #1226 zdftke_chain_walk (STAGE 2 finding): faithful transcription
        # of the FULL nn_pdl==1 conditional, zdftke.F90:459-476 (DINO
        # MY_SRC copy):
        #   IF (rn2b <= 0)      THEN zri = 0
        #   ELSE
        #     zdiv = p_sh2 + rn_bshear
        #     IF (zdiv == 0)    THEN zri = rn2b*p_avm / rn_bshear
        #     ELSE                   zri = rn2b*p_avm / zdiv   (zdiv may be
        #                            NEGATIVE — p_sh2 can be a tiny negative
        #                            float-noise value, |p_sh2| > rn_bshear;
        #                            NEMO takes the division AS-IS, no
        #                            positivity clamp on zdiv)
        # The old code applied ``jnp.maximum(p_sh2 + bshear, 1e-30)``
        # unconditionally, which FLIPS a genuinely negative zdiv to a tiny
        # POSITIVE floor — turning a negative zri (-> pdlr=1.0, Pr=1) into a
        # huge positive one (-> pdlr=0.1, Pr=10): the opposite end of the
        # same clamp. Fix: only the exact-zero special case divides by
        # rn_bshear; a negative zdiv is used AS-IS (matching NEMO's sign).
        #
        # AD safety (JAX where-NaN-grad trap): jnp.where evaluates BOTH
        # branches, so a raw division by a possibly-zero zdiv in the
        # untaken branch can still produce inf/NaN and NaN gradients. Use
        # the double-where idiom: substitute a safe (nonzero) denominator
        # in the branch that will be masked out, then select the branch
        # with the ORIGINAL (correctly-signed) value on the taken side —
        # never let the safe substitute leak into the selected result.
        is_zero_zdiv = p_sh2 + bshear == 0.0
        safe_zdiv = jnp.where(is_zero_zdiv, 1.0, p_sh2 + bshear)  # avoid 1/0
        numerator = N2 * kappaM
        if cfg.tke_n2_evaluation_stage == "step_entry":
            # DINO/NEMO literal evaluation order, zdftke.F90:489-495.
            # Do not replace division with multiplication by a reciprocal:
            # the matched-state row-11 bar resolves that one-ulp difference.
            zri_stratified = jnp.where(
                is_zero_zdiv, numerator / bshear, numerator / safe_zdiv)
        else:
            # Historical association retained exactly for every card outside
            # the two complete DINO NEMO recipes.
            zri_stratified = numerator * jnp.where(
                is_zero_zdiv, 1.0 / bshear, 1.0 / safe_zdiv)
        zri = jnp.where(N2 > 0.0, zri_stratified, 0.0)
        if cfg.tke_n2_evaluation_stage == "step_entry":
            ri_cri = jnp.asarray(1.0 / cfg.prandtl_ri_coeff, dtype=N2.dtype)
            pdlr = jnp.maximum(0.1, ri_cri / jnp.maximum(ri_cri, zri))  # coeff-ok: NEMO zdfric pdlr floor
            return 1.0 / pdlr
        return jnp.maximum(
            1.0, jnp.minimum(10.0, cfg.prandtl_ri_coeff * zri))
    raise ValueError(
        f"Unknown prandtl_mode={cfg.prandtl_mode!r}; expected 'unit', "
        f"'constant', 'richardson' or 'nemo_ri'."
    )


def _bryan_lewis_kappaH_floor(
    z_interface: jnp.ndarray, cfg: TKEConfig,
) -> jnp.ndarray:
    r"""Bryan & Lewis (1979) depth-dependent tracer-diffusivity floor.

    Veros ``enable_kappaH_profile`` (``veros/core/tke.py:94-102``):

    .. math::

        \kappa_H^{floor}(z) = \left(0.8 + \frac{1.05}{\pi}\,
            \arctan\!\frac{-z - 2500}{222.2}\right) \times 10^{-4}

    with ``z`` the interface position [m] (negative downward; Veros uses
    ``-zw`` with ``zw < 0``). Mainly raises the abyssal diffusivity below
    ~2500 m. Pure arithmetic -> differentiable.  The fit coefficients are
    ``TKEConfig.bg_diff_*`` (#518 item 10): the amp/arctan-coeff/depth/width
    are the fixed Bryan-Lewis published profile, ``bg_diff_scale`` the
    abyssal-floor amplitude.
    """
    # -z = depth (positive); Veros's argument is (-zw - 2500)/222.2 with
    # zw the (negative) interface height -> here z_interface plays zw.
    depth = -z_interface
    return (cfg.bg_diff_amp + cfg.bg_diff_arctan_coeff / jnp.pi
            * jnp.arctan((depth - cfg.bg_diff_depth_m) / cfg.bg_diff_width_m)
            ) * cfg.bg_diff_scale


def compute_K_from_tke(
    e: jnp.ndarray,
    l_k: jnp.ndarray,
    cfg: TKEConfig,
    *,
    N2: jnp.ndarray | None = None,
    shear_sq: jnp.ndarray | None = None,
    z_interface: jnp.ndarray | None = None,
    N2_prandtl: jnp.ndarray | None = None,
    p_sh2_override: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    prandtl_K_M: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Compute K_M and K_H from TKE and the mixing length.

    .. math::

        K_M = c_k \, l_k \, \sqrt{2 e}, \qquad
        K_M \leftarrow \min(\kappa_{M,\max},\, K_M)

    The ``kappaM_max`` ceiling (Veros default 100 m²/s) caps the
    convective viscosity when ``l_k`` blows up over a statically-unstable
    column. ``K_M`` is then floored at ``kappaM_min``.

    Tracer diffusivity ``K_H`` (Prandtl chain — the abyssal
    over-diffusion fix):

    - ``prandtl_mode="unit"`` (default, BIT-IDENTICAL legacy):
      ``K_H = max(K_M, kappaH_min)`` — the MOMENTUM floor ``kappaM_min``
      leaks into the tracer floor.
    - ``prandtl_mode in {"constant", "richardson"}``: Veros's
      ``K_H = max(kappaH_min, K_M / Pr)`` where ``K_M`` here is the
      ceilinged-but-UN-``kappaM_min``-floored viscosity, so the momentum floor
      does NOT leak into the tracer floor (NEMO floors ``avt`` at ``avtb``
      INDEPENDENTLY of ``avm`` at ``avmb``); requires ``N2`` and ``shear_sq``
      for the ``"richardson"`` Pr.

    Amplitude convention (``cfg.kappa_convention``):

    - ``"gaspar_sqrt2e"`` (default, BIT-IDENTICAL legacy): the Gaspar form
      ``K_M = c_k·l_k·sqrt(2·max(e, tke_background))``. On the signed-N²
      path the Veros buoyancy length already carries the ``sqrt(2)``
      (``mxl = √2·√e/√N̄``), so this DOUBLE-COUNTS it — K_M ×1.414 vs the
      oracle wherever the caps/floors don't bind.
    - ``"veros_sqrte"``: Veros tke.py:73 ``kappaM = c_k·mxl·sqrttke`` with
      ``sqrttke = sqrt(max(0, e))`` (zero in negative-TKE debt regions).

    N2_prandtl : (..., nlev-1) or None
        N² fed to the Prandtl-number computation (T8: NEMO's ``zri`` reads
        ``rn2b``, the BEFORE/Nbb level — genuinely different from the
        ``rn2``/Nnow ``N2`` used for the mixing length / buoyancy sink).
        None (default, BIT-IDENTICAL) ⇒ falls back to ``N2``.

    Returns
    -------
    K_M, K_H : (..., nlev-1) at interfaces.
    """
    kappa_convention = getattr(cfg, "kappa_convention", "gaspar_sqrt2e")
    if kappa_convention == "gaspar_sqrt2e":
        K_M = cfg.c_k * l_k * jnp.sqrt(
            2.0 * jnp.maximum(e, cfg.tke_background))
    elif kappa_convention == "veros_sqrte":
        # Double-``where`` for an AD-safe sqrt where the carried TKE is
        # negative (primal BIT-IDENTICAL to sqrt(max(0,e)); see the note in
        # ``_veros_buoyancy_length``).
        K_M = cfg.c_k * l_k * jnp.where(
            e > 0.0, jnp.sqrt(jnp.where(e > 0.0, e, 1.0)), 0.0)
    else:
        raise ValueError(
            f"Unknown TKEConfig.kappa_convention={kappa_convention!r}; "
            f"expected 'gaspar_sqrt2e' or 'veros_sqrte'."
        )

    if cfg.prandtl_mode == "unit":
        # Legacy path — BIT-IDENTICAL: no kappaM_max ceiling, tracer floor
        # inherits the momentum floor (kappaH_min dead).
        K_M = jnp.maximum(K_M, cfg.kappaM_min)
        K_H = jnp.maximum(K_M, cfg.kappaH_min)
    else:
        # Convective ceiling BEFORE the floor (Veros: kappaM = min(kappaM_max,
        # c_k*mxl*sqrttke) then max(kappaM_min, kappaM)). Only on the
        # opt-in Prandtl path so the default stays bit-identical.
        K_M = jnp.minimum(cfg.kappaM_max, K_M)
        if N2 is None or shear_sq is None:
            raise ValueError(
                f"prandtl_mode={cfg.prandtl_mode!r} requires N2 and "
                f"shear_sq for the Prandtl-number computation."
            )
        _N2_pr = N2 if N2_prandtl is None else N2_prandtl
        _K_M_pr = K_M if prandtl_K_M is None else prandtl_K_M
        if _K_M_pr.shape != K_M.shape:
            raise ValueError(
                "compute_K_from_tke: prandtl_K_M must match K_M shape; "
                f"got {_K_M_pr.shape} vs {K_M.shape}.")
        _p_sh2_pr = (None if p_sh2_override is None
                     else p_sh2_override(_K_M_pr))
        Pr = _prandtl_number(_N2_pr, shear_sq, _K_M_pr, cfg, _p_sh2_pr)
        # Tracer floor is INDEPENDENT of the momentum floor (NEMO zdftke:
        # avt = max(avtb, pdlr*zav), avm = max(avmb, zav), both from the raw K).
        # Divide the ceilinged-but-UN-kappaM_min-floored K_M by Pr, then floor at
        # kappaH_min — else the momentum floor kappaM_min leaks into the tracer
        # floor (K_H -> kappaM_min/Pr > kappaH_min) in quiescent cells where the
        # raw K < kappaM_min (the abyss). Only those cells change; active/interior
        # cells (raw K >= kappaM_min) are unaffected, so the verified avt match
        # (corr 0.9998) holds. NB: the quiescent deep K_H settles to kappaH_min
        # ONLY when the Bryan-Lewis profile below is off (enable_kappaH_profile=
        # False, as the NEMO recipe sets for NEMO's constant avtb); with BL on the
        # BL depth floor becomes the binding deep floor instead.
        K_H = jnp.maximum(cfg.kappaH_min, K_M / Pr)
        # Bryan-Lewis (1979) arctan depth floor on K_H (Veros
        # enable_kappaH_profile). Previously recorded-but-ignored; now wired
        # on the opt-in Prandtl path. Low impact in shallow domains; raises
        # the abyssal tracer floor below ~2500 m. Requires the interface
        # depths.
        if cfg.enable_kappaH_profile and z_interface is not None:
            K_H = jnp.maximum(K_H, _bryan_lewis_kappaH_floor(z_interface, cfg))
        # Momentum floor, applied AFTER K_H so kappaM_min stays out of the tracer
        # floor (NEMO avm = max(avmb, zav)).
        K_M = jnp.maximum(K_M, cfg.kappaM_min)
    return K_M, K_H


# ---------------------------------------------------------------------------
# NEMO zdftke surface terms: Langmuir source (ln_lc) + sub-ML penetration
# (nn_etau).  Faithful ports of NEMO 5.0.1 zdftke.F90; see TKEConfig.
# ---------------------------------------------------------------------------


def nemo_tke_effective_ice_fraction(
    ice_fraction: jnp.ndarray,
    nn_eice: int,
) -> jnp.ndarray:
    """Return NEMO ``zice_fra`` for the selected ``nn_eice`` arm.

    This is the single shared transcription of ``zdftke.F90:246,253-258``.
    In particular, mode 1 is ``TANH(fr_i*10._wp)``; it is *not* the raw ice
    fraction (that is NEMO mode 2).  The multiplication is materialized at the
    Fortran source-statement boundary and TANH follows the active scalar-libm
    precision policy.  Modes 0 and 3 preserve their established expressions.
    """
    mode = int(nn_eice)
    value = jnp.asarray(ice_fraction)
    if mode == 0:
        return jnp.zeros_like(value)
    if mode == 1:
        argument = nemo_source_round(
            value * jnp.asarray(10.0, dtype=value.dtype))
        return nemo_source_round(precision_tanh(argument))
    if mode == 2:
        # NEMO zdftke.F90:256 assigns the resolved sea-ice fraction without
        # transformation.  Keeping this as its own arm preserves NEMO's
        # numbering: mode 1 is tanh(10*fi), while raw fi is mode 2.
        return value
    if mode == 3:
        return jnp.minimum(
            jnp.asarray(4.0, dtype=value.dtype) * value,
            jnp.asarray(1.0, dtype=value.dtype),
        )
    raise ValueError(
        f"Unknown TKEConfig.eice={mode!r}; expected NEMO nn_eice 0, 1, 2 or 3.")


def _nemo_literal_langmuir_operands(
    taum: jnp.ndarray,
    N2: jnp.ndarray,
    depth_w: jnp.ndarray,
    dz_w: jnp.ndarray,
    cfg: TKEConfig,
    ice_frac: jnp.ndarray | None,
    bottom_level: jnp.ndarray | None,
    w_active: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return literal ``zus3, zwlc, zhlc, apply`` operands for line 463."""
    if bottom_level is None or w_active is None:
        raise ValueError(
            "tke_langmuir_evaluation='nemo_literal' requires bottom_level "
            "and w_active for the per-column mbkt+1 fallback and wmask.")
    if w_active.shape != N2.shape:
        raise ValueError(
            "literal Langmuir w_active must match N2 shape; got "
            f"{w_active.shape} vs {N2.shape}.")
    if bottom_level.shape != N2.shape[:-1]:
        raise ValueError(
            "literal Langmuir bottom_level must match the horizontal N2 "
            f"shape; got {bottom_level.shape} vs {N2.shape[:-1]}.")

    # Literal active DINO arm, zdftke.F90:422-463. Depth is positive down;
    # the vertical loops are deliberately scans so floating-point association
    # follows the oracle source order.
    half_wlc2 = _NEMO_TKE_LC_CSD * taum
    depth_b = jnp.broadcast_to(depth_w, N2.shape)
    dz_b = jnp.broadcast_to(dz_w, N2.shape)
    pe_term = jnp.maximum(N2, 0.0) * depth_b * dz_b

    def accumulate_pe(carry, term):
        updated = carry + term
        return updated, updated

    _, pe_vertical = jax.lax.scan(
        accumulate_pe,
        jnp.zeros_like(pe_term[..., 0]),
        jnp.moveaxis(pe_term, -1, 0),
    )
    pe = jnp.moveaxis(pe_vertical, 0, -1)

    # NEMO initializes imlc=mbkt+1 independently in every column, then walks
    # jk=jpkm1..2. Since this array starts at Fortran jk=2, bottom_level is
    # already the matching Python fallback index and the last array row
    # (Fortran jpk) is excluded from the reverse threshold scan.
    fallback = jnp.clip(
        jnp.asarray(bottom_level, dtype=jnp.int32), 0, N2.shape[-1] - 1)
    reverse_pe = jnp.moveaxis(pe[..., :-1][..., ::-1], -1, 0)
    reverse_k = jnp.arange(N2.shape[-1] - 2, -1, -1, dtype=jnp.int32)

    def select_imlc(imlc, operands):
        pe_at_k, k = operands
        return jnp.where(pe_at_k > half_wlc2, k, imlc), None

    imlc, _ = jax.lax.scan(select_imlc, fallback, (reverse_pe, reverse_k))
    h_lc = jnp.take_along_axis(
        depth_b, imlc[..., None], axis=-1)[..., 0]
    h_lc = jnp.maximum(h_lc, _EPS)

    # NEMO: zus = SQRT( 2. * zcof * taum ) (zdftke.F90:447).  A bare sqrt has
    # an INFINITE derivative at zero stress, so reverse mode returns NaN over
    # land and in calm columns and that NaN survives the ``apply`` mask below
    # (0 * inf = NaN).  The double-``where`` is the AD-safe sqrt pattern also
    # used by ``_safe_stress_modulus`` and ``_veros_buoyancy_length`` in this
    # module, but the guard here is deliberately ``!= 0`` where those two use
    # ``> 0``: a NEGATIVE argument is not physical for a stress modulus, and
    # ``!= 0`` lets it keep reaching the sqrt instead of being silently
    # rewritten to a valid 0, so it stays distinguishable from a genuine calm
    # column.
    #
    # Primal equivalence to the bare sqrt holds at every input EXCEPT negative
    # zero, where the bare root returns -0.0 and this guard returns +0.0.  That
    # is inert here: the root is only ever cubed behind the ``zus3 != 0.0``
    # mask below, which rejects both signed zeros, so no module output moves.
    # Only the derivative AT exactly zero changes, from +inf to the correct
    # limit 0 (zus3 ~ taum^{3/2}).
    #
    # Three measured behaviour facts of this arm, recorded not fixed:
    #   * Negative stress does NOT surface in the primal.  ``half_wlc2 < 0``
    #     makes every cumulative-PE level exceed the threshold, so imlc lands
    #     on the shallowest interface, ``apply`` is empty and the source is
    #     exactly 0.0 while the gradient is NaN.  Corruption is therefore
    #     visible in the gradient only, not in the forward solution.
    #   * This literal arm carries NO ``max(taum, 0)`` clamp, unlike the
    #     compact "vectorized" arm above; NEMO has no clamp either, so the
    #     omission is fidelity-correct, but any card switched from the compact
    #     arm to this one LOSES that non-negativity guard.
    #   * Second-order AD at zero stress now reports zero curvature where the
    #     true curvature is infinite, so a Hessian-based calibration reads calm
    #     columns as flat rather than as NaN.  KNOWN OPEN ITEM, not fixed here:
    #     the compact arm on the library default still returns NaN at second
    #     order at zero stress.
    _zus_arg = 2.0 * half_wlc2
    _zus_nonzero = _zus_arg != 0.0
    zus = jnp.where(
        _zus_nonzero, jnp.sqrt(jnp.where(_zus_nonzero, _zus_arg, 1.0)), 0.0)
    ice_scale = (jnp.ones_like(zus) if ice_frac is None
                 else jnp.maximum(0.0, 1.0 - ice_frac))
    surface_wet = jnp.asarray(w_active[..., 0], dtype=N2.dtype)
    zus3 = ice_scale * zus * zus * zus * surface_wet
    zwlc = cfg.lc_coeff * jnp.sin(
        jnp.pi * depth_b / h_lc[..., None])
    apply = ((zus3[..., None] != 0.0)
             & ((depth_b - h_lc[..., None]) < 0.0)
             & jnp.asarray(w_active, dtype=bool))
    return zus3, zwlc, h_lc, apply


def nemo_langmuir_tke_source(
    taum: jnp.ndarray,
    N2: jnp.ndarray,
    depth_w: jnp.ndarray,
    dz_w: jnp.ndarray,
    cfg: TKEConfig,
    ice_frac: jnp.ndarray | None = None,
    bottom_level: jnp.ndarray | None = None,
    w_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Langmuir-circulation TKE source [m²/s³] (Axell 2002; NEMO ln_lc).

    Faithful port of ``zdftke.F90`` lines 305-370 (the no-Stokes-coupling
    branch — Stokes drift deduced from the surface stress):

    1. ``½W_lc² = ½·0.016²·|τ| / (ρ_air·C_d)``   (Axell Eq. 44 via
       ``|τ| = ρ_air·C_d·U₁₀²`` and ``u_s = 0.016·U₁₀``);
    2. LC depth ``h_lc`` = shallowest interface where the cumulative
       potential energy ``Σ_k max(N²,0)·z_w·Δz_w`` exceeds ``½W_lc²``
       (Axell Eq. 47; whole column stable ⇒ deepest interface);
    3. source(k) = ``u_s³ · (lc_coeff·sin(π z_k/h_lc))³ / h_lc`` for
       ``z_k < h_lc``, else 0 — with ``u_s = √(2·½W_lc²)`` and an
       ice-fraction attenuation ``(1-fice)`` (0 when ``ice_frac=None``).

    Sign convention: z positive DOWN (``depth_w`` > 0 below the surface);
    the source is ≥ 0 (sin > 0 on (0, h_lc)) — pure TKE production.

    Parameters
    ----------
    taum : (...,) surface stress modulus |τ| [N/m²] (≥ 0).
    N2 : (..., nlev-1) buoyancy frequency² at interior interfaces
        (may be signed; clipped ≥ 0 as NEMO uses ``MAX(rn2b, 0)``).
    depth_w : (nlev-1,) or (..., nlev-1) interface depths [m, positive down].
    dz_w : (..., nlev-1) interface control-volume thicknesses (≈ e3w).
    cfg : TKEConfig (uses ``lc_coeff``).
    bottom_level : (...,) int or None
        Python 0-based deepest wet T-cell index. Required by
        ``tke_langmuir_evaluation="nemo_literal"``: it maps exactly to
        NEMO's per-column ``imlc=mbkt+1`` no-crossing W-level fallback.
    w_active : (..., nlev-1) bool or None
        NEMO interior ``wmask``. Required by the literal evaluation.

    Returns
    -------
    (..., nlev-1) TKE source, feedable as ``external_source`` (the RHS gets
    ``+dt·source``, exactly NEMO's ``en += rn_Dt·source``).
    """
    evaluation = getattr(cfg, "tke_langmuir_evaluation", "vectorized")
    if evaluation not in ("vectorized", "nemo_literal"):
        raise ValueError(
            "Unknown TKEConfig.tke_langmuir_evaluation: expected "
            f"'vectorized' or 'nemo_literal', got {evaluation!r}.")

    if evaluation == "vectorized":
        # Historical shared construction. Keep this arm expression-for-
        # expression unchanged: every non-DINO card remains byte-identical.
        half_wlc2 = _NEMO_TKE_LC_CSD * jnp.maximum(taum, 0.0)
        pe = jnp.cumsum(
            jnp.maximum(N2, 0.0) * depth_w * dz_w, axis=-1)
        exceeded = pe > half_wlc2[..., None]
        first = jnp.argmax(exceeded, axis=-1)
        depth_b = jnp.broadcast_to(depth_w, pe.shape)
        h_first = jnp.take_along_axis(
            depth_b, first[..., None], axis=-1)[..., 0]
        h_lc = jnp.where(
            jnp.any(exceeded, axis=-1), h_first, depth_b[..., -1])
        h_lc = jnp.maximum(h_lc, _EPS)
        us3 = (2.0 * half_wlc2) ** 1.5
        if ice_frac is not None:
            us3 = us3 * jnp.maximum(0.0, 1.0 - ice_frac)
        w_lc = cfg.lc_coeff * jnp.sin(
            jnp.pi * depth_b / h_lc[..., None])
        src = us3[..., None] * (w_lc ** 3) / h_lc[..., None]
        return jnp.where(depth_b < h_lc[..., None], src, 0.0)

    zus3, zwlc, h_lc, apply = _nemo_literal_langmuir_operands(
        taum, N2, depth_w, dz_w, cfg, ice_frac, bottom_level, w_active)
    src = (zus3[..., None] * (zwlc * zwlc * zwlc)
           / h_lc[..., None])
    return jnp.where(apply, src, 0.0)


def nemo_literal_langmuir_tke_update(
    e_old: jnp.ndarray,
    dt: float,
    taum: jnp.ndarray,
    N2: jnp.ndarray,
    depth_w: jnp.ndarray,
    dz_w: jnp.ndarray,
    cfg: TKEConfig,
    ice_frac: jnp.ndarray | None = None,
    bottom_level: jnp.ndarray | None = None,
    w_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply active ``zdftke.F90:463`` in its literal association order."""
    zus3, zwlc, h_lc, apply = _nemo_literal_langmuir_operands(
        taum, N2, depth_w, dz_w, cfg, ice_frac, bottom_level, w_active)
    # NEMO: en = en + rn_Dt * zus3 * (zwlc*zwlc*zwlc) / zhlc.
    increment = (((jnp.asarray(dt, dtype=e_old.dtype) * zus3[..., None])
                  * (zwlc * zwlc * zwlc)) / h_lc[..., None])
    return jnp.where(apply, e_old + increment, e_old)


@jax.custom_jvp
def _nemo_binary64_round(value: jnp.ndarray) -> jnp.ndarray:
    """Materialise one binary64 rounding point while retaining identity AD.

    XLA is otherwise free to contract adjacent multiply/add expressions.  The
    glibc vector routine used separate SSE2 instructions, so its source-order
    rounding points are observable at the last bit.  ``nextafter(x, x)`` is
    value-identical but remains an operation boundary; the custom JVP records
    the mathematical identity derivative instead of differentiating the
    representational barrier.
    """
    return jnp.nextafter(value, value)


@_nemo_binary64_round.defjvp
def _nemo_binary64_round_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    return _nemo_binary64_round(value), tangent


def _nemo_glibc234_vector_exp(argument: jnp.ndarray) -> jnp.ndarray:
    """DINO oracle's ordinary-range glibc-2.34 two-lane EXP arithmetic.

    This is a pure-JAX transcription of ``_ZGVbN2v_exp`` as linked into the
    instrumented NEMO executable: 1024-bin reduction, split ln(2)/1024,
    cubic residual polynomial, and integer exponent assembly.  The exact
    lookup bits live in :mod:`_glibc234_exp_table`; no host libm is called.
    glibc's exceptional-input arm is deliberately left to ``jnp.exp``.
    """
    if argument.dtype != jnp.float64:
        return jnp.exp(argument)

    bits = jax.lax.bitcast_convert_type(argument, jnp.uint64)
    high_word = ((bits & jnp.uint64(0x7FFF_FFFF_FFFF_FFFF))  # coeff-ok: sign-bit mask
                 >> jnp.uint64(32))  # coeff-ok: high-word shift
    # Exact pcmpgtd cutoff in the linked routine.  Supplying zero to the
    # unselected literal arm prevents invalid exponent-bit assembly.
    ordinary = high_word <= jnp.uint64(0x4086_232A)  # coeff-ok: glibc pcmpgtd cutoff
    x = jnp.where(ordinary, argument, jnp.asarray(0.0, argument.dtype))
    rounded = _nemo_binary64_round

    product = rounded(x * float.fromhex("0x1.71547652b82fep+10"))
    n = rounded(jnp.rint(product))
    encoded_n = rounded(product + float.fromhex("0x1.8p+52"))
    n_hi = rounded(n * float.fromhex("0x1.62e42fec00000p-11"))
    residual_hi = rounded(x - n_hi)
    n_lo = rounded(n * float.fromhex("0x1.d1cf79abc9e3bp-42"))
    residual = rounded(residual_hi - n_lo)

    poly = rounded(
        float.fromhex("0x1.5555555555556p-3") * residual)
    poly = rounded(poly + float.fromhex("0x1.0000001ebfbe0p-1"))
    poly = rounded(poly * residual)
    poly = rounded(poly + 1.0)
    poly = rounded(poly * residual)
    poly = rounded(poly + 1.0)

    encoded_bits = jax.lax.bitcast_convert_type(encoded_n, jnp.uint64)
    table_index = encoded_bits & jnp.uint64(0x3FF)  # coeff-ok: 1024-entry table index
    exponent_bits = (
        (encoded_bits & jnp.uint64(0xFFFF_FFFF_FFFF_FC00))  # coeff-ok: exponent mask
        << jnp.uint64(42))  # coeff-ok: exponent-field shift
    table_bits = jnp.asarray(
        GLIBC234_EXP_TABLE_BITS, dtype=jnp.uint64)[table_index]
    scale = jax.lax.bitcast_convert_type(
        table_bits + exponent_bits, jnp.float64)
    literal = rounded(scale * poly)
    return jnp.where(ordinary, literal, jnp.exp(argument))


def _nemo_glibc234_vector_sin(argument: jnp.ndarray) -> jnp.ndarray:
    """DINO oracle's glibc-2.34 two-lane SIN arithmetic for latitudes.

    The active host IFUNC resolves ``_ZGVbN2v_sin`` to the SSE4 implementation
    at ``libmvec.so.1+0x3d70``.  Geographic latitude phases lie in
    ``[-pi/2, pi/2]``, where its integer range-reduction index is zero.  This
    pure-JAX transcription therefore preserves the linked routine's seven
    Horner coefficients and separate binary64 MUL/ADD rounding points without
    carrying its irrelevant large-argument scalar fallback.  Values outside
    the geographic interval retain the generic JAX sine.
    """
    if argument.dtype != jnp.float64:
        return jnp.sin(argument)

    rounded = _nemo_binary64_round
    magnitude = jnp.abs(argument)
    ordinary = magnitude <= float.fromhex("0x1.921fb54442d18p+0")
    x = jnp.where(ordinary, magnitude, jnp.asarray(0.0, argument.dtype))
    squared = rounded(x * x)

    # libmvec.so.1 .rodata f840,f800,...,f6c0; each instruction below is
    # one mulpd/addpd in the active glibc-2.34 SSE4 implementation.
    poly = rounded(float.fromhex("-0x1.9f1517e9f65f0p-41") * squared)
    poly = rounded(poly + float.fromhex("0x1.60e6bee01d83ep-33"))
    poly = rounded(poly * squared)
    poly = rounded(poly + float.fromhex("-0x1.ae6355aaa4a53p-26"))
    poly = rounded(poly * squared)
    poly = rounded(poly + float.fromhex("0x1.71de3806add1ap-19"))
    poly = rounded(poly * squared)
    poly = rounded(poly + float.fromhex("-0x1.a01a019a659ddp-13"))
    poly = rounded(poly * squared)
    poly = rounded(poly + float.fromhex("0x1.111111110a573p-7"))
    poly = rounded(poly * squared)
    poly = rounded(poly + float.fromhex("-0x1.55555555554a8p-3"))

    correction = rounded(squared * poly)
    correction = rounded(x * correction)
    literal = rounded(x + correction)
    literal = jnp.copysign(literal, argument)
    return jnp.where(ordinary, literal, jnp.sin(argument))


def _nemo_etau_htau(
    lat_deg: jnp.ndarray,
    cfg: TKEConfig,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Evaluate the nn_htau=1 latitude profile under its selected arithmetic."""
    evaluation = getattr(cfg, "tke_htau_evaluation", "jax_expression")
    if evaluation == "jax_expression":
        # Historical expression: deliberately unchanged for every generic and
        # non-oracle card.
        sine = jnp.sin(jnp.deg2rad(lat_deg))
    elif evaluation == "nemo_literal":
        # NEMO zdftke.F90:493: rpi / 180._wp * gphit.  The caller guarantees
        # that lat_deg is the bridge-carried native gphit array in this arm.
        phase = _nemo_binary64_round(
            jnp.asarray(lat_deg, dtype=dtype)
            * float.fromhex("0x1.1df46a2529d39p-6"))
        sine = _nemo_glibc234_vector_sin(phase)
    else:
        raise ValueError(
            "Unknown TKEConfig.tke_htau_evaluation: expected "
            f"'jax_expression' or 'nemo_literal', got {evaluation!r}.")
    return jnp.maximum(
        _NEMO_TKE_HTAU_MIN_M,
        jnp.minimum(
            _NEMO_TKE_HTAU_MAX_M,
            _NEMO_TKE_HTAU_SLOPE_M * jnp.abs(sine),
        ),
    )


def nemo_etau_injection(
    e: jnp.ndarray,
    taum: jnp.ndarray,
    depth_w: jnp.ndarray,
    cfg: TKEConfig,
    rho_0: float = constants.rho_ocean,
    lat_deg: jnp.ndarray | None = None,
    ice_frac: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Sub-ML penetration of surface TKE (NEMO nn_etau=1, zdftke.F90:492-496).

    ``e(k) += etau_frac · e_sfc · exp(-z_k / h_tau) · (1-fice)`` at every
    interior interface, with the NEMO surface-TKE Dirichlet value
    ``e_sfc = max(rn_emin0, rn_ebb·|τ|/ρ0)`` (zdftke.F90:265) and

    - ``etau_htau_mode="constant10m"`` (nn_htau=0): h_tau = 10 m;
    - ``etau_htau_mode="latitude"``   (nn_htau=1):
      h_tau = max(0.5, min(30, 45·|sin φ|)) m — requires ``lat_deg``.

    Applied AFTER the TKE solve and BEFORE the K_M/K_H computation (the
    NEMO step order: ``tke_tke`` ends with the etau block, then ``tke_avn``
    derives the mixing coefficients).  Additive and ≥ 0 ⇒ preserves TKE
    positivity.  Unknown modes raise (dispatch hardening).
    """
    exp_evaluation = getattr(
        cfg, "tke_etau_exponential_evaluation", "jax_expression")
    if exp_evaluation not in ("jax_expression", "nemo_literal"):
        raise ValueError(
            "Unknown TKEConfig.tke_etau_exponential_evaluation: expected "
            f"'jax_expression' or 'nemo_literal', got {exp_evaluation!r}.")
    mode = cfg.etau_mode
    if mode == "none":
        return e
    if mode != "below_ml":
        raise ValueError(
            f"Unknown TKEConfig.etau_mode={mode!r}; expected 'none' or "
            f"'below_ml' (NEMO nn_etau 0/1).")
    htau_mode = cfg.etau_htau_mode
    if htau_mode == "constant10m":
        htau = jnp.asarray(_NEMO_TKE_HTAU_CONST_M, dtype=e.dtype)
    elif htau_mode == "latitude":
        if lat_deg is None:
            raise ValueError(
                "TKEConfig.etau_htau_mode='latitude' requires lat_deg (the "
                "column latitudes in degrees) to be threaded to the TKE "
                "closure; got None. Use 'constant10m' or pass lat_deg.")
        htau = _nemo_etau_htau(lat_deg, cfg, e.dtype)[..., None]
    else:
        raise ValueError(
            f"Unknown TKEConfig.etau_htau_mode={htau_mode!r}; expected "
            f"'constant10m' or 'latitude' (NEMO nn_htau 0/1).")
    e_sfc = jnp.maximum(_NEMO_TKE_EMIN0, _NEMO_TKE_EBB / rho_0
                        * jnp.maximum(taum, 0.0))
    if exp_evaluation == "jax_expression":
        profile = jnp.exp(-depth_w / htau)
    else:  # validated above
        profile = _nemo_glibc234_vector_exp(-depth_w / htau)
    inj = cfg.etau_frac * e_sfc[..., None] * profile
    if ice_frac is not None:
        inj = inj * jnp.maximum(0.0, 1.0 - ice_frac[..., None])
    return e + inj


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def tke_vertical_mixing(
    u_cell: jnp.ndarray,
    v_cell: jnp.ndarray,
    T_cell: jnp.ndarray,
    S_cell: jnp.ndarray,
    rho_cell: jnp.ndarray,
    dz_half: jnp.ndarray,
    tke_old: jnp.ndarray | None,
    tau_x_surface: jnp.ndarray | None,
    tau_y_surface: jnp.ndarray | None,
    dt: float,
    cfg: TKEConfig,
    rho_0: float = constants.rho_ocean,
    g: float = constants.g,
    n_iterations: int = 1,
    *,
    taum_surface: jnp.ndarray | None = None,
    p_cell: jnp.ndarray | None = None,
    dz_ref: jnp.ndarray | None = None,
    jacobian: jnp.ndarray | None = None,
    eos_fn=None,
    z_interface: jnp.ndarray | None = None,
    external_source: jnp.ndarray | None = None,
    dz_surface: jnp.ndarray | None = None,
    boundary_cap: jnp.ndarray | None = None,
    lat_deg: jnp.ndarray | None = None,
    T_n2: jnp.ndarray | None = None,
    S_n2: jnp.ndarray | None = None,
    t_depth: jnp.ndarray | None = None,
    w_depth: jnp.ndarray | None = None,
    e3w_int: jnp.ndarray | None = None,
    ice_frac: jnp.ndarray | None = None,
    surface_tmask: jnp.ndarray | None = None,
    bottom_dirichlet: jnp.ndarray | None = None,
    bottom_level: jnp.ndarray | None = None,
    T_n2b: jnp.ndarray | None = None,
    S_n2b: jnp.ndarray | None = None,
    u_before_cell: jnp.ndarray | None = None,
    v_before_cell: jnp.ndarray | None = None,
    u_face_now: jnp.ndarray | None = None,
    v_face_now: jnp.ndarray | None = None,
    u_face_before: jnp.ndarray | None = None,
    v_face_before: jnp.ndarray | None = None,
    face_masks_3d: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    w_active: jnp.ndarray | None = None,
    preclosure_K_M: jnp.ndarray | None = None,
    preclosure_K_H: jnp.ndarray | None = None,
    preclosure_K_M_surface: jnp.ndarray | None = None,
    preclosure_dissl: jnp.ndarray | None = None,
    precomputed_p_sh2: jnp.ndarray | None = None,
    precomputed_n2_bundle: TKEEntryN2Bundle | None = None,
    return_statement_trace: bool = False,
    rhs_materialization: str = "",
    rhs_intermediate: str = "",
) -> TKEOutput:
    """Advance the TKE closure and return new K_M, K_H, TKE.

    Mode A — **prognostic** (recommended when state-pytree wiring is
    available): caller passes ``tke_old`` from the previous step and
    ``n_iterations=1``. One backward-Euler tridiagonal solve carries
    TKE forward one ``dt``.

    Mode B — **diagnostic / quasi-steady-state** (used when no prognostic
    TKE state is carried on the ocean state pytree yet): caller passes
    ``tke_old=None`` and ``n_iterations=3`` (or more). The function
    seeds TKE at ``cfg.tke_background`` and iterates the same backward
    Euler step ``n_iterations`` times to converge toward the local
    quasi-steady-state. This produces K_M / K_H within a few percent
    of the prognostic equilibrium for typical ocean shear/stratification,
    at the cost of repeating the tridiagonal solve.

    Mode B is the appropriate default until ``state.tke`` lands on
    :class:`legoesm.ocean.state.LatLonCGridOceanState`. The audit
    documents this as Phase G.1a follow-up.

    Parameters
    ----------
    u_cell, v_cell : (..., nlev)
        Velocity components at cell centres (interpolated from the
        appropriate staggered grid by the caller).
    T_cell, S_cell : (..., nlev)
        Tracer fields at cell centres (accepted for signature parity
        with the other vertical-mixing schemes; unused inside this
        function).
    rho_cell : (..., nlev)
        In-situ density at cell centres — used to compute ``N^2``.
    dz_half : (..., nlev-1)
        Distance between adjacent cell centres.
    tke_old : (..., nlev-1) or None
    tau_x_surface, tau_y_surface : (...)
    bottom_dirichlet : (...,) or None — NEMO bottom TKE BC value (T15;
        :func:`nemo_bottom_tke_dirichlet`), held at the DEEPEST carried
        interface (or, when ``bottom_level`` is also given, at the
        PER-COLUMN seafloor interface — T15-exact). REQUIRED when
        ``cfg.bottom_tke_bc=True`` (raises otherwise — the gate has no
        meaning without a value); ignored (and must be None) when the gate
        is off.
    bottom_level : (...,) int or None — per-column T-point bottom-cell
        index (:class:`OceanPartialCellCoordinate.bottom_level`). None
        (default, BIT-IDENTICAL) ⇒ ``bottom_dirichlet`` pins the array's
        last interface unconditionally (exact on a flat-bottom column,
        the historical behaviour). Ignored when ``bottom_dirichlet`` is
        None.
    dt : float
    cfg : TKEConfig
    n_iterations : int
        See Mode A / Mode B above.
    external_source : (..., nlev-1) or None
        Additive energy-recycling TKE source ``forc`` [m²/s³] at the interior
        interfaces (Veros integrate_tke ``forc = ... + eke_diss_iw + K_diss_bot``),
        applied EXPLICITLY in every sub-iteration's backward-Euler RHS. ``None``
        ⇒ bit-identical to the closure without recycled sources.
    dz_surface : (...) or None
        Surface W control volume ``0.5·dzw_top = -z_full_ref[0]·J`` [m]
        (Veros tke.py:225) — REQUIRED when ``cfg.veros_dz_slots=True``
        (together with ``dz_ref`` and ``jacobian``), ignored otherwise.

    Veros metric slots (``cfg.veros_dz_slots=True``): the closure chain is
    evaluated over the slots Veros actually uses —

    - adiabatic N² over the centre spacing ``dz_half`` (= Veros dzw,
      thermodynamics.py:99) instead of the midpoint reconstruction;
    - buoyancy-length growth allowance = cell thickness ``dzt`` with the
      Veros pass order (tke.py:54-65);
    - TKE-diffusion face gradients over ``dzt`` and per-interface control
      volumes = ``dz_half`` (tke.py:193-222);
    - surface injection over ``0.5·dzw_top`` (tke.py:225).

    Default False ⇒ every path BIT-IDENTICAL legacy.

    ``w_active`` ((..., nlev-1) or None, ``TKEConfig.tke_dry_wmask``) is
    NEMO's ``wmask`` at the interior w-interfaces, forwarded UNCHANGED to
    :func:`_solve_tke_backward_euler` — the one line that transcribes
    ``en = MAX( en, rn_emin ) * wmask``
    (``cfgs/DINO/MY_SRC/zdftke.F90:565`` = upstream
    ``src/OCE/ZDF/zdftke.F90:469``). The mixing lengths are NOT special-cased:
    with ``e == 0`` below the seafloor they fall out at ``rmxl_min`` on their
    own, exactly as in ``tke_avn``. ``None`` (default) ⇒ BIT-IDENTICAL legacy.

    Returns
    -------
    TKEOutput
    """
    _bottom_bc_on = bool(getattr(cfg, "bottom_tke_bc", False))
    if _bottom_bc_on and bottom_dirichlet is None:
        raise ValueError(
            "TKEConfig.bottom_tke_bc=True requires bottom_dirichlet (the "
            "NEMO bottom-friction TKE value, nemo_bottom_tke_dirichlet) to "
            "be passed to tke_vertical_mixing.")
    if not _bottom_bc_on and bottom_dirichlet is not None:
        raise ValueError(
            "bottom_dirichlet was passed but TKEConfig.bottom_tke_bc=False "
            "— set the gate True to actually use it (silent-no-op guard)."
        )
    _literal_langmuir_uses_bottom = (
        bool(getattr(cfg, "lc", False))
        and getattr(cfg, "tke_langmuir_evaluation", "vectorized")
        == "nemo_literal")
    if (bottom_dirichlet is None and bottom_level is not None
            and not _literal_langmuir_uses_bottom):
        raise ValueError(
            "bottom_level was passed but bottom_dirichlet is None — "
            "neither the bottom Dirichlet pin nor literal Langmuir consumes "
            "it (silent-no-op guard).")
    if return_statement_trace:
        _trace_requirements = {
            "one prognostic iteration": int(n_iterations) == 1,
            "nemo_literal matrix": (
                getattr(cfg, "tke_matrix_evaluation", "factored")
                == "nemo_literal"),
            "nemo_literal solver": (
                getattr(cfg, "tke_solver_evaluation", "shared_thomas")
                == "nemo_literal"),
            "active Langmuir": bool(getattr(cfg, "lc", False)),
            "separate z=0 row": (
                getattr(cfg, "tke_surface_bc_level", "interior_pinned")
                == "nemo_z0"),
        }
        if not all(_trace_requirements.values()):
            raise ValueError(
                "return_statement_trace requires the complete literal "
                f"NEMO program; got {_trace_requirements}")
    if (getattr(cfg, "buoyancy_timing", "pre_mixing")
            == "post_mixing_veros"):
        # This orchestrator IS the pre-mixing solve (the TKE budget charged
        # before the tracer implicit mixing). The post-mixing Veros order
        # runs tke_set_diffusivities + tke_integrate_post_mixing from the
        # model step instead — reaching here means a caller (Mode B, SCM,
        # the explicit physics pipeline) cannot honour the option.
        raise ValueError(
            "TKEConfig.buoyancy_timing='post_mixing_veros' cannot be solved "
            "by tke_vertical_mixing (the pre-mixing orchestrator); it "
            "requires the prognostic model-step ordering "
            "(_apply_implicit_vertical_mixing) via tke_set_diffusivities + "
            "tke_integrate_post_mixing."
        )
    _validate_post_mixing_cfg(cfg)
    positivity = getattr(cfg, "positivity", "floor")
    if positivity not in ("floor", "veros_surface_correction"):
        raise ValueError(
            f"Unknown TKEConfig.positivity={positivity!r}; expected 'floor' "
            f"or 'veros_surface_correction'."
        )
    if positivity == "veros_surface_correction" and cfg.n2_mode != "adiabatic":
        # The Veros surface correction lets the interior TKE carry a negative
        # energy debt; only the signed-N² adiabatic buoyancy length handles
        # it (sqrt(max(0,e)) clamp). tke_mxl_choice ∈ {1, 2, 3} are all
        # debt-safe on the adiabatic path: choice=2 via _veros_buoyancy_length's
        # recursion, choice=1 via the distance-to-boundary cap
        # (veros_mxl_choice1_boundary_cap, supplied by the orchestrator),
        # choice=3 via its own double-where sqrt(2e) + the lup/ldown caps. The
        # in-situ N² branch (n2_mode != 'adiabatic') is NOT debt-safe (raw e
        # inside the closed-form sqrt) — fail loudly at config time.
        raise ValueError(
            "TKEConfig.positivity='veros_surface_correction' requires "
            f"n2_mode='adiabatic'; got n2_mode={cfg.n2_mode!r}."
        )

    veros_slots = bool(getattr(cfg, "veros_dz_slots", False))
    if veros_slots:
        if dz_ref is None or jacobian is None or dz_surface is None:
            raise ValueError(
                "TKEConfig.veros_dz_slots=True requires dz_ref, jacobian "
                "and dz_surface (= -z_full_ref[0]·J, the Veros 0.5·dzw_top "
                "surface W volume) to be passed to tke_vertical_mixing."
            )
        # Actual cell thicknesses dzt·J, (..., nlev).
        dz_cell = dz_ref * jacobian[..., jnp.newaxis]
    else:
        dz_cell = None

    _surf_bc_level = getattr(cfg, "tke_surface_bc_level", "interior_pinned")
    # NEMO's virtual-surface face distance is e3t(1) -- the LIVE top-cell
    # thickness -- because both the z=0 row and interior interface 0 are
    # W-points and gdepw(2) - gdepw(1) = e3t(1) exactly (zdftke.F90:403-410,
    # zzd_lw(jk=2) denominator e3t(1)*e3w(2)).  Derived HERE from the
    # dz_ref/jacobian the caller already threads, so the metric has ONE
    # owner.  It is deliberately NOT dz_surface (= -z_full_ref[0]*J), which
    # is the top cell's MIDPOINT depth = half e3t(1) on a midpoint grid and
    # doubled the surface coupling until #1690.
    dz_face_surface = None
    if _surf_bc_level == "nemo_z0":
        _entry_stage = (
            getattr(cfg, "tke_n2_evaluation_stage", "implicit_solve_state")
            == "step_entry" and precomputed_n2_bundle is not None)
        if _entry_stage:
            # n2_evaluation_stage="step_entry" freezes the whole vertical
            # metric at Kmm (the solve below takes dz_half from
            # bundle.e3w_Kmm), so e3t(1) must come from the SAME frozen
            # ladder -- mixing a current-state e3t against a frozen e3w
            # would be a metric inconsistency (codex review, #1690).
            dz_face_surface = precomputed_n2_bundle.e3t_Kmm[..., 0]
        elif dz_ref is None or jacobian is None:
            raise ValueError(
                "TKEConfig.tke_surface_bc_level='nemo_z0' requires dz_ref "
                "and jacobian (the live top-cell thickness e3t(1) = "
                "dz_ref[0]*J is the virtual-surface face distance) to be "
                "passed to tke_vertical_mixing."
            )
        else:
            # Partial cells sit at the SEAFLOOR: h_partial[k] = dz_ref[k] for
            # every k < bottom_level (vertical.py:833), so the top cell is a
            # full cell on any column with more than one wet level.  The sole
            # exception -- bottom_level == 0 -- is the degenerate column whose
            # surface coupling the solver already zeroes outright.
            dz_face_surface = dz_ref[0] * jacobian

    # NEMO nn_mxl=3 (tke_mxl_choice=3) needs the e3t cell thicknesses for the
    # lup/ldown |dl/dz|<=e3t mixing-length sweeps (zdftke.F90:690-704) even
    # when the Veros metric-slot feature (veros_dz_slots) is OFF — the two are
    # independent (choice 3 is the length construction; veros_dz_slots is the
    # TKE-diffusion/injection metric). Derive the cell thicknesses here from the
    # reference thickness + z-star Jacobian the caller already threads, and feed
    # ONLY the mixing-length call — the backward-Euler solver keeps its own
    # veros_slots-gated (dz_cell, dz_surface) pair untouched, so choices 1/2 and
    # every non-veros_slots recipe stay BIT-IDENTICAL.
    dz_cell_mxl = dz_cell
    if cfg.tke_mxl_choice in (3, 4) and dz_cell_mxl is None:
        if dz_ref is None or jacobian is None:
            raise ValueError(
                "tke_mxl_choice=3/4 (NEMO nn_mxl) requires dz_ref and jacobian "
                "(the e3t cell thicknesses) for the lup/ldown mixing-length "
                "sweeps; pass them to tke_vertical_mixing."
            )
        dz_cell_mxl = dz_ref * jacobian[..., jnp.newaxis]

    if tke_old is None:
        leading_shape = rho_cell.shape[:-1]
        nlev = rho_cell.shape[-1]
        tke_old = jnp.full(
            leading_shape + (nlev - 1,),
            cfg.tke_background,
            dtype=rho_cell.dtype,
        )

    _n2_stage = getattr(
        cfg, "tke_n2_evaluation_stage", "implicit_solve_state")
    if _n2_stage not in ("implicit_solve_state", "step_entry"):
        raise ValueError(
            "Unknown TKEConfig.tke_n2_evaluation_stage: expected "
            "'implicit_solve_state' or 'step_entry', got "
            f"{_n2_stage!r}.")
    if _n2_stage == "implicit_solve_state" and precomputed_n2_bundle is not None:
        raise ValueError(
            "precomputed_n2_bundle was supplied while "
            "tke_n2_evaluation_stage='implicit_solve_state'; select "
            "'step_entry' to consume the frozen operands.")
    if _n2_stage == "step_entry":
        if precomputed_n2_bundle is None:
            raise ValueError(
                "tke_n2_evaluation_stage='step_entry' requires "
                "precomputed_n2_bundle from the physical step entry.")
        for _name in ("rn2", "rn2b", "gdepw_Kmm", "e3w_Kmm"):
            _value = getattr(precomputed_n2_bundle, _name)
            if _value.shape != tke_old.shape:
                raise ValueError(
                    f"precomputed_n2_bundle.{_name} must match tke_old "
                    f"shape; got {_value.shape} vs {tke_old.shape}.")
        _e3t_shape = tke_old.shape[:-1] + (tke_old.shape[-1] + 1,)
        if precomputed_n2_bundle.e3t_Kmm.shape != _e3t_shape:
            raise ValueError(
                "precomputed_n2_bundle.e3t_Kmm must contain every live "
                f"T-cell slot; got {precomputed_n2_bundle.e3t_Kmm.shape} "
                f"vs {_e3t_shape}.")
        # zdf_mxl executes in the same carried Kmm geometry as rn2/gdepw/e3w.
        # The prior step-entry implementation validated e3t_Kmm but left the
        # length scans on the later implicit-solve Jacobian, so every sloping
        # column used the wrong e3t operand at zdftke.F90:800-806.
        dz_cell_mxl = precomputed_n2_bundle.e3t_Kmm

    _shear_stage = getattr(
        cfg, "tke_shear_evaluation_stage", "implicit_solve_state")
    if _shear_stage not in ("implicit_solve_state", "step_entry"):
        raise ValueError(
            "Unknown TKEConfig.tke_shear_evaluation_stage: expected "
            "'implicit_solve_state' or 'step_entry', got "
            f"{_shear_stage!r}.")
    if _shear_stage == "implicit_solve_state" and precomputed_p_sh2 is not None:
        raise ValueError(
            "precomputed_p_sh2 was supplied while "
            "tke_shear_evaluation_stage='implicit_solve_state'; select "
            "'step_entry' to consume the frozen operand.")
    if _shear_stage == "step_entry":
        if precomputed_p_sh2 is None:
            raise ValueError(
                "tke_shear_evaluation_stage='step_entry' requires "
                "precomputed_p_sh2 from the selected step-entry face levels.")
        if precomputed_p_sh2.shape != tke_old.shape:
            raise ValueError(
                "precomputed_p_sh2 must match tke_old shape; got "
                f"{precomputed_p_sh2.shape} vs {tke_old.shape}.")

    _coeff_source = getattr(
        cfg, "tke_preclosure_coeff_source", "current_subiteration")
    if _coeff_source not in ("current_subiteration", "carried_previous_step"):
        raise ValueError(
            "Unknown TKEConfig.tke_preclosure_coeff_source: expected "
            "'current_subiteration' or 'carried_previous_step', got "
            f"{_coeff_source!r}.")
    _carried_coeffs = _coeff_source == "carried_previous_step"
    _matrix_eval = getattr(cfg, "tke_matrix_evaluation", "factored")
    if _matrix_eval not in ("factored", "nemo_literal"):
        raise ValueError(
            "Unknown TKEConfig.tke_matrix_evaluation: expected 'factored' "
            f"or 'nemo_literal', got {_matrix_eval!r}.")
    _lc_eval = getattr(cfg, "tke_langmuir_evaluation", "vectorized")
    if _lc_eval not in ("vectorized", "nemo_literal"):
        raise ValueError(
            "Unknown TKEConfig.tke_langmuir_evaluation: expected "
            f"'vectorized' or 'nemo_literal', got {_lc_eval!r}.")
    _supplied_carry = (preclosure_K_M, preclosure_K_H,
                       preclosure_K_M_surface, preclosure_dissl)
    if not _carried_coeffs and any(x is not None for x in _supplied_carry):
        raise ValueError(
            "preclosure K fields were supplied while "
            "tke_preclosure_coeff_source='current_subiteration'; select "
            "'carried_previous_step' to consume them.")
    if _carried_coeffs:
        if not bool(getattr(cfg, "prognostic", False)):
            raise ValueError(
                "tke_preclosure_coeff_source='carried_previous_step' requires "
                "TKEConfig.prognostic=True.")
        if int(n_iterations) != 1:
            raise ValueError(
                "carried_previous_step is a one-physical-step NEMO lifetime "
                "and requires n_iterations=1.")
        if preclosure_K_M is None or preclosure_K_H is None:
            raise ValueError(
                "carried_previous_step requires both preclosure_K_M (avm_k) "
                "and preclosure_K_H (avt_k).")
        if (preclosure_K_M.shape != tke_old.shape
                or preclosure_K_H.shape != tke_old.shape):
            raise ValueError(
                "carried preclosure avm_k/avt_k must match tke_old shape; "
                f"got {preclosure_K_M.shape}/{preclosure_K_H.shape} vs "
                f"{tke_old.shape}.")
    if _matrix_eval == "nemo_literal":
        if not _carried_coeffs:
            raise ValueError(
                "tke_matrix_evaluation='nemo_literal' requires "
                "tke_preclosure_coeff_source='carried_previous_step'.")
        if _n2_stage != "step_entry":
            raise ValueError(
                "tke_matrix_evaluation='nemo_literal' requires "
                "tke_n2_evaluation_stage='step_entry' for live e3t/e3w.")
        if preclosure_dissl is None or preclosure_dissl.shape != tke_old.shape:
            raise ValueError(
                "tke_matrix_evaluation='nemo_literal' requires carried "
                "preclosure_dissl matching tke_old shape; got "
                f"{None if preclosure_dissl is None else preclosure_dissl.shape} "
                f"vs {tke_old.shape}.")
    elif preclosure_dissl is not None:
        raise ValueError(
            "preclosure_dissl was supplied while "
            "tke_matrix_evaluation='factored'; select 'nemo_literal' to "
            "consume it.")

    # Shear-production discretization (T4, Phase-2 #1317; face-native
    # #1226 sh2_walk.py Candidate E/F): "squared_centered" (default,
    # BIT-IDENTICAL) is the now-only squared form; "nemo_burchard" is the
    # Burchard (2002) now×before energy-conserving cross term (still
    # T-point-collapsed); "nemo_face_native" is the FULL zdfsh2.F90:78-94
    # transcription (face-native now×before + wet-only coast-doubling).
    # NEMO feeds the SAME p_sh2 to BOTH the shear-production source AND the
    # Prandtl zri (zdftke.F90:392-395) — so ``shear_sq`` below feeds both
    # consumers identically to NEMO either way.
    _shear_disc = getattr(cfg, "tke_shear_production", "squared_centered")
    if _shear_disc == "nemo_face_native_nbb2":
        _shear_disc = "nemo_face_native_now2"
    if _shear_disc not in (
            "squared_centered", "nemo_burchard", "nemo_face_native",
            "nemo_face_native_now2"):
        raise ValueError(
            "Unknown TKEConfig.tke_shear_production shear-discretization: "
            "must be one of ('squared_centered', 'nemo_burchard', "
            "'nemo_face_native', 'nemo_face_native_now2'), "
            f"got {_shear_disc!r}.")
    _face_native_inputs = (u_face_now, v_face_now, u_face_before,
                          v_face_before, face_masks_3d)
    if _shear_disc in ("nemo_face_native", "nemo_face_native_now2"):
        if (_shear_stage == "implicit_solve_state"
                and _shear_disc == "nemo_face_native" and (
                    u_before_cell is None or v_before_cell is None)):
            raise ValueError(
                "TKEConfig.tke_shear_production='nemo_face_native' "
                "requires u_before_cell and v_before_cell (the carried "
                "leap-frog before-velocities) to be passed to "
                "tke_vertical_mixing.")
        if (_shear_stage == "implicit_solve_state"
                and any(x is None for x in _face_native_inputs)):
            raise ValueError(
                "TKEConfig.tke_shear_production='nemo_face_native' requires "
                "u_face_now, v_face_now, u_face_before, v_face_before and "
                "face_masks_3d (the RAW C-grid face state + per-level "
                "wumask/wvmask/coast masks, zdfsh2.F90:78-94) to be passed "
                "to tke_vertical_mixing.")
        if _shear_stage == "implicit_solve_state":
            from legoesm.ocean.physics.vertical_mixing._shared import (
                vertical_shear_face_native as _vertical_shear_face_native,
            )
            u_mask_3d, v_mask_3d = face_masks_3d
            shear_sq = _vertical_shear_face_native(
                u_face_now, v_face_now, u_face_before, v_face_before,
                dz_half, u_mask_3d, v_mask_3d)
        else:
            # A placeholder only: both p_sh2 consumers below use the frozen
            # field.  Do not touch the implicit-solve velocity operands.
            shear_sq = jnp.zeros_like(precomputed_p_sh2)
    elif _shear_disc == "nemo_burchard":
        if (_shear_stage == "implicit_solve_state"
                and (u_before_cell is None or v_before_cell is None)):
            raise ValueError(
                "TKEConfig.tke_shear_production='nemo_burchard' requires "
                "u_before_cell and v_before_cell (the carried leap-frog "
                "before-velocities, state.u_before/v_before) to be passed "
                "to tke_vertical_mixing.")
        if any(x is not None for x in _face_native_inputs):
            raise ValueError(
                "u_face_now/v_face_now/u_face_before/v_face_before/"
                "face_masks_3d were passed but "
                "TKEConfig.tke_shear_production='nemo_burchard' — set "
                "tke_shear_production='nemo_face_native' to actually use "
                "them (silent-no-op guard).")
        if _shear_stage == "step_entry":
            shear_sq = jnp.zeros_like(precomputed_p_sh2)
        else:
            from legoesm.ocean.physics.vertical_mixing._shared import (
                vertical_shear_burchard as _vertical_shear_burchard,
            )
            shear_sq = _vertical_shear_burchard(
                u_cell, v_cell, u_before_cell, v_before_cell, dz_half)
    else:
        if (_shear_stage == "implicit_solve_state"
                and (u_before_cell is not None or v_before_cell is not None)):
            raise ValueError(
                "u_before_cell/v_before_cell were passed but "
                "TKEConfig.tke_shear_production='squared_centered' — set "
                "tke_shear_production='nemo_burchard' or "
                "'nemo_face_native' to actually use them (silent-no-op "
                "guard).")
        if (_shear_stage == "implicit_solve_state"
                and any(x is not None for x in _face_native_inputs)):
            raise ValueError(
                "u_face_now/v_face_now/u_face_before/v_face_before/"
                "face_masks_3d were passed but "
                "TKEConfig.tke_shear_production='squared_centered' — set "
                "tke_shear_production='nemo_face_native' to actually use "
                "them (silent-no-op guard).")
        shear_sq = (jnp.zeros_like(precomputed_p_sh2)
                    if _shear_stage == "step_entry"
                    else _vertical_shear_squared(u_cell, v_cell, dz_half))

    # avm face-averaging inside p_sh2 (#1455 sh2 chain-walk avm-weighting
    # gap, unpark attempt): "tpoint" (default, BIT-IDENTICAL) keeps the
    # existing K_M*shear_sq external multiply; "nemo_face" instead computes
    # the full zdfsh2.F90:80-94 p_sh2 (avm face-averaged INSIDE the face
    # sum) per sub-iteration from K_M_curr, via
    # _shared.avm_weighted_shear_production. Only meaningful paired with
    # tke_shear_production="nemo_face_native" (same face-native geometry;
    # avm-weighting a T-collapsed shear_sq would double-apply the T-point
    # combine) — raise otherwise (dispatch hardening).
    _avm_weighting = getattr(cfg, "tke_shear_avm_weighting", "tpoint")
    if _avm_weighting not in ("tpoint", "nemo_face"):
        raise ValueError(
            "Unknown TKEConfig.tke_shear_avm_weighting: must be one of "
            f"('tpoint', 'nemo_face'), got {_avm_weighting!r}.")
    _p_sh2_face_fn = None
    if _shear_stage == "step_entry":
        def _p_sh2_face_fn(_kappaM_T):
            return precomputed_p_sh2
    elif _avm_weighting == "nemo_face":
        if _shear_disc != "nemo_face_native":
            raise ValueError(
                "TKEConfig.tke_shear_avm_weighting='nemo_face' requires "
                "tke_shear_production='nemo_face_native' (the avm "
                f"face-averaging is only meaningful with the matching "
                f"face-native shear geometry), got tke_shear_production="
                f"{_shear_disc!r}.")
        from legoesm.ocean.physics.vertical_mixing._shared import (
            avm_weighted_shear_production as _avm_weighted_shear_production,
        )

        def _p_sh2_face_fn(kappaM_T):
            return _avm_weighted_shear_production(
                u_face_now, v_face_now, u_face_before, v_face_before,
                dz_half, u_mask_3d, v_mask_3d, kappaM_T)

    # Static stability N^2. ``"insitu"`` (default) is the clipped in-situ
    # form (BIT-IDENTICAL); ``"adiabatic"`` is the SIGNED Veros parcel-
    # displacement form that lets the TKE convect (N^2 < 0).
    signed_n2 = cfg.n2_mode in ("adiabatic", "nemo_bn2")
    # Diffusivity-stage N² time level (TKEConfig.n2_before_advection): the
    # before-advection (Nnow) T/S override, when supplied by the caller (see
    # tke_set_diffusivities). Python-static; None ⇒ BIT-IDENTICAL.
    _Tn2 = T_cell if T_n2 is None else T_n2
    _Sn2 = S_cell if S_n2 is None else S_n2
    if _n2_stage == "step_entry":
        N2 = precomputed_n2_bundle.rn2
    else:
        N2 = _compute_N2(
            rho_cell, dz_half, rho_0, g,
            T_cell=_Tn2, S_cell=_Sn2, p_cell=p_cell,
            dz_ref=dz_ref, jacobian=jacobian, eos_fn=eos_fn,
            n2_mode=cfg.n2_mode,
            n2_eos_form=getattr(cfg, "n2_eos_form", "seos"),
            adiabatic_over_dz_half=veros_slots,
            t_depth=t_depth, w_depth=w_depth,
            e3w_int=e3w_int,
        )

    # ----- rn2b (T8/T13, NEMO's TRUE before/Nbb level) -----
    # NEMO's ``rn2`` (this ``N2`` — step-entry Nnow T/S, via T_n2/S_n2) feeds
    # the en-equation buoyancy sink AND the mixing length (zdftke.F90:418,650)
    # — the ONLY terms N2 (as computed above) may be used for. ``rn2b`` (the
    # leap-frog BEFORE/Nbb level, stpmlf.F90:186 ``bn2(ts(Nbb))`` — one full
    # leap-frog step behind Nnow) feeds the Prandtl zri (:384-395) and the
    # Langmuir PE integral (:340-344) — genuinely DIFFERENT time levels, not
    # a relabelling of the same tracers. ``T_n2b``/``S_n2b`` (None ⇒
    # BIT-IDENTICAL: rn2b consumers fall back to using this N2, the PRIOR
    # behaviour) let the caller supply the true Nbb tracers (leapfrog
    # ``state.T_before``/``S_before``) so those two consumers get NEMO's
    # actual time level instead.
    if _n2_stage == "step_entry":
        N2b = precomputed_n2_bundle.rn2b
    elif T_n2b is not None or S_n2b is not None:
        if T_n2b is None or S_n2b is None:
            raise ValueError(
                "tke_vertical_mixing: T_n2b and S_n2b must be supplied "
                "together (rn2b needs both tracers).")
        N2b = _compute_N2(
            rho_cell, dz_half, rho_0, g,
            T_cell=T_n2b, S_cell=S_n2b, p_cell=p_cell,
            dz_ref=dz_ref, jacobian=jacobian, eos_fn=eos_fn,
            n2_mode=cfg.n2_mode,
            n2_eos_form=getattr(cfg, "n2_eos_form", "seos"),
            adiabatic_over_dz_half=veros_slots,
            t_depth=t_depth, w_depth=w_depth,
            e3w_int=e3w_int,
        )
    else:
        N2b = N2

    if taum_surface is not None:
        # NEMO taum channel: the caller supplies the surface stress MODULUS
        # directly (e.g. the DINO usrdef x1.3 westerly boost, which enters
        # the TKE input but NOT the momentum stress).
        taum = jnp.maximum(jnp.asarray(taum_surface), 0.0)
        surface_flux = cfg.surface_flux_coeff * (taum / rho_0) ** 1.5
    elif tau_x_surface is None and tau_y_surface is None:
        surface_flux = jnp.zeros(rho_cell.shape[:-1], dtype=rho_cell.dtype)
        taum = surface_flux  # |τ| = 0 (unforced)
    else:
        tx = tau_x_surface if tau_x_surface is not None else jnp.zeros_like(rho_cell[..., 0])
        ty = tau_y_surface if tau_y_surface is not None else jnp.zeros_like(rho_cell[..., 0])
        taum = _safe_stress_modulus(tx, ty)
        surface_flux = cfg.surface_flux_coeff * (taum / rho_0) ** 1.5

    # Surface TKE BC dispatch (single owner; raises on unknown).
    surface_dirichlet = _surface_tke_dirichlet(cfg, taum, rho_0)

    # --- NEMO zdftke surface terms (static feature gates; see TKEConfig) ---
    _lc_on = bool(getattr(cfg, "lc", False))
    _etau_on = getattr(cfg, "etau_mode", "none") != "none"
    if _lc_on or _etau_on:
        if z_interface is None:
            raise ValueError(
                "TKEConfig.lc / etau_mode require z_interface (the interior "
                "interface reference heights) so the NEMO surface terms know "
                "the interface depths; the k_profiles caller passes "
                "z_coord.z_half_ref[1:-1].")
        # z_interface holds NEGATIVE reference heights; NEMO's gdepw is
        # positive-down depth.
        _depth_w = (-z_interface if _n2_stage == "implicit_solve_state"
                    else precomputed_n2_bundle.gdepw_Kmm)
        _surface_e3w = (dz_half if _n2_stage == "implicit_solve_state"
                        else precomputed_n2_bundle.e3w_Kmm)
    if _lc_on:
        # Langmuir source enters the RHS as +dt·source — exactly NEMO's
        # pre-solve ``en += rn_Dt·source`` (zdftke.F90:367). NEMO's PE
        # integral (:340,344) reads ``rn2b`` (the BEFORE/Nbb level) — N2b
        # (defaults to N2 when T_n2b/S_n2b are not supplied, T8/T13).
        if _lc_eval == "nemo_literal":
            if _n2_stage != "step_entry":
                raise ValueError(
                    "tke_langmuir_evaluation='nemo_literal' requires "
                    "tke_n2_evaluation_stage='step_entry' for carried live "
                    "rn2b/gdepw/e3w operands.")
            if bottom_level is None or w_active is None:
                raise ValueError(
                    "tke_langmuir_evaluation='nemo_literal' requires "
                    "bottom_level and w_active.")
            if external_source is not None:
                raise ValueError(
                    "tke_langmuir_evaluation='nemo_literal' cannot merge "
                    "Langmuir with a generic external TKE source; keep the "
                    "source channels separate before selecting this path.")
            if _matrix_eval != "nemo_literal" or int(n_iterations) != 1:
                raise ValueError(
                    "tke_langmuir_evaluation='nemo_literal' requires the "
                    "literal TKE matrix and one prognostic iteration so the "
                    "line-463 en update is consumed exactly once.")
        _lc_src = nemo_langmuir_tke_source(
            taum, N2b, _depth_w, _surface_e3w, cfg,
            ice_frac=ice_frac, bottom_level=bottom_level,
            w_active=w_active)
        external_source = (_lc_src if external_source is None
                           else external_source + _lc_src)

    # Sub-iteration loop (Mode B convergence; Mode A uses n_iterations=1).
    tke_curr = tke_old
    _statement_entry = tke_curr if return_statement_trace else None
    _statement_after_boundaries = None
    _statement_after_langmuir = None
    _statement_rhs = None
    _statement_post_sweep = None
    _statement_matrix_upper = None
    _statement_matrix_lower = None
    _statement_matrix_diag = None
    _statement_shear = None
    _statement_rhs_intermediate = None
    # NEMO ln_mxl0 anchor for nn_mxl=3: l_sfc = max(rn_mxl0, vkarmn*2e5/(rho0*g)*taum)
    _l_anchor = _mxl0_surface_anchor(cfg, taum, rho_0, g, surface_tmask)
    # T3-exact: NEMO's TRUE surface-w-level viscosity avm(jk=1)
    # (:func:`nemo_surface_avm`), consulted only by the nemo_z0 face
    # assembly. Requires the ln_mxl0 anchor (tke_mxl_choice=3) and a held
    # surface value (surface_dirichlet) — both already validated above when
    # surface_bc_level="nemo_z0" is selected. None otherwise ⇒ the
    # documented approximation, bit-identical to the prior behaviour.
    _K_M_surface = None
    if _surf_bc_level == "nemo_z0" and _l_anchor is not None:
        _K_M_surface = nemo_surface_avm(cfg, surface_dirichlet, _l_anchor)
    if _carried_coeffs and _surf_bc_level == "nemo_z0":
        if preclosure_K_M_surface is None:
            raise ValueError(
                "carried_previous_step with tke_surface_bc_level='nemo_z0' "
                "requires preclosure_K_M_surface (restart/previous avm_k at "
                "the surface W level).")
        if preclosure_K_M_surface.shape != tke_old.shape[:-1]:
            raise ValueError(
                "preclosure_K_M_surface must match the horizontal TKE shape; "
                f"got {preclosure_K_M_surface.shape} vs {tke_old.shape[:-1]}.")
    for _ in range(max(1, int(n_iterations))):
        l_k, l_eps = compute_mixing_lengths(
            tke_curr, N2, dz_half, cfg, signed_n2=signed_n2,
            dz_cell=dz_cell_mxl, boundary_cap=boundary_cap,
            l_surface_anchor=_l_anchor)
        K_M_curr, K_H_curr = compute_K_from_tke(
            tke_curr, l_k, cfg, N2=N2, shear_sq=shear_sq,
            z_interface=z_interface, N2_prandtl=N2b,
            p_sh2_override=_p_sh2_face_fn,
            prandtl_K_M=(preclosure_K_M if _carried_coeffs else None))
        _K_M_pre = preclosure_K_M if _carried_coeffs else K_M_curr
        _K_H_pre = preclosure_K_H if _carried_coeffs else K_H_curr
        P_s_curr = (_K_M_pre * shear_sq if _p_sh2_face_fn is None
                    else _p_sh2_face_fn(_K_M_pre))
        _literal_external_rhs = None
        if return_statement_trace:
            _surface_row = jnp.asarray(
                surface_dirichlet, dtype=tke_curr.dtype)[..., None]
            # The current model program holds the surface row separately and
            # delays its bottom identity scatter until matrix assembly.  This
            # is the actual value presented to its Langmuir statement, not a
            # replay with the NEMO bottom assignment substituted.
            _statement_after_boundaries = jnp.concatenate(
                [_surface_row, tke_curr], axis=-1)
        if _lc_on and _lc_eval == "nemo_literal":
            _literal_external_rhs = nemo_literal_langmuir_tke_update(
                tke_curr, dt, taum, N2b, _depth_w, _surface_e3w, cfg,
                ice_frac=ice_frac, bottom_level=bottom_level,
                w_active=w_active)
        _solve_result = _solve_tke_backward_euler(
            e_old=tke_curr,
            K_M_old=_K_M_pre, K_H_old=_K_H_pre,
            P_s=P_s_curr, N2=N2, l_eps=l_eps,
            dz_half=(precomputed_n2_bundle.e3w_Kmm
                     if _n2_stage == "step_entry" else dz_half),
            surface_flux=surface_flux,
            dt=dt, cfg=cfg,
            external_source=external_source,
            literal_external_rhs=_literal_external_rhs,
            dz_cell=dz_cell,
            dz_surface=(dz_surface if veros_slots else None),
            dz_face_surface=dz_face_surface,
            surface_dirichlet=surface_dirichlet,
            surface_bc_level=_surf_bc_level,
            bottom_dirichlet=bottom_dirichlet,
            K_M_surface=(preclosure_K_M_surface
                         if _carried_coeffs else _K_M_surface),
            bottom_level=bottom_level,
            w_active=w_active,
            nemo_e3t=(precomputed_n2_bundle.e3t_Kmm
                      if _matrix_eval == "nemo_literal" else None),
            dissl_old=(preclosure_dissl
                       if _matrix_eval == "nemo_literal" else None),
            return_statement_trace=return_statement_trace,
            rhs_materialization=rhs_materialization,
            rhs_intermediate=rhs_intermediate,
        )
        if return_statement_trace:
            (tke_curr, _statement_langmuir_interior, _statement_rhs,
             _statement_matrix_upper, _statement_matrix_lower,
             _statement_matrix_diag,
             _statement_rhs_intermediate) = _solve_result
            _statement_shear = P_s_curr
            _statement_after_langmuir = jnp.concatenate(
                [_surface_row, _statement_langmuir_interior], axis=-1)
            _statement_post_sweep = jnp.concatenate(
                [_surface_row, tke_curr], axis=-1)
        else:
            tke_curr = _solve_result

    if _etau_on:
        # NEMO step order: the etau injection closes tke_tke (AFTER the
        # implicit solve), then tke_avn derives K_M/K_H from the updated
        # field (zdftke.F90:492-496). Applied ONCE per call — NOT per
        # Mode-B sub-iteration (that would triple the non-time-scaled
        # injection in the diagnostic n_iterations=3 chain; codex P1/P2
        # review finding #2). Prognostic mode (n_iterations=1) identical.
        tke_curr = nemo_etau_injection(
            tke_curr, taum, _depth_w, cfg, rho_0=rho_0, lat_deg=lat_deg,
            ice_frac=ice_frac)

    # Final K from converged TKE.
    l_k_final, l_eps_final = compute_mixing_lengths(
        tke_curr, N2, dz_half, cfg, signed_n2=signed_n2,
        dz_cell=dz_cell_mxl, boundary_cap=boundary_cap,
        l_surface_anchor=_l_anchor)
    K_M, K_H = compute_K_from_tke(
        tke_curr, l_k_final, cfg, N2=N2, shear_sq=shear_sq,
        z_interface=z_interface, N2_prandtl=N2b,
        p_sh2_override=_p_sh2_face_fn,
        prandtl_K_M=(preclosure_K_M if _carried_coeffs else None))

    dissl_new = None
    if _matrix_eval == "nemo_literal":
        # Post-solve tke_avn overwrite, MY_SRC/zdftke.F90:832-837.  Keep the
        # source association: zsqen=SQRT(en), then dissl=zsqen/zmxld.
        dissl_new = jnp.sqrt(tke_curr) / l_eps_final
    _statement_trace = None
    if return_statement_trace:
        if any(value is None for value in (
                _statement_entry, _statement_after_boundaries,
                _statement_after_langmuir, _statement_rhs,
                _statement_post_sweep, _statement_matrix_upper,
                _statement_matrix_lower, _statement_matrix_diag,
                _statement_shear)):
            raise ValueError("requested TKE statement trace is incomplete")
        _statement_trace = TKEStatementTrace(
            en_entry=_statement_entry,
            taum_surface=taum,
            surface_dirichlet=surface_dirichlet,
            en_after_boundaries=_statement_after_boundaries,
            en_after_langmuir=_statement_after_langmuir,
            rhs_pre_sweep=_statement_rhs,
            en_post_sweep=_statement_post_sweep,
            matrix_upper=_statement_matrix_upper,
            matrix_lower=_statement_matrix_lower,
            matrix_diag=_statement_matrix_diag,
            rhs_shear=_statement_shear,
            rhs_intermediate=_statement_rhs_intermediate,
        )
    return TKEOutput(K_M=K_M, K_H=K_H, tke_new=tke_curr,
                     l_eps=l_eps_final, K_M_surface=_K_M_surface,
                     dissl=dissl_new, statement_trace=_statement_trace)


# ---------------------------------------------------------------------------
# POST-MIXING Veros step order (TKEConfig.buoyancy_timing="post_mixing_veros")
# ---------------------------------------------------------------------------


def _validate_post_mixing_cfg(cfg: TKEConfig) -> None:
    """Fail loudly unless the post-mixing prerequisites hold (see config doc)."""
    eice = getattr(cfg, "eice", 0)
    if eice not in (0, 1, 2, 3):
        raise ValueError(
            f"Unknown TKEConfig.eice={eice!r}; expected NEMO nn_eice "
            "0, 1, 2 or 3.")
    timing = getattr(cfg, "buoyancy_timing", "pre_mixing")
    if timing not in ("pre_mixing", "post_mixing_veros"):
        raise ValueError(
            f"Unknown TKEConfig.buoyancy_timing={timing!r}; expected "
            f"'pre_mixing' or 'post_mixing_veros'."
        )
    shear = getattr(cfg, "shear_production", "pre_solve")
    if shear not in ("pre_solve", "realized_veros"):
        raise ValueError(
            f"Unknown TKEConfig.shear_production={shear!r}; expected "
            f"'pre_solve' or 'realized_veros'."
        )
    _tke_shear = getattr(cfg, "tke_shear_production", "squared_centered")
    if _tke_shear == "nemo_face_native_nbb2":
        _tke_shear = "nemo_face_native_now2"
    if _tke_shear not in ("squared_centered", "nemo_burchard",
                          "nemo_face_native", "nemo_face_native_now2"):
        raise ValueError(
            "Unknown TKEConfig.tke_shear_production shear-discretization: "
            "must be one of ('squared_centered', 'nemo_burchard', "
            "'nemo_face_native', 'nemo_face_native_now2'), "
            f"got {_tke_shear!r}.")
    _avm_w = getattr(cfg, "tke_shear_avm_weighting", "tpoint")
    if _avm_w not in ("tpoint", "nemo_face"):
        raise ValueError(
            "Unknown TKEConfig.tke_shear_avm_weighting: must be one of "
            f"('tpoint', 'nemo_face'), got {_avm_w!r}.")
    if timing == "post_mixing_veros" and _avm_w == "nemo_face":
        raise ValueError(
            "TKEConfig.tke_shear_avm_weighting='nemo_face' is not supported "
            "with buoyancy_timing='post_mixing_veros' — tke_set_diffusivities "
            "never assembles the face-weighted p_sh2 and would silently keep "
            "the tpoint weighting. Use the standard pre_mixing path.")
    if timing == "post_mixing_veros" and _tke_shear in (
            "nemo_burchard", "nemo_face_native", "nemo_face_native_now2"):
        raise ValueError(
            f"TKEConfig.tke_shear_production={_tke_shear!r} is not "
            "supported with buoyancy_timing='post_mixing_veros' — "
            "tke_set_diffusivities does not accept u_before_cell/"
            "v_before_cell (or the raw face state nemo_face_native needs) "
            "and would silently keep squared_centered. Disable "
            "tke_shear_production or use the standard pre_mixing path.")
    if timing == "post_mixing_veros":
        if not (getattr(cfg, "prognostic", False)
                and getattr(cfg, "veros_dz_slots", False)
                and cfg.n2_mode == "adiabatic"
                and getattr(cfg, "positivity", "floor")
                == "veros_surface_correction"):
            raise ValueError(
                "TKEConfig.buoyancy_timing='post_mixing_veros' requires "
                "prognostic=True, veros_dz_slots=True, n2_mode='adiabatic' "
                "and positivity='veros_surface_correction' (the Veros "
                "integrate_tke form); got prognostic="
                f"{getattr(cfg, 'prognostic', False)!r}, veros_dz_slots="
                f"{getattr(cfg, 'veros_dz_slots', False)!r}, n2_mode="
                f"{cfg.n2_mode!r}, positivity="
                f"{getattr(cfg, 'positivity', 'floor')!r}."
            )
        # tke_mxl_choice ∈ {1, 2} are both debt-safe under the post-mixing
        # surface correction. choice=2 is bounded by the MITgcm/OPA recursion
        # (_veros_buoyancy_length); choice=1 is bounded by the Veros
        # distance-to-boundary cap (veros_mxl_choice1_boundary_cap, wired in by
        # the orchestrator). Both match Veros (only global_1deg selects
        # choice=1). compute_mixing_lengths raises on any other value.
        #
        # Mixed-oracle guard (RELAXED for lc 2026-07-16: the Langmuir source is
        # now computed in tke_set_diffusivities and applied pre-solve inside
        # tke_integrate_post_mixing — NEMO zdftke:367 en += rDt*source — so it
        # no longer silently no-ops). etau stays blocked: the NEMO zdftke etau term is
        # implemented on the standard orchestrator path only — the Veros
        # post-mixing step order has no such terms (Veros has no ln_lc /
        # nn_etau). Combining them would silently no-op (this path never
        # calls the injections) or mix oracle semantics — raise instead.
        if getattr(cfg, "etau_mode", "none") != "none":
            raise ValueError(
                "TKEConfig.etau_mode (the NEMO zdftke sub-ML TKE penetration) is "
                "not supported with buoyancy_timing='post_mixing_veros' "
                "(the Veros-faithful step order has no Langmuir/etau terms; "
                "they would silently not be applied). Disable lc/etau or "
                "use the standard pre_mixing path.")
        if getattr(cfg, "tke_surface_bc_level",
                   "interior_pinned") != "interior_pinned":
            raise ValueError(
                "TKEConfig.tke_surface_bc_level='nemo_z0' is not supported "
                "with buoyancy_timing='post_mixing_veros' — the post-mixing "
                "solve (tke_integrate_post_mixing) does not accept "
                "surface_bc_level and would silently keep the "
                "interior_pinned Dirichlet placement. Disable "
                "tke_surface_bc_level or use the standard pre_mixing path.")
        if getattr(cfg, "tke_buoyancy_sink",
                   "implicit_linearized") != "implicit_linearized":
            raise ValueError(
                "TKEConfig.tke_buoyancy_sink='nemo_explicit' is not "
                "supported with buoyancy_timing='post_mixing_veros' — the "
                "post-mixing solve already charges P_diss_v = kappaH*N^2 "
                "fully explicitly (Veros integrate_tke forc), so this axis "
                "is not consulted there and would silently no-op. Disable "
                "tke_buoyancy_sink or use the standard pre_mixing path.")
        if getattr(cfg, "bottom_tke_bc", False):
            raise ValueError(
                "TKEConfig.bottom_tke_bc=True is not supported with "
                "buoyancy_timing='post_mixing_veros' — tke_integrate_post_"
                "mixing does not accept bottom_dirichlet and would silently "
                "keep the natural no-flux bottom row. Disable "
                "bottom_tke_bc or use the standard pre_mixing path.")
        if getattr(cfg, "tke_dry_wmask", False):
            raise ValueError(
                "TKEConfig.tke_dry_wmask=True is not supported with "
                "buoyancy_timing='post_mixing_veros' — the `* wmask` it "
                "transcribes rides on the `MAX(en, rn_emin)` post-solve floor "
                "(MY_SRC/zdftke.F90:565), and tke_integrate_post_mixing is "
                "the VEROS integrate_tke form, which leaves the interior "
                "UNFLOORED and never sees this axis, so it would silently "
                "no-op. Disable tke_dry_wmask or use the standard "
                "pre_mixing path.")
    elif shear == "realized_veros":
        raise ValueError(
            "TKEConfig.shear_production='realized_veros' requires "
            "buoyancy_timing='post_mixing_veros' (the realized implicit-"
            "friction increments only exist in the reordered step); got "
            f"buoyancy_timing={timing!r}."
        )


def tke_set_diffusivities(
    u_cell: jnp.ndarray,
    v_cell: jnp.ndarray,
    T_cell: jnp.ndarray,
    S_cell: jnp.ndarray,
    rho_cell: jnp.ndarray,
    dz_half: jnp.ndarray,
    tke_old: jnp.ndarray,
    tau_x_surface: jnp.ndarray | None,
    tau_y_surface: jnp.ndarray | None,
    cfg: TKEConfig,
    rho_0: float,
    g: float,
    *,
    taum_surface: jnp.ndarray | None = None,
    p_cell: jnp.ndarray,
    dz_ref: jnp.ndarray,
    jacobian: jnp.ndarray,
    eos_fn,
    z_interface: jnp.ndarray,
    dz_surface: jnp.ndarray,
    boundary_cap: jnp.ndarray | None = None,
    T_n2: jnp.ndarray | None = None,
    S_n2: jnp.ndarray | None = None,
    t_depth: jnp.ndarray | None = None,
    w_depth: jnp.ndarray | None = None,
    e3w_int: jnp.ndarray | None = None,
    ice_frac: jnp.ndarray | None = None,
    surface_tmask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, TKEPostMixingContext]:
    """Veros ``set_tke_diffusivities`` (tke.py:20-113) from the CARRIED TKE.

    Phase 1 of the post-mixing Veros step order: derive ``K_M``/``K_H`` (and
    the mxl/sqrttke linearisation points of the later TKE solve) from the
    PREVIOUS step's TKE — the kappa profiles the TRACER and MOMENTUM
    implicit solves consume — WITHOUT advancing the TKE field. Composes the
    same shared helpers the legacy orchestrator uses
    (:func:`_compute_N2` / :func:`_vertical_shear_squared` /
    :func:`compute_mixing_lengths` / :func:`compute_K_from_tke`); no
    numerics are re-derived.

    Returns ``(K_M, K_H, ctx)``; ``ctx`` packages every phase-1 ingredient
    :func:`tke_integrate_post_mixing` needs after the tracer solve.
    """
    _validate_post_mixing_cfg(cfg)
    if getattr(cfg, "buoyancy_timing", "pre_mixing") != "post_mixing_veros":
        raise ValueError(
            "tke_set_diffusivities is the post-mixing phase-1 entry point; "
            f"got buoyancy_timing={getattr(cfg, 'buoyancy_timing', None)!r}."
        )
    dz_cell = dz_ref * jacobian[..., jnp.newaxis]
    # _validate_post_mixing_cfg (above) raises for tke_shear_production in
    # ("nemo_burchard", "nemo_face_native") under post_mixing_veros — this
    # entry point has no u_before_cell/face-native inputs to honour either,
    # so "squared_centered" is the ONLY value that can reach here; the call
    # below is provably not a silent dispatch gap (unlike the pre-mixing
    # orchestrator, which dispatches on tke_shear_production explicitly).
    shear_sq = _vertical_shear_squared(u_cell, v_cell, dz_half)
    # Diffusivity-stage N² time level (TKEConfig.n2_before_advection, NEMO
    # eosbn2 Nnow sequencing): when the caller supplies the BEFORE-advection
    # T/S (``T_n2``/``S_n2``), the static-stability contrast is evaluated on
    # them instead of the post-advection ``T_cell``/``S_cell``. Only the T/S
    # parcel pair changes; ``p_cell`` (the reference pressure) stays as-is —
    # sign-neutral by construction (BOTTOM_N2_DIAGNOSIS_FINDINGS.md: the deep
    # marginal interface flips on the T/S contrast, not the reference
    # pressure). Python-static (None ⇒ BIT-IDENTICAL), not a traced branch.
    _Tn2 = T_cell if T_n2 is None else T_n2
    _Sn2 = S_cell if S_n2 is None else S_n2
    N2 = _compute_N2(
        rho_cell, dz_half, rho_0, g,
        T_cell=_Tn2, S_cell=_Sn2, p_cell=p_cell,
        dz_ref=dz_ref, jacobian=jacobian, eos_fn=eos_fn,
        n2_mode=cfg.n2_mode, adiabatic_over_dz_half=True,
        t_depth=t_depth, w_depth=w_depth, e3w_int=e3w_int,
    )
    if taum_surface is not None:
        # NEMO taum channel (see tke_vertical_mixing).
        taum = jnp.maximum(jnp.asarray(taum_surface), 0.0)
    elif tau_x_surface is None and tau_y_surface is None:
        taum = jnp.zeros(rho_cell.shape[:-1], dtype=rho_cell.dtype)
    else:
        tx = (tau_x_surface if tau_x_surface is not None
              else jnp.zeros_like(rho_cell[..., 0]))
        ty = (tau_y_surface if tau_y_surface is not None
              else jnp.zeros_like(rho_cell[..., 0]))
        taum = _safe_stress_modulus(tx, ty)
    surface_flux = cfg.surface_flux_coeff * (taum / rho_0) ** 1.5
    # NEMO nn_bc_surf=1 option (TKEConfig.surface_bc; single-owner dispatch).
    surface_dirichlet = _surface_tke_dirichlet(cfg, taum, rho_0)
    # NEMO ln_lc Langmuir source (zdftke:332-370), applied pre-solve in
    # tke_integrate_post_mixing (en += rDt*source).
    if getattr(cfg, "lc", False):
        if z_interface is None:
            raise ValueError(
                "TKEConfig.lc requires z_interface (interface reference "
                "heights) so the Langmuir source knows the depths.")
        # NB computed from the PRE-mixing N2 and applied in the POST-mixing
        # solve — the same timing offset the post_mixing_veros path accepts
        # for the diffusivities themselves (review note 2026-07-16).
        langmuir_source = nemo_langmuir_tke_source(
            taum, N2, -z_interface, dz_half, cfg, ice_frac=ice_frac)
    else:
        langmuir_source = None

    _l_anchor = _mxl0_surface_anchor(cfg, taum, rho_0, g, surface_tmask)
    l_k, l_eps = compute_mixing_lengths(
        tke_old, N2, dz_half, cfg, signed_n2=True, dz_cell=dz_cell,
        boundary_cap=boundary_cap, l_surface_anchor=_l_anchor)
    K_M, K_H = compute_K_from_tke(
        tke_old, l_k, cfg, N2=N2, shear_sq=shear_sq, z_interface=z_interface)
    ctx = TKEPostMixingContext(
        K_M_old=K_M, K_H_old=K_H, mxl=l_k, l_eps=l_eps,
        # Double-``where`` for an AD-safe sqrt at the negative-TKE energy
        # debt (primal BIT-IDENTICAL to sqrt(max(0,e)); the plain form has a
        # NaN derivative at e <= 0 — see the note in
        # ``_veros_buoyancy_length``).
        sqrttke=jnp.where(
            tke_old > 0.0,
            jnp.sqrt(jnp.where(tke_old > 0.0, tke_old, 1.0)), 0.0),
        shear_sq=shear_sq, tke_old=tke_old, surface_flux=surface_flux,
        dz_half=dz_half, dz_cell=dz_cell,
        dz_surface=jnp.asarray(dz_surface, dtype=rho_cell.dtype),
        p_cell=p_cell, eos_fn=eos_fn, rho_0=rho_0, g=g,
        surface_dirichlet=surface_dirichlet,
        langmuir_source=langmuir_source,
    )
    return K_M, K_H, ctx


def compute_surface_buoyancy_P_diss_v(
    T_sfc: jnp.ndarray,
    S_sfc: jnp.ndarray,
    p_sfc: jnp.ndarray,
    forc_temp_surface: jnp.ndarray,
    forc_salt_surface: jnp.ndarray,
    eos_fn,
    rho_0: float,
    g: float = constants.g,
    eos_salinity_floor: float = 1.0e-3,  # coeff-ok: EOS salinity floor [PSU]
) -> jnp.ndarray:
    r"""Surface buoyancy-flux ``P_diss_v`` slot (Veros thermodynamics.py:304-317,
    386-388).

    .. math::

        \text{forc\_rho\_surface} =
            \frac{\partial\rho}{\partial T}\,F_T^{sfc}
          + \frac{\partial\rho}{\partial S}\,F_S^{sfc},
        \qquad
        P_{diss,v}^{sfc} = -\frac{g}{\rho_0}\,\text{forc\_rho\_surface}

    with the EOS derivatives evaluated at the POST-MIXING (taup1) surface
    T/S (``surf_densityf``) via the shared
    :func:`legoesm.ocean.eos.eos_density_derivatives` (EOS-generic autodiff —
    Veros ``get_drhodT``/``get_drhodS``). ``F_T^{sfc}``/``F_S^{sfc}`` are the
    surface KINEMATIC fluxes [K·m/s] / [PSU·m/s] (Veros
    ``forc_temp_surface``/``forc_salt_surface`` = legoESM's implicit surface
    rate × top-cell thickness). A destabilising flux (surface cooling /
    brine) gives ``forc_rho_surface > 0`` ⇒ ``P_diss_v < 0`` ⇒ the TKE
    forcing ``-P_diss_v > 0`` (buoyancy-driven TKE production). [m²/s³]

    ``eos_salinity_floor``: the EOS DERIVATIVE is evaluated at
    ``max(S_sfc, floor)`` — the TEOS-10/gsw polynomials carry ``√S`` terms
    whose S-derivative is +∞ at S=0 (the land-fill salinity) and NaN for
    S<0, and ``NaN·mask`` does NOT mask NaN out (the global_4deg day-5
    blowup: land-column NaN advected into wet columns by the TKE superbee
    advection). Physically S ≥ O(1) PSU in every wet cell, so the floor
    only guards the autodiff evaluation point on land / unphysical
    transients; the surface FLUXES are not modified. [PSU]
    """
    from legoesm.ocean.eos import eos_density_derivatives
    S_eval = jnp.maximum(S_sfc, jnp.asarray(eos_salinity_floor,
                                            dtype=S_sfc.dtype))
    drho_dT, drho_dS = eos_density_derivatives(eos_fn, T_sfc, S_eval, p_sfc)
    forc_rho = (drho_dT * forc_temp_surface.astype(T_sfc.dtype)
                + drho_dS * forc_salt_surface.astype(T_sfc.dtype))
    return -(g / rho_0) * forc_rho


def realized_implicit_friction_dissipation(
    u_old: jnp.ndarray,
    u_new: jnp.ndarray,
    A_v_face: jnp.ndarray,
    dz_half: jnp.ndarray,
) -> jnp.ndarray:
    r"""Realized implicit vertical-friction dissipation at interfaces.

    Veros ``implicit_vert_friction`` diagnoses the K_diss_v contribution as
    the product of the POST-solve friction flux with the PRE-solve shear
    (friction.py:131-151)::

        flux[k] = κ_f[k] · (u_new[k] − u_new[k+1]) / dzw[k]
        diss[k] = (u_old[k] − u_old[k+1]) · flux[k] / dzw[k]
                = κ_f[k] · g_new[k] · g_old[k]            [m²/s³]

    (legoESM top-down indexing; the implicit solve damps the shear so
    ``g_new·g_old ≥ 0`` generically — Veros applies no clamp, neither does
    this). Grid-agnostic on the last axis: call once per velocity component
    on its own face stagger with the SAME ``A_v_face``/``dz_half`` the
    friction solve used, then average faces→centres at the caller.

    Parameters
    ----------
    u_old, u_new : (..., nlev) — pre-/post-solve velocity on one stagger.
    A_v_face : (..., nlev-1) — the interface viscosity the solve used.
    dz_half : (..., nlev-1) — centre-spacing metric of the solve.

    Returns
    -------
    diss : (..., nlev-1) at the interior interfaces [m²/s³].
    """
    dz_safe = jnp.maximum(dz_half, _EPS)
    g_old = (u_old[..., :-1] - u_old[..., 1:]) / dz_safe
    g_new = (u_new[..., :-1] - u_new[..., 1:]) / dz_safe
    return A_v_face * g_new * g_old


def tke_integrate_post_mixing(
    ctx: TKEPostMixingContext,
    N2_post: jnp.ndarray,
    K_diss_v: jnp.ndarray,
    P_diss_v_surface: jnp.ndarray,
    dt: float,
    cfg: TKEConfig,
    external_source: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Veros ``integrate_tke`` (tke.py:116-244) on the POST-MIXING state.

    Solves ONE backward-Euler TKE step charging

    .. math::

        \text{forc} = K_{diss,v} - P_{diss,v}, \qquad
        P_{diss,v} = \kappa_H\,N^2_{\text{post}}

    with ``κ_H`` from the CARRIED TKE (``ctx.K_H_old``; Veros kappaH from
    tke[tau]) and ``N²_post`` the POST-tracer-mixing stratification
    (thermodynamics.py:385) — plus the surface buoyancy-flux ``P_diss_v``
    slot at an INTERNAL surface-W row (thermodynamics.py:386-388).

    The tridiagonal is the Veros assembly (tke.py:185-227) over ``nlev``
    W rows top-down: row 0 = the surface half-volume W point (volume
    ``ctx.dz_surface`` = 0.5·dzw_top; Veros's last W level), rows
    ``1..nlev-1`` = the carried interior interfaces (volumes ``ctx.dz_half``
    = Veros dzw), face gradients over the intervening CELL thickness
    ``ctx.dz_cell`` (Veros dzt). The surface row is NOT carried across steps
    (legoESM's prognostic TKE lives on the nlev−1 interior interfaces): it
    is seeded from the topmost interior value each step — the SAME seeding
    the banked term-level oracle probe uses
    (.physics-validator/tke_metric_fix/probe_tke_term_level_postfix.py) —
    and its solved value is returned only through the implicit coupling.
    Veros's surface-only positivity clamp (tke.py:238-244) therefore acts on
    a discarded row; the interior may go negative (the energy debt of
    ``positivity="veros_surface_correction"``), exactly as in Veros.

    Parameters
    ----------
    ctx : TKEPostMixingContext — phase-1 ingredients from
        :func:`tke_set_diffusivities` (kappa/mxl/sqrttke at tau).
    N2_post : (..., nlev-1) — SIGNED adiabatic N² from the MIXED T/S over
        the dzw slot (the caller recomputes it after the tracer solve).
    K_diss_v : (..., nlev-1) — shear-production forcing: ``K_M_old·S²``
        (``shear_production="pre_solve"``) or the realized implicit-friction
        dissipation (``"realized_veros"``), at cell-centre interfaces.
    P_diss_v_surface : (...) — the surface buoyancy-flux slot from
        :func:`compute_surface_buoyancy_P_diss_v` (zero array when no
        surface forcing).
    dt : float — the TKE timestep (= dt_mom, Veros tke.py:137).
    cfg : TKEConfig.
    external_source : (..., nlev-1) or None — energy-recycling ``forc``
        terms (eke_diss_iw + K_diss_bot) at the interior interfaces.

    Returns
    -------
    tke_new : (..., nlev-1) — the carried interior interfaces (signed; no
        interior floor).
    """
    _validate_post_mixing_cfg(cfg)
    e_old = ctx.tke_old
    n_int = e_old.shape[-1]          # nlev - 1 interior interfaces
    n_w = n_int + 1                  # + the internal surface W row
    dtype = e_old.dtype

    # --- forc (Veros tke.py:142): interior rows ---
    forc_int = K_diss_v.astype(dtype) - ctx.K_H_old * N2_post.astype(dtype)
    if external_source is not None:
        forc_int = forc_int + external_source.astype(dtype)
    # Surface W row: K_diss_v ≡ 0 there (friction diss[..., surface] = 0,
    # friction.py:149) − the surface buoyancy-flux P_diss_v slot. The
    # interior-shaped external_source has no surface entry (legoESM's
    # eke_diss/K_diss_bot live on the carried interfaces) — documented.
    forc_srf = -P_diss_v_surface.astype(dtype)
    forc_w = jnp.concatenate([forc_srf[..., jnp.newaxis], forc_int], axis=-1)

    # --- W-row state/closure arrays (surface row seeded by top-interior copy,
    #     matching the banked oracle probe; Veros carries a real value) ---
    def _w(x):
        return jnp.concatenate([x[..., :1], x], axis=-1)

    e_w = _w(e_old)
    kM_w = _w(ctx.K_M_old)
    sqrttke_w = _w(ctx.sqrttke)
    # Dissipation length: l_eps = sqrt(lup*ldown) (NEMO zmxld -> dissl,
    # zdftke:672/735), NOT the eddy-coefficient l_k = min(lup,ldown). The two
    # coincide for tke_mxl_choice 1/2; for choice 3 using l_k here would
    # OVER-dissipate exactly in the winter ML (anchored lup small, bottom-
    # grown ldown large). None => older ctx / equal-length fallback.
    _l_diss = ctx.l_eps if getattr(ctx, "l_eps", None) is not None else ctx.mxl
    mxl_w = _w(_l_diss)

    # --- Veros tridiagonal assembly (tke.py:185-227), top-down ---
    # delta[w] couples W rows w and w+1 through cell w (thickness dzt[w]·J):
    # surface W (z=0) and interface 0 sandwich cell 0, generally cell w.
    dz_face = jnp.maximum(ctx.dz_cell[..., :n_w - 1], _EPS)   # (..., n_w-1)
    delta = (dt * cfg.alpha_tke * 0.5
             * (kM_w[..., :-1] + kM_w[..., 1:]) / dz_face)
    vol = jnp.concatenate(
        [jnp.maximum(ctx.dz_surface, _EPS)[..., jnp.newaxis],
         jnp.maximum(ctx.dz_half[..., :n_int], _EPS)], axis=-1)
    a = jnp.concatenate(
        [jnp.zeros_like(delta[..., :1]), -delta / vol[..., 1:]], axis=-1)
    c = jnp.concatenate(
        [-delta / vol[..., :n_w - 1], jnp.zeros_like(delta[..., :1])],
        axis=-1)
    _diss_w = cfg.c_eps * sqrttke_w / jnp.maximum(
        mxl_w, _mixing_length_floor(cfg))
    _disc = getattr(cfg, "dissipation_discretization", "backward_euler")
    if _disc == "nemo_1p5_split":
        # NEMO zdftke semi-implicit dissipation split (zdftke.F90:241-242,
        # 414, 419): 1.5x the linearized dissipation on the diagonal
        # (zfact2 = 1.5*rn_Dt*rn_ediss) and +0.5x added back EXPLICITLY to
        # the RHS (zfact3 = 0.5*rn_ediss), both linearized at the CARRIED
        # sqrt(e)/l_eps. Net first-order dissipation identical; the discrete
        # decay factor differs from plain backward-Euler at large dt*diss
        # (NEMO: (1+0.5a)/(1+1.5a) -> 1/3; backward-Euler: 1/(1+a) -> 0).
        b = 1.0 - (a + c) + 1.5 * dt * _diss_w  # coeff-ok: NEMO zdftke semi-implicit split zfact2=1.5*rn_Dt*rn_ediss (zdftke.F90:241)
        forc_w = forc_w + 0.5 * _diss_w * e_w
    elif _disc == "backward_euler":
        b = 1.0 - (a + c) + dt * _diss_w
    else:
        raise ValueError(
            "Unknown TKEConfig.dissipation_discretization: must be one of "
            f"('backward_euler', 'nemo_1p5_split'), got {_disc!r}")

    if getattr(ctx, "langmuir_source", None) is not None:
        # NEMO ln_lc: en += rDt * source BEFORE the implicit solve
        # (zdftke.F90:367). The source lives on the interior interfaces;
        # pad the (discarded) surface W row with zero.
        _lc_w = jnp.concatenate(
            [jnp.zeros_like(ctx.langmuir_source[..., :1]),
             ctx.langmuir_source], axis=-1).astype(dtype)
        forc_w = forc_w + _lc_w
    d = e_w + dt * forc_w
    if getattr(ctx, "surface_dirichlet", None) is not None:
        # NEMO nn_bc_surf=1: hold the surface W row at
        # en(1)=max(rn_emin0, rn_ebb*|tau|/rho0) (identity row; the interior
        # row 1 couples to the held value through a[...,1] — the standard
        # Dirichlet tridiagonal). No wind-work flux injection in this mode.
        b = b.at[..., 0].set(1.0)
        c = c.at[..., 0].set(0.0)
        d = d.at[..., 0].set(ctx.surface_dirichlet.astype(dtype))
    else:
        # Wind-work surface injection over the surface half-volume (tke.py:225).
        d = d.at[..., 0].add(dt * ctx.surface_flux.astype(dtype) / vol[..., 0])

    e_new_w = _tridiag_thomas(a, b, c, d)
    # Veros surface clamp (tke.py:238-244) acts on the (discarded) surface
    # row; the carried interior stays UNFLOORED (energy debt allowed).
    return e_new_w[..., 1:]


__all__ = (
    "TKEOutput",
    "TKEPostMixingContext",
    "compute_K_from_tke",
    "compute_mixing_lengths",
    "compute_surface_buoyancy_P_diss_v",
    "nemo_tke_effective_ice_fraction",
    "realized_implicit_friction_dissipation",
    "tke_integrate_post_mixing",
    "tke_set_diffusivities",
    "tke_vertical_mixing",
)


def _nemo_literal_rhs_materialized(
    en, dt, p_sh2, p_avt, rn2, diss_rate, dissl, rn_ediss, wmask, mode: str,
    *, intermediate: str = "",
):
    """Private full-step discriminator for compiled ``zdftke`` RHS order.

    Each named arm adds one IEEE-identity source boundary to the expression
    at ``R101TKEW/BLD/ppsrc/nemo/zdftke.f90:439-442``.  The selector is
    reachable only through :class:`_NEMOWSRK3TestHooks`; it is deliberately
    not a physics configuration.  ``all`` is the preregistered fallback that
    materializes every listed boundary.  The default production path never
    calls this helper.
    """
    from legoesm.core.source_rounding import nemo_source_round

    valid = {
        "p_avt_rn2", "zfact3_dissl", "dissipation_product",
        "after_stratification", "parenthesized_sum", "dt_product",
        "masked_increment", "all", "nemo_dissipation_tree",
        "nemo_dissipation_tree_materialized",
    }
    intermediates = {
        "p_avt_operand", "rn2_operand",
        "p_avt_rn2", "zfact3_dissl", "dissipation_product",
        "after_stratification", "parenthesized_sum", "dt_product",
        "masked_increment", "final_accumulation",
    }
    if intermediate:
        if mode:
            raise ValueError(
                "private TKE RHS materialization and intermediate selectors "
                "are mutually exclusive")
        if intermediate not in intermediates:
            raise ValueError(
                "unknown private TKE RHS intermediate: "
                f"{intermediate!r}; expected one of {sorted(intermediates)}")
        zfact3 = 0.5 * jnp.asarray(rn_ediss, dtype=en.dtype)
        p_avt_rn2 = p_avt * rn2
        zfact3_dissl = zfact3 * jnp.asarray(dissl, dtype=en.dtype)
        dissipation = zfact3_dissl * en
        stratified = p_sh2 - p_avt_rn2
        parenthesized = stratified + dissipation
        scaled = dt * parenthesized
        increment = scaled * jnp.asarray(wmask, dtype=en.dtype)
        final = en + increment
        values = {
            "p_avt_operand": p_avt,
            "rn2_operand": rn2,
            "p_avt_rn2": p_avt_rn2,
            "zfact3_dissl": zfact3_dissl,
            "dissipation_product": dissipation,
            "after_stratification": stratified,
            "parenthesized_sum": parenthesized,
            "dt_product": scaled,
            "masked_increment": increment,
            "final_accumulation": final,
        }
        return final, values[intermediate]
    if mode not in valid:
        raise ValueError(
            "unknown private TKE RHS materialization boundary: "
            f"{mode!r}; expected one of {sorted(valid)}")

    def boundary(name, value):
        return nemo_source_round(value) if mode in (name, "all") else value

    if mode in (
        "nemo_dissipation_tree", "nemo_dissipation_tree_materialized",
    ):
        materialize = mode == "nemo_dissipation_tree_materialized"
        zfact3 = 0.5 * jnp.asarray(rn_ediss, dtype=en.dtype)
        if materialize:
            zfact3 = nemo_source_round(zfact3)
        zfact3_dissl = zfact3 * jnp.asarray(dissl, dtype=en.dtype)
        if materialize:
            zfact3_dissl = nemo_source_round(zfact3_dissl)
        dissipation = zfact3_dissl * en
        if materialize:
            dissipation = nemo_source_round(dissipation)
        return en + dt * (
            p_sh2 - p_avt * rn2 + dissipation
        ) * jnp.asarray(wmask, dtype=en.dtype)

    p_avt_rn2 = boundary("p_avt_rn2", p_avt * rn2)
    zfact3_dissl = boundary("zfact3_dissl", 0.5 * diss_rate)
    dissipation = boundary(
        "dissipation_product", zfact3_dissl * en)
    stratified = boundary("after_stratification", p_sh2 - p_avt_rn2)
    parenthesized = boundary(
        "parenthesized_sum", stratified + dissipation)
    scaled = boundary("dt_product", dt * parenthesized)
    increment = boundary(
        "masked_increment", scaled * jnp.asarray(wmask, dtype=en.dtype))
    return en + increment
