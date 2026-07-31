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

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
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
_NEMO_MXL0_LENGTH_SCALE = 2.0e5  # zraug numerator [m*kg/(m*s^2)^-1... NEMO zdftke:575]


def _mxl0_surface_anchor(cfg: "TKEConfig", taum, rho_0: float, g: float):
    """ln_mxl0 surface mixing-length anchor (single owner; zdftke:575+602):
    l_sfc = max(rn_mxl0, vkarmn*2e5/(rho0*g)*taum). None unless the choice is a
    NEMO nn_mxl scheme (3 = nn_mxl=3, 4 = nn_mxl=2); ORCA1 sets ln_mxl0=.true.
    independently of nn_mxl, so BOTH need the anchor."""
    if cfg.tke_mxl_choice not in (3, 4):
        return None
    return jnp.maximum(
        jnp.asarray(cfg.mxl0_min_m),
        _NEMO_MXL0_VKARMN * _NEMO_MXL0_LENGTH_SCALE / (rho_0 * g)
        * jnp.maximum(taum, 0.0))
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
            jnp.asarray(_NEMO_TKE_EMIN0, dtype=taum.dtype),
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
    single-owner doctrine, no re-derived drag coefficient) and ``u_bot``/
    ``v_bot`` the bottom-cell T-point velocity components. ``rn_emin`` is
    ``cfg.tke_background`` (legoESM's interior TKE floor — same value as
    NEMO's namelist default 1e-6 m²/s²).

    Parameters
    ----------
    r_bottom_drag : (...,) — bottom-drag rate at T-points [m/s], >= 0.
    u_bot, v_bot : (...,) — T-point bottom-cell velocity components [m/s].
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

    Returns
    -------
    l_k, l_eps : (..., nlev-1) — for use in K = c_k·l_k·sqrt(2e) and
        eps = c_eps·e^{3/2} / l_eps respectively.
    """
    if cfg.tke_mxl_choice == 2:
        if signed_n2:
            # Veros mxl_choice=2: a single length used for BOTH K_M
            # (l_k) and dissipation (l_eps), as in veros/core/tke.py.
            l_buoy = _veros_buoyancy_length(
                e, N2, dz_half, cfg.mxl_min, dz_cell=dz_cell)
            return l_buoy, l_buoy
        l_up, l_dn = _bougeault_lacarrere_lengths(
            e, N2, dz_half, cfg.mxl_min,
        )
        l_k = jnp.sqrt(jnp.maximum(l_up * l_dn, cfg.mxl_min ** 2))
        l_eps = jnp.maximum(l_up, l_dn)
    elif cfg.tke_mxl_choice in (3, 4):
        # --- NEMO nn_mxl=3 (choice 3) / nn_mxl=2 (choice 4) + ln_mxl0 ---
        # (zdftke.F90:575, 588-614, 658-690).  Both share the lup/ldown sweeps;
        # they differ ONLY in the final l_eps (see below).
        if dz_cell is None:
            raise ValueError(
                f"tke_mxl_choice={cfg.tke_mxl_choice} (NEMO nn_mxl) requires "
                "dz_cell (the e3t cell thicknesses) for the |dl/dz|<=e3t "
                "bounding sweeps.")
        # buoyancy length sqrt(2e)/N at interior interfaces, AD-safe at the
        # negative-TKE debt (same double-where idiom as choice 1/2).
        sqrt2e = jnp.sqrt(2.0) * jnp.where(
            e > 0.0, jnp.sqrt(jnp.where(e > 0.0, e, 1.0)), 0.0)
        N_safe = jnp.sqrt(jnp.maximum(N2, 1.0e-12))
        l_int = jnp.maximum(sqrt2e / N_safe, cfg.mxl_min)     # (..., nlev-1)
        # ln_mxl0 surface anchor l_sfc = max(rn_mxl0, vkarmn*2e5/(rho0*g)*taum)
        # (zdftke:575+602), computed by the CALLER (which owns taum/rho_0/g)
        # and passed via l_surface_anchor; None => the rn_mxl0 floor (windless).
        if l_surface_anchor is not None:
            l_sfc = jnp.asarray(l_surface_anchor, dtype=l_int.dtype)
        else:
            l_sfc = jnp.full(l_int.shape[:-1], cfg.mxl0_min_m,
                             dtype=l_int.dtype)
        # W-row stack: surface anchor + interior interfaces
        l_w = jnp.concatenate([l_sfc[..., None], l_int], axis=-1)  # (..., nlev)
        e3t = dz_cell                                              # (..., nlev)
        # lup: downward scan  l(k) = min(l(k-1) + e3t(k-1), l(k))
        def _down(carry, xs):
            l_km1 = carry
            l_k_, e3_km1 = xs
            out = jnp.minimum(l_km1 + e3_km1, l_k_)
            return out, out
        lT = jnp.moveaxis(l_w, -1, 0)                              # (nlev, ...)
        e3T = jnp.moveaxis(e3t, -1, 0)
        _, lup_rest = jax.lax.scan(_down, lT[0], (lT[1:], e3T[:-1]))
        lup = jnp.concatenate([lT[:1], lup_rest], axis=0)
        # ldown: upward scan  l(k) = min(l(k+1) + e3t(k+1), l(k))
        _, ldn_rest = jax.lax.scan(
            _down, lT[-1], (lT[:-1][::-1], e3T[1:][::-1]))
        ldn = jnp.concatenate([lT[-1:], ldn_rest], axis=0)[::-1]
        lup = jnp.moveaxis(lup, 0, -1)[..., 1:]                    # interior
        ldn = jnp.moveaxis(ldn, 0, -1)[..., 1:]
        l_k = jnp.maximum(jnp.minimum(lup, ldn), cfg.mxl_min)
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
            # SMALLER dissipation length, hence LARGER eps =
            # c_eps*e^{3/2}/l_eps.
            #
            # PRECISION (codex): this does NOT directly "mix less".  At a FIXED
            # TKE the eddy coefficients avm/avt use l_k, which is IDENTICAL in
            # choices 3 and 4 (it is computed before the split), so the
            # instantaneous diffusivity is unchanged.  What changes directly is
            # the DISSIPATION; a shallower mixed layer is a subsequent coupled
            # effect once the larger eps has drawn TKE down, not an algebraic
            # consequence of the branch.  The two coincide where lup ~ ldown
            # (strong stratification, both branches locally limited) and differ
            # most where they diverge (weakly stratified deep columns far from
            # both boundaries) -- which is why this is a HIGH-LATITUDE-selective
            # lever, not a global one.  ORCA1's namelist runs nn_mxl=2.
            l_eps = l_k
        else:
            l_eps = jnp.maximum(jnp.sqrt(lup * ldn), cfg.mxl_min)
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
        l_k = jnp.maximum(l_k, cfg.mxl_min)
        l_eps = l_k
    else:
        raise ValueError(
            f"Unknown tke_mxl_choice={cfg.tke_mxl_choice!r}; expected 1 or 2 "
            "(Veros), 3 (NEMO nn_mxl=3) or 4 (NEMO nn_mxl=2)."
        )
    return l_k, l_eps


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
    dz_cell: jnp.ndarray | None = None,
    dz_surface: jnp.ndarray | None = None,
    surface_dirichlet: jnp.ndarray | None = None,
    surface_bc_level: str = "interior_pinned",
    bottom_dirichlet: jnp.ndarray | None = None,
    K_M_surface: jnp.ndarray | None = None,
    bottom_level: jnp.ndarray | None = None,
) -> jnp.ndarray:
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
        and ``dz_surface`` (the z=0-to-first-cell-centre distance, the
        surface row's control volume / face spacing — same slot
        ``-z_full_ref[0]·J`` used elsewhere). The surface face is
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

    Returns
    -------
    e_new : (..., nlev-1)
    """
    if dz_cell is not None and dz_surface is None:
        raise ValueError(
            "_solve_tke_backward_euler: dz_cell requires dz_surface too "
            "(TKEConfig.veros_dz_slots); got dz_cell=set, dz_surface=None."
        )
    # dz_surface MAY be passed alone (dz_cell=None) for
    # surface_bc_level="nemo_z0" without veros_dz_slots — the virtual
    # surface row only needs the z=0-to-interface-0 distance, not the full
    # Veros metric-slot machinery dz_cell gates. veros_slots (the metric-
    # slot feature) is keyed off dz_cell alone, unaffected.
    veros_slots = dz_cell is not None
    N = e_old.shape[-1]   # number of interfaces
    # Dtype hygiene: surface_flux / external_source can promote to f64 (tau or
    # the EKE-diss source built at default precision) while e_old runs at the
    # storage policy's f32 — cast them down so the tridiagonal RHS scatter does
    # not raise the JAX implicit-downcast FutureWarning. No numeric change when
    # dtypes already match (the default path).
    surface_flux = surface_flux.astype(e_old.dtype)
    if external_source is not None:
        external_source = external_source.astype(e_old.dtype)
    positivity = getattr(cfg, "positivity", "floor")
    if positivity not in ("floor", "veros_surface_correction"):
        raise ValueError(
            f"Unknown TKEConfig.positivity={positivity!r}; expected 'floor' "
            f"or 'veros_surface_correction'."
        )
    veros_positivity = positivity == "veros_surface_correction"
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
    # Linearised dissipation rate (per unit e_new):
    diss_rate = cfg.c_eps * e_sqrt / jnp.maximum(l_eps, cfg.mxl_min)
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

    if veros_slots and N >= 2:
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
    if _disc == "nemo_1p5_split":
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
    rhs = e_old + dt * (P_s + buoy_source)
    if _disc == "nemo_1p5_split":
        # zfact3·dissl·en explicit add-back (NEMO zdftke.F90:419).
        rhs = rhs + dt * 0.5 * diss_rate * e_old
    if external_source is not None:
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
        if dz_surface is None:
            raise ValueError(
                "TKEConfig.tke_surface_bc_level='nemo_z0' requires "
                "dz_surface (the z=0-to-first-cell-centre distance).")
        e_sfc = jnp.asarray(surface_dirichlet, dtype=e_old.dtype)
        dz_face_sfc = jnp.maximum(
            jnp.asarray(dz_surface, dtype=e_old.dtype), _EPS)
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
        K_face_sfc = cfg.alpha_tke * 0.5 * (K_M_old[..., 0] + _avm1)
        delta_sfc = dt * K_face_sfc / dz_face_sfc      # (...,) flux coeff.
        row0_coupling = delta_sfc / vol0               # a[row 0] magnitude
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
        b0 = diag[..., 0] + row0_coupling

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
    e_new = jnp.maximum(e_new, cfg.tke_background)
    # Floor at surface_min on the topmost interface only.
    e_new = e_new.at[..., 0].set(
        jnp.maximum(e_new[..., 0], cfg.tke_surface_min),
    )
    return e_new


# ---------------------------------------------------------------------------
# K_M, K_H from TKE
# ---------------------------------------------------------------------------


def _prandtl_number(
    N2: jnp.ndarray,
    shear_sq: jnp.ndarray,
    kappaM: jnp.ndarray,
    cfg: TKEConfig,
) -> jnp.ndarray:
    r"""Turbulent Prandtl number for the K_H = K_M / Pr relation.

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
    - ``prandtl_mode="nemo_ri"`` (Phase-2 #1317 T8, fixed #1226 item 11):
      NEMO's EXACT nn_pdl=1 form (zdftke.F90:381-401):
      ``zri = rn2b·p_avm / (p_sh2 + rn_bshear)``, ``pdlr = max(0.1,
      ri_cri/max(ri_cri,zri))``, ``Pr = 1/pdlr`` — i.e.
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
      80-94) has no face-averaging analog — same documented simplification
      as :func:`legoesm.ocean.physics.vertical_mixing._shared.
      vertical_shear_burchard` — so the faithful transcription forms
      ``p_sh2 ≈ kappaM·shear_sq`` (the AVM-WEIGHTED shear, matching units
      [m²/s³]) before adding ``bshear_floor`` (now in the SAME m²/s³ units
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
        # face-averages OLD avm onto the shear product; legoESM's single
        # per-interface K_M has no face-avg analog, so kappaM*shear_sq is
        # the faithful cell-centred transcription (== P_s_curr at the
        # call site, tke.py:1932).
        p_sh2 = kappaM * shear_sq
        # NB with NEMO's default rn_bshear = 1e-20 the kappaM factors cancel
        # almost everywhere (bshear is ~9 decades below kappaM*shear_sq in any
        # realistic regime), so zri ~= N2/shear_sq: nemo_ri is then
        # NEAR-DEGENERATE with "richardson" up to the ri_cri-vs-6.6 scaling.
        # The weighted form matters only where the floor competes (kappaM or
        # shear ~ 0) — keep it for faithfulness, but do not expect materially
        # different production behaviour (review of 4aeeb867d, #1226).
        zri = N2 * kappaM / jnp.maximum(p_sh2 + bshear, 1e-30)
        return jnp.maximum(1.0, jnp.minimum(10.0, cfg.prandtl_ri_coeff * zri))
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
        Pr = _prandtl_number(_N2_pr, shear_sq, K_M, cfg)
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


def nemo_langmuir_tke_source(
    taum: jnp.ndarray,
    N2: jnp.ndarray,
    depth_w: jnp.ndarray,
    dz_w: jnp.ndarray,
    cfg: TKEConfig,
    ice_frac: jnp.ndarray | None = None,
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

    Returns
    -------
    (..., nlev-1) TKE source, feedable as ``external_source`` (the RHS gets
    ``+dt·source``, exactly NEMO's ``en += rn_Dt·source``).
    """
    half_wlc2 = _NEMO_TKE_LC_CSD * jnp.maximum(taum, 0.0)         # ½W_lc² ≥ 0
    # Axell Eq. 47 LHS: cumulative PE from the surface down (z positive down).
    pe = jnp.cumsum(jnp.maximum(N2, 0.0) * depth_w * dz_w, axis=-1)
    # h_lc: first (shallowest) interface where PE exceeds ½W_lc²; fall back to
    # the deepest interface when the whole column's PE is below it (NEMO
    # initialises imlc to the bottom).  argmax(mask) returns the FIRST True.
    exceeded = pe > half_wlc2[..., None]
    first = jnp.argmax(exceeded, axis=-1)
    depth_b = jnp.broadcast_to(depth_w, pe.shape)
    h_first = jnp.take_along_axis(depth_b, first[..., None], axis=-1)[..., 0]
    h_lc = jnp.where(jnp.any(exceeded, axis=-1), h_first, depth_b[..., -1])
    h_lc = jnp.maximum(h_lc, _EPS)
    # u_s³ = (2·½W_lc²)^{3/2}; d/dx x^{3/2} → 0 at 0, AD-safe without a guard.
    us3 = (2.0 * half_wlc2) ** 1.5
    if ice_frac is not None:
        us3 = us3 * jnp.maximum(0.0, 1.0 - ice_frac)
    w_lc = cfg.lc_coeff * jnp.sin(jnp.pi * depth_b / h_lc[..., None])
    src = us3[..., None] * (w_lc ** 3) / h_lc[..., None]
    return jnp.where(depth_b < h_lc[..., None], src, 0.0)


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
        htau = jnp.maximum(
            _NEMO_TKE_HTAU_MIN_M,
            jnp.minimum(
                _NEMO_TKE_HTAU_MAX_M,
                _NEMO_TKE_HTAU_SLOPE_M
                * jnp.abs(jnp.sin(jnp.deg2rad(lat_deg)))),
        )[..., None]
    else:
        raise ValueError(
            f"Unknown TKEConfig.etau_htau_mode={htau_mode!r}; expected "
            f"'constant10m' or 'latitude' (NEMO nn_htau 0/1).")
    e_sfc = jnp.maximum(_NEMO_TKE_EMIN0, _NEMO_TKE_EBB / rho_0
                        * jnp.maximum(taum, 0.0))
    inj = cfg.etau_frac * e_sfc[..., None] * jnp.exp(-depth_w / htau)
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
    ice_frac: jnp.ndarray | None = None,
    bottom_dirichlet: jnp.ndarray | None = None,
    bottom_level: jnp.ndarray | None = None,
    T_n2b: jnp.ndarray | None = None,
    S_n2b: jnp.ndarray | None = None,
    u_before_cell: jnp.ndarray | None = None,
    v_before_cell: jnp.ndarray | None = None,
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
    if bottom_dirichlet is None and bottom_level is not None:
        raise ValueError(
            "bottom_level was passed but bottom_dirichlet is None — "
            "bottom_level only selects WHERE the bottom Dirichlet pin lands, "
            "it does not supply one (silent-no-op guard).")
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
    if _surf_bc_level == "nemo_z0" and dz_surface is None:
        raise ValueError(
            "TKEConfig.tke_surface_bc_level='nemo_z0' requires dz_surface "
            "(= -z_full_ref[0]·J, the z=0-to-first-cell-centre distance) to "
            "be passed to tke_vertical_mixing."
        )

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

    # Shear-production discretization (T4, Phase-2 #1317): "squared_centered"
    # (default, BIT-IDENTICAL) is the now-only squared form; "nemo_burchard"
    # is the Burchard (2002) now×before energy-conserving cross term, which
    # NEMO feeds to BOTH the shear-production source AND the Prandtl zri
    # (zdftke.F90:392-395 reads the SAME p_sh2) — so ``shear_sq`` below feeds
    # both consumers identically to NEMO either way.
    _shear_disc = getattr(cfg, "tke_shear_production", "squared_centered")
    if _shear_disc not in ("squared_centered", "nemo_burchard"):
        raise ValueError(
            "Unknown TKEConfig.tke_shear_production: must be one of "
            f"('squared_centered', 'nemo_burchard'), got {_shear_disc!r}.")
    if _shear_disc == "nemo_burchard":
        if u_before_cell is None or v_before_cell is None:
            raise ValueError(
                "TKEConfig.tke_shear_production='nemo_burchard' requires "
                "u_before_cell and v_before_cell (the carried leap-frog "
                "before-velocities, state.u_before/v_before) to be passed "
                "to tke_vertical_mixing.")
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_burchard as _vertical_shear_burchard,
        )
        shear_sq = _vertical_shear_burchard(
            u_cell, v_cell, u_before_cell, v_before_cell, dz_half)
    else:
        if u_before_cell is not None or v_before_cell is not None:
            raise ValueError(
                "u_before_cell/v_before_cell were passed but "
                "TKEConfig.tke_shear_production='squared_centered' — set "
                "tke_shear_production='nemo_burchard' to actually use them "
                "(silent-no-op guard).")
        shear_sq = _vertical_shear_squared(u_cell, v_cell, dz_half)

    # Static stability N^2. ``"insitu"`` (default) is the clipped in-situ
    # form (BIT-IDENTICAL); ``"adiabatic"`` is the SIGNED Veros parcel-
    # displacement form that lets the TKE convect (N^2 < 0).
    signed_n2 = cfg.n2_mode in ("adiabatic", "nemo_bn2")
    # Diffusivity-stage N² time level (TKEConfig.n2_before_advection): the
    # before-advection (Nnow) T/S override, when supplied by the caller (see
    # tke_set_diffusivities). Python-static; None ⇒ BIT-IDENTICAL.
    _Tn2 = T_cell if T_n2 is None else T_n2
    _Sn2 = S_cell if S_n2 is None else S_n2
    N2 = _compute_N2(
        rho_cell, dz_half, rho_0, g,
        T_cell=_Tn2, S_cell=_Sn2, p_cell=p_cell,
        dz_ref=dz_ref, jacobian=jacobian, eos_fn=eos_fn,
        n2_mode=cfg.n2_mode,
        adiabatic_over_dz_half=veros_slots,
        t_depth=t_depth, w_depth=w_depth,
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
    if T_n2b is not None or S_n2b is not None:
        if T_n2b is None or S_n2b is None:
            raise ValueError(
                "tke_vertical_mixing: T_n2b and S_n2b must be supplied "
                "together (rn2b needs both tracers).")
        N2b = _compute_N2(
            rho_cell, dz_half, rho_0, g,
            T_cell=T_n2b, S_cell=S_n2b, p_cell=p_cell,
            dz_ref=dz_ref, jacobian=jacobian, eos_fn=eos_fn,
            n2_mode=cfg.n2_mode,
            adiabatic_over_dz_half=veros_slots,
            t_depth=t_depth, w_depth=w_depth,
        )
    else:
        N2b = N2

    if taum_surface is not None:
        # NEMO taum channel: the caller supplies the surface stress MODULUS
        # directly (e.g. the DINO usrdef x1.3 westerly boost, which enters
        # the TKE input but NOT the momentum stress).
        taum = jnp.maximum(jnp.asarray(taum_surface), 0.0)
        surface_flux = (taum / rho_0) ** 1.5
    elif tau_x_surface is None and tau_y_surface is None:
        surface_flux = jnp.zeros(rho_cell.shape[:-1], dtype=rho_cell.dtype)
        taum = surface_flux  # |τ| = 0 (unforced)
    else:
        tx = tau_x_surface if tau_x_surface is not None else jnp.zeros_like(rho_cell[..., 0])
        ty = tau_y_surface if tau_y_surface is not None else jnp.zeros_like(rho_cell[..., 0])
        taum = _safe_stress_modulus(tx, ty)
        surface_flux = (taum / rho_0) ** 1.5

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
        _depth_w = -z_interface
    if _lc_on:
        # Langmuir source enters the RHS as +dt·source — exactly NEMO's
        # pre-solve ``en += rn_Dt·source`` (zdftke.F90:367). NEMO's PE
        # integral (:340,344) reads ``rn2b`` (the BEFORE/Nbb level) — N2b
        # (defaults to N2 when T_n2b/S_n2b are not supplied, T8/T13).
        _lc_src = nemo_langmuir_tke_source(taum, N2b, _depth_w, dz_half, cfg,
                                           ice_frac=ice_frac)
        external_source = (_lc_src if external_source is None
                           else external_source + _lc_src)

    # Sub-iteration loop (Mode B convergence; Mode A uses n_iterations=1).
    tke_curr = tke_old
    # NEMO ln_mxl0 anchor for nn_mxl=3: l_sfc = max(rn_mxl0, vkarmn*2e5/(rho0*g)*taum)
    _l_anchor = _mxl0_surface_anchor(cfg, taum, rho_0, g)
    # T3-exact: NEMO's TRUE surface-w-level viscosity avm(jk=1)
    # (:func:`nemo_surface_avm`), consulted only by the nemo_z0 face
    # assembly. Requires the ln_mxl0 anchor (tke_mxl_choice=3) and a held
    # surface value (surface_dirichlet) — both already validated above when
    # surface_bc_level="nemo_z0" is selected. None otherwise ⇒ the
    # documented approximation, bit-identical to the prior behaviour.
    _K_M_surface = None
    if _surf_bc_level == "nemo_z0" and _l_anchor is not None:
        _K_M_surface = nemo_surface_avm(cfg, surface_dirichlet, _l_anchor)
    for _ in range(max(1, int(n_iterations))):
        l_k, l_eps = compute_mixing_lengths(
            tke_curr, N2, dz_half, cfg, signed_n2=signed_n2,
            dz_cell=dz_cell_mxl, boundary_cap=boundary_cap,
            l_surface_anchor=_l_anchor)
        K_M_curr, K_H_curr = compute_K_from_tke(
            tke_curr, l_k, cfg, N2=N2, shear_sq=shear_sq,
            z_interface=z_interface, N2_prandtl=N2b)
        P_s_curr = K_M_curr * shear_sq
        tke_curr = _solve_tke_backward_euler(
            e_old=tke_curr,
            K_M_old=K_M_curr, K_H_old=K_H_curr,
            P_s=P_s_curr, N2=N2, l_eps=l_eps,
            dz_half=dz_half,
            surface_flux=surface_flux,
            dt=dt, cfg=cfg,
            external_source=external_source,
            dz_cell=dz_cell,
            dz_surface=(dz_surface if (veros_slots
                                        or _surf_bc_level == "nemo_z0")
                       else None),
            surface_dirichlet=surface_dirichlet,
            surface_bc_level=_surf_bc_level,
            bottom_dirichlet=bottom_dirichlet,
            K_M_surface=_K_M_surface,
            bottom_level=bottom_level,
        )

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
        z_interface=z_interface, N2_prandtl=N2b)

    return TKEOutput(K_M=K_M, K_H=K_H, tke_new=tke_curr, l_eps=l_eps_final)


# ---------------------------------------------------------------------------
# POST-MIXING Veros step order (TKEConfig.buoyancy_timing="post_mixing_veros")
# ---------------------------------------------------------------------------


def _validate_post_mixing_cfg(cfg: TKEConfig) -> None:
    """Fail loudly unless the post-mixing prerequisites hold (see config doc)."""
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
    if _tke_shear not in ("squared_centered", "nemo_burchard"):
        raise ValueError(
            "Unknown TKEConfig.tke_shear_production: must be one of "
            f"('squared_centered', 'nemo_burchard'), got {_tke_shear!r}.")
    if timing == "post_mixing_veros" and _tke_shear == "nemo_burchard":
        raise ValueError(
            "TKEConfig.tke_shear_production='nemo_burchard' is not "
            "supported with buoyancy_timing='post_mixing_veros' — "
            "tke_set_diffusivities does not accept u_before_cell/"
            "v_before_cell and would silently keep squared_centered. "
            "Disable tke_shear_production or use the standard pre_mixing "
            "path.")
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
    ice_frac: jnp.ndarray | None = None,
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
    surface_flux = (taum / rho_0) ** 1.5
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

    _l_anchor = _mxl0_surface_anchor(cfg, taum, rho_0, g)
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
    _diss_w = cfg.c_eps * sqrttke_w / jnp.maximum(mxl_w, cfg.mxl_min)
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
    "realized_implicit_friction_dissipation",
    "tke_integrate_post_mixing",
    "tke_set_diffusivities",
    "tke_vertical_mixing",
)
