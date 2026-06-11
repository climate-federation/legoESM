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

_EPS = float(jnp.finfo(jnp.float32).eps)


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
    l_up_raw = jnp.sqrt(2.0 * e / N2_safe + _EPS)
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


def compute_mixing_lengths(
    e: jnp.ndarray,
    N2: jnp.ndarray,
    dz_half: jnp.ndarray,
    cfg: TKEConfig,
    *,
    signed_n2: bool = False,
    dz_cell: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (l_k, l_eps) for the chosen ``tke_mxl_choice``.

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
    elif cfg.tke_mxl_choice == 1:
        # Simple parabolic / linear length scale.
        # l = max(mxl_min, min(kappa_vk * z, sqrt(2e / max(N2, eps))))
        # We use the closed-form sqrt(2e/N^2) consistent with choice=2
        # but without the symmetric average.
        N2_safe = jnp.maximum(N2, _EPS)
        l_k = jnp.maximum(jnp.sqrt(2.0 * e / N2_safe + _EPS), cfg.mxl_min)
        l_eps = l_k
    else:
        raise ValueError(
            f"Unknown tke_mxl_choice={cfg.tke_mxl_choice!r}; expected 1 or 2."
        )
    return l_k, l_eps


# ---------------------------------------------------------------------------
# Shear, buoyancy, dissipation at interfaces
# ---------------------------------------------------------------------------


def _vertical_shear_squared(
    u_cell: jnp.ndarray, v_cell: jnp.ndarray, dz_half: jnp.ndarray,
) -> jnp.ndarray:
    """Compute ``|du/dz|^2 + |dv/dz|^2`` at interfaces.

    Parameters
    ----------
    u_cell, v_cell : (..., nlev) — cell-centre velocities.
    dz_half : (..., nlev-1) — distance between cell centres.

    Returns
    -------
    S2 : (..., nlev-1) — squared vertical shear at interfaces.
    """
    dz_safe = jnp.maximum(dz_half, _EPS)
    du = (u_cell[..., 1:] - u_cell[..., :-1]) / dz_safe
    dv = (v_cell[..., 1:] - v_cell[..., :-1]) / dz_safe
    return du * du + dv * dv


def _compute_N2(
    rho_cell: jnp.ndarray, dz_half: jnp.ndarray, rho_0: float,
    g: float = constants.g,
    *,
    T_cell: jnp.ndarray | None = None,
    S_cell: jnp.ndarray | None = None,
    p_cell: jnp.ndarray | None = None,
    dz_ref: jnp.ndarray | None = None,
    jacobian: jnp.ndarray | None = None,
    eos_fn=None,
    n2_mode: str = "insitu",
    adiabatic_over_dz_half: bool = False,
) -> jnp.ndarray:
    """N^2 at interfaces.

    ``n2_mode="insitu"`` (default, BIT-IDENTICAL legacy): from cell-centre
    *in-situ* density, ``N^2 = -(g/rho_0) drho/dz`` with z positive upward.
    legoESM's cell index k = 0 is the surface (top) and k = nlev-1 is the
    bottom, so z is decreasing with k. Discretely, between cell centres k
    and k+1 a distance ``dz_half[k]`` apart::

        drho/dz = (rho[k] - rho[k+1]) / dz_half[k]

    For stable stratification this is negative (light water on top), so
    ``N^2 = -g/rho_0 * drho/dz > 0``. The result is **clipped >= 0**: the
    in-situ density difference carries compressibility and is biased too
    stable, so its sign is not a reliable convection trigger — the
    unstable case is handled by a separate convective-adjustment scheme.

    ``n2_mode="adiabatic"``: the **true static stability** via adiabatic
    parcel displacement to the upper cell's pressure (Veros
    thermodynamics.py:99-103), delegating to the shared
    :func:`legoesm.ocean.eos.compute_buoyancy_frequency_adiabatic` (no
    duplicate numerics). The result is **SIGNED** (not clipped) — N^2 < 0
    marks a statically unstable interface, which is exactly the convection
    trigger the TKE closure needs. Requires ``T_cell``, ``S_cell``,
    ``p_cell`` (cell-centre pressure [Pa]), the reference layer thickness
    ``dz_ref`` (shape ``(nlev,)``) and the ``jacobian`` (shape ``(...)``)
    so the shared helper's interface thickness
    ``0.5*(dz_ref*J)[k] + 0.5*(dz_ref*J)[k+1]`` reproduces ``dz_half[k]``
    exactly, plus an ``eos_fn``.

    ``adiabatic_over_dz_half=True`` (the ``TKEConfig.veros_dz_slots``
    metric fix) instead divides the adiabatic density contrast by the
    CALLER'S ``dz_half`` — the Veros ``dzw`` slot (thermodynamics.py:99),
    which on a stretched u_centered coordinate is NOT the midpoint
    interface spacing the shared helper would reconstruct. ``False``
    (default) is BIT-IDENTICAL legacy (midpoint reconstruction).

    Returns
    -------
    N2 : (..., nlev-1). Clipped >= 0 for ``"insitu"``; signed for
        ``"adiabatic"``.
    """
    if n2_mode == "insitu":
        dz_safe = jnp.maximum(dz_half, _EPS)
        # drho/dz with z positive upward — negative for stable stratification.
        drho_dz = (rho_cell[..., :-1] - rho_cell[..., 1:]) / dz_safe
        N2 = -g / rho_0 * drho_dz
        return jnp.maximum(N2, 0.0)
    if n2_mode == "adiabatic":
        if (T_cell is None or S_cell is None or p_cell is None
                or dz_ref is None or jacobian is None):
            raise ValueError(
                "n2_mode='adiabatic' requires T_cell, S_cell, p_cell "
                "(cell-centre pressure [Pa]), dz_ref and jacobian to "
                "displace parcels through the EOS."
            )
        from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
        return compute_buoyancy_frequency_adiabatic(
            T_cell, S_cell, p_cell, dz_ref, jacobian,
            eos_fn=eos_fn, rho_ref=rho_0, g=g,
            dz_half=dz_half if adiabatic_over_dz_half else None,
        )
    raise ValueError(
        f"Unknown n2_mode={n2_mode!r}; expected 'insitu' or 'adiabatic'."
    )


# ---------------------------------------------------------------------------
# Prognostic TKE backward-Euler step
# ---------------------------------------------------------------------------


def _tridiag_thomas(a, b, c, d):
    """Solve a tridiagonal system A x = d via the Thomas algorithm.

    a, b, c, d each have shape ``(..., N)`` and ``a[..., 0]``,
    ``c[..., -1]`` are unused (left as zero by the caller). Returns
    ``x`` of shape ``(..., N)``.
    """
    N = b.shape[-1]

    def step(carry, k):
        c_prev, d_prev = carry
        denom = b[..., k] - a[..., k] * c_prev
        denom_safe = jnp.where(jnp.abs(denom) > _EPS, denom, _EPS)
        cp = c[..., k] / denom_safe
        dp = (d[..., k] - a[..., k] * d_prev) / denom_safe
        return (cp, dp), (cp, dp)

    # Forward sweep
    init_c = jnp.zeros_like(b[..., 0])
    init_d = jnp.zeros_like(d[..., 0])
    _, (cp_all, dp_all) = jax.lax.scan(
        step, (init_c, init_d), jnp.arange(N),
    )
    # cp_all, dp_all have shape (N, ...); transpose so trailing axis is N.
    cp_all = jnp.moveaxis(cp_all, 0, -1)
    dp_all = jnp.moveaxis(dp_all, 0, -1)

    # Back substitution
    def back(carry, k_rev):
        x_next = carry
        k = N - 1 - k_rev
        x = jnp.where(
            k_rev == 0, dp_all[..., k],
            dp_all[..., k] - cp_all[..., k] * x_next,
        )
        return x, x

    x_init = jnp.zeros_like(b[..., 0])
    _, x_rev = jax.lax.scan(back, x_init, jnp.arange(N))
    x_rev = jnp.moveaxis(x_rev, 0, -1)
    # Reverse the back-sub output to get x in natural index order.
    return x_rev[..., ::-1]


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

    Returns
    -------
    e_new : (..., nlev-1)
    """
    if (dz_cell is None) != (dz_surface is None):
        raise ValueError(
            "_solve_tke_backward_euler: dz_cell and dz_surface must be "
            "passed together (TKEConfig.veros_dz_slots) or both omitted "
            f"(legacy); got dz_cell={'set' if dz_cell is not None else None}, "
            f"dz_surface={'set' if dz_surface is not None else None}."
        )
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
    #   (1 + dt * (diss_rate + buoy_sink_rate)) * e_new
    #     + diffusion contribution = e_old + dt * P_s + flux BC
    diag = 1.0 + dt * (diss_rate + buoy_sink_rate) + b_diff

    # RHS: explicit shear-production source + explicit convective buoyancy
    # production (zero in the default in-situ mode) + previous-step e
    # + the external energy-recycling source ``forc`` (eke_diss_iw + K_diss_bot,
    # Veros integrate_tke; zero / None ⇒ bit-identical).
    rhs = e_old + dt * (P_s + buoy_source)
    if external_source is not None:
        rhs = rhs + dt * external_source

    # Surface flux BC at interface k=0: add the flux divergence with
    # ``forc_tke_surface``-style energy input. Veros injects over the
    # surface W half-volume 0.5·dzw_top (tke.py:225) — the distance from
    # z=0 to the top cell centre — NOT the first interior centre spacing
    # dz_half[0] (the legacy denominator, kept bit-identical by default).
    if veros_slots:
        inj_vol = jnp.maximum(jnp.asarray(dz_surface, dtype=e_old.dtype), _EPS)
    else:
        inj_vol = jnp.maximum(dz_half[..., 0], _EPS)
    rhs = rhs.at[..., 0].add(dt * surface_flux / inj_vol)

    # Solve tridiagonal system.
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

    Differentiable; the ``min``/``max`` clamps are sub-gradient-safe and
    the ``6.6`` / ``1`` / ``10`` are Veros's fixed scheme constants.
    """
    if cfg.prandtl_mode == "constant":
        return jnp.full_like(kappaM, cfg.Prandtl_tke0)
    if cfg.prandtl_mode == "richardson":
        Ri = N2 / jnp.maximum(shear_sq, 1e-12)
        return jnp.maximum(1.0, jnp.minimum(10.0, 6.6 * Ri))
    raise ValueError(
        f"Unknown prandtl_mode={cfg.prandtl_mode!r}; expected 'unit', "
        f"'constant' or 'richardson'."
    )


def _bryan_lewis_kappaH_floor(z_interface: jnp.ndarray) -> jnp.ndarray:
    r"""Bryan & Lewis (1979) depth-dependent tracer-diffusivity floor.

    Veros ``enable_kappaH_profile`` (``veros/core/tke.py:94-102``):

    .. math::

        \kappa_H^{floor}(z) = \left(0.8 + \frac{1.05}{\pi}\,
            \arctan\!\frac{-z - 2500}{222.2}\right) \times 10^{-4}

    with ``z`` the interface position [m] (negative downward; Veros uses
    ``-zw`` with ``zw < 0``). Mainly raises the abyssal diffusivity below
    ~2500 m. Pure arithmetic -> differentiable.
    """
    # -z = depth (positive); Veros's argument is (-zw - 2500)/222.2 with
    # zw the (negative) interface height -> here z_interface plays zw.
    depth = -z_interface
    return (0.8 + 1.05 / jnp.pi
            * jnp.arctan((depth - 2500.0) / 222.2)) * 1.0e-4


def compute_K_from_tke(
    e: jnp.ndarray,
    l_k: jnp.ndarray,
    cfg: TKEConfig,
    *,
    N2: jnp.ndarray | None = None,
    shear_sq: jnp.ndarray | None = None,
    z_interface: jnp.ndarray | None = None,
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
      ``K_H = max(kappaH_min, K_M / Pr)`` (see :func:`_prandtl_number`);
      requires ``N2`` and ``shear_sq`` for the ``"richardson"`` Pr.

    Amplitude convention (``cfg.kappa_convention``):

    - ``"gaspar_sqrt2e"`` (default, BIT-IDENTICAL legacy): the Gaspar form
      ``K_M = c_k·l_k·sqrt(2·max(e, tke_background))``. On the signed-N²
      path the Veros buoyancy length already carries the ``sqrt(2)``
      (``mxl = √2·√e/√N̄``), so this DOUBLE-COUNTS it — K_M ×1.414 vs the
      oracle wherever the caps/floors don't bind.
    - ``"veros_sqrte"``: Veros tke.py:73 ``kappaM = c_k·mxl·sqrttke`` with
      ``sqrttke = sqrt(max(0, e))`` (zero in negative-TKE debt regions).

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
        K_M = jnp.maximum(K_M, cfg.kappaM_min)
        if N2 is None or shear_sq is None:
            raise ValueError(
                f"prandtl_mode={cfg.prandtl_mode!r} requires N2 and "
                f"shear_sq for the Prandtl-number computation."
            )
        Pr = _prandtl_number(N2, shear_sq, K_M, cfg)
        K_H = jnp.maximum(cfg.kappaH_min, K_M / Pr)
        # Bryan-Lewis (1979) arctan depth floor on K_H (Veros
        # enable_kappaH_profile). Previously recorded-but-ignored; now wired
        # on the opt-in Prandtl path. Low impact in shallow domains; raises
        # the abyssal tracer floor below ~2500 m. Requires the interface
        # depths.
        if cfg.enable_kappaH_profile and z_interface is not None:
            K_H = jnp.maximum(K_H, _bryan_lewis_kappaH_floor(z_interface))
    return K_M, K_H


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
    p_cell: jnp.ndarray | None = None,
    dz_ref: jnp.ndarray | None = None,
    jacobian: jnp.ndarray | None = None,
    eos_fn=None,
    z_interface: jnp.ndarray | None = None,
    external_source: jnp.ndarray | None = None,
    dz_surface: jnp.ndarray | None = None,
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
    if positivity == "veros_surface_correction" and (
            cfg.n2_mode != "adiabatic" or cfg.tke_mxl_choice != 2):
        # The non-Veros mixing-length branches (Bougeault-Lacarrere /
        # mxl_choice=1) evaluate sqrt(2e/N²) on the RAW TKE, which is NaN
        # once the interior is allowed to go negative; only the Veros
        # buoyancy length (signed-N² adiabatic, mxl_choice=2) handles the
        # energy debt via sqrt(max(0,e)). Fail loudly at config time.
        raise ValueError(
            "TKEConfig.positivity='veros_surface_correction' requires "
            "n2_mode='adiabatic' and tke_mxl_choice=2 (the Veros buoyancy "
            f"length); got n2_mode={cfg.n2_mode!r}, "
            f"tke_mxl_choice={cfg.tke_mxl_choice!r}."
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

    if tke_old is None:
        leading_shape = rho_cell.shape[:-1]
        nlev = rho_cell.shape[-1]
        tke_old = jnp.full(
            leading_shape + (nlev - 1,),
            cfg.tke_background,
            dtype=rho_cell.dtype,
        )

    shear_sq = _vertical_shear_squared(u_cell, v_cell, dz_half)

    # Static stability N^2. ``"insitu"`` (default) is the clipped in-situ
    # form (BIT-IDENTICAL); ``"adiabatic"`` is the SIGNED Veros parcel-
    # displacement form that lets the TKE convect (N^2 < 0).
    signed_n2 = cfg.n2_mode == "adiabatic"
    N2 = _compute_N2(
        rho_cell, dz_half, rho_0, g,
        T_cell=T_cell, S_cell=S_cell, p_cell=p_cell,
        dz_ref=dz_ref, jacobian=jacobian, eos_fn=eos_fn,
        n2_mode=cfg.n2_mode,
        adiabatic_over_dz_half=veros_slots,
    )

    if tau_x_surface is None and tau_y_surface is None:
        surface_flux = jnp.zeros(rho_cell.shape[:-1], dtype=rho_cell.dtype)
    else:
        tx = tau_x_surface if tau_x_surface is not None else jnp.zeros_like(rho_cell[..., 0])
        ty = tau_y_surface if tau_y_surface is not None else jnp.zeros_like(rho_cell[..., 0])
        surface_flux = (jnp.sqrt(tx * tx + ty * ty) / rho_0) ** 1.5

    # Sub-iteration loop (Mode B convergence; Mode A uses n_iterations=1).
    tke_curr = tke_old
    for _ in range(max(1, int(n_iterations))):
        l_k, l_eps = compute_mixing_lengths(
            tke_curr, N2, dz_half, cfg, signed_n2=signed_n2,
            dz_cell=dz_cell)
        K_M_curr, K_H_curr = compute_K_from_tke(
            tke_curr, l_k, cfg, N2=N2, shear_sq=shear_sq,
            z_interface=z_interface)
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
            dz_surface=dz_surface if veros_slots else None,
        )

    # Final K from converged TKE.
    l_k_final, l_eps_final = compute_mixing_lengths(
        tke_curr, N2, dz_half, cfg, signed_n2=signed_n2,
        dz_cell=dz_cell)
    K_M, K_H = compute_K_from_tke(
        tke_curr, l_k_final, cfg, N2=N2, shear_sq=shear_sq,
        z_interface=z_interface)

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
    p_cell: jnp.ndarray,
    dz_ref: jnp.ndarray,
    jacobian: jnp.ndarray,
    eos_fn,
    z_interface: jnp.ndarray,
    dz_surface: jnp.ndarray,
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
    N2 = _compute_N2(
        rho_cell, dz_half, rho_0, g,
        T_cell=T_cell, S_cell=S_cell, p_cell=p_cell,
        dz_ref=dz_ref, jacobian=jacobian, eos_fn=eos_fn,
        n2_mode=cfg.n2_mode, adiabatic_over_dz_half=True,
    )
    if tau_x_surface is None and tau_y_surface is None:
        surface_flux = jnp.zeros(rho_cell.shape[:-1], dtype=rho_cell.dtype)
    else:
        tx = (tau_x_surface if tau_x_surface is not None
              else jnp.zeros_like(rho_cell[..., 0]))
        ty = (tau_y_surface if tau_y_surface is not None
              else jnp.zeros_like(rho_cell[..., 0]))
        surface_flux = (jnp.sqrt(tx * tx + ty * ty) / rho_0) ** 1.5

    l_k, _l_eps = compute_mixing_lengths(
        tke_old, N2, dz_half, cfg, signed_n2=True, dz_cell=dz_cell)
    K_M, K_H = compute_K_from_tke(
        tke_old, l_k, cfg, N2=N2, shear_sq=shear_sq, z_interface=z_interface)
    ctx = TKEPostMixingContext(
        K_M_old=K_M, K_H_old=K_H, mxl=l_k,
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
    eos_salinity_floor: float = 1.0e-3,
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
    mxl_w = _w(ctx.mxl)

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
    b = 1.0 - (a + c) + dt * cfg.c_eps * sqrttke_w / jnp.maximum(
        mxl_w, cfg.mxl_min)

    d = e_w + dt * forc_w
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
