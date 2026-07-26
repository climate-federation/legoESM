"""Hydrostatic Primitive Equations on MPAS Voronoi meshes.

Implements the vector-invariant form of the hydrostatic PE using the
TRiSK discretization (Ringler et al. 2010) on the C-grid.  Reuses
the existing Voronoi operators and shape-agnostic vertical coordinate
functions (sigma and hybrid) from the core library.

Equations (per level k):
    du/dt = F_pv - grad(B_k) - R_d T_k grad(ln p_s) + visc + vert_adv
    dT/dt = -v·∇T - σ̇ ∂T/∂σ + κ T ω/p
    dp_s/dt = -(1/σ_range) ∫ div(p_s v) dσ

where B_k = KE + Φ_k (Bernoulli function), F_pv is the potential-
vorticity flux, and σ̇ is diagnosed from the continuity equation.

References
----------
- Ringler, T. D., et al. (2010). J. Comput. Phys., 229(9), 3065-3090.
- Skamarock, W. C., et al. (2012). Mon. Wea. Rev., 140, 3090-3105.
- Simmons & Burridge (1981). Mon. Wea. Rev., 109, 758-766.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.conservation import conservative_positive_clip
from legoesm.core.precision import cast_pytree

from legoesm.core.field import Field
from legoesm.core.state import MPASHydrostaticState, MPASHydrostaticTendencies
from legoesm.core.operators_voronoi import (
    # 2D operators used for surface-pressure-only fields (ln_ps, p_s).
    divergence_cell,
    gradient_edge,
    divergence_cell_3d,
    gradient_edge_3d,
    kinetic_energy_cell_3d,
    potential_vorticity_vertex_3d,
    pv_flux_energy_conserving_3d,
    pv_flux_enstrophy_conserving_3d,
    vector_laplacian_del2_3d,
    vector_laplacian_del4_3d,
    cell_to_edge_avg_3d,
    apvm_correction_3d,
)
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
    pressure_from_hybrid,
    dp_from_hybrid,
    compute_geopotential,
    compute_geopotential_hybrid,
    compute_sigma_dot_from_cumsum,
    compute_mass_flux_from_cumsum,
    vertical_advection,
    vertical_advection_hybrid,
    vertical_advection_theta,
    vertical_advection_theta_hybrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import (
    IntegrationMixin,
    refuse_unthreaded_stateful_physics,
)
from legoesm.parallel.reductions import global_sum_mpi, is_multi_process
from legoesm import constants


class MPASPrimitiveEquationConfig(NamedTuple):
    """Configuration for the MPAS hydrostatic primitive equation model."""
    g: float = constants.g
    nu_del2: float = 0.0          # del2 viscosity [m²/s]
    nu_del4: float = 0.0          # del4 viscosity [m⁴/s]
    nu_del4_ps: float = 0.0       # del4 diffusion for surface pressure [m⁴/s]
    K_h: float = 0.0              # scalar diffusion [m²/s]
    T_min: float = 50.0           # temperature floor [K]
    p_floor: float = 100.0        # pressure floor [Pa] for adiabatic heating (limits 1/p)
    pv_scheme: str = "energy"     # "energy" or "enstrophy"
    apvm_scale: float = 0.0       # APVM upwinding (0 = off)
    fix_mass: bool = True
    # Column-conserving tracer positivity clamp (borrow the clipped deficit
    # from the positives) instead of the mass-CREATING plain max(q, 0).  See
    # the "--- 3. Floors ---" note: the naive clamp invents ~+30 kg/m2/yr of
    # water on the AMIP century.  Default False keeps existing MPAS results
    # bit-identical; flip after validation.
    conservative_tracer_clamp: bool = False
    anchor_mass_to_initial: bool = False  # iter-11: mirror PE/SW anchor pattern
    # Default integrator is the 5-stage 4th-order SSP scheme — NOT the
    # 3-stage ``ssp_rk3`` — because ``ssp_rk3`` has the smaller absolute-
    # stability region and cannot tolerate the operational ``del2``/``del4``
    # hyperdiffusion at the time steps used here.  Measured (GPU jobs
    # 8087100/8088578, reproduced CPU x64 in
    # ``tests/unit/test_mpas_atmosphere.py::TestMPASHydrostaticIntegratorStability``):
    #   * With ZERO dissipation (nu_del2 = nu_del4 = 0, the bare config),
    #     BOTH ssp_rk3 and ssp_rk54 are stable to dt >= 600 s on L4/nlev30
    #     from a near-rest IC — the integrator choice is irrelevant.
    #   * With hyperdiffusion ON, ssp_rk3 diverges within ~3 steps at
    #     dt = 600 s while ssp_rk54 stays bounded.  The biharmonic ∇⁴
    #     operator's (negative-real-axis) eigenvalues fall outside ssp_rk3's
    #     stability region at this dt but inside the 5-stage ssp_rk54 one.
    # A long run should damp grid-scale noise with hyperdiffusion, so the
    # default must cope with it; ssp_rk54 also matches the spectral-PE
    # default (``component_factory`` builds Spectral PE with ``ssp_rk54``).
    #
    # This supersedes the earlier "undamped gravity wave on the imaginary
    # axis / hidden-CFL at ~300 s any resolution" explanation, which the
    # zero-dissipation sweep disproved.  The separate ~450 s dt-ceiling seen
    # with the gray AMIP deck (driver ``_run_mpas``) is a radiative startup
    # transient (T=300 K isothermal IC), integrator-independent — and well
    # above the production dt=240 s, which is stable for both integrators.
    #
    # We default to the SCAN-FOLDED variant ``ssp_rk54_scan`` (identical
    # Spiteri-Ruuth scheme and stability region as ``ssp_rk54`` — only the
    # tendency is compiled once instead of inlined five times).  The inlined
    # form makes XLA-CPU cross an op-count threshold that de-vectorizes the
    # TRiSK indirect-addressing gathers, costing ~8x its nominal 5-evaluation
    # work (439 ms/step vs 173 ms/step at ico5/nlev=40).  The fold removes
    # that blowup at no stability cost.  Use ``ssp_rk54`` for a bit-exact
    # reference run.  See ``timestepping/ssp_rk54.py``.
    time_integrator: str = "ssp_rk54_scan"
    p_ceil: float = 2.0e6          # Surface-pressure ceiling [Pa] (~20-bar overflow guard for omega/p).
    # Vertical biharmonic (∂⁴/∂σ⁴) hyperdiffusion RATE for T [1/s] — an
    # index-space fourth-difference filter (NOT a physical hyperdiffusivity;
    # cf. the horizontal ``nu_del4`` [m⁴/s]).  Scale-selective damping of the
    # grid-scale 2Δσ vertical mode.  Cures the #930 vertical checkerboard: the
    # thermodynamic equation's adiabatic term κ·T·ω/p amplifies vertical T
    # structure in subsidence (1/p explodes at the low-pressure top levels),
    # and NOTHING else in this dycore damps a 2Δσ mode in T (vertical advection
    # is upwind but vanishes where the mass flux is weak; ω is smoothed by the
    # ½ half→full average).  A 2Δσ mode grows until the silent ``T_min`` floor
    # pins its cold levels and rectifies it into an even/odd checkerboard (#915
    # autopsy: even levels pinned at 50 K, odd exploding to 8e8 K).  Damping
    # rate is 16·ν interior / 8·ν at the top+bottom boundary (τ = 1/(16ν),
    # 1/(8ν); e.g. ν=2e-6 ⇒ ~8.7 h interior, ~17 h boundary — fast vs the
    # day-20 blowup).  del4 damps 2Δσ ~47× faster than an 8Δσ resolved wave, so
    # resolved vertical structure is essentially untouched, and it conserves
    # column-integrated T to machine precision (flux form).  0.0 (default)
    # reproduces the pre-fix dycore bit-for-bit (matches the ``nu_del2``/
    # ``nu_del4`` "off by default, set in production" convention); the
    # coupled/AMIP path sets a small value.  Last field to preserve positional ABI.
    nu_vert4_T: float = 0.0


# ============================================================================
# Tendency computation
# ============================================================================

def vertical_del4_T_tendency(
    T_3d: jax.Array, nu_vert4_T: float, layer_mass: jax.Array | None = None,
) -> jax.Array:
    """Scale-selective vertical biharmonic damping of the grid-scale T mode.

    Returns the tendency ``-nu · ∂⁴T/∂σ⁴`` (a discrete fourth-difference on the
    level INDEX — so ``nu_vert4_T`` is a filter RATE [1/s], NOT a physical
    hyperdiffusivity like the horizontal ``nu_del4`` [m⁴/s]).  It damps the 2Δσ
    (Nyquist) vertical mode while leaving resolved vertical structure
    essentially untouched — the vertical analogue of the dycore's horizontal
    ``nu_del4`` biharmonic hyperdiffusion.

    Implemented as del2∘del2 (Laplacian of the Laplacian).  Boundary treatment:
    the INNER Laplacian is ``reflect``-padded (so a 2Δσ mode keeps its full
    ``-4`` Laplacian at the top/bottom levels — where the #930 checkerboard is
    worst, at the low-pressure top), while the OUTER Laplacian is ``edge``
    (zero-gradient) padded (a no-flux boundary → the INDEX-space sum
    ``Σ_k tendency_k`` is ZERO to machine precision for ANY profile).
    That is NOT the same as column conservation: the conserved quantity is the
    MASS-weighted ``Σ_k tendency_k · Δσ_k``, and the two coincide only when
    ``Δσ`` is constant.  On a stretched grid the unweighted-zero operator is a
    spurious column source/sink — measured on the tropopause-refined σ grid
    (``grids.vertical.tropopause_refined_sigma_half``, refine=3, nlev=30) at
    -7.86 W/m² of column enthalpy and -0.135 mm/day of column water for a
    ±5 K / ±1 g/kg 2Δσ checkerboard, and -0.99 W/m² on the shipped stretched
    HYBRID L40 grid.  Pass ``dsigma`` to remove it: the mass-weighted mean of
    the tendency is subtracted, which is the minimum-norm conservative
    projection (it leaves every vertical DIFFERENCE — hence the filter's
    variance damping — untouched and only cancels the spurious column mean).
    On a uniform grid the correction is identically zero to round-off
    (measured ≤1.4e-20 K/s, far below the float32 ULP of the tendency), so the
    uniform and ``dsigma=None`` paths stay bit-identical.  Discrete 2Δσ
    ``(-1)^k`` response: ``-16·nu`` in the interior, ``-8·nu`` at the top/bottom
    (½ the interior rate — a boundary no-flux constraint of any conservative
    biharmonic; still strong).  An 8Δσ resolved wave sees ``≈-0.34·nu``
    (≈47× weaker than 2Δσ), so the filter is grid-scale-selective.

    Parameters
    ----------
    T_3d : jax.Array
        Temperature, shape ``(..., nlev)``.
    nu_vert4_T : float
        Biharmonic filter rate [1/s].  ``0.0`` ⇒ exact zero tendency.
    layer_mass : jax.Array or None
        Per-layer mass weight, either ``(nlev,)`` or per-column
        ``(..., nlev)``.  Any positive multiple of the true layer mass works
        (it is normalised): in σ pass ``Δσ`` — ``dp_k = p_s·Δσ_k`` and p_s
        cancels — but in HYBRID pass the actual ``dp = dA·p_ref + dB·p_s``,
        NOT ``dA + dB``, which is the thickness only at ``p_s = p_ref``.
        When given, the mass-weighted column mean is removed so
        ``Σ_k tendency_k · dp_k == 0`` to machine precision on ANY grid.
        ``None`` keeps the legacy index-space-only behaviour (correct only for
        a uniform grid).

    Returns
    -------
    jax.Array
        Vertical-hyperdiffusion tendency of T, same shape as ``T_3d``.
    """
    pad_axes = ((0, 0),) * (T_3d.ndim - 1)
    # Inner Laplacian: reflect BC keeps the FULL 2Δσ response at the boundary
    # levels (a plain edge/no-flux inner BC halves it again and leaves a slowly
    # decaying top boundary mode).
    Tp = jnp.pad(T_3d, (*pad_axes, (1, 1)), mode="reflect")
    lap = Tp[..., :-2] - 2.0 * Tp[..., 1:-1] + Tp[..., 2:]      # ∂²/∂σ²
    # Outer Laplacian: edge (zero-gradient / no-flux) BC ⇒ Σ_k tendency = 0
    # exactly (flux form), so the filter conserves column-integrated T.
    lap_p = jnp.pad(lap, (*pad_axes, (1, 1)), mode="edge")
    bih = lap_p[..., :-2] - 2.0 * lap_p[..., 1:-1] + lap_p[..., 2:]  # ∂⁴/∂σ⁴ (>0 at 2Δσ)
    tend = -nu_vert4_T * bih
    if layer_mass is None:
        return tend
    # Conservative projection onto the mass-weighted zero-mean subspace.
    # Sign convention: ``tend`` is a source term added to dX/dt, so removing
    # its mass-weighted mean makes the filter a pure REDISTRIBUTOR of X within
    # the column — no net column source or sink, either sign.
    # ``keepdims`` on BOTH reductions so a per-column weight (..., nlev) and a
    # coordinate-constant weight (nlev,) both normalise by their OWN column
    # sum; a scalar ``.sum()`` would silently sum over cells for the former.
    # PRECISION: the weight is cast DOWN to the tendency dtype, so the closure
    # is exact only to that dtype's reduction round-off — measured residual
    # 4e-14 W/m² in fp64 and 3.6e-5 W/m² in fp32, against the 7.9 W/m² leak
    # this replaces.  fp32 is therefore 2e5x better than the status quo and
    # 4 orders below the <1 W/m² TOA-imbalance target, not an exact fp64
    # guarantee (codex round 3).
    _w = layer_mass.astype(tend.dtype)
    return tend - ((tend * _w).sum(axis=-1, keepdims=True)
                   / _w.sum(axis=-1, keepdims=True))


def mpas_hydrostatic_tendencies(
    state: MPASHydrostaticState,
    mesh: VoronoiMesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    config: MPASPrimitiveEquationConfig = MPASPrimitiveEquationConfig(),
    physics_tendency: MPASHydrostaticTendencies | None = None,
    dt: float = 0.0,
) -> MPASHydrostaticTendencies:
    """Compute tendencies for the hydrostatic PE on an MPAS mesh.

    Parameters
    ----------
    state : MPASHydrostaticState
    mesh : VoronoiMesh
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : MPASPrimitiveEquationConfig
    physics_tendency : MPASHydrostaticTendencies, optional
    dt : float
        Time step (needed for APVM correction).

    Returns
    -------
    MPASHydrostaticTendencies
    """
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    u_3d = state.u.data        # (nEdges, nlev)
    T_3d = state.T.data        # (nCells, nlev)
    p_s = state.p_s.data       # (nCells,)
    phis = state.phis.data     # (nCells,)

    R_d = constants.R_d
    kappa = constants.kappa
    T_3d.shape[-1]
    p_s = jnp.clip(p_s, config.p_floor, config.p_ceil)

    # --- 1. Pressure at full levels ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)   # (nCells, nlev)
        dp = dp_from_hybrid(sigma_coord, p_s)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 2. Geopotential ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T_3d, p_s, sigma_coord, phis)
    else:
        Phi = compute_geopotential(T_3d, p_s, sigma_coord, phis)

    ln_ps = jnp.log(p_s)

    # --- Batched 3D tendencies (single gather for all levels) ---
    # Kinetic energy at all levels
    ke_3d = kinetic_energy_cell_3d(u_3d, mesh)  # (nCells, nlev)

    # Bernoulli function: KE + Phi
    bernoulli_3d = ke_3d + Phi  # (nCells, nlev)

    # Bernoulli + ln(p_s) [+ T] gradient batch.  All three quantities use
    # the same ``cellsOnEdge`` gather + ``dcEdge`` divide; the trailing
    # axis is purely passive.  Promote ``ln_ps`` to a single-level slot
    # via ``[..., None]`` and concatenate along the trailing axis with
    # the (bernoulli, T) batch.  When ``K_h > 0`` the batch has 2
    # 3D channels + 1 2D channel = ``nlev*2 + 1`` slots; when ``K_h = 0``
    # it has 1 + 1 = ``nlev + 1`` slots.  Saves one full
    # ``gradient_edge`` (2D) call per RHS evaluation — same Loop 148/159
    # exploit as the latlon PE (B, ln_ps) batch.
    n_cells_BT = bernoulli_3d.shape[0]
    nlev_BT = bernoulli_3d.shape[-1]
    if config.K_h > 0:
        _BT_stack = jnp.stack(
            [bernoulli_3d, T_3d], axis=-1,
        )  # (nCells, nlev, 2)
        _BT_flat = _BT_stack.reshape(n_cells_BT, nlev_BT * 2)
    else:
        _BT_flat = bernoulli_3d  # (nCells, nlev)
    _BTln_input = jnp.concatenate(
        [_BT_flat, ln_ps[:, jnp.newaxis]], axis=-1,
    )  # (nCells, nlev*K + 1)
    _BTln_grad = gradient_edge_3d(_BTln_input, mesh)
    if config.K_h > 0:
        _grad_BT = _BTln_grad[:, : nlev_BT * 2].reshape(-1, nlev_BT, 2)
        grad_B_3d = _grad_BT[..., 0]
        grad_T_3d_pre = _grad_BT[..., 1]
    else:
        grad_B_3d = _BTln_grad[:, :nlev_BT]
        grad_T_3d_pre = None
    grad_ln_ps = _BTln_grad[:, -1]  # (nEdges,)

    # Pressure gradient correction: R_d * T_edge * grad_eta(ln p)
    # In sigma coords: grad_eta(ln p) = grad(ln p_s).
    # In hybrid coords: grad_eta(ln p) = (B*p_s/p) * grad(ln p_s).
    if _hybrid:
        # Batch the cell-to-edge gathers — three 3D fields (T_3d, p_full,
        # dp) and two 2D fields (p_s, ln_ps) — into a single
        # ``cell_to_edge_avg_3d`` call.  The 2D fields are promoted to
        # single-level slots via ``[..., None]`` and concatenated along
        # the trailing axis, so total trailing axis = ``nlev*3 + 2``.
        # ``cell_to_edge_avg_3d`` is a pure ``cellsOnEdge`` gather + average
        # — the trailing axis is purely passive.  Saves *two* full
        # ``cell_to_edge_avg`` (2D) calls per RHS evaluation.  Same
        # exploit as Loop 154 / 174.
        nlev_te = T_3d.shape[-1]
        _Tpd_stack = jnp.stack([T_3d, p_full, dp], axis=-1)  # (nCells, nlev, 3)
        _Tpd_flat = _Tpd_stack.reshape(_Tpd_stack.shape[0], nlev_te * 3)
        _Tpd_pl_input = jnp.concatenate(
            [_Tpd_flat, p_s[:, jnp.newaxis], ln_ps[:, jnp.newaxis]], axis=-1,
        )  # (nCells, nlev*3 + 2)
        _Tpd_pl_edge = cell_to_edge_avg_3d(_Tpd_pl_input, mesh)
        _Tpd_edge = _Tpd_pl_edge[:, : nlev_te * 3].reshape(-1, nlev_te, 3)
        T_edge_3d = _Tpd_edge[..., 0]
        p_full_edge = _Tpd_edge[..., 1]
        dp_edge_3d = _Tpd_edge[..., 2]  # consumed in the divergence batch below
        p_s_edge_scalar = _Tpd_pl_edge[:, -2]  # (nEdges,)
        ln_ps_edge_pre = _Tpd_pl_edge[:, -1]   # (nEdges,) — reused below
        B_full = sigma_coord.B_full  # (nlev,)
        hybrid_factor_edge = B_full * p_s_edge_scalar[:, None] / jnp.maximum(p_full_edge, 1e-10)
        pg_corr_3d = R_d * T_edge_3d * grad_ln_ps[:, None] * hybrid_factor_edge
    else:
        # Batch (T_3d, p_s, ln_ps) into a single cell_to_edge_avg_3d call.
        # ``p_s`` rides as one extra passive slot so the σ-branch flux-form
        # continuity below gets the SAME edge-averaged p_s the hybrid branch
        # uses for its dp_edge — no extra gather.
        nlev_te = T_3d.shape[-1]
        _T_ln_input = jnp.concatenate(
            [T_3d, p_s[:, jnp.newaxis], ln_ps[:, jnp.newaxis]], axis=-1,
        )  # (nCells, nlev + 2)
        _T_ln_edge = cell_to_edge_avg_3d(_T_ln_input, mesh)
        T_edge_3d = _T_ln_edge[:, :nlev_te]
        p_s_edge_scalar = _T_ln_edge[:, -2]  # (nEdges,)
        ln_ps_edge_pre = _T_ln_edge[:, -1]
        # σ-coordinate layer thickness at edges: dp_k = p_s·Δσ_k (astype:
        # keep the state dtype when the σ arrays are f32).
        dp_edge_3d = (
            p_s_edge_scalar[:, None]
            * sigma_coord.dsigma.astype(p_s.dtype)[None, :]
        )  # (nEdges, nlev)
        pg_corr_3d = R_d * T_edge_3d * grad_ln_ps[:, None]  # (nEdges, nlev)

    # PV flux: h_proxy = dp/g (pressure thickness).  For the hybrid
    # branch ``dp_edge_3d`` was produced by the batched cell-to-edge
    # gather above, so ``h_proxy_edge = dp_edge_3d / g`` is free —
    # forward it to ``pv_flux_*_conserving_3d`` via ``h_edge_3d=`` to
    # skip one redundant ``cellsOnEdge`` gather.
    if _hybrid:
        h_proxy_3d = dp / config.g  # (nCells, nlev)
        h_proxy_edge_3d = dp_edge_3d / config.g
    else:
        h_proxy_3d = p_s[:, None] * sigma_coord.dsigma[None, :] / config.g
        h_proxy_edge_3d = None

    q_v_3d = potential_vorticity_vertex_3d(u_3d, h_proxy_3d, mesh.fVertex, mesh)

    if config.apvm_scale > 0:
        q_v_3d = apvm_correction_3d(q_v_3d, u_3d, mesh, config.apvm_scale * dt)

    if config.pv_scheme == "energy":
        pv_flux_3d = pv_flux_energy_conserving_3d(
            u_3d, h_proxy_3d, q_v_3d, mesh, h_edge_3d=h_proxy_edge_3d,
        )
    elif config.pv_scheme == "enstrophy":
        pv_flux_3d = pv_flux_enstrophy_conserving_3d(
            u_3d, h_proxy_3d, q_v_3d, mesh, h_edge_3d=h_proxy_edge_3d,
        )
    else:
        raise ValueError(
            f"Unknown pv_scheme {config.pv_scheme!r}; "
            "expected one of: 'energy', 'enstrophy'."
        )

    # Momentum tendency
    du_dt_3d = -grad_B_3d - pg_corr_3d + pv_flux_3d  # (nEdges, nlev)

    # Viscosity — when both del2 and del4 are active, the biharmonic
    # ``vector_laplacian_del4_3d`` is defined as
    # ``-vector_laplacian_del2_3d(vector_laplacian_del2_3d(u))``, so
    # the *inner* del2 is identical to the explicit del2 viscosity.
    # Compute it once and reuse — saves one full
    # ``vector_laplacian_del2_3d`` call (1 div + 1 curl + 1 grad +
    # 1 tangential-curl difference) per RHS evaluation.  Same exploit
    # as Loop 135 for the latlon ocean K_h+K_bih sharing.
    if config.nu_del2 > 0 and config.nu_del4 > 0:
        _del2_u = vector_laplacian_del2_3d(u_3d, mesh)
        du_dt_3d = du_dt_3d + config.nu_del2 * _del2_u
        du_dt_3d = du_dt_3d - config.nu_del4 * vector_laplacian_del2_3d(_del2_u, mesh)
    elif config.nu_del2 > 0:
        du_dt_3d = du_dt_3d + config.nu_del2 * vector_laplacian_del2_3d(u_3d, mesh)
    elif config.nu_del4 > 0:
        du_dt_3d = du_dt_3d + config.nu_del4 * vector_laplacian_del4_3d(u_3d, mesh)

    # Batched divergences.  ``divergence_cell_3d`` shares the same
    # MPAS edgesOnCell gather + reduce on the leading edge axis (the
    # trailing nlev axis is purely passive), so all the divergences
    # the dycore needs at this stage can fold into a single call:
    #
    #   * div(u)                 — horizontal-advection / PV bookkeeping
    #   * div(u * T_edge)        — temperature flux divergence
    #   * div(u * ln_ps_edge)    — v·∇(ln p_s) thermodynamic correction
    #   * div(u * dp_edge)       — flux-form layer-mass continuity
    #                              (BOTH branches; σ uses dp = p_s_edge·Δσ)
    #   * div(grad_T)            — K_h scalar Laplacian (only when ``K_h > 0``)
    #
    # Pull ``ln_ps_edge`` and (when hybrid) ``u*dp_edge_3d`` and (when
    # K_h > 0) ``grad_T_3d_pre`` up into the batch so the standalone
    # divergence calls that previously fired later in the function are
    # eliminated.
    flux_T_3d = u_3d * T_edge_3d  # (nEdges, nlev)
    # ``ln_ps_edge_pre`` was already produced by the batched
    # ``cell_to_edge_avg_3d`` block above (Loop 175); reuse it here so
    # the standalone 2D ``cell_to_edge_avg(ln_ps)`` call is eliminated.
    ln_ps_edge = ln_ps_edge_pre
    flux_lnps_3d = u_3d * ln_ps_edge[:, None]   # (nEdges, nlev)
    n_edges_d, nlev_d = u_3d.shape

    _div_input_list = [u_3d, flux_T_3d, flux_lnps_3d]
    _idx_u, _idx_uT, _idx_ulnps = 0, 1, 2
    _idx_gradT = -1
    # Flux-form layer-mass divergence for BOTH vertical-coordinate branches
    # (dp_edge_3d is dA+dB·p_s at edges when hybrid, p_s_edge·Δσ when σ).
    _div_input_list.append(u_3d * dp_edge_3d)
    _idx_udp = len(_div_input_list) - 1
    if config.K_h > 0:
        _div_input_list.append(grad_T_3d_pre)
        _idx_gradT = len(_div_input_list) - 1

    _n_div = len(_div_input_list)
    _div_inputs = jnp.stack(_div_input_list, axis=-1)  # (nEdges, nlev, K)
    _div_outputs = divergence_cell_3d(
        _div_inputs.reshape(n_edges_d, nlev_d * _n_div), mesh,
    ).reshape(-1, nlev_d, _n_div)
    div_3d = _div_outputs[..., _idx_u]
    div_uT_3d = _div_outputs[..., _idx_uT]
    div_flux_lnps = _div_outputs[..., _idx_ulnps]
    div_dp_3d = _div_outputs[..., _idx_udp]  # flux-form div(u·dp), both branches
    if config.K_h > 0:
        _div_grad_T = _div_outputs[..., _idx_gradT]
    horiz_adv_T_3d = -div_uT_3d + T_3d * div_3d  # (nCells, nlev)

    # Scalar diffusion — ``grad_T_3d`` and its divergence were already
    # computed in the batched blocks above; reuse the cached results.
    if config.K_h > 0:
        horiz_adv_T_3d = horiz_adv_T_3d + config.K_h * _div_grad_T

    # --- 4. Surface pressure tendency and vertical velocity ---
    # Flux-form continuity (both branches): ``div_dp_3d = div(u·dp_edge)``
    # from the batched divergence block above.  The cumsum is shared between
    # ``dp_s_dt`` (last entry) and the σ̇ / mass-flux integration (iter-53/54
    # pattern: one cross-cell-shard reduction per RK stage).  Positive
    # divergence (mass export) ⇒ dp_s/dt < 0.
    _cumsum_dp = jnp.cumsum(div_dp_3d, axis=-1)  # (nCells, nlev)
    _D_total_p = _cumsum_dp[..., -1:]            # (nCells, 1)  [Pa/s]
    if _hybrid:
        # Hybrid closure on MPAS:
        #   B_range * dp_s/dt = -sum_k div(dp_k * v_k)
        dp_s_dt = -_D_total_p[..., 0] / sigma_coord.B_range

        # Flux-form mass flux from the SAME cumsum — consistent with the
        # flux-form dp_s_dt above.  (compute_mass_flux_hybrid would rebuild
        # it from the ADVECTIVE div(v)·dp, which differs wherever ∇p_s ≠ 0.)
        mass_flux = compute_mass_flux_from_cumsum(
            _cumsum_dp, _D_total_p, sigma_coord,
        )
        # θ-form vertical thermodynamic transport (cancellation-free, #930):
        #   -F·∂T/∂p + κ·T·F/p  ==  -exner·F·∂θ/∂p   (θ = T·(p₀/p)^κ).
        # Advecting θ cancels the two large near-equal terms BEFORE
        # discretization, killing the 2Δz residual the 1/p prefactor
        # amplified at the stretched top levels.  See the σ branch below.
        vert_thermo_T = vertical_advection_theta_hybrid(
            T_3d, mass_flux, p_s, sigma_coord)
        # Only the surface-pressure-tendency part of ω stays in ``adiabatic``:
        #   ω = B·dp_s/dt + F  ⇒  ω_ps = B·dp_s/dt.  The F (mass-flux) part
        # κ·T·F/p is now folded into ``vert_thermo_T`` above — NO double-count.
        omega_ps = sigma_coord.B_full * dp_s_dt[:, None]
    else:
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top

        # σ closure: (1 - σ_top)·dp_s/dt = -Σ_k div(u·p_s_edge·Δσ_k).
        # ``div_dp`` already carries the p_s factor (flux form) — no extra
        # p_s multiply, unlike the old advective div(v)·Δσ closure.
        dp_s_dt = -_D_total_p[..., 0] / sigma_range

        # Flux-form σ̇ from the SAME cumsum (shared closure; mirrors the
        # lat-lon C-grid reference implementation).
        sigma_dot = compute_sigma_dot_from_cumsum(
            _cumsum_dp, _D_total_p, p_s, sigma_coord,
        )

        # θ-form vertical thermodynamic transport (cancellation-free, #930):
        #   -σ̇·∂T/∂σ + κ·T·σ̇/σ  ==  -exner·σ̇·∂θ/∂σ   (θ = T·(p₀/p)^κ).
        # The split form computes -σ̇·∂T/∂σ (first-diff, ×2 at the 2Δz
        # Nyquist) and κ·T·σ̇/σ (point value, ×1) separately, leaving a
        # spurious 2Δz residual that the 1/σ prefactor amplifies at the
        # stretched top levels.  Advecting θ cancels the two large terms
        # BEFORE discretization.  σ-convention (index 0 top, σ̇>0 downward)
        # is inherited verbatim from the reused ``vertical_advection``.
        vert_thermo_T = vertical_advection_theta(
            T_3d, sigma_dot, p_s, sigma_coord)
        # Only the surface-pressure-tendency part of ω stays in ``adiabatic``:
        #   ω = σ·dp_s/dt + p_s·σ̇  ⇒  ω_ps = σ·dp_s/dt.  The σ̇ part
        # κ·T·σ̇/σ is now folded into ``vert_thermo_T`` above — NO double-count.
        omega_ps = sigma_coord.sigma_full * dp_s_dt[:, None]

    # Surface pressure hyperdiffusion: -nu * del2(del2(p_s))
    if config.nu_del4_ps > 0:
        del2_ps = divergence_cell(gradient_edge(p_s, mesh), mesh)
        del4_ps = divergence_cell(gradient_edge(del2_ps, mesh), mesh)
        dp_s_dt = dp_s_dt - config.nu_del4_ps * del4_ps

    # Vertical advection of u: approximate via edge-averaged sigma-dot
    if _hybrid:
        vert_adv_u = _vertical_advection_edge(u_3d, mass_flux, sigma_coord, mesh, hybrid=True, p_s=p_s)
    else:
        vert_adv_u = _vertical_advection_edge(u_3d, sigma_dot, sigma_coord, mesh, hybrid=False)

    du_dt_3d = du_dt_3d + vert_adv_u

    # --- 5. Thermodynamic equation ---
    # Adiabatic heating from the surface-pressure tendency ONLY: κ·T·ω_ps/p.
    # The σ̇/mass-flux part of the adiabatic term (κ·T·σ̇/σ resp. κ·T·F/p)
    # now lives inside ``vert_thermo_T`` via the θ-form (#930) — using the
    # full ω here would double-count it.
    # ``p_floor`` still caps THIS term because it forms an explicit 1/p; the
    # σ̇-part inside ``vert_thermo_T`` needs no such cap — the θ-form carries
    # it as exner·∂θ/∂σ (exner = (p/p₀)^κ → 0 at the top), so it is bounded by
    # construction rather than by a 1/p clip.  (For the standard σ / hybrid
    # coordinate builders p_full stays well above ``p_floor`` at every full
    # level, so the two paths' top-level behaviour coincides in practice.)
    p_adiab = jnp.maximum(p_full, config.p_floor)
    adiabatic = kappa * T_3d * omega_ps / p_adiab

    # v·∇(ln p_s) at cells: div(u * ln_ps_edge) - ln_ps * div(u).
    # ``div_flux_lnps`` was already computed via the batched divergence
    # block above; just combine it with ``div_3d`` here.
    v_grad_lnps = div_flux_lnps - ln_ps[:, None] * div_3d  # (nCells, nlev)

    # In hybrid coords: grad_eta(ln p) = (B*p_s/p) * grad(ln p_s),
    # so the adiabatic correction needs the same factor.
    if _hybrid:
        v_grad_lnps = v_grad_lnps * (sigma_coord.B_full * p_s[:, None] / p_adiab)
    adiabatic = adiabatic + kappa * T_3d * v_grad_lnps

    dT_dt_3d = horiz_adv_T_3d + vert_thermo_T + adiabatic

    # Vertical biharmonic hyperdiffusion of T (#930 cure): damp the grid-scale
    # 2Δσ vertical mode that the adiabatic κ·T·ω/p term amplifies but no other
    # vertical operator in this dycore opposes.  Zero when nu_vert4_T == 0.
    # Mass weight for the filter's conservative projection.  Must be the TRUE
    # layer mass dp_k, not a coordinate proxy: in σ, dp_k = p_s·Δσ_k and p_s
    # cancels between numerator and denominator, so Δσ is EXACT and cheaper;
    # in hybrid, dp_k = dA_k·p_ref + dB_k·p_s varies by column and
    # ``sigma_coord.dsigma`` (= dA + dB) is the thickness only at p_s = p_ref
    # — using it would leave the filter conservative only to the p_s/p_ref
    # departure (~50% of the dB share at p_s = 500 hPa).  ``dp`` is already
    # built above in the hybrid branch, so this is free.
    _filter_weight = dp if _hybrid else sigma_coord.dsigma
    if config.nu_vert4_T > 0.0 and T_3d.shape[-1] > 2:
        dT_dt_3d = dT_dt_3d + vertical_del4_T_tendency(
            T_3d, config.nu_vert4_T, _filter_weight)

    # --- 6. Add physics tendencies ---
    if physics_tendency is not None:
        du_dt_3d = du_dt_3d + physics_tendency.du_dt.data
        dT_dt_3d = dT_dt_3d + physics_tendency.dT_dt.data
        dp_s_dt = dp_s_dt + physics_tendency.dp_s_dt.data

    # --- 7. Tracer transport (moisture etc.) ---
    # Advect prognostic tracers with the dycore's OWN edge wind (u_3d) and
    # vertical mass flux (mass_flux for hybrid / sigma_dot for σ), so moisture
    # transport is MASS-CONSISTENT with the thermodynamics — same horizontal
    # operator (shared ``tracer_horizontal_advection``) and the SAME vertical
    # operator the dycore uses for T.  Physics (microphysics/convection)
    # tracer tendencies add on.  ``tracers=None`` ⇒ dry, no extra work.
    tracer_tends_out = None
    if state.tracers is not None and len(state.tracers) > 0:
        from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
            tracer_horizontal_advection,
        )
        _tnames = list(state.tracers.keys())
        q = jnp.stack([state.tracers[k].data for k in _tnames], axis=-1)
        dq = tracer_horizontal_advection(q, u_3d, mesh)
        if _hybrid:
            dq = dq + jax.vmap(
                lambda qk: vertical_advection_hybrid(qk, mass_flux, p_s, sigma_coord),
                in_axes=-1, out_axes=-1)(q)
        else:
            dq = dq + jax.vmap(
                lambda qk: vertical_advection(qk, sigma_dot, sigma_coord),
                in_axes=-1, out_axes=-1)(q)
        # #930 vertical checkerboard damper on TRACERS (same operator + rate
        # as the T filter above).  The 2026-07-23 moist-AMIP blowup forensics
        # (Tibetan-plateau cell, ±140 K 2Δσ T zigzag with 66 g/kg q_v pooling
        # at the hot levels) show the moist checkerboard's primary oscillator
        # is the TRACER field: tracer vertical transport reuses the
        # checkerboard-prone ``vertical_advection`` form (only T got the #962
        # θ-form rewrite), so damping T alone cannot stabilize the coupled
        # q↔latent-heating mode.  ``dsigma`` is passed so the MASS-weighted
        # column integral (not merely the index-space sum) is zero to machine
        # precision → conserves column moisture on the stretched hybrid and
        # tropopause-refined σ grids too, not just on a uniform grid.
        # Zero when nu_vert4_T == 0 (bit-identical).
        if config.nu_vert4_T > 0.0 and q.shape[-2] > 2:
            dq = dq + jax.vmap(
                lambda qk: vertical_del4_T_tendency(
                    qk, config.nu_vert4_T, _filter_weight),
                in_axes=-1, out_axes=-1)(q)
        _phys_tt = (physics_tendency.tracer_tendencies
                    if physics_tendency is not None else None)
        for _i, k in enumerate(_tnames):
            if _phys_tt is not None and k in _phys_tt:
                dq = dq.at[..., _i].add(_phys_tt[k].data)
        tracer_tends_out = {
            k: Field(data=dq[..., _i], name=f"d{k}_dt",
                     dims=("nCells", "nlev"), units="kg/kg/s")
            for _i, k in enumerate(_tnames)
        }

    return MPASHydrostaticTendencies(
        du_dt=Field(data=du_dt_3d, name="du_dt",
                    dims=("nEdges", "nlev"), units="m/s²"),
        dT_dt=Field(data=dT_dt_3d, name="dT_dt",
                    dims=("nCells", "nlev"), units="K/s"),
        dp_s_dt=Field(data=dp_s_dt, name="dp_s_dt",
                      dims=("nCells",), units="Pa/s"),
        dphis_dt=Field(data=jnp.zeros_like(phis), name="dphis_dt",
                       dims=("nCells",), units="m²/s³"),
        tracer_tendencies=tracer_tends_out,
    )


def _vertical_advection_edge(
    u_3d, vert_vel, sigma_coord, mesh, hybrid=False, p_s=None,
):
    """Vertical advection of edge-based velocity.

    Interpolates the vertical velocity (sigma-dot or mass flux) from
    cells to edges, then applies the standard vertical advection.
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]

    if hybrid:
        # mass_flux is (nCells, nlev+1), average to edges
        mass_flux_edge = 0.5 * (vert_vel[c1] + vert_vel[c2])
        p_s_edge = 0.5 * (p_s[c1] + p_s[c2])
        return vertical_advection_hybrid(u_3d, mass_flux_edge, p_s_edge, sigma_coord)
    else:
        # sigma_dot is (nCells, nlev+1), average to edges
        sigma_dot_edge = 0.5 * (vert_vel[c1] + vert_vel[c2])
        return vertical_advection(u_3d, sigma_dot_edge, sigma_coord)


# ============================================================================
# Model class
# ============================================================================

class MPASPrimitiveEquationModel(IntegrationMixin):
    """Hydrostatic primitive equation model on an MPAS Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : MPASPrimitiveEquationConfig, optional
    """

    #: The free RHS function ``(state, mesh, sigma_coord, config, *, dt, ...)``,
    #: exposed on the model so the shared-substrate multi-device sharder
    #: (``parallel.sharded_dynamics``) can compute tendencies on a rank-LOCAL,
    #: traced mesh WITHOUT importing this atmosphere module — keeping
    #: legoesm-core free of an upward atmosphere dependency.  (The ``.tendencies``
    #: method binds ``self.mesh``; the sharder needs the per-device mesh, so it
    #: needs the unbound function.)
    sharded_tendency_fn = staticmethod(mpas_hydrostatic_tendencies)

    def __init__(
        self,
        mesh: VoronoiMesh,
        sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
        config: MPASPrimitiveEquationConfig | None = None,
    ):
        self.mesh = mesh
        self.sigma_coord = sigma_coord
        self.config = config or MPASPrimitiveEquationConfig()
        # Pre-compute the global total area once at construction time so
        # the per-step mass fixer does not include this constant in its
        # cross-device reduction payload (drops 3-element allreduce → 2).
        # ``mesh`` here is the full global mesh (the MPI-partitioned step
        # lives in ``parallel/voronoi_mpi.py`` and has its own constant);
        # under SPMD sharding XLA folds this value as a compile-time
        # constant and avoids the live-time reduction.
        self._total_area = float(jnp.sum(mesh.areaCell))
        # iter-11: lazy fp64 mass snapshot for anchor-to-initial.
        self._target_mass: jax.Array | None = None
        # Operator-split physics carry (prognostic TKE / convection state).
        # ``step`` stashes the physics-state OUT here each call so the driver
        # can feed it back in via ``phys_state`` next step, while ``step``
        # itself still returns just the dynamical state (backward-compatible).
        self._phys_state = None



    def compute_mass(self, state) -> jax.Array:
        """Compute global ``∫ p_s dA`` in the fp64 budget accumulator."""
        return jnp.sum(
            state.p_s.data.astype(jnp.float64)
            * self.mesh.areaCell.astype(jnp.float64),
        )

    def tendencies(
        self,
        state: MPASHydrostaticState,
        dt: float = 0.0,
        physics_tendency: MPASHydrostaticTendencies | None = None,
    ) -> MPASHydrostaticTendencies:
        return mpas_hydrostatic_tendencies(
            state, self.mesh, self.sigma_coord, self.config,
            physics_tendency=physics_tendency, dt=dt,
        )

    def step(
        self,
        state: MPASHydrostaticState,
        dt: float,
        physics_fn=None,
        forcing=None,
        phys_state=None,
    ):
        """Outer wrapper: snapshots initial mass on first call when
        ``anchor_mass_to_initial`` is on (fp64, outside JIT).

        ``forcing`` is an optional TRACED pytree (e.g. ``{"T_sfc": (nCells,)}``)
        threaded to ``physics_fn`` — the AMIP path passes a time-varying
        prescribed SST through here WITHOUT retracing (jit argument, not a
        static closure).  ``phys_state`` is the operator-split physics carry
        (prognostic TKE / convection state); pass back ``model._phys_state``
        from the previous step.  ``step`` returns just the dynamical state
        (backward-compatible) and stashes the physics-state OUT on
        ``self._phys_state`` — EAGER calls only: under an outer trace the
        side-channel is skipped (a stashed tracer leaks into the next
        trace; gh-417), so traced callers must thread ``phys_state``
        explicitly through their own carry.
        """
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="MPAS step()")
        target_mass = self._target_mass
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and target_mass is None):
            target_mass = self.compute_mass(state)
            if not isinstance(target_mass, jax.core.Tracer):
                # Designed eager path: cache the concrete t=0 mass so
                # later segments keep anchoring to the same constant.
                self._target_mass = target_mass
            # Traced path (step() inside an OUTER jit/grad/scan): NEVER
            # cache — a tracer stored on self leaks into the next trace
            # (UnexpectedTracerError; gh-417, same class as the
            # primitive_eq_cdgrid A1-gate bug).  Thread the per-call
            # pre-step mass instead (telescoping fixer semantics).
        state_new, phys_out, sfc_diag = self._step_jit(
            state, dt, physics_fn, target_mass, forcing, phys_state)
        if not any(isinstance(leaf, jax.core.Tracer)
                   for leaf in jax.tree_util.tree_leaves(phys_out)):
            self._phys_state = phys_out
        # Stash the surface net radiative fluxes (sw/lw net [W/m^2, +into
        # surface]) so the coupled MPAS loop can export them to the coupler.
        # Eager-only (same tracer guard as _phys_state — a stashed tracer leaks
        # into the next trace, gh-417); keep the last radiation-step value
        # across held-radiation sub-steps (sfc_diag is None then).
        if sfc_diag is not None and not any(
                isinstance(leaf, jax.core.Tracer)
                for leaf in jax.tree_util.tree_leaves(sfc_diag)):
            # Merge ELEMENT-WISE into the persisted bundle: sw/lw refresh only on
            # a full-radiation step, precip every step, so a wholesale overwrite
            # on a held-radiation step (sw/lw None) would DROP the last radiation
            # fluxes. Keep the last non-None value per slot (sw, lw, precip).
            _prev = getattr(self, "_sfc_diag", None) or ()
            # Pad the shorter of (prev, new) so a session that grows the
            # tuple contract (3-slot legacy -> 8-slot with TOA/turb-flux
            # extras) merges slot-wise instead of truncating.
            _n = max(len(sfc_diag), len(_prev))
            _prev = _prev + (None,) * (_n - len(_prev))
            sfc_diag = sfc_diag + (None,) * (_n - len(sfc_diag))
            self._sfc_diag = tuple(
                new if new is not None else old
                for new, old in zip(sfc_diag, _prev))
        return state_new

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_jit(
        self,
        state: MPASHydrostaticState,
        dt: float,
        physics_fn=None,
        target_mass: jax.Array | None = None,
        forcing=None,
        phys_state=None,
    ):
        """Advance one time step with OPERATOR-SPLIT physics.

        Dynamics (incl. tracer ADVECTION) are RK-integrated first with NO
        physics in the per-stage tendency; the physics package is then
        evaluated ONCE on the post-dynamics state and its tendencies applied
        forward over ``dt``.  This is the correct coupling for implicit /
        stateful physics (turbulent vertical diffusion, microphysics
        saturation adjustment, prognostic TKE), which assume a single forward
        application per step — mixing them into the per-RK-stage tendency
        (the previous scheme) integrates an adjustment scheme with sub-dt RK
        weights, which is wrong, and re-evaluates expensive physics
        (radiation) once per stage.  Additive physics (radiation heating) is
        unaffected to leading order.

        Returns ``(state_new, phys_state_out)``.  ``phys_state`` carries the
        prognostic physics state (TKE etc.) across steps; ``forcing`` carries
        per-step traced forcing (e.g. prescribed ``T_sfc``).  Both are jit
        arguments (NOT static), so new values each step do not retrace.

        NB: the spectral and cubed-sphere PE paths add physics INSIDE the
        per-RK-stage tendency (see ``spectral_pe.py`` / ``primitive_eq.py``);
        MPAS deliberately diverges to operator-split because its physics now
        includes the stateful / implicit turbulence (TKE + vertical diffusion)
        the others do not run.  Converging those paths onto operator-split is
        a future-consistency item, not a correctness issue here.
        """
        state = cast_pytree(state, None, "compute")

        # --- 1. Dynamics: RK-integrate dynamics-only tendencies (physics OFF;
        #        tracer ADVECTION stays in the dynamics, mass-consistently). ---
        def dyn_tendency_fn(s):
            tend = mpas_hydrostatic_tendencies(
                s, self.mesh, self.sigma_coord, self.config,
                physics_tendency=None, dt=dt,
            )
            # Pin every tendency leaf to the state's dtype.  Under an fp32
            # compute policy with x64 enabled, the f64 mesh-geometry
            # closure constants promote the tendency products to f64; the
            # scan-folded integrators (ssp_rk54_scan — the MPAS default via
            # time_integrator="auto") have a fixed-dtype scan carry and
            # REFUSE a tendency wider than the state (the unrolled ssp_rk3
            # used to promote silently instead).  Scan-carry dtype
            # stability rule: outputs at result dtype of the state.
            _new = MPASHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data.astype(s.u.data.dtype)),
                T=s.T.replace(data=tend.dT_dt.data.astype(s.T.data.dtype)),
                p_s=s.p_s.replace(
                    data=tend.dp_s_dt.data.astype(s.p_s.data.dtype)),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )
            # Carry tracer ADVECTION tendencies as a tendency-shaped state so
            # the pytree RK integrator advances moisture with u/T/p_s.  Same
            # tracer keys as the input state ⇒ tree-axpy lines up.
            if s.tracers is not None and tend.tracer_tendencies is not None:
                _new = _new._replace(tracers={
                    k: s.tracers[k].replace(
                        data=tend.tracer_tendencies[k].data.astype(
                            s.tracers[k].data.dtype))
                    for k in s.tracers
                })
            return _new

        state_new = dispatch_integrator(
            state, dyn_tendency_fn, dt, self.config.time_integrator,
        )

        # --- 2. Operator-split physics: evaluate ONCE on the post-dynamics
        #        state, apply forward over dt.  ``state += dt * tendency``
        #        recovers a scheme's internal dt integration when its tendency
        #        is defined as (post-physics - pre)/dt (the package's
        #        convention). ---
        phys_state_out = phys_state
        sfc_diag = None
        if physics_fn is not None:
            _pr = physics_fn(state_new, self.mesh, self.sigma_coord,
                             phys_state=phys_state, forcing=forcing)
            if type(_pr) is tuple:
                _pt, phys_state_out = _pr[0], _pr[1]
            else:
                _pt = _pr
            # Export the surface net radiative fluxes the physics computed
            # (sw/lw net [W/m^2, +into surface]); the lean loop has no
            # PhysicsOutput channel, so without this the coupled-voronoi ocean
            # and land tiles were forced with zero shortwave. None (static,
            # radiation inactive / held-radiation sub-step) leaves it unset.
            _sw_sfc = getattr(_pt, "sw_net_sfc", None)
            _lw_sfc = getattr(_pt, "lw_net_sfc", None)
            _pr_sfc = getattr(_pt, "precip", None)   # surface precip [kg/m^2/s]
            # CMOR-feed diagnostic extras: TOA fluxes (radiation steps only)
            # + surface turbulent fluxes (every step). Slot ORDER is the
            # sfc_diag tuple contract shared with the driver feed:
            # (sw_net, lw_net, precip, lw_up_toa, sw_up_toa, sw_down_toa,
            #  shflx, lhflx, sw_down_sfc, lw_down_sfc).
            # ...appended (slots 8/9): surface DOWNWELLING sw/lw — the
            # interactive multilayer land forcing (AtmToSurface.sw_down/
            # lw_down; model_driver._marshal_land_forcing reads these slots).
            _extras = tuple(getattr(_pt, _k, None) for _k in (
                "lw_up_toa", "sw_up_toa", "sw_down_toa",
                "shflx_sfc", "lhflx_sfc",
                "sw_down_sfc", "lw_down_sfc"))
            # Publish when ANY surface diagnostic is fresh — precip (microphysics)
            # advances every step even on a held-radiation sub-step or a
            # radiation=none run where sw/lw are None, so gating on sw/lw would
            # stash stale precip. Each element is None-guarded by the consumer.
            if (_sw_sfc is not None or _lw_sfc is not None
                    or _pr_sfc is not None
                    or any(_e is not None for _e in _extras)):
                sfc_diag = (_sw_sfc, _lw_sfc, _pr_sfc) + _extras
            state_new = MPASHydrostaticState(
                u=state_new.u.replace(data=state_new.u.data + dt * _pt.du_dt.data),
                T=state_new.T.replace(data=state_new.T.data + dt * _pt.dT_dt.data),
                p_s=state_new.p_s.replace(
                    data=state_new.p_s.data + dt * _pt.dp_s_dt.data),
                phis=state_new.phis,
                tracers=state_new.tracers,
            )
            if (state_new.tracers is not None
                    and _pt.tracer_tendencies is not None):
                state_new = state_new._replace(tracers={
                    k: (state_new.tracers[k].replace(
                            data=state_new.tracers[k].data
                            + dt * _pt.tracer_tendencies[k].data)
                        if k in _pt.tracer_tendencies else state_new.tracers[k])
                    for k in state_new.tracers
                })

        # --- 3. Floors ---
        # Last-resort NaN-safety guard, now BEHIND the #930 cure (``nu_vert4_T``
        # damps the 2Δσ mode so this floor is dead in normal operation — the
        # integration test asserts T_min never binds once the cure is on).  It
        # is no longer the SILENT masker it was: #915's daily ``T < 100`` bounds
        # guard aborts the run on ANY floor activation, so a clamp to 50 K can
        # never again hide a runaway (#871/#912/#915).
        if self.config.T_min > 0:
            state_new = state_new._replace(
                T=state_new.T.replace(
                    data=jnp.maximum(state_new.T.data, self.config.T_min)))
        # Tracer non-negativity: advection is not positive-definite and
        # microphysics can leave tiny undershoots; clamp before they feed
        # saturation calculations.
        #
        # The plain ``maximum(f, 0.0)`` below is NOT mass-neutral, contrary to
        # the "negligible mass impact" this comment used to claim.  MEASURED
        # 2026-07-26 on the AMIP century (checkpoint day 520, dt=75 s):
        # horizontal advection ALONE leaves enough undershoot that the clamp
        # invents **7.13e-5 kg/m2/step = +0.0822 kg/m2/day = +30 kg/m2/yr** of
        # water — 96% of it from the spiky ``q_i`` (42773 cells) and ``q_c``
        # (26214 cells) fields, vs only 23 cells of ``q_v``.  That is 2.2x the
        # +0.0367 kg/m2/day total-water residual measured from the checkpoints,
        # and it compounded into column water 23->42 kg/m2, OLR 199->109 W/m2
        # and +10 K/yr of warming.
        #
        # ``conservative_tracer_clamp`` swaps the naive clamp for the
        # column-conserving borrow (shared with the LES lane).  Default False
        # so existing MPAS results are bit-identical until the flag is set;
        # the default is known-wrong and should flip once validated.
        if state_new.tracers is not None:
            if self.config.conservative_tracer_clamp:
                # ONLY the water MASS tracers get the conserving borrow.
                # Number concentrations (N_c/N_i/N_r) are NOT conserved
                # quantities — rescaling them to preserve a column integral is
                # unphysical and perturbs the microphysics directly (M2005 ice
                # deposition goes as N_i^(2/3)).  The LES reference makes the
                # same split ("number slots clip freely; their conservation is
                # not physically required"), so keep the two lanes consistent.
                _dsig = jnp.asarray(self.sigma_coord.dsigma)
                state_new = state_new._replace(tracers={
                    k: f.replace(data=(
                        conservative_positive_clip(f.data, _dsig, axis=-1)[0]
                        if _is_water_mass_tracer(k)
                        else jnp.maximum(f.data, 0.0)))
                    for k, f in state_new.tracers.items()
                })
            else:
                state_new = state_new._replace(tracers={
                    k: f.replace(data=jnp.maximum(f.data, 0.0))
                    for k, f in state_new.tracers.items()
                })

        if self.config.fix_mass:
            state_new = _fix_mass_mpas_hydro(
                state_new, state, self.mesh,
                total_area=self._total_area,
                target_mass=target_mass,
            )

        return cast_pytree(state_new, None, "storage"), phys_state_out, sfc_diag

    # integrate() and integrate_scan() inherited from IntegrationMixin


#: Water MASS mixing ratios [kg/kg] — the only tracers whose column integral
#: is a conserved quantity.  Number concentrations (``N_c``/``N_i``/``N_r``)
#: and any non-water tracer are deliberately excluded: rescaling a number to
#: preserve an integral is unphysical and feeds straight into the microphysics
#: (M2005 deposition ~ N_i^(2/3)).  Mirrors the LES lane's ``n_water`` split.
_WATER_MASS_TRACERS = frozenset(
    {"q_v", "q_c", "q_r", "q_i", "q_s", "q_g"})


def _is_water_mass_tracer(name: str) -> bool:
    """True for a water mass mixing ratio, tolerating a ``trc_`` prefix."""
    return str(name).removeprefix("trc_") in _WATER_MASS_TRACERS


def _fix_mass_mpas_hydro(
    state_new, state_old, mesh, total_area=None, target_mass=None,
):
    """Fix mass conservation: uniform additive correction to p_s.

    iter-11: cast both p_s fields to the fp64 budget accumulator before
    the area-weighted sum.  Plain fp32 reductions over ~10^4–10^5
    Voronoi cells leak ~N·eps noise into ``mass_old``/``mass_new`` and
    leave a residual O(10^-7) random walk in the global integral.
    When ``target_mass`` is provided (anchor-to-initial path), skip the
    ``state_old`` reduction entirely.

    Parameters
    ----------
    total_area : float or jax.Array, optional
        Pre-allreduced global ``sum(areaCell)``.  Defaults to a fresh
        ``jnp.sum(mesh.areaCell)`` (correct for single-rank /
        non-sharded; redundant work under MPI when *total_area* is
        already known at the call site).
    target_mass : jax.Array, optional
        Anchored initial mass; when supplied (and only-then),
        ``mass_old`` is replaced by this fp64 scalar.
    """
    area = mesh.areaCell
    if total_area is None:
        total_area = jnp.sum(area)
    acc = jnp.float64
    area_acc = area.astype(acc)
    if target_mass is not None:
        mass_new = jnp.sum(state_new.p_s.data.astype(acc) * area_acc)
        # is_multi_process(), NOT jax.process_count() > 1: the mpi4jax
        # allreduce is correct only when each rank holds a LOCAL partition
        # (route-A MPI).  Under multi-controller SPMD (route-B, federated
        # jax.distributed) every process traces the GLOBAL sharded arrays —
        # the sum above is already global via GSPMD, and routing it through
        # mpi4jax would arm the forbidden mixed stack AND over-count by the
        # world size (#751 latent-bug class).
        if is_multi_process():
            mass_new = global_sum_mpi(mass_new)
        mass_old = target_mass
    else:
        _ps_stack = jnp.stack(
            [
                state_old.p_s.data.astype(acc),
                state_new.p_s.data.astype(acc),
            ],
            axis=-1,
        ) * area_acc[..., None]
        local = jnp.sum(_ps_stack, axis=tuple(range(area.ndim)))
        if is_multi_process():  # route-A local partitions only (see above)
            local = global_sum_mpi(local)
        mass_old, mass_new = local[0], local[1]
    correction = (mass_old - mass_new) / total_area
    # iter-11: drop ``.astype(p_s.data.dtype)`` — fp64 correction
    # promotes the add, matches ``fix_ps_mass`` cubed-sphere convention.
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)
    return state_new._replace(p_s=p_s_fixed)
