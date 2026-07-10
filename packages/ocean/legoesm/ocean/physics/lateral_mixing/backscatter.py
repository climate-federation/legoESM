"""Deterministic energy backscatter (Jansen & Held 2014) for the ocean.

The backscatter scheme re-injects kinetic energy that has been removed by
the resolved-scale viscous closures (Smagorinsky, Leith, biharmonic).  A
prognostic, depth-MEAN (per-mass) subgrid kinetic-energy reservoir ``E(x, y)``
[m²/s²] evolves under

    dE/dt = ε_dissipation(x, y)        (source — from the resolved dyn)
          - ε_backscatter(x, y)        (sink   — returned to the resolved dyn)
          - E / τ_relax                (linear damping of the reservoir)

with ``E`` clamped to ``[E_min, E_max]`` each step.  The backscatter
momentum tendency is a NEGATIVE (anti-diffusive) Laplacian viscosity

    ∂u/∂t|_bs = -∇·(ν_bs ∇u) = -ν_bs ∇²u ,   ν_bs = c_bs · Δ · √E ≥ 0

i.e. an ordinary Laplacian viscosity applied with the *opposite* sign to
diffusion, so it INJECTS kinetic energy into the resolved flow (growth
rate ∝ +ν_bs k²).  This is the Jansen & Held (2014) / Bachman (2019)
negative-Laplacian ("negative viscosity") backscatter — NOT a biharmonic.
The coefficient is a HARMONIC viscosity ``ν_bs = c_bs · Δ · √E`` [m²/s] on
BOTH grids (``Δ = √area`` the local grid length [m], ``√E`` [m/s]), so —
unlike the earlier per-path ``Δ¹``/``Δ³`` split — ``c_bs`` carries the same
meaning on the C-grid and MPAS paths.

STABILITY (read before enabling).  A negative Laplacian on its own is
unconditionally unstable — its growth ∝ +k² is largest at the grid scale.
Per Jansen & Held (2014) it is stabilised ONLY when paired with a
scale-SELECTIVE dissipation (a biharmonic ∇⁴ or Leith closure, whose
grid-scale damping ∝ k⁴ overwhelms the k² injection at high k) AND when the
reservoir ``E`` — hence ν_bs — is bounded (the ``E_max`` clamp).  Do NOT
enable backscatter against a purely harmonic (∇²) dissipation: the two
operators are the same order, so the flow noises up at the grid scale.

    A PRIOR implementation returned ``+∇²(ν_bs ∇²u)`` — an *anti-biharmonic*
    (growth ∝ +k⁴).  That is the WRONG scale selectivity: +k⁴ amplifies the
    GRID scale MOST (not the large scales), the opposite of a backscatter's
    purpose and worse at the grid scale than the negative Laplacian it was
    meant to tame.  Corrected here to the faithful negative-Laplacian form.

STATUS: opt-in (``BackscatterConfig.enabled=False`` by default) and NOT yet
wired into the production momentum update — the operators here are exercised
by the unit tests only, pending an eddy-permitting stability validation
before coupling into the ocean dynamics.

References
----------
- Jansen, M. F. & Held, I. M. (2014)  Parameterizing subgrid-scale eddy
  effects using energetically consistent backscatter.  Ocean Modelling 80,
  36–48.  doi:10.1016/j.ocemod.2014.06.002
- Bachman, S. D. (2019)  The GM+E closure: A framework for
  coupling backscatter with the Gent and McWilliams parameterization.
  Ocean Modelling 136, 85–106.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    viscous_tendency_cgrid,
    vertex_area_1d,
)
from legoesm.core.operators_voronoi import vector_laplacian_del2_3d

__physics_contract__ = {
    "summary": (
        "Jansen & Held (2014) deterministic energy backscatter: a negative "
        "(anti-diffusive) Laplacian viscosity du/dt = -nu_bs*grad^2(u) "
        "(nu_bs = c_bs*Delta*sqrt(E) >= 0) re-injects resolved kinetic energy "
        "drawn from a prognostic subgrid-KE reservoir E."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "E": "m^2/s^2 (subgrid KE reservoir)",
        "cfg.c_bs": "1 (dimensionless)",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "E_new": "m^2/s^2",
    },
    "sign_convention": (
        "nu_bs >= 0 but applied with the OPPOSITE sign to diffusion (negative "
        "Laplacian), so the tendency INJECTS resolved KE (growth ~ +nu_bs*k^2) "
        "and dissipates enstrophy; the reservoir E evolves as (resolved viscous "
        "dissipation source) - (backscatter sink) - (linear damping), clamped "
        "to [E_min, E_max]. Does NOT conserve resolved KE (energy is drawn FROM "
        "the subgrid reservoir); energetic consistency holds only over the full "
        "dissipation<->reservoir<->backscatter cycle."
    ),
    # Negative viscosity INJECTS resolved KE — not a conservative operator.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Jansen, M. F. & Held, I. M. (2014), Ocean Modelling 80, 36-48, "
        "doi:10.1016/j.ocemod.2014.06.002; Bachman (2019), Ocean Modelling 136"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_backscatter.py — the momentum tendency INCREASES "
        "resolved KE (sum u.du_dt.area >= 0) for E>0; E=0 or disabled gives zero "
        "tendency; the reservoir stays within [E_min, E_max]."
    ),
}

_EPS = float(jnp.finfo(jnp.float32).eps)


__param_spec__ = {
    "BackscatterConfig": {
        "scheme_key": "ocean.backscatter",
        "excluded": {
            "E_min": "default 0 = disabled/off (enable via config, not training)",
            "nu_bs_cfl_safety": "numerics viscous-CFL safety factor that BOUNDS the negative viscosity (a stability cap, not a tunable closure knob)",
        },
        "params": {
            "E_max": {"units": "m^2/s^2", "bounds": (0.033, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Jansen-Held energy backscatter", "shape": None},
            "c_bs": {"units": "1", "bounds": (0.0033, 0.03), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Jansen-Held energy backscatter", "shape": None},
            "efficiency": {"units": "1", "bounds": (0.3, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Jansen-Held energy backscatter", "shape": None},
            "tau_relax_days": {"units": "days", "bounds": (3.3, 30.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Jansen-Held energy backscatter", "shape": None},
        },
    },
}


class BackscatterConfig(NamedTuple):
    """Configuration for the Jansen–Held (2014) energy-backscatter closure.

    Parameters
    ----------
    enabled : bool
        If ``False`` (default) the closure is a no-op and neither the
        momentum tendency nor the reservoir are modified.
    c_bs : float
        Dimensionless backscatter coefficient setting the HARMONIC
        (negative-Laplacian) viscosity ``ν_bs = c_bs · Δ · √E`` [m²/s]
        (``Δ = √area`` the grid length; same meaning on the C-grid and
        MPAS paths).  Typical values: 0.001 – 0.05.  Must be non-negative.
    tau_relax_days : float
        e-folding timescale for the linear damping of ``E`` [days].
    E_min : float
        Floor on the reservoir energy [m²/s²]; 0.0 is fine.
    E_max : float
        Ceiling on the reservoir energy [m²/s²].  The exact value is
        not critical; it only prevents runaway if the dissipation
        estimate is noisy.
    efficiency : float
        Fraction of the resolved viscous dissipation routed into ``E``
        (the remainder is treated as genuine dissipation).  Default
        0.9 following Jansen & Held 2014.  Must be in ``[0, 1]``.
    nu_bs_cfl_safety : float
        Viscous-CFL safety factor bounding the backscatter viscosity
        ``ν_bs`` [-].  The negative Laplacian is destabilising, so ``ν_bs``
        is capped per-cell at ``nu_bs_cfl_safety · Δ² / Δt`` (the SAME
        Laplacian viscous-CFL ceiling used by the Smagorinsky closure via
        ``laplacian_smag_cfl_cap``) in addition to the ``E_max`` clamp.
        This is a numerics stability floor, NOT a tunable closure knob.
    """
    enabled: bool = False
    c_bs: float = 0.01
    tau_relax_days: float = 10.0
    E_min: float = 0.0
    E_max: float = 0.1
    efficiency: float = 0.9
    nu_bs_cfl_safety: float = 0.5


# ---------------------------------------------------------------------------
# LatLon C-grid backscatter
# ---------------------------------------------------------------------------


def _A_bs_h_cgrid(E: jnp.ndarray, grid, c_bs: float) -> jnp.ndarray:
    """Backscatter coefficient ``ν_bs = c_bs · Δ · √E`` at h-points.

    Units: ``Δ = √area`` is in m, ``√E`` is in m/s, so ``ν_bs`` has
    harmonic-viscosity units m²/s — consumed by the SINGLE (negated)
    stress-divergence to give the negative-Laplacian ``-ν_bs ∇²u``.
    """
    Delta = jnp.sqrt(grid.area)                         # Δ = length [m]
    return c_bs * Delta * jnp.sqrt(jnp.maximum(E, 0.0) + 1e-30)


def _A_bs_q_cgrid(E: jnp.ndarray, grid, c_bs: float) -> jnp.ndarray:
    """Backscatter coefficient ``ν_bs = c_bs · Δ_q · √E_q`` at q-points.

    Interpolates ``E`` from cell centres to vertices via the 4-cell
    average, pads pole rows with zero, and appends the wrap column so
    the shape matches ``(n_lat+1, n_lon+1)``.  Uses ``Δ_q = √A_vertex``
    (length) to match ``_A_bs_h_cgrid``.
    """
    n_lat, n_lon = E.shape[-2:]
    # 4-cell average interior
    E_wrap = jnp.roll(E, 1, axis=-1)
    E_q_int = 0.25 * (E[..., :-1, :] + E[..., 1:, :]
                      + E_wrap[..., :-1, :] + E_wrap[..., 1:, :])
    # Pole rows zero (wall BC); single Pad HLO op replaces alloc-zeros
    # + concatenate-of-three.
    pad_axes = ((0, 0),) * (E_q_int.ndim - 2)
    E_q = jnp.pad(E_q_int, (*pad_axes, (1, 1), (0, 0)))
    E_q = jnp.concatenate([E_q, E_q[..., 0:1]], axis=-1)

    A_vert = vertex_area_1d(grid)                         # (n_lat+1,) [m²]
    Delta_q = jnp.sqrt(A_vert)[:, jnp.newaxis]           # length [m]
    return c_bs * Delta_q * jnp.sqrt(jnp.maximum(E_q, 0.0) + 1e-30)


def backscatter_tendency_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    E: jnp.ndarray,
    grid,
    cfg: BackscatterConfig,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Energy-backscatter momentum tendency on the lat-lon C-grid.

    Negative-Laplacian (anti-diffusive) backscatter: returns the tendency
    the caller should ADD to ``du/dt`` (``du/dt += tend`` realises
    ``-ν_bs ∇²u``, an energy INJECTION ∝ +ν_bs k²).  A SINGLE
    ``viscous_tendency_cgrid`` harmonic stress-divergence — the exact
    discrete adjoint of ``strain_rate_cgrid`` — is negated, so the
    injection is energy-consistent with the resolved-scale viscous
    closures to machine precision.  (This replaced a two-pass
    ``+∇²(A∇²u)`` anti-biharmonic; see the module docstring.)

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    E : (n_lat, n_lon) — depth-mean (per-mass) subgrid KE reservoir [m²/s²].
    grid : LatLonGrid.
    cfg : BackscatterConfig.
    """
    if not cfg.enabled or cfg.c_bs == 0.0:
        return jnp.zeros_like(u), jnp.zeros_like(v)

    is_3d = u.ndim == 3

    # A_bs at h-points and q-points.
    A_h = _A_bs_h_cgrid(E, grid, cfg.c_bs)              # (n_lat, n_lon)
    A_q = _A_bs_q_cgrid(E, grid, cfg.c_bs)              # (n_lat+1, n_lon+1)

    if is_3d:
        # ``viscous_tendency_cgrid`` vmaps over the last axis of 3-D
        # coefficients; a trailing singleton would give vmap-size 1
        # which mismatches the velocity's ``nlev > 1`` axis.  Broadcast
        # to the full nlev so each level sees the same depth-
        # mean coefficient (E is a per-mass, depth-mean reservoir).
        nlev = u.shape[-1]
        A_h = jnp.broadcast_to(A_h[..., jnp.newaxis],
                                A_h.shape + (nlev,))
        A_q = jnp.broadcast_to(A_q[..., jnp.newaxis],
                                A_q.shape + (nlev,))

    # Negative-Laplacian backscatter: a SINGLE A_bs-weighted harmonic
    # stress-divergence.  ``viscous_tendency_cgrid`` returns the harmonic
    # viscous tendency that is DISSIPATIVE when added
    # (``sum u·tend·area = -sum A_h·D_T²·area - sum A_q·D_S²·A_vert ≤ 0``);
    # NEGATING it and letting the caller ADD (``du/dt += tend``) flips the
    # sign to an anti-diffusion ``du/dt = -∇·(A_bs ∇u) = -A_bs ∇²u`` that
    # INJECTS energy (``sum u·(-tend)·area ≥ 0``, growth ∝ +A_bs k²).  This
    # is the Jansen & Held (2014) negative-viscosity backscatter — a
    # harmonic (∇²) operator, NOT the earlier anti-biharmonic two-pass
    # ``+∇²(A∇²u)`` (∝ +k⁴) which amplified the grid scale most.
    diss_u, diss_v = viscous_tendency_cgrid(
        u, v, grid, A_h, A_q,
        mask=mask, u_mask=u_mask, v_mask=v_mask,
        normalize=True)

    return -diss_u, -diss_v


# ---------------------------------------------------------------------------
# MPAS (TRiSK) backscatter
# ---------------------------------------------------------------------------


def backscatter_tendency_mpas(
    u_edge_3d: jnp.ndarray,
    E_cell: jnp.ndarray,
    mesh,
    cfg: BackscatterConfig,
) -> jnp.ndarray:
    """Energy-backscatter edge-normal velocity tendency on an MPAS mesh.

    ``-ν_bs ∇²u`` (negative-Laplacian backscatter) with the HARMONIC
    coefficient ``ν_bs = c_bs · Δ_e · √E`` [m²/s] at edges.  ``E_cell`` is
    interpolated to edges by a two-cell average.

    Parameters
    ----------
    u_edge_3d : (nEdges, nlev) edge-normal velocity.
    E_cell : (nCells,) or (nCells, nlev) subgrid-KE reservoir.  If 2D
        a per-level backscatter is produced; if 1D the same coefficient
        is used at every vertical level.
    mesh : VoronoiMesh.
    cfg : BackscatterConfig.
    """
    if not cfg.enabled or cfg.c_bs == 0.0:
        return jnp.zeros_like(u_edge_3d)

    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    # Interpolate E to edges
    if E_cell.ndim == 1:
        E_edge = 0.5 * (E_cell[c1] + E_cell[c2])
        E_edge = E_edge[:, jnp.newaxis]
    else:
        E_edge = 0.5 * (E_cell[c1] + E_cell[c2])         # (nEdges, nlev)

    # Negative-Laplacian backscatter: -ν_bs ∇²u, a SINGLE vector Laplacian
    # (vector_laplacian_del2_3d ONCE) with the anti-diffusive sign.  ∇²
    # carries units [1/m²], so the HARMONIC coefficient ν_bs = c_bs·Δ·√E
    # [m²/s] gives a momentum tendency [u/s].  ``c_bs`` now has the SAME
    # meaning as the C-grid path.  Added by the caller (``du/dt += tend``)
    # the minus sign injects energy (growth ∝ +ν_bs k²) — Jansen & Held
    # (2014).  (The prior form ``+∇²(A∇²u)`` with A = c_bs·Δ³·√E [m⁴/s] was
    # an anti-biharmonic ∝ +k⁴ that amplified the grid scale most; see the
    # module docstring.)
    #
    # COEFFICIENT PLACEMENT (ν_bs OUTSIDE the Laplacian, at edges): this is
    # deliberately the exact NEGATION of the codebase's wired MPAS harmonic
    # viscosity ``smagorinsky_laplacian_3d`` (``A_e·∇²u``,
    # operators_voronoi.py; used in ocean_pe_mpas.py), so backscatter returns
    # energy in the SAME discretization the paired MPAS dissipation removed it
    # — the energetically-consistent (Jansen & Held) requirement.  Using the
    # strict exact-adjoint ``-∇·(ν∇u)`` here instead would make injection and
    # dissipation use DIFFERENT operators and break that consistency.  The
    # cost: for spatially varying ν_bs the edge-coefficient form is
    # sign-definite only to O(∇ν_bs·∇u) (the strict pointwise stress-tensor
    # identity holds on the C-grid path via ``viscous_tendency_cgrid``, whose
    # cgrid dissipation IS exact-adjoint); on MPAS, as for every
    # edge-coefficient viscosity here, the net injection is bounded globally
    # by the ``E_max`` reservoir cap rather than guaranteed pointwise.
    delta_edge = jnp.sqrt(mesh.dcEdge * mesh.dvEdge)     # Δ [m]
    nu_bs = cfg.c_bs * delta_edge[:, jnp.newaxis] * jnp.sqrt(
        jnp.maximum(E_edge, 0.0) + 1e-30)                # (nEdges, nlev) [m²/s]

    del2_u = vector_laplacian_del2_3d(u_edge_3d, mesh)   # ∇²u  [u/m²]
    return -nu_bs * del2_u                                # -ν_bs ∇²u  [u/s]


# ---------------------------------------------------------------------------
# Energy budget helpers
# ---------------------------------------------------------------------------


def backscatter_power_density_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    tend_u: jnp.ndarray,
    tend_v: jnp.ndarray,
    grid,
    *,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    dz: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-cell backscatter power density ε_bs(x, y) [m²/s³].

    Computed as ``u · tend_u + v · tend_v`` averaged to cell centres.
    Sign convention: positive ``tend`` means the backscatter ADDS
    energy to the resolved flow, so ``ε_bs > 0`` is an energy sink
    for the reservoir.

    For 3-D inputs the result is the depth-MEAN per-mass power
    ``Σ_k (u·t_u + v·t_v)_k · dz_k / Σ_k dz_k`` [m²/s³] — the per-mass
    quantity that shares units with the per-mass EKE reservoir budget in
    ``update_eddy_energy`` (NOT the depth-INTEGRATED column power [m³/s³],
    which would be a factor of the column thickness too large).  When ``dz``
    is ``None`` a uniform ``dz = 1`` is used, so the depth-mean reduces to the
    level-mean.  Pass the actual layer-thickness array ``(nlev,)`` or
    ``(n_lat, n_lon, nlev)`` for z*-ocean grids with non-uniform vertical
    spacing.
    """
    is_3d = u.ndim == 3

    # Promote 2-D face masks to broadcast against 3-D fields so the
    # helper works with the standard ``compute_face_masks()`` output.
    if u_mask is not None and is_3d and u_mask.ndim == 2:
        u_mask_b = u_mask[..., None]
    else:
        u_mask_b = u_mask
    if v_mask is not None and is_3d and v_mask.ndim == 2:
        v_mask_b = v_mask[..., None]
    else:
        v_mask_b = v_mask

    u_eff = u if u_mask_b is None else u * u_mask_b
    v_eff = v if v_mask_b is None else v * v_mask_b

    # <u, tend_u> at u-face ⇒ average to cell centre.  ``u`` lives on
    # east faces along axis 1 (shape n_lon+1); ``v`` lives on north
    # faces along axis 0 (shape n_lat+1).  Use explicit axis slices so
    # the helper works for both 2-D ``(n_lat, n_lon±)`` and 3-D
    # ``(n_lat, n_lon±, nlev)`` inputs.
    up_tu = u_eff * tend_u
    vp_tv = v_eff * tend_v

    # u at cell centres: average the two bracketing u-faces.
    u_cell = 0.5 * (jnp.take(up_tu, indices=jnp.arange(up_tu.shape[1] - 1),
                             axis=1)
                    + jnp.take(up_tu,
                                indices=jnp.arange(1, up_tu.shape[1]),
                                axis=1))
    # v at cell centres: average the two bracketing v-faces.
    v_cell = 0.5 * (jnp.take(vp_tv, indices=jnp.arange(vp_tv.shape[0] - 1),
                             axis=0)
                    + jnp.take(vp_tv,
                                indices=jnp.arange(1, vp_tv.shape[0]),
                                axis=0))

    power = u_cell + v_cell                              # (n_lat, n_lon[, nlev])
    if is_3d:
        # Depth-MEAN ``u·tend`` — a PER-MASS power density [m²/s³] that
        # matches the units of the per-mass EKE reservoir budget consumed in
        # ``update_eddy_energy`` (dE/dt = η·ε_diss − ε_bs − E/τ, with E in
        # [m²/s²]).  Returning the depth-INTEGRATED column power
        # (Σ_k u·tend·dz_k, units [m³/s³]) would be a factor of the column
        # thickness H too large and would immediately pin E to E_max on a deep
        # column.  Divide the dz-weighted sum by the column thickness Σ_k dz_k
        # (guarded strictly positive so a dry/zero-thickness column stays 0).
        if dz is None:
            # Uniform dz = 1 ⇒ depth-mean = level-mean.
            power = jnp.mean(power, axis=-1)
        else:
            dz_b = jnp.asarray(dz)                       # (nlev,) or (…, nlev)
            col_thickness = jnp.maximum(
                jnp.sum(jnp.broadcast_to(dz_b, power.shape), axis=-1), _EPS)
            power = jnp.sum(power * dz_b, axis=-1) / col_thickness
    return power


def update_eddy_energy(
    E: jnp.ndarray,
    dt: float,
    eps_dissipation: jnp.ndarray,
    eps_backscatter: jnp.ndarray,
    cfg: BackscatterConfig,
) -> jnp.ndarray:
    """Forward-Euler update of the subgrid-KE reservoir.

    ``E_new = clamp(E + dt·(η · ε_diss - ε_bs - E/τ), E_min, E_max)``
    where ``η = cfg.efficiency`` and ``τ = cfg.tau_relax_days · 86400``.

    When ``cfg.enabled = False`` — or when ``cfg.c_bs == 0.0`` so the
    momentum tendency is zero — the helper is a no-op.  Callers can
    invoke it unconditionally each step without perturbing ``E``.

    All non-disabled inputs share the same horizontal shape (typically
    ``(n_lat, n_lon)`` or ``(nCells,)``).
    """
    if (not cfg.enabled) or cfg.c_bs == 0.0:
        return E
    tau = cfg.tau_relax_days * 86400.0
    eta = cfg.efficiency
    dE = eta * eps_dissipation - eps_backscatter - E / jnp.maximum(tau, _EPS)
    E_new = E + dt * dE
    return jnp.clip(E_new, cfg.E_min, cfg.E_max)


# ---------------------------------------------------------------------------
# Diagnostic (no-carry) reservoir + CFL bound + C-grid wiring
# ---------------------------------------------------------------------------
#
# The production momentum update wires backscatter through a DIAGNOSTIC
# (standing-equilibrium) reservoir E rather than the prognostic, time-
# integrated E of ``update_eddy_energy``.  A carried E would need a new field
# threaded through the outer ``lax.scan`` and seeded to a constant pytree
# BEFORE the scan — the None→Field carry-structure hazard documented in
# ``ocean_pe_latlon_cgrid`` (the F_slow_prev seed).  The diagnostic reservoir
# needs NO carry: it is the local equilibrium of the reservoir budget, so the
# wiring is a pure function of the instantaneous state.  FOLLOW-UP: swap
# ``diagnostic_eddy_energy`` for a carried E updated by ``update_eddy_energy``
# (constant-pytree seed, mirroring the ``state.eke`` carry) once an eddy-
# permitting stability validation of the prognostic path is available.


def diagnostic_eddy_energy(
    eps_diss: jnp.ndarray, cfg: BackscatterConfig
) -> jnp.ndarray:
    """DIAGNOSTIC (no-carry) subgrid-KE reservoir from the instantaneous
    resolved scale-selective dissipation.

    Local-equilibrium closure of the prognostic reservoir budget
    ``dE/dt = η·ε_diss − ε_bs − E/τ``: dropping the (small) backscatter
    return ``ε_bs`` and setting ``dE/dt = 0`` gives the standing reservoir
    ``E_eq = η·τ·ε_diss`` [m²/s²] (units: [1]·[s]·[m²/s³]).  Used INSTEAD of a
    carried, time-integrated ``E`` (``update_eddy_energy``) so the closure
    needs NO prognostic field threaded through the outer ``lax.scan`` —
    avoiding the None→Field carry-structure hazard.  Clamped to
    ``[E_min, E_max]``.

    Parameters
    ----------
    eps_diss : resolved scale-selective (biharmonic/Leith) dissipation power
        density [m²/s³], ``≥ 0`` (energy REMOVED from the resolved flow).
    cfg : BackscatterConfig.

    Returns
    -------
    E : subgrid-KE reservoir [m²/s²], clamped to ``[E_min, E_max]``.
    """
    tau = cfg.tau_relax_days * 86400.0
    E = cfg.efficiency * tau * jnp.maximum(eps_diss, 0.0)
    return jnp.clip(E, cfg.E_min, cfg.E_max)


def cfl_cap_eddy_energy(
    E: jnp.ndarray,
    nu_max: jnp.ndarray,
    delta: jnp.ndarray,
    cfg: BackscatterConfig,
) -> jnp.ndarray:
    """Cap ``E`` so the backscatter viscosity ``ν_bs = c_bs·Δ·√E`` never
    exceeds the per-cell viscous-CFL ceiling ``nu_max`` [m²/s].

    A negative Laplacian is destabilising (module docstring), so ``ν_bs``
    MUST be bounded.  ``ν_bs ≤ nu_max`` ⇔ ``E ≤ (nu_max/(c_bs·Δ))²``; the
    returned ``min(E, E_cap)`` enforces this POINTWISE, while the ``E_max``
    clamp in ``diagnostic_eddy_energy``/``update_eddy_energy`` supplies the
    second, dt-INDEPENDENT ceiling.  ``Δ`` MUST be the SAME grid length the
    operator uses (``√area`` at the C-grid h-points, ``_A_bs_h_cgrid``), so
    the cap cancels the operator's ``Δ`` exactly and bounds ``ν_bs`` to
    ``nu_max`` regardless of the metric ``nu_max`` was built from.

    Parameters
    ----------
    E : candidate reservoir [m²/s²] (e.g. from ``diagnostic_eddy_energy``).
    nu_max : per-cell viscous-CFL ceiling on ``ν_bs`` [m²/s]
        (e.g. ``laplacian_smag_cfl_cap(grid, dt, cfg.nu_bs_cfl_safety)[0]``).
    delta : grid length ``Δ`` [m] at the same points ``E`` lives on.
    cfg : BackscatterConfig.
    """
    denom = jnp.maximum(cfg.c_bs * delta, _EPS)
    E_cap = (nu_max / denom) ** 2
    return jnp.minimum(E, E_cap)


def diagnostic_backscatter_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    diss_u: jnp.ndarray,
    diss_v: jnp.ndarray,
    nu_max_h: jnp.ndarray,
    grid,
    cfg: BackscatterConfig,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    dz: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Diagnostic-``E`` (no-carry), CFL-bounded Jansen–Held backscatter
    momentum tendency on the lat-lon C-grid.

    Orchestrates the reuse-only pipeline (no new numerics):

    1. resolved scale-selective dissipation power ``ε_diss = −⟨u·D_diss⟩``
       [m²/s³] at cell centres from the APPLIED biharmonic/Leith tendency
       ``(diss_u, diss_v)`` via :func:`backscatter_power_density_cgrid`;
    2. standing-equilibrium reservoir ``E = clamp(η·τ·ε_diss)`` via
       :func:`diagnostic_eddy_energy`;
    3. per-cell CFL cap ``ν_bs ≤ nu_max_h`` via :func:`cfl_cap_eddy_energy`
       (``Δ = √area``, the h-point grid length
       :func:`backscatter_tendency_cgrid` uses);
    4. the negative-Laplacian tendency via the existing
       :func:`backscatter_tendency_cgrid` (UNCHANGED).

    ``diss_u/diss_v`` are the APPLIED (already-signed, dissipative) lateral
    tendencies [m/s²]; a dissipative tendency gives ``⟨u·D⟩ ≤ 0`` so
    ``ε_diss ≥ 0`` and backscatter injects ONLY where a scale-selective sink
    is active — the Jansen & Held (2014) energetic-consistency requirement
    (and the stability pairing the module docstring mandates).

    Returns ``(tend_u, tend_v, E)`` — add ``tend_u/tend_v`` to ``du/dt``,
    ``dv/dt``; ``E`` [m²/s²] is the CFL-capped reservoir (diagnostic output).
    A no-op (zeros) when ``cfg.enabled`` is False or ``cfg.c_bs == 0``.
    """
    if not cfg.enabled or cfg.c_bs == 0.0:
        return (jnp.zeros_like(u), jnp.zeros_like(v),
                jnp.zeros_like(grid.area))
    # (1) resolved scale-selective dissipation power ε_diss ≥ 0 [m²/s³].
    #     ⟨u·D⟩ ≤ 0 for a dissipative tendency ⇒ ε_diss = −⟨u·D⟩ ≥ 0.
    power = backscatter_power_density_cgrid(
        u, v, diss_u, diss_v, grid,
        u_mask=u_mask, v_mask=v_mask, dz=dz)
    eps_diss = jnp.maximum(-power, 0.0)                  # (n_lat, n_lon)
    # (2) standing-equilibrium reservoir E = clamp(η·τ·ε_diss, E_min, E_max).
    E = diagnostic_eddy_energy(eps_diss, cfg)
    # (3) per-cell CFL cap: ν_bs = c_bs·√area·√E ≤ nu_max_h.
    delta_h = jnp.sqrt(grid.area)                        # h-point Δ [m]
    E = cfl_cap_eddy_energy(E, nu_max_h, delta_h, cfg)
    # (4) negative-Laplacian backscatter tendency (existing operator).
    tend_u, tend_v = backscatter_tendency_cgrid(
        u, v, E, grid, cfg, mask=mask, u_mask=u_mask, v_mask=v_mask)
    return tend_u, tend_v, E
