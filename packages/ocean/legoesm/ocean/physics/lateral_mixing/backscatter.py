"""Deterministic energy backscatter (Jansen & Held 2014) for the ocean.

The backscatter scheme re-injects kinetic energy that has been removed by
the resolved-scale viscous closures (Smagorinsky, Leith, biharmonic).  A
prognostic, depth-integrated subgrid kinetic-energy reservoir ``E(x, y)``
evolves under

    dE/dt = ε_dissipation(x, y)        (source — from the resolved dyn)
          - ε_backscatter(x, y)        (sink   — returned to the resolved dyn)
          - E / τ_relax                (linear damping of the reservoir)

with ``E`` clamped to ``[E_min, E_max]`` each step.  The backscatter
momentum tendency is the *negative* (sign-flipped) biharmonic Laplacian

    ∂u/∂t|_bs = + ∇²(ν_bs ∇² u)

where the coefficient is driven by the local eddy-energy level:

    ν_bs(x, y) = c_bs · Δ² · √E(x, y),   ν_bs ≥ 0.

The "+" sign (as opposed to the "-" used by the Smagorinsky biharmonic)
makes this an *anti*-biharmonic that injects energy into the resolved
flow.  The biharmonic form is scale-selective — it amplifies large
wavelengths much more than the grid scale, avoiding the unconditional
instability of plain anti-Laplacian backscatter.

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

import jax
import jax.numpy as jnp

from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    stress_divergence_cgrid,
    strain_rate_cgrid,
    viscous_tendency_cgrid,
    _vertex_area,
)
from legoesm.core.operators_voronoi import vector_laplacian_del2_3d

_EPS = float(jnp.finfo(jnp.float32).eps)


class BackscatterConfig(NamedTuple):
    """Configuration for the Jansen–Held (2014) energy-backscatter closure.

    Parameters
    ----------
    enabled : bool
        If ``False`` (default) the closure is a no-op and neither the
        momentum tendency nor the reservoir are modified.
    c_bs : float
        Dimensionless backscatter coefficient applied to the
        biharmonic coefficient ``ν_bs = c_bs · Δ² · √E``.  Typical
        values: 0.001 – 0.05.  Must be non-negative.
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
    """
    enabled: bool = False
    c_bs: float = 0.01
    tau_relax_days: float = 10.0
    E_min: float = 0.0
    E_max: float = 0.1
    efficiency: float = 0.9


# ---------------------------------------------------------------------------
# LatLon C-grid backscatter
# ---------------------------------------------------------------------------


def _A_bs_h_cgrid(E: jnp.ndarray, grid, c_bs: float) -> jnp.ndarray:
    """Backscatter coefficient ``ν_bs = c_bs · Δ · √E`` at h-points.

    Units: ``Δ = √area`` is in m, ``√E`` is in m/s, so ``ν_bs`` has
    harmonic-viscosity units m²/s.  The two-pass stress-divergence that
    consumes it then produces biharmonic-scaling tendencies.
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

    A_vert = _vertex_area(grid)                         # (n_lat+1,) [m²]
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

    Returns the tendency that the caller should ADD to ``du/dt`` (note
    the sign convention differs from the Smagorinsky-biharmonic helper,
    which is subtracted).  Implementation mirrors the MOM6 two-pass
    stress formulation used by
    ``smagorinsky_biharmonic_tendency_cgrid`` so the operator is the
    exact discrete adjoint of ``strain_rate_cgrid`` and therefore
    energy-consistent with the resolved-scale viscous closures.

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    E : (n_lat, n_lon) — depth-integrated subgrid KE reservoir [m²/s²].
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
        # integrated coefficient (E is already column-integrated).
        nlev = u.shape[-1]
        A_h = jnp.broadcast_to(A_h[..., jnp.newaxis],
                                A_h.shape + (nlev,))
        A_q = jnp.broadcast_to(A_q[..., jnp.newaxis],
                                A_q.shape + (nlev,))

    # First pass: unit-coefficient, UNNORMALISED stress-divergence.
    u_star, v_star = viscous_tendency_cgrid(
        u, v, grid, 1.0, 1.0,
        mask=mask, u_mask=u_mask, v_mask=v_mask,
        normalize=False)

    # Second pass: A_bs-weighted NORMALISED stress-divergence.  The
    # returned ``(tend_u, tend_v)`` has the SAME sign as the
    # Smagorinsky-biharmonic tendency — which the Smag caller
    # *subtracts* (``du/dt -= tend``) to dissipate.  Our caller instead
    # *adds* (``du/dt += tend``), which gives ``du/dt = +∇²(A ∇²u)``
    # and injects kinetic energy into the resolved flow.
    tend_u, tend_v = viscous_tendency_cgrid(
        u_star, v_star, grid, A_h, A_q,
        mask=mask, u_mask=u_mask, v_mask=v_mask,
        normalize=True)

    return tend_u, tend_v


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

    ``+∇²(ν_bs · ∇²u)`` with ``ν_bs = c_bs · Δ_e² · √E`` at edges.
    ``E_cell`` is interpolated to edges by a two-cell average.

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

    # Geometric-mean edge length — already a length (m), not area.
    # Units: ``√(E_edge)`` [m/s] × ``delta_edge`` [m] = ``m²/s``
    # (harmonic viscosity), which the biharmonic two-pass operator
    # consumes.
    delta_edge = jnp.sqrt(mesh.dcEdge * mesh.dvEdge)
    Delta = delta_edge[:, jnp.newaxis]
    A_bs = cfg.c_bs * Delta * jnp.sqrt(
        jnp.maximum(E_edge, 0.0) + 1e-30)                # (nEdges, nlev)

    del2_u = vector_laplacian_del2_3d(u_edge_3d, mesh)
    intermediate = A_bs * del2_u
    # Smag-biharm returns -∇²(A ∇²u); we want +∇²(A ∇²u), hence the
    # explicit positive sign.
    return vector_laplacian_del2_3d(intermediate, mesh)


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

    For 3-D inputs the result is the depth-integrated per-column power
    ``Σ_k (u·t_u + v·t_v)_k · dz_k``.  When ``dz`` is ``None`` a
    uniform ``dz = 1`` is used (equivalent to summing the per-level
    densities).  Pass the actual layer-thickness array
    ``(nlev,)`` or ``(n_lat, n_lon, nlev)`` for z*-ocean grids with
    non-uniform vertical spacing.
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
        # Depth-integrate ``u·tend`` with layer thickness ``dz`` to
        # get the column total used as the reservoir source/sink.
        # When ``dz=None`` we fall back to unit spacing, equivalent to
        # summing per-level densities (only exact for equal-thickness
        # layers).
        if dz is None:
            power = jnp.sum(power, axis=-1)
        else:
            dz_arr = jnp.asarray(dz)
            # Broadcast 1-D dz against (n_lat, n_lon, nlev).
            if dz_arr.ndim == 1:
                dz_b = dz_arr
            else:
                dz_b = dz_arr
            power = jnp.sum(power * dz_b, axis=-1)
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
