"""Rigid-lid barotropic streamfunction solver (lat-lon C-grid).

A faithful, differentiable re-implementation of Veros's rigid-lid barotropic
mode (``veros/core/external/solve_stream.py``) as a selectable
``barotropic_solver = "rigid_lid"`` option, for apples-to-apples fidelity with
the Veros oracle (the ACC transport's response to bottom drag differs between a
free surface and a rigid lid; see docs/ocean/fidelity/oracle_recipe_strategy.md).

Formulation
-----------
The rigid lid removes the free surface entirely.  The depth-integrated flow is
non-divergent and carried by a streamfunction ``ψ`` on vertex (corner) points::

    (H·u_bt, H·v_bt) = ∇⊥ψ      ⇒   u_bt = -(1/H)∂ψ/∂y ,  v_bt = +(1/H)∂ψ/∂x

with the column depth ``H`` FIXED at the bathymetry (no eta dependence).  Taking
the curl of the depth-integrated momentum equation eliminates the unknown
surface (lid) pressure and yields the elliptic vorticity equation for the
streamfunction tendency ``dψ ≡ ∂ψ/∂t``::

    L(dψ) = curl((1/H)∫F dz)         L(ψ) ≡ ∇·((1/H)∇ψ)

where ``F`` is the depth-integrated momentum forcing (every tendency except the
barotropic lid-pressure gradient).  ``L`` is the weighted vertex Laplacian
``streamfunction_vorticity_operator`` (= the vertex curl of the recovered
velocity, so operator and RHS share identical stencils).  ``ψ`` is then advanced
with Adams-Bashforth-2 (Veros ``AB_eps`` = ``config.ab2_epsilon``).

Multiply-connected domains
--------------------------
``L`` is singular: its null space is the constant on each disconnected land mass
(``island``).  The interior solve pins ``ψ = 0`` on all land (Dirichlet); the
per-island constants ``dpsin`` are recovered from the line-integral (circulation)
constraints ``line_psin · dpsin = line_forc`` (one constraint per free island).
For a periodic re-entrant channel (the ACC) the single free island constant *is*
the net channel transport — exactly the mode whose forcing balance (including
bottom drag, as a circulation integral) differs from the free-surface dynamics.

The static topology/depth data (``RigidLidStaticData``) is built once at model
construction (``rigid_lid_islands.build_rigid_lid_data``) from the fixed
bathymetry + land mask; it is captured by the model (a compile-time constant),
not carried in the ``lax.scan`` state.  Only ``psi`` and the tendency histories
``dpsi/dpsi_prev`` (interior) and ``dpsin/dpsin_prev`` (islands) are prognostic.

Differentiability
-----------------
The elliptic solve uses ``jax.scipy.sparse.linalg.cg`` on the symmetrized SPD
operator ``S = -A_vertex·L`` (see ``_symmetric_solve_operator``; CG replaced
Veros's BiCG-STAB because BiCG-STAB's adjoint broke down to NaN cotangent-
dependently), differentiable via the implicit-function theorem.  The VJP is
overridden to solve the adjoint system in the seam-REDUCED vertex space, where
``S`` is exactly Euclidean-symmetric — on the seam-redundant (n_lon+1) layout
it is NOT, which silently biased every reverse-mode gradient through the
ψ-solve by a few % up to 25% (see ``solve_streamfunction_interior``).  The
forward solve is bit-identical to the plain CG call.  The per-step dense
island solve is ``jnp.linalg.solve``.  The static data is not differentiated
through.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.ocean.vertical import compute_layer_thickness
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    curl_vertex_cgrid,
    recover_velocity_from_streamfunction,
    streamfunction_vorticity_operator,
    coriolis_cgrid,
    min_cell_to_uface,
    min_cell_to_vface,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    ab2_blend,
    column_depth,
    depth_average_to_faces,
)
from legoesm.parallel.reductions import is_multi_process

# Numerical floor for reciprocal depths / empty cells.
_DEPTH_FLOOR = 1.0e-10


def rigid_lid_is_decomposed() -> bool:
    """True if the domain is split across MPI ranks or an SPMD latitude-band mesh.

    The rigid-lid streamfunction solve is GLOBAL and single-rank only:
    ``solve_streamfunction_interior`` calls the stock
    ``jax.scipy.sparse.linalg.cg``, whose dot products are rank-local and whose
    residual-terminated ``while_loop`` runs a data-dependent iteration count, and
    ``island_line_integrals`` / ``_solve_island_constants`` reduce over the whole
    domain with a plain ``jnp.sum``.  None of these is MPI/SPMD-reduced, so a
    decomposed run silently converges each rank/band to its OWN sub-system (and
    the residual-dependent ``while_loop`` can desynchronise the collective
    schedule / deadlock).  This mirrors the multi-rank predicate used by
    ``barotropic_common._global_dot_batch`` / ``eta_floor._global_sum_pair`` —
    the SPMD lat-band shard_map path (single process, ``is_multi_process()`` is
    False) is caught FIRST via the ``"spmd"`` halo backend, then the mpi4jax /
    multi-host / Voronoi paths via ``is_multi_process()``.
    """
    # Function-scope import (matches eta_floor / barotropic_common): the halo
    # accessors live in ``grids.halo`` and are imported lazily to avoid an
    # import cycle at module load.
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh

    if get_halo_backend() == "spmd":
        mesh = get_spmd_mesh()
        # Backend armed "spmd" but no mesh set is an invalid state; the ocean
        # SPMD reductions FAIL FAST on it, so mirror that rather than silently
        # treating it as single-rank.
        if mesh is None:
            raise RuntimeError(
                "rigid_lid_is_decomposed: halo backend is 'spmd' but no SPMD "
                "mesh is set; arm it via activate_latlon_spmd_halo(mesh).")
        # A >1-device latitude axis is a real decomposition.  Keyed on "lat" BY
        # NAME so a coupled cube-atm SPMD mesh (ocean is never cube-sharded)
        # falls through to the is_multi_process() check below.
        if dict(mesh.shape).get("lat", 1) > 1:
            return True
    return is_multi_process()


def assert_rigid_lid_single_rank() -> None:
    """Raise if the rigid-lid barotropic solver is used on a decomposed domain.

    Fail-fast dispatch hardening: converts a silent wrong-answer / potential
    collective deadlock (see ``rigid_lid_is_decomposed``) into a loud error,
    pointing the user at the barotropic solvers that DO support MPI/SPMD
    decomposition.  Call it EAGERLY (host-side, before the jitted step / the
    rigid-lid data build) so a distributed run cannot reuse a serially-traced
    compiled step, and also in-body so direct traced entry points are covered.
    """
    if rigid_lid_is_decomposed():
        raise NotImplementedError(
            "barotropic_solver='rigid_lid' is single-rank only: the global "
            "streamfunction elliptic solve uses rank-local CG dot products in a "
            "residual-terminated while_loop, and the island line integrals sum "
            "over the whole domain — neither is MPI/SPMD-reduced, so a "
            "decomposed run silently converges each rank/band to its own "
            "sub-system (and the residual-dependent while_loop can desynchronise "
            "the collective schedule / deadlock). Use "
            "barotropic_solver='implicit_cn' (distributed fixed-iteration PCG) "
            "or 'explicit_substep' for multi-rank / SPMD-sharded ocean runs, or "
            "run the rigid lid on a single rank with an undecomposed mesh."
        )


class RigidLidStaticData(NamedTuple):
    """Static (time-independent) data for the rigid-lid streamfunction solve.

    Built once at model construction from the fixed bathymetry + land mask
    (``rigid_lid_islands.build_rigid_lid_data``).  All arrays are constants in
    the traced step — captured by the model, never carried in the scan state.

    Fields
    ------
    inv_H_u : (n_lat, n_lon+1) reciprocal column depth at u-faces [1/m].
    inv_H_v : (n_lat+1, n_lon) reciprocal column depth at v-faces [1/m].
    u_mask : (n_lat, n_lon+1) u-face ocean mask.
    v_mask : (n_lat+1, n_lon) v-face ocean mask.
    solve_mask : (n_lat+1, n_lon+1) vertex interior-solve mask — 1 where all 4
        surrounding cells are ocean (ψ solved), 0 elsewhere (ψ pinned: coast/
        land carry the island constant, = 0 for the interior particular solve).
    A_vertex : (n_lat+1, n_lon+1) vertex dual-cell area [m^2].
    inv_diag : (n_lat+1, n_lon+1) Jacobi preconditioner 1/diag(S) for the
        symmetric SPD solve operator S = -A_vertex·L (1 on pinned rows).  The
        elliptic solve uses CONJUGATE GRADIENT on S (not BiCG-STAB on L): L is
        self-adjoint only in the area-weighted inner product, so A_vertex·L is
        the standard-symmetric form and -A_vertex·L is SPD (the Laplacian is
        negative-definite).  CG's ``custom_linear_solve`` VJP solves the SAME
        symmetric system for the adjoint, so it is robust for every cotangent —
        unlike BiCG-STAB, whose adjoint breaks down to NaN cotangent-dependently
        (which broke the end-to-end jax.grad design goal).  diag(S) mixes O(1)
        pinned rows with the Laplacian rows, so the Jacobi preconditioner is also
        needed for conditioning.
    psin : (n_lat+1, n_lon+1, nisle) per-island streamfunction basis functions
        (psin[..., k] solves L=0 with value 1 on island k's vertices, 0 else).
    line_psin : (nisle, nisle) island circulation-coupling matrix
        line_psin[i, k] = circulation of psin[...,k]'s velocity around island i.
    island_vertex_masks : (nisle, n_lat+1, n_lon+1) — mask of vertices belonging
        to (being a corner of) each island's land, for the Stokes line integral.
    nisle : int — number of disconnected land masses (boundary components).
        Island 0 is the reference (dpsin[0] ≡ 0); islands 1.. carry free
        constants determined by the line-integral constraints.
    """

    inv_H_u: jnp.ndarray
    inv_H_v: jnp.ndarray
    u_mask: jnp.ndarray
    v_mask: jnp.ndarray
    solve_mask: jnp.ndarray
    A_vertex: jnp.ndarray
    inv_diag: jnp.ndarray
    psin: jnp.ndarray
    line_psin: jnp.ndarray
    island_vertex_masks: jnp.ndarray
    nisle: int


def barotropic_face_depths(H_bathy, land_mask, grid):
    """Fixed column depths + reciprocals at C-grid faces (rigid lid).

    The rigid-lid column depth is the (static) bathymetry — no free-surface
    dependence.  Face depths use the MOM6/MITgcm min-rule (the face is only as
    deep as its shallower neighbour cell), matching
    ``diagnostics_streamfunction.barotropic_streamfunction``.

    Parameters
    ----------
    H_bathy : (n_lat, n_lon) bathymetry depth [m], positive down.
    land_mask : (n_lat, n_lon) cell-center ocean mask (1=ocean, 0=land).
    grid : LatLonGrid.

    Returns
    -------
    (H_u, H_v, inv_H_u, inv_H_v) : column depths at u-/v-faces and reciprocals
        (inv = 0 on dry faces — guarded against the division by zero).
    """
    H = H_bathy * land_mask  # zero depth on land cells

    # u-faces (n_lat, n_lon+1): min of the two cells sharing the face; west edge
    # of cell i is stored at column i, periodic wrap appended as the last column.
    H_E = H
    H_W = jnp.roll(H, 1, axis=1)
    H_u_int = jnp.minimum(H_E, H_W)                              # (n_lat, n_lon)
    H_u = jnp.concatenate([H_u_int, H_u_int[:, 0:1]], axis=1)    # (n_lat, n_lon+1)

    # v-faces (n_lat+1, n_lon): min of the south/north cells; walls (poles) dry.
    H_v_int = jnp.minimum(H[:-1], H[1:])                         # (n_lat-1, n_lon)
    zero_row = jnp.zeros((1, H.shape[1]), dtype=H.dtype)
    H_v = jnp.concatenate([zero_row, H_v_int, zero_row], axis=0)  # (n_lat+1, n_lon)

    inv_H_u = jnp.where(H_u > _DEPTH_FLOOR, 1.0 / jnp.maximum(H_u, _DEPTH_FLOOR), 0.0)
    inv_H_v = jnp.where(H_v > _DEPTH_FLOOR, 1.0 / jnp.maximum(H_v, _DEPTH_FLOOR), 0.0)
    return H_u, H_v, inv_H_u, inv_H_v


def _symmetric_solve_operator(psi, rl_data, grid):
    """Symmetric SPD elliptic operator for the interior solve.

    ``S(ψ) = -A_vertex·L(ψ)`` on the interior-solve rows, identity on the pinned
    (coast/land) rows.  ``L = ∇·((1/H)∇)`` is self-adjoint only in the
    area-weighted inner product, so ``A_vertex·L`` is the standard-symmetric
    form and ``-A_vertex·L`` is symmetric positive-definite (the Laplacian is
    negative-definite); the identity on pinned rows keeps the system non-singular
    and SPD.  This is the operator solved by CG (its VJP is the same robust
    symmetric solve).  ψ is masked to the solve domain before applying ``L`` so
    the interior rows never couple to the pinned ψ (block-diagonal ⇒ symmetric).
    """
    psi_in = psi * rl_data.solve_mask
    Lpsi = streamfunction_vorticity_operator(
        psi_in, rl_data.inv_H_u, rl_data.inv_H_v, grid,
        u_mask=rl_data.u_mask, v_mask=rl_data.v_mask,
    )
    return jnp.where(rl_data.solve_mask > 0.5, -rl_data.A_vertex * Lpsi, psi)


def _seam_expand(v):
    """Expand a seam-reduced vertex field (n_lat+1, n_lon) to the redundant
    layout (n_lat+1, n_lon+1) by re-appending the periodic duplicate of
    column 0 as the wrap column."""
    return jnp.concatenate([v, v[:, :1]], axis=1)


def solve_streamfunction_interior(rhs, rl_data, grid, x0, *, tol, maxiter):
    """Solve L(dψ) = rhs on the wet interior (ψ=0 on land), via preconditioned CG.

    The system is recast in the symmetric SPD form ``S(dψ) = -A_vertex·rhs`` on
    the solve rows (``S = -A_vertex·L``, see ``_symmetric_solve_operator``) — the
    same solution as ``L(dψ)=rhs`` (both sides scaled by ``-A_vertex``).  CG with
    a Jacobi preconditioner (``rl_data.inv_diag = 1/diag(S)``) is used instead of
    BiCG-STAB precisely because CG's ``custom_linear_solve`` VJP solves the same
    symmetric system for the adjoint and is therefore robust for every cotangent
    (BiCG-STAB's adjoint breaks down to NaN cotangent-dependently, which broke
    the end-to-end jax.grad design goal).  Returns dψ (n_lat+1, n_lon+1), zero on
    land.

    Seam-reduced adjoint (custom VJP)
    ---------------------------------
    The vertex layout (n_lat+1, n_lon+1) stores the periodic wrap REDUNDANTLY:
    column ``n_lon`` duplicates column 0.  ``S`` maps wrap-consistent fields to
    wrap-consistent fields and equals the physical operator there, so the
    FORWARD solve is correct.  But as a matrix on the redundant space ``S`` is
    NOT Euclidean-symmetric: the operator/RHS stencils park the seam coupling
    on different redundant columns than their transpose partners — e.g. the row
    at vertex column 0 reads its west neighbour through the wrapped v-face
    (``jnp.roll``), whose stencil touches ψ columns ``n_lon-1`` **and**
    ``n_lon`` (the duplicate), while the row at column ``n_lon-1`` reads its
    east neighbour from column ``n_lon`` only — so ``S[0, n_lon-1] != 0`` but
    ``S[n_lon-1, 0] == 0`` (that weight sits in ``S[n_lon-1, n_lon]``).
    Measured: rel asymmetry up to 0.42 on wrap-consistent vectors, exactly
    symmetric (~1e-15) once the duplicate column is folded onto column 0
    (probe: tests/ocean/unit/test_rigid_lid_seam_adjoint.py).

    ``jax.scipy.sparse.linalg.cg``'s VJP (``lax.custom_linear_solve`` with
    ``transpose_solve = solve``) re-solves with ``S`` assuming Euclidean
    symmetry, so every reverse-mode gradient through the ψ-solve was biased —
    measured AD/FD 1.061 on a single-solve functional and 0.971 (KE) / 0.82
    (transport) on multi-step objectives in the adjoint-matching harness
    (.physics-validator/adjoint_oracle_match/RESULTS.md), eps-stable over 3
    decades of FD step.

    Fix (backward-only): the forward CG call is kept byte-identical (running
    the forward in the reduced space would change the Krylov path — the
    redundant inner products double-count the seam column — and perturb the
    solution at the tolerance level, ~1e-10 rel).  Only the VJP is overridden
    via ``jax.custom_vjp`` whose primal is the UNCHANGED stock CG call.
    (``lax.custom_linear_solve`` with an explicit ``transpose_solve`` was
    tried first and is mathematically equivalent, but the extra/restructured
    linear_solve primitive perturbs XLA fusion of ADJACENT step ops — the
    faithful ACC stack drifted in the last bit of T from step 2 while u/ψ
    stayed identical.  ``custom_vjp`` inlines the primal transparently, which
    A/B-verified bit-identical over 12 faithful-ACC steps + a global_4deg
    step.  CLOSURE RULE for scan-compatibility: the fwd/bwd closures must
    capture only CONCRETE arrays — any value COMPUTED from them in this
    function body, e.g. ``inv_diag[:, :-1]``, becomes a tracer when the step
    is traced inside ``lax.scan`` and a tracer captured in the bwd closure
    fails at lowering ("No constant handler for DynamicJaxprTracer"); slice
    INSIDE the closure body instead.)  With ``E`` (duplicate col 0 →
    wrap col) and ``P`` (drop wrap col), the solution map on the wrap-
    consistent inputs the model produces is ``F(rhs_sym) = E·S_r⁻¹·P·rhs_sym``
    with ``S_r = P·S·E`` exactly symmetric, so the adjoint is
    ``Fᵀ = Pᵀ·S_r⁻¹·Eᵀ``: fold the seam cotangent onto column 0 (``Eᵀ``), run
    the SAME preconditioned CG on the seam-reduced operator, and zero-pad the
    duplicate column (``Pᵀ`` — each physical dof's cotangent counted exactly
    once).  ``x0`` receives no cotangent, identical to stock ``cg`` (the
    Krylov guess does not affect the converged solution; IFT).  Verified
    AD/FD = 1 ± 2e-8 (was 1.061) with the forward bit-identical.

    Known limitation: ``custom_vjp`` does not support forward-mode AD, so
    ``jax.jvp``/``jacfwd`` through the ψ-solve now raises (stock ``cg``
    supported it).  Reverse mode is the end-to-end design goal and no repo
    path uses forward-mode through the ocean step (grep 2026-06-11); if one
    ever does, add a paired ``custom_jvp`` solving the tangent system with
    the same reduced operator.
    """
    # Leaf-level dispatch hardening: the stock CG below (and rigid_lid_step,
    # which reaches here) is single-rank only. Guard here so a DIRECT call
    # under decomposition also fails loudly, not just the model wrapper.
    assert_rigid_lid_single_rank()

    sm = rl_data.solve_mask
    rhs_sym = jnp.where(sm > 0.5, -rl_data.A_vertex * rhs, 0.0)
    x0_in = x0 * sm

    def op(psi):
        return _symmetric_solve_operator(psi, rl_data, grid)

    def precond(r):
        return r * rl_data.inv_diag

    # Seam-REDUCED operator/preconditioner for the adjoint solve: S_r = P·S·E
    # (drop the duplicate wrap column from the operator's domain and range).
    # Exactly Euclidean-symmetric, unlike S on the redundant layout.
    def op_reduced(v):
        return op(_seam_expand(v))[:, :-1]

    def precond_reduced(r):
        # Slice INSIDE the closure (see docstring CLOSURE RULE): hoisting
        # ``inv_diag[:, :-1]`` out captures a tracer under lax.scan tracing
        # and breaks jax.grad-through-scan at lowering.
        return r * rl_data.inv_diag[:, :-1]

    def _forward_cg(rhs_sym_in, x0_v):
        # The pre-fix forward, verbatim — custom_vjp inlines this primal
        # transparently (bit-identical; A/B-verified, see docstring).
        dpsi_f, _info = jax.scipy.sparse.linalg.cg(
            op, rhs_sym_in, x0=x0_v, tol=tol, atol=0.0, maxiter=maxiter,
            M=precond,
        )
        return dpsi_f

    @jax.custom_vjp
    def _cg_seam_adjoint(rhs_sym_in, x0_v):
        return _forward_cg(rhs_sym_in, x0_v)

    def _cg_fwd(rhs_sym_in, x0_v):
        return _forward_cg(rhs_sym_in, x0_v), None

    def _cg_bwd(_res, dpsi_bar):
        # Fᵀ = Pᵀ·S_r⁻¹·Eᵀ, solved in the seam-reduced space where S is
        # truly symmetric (re-solving with S on the redundant layout — what
        # stock cg's VJP does — IS the bug this fixes).
        # Eᵀ: fold the duplicate-column cotangent onto column 0, drop wrap col.
        g_reduced = dpsi_bar.at[:, 0].add(dpsi_bar[:, -1])[:, :-1]
        lam_reduced, _info = jax.scipy.sparse.linalg.cg(
            op_reduced, g_reduced, x0=jnp.zeros_like(g_reduced),
            tol=tol, atol=0.0, maxiter=maxiter, M=precond_reduced,
        )
        # Pᵀ: zero-pad the duplicate column — each physical dof exactly once.
        rhs_sym_bar = jnp.concatenate(
            [lam_reduced, jnp.zeros_like(lam_reduced[:, :1])], axis=1)
        # x0 cotangent is exactly zero (matches stock cg; IFT — the Krylov
        # guess does not affect the converged solution).
        return rhs_sym_bar, jnp.zeros_like(rhs_sym_bar)

    _cg_seam_adjoint.defvjp(_cg_fwd, _cg_bwd)

    dpsi = _cg_seam_adjoint(rhs_sym, x0_in)
    return dpsi * sm


def compute_operator_inv_diag(inv_H_u, inv_H_v, u_mask, v_mask, solve_mask,
                              A_vertex, grid):
    """1/diag(S) on solve vertices (1 on pinned), the CG Jacobi preconditioner.

    ``S = -A_vertex·L`` (the symmetric SPD solve operator) is a 5-point stencil
    (vertex v couples only to its N/S/E/W neighbours), so its diagonal is
    recovered EXACTLY by 4 colored probes — a 2x2 (i%2, j%2) checkerboard.  For
    each colour the probed vertices have no probed neighbour, so ``(S·probe)``
    restricted to the probed set equals the diagonal there.  Used only as a
    preconditioner, so the slight periodic-seam approximation when ``n_lon`` is
    odd is harmless (it only affects CG convergence rate, never the answer).
    """
    nlat1, nlon1 = solve_mask.shape
    ii, jj = jnp.meshgrid(jnp.arange(nlat1), jnp.arange(nlon1), indexing="ij")
    diag = jnp.zeros_like(solve_mask)
    for a in (0, 1):
        for b in (0, 1):
            color = (((ii % 2) == a) & ((jj % 2) == b)).astype(solve_mask.dtype)
            color = color * solve_mask
            Lc = streamfunction_vorticity_operator(
                color, inv_H_u, inv_H_v, grid, u_mask=u_mask, v_mask=v_mask)
            # diag(S) = -A_vertex·diag(L); S is identity (diag 1) on pinned rows.
            diag = diag + jnp.where(color > 0.5, -A_vertex * Lc, 0.0)
    safe = jnp.where(jnp.abs(diag) > 0.0, diag, 1.0)
    return jnp.where(solve_mask > 0.5, 1.0 / safe, 1.0)


def island_line_integrals(u_face, v_face, rl_data, grid):
    """Circulation of a velocity field around each island (discrete Stokes).

    The circulation around island ``i`` equals the sum of the vertex vorticity
    times the dual-cell area over the vertices belonging to island ``i``::

        ∮_{∂island_i} u·dl = Σ_v island_vertex_mask_i[v] · ζ_v · A_vertex_v

    (the union of those vertices' dual cells is the island's land + a half-cell
    collar, whose boundary is a loop in the surrounding ocean).  Reuses
    ``curl_vertex_cgrid`` for ζ, so the periodic-wrap / pole handling matches the
    operator.

    Returns
    -------
    circ : (nisle,) circulation per island [m^3/s^2 for a forcing field; m^3/s
        for a velocity field].
    """
    # Leaf-level dispatch hardening: the jnp.sum below is a domain-wide island
    # reduction (rank-local under decomposition). Guard direct callers
    # (build_line_psin reaches here); rigid_lid_step hits the solve guard first.
    assert_rigid_lid_single_rank()

    zeta = curl_vertex_cgrid(u_face, v_face, grid)        # (n_lat+1, n_lon+1)
    contrib = zeta * rl_data.A_vertex                      # circulation density
    # Σ over (lat, lon) for each island: (nisle, n_lat+1, n_lon+1) · (n_lat+1, n_lon+1)
    return jnp.sum(rl_data.island_vertex_masks * contrib[jnp.newaxis], axis=(1, 2))


def _solve_island_constants(line_forc, rl_data):
    """Solve line_psin[1:,1:] · dpsin[1:] = line_forc[1:] for the free islands.

    Island 0 is the reference (dpsin[0] ≡ 0).  Returns dpsin (nisle,).  When
    there are no free islands (nisle <= 1) returns zeros (no transport mode).
    """
    nisle = rl_data.nisle
    if nisle <= 1:
        return jnp.zeros((nisle,), dtype=line_forc.dtype)
    A = rl_data.line_psin[1:, 1:]            # (nisle-1, nisle-1)
    b = line_forc[1:]                        # (nisle-1,)
    x = jnp.linalg.solve(A, b)               # free-island constants
    return jnp.concatenate([jnp.zeros((1,), dtype=x.dtype), x], axis=0)


def rigid_lid_step(psi, dpsi, dpsi_prev, dpsin, dpsin_prev,
                   F_u_avg, F_v_avg, rl_data, dt, config, grid):
    """One rigid-lid barotropic step: solve dψ, advance ψ (AB2), recover (u,v).

    Parameters
    ----------
    psi : (n_lat+1, n_lon+1) streamfunction at time n [m^3/s].
    dpsi, dpsi_prev : (n_lat+1, n_lon+1) interior streamfunction tendency at the
        previous two solved steps (AB2 history + leapfrog CG-guess history).
    dpsin, dpsin_prev : (nisle,) per-island constant tendencies, two steps back.
    F_u_avg : (n_lat, n_lon+1) DEPTH-AVERAGED zonal momentum forcing at u-faces
        [m/s^2] — Σ_k du_k·h_u_k / H_u (all tendencies except the barotropic lid
        PGF), i.e. the model's ``F_slow_u``.
    F_v_avg : (n_lat+1, n_lon) depth-averaged meridional forcing at v-faces.
    rl_data : RigidLidStaticData.
    dt : barotropic timestep [s].
    config : LatLonCGridOceanConfig (uses ab2_epsilon, rigid_lid_cg_tol/maxiter).
    grid : LatLonGrid.

    Returns
    -------
    (psi_new, dpsi_new, dpsi_shift, dpsin_new, dpsin_shift, u_bt, v_bt) :
        the advanced streamfunction, the new + shifted tendency histories
        (dpsi_shift/dpsin_shift become *_prev next step), and the recovered
        barotropic velocity at the u-/v-faces.
    """
    eps = config.ab2_epsilon

    # 1. RHS = curl of the depth-averaged forcing = curl_vertex((1/H)∫F dz).
    #    The forcing is already depth-averaged (model F_slow_u); the curl of the
    #    depth-averaged momentum tendency is the streamfunction-tendency source.
    F_u_avg = F_u_avg * rl_data.u_mask
    F_v_avg = F_v_avg * rl_data.v_mask
    rhs = curl_vertex_cgrid(F_u_avg, F_v_avg, grid)        # (n_lat+1, n_lon+1)

    # 2. Leapfrog extrapolation as the Krylov initial guess (Veros solve_stream).
    guess = 2.0 * dpsi - dpsi_prev

    # 3. Interior solve: L(dψ_new) = rhs, ψ=0 on land.
    dpsi_new = solve_streamfunction_interior(
        rhs, rl_data, grid, guess,
        tol=config.barotropic.rigid_lid_cg_tol, maxiter=config.barotropic.rigid_lid_cg_maxiter,
    )

    # 4. Island constants from the circulation constraints:
    #    line_psin·dpsin = ∮(forcing) - ∮(interior-solution velocity).
    u_int, v_int = recover_velocity_from_streamfunction(
        dpsi_new, rl_data.inv_H_u, rl_data.inv_H_v, grid,
        u_mask=rl_data.u_mask, v_mask=rl_data.v_mask,
    )
    line_forc = (island_line_integrals(F_u_avg, F_v_avg, rl_data, grid)
                 - island_line_integrals(u_int, v_int, rl_data, grid))
    dpsin_new = _solve_island_constants(line_forc, rl_data)

    # 5. AB2 integrate ψ (interior + island contributions).
    # (#517 item 8: shared ab2_blend; eps passed verbatim → bit-identical.)
    ab2_int = ab2_blend(dpsi_new, dpsi, eps)
    psi_new = psi + dt * ab2_int
    if rl_data.nisle > 1:
        ab2_isle = ab2_blend(dpsin_new, dpsin, eps)                    # (nisle,)
        # Σ_k ab2_isle[k] · psin[...,k]
        psi_new = psi_new + dt * jnp.tensordot(rl_data.psin, ab2_isle, axes=([2], [0]))

    # 6. Recover the new barotropic velocity from ψ^{n+1}.
    u_bt, v_bt = recover_velocity_from_streamfunction(
        psi_new, rl_data.inv_H_u, rl_data.inv_H_v, grid,
        u_mask=rl_data.u_mask, v_mask=rl_data.v_mask,
    )

    # 7. Shift histories: *_prev <- (old current), current <- new.
    return psi_new, dpsi_new, dpsi, dpsin_new, dpsin, u_bt, v_bt


def barotropic_rigid_lid_latlon_cgrid(state, dt, grid, z_coord, config, rl_data,
                                      *, F_slow_u, F_slow_v,
                                      add_barotropic_coriolis=True):
    """Rigid-lid barotropic step with the standard barotropic-solver contract.

    Drop-in for ``barotropic_implicit_latlon_cgrid`` / the explicit-substep
    solver: takes the predicted state (3D velocity already advanced by the
    baroclinic perturbation tendency *and* the planetary-Coriolis predictor) +
    the depth-averaged baroclinic slow forcing ``F_slow_u/v`` (which already
    carries wind stress and bottom drag — they live in ``du_dt``), and returns
    ``(state_new, (Hu_avg, Hv_avg))`` with the barotropic mode diagnosed from
    the streamfunction.  ``eta`` is unchanged (the lid is rigid).

    The streamfunction RHS needs the FULL depth-averaged barotropic forcing, so
    the planetary Coriolis (the Sverdrup/β term) is added here, evaluated on the
    rigid-lid barotropic velocity at time n (recovered from ``ψ^n``) — the
    self-consistent value, since the rigid lid keeps the barotropic mode equal to
    ``u_bt(ψ)``.  It then flows through the AB2 (via dψ), matching Veros, where
    Coriolis is part of ``du`` and is Adams-Bashforth-extrapolated.  The
    line-813 3D Coriolis predictor remains correct for the baroclinic deviation
    (whose barotropic part is discarded and replaced by ``u_bt(ψ^{n+1})``).

    Parameters
    ----------
    state : LatLonCGridOceanState (state_mid — u/v carry du_dt_pert + Coriolis;
        ψ/dψ/dψ_prev/dpsin/dpsin_prev seeded to zero if None).
    dt : timestep [s].
    grid, z_coord, config : model components.
    rl_data : RigidLidStaticData (static topology/depth; built at construction).
    F_slow_u : (n_lat, n_lon+1) depth-averaged zonal slow forcing [m/s^2].
    F_slow_v : (n_lat+1, n_lon) depth-averaged meridional slow forcing.

    Returns
    -------
    (state_new, (Hu_avg, Hv_avg)).
    """
    from legoesm.grids.operators_latlon_cgrid import refuse_fpivot
    refuse_fpivot(getattr(grid, "fold", None), "barotropic_rigid_lid_latlon_cgrid")
    # Dispatch hardening: the streamfunction solve is single-rank only (global
    # elliptic CG with rank-local dots + domain-wide island line integrals).
    # Fail LOUDLY at trace time rather than return a silently per-rank-wrong
    # answer under MPI / SPMD decomposition.  (Also guarded EAGERLY at the
    # model's step / rigid-lid-data-build entry points; this in-body call covers
    # direct/shard_map traces of the solver.)
    assert_rigid_lid_single_rank()

    u_3d = state.u.data
    v_3d = state.v.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    dtype = u_3d.dtype

    # Fixed (eta-independent) layer thickness + face depths for the rigid lid.
    h_k = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, z_coord,
        min_water_column_m=config.min_water_column_m)
    h_u = min_cell_to_uface(h_k)                              # (n_lat, n_lon+1, nlev)
    h_v = min_cell_to_vface(h_k, grid)                        # (n_lat+1, n_lon, nlev)
    # Floored face-column depths (#517 item 5: shared column_depth; floor
    # = _DEPTH_FLOOR → bit-identical).  H_u / H_v are reused below to
    # rebuild the depth-integrated transport, so keep them named.
    H_u = column_depth(h_u, _DEPTH_FLOOR)                     # (n_lat, n_lon+1)
    H_v = column_depth(h_v, _DEPTH_FLOOR)                     # (n_lat+1, n_lon)

    # Old barotropic (depth-mean) of the predicted velocity -> baroclinic dev.
    # (#517 items 1/5: shared depth_average_to_faces; floor = _DEPTH_FLOOR
    # passed verbatim → bit-identical.)  U_old/V_old were open-coded as TWO
    # separate sums → fused=False (byte-identity reduction topology).
    U_old = depth_average_to_faces(u_3d, h_u, u_mask, _DEPTH_FLOOR,
                                   fused=False)  # (n_lat, n_lon+1)
    V_old = depth_average_to_faces(v_3d, h_v, v_mask, _DEPTH_FLOOR,
                                   fused=False)  # (n_lat+1, n_lon)
    u_prime = u_3d - U_old[..., jnp.newaxis]
    v_prime = v_3d - V_old[..., jnp.newaxis]

    # Streamfunction histories (cold start -> zeros).
    nV = (grid.n_lat + 1, grid.n_lon + 1)
    psi = state.psi if state.psi is not None else jnp.zeros(nV, dtype=dtype)
    dpsi = state.dpsi if state.dpsi is not None else jnp.zeros(nV, dtype=dtype)
    dpsi_prev = state.dpsi_prev if state.dpsi_prev is not None else jnp.zeros(nV, dtype=dtype)
    dpsin = state.dpsin if state.dpsin is not None else jnp.zeros((rl_data.nisle,), dtype=dtype)
    dpsin_prev = state.dpsin_prev if state.dpsin_prev is not None else jnp.zeros((rl_data.nisle,), dtype=dtype)

    # Add the planetary Coriolis to the barotropic forcing, evaluated on the
    # rigid-lid barotropic velocity at time n (recovered from ψ^n).
    #
    # GATED OFF under coriolis_scheme="explicit_ab2" (add_barotropic_coriolis=
    # False): there the 3-D Coriolis tendency f×u already entered du_dt, so its
    # depth-mean is INSIDE F_slow_u/v (= Veros solve_stream.py uloc/vloc =
    # depth-integral of du including Coriolis). Adding it again here would
    # double-count the barotropic Coriolis. The depth-mean of the 3-D Coriolis
    # tendency on u^n equals coriolis_cgrid(U_bar^n) only up to the C-grid
    # averaging order (depth-mean of a 4-pt average vs 4-pt average of the
    # depth-mean) — under the rigid lid eta≡0 the column depth is fixed so the
    # two barotropic-Coriolis forcings agree to the metric-weight level; routing
    # it through F_slow is the faithful (Veros) structure either way.
    if add_barotropic_coriolis:
        u_bt_n, v_bt_n = recover_velocity_from_streamfunction(
            psi, rl_data.inv_H_u, rl_data.inv_H_v, grid,
            u_mask=u_mask, v_mask=v_mask)
        cor_u, cor_v = coriolis_cgrid(
            u_bt_n, v_bt_n, grid, u_mask=u_mask, v_mask=v_mask)
        F_u = F_slow_u + cor_u
        F_v = F_slow_v + cor_v
    else:
        F_u = F_slow_u
        F_v = F_slow_v

    psi_new, dpsi_new, dpsi_shift, dpsin_new, dpsin_shift, u_bt, v_bt = rigid_lid_step(
        psi, dpsi, dpsi_prev, dpsin, dpsin_prev, F_u, F_v, rl_data, dt, config, grid)

    # Recombine: baroclinic deviation + new barotropic mode (all active levels).
    h_active_3d = (h_k > 0).astype(dtype)
    u_active_inner = h_active_3d * jnp.roll(h_active_3d, 1, axis=1)
    u_active_3d = jnp.concatenate([u_active_inner, u_active_inner[:, 0:1, :]], axis=1)
    v_active_int = h_active_3d[:-1] * h_active_3d[1:]
    zero_row = jnp.zeros_like(v_active_int[:1])
    v_active_3d = jnp.concatenate([zero_row, v_active_int, zero_row], axis=0)

    u_mask_3d = u_mask[..., jnp.newaxis]
    v_mask_3d = v_mask[..., jnp.newaxis]
    u_new_3d = (u_prime + u_bt[..., jnp.newaxis]) * u_mask_3d * u_active_3d
    v_new_3d = (v_prime + v_bt[..., jnp.newaxis]) * v_mask_3d * v_active_3d

    # Tracer-step transport: depth-integrated barotropic transport, using the
    # SAME dynamic H as the tracer column-sum (so Σ_k h_k·u_new_k = Hu_avg
    # holds).  Under the rigid-lid invariant eta≡0 (which the recipe starts at
    # and the rigid lid never changes, since it does not evolve eta), the static
    # face depth equals the dynamic one bit-for-bit, so H·u_bt = -∂ψ/∂y exactly
    # and div(Hu_avg)=0 — consistent with the unchanged eta/thickness.  (If the
    # state is ever seeded with eta≠0, e.g. a free-surface warm restart, the
    # non-divergence degrades by O(eta/H); keep eta=0 for a clean rigid-lid run.)
    Hu_avg = H_u * u_bt * u_mask
    Hv_avg = H_v * v_bt * v_mask

    state_new = state._replace(
        u=state.u.replace(data=u_new_3d),
        v=state.v.replace(data=v_new_3d),
        psi=psi_new, dpsi=dpsi_new, dpsi_prev=dpsi_shift,
        dpsin=dpsin_new, dpsin_prev=dpsin_shift,
    )
    return state_new, (Hu_avg, Hv_avg)
