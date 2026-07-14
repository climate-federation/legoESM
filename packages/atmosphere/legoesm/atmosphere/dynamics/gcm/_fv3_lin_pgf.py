"""FV3-faithful Lin (1997) hydrostatic pressure-gradient force.

Direct port of GFDL FV3 ``dyn_core.F90:p_grad_c`` (line 2073) with the
companion gz-half / pk-half computation from ``a_p_pe`` / hydrostatic
geopotential calculation in the same file (line 2767-2778).

Why this exists
---------------

Our existing 3D atmospheric pressure-gradient force in
``primitive_eq_cdgrid.fv3_hydrostatic_tendencies`` splits the PGF into
two terms at D-grid corners::

    du_d/dt = ζ_corner * v_d - dB/dx - pg_corr_x

where ``B = KE + Φ`` and ``pg_corr_x = R_d * T_corner * ∇(ln p_s)`` is
the η-coordinate correction.  These two terms must cancel exactly in
hydrostatic balance.  In CONTINUUM they do; in the DISCRETE
Arakawa-Lamb 4-cell stencil at corners they only cancel in the deep
interior.  At face boundaries the stencil reads halo-interpolated
neighbour-face values whose O(dx²) interpolation error is amplified by
the A-L Cartesian matrix's off-diagonal terms (c01, c10) to O(dx)
gradient error — see ``docs/cubed_sphere_edge_artifacts.md`` iter
1-14 for the full derivation.

The Lin (1997) cross-product PGF computes the entire hydrostatic
pressure-gradient force as a SINGLE cell-pair operation at C-grid
faces using the layer-edge geopotential (gz) and layer-edge
p^κ (pkc).  In any pure hydrostatic column the cross-product is
ZERO BY CONSTRUCTION (no continuum-level cancellation that
discretisation can break).  The horizontal pressure gradient at the
face is then well-conditioned with no A-L matrix amplification.

This module provides:

- :func:`compute_pkappa_half`: ``p^κ`` at half-levels (cell centres).
- :func:`compute_geopotential_half_fv3`: FV3-style geopotential at
  half-levels via the bottom-up Φ recurrence consistent with the
  cross-product PGF (matches ``dyn_core.F90:2767-2778``).
- :func:`fv3_lin1997_pgf_3d_cgrid`: cross-product PGF at C-grid faces
  (``u_c`` and ``v_c`` positions), faithful to ``p_grad_c``.
- :func:`project_cgrid_pgf_to_dgrid_corners`: 2-point average from
  C-grid faces to D-grid corners (consistent with our prognostic
  storage of u_d, v_d at corners).

These are kept in a dedicated module rather than added to
``operators_cdgrid.py`` because they are the FV3 *architecture*
contribution to the 3D dycore and have no analog in the SW or ocean
paths (which run on the existing Arakawa-Lamb gradient).  Wiring this
into ``fv3_hydrostatic_tendencies`` is a separate, opt-in change
behind a config flag — see iteration 3 of ``FV3_3D.md``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.halo import pad_halo_4d as _pad_halo_4d
from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.vertical import HybridSigmaPressureCoordinate

# --- numerics floors (one-off; not a tunable scheme coefficient) ---
# Strictly-positive denominator floor for the Lin (1997) cross-product PGF
# divide.  FV3 uses no floor; we add one for autodiff / float32 NaN-safety.
_PGF_DENOM_FLOOR: float = 1e-12  # coeff-ok: numerics divide-by-zero floor


def compute_pkappa_half(
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
) -> jax.Array:
    """``p^κ`` at half-levels (cell centres).

    Mirrors FV3 ``dyn_core.F90:2746`` ``pk(i,j,k) = exp(akap*log(p))``.

    Parameters
    ----------
    p_s : jax.Array, shape ``(..., 6, n, n)``  -- but we accept the
        canonical legoESM 2D shape ``(6, n, n)`` (Pa).
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    pk_half : jax.Array, shape ``(..., nlev+1)``  (Pa^κ).
    """
    p_half = (
        coord.A_half * coord.p_ref + coord.B_half * p_s[..., None]
    )  # (6, n, n, nlev+1)
    p_half_safe = jnp.clip(p_half, 1.0, None)
    return p_half_safe ** constants.kappa


def compute_geopotential_half_fv3(
    T: jax.Array,
    p_s: jax.Array,
    phis: jax.Array,
    coord: HybridSigmaPressureCoordinate,
) -> jax.Array:
    """FV3-style geopotential at half-levels (cell centres).

    Derived from the hydrostatic relation ``∂Φ/∂p = -RT/p`` rewritten
    in p^κ coordinates:

        dΦ = -R*T * dp/p
           = -R*T * (1/κ) * d(p^κ)/p^κ      (since dp = (1/κ)*p^(1-κ)*dpk)
           = -cp*T*p^(-κ) * d(p^κ)         (since R/κ = cp)

    so the bottom-up half-level recurrence is::

        gz(k) = gz(k+1) + cp * T(k) * p_full(k)^(-κ) * (pk(k+1) - pk(k))

    and ``gz_half[..., nlev] = phis`` (surface).

    This is mathematically equivalent to FV3's ``dyn_core.F90:2775``
    formula ``gz(k) = gz(k+1) + cp*pt*(pk(k+1)-pk(k))`` *if* FV3's
    ``pt`` is the THERMODYNAMICALLY TRANSFORMED variable
    ``pt = T * p^(-κ)`` (after the line-403 `pt /= pkz` transform in
    ``fv_dynamics.F90``), NOT the bare potential temperature.  Our
    state stores absolute temperature ``T``, so we apply the
    ``p_full^(-κ)`` factor explicitly here.

    A naive port that uses ``cp * θ * dpk`` (where θ is bare
    potential temperature) gives a gz that is WRONG by a factor of
    ``p_ref^κ ≈ 21`` (verified by an iter-4 magnitude check against
    Simmons-Burridge geopotential — the bare-θ formula gave
    9.5e6 m²/s² while the correct one matches Simmons-Burridge ~2e5
    m²/s² for a uniform 300 K column).

    Parameters
    ----------
    T : jax.Array, shape ``(6, n, n, nlev)`` (K).
    p_s : jax.Array, shape ``(6, n, n)`` (Pa).
    phis : jax.Array, shape ``(6, n, n)``  (m^2/s^2) -- surface
        geopotential ``g * z_s``.
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    gz_half : jax.Array, shape ``(6, n, n, nlev+1)`` (m^2/s^2).  Index
        0 is the model top, index ``nlev`` is the surface
        (``gz_half[..., -1] = phis``).
    """
    p_ref = constants.p_ref
    cp = constants.c_pd
    kappa = constants.kappa

    # δp^κ per layer at full level k.
    pk_half = compute_pkappa_half(p_s, coord)  # (6, n, n, nlev+1)
    dpk = pk_half[..., 1:] - pk_half[..., :-1]  # (6, n, n, nlev)

    # p_full at layer mid-pressures (used as the p^(-κ) weight).
    p_full = coord.A_full * p_ref + coord.B_full * p_s[..., None]
    p_full_safe = jnp.clip(p_full, 1.0, None)

    # δgz per layer (positive — gz increases upward, layer index k
    # increases downward).  ``cp * T * p_full^(-κ) * dpk`` per the
    # hydrostatic relation derivation above.
    dgz = cp * T * p_full_safe ** (-kappa) * dpk  # (6, n, n, nlev)

    # Cumsum from the SURFACE upward.  Reverse along level axis,
    # cumsum, reverse back.
    cum_from_bottom = jnp.cumsum(dgz[..., ::-1], axis=-1)[..., ::-1]
    # ``cum_from_bottom[..., k]`` = sum_{l=k}^{nlev-1} dgz(l)
    # = gz_half(k) - gz_half(nlev) = gz_half(k) - phis.

    # Pre-allocate gz_half then fill: index nlev is phis, indices 0..nlev-1
    # are phis + cum_from_bottom.
    gz_half_top = phis[..., None] + cum_from_bottom         # (6, n, n, nlev)
    gz_half = jnp.concatenate(
        [gz_half_top, phis[..., None]], axis=-1,
    )                                                        # (6, n, n, nlev+1)
    return gz_half


def fv3_lin1997_pgf_3d_cgrid(
    T: jax.Array,
    p_s: jax.Array,
    phis: jax.Array,
    coord: HybridSigmaPressureCoordinate,
    cdgrid: CubedSphereCDGrid,
) -> tuple[jax.Array, jax.Array]:
    """FV3 Lin (1997) cross-product hydrostatic PGF at C-grid faces.

    Faithful port of ``dyn_core.F90:p_grad_c`` (line 2073-2129) for
    the hydrostatic branch (``hydrostatic = .true.`` so ``wk`` is
    ``δp^κ`` rather than ``delpc``).

    For each level ``k`` (0..nlev-1) and each x-face between cells
    (i-1, j) and (i, j), the C-grid u tendency is::

        ∂u_c/∂t (i,j,k) = -rdxc(i,j) / (wk_W + wk_E) * (
              (gz_W(k+1) - gz_E(k))   * (pkc_E(k+1) - pkc_W(k))
            + (gz_W(k)   - gz_E(k+1)) * (pkc_W(k+1) - pkc_E(k))
        )

    where ``W = (i-1, j)``, ``E = (i, j)``, ``wk = pkc(k+1) - pkc(k)``
    is δp^κ at the cell.  The OUTER negative sign converts the FV3
    "uc += dt * RHS" form to the "du_c/dt = RHS" tendency convention
    used in legoESM (the Fortran formula is the time-update; we want
    the rate that integrates to the same update).

    The y-face formula is the analogous expression in j with rdyc.

    Parameters
    ----------
    T : jax.Array, shape ``(6, n, n, nlev)``       (K).
    p_s : jax.Array, shape ``(6, n, n)``           (Pa).
    phis : jax.Array, shape ``(6, n, n)``          (m^2/s^2).
    coord : HybridSigmaPressureCoordinate
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    pgf_x_c : jax.Array, shape ``(6, n+1, n, nlev)`` (m/s^2).
        Tendency contribution to ``u_c`` (C-grid x-face midpoints).
        Add directly: ``du_c/dt += pgf_x_c``.
    pgf_y_c : jax.Array, shape ``(6, n, n+1, nlev)`` (m/s^2).
        Tendency contribution to ``v_c`` (C-grid y-face midpoints).

    Notes
    -----
    - Hydrostatic only (no pkc for non-hydrostatic delpc form).
    - Returns C-grid tendencies; project to D-grid via
      :func:`project_cgrid_pgf_to_dgrid_corners` to consume in the
      D-grid prognostic-wind path.
    - In any pure hydrostatic column (uniform T, p_s, phis),
      ``pgf_x_c`` and ``pgf_y_c`` are exactly zero in machine
      precision (verified by the unit test).
    """
    n = T.shape[1]
    T.shape[-1]

    # Build half-level fields.
    pk_half = compute_pkappa_half(p_s, coord)                  # (6, n, n, nlev+1)
    gz_half = compute_geopotential_half_fv3(T, p_s, phis, coord)  # (6, n, n, nlev+1)

    # 1-cell halo so we can read the i±1 and j±1 neighbour cells.
    pk_pad = _pad_halo_4d(pk_half)   # (6, n+2, n+2, nlev+1)
    gz_pad = _pad_halo_4d(gz_half)

    # Slice the W/E pair for each x-face i ∈ [0, n].  In padded coords
    # the interior cells live at index 1..n.  X-face i sits between
    # padded cells i (=W) and i+1 (=E), and at j-cell index j_c ∈ [0, n-1]
    # which corresponds to padded j-index j_c+1.
    # Slicing windows: W = pk_pad[:, 0:n+1, 1:n+1, :]  (n+1 by n)
    #                  E = pk_pad[:, 1:n+2, 1:n+1, :]
    pk_W = pk_pad[:, 0:n + 1, 1:n + 1, :]    # (6, n+1, n, nlev+1)
    pk_E = pk_pad[:, 1:n + 2, 1:n + 1, :]
    gz_W = gz_pad[:, 0:n + 1, 1:n + 1, :]
    gz_E = gz_pad[:, 1:n + 2, 1:n + 1, :]

    # Layer δp^κ at each side cell (Fortran ``wk(i-1) = pkc(i-1, k+1) - pkc(i-1, k)``).
    wk_W = pk_W[..., 1:] - pk_W[..., :-1]    # (6, n+1, n, nlev)
    wk_E = pk_E[..., 1:] - pk_E[..., :-1]

    # Cross-product (Fortran:
    #   (gz_W(k+1) - gz_E(k))   * (pkc_E(k+1) - pkc_W(k))
    # + (gz_W(k)   - gz_E(k+1)) * (pkc_W(k+1) - pkc_E(k))).
    cross_x = (
        (gz_W[..., 1:] - gz_E[..., :-1]) * (pk_E[..., 1:] - pk_W[..., :-1])
        + (gz_W[..., :-1] - gz_E[..., 1:]) * (pk_W[..., 1:] - pk_E[..., :-1])
    )  # (6, n+1, n, nlev)

    denom_x = wk_W + wk_E
    # Safety floor for the denominator (FV3 uses no floor — relies on the
    # dynamics never producing zero δp^κ in a stable atmosphere; we add a
    # conservative epsilon to keep autodiff and float32 paths NaN-safe).
    # NOTE: ``jnp.sign(0) == 0`` would zero the floor and reintroduce a
    # 0/0 (NaN value AND NaN gradient) exactly at ``denom == 0``.  Map the
    # sign-of-zero to +1 so the floor is ALWAYS strictly nonzero.
    _sgn_x = jnp.sign(denom_x)
    _floor_sign_x = _sgn_x + (1.0 - jnp.abs(_sgn_x))  # +1 where denom_x == 0
    denom_x_safe = jnp.where(
        jnp.abs(denom_x) > _PGF_DENOM_FLOOR,
        denom_x,
        _floor_sign_x * _PGF_DENOM_FLOOR,
    )

    # rdxc shape: (6, n+1, n).  Broadcast over the trailing nlev axis.
    rdxc = cdgrid.rdxc[..., None]                       # (6, n+1, n, 1)

    # Sign convention: Fortran is ``u += dt * (rdxc/sum) * cross``,
    # i.e. ``du/dt = +rdxc/sum * cross``.  The cross-product itself
    # already carries the correct PHYSICAL sign — for an eastward
    # surface-pressure ridge (``p_s`` increases with i, with constant
    # ``phis``), the cross-product is dominantly NEGATIVE (the
    # geopotential at the top of the lowest layer is higher in the
    # high-``p_s`` column → ``(gz_W(k+1) - gz_E(k))`` is large
    # negative → driver term in ``cross_x``), so ``du_c/dt`` is
    # westward as expected.  Match Fortran sign exactly: NO negation.
    # The caller then ADDS this directly: ``du_c/dt += pgf_x_c``.
    pgf_x_c = rdxc * cross_x / denom_x_safe             # (6, n+1, n, nlev)

    # Y-faces: between cells (i, j-1) and (i, j).  X-cell i ∈ [0, n-1]
    # → padded i+1.  Y-face j ∈ [0, n] → between padded j (=S) and j+1 (=N).
    pk_S = pk_pad[:, 1:n + 1, 0:n + 1, :]    # (6, n, n+1, nlev+1)
    pk_N = pk_pad[:, 1:n + 1, 1:n + 2, :]
    gz_S = gz_pad[:, 1:n + 1, 0:n + 1, :]
    gz_N = gz_pad[:, 1:n + 1, 1:n + 2, :]

    wk_S = pk_S[..., 1:] - pk_S[..., :-1]
    wk_N = pk_N[..., 1:] - pk_N[..., :-1]

    cross_y = (
        (gz_S[..., 1:] - gz_N[..., :-1]) * (pk_N[..., 1:] - pk_S[..., :-1])
        + (gz_S[..., :-1] - gz_N[..., 1:]) * (pk_S[..., 1:] - pk_N[..., :-1])
    )

    denom_y = wk_S + wk_N
    _sgn_y = jnp.sign(denom_y)
    _floor_sign_y = _sgn_y + (1.0 - jnp.abs(_sgn_y))  # +1 where denom_y == 0
    denom_y_safe = jnp.where(
        jnp.abs(denom_y) > _PGF_DENOM_FLOOR,
        denom_y,
        _floor_sign_y * _PGF_DENOM_FLOOR,
    )

    rdyc = cdgrid.rdyc[..., None]                       # (6, n, n+1, 1)
    pgf_y_c = rdyc * cross_y / denom_y_safe             # (6, n, n+1, nlev)

    return pgf_x_c, pgf_y_c


def project_cgrid_pgf_to_dgrid_corners(
    pgf_x_c: jax.Array,
    pgf_y_c: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Project C-grid face PGF tendencies to D-grid corner tendencies.

    Our prognostic winds ``u_d, v_d`` live at D-grid corners
    ``(6, n+1, n+1, nlev)``.  The Lin (1997) PGF is naturally at C-grid
    faces: ``pgf_x_c`` at u-faces ``(6, n+1, n, nlev)`` and ``pgf_y_c``
    at v-faces ``(6, n, n+1, nlev)``.  Convert via 2-point averaging
    along the OFF-DIRECTION axis::

        pgf_x_d(i, j) = 0.5 * (pgf_x_c(i, j-1) + pgf_x_c(i, j))
        pgf_y_d(i, j) = 0.5 * (pgf_y_c(i-1, j) + pgf_y_c(i, j))

    Boundary rows/cols (j=0, j=n in pgf_x; i=0, i=n in pgf_y) read the
    TRUE cross-face cubed-sphere neighbour via the staggered D-grid
    vector halo (:func:`pad_halo_dgrid_vector_4d`), so the corner-row
    PGF is second-order accurate at every panel seam.

    Cube-seam correctness (sign/metric convention)
    ----------------------------------------------
    The PGF acceleration ``(pgf_x_c, pgf_y_c)`` is a physical (geographic)
    vector whose face-local components transform across cube-panel seams
    exactly like the prognostic D-grid wind ``(u_d, v_d)``: the 16/24
    same-axis seams copy the component unchanged; the 8/24 axis-swap
    seams (faces 1,3 N/S ↔ faces 4,5 E/W) rotate 90° so the
    x-component on one face maps to the (signed) y-component on the
    neighbour.  We therefore reuse the FV3-faithful, MPI-validated
    ``pad_halo_dgrid_vector_4d`` (the same primitive the live PE/NH
    dycores use for the ``du_normal``/``dv_normal`` D-grid wind-increment
    projection in ``primitive_eq_cdgrid`` and ``compressible_euler_cdgrid``).

    STAGGERING MAP (this is the crux — the shapes cross over):
    ``pad_halo_dgrid_vector_4d`` is defined for the FV3 DGRID_NE
    convention ``u_d`` at v-edges ``(6, n, n+1, nlev)`` and ``v_d`` at
    u-edges ``(6, n+1, n, nlev)``.  Our ``pgf_y_c`` (x-cell-centre /
    y-face, ``(6, n, n+1, nlev)``) matches the ``u_d`` slot, and our
    ``pgf_x_c`` (x-face / y-cell-centre, ``(6, n+1, n, nlev)``) matches
    the ``v_d`` slot.  Hence the call is
    ``pad_halo_dgrid_vector_4d(u_d=pgf_y_c, v_d=pgf_x_c)``.

    Previously these boundary rows used ``jnp.pad(..., mode="edge")``,
    which REPLICATES the same-face edge value instead of reading the
    neighbour face — a first-order-wrong corner PGF and a direct source
    of the cube-imprint the W2 v-wind visual-regression gate guards
    against (measured ~33 % of the interior PGF signal at the seam row).

    Note: this is a SIMPLE 2-point average, NOT the A-L 4-point
    matrix.  It does not amplify halo errors at panel boundaries the
    way A-L does — that's the point of the cross-product PGF approach.

    In a uniform hydrostatic state the cross-face neighbour PGF is also
    exactly zero, so the projection preserves the zero-corner property
    by construction (verified by the unit test).

    Parameters
    ----------
    pgf_x_c : (6, n+1, n, nlev) — u-face midpoint PGF
    pgf_y_c : (6, n, n+1, nlev) — v-face midpoint PGF

    Returns
    -------
    pgf_x_d : (6, n+1, n+1, nlev) — D-grid corner u tendency
    pgf_y_d : (6, n+1, n+1, nlev) — D-grid corner v tendency
    """
    from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d

    # Cross-face staggered vector halo.  ``pgf_y_c`` fills the DGRID_NE
    # ``u_d`` slot (v-edges, (6,n,n+1)); ``pgf_x_c`` fills the ``v_d``
    # slot (u-edges, (6,n+1,n)).  The 8 axis-swap seams apply the
    # component swap + sign internally (FV3-faithful, bit-for-bit
    # validated incl. MPI — see legoesm.grids.dgrid_halo).
    pgf_y_full, pgf_x_full = pad_halo_dgrid_vector_4d(pgf_y_c, pgf_x_c)
    # pgf_y_full: (6, n+2, n+3, nlev); pgf_x_full: (6, n+3, n+2, nlev).

    # x-PGF: u-faces (n+1, n) → corners (n+1, n+1).  Average over j using
    # the now-populated south/north halo rows (trim the i-halo first).
    pgf_x_pad_j = pgf_x_full[:, 1:-1, :, :]            # (6, n+1, n+2, nlev)
    pgf_x_d = 0.5 * (pgf_x_pad_j[:, :, :-1, :] + pgf_x_pad_j[:, :, 1:, :])

    # y-PGF: v-faces (n, n+1) → corners (n+1, n+1).  Average over i using
    # the now-populated west/east halo cols (trim the j-halo first).
    pgf_y_pad_i = pgf_y_full[:, :, 1:-1, :]            # (6, n+2, n+1, nlev)
    pgf_y_d = 0.5 * (pgf_y_pad_i[:, :-1, :, :] + pgf_y_pad_i[:, 1:, :, :])

    return pgf_x_d, pgf_y_d
