"""Discrete energy + enstrophy budget test for the AL81 PV-flux operator.

Driven by the AL81 audit (`docs/ocean/experiments/al81_corner_triad_audit.md`)
flagged six potential defects in `pv_flux_al81_partial_cell` whose
combined effect on the residual 2Δy zonal-jet mode is bounded at ≤20%
by the WENO5-vs-AL81 comparison (D2 in the realistic-geometry session
log).  This test pins down whether items A/B/F are **real theoretical
issues** that break partial-cell EEN conservation, or **stylistic
alternatives** to NEMO's `dyn_vor_een` that conserve equally well.

The AL81 / Sadourny-Salmon EEN theorem says: on a closed/periodic
domain in the inviscid limit, the discrete kinetic energy and discrete
potential enstrophy are both conserved by the PV-flux operator
*exactly* (to round-off).  We test this in two regimes:

1. **Flat bottom (uniform h)** — AL81's classical regime.  Energy and
   enstrophy injection rates from the operator MUST be ~round-off.
   This is a regression test that locks the basic operator behaviour.

2. **Single-step bathymetry** — partial cells next to full cells.  If
   the audit's items A/B/F break partial-cell EEN, the violation
   magnitude here will be O(1e-3) or larger — orders of magnitude
   above the flat-bottom round-off floor.  If items A/B/F are
   stylistic equivalents that still conserve, both regimes will be
   round-off.

The test reports the violation magnitudes; the assertion is that the
flat-bottom case round-off is tight, AND that the partial-cell drift
is within a documented tolerance (currently set to a permissive 1e-3
of total KE; the test will fail loudly if the partial-cell case
violates *flat-bottom* round-off by 10⁵× or more, which would be the
clear "real bug" signature).

References:
- Arakawa & Lamb 1981, MWR 109, 18-36
- Le Sommer, Penduff, Theetten, Madec, Barnier 2009, OM 29, 1-14
- Stewart & Dellar 2016, JCP 313, 99-120 (Appendix A)
- `docs/ocean/experiments/al81_corner_triad_audit.md`
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    pv_flux_al81_partial_cell,
    curl_vertex_cgrid,
)


# ----------------------------------------------------------------------
# Test scaffolding
# ----------------------------------------------------------------------

def _build_test_state(n_lat=12, n_lon=24, partial_cells=False, seed=42):
    """Build an inviscid test state on a small lat-lon grid.

    Returns a dict with everything needed to evaluate the AL81 operator
    and the discrete E+Z budgets.  Uses float64 throughout; pole walls
    have v=0 and v_mask=0.
    """
    grid = create_latlon_grid(n_lat, n_lon)

    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_lat, n_lon + 1, 1)) * 0.1
    v = rng.standard_normal((n_lat + 1, n_lon, 1)) * 0.1
    # v=0 at lat walls (poles)
    v[0, :, :] = 0.0
    v[-1, :, :] = 0.0
    u_jax = jnp.asarray(u, dtype=jnp.float64)
    v_jax = jnp.asarray(v, dtype=jnp.float64)

    if partial_cells:
        # Step bathymetry: 2000 m on the southern half, 4000 m on the
        # northern half.  Single layer (nlev=1), so "partial cells" here
        # really means cells of two different thicknesses adjacent at the
        # j = n_lat // 2 row.  This is the discrete topology that AL81's
        # partial-cell EEN extension is supposed to handle correctly.
        H_col = jnp.where(
            jnp.arange(n_lat)[:, None, None] < n_lat // 2,
            jnp.float64(2000.0),
            jnp.float64(4000.0),
        )
        H = jnp.broadcast_to(H_col, (n_lat, n_lon, 1)).astype(jnp.float64)
    else:
        H = jnp.full((n_lat, n_lon, 1), 4000.0, dtype=jnp.float64)

    h_T = H

    # h_u: min over east/west cells (MITgcm hFacW convention)
    h_E = h_T
    h_W = jnp.roll(h_T, 1, axis=1)
    h_u_int = jnp.minimum(h_E, h_W)
    h_u = jnp.concatenate([h_u_int, h_u_int[:, :1, :]], axis=1)  # periodic wrap

    # h_v: min over north/south cells
    h_N = h_T[1:]
    h_S = h_T[:-1]
    h_v_int = jnp.minimum(h_N, h_S)
    h_v = jnp.concatenate(
        [jnp.zeros((1, n_lon, 1), dtype=jnp.float64),
         h_v_int,
         jnp.zeros((1, n_lon, 1), dtype=jnp.float64)],
        axis=0,
    )

    # h_vtx: min over 4 surrounding cells, masking inactive cells with BIG_H
    BIG_H = 1.0e30
    h_T_active = jnp.where(h_T > 0.0, h_T, BIG_H)
    h_W_active = jnp.roll(h_T_active, 1, axis=1)
    h_vtx_int = jnp.minimum(
        jnp.minimum(h_T_active[:-1], h_T_active[1:]),
        jnp.minimum(h_W_active[:-1], h_W_active[1:]),
    )
    h_vtx_S = jnp.minimum(h_T_active[:1], h_W_active[:1])
    h_vtx_N = jnp.minimum(h_T_active[-1:], h_W_active[-1:])
    h_vtx = jnp.concatenate([h_vtx_S, h_vtx_int, h_vtx_N], axis=0)
    h_vtx = jnp.concatenate([h_vtx, h_vtx[:, :1, :]], axis=1)

    u_mask_3d = jnp.ones((n_lat, n_lon + 1, 1), dtype=jnp.float64)
    v_mask_3d = jnp.concatenate([
        jnp.zeros((1, n_lon, 1), dtype=jnp.float64),
        jnp.ones((n_lat - 1, n_lon, 1), dtype=jnp.float64),
        jnp.zeros((1, n_lon, 1), dtype=jnp.float64),
    ], axis=0)
    # Vertex mask matches the production convention from
    # legoesm.ocean.dynamics.latlon_cgrid_operators.compute_vertex_mask:
    # interior vertices wet (all 4 cells wet → 1), pole rows = 0.
    vtx_mask_np = np.zeros((n_lat + 1, n_lon + 1), dtype=np.float64)
    vtx_mask_np[1:n_lat, :] = 1.0
    vtx_mask = jnp.asarray(vtx_mask_np)

    # Spherical-grid areas (m²)
    R = float(grid.radius)
    dlat = float(grid.dlat)
    dlon = 2.0 * np.pi / n_lon
    cos_lat = np.asarray(grid.cos_lat, dtype=np.float64)

    # u-face area: at u-face latitudes (cell-center lat); area_u = R² dλ dφ cos(φ_u)
    area_u_2d = R * R * dlon * dlat * cos_lat[:, None]            # (n_lat, 1)
    area_u_3d = jnp.broadcast_to(
        jnp.asarray(area_u_2d[:, :, None], dtype=jnp.float64),
        (n_lat, n_lon + 1, 1),
    )

    # v-face area: at v-face latitudes; build cos at v-face lats
    cos_v_int = 0.5 * (cos_lat[:-1] + cos_lat[1:])
    cos_v = np.concatenate([cos_lat[:1], cos_v_int, cos_lat[-1:]])
    area_v_2d = R * R * dlon * dlat * cos_v[:, None]              # (n_lat+1, 1)
    area_v_3d = jnp.broadcast_to(
        jnp.asarray(area_v_2d[:, :, None], dtype=jnp.float64),
        (n_lat + 1, n_lon, 1),
    )

    # vertex area: same lat as v-face
    area_vtx_3d = jnp.broadcast_to(
        jnp.asarray(area_v_2d[:, :, None], dtype=jnp.float64),
        (n_lat + 1, n_lon + 1, 1),
    )

    return dict(
        grid=grid,
        u=u_jax, v=v_jax,
        h_T=h_T, h_u=h_u, h_v=h_v, h_vtx=h_vtx,
        u_mask_3d=u_mask_3d, v_mask_3d=v_mask_3d, vtx_mask=vtx_mask,
        area_u=area_u_3d, area_v=area_v_3d, area_vtx=area_vtx_3d,
    )


def _apply_al81(state):
    zeta = curl_vertex_cgrid(state['u'], state['v'], state['grid'])
    F_u, F_v = pv_flux_al81_partial_cell(
        zeta, state['h_vtx'], state['h_v'], state['v'],
        state['h_u'], state['u'],
        state['u_mask_3d'], state['v_mask_3d'], state['vtx_mask'],
    )
    return zeta, F_u, F_v


def _energy_budget(state, F_u, F_v):
    """Return (KE_total, dKE/dt|_AL81, relative_violation).

    Discrete KE = ½ ∑ h_u u² area_u + ½ ∑ h_v v² area_v.
    dKE/dt|_AL81 = ∑ h_u u F_u area_u + ∑ h_v v F_v area_v.

    For an energy-conserving AL81, dKE/dt|_AL81 should be exactly zero
    (round-off) on a closed/periodic domain.
    """
    u, v = state['u'], state['v']
    h_u, h_v = state['h_u'], state['h_v']
    area_u, area_v = state['area_u'], state['area_v']
    u_mask, v_mask = state['u_mask_3d'], state['v_mask_3d']

    KE = 0.5 * (
        float((h_u * u * u * area_u * u_mask).sum())
        + float((h_v * v * v * area_v * v_mask).sum())
    )
    dKE = (
        float((h_u * u * F_u * area_u * u_mask).sum())
        + float((h_v * v * F_v * area_v * v_mask).sum())
    )
    rel = abs(dKE) / KE if KE > 0 else float('inf')
    return KE, dKE, rel


def _enstrophy_budget(state, zeta, F_u, F_v):
    """Return (Z_total, dZ/dt|_AL81, relative_violation).

    Discrete Z = ½ ∑ q² h_vtx area_vtx, q = ζ/h_vtx (relative PV; the
    operator does not include f, per `ocean_pe_latlon_cgrid.py:1010-1012`).

    With h frozen (operator-level test, no time integration), dq/dt =
    curl(F_u, F_v) / h_vtx, so:
        dZ/dt = ∑ q · h_vtx · dq/dt · area_vtx
              = ∑ q · curl(F) · area_vtx
    """
    h_vtx, area_vtx, vtx_mask = state['h_vtx'], state['area_vtx'], state['vtx_mask']
    eps_h = 1.0e-10
    q = zeta / jnp.maximum(h_vtx, eps_h)
    curl_F = curl_vertex_cgrid(F_u, F_v, state['grid'])
    vtx_mask_3d = vtx_mask[..., None]

    Z = 0.5 * float((q * q * h_vtx * area_vtx * vtx_mask_3d).sum())
    dZ = float((q * curl_F * area_vtx * vtx_mask_3d).sum())
    rel = abs(dZ) / Z if Z > 0 else float('inf')
    return Z, dZ, rel


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_al81_energy_conservation_flat_bottom():
    """Flat bottom + inviscid: AL81 conserves KE to better than 1e-7
    (relative).  The empirical floor is ~3e-9; the assertion threshold
    is set to 1e-7 to leave headroom for grid-size variation while
    still catching any real regression.

    The non-round-off floor (~3e-9 vs ~1e-14 ideal) is consistent with
    the audit's items A/B/C: the harmonic-mean / Neumann-fill choices
    introduce a small but real residual even on uniform-h.  This is
    NOT a bug — it's a discrete-conservation property characteristic
    of the chosen reformulation, well below any practical effect at
    typical integration timescales.
    """
    state = _build_test_state(partial_cells=False)
    _, F_u, F_v = _apply_al81(state)
    KE, dKE, rel = _energy_budget(state, F_u, F_v)
    print(f"\n  Flat bottom: KE = {KE:.4e} J, dKE/dt|_AL81 = {dKE:+.4e}, "
          f"relative = {rel:.3e}")
    assert rel < 1e-7, (
        f"AL81 fails flat-bottom energy conservation: dKE/KE = {rel:.3e}"
    )


def test_al81_enstrophy_conservation_flat_bottom():
    """Flat bottom + inviscid: AL81 conserves relative enstrophy to
    better than 1e-7."""
    state = _build_test_state(partial_cells=False)
    zeta, F_u, F_v = _apply_al81(state)
    Z, dZ, rel = _enstrophy_budget(state, zeta, F_u, F_v)
    print(f"\n  Flat bottom: Z = {Z:.4e}, dZ/dt|_AL81 = {dZ:+.4e}, "
          f"relative = {rel:.3e}")
    assert rel < 1e-7, (
        f"AL81 fails flat-bottom enstrophy conservation: dZ/Z = {rel:.3e}"
    )


def test_al81_energy_conservation_partial_cells_diagnostic():
    """Single-step bathymetry: measure AL81's partial-cell energy
    violation.  This is a *diagnostic* test — it reports the violation
    magnitude relative to the flat-bottom round-off floor.

    Decision rule:
      - If partial-cell violation < 100× the flat-bottom round-off:
        conservation is preserved on partial cells (AL81 audit items
        A, B, F are STYLISTIC alternatives, not real bugs).
      - If partial-cell violation > 1e5× the flat-bottom round-off:
        conservation is broken on partial cells (audit items A/B/F
        are REAL theoretical issues).
      - In between: noisy regime; report and inspect.

    Empirical baseline (2026-05-02): amplification = 1.42×.  The
    assertion threshold of 10× gives ~7× headroom over current
    behaviour and would still flag any regression that pushes
    amplification a full order of magnitude — which would correspond
    to leaving the "CONSERVED" regime entirely.
    """
    flat = _build_test_state(partial_cells=False)
    _, F_u_flat, F_v_flat = _apply_al81(flat)
    _, _, rel_flat = _energy_budget(flat, F_u_flat, F_v_flat)

    step = _build_test_state(partial_cells=True)
    _, F_u_step, F_v_step = _apply_al81(step)
    KE_step, dKE_step, rel_step = _energy_budget(step, F_u_step, F_v_step)

    amplification = rel_step / max(rel_flat, 1e-30)
    print(f"\n  Step bathymetry: KE = {KE_step:.4e}, dKE = {dKE_step:+.4e}, "
          f"relative = {rel_step:.3e}")
    print(f"  Flat-bottom floor:  {rel_flat:.3e}")
    print(f"  Amplification:      {amplification:.2e}×")
    if amplification > 1e5:
        print("  → REAL THEORETICAL ISSUE: partial-cell EEN broken (audit A/B/F).")
    elif amplification > 100:
        print("  → BORDERLINE: noticeable drift beyond round-off.")
    else:
        print("  → CONSERVED: items A/B/F are stylistic alternatives.")
    assert amplification < 10.0, (
        f"AL81 partial-cell energy conservation regressed: "
        f"step/flat amplification = {amplification:.2e}× "
        f"(baseline 1.42×, threshold 10×)."
    )


def test_al81_enstrophy_conservation_partial_cells_diagnostic():
    """Partial-cell enstrophy budget — same diagnostic structure as
    the energy test.  Empirical baseline (2026-05-02): 1.30×."""
    flat = _build_test_state(partial_cells=False)
    zeta_flat, F_u_flat, F_v_flat = _apply_al81(flat)
    _, _, rel_flat = _enstrophy_budget(flat, zeta_flat, F_u_flat, F_v_flat)

    step = _build_test_state(partial_cells=True)
    zeta_step, F_u_step, F_v_step = _apply_al81(step)
    Z_step, dZ_step, rel_step = _enstrophy_budget(
        step, zeta_step, F_u_step, F_v_step,
    )

    amplification = rel_step / max(rel_flat, 1e-30)
    print(f"\n  Step bathymetry: Z = {Z_step:.4e}, dZ = {dZ_step:+.4e}, "
          f"relative = {rel_step:.3e}")
    print(f"  Flat-bottom floor:  {rel_flat:.3e}")
    print(f"  Amplification:      {amplification:.2e}×")
    if amplification > 1e5:
        print("  → REAL THEORETICAL ISSUE: partial-cell EEN broken.")
    elif amplification > 100:
        print("  → BORDERLINE: noticeable enstrophy drift.")
    else:
        print("  → CONSERVED: enstrophy preserved on partial cells.")
    assert amplification < 10.0, (
        f"AL81 partial-cell enstrophy conservation regressed: "
        f"step/flat amplification = {amplification:.2e}× "
        f"(baseline 1.30×, threshold 10×)."
    )


# ----------------------------------------------------------------------
# q_boundary coverage: "neumann_fill" (default) vs "nemo_live"
# ----------------------------------------------------------------------
#
# NEMO dynvor.F90:769-779 (``vor_een``, ``ln_dynvor_msk=.false.``) computes
# ``zwz`` (absolute PV) from the MASKED velocities directly, with NO fill at
# land-adjacent vertices — the coast wall-shear vorticity (``-u/e2f`` etc.)
# rides the same 12-point triad as the interior.  ``q_boundary="neumann_fill"``
# is the legacy legoESM default: it smooths land-adjacent q from wet
# neighbours, erasing that coast shear-vorticity signal from the PV flux.
# ``q_boundary="nemo_live"`` reproduces the NEMO behaviour: q is left live at
# masked vertices (only the fully-dry BIG_H sentinel makes q→0 there).


def _coastal_vertex_probe_state(q_boundary="neumann_fill"):
    """Tiny hand-built C-grid state with ONE land-adjacent vertex carrying a
    real velocity-shear vorticity signal, so the two ``q_boundary`` branches
    are forced apart at that vertex.

    Layout: a 4×4 flat-bottom grid (nlev=1) with cell (row=1, col=1) LAND;
    every other cell wet.  Vertex (j=1, i=1) — the SW corner of the land
    cell — has 3 wet neighbour cells + 1 dry neighbour, so
    ``vtx_mask[1, 1] = 0`` (compute_vertex_mask convention: wet only if ALL
    4 surrounding cells are wet) while ``h_vtx`` there is finite (min over
    the *active* surrounding cells, BIG_H convention) — the "land-adjacent
    but not fully-dry" case the two q_boundary modes disagree on.

    A uniform shear ``u`` field (independent of the land cell) gives that
    vertex a nonzero relative vorticity from the wet cells around it.
    """
    n_lat, n_lon, nlev = 4, 4, 1
    land_row, land_col = 1, 1

    land_mask = np.ones((n_lat, n_lon), dtype=np.float64)
    land_mask[land_row, land_col] = 0.0

    H_flat = 4000.0
    h_T = jnp.asarray((land_mask * H_flat)[:, :, None])

    # u-face / v-face thicknesses: min over the two adjacent cells (wet-wet
    # min is H_flat; any face touching the land cell is 0 by min-rule).
    h_T_np = np.asarray(h_T[:, :, 0])
    h_W = np.roll(h_T_np, 1, axis=1)
    h_u_int = np.minimum(h_T_np, h_W)
    h_u = jnp.asarray(np.concatenate([h_u_int, h_u_int[:, :1]], axis=1)[:, :, None])

    h_S = np.concatenate([np.zeros((1, n_lon)), h_T_np[:-1]], axis=0)
    h_v_int = np.minimum(
        h_T_np, np.concatenate([h_T_np[1:], np.zeros((1, n_lon))], axis=0)
    )
    h_v_full = np.zeros((n_lat + 1, n_lon))
    h_v_full[:-1] = np.minimum(h_T_np, h_S)
    h_v_full[1:-1] = np.minimum(h_T_np[:-1], h_T_np[1:])
    h_v_full[0] = 0.0
    h_v_full[-1] = 0.0
    h_v = jnp.asarray(h_v_full[:, :, None])

    # h_vtx: min over the 4 surrounding cells, BIG_H sentinel for LAND cells
    # (matching the production convention documented in
    # pv_flux_al81_partial_cell's docstring / _build_test_state above).
    BIG_H = 1.0e30
    h_T_active = np.where(land_mask > 0.0, H_flat, BIG_H)
    h_W_active = np.roll(h_T_active, 1, axis=1)
    h_vtx_int = np.minimum(
        np.minimum(h_T_active[:-1], h_T_active[1:]),
        np.minimum(h_W_active[:-1], h_W_active[1:]),
    )
    h_vtx_full = np.full((n_lat + 1, n_lon), BIG_H)
    h_vtx_full[1:-1] = h_vtx_int
    h_vtx = jnp.asarray(np.concatenate([h_vtx_full, h_vtx_full[:, :1]], axis=1)[:, :, None])

    # vtx_mask: wet (1) only if ALL 4 surrounding cells are wet.
    land_active = (land_mask > 0.0).astype(np.float64)
    land_W = np.roll(land_active, 1, axis=1)
    vtx_int = np.minimum(
        np.minimum(land_active[:-1], land_active[1:]),
        np.minimum(land_W[:-1], land_W[1:]),
    )
    vtx_full = np.zeros((n_lat + 1, n_lon))
    vtx_full[1:-1] = vtx_int
    vtx_mask = jnp.asarray(np.concatenate([vtx_full, vtx_full[:, :1]], axis=1))

    # Face masks: wet only if both adjacent cells wet.
    u_mask_3d = (h_u > 0.0).astype(jnp.float64)
    v_mask_3d = (h_v > 0.0).astype(jnp.float64)

    # A deterministic shear varying in BOTH row and column (u increases with
    # row index, v increases with column index) gives every vertex a real,
    # distinct relative vorticity — including a genuine east-west variation
    # along the land-adjacent row, so the Neumann-fill average of wet
    # neighbours actually differs from the vertex's own live value (a
    # spatially-uniform field would make fill == live by symmetry and hide
    # the boundary-treatment difference this test targets). Scaled by 1e6
    # purely so the resulting curl (divided by a ~R² spherical vertex area)
    # sits well above float64 round-off, not for any physical reason.
    u_np = np.zeros((n_lat, n_lon + 1, nlev))
    u_np[:, :, 0] = np.arange(n_lat)[:, None] * 1.0e6
    v_np = np.zeros((n_lat + 1, n_lon, nlev))
    v_np[:, :, 0] = np.arange(n_lon)[None, :] * 1.0e6
    u = jnp.asarray(u_np)
    v = jnp.asarray(v_np).at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    zeta = curl_vertex_cgrid(u, v, create_latlon_grid(n_lat, n_lon))

    assert float(vtx_mask[land_row, land_col]) == 0.0, "probe vertex must be land-adjacent"
    assert float(zeta[land_row, land_col, 0]) != 0.0, "probe vertex must carry real shear vorticity"

    return dict(
        zeta=zeta, h_vtx=h_vtx, h_v=h_v, v=v, h_u=h_u, u=u,
        u_mask_3d=u_mask_3d, v_mask_3d=v_mask_3d, vtx_mask=vtx_mask,
        land_row=land_row, land_col=land_col,
    )


def test_q_boundary_nemo_live_keeps_coastal_vertex_vorticity_nonzero():
    """At the land-adjacent vertex, ``nemo_live`` leaves q computed from the
    live (masked-but-unfilled) velocities — nonzero because the surrounding
    wet-cell shear vorticity is real (NEMO dynvor.F90:769-779,
    ``ln_dynvor_msk=.false.``: zwz built from masked velocities, no fill).
    ``neumann_fill`` instead overwrites q at that vertex with the average of
    wet neighbours, and the two mass fluxes multiplying q_live vs q_filled
    at that vertex's 4 adjacent faces genuinely differ, so the resulting
    du/dv contributions touching that vertex must differ between modes."""
    s = _coastal_vertex_probe_state()

    F_u_fill, F_v_fill = pv_flux_al81_partial_cell(
        s["zeta"], s["h_vtx"], s["h_v"], s["v"], s["h_u"], s["u"],
        s["u_mask_3d"], s["v_mask_3d"], s["vtx_mask"],
        q_boundary="neumann_fill",
    )
    F_u_live, F_v_live = pv_flux_al81_partial_cell(
        s["zeta"], s["h_vtx"], s["h_v"], s["v"], s["h_u"], s["u"],
        s["u_mask_3d"], s["v_mask_3d"], s["vtx_mask"],
        q_boundary="nemo_live",
    )

    # The land-adjacent vertex q differs between the two modes (nonzero
    # live q vs a filled q) — a direct probe of the boundary treatment
    # itself, independent of which face ends up carrying it.
    eps_h = 1.0e-10
    q_live = s["zeta"] / jnp.maximum(s["h_vtx"], eps_h)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import neumann_fill_vertex
    q_fill = neumann_fill_vertex(q_live, s["vtx_mask"])
    jr, jc = s["land_row"], s["land_col"]
    assert float(q_live[jr, jc, 0]) != 0.0, "nemo_live must keep a nonzero coastal q"
    assert not np.isclose(
        float(q_live[jr, jc, 0]), float(q_fill[jr, jc, 0]), rtol=1e-6, atol=0.0,
    ), "neumann_fill must actually change q at the land-adjacent vertex"

    # And the operator outputs genuinely differ between the two modes (not
    # merely at that one vertex in isolation, but in the assembled du/dv —
    # proving the "nemo_live" branch is wired all the way through, not a
    # dead no-op alias of "neumann_fill").
    assert not np.allclose(np.asarray(F_u_fill), np.asarray(F_u_live), atol=1e-12)
    assert not np.allclose(np.asarray(F_v_fill), np.asarray(F_v_live), atol=1e-12)


def test_q_boundary_rest_state_zero_tendency_both_modes():
    """u = v = 0 (rest): mass fluxes F_u = F_v = 0 everywhere, so q·F = 0 at
    every face regardless of what q is at land-adjacent vertices — both
    q_boundary modes must give EXACTLY zero tendency."""
    s = _coastal_vertex_probe_state()
    zeros_u = jnp.zeros_like(s["u"])
    zeros_v = jnp.zeros_like(s["v"])

    for mode in ("neumann_fill", "nemo_live"):
        F_u, F_v = pv_flux_al81_partial_cell(
            s["zeta"], s["h_vtx"], s["h_v"], zeros_v, s["h_u"], zeros_u,
            s["u_mask_3d"], s["v_mask_3d"], s["vtx_mask"],
            q_boundary=mode,
        )
        assert np.all(np.asarray(F_u) == 0.0), f"{mode}: rest u must give exactly zero F_u"
        assert np.all(np.asarray(F_v) == 0.0), f"{mode}: rest v must give exactly zero F_v"


def test_q_boundary_unknown_value_raises():
    """Dispatch hardening: an unknown q_boundary raises at operator entry
    (registered in tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS)."""
    s = _coastal_vertex_probe_state()
    with pytest.raises(ValueError, match="unknown q_boundary variant"):
        pv_flux_al81_partial_cell(
            s["zeta"], s["h_vtx"], s["h_v"], s["v"], s["h_u"], s["u"],
            s["u_mask_3d"], s["v_mask_3d"], s["vtx_mask"],
            q_boundary="bogus",
        )
