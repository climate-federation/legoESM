"""NEMO state -> legoESM lat-lon C-grid state bridge (differentiable-NEMO harness).

Assembles a :class:`~legoesm.ocean.state.LatLonCGridOceanState` (plus its
beta-plane geometry and z-coordinate) from the halo-stripped :class:`NemoGrid` /
:class:`NemoState` produced by :mod:`nemo_io`, so legoESM can be handed the exact
state a NEMO run is in for a single-step tendency comparison.

Conventions reconciled here (the model/harness split of
``docs/ocean/fidelity/oracle_recipe_strategy.md`` keeps these in the harness):

* **Geometry.** GYRE (and NEMO idealised beta-plane configs) use a uniform
  Cartesian metric + ``f(y) = f0 + beta*y`` — exactly
  :func:`create_beta_plane_cgrid_geometry`. ``f0``/``beta`` are recovered from
  NEMO's own ``ff_t`` (linear in y) and the build is VERIFIED against it.
* **Velocity staggering.** NEMO ``u(i)`` sits on the **east** face of T-cell
  ``i``; legoESM ``u_face`` is the **west**-face array of length ``n_lon+1``. So
  NEMO ``u`` maps to ``u_face[:, 1:]`` and ``u_face[:, 0]`` is the west wall
  (prepend a zero column) — the OPPOSITE end from the MITgcm bridge (MITgcm ``U``
  is the west face). ``v`` analogously: NEMO north-face ``v`` -> prepend a south
  wall row. GYRE is a closed basin, so all four boundary faces are walls (0).
* **Vertical.** ``z*`` from NEMO ``e3t_1d``. GYRE is ``key_linssh`` (thicknesses
  FIXED, do not move with eta), whereas legoESM z* moves them. The *thickness
  metric* error is tiny (``~eta/H ~ 1e-5``), but the *free-surface / continuity
  formulation* differs (linssh keeps thicknesses fixed in the ssh and surface-w
  equations) — invisible at correlation tier, but it caps a machine-precision
  tier-3 tendency match. A fixed-thickness (linssh) z-coord option is a later
  step; until then this residual is expected and must not be read as a physics bug.

Scope: **flat-bottom only** (``key_vco_1d``). A NEMO config with
topography/partial cells is rejected (see :func:`bridge_nemo_to_legoesm`).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.grids.latlon import (
    LatLonCGridGeometry,
    create_beta_plane_cgrid_geometry,
    create_latlon_geometry,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid
from legoesm.ocean.fidelity.nemo_io import NemoBeforeState, NemoGrid, NemoState
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanState
from legoesm.ocean.vertical import (
    create_full_step_coordinate,
    create_z_star_from_thicknesses,
)


class NemoBridgeOutput(NamedTuple):
    geometry: LatLonCGridGeometry
    z_coord: object
    state: LatLonCGridOceanState
    land_mask: np.ndarray          # (n_lat, n_lon) surface wet mask
    f_match_max_abs: float         # max|geom.f_T - NEMO ff_t| — build self-check


def _beta_plane_params(grid: NemoGrid):
    """Recover (n_lat, n_lon, dx_m, dy_m, f0, beta) from a NEMO beta-plane mesh.

    ``ff_t`` is linear in y and constant along x on a beta-plane; fit f0/beta with
    ``y_origin_m = 0`` and cell-centre ``y_c[j] = (j + 1/2) dy`` so the resulting
    ``create_beta_plane_cgrid_geometry`` reproduces ``ff_t`` exactly.
    """
    n_lat, n_lon = grid.gphit.shape
    dx_m = float(np.mean(grid.e1t))
    dy_m = float(np.mean(grid.e2t))
    ff_col = np.asarray(grid.ff_t)[:, 0]                     # (n_lat,)
    beta = float((ff_col[-1] - ff_col[0]) / ((n_lat - 1) * dy_m))
    f0 = float(ff_col[0] - beta * 0.5 * dy_m)               # f at y_c[0] = ff_col[0]
    return n_lat, n_lon, dx_m, dy_m, f0, beta


def _u_east_to_face(nemo_u: np.ndarray) -> np.ndarray:
    """NEMO east-face u ``(n_lat, n_lon, nz)`` -> legoESM ``u_face (n_lat, n_lon+1, nz)``.

    ``u_face[:, 0]`` = west wall (0); ``u_face[:, 1:]`` = NEMO u (whose last
    column is the east wall). Closed basin -> the west wall is a true wall.
    """
    wall = np.zeros_like(nemo_u[:, :1, :])
    return np.concatenate([wall, nemo_u], axis=1)


def _v_north_to_face(nemo_v: np.ndarray) -> np.ndarray:
    """NEMO north-face v ``(n_lat, n_lon, nz)`` -> legoESM ``v_face (n_lat+1, n_lon, nz)``."""
    wall = np.zeros_like(nemo_v[:1, :, :])
    return np.concatenate([wall, nemo_v], axis=0)


def _u_east_to_face_periodic(nemo_u: np.ndarray) -> np.ndarray:
    """NEMO east-face u -> legoESM ``u_face`` for an i-PERIODIC (re-entrant) grid.

    DINO and other ``ln_Iperio=T`` configs are zonally re-entrant: the west face
    of cell 0 is the east face of the last cell (periodic wrap), NOT a wall.  So
    ``u_face[:, 1:] = nemo_u`` and ``u_face[:, 0] = nemo_u[:, -1]`` (the periodic
    image).  Contrast :func:`_u_east_to_face`, which prepends a zero wall for a
    closed basin (GYRE).  Longitude is axis 1 for both 2-D ``(lat,lon)`` masks and
    3-D ``(lat,lon,nz)`` fields.
    """
    nemo_u = np.asarray(nemo_u)
    return np.concatenate([nemo_u[:, -1:], nemo_u], axis=1)


def bridge_nemo_to_legoesm(
    grid: NemoGrid,
    state: NemoState,
    *,
    f_tol: float = 1e-9,
) -> NemoBridgeOutput:
    """Build a legoESM C-grid ocean state from a halo-stripped NEMO grid+state.

    Raises ``ValueError`` if the reconstructed beta-plane Coriolis does not match
    NEMO's ``ff_t`` to ``f_tol`` (a guard against a wrong f0/beta/dy).
    """
    n_lat, n_lon, dx_m, dy_m, f0, beta = _beta_plane_params(grid)

    geom = create_beta_plane_cgrid_geometry(
        n_lat, n_lon, dx_m=dx_m, dy_m=dy_m, f0=f0, beta=beta,
        y_origin_m=0.0, cartesian_pseudo_lat=True,
    )
    f_err = float(np.max(np.abs(np.asarray(geom.f_T) - np.asarray(grid.ff_t))))
    if f_err > f_tol:
        raise ValueError(
            f"beta-plane Coriolis mismatch vs NEMO ff_t: max|Δ|={f_err:.3e} > "
            f"{f_tol:.1e}. f0={f0:.6e}, beta={beta:.6e}, dy={dy_m:.1f}."
        )
    # f_v (vorticity-point Coriolis, used by the rel-vort flux) is f0+beta*y_g by
    # construction; f_T matching confirms the LAW, but verify NEMO's ff_f obeys the
    # SAME (f0, beta) so a y_g stagger error can't slip through the f_T gate.
    ff_f_col = np.asarray(grid.ff_f)[:, 0]
    if ff_f_col.size > 1:
        beta_f = float((ff_f_col[-1] - ff_f_col[0]) / ((ff_f_col.size - 1) * dy_m))
        if abs(beta_f - beta) > max(f_tol, 1e-6 * abs(beta)):
            raise ValueError(
                f"NEMO ff_f slope {beta_f:.6e} != ff_t slope {beta:.6e}; "
                "f-point / T-point Coriolis are inconsistent — check the mesh."
            )

    # Flat-bottom guard: this bridge builds a single-depth basin from the SURFACE
    # mask (key_vco_1d GYRE). "Flat" = every wet column is wet for the TOP n_wet
    # levels and dry below — a UNIFORM bottom depth (GYRE has 30 wet of 31 levels,
    # the 31st below-bottom). Topography/partial cells (varying n_wet, or interior
    # holes) would corrupt PGF/continuity — reject them loudly.
    tmask = np.asarray(grid.tmask) > 0.5                     # (n_lat, n_lon, nlev)
    land_mask = tmask[:, :, 0]                               # surface-wet (n_lat, n_lon)
    n_wet_col = tmask.sum(axis=2)                            # wet levels per column
    n_wet = int(n_wet_col[land_mask].max()) if land_mask.any() else 0
    # (a) every wet column has the SAME bottom depth; (b) wet cells are the top
    # n_wet (contiguous from the surface, no interior/topographic holes).
    top_contig = np.zeros_like(tmask)
    top_contig[:, :, :n_wet] = land_mask[:, :, None]
    if not (np.array_equal(n_wet_col[land_mask], np.full(int(land_mask.sum()), n_wet))
            and np.array_equal(tmask & land_mask[:, :, None], top_contig)):
        raise ValueError(
            "NEMO tmask is not flat-bottomed (varying bottom depth or interior "
            "masked cells): this bridge only supports uniform-depth bathymetry "
            "(key_vco_1d). Add partial-cell/mbathy H_bathy support before "
            "bridging topographic configs."
        )

    # Supply NEMO's EXACT analytic T-point depths (gdept_1d) so the
    # nemo_trapezoid PGF quadrature reconstructs NEMO's e3w W-spacings
    # (e3w(1)=2·gdept(1), e3w(k)=gdept(k)−gdept(k−1)) to roundoff.  These
    # differ from the interface-midpoint z_full_ref by up to ~3.6 m on the
    # stretched MI96 grid, which is the entire ~0.5% depth-signed PGF gap.
    z_coord = create_z_star_from_thicknesses(
        np.asarray(grid.e3t_1d),
        t_depth_ref_m=np.asarray(grid.gdept_1d).ravel(),
    )
    H_max = float(np.sum(np.asarray(grid.e3t_1d)[:n_wet]))   # depth of the n_wet wet cells

    base = rest_state_latlon_cgrid_ocean(
        geom, z_coord, H_max=H_max, land_mask_override=jnp.asarray(land_mask),
    )

    # Neumann-fill T/S over land so the 0.0 NEMO stores on masked cells cannot
    # contaminate legoESM's wide high-order tracer stencils (the #480 T=0 bug).
    mask3 = jnp.asarray(tmask)                               # (n_lat, n_lon, nlev) bool
    T_fill = neumann_fill_cgrid(jnp.asarray(state.T), mask3, geom)
    S_fill = neumann_fill_cgrid(jnp.asarray(state.S), mask3, geom)

    u_face = _u_east_to_face(np.asarray(state.u))
    v_face = _v_north_to_face(np.asarray(state.v))
    st = base._replace(
        T=base.T.replace(data=T_fill),
        S=base.S.replace(data=S_fill),
        u=base.u.replace(data=jnp.asarray(u_face)),
        v=base.v.replace(data=jnp.asarray(v_face)),
        eta=base.eta.replace(data=jnp.asarray(state.ssh)),
    )

    return NemoBridgeOutput(
        geometry=geom, z_coord=z_coord, state=st,
        land_mask=land_mask, f_match_max_abs=f_err,
    )


def effective_vertical_scale_factors(grid, tmask, mode=None):
    """Per-level thickness + T-depth the NEMO run ACTUALLY integrates with.

    NEMO integrates with the 3-D scale factors ``e3t_0`` (``key_vco_3d``).
    ``e3t_1d`` is a DIFFERENT, unstretched reference ladder. For DINO they agree
    in the upper ocean and diverge below ~2000 m by up to 12.9%: ``e3t_1d`` sums
    to 4506.375 m while ``e3t_0`` is stretched so the deepest wet column is
    exactly the 4000 m domain depth. Building legoESM's grid from ``e3t_1d`` put
    its abyssal layers 7-13% off and its water columns ~22 m too deep --
    precisely where #1226's ACC deficit is sourced (80% of the missing thermal
    wind below 2000 m), and thermal wind integrates density x THICKNESS.

    Falls back to the 1-D ladder when the mesh_mask predates ``e3t_0`` (GYRE,
    ``key_linssh``, where the two coincide -- which is why this went unnoticed).

    Raises
    ------
    ValueError
        If ``e3t_0`` varies horizontally over wet cells, i.e. the config has
        PARTIAL CELLS (``ln_zps``), which this bridge does not support. Silently
        averaging a thinned bottom cell into a full one would yield a
        plausible-looking but wrong bathymetry.
    """
    e3t = np.asarray(grid.e3t_1d).ravel().astype(np.float64)
    t_depth = np.asarray(grid.gdept_1d).ravel().astype(np.float64)
    # DIAGNOSTIC (#1226, temporary): LEGOESM_NEMO_E3T isolates which half of
    # NEMO's 3-D geometry drives a regression -- the thickness ladder or the
    # T-depth ladder.  "both" (default) | "e3t_only" | "gdept_only" | "off"
    import os as _os
    # DEFAULT IS "off" -- i.e. the KNOWN-WRONG 1-D ladder. This is deliberate
    # and temporary. Adopting NEMO's true e3t_0 thicknesses is CORRECT (it makes
    # legoESM's geometry match NEMO to roundoff: volume 4.7e-03 -> 6.0e-09) but
    # it DESTABILISES the model: from a bit-exact NEMO restart, max|u| grows
    # 0.66 -> 2.2 m/s over 20 days and saturates near 3 m/s, where the 1-D
    # ladder holds 0.60-0.69 indefinitely. Isolated to the THICKNESS ladder --
    # "gdept_only" (NEMO T-depths, 1-D thicknesses) is stable at 0.61, so the
    # depth ladder is innocent.
    # => legoESM is UNSTABLE ON NEMO'S ACTUAL GRID and was stable only because
    #    it ran on a wrong one. That second defect must be found before this can
    #    default to "both". Do NOT flip this default to hide the instability.
    _mode = (mode if mode is not None
             else _os.environ.get("LEGOESM_NEMO_E3T", "off"))
    if _mode not in ("off", "e3t_only", "gdept_only", "both"):
        raise ValueError(
            f"unknown vertical-scale-factor mode {_mode!r}; expected "
            '"off", "e3t_only", "gdept_only" or "both"')
    e3t3 = getattr(grid, "e3t_0", None)
    if e3t3 is None or _mode == "off":
        return e3t, t_depth, "e3t_1d"
    e3t3 = np.asarray(e3t3)
    nlev = e3t3.shape[-1]
    spread = np.zeros(nlev)
    for k in range(nlev):
        w = tmask[:, :, k]
        if w.any():
            v = e3t3[:, :, k][w]
            spread[k] = float(v.max() - v.min())
    if spread.max() > 1.0e-6:
        raise ValueError(
            f"mesh_mask e3t_0 varies horizontally (max spread {spread.max():.3e} "
            "m over wet cells): this is a PARTIAL-CELL (ln_zps) grid, which "
            "bridge_nemo_to_legoesm_topo does not support."
        )
    lev_any = tmask.any(axis=(0, 1))
    out_e3t = e3t.copy()
    for k in range(nlev):
        if lev_any[k]:
            out_e3t[k] = float(e3t3[:, :, k][tmask[:, :, k]].mean())
    if _mode == "gdept_only":
        out_e3t = e3t.copy()            # keep the 1-D thickness ladder
    gd3 = getattr(grid, "gdept_0", None)
    out_td = t_depth.copy()
    if gd3 is not None and _mode in ("both", "gdept_only"):
        gd3 = np.asarray(gd3)
        for k in range(nlev):
            if lev_any[k]:
                out_td[k] = float(gd3[:, :, k][tmask[:, :, k]].mean())
    return out_e3t, out_td, "e3t_0"


_FP32_BRIDGE_WARNED = False


def _warn_if_not_fp64() -> None:
    """One-shot loud warning when an oracle bridge is built in single precision.

    NEMO/MITgcm/Veros are fp64.  RUNNING legoESM in fp32 is a legitimate
    performance choice, so this is NOT an error -- but COMPARING against an
    oracle in fp32 measures our own rounding (f32 eps = 1.19e-7), and that is
    how the f32 depth ladder silently corrupted every #1226 measurement for
    weeks (it rounded NEMO's f64 gdept_1d to ~7 digits; median |rel| 2.555e-8 =
    0.21 x f32 eps).  ``JAX_ENABLE_X64=1`` does NOT change the policy.

    Comparison probes should use the HARD gate
    :func:`legoesm.ocean.fidelity.precision_gate.require_fp64` instead of
    relying on this warning.
    """
    global _FP32_BRIDGE_WARNED
    if _FP32_BRIDGE_WARNED:
        return
    from legoesm.core.precision import get_policy
    import jax.numpy as _jnp
    if _jnp.dtype(get_policy().control) == _jnp.float64:
        return
    _FP32_BRIDGE_WARNED = True
    import warnings
    warnings.warn(
        "NEMO oracle bridge built under a NON-fp64 precision policy "
        f"(control={_jnp.dtype(get_policy().control).name}). Model RUNS in "
        "fp32 are fine, but any COMPARISON against the oracle is then "
        "measuring float32 rounding, not physics (f32 eps = 1.19e-7). "
        "JAX_ENABLE_X64=1 does NOT change this -- set "
        "PrecisionPolicy.fp64() via legoesm.core.precision.set_policy, and "
        "use ocean.fidelity.precision_gate.require_fp64 for a hard gate.",
        RuntimeWarning, stacklevel=3,
    )


def bridge_nemo_to_legoesm_topo(
    grid: NemoGrid,
    state: NemoState,
    *,
    periodic_i: bool = True,
    omega: float = constants.Omega,
    radius: float = constants.R_earth,
    f_rtol: float = 1e-3,
    full_step: bool = False,
    metric_convention: str = "exact",
) -> NemoBridgeOutput:
    """Bridge a NEMO **Mercator + topography** config (e.g. DINO) to legoESM.

    Unlike :func:`bridge_nemo_to_legoesm` — which reconstructs a *beta-plane*
    geometry and rejects non-flat bathymetry (GYRE, ``key_vco_1d``) — this builds
    the legoESM geometry directly from NEMO's own mesh arrays and carries the
    column-varying bottom depth, so it handles:

    * **Mercator geometry** with real ``f(φ) = 2Ω sin φ`` via
      :func:`create_latlon_geometry` (``lat_1d``/``lon_1d`` from ``gphit``/``glamt``,
      exact meridional faces from ``gphiv``).  The built ``dx_T``/``f_T`` match
      NEMO's ``e1t``/``ff_t`` by construction (verified to ``f_rtol``).
    * **Full-step-z topography** (``ln_zco``, ``ln_zps=F``): a per-column bottom
      depth ``H_bathy`` from the 3-D ``tmask`` (no partial cells), so bowl / ridge
      / sill bathymetry is represented.  The guard below checks mask TOPOLOGY only
      (no interior holes); it CANNOT detect ``ln_zps`` partial cells (their mask is
      identical to full-step), and ``H_bathy`` uses the 1-D reference ``e3t_1d`` —
      so the **caller must guarantee ``ln_zps=F``**.  A per-cell ``e3t`` path
      (not read by :mod:`nemo_io`) would be needed for partial cells.
    * **i-periodic** (``ln_Iperio``) zonal boundaries via
      :func:`_u_east_to_face_periodic`; set ``periodic_i=False`` for a closed
      basin.

    Certified against a NEMO DINO 12-step + 2000-step trend dump: the interior
    hydrostatic-PGF, Coriolis, EEN-vorticity and Hollingsworth-KE tendencies match
    to correlation 1.000 (see ``docs/ocean/fidelity``).  KNOWN LIMITATIONS: (a) the
    single redundant periodic-wrap u-face (columns 0 / n_lon, which are the same
    physical face) uses legoESM's closed-basin face-storage convention rather than
    the periodic roll — exclude it from a full-domain face comparison; (b) the
    caller MUST set ``eos_depth="geometric"`` on the probe/model config for the
    NEMO S-EOS depth argument to match (the ``insitu`` default gives a
    depth-proportional density error via the thermobaric ``μ1·zh`` term); (c) z*
    thickness metric moves with η whereas NEMO ``key_qco`` differs at the
    machine-precision tier (same caveat as the flat-bottom bridge).

    Parameters
    ----------
    grid, state : NemoGrid, NemoState
        Halo-stripped NEMO mesh + restart from :mod:`nemo_io`.  ``grid.gphiv``
        (V-point latitudes) is required for exact meridional faces.
    periodic_i : bool
        ``True`` for a zonally re-entrant grid (``ln_Iperio``); ``False`` closes
        the west/east boundaries with walls.
    f_rtol : float
        Max relative error tolerance between the built ``f_T`` and NEMO ``ff_t``.
    metric_convention : {"exact", "nemo_isotropic"}, optional (#1226)
        Forwarded to :func:`create_latlon_geometry`. Default ``"exact"``
        (the true finite-difference T/u-face metric legoESM has always
        built here — BIT-IDENTICAL for every existing caller of this
        bridge). ``"nemo_isotropic"`` reproduces NEMO's own
        ``usr_def_hgr.F90`` DINO closed-form T/u-face metric
        (``pe1t = pe2t``) instead of the exact one this bridge computes
        from ``gphiv`` -- lets a fidelity probe compare against NEMO on
        NEMO's OWN metric convention rather than legoESM's (geometrically
        more exact but less NEMO-faithful) reconstruction. Does not touch
        the v-face metric (#516) or the Coriolis/``f_rtol`` check below,
        which reads ``geom.f_T`` (unaffected by this flag).

    Raises
    ------
    ValueError
        If ``gphiv`` is missing, the bathymetry is not full-step, or the built
        Coriolis does not match NEMO ``ff_t`` to ``f_rtol``.
    """
    _warn_if_not_fp64()
    if grid.gphiv is None:
        raise ValueError(
            "bridge_nemo_to_legoesm_topo requires grid.gphiv (V-point latitudes) "
            "for exact meridional cell faces; read the mesh_mask with a build that "
            "carries gphiv (read_nemo_mesh_mask populates it when present)."
        )

    gphit = np.asarray(grid.gphit)
    glamt = np.asarray(grid.glamt)
    n_lat, n_lon = gphit.shape
    lat_1d = np.deg2rad(gphit[:, 0])            # Mercator: lat varies with j only
    lon_1d = np.deg2rad(glamt[0, :])            #           lon varies with i only
    # Face latitudes: gphiv[j] = north face of cell j -> face[j+1]; the south face
    # of cell 0 by half-cell reflection about the cell centre.
    gphiv = np.deg2rad(np.asarray(grid.gphiv)[:, 0])
    lat_face = np.concatenate([[2.0 * lat_1d[0] - gphiv[0]], gphiv])  # (n_lat+1,)

    geom = create_latlon_geometry(
        n_lat, n_lon, radius=radius, omega=omega,
        lat_1d=jnp.asarray(lat_1d), lon_1d=jnp.asarray(lon_1d),
        lat_face_1d=jnp.asarray(lat_face),
        metric_convention=metric_convention,
    )
    # Partial-periodic seam wall (NEMO DINO): ALL interior cells are wet,
    # but the zonal seam u-face is closed outside the ACC channel — carried
    # on the geometry so every mask derivation (2-D/3-D face, vertex,
    # barotropic diffusion) reads it via ``getattr(grid, "seam_wall_rows")``.
    # Only meaningful for a re-entrant (periodic_i) grid; a closed basin
    # already has real west/east walls.  None on grids without a partial
    # seam → fully periodic (byte-identical).
    seam_wall_rows = getattr(grid, "seam_wall_rows", None) if periodic_i else None
    if seam_wall_rows is not None:
        seam_wall_rows = jnp.asarray(seam_wall_rows)
        if seam_wall_rows.shape != (n_lat,):
            raise ValueError(
                f"NEMO seam_wall_rows shape {seam_wall_rows.shape} != (n_lat,)="
                f"({n_lat},); the halo-derived seam profile must span the "
                "interior latitude rows."
            )
        geom = geom._replace(seam_wall_rows=seam_wall_rows)
    # Verify the built metrics + Coriolis reproduce NEMO's own arrays (guards a
    # wrong omega/lat/lon/radius/face build).  Relative because Mercator f/e1
    # span the whole latitude range.  dx_T (= R·dλ·cos φ) matches NEMO e1t to
    # roundoff; dy_T (from the reconstructed cell faces) matches e2t only to the
    # Mercator centre-vs-face residual (~0.4% on the stretched grid), so its guard
    # is loose — tight enough to catch a face sign-flip / off-by-one (which is
    # O(100%)), loose enough to pass the reconstruction residual.
    f_built = np.asarray(geom.f_T)
    f_nemo = np.asarray(grid.ff_t)
    f_scale = float(np.max(np.abs(f_nemo)))
    f_err = float(np.max(np.abs(f_built - f_nemo)))
    if f_err > f_rtol * f_scale:
        raise ValueError(
            f"Mercator Coriolis mismatch vs NEMO ff_t: max|Δ|={f_err:.3e} > "
            f"{f_rtol:.1e}·{f_scale:.3e}. Check gphit/omega."
        )
    dx_err = float(np.max(np.abs(np.asarray(geom.dx_T) - grid.e1t)))
    if dx_err > f_rtol * float(np.max(np.abs(grid.e1t))):
        raise ValueError(
            f"Mercator dx_T mismatch vs NEMO e1t: max|Δ|={dx_err:.3e}. Check "
            "glamt (lon-separable?) / radius."
        )
    dy_err = float(np.max(np.abs(np.asarray(geom.dy_T) - grid.e2t)))
    if dy_err > 5e-2 * float(np.max(np.abs(grid.e2t))):   # loose: catches face flip
        raise ValueError(
            f"Mercator dy_T mismatch vs NEMO e2t: max|Δ|={dy_err:.3e}. Check "
            "gphiv / lat_face reflection."
        )

    # --- Full-step-z topography from the 3-D tmask -------------------------
    # NB this checks tmask TOPOLOGY only — that every wet column is wet for its
    # top k_bot cells with no interior holes / dry-surface-over-wet.  It does NOT
    # (and cannot) detect ``ln_zps`` PARTIAL cells: a partial cell keeps tmask=1
    # on the thinned bottom cell, so its mask is byte-identical to a full-step
    # column.  H_bathy here is built from the 1-D reference e3t_1d, i.e. the
    # FULL-STEP bottom depth — a real ln_zps config would silently get the wrong
    # bathymetry.  The caller MUST guarantee ln_zps=F / ln_zco=T (DINO is ln_zco).
    # Detecting/correcting partial cells needs the 3-D e3t (not read by nemo_io).
    tmask = np.asarray(grid.tmask) > 0.5                 # (n_lat, n_lon, nlev)
    e3t_1d = np.asarray(grid.e3t_1d).ravel()
    land_mask = tmask[:, :, 0]                           # surface wet
    k_bot = tmask.sum(axis=2).astype(int)               # wet levels per column
    kk = np.arange(tmask.shape[2])[None, None, :]
    top_contig = (kk < k_bot[:, :, None]) & land_mask[:, :, None]
    if not np.array_equal(tmask, top_contig):
        raise ValueError(
            "NEMO tmask topology is not full-step (interior masked cells / "
            "dry-surface-over-wet): bridge_nemo_to_legoesm_topo assumes ln_zco "
            "full-step-z. NB partial cells (ln_zps) are NOT detectable from the "
            "mask — the caller must guarantee ln_zps=F."
        )
    # NEMO integrates with e3t_0, not the 1-D ladder e3t_1d -- see
    # effective_vertical_scale_factors for why this matters (#1226).
    e3t_1d, _t_depth, _e3t_src = effective_vertical_scale_factors(grid, tmask)

    depth_cum = np.cumsum(e3t_1d)                        # bottom-interface depth
    H_bathy = np.where(
        k_bot > 0, depth_cum[np.clip(k_bot - 1, 0, len(e3t_1d) - 1)], 0.0)

    z_coord = create_z_star_from_thicknesses(
        e3t_1d, t_depth_ref_m=_t_depth,
    )

    # NEMO ln_zco FULL-STEP-z: fixed reference levels everywhere + a
    # STAIRCASE of dry bottom cells below k_bot (usrdef_zgr.F90 zgr_zco_3d
    # e3t=pe3t_1d + zgr_msk_top_bot k_bot).  Wrap the plain z* coord into a
    # full-step OceanPartialCellCoordinate built DIRECTLY from NEMO's own
    # tmask column count (k_bot) — bit-faithful to the staircase, no float
    # rounding at the level interfaces.  Default off ⇒ the legacy pure-z*
    # (all levels stretched, no dry cells) path is byte-identical.
    if full_step:
        z_coord = create_full_step_coordinate(
            z_coord, bottom_level=jnp.asarray(k_bot - 1, dtype=jnp.int32),
        )

    base = rest_state_latlon_cgrid_ocean(
        geom, z_coord,
        land_mask_override=jnp.asarray(land_mask),
        H_bathy_override=jnp.asarray(H_bathy.astype(np.float64)),
    )

    # Surface face masks from NEMO umask/vmask (periodic-wrap for re-entrant i).
    umap = _u_east_to_face_periodic if periodic_i else _u_east_to_face
    umask_s = (np.asarray(grid.umask)[:, :, 0] > 0.5).astype(np.float64)
    vmask_s = (np.asarray(grid.vmask)[:, :, 0] > 0.5).astype(np.float64)
    umask_face = umap(umask_s[:, :, None])[:, :, 0]
    vmask_face = _v_north_to_face(vmask_s[:, :, None])[:, :, 0]
    # Close the seam u-face on walled rows so the bridge's own state is
    # self-consistent with the geometry seam wall (NEMO's interior umask
    # is filled wet at the seam by the periodic lbc_lnk — the wall lives
    # only in the halo tmask, so re-impose it here).
    if seam_wall_rows is not None:
        _open = (1.0 - np.asarray(seam_wall_rows)).astype(umask_face.dtype)
        umask_face[:, 0] *= _open
        umask_face[:, -1] *= _open

    # Neumann-fill T/S over land (NEMO stores 0.0 on masked cells; the wide
    # high-order tracer stencils must not see it — the #480 T=0 bug).
    mask3 = jnp.asarray(tmask)
    T_fill = neumann_fill_cgrid(jnp.asarray(state.T), mask3, geom)
    S_fill = neumann_fill_cgrid(jnp.asarray(state.S), mask3, geom)
    u_face = umap(np.asarray(state.u))
    v_face = _v_north_to_face(np.asarray(state.v))

    st = base._replace(
        T=base.T.replace(data=T_fill),
        S=base.S.replace(data=S_fill),
        u=base.u.replace(data=jnp.asarray(u_face)),
        v=base.v.replace(data=jnp.asarray(v_face)),
        eta=base.eta.replace(data=jnp.asarray(state.ssh)),
        u_mask=base.u_mask.replace(data=jnp.asarray(umask_face)),
        v_mask=base.v_mask.replace(data=jnp.asarray(vmask_face)),
    )

    return NemoBridgeOutput(
        geometry=geom, z_coord=z_coord, state=st,
        land_mask=land_mask, f_match_max_abs=f_err,
    )


def bridge_before_state_topo(
    br: NemoBridgeOutput,
    grid: NemoGrid,
    before: NemoBeforeState,
    *,
    periodic_i: bool = True,
) -> LatLonCGridOceanState:
    """Populate ``br.state``'s leap-frog BEFORE fields (Nbb) from a NEMO restart.

    NEMO's Modified-Leap-Frog restart always carries a THIRD time level
    (``tb/sb/ub/vb`` (+``sshb``/``utau_b``/``vtau_b``)), one full step behind
    the now-level (``tn/sn/...``) fields :func:`bridge_nemo_to_legoesm_topo`
    already bridges onto ``br.state``. This populates
    ``state.{T,S,u,v,eta}_before`` (+ ``tau_x_prev``/``tau_y_prev`` when the
    restart carries ``utau_b``/``vtau_b``) with the SAME face-staggering /
    Neumann-land-fill conventions as the now-level bridge, so a twin using
    this state is an EXACT leap-frog entry state (matches NEMO's own three
    time levels), not a forward-Euler cold start.

    Must be called with the SAME ``grid``/``periodic_i`` used to build
    ``br`` (no independent re-derivation of the mesh/mask).

    Parameters
    ----------
    br : NemoBridgeOutput
        Output of :func:`bridge_nemo_to_legoesm_topo` on the SAME ``grid``.
    grid : NemoGrid
        The mesh_mask this ``br`` was bridged from (for ``tmask``).
    before : NemoBeforeState
        From :func:`nemo_io.read_nemo_restart_before` on the SAME restart
        file ``br.state`` was bridged from.
    """
    tmask = np.asarray(grid.tmask) > 0.5
    mask3 = jnp.asarray(tmask)
    umap = _u_east_to_face_periodic if periodic_i else _u_east_to_face

    T_fill = neumann_fill_cgrid(jnp.asarray(before.T), mask3, br.geometry)
    S_fill = neumann_fill_cgrid(jnp.asarray(before.S), mask3, br.geometry)
    u_face = umap(np.asarray(before.u))
    v_face = _v_north_to_face(np.asarray(before.v))

    st = br.state
    replacements = dict(
        T_before=st.T.replace(data=T_fill),
        S_before=st.S.replace(data=S_fill),
        u_before=st.u.replace(data=jnp.asarray(u_face)),
        v_before=st.v.replace(data=jnp.asarray(v_face)),
        eta_before=st.eta.replace(
            data=jnp.asarray(before.ssh if before.ssh is not None else st.eta.data)),
    )
    # utau_b/vtau_b (T-point, before-level wind stress): only set when the
    # restart carries them. When absent (older NEMO builds without these
    # fields), leave tau_x_prev/tau_y_prev at their None default --
    # ``_leapfrog_step``'s own forward-Euler-start branch (state.u_before is
    # the ONLY None-gate it checks) then seeds "before := now" on step 1
    # (NEMO nit000 convention, sbcmod.F90:568-573) exactly as an un-bridged
    # cold start would, so barotropic_forcing_centred=True still gets a
    # defined ½(before+now) average rather than an AttributeError.
    if before.tau_x is not None:
        replacements["tau_x_prev"] = jnp.asarray(before.tau_x)
    if before.tau_y is not None:
        replacements["tau_y_prev"] = jnp.asarray(before.tau_y)
    return st._replace(**replacements)


__all__ = (
    "NemoBridgeOutput",
    "bridge_nemo_to_legoesm",
    "bridge_nemo_to_legoesm_topo",
    "bridge_before_state_topo",
)
