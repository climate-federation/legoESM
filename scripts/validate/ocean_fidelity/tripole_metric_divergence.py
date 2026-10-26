"""How much does the eORCA1 ZONAL metric misalignment change model continuity?

DESIGN (GLM-5.2's, after it correctly rejected my first one as a near-tautology
-- that version divided a synthetic transport by the TRUE face length and let
the operator multiply by the SHIFTED one, so a northern residual was
foreordained and its size was set by an arbitrary streamfunction rather than by
the grid; do not resurrect it):

    delta = div(u, v; production grid) - div(u, v; grid with dy_u corrected)

ONE velocity field -- the PRODUCTION RESTART, not a synthetic one -- through the
model's own ``divergence_cgrid``.  The operator is linear in the metric, so
``delta`` IS the error the face-length choice injects into continuity for the
state the model actually carries.

MEASURED INPUT, not assumed (job 9834277): glamu - glamt = +0.5 deg on eORCA1,
so NEMO's u-point is EAST of the cell centre and our u-face ``i`` must carry
NEMO's ``e2u[i-1]``.  ``tripole.py:543`` builds ``dy_u = concat([e2u,
e2u[:, 0:1]])``, giving face ``i`` the length of the face one column EAST.

ONLY ``dy_u`` IS PERTURBED.  ``tripole.py:554`` builds ``dx_v`` by PREPENDING a
zero row, which is already the correct convention (v-face ``j`` takes
``e1v[j-1]``) -- codex confirmed this independently.  So the meridional metric
is right and the defect is zonal only.  Perturbing one array keeps this a
one-variable test.

FIXES FROM CODEX REVIEW OF THE PREVIOUS DRAFT, each of which would have
produced a believable wrong number:
  * ``create_tripole_grid`` retains the FULL halo-inclusive mesh, so a
    hard-coded de-halo slice aborted the run.  Shapes are now taken from the
    GRID and the mesh is matched to it.
  * ``grid.lat_T`` is in RADIANS; degree thresholds put every cell in one band
    and emptied the null.  Converted explicitly, with the unit inferred and
    printed.
  * production FLOORS every face length at ``min_dx_m`` (1000 m,
    ``tripole.py:654-660``).  An unfloored control would differ in TWO ways, so
    the same floor is applied to the corrected array.
  * grid metrics may be stored fp32 while the raw mesh is fp64; the control is
    cast to the PRODUCTION dtype so precision is not a second variable.
  * statistics now cover WET cells only and exclude the halo columns.

WHAT THIS CAN AND CANNOT SAY.  It measures whether the error reaches the
solver's continuity operator -- the question asked.  It does NOT show whether
the dynamics later compensate; only running the barotropic solve to convergence
on both grids and comparing sea surface height could, and that is a larger
experiment.  Both reviewers also agree the 30S-60S band CANNOT detect the shift
(the metrics are identical there): it is a probe-contamination check with no
evidential weight on the hypothesis.
"""
from __future__ import annotations

import numpy as np

MESH = ("/burg-archive/glab/users/pg2328/legoESM/data/grids/"
        "eORCA1.2_mesh_mask.nc")
RESTART = ("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
           "trp_orca1faithtint_d30/restart_leg2.npz")
BANDS = [(-90, -60), (-60, -30), (-30, 30), (30, 60), (60, 75), (75, 90)]
NULL_BAND = (-60, -30)


def _sq(a, ndim=2):
    a = np.asarray(a)
    while a.ndim > ndim:
        a = a[0]
    return a


def main() -> int:
    import jax.numpy as jnp
    import xarray as xr

    from legoesm.grids.operators_latlon_cgrid import divergence_cgrid
    from legoesm.grids.tripole import DEFAULT_MIN_DX_M, create_tripole_grid

    grid = create_tripole_grid(MESH)
    dy_u_prod = np.asarray(grid.dy_u)
    prod_dtype = dy_u_prod.dtype
    area_t = np.asarray(grid.area_T, dtype=np.float64)
    n_lat, n_lon = area_t.shape
    print(f"[grid] n_lat={n_lat} n_lon={n_lon} dy_u{dy_u_prod.shape} "
          f"dtype={prod_dtype} floor={DEFAULT_MIN_DX_M}")

    lat_raw = np.asarray(getattr(grid, "lat_T"), dtype=np.float64)
    in_radians = np.nanmax(np.abs(lat_raw)) <= (np.pi + 1e-6)
    lat_t = np.degrees(lat_raw) if in_radians else lat_raw
    print(f"[grid] lat_T was in {'RADIANS' if in_radians else 'DEGREES'}; "
          f"range now {np.nanmin(lat_t):.2f}..{np.nanmax(lat_t):.2f} deg")
    if lat_t.ndim == 1:
        lat_t = np.repeat(lat_t[:, None], n_lon, axis=1)

    dm = xr.open_dataset(MESH, decode_times=False)
    e2u = _sq(dm["e2u"].values).astype(np.float64)
    tmask = _sq(dm["tmask"].values, ndim=3)[0].astype(np.float64)
    if e2u.shape != (n_lat, n_lon) or tmask.shape != (n_lat, n_lon):
        raise SystemExit(
            f"FATAL: mesh e2u{e2u.shape} tmask{tmask.shape} != grid "
            f"{(n_lat, n_lon)}. The grid is not on the raw mesh extent; align "
            "them explicitly rather than guessing a slice.")

    # our u-face i carries NEMO's e2u[i-1]; face 0 wraps to the last column
    true_dy_u = np.concatenate([e2u[:, -1:], e2u], axis=1)
    if DEFAULT_MIN_DX_M > 0.0:                      # SAME floor as production
        true_dy_u = np.maximum(true_dy_u, DEFAULT_MIN_DX_M)
    true_dy_u = true_dy_u.astype(prod_dtype)        # SAME dtype as production
    if true_dy_u.shape != dy_u_prod.shape:
        raise SystemExit(f"FATAL: corrected dy_u{true_dy_u.shape} != "
                         f"production{dy_u_prod.shape}")

    diff = np.abs(true_dy_u.astype(np.float64) - dy_u_prod.astype(np.float64))
    print(f"[ctrl ] dy_u changed on {int((diff > 0).sum())} of {diff.size} "
          f"faces; max |delta| = {diff.max():.3e} m")
    if diff.max() == 0.0:
        raise SystemExit("FATAL: control identical to production — vacuous.")

    z = np.load(RESTART, allow_pickle=True)
    u = np.asarray(z["u"]); v = np.asarray(z["v"])
    print(f"[state] restart u{u.shape} v{v.shape} "
          f"|u|max={np.abs(u).max():.4f} |v|max={np.abs(v).max():.4f} m/s")
    if u.shape[:2] != (n_lat, n_lon + 1) or v.shape[:2] != (n_lat + 1, n_lon):
        raise SystemExit(f"FATAL: restart shapes {u.shape}/{v.shape} do not "
                         f"match the grid {(n_lat, n_lon)}.")

    for name in ("_replace", "replace"):
        fn = getattr(grid, name, None)
        if callable(fn):
            grid_ctrl = fn(dy_u=jnp.asarray(true_dy_u))
            break
    else:
        import dataclasses
        if not dataclasses.is_dataclass(grid):
            raise SystemExit(f"FATAL: cannot rebuild {type(grid).__name__}.")
        grid_ctrl = dataclasses.replace(grid, dy_u=jnp.asarray(true_dy_u))

    uj, vj = jnp.asarray(u), jnp.asarray(v)
    d_prod = np.asarray(divergence_cgrid(uj, vj, grid), dtype=np.float64)
    d_ctrl = np.asarray(divergence_cgrid(uj, vj, grid_ctrl), dtype=np.float64)
    delta = d_prod - d_ctrl
    if delta.ndim == 3:
        print(f"[note ] 3-D velocities (nlev={delta.shape[-1]}); "
              "reporting the SURFACE level")
        delta, d_prod = delta[..., 0], d_prod[..., 0]

    wet = tmask > 0.5
    wet[:, 0] = False                       # halo columns carry no solver cell
    wet[:, -1] = False
    print(f"[stats] WET cells only, halo columns excluded: {int(wet.sum())}")

    print("\n[delta] |div(production) - div(corrected)| [1/s], wet cells, and "
          "as a fraction of |div(production)|")
    print("   band           median       p99         max     "
          "  rel p50    rel p99       n")
    for lo, hi in BANDS:
        m = wet & (lat_t >= lo) & (lat_t < hi)
        if not m.any():
            continue
        a = np.abs(delta[m])
        den = np.abs(d_prod[m])
        rel = np.where(den > 0, a / np.where(den > 0, den, 1.0), np.nan)
        print(f" {lo:+4d}..{hi:+4d} {np.median(a):11.4e} "
              f"{np.percentile(a, 99):11.4e} {a.max():11.4e} "
              f"{np.nanmedian(rel):10.3e} {np.nanpercentile(rel, 99):10.3e} "
              f"{int(m.sum()):7d}")

    m0 = wet & (lat_t >= NULL_BAND[0]) & (lat_t < NULL_BAND[1])
    print(f"\n[check] CONTAMINATION CHECK ONLY — both reviewers agree this "
          f"band cannot detect the shift.")
    print(f"[check] 30S-60S max |delta| = {np.abs(delta[m0]).max():.4e} "
          f"over {int(m0.sum())} wet cells; non-zero means THIS PROBE is broken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
