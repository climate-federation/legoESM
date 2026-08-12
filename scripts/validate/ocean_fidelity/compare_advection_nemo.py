"""Stage C: our tracer-ADVECTION operator vs NEMO's ``ttrd_totad``.

Stage B/B2 verified ONE operator (implicit vertical diffusion).  This is the
next per-process term, and unlike B2 it needs NO reconstruction from the
target: NEMO's RK3 stage applies the tracer operators in the order

    tra_adv -> tra_sbc -> tra_qsr -> tra_ldf -> tra_bbl -> tra_zdf

(``stprk3_stg.F90`` lines 519, 521, 585-598), so ADVECTION IS FIRST and the
field it acts on is the state at the start of the step.  With the measured
time pairing (trend record ``r`` <-> state change ``r -> r+1``;
``nemo_trend_closure.py``), that field is simply state record ``r``.  This is
therefore a DIRECT operator comparison: NEMO's own state and transports in,
our flux-divergence out, against NEMO's own advection trend.

SCHEME NOTE, stated with every number: ORCA1 runs ``ln_traadv_fct`` with
``nn_fct_h = nn_fct_v = 2`` (centred-2 FCT).  Our production tripole arm runs
``superbee``.  Both are scored here: ``fct2`` measures whether our FCT
IMPLEMENTATION matches NEMO's, and ``superbee`` measures what the production
scheme CHOICE costs against the oracle -- two different questions that must
not be reported as one.

GEOMETRY APPROXIMATION (declared): NEMO's transports are ``e3u*u`` at u-faces;
the trend file carries ``uoce``/``voce`` and ``e3t``, so face thicknesses are
built by the two-point average of ``e3t`` (NEMO's own ``e3u`` uses the same
mean with a min-rule at steps).  The vertical transport is reconstructed from
continuity, not read.  These affect the flux magnitudes at bottom steps and
are the reason this probe reports a correlation/nrmse verdict rather than a
bitwise one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
from compare_tendencies_nemo import REGIONS, _fill, region_report  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tfile", required=True)
    p.add_argument("--ufile", required=True)
    p.add_argument("--vfile", required=True)
    p.add_argument("--rec", type=int, default=1)
    p.add_argument("--dt", type=float, default=3600.0)
    p.add_argument("--mesh-mask", required=True,
                   help="eORCA1 mesh_mask.nc -- the tripolar metrics the model "
                        "itself uses (a regular lat-lon geometry is WRONG here).")
    p.add_argument("--schemes", nargs="+", default=["fct2", "superbee"])
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import jax.numpy as jnp
    import netCDF4 as nc
    from legoesm.grids.tripole import create_tripole_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _compute_advection_flux_div,
    )

    dsT, dsU, dsV = (nc.Dataset(a.tfile), nc.Dataset(a.ufile), nc.Dataset(a.vfile))
    r = a.rec
    T = _fill(dsT.variables["votemper"][r])          # (z, y, x) at the state
    S = _fill(dsT.variables["vosaline"][r])
    e3t = _fill(dsT.variables["e3t"][r])
    ttrd = _fill(dsT.variables["ttrd_totad"][r])
    strd = _fill(dsT.variables["strd_totad"][r])
    uo = _fill(dsU.variables["uoce"][r])
    vo = _fill(dsV.variables["voce"][r])
    lat = _fill(dsT.variables["nav_lat_grid_T"][:])
    dsT.close(); dsU.close(); dsV.close()

    # GEOMETRY: the eORCA1 tripolar metrics, NOT a regular lat-lon grid.
    # Driving the C-grid operators with create_latlon_geometry(331, 360)
    # produced advection tendencies ~1e6 too large (measured 2026-08-12:
    # rms 1.3e-1 K/s vs NEMO 2.8e-7) because every cell length/area was the
    # wrong mesh.  create_tripole_grid reads the mesh mask the model itself
    # uses.
    grid = create_tripole_grid(a.mesh_mask)
    e1t_full = np.asarray(grid.dx_T, dtype=np.float64)
    e2t_full = np.asarray(grid.dy_T, dtype=np.float64)
    print(f"[grid] tripole metrics {e1t_full.shape}; trend arrays are the "
          "native eORCA1 frame")

    z, ny, nx = T.shape
    wet = np.isfinite(T)
    # The GRID OBJECT carries the model's halo frame (332, 362) and its fold
    # logic; slicing the metrics is not enough because the operators index
    # grid.area_T etc. internally.  So lift the native NEMO fields INTO the
    # model frame instead, run there, and slice the result back.
    NY, NX = e1t_full.shape
    if (NY, NX) != (ny + 1, nx + 2):
        raise SystemExit(f"unexpected model frame {e1t_full.shape} for a "
                         f"native trend frame ({ny}, {nx})")
    e1t, e2t = e1t_full, e2t_full

    def lift(native):
        """(ny, nx, z) native -> (NY, NX, z) model frame with the cyclic
        overlap columns filled; the north-fold ghost row stays zero and rows
        >= ny-1 are excluded from scoring below."""
        full = np.zeros((NY, NX, native.shape[-1]), dtype=np.float64)
        full[0:ny, 1:nx + 1, :] = native
        full[:, 0, :] = full[:, nx, :]
        full[:, nx + 1, :] = full[:, 1, :]
        return full
    # (z, y, x) -> (y, x, z): the C-grid operators take level-last.
    def yxz(x):
        return np.transpose(np.nan_to_num(x, nan=0.0), (1, 2, 0))

    T3, S3, h3 = lift(yxz(T)), lift(yxz(S)), lift(yxz(e3t))
    u3, v3 = lift(yxz(uo)), lift(yxz(vo))
    h3 = np.where(h3 > 0, h3, 0.0)

    # STAGGERING (the array layout is an API): the C-grid operators want
    # mass_flux_u on (n_lat, n_lon+1, nlev) and mass_flux_v on
    # (n_lat+1, n_lon, nlev), i.e. faces INCLUDING both domain edges, while
    # NEMO's uoce/voce sit on (n_lat, n_lon)/(n_lat, n_lon) cell-indexed
    # faces.  NEMO's u(i) is the face between T(i) and T(i+1), so our face j
    # (between T(j-1) and T(j)) is NEMO's u(j-1): prepend the periodic wrap
    # in x, and pad the southern/northern v rows with zero (closed).
    h_u = 0.5 * (h3 + np.roll(h3, -1, axis=1))
    h_v = 0.5 * (h3 + np.roll(h3, -1, axis=0))
    e2u = 0.5 * (e2t + np.roll(e2t, -1, axis=1))
    e1v = 0.5 * (e1t + np.roll(e1t, -1, axis=0))
    # UNITS (read from the model, not inferred): the driver builds
    #     mass_flux_u = h_u_old * u          (ocean_model_latlon_cgrid.py:4250)
    # i.e. a transport PER UNIT WIDTH [m^2/s] -- divergence_cgrid multiplies
    # by the face length itself.  Multiplying by e2u here first overshot the
    # tendency by ~1e5-1e6 (measured: rms 1.8 K/s vs NEMO 2e-6).
    mf_u_c = h_u * u3                            # (ny, nx, z) at NEMO u-points
    mf_v_c = h_v * v3                            # (ny, nx, z) at NEMO v-points
    # x is periodic: face 0 == face nx.
    mf_u = np.concatenate([mf_u_c[:, -1:, :], mf_u_c], axis=1)      # (ny, nx+1, z)
    h_u_f = np.concatenate([h_u[:, -1:, :], h_u], axis=1)
    # y is not periodic: the southern face carries no transport.
    mf_v = np.concatenate([np.zeros_like(mf_v_c[:1]), mf_v_c], axis=0)  # (ny+1, nx, z)
    h_v_f = np.concatenate([np.zeros_like(h_v[:1]), h_v], axis=0)
    # Vertical transport from continuity.  w is a VELOCITY [m/s], so the
    # horizontal divergence must be the VOLUME one: multiply the per-width
    # transports by their face lengths, difference, and divide by cell area.
    e2u_f = np.concatenate([e2u[:, -1:], e2u], axis=1)[:, :, None]
    e1v_f = np.concatenate([np.zeros_like(e1v[:1]), e1v], axis=0)[:, :, None]
    div_h = (((mf_u * e2u_f)[:, 1:, :] - (mf_u * e2u_f)[:, :-1, :])
             + ((mf_v * e1v_f)[1:, :, :] - (mf_v * e1v_f)[:-1, :, :]))
    area = (e1t * e2t)[:, :, None]
    w_int = -np.flip(np.cumsum(np.flip(div_h, axis=-1), axis=-1), axis=-1) / area
    w_half = np.concatenate([np.zeros_like(w_int[..., :1]), w_int], axis=-1)[..., :z + 1]

    lat_col = lat.reshape(ny * nx)
    result = {"rec": r, "schemes": {},
              "scheme_note": ("NEMO ORCA1 runs FCT with nn_fct_h=nn_fct_v=2; "
                              "our production tripole arm runs superbee. "
                              "fct2 tests our IMPLEMENTATION, superbee tests "
                              "the production scheme CHOICE."),
              "geometry_note": ("face thicknesses from the two-point e3t mean; "
                                "vertical transport reconstructed from "
                                "continuity; not a bitwise test")}

    def unlift(full):
        """(NY, NX, z) model frame -> (ny, nx, z) native."""
        return full[0:ny, 1:nx + 1, :]

    def cols(x):   # (ny, nx, z) -> (ncol, z)
        return x.reshape(ny * nx, z)

    # The north-fold ghost row is not reproduced by this single-call
    # diagnostic, so the two rows adjacent to it are excluded from scoring.
    fold_excl = np.ones((ny, nx), dtype=bool)
    fold_excl[ny - 2:, :] = False

    for scheme in a.schemes:
        dT_dt, dS_dt = [], []
        for tr in (T3, S3):
            div_hut, vfd = _compute_advection_flux_div(
                jnp.asarray(tr), scheme, jnp.asarray(mf_u), jnp.asarray(mf_v),
                jnp.asarray(w_half), jnp.asarray(h3), jnp.asarray(h_u_f),
                jnp.asarray(h_v_f), grid, a.dt)
            inv_h = np.where(h3 > 0, 1.0 / np.maximum(h3, 1e-12), 0.0)
            dT_dt.append(unlift(-(np.asarray(div_hut) + np.asarray(vfd)) * inv_h))
        ours_T, ours_S = cols(dT_dt[0]), cols(dT_dt[1])
        nemo_T, nemo_S = cols(yxz(ttrd)), cols(yxz(strd))
        wet_c = (cols(yxz(wet.astype(float))) > 0.5) & cols(
            np.broadcast_to(fold_excl[:, :, None], (ny, nx, z)))
        print(f"\n########## scheme = {scheme}")
        result["schemes"][scheme] = {
            "T": region_report(f"Stage C ({scheme}): advection dT vs ttrd_totad [K/s]",
                               ours_T, nemo_T, wet_c & np.isfinite(nemo_T), lat_col),
            "S": region_report(f"Stage C ({scheme}): advection dS vs strd_totad [PSU/s]",
                               ours_S, nemo_S, wet_c & np.isfinite(nemo_S), lat_col),
        }

    (out / f"advection_match_rec{r}.json").write_text(json.dumps(result, indent=1))
    print(f"\n[report] {out / f'advection_match_rec{r}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
