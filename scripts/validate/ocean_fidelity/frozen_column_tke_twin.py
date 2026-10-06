#!/usr/bin/env python
"""OUR TKE closure K on a frozen equatorial column — instant-in, K-out vs NEMO.

We read the diffusivities OUR closure would assign to an instantaneous column
WITH A GIVEN turbulent energy ``en``, and compare to NEMO's own closure
diffusivities on the SAME instantaneous column with NEMO's OWN ``en``.

WHY THE DIRECT CALL, NOT ``tke_vertical_mixing`` (codex #1, the decisive fix)
----------------------------------------------------------------------------
``tke_vertical_mixing`` ALWAYS advances ``en`` one backward-Euler step and
returns K from the UPDATED en (tke.py:2252,2288) — so even NEMO's restart en
with ``n_iterations=1`` gives K at restart+dt, not the instant.  Instead we call
the model's OWN diagnostic pair, exactly as that function calls them just before
its solve (tke.py:2242-2249), with NO time step:

    l_k, l_eps = compute_mixing_lengths(en, N2, dz_half, cfg,
                    signed_n2=?, dz_cell=?, boundary_cap=?, l_surface_anchor=?)
    K_M, K_H  = compute_K_from_tke(en, l_k, cfg, N2=N2, shear_sq=shear_sq,
                    z_interface=?, N2_prandtl=N2b, p_sh2_override=?)

The N2 / shear_sq / anchor inputs are built with the model's OWN helpers
(``_shared.compute_N2``, ``_shared.vertical_shear_squared``,
``tke._mxl0_surface_anchor``) the way the pre-loop block builds them — nothing
about the closure is hand-rolled.

NEMO REFERENCE (codex #6/#8): a day-30 INSTANTANEOUS restart carries tn/sn/un/vn
AND en, and ``avm_k``/``avt_k`` = NEMO field_def "vertical eddy viscosity/
diffusivity FROM CLOSURE SCHEMES" — the pure TKE part, before the additive IWM/
background and before EVD.  So ours-vs-avm_k/avt_k is closure-vs-closure, instant
vs instant, same en — no 5-day-mean K(mean state) artifact and no IWM confound.

CONFIG + GEOMETRY (codex #10/#2/#3): the resolved TKEConfig is DESERIALISED from
the arm's run_manifest (not a flag reparse); eos/rho_0/g come from the same
resolved config and are passed, not defaulted.  Vertical geometry is NEMO's
NATIVE ladders — gdept_1d/gdepw_1d/e3t_1d from an ORCA1 mesh_mask for the NEMO
side, and the snapshot's stored z_center_ref/z_interface_ref for the control —
never a midpoint reconstruction.  ``eta`` is REQUIRED on both sides (NEMO sshn /
our snapshot eta), never zero-defaulted.

CONTROL (instrument gate): feed OUR snapshot's own (en,T,S,u,v) through the SAME
direct pair and reproduce OUR stored K_M_diag/K_H_diag.  Our stored K is the FULL
diffusivity (closure + additive IWM); the direct call is closure-only, so the
band ratio carries the IWM shortfall — PRINTED, and the gate tolerance separates
"plumbing correct, small IWM offset" from "plumbing broken".  Non-vacuity is a
DETERMINISTIC N2 perturbation whose perturbed band-K must fall clearly BELOW the
UNPERTURBED band-K (a direct closure/closure ratio, not a stored comparison).

PRINTS NUMBERS AND RATIOS ONLY — no verdict; interpretation is the reader's.

Usage (compute node / sbatch ONLY):
    python frozen_column_tke_twin.py --mode control --snapshot <npz> \
        --manifest <run_manifest.json> --nemo-sbc <*_SBC.nc>
    python frozen_column_tke_twin.py --mode nemo \
        --restart-glob '.../ORCA1_00000720_restart_oce_*.nc' \
        --nemo-meshmask '.../mesh_mask_0001.nc' \
        --manifest <run_manifest.json> --nemo-sbc <*_SBC.nc>
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_FIDELITY_DIR = _HERE.parent
_RUN_DIR = _REPO_ROOT / "scripts" / "run"

DEFAULT_LAT_HALFWIDTH = 2.0
DEFAULT_LON_WEST = 200.0
DEFAULT_LON_EAST = 260.0
SURFACE_LO, SURFACE_HI = 5.0, 65.0
ENTRAINMENT_LO, ENTRAINMENT_HI = 65.0, 105.0     # the reported band
MIN_COLUMNS = 20                                 # gate: too few columns => FAIL

_T_CANDS = ("tn", "thetao", "votemper", "toce", "to")
_S_CANDS = ("sn", "so", "vosaline", "soce")
_U_CANDS = ("un", "uo", "vozocrtx")
_V_CANDS = ("vn", "vo", "vomecrty")
_EN_CANDS = ("en",)
_AVTK_CANDS = ("avt_k",)
_AVMK_CANDS = ("avm_k",)
_SSH_CANDS = ("sshn", "ssh", "ssh_m")
_TAUM_CANDS = ("taum", "taum_oce")


def _import_reused():
    for d in (_FIDELITY_DIR, _RUN_DIR):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import equatorial_diffusivity_oracle as oracle
    import run_scm_column_twins as twins
    return oracle, twins


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception as exc:                       # pragma: no cover
        return f"UNKNOWN ({exc})"


# ===========================================================================
# resolved config from the manifest (deserialise, do not reparse flags)
# ===========================================================================
def _get(o, dotted):
    for p in dotted.split("."):
        o = o[p]
    return o


def load_resolved_config(manifest_path: Path, allow_z0_direct: bool = False):
    """Deserialise the arm's resolved TKEConfig + eos/rho_0/g from the manifest.

    The manifest stores the FULLY RESOLVED runtime config (not just the command
    line), so the closure config is the run's config exactly — no flag reparse,
    no ``iwm=True`` bool/object mismatch (codex #10).
    """
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

    m = json.loads(Path(manifest_path).read_text())
    base = "config.resolved_config.runtime_config"
    tke_d = _get(m, base + ".physics.vertical_mixing.tke")
    if not isinstance(tke_d, dict) or "c_k" not in tke_d:
        raise SystemExit(f"{manifest_path}: resolved TKE config not found at "
                         f"{base}.physics.vertical_mixing.tke")
    fields = {k: v for k, v in tke_d.items() if k in TKEConfig._fields}
    cfg = TKEConfig(**fields)
    eos = str(_get(m, base + ".eos"))
    rho0 = float(_get(m, base + ".constants.rho_0"))
    g = float(_get(m, base + ".constants.g"))
    # This harness only replicates the pre-loop inputs for the DEFAULT shear /
    # weighting / surface-BC-level; anything else needs before-fields / face
    # state / pressure we do not construct — refuse rather than run wrong.
    unsupported = []
    if getattr(cfg, "tke_shear_production", "squared_centered") != "squared_centered":
        unsupported.append(f"tke_shear_production={cfg.tke_shear_production}")
    if getattr(cfg, "tke_shear_avm_weighting", "tpoint") != "tpoint":
        unsupported.append(f"tke_shear_avm_weighting={cfg.tke_shear_avm_weighting}")
    # nemo_z0 only adds the virtual z=0 row to the en SOLVE (tke.py
    # tke_vertical_mixing, surface_bc_level block); the direct pair
    # compute_mixing_lengths/compute_K_from_tke never reads it, so the
    # strip mode (direct K from a GIVEN en, no solve) may accept it.
    if (getattr(cfg, "tke_surface_bc_level", "interior_pinned") == "nemo_z0"
            and not allow_z0_direct):
        unsupported.append("tke_surface_bc_level=nemo_z0")
    if bool(getattr(cfg, "bottom_tke_bc", False)):
        unsupported.append("bottom_tke_bc=True")
    if cfg.n2_mode not in ("insitu", "nemo_bn2"):
        unsupported.append(f"n2_mode={cfg.n2_mode}")
    if unsupported:
        raise SystemExit(
            "frozen_column_tke_twin supports only the default direct-K inputs; "
            f"this arm needs {unsupported} which require closure state this "
            "harness does not build. Extend the harness before trusting it.")
    return cfg, eos, rho0, g


def _echo_cfg(cfg, eos, rho0, g):
    print(f"[cfg] eos={eos!r} rho_0={rho0} g={g}")
    for f in ("prognostic", "tke_mxl_choice", "n2_mode", "n2_eos_form",
              "kappa_convention", "prandtl_mode", "prandtl_ri_coeff", "c_k",
              "c_eps", "tke_background", "tke_surface_min", "kappaM_min",
              "kappaH_min", "kappaM_max", "tke_shear_production",
              "tke_surface_bc_level", "bottom_tke_bc"):
        if hasattr(cfg, f):
            print(f"[cfg]   {f} = {getattr(cfg, f)!r}")


# ===========================================================================
# velocity centering (mirror the model exactly)
# ===========================================================================
def centre_uv_extra_column(u, v):
    """OUR C-grid staggered u/v (extra face column/row) -> T-points.

    ocean_model_latlon_cgrid.py:6375-6376: u is (ny, nx+1, nlev); v is
    (ny+1, nx, nlev).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return 0.5 * (u[:, :-1, :] + u[:, 1:, :]), 0.5 * (v[:-1, :, :] + v[1:, :, :])


def centre_uv_collocated(un, vn):
    """NEMO collocated restart un/vn (same shape as tn) -> T-points.

    un(i) is the EAST face of T(i) so T(i) sits between un(i-1) and un(i);
    vn(j) the NORTH face of T(j).  Interior box, so edge columns keep one face.
    """
    un = np.asarray(un, dtype=np.float64)
    vn = np.asarray(vn, dtype=np.float64)
    u_cell = np.array(un); v_cell = np.array(vn)
    u_cell[:, 1:, :] = 0.5 * (un[:, :-1, :] + un[:, 1:, :])
    v_cell[1:, :, :] = 0.5 * (vn[:-1, :, :] + vn[1:, :, :])
    return u_cell, v_cell


# ===========================================================================
# NEMO tiled restart reassembly (no repo reader exists — minimal stitcher)
# ===========================================================================
def _var_candidates(v):
    return {"T": _T_CANDS, "S": _S_CANDS, "U": _U_CANDS, "V": _V_CANDS,
            "en": _EN_CANDS, "avt_k": _AVTK_CANDS, "avm_k": _AVMK_CANDS,
            "ssh": _SSH_CANDS, "dissl": ("dissl",)}[v]


def reassemble_restart(glob_pat, varnames, twins):
    """Stitch a global field per var from the DOMAIN_position_first/last tiles.

    NEMO XIOS metadata: DOMAIN_size_global (nx, ny), DOMAIN_position_first/last
    (1-based global i/j corners, no halos).  NEMO ELIMINATES all-land tiles, so
    uncovered cells stay NaN (land) and drop out of the wet box — not a bug.
    3-D vars -> (y, x, z); 2-D (ssh) -> (y, x).
    """
    import netCDF4 as nc
    files = sorted(_glob.glob(str(glob_pat)))
    if not files:
        raise SystemExit(f"no restart tiles match {glob_pat!r}")
    d0 = nc.Dataset(files[0])
    nx_g, ny_g = (int(x) for x in d0.getncattr("DOMAIN_size_global"))
    nlev = len(d0.dimensions["nav_lev"])
    d0.close()
    out = {}
    is3d = {}
    for f in files:
        d = nc.Dataset(f)
        i0, j0 = (int(x) for x in d.getncattr("DOMAIN_position_first"))
        i1, j1 = (int(x) for x in d.getncattr("DOMAIN_position_last"))
        ys, xs = slice(j0 - 1, j1), slice(i0 - 1, i1)
        for v in varnames:
            name = twins._find_data_var(d, _var_candidates(v))
            a = np.asarray(d[name][:], dtype=np.float64)
            a = a[0] if a.shape[0] == 1 else a[0]
            a = np.where(np.abs(a) > 1e10, np.nan, a)
            if a.ndim == 3:                         # (z, y, x) -> (y, x, z)
                a = np.moveaxis(a, 0, -1); v3 = True
            else:                                   # (y, x)
                v3 = False
            is3d[v] = v3
            if v not in out:
                out[v] = np.full((ny_g, nx_g, nlev) if v3 else (ny_g, nx_g),
                                 np.nan, dtype=np.float64)
            out[v][ys, xs] = a
        if "nav_lat" not in out:
            out["nav_lat"] = np.full((ny_g, nx_g), np.nan)
            out["nav_lon"] = np.full((ny_g, nx_g), np.nan)
        out["nav_lat"][ys, xs] = np.asarray(d["nav_lat"][:], dtype=np.float64)
        out["nav_lon"][ys, xs] = np.asarray(d["nav_lon"][:], dtype=np.float64)
        d.close()
    out["nav_lon"] = out["nav_lon"] % 360.0
    n_gap = int(np.isnan(out["nav_lat"]).sum())
    print(f"[restart] stitched {len(files)} tiles -> ({ny_g},{nx_g},{nlev}); "
          f"{n_gap} land cells eliminated by NEMO (not a gap)")
    return out


# ===========================================================================
# native vertical geometry
# ===========================================================================
def native_ladders_meshmask(mesh_path, twins):
    """ORCA1 native (dz_ref=e3t_1d, t_depth=gdept_1d, w_interior=gdepw_1d[1:])."""
    import netCDF4 as nc
    d = nc.Dataset(mesh_path)

    def g1d(name):
        return np.abs(np.asarray(d[name][:], dtype=np.float64).ravel())
    dz_ref = g1d("e3t_1d")
    t_depth = g1d("gdept_1d")
    gdepw = g1d("gdepw_1d")
    d.close()
    return dz_ref, t_depth, gdepw[1:]                # interior W-depths (nlev-1)


def native_ladders_snapshot(z):
    """OUR native ladders from the snapshot's stored centre + interior interfaces.

    z_center_ref (nlev) and z_interface_ref (nlev-1 INTERIOR) are the run's OWN
    geometry — the full interface ladder is [0, z_interface_ref, bottom] with the
    bottom reflected from the last centre, so dz_ref is exact (no midpointing).
    """
    zc = np.abs(np.asarray(z["z_center_ref"], dtype=np.float64))
    zi = np.abs(np.asarray(z["z_interface_ref"], dtype=np.float64))
    if zi.size != zc.size - 1:
        raise SystemExit("z_interface_ref must be nlev-1 (interior interfaces)")
    bottom = 2.0 * zc[-1] - zi[-1]
    full = np.concatenate([[0.0], zi, [bottom]])
    dz_ref = np.diff(full)
    if not np.all(dz_ref > 0):
        raise SystemExit("reconstructed dz_ref not all positive")
    return dz_ref, zc, zi                            # w-interior = z_interface_ref


# ===========================================================================
# SBC wind-stress modulus (fail on out-of-range record, do not clamp)
# ===========================================================================
def load_sbc_taum(sbc_path, rec, twins):
    """Return (taum_field, sbc_lat, sbc_lon) at record ``rec``.

    The SBC field is on the NEMO output grid, whose shape differs from both our
    snapshot grid and the restart grid, so its OWN nav_lat/nav_lon MUST travel
    with it — sampling it against another grid's coordinates is a shape/geo bug.
    """
    import netCDF4 as nc
    d = nc.Dataset(sbc_path)
    name = twins._find_data_var(d, _TAUM_CANDS)
    a = np.asarray(d[name][:], dtype=np.float64)
    sbc_lat = np.asarray(d["nav_lat"][:], dtype=np.float64)
    sbc_lon = np.asarray(d["nav_lon"][:], dtype=np.float64) % 360.0
    d.close()
    nt = a.shape[0]
    if not (0 <= rec < nt):
        raise SystemExit(f"SBC record {rec} out of range [0,{nt}) — no clamp")
    field = np.where(np.abs(a[rec]) > 1e10, np.nan, a[rec])
    print(f"[stress] SBC {name} record {rec}/{nt - 1} (5-day mean)")
    return field, sbc_lat, sbc_lon


# ===========================================================================
# the DIRECT closure-K read (instant-in, K-out; no solve)
# ===========================================================================
def direct_K(*, T, S, u_cell, v_cell, en, taum, eta, lat, dz_ref, t_depth_ref,
             cfg, eos_name, rho0, g, n2_scale=1.0, quiet=False):
    """(avm, avt) at interior interfaces from a GIVEN en — no en advance.

    Replicates the pre-loop inputs of ``tke_vertical_mixing`` (tke.py:2018-2231)
    with the model's OWN helpers, then calls the same diagnostic pair it calls
    at tke.py:2242-2249.  Inputs are (1, ncol, nlev) cell-centre arrays; en is
    (1, ncol, nlev-1); taum/eta/lat are (1, ncol).
    """
    import jax.numpy as jnp
    from legoesm.ocean.eos import (
        compute_ocean_rho, make_eos_fn, nemo_bn2_live_geometry,
    )
    from legoesm.ocean.vertical import (
        compute_ocean_jacobian, create_z_star_from_thicknesses,
    )
    from legoesm.ocean.physics.vertical_mixing._shared import (
        compute_N2, vertical_shear_squared,
    )
    from legoesm.ocean.physics.vertical_mixing.tke import (
        compute_K_from_tke, compute_mixing_lengths, _mxl0_surface_anchor,
    )

    dz_ref_j = jnp.asarray(dz_ref)
    z_coord = create_z_star_from_thicknesses(dz_ref_j, jnp.asarray(t_depth_ref), nemo_e3w_source="depth_difference")
    Tj = jnp.asarray(T); Sj = jnp.asarray(S)
    uj = jnp.asarray(u_cell); vj = jnp.asarray(v_cell)
    etaj = jnp.asarray(eta)
    H = jnp.full(etaj.shape, float(t_depth_ref[-1] + 0.5 * dz_ref[-1]))
    J = compute_ocean_jacobian(etaj, H, z_coord, 1.0)
    eos_fn = make_eos_fn(eos=eos_name)

    class _F:
        __slots__ = ("data",)
        def __init__(self, d): self.data = d

    class _St:
        pass
    st = _St(); st.T = _F(Tj); st.S = _F(Sj); st.eta = _F(etaj)
    rho = compute_ocean_rho(st, z_coord, J, eos_fn, rho0=rho0, g=g)

    dz_half = jnp.asarray(z_coord.dz_half_ref) * J[..., None]
    z_int = jnp.asarray(z_coord.z_half_ref[1:-1])
    # the model's caller (k_profiles.py) threads t_depth, w_depth AND the raw-mesh e3w
    t_depth, w_depth, e3w_int = nemo_bn2_live_geometry(z_coord, etaj, H)

    # --- N2 exactly as the pre-loop block builds it (tke.py:2148) ---
    N2 = compute_N2(
        rho, dz_half, float(rho0), float(g),
        T_cell=Tj, S_cell=Sj, dz_ref=dz_ref_j, jacobian=J, eos_fn=eos_fn,
        n2_mode=cfg.n2_mode, n2_eos_form=getattr(cfg, "n2_eos_form", "seos"),
        adiabatic_over_dz_half=False, t_depth=t_depth, w_depth=w_depth,
        e3w_int=e3w_int)
    # n2_scale: layering probe only (uniform N2 scaling, en/shear held fixed)
    N2 = N2 * n2_scale
    signed_n2 = cfg.n2_mode in ("adiabatic", "nemo_bn2")
    # squared_centered shear (the only supported discretization here; tke.py:2105)
    shear_sq = vertical_shear_squared(uj, vj, dz_half)
    # tke_mxl_choice 3/4 cell thickness for the lup/ldown sweeps (tke.py:2018)
    dz_cell_mxl = dz_ref_j * J[..., None]
    # ln_mxl0 surface anchor (tke.py:2231)
    l_anchor = _mxl0_surface_anchor(cfg, jnp.asarray(taum), float(rho0), float(g))

    enj = jnp.asarray(en)
    l_k, _l_eps = compute_mixing_lengths(
        enj, N2, dz_half, cfg, signed_n2=signed_n2,
        dz_cell=dz_cell_mxl, boundary_cap=None, l_surface_anchor=l_anchor)
    K_M, K_H = compute_K_from_tke(
        enj, l_k, cfg, N2=N2, shear_sq=shear_sq, z_interface=z_int,
        N2_prandtl=N2, p_sh2_override=None)
    zk = np.abs(np.asarray(z_coord.z_half_ref[1:-1]))
    # State-swap decomposition (GLM 2026-08-27): band-median en, N2, S2, Ri of
    # the state being fed to the closure, so a control run (our snapshot state)
    # and a nemo run (NEMO restart state) reveal WHICH input carries the coupled
    # avm gap. l_k is master length; K_M ~ c_k*l_k*sqrt(2 en).
    _en = np.asarray(enj)[0]; _n2 = np.asarray(N2)[0]
    _s2 = np.asarray(shear_sq)[0]; _lk = np.asarray(l_k)[0]
    _bk = (zk >= ENTRAINMENT_LO) & (zk <= ENTRAINMENT_HI)
    if _bk.any() and not quiet:
        def _bm(a):
            v = a[:, _bk]; v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else float("nan")
        _ri = _bm(np.where(_s2 > 0, _n2 / np.where(_s2 > 0, _s2, np.nan),
                           np.nan))
        print(f"[state] band 65-105 m: en={_bm(_en):.3e}  N2={_bm(_n2):.3e}  "
              f"S2={_bm(_s2):.3e}  Ri={_ri:.3f}  l_k={_bm(_lk):.3e} m  "
              f"sqrt(2en)={_bm(np.sqrt(2*np.abs(_en))):.3e}")
    return np.asarray(K_M)[0], np.asarray(K_H)[0], zk, np.asarray(N2)[0]


# ===========================================================================
# ONE reduction, matching the oracle (spatial median per depth -> band median),
# through a COMMON validity mask so ours/stored/NEMO drop the SAME cells.
# ===========================================================================
def common_mask(*arrs):
    """(ncol, nk) boolean: finite AND positive on EVERY operand.

    The operands share the interior-interface indexing (k -> the same physical
    interface), so a single per-(column, interface) mask applies to all — our
    floored-positive sub-seafloor rows and NEMO's zeroed ones drop together, and
    no operand independently keeps a cell another one lacks (codex #4).  NEMO's
    wet/tmask is captured by ``avm_k > 0`` (NEMO zeroes dry interfaces).
    """
    m = None
    for a in arrs:
        a = np.asarray(a, dtype=np.float64)
        ok = np.isfinite(a) & (a > 0)
        m = ok if m is None else (m & ok)
    return m


def band_reduce(K_cols, zk, lo, hi, mask=None):
    """Spatial median per depth over the masked columns, then median over band."""
    K = np.asarray(K_cols, dtype=np.float64)
    zk = np.abs(np.asarray(zk, dtype=np.float64))
    prof = []
    for k in range(K.shape[1]):
        sel = np.isfinite(K[:, k]) & (K[:, k] > 0)
        if mask is not None:
            sel = sel & mask[:, k]
        col = K[sel, k]
        if col.size:
            prof.append((zk[k], float(np.median(col))))
    band = [p for d, p in prof if lo <= d <= hi]
    return float(np.median(band)) if band else float("nan")


def _finite_positive(*vals):
    return all(np.isfinite(v) and v > 0 for v in vals)


def sample_at_columns(field2d, flat_lat, flat_lon, col_lat, col_lon, twins):
    fin = np.isfinite(field2d)
    out = np.empty(col_lat.size)
    for c in range(col_lat.size):
        dd = twins.great_circle_deg(flat_lat, flat_lon,
                                    float(col_lat[c]), float(col_lon[c]))
        j, i = np.unravel_index(
            int(np.argmin(np.where(fin, dd, np.inf))), dd.shape)
        out[c] = field2d[j, i]
    return out


def _box(lat, lon, args):
    return ((np.abs(lat) <= args.lat_halfwidth)
            & (lon >= args.lon_west) & (lon <= args.lon_east))


# ===========================================================================
# modes
# ===========================================================================
def run_control(args, oracle, twins):
    cfg, eos, rho0, g = load_resolved_config(Path(args.manifest))
    _echo_cfg(cfg, eos, rho0, g)

    z = np.load(args.snapshot)
    need = ("T", "S", "u", "v", "tke", "eta", "z_center_ref",
            "z_interface_ref", "land_mask", "lat_T", "lon_T")
    miss = [k for k in need if k not in z]
    if miss:
        raise SystemExit(f"{args.snapshot}: missing {miss}")
    lat = np.asarray(z["lat_T"], dtype=np.float64)
    lon = np.asarray(z["lon_T"], dtype=np.float64) % 360.0
    if np.nanmax(np.abs(lat)) <= np.pi + 1e-6:
        lat, lon = np.degrees(lat), np.degrees(lon) % 360.0
    wet = np.asarray(z["land_mask"], dtype=np.float64) > 0.5
    band = _box(lat, lon, args) & wet
    ncol = int(band.sum())
    if ncol < MIN_COLUMNS:
        raise SystemExit(f"only {ncol} columns in box (< {MIN_COLUMNS}) — FAIL")

    dz_ref, t_depth_ref, _wint = native_ladders_snapshot(z)
    u_cell, v_cell = centre_uv_extra_column(z["u"], z["v"])
    taum, sbc_lat, sbc_lon = load_sbc_taum(args.nemo_sbc, args.rec, twins)
    tau_cols = sample_at_columns(taum, sbc_lat, sbc_lon, lat[band], lon[band],
                                 twins)
    print(f"[stress] control |tau| median {np.nanmedian(tau_cols):.4f} N/m^2")
    print("[caveat] taum is a 5-DAY-MEAN SBC record and H is a global-constant "
          "depth; both feed only the ln_mxl0 SURFACE anchor, which has decayed "
          "by the 65-105 m entrainment band, so the reported band K is weakly "
          "sensitive to either (no per-column bathymetry invented).")

    def call(Tc, Sc, en_scale=1.0):
        return direct_K(
            T=Tc[band][None], S=Sc[band][None],
            u_cell=u_cell[band][None], v_cell=v_cell[band][None],
            en=np.asarray(z["tke"])[band][None] * en_scale, taum=tau_cols[None, :],
            eta=np.asarray(z["eta"])[band][None], lat=lat[band][None, :],
            dz_ref=dz_ref, t_depth_ref=t_depth_ref, cfg=cfg,
            eos_name=eos, rho0=rho0, g=g)

    Tc = np.asarray(z["T"], dtype=np.float64)
    Sc = np.asarray(z["S"], dtype=np.float64)
    # State fed to the gate: perturbed under --perturb-en (non-vacuity), else the
    # true snapshot state.  Non-vacuity scales the INPUT turbulent energy en:
    # K_M = c_k * l_k * sqrt(2 en) with l_k ~ sqrt(2 en)/N, so K ~ en — a strong,
    # monotone, deterministic coupling (the earlier T,S->N2 perturbation moved the
    # band N2 only ~1.24x, codex #6).  The resulting band-K ratio is measured, and
    # the REAL STEP-1 gate must reject.
    avm, avt, zk, N2g = call(Tc, Sc, en_scale=args.perturb_en)

    # our stored K (FULL = closure + additive IWM), reduced over the SAME wet box
    # columns and the SAME interior interfaces as ours (codex #4 common mask).
    oavm, oavt, _b, _olat, _olon, ozk = oracle._load_ours(Path(args.snapshot))
    s_avm_cols, s_avt_cols = oavm[band], oavt[band]
    maskM = common_mask(avm, s_avm_cols)
    maskH = common_mask(avt, s_avt_cols)
    c_avm = band_reduce(avm, zk, ENTRAINMENT_LO, ENTRAINMENT_HI, maskM)
    c_avt = band_reduce(avt, zk, ENTRAINMENT_LO, ENTRAINMENT_HI, maskH)
    st_avm = band_reduce(s_avm_cols, ozk, ENTRAINMENT_LO, ENTRAINMENT_HI, maskM)
    st_avt = band_reduce(s_avt_cols, ozk, ENTRAINMENT_LO, ENTRAINMENT_HI, maskH)

    print("\n[control] entrainment band 65-105 m (common wet/interface mask):")
    print(f"[control]   our closure   avm={c_avm:.4e}  avt={c_avt:.4e}")
    print(f"[control]   our stored    avm={st_avm:.4e}  avt={st_avt:.4e}  "
          "(FULL = closure + additive IWM)")
    if not _finite_positive(c_avm, c_avt, st_avm, st_avt):
        print("[control]   non-finite/empty band reduction -> FAIL")
        return 4
    r_m, r_t = c_avm / st_avm, c_avt / st_avt
    lo, hi = 1.0 / args.control_tol, args.control_tol
    in_band = (lo <= r_m <= hi) and (lo <= r_t <= hi)
    print(f"[control]   ratio closure/stored  avm x{r_m:.3f}  avt x{r_t:.3f}  "
          f"(in [x{1/args.control_tol:.2f}, x{args.control_tol}] = {in_band}; "
          "shortfall = additive-IWM fraction)")

    if args.perturb_en != 1.0:
        # Real-gate non-vacuity (codex #5): rerun the ACTUAL STEP-1 assertion
        # (closure/stored in the band) under an en perturbation and require it to
        # now FAIL.  MEASURE the resulting band-K ratio so the effect is verified.
        avm_b, _, _, _ = call(Tc, Sc)
        k_base = band_reduce(avm_b, zk, ENTRAINMENT_LO, ENTRAINMENT_HI, maskM)
        k_ratio = c_avm / k_base if k_base else float("nan")
        print(f"[control] NON-VACUITY en x{args.perturb_en}; MEASURED band avm "
              f"ratio perturbed/control = x{k_ratio:.3f}")
        broke = not in_band
        print(f"[control] NON-VACUITY: STEP-1 assertion now FAILS = {broke} "
              "(must be True -> the real gate can reject)")
        return 0 if broke else 3

    print(f"[control]   CONFIRMED: closure reproduces stored K within x"
          f"{args.control_tol} = {in_band}")
    return 0 if in_band else 4


def run_nemo(args, oracle, twins):
    cfg, eos, rho0, g = load_resolved_config(Path(args.manifest))
    _echo_cfg(cfg, eos, rho0, g)

    R = reassemble_restart(
        args.restart_glob, ("T", "S", "U", "V", "en", "avt_k", "avm_k", "ssh"),
        twins)
    nav_lat, nav_lon = R["nav_lat"], R["nav_lon"]
    wet = np.isfinite(R["T"][..., 0])
    band = _box(nav_lat, nav_lon, args) & wet
    ncol = int(band.sum())
    if ncol < MIN_COLUMNS:
        raise SystemExit(f"only {ncol} NEMO columns in box (< {MIN_COLUMNS}) — FAIL")
    print(f"[nemo] {ncol} equatorial columns")

    dz_ref, t_depth_ref, w_interior = native_ladders_meshmask(
        args.nemo_meshmask, twins)
    u_cell, v_cell = centre_uv_collocated(R["U"], R["V"])
    if "ssh" not in R or np.isnan(R["ssh"][band]).all():
        raise SystemExit("restart carries no usable sshn (eta) — required")
    eta = R["ssh"][band][None, :]
    taum, sbc_lat, sbc_lon = load_sbc_taum(args.nemo_sbc, args.rec, twins)
    tau_cols = sample_at_columns(taum, sbc_lat, sbc_lon,
                                 nav_lat[band], nav_lon[band], twins)
    print(f"[stress] NEMO |tau| median {np.nanmedian(tau_cols):.4f} N/m^2")
    print("[caveat] taum is a 5-DAY-MEAN SBC record and H is a global-constant "
          "depth; both feed only the ln_mxl0 SURFACE anchor, weak at the "
          "65-105 m band (no per-column bathymetry invented).")

    avm, avt, zk, _N2 = direct_K(
        T=R["T"][band][None], S=R["S"][band][None],
        u_cell=u_cell[band][None], v_cell=v_cell[band][None],
        en=R["en"][band][:, 1:][None], taum=tau_cols[None, :], eta=eta,
        lat=nav_lat[band][None, :], dz_ref=dz_ref, t_depth_ref=t_depth_ref,
        cfg=cfg, eos_name=eos, rho0=rho0, g=g)

    # NEMO's OWN closure diffusivities (w-points; drop surface k=0 -> interior).
    n_avmk = R["avm_k"][band][:, 1:]
    n_avtk = R["avt_k"][band][:, 1:]
    ok_all = True
    for name, lo, hi in (("SURFACE 5-65 m", SURFACE_LO, SURFACE_HI),
                         ("ENTRAINMENT 65-105 m", ENTRAINMENT_LO, ENTRAINMENT_HI)):
        # PRIMARY: avm (viscosity) — the EUC-relevant quantity, and apples-to-
        # apples (no Prandtl time-level difference).  Common mask across ours+NEMO.
        maskM = common_mask(avm, n_avmk)
        o_m = band_reduce(avm, zk, lo, hi, maskM)
        n_m = band_reduce(n_avmk, w_interior, lo, hi, maskM)
        print(f"\n[nemo] {name}:")
        print(f"[nemo]   PRIMARY avm  ours={o_m:.4e}  NEMO avm_k={n_m:.4e}", end="")
        if _finite_positive(o_m, n_m):
            print(f"  ->  ratio ours/NEMO x{o_m / n_m:.3f}")
        else:
            print("  ->  non-finite/empty band reduction"); ok_all = False
        # SECONDARY: avt (heat) — NEMO's avt_k applies a post-solve Prandtl
        # pdlr from rn2b (BEFORE-level N2) and face-native Burchard shear, while
        # our direct call uses current N2 + centered u/v, so this is NOT
        # apples-to-apples (codex #1). Reported, caveated, not gated.
        maskH = common_mask(avt, n_avtk)
        o_t = band_reduce(avt, zk, lo, hi, maskH)
        n_t = band_reduce(n_avtk, w_interior, lo, hi, maskH)
        print(f"[nemo]   avt (heat, Prandtl-time-level CAVEAT — NOT apples-to-"
              f"apples)  ours={o_t:.4e}  NEMO avt_k={n_t:.4e}", end="")
        if _finite_positive(o_t, n_t):
            print(f"  ->  ratio x{o_t / n_t:.3f}")
        else:
            print("  ->  non-finite/empty band reduction")
    # Per-interface box means (common mask): where in the column does our closure
    # disagree with NEMO's on NEMO's own state?  Ratio-of-means, top 20 interfaces.
    maskM = common_mask(avm, n_avmk)
    zk_np = np.asarray(zk).ravel()
    print("\n[nemo] per-interface box means (ours on NEMO state vs NEMO avm_k/avt_k; l = K_M/(c_k sqrt(en)))")
    print("   k    z_w m |  avm ours  avm_k NEMO  ratio |  avt ours  avt_k NEMO  ratio |   l ours   l NEMO")
    en_int = np.asarray(R["en"][band][:, 1:], dtype=np.float64)
    ck = float(getattr(cfg, "c_k", 0.1))
    for k in range(min(20, zk_np.size)):
        mk = np.asarray(maskM)[..., k].ravel()
        if not mk.any():
            continue
        o = float(np.mean(np.asarray(avm)[..., k].ravel()[mk])); n_ = float(np.mean(np.asarray(n_avmk)[..., k].ravel()[mk]))
        ot = float(np.mean(np.asarray(avt)[..., k].ravel()[mk])); nt = float(np.mean(np.asarray(n_avtk)[..., k].ravel()[mk]))
        e_ = float(np.mean(en_int[..., k].ravel()[mk]))
        lo_ = o / (ck * np.sqrt(max(e_, 1e-30))); ln_ = n_ / (ck * np.sqrt(max(e_, 1e-30)))
        print(f"  {k + 1:2d}  {zk_np[k]:7.2f} | {o:9.3e} {n_:10.3e} {o / n_ if n_ > 0 else float('nan'):6.3f} | {ot:9.3e} {nt:10.3e} {ot / nt if nt > 0 else float('nan'):6.3f} | {lo_:8.3f} {ln_:8.3f}")
    print("\n[nemo] instant-vs-instant, NEMO's OWN en, closure-vs-closure. "
          "avm is the PRIMARY trusted number; avt carries the Prandtl "
          "time-level caveat. Numbers only, no verdict.")
    return 0 if ok_all else 5


def load_hourly_taum(path, hour_index):
    """Hourly-mean |tau| (NEMO trd1h_T ``taum``) at record ``hour_index``, with
    the file's own nav_lat/nav_lon (output grid). No clamp on the record."""
    import netCDF4 as nc
    d = nc.Dataset(path)
    nt = d["taum"].shape[0]
    if not (0 <= hour_index < nt):
        raise SystemExit(f"taum record {hour_index} out of range [0,{nt})")
    a = np.asarray(d["taum"][hour_index], dtype=np.float64)
    lat = np.asarray(d["nav_lat_grid_T"][:], dtype=np.float64)
    lon = np.asarray(d["nav_lon_grid_T"][:], dtype=np.float64) % 360.0
    d.close()
    return np.where(np.abs(a) > 1e10, np.nan, a), lat, lon


def ratio_class(r, tol=1.2, out=2.0):
    """Symmetric pre-registered class of a ratio: within [1/tol, tol],
    INCONCLUSIVE up to ``out``x, else OUTSIDE."""
    a = max(r, 1.0 / r) if (np.isfinite(r) and r > 0) else float("inf")
    return "within" if a <= tol else ("INCONCLUSIVE" if a < out else "OUTSIDE")


def _med(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def run_strip(args, oracle, twins):
    """Instant closure K, OURS (production config) vs NEMO avm_k/avt_k, on
    NEMO's own restart columns in a strip, per interface (2-30 m).

    Per-column ratios (median over columns) per interface; the 8-18 m band;
    the top-decile |dN2/dk| subset (where staircases nucleate); rank
    correlation of log(avt ratio) with N2; sign alternation of the log ratio
    across adjacent interfaces; fixed-en layering slope dln(K_H N2)/dln N2 for
    ours and for a NEMO-Prandtl variant (prandtl_mode=nemo_ri), the variant
    first scored against NEMO avt_k on the same state. Numbers only.
    """
    cfg, eos, rho0, g = load_resolved_config(Path(args.manifest), allow_z0_direct=True)
    _echo_cfg(cfg, eos, rho0, g)
    R = reassemble_restart(
        args.restart_glob, ("T", "S", "U", "V", "en", "avt_k", "avm_k", "ssh"),
        twins)
    nav_lat, nav_lon = R["nav_lat"], R["nav_lon"]
    wet = np.isfinite(R["T"][..., 0])
    band = _box(nav_lat, nav_lon, args) & wet
    ncol = int(band.sum())
    if ncol < MIN_COLUMNS:
        raise SystemExit(f"only {ncol} NEMO columns in strip (< {MIN_COLUMNS}) — FAIL")
    dz_ref, t_depth_ref, w_interior = native_ladders_meshmask(args.nemo_meshmask, twins)
    u_cell, v_cell = centre_uv_collocated(R["U"], R["V"])
    taum, tlat, tlon = load_hourly_taum(args.taum_file, args.taum_hour)
    tau_cols = sample_at_columns(taum, tlat, tlon, nav_lat[band], nav_lon[band], twins)
    print(f"[strip] {ncol} columns; |tau| median {np.nanmedian(tau_cols):.4f} N/m^2 "
          f"(hourly mean record {args.taum_hour}); dtype T {R['T'].dtype}")

    def call(c, n2s=1.0, en_scale=1.0):
        return direct_K(
            T=R["T"][band][None], S=R["S"][band][None],
            u_cell=u_cell[band][None], v_cell=v_cell[band][None],
            en=R["en"][band][:, 1:][None] * en_scale, taum=tau_cols[None, :],
            eta=R["ssh"][band][None, :], lat=nav_lat[band][None, :],
            dz_ref=dz_ref, t_depth_ref=t_depth_ref, cfg=c, eos_name=eos,
            rho0=rho0, g=g, n2_scale=n2s, quiet=True)

    # NEMO floors avm_k/avt_k at avmb/avtb; with ln_zdfiwm=T those are 1.4e-6 /
    # 1e-10 (zdfiwm init) = our kappaM_min / kappaH_min, so floors match.
    avm, avt, zk, N2 = call(cfg, en_scale=args.perturb_en)
    print(f"[strip] dtypes: avm {avm.dtype} N2 {N2.dtype} dz_ref {np.asarray(dz_ref).dtype}")
    n_avm = R["avm_k"][band][:, 1:]
    n_avt = R["avt_k"][band][:, 1:]
    # index-mapping check only: cumsum(e3t_1d) and gdepw_1d differ by <1 cm
    if not np.allclose(np.asarray(zk)[:20], np.asarray(w_interior)[:20], atol=0.05):
        raise SystemExit("our interface ladder != NEMO gdepw interior (k mapping broken)")
    kmax = int(np.searchsorted(np.asarray(zk), args.zmax))
    mM = common_mask(avm, n_avm)
    mH = common_mask(avt, n_avt)
    rM = np.where(mM, avm / np.where(mM, n_avm, 1.0), np.nan)
    rH = np.where(mH, avt / np.where(mH, n_avt, 1.0), np.nan)
    print("\n   k    z_w | avm ours  avm_k NEMO  med(r) | avt ours  avt_k NEMO  med(r) |  N2 med")
    for k in range(kmax):
        print(f"  {k + 1:2d} {zk[k]:6.2f} | {_med(avm[:, k][mM[:, k]]):9.3e} {_med(n_avm[:, k][mM[:, k]]):10.3e} "
              f"{_med(rM[:, k]):6.3f} | {_med(avt[:, k][mH[:, k]]):9.3e} {_med(n_avt[:, k][mH[:, k]]):10.3e} "
              f"{_med(rH[:, k]):6.3f} | {_med(N2[:, k]):.2e}")
    bk = (zk >= args.band_lo) & (zk <= args.band_hi)
    def q(x):
        x = np.asarray(x)[np.isfinite(x)]
        return "nan" if not x.size else "/".join(f"{v:.3f}" for v in np.percentile(x, [25, 50, 75]))
    mb_m, mb_t = _med(rM[:, bk]), _med(rH[:, bk])
    print(f"\n[band {args.band_lo:g}-{args.band_hi:g} m] per-column ratio p25/p50/p75  avm {q(rM[:, bk])}  "
          f"avt {q(rH[:, bk])}  (n={int(np.isfinite(rH[:, bk]).sum())})")
    # per-COLUMN band summary (median over the column's band interfaces), then
    # quartiles over columns (codex: pooled column-interface samples overweight)
    colM, colH = np.nanmedian(rM[:, bk], axis=1), np.nanmedian(rH[:, bk], axis=1)
    print(f"[band] per-column band medians, quartiles over {int(np.isfinite(colH).sum())} columns: "
          f"avm {q(colM)}  avt {q(colH)}")
    # floor occupancy + unfloored score (samples where both sides >= 10x floor)
    fm, fh = float(cfg.kappaM_min), float(cfg.kappaH_min)
    flM = (avm[:, bk] <= 1.0001 * fm) | (n_avm[:, bk] <= 1.0001 * fm)
    flH = (avt[:, bk] <= 1.0001 * fh) | (n_avt[:, bk] <= 1.0001 * fh)
    upM = (avm[:, bk] >= 10 * fm) & (n_avm[:, bk] >= 10 * fm)
    upH = (avt[:, bk] >= 10 * fh) & (n_avt[:, bk] >= 10 * fh)
    print(f"[band] floor-bound fraction avm {float(flM.mean()):.3f} avt {float(flH.mean()):.3f}; "
          f"unfloored (both >=10x floor) avm {q(rM[:, bk][upM])}  avt {q(rH[:, bk][upH])}")
    cM, cH = ratio_class(_med(colM)), ratio_class(_med(colH))
    print(f"[band] pre-registered class (per-column median, [1/1.2,1.2]; INCONCLUSIVE <2x): avm {cM}  avt {cH}")
    planted_ok = None
    if args.perturb_en != 1.0:
        planted_ok = (cM != "within") and (cH != "within")
        print(f"[band] PLANTED en x{args.perturb_en}: gate rejects = {planted_ok} (must be True)")
    # staircase nucleation sites: top-decile |N2(k) - mean(N2(k-1), N2(k+1))|
    curv = np.full_like(N2, np.nan)
    zz = np.asarray(zk, dtype=np.float64)
    wgt = (zz[1:-1] - zz[:-2]) / (zz[2:] - zz[:-2])          # depth-linear interp weight
    curv[:, 1:-1] = np.abs(N2[:, 1:-1] - ((1 - wgt) * N2[:, :-2] + wgt * N2[:, 2:]))
    cb = np.where(bk[None, :] & np.isfinite(rH), curv, np.nan)
    thr = np.nanpercentile(cb, 90)
    top = cb >= thr
    print(f"[band] top-decile |N2 curvature| interfaces: avm {_med(rM[top]):.3f}  avt {_med(rH[top]):.3f}  (n={int(top.sum())})")
    lr = np.log(rH[:, bk]); n2b = N2[:, bk]
    ok = np.isfinite(lr) & np.isfinite(n2b)
    if ok.sum() > 10:
        from scipy.stats import spearmanr                   # tie-aware ranks
        print(f"[band] Spearman(log avt ratio, N2) = {spearmanr(lr[ok], n2b[ok]).statistic:+.3f}")
    d = np.diff(np.sign(np.log(rH[:, bk])), axis=1)
    alt = np.isfinite(d)
    print(f"[band] adjacent-interface sign flips of log(avt ratio): {float((np.abs(d[alt]) > 1).mean()):.3f} "
          "(0 = smooth, ~0.5 = random, 1 = alternating)")
    # Prandtl branch occupancy in the band (ours: Pr = clip(c*N2/S2, 1, 10))
    with np.errstate(divide="ignore", invalid="ignore"):
        prr = np.where(avt[:, bk] > 1.0001 * fh, avm[:, bk] / avt[:, bk], np.nan)
    fin = np.isfinite(prr)
    print(f"[band] Prandtl branch occupancy: Pr~1 {float((prr[fin] < 1.01).mean()):.3f}  "
          f"linear {float(((prr[fin] >= 1.01) & (prr[fin] <= 9.9)).mean()):.3f}  Pr~10 {float((prr[fin] > 9.9).mean()):.3f}")
    # fixed-en LOCAL layering slope: perturb N2 at ONE interface k (x0.9/x1.1),
    # read the buoyancy flux K_H*N2 at that same k (mixing-length sweeps couple
    # neighbours, so a uniform scaling is not a local test). CAVEAT: en fixed.
    # OURS only: NEMO's pdlr = ri_c/max(ri_c, zri) is the same law (zdftke.F90
    # nn_pdl=1), so a 'NEMO variant' of our code is not independent evidence.
    for eps in (0.1, 0.02):
        sls = []
        for k in np.where(bk)[0]:
            fac = np.ones(N2.shape[1]); fac[k] = 1.0 - eps
            _, a_lo, _, n_lo = call(cfg, fac[None, None, :])
            fac[k] = 1.0 + eps
            _, a_hi, _, n_hi = call(cfg, fac[None, None, :])
            with np.errstate(divide="ignore", invalid="ignore"):
                sl = (np.log(a_hi[:, k] * n_hi[:, k]) - np.log(a_lo[:, k] * n_lo[:, k])) / (np.log(1 + eps) - np.log(1 - eps))
            sls.append(np.where((n_lo[:, k] > 0) & (n_hi[:, k] > 0), sl, np.nan))
        sl = np.stack(sls, axis=1)
        print(f"[layering local fixed-en, ours, +/-{eps:g}] slope p25/p50/p75 {q(sl)}  frac<0 "
              f"{float((sl[np.isfinite(sl)] < 0).mean()):.3f}")
    print("\n[strip] instant closure map on NEMO's own state + en (restart T/S/u/v are one "
          "RK3 step newer than NEMO's closure inputs); numbers only, no verdict.")
    if planted_ok is False:
        return 6
    return 0


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("control", "nemo", "strip"))
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--nemo-sbc", type=Path, default=None,
                   help="control/nemo modes: 5-day SBC file (taum)")
    p.add_argument("--rec", type=int, default=5,
                   help="SBC record (fails if out of range; no clamp)")
    p.add_argument("--lat-halfwidth", type=float, default=DEFAULT_LAT_HALFWIDTH)
    p.add_argument("--lon-west", type=float, default=DEFAULT_LON_WEST)
    p.add_argument("--lon-east", type=float, default=DEFAULT_LON_EAST)
    p.add_argument("--snapshot", type=Path, default=None)
    p.add_argument("--perturb-en", type=float, default=1.0,
                   help="non-vacuity: scale input TKE en by this (K ~ en); the "
                        "STEP-1 closure/stored band assertion must then FAIL "
                        "(real gate)")
    p.add_argument("--control-tol", type=float, default=1.5)
    p.add_argument("--restart-glob", type=str, default=None)
    p.add_argument("--taum-file", type=Path, default=None,
                   help="strip mode: NEMO trd1h_T file carrying hourly taum")
    p.add_argument("--taum-hour", type=int, default=None,
                   help="strip mode: hourly record (0-based) for the restart's hour")
    p.add_argument("--band-lo", type=float, default=8.0)
    p.add_argument("--band-hi", type=float, default=18.0)
    p.add_argument("--zmax", type=float, default=30.0)
    p.add_argument("--nemo-meshmask", type=Path, default=None,
                   help="an ORCA1 mesh_mask tile (native gdept_1d/gdepw_1d/e3t_1d)")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    oracle, twins = _import_reused()
    print(f"[prov] git {_git_sha()}")
    print(f"[prov] mode={args.mode} box |lat|<={args.lat_halfwidth} "
          f"{args.lon_west:.0f}-{args.lon_east:.0f}E")
    if args.mode in ("control", "nemo") and args.nemo_sbc is None:
        raise SystemExit(f"{args.mode} mode needs --nemo-sbc")
    if args.mode == "control":
        if args.snapshot is None:
            raise SystemExit("control mode needs --snapshot")
        print(f"[prov] snapshot {args.snapshot}")
        return run_control(args, oracle, twins)
    if args.restart_glob is None or args.nemo_meshmask is None:
        raise SystemExit("nemo/strip mode needs --restart-glob and --nemo-meshmask")
    if args.mode == "strip":
        # oracle-fidelity rule 1c: fp64 policy, not only JAX x64 (constructors
        # cast geometry to the policy control dtype).
        if args.taum_file is None or args.taum_hour is None:
            raise SystemExit("strip mode needs --taum-file and --taum-hour")
        from legoesm.core.precision import PrecisionPolicy, set_policy
        set_policy(PrecisionPolicy.fp64())
        print(f"[prov] restart {args.restart_glob}  taum {args.taum_file}[{args.taum_hour}]")
        return run_strip(args, oracle, twins)
    print(f"[prov] restart {args.restart_glob}")
    print(f"[prov] meshmask {args.nemo_meshmask}")
    return run_nemo(args, oracle, twins)


if __name__ == "__main__":
    raise SystemExit(main())
