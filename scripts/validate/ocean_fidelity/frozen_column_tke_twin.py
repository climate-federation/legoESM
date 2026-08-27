#!/usr/bin/env python
"""OUR TKE closure on a frozen equatorial column — instant-vs-instant vs NEMO.

THE DECISIVE TEST this campaign did not yet have.  We proved our TKE closure is
INTERNALLY self-consistent (implied mixing length matches buoyancy length), but
that never proved it equals NEMO's closure when BOTH see the SAME instantaneous
state WITH THE SAME turbulent energy.  Here we feed NEMO's OWN instantaneous
equatorial column (tn, sn, un, vn) AND NEMO's OWN turbulent energy ``en`` into
our closure, read avm (K_M) / avt (K_H) from a SINGLE evaluation, and compare to
NEMO's OWN closure diffusivities ``avm_k`` / ``avt_k`` at the same column.

WHY THIS IS THE RIGHT COMPARISON (codex findings #6 and #8, resolved)
--------------------------------------------------------------------
* #6 — the old grid_W ``avm``/``avt`` are 5-DAY MEANS, and K(mean state) is not
  mean(K(state)); that comparison is dead.  A NEMO INSTANTANEOUS restart exists
  (step 720 = day 30, dt=3600 s), carrying tn/sn/un/vn AND en/avt_k/avm_k on the
  native grid.  We compare instant-to-instant, with NEMO's OWN en, so there is
  no free-spin and no equilibrium assumption on the NEMO side — using NEMO's en
  is the point.
* #8 — the IWM confound.  NEMO field_def_nemo-oce.xml (5.0.1):
    avt_k = "vertical eddy diffusivity FROM CLOSURE SCHEMES"
    avm_k = "vertical eddy viscosity   FROM CLOSURE SCHEMES"
  i.e. the PURE TKE-closure part, BEFORE the additive internal-wave/background
  mixing (avt total) and BEFORE the convective enhancement (avt_evd).  That is
  exactly what ``tke_vertical_mixing`` returns, so ours-vs-avt_k is apples to
  apples — the additive IWM K never enters this comparison.

SELF-RECOVERY CONTROL (the instrument gate, kept — it is instant-vs-instant too)
--------------------------------------------------------------------------------
Feed OUR snapshot's own column + OUR own stored ``tke`` into our closure and
reproduce OUR stored ``K_M_diag``/``K_H_diag``.  A plumbing defect (wrong
geometry, config, centering, N2 ladders, EOS) shows up as an order-of-magnitude
band mismatch here, so the NEMO number is trusted only after this passes.  Our
stored K is the FULL diffusivity (closure + additive IWM), so the control ratio
carries the additive-IWM offset — it is PRINTED, and the gate tolerance is set
to separate "plumbing correct, small IWM offset" from "plumbing broken", never
to hide it.  Non-vacuity uses a DETERMINISTIC N2 perturbation with a stated
expected band-K effect, not a shear scale (at floored en the band K is
insensitive to shear — codex #2 — but K ~ 1/sqrt(N2) via the buoyancy length
holds even at the floor).

This probe PRINTS NUMBERS AND RATIOS ONLY.  No verdict; interpretation is the
reader's.

WHAT IT REUSES (RULE 4 — searched, nothing to reuse for the tiled restart)
--------------------------------------------------------------------------
* NEMO var-name detection / nearest-wet: ``run_scm_column_twins`` helpers.
* stored-K box loader: ``equatorial_diffusivity_oracle._load_ours``.
* the closure: the model's OWN ``tke_vertical_mixing``, called as the production
  prognostic step calls it (``vertical_mixing/k_profiles.py`` L926).
* config/EOS/geometry: ``run_omip_core2.build_tripole_vmix_config`` +
  ``ocean.eos`` (``make_eos_fn``, ``compute_ocean_rho``, ``nemo_bn2_live_ladders``)
  + ``ocean.vertical`` — with eos/rho_0/g READ FROM the arm manifest and asserted.
* NEMO tiled restart reassembly: no repo reader exists (searched restart/nimpp/
  njmpp/rebuild) — a minimal DOMAIN_position_first/last stitcher is written here.

Usage (compute node / sbatch ONLY):
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -u frozen_column_tke_twin.py \
        --mode control --snapshot <snap.npz> --manifest <run_manifest.json> \
        --nemo-sbc <..._SBC.nc>
    ... --mode nemo --restart-glob '.../ORCA1_00000720_restart_oce_*.nc' \
        --manifest <run_manifest.json> --nemo-sbc <..._SBC.nc>
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_FIDELITY_DIR = _HERE.parent
_RUN_DIR = _REPO_ROOT / "scripts" / "run"

# Equatorial cold-tongue box (oracle defaults).
DEFAULT_LAT_HALFWIDTH = 2.0
DEFAULT_LON_WEST = 200.0
DEFAULT_LON_EAST = 260.0

SURFACE_LO, SURFACE_HI = 5.0, 65.0
ENTRAINMENT_LO, ENTRAINMENT_HI = 65.0, 105.0     # the reported band

# NEMO variable-name candidates (first present wins).
_T_CANDS = ("tn", "thetao", "votemper", "toce", "to")
_S_CANDS = ("sn", "so", "vosaline", "soce")
_U_CANDS = ("un", "uo", "vozocrtx")
_V_CANDS = ("vn", "vo", "vomecrty")
_EN_CANDS = ("en",)
_AVTK_CANDS = ("avt_k",)
_AVMK_CANDS = ("avm_k",)
_TAUM_CANDS = ("taum", "taum_oce")
_TAUX_CANDS = ("utau_ao", "utau", "tauuo", "sozotaux")
_TAUY_CANDS = ("vtau_ao", "vtau", "tauvo", "sometauy")


def _import_reused():
    for d in (_FIDELITY_DIR, _RUN_DIR):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import equatorial_diffusivity_oracle as oracle
    import run_scm_column_twins as twins
    import run_omip_core2 as runner
    return oracle, twins, runner


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception as exc:                       # pragma: no cover
        return f"UNKNOWN ({exc})"


# ===========================================================================
# manifest -> config (no hidden choice; assert the physical constants)
# ===========================================================================
def _walk_find(o, key, path=""):
    """Yield (dotted-path, value) for every scalar leaf named ``key``."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key and not isinstance(v, (dict, list)):
                yield path + "." + k, v
            yield from _walk_find(v, key, path + "." + k)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_find(v, key, path)


def _one(manifest, key):
    vals = {v for _p, v in _walk_find(manifest, key)}
    if len(vals) == 1:
        return next(iter(vals))
    return vals                                    # empty or ambiguous


def manifest_constants(manifest_path: Path):
    """Resolved (eos, rho_0, g) from the arm's manifest — asserted, not guessed.

    The frozen-column N2/density MUST use the SAME EOS and reference density the
    arm ran, or the closure sees a different stratification and the twin number
    is wrong for a reason unrelated to the closure.
    """
    m = json.loads(Path(manifest_path).read_text())
    eos = _one(m, "eos")
    rho0 = _one(m, "rho_0")
    g = _one(m, "g")
    # eos/rho_0/g appear in several sub-configs; require a single consistent value.
    for name, val in (("eos", eos), ("rho_0", rho0), ("g", g)):
        if not isinstance(val, (str, int, float)):
            raise SystemExit(
                f"{manifest_path}: {name} is ambiguous/absent in the resolved "
                f"config ({val!r}); refusing to guess a physical constant.")
    return str(eos), float(rho0), float(g), m


def tke_config_from_manifest(manifest, runner):
    """Rebuild the arm's VerticalMixingConfig from its recorded command line."""
    cl = manifest.get("run", {}).get("command_line", manifest.get("command_line"))
    if cl is None:
        raise SystemExit("manifest has no run.command_line to parse")
    toks = shlex.split(cl) if isinstance(cl, str) else list(cl)

    def val(name):
        return toks[toks.index(name) + 1] if name in toks else None

    def has(name):
        return name in toks

    vmix = runner.build_tripole_vmix_config(
        tripole_vmix=val("--tripole-vmix") or "tke",
        iwm=has("--iwm"),
        tke_eice=(int(val("--tke-eice")) if has("--tke-eice") else None),
        tke_surface_bc=val("--tke-surface-bc"),
        tke_mxl_choice=(int(val("--tke-mxl-choice"))
                        if has("--tke-mxl-choice") else None),
        tke_n2_mode=val("--tke-n2-mode"),
        tke_n2_eos_form=val("--tke-n2-eos-form"),
        tke_prognostic=(True if has("--tke-prognostic") else None),
        tke_kappa_convention=val("--tke-kappa-convention"),
        tke_shear_production=val("--tke-shear-production"),
        tke_lc=(True if has("--tke-lc") else None),
        tke_etau=val("--tke-etau"),
    )
    return vmix, cl


def _echo_cfg(vmix, eos, rho0, g):
    t = vmix.tke
    print(f"[cfg] eos={eos!r} rho_0={rho0} g={g}")
    for f in ("prognostic", "tke_mxl_choice", "surface_bc", "n2_mode",
              "n2_eos_form", "kappa_convention", "c_k", "c_eps",
              "tke_background", "tke_surface_min", "kappaM_min", "kappaH_min",
              "eice", "prandtl_mode"):
        if hasattr(t, f):
            print(f"[cfg]   {f} = {getattr(t, f)!r}")


# ===========================================================================
# geometry + velocity centering (mirror the production model exactly)
# ===========================================================================
def full_interfaces_from_centres(z_centre):
    """Full interface ladder (nlev+1, >0) from centre depths (midpoint rule).

    interfaces[0]=0, interior at centre midpoints, bottom reflected — the SAME
    construction as ``run_scm_column_twins.dz_from_center_depths`` uses; the
    interior interfaces (nlev-1) are ``[1:-1]``.  z_center_ref is stored by the
    snapshot; nav_lev/gdept give it for the NEMO restart.
    """
    zc = np.abs(np.asarray(z_centre, dtype=np.float64))
    if zc.ndim != 1 or zc.size < 2 or not np.all(np.diff(zc) > 0):
        raise ValueError(f"centre depths must be 1-D increasing, got {zc}")
    itf = np.empty(zc.size + 1)
    itf[0] = 0.0
    itf[1:-1] = 0.5 * (zc[:-1] + zc[1:])
    itf[-1] = 2.0 * zc[-1] - itf[-2]
    return itf                                     # (nlev+1,)


def centre_uv_extra_column(u, v):
    """Center OUR C-grid staggered u/v (extra face column) to T-points.

    EXACTLY the model's own centering (ocean_model_latlon_cgrid.py:6375-6376):
    u has an extra x-column (ny, nx+1, nlev) -> average adjacent U-faces along
    x; v has an extra y-row (ny+1, nx, nlev) -> average adjacent V-faces along y.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u_cell = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    v_cell = 0.5 * (v[:-1, :, :] + v[1:, :, :])
    return u_cell, v_cell


def centre_uv_collocated(un, vn):
    """Center NEMO's COLLOCATED restart un/vn (same shape as tn) to T-points.

    NEMO stores un/vn on the same (jpj, jpi) array as tn, staggered by C-grid
    CONVENTION not by array size: un(i) is the EAST face of T(i), so T(i) sits
    between un(i-1) and un(i); vn(j) is the NORTH face of T(j).  We average the
    two faces bracketing each T-cell; the first interior column/row keeps its
    single face (the equatorial box is interior, far from these edges).
    """
    un = np.asarray(un, dtype=np.float64)
    vn = np.asarray(vn, dtype=np.float64)
    u_cell = np.array(un)
    u_cell[:, 1:, :] = 0.5 * (un[:, :-1, :] + un[:, 1:, :])
    v_cell = np.array(vn)
    v_cell[1:, :, :] = 0.5 * (vn[:-1, :, :] + vn[1:, :, :])
    return u_cell, v_cell


# ===========================================================================
# NEMO tiled restart reassembly (no repo reader exists — minimal stitcher)
# ===========================================================================
def reassemble_restart(glob_pat, varnames, twins):
    """Stitch a global field per var from the DOMAIN_position_first/last tiles.

    Standard NEMO XIOS restart metadata: each tile carries DOMAIN_size_global
    (nx, ny), DOMAIN_position_first/last (1-based global i/j of the tile's
    corners, no halos here — DOMAIN_halo_size_* == 0).  We place each tile's
    (y, x[, z]) block at [j0-1:j1, i0-1:i1].  Returns dict name->global array
    plus nav_lat/nav_lon (2-D) and nav_lev (centre depths).
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
    for v in varnames:
        out[v] = np.full((ny_g, nx_g, nlev), np.nan, dtype=np.float64)
    nav_lat = np.full((ny_g, nx_g), np.nan)
    nav_lon = np.full((ny_g, nx_g), np.nan)
    nav_lev = None
    for f in files:
        d = nc.Dataset(f)
        i0, j0 = (int(x) for x in d.getncattr("DOMAIN_position_first"))
        i1, j1 = (int(x) for x in d.getncattr("DOMAIN_position_last"))
        ys, xs = slice(j0 - 1, j1), slice(i0 - 1, i1)
        resolved = {}
        for v in varnames:
            name = twins._find_data_var(d, _var_candidates(v))
            resolved[v] = name
            a = np.asarray(d[name][:], dtype=np.float64)   # (t, z, y, x) or (t,y,x)
            a = np.squeeze(a, axis=0) if a.shape[0] == 1 else a[0]
            a = np.where(np.abs(a) > 1e10, np.nan, a)
            if a.ndim == 3:                         # (z, y, x) -> (y, x, z)
                a = np.moveaxis(a, 0, -1)
            else:                                   # (y, x) -> (y, x, 1)
                a = a[..., None]
                if out[v].shape[-1] != 1:
                    out[v] = out[v][..., :1]
            out[v][ys, xs, :] = a
        nav_lat[ys, xs] = np.asarray(d["nav_lat"][:], dtype=np.float64)
        nav_lon[ys, xs] = np.asarray(d["nav_lon"][:], dtype=np.float64)
        if nav_lev is None:
            nav_lev = np.abs(np.asarray(d["nav_lev"][:], dtype=np.float64))
        d.close()
    # NEMO ELIMINATES all-land subdomains, so uncovered cells are land (normal),
    # NOT a stitching bug — they stay NaN and drop out of the wet box selection.
    n_gap = int(np.isnan(nav_lat).sum())
    print(f"[restart] stitched {len(files)} tiles -> ({ny_g},{nx_g},{nlev}); "
          f"{n_gap} land cells eliminated by NEMO (not stitched); "
          f"vars {[(v, resolved[v]) for v in varnames]}")
    return out, nav_lat, nav_lon % 360.0, nav_lev


def _var_candidates(v):
    return {"T": _T_CANDS, "S": _S_CANDS, "U": _U_CANDS, "V": _V_CANDS,
            "en": _EN_CANDS, "avt_k": _AVTK_CANDS, "avm_k": _AVMK_CANDS}[v]


# ===========================================================================
# surface stress modulus from the SBC (taum), matched record
# ===========================================================================
def load_sbc_taum(sbc_path, rec, twins):
    """Wind-stress MODULUS field |tau| from the SBC file (taum), record ``rec``.

    NEMO's own taum is fed straight to the closure's ``taum_surface`` channel
    (the modulus form the surface BC / mxl0 anchor consume), so no tau_x/tau_y
    rotation or sign convention is introduced.  ``rec`` clamped to the last
    record (closest 5-day mean to the day-30 instant).
    """
    import netCDF4 as nc
    d = nc.Dataset(sbc_path)
    name = twins._find_data_var(d, _TAUM_CANDS)
    a = np.asarray(d[name][:], dtype=np.float64)
    d.close()
    nt = a.shape[0]
    r = min(rec, nt - 1)
    field = np.where(np.abs(a[r]) > 1e10, np.nan, a[r])
    print(f"[stress] SBC {name} record {r}/{nt - 1} (5-day mean; entrainment-band "
          f"K depends only weakly on it via the mxl0 surface anchor)")
    return field, name                              # (ny, nx)


# ===========================================================================
# ONE reduction, matching the oracle (_table): spatial median per depth,
# then median over the depth band.  Used on OURS, STORED, and NEMO alike.
# ===========================================================================
def oracle_band_reduce(K_cols, zk, lo, hi):
    """K_cols: (ncol, nk) at interface depths zk (nk,). Returns band median.

    For each interface depth: median over the finite, >0 columns (matching the
    oracle's convection-robust per-depth spatial median). Then median over the
    depths inside [lo, hi].  NEVER a mean, NEVER a ratio-of-reductions.
    """
    K = np.asarray(K_cols, dtype=np.float64)
    zk = np.abs(np.asarray(zk, dtype=np.float64))
    prof = []
    for k in range(K.shape[1]):
        col = K[:, k]
        col = col[np.isfinite(col) & (col > 0)]
        if col.size:
            prof.append((zk[k], float(np.median(col))))
    band = [p for d, p in prof if lo <= d <= hi]
    return float(np.median(band)) if band else float("nan")


# ===========================================================================
# the single closure evaluation (production prognostic call, one step)
# ===========================================================================
def closure_K(*, T, S, u_cell, v_cell, en, taum, eta, lat, z_centre, dz_ref,
              vmix_cfg, eos_name, rho0, g, dt):
    """One evaluation of OUR TKE closure -> (avm, avt) at interior interfaces.

    Mirrors the production prognostic call (k_profiles.py L926) with
    ``n_iterations=1`` and ``tke_old=en``: the closure computes K from the
    SUPPLIED en (NEMO's own, or our own stored) with a single backward-Euler
    step, so K reflects that en rather than a free-spun equilibrium.  Inputs are
    (1, ncol, nlev) cell-centre arrays; taum is (1, ncol) [N/m^2].

    Returns (avm, avt) each (ncol, nlev-1) at interior interfaces, and the
    interior-interface depths.
    """
    import jax.numpy as jnp
    from legoesm.ocean.eos import (
        compute_ocean_rho, make_eos_fn, nemo_bn2_live_ladders,
    )
    from legoesm.ocean.vertical import (
        compute_ocean_jacobian, create_z_star_from_thicknesses,
    )
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing

    dz_ref_j = jnp.asarray(dz_ref)
    z_coord = create_z_star_from_thicknesses(dz_ref_j, jnp.asarray(z_centre))
    Tj = jnp.asarray(T); Sj = jnp.asarray(S)
    uj = jnp.asarray(u_cell); vj = jnp.asarray(v_cell)
    etaj = jnp.asarray(eta)
    H = jnp.full(etaj.shape, float(z_centre[-1] + 0.5 * dz_ref[-1]))
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
    # nemo_bn2 needs the LIVE geometric depth ladders (gdept, interior gdepw).
    t_depth, w_depth = nemo_bn2_live_ladders(z_coord, etaj, H)

    out = tke_vertical_mixing(
        uj, vj, Tj, Sj, rho, dz_half,
        tke_old=jnp.asarray(en),
        tau_x_surface=None, tau_y_surface=None,
        dt=float(dt), cfg=vmix_cfg.tke,
        rho_0=float(rho0), g=float(g), n_iterations=1,
        taum_surface=jnp.asarray(taum),
        dz_ref=dz_ref_j, jacobian=J, eos_fn=eos_fn, z_interface=z_int,
        lat_deg=jnp.asarray(lat), t_depth=t_depth, w_depth=w_depth,
    )
    avm = np.asarray(out.K_M)[0]                    # (ncol, nlev-1)
    avt = np.asarray(out.K_H)[0]
    zk = np.abs(np.asarray(z_coord.z_half_ref[1:-1]))
    return avm, avt, zk


# ===========================================================================
# stress sampling helper (nearest SBC cell per column)
# ===========================================================================
def sample_field_at_columns(field2d, flat_lat, flat_lon, col_lat, col_lon,
                            twins):
    """Nearest-cell value of ``field2d`` at each (col_lat, col_lon)."""
    ncol = col_lat.size
    out = np.empty(ncol)
    fin = np.isfinite(field2d)
    for c in range(ncol):
        dd = twins.great_circle_deg(flat_lat, flat_lon,
                                    float(col_lat[c]), float(col_lon[c]))
        j, i = np.unravel_index(
            int(np.argmin(np.where(fin, dd, np.inf))), dd.shape)
        out[c] = field2d[j, i]
    return out


# ===========================================================================
# modes
# ===========================================================================
def _box_mask(lat, lon, args):
    return ((np.abs(lat) <= args.lat_halfwidth)
            & (lon >= args.lon_west) & (lon <= args.lon_east))


def run_control(args, oracle, twins, runner):
    """Instrument gate: reproduce OUR stored K from OUR snapshot column + en."""
    eos, rho0, g, manifest = manifest_constants(Path(args.manifest))
    vmix, cl = tke_config_from_manifest(manifest, runner)
    _echo_cfg(vmix, eos, rho0, g)

    z = np.load(args.snapshot)
    need = ("T", "S", "u", "v", "tke", "z_center_ref", "land_mask",
            "lat_T", "lon_T")
    miss = [k for k in need if k not in z]
    if miss:
        raise SystemExit(f"{args.snapshot}: missing {miss} (need full column "
                         "state + carried tke + z_center_ref)")
    lat = np.asarray(z["lat_T"], dtype=np.float64)
    lon = np.asarray(z["lon_T"], dtype=np.float64) % 360.0
    if np.nanmax(np.abs(lat)) <= np.pi + 1e-6:
        lat, lon = np.degrees(lat), np.degrees(lon) % 360.0
    wet = np.asarray(z["land_mask"], dtype=np.float64) > 0.5
    band = _box_mask(lat, lon, args) & wet
    ncol = int(band.sum())
    if ncol == 0:
        raise SystemExit("no wet columns in the box")

    z_centre = np.abs(np.asarray(z["z_center_ref"], dtype=np.float64))
    dz_ref = np.diff(full_interfaces_from_centres(z_centre))
    u_cell, v_cell = centre_uv_extra_column(z["u"], z["v"])     # fix codex #2
    # deterministic N2 perturbation (non-vacuity): scale T,S deviation from the
    # surface value by sqrt(f) so d(T,S)/dz -> sqrt(f)* and N2 -> f* (T/S-linear
    # thermocline); K ~ 1/sqrt(N2) via the buoyancy length -> band K ~ /sqrt(f).
    Tc = np.asarray(z["T"], dtype=np.float64)
    Sc = np.asarray(z["S"], dtype=np.float64)
    if args.perturb_n2 != 1.0:
        s = np.sqrt(args.perturb_n2)
        Tc = Tc[..., :1] + s * (Tc - Tc[..., :1])
        Sc = Sc[..., :1] + s * (Sc - Sc[..., :1])
        print(f"[control] NON-VACUITY: N2 x{args.perturb_n2} "
              f"(T,S deviation x{s:.3f}); expected band K x~{1/s:.3f} "
              "-> reproduction MUST break")

    def stk(a):
        return a[band][None, ...]
    eta = (np.asarray(z["eta"])[band][None, ...] if "eta" in z
           else np.zeros((1, ncol)))
    taum, tname = load_sbc_taum(args.nemo_sbc, args.rec, twins)
    tau_cols = sample_field_at_columns(
        taum, lat, lon, lat[band], lon[band], twins)
    print(f"[stress] control |tau| median {np.nanmedian(tau_cols):.4f} N/m^2 "
          f"(NEMO {tname}; our run is CORE-II forced too)")

    avm, avt, zk = closure_K(
        T=stk(Tc), S=stk(Sc), u_cell=stk(u_cell), v_cell=stk(v_cell),
        en=stk(np.asarray(z["tke"])), taum=tau_cols[None, :], eta=eta,
        lat=lat[band][None, :], z_centre=z_centre, dz_ref=dz_ref, vmix_cfg=vmix,
        eos_name=eos, rho0=rho0, g=g, dt=args.dt)

    # stored K (full: closure + additive IWM) via the oracle loader.
    oavm, oavt, _b, olat, olon, ozk = oracle._load_ours(Path(args.snapshot))
    oband = _box_mask(olat, olon, args)
    s_avm = oracle_band_reduce(oavm[oband], ozk, ENTRAINMENT_LO, ENTRAINMENT_HI)
    s_avt = oracle_band_reduce(oavt[oband], ozk, ENTRAINMENT_LO, ENTRAINMENT_HI)
    c_avm = oracle_band_reduce(avm, zk, ENTRAINMENT_LO, ENTRAINMENT_HI)
    c_avt = oracle_band_reduce(avt, zk, ENTRAINMENT_LO, ENTRAINMENT_HI)

    print("\n[control] entrainment band 65-105 m (spatial median per depth, "
          "then median over band):")
    print(f"[control]   our closure   avm={c_avm:.4e}  avt={c_avt:.4e}")
    print(f"[control]   our stored    avm={s_avm:.4e}  avt={s_avt:.4e}  "
          "(FULL = closure + additive IWM)")
    r_m = c_avm / s_avm if s_avm else float("nan")
    r_t = c_avt / s_avt if s_avt else float("nan")
    print(f"[control]   ratio closure/stored  avm x{r_m:.3f}  avt x{r_t:.3f}  "
          f"(shortfall = additive-IWM fraction)")

    lo, hi = 1.0 / args.control_tol, args.control_tol
    ok = (lo <= r_m <= hi) and (lo <= r_t <= hi)
    if args.perturb_n2 != 1.0:
        print(f"[control]   NON-VACUITY: reproduction broke = {not ok} "
              "(must be True)")
        return 0 if not ok else 3
    print(f"[control]   CONFIRMED: closure reproduces stored K within x"
          f"{args.control_tol} = {ok}")
    return 0 if ok else 4


def run_nemo(args, oracle, twins, runner):
    """The test: OUR closure on NEMO's instant column+en vs NEMO avm_k/avt_k."""
    eos, rho0, g, manifest = manifest_constants(Path(args.manifest))
    vmix, cl = tke_config_from_manifest(manifest, runner)
    _echo_cfg(vmix, eos, rho0, g)

    fields, nav_lat, nav_lon, nav_lev = reassemble_restart(
        args.restart_glob, ("T", "S", "U", "V", "en", "avt_k", "avm_k"), twins)
    wet = np.isfinite(fields["T"][..., 0])
    band = _box_mask(nav_lat, nav_lon, args) & wet
    ncol = int(band.sum())
    if ncol == 0:
        raise SystemExit("no wet NEMO columns in the box")
    print(f"[nemo] {ncol} equatorial columns; nav_lev {nav_lev[0]:.2f}.."
          f"{nav_lev[-1]:.0f} m")

    z_centre = nav_lev
    dz_ref = np.diff(full_interfaces_from_centres(z_centre))
    u_cell, v_cell = centre_uv_collocated(fields["U"], fields["V"])

    def stk(a):
        return a[band][None, ...]
    taum, tname = load_sbc_taum(args.nemo_sbc, args.rec, twins)
    tau_cols = sample_field_at_columns(
        taum, nav_lat, nav_lon, nav_lat[band], nav_lon[band], twins)
    print(f"[stress] NEMO |tau| median {np.nanmedian(tau_cols):.4f} N/m^2")

    avm, avt, zk = closure_K(
        T=stk(fields["T"]), S=stk(fields["S"]),
        u_cell=stk(u_cell), v_cell=stk(v_cell),
        en=stk(_en_to_interior(fields["en"], band)),
        taum=tau_cols[None, :], eta=np.zeros((1, ncol)),
        lat=nav_lat[band][None, :], z_centre=z_centre, dz_ref=dz_ref,
        vmix_cfg=vmix, eos_name=eos, rho0=rho0, g=g, dt=args.dt)

    # NEMO's OWN closure diffusivities avt_k/avm_k (w-points), banded on the
    # geometric interface depths derived from nav_lev.
    w_depth = full_interfaces_from_centres(z_centre)[1:-1]      # interior (nlev-1)
    n_avmk = _avk_to_interior(fields["avm_k"], band)
    n_avtk = _avk_to_interior(fields["avt_k"], band)

    for name, lo, hi in (("SURFACE 5-65 m", SURFACE_LO, SURFACE_HI),
                         ("ENTRAINMENT 65-105 m", ENTRAINMENT_LO,
                          ENTRAINMENT_HI)):
        o_m = oracle_band_reduce(avm, zk, lo, hi)
        o_t = oracle_band_reduce(avt, zk, lo, hi)
        n_m = oracle_band_reduce(n_avmk, w_depth, lo, hi)
        n_t = oracle_band_reduce(n_avtk, w_depth, lo, hi)
        print(f"\n[nemo] {name}:")
        print(f"[nemo]   ours (closure)  avm={o_m:.4e}  avt={o_t:.4e}")
        print(f"[nemo]   NEMO avm_k/avt_k avm={n_m:.4e}  avt={n_t:.4e}")
        rm = o_m / n_m if n_m else float("nan")
        rt = o_t / n_t if n_t else float("nan")
        print(f"[nemo]   ratio ours/NEMO  avm x{rm:.3f}  avt x{rt:.3f}  "
              f"(median-local Pr ratio x{(rm / rt) if rt else float('nan'):.3f})")
    print("\n[nemo] instant-vs-instant, NEMO's OWN en, closure-vs-closure "
          "(avm_k/avt_k); no free-spin, no IWM confound. Numbers only.")
    return 0


def _en_to_interior(en3d, band):
    """NEMO en (w-points, nlev) at box columns, dropped to interior interfaces.

    en is at w-levels k=0..nlev-1 (k=0 = surface).  Our closure carries en at
    the nlev-1 INTERIOR interfaces (surface dropped), so take k=1..nlev-1.
    """
    a = en3d[band]                                  # (ncol, nlev)
    return a[:, 1:]                                 # (ncol, nlev-1)


def _avk_to_interior(avk3d, band):
    """NEMO avm_k/avt_k (w-points, nlev) at box columns -> interior (nlev-1)."""
    a = avk3d[band]
    return a[:, 1:]


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("control", "nemo"))
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--nemo-sbc", required=True, type=Path,
                   help="NEMO *_SBC.nc (taum wind-stress modulus)")
    p.add_argument("--rec", type=int, default=5,
                   help="SBC record (clamped to last; 5-day mean near day 30)")
    p.add_argument("--lat-halfwidth", type=float, default=DEFAULT_LAT_HALFWIDTH)
    p.add_argument("--lon-west", type=float, default=DEFAULT_LON_WEST)
    p.add_argument("--lon-east", type=float, default=DEFAULT_LON_EAST)
    p.add_argument("--dt", type=float, default=3600.0,
                   help="one closure BE step [s] (NEMO restart dt=3600)")
    # control
    p.add_argument("--snapshot", type=Path, default=None)
    p.add_argument("--perturb-n2", type=float, default=1.0,
                   help="non-vacuity: scale N2 by this (T,S deviation x sqrt); "
                        "expected band K x 1/sqrt -> control must break")
    p.add_argument("--control-tol", type=float, default=1.5,
                   help="entrainment-band ratio tol (x); IWM shortfall allowed "
                        "within, plumbing errors are order-of-magnitude")
    # nemo
    p.add_argument("--restart-glob", type=str, default=None,
                   help="glob for the ORCA1_*_restart_oce_*.nc tiles")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    oracle, twins, runner = _import_reused()
    print(f"[prov] git {_git_sha()}")
    print(f"[prov] mode={args.mode} box |lat|<={args.lat_halfwidth} "
          f"{args.lon_west:.0f}-{args.lon_east:.0f}E")
    if args.mode == "control":
        if args.snapshot is None:
            raise SystemExit("control mode needs --snapshot")
        print(f"[prov] snapshot {args.snapshot}")
        return run_control(args, oracle, twins, runner)
    if args.restart_glob is None:
        raise SystemExit("nemo mode needs --restart-glob")
    print(f"[prov] restart {args.restart_glob}")
    return run_nemo(args, oracle, twins, runner)


if __name__ == "__main__":
    raise SystemExit(main())
