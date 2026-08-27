#!/usr/bin/env python
"""Run OUR TKE closure on a FROZEN equatorial column and compare avm/avt.

THE DECISIVE TEST this campaign did not yet have.  We proved our TKE closure is
INTERNALLY self-consistent (implied mixing length matches buoyancy length), but
that never proved it equals NEMO's closure when BOTH see the SAME frozen state.
Here we take a frozen background (T, S, u, v, surface wind stress) at the
equatorial cold tongue, spin OUR prognostic TKE energy ``en`` to statistical
equilibrium under it, and read the resulting avm (K_M) / avt (K_H).  MATCH to
NEMO's published grid_W avm/avt (within a few %) => the closure is faithful and
the ~8.6x coupled-run gap is a STATE difference; MISMATCH on a frozen state =>
a closure CODE defect.

This probe PRINTS NUMBERS AND RATIOS ONLY.  It renders no verdict — the word
"faithful" appears nowhere in its output.  Interpretation is the reader's.

WHY A SEPARATE HARNESS, AND WHAT IT REUSES (RULE 4 — search before build)
------------------------------------------------------------------------
* NEMO grid_W avm/avt/bn2 box reduction: reuses ``equatorial_diffusivity_oracle``
  (``_load``, ``_table``, ``_load_ours``, ``CONVECTIVE_AVT`` — the SAME box,
  window, and per-column-median reduction, so both sides are the same quantity).
* NEMO column extraction / nearest-wet / var-name detection: reuses
  ``run_scm_column_twins`` (``_find_data_var``, ``nearest_wet_column``).
* The TKE closure itself: the model's OWN ``tke_vertical_mixing`` (never
  re-implemented), called argument-for-argument as the production prognostic
  step calls it (``vertical_mixing/k_profiles.py`` ~L926).
* The production TKE config: ``run_omip_core2.build_tripole_vmix_config``,
  parameterised FROM the snapshot arm's ``run_manifest.json`` (no hidden config
  choice — the resolved TKEConfig fields are echoed).
* rho / N2 / z-coord / Jacobian: the model's own ``eos`` and ``vertical``.
  The SCM ``OceanColumnModel`` cannot host this closure (it rejects tke/catke),
  so the bare column-local ``tke_vertical_mixing`` is the reuse seam.

THE ONE KNOWN GAP, MEASURED NOT ASSUMED (background is molecular under --iwm)
----------------------------------------------------------------------------
The production arm ran ``--iwm`` (internal-wave mixing on) and ``--tke-prognostic``.
The stored ``K_M_diag``/``K_H_diag`` are the FULL solve's coefficients:
closure PLUS the additive internal-wave-mixing field that
``_apply_implicit_vertical_mixing`` adds AFTER the closure.  ``tke_vertical_mixing``
returns the closure part only.  The namzdf backgrounds live INSIDE the closure
(kappaM_min/kappaH_min = molecular under --iwm), so the residual between our
spun closure K and the stored K is the ADDITIVE IWM K field.  The self-recovery
control MEASURES that residual in the entrainment band and GATES on it: if it is
larger than the tolerance, this bare-column instrument is NOT a faithful proxy
for the published quantity there and the run aborts (escalate to the full-model
``diagnose_vertical_K`` path).  It is never silently ignored.

Usage (compute node / sbatch ONLY — never the login node):
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -u \
        scripts/validate/ocean_fidelity/frozen_column_tke_twin.py \
        --mode control --snapshot <snap.npz> --manifest <run_manifest.json>
    ...   --mode nemo --nemo-gridt ... --nemo-gridu ... --nemo-gridv ... \
          --nemo-gridw ... --rec 5 --manifest <run_manifest.json>
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_FIDELITY_DIR = _HERE.parent                       # scripts/validate/ocean_fidelity
_RUN_DIR = _REPO_ROOT / "scripts" / "run"

# Equatorial cold-tongue box (oracle defaults) + a single 140W column option.
DEFAULT_LAT_HALFWIDTH = 2.0
DEFAULT_LON_WEST = 200.0
DEFAULT_LON_EAST = 260.0
NINO140W = (0.0, 220.0)                            # 0N, 140W = 220E

# Entrainment band where the cooling budget lives and where EVD is OFF (stably
# stratified thermocline) — the band the control gates on and the test reports.
ENTRAINMENT_LO, ENTRAINMENT_HI = 65.0, 105.0
SURFACE_LO, SURFACE_HI = 5.0, 65.0

# NEMO variable-name candidates (first present wins; verified names first).
_T_CANDS = ("thetao", "votemper", "toce", "to")
_S_CANDS = ("so", "vosaline", "soce")
_U_CANDS = ("uo", "vozocrtx")
_V_CANDS = ("vo", "vomecrty")
_TAUX_CANDS = ("tauuo", "utau", "sozotaux")
_TAUY_CANDS = ("tauvo", "vtau", "sometauy")


# ===========================================================================
# reuse imports (oracle + SCM-twin helpers + model + runner config builder)
# ===========================================================================
def _import_reused():
    """Import the sibling scripts' reusable helpers (white-box script reuse).

    ``scripts/validate/ocean_fidelity`` and ``scripts/run`` are on PYTHONPATH
    in the sbatch env; insert them so the harness also runs standalone.
    """
    for d in (_FIDELITY_DIR, _RUN_DIR):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import equatorial_diffusivity_oracle as oracle
    import run_scm_column_twins as twins
    import run_omip_core2 as runner
    return oracle, twins, runner


# ===========================================================================
# provenance
# ===========================================================================
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception as exc:                       # pragma: no cover
        return f"UNKNOWN ({exc})"


# ===========================================================================
# TKE config from the snapshot arm's manifest (no hidden config choice)
# ===========================================================================
def tke_config_from_manifest(manifest_path: Path, runner):
    """Rebuild the arm's VerticalMixingConfig from ``run_manifest.json``.

    Parses the recorded command line for the ``--tke-*`` / ``--iwm`` /
    ``--tripole-vmix`` flags and feeds them to the SAME
    ``build_tripole_vmix_config`` the production run used, so the closure
    config is the run's config, not a guessed default.  The resolved TKEConfig
    fields are returned for echoing (provenance).
    """
    m = json.loads(Path(manifest_path).read_text())
    cl = m.get("run", {}).get("command_line", m.get("command_line"))
    if cl is None:
        raise SystemExit(f"{manifest_path}: no run.command_line to parse")
    toks = shlex.split(cl) if isinstance(cl, str) else list(cl)

    def _flag_val(name, default=None):
        if name in toks:
            i = toks.index(name)
            return toks[i + 1] if i + 1 < len(toks) else True
        return default

    def _present(name):
        return name in toks

    tripole_vmix = _flag_val("--tripole-vmix", "tke")
    kwargs = dict(
        tripole_vmix=tripole_vmix,
        iwm=_present("--iwm"),
        tke_eice=(int(_flag_val("--tke-eice")) if _present("--tke-eice")
                  else None),
        tke_surface_bc=_flag_val("--tke-surface-bc"),
        tke_mxl_choice=(int(_flag_val("--tke-mxl-choice"))
                        if _present("--tke-mxl-choice") else None),
        tke_n2_mode=_flag_val("--tke-n2-mode"),
        tke_n2_eos_form=_flag_val("--tke-n2-eos-form"),
        tke_prognostic=(True if _present("--tke-prognostic") else None),
        tke_kappa_convention=_flag_val("--tke-kappa-convention"),
        tke_shear_production=_flag_val("--tke-shear-production"),
        tke_lc=(True if _present("--tke-lc") else None),
        tke_etau=_flag_val("--tke-etau"),
    )
    vmix = runner.build_tripole_vmix_config(**kwargs)
    return vmix, kwargs, cl


def _echo_tke_cfg(vmix):
    t = vmix.tke
    fields = ("prognostic", "tke_mxl_choice", "surface_bc", "n2_mode",
              "kappa_convention", "c_k", "c_eps", "tke_background",
              "tke_surface_min", "kappaM_min", "kappaH_min", "eice",
              "prandtl_mode")
    print("[cfg] resolved TKEConfig (from manifest):")
    for f in fields:
        if hasattr(t, f):
            print(f"[cfg]   {f} = {getattr(t, f)!r}")


# ===========================================================================
# column-set construction  (returns everything tke_vertical_mixing needs)
# ===========================================================================
class ColumnSet:
    """A stack of frozen equatorial columns and their static geometry.

    Fields are (1, ncol, nlev) cell-centre arrays (T, S, u, v, rho), (1, ncol)
    surface stress, the reference z-coordinate and per-column Jacobian, plus
    the box band mask and level depths for the oracle-style reduction.
    """
    def __init__(self, *, T, S, u, v, eta, dz_ref, t_depth_ref,
                 z_interface_interior, tau_x, tau_y, lat, lon,
                 stored_KM=None, stored_KH=None, source=""):
        self.T = T; self.S = S; self.u = u; self.v = v; self.eta = eta
        self.dz_ref = dz_ref; self.t_depth_ref = t_depth_ref
        self.z_interface_interior = z_interface_interior      # (nlev-1,) >0 [m]
        self.tau_x = tau_x; self.tau_y = tau_y
        self.lat = lat; self.lon = lon
        self.stored_KM = stored_KM; self.stored_KH = stored_KH
        self.source = source


def _centre_dz_and_depth(z_interface_ref):
    """dz_ref (nlev), t_depth_ref (nlev, centre depths >0) from interface depths.

    ``z_interface_ref`` here is the FULL interface ladder (nlev+1, incl. 0 and
    the deepest), positive-down.  The closure's interior interfaces are the
    (nlev-1) between cell centres.
    """
    zi = np.abs(np.asarray(z_interface_ref, dtype=np.float64))
    zi = np.sort(zi)
    dz = np.diff(zi)                                # (nlev,)
    tc = 0.5 * (zi[:-1] + zi[1:])                  # (nlev,) centre depths
    return dz, tc, zi[1:-1]                         # interior interfaces (nlev-1)


def build_columns_from_snapshot(snapshot_path, box_fn, oracle):
    """CONTROL: our own frozen columns + the stored K_M_diag/K_H_diag.

    Reuses ``oracle._load_ours`` to pull the SAME stored diffusivities the
    oracle reads (so the control's reference is exactly the stored quantity),
    and reads T/S/u/v/eta/z from the snapshot arrays directly.  A snapshot that
    lacks any of T/S/u/v/tke is a hard error (the frozen spin is impossible
    without them) — never a defaulted column.
    """
    z = np.load(snapshot_path)
    need = ("T", "S", "u", "v", "z_interface_ref", "land_mask",
            "lat_T", "lon_T")
    missing = [k for k in need if k not in z]
    if missing:
        raise SystemExit(
            f"{snapshot_path}: missing {missing}. The frozen-column spin needs "
            "the full column state (T,S,u,v) AND the interface depths; a "
            "snapshot without them cannot be used. Re-run the arm with "
            "--kprofile-snapshots (which stores z_interface_ref) and confirm "
            "T/S/u/v are present.")
    # Stored diffusivities via the oracle loader (same box reduction later).
    oavm, oavt, _bn2, olat, olon, ozk = oracle._load_ours(Path(snapshot_path))

    T = np.asarray(z["T"], dtype=np.float64)          # (ny, nx, nlev)
    S = np.asarray(z["S"], dtype=np.float64)
    u = np.asarray(z["u"], dtype=np.float64)
    v = np.asarray(z["v"], dtype=np.float64)
    eta = (np.asarray(z["eta"], dtype=np.float64) if "eta" in z
           else np.zeros(T.shape[:-1], dtype=np.float64))
    lat = np.asarray(z["lat_T"], dtype=np.float64)
    lon = np.asarray(z["lon_T"], dtype=np.float64) % 360.0
    if np.nanmax(np.abs(lat)) <= np.pi + 1e-6:
        lat, lon = np.degrees(lat), np.degrees(lon) % 360.0
    dz_ref, t_depth_ref, z_int_int = _centre_dz_and_depth(z["z_interface_ref"])
    wet = np.asarray(z["land_mask"], dtype=np.float64) > 0.5

    band = box_fn(lat, lon) & wet
    ncol = int(band.sum())
    if ncol == 0:
        raise SystemExit(f"{snapshot_path}: no wet columns in the box")

    def _stack(a):                                  # (ncol, nlev) -> (1, ncol, nlev)
        return a[band][None, ...]
    tau_x, tau_y = _stress_placeholder(ncol)
    return ColumnSet(
        T=_stack(T), S=_stack(S), u=_stack(u), v=_stack(v),
        eta=eta[band][None, ...], dz_ref=dz_ref, t_depth_ref=t_depth_ref,
        z_interface_interior=z_int_int, tau_x=tau_x, tau_y=tau_y,
        lat=lat[band][None, ...], lon=lon[band][None, ...],
        stored_KM=oavm[band][None, ...], stored_KH=oavt[band][None, ...],
        source=str(snapshot_path),
    ), band, (olat, olon, ozk)


def _stress_placeholder(ncol):
    # Surface stress is supplied EXPLICITLY (never defaulted to a physical
    # value): the caller overwrites these before the spin. Left as NaN so a
    # missing wiring FAILS the finite-check instead of silently running tau=0.
    nan = np.full((1, ncol), np.nan, dtype=np.float64)
    return nan.copy(), nan.copy()


def load_nemo_stress_for_columns(gridu, gridv, rec, lat_box, lon_box,
                                 twins):
    """Surface wind stress (tauuo/tauvo) at the box columns, NEMO record ``rec``.

    Both our OMIP run and NEMO are CORE-II-forced, so NEMO's own surface stress
    at the matched day is the stress our frozen column should see — sourced here
    from grid_U/grid_V, never defaulted.  Nearest-cell to each requested column
    (great-circle), matching the oracle box's cells.
    """
    import xarray as xr
    out = {}
    for path, cands, key in ((gridu, _TAUX_CANDS, "tx"),
                             (gridv, _TAUY_CANDS, "ty")):
        ds = xr.open_dataset(path, decode_times=False)
        name = twins._find_data_var(ds, cands)
        a = np.asarray(ds[name].values, dtype=np.float64)
        if a.ndim == 3:
            a = a[rec]
        a = np.where(np.abs(a) > 1e10, np.nan, a)
        nav_lat = np.asarray(ds["nav_lat"].values, dtype=np.float64)
        nav_lon = np.asarray(ds["nav_lon"].values, dtype=np.float64) % 360.0
        out[key] = (a, nav_lat, nav_lon, name)
        ds.close()
    return out


# ===========================================================================
# the frozen TKE spin  (model's own closure, prognostic en carried by us)
# ===========================================================================
def spin_tke(cols, vmix_cfg, runner, *, dt, max_iter, tol, band_lo, band_hi,
             report_every=25):
    """Spin prognostic ``en`` under the frozen background until box K converges.

    Mirrors the production prognostic call (vertical_mixing/k_profiles.py L926):
    ONE backward-Euler TKE step per iteration, feeding ``tke_new`` back as
    ``tke_old``, holding T/S/u/v (and hence rho, shear, N2) FROZEN.  The
    background is never advanced, so this is a pure en-relaxation to the
    quasi-steady equilibrium of the frozen state — the operation the science
    question specifies.  Convergence: the ENTRAINMENT-band per-column-median
    of both avm and avt changes by < ``tol`` (relative) between iterations.

    Returns (avm, avt, tke_final, n_iter, converged) with avm/avt shape
    (1, ncol, nlev-1) at interior interfaces.
    """
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_ocean_rho, make_eos_fn
    from legoesm.ocean.vertical import (
        compute_ocean_jacobian, create_z_star_from_thicknesses,
    )
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing

    tke_cfg = vmix_cfg.tke
    if not bool(getattr(tke_cfg, "prognostic", False)):
        raise SystemExit(
            "manifest TKEConfig has prognostic=False (Mode-B diagnostic). The "
            "frozen-en SPIN only means something for a prognostic closure; this "
            "arm carries no en. Re-point at a --tke-prognostic snapshot arm, or "
            "the single quasi-steady evaluation IS the answer (n_iterations=3).")

    # --- static geometry: z-coord, Jacobian, rho, tau ---------------------
    z_coord = create_z_star_from_thicknesses(
        jnp.asarray(cols.dz_ref), jnp.asarray(cols.t_depth_ref))
    T = jnp.asarray(cols.T); S = jnp.asarray(cols.S)
    u = jnp.asarray(cols.u); v = jnp.asarray(cols.v)
    eta = jnp.asarray(cols.eta)
    # H_bathy from the deepest interface (full-depth open-ocean columns in the
    # equatorial box; the 5-105 m band is far above the seafloor everywhere).
    H = jnp.full(eta.shape, float(cols.t_depth_ref[-1]
                                  + 0.5 * cols.dz_ref[-1]))
    J = compute_ocean_jacobian(eta, H, z_coord, 1.0)

    class _F:                                       # minimal Field shim (.data)
        __slots__ = ("data",)
        def __init__(self, d): self.data = d

    class _St:                                      # duck state for eos
        pass
    st = _St()
    st.T = _F(T); st.S = _F(S); st.eta = _F(eta)
    eos_fn = make_eos_fn(eos="nemo_seos")           # ORCA1 S-EOS family; see note
    rho = compute_ocean_rho(st, z_coord, J, eos_fn)

    dz_half = (jnp.asarray(z_coord.dz_half_ref) * J[..., None])
    z_int = jnp.asarray(z_coord.z_half_ref[1:-1])   # interior interface heights
    tau_x = jnp.asarray(cols.tau_x); tau_y = jnp.asarray(cols.tau_y)
    if not bool(np.all(np.isfinite(np.asarray(tau_x)))
                and np.all(np.isfinite(np.asarray(tau_y)))):
        raise SystemExit(
            "surface stress tau_x/tau_y is not finite — it must be wired "
            "explicitly (from NEMO tauuo/tauvo) before the spin; a defaulted "
            "physical input is a defect (no silent tau=0).")

    zk = np.abs(np.asarray(cols.z_interface_interior, dtype=np.float64))
    band_k = (zk >= band_lo) & (zk <= band_hi)
    if not band_k.any():
        raise SystemExit(f"no interior interfaces in {band_lo}-{band_hi} m")

    @jax.jit
    def _one(tke_old):
        # rho_0 / g omitted -> tke_vertical_mixing defaults (constants.rho_ocean,
        # constants.g), matching the production call's constants_config values.
        out = tke_vertical_mixing(
            u, v, T, S, rho, dz_half,
            tke_old=tke_old,
            tau_x_surface=tau_x, tau_y_surface=tau_y,
            dt=float(dt), cfg=tke_cfg,
            n_iterations=1,
            dz_ref=jnp.asarray(z_coord.dz_ref), jacobian=J, eos_fn=eos_fn,
            z_interface=z_int, lat_deg=jnp.asarray(cols.lat),
        )
        return out.K_M, out.K_H, out.tke_new

    def _band_median(K):
        a = np.asarray(K)[0][:, band_k]             # (ncol, nband)
        a = np.where(a > 0, a, np.nan)
        return float(np.nanmedian(a))

    tke_old = jnp.full(dz_half.shape[:-1] + (dz_half.shape[-1],),
                       float(tke_cfg.tke_background), dtype=T.dtype)
    prev_m = prev_t = None
    converged = False
    for it in range(1, int(max_iter) + 1):
        KM, KH, tke_old = _one(tke_old)
        m, t = _band_median(KM), _band_median(KH)
        if prev_m is not None:
            dm = abs(m - prev_m) / max(prev_m, 1e-30)
            dt_ = abs(t - prev_t) / max(prev_t, 1e-30)
            if it % report_every == 0 or it == 1:
                print(f"[spin] it={it:4d}  avm~{m:.3e} (d={dm:.2e})  "
                      f"avt~{t:.3e} (d={dt_:.2e})", flush=True)
            if dm < tol and dt_ < tol:
                converged = True
                print(f"[spin] CONVERGED at it={it}: d(avm)={dm:.2e} "
                      f"d(avt)={dt_:.2e} < tol={tol:.0e}", flush=True)
                break
        prev_m, prev_t = m, t
    if not converged:
        print(f"[spin] NOT converged in {max_iter} iters "
              f"(last avm~{prev_m:.3e} avt~{prev_t:.3e}) — report as PLAUSIBLE",
              flush=True)
    return np.asarray(KM), np.asarray(KH), np.asarray(tke_old), it, converged


# ===========================================================================
# oracle-style box reduction of an interior-interface K field
# ===========================================================================
def band_median_profile(K, zk, lo, hi):
    """Per-column median over the band, on interior interfaces (oracle rule).

    K: (1, ncol, nlev-1); zk: (nlev-1,) interface depths [m].  Returns the
    band-median of the per-column medians (medians throughout, matching the
    oracle's convection-robust reduction).
    """
    a = np.asarray(K)[0]
    a = np.where(a > 0, a, np.nan)
    band_k = (zk >= lo) & (zk <= hi)
    if not band_k.any():
        return float("nan")
    per_col = np.nanmedian(a[:, band_k], axis=1)    # (ncol,)
    return float(np.nanmedian(per_col))


# ===========================================================================
# modes
# ===========================================================================
def run_control(args, oracle, twins, runner):
    """Self-recovery gate: reproduce the snapshot's stored K in the band.

    Feeds OUR OWN frozen column into OUR closure, spins en, and compares the
    spun avm/avt to the snapshot's stored K_M_diag/K_H_diag in the entrainment
    band.  A defect in this harness (wrong config, wrong geometry, wrong
    stress) shows up as a band mismatch here — so the NEMO number is only
    trusted after this passes.  The RESIDUAL between spun-closure K and stored-
    full K is the additive IWM contribution (the known gap); it is printed and
    gated.  ``--perturb-shear`` scales u,v to prove the gate is non-vacuous
    (a 2x shear must break the reproduction).
    """
    def box_fn(lat, lon):
        return ((np.abs(lat) <= args.lat_halfwidth)
                & (lon >= args.lon_west) & (lon <= args.lon_east))

    cols, band, _ours_axes = build_columns_from_snapshot(
        Path(args.snapshot), box_fn, oracle)
    # Surface stress: NEMO tauuo/tauvo at the matched record (both runs CORE-II
    # forced — stated, not hidden). Required for the nemo_dirichlet surface BC.
    _wire_stress(cols, args, twins)
    if args.perturb_shear != 1.0:
        print(f"[control] NON-VACUITY: scaling u,v by {args.perturb_shear} — "
              "the reproduction MUST fail (proves the gate can fail)")
        cols.u = cols.u * args.perturb_shear
        cols.v = cols.v * args.perturb_shear

    vmix, kwargs, cl = tke_config_from_manifest(Path(args.manifest), runner)
    _echo_tke_cfg(vmix)
    zk = np.abs(np.asarray(cols.z_interface_interior))

    avm, avt, _tke, n_it, conv = spin_tke(
        cols, vmix, runner, dt=args.dt, max_iter=args.max_iter, tol=args.tol,
        band_lo=ENTRAINMENT_LO, band_hi=ENTRAINMENT_HI)

    spun_avm = band_median_profile(avm, zk, ENTRAINMENT_LO, ENTRAINMENT_HI)
    spun_avt = band_median_profile(avt, zk, ENTRAINMENT_LO, ENTRAINMENT_HI)
    # Stored K is on the oracle's interior-interface depths (ozk); reduce it in
    # the SAME band with the SAME per-column-median.
    stored_avm = band_median_profile(cols.stored_KM, _ours_axes[2],
                                     ENTRAINMENT_LO, ENTRAINMENT_HI)
    stored_avt = band_median_profile(cols.stored_KH, _ours_axes[2],
                                     ENTRAINMENT_LO, ENTRAINMENT_HI)

    print("\n[control] ENTRAINMENT band 65-105 m, per-column median:")
    print(f"[control]   spun closure   avm={spun_avm:.4e}  avt={spun_avt:.4e}")
    print(f"[control]   stored (full)  avm={stored_avm:.4e}  avt={stored_avt:.4e}")
    r_m = spun_avm / stored_avm if stored_avm else float("nan")
    r_t = spun_avt / stored_avt if stored_avt else float("nan")
    print(f"[control]   ratio spun/stored  avm x{r_m:.3f}  avt x{r_t:.3f}")
    print(f"[control]   residual (stored-spun) = additive IWM+model K: "
          f"avm {stored_avm - spun_avm:+.3e}  avt {stored_avt - spun_avt:+.3e}")

    ok_m = (1.0 / args.control_tol) <= r_m <= args.control_tol
    ok_t = (1.0 / args.control_tol) <= r_t <= args.control_tol
    gate_pass = bool(ok_m and ok_t and conv) and (args.perturb_shear == 1.0)
    print(f"[control]   n_iter={n_it} converged={conv} "
          f"control_tol=x{args.control_tol}")
    print(f"[control]   CONFIRMED: instrument reproduces stored K in band = "
          f"{gate_pass}")
    if args.perturb_shear != 1.0:
        # Non-vacuity run: PASS means the gate FAILED to reproduce (as it must).
        broke = not (ok_m and ok_t)
        print(f"[control]   NON-VACUITY result: reproduction broke = {broke} "
              "(must be True)")
        return 0 if broke else 3
    return 0 if gate_pass else 4


def run_nemo(args, oracle, twins, runner):
    """The test: OUR closure on NEMO's frozen equatorial column vs NEMO grid_W.

    Builds the frozen column from NEMO grid_T (T/S) + grid_U/grid_V (u/v +
    surface stress) at the matched record, spins en, and prints the box-median
    avm/avt beside NEMO's published grid_W avm/avt (oracle box reduction) with
    the ratio.  All four NEMO grids at the SAME record/day.
    """
    def box_fn(lat, lon):
        return ((np.abs(lat) <= args.lat_halfwidth)
                & (lon >= args.lon_west) & (lon <= args.lon_east))

    cols, band = build_columns_from_nemo(args, box_fn, twins)
    _wire_stress(cols, args, twins)
    vmix, kwargs, cl = tke_config_from_manifest(Path(args.manifest), runner)
    _echo_tke_cfg(vmix)
    zk = np.abs(np.asarray(cols.z_interface_interior))

    avm, avt, _tke, n_it, conv = spin_tke(
        cols, vmix, runner, dt=args.dt, max_iter=args.max_iter, tol=args.tol,
        band_lo=ENTRAINMENT_LO, band_hi=ENTRAINMENT_HI)

    # NEMO grid_W published avm/avt via the oracle's own loader + box.
    n_avm, n_avt, n_bn2, nlat, nlon, nz = oracle._load(
        Path(args.nemo_gridw), args.rec, args.lat_halfwidth)
    nband = box_fn(nlat, nlon)

    def _nemo_band(field):
        band_k = (nz >= ENTRAINMENT_LO) & (nz <= ENTRAINMENT_HI)
        per = []
        for k in np.where(band_k)[0]:
            a = field[..., k][nband]
            a = a[np.isfinite(a) & (a > 0)]
            if a.size:
                per.append(np.median(a))
        return float(np.median(per)) if per else float("nan")

    for band_name, lo, hi in (("SURFACE 5-65 m", SURFACE_LO, SURFACE_HI),
                              ("ENTRAINMENT 65-105 m",
                               ENTRAINMENT_LO, ENTRAINMENT_HI)):
        ours_m = band_median_profile(avm, zk, lo, hi)
        ours_t = band_median_profile(avt, zk, lo, hi)
        bk = (nz >= lo) & (nz <= hi)
        nemo_m = _band_reduce(n_avm, nband, nz, lo, hi)
        nemo_t = _band_reduce(n_avt, nband, nz, lo, hi)
        rm = ours_m / nemo_m if nemo_m else float("nan")
        rt = ours_t / nemo_t if nemo_t else float("nan")
        print(f"\n[nemo] {band_name}:")
        print(f"[nemo]   ours (frozen spin)  avm={ours_m:.4e}  avt={ours_t:.4e}")
        print(f"[nemo]   NEMO grid_W          avm={nemo_m:.4e}  avt={nemo_t:.4e}")
        print(f"[nemo]   ratio ours/NEMO      avm x{rm:.3f}  avt x{rt:.3f}  "
              f"Pr ours/NEMO x{(rm/rt) if rt else float('nan'):.3f}")
    print(f"\n[nemo]   n_iter={n_it} converged={conv}  "
          "(PLAUSIBLE if not converged; numbers above are ratios, no verdict)")
    return 0


def _band_reduce(field, nband, nz, lo, hi):
    band_k = (nz >= lo) & (nz <= hi)
    per = []
    for k in np.where(band_k)[0]:
        a = field[..., k][nband]
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            per.append(np.median(a))
    return float(np.median(per)) if per else float("nan")


def build_columns_from_nemo(args, box_fn, twins):
    """Frozen columns from NEMO grid_T/U/V at the matched record.

    T/S from grid_T (thetao/so), u/v from grid_U/grid_V (uo/vo, taken at the
    C-grid face nearest each T column — a half-cell horizontal offset that is
    second-order for the VERTICAL shear du/dz the closure reads; flagged).  dz
    from grid_T e3t when present, else deptht bounds.  All at record ``rec``.
    """
    import xarray as xr

    def _grab3d(path, cands):
        ds = xr.open_dataset(path, decode_times=False)
        name = twins._find_data_var(ds, cands)
        a = np.asarray(ds[name].values, dtype=np.float64)
        if a.ndim == 4:
            a = a[args.rec]
        a = np.where(np.abs(a) > 1e10, np.nan, a)
        a = np.moveaxis(a, 0, -1)                    # (y, x, z)
        lat = np.asarray(ds["nav_lat"].values, dtype=np.float64)
        lon = np.asarray(ds["nav_lon"].values, dtype=np.float64) % 360.0
        dname = "deptht" if "deptht" in ds else (
            "depthu" if "depthu" in ds else "depthv")
        zc = np.abs(np.asarray(ds[dname].values, dtype=np.float64))
        e3t = None
        if "e3t" in ds.variables:
            e3 = np.asarray(ds["e3t"].values, dtype=np.float64)
            if e3.ndim == 4:
                e3 = e3[args.rec]
            e3t = np.moveaxis(np.where(np.abs(e3) > 1e10, np.nan, e3), 0, -1)
        ds.close()
        return a, lat, lon, zc, name, e3t

    T, lat, lon, zc, tname, e3t = _grab3d(args.nemo_gridt, _T_CANDS)
    S, _, _, _, sname, _ = _grab3d(args.nemo_gridt, _S_CANDS)
    u, _, _, _, uname, _ = _grab3d(args.nemo_gridu, _U_CANDS)
    v, _, _, _, vname, _ = _grab3d(args.nemo_gridv, _V_CANDS)

    wet2d = np.isfinite(T[..., 0])
    band = box_fn(lat, lon) & wet2d
    ncol = int(band.sum())
    if ncol == 0:
        raise SystemExit("no wet NEMO columns in the box")
    print(f"[nemo] grid_T T={tname} S={sname}; grid_U u={uname}; "
          f"grid_V v={vname}; rec={args.rec}; {ncol} columns")

    # Vertical grid from e3t (per-column) if present, else global centre depths.
    if e3t is not None:
        # box-median thickness profile (open equatorial columns share geometry)
        dzc = np.nanmedian(e3t[band], axis=0)       # (nlev,)
        zi = np.concatenate([[0.0], np.cumsum(dzc)])
    else:
        zc1 = zc
        zi = np.empty(zc1.size + 1)
        zi[0] = 0.0
        zi[1:-1] = 0.5 * (zc1[:-1] + zc1[1:])
        zi[-1] = 2.0 * zc1[-1] - zi[-2]
    dz_ref, t_depth_ref, z_int_int = _centre_dz_and_depth(zi)

    def _stack(a):
        return a[band][None, ...]
    tau_x, tau_y = _stress_placeholder(ncol)
    cols = ColumnSet(
        T=_stack(T), S=_stack(S), u=_stack(u), v=_stack(v),
        eta=np.zeros((1, ncol)), dz_ref=dz_ref, t_depth_ref=t_depth_ref,
        z_interface_interior=z_int_int, tau_x=tau_x, tau_y=tau_y,
        lat=lat[band][None, ...], lon=lon[band][None, ...],
        source=str(args.nemo_gridt),
    )
    return cols, band


def _wire_stress(cols, args, twins):
    """Overwrite the placeholder NaN stress with NEMO tauuo/tauvo at the box."""
    ncol = cols.T.shape[1]
    stress = load_nemo_stress_for_columns(
        args.nemo_gridu, args.nemo_gridv, args.rec, None, None, twins)
    tx_f, la_u, lo_u, txn = stress["tx"]
    ty_f, la_v, lo_v, tyn = stress["ty"]
    latc = np.asarray(cols.lat)[0]                  # (ncol,)
    lonc = np.asarray(cols.lon)[0] % 360.0
    txo = np.empty(ncol); tyo = np.empty(ncol)
    for c in range(ncol):
        d = twins.great_circle_deg(la_u, lo_u, float(latc[c]), float(lonc[c]))
        j, i = np.unravel_index(int(np.nanargmin(np.where(np.isfinite(tx_f), d,
                                                          np.inf))), d.shape)
        txo[c] = tx_f[j, i]
        d2 = twins.great_circle_deg(la_v, lo_v, float(latc[c]), float(lonc[c]))
        j2, i2 = np.unravel_index(int(np.nanargmin(np.where(np.isfinite(ty_f),
                                                            d2, np.inf))),
                                  d2.shape)
        tyo[c] = ty_f[j2, i2]
    cols.tau_x = txo[None, :]
    cols.tau_y = tyo[None, :]
    print(f"[stress] wired NEMO {txn}/{tyn} rec {args.rec}: "
          f"|tau| median {np.nanmedian(np.hypot(txo, tyo)):.4f} N/m^2")


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("control", "nemo"))
    p.add_argument("--manifest", required=True, type=Path,
                   help="run_manifest.json of the snapshot arm (TKE config "
                        "source — no hidden config choice)")
    # column / box
    p.add_argument("--lat-halfwidth", type=float, default=DEFAULT_LAT_HALFWIDTH)
    p.add_argument("--lon-west", type=float, default=DEFAULT_LON_WEST)
    p.add_argument("--lon-east", type=float, default=DEFAULT_LON_EAST)
    # control inputs
    p.add_argument("--snapshot", type=Path, default=None,
                   help="our own --kprofile-snapshots npz (control mode)")
    p.add_argument("--perturb-shear", type=float, default=1.0,
                   help="scale u,v by this (non-vacuity: 2.0 must break the "
                        "control reproduction)")
    p.add_argument("--control-tol", type=float, default=1.25,
                   help="entrainment-band ratio tolerance (x); 1.25 = +-25%%")
    # nemo inputs
    p.add_argument("--nemo-gridt", type=Path, default=None)
    p.add_argument("--nemo-gridu", type=Path, default=None)
    p.add_argument("--nemo-gridv", type=Path, default=None)
    p.add_argument("--nemo-gridw", type=Path, default=None)
    p.add_argument("--rec", type=int, default=5,
                   help="NEMO 5-day record (5 = days 26-30)")
    # spin control
    p.add_argument("--dt", type=float, default=150.0,
                   help="TKE step [s] (match the arm's --dt for the en carry)")
    p.add_argument("--max-iter", type=int, default=4000)
    p.add_argument("--tol", type=float, default=1e-3,
                   help="relative band-K change for spin convergence")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    oracle, twins, runner = _import_reused()
    print(f"[prov] git {_git_sha()}")
    print(f"[prov] mode={args.mode}  box |lat|<={args.lat_halfwidth} "
          f"{args.lon_west:.0f}-{args.lon_east:.0f}E  rec={args.rec}")
    if args.mode == "control":
        if args.snapshot is None or args.nemo_gridu is None:
            raise SystemExit("control mode needs --snapshot and "
                             "--nemo-gridu/--nemo-gridv (for the surface stress)")
        print(f"[prov] snapshot {args.snapshot}")
        return run_control(args, oracle, twins, runner)
    for req in ("nemo_gridt", "nemo_gridu", "nemo_gridv", "nemo_gridw"):
        if getattr(args, req) is None:
            raise SystemExit(f"nemo mode needs --{req.replace('_', '-')}")
    print(f"[prov] NEMO T={args.nemo_gridt}")
    print(f"[prov] NEMO U={args.nemo_gridu}  V={args.nemo_gridv}  "
          f"W={args.nemo_gridw}")
    return run_nemo(args, oracle, twins, runner)


if __name__ == "__main__":
    raise SystemExit(main())
