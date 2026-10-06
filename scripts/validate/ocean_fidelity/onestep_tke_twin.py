#!/usr/bin/env python
"""ONE prognostic TKE step, OURS vs NEMO, from IDENTICAL input (equatorial strip).

NEMO (RK3, stprk3.F90:156-165) computes rn2b = bn2(ts(Nbb)) and calls
zdf_phy(kstp, Nbb, Nbb, Nrhs) ONCE at step entry: zdf_sh2(Kbb, Kmm, avm_k)
-> zdf_tke(sh2, avm_k, avt_k) advances en and rebuilds avm_k/avt_k
(zdfphy.F90:268-286).  rst_write runs AFTER the RK3 swap, so the restart
written at the end of step kt holds exactly the inputs of step kt+1:
tn, sn, un, vn (Kbb), en, avm_k, avt_k, dissl, sshn.  The restart written at
the end of step kt+1 holds NEMO's en after that step.  taum of step kt+1 comes
from the every-step strip surface file (sbc is called at step entry).

So R(kt) -> one call of OUR ``tke_vertical_mixing`` (production resolved
TKEConfig, dt = rn_Dt, n_iterations=1) -> compare with R(kt+1).en, per
interface, two-sided, with tails.  The frozen closure (frozen_column_tke_twin
--mode strip) already matches avm_k/avt_k to 1.000 on NEMO's state; this asks
whether the ENERGY STEP itself evolves en like NEMO's.

Shear: OUR production shear (squared_centered, t-point avm weighting) from
NEMO's collocated un/vn centred to T-points; NEMO's zdf_sh2 uses face-averaged
avm_k and face shear.  The four operand arms showed those choices do not move
the outcome; the residual of THIS step is printed per interface either way.

PLANTED FAILURE (gate): the same call with the input en scaled by
``--plant`` at ONE interface must move that interface's output ratio by more
than the reported tolerance; otherwise the instrument cannot see a local error
and the run exits non-zero.  Numbers only — no verdict.

NEMO writes no restart for a step that directly follows a listed one
(restart.F90:104-121 opens the file at nitrst-1), so kt and kt+1 come from two
identical runs ending at kt and kt+1 (end-of-run restarts).

Usage (compute node only):
  python onestep_tke_twin.py --run-dir <..._k95> --run-dir-next <..._k96> --kt 95 \
      --nemo-meshmask <mesh_mask.nc> --manifest <run_manifest.json>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
import frozen_column_tke_twin as fz  # noqa: E402  (shared loaders; extend, not duplicate)

STEP_DT = 150.0
PLANT_K = 8                      # interior interface index (0-based) ~ 12.8 m


def load_strip_taum(run_dir: Path, step: int):
    """|tau| of model step ``step`` from the every-step strip 2D file.

    Record r of a freq_op=1ts instant file is step nit000 + r (nit000 = 1);
    the record's own time stamp is checked against step*rn_Dt, never assumed.
    """
    import netCDF4 as nc
    f = sorted(run_dir.glob("ORCA1_1ts_*_eqs1ts_2D.nc"))
    if len(f) != 1:
        raise SystemExit(f"need exactly one eqs1ts_2D file in {run_dir}, got {f}")
    d = nc.Dataset(f[0])
    rec = step - 1
    t = np.asarray(d["time_instant"][:], dtype=np.float64)
    # absolute: the run starts at 00:00 (nn_date0), so record 0 must end 1*rn_Dt after midnight
    if abs((t[0] % 86400.0) - STEP_DT) > 1e-6:
        raise SystemExit(f"strip record 0 ends at {t[0] % 86400.0} s after midnight, not step 1")
    t_rel = t - t[0] + STEP_DT                         # record 0 = end of step 1
    if abs(t_rel[rec] - step * STEP_DT) > 1e-6:
        raise SystemExit(f"strip record {rec} time {t_rel[rec]} != step {step}*dt")
    a = np.asarray(d["taum"][rec], dtype=np.float64)
    lat = np.asarray(d["nav_lat_eqsT"][:], dtype=np.float64)
    lon = np.asarray(d["nav_lon_eqsT"][:], dtype=np.float64) % 360.0
    d.close()
    return np.where(np.abs(a) > 1e10, np.nan, a), lat, lon


def our_step(*, R, band, cfg, eos, rho0, g, dz_ref, t_depth_ref, taum_cols,
             en_in, dt, p_sh2=None, carry=None):
    """One TKE step on the strip columns; returns (en_new, zk).

    ``p_sh2`` (ncol, nlev-1): NEMO's own shear production (strip eshear_k =
    zdf_sh2 output, zdfphy.F90:363) injected with
    tke_shear_evaluation_stage='step_entry' -- isolates the non-shear part of
    the step (GLM arm a)."""
    import jax.numpy as jnp
    from legoesm.ocean.eos import (
        compute_ocean_rho, make_eos_fn, nemo_bn2_live_geometry,
    )
    from legoesm.ocean.vertical import (
        compute_ocean_jacobian, create_z_star_from_thicknesses,
    )
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing

    u_cell, v_cell = fz.centre_uv_collocated(R["U"], R["V"])
    T = jnp.asarray(R["T"][band][None]); S = jnp.asarray(R["S"][band][None])
    eta = jnp.asarray(R["ssh"][band][None, :])
    z_coord = create_z_star_from_thicknesses(
        jnp.asarray(dz_ref), jnp.asarray(t_depth_ref),
        nemo_e3w_source="depth_difference")
    H = jnp.full(eta.shape, float(t_depth_ref[-1] + 0.5 * dz_ref[-1]))
    J = compute_ocean_jacobian(eta, H, z_coord, 1.0)
    eos_fn = make_eos_fn(eos=eos)

    class _F:
        __slots__ = ("data",)
        def __init__(self, d): self.data = d

    class _St:
        pass
    st = _St(); st.T = _F(T); st.S = _F(S); st.eta = _F(eta)
    rho = compute_ocean_rho(st, z_coord, J, eos_fn, rho0=rho0, g=g)
    dz_half = jnp.asarray(z_coord.dz_half_ref) * J[..., None]
    t_depth, w_depth, e3w_int = nemo_bn2_live_geometry(z_coord, eta, H)
    out = tke_vertical_mixing(
        jnp.asarray(u_cell[band][None]), jnp.asarray(v_cell[band][None]),
        T, S, rho, dz_half,
        tke_old=jnp.asarray(en_in[None]),
        tau_x_surface=None, tau_y_surface=None,
        dt=float(dt), cfg=cfg, rho_0=rho0, g=g, n_iterations=1,
        taum_surface=jnp.asarray(taum_cols[None, :]),
        dz_ref=jnp.asarray(dz_ref), jacobian=J, eos_fn=eos_fn,
        z_interface=jnp.asarray(z_coord.z_half_ref[1:-1]),
        dz_surface=(-jnp.asarray(z_coord.z_full_ref[0])) * J,
        lat_deg=jnp.asarray(R["nav_lat"][band][None, :]),
        T_n2=T, S_n2=S, t_depth=t_depth, w_depth=w_depth, e3w_int=e3w_int,
        precomputed_p_sh2=(None if p_sh2 is None else jnp.asarray(p_sh2[None])),
        **({} if carry is None else {k: jnp.asarray(v[None]) for k, v in carry.items()}))
    zk = np.abs(np.asarray(z_coord.z_half_ref[1:-1]))
    our_step.last = out
    return np.asarray(out.tke_new)[0], zk


TRACER_OP_TOL_K = 1e-6          # pre-registered operator-equivalence floor


def _strip_file(run_dir: Path, grid: str):
    f = sorted(run_dir.glob(f"ORCA1_1ts_*_eqs1ts_{grid}.nc"))
    if len(f) != 1:
        raise SystemExit(f"need exactly one eqs1ts_{grid} file in {run_dir}, got {f}")
    return f[0]


def tracer_operator_check(run_next: Path, step: int, sj, si, ssh_mid, e3w_1d, zmax_k):
    """OPERATOR equivalence of the tracer vertical solve (not a forward replay).

    NEMO trazdf.F90:207-221,272-285 (Aimp and the MSC/akz add-on are not in the
    output, stated as a limitation):
      (A x)_k = e3t_k(Kaa) x_k - dt*[avt_k+1/e3w_k+1(Kmm) (x_k+1 - x_k)
                                   - avt_k/e3w_k(Kmm) (x_k - x_k-1)]
    avt  = strip avt at ``step`` (set at that step's entry, zdfphy.F90:313-336);
    e3t(Kaa) = strip e3t at the end of ``step`` (written after the RK3 swap);
    e3w(Kmm) = mesh e3w_1d * (1 + ssh(N+1/2)/H) (qco z-star; deep strip columns
    have full cells in the scored band).  b = A_NEMO @ T_after is formed HERE in
    numpy, independently of our code, over the FULL wet column (avt is masked
    to 0 at the sea floor, so no artificial truncation); OUR production solver
    (implicit_vertical_diffusion_ocean, dz=e3t(Kaa), dz_half=e3w(Kmm)) must
    return T_after.  Records are time-checked; T and W strips must share
    coordinates."""
    import netCDF4 as nc
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import implicit_vertical_diffusion_ocean
    dT, dW = nc.Dataset(_strip_file(run_next, "T")), nc.Dataset(_strip_file(run_next, "W"))
    for d in (dT, dW):
        t = np.asarray(d["time_instant"][:], dtype=np.float64)
        if abs((t[step - 1] - t[0]) - (step - 1) * STEP_DT) > 1e-6 or abs((t[0] % 86400.0) - STEP_DT) > 1e-6:
            raise SystemExit(f"strip record {step - 1} of {d.filepath()} is not step {step}")
    if not (np.allclose(dT["nav_lat_eqsT"][:], dW["nav_lat_eqsW"][:]) and
            np.allclose(dT["nav_lon_eqsT"][:], dW["nav_lon_eqsW"][:])):
        raise SystemExit("T and W strips are not co-located")
    rec = step - 1
    g = lambda d, v: np.ma.filled(d[v][rec], np.nan).astype(np.float64)[:, sj, si].T   # (ncol, z)
    e3a, Ta, avt = g(dT, "e3t"), g(dT, "votemper"), g(dW, "avt")
    dT.close(); dW.close()
    nz = Ta.shape[1]
    wet = np.isfinite(Ta) & np.isfinite(e3a) & (e3a > 0)
    if not np.all(wet[:, :zmax_k + 1]):
        raise SystemExit("a matched strip column is dry inside the scored band")
    H = np.sum(np.where(wet, e3a, 0.0), axis=1)                                  # live column depth
    e3w = np.asarray(e3w_1d)[None, 1:nz] * (1.0 + ssh_mid[:, None] / H[:, None])     # interfaces 2..nz
    K = np.where(wet[:, 1:] & wet[:, :-1] & np.isfinite(avt[:, 1:]), avt[:, 1:], 0.0)
    e3a = np.where(wet, e3a, 1.0)
    Tf = np.where(wet, Ta, 0.0)
    flux = K / e3w * (Tf[:, 1:] - Tf[:, :-1])
    div = np.zeros_like(Tf); div[:, :-1] += flux; div[:, 1:] -= flux
    dt = STEP_DT
    b = e3a * Tf - dt * div
    solve = lambda KK: np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(b / e3a), jnp.asarray(KK), jnp.asarray(e3a), jnp.asarray(e3w), dt))
    x = solve(K)
    err = np.abs(x - Tf)[:, :zmax_k]
    inc = np.abs(dt * div / e3a)[:, :zmax_k]
    Kp = K.copy(); Kp[:, PLANT_K] *= 1.01
    eplant = float(np.abs(solve(Kp) - Tf)[:, :zmax_k].max())
    print(f"\n[tracer-op] step {step}: {Tf.shape[0]} columns, full wet columns, scored top {zmax_k} levels; "
          f"max|x-T| {err.max():.3e} K (p99 {np.percentile(err, 99):.3e}); diffusive increment median "
          f"{np.median(inc):.3e} K max {inc.max():.3e} K; plant K x1.01 at interface {PLANT_K + 1}: {eplant:.3e} K; "
          f"dtype {x.dtype}")
    return float(err.max()), eplant


def stats(ours, nemo, zk, kmax, label):
    """Per-interface two-sided relative error of en, ours vs NEMO."""
    m = np.isfinite(ours) & np.isfinite(nemo) & (nemo > 0)
    rel = np.where(m, ours / np.where(m, nemo, 1.0) - 1.0, np.nan)
    print(f"\n[{label}]   k   z_w |  en NEMO med  en ours med | rel p10 / p50 / p90 | max|rel| | frac|rel|>2%")
    for k in range(kmax):
        r = rel[:, k][np.isfinite(rel[:, k])]
        if not r.size:
            continue
        p = np.percentile(r, [10, 50, 90])
        print(f"  {k + 1:2d} {zk[k]:6.2f} | {np.median(nemo[m[:, k], k]):12.4e} {np.median(ours[m[:, k], k]):12.4e} "
              f"| {p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f} | {np.max(np.abs(r)):.4f} | {np.mean(np.abs(r) > 0.02):.3f}")
    return rel


def budget_compare(budget, run_next, step, sj, si, dissl_rst, *, our_step_e, cfg, zk, kmax, post):
    """Arm a's budget terms vs NEMO's own strip diagnostics of the SAME step.

    eshear_k = p_sh2 (injected, so ~identity: a self-check), estrat_k = -K_H*N2
    with post-avn K_H, ediss_k = 0.5*rn_ediss*dissl_old*en_post.  The implied
    dissl_old (ours: ediss/(0.5 c_eps e_post)) is also compared with
    restart(kt) dissl, which is what NEMO's matrix consumes."""
    ours = dict(eshear_k=budget[0], estrat_k=budget[1], ediss_k=budget[2])
    print("\n[budget arm a] per-interface median ratio ours/NEMO (same step; NEMO strip W)")
    print("   k   z_w | eshear | estrat | ediss  | dissl_used/dissl_rst(kt) | K_M/avm_k(kt+1) | K_H/avt_k(kt+1) | implied N2")
    nemo = {n: load_strip_w(run_next, step, n)[:, sj, si].T[:, 1:] for n in ours}
    dissl_ours = np.asarray(budget[2])[0] / np.maximum(0.5 * cfg.c_eps * our_step_e, 1e-300)
    def ratio(a, b):
        ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
        return float(np.median(a[ok] / b[ok])) if ok.any() else np.nan
    for k in range(kmax):
        row = [ratio(np.asarray(ours[n])[0][:, k], nemo[n][:, k]) for n in ours]
        d = dissl_rst[:, k]
        rd = ratio(dissl_ours[:, k], d)
        km, kh = ratio(post[0][:, k], post[2][:, k]), ratio(post[1][:, k], post[3][:, k])
        n2 = ratio(-np.asarray(ours["estrat_k"])[0][:, k] / np.where(post[1][:, k] > 0, post[1][:, k], np.nan),
                   -nemo["estrat_k"][:, k] / np.where(post[3][:, k] > 0, post[3][:, k], np.nan))
        print(f"  {k + 1:2d} {zk[k]:6.2f} | " + " | ".join(f"{v:6.3f}" for v in row)
              + f" | {rd:6.3f} | {km:6.3f} | {kh:6.3f} | {n2:6.3f}")


def _mesh_e3w(mesh):
    import netCDF4 as nc
    d = nc.Dataset(mesh)
    e3t = np.abs(np.asarray(d["e3t_1d"][:], dtype=np.float64).ravel())
    e3w = np.abs(np.asarray(d["e3w_1d"][:], dtype=np.float64).ravel())
    d.close()
    return e3t, e3w


def check_runs_identical(run_a: Path, run_b: Path, kt: int):
    """The two NEMO runs must differ ONLY in nn_itend, and share the trajectory
    up to step kt bitwise (strip votemper/avt at step kt) -- codex P1."""
    import re
    import netCDF4 as nc
    strip = lambda f: [ln for ln in Path(f).read_text().splitlines()
                       if not re.match(r"\s*nn_itend\s*=", ln)]
    if strip(run_a / "namelist_cfg") != strip(run_b / "namelist_cfg"):
        raise SystemExit("the two runs' namelist_cfg differ beyond nn_itend")
    if (run_a / "nemo.exe").resolve() != (run_b / "nemo.exe").resolve():
        raise SystemExit("the two runs use different nemo.exe")
    for grid, var in (("T", "votemper"), ("W", "avt")):
        x = [np.ma.filled(nc.Dataset(_strip_file(r, grid))[var][kt - 1], np.nan) for r in (run_a, run_b)]
        if not np.array_equal(x[0], x[1], equal_nan=True):
            raise SystemExit(f"runs diverge before step {kt}: strip {var} at step {kt} differs")
    print(f"[runs] namelists differ only in nn_itend; strip votemper/avt at step {kt} bitwise equal")


def match_columns(lat, lon, band, slat, slon):
    """Exact (1e-4 deg) T-point match of restart box columns to strip cells.

    Returns (sj, si, keep): strip indices per kept column (row-major band
    order) and the keep mask over np.nonzero(band).  No nearest-neighbour fill.
    """
    jj, ii = np.nonzero(band)
    sj = np.full(jj.size, -1); si = np.full(jj.size, -1)
    for c in range(jj.size):
        hit = (np.abs(slat - lat[jj[c], ii[c]]) < 1e-4) & \
              (np.abs(((slon - lon[jj[c], ii[c]] + 180.0) % 360.0) - 180.0) < 1e-4)
        if hit.sum() == 1:
            sj[c], si[c] = (int(x[0]) for x in np.nonzero(hit))
    return sj, si, sj >= 0


def load_strip_w(run_next: Path, step: int, var: str):
    """(z, y, x) strip W field of step ``step`` (record step-1; time checked by
    load_strip_taum on the same run)."""
    import netCDF4 as nc
    f = sorted(run_next.glob("ORCA1_1ts_*_eqs1ts_W.nc"))[0]
    d = nc.Dataset(f)
    a = np.ma.filled(d[var][step - 1], np.nan).astype(np.float64)
    d.close()
    return np.where(np.abs(a) > 1e10, np.nan, a)


def band_report(rel, chg, zk, lo, hi, label):
    """Signed bias vs NEMO's own per-step change, and sign coherence (GLM)."""
    bk = (zk >= lo) & (zk <= hi)
    r = rel[:, bk]; c = chg[:, bk]
    fin = np.isfinite(r)
    med = float(np.nanmedian(r)); sig = float(np.nanmedian(np.abs(c)))
    pos = float(np.mean(r[fin] > 0))
    print(f"[{label}] band {lo:g}-{hi:g} m: signed median rel {med:+.5f}  |  NEMO median |den/en| per step "
          f"{sig:.5f}  |  bias/signal {abs(med) / max(sig, 1e-30):.3f}  |  fraction rel>0 {pos:.3f} (n={int(fin.sum())})")
    return med, sig, pos


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="NEMO run whose END-OF-RUN restart is step kt (input)")
    p.add_argument("--run-dir-next", required=True, type=Path,
                   help="NEMO run ending at kt+1 (output en + strip taum/eshear_k of step kt+1)")
    p.add_argument("--kt", type=int, required=True, help="input restart step; output = kt+1")
    p.add_argument("--nemo-meshmask", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--lat-halfwidth", type=float, default=2.0)
    p.add_argument("--lon-west", type=float, default=225.0)
    p.add_argument("--lon-east", type=float, default=256.0)
    p.add_argument("--zmax", type=float, default=30.0)
    p.add_argument("--band-lo", type=float, default=8.0)
    p.add_argument("--band-hi", type=float, default=18.0)
    p.add_argument("--plant", type=float, default=1.10,
                   help="planted local en scale at interface PLANT_K (gate)")
    p.add_argument("--plant-uniform", type=float, default=1.005,
                   help="planted UNIFORM en scale at every interface (distributed-bias gate)")
    a = p.parse_args(argv)

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    import jax
    jax.config.update("jax_enable_x64", True)
    oracle, twins = fz._import_reused()
    # nemo_z0 surface row is handled INSIDE tke_vertical_mixing (the solve), so
    # the direct-K-only restriction in load_resolved_config does not apply here.
    cfg, eos, rho0, g = fz.load_resolved_config(Path(a.manifest), allow_z0_direct=True)
    fz._echo_cfg(cfg, eos, rho0, g)
    names = ("T", "S", "U", "V", "en", "avt_k", "avm_k", "dissl", "ssh")
    R0 = fz.reassemble_restart(a.run_dir / f"ORCA1_{a.kt:08d}_restart_oce_*.nc", names, twins)
    R1 = fz.reassemble_restart(a.run_dir_next / f"ORCA1_{a.kt + 1:08d}_restart_oce_*.nc", ("en", "avm_k", "avt_k", "ssh"), twins)
    check_runs_identical(a.run_dir, a.run_dir_next, a.kt)
    box_args = argparse.Namespace(lat_halfwidth=a.lat_halfwidth, lon_west=a.lon_west, lon_east=a.lon_east)
    # deep columns only (>= 45 wet levels, ~ >1000 m): the 1-D ladder is exact in
    # the scored band and no bottom constraint reaches it (codex P1 geometry)
    band = (fz._box(R0["nav_lat"], R0["nav_lon"], box_args)
            & (np.isfinite(R0["T"]).sum(axis=-1) >= 45))
    dz_ref, t_depth_ref, w_interior = fz.native_ladders_meshmask(a.nemo_meshmask, twins)
    import netCDF4 as nc
    rdt = {float(np.ravel(nc.Dataset(sorted(r.glob(f"ORCA1_{k:08d}_restart_oce_*.nc"))[0])["rdt"][:])[0])
           for r, k in ((a.run_dir, a.kt), (a.run_dir_next, a.kt + 1))}
    if rdt != {STEP_DT}:
        raise SystemExit(f"restart rdt {rdt} != {STEP_DT} s (STEP_DT is fixed in this probe)")
    taum, tlat, tlon = load_strip_taum(a.run_dir_next, a.kt + 1)
    sj, si, keep = match_columns(R0["nav_lat"], R0["nav_lon"], band, tlat, tlon)
    jj, ii = np.nonzero(band)
    band[jj[~keep], ii[~keep]] = False
    sj, si = sj[keep], si[keep]
    ncol = int(band.sum())
    if ncol < fz.MIN_COLUMNS:
        raise SystemExit(f"only {ncol} strip columns matched exactly (< {fz.MIN_COLUMNS})")
    tau_cols = taum[sj, si]
    # NEMO's own p_sh2 of step kt+1 at interior w-levels 2..jpk -> our interfaces
    esh = load_strip_w(a.run_dir_next, a.kt + 1, "eshear_k")[:, sj, si].T     # (ncol, nz_strip)
    print(f"[twin] kt {a.kt} -> {a.kt + 1}; {ncol} columns; |tau| median {np.median(tau_cols):.4f} N/m2; "
          f"T dtype {R0['T'].dtype}; eshear_k levels {esh.shape[1]}")

    en_in = R0["en"][band][:, 1:]                    # interior w-levels 2..jpk -> our interfaces
    en_nemo = R1["en"][band][:, 1:]
    kw = dict(R=R0, band=band, eos=eos, rho0=rho0, g=g, dz_ref=dz_ref,
              t_depth_ref=t_depth_ref, taum_cols=tau_cols, dt=STEP_DT)
    ours, zk = our_step(cfg=cfg, en_in=en_in, **kw)
    print(f"[twin] dtypes: en_out {ours.dtype} dz_ref {np.asarray(dz_ref).dtype}")
    if not np.allclose(zk[:20], np.asarray(w_interior)[:20], atol=0.05):
        raise SystemExit("our interface ladder != NEMO gdepw interior (k mapping broken)")
    kmax = int(np.searchsorted(zk, a.zmax))
    chg = np.where(en_in > 0, en_nemo / np.where(en_in > 0, en_in, 1.0) - 1.0, np.nan)
    print("\n[signal] NEMO en(kt+1)/en(kt)-1 median per interface: "
          + " ".join(f"{np.nanmedian(chg[:, k]):+.4f}" for k in range(kmax)))
    rel = stats(ours, en_nemo, zk, kmax, "arm b: production (own shear)")
    band_report(rel, chg, zk, a.band_lo, a.band_hi, "arm b")
    # arm a: NEMO's p_sh2 injected -> residual = non-shear part of the step
    nsh = min(esh.shape[1] - 1, en_in.shape[1])
    p_sh2 = np.zeros_like(en_in); p_sh2[:, :nsh] = esh[:, 1:nsh + 1]
    p_sh2 = np.where(np.isfinite(p_sh2), p_sh2, 0.0)
    cfg_a = cfg._replace(tke_shear_evaluation_stage="step_entry")
    ours_a, _ = our_step(cfg=cfg_a, en_in=en_in, p_sh2=p_sh2, **kw)
    rel_a = stats(ours_a, en_nemo, zk, min(kmax, nsh), "arm a: NEMO p_sh2 injected")
    band_report(rel_a, chg, zk, a.band_lo, a.band_hi, "arm a")
    budget_a = our_step.last.budget
    post_a = (np.asarray(our_step.last.K_M)[0], np.asarray(our_step.last.K_H)[0],
              R1["avm_k"][band][:, 1:], R1["avt_k"][band][:, 1:])
    # arm c: NEMO's time level for the closure coefficients -- avm_k/avt_k
    # (and surface avm_k) CARRIED from restart(kt), as zdftke consumes them
    carry = dict(preclosure_K_M=R0["avm_k"][band][:, 1:], preclosure_K_H=R0["avt_k"][band][:, 1:],
                 preclosure_K_M_surface=R0["avm_k"][band][:, 0])
    cfg_c = cfg_a._replace(tke_preclosure_coeff_source="carried_previous_step")
    ours_c, _ = our_step(cfg=cfg_c, en_in=en_in, p_sh2=p_sh2, carry=carry, **kw)
    rel_c = stats(ours_c, en_nemo, zk, min(kmax, nsh), "arm c: NEMO p_sh2 + carried avm_k/avt_k")
    band_report(rel_c, chg, zk, a.band_lo, a.band_hi, "arm c")
    kh_nemo = R1["avt_k"][band][:, 1:]
    def kh_line(lab):
        kh = np.asarray(our_step.last.K_H)[0]
        ok = np.isfinite(kh) & np.isfinite(kh_nemo) & (kh_nemo > 0)
        print(f"[{lab}] K_H/avt_k(kt+1) median per interface: " + " ".join(
            f"{np.median(kh[ok[:, k], k] / kh_nemo[ok[:, k], k]):.3f}" for k in range(min(kmax, nsh))))
    kh_line("arm c")
    # arm d: arm c + NEMO's Richardson form (rn2b*avm_old/p_sh2, nemo_ri)
    ours_d, _ = our_step(cfg=cfg_c._replace(prandtl_mode="nemo_ri"), en_in=en_in, p_sh2=p_sh2, carry=carry, **kw)
    band_report(stats(ours_d, en_nemo, zk, min(kmax, nsh), "arm d: arm c + prandtl nemo_ri"),
                chg, zk, a.band_lo, a.band_hi, "arm d")
    kh_line("arm d")
    # arm e: arm c with Langmuir off -- magnitude of that source in the band only
    ours_e, _ = our_step(cfg=cfg_c._replace(lc=False), en_in=en_in, p_sh2=p_sh2, carry=carry, **kw)
    band_report(stats(ours_e, en_nemo, zk, min(kmax, nsh), "arm e: arm c, Langmuir off"),
                chg, zk, a.band_lo, a.band_hi, "arm e")
    budget_compare(budget_a, a.run_dir_next, a.kt + 1, sj, si, R0["dissl"][band][:, 1:],
                   our_step_e=ours_a, cfg=cfg_a, zk=zk, kmax=min(kmax, nsh), post=post_a)
    # GATES: a local plant and a uniform (distributed) plant, both on arm a
    bk = (zk >= a.band_lo) & (zk <= a.band_hi)
    en_p = en_in.copy(); en_p[:, PLANT_K] *= a.plant
    ours_p, _ = our_step(cfg=cfg_a, en_in=en_p, p_sh2=p_sh2, **kw)
    rel_p = ours_p / np.where(en_nemo > 0, en_nemo, np.nan) - 1.0
    shift = float(np.nanmedian(rel_p[:, PLANT_K]) - np.nanmedian(rel_a[:, PLANT_K]))
    ours_u, _ = our_step(cfg=cfg_a, en_in=en_in * a.plant_uniform, p_sh2=p_sh2, **kw)
    rel_u = ours_u / np.where(en_nemo > 0, en_nemo, np.nan) - 1.0
    ushift = float(np.nanmedian(rel_u[:, bk]) - np.nanmedian(rel_a[:, bk]))
    print(f"\n[plant] local x{a.plant} at interface {PLANT_K + 1} ({zk[PLANT_K]:.2f} m): median shift {shift:+.5f}; "
          f"uniform x{a.plant_uniform}: band median shift {ushift:+.5f}")
    rc = 0
    if abs(shift) <= 0.02:
        print("[plant] FAIL: a planted local input error is invisible at the 2% tolerance"); rc = 2
    if abs(ushift) < 0.5 * (a.plant_uniform - 1.0):
        print("[plant] FAIL: a planted uniform input bias is not resolved in the band"); rc = 2
    if not all(np.isfinite(v) for v in (shift, ushift)):
        print("[plant] FAIL: non-finite plant response"); rc = 2
    for lab, r in (("arm b", rel), ("arm a", rel_a)):
        cov = float(np.isfinite(r[:, bk]).mean())
        if cov < 0.95:
            print(f"[coverage] FAIL: {lab} scores only {cov:.3f} of band samples"); rc = rc or 4
    _, e3w_1d = _mesh_e3w(a.nemo_meshmask)
    ssh_mid = 0.5 * (R0["ssh"][band] + R1["ssh"][band])
    emax, eplant = tracer_operator_check(a.run_dir_next, a.kt + 1, sj, si, ssh_mid, e3w_1d, kmax)
    if not (np.isfinite(emax) and np.isfinite(eplant)) or eplant <= 10.0 * emax:
        print("[tracer-op] FAIL: planted K error not >> the operator floor (or non-finite)"); rc = rc or 3
    print(f"[tracer-op] pre-registered floor {TRACER_OP_TOL_K:.0e} K: "
          + ("within" if emax < TRACER_OP_TOL_K else "EXCEEDED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
