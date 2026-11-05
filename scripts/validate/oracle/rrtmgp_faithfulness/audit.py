"""Numerical audit of legoESM RRTMGP against the latest RTE+RRTMGP Fortran.

Both models get identical inputs; every comparison is printed as one line and
written to <out>/audit.json.  No verdicts here: bars live in the test that
consumes these numbers.

Levels, one variable changed at a time:
  tables   lookup tables, ours vs upstream, variable by variable
  optics   gas optical depth, ssa, Planck sources, solar weights, same state
  solver   our production solve_lw / solve_sw fed the ORACLE's optics, so any
           difference is in the transport alone (real columns + synthetic sweep)
  e2e      production solve_columns vs the oracle chain, clear and all-sky,
           float64 and the production float32

usage (sbatch, JAX_ENABLE_X64=1):
  python audit.py --oracle <driver> --data <rrtmgp-data> --out <dir> [--tables ours|upstream]
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp import constants as RC  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp as R  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import atmospheric_state  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cloud_optics  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (  # noqa: E402
    monochromatic_two_stream as MTS, two_stream as TS)

RFMIP = "examples/rfmip-clear-sky/inputs/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
OURS_DATA = Path(cloud_optics.__file__).parent / "rrtmgp_data"
RESULTS = {}
SECONDS_PER_DAY = 86400.0


def cmp(name, ours, ref, floor=0.0):
    """Print max |diff|, max relative diff above ``floor``, and rms diff."""
    a, b = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), f"non-finite in {name}"
    d = np.abs(a - b)
    i = np.unravel_index(int(np.argmax(d)), d.shape)
    m = np.abs(b) > floor
    rel = float(np.max(d[m] / np.abs(b[m]))) if m.any() else 0.0
    RESULTS[name] = dict(max_abs=float(d.max()), max_rel=rel, rms=float(np.sqrt(np.mean(d * d))),
                         at=[int(x) for x in i], ours=float(a[i]), ref=float(b[i]),
                         ref_max=float(np.abs(b).max()), n=int(d.size))
    print(f"{name:58s} max|d|={d.max():.3e} rel={rel:.3e} rms={RESULTS[name]['rms']:.3e} "
          f"ref_max={np.abs(b).max():.3e} at={tuple(int(x) for x in i)}", flush=True)


def compare_tables(data):
    pairs = [("rrtmgp-gas-lw-g128.nc", "rrtmgp-gas-lw-g128.nc"),
             ("rrtmgp-gas-sw-g112.nc", "rrtmgp-gas-sw-g112.nc"),
             ("cloudysky_lw.nc", "rrtmgp-clouds-lw-bnd.nc"),
             ("cloudysky_sw.nc", "rrtmgp-clouds-sw-bnd.nc")]
    for mine, theirs in pairs:
        a, b = netCDF4.Dataset(OURS_DATA / mine), netCDF4.Dataset(Path(data) / theirs)
        va, vb = set(a.variables), set(b.variables)
        print(f"TABLE {mine} vs {theirs}: only ours {sorted(va - vb)} only upstream {sorted(vb - va)}")
        print(f"  attrs ours {dict((k, str(a.getncattr(k))[:70]) for k in a.ncattrs())}")
        print(f"  attrs upstream {dict((k, str(b.getncattr(k))[:70]) for k in b.ncattrs())}")
        for v in sorted(va & vb):
            x, y = a[v][:], b[v][:]
            if x.shape != y.shape:
                print(f"  {v}: SHAPE {x.shape} vs {y.shape}")
            elif x.dtype.kind in "fi":
                cmp(f"table/{mine}/{v}", np.ma.filled(x.astype(float), 0.0),
                    np.ma.filled(y.astype(float), 0.0), 1e-300)
            elif not np.array_equal(np.ma.filled(x), np.ma.filled(y)):
                print(f"  {v}: non-numeric content differs")
        for v in sorted(va - vb):  # old cloud file: lut_* / radice_* are upstream's * / diamice_*
            new = v.removeprefix("lut_").replace("radice_", "diamice_")
            if new != v and new in vb:
                x, y = a[v][:], b[new][:]
                if x.shape != y.shape:
                    print(f"  {v} vs {new}: SHAPE {x.shape} vs {y.shape}")
                else:
                    cmp(f"table/{mine}/{v}->{new}", np.ma.filled(x.astype(float), 0.0),
                        np.ma.filled(y.astype(float), 0.0), 1e-300)
        a.close(); b.close()


def columns(data, p_top_target):
    ds = netCDF4.Dataset(Path(data) / RFMIP)
    g = lambda k: np.asarray(ds[k][:], dtype=np.float64)  # noqa: E731
    p_lay, p_lev, t_lay = g("pres_layer"), g("pres_level"), g("temp_layer")[0]
    h2o = g("water_vapor")[0] * float(ds["water_vapor"].units)
    o3 = g("ozone")[0] * float(ds["ozone"].units)
    gm = {k: float(g(f"{k}_GM")[0]) * float(ds[f"{k}_GM"].units)
          for k in ("carbon_dioxide", "methane", "nitrous_oxide", "cfc11", "cfc12")}
    c = dict(tsfc=g("surface_temperature")[0], emis=g("surface_emissivity"), alb=g("surface_albedo"),
             mu0=np.cos(np.deg2rad(g("solar_zenith_angle"))), tsi=g("total_solar_irradiance"))
    ds.close()
    if p_lev[0, 0] > p_lev[0, -1]:  # index 0 = TOA
        p_lay, p_lev, t_lay, h2o, o3 = (a[:, ::-1] for a in (p_lay, p_lev, t_lay, h2o, o3))
    dp = np.diff(p_lev, axis=1)
    k0 = int(np.max(np.argmax(p_lev >= p_top_target, axis=1)))
    n = p_lay.shape[0]
    low = np.array([0.5, 0.02, 0.005, 0.002])[np.arange(n) % 4]
    c["mu0"] = np.where(c["mu0"] > 0.05, c["mu0"], low)
    r = RC.R_V / RC.R_D
    c.update(p_half=p_lev[:, k0:], p_full=p_lay[:, k0:], T=t_lay[:, k0:], o3=o3[:, k0:],
             q=h2o[:, k0:] / (r + h2o[:, k0:]))
    c["o3_top"] = (np.sum(o3[:, :k0] * dp[:, :k0], axis=1) / np.sum(dp[:, :k0], axis=1)
                   if k0 else o3[:, 0])
    ghg = {"co2": gm["carbon_dioxide"], "ch4": gm["methane"], "n2o": gm["nitrous_oxide"],
           "cfc11": gm["cfc11"], "cfc12": gm["cfc12"]}
    return c, ghg


def add_clouds(c):
    """Overcast liquid and ice layers in every other column, inside table range."""
    pf = c["p_full"]
    i = np.arange(pf.shape[0])[:, None]
    on = (i % 2 == 0)
    liq = np.where(on & (pf > 8.5e4) & (pf < 9.2e4), 0.03 * (1 + i % 4), 0.0)
    ice = np.where(on & (pf > 2.0e4) & (pf < 3.0e4), 0.01 * (1 + i % 3), 0.0)
    return dict(cloud_path_liq=liq, cloud_path_ice=ice,
                cloud_r_eff_liq=np.broadcast_to((6.0 + 3.0 * (i % 5)) * 1e-6, pf.shape).copy(),
                cloud_r_eff_ice=np.broadcast_to((20.0 + 10.0 * (i % 6)) * 1e-6, pf.shape).copy())


def convert_cloud_table(src, dst):
    """Upstream cloud table under the variable names our reader expects."""
    rename = {v: "lut_" + v for v in ("extliq", "ssaliq", "asyliq", "extice", "ssaice", "asyice")}
    rename.update(diamice_lwr="radice_lwr", diamice_upr="radice_upr")
    with netCDF4.Dataset(src) as a, netCDF4.Dataset(dst, "w") as b:
        for n, d in a.dimensions.items():
            b.createDimension(n, len(d))
        for n, v in a.variables.items():
            b.createVariable(rename.get(n, n), v.dtype, v.dimensions)[:] = v[:]
    return str(dst)


def run_ours(c, ghg, cloud, overhead, fp32, s0, capture=None, cloud_files=("", "")):
    cfg = RRTMGPConfig(lw_cloud_file=cloud_files[0], sw_cloud_file=cloud_files[1], S_0=s0, include_clouds=cloud is not None, overhead_layer=overhead,
                       compute_fp32=fp32, gpoint_batch_size=0, gpoint_checkpoint=False,
                       use_optimal_angle=False, use_scan=True, co2_ppmv=ghg["co2"] * 1e6,
                       ch4_ppbv=ghg["ch4"] * 1e9, n2o_ppbv=ghg["n2o"] * 1e9)
    orig = TS.solve_lw

    def cap(pressure, temperature, molecules, optics_lib, atmos_state, vmr_fields=None,
            sfc_temperature=None, **kw):
        capture.update(p=pressure, T=temperature, mol=molecules, vmr=dict(vmr_fields),
                       sfcT=sfc_temperature, optics=optics_lib, kw=kw)
        return orig(pressure, temperature, molecules, optics_lib, atmos_state, vmr_fields,
                    sfc_temperature, **kw)

    if capture is not None:
        TS.solve_lw = cap
    try:
        solver = R.RRTMGP.from_legoesm_config(cfg)
        kw = {k: jnp.asarray(v) for k, v in (cloud or {}).items()}
        out = solver.solve_columns(
            T=jnp.asarray(c["T"]), p_full=jnp.asarray(c["p_full"]), p_half=jnp.asarray(c["p_half"]),
            sfc_temperature=jnp.asarray(c["tsfc"]), q_v=jnp.asarray(c["q"]),
            cos_zenith=jnp.asarray(c["mu0"]), sfc_albedo=jnp.asarray(c["alb"]),
            sfc_emissivity=jnp.asarray(c["emis"]), o3_vmr=jnp.asarray(c["o3"]),
            o3_top_vmr=jnp.asarray(c["o3_top"]), ghg_vmr_override=ghg, **kw)
    finally:
        TS.solve_lw = orig
    return out


def write_f(path, arrays):
    with open(path, "wb") as f:
        for a in arrays:
            np.asarray(a, dtype="<f8").ravel(order="F").tofile(f)


class Reader:
    def __init__(self, path):
        self.raw, self.o = np.fromfile(path, dtype="<f8"), 0

    def take(self, *shape):
        n = int(np.prod(shape))
        a = self.raw[self.o:self.o + n].reshape(shape, order="F")
        self.o += n
        return a

    def done(self):
        assert self.o == self.raw.size, (self.o, self.raw.size)


class OracleOptics:
    """Hands our production solvers the oracle's optics, one g-point at a time."""

    def __init__(self, dtype, lw=None, sw=None, lev=None, sfc=None):
        pad = lambda a: jnp.asarray(  # noqa: E731  (g, col, lay) -> (g, col, 1, lay+2)
            np.pad(a, ((0, 0), (0, 0), (1, 1)))[:, :, None, :], dtype)
        if lw is not None:
            self.lw = {k: pad(v) for k, v in lw.items()}
            self.n_gpt_lw = lw["optical_depth"].shape[0]
            # Our solver takes the geometric mean of the two sources meeting at a
            # level, so handing it lev on both sides reproduces lev exactly.
            bottom = np.concatenate([np.zeros_like(lev[:, :, :1]), lev], axis=2)
            top = np.concatenate([lev[:, :, :1], lev[:, :, 1:], np.zeros_like(lev[:, :, :1])], axis=2)
            self.bottom = jnp.asarray(bottom[:, :, None, :], dtype)
            self.top = jnp.asarray(top[:, :, None, :], dtype)
            self.sfc = jnp.asarray(sfc[:, :, None], dtype)
        if sw is not None:
            self.sw = {k: pad(v) for k, v in sw.items()}
            self.n_gpt_sw = sw["optical_depth"].shape[0]

    def compute_lw_optical_properties(self, p, T, mol, igpt, vmr=None, *a, **k):
        return {n: v[igpt] for n, v in self.lw.items()}

    def compute_planck_sources(self, p, T, igpt, vmr=None, sfc_temperature=None,
                               level_temperature=None):
        return {"planck_src_top": self.top[igpt], "planck_src_bottom": self.bottom[igpt],
                "planck_src_sfc": self.sfc[igpt]}

    def compute_sw_optical_properties(self, p, T, mol, igpt, vmr=None, *a, **k):
        return {n: v[igpt] for n, v in self.sw.items()}


def solve_with_oracle_optics(tau_lw, ssa_lw, g_lw, lev, sfc, tau_sw, ssa_sw, g_sw, toa, emis, alb_dir,
                             alb_dif, mu0, dtype):
    """Run production solve_lw/solve_sw. Optics arrays are (g, col, lay)."""
    opt = OracleOptics(dtype, dict(optical_depth=tau_lw, ssa=ssa_lw, asymmetry_factor=g_lw),
                       dict(optical_depth=tau_sw, ssa=ssa_sw, asymmetry_factor=g_sw), lev, sfc)
    ncol, nz = tau_lw.shape[1], tau_lw.shape[2] + 2
    z = jnp.zeros((ncol, 1, nz), dtype)
    col = lambda a: jnp.asarray(a, dtype).reshape(ncol, 1)  # noqa: E731
    st = atmospheric_state.AtmosphericState(
        sfc_emis=col(emis), sfc_alb=col(alb_dif), sfc_alb_dir=col(alb_dir),
        zenith=jnp.arccos(jnp.asarray(mu0, dtype)).reshape(ncol, 1, 1), irrad=1.0, vmr=None,
        toa_flux_lw=0.0)
    kw = dict(use_scan=True, gpoint_batch_size=0, gpoint_checkpoint=False)
    lw = TS.solve_lw(z, z, z, opt, st, None, jnp.zeros((ncol, 1), dtype), **kw)
    sw = TS.solve_sw(z, z, z, opt, st, None, solar_fraction_by_gpt=jnp.asarray(toa, dtype), **kw)
    f = lambda d, k: np.asarray(d[k][:, 0, 1:], np.float64)  # noqa: E731
    return f(lw, "flux_up"), f(lw, "flux_down"), f(sw, "flux_up"), f(sw, "flux_down")


def synthetic(oracle, tag, args, n=4096, nlay=12, seed=0):
    """Solver-only sweep, one g-point per column.

    Quarters: random legal optics; thin layers (tau 1e-9..1e-3); conservative
    thick cloud over a bright surface (where 1 - R A is smallest); the direct-beam
    resonance k mu0 = 1.
    """
    r = np.random.default_rng(seed)
    q = n // 4
    tau = 10.0 ** r.uniform(-10, 3, (n, nlay))
    ssa = np.select([r.random((n, nlay)) < 0.2, r.random((n, nlay)) < 0.2],
                    [0.0, 1.0], 1.0 - 10.0 ** r.uniform(-8, 0, (n, nlay)))
    g = r.uniform(-0.9, 0.99, (n, nlay))
    lev = np.cumsum(10.0 ** r.uniform(-8, 1, (n, nlay + 1)), axis=1)[:, ::-1].copy()
    emis, alb_dir, alb_dif = r.uniform(0.6, 1, n), r.uniform(0, 0.9, n), r.uniform(0, 0.9, n)
    sfc = lev[:, 0] * r.uniform(0.9, 1.1, n)
    mu0 = 10.0 ** r.uniform(-4, 0, n)
    tau[q:2 * q] = 10.0 ** r.uniform(-9, -3, (q, nlay))
    tau[2 * q:3 * q] = 10.0 ** r.uniform(1, 3, (q, nlay))
    ssa[2 * q:3 * q], g[2 * q:3 * q] = 1.0, r.uniform(0.0, 0.9, (q, nlay))
    alb_dir[2 * q:3 * q], alb_dif[2 * q:3 * q], emis[2 * q:3 * q] = 0.99, 0.99, 0.6
    s3 = slice(3 * q, n)
    g[s3], ssa[s3], tau[s3] = 0.5, 0.6, 10.0 ** r.uniform(-3, 1, (n - 3 * q, nlay))
    g1, g2 = (8.0 - 0.6 * 6.5) * 0.25, 3.0 * 0.6 * 0.5 * 0.25
    mu0[s3] = (1.0 + r.uniform(-1e-6, 1e-6, n - 3 * q)) / np.sqrt((g1 - g2) * (g1 + g2))
    inc = np.full(n, 100.0)
    with tempfile.TemporaryDirectory(dir=args.out) as d:
        Path(d, "synth_dims.txt").write_text(f"{n} {nlay}\n")
        write_f(Path(d, "synth.bin"), [tau, ssa, g, lev, emis, sfc, mu0, alb_dir, alb_dif, inc])
        Path(d, "dims.txt").write_text("0 0 0\n")
        run = subprocess.run([oracle, d, "x", "x", "x", "x", "2"], capture_output=True, text=True)
        assert run.returncode == 0 and "ORACLE_SYNTH_DONE" in run.stdout, run.stdout + run.stderr
        rd = Reader(Path(d, "synth_out.bin"))
        lu, ld = rd.take(n, nlay + 1), rd.take(n, nlay + 1)
        su, sd, _ = rd.take(n, nlay + 1), rd.take(n, nlay + 1), rd.take(n, nlay + 1)
        rd.done()
    e = lambda a: a[None]  # noqa: E731
    for dt, nm in ((jnp.float64, "f64"), (jnp.float32, "f32")):
        o = solve_with_oracle_optics(e(tau), e(ssa), e(g), e(lev), sfc[None], e(tau), e(ssa), e(g),
                                     np.array([100.0]), emis, alb_dir, alb_dif, mu0, dt)
        for key, ours, ref in zip(("lw_up", "lw_dn", "sw_up", "sw_dn"), o, (lu, ld, su, sd)):
            cmp(f"solver/synth/{tag}/{nm}/{key}", ours, ref, 1e-6)
        for lab, sl in (("random", slice(0, q)), ("thin", slice(q, 2 * q)),
                        ("conservative", slice(2 * q, 3 * q)), ("resonance", s3)):
            for key, ours, ref in zip(("lw_up", "lw_dn", "sw_up", "sw_dn"), o, (lu, ld, su, sd)):
                cmp(f"solver/synth/{tag}/{nm}/{lab}/{key}", ours[sl], ref[sl], 1e-6)


def heating(up, dn, p_half_sfc_first):
    net = up - dn
    dp = np.abs(np.diff(p_half_sfc_first, axis=1))
    return -constants.g * np.diff(net, axis=1) / dp / constants.c_pd * SECONDS_PER_DAY


def audit_case(name, args, p_top, overhead, data):
    print(f"\n=== case {name}: p_top target {p_top} Pa, overhead layer {overhead}, tables {args.tables}")
    c, ghg = columns(args.data, p_top)
    s0 = float(np.mean(c["tsi"]))
    cloud = add_clouds(c)
    cap, cap_c = {}, {}
    ours = {("clear", "f64"): run_ours(c, ghg, None, overhead, False, s0, cap),
            ("cloud", "f64"): run_ours(c, ghg, cloud, overhead, False, s0, cap_c),
            ("clear", "f32"): run_ours(c, ghg, None, overhead, True, s0),
            ("cloud", "f32"): run_ours(c, ghg, cloud, overhead, True, s0)}
    up = tuple(convert_cloud_table(Path(args.data) / f"rrtmgp-clouds-{b}-bnd.nc",
                                   Path(args.out) / f"converted_{b}.nc") for b in ("lw", "sw"))
    ours[("cloud_upstream_tables", "f64")] = run_ours(c, ghg, cloud, overhead, False, s0, cloud_files=up)
    opt = cap["optics"]
    interior = lambda a: np.asarray(a, np.float64)[:, 0, 1:-1]  # noqa: E731
    play, tlay = interior(cap["p"]), interior(cap["T"])
    ncol, nlay = play.shape
    col_dry = interior(cap["mol"]) / 1.0e4  # molecules/m2 -> molecules/cm2
    ph = c["p_half"]
    if overhead:
        ph = np.concatenate([np.minimum(ph[:, :1], 1.0), ph], axis=1)
    plev = ph[:, ::-1]
    tlev = np.asarray(cap["kw"]["level_temperature"], np.float64)[:, 0, :]
    tsfc = np.asarray(cap["sfcT"], np.float64).reshape(-1)
    print(f"ncol {ncol} nlay {nlay} p {play.min():.3f}..{play.max():.1f} T {tlay.min():.1f}..{tlay.max():.1f}"
          f" dtype table {opt.gas_optics_lw.kmajor.dtype} molecules {cap['mol'].dtype}")
    gases = {}
    for table in (opt.gas_optics_lw, opt.gas_optics_sw):
        for gname in (str(x) for x in table.idx_gases):
            if gname in ("dry_air", "h2o_frgn", "h2o_self") or gname in gases:
                continue
            if gname in cap["vmr"]:
                gases[gname] = interior(cap["vmr"][gname])
            elif gname in opt.vmr_lib.global_means:
                gases[gname] = np.full(play.shape, float(opt.vmr_lib.global_means[gname]))
    print("gases passed:", {k: float(np.mean(v)) for k, v in gases.items()})
    ngl, ngs = opt.n_gpt_lw, opt.n_gpt_sw
    solar = np.asarray(opt.solar_fraction_by_gpt, np.float64)
    toa = np.outer(np.ones(ncol), s0 * solar)
    mu0, emis, alb = c["mu0"], c["emis"], c["alb"]
    # Cloud files always upstream's: the current reader rejects our older format;
    # the tables stage compares the two contents.
    gas_dir = OURS_DATA if args.tables == "ours" else Path(args.data)
    lwg, swg = gas_dir / "rrtmgp-gas-lw-g128.nc", gas_dir / "rrtmgp-gas-sw-g112.nc"
    lwc, swc = Path(args.data) / "rrtmgp-clouds-lw-bnd.nc", Path(args.data) / "rrtmgp-clouds-sw-bnd.nc"
    with tempfile.TemporaryDirectory(dir=args.out) as d:
        Path(d, "dims.txt").write_text(f"{ncol} {nlay} {len(gases)}\n" + "\n".join(gases) + "\n")
        write_f(Path(d, "in.bin"), [play, plev, tlay, tlev, tsfc, col_dry, np.stack(list(gases.values()), -1),
                                    emis, mu0, alb, alb, toa])
        k = cap_c["kw"]
        hw = interior
        write_f(Path(d, "cloud.bin"), [
            hw(k["cloud_path_liq"]) * 1e3, np.clip(hw(k["cloud_r_eff_liq"]) * 1e6, 2.5, 21.5),
            hw(k["cloud_path_ice"]) * 1e3,
            np.clip(hw(k["cloud_r_eff_ice"]) * 1e6 * cloud_optics.RHO_ICE_BULK_CAM_MG2
                    / constants.rho_ice * 2.0, 10.0, 180.0)])
        run = subprocess.run([args.oracle, d, str(lwg), str(swg), str(lwc), str(swc), "2"],
                             capture_output=True, text=True)
        print(run.stdout[-600:], run.stderr[-300:])
        assert run.returncode == 0 and "ORACLE_DONE" in run.stdout, "oracle failed"
        L = Reader(Path(d, "out_lw.bin"))
        lev_shape, lay_shape = (ncol, nlay + 1, ngl), (ncol, nlay, ngl)
        tau, lay, lev, sfc = L.take(*lay_shape), L.take(*lay_shape), L.take(*lev_shape), L.take(ncol, ngl)
        flw = {}
        for key in ("2str", "noscat1", "noscat3", "2str_own_tlev"):
            flw[key] = (L.take(ncol, nlay + 1), L.take(ncol, nlay + 1))
        tau_own_coldry = L.take(*lay_shape)
        flw["2str_own_coldry"] = (L.take(ncol, nlay + 1), L.take(ncol, nlay + 1))
        L.done()
        S = Reader(Path(d, "out_sw.bin"))
        sw_shape = (ncol, nlay, ngs)
        tau_s, ssa_s, g_s, toa_src = S.take(*sw_shape), S.take(*sw_shape), S.take(*sw_shape), S.take(ncol, ngs)
        fsw = {key: (S.take(ncol, nlay + 1), S.take(ncol, nlay + 1), S.take(ncol, nlay + 1))
               for key in ("ours_toa", "own_toa")}
        S.done()
        C = Reader(Path(d, "out_cld.bin"))
        flw_c = (C.take(ncol, nlay + 1), C.take(ncol, nlay + 1))
        flw_c_noscat = (C.take(ncol, nlay + 1), C.take(ncol, nlay + 1))
        fsw_c = (C.take(ncol, nlay + 1), C.take(ncol, nlay + 1), C.take(ncol, nlay + 1))
        C.done()

    # ---- optics, same state ----
    lwf = TS.compute_lw_optical_field(cap["p"], cap["T"], cap["mol"], opt, cap["sfcT"], cap["vmr"],
                                      level_temperature=cap["kw"]["level_temperature"])
    swf = TS.compute_sw_optical_field(cap["p"], cap["T"], cap["mol"], opt, cap["vmr"])
    to_cl = lambda a: np.moveaxis(np.asarray(a, np.float64)[:, :, 0, 1:-1], 0, -1)  # noqa: E731
    cmp(f"optics/{name}/lw_tau", to_cl(lwf["abs_optical_depth"]), tau, 1e-30)
    cmp(f"optics/{name}/lay_source", to_cl(lwf["planck_src"]), lay, 1e-30)
    comb = jax.vmap(lambda t, b: MTS.lw_combine_sources({"planck_src_top": t, "planck_src_bottom": b}))(
        lwf["planck_src_top"], lwf["planck_src_bottom"])["planck_src_top"]
    lev_ours = np.moveaxis(np.asarray(comb, np.float64)[:, :, 0, :-1], 0, -1)
    cmp(f"optics/{name}/lev_source", lev_ours, lev, 1e-30)
    binds = lev ** 2 < MTS._EPSILON
    print(f"source floor binds on {binds.mean():.3%} of (col, level, g-point)")
    cmp(f"optics/{name}/lev_source_unfloored_cells", lev_ours[~binds], lev[~binds], 1e-30)
    cmp(f"optics/{name}/sfc_source", np.moveaxis(np.asarray(lwf["planck_src_sfc"], np.float64)[:, :, 0], 0, -1),
        sfc, 1e-30)
    cmp(f"optics/{name}/sw_tau", to_cl(swf["optical_depth"]), tau_s, 1e-30)
    cmp(f"optics/{name}/sw_ssa", to_cl(swf["ssa"]), ssa_s, 1e-30)
    cmp(f"optics/{name}/solar_weights", solar, toa_src[0] / np.sum(toa_src[0]), 1e-30)
    print(f"oracle sum(toa_src) {np.sum(toa_src[0]):.6f}")
    cmp(f"optics/{name}/tau_col_dry_ours_vs_oracle_default", tau, tau_own_coldry, 1e-30)

    # ---- solver alone, oracle optics ----
    tr = lambda a: np.moveaxis(a, -1, 0)  # noqa: E731
    zl = np.zeros_like(tr(tau))
    for dt, nm in ((jnp.float64, "f64"), (jnp.float32, "f32")):
        lu, ld, su, sd = solve_with_oracle_optics(
            tr(tau), zl, zl, tr(lev), tr(sfc), tr(tau_s), tr(ssa_s), tr(g_s), toa[0], emis, alb, alb, mu0, dt)
        cmp(f"solver/{name}/{nm}/lw_up", lu, flw["2str"][0], 1e-6)
        cmp(f"solver/{name}/{nm}/lw_dn", ld, flw["2str"][1], 1e-6)
        cmp(f"solver/{name}/{nm}/sw_up", su, fsw["ours_toa"][0], 1e-6)
        cmp(f"solver/{name}/{nm}/sw_dn", sd, fsw["ours_toa"][1], 1e-6)

    # ---- end to end, production entry ----
    keep = (lambda a: np.concatenate([a[:, :1], a[:, 2:]], axis=1)) if overhead else (lambda a: a)
    toa_first = lambda a: keep(a[:, ::-1])  # noqa: E731
    hr_ph = plev
    sun = mu0 >= 0.01
    print(f"columns with mu0 >= 0.01: {sun.sum()} of {ncol}")
    for sky, ref_lw, ref_sw in (("clear", flw["2str"], fsw["ours_toa"]), ("cloud", flw_c, fsw_c),
                                ("cloud_upstream_tables", flw_c, fsw_c)):
        hr_lw, hr_sw = heating(*ref_lw, hr_ph), heating(ref_sw[0], ref_sw[1], hr_ph)
        for nm in ("f64", "f32"):
            if (sky, nm) not in ours:
                continue
            o = ours[(sky, nm)]
            cmp(f"e2e/{name}/{sky}/{nm}/sw_up/mu>=0.01", np.asarray(o.sw_flux_up)[sun], toa_first(ref_sw[0])[sun], 1e-6)
            cmp(f"e2e/{name}/{sky}/{nm}/sw_dn/mu>=0.01", np.asarray(o.sw_flux_down)[sun], toa_first(ref_sw[1])[sun], 1e-6)
            cmp(f"e2e/{name}/{sky}/{nm}/sw_hr_Kday/mu>=0.01", (np.asarray(o.sw_heating_rate) * SECONDS_PER_DAY)[sun],
                hr_sw[:, ::-1][:, (1 if overhead else 0):][sun])
            cmp(f"e2e/{name}/{sky}/{nm}/lw_up", o.lw_flux_up, toa_first(ref_lw[0]), 1e-6)
            cmp(f"e2e/{name}/{sky}/{nm}/lw_dn", o.lw_flux_down, toa_first(ref_lw[1]), 1e-6)
            cmp(f"e2e/{name}/{sky}/{nm}/sw_up", o.sw_flux_up, toa_first(ref_sw[0]), 1e-6)
            cmp(f"e2e/{name}/{sky}/{nm}/sw_dn", o.sw_flux_down, toa_first(ref_sw[1]), 1e-6)
            drop = 1 if overhead else 0
            cmp(f"e2e/{name}/{sky}/{nm}/lw_hr_Kday", np.asarray(o.lw_heating_rate) * SECONDS_PER_DAY,
                hr_lw[:, ::-1][:, drop:])
            cmp(f"e2e/{name}/{sky}/{nm}/sw_hr_Kday", np.asarray(o.sw_heating_rate) * SECONDS_PER_DAY,
                hr_sw[:, ::-1][:, drop:])

    # ---- what our choice of solver and inputs costs, all oracle runs ----
    top = -1
    for key in ("noscat1", "noscat3", "2str_own_tlev", "2str_own_coldry"):
        cmp(f"oracle_choice/{name}/{key}/olr_vs_2str", flw[key][0][:, top], flw["2str"][0][:, top])
        cmp(f"oracle_choice/{name}/{key}/sfc_dn_vs_2str", flw[key][1][:, 0], flw["2str"][1][:, 0])
        drop = 1 if overhead else 0  # the overhead layer's heating is discarded
        cmp(f"oracle_choice/{name}/{key}/lw_hr_vs_2str", heating(*flw[key], hr_ph)[:, ::-1][:, drop:],
            heating(*flw["2str"], hr_ph)[:, ::-1][:, drop:])
    cmp(f"oracle_choice/{name}/allsky_noscat1/olr_vs_2str", flw_c_noscat[0][:, top], flw_c[0][:, top])
    cmp(f"oracle_choice/{name}/own_solar_weights/sw_up_vs_ours", fsw["own_toa"][0][:, top],
        fsw["ours_toa"][0][:, top])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", choices=("ours", "upstream"), default="ours")
    ap.add_argument("--min-tau", type=float, default=None, help="override the LW source cutoff")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("devices", jax.devices(), "x64", jax.config.jax_enable_x64)
    compare_tables(args.data)
    synthetic(args.oracle, "base", args)
    if args.min_tau is not None:
        MTS._MIN_TAU_FOR_LW_SRC = args.min_tau
        synthetic(args.oracle, f"min_tau_{args.min_tau:g}", args)
    audit_case("full", args, 0.0, False, args.data)
    audit_case("prod", args, 225.0, True, args.data)
    git = lambda p: subprocess.run(["git", "-c", "safe.directory=*", "-C", str(p), "rev-parse", "HEAD"],  # noqa: E731
                                   capture_output=True, text=True).stdout.strip()
    RESULTS["_provenance"] = dict(
        legoesm=git(Path(__file__).parent), rte_rrtmgp=git(Path(args.oracle).parents[2] / "ref/rte-rrtmgp"),
        rrtmgp_data=git(args.data), tables=args.tables, jax=jax.__version__,
        devices=str(jax.devices()), oracle_note="unpatched kernels, per-g-point two-stream driver")
    Path(args.out, "audit.json").write_text(json.dumps(RESULTS, indent=1))
    print("AUDIT_DONE", len(RESULTS))


if __name__ == "__main__":
    main()
