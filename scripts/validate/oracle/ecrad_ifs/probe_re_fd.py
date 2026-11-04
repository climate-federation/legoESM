"""Diagnosis probe (no code change): FD vs AD of ecRad-core outputs w.r.t. the
effective radii, stage by stage, on the 8-column fixture.

Direction: relative, d = re (each of re_liquid / re_ice separately), so x(h) =
re*(1+h).  Per stage and output: F = sum(W*out) with fixed random weights W
(seed 0, |W| ~ U(0.5,1.5), sign random).  AD = jax.jvp along d.  FD = central
(F(h)-F(-h))/2h.  Also D3 = (F(2h)-2F(h)+2F(-h)-F(-2h))/(2h^3) and the predicted
smooth truncation h^2*D3/6 (mechanism 2) next to the measured |FD-AD|.

Stages: cloud optics (6 arrays), sub-column cloud od (od_scaling x od_cloud(band
of g), LW and SW; od_scaling fixed = generator output, independent of re), LW
solver, SW solver (fluxes; gas optics and generator fixed), and the two-stream
kernels ELEMENTWISE on cloudy layers (localisation: worst elements printed with
od, ssa, g, distance to the od 1e-3 thin/thick band, and k^2).
usage: probe_re_fd.py <sharpness>   (100 or inf)

Provenance (vote C10): copied unchanged from /burg/glab/users/pg2328/oracle_audit/ifs_port/ecrad/agents/end_to_end/probe_re_fd.py (end-to-end agent's
diagnosis of the FD-vs-AD finding of vote C7; sha256 prefix ec174b7a1a5b0321); only this
paragraph is added.
"""
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from legoesm.atmosphere.physics.radiation.ecrad_ifs import cloud_optics as co  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ecrad  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_generator as mg  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_solvers as ms  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import rrtmg_lw, rrtmg_sw  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import rrtmg_prepare as prep  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import two_stream as ts  # noqa: E402
from tests.atmosphere import test_ecrad_ifs_end_to_end_oracle as t  # noqa: E402

S = float(sys.argv[1])
HS = (3e-5, 1e-5, 3e-6, 1e-6)
THR = ecrad._CLOUD_FRACTION_THRESHOLD

z = t.load(t.FIXTURE)
inp = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in t.inputs(z).items()}
ulw, _, _ = t.replay(z, t.inputs(z), "lw")
usw, _, _ = t.replay(z, t.inputs(z), "sw")

# fixed (re-independent) pieces, exactly as ecrad.radiation wires them
lw_alb = ecrad.lw_albedo_from_emissivity(inp["lw_emissivity"])
B, I = ecrad.SW_ALBEDO_WAVELENGTH_BOUND, ecrad.I_SW_ALBEDO_INDEX
alb_dif = ms.sw_albedo_to_gpoints(inp["sw_albedo"], B, I)
alb_dir = ms.sw_albedo_to_gpoints(inp["sw_albedo_direct"], B, I)
pfl, tfl = prep.full_levels(inp["pressure_hl"], inp["temperature_hl"])
gases = prep.rrtm_prepare_gases(inp["pressure_hl"], pfl, inp["temperature_hl"], tfl, inp["gas_mmr"], ecrad._GRAVITY)
lw = rrtmg_lw.gas_optics_lw(gases, inp["temperature_hl"], inp["skin_temperature"], lw_alb, ecrad._MIN_GAS_OD_LW, S)
od_sw, ssa_sw, inc_sw = rrtmg_sw.gas_optics_sw(gases.pavel, gases.tavel, gases.coldry, gases.wkl, inp["cos_sza"],
                                               inp["solar_irradiance"], ecrad._MIN_GAS_OD_SW, S)
frac = co.crop_cloud_fraction(inp["cloud_fraction"], inp["q_liquid"], inp["q_ice"], S)[0]


def gen(u):
    sc, tcc = mg.cloud_generator(frac, inp["overlap_param"], inp["fractional_std"], *map(jnp.asarray, u),
                                 decorrelation_scaling=ecrad._CLOUD_INHOM_DECORR_SCALING, frac_threshold=THR,
                                 sharpness=S)
    return jnp.swapaxes(sc, 1, 2), tcc


sc_lw, tcc_lw = gen(ulw)
sc_sw, tcc_sw = gen(usw)
band_lw, band_sw = ms.band_from_g_lw(), ms.band_from_g_sw()


def cloud(rl, ri):
    return co.cloud_optics(inp["pressure_hl"], inp["cloud_fraction"], inp["q_liquid"], inp["q_ice"], rl, ri, S)


def stages(rl, ri):
    c = cloud(rl, ri)
    out = {f"cloud.{n}": getattr(c, n) for n in ("od_lw", "ssa_lw", "g_lw", "od_sw", "ssa_sw", "g_sw")}
    out["subcol.od_lw"] = sc_lw * c.od_lw[..., band_lw]
    out["subcol.od_sw"] = sc_sw * c.od_sw[..., band_sw]
    fl = ms.solver_mcica_lw(lw.od_lw, c.od_lw, c.ssa_lw, c.g_lw, lw.planck_hl, lw.lw_emission, lw_alb,
                            c.cloud_fraction, sc_lw, tcc_lw, cloud_fraction_threshold=THR, sharpness=S)
    fs = ms.solver_mcica_sw(od_sw, ssa_sw, jnp.zeros_like(od_sw), c.od_sw, c.ssa_sw, c.g_sw, alb_dir, alb_dif,
                            inc_sw, inp["cos_sza"], c.cloud_fraction, sc_sw, tcc_sw,
                            cloud_fraction_threshold=THR, sharpness=S)
    for k in ("flux_up_lw", "flux_dn_lw", "lw_derivative"):
        out["lwsolver." + k] = fl[k]
    for k in ("flux_up_sw", "flux_dn_sw", "flux_dn_direct_sw"):
        out["swsolver." + k] = fs[k]
    return out


def kernels(rl, ri):
    """Cloudy-layer two-stream kernel outputs elementwise (same combination as the solvers)."""
    c = cloud(rl, ri)
    o = {}
    # solver dead-branch masks (mcica_solvers._solver_*_column): od_scaling := 0 where
    # the layer or column weight is exactly 0 (and, SW, where the day weight is 0)
    wl = ts.step(c.cloud_fraction, THR, THR, S, strict=False)
    live_lw = ((wl > 0) & (ts.step(tcc_lw, THR, THR, S, strict=False) > 0)[:, None])[..., None]
    w_day = (inp["cos_sza"] > 0.0) if math.isinf(S) else (1.0 + inp["cos_sza"] * S > 0.0)
    live_sw = ((wl > 0) & (ts.step(tcc_sw, THR, THR, S, strict=False) > 0)[:, None] & w_day[:, None])[..., None]
    ocn = jnp.where(live_lw, sc_lw, 0.0) * c.od_lw[..., band_lw]
    odt = lw.od_lw + ocn
    scat = c.ssa_lw[..., band_lw] * ocn
    ssa = jnp.where(odt > 0, scat / jnp.where(odt > 0, odt, 1.0), 0.0)
    g = jnp.where(scat > 0, c.g_lw[..., band_lw] * scat / jnp.where(scat > 0, scat, 1.0), 0.0)
    pt, pb = lw.planck_hl[:, :-1], lw.planck_hl[:, 1:]
    for n, v in zip(("R", "T", "su", "sd"), ts.calc_ref_trans_lw(odt, ssa, g, pt, pb, S)):
        o["lw." + n] = v
    o["lw.od"], o["lw.ssa"], o["lw.g"] = odt, ssa, g
    ocn = jnp.where(live_sw, sc_sw, 0.0) * c.od_sw[..., band_sw]
    odt = od_sw + ocn
    scat = ssa_sw * od_sw + c.ssa_sw[..., band_sw] * ocn
    ssa = jnp.where(odt > 0, scat / jnp.where(odt > 0, odt, 1.0), 0.0)
    gs = (c.g_sw[..., band_sw] * c.ssa_sw[..., band_sw]) * ocn
    g = jnp.where(scat > 0, gs / jnp.where(scat > 0, scat, 1.0), 0.0)
    mu0 = jnp.where(inp["cos_sza"] > 0, inp["cos_sza"], 1.0)[:, None, None]
    for n, v in zip(("Rd", "Td", "Rdir", "Tdd", "Tdir"), ts.calc_ref_trans_sw(mu0, odt, ssa, g, S)):
        o["sw." + n] = v
    o["sw.od"], o["sw.ssa"], o["sw.g"] = odt, ssa, g
    f = 0.75 * g
    g1, g2 = 2.0 - ssa * (1.25 + f), ssa * (0.75 - f)
    o["sw.k2"] = (g1 - g2) * (g1 + g2)          # two_stream._k_exponent argument
    fl = (1.66 * 0.5) * o["lw.ssa"]
    g1, g2 = 1.66 - fl * (1.0 + o["lw.g"]), fl * (1.0 - o["lw.g"])
    o["lw.k2"] = (g1 - g2) * (g1 + g2)
    return o


rng = np.random.default_rng(0)
re0 = {"liq": inp["re_liquid"], "ice": inp["re_ice"]}


def along(fn, which):
    def f(h):
        rl = re0["liq"] * (1.0 + h) if which == "liq" else re0["liq"]
        ri = re0["ice"] * (1.0 + h) if which == "ice" else re0["ice"]
        return fn(rl, ri)
    return f


for which in ("liq", "ice"):
    # ---- stage-level scalar functionals
    f = jax.jit(along(stages, which))
    base = f(0.0)
    W = {k: jnp.asarray(rng.uniform(0.5, 1.5, v.shape) * rng.choice([-1.0, 1.0], v.shape)) for k, v in base.items()}
    F = jax.jit(lambda h: {k: jnp.sum(W[k] * v) for k, v in f(h).items()})
    _, ad = jax.jvp(F, (0.0,), (1.0,))
    vals = {h: (F(h), F(-h), F(2 * h), F(-2 * h)) for h in HS}
    print(f"=== sharpness {S} re_{which}: stage functionals (rel |FD-AD|/|AD|; pred = h^2 D3/6 / |AD|)")
    for k in base:
        a = float(ad[k])
        row = []
        for h in HS:
            p, m, p2, m2 = (float(v[k]) for v in vals[h])
            fd = (p - m) / (2 * h)
            d3 = (p2 - 2 * p + 2 * m - m2) / (2 * h ** 3)
            den = abs(a) if a != 0 else 1.0
            row.append(f"h={h:.0e}: {abs(fd - a) / den:.1e} (pred {h * h * abs(d3) / 6 / den:.1e})")
        print(f"  {k:28s} AD={a:+.6e}  " + "  ".join(row))
    # ---- elementwise kernels: localisation
    g = jax.jit(along(kernels, which))
    k0 = g(0.0)
    _, kad = jax.jvp(g, (0.0,), (1.0,))
    print(f"--- kernels elementwise, re_{which}: elements with |AD| > 1e-3 max|AD| of that output, "
          f"day columns only for SW; err = |FD-AD|/|AD| per element")
    day = np.asarray(inp["cos_sza"]) > 0.0
    for name in ("lw.R", "lw.T", "lw.su", "lw.sd", "sw.Rd", "sw.Td", "sw.Rdir", "sw.Tdd", "sw.Tdir"):
        spec = name.split(".")[0]
        a = np.asarray(kad[name])
        sel = np.abs(a) > 1e-3 * np.max(np.abs(a))
        if spec == "sw":
            sel &= day[:, None, None]
        k2 = np.asarray(k0[f"{spec}.k2"])
        for h in HS:
            fd = (np.asarray(g(h)[name]) - np.asarray(g(-h)[name])) / (2 * h)
            err = np.where(sel, np.abs(fd - a) / np.where(sel, np.abs(a), 1.0), 0.0)
            i = np.unravel_index(np.argmax(err), err.shape)
            od, ssa, gg = (float(np.asarray(k0[f"{spec}.{q}"])[i]) for q in ("od", "ssa", "g"))
            thin = f" (od-1e-3)/1e-3={(od - 1e-3) / 1e-3:+.2e}" if spec == "lw" else ""
            bad = err > 1e-6
            nfl = int((bad & (k2 < 1e-12)).sum())
            print(f"  {name:8s} h={h:.0e} n_sel={int(sel.sum())} max err={err[i]:.1e} at {tuple(int(x) for x in i)} "
                  f"od={od:.4e}{thin} 1-ssa={1.0 - ssa:.3e} g={gg:.4f} k2={k2[i]:.3e} "
                  f"n_err>1e-6={int(bad.sum())} of which k2<1e-12: {nfl}")
    # ---- residual scan of the SW solver functional: jumps vs smooth curvature
    if which == "liq":
        Fs = jax.jit(lambda h: F(h)["swsolver.flux_up_sw"])
        a0 = float(ad["swsolver.flux_up_sw"])
        f0 = float(Fs(0.0))
        hs = np.linspace(-3e-6, 3e-6, 25)
        res = [float(Fs(h)) - f0 - a0 * h for h in hs]
        print("--- scan swsolver.flux_up_sw: F(h)-F(0)-AD*h, h in [-3e-6,3e-6] step 2.5e-7 (|F(0)|=%.3e)" % abs(f0))
        print("   ", " ".join(f"{r:+.2e}" for r in res))
print("PROBE_DONE", math.isinf(S))
