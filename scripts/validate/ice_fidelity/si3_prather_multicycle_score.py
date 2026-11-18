"""PM: Prather multi-cycle row vs NEMO 5.0.1 ORCA1 (tier-4 build, kt 17281).

Arms: ctl_on (icycle 1, tier-4), cyc2 / cyc3 (same step, restart DELAY__r8_cflice set to 1.0 / 2.0 so NEMO's own
CFL branch picks icycle 2 / 3; NEMO code unchanged).  For each arm two port compositions, all limits hard (NEMO):
  before = advect all cycles, then corrections + open water ONCE (the composition on main, D2/tier-4 harness);
  after  = advect_si3_prather_with_cycle_corrections (corrections + open water + ledgers inside every cycle).
Both use north_fold='orca_f_pivot'.  Scores: max rel error vs NEMO on the ice region (entry a_i>0, interior, fold row
dropped) and wet region; the icycle=1 20-seed field spread (tier4 spread.json) is printed as the reference.  Budget:
inner-wet totals of water/heat/salt change + residual*dt for port and NEMO.
usage: si3_prather_multicycle_score.py <out.json>  (branch on PYTHONPATH); importable by the extractor
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, "/burg/glab/users/pg2328/oracle_audit/tier4/dyn")
import t4lib as L  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ice.transport import (advect_si3_prather_2d, si3_prather_open_water,  # noqa: E402
                                   si3_prather_pack_intensives, si3_prather_unpack_intensives)
from legoesm.ice.fidelity import nemo_adv2d_testcase_recipe as R  # noqa: E402


pol = PrecisionPolicy.fp64(transcendentals="libm"); set_policy(pol); assert get_policy() == pol
D = "/burg/glab/users/pg2328/oracle_audit/tier4/dyn"
ARMS = {"cyc1": f"{D}/ctl_on", "cyc2": "/burg/glab/users/pg2328/oracle_audit/si3dyn_fix/PM/cyc2",
        "cyc3": "/burg/glab/users/pg2328/oracle_audit/si3dyn_fix/PM/cyc3"}
SPREAD = json.load(open(f"{D}/spread.json"))
TR = ["v_i", "v_s", "a_i", "oa_i", *[f"e_s_l{k:02d}" for k in (1, 2, 3)], *[f"e_i_l{k:02d}" for k in (1, 2, 3)],
      "szv_i_l01", "szv_i_l02", "szv_i_l03", "a_ip", "v_ip", "v_il"]
MB = {"v_i": "ice", "v_s": "sn", "a_i": "a", "oa_i": "age", "szv_i_l01": "sal"}
NEMO_OUT = {"v_i": "v_i_out", "v_s": "v_s_out", "a_i": "a_i_out", "oa_i": "oa_i_out", "szv_i_l01": "sv_i_out"}
RES = ("wfx_res", "hfx_res", "sfx_res")


def load(rd):
    recs = L.load(rd, "adv"); L.set_guard(L.load(rd, "rhg"))
    h0, _, I0, _, O0 = next(iter(recs.values()))
    G = lambda k, w="in", nk=None: L.assemble(recs, k, w, nk)[0]  # noqa: E731
    a = dict(h0=h0, dt=float(I0["rDt_ice"][0]), icycle=int(O0["cfl_icycle"][1]), cfl=float(O0["cfl_icycle"][0]),
             sc={k: float(I0[k][0]) for k in ("rhoi", "rhos", "rhow")}, G=G)
    f2 = {k: G(k) for k in ("u_ice", "v_ice", "ato_i", "e1e2t", "e2u", "e1v", "tmask1")}
    f3 = {k: G(k) for k in ("v_i", "v_s", "sv_i", "oa_i", "a_i", "a_ip", "v_ip", "v_il")}
    es, ei = G("e_s", nk=3), G("e_i", nk=3)
    zero = np.zeros_like(f2["e1e2t"])
    inten = {"v_i": f3["v_i"], "v_s": f3["v_s"], "a_i": f3["a_i"], "oa_i": f3["oa_i"], "szv_i_l01": f3["sv_i"],
             "szv_i_l02": zero, "szv_i_l03": zero, "a_ip": f3["a_ip"], "v_ip": f3["v_ip"], "v_il": f3["v_il"]}
    for k in range(3):
        inten[f"e_s_l{k + 1:02d}"] = es[..., k]; inten[f"e_i_l{k + 1:02d}"] = ei[..., k]
    assert all(float(np.abs(inten[k]).max()) == 0.0 for k in ("a_ip", "v_ip", "v_il"))
    mom = []
    for m in ("sx", "sy", "sxx", "syy", "sxy"):
        c0, e = G(m + "c0", nk=3), G(m + "e", nk=3)
        mom.append(np.stack([G(m + MB[t]) if t in MB else c0[..., int(t[-2:]) - 1] if t.startswith("e_s_l")
                             else e[..., int(t[-2:]) - 1] if t.startswith("e_i_l") else zero for t in TR], -1))
    nemo = {t: G(v, "out") for t, v in NEMO_OUT.items()}
    eso, eio = G("e_s_out", "out", 3), G("e_i_out", "out", 3)
    for k in range(3):
        nemo[f"e_s_l{k + 1:02d}"] = eso[..., k]; nemo[f"e_i_l{k + 1:02d}"] = eio[..., k]
    nemo["ato_i"] = G("ato_i_out", "out")
    for k in RES:
        nemo[k] = G(k + "_out", "out"); nemo[k + "_in"] = G(k)
    nmom = {}
    for mi, m in enumerate(("sx", "sy", "sxx", "syy", "sxy")):
        c0, e = G(m + "c0_out", "out", 3), G(m + "e_out", "out", 3)
        for t in TR:
            if t in MB:
                nmom[f"{m}{MB[t]}"] = (G(m + MB[t] + "_out", "out"), mi, TR.index(t))
        nmom[f"{m}c0"] = (c0, mi, [TR.index(f"e_s_l{k:02d}") for k in (1, 2, 3)])
        nmom[f"{m}e"] = (e, mi, [TR.index(f"e_i_l{k:02d}") for k in (1, 2, 3)])
    a.update(f2=f2, f3=f3, X=np.stack([inten[k] for k in TR], -1), MOM=tuple(mom), nemo=nemo, nmom=nmom)
    return a


def port_before(a):
    f2, X, dt, n = a["f2"], jnp.asarray(a["X"]), a["dt"], a["icycle"]
    area = jnp.asarray(f2["e1e2t"]); wet = jnp.asarray(f2["tmask1"] > 0)
    c, m, _ = advect_si3_prather_2d(
        si3_prather_pack_intensives(X, area), tuple(jnp.asarray(q) for q in a["MOM"]), jnp.asarray(f2["u_ice"]),
        jnp.asarray(f2["v_ice"]), area, wet, dt, dx=jnp.asarray(f2["e1v"]), dy=jnp.asarray(f2["e2u"]),
        ice_step_index=a["h0"]["kt"], nn_fsbc=a["h0"]["nn_fsbc"], halo_width=L.H, ice_volume_index=TR.index("v_i"),
        concentration_index=TR.index("a_i"), subcycles=n, north_fold="orca_f_pivot", limit_sharpness_scale=math.inf)
    raw = si3_prather_unpack_intensives(c, area, wet)
    cor, res = R.apply_si3_prather_source_corrections(
        X, raw, tracer_names=tuple(TR), nlay_i=3, nlay_s=3, cell_area_m2=1.0, halo_width=L.H, ponds_active=False,
        entry_intensive_contents=X, contents_are_intensive=True, negative_area_ramp_width=0.0, dt_s=dt,
        return_residual_fluxes=True)
    ow = si3_prather_open_water(jnp.asarray(f2["ato_i"]), X[..., TR.index("a_i")], cor[..., TR.index("a_i")],
                                jnp.asarray(f2["u_ice"]), jnp.asarray(f2["v_ice"]), dy_u=jnp.asarray(f2["e2u"]),
                                dx_v=jnp.asarray(f2["e1v"]), cell_area=area, dt=dt / n, floor_sharpness=math.inf)
    return np.asarray(cor), m, np.asarray(ow), {k: np.asarray(getattr(res, k)) for k in RES}


def port_after(a):
    f2, X, dt = a["f2"], jnp.asarray(a["X"]), a["dt"]
    area = jnp.asarray(f2["e1e2t"]); wet = jnp.asarray(f2["tmask1"] > 0)
    s = R.advect_si3_prather_with_cycle_corrections(
        X, tuple(jnp.asarray(q) for q in a["MOM"]), jnp.asarray(f2["ato_i"]), jnp.asarray(f2["u_ice"]),
        jnp.asarray(f2["v_ice"]), area, wet, dt, dx=jnp.asarray(f2["e1v"]), dy=jnp.asarray(f2["e2u"]),
        ice_step_index=a["h0"]["kt"], nn_fsbc=a["h0"]["nn_fsbc"], halo_width=L.H, subcycles=a["icycle"],
        north_fold="orca_f_pivot", tracer_names=tuple(TR), nlay_i=3, nlay_s=3, ponds_active=False,
        limit_sharpness_scale=math.inf, negative_area_ramp_width=0.0, open_water_floor_sharpness=math.inf)
    return (np.asarray(s.intensives), s.moments, np.asarray(s.open_water),
            {k: np.asarray(getattr(s.residual_fluxes, k)) for k in RES})


def budgets(a, X_out, res):
    """Inner-wet totals [kg, J, kg salt-units]: change of stored content + residual*dt (0 if closed, up to fold flux)."""
    sc, dt, area = a["sc"], a["dt"], a["f2"]["e1e2t"]
    inner = np.zeros(area.shape, bool); inner[L.H:-L.H, L.H:-L.H] = True
    m = inner & (a["f2"]["tmask1"] > 0)
    X0 = a["X"]
    def tot(Y):
        w = (Y[..., TR.index("v_i")] * sc["rhoi"] + Y[..., TR.index("v_s")] * sc["rhos"]) * area
        h = -sum(Y[..., TR.index(n)] for n in TR if n.startswith("e_")) * area
        s = Y[..., TR.index("szv_i_l01")] * sc["rhoi"] * area
        return {"water": float(w[m].sum()), "heat": float(h[m].sum()), "salt": float(s[m].sum())}
    t0, t1 = tot(X0), tot(X_out)
    led = {"water": float((res["wfx_res"] * area)[m].sum() * dt), "heat": float((res["hfx_res"] * area)[m].sum() * dt),
           "salt": float((res["sfx_res"] * area)[m].sum() * dt)}
    return {k: dict(stored_change=t1[k] - t0[k], ledger=led[k], residual=t1[k] - t0[k] + led[k],
                    rel=(t1[k] - t0[k] + led[k]) / max(abs(t0[k]), 1e-300)) for k in t0}


def nemo_out_state(a):
    Y = a["X"].copy()
    for t, v in a["nemo"].items():
        if t in TR:
            Y[..., TR.index(t)] = v
    return Y


def main(OUT):
    report = {}
    for arm, rd in ARMS.items():
        a = load(rd)
        at_int, tm_int = L.interior(a["f3"]["a_i"]), L.interior(a["f2"]["tmask1"])
        reg = L.regions(at_int, tm_int, "T")
        row = dict(icycle=a["icycle"], cfl=a["cfl"])
        for side, fn in (("before", port_before), ("after", port_after)):
            Xo, mom, ow, res = fn(a)
            sc = {}
            for t in [*NEMO_OUT, *[f"e_s_l{k:02d}" for k in (1, 2, 3)], *[f"e_i_l{k:02d}" for k in (1, 2, 3)]]:
                r = L.rel(L.interior(Xo[..., TR.index(t)]), L.interior(a["nemo"][t]))
                sc[t] = {nm: float(r[mk].max()) for nm, mk in (("ice", reg["ice"]), ("wet", reg["wet"]))}
            for k, (nv, mi, ti) in a["nmom"].items():
                p = np.asarray(mom[mi])[..., ti]
                r = L.rel(L.interior(p), L.interior(nv))
                sc["mom:" + k] = {nm: float(r[mk].max()) for nm, mk in (("ice", reg["ice"]), ("wet", reg["wet"]))}
            r = L.rel(L.interior(ow), L.interior(a["nemo"]["ato_i"]))
            sc["ato_i"] = {nm: float(r[mk].max()) for nm, mk in (("ice", reg["ice"]), ("wet", reg["wet"]))}
            for k in RES:
                nv = a["nemo"][k] - a["nemo"][k + "_in"]   # NEMO accumulates into the step's ledger
                r = L.rel(L.interior(res[k]), L.interior(nv))
                sc[k] = {nm: float(r[mk].max()) for nm, mk in (("ice", reg["ice"]), ("wet", reg["wet"]))}
                sc[k]["abs_wet"] = float(np.abs(L.interior(res[k]) - L.interior(nv))[reg["wet"]].max())
                sc[k]["nemo_absmax"] = float(np.abs(L.interior(nv))[reg["wet"]].max())
            row[side] = dict(scores=sc, budget=budgets(a, Xo, res))
        nres = {k: a["nemo"][k] - a["nemo"][k + "_in"] for k in RES}
        row["nemo_budget"] = budgets(a, nemo_out_state(a), nres)
        report[arm] = row
        worst = lambda side: max((v["ice"], k) for k, v in row[side]["scores"].items())  # noqa: E731
        print(arm, "icycle", a["icycle"], "worst ice before", worst("before"), "after", worst("after"), flush=True)
    spread_ref = {k: v["ice"]["max"] for k, v in SPREAD.items() if k.startswith("adv:")}
    json.dump(dict(report=report, spread_icycle1_ice_max=spread_ref), open(OUT, "w"), indent=1)
    for arm, row in report.items():
        print("==", arm, "icycle", row["icycle"])
        for k in row["after"]["scores"]:
            b, f = row["before"]["scores"][k], row["after"]["scores"][k]
            print(f"  {k:14s} before ice {b['ice']:.2e} wet {b['wet']:.2e} | after ice {f['ice']:.2e} wet {f['wet']:.2e}")
        for side in ("before", "after", "nemo_budget"):
            bud = row[side]["budget"] if side != "nemo_budget" else row[side]
            print("  budget", side, {k: f"{v['rel']:.2e}" for k, v in bud.items()})


if __name__ == "__main__":
    main(sys.argv[1])
