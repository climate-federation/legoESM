"""D5: full-field score of SI3 jpl1 ridging open water (base vs branch) + small fixture extraction.

usage: extract.py <fixture.npz>   (PYTHONPATH = branch worktree packages; base module loaded from D5/ridging_base.py)
Config: validator-allowed default (retention 1.0).  Open water / ice area / ice volume / ice enthalpy / age are
retention-independent (retention only scales snow & pond inventories), so they are scored vs NEMO (retention 0.5).
Per-cell spread: max over 20 seeds of rel(seed ato_i_out, NEMO stop-run ato_i_out) (tier4/dyn/rdg/closure_arm.py).
"""
import glob
import hashlib
import importlib.util
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/burg/glab/users/pg2328/oracle_audit/tier4/dyn")
sys.path.insert(0, "/burg/glab/users/pg2328/oracle_audit/spread")
import t4lib as L  # noqa: E402
import perturb  # noqa: E402

import jax.numpy as jnp  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
import legoesm.ice.ridging as RB  # noqa: E402  (branch)

set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
spec = importlib.util.spec_from_file_location("ridging_base", "/burg/glab/users/pg2328/oracle_audit/si3dyn_fix/D5/ridging_base.py")
RA = importlib.util.module_from_spec(spec); spec.loader.exec_module(RA)

OUT = sys.argv[1]
RD = "/burg/glab/users/pg2328/oracle_audit/tier4/dyn/ctl_on"
recs = L.load(RD, "rdg")
L.set_guard(L.load(RD, "rhg"))
h0, _, I0, _, _ = next(iter(recs.values()))
dt = float(I0["rDt_ice"][0])
I = lambda k, nk=None, w="in": L.interior(L.assemble(recs, k, w, nk)[0])  # noqa: E731
IN = dict(a_i=I("a_i"), ato_i=I("ato_i"), v_i=I("v_i"), v_s=I("v_s"), oa_i=I("oa_i"), a_ip=I("a_ip"), v_ip=I("v_ip"),
          v_il=I("v_il"), e_s=I("e_s", 3), e_i=I("e_i", 3), szv_i=I("szv_i", 3), divu_i=I("divu_i"),
          delta_i=I("delta_i"))
NO = dict(a_i_out=I("a_i_out", w="out"), ato_i_out=I("ato_i_out", w="out"), v_i_out=I("v_i_out", w="out"),
          oa_i_out=I("oa_i_out", w="out"), e_i_out=I("e_i_out", 3, "out"))


def run(R, d, sl=(...,)):
    st = R.SI3JPL1RidgingState(
        ice_area=jnp.asarray(d["a_i"][sl]), open_water_area=jnp.asarray(d["ato_i"][sl]),
        ice_volume=jnp.asarray(d["v_i"][sl]), snow_volume=jnp.asarray(d["v_s"][sl]),
        age_content=jnp.asarray(d["oa_i"][sl]), pond_area=jnp.asarray(d["a_ip"][sl]),
        pond_volume=jnp.asarray(d["v_ip"][sl]), pond_lid_volume=jnp.asarray(d["v_il"][sl]),
        snow_enthalpy=jnp.asarray(d["e_s"][sl]), ice_enthalpy=jnp.asarray(d["e_i"][sl]),
        ice_salt_content=jnp.asarray(d["szv_i"][sl]))
    return R.apply_si3_jpl1_ridging(st, jnp.asarray(d["divu_i"][sl]), jnp.asarray(d["delta_i"][sl]), dt)


# per-cell NEMO spread of ato_i_out / a_i_out (20 seeds vs stop-run)
b = L.load("/burg/glab/users/pg2328/oracle_audit/tier4/dyn/ctl_stop", "rdg")
sp = {}
for k in ("ato_i_out", "a_i_out"):
    Bx = L.interior(L.assemble(b, k, "out")[0])
    assert np.array_equal(Bx, NO[k]), k
    s = np.zeros_like(Bx)
    for sd in perturb.SEEDS:
        z = np.load(f"/burg/glab/users/pg2328/oracle_audit/tier4/dyn/seeds/out_{sd:02d}.npz")
        pr = {r: (h, mb, inp, o, {**out, k: z[f"rdg:{r}:{k}"]}) for r, (h, mb, inp, o, out) in b.items()}
        s = np.maximum(s, L.rel(L.interior(L.assemble(pr, k, "out")[0]), Bx))
    sp[k] = s

base, base_l = run(RA, IN)
fix, fix_l = run(RB, IN)
mask = np.ones_like(IN["a_i"], bool); mask[:, -1] = False     # G1a: fold row dropped
summary = {}
for nm, (st, ls) in (("base", (base, base_l)), ("branch", (fix, fix_l))):
    row = {}
    for k, pk in (("ato_i_out", "open_water_area"), ("a_i_out", "ice_area"), ("v_i_out", "ice_volume"),
                  ("oa_i_out", "age_content"), ("e_i_out", "ice_enthalpy")):
        p = np.asarray(getattr(st, pk)); r = L.rel(p, NO[k])
        rr = r if r.ndim == 2 else r.max(-1)
        e = dict(max=float(rr[mask].max()), n_gt_1e12=int((rr[mask] > 1e-12).sum()))
        if k in sp:
            e["n_gt_cell_spread"] = int(((rr > sp[k]) & mask).sum())
            e["max_excess_over_spread"] = float((rr - sp[k])[mask].max())
        row[k] = e
    row["iterations_max"] = int(np.asarray(ls.iterations).max())
    summary[nm] = row
    print(nm, json.dumps(row))

# ---- fixture cells ----
a, ato = IN["a_i"], IN["ato_i"]
rel_b = L.rel(np.asarray(base.open_water_area), NO["ato_i_out"])
rel_f = L.rel(np.asarray(fix.open_water_area), NO["ato_i_out"])
ridged = (np.asarray(fix_l.iterations) >= 1) & mask
resid = 1.0 - (ato + a)
rng = np.random.default_rng(0)
pick = []
def take(m, n, key=None):
    idx = np.argwhere(m & ~np.isin(np.arange(m.size).reshape(m.shape), [i * m.shape[1] + j for i, j in pick]))
    if len(idx) == 0:
        return 0
    if key is not None:
        idx = idx[np.argsort(-key[tuple(idx.T)])][:n]
    else:
        idx = idx[rng.choice(len(idx), min(n, len(idx)), replace=False)]
    pick.extend(map(tuple, idx)); return len(idx)
cnt = {}
cnt["deficit_base_miss"] = take(ridged & (rel_b > sp["ato_i_out"]) & (resid > 0), 16, rel_b)
cnt["excess_entry"] = take(ridged & (resid < -1e-10), 4)
cnt["deficit_base_match"] = take(ridged & (resid > 0) & (rel_b <= sp["ato_i_out"]), 6)
exb = (rel_f - sp["ato_i_out"]) * mask
cnt["branch_worst"] = take(exb == exb.max(), 1)
wi, wj = pick[-1]
print("worst cell", wi, wj, "nemo", NO["ato_i_out"][wi, wj], "branch", np.asarray(fix.open_water_area)[wi, wj], "base", np.asarray(base.open_water_area)[wi, wj], "spread", sp["ato_i_out"][wi, wj], "a_in", a[wi, wj], "ato_in", ato[wi, wj], "nemo a_out", NO["a_i_out"][wi, wj])
cnt["low_conc_ridged"] = take(ridged & (a < 0.15) & (resid > 0), 4)
cnt["ice_not_ridged"] = take(mask & (a > 1e-10) & ~ridged, 4)
cnt["no_ice"] = take(mask & (a == 0) & (ato > 0), 2)
ii, jj = np.array(pick).T
fx = {k: v[ii, jj] for k, v in IN.items()} | {k: v[ii, jj] for k, v in NO.items()}
fx |= {"spread_ato_i_out": sp["ato_i_out"][ii, jj], "spread_a_i_out": sp["a_i_out"][ii, jj],
       "base_ato_i_out": np.asarray(base.open_water_area)[ii, jj], "rDt_ice": np.array(dt),
       "global_i": ii, "global_j": jj}
np.savez_compressed(OUT, **fx)
files = sorted(glob.glob(f"{RD}/t4_rdg_record_*.bin"))
md5 = hashlib.md5(b"".join(hashlib.md5(open(f, "rb").read()).digest() for f in files)).hexdigest()
fix_sub, _ = run(RB, fx)
prov = dict(
    source="NEMO 5.0.1 SI3, cfg ORCA1_T4DYN (eORCA1 360x331, 32 ranks, nn_hls=2), build job 11022552, kt=17281=nit000, jpl=1",
    record=f"{RD}/t4_rdg_record_*.bin (32 files, md5-of-md5s {md5}); hook MY_SRC t4_dyn.F90 channel rdg (ice_dyn_rdgrft entry/exit)",
    spread="20-seed 1-ulp per-cell spread, tier4/dyn/seeds/out_*.npz vs ctl_stop (shared perturb.py)",
    harness="tier4/dyn/t4lib.py assemble/interior (ext idx -> interior global 0-based i,j)",
    extractor="/burg/glab/users/pg2328/oracle_audit/si3dyn_fix/D5/extract.py",
    legoesm_branch_sha=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    nemo_retention="rn_fsnwrdg=rn_fsnwrft=rn_fpndrdg=rn_fpndrft=0.5 (snow/pond only; not in fixture outputs)",
    cell_classes=cnt, full_field_summary=summary,
    fixture_subset_branch_max_rel_ato=float(L.rel(np.asarray(fix_sub.open_water_area), fx["ato_i_out"]).max()))
json.dump(prov, open(OUT.replace(".npz", ".json"), "w"), indent=1)
print(json.dumps(cnt)); print("fixture branch max rel ato", prov["fixture_subset_branch_max_rel_ato"])
print("fixture base max rel ato", float(L.rel(fx["base_ato_i_out"], fx["ato_i_out"]).max()))
