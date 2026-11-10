"""D5 smooth-default budget on the ORCA1 tier-4 rdg record (branch: default sharpness 100 vs math.inf hard),
plus the stored smooth outputs appended to the committed fixture.  usage: budget.py <fixture.npz>"""
import json, math, sys
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/burg/glab/users/pg2328/oracle_audit/tier4/dyn")
import t4lib as L
from legoesm.core.precision import PrecisionPolicy, set_policy
import legoesm.ice.ridging as R
set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
RD = "/burg/glab/users/pg2328/oracle_audit/tier4/dyn/ctl_on"
recs = L.load(RD, "rdg"); rh = L.load(RD, "rhg"); L.set_guard(rh)
I = lambda k, nk=None, w="in", r=recs: L.interior(L.assemble(r, k, w, nk)[0])
area = I("e1e2t", r=rh)
dt = float(next(iter(recs.values()))[2]["rDt_ice"][0])
KEYS = (("a_i",), ("ato_i",), ("v_i",), ("v_s",), ("oa_i",), ("a_ip",), ("v_ip",), ("v_il",), ("e_s", 3), ("e_i", 3), ("szv_i", 3))
def run(src, s, sl=(...,)):
    st = R.SI3JPL1RidgingState(*(jnp.asarray(src(k, *a)[sl]) for k, *a in KEYS))
    return R.apply_si3_jpl1_ridging(st, jnp.asarray(src("divu_i")[sl]), jnp.asarray(src("delta_i")[sl]), dt,
                                    config=R.SI3JPL1RidgingConfig(open_water_limit_sharpness=s))
m = np.ones_like(area, bool); m[:, -1] = False
H, _ = run(I, math.inf); S, _ = run(I, R.SI3JPL1RidgingConfig().open_water_limit_sharpness)
out = {}
for nm, f in (("open_water", lambda o: o.open_water_area), ("ice_area", lambda o: o.ice_area),
              ("ice_volume", lambda o: o.ice_volume), ("snow_volume", lambda o: o.snow_volume),
              ("ice_enthalpy", lambda o: o.ice_enthalpy.sum(-1)), ("snow_enthalpy", lambda o: o.snow_enthalpy.sum(-1)),
              ("ice_salt", lambda o: o.ice_salt_content.sum(-1))):
    h, s = np.asarray(f(H)), np.asarray(f(S)); d = (s - h) * m
    tot = float(np.abs(np.asarray(f(H))[m] * area[m]).sum())
    out[nm] = dict(n_changed=int((d != 0).sum()), max_abs=float(np.abs(d).max()),
                   net_area_weighted=float((d * area).sum()), abs_area_weighted=float(np.abs(d * area).sum()),
                   rel_to_global_hard_inventory=float(np.abs(d * area).sum() / tot) if tot > 0 else 0.0)
clo = lambda o: (1.0 - np.asarray(o.open_water_area) - np.asarray(o.ice_area))
dc = (clo(S) - clo(H)) * m
out["closure_1_minus_ato_minus_a"] = dict(n_changed=int((np.abs(dc) > 1e-10).sum()), max_abs=float(np.abs(dc).max()),
                                          net_area_weighted_m2=float((dc * area).sum()))
out["ocean_area_m2_interior"] = float((area * m).sum())
print(json.dumps(out, indent=1))
FX = sys.argv[1]; fx = dict(np.load(FX)); fxsrc = lambda k, *a: fx[k]
S2, _ = run(fxsrc, R.SI3JPL1RidgingConfig().open_water_limit_sharpness)
fx |= {"smooth_ato_i_out": np.asarray(S2.open_water_area), "smooth_a_i_out": np.asarray(S2.ice_area),
       "smooth_v_i_out": np.asarray(S2.ice_volume), "smooth_e_i_out": np.asarray(S2.ice_enthalpy)}
np.savez_compressed(FX, **fx)
j = json.load(open(FX.replace(".npz", ".json")))
j["smooth_outputs"] = "smooth_* keys: branch port at default open_water_limit_sharpness=100 on the fixture cells (stored gap vs NEMO), budget.py"
j["smooth_budget_full_field"] = out
json.dump(j, open(FX.replace(".npz", ".json"), "w"), indent=1)
print("fixture smooth max |gap| open water", float(np.abs(fx["smooth_ato_i_out"] - fx["ato_i_out"]).max()))
