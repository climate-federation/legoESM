"""PM fixture extraction (votes V1 a, V7 a) from the NEMO 5.0.1 ORCA1 multi-cycle records (cyc2, cyc3).

windows: per arm one WxW window (inner (W-2M)^2 scored) where the main-branch composition (corrections once) misses
NEMO, all 18 tracers' inputs + moments + metrics + NEMO outputs on the inner cells; the port run on the window with the
periodic refresh must equal the full-domain ORCA run on the inner cells BITWISE (asserted here).
fold: cyc3 top band (all 364 columns, 2 rows below + R interior + 2 fold-halo rows), v_i and a_i only, NEMO's
output moments on the fold-halo rows and the S interior rows they fold; band run == full run bitwise there.
usage: si3_prather_multicycle_extract.py <fixture_dir> <git sha>
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import si3_prather_multicycle_score as S  # noqa: E402
import t4lib as L  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from legoesm.ice.transport import advect_si3_prather_2d, si3_prather_pack_intensives  # noqa: E402
from legoesm.ice.fidelity import nemo_adv2d_testcase_recipe as R  # noqa: E402

FX, SHA = sys.argv[1], sys.argv[2]
W, M = 20, 7
TR = S.TR
KEEP = ["v_i", "v_s", "a_i", "oa_i", *[f"e_s_l{k:02d}" for k in (1, 2, 3)], *[f"e_i_l{k:02d}" for k in (1, 2, 3)],
        "szv_i_l01"]


def run_window(a, sl, fold):
    f2 = a["f2"]
    g = lambda x: jnp.asarray(x[sl])  # noqa: E731
    X = jnp.asarray(a["X"][sl])
    s = R.advect_si3_prather_with_cycle_corrections(
        X, tuple(jnp.asarray(q[sl]) for q in a["MOM"]), g(f2["ato_i"]), g(f2["u_ice"]), g(f2["v_ice"]),
        g(f2["e1e2t"]), jnp.asarray(f2["tmask1"][sl] > 0), a["dt"], dx=g(f2["e1v"]), dy=g(f2["e2u"]),
        ice_step_index=a["h0"]["kt"], nn_fsbc=a["h0"]["nn_fsbc"], halo_width=L.H, subcycles=a["icycle"],
        north_fold=fold, tracer_names=tuple(TR), nlay_i=3, nlay_s=3, ponds_active=False,
        limit_sharpness_scale=math.inf, negative_area_ramp_width=0.0, open_water_floor_sharpness=math.inf)
    return np.asarray(s.intensives), np.asarray(s.open_water), {k: np.asarray(getattr(s.residual_fluxes, k)) for k in S.RES}


fx, meta = {}, dict(oracle="NEMO 5.0.1 SI3 ice_dyn_adv_pra, cfg ORCA1_T4DYN (eORCA1, 32 ranks, nn_hls 2, F-pivot), build job "
                    "11022552, kt 17281 = nit000, rDt_ice 14400 s, nn_fsbc 96, jpl 1, nn_icesal 2, ln_pnd F",
                    arms={}, legoesm_sha=SHA, window=dict(W=W, M=M), extractor="scripts/validate/ice_fidelity/si3_prather_multicycle_extract.py")
for arm in ("cyc2", "cyc3"):
    a = S.load(S.ARMS[arm])
    Xa, _, owa, resa = S.port_after(a)
    Xb, _, owb, resb = S.port_before(a)
    nemo = a["nemo"]
    NI, NJ = Xa.shape[:2]
    inner = np.zeros((NI, NJ), bool); inner[L.H:-L.H, L.H:-L.H] = True
    wet = inner & (a["f2"]["tmask1"] > 0)
    miss = np.zeros((NI, NJ), bool)
    for t in KEEP:
        miss |= L.rel(Xb[..., TR.index(t)], nemo[t]) > 1e-6
    for k in S.RES:
        miss |= np.abs(resb[k] - (nemo[k] - nemo[k + "_in"])) > 1e-9 * np.abs(nemo[k] - nemo[k + "_in"]).max()
    miss &= wet
    best = None
    for i0 in range(L.H, NI - L.H - W, 2):
        for j0 in range(L.H, NJ - 70 - W, 2):
            c = int(miss[i0 + M:i0 + W - M, j0 + M:j0 + W - M].sum())
            if best is None or c > best[0]:
                best = (c, i0, j0)
    c, i0, j0 = best
    sl = (slice(i0, i0 + W), slice(j0, j0 + W)); isl = (slice(M, W - M), slice(M, W - M))
    Xw, oww, resw = run_window(a, sl, "none")
    full_in = (slice(i0 + M, i0 + W - M), slice(j0 + M, j0 + W - M))
    assert np.array_equal(Xw[isl], Xa[full_in]), (arm, float(np.abs(Xw[isl] - Xa[full_in]).max()))
    assert np.array_equal(oww[isl], owa[full_in]) and all(np.array_equal(resw[k][isl], resa[k][full_in]) for k in S.RES)
    p = f"{arm}_"
    fx[p + "X"] = a["X"][sl][..., [TR.index(t) for t in KEEP]]
    fx[p + "MOM"] = np.stack([q[sl][..., [TR.index(t) for t in KEEP]] for q in a["MOM"]], 0)
    for k in ("u_ice", "v_ice", "e1e2t", "e2u", "e1v", "tmask1", "ato_i"):
        fx[p + k] = a["f2"][k][sl]
    fx[p + "nemo_X"] = np.stack([nemo[t][full_in] for t in KEEP], -1)
    fx[p + "nemo_ato_i"] = nemo["ato_i"][full_in]
    for k in S.RES:
        fx[p + "nemo_" + k] = (nemo[k] - nemo[k + "_in"])[full_in]
    fx[p + "scalars"] = np.array([a["dt"], a["h0"]["kt"], a["h0"]["nn_fsbc"], a["icycle"], a["cfl"]])
    meta["arms"][arm] = dict(record=S.ARMS[arm], icycle=a["icycle"], restart_delay_cflice=a["cfl"],
                             window_ext_origin=[i0, j0], inner_cells_main_misses=c,
                             windowed_equals_full_orca_run="bitwise")
    print(arm, "icycle", a["icycle"], "window", i0, j0, "inner cells main misses", c, flush=True)

# --- fold band (cyc3) ---
a = S.load(S.ARMS["cyc3"])
NI, NJ = a["X"].shape[:2]
R_INT, S_ROWS = 12, 4
j_lo = NJ - L.H - R_INT - L.H          # 2 rows below act as the band's south halo
band = (slice(None), slice(j_lo, NJ))
two = [TR.index("v_i"), TR.index("a_i")]


def adv2(sl, fold):
    f2 = a["f2"]
    area = jnp.asarray(f2["e1e2t"][sl]); wet = jnp.asarray(f2["tmask1"][sl] > 0)
    x = jnp.asarray(a["X"][sl][..., two])
    c, m, _ = advect_si3_prather_2d(
        si3_prather_pack_intensives(x, area), tuple(jnp.asarray(q[sl][..., two]) for q in a["MOM"]),
        jnp.asarray(f2["u_ice"][sl]), jnp.asarray(f2["v_ice"][sl]), area, wet, a["dt"], dx=jnp.asarray(f2["e1v"][sl]),
        dy=jnp.asarray(f2["e2u"][sl]), ice_step_index=a["h0"]["kt"], nn_fsbc=a["h0"]["nn_fsbc"], halo_width=L.H,
        ice_volume_index=0, concentration_index=1, subcycles=a["icycle"], north_fold=fold,
        limit_sharpness_scale=math.inf)
    return np.stack([np.asarray(q) for q in m], 0)


full = adv2((slice(None), slice(None)), "orca_f_pivot")[:, :, j_lo:]
bnd = adv2(band, "orca_f_pivot")
top = slice(-L.H - S_ROWS, None)           # S interior rows + 2 fold-halo rows
assert np.array_equal(bnd[:, L.H:-L.H, top], full[:, L.H:-L.H, top]), float(np.abs(bnd - full)[:, L.H:-L.H, top].max())
fx["fold_X"] = a["X"][band][..., two]
fx["fold_MOM"] = np.stack([q[band][..., two] for q in a["MOM"]], 0)
for k in ("u_ice", "v_ice", "e1e2t", "e2u", "e1v", "tmask1"):
    fx["fold_" + k] = a["f2"][k][band]
nm = []
for m in ("sx", "sy", "sxx", "syy", "sxy"):
    nm.append(np.stack([a["G"](m + "ice_out", "out")[band], a["G"](m + "a_out", "out")[band]], -1))
fx["fold_nemo_MOM"] = np.stack(nm, 0)
fx["fold_scalars"] = np.array([a["dt"], a["h0"]["kt"], a["h0"]["nn_fsbc"], a["icycle"], S_ROWS])
meta["fold"] = dict(record=S.ARMS["cyc3"], icycle=a["icycle"], rows=f"global extended rows {j_lo}..{NJ - 1}",
                    scored="2 fold-halo rows + 4 interior rows below them, interior columns",
                    band_equals_full_run="bitwise on scored rows")
import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402

_HERE = Path(__file__).resolve().parent
meta["producer_scripts"] = {"extract": "scripts/validate/ice_fidelity/si3_prather_multicycle_extract.py",
                            "score": "scripts/validate/ice_fidelity/si3_prather_multicycle_score.py"}
meta["producer_sha256"] = {k: hashlib.sha256((_HERE / Path(v).name).read_bytes()).hexdigest()
                           for k, v in meta["producer_scripts"].items()}
np.savez_compressed(f"{FX}/prather_multicycle_orca1.npz", **fx)
json.dump(meta, open(f"{FX}/prather_multicycle_orca1.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
