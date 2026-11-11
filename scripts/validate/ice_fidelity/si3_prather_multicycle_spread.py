"""NEMO 1-ulp spread of the multi-cycle Prather records and the class of the per-cycle port (vote V2 b).

Spread per arm (icycle 2, 3) and field = max over 4 seeds (si3_prather_multicycle_seed.py, shared perturb.py
semantics) of the max relative difference seed-vs-base output on the ice region (tier-4 G1a: interior, fold row
dropped, entry a_i > 0); ledgers absolute on (out - in) over the wet region.  Class per field: A if the per-cycle
port's error <= spread.  With --fixture-json it writes the state/open-water spreads into the fixture json
(``spread_4seed``) for the per-field test bound (vote T5 b).
usage: si3_prather_multicycle_spread.py <out.json> [--fixture-json <path>]   (branch on PYTHONPATH)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import si3_prather_multicycle_score as S  # noqa: E402
import t4lib as L  # noqa: E402

SEEDS_DIR = "/burg/glab/users/pg2328/oracle_audit/si3dyn_fix/PM/seeds"
SEEDS = (0, 1, 2, 3)
MOM_NAMES = ("sx", "sy", "sxx", "syy", "sxy")
BULK = {"ice": "v_i", "sn": "v_s", "a": "a_i", "sal": "szv_i_l01", "age": "oa_i"}
KEEP = ["v_i", "v_s", "a_i", "oa_i", *[f"e_s_l{k:02d}" for k in (1, 2, 3)], *[f"e_i_l{k:02d}" for k in (1, 2, 3)],
        "szv_i_l01"]
OUTKEY = {"v_i": ("v_i_out", None, None), "v_s": ("v_s_out", None, None), "a_i": ("a_i_out", None, None),
          "oa_i": ("oa_i_out", None, None), "szv_i_l01": ("sv_i_out", None, None), "ato_i": ("ato_i_out", None, None)}
for _k in (1, 2, 3):
    OUTKEY[f"e_s_l{_k:02d}"] = ("e_s_out", 3, _k - 1)
    OUTKEY[f"e_i_l{_k:02d}"] = ("e_i_out", 3, _k - 1)
MOMS = {f"mom:{m}{b}": (m, b) for m in MOM_NAMES for b in (*BULK, "c0", "e")}


def main(out_path, fixture_json=None):
    report = {}
    for arm in ("cyc2", "cyc3"):
        a = S.load(S.ARMS[arm])
        recs = L.load(S.ARMS[arm], "adv")
        reg = L.regions(L.interior(a["f3"]["a_i"]), L.interior(a["f2"]["tmask1"]), "T")
        ice, wet = reg["ice"], reg["wet"]
        Xa, mom, ow, res = S.port_after(a)

        def field(rs, key, nk, lay=None):
            v = L.assemble(rs, key, "out", nk)[0]
            return v if lay is None else v[..., lay]

        def with_seed(z, key):
            return {r: (h, mb, inp, o, {**out, key: z[f"adv:{r}:{key}"]}) for r, (h, mb, inp, o, out) in recs.items()}

        base = {f: field(recs, k, nk, lay) for f, (k, nk, lay) in OUTKEY.items()}
        for f, (m, b) in MOMS.items():
            base[f] = field(recs, f"{m}{b}_out", None if b in BULK else 3)
        for k in S.RES:
            base[k] = field(recs, k + "_out", None) - L.assemble(recs, k)[0]
        spread = dict.fromkeys(base, 0.0)
        for sd in SEEDS:
            z = np.load(f"{SEEDS_DIR}/out_{arm}_{sd:02d}.npz")
            if z["c3_bad"].size:
                raise RuntimeError(f"{arm} seed {sd}: C3 failed {z['c3_bad'][:3]}")
            for f, (k, nk, lay) in OUTKEY.items():
                v = field(with_seed(z, k), k, nk, lay)
                spread[f] = max(spread[f], float(L.rel(L.interior(v), L.interior(base[f]))[ice].max()))
            for f, (m, b) in MOMS.items():
                k = f"{m}{b}_out"
                v = field(with_seed(z, k), k, None if b in BULK else 3)
                spread[f] = max(spread[f], float(L.rel(L.interior(v), L.interior(base[f]))[ice].max()))
            for k in S.RES:
                v = field(with_seed(z, k + "_out"), k + "_out", None) - L.assemble(recs, k)[0]
                spread[k] = max(spread[k], float(np.abs(L.interior(v) - L.interior(base[k]))[wet].max()))
        err = {f: float(L.rel(L.interior(Xa[..., S.TR.index(f)]), L.interior(base[f]))[ice].max()) for f in KEEP}
        err["ato_i"] = float(L.rel(L.interior(ow), L.interior(base["ato_i"]))[ice].max())
        for f, (m, b) in MOMS.items():
            ti = S.TR.index(BULK[b]) if b in BULK else [S.TR.index(f"{'e_s' if b == 'c0' else 'e_i'}_l{k:02d}")
                                                         for k in (1, 2, 3)]
            p = np.asarray(mom[MOM_NAMES.index(m)])[..., ti]
            err[f] = float(L.rel(L.interior(p), L.interior(base[f]))[ice].max())
        for k in S.RES:
            err[k] = float(np.abs(L.interior(res[k]) - L.interior(base[k]))[wet].max())
        report[arm] = {f: dict(port=err[f], spread=spread[f], cls="A" if err[f] <= spread[f] else "over") for f in err}
        print("==", arm, "icycle", a["icycle"], "fields", len(err), "class A",
              sum(r["cls"] == "A" for r in report[arm].values()), flush=True)
        for f, r in report[arm].items():
            print(f"  {f:14s} port {r['port']:.2e} spread {r['spread']:.2e} {r['cls']}")
    json.dump(report, open(out_path, "w"), indent=1)
    if fixture_json:
        meta = json.loads(Path(fixture_json).read_text())
        meta["spread_4seed"] = {arm: {f: report[arm][f]["spread"] for f in (*KEEP, "ato_i")} for arm in report}
        meta["spread_4seed_definition"] = ("max over seeds 0-3 (shared perturb.py, 1 ulp on every recorded input, "
                                           "zeros held) of max rel(seed, base) on the ice region of the full ORCA1 "
                                           "domain")
        here = Path(__file__).resolve()
        meta["producer_scripts"]["spread"] = "scripts/validate/ice_fidelity/" + here.name
        meta["producer_scripts"]["seed"] = "scripts/validate/ice_fidelity/si3_prather_multicycle_seed.py"
        meta["producer_scripts"]["seed_run"] = "scripts/validate/ice_fidelity/si3_prather_multicycle_seed_run.sh"
        meta["producer_sha256"] = {k: hashlib.sha256((here.parent / Path(v).name).read_bytes()).hexdigest()
                                   for k, v in meta["producer_scripts"].items()}
        Path(fixture_json).write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    fx = sys.argv[sys.argv.index("--fixture-json") + 1] if "--fixture-json" in sys.argv else None
    main(sys.argv[1], fx)
