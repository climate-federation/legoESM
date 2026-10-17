#!/usr/bin/env python3
"""V1 substitution probe: which of NEMO's fields makes the card's step k exact.

One step k of the ORCA2V1-tke card from the card's own chain state after
step k - 1 (the ladder's --state-out pickle), with NEMO's fields swapped in.
Each arm is a fresh model; the result is scored on every level against
NEMO's restart of step k.

  --kind entry    NEMO's restart of step k - 1 replaces the card's entry:
                  arms plain | self (the card's own arrays re-wrapped:
                  identity control) | nemo_ts | nemo_tke | nemo_all.
                  Layout control: the probe reports, per swapped field, the
                  cells where NEMO's entry differs from the card's; these
                  must be the ladder's residue counts at step k - 1.
  --kind closure  NEMO's restart avm_k / avt_k of step k replaces the
                  vertical-mixing closure's output where the solves read it:
                  arms plain | self | nemo_avm | nemo_avt | nemo_both.
                  --entry-source nemo starts every closure arm from NEMO's
                  restart of step k - 1 (all prognostics) instead of the
                  card's chain state; the pickle then supplies only the
                  state's structure.

  --kind trace    one step from NEMO's restart of step k - 1 (entry source
                  nemo) with the TKE statement trace exposed; the traced
                  step must be bit-identical to the plain one (else exit 3).
                  Writes the trace, the published closure and NEMO's step-k
                  closure to <output>.npz for offline statement replays.

The probe prints no verdict; it writes counts, maxima and unequal cells.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "nemo_testcase_g5_orca2ice"))

ENTRY_ARMS = ("plain", "self", "nemo_ts", "nemo_tke", "nemo_all")
CLOSURE_ARMS = ("plain", "self", "nemo_avm", "nemo_avt", "nemo_both")
TS = ("T", "S")
TKE = ("tke", "tke_avm", "tke_avt", "tke_dissl", "tke_avm_surface")
DYN = ("u", "v", "eta", "uu_b", "vv_b", "bt_hist")
ARM_FIELDS = {"self": TS + TKE + DYN, "nemo_ts": TS, "nemo_tke": TKE,
              "nemo_all": TS + TKE + DYN}


def w_layout(full: np.ndarray, last: int, nk: int) -> np.ndarray:
    """NEMO's w-level profile 1..nk+1 (levels last) cut to a closure array
    whose last axis has `last` entries."""
    if last == nk + 1:
        return full
    if last == nk:
        return full[..., :nk]
    if last == nk - 1:
        return full[..., 1:nk]
    raise SystemExit(f"unknown closure layout {last}")


def unequal(ref: np.ndarray, cand: np.ndarray, top: int = 12) -> dict:
    ref, cand = np.asarray(ref, np.float64), np.asarray(cand, np.float64)
    if ref.shape != cand.shape:
        raise SystemExit(f"shape {cand.shape} != NEMO's {ref.shape}")
    d = np.abs(cand - ref)
    cells = np.argwhere(d != 0)
    order = np.argsort(-d[tuple(cells.T)])[:top] if cells.size else []
    scale = float(np.abs(ref).max())
    return {"unequal": int(cells.shape[0]), "max_abs": float(d.max()),
            "max_norm": float(d.max() / scale) if scale > 0 else float(d.max()),
            "scale": scale, "cells": [[int(x) for x in cells[i]] for i in order]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="the V1 twin_a deck")
    ap.add_argument("--restart-root", type=Path, action="append", required=True)
    ap.add_argument("--kind", choices=("entry", "closure", "trace"), required=True)
    ap.add_argument("--kt", type=int, required=True)
    ap.add_argument("--state-in", type=Path, required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--entry-source", choices=("card", "nemo"), default="card")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    arms = a.arms.split(",")
    allowed = {"entry": ENTRY_ARMS, "closure": CLOSURE_ARMS, "trace": ("plain",)}[a.kind]
    if arms[0] != "plain" or any(x not in allowed for x in arms):
        raise SystemExit(f"arms must start with plain and lie in {allowed}")
    if a.entry_source == "nemo" and a.kind == "entry":
        raise SystemExit("--entry-source nemo is a closure-kind option")
    if a.kind == "trace" and a.entry_source != "nemo":
        raise SystemExit("--kind trace starts from NEMO's entry (--entry-source nemo)")

    import nemo_testcase_g2_orca2bathy_ladder as L
    L.g1._fp64()
    L._pickle_fields()
    import g5_records as g5
    import jax
    import jax.numpy as jnp
    import netCDF4
    import legoesm.ocean.physics.vertical_mixing as VM
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.fidelity.provenance import git_sha

    card = recipe.build_nemo_testcase_card("ORCA2V1-tke", deck_root=a.root)
    cfg, z = card.recipe.model_config, card.recipe.z_coord
    nk = int(z.n_levels)
    saved = pickle.loads(a.state_in.read_bytes())
    if saved["after_kt"] != a.kt - 1:
        raise SystemExit(f"--state-in is after step {saved['after_kt']}, not {a.kt - 1}")
    entry = saved["state"]
    fw, sf = g5.nemo_supplied_forcing(card, a.root, a.kt)
    sf = sf._replace(ice_mean_thickness_hm_i=g5.read_v1_hm_i(a.root, a.kt))

    def restart(kt):
        path = L.restart_path(a.restart_root, kt)
        if path is None:
            raise SystemExit(f"no NEMO restart of step {kt}")
        with netCDF4.Dataset(path) as h:
            var = lambda n: np.asarray(h.variables[n][0], np.float64)  # noqa: E731
            lev = lambda n: L.levels_last(var(n), nk)                    # noqa: E731
            w = lambda n: L.levels_last(var(n), nk + 1)                  # noqa: E731
            out = {"T": lev("tn"), "S": lev("sn"), "u": L.g1.pad_u(lev("un")),
                   "v": L.g1.pad_v(lev("vn")), "eta": var("sshn"),
                   "uu_b": L.g1.pad_u(var("uu_n")), "vv_b": L.g1.pad_v(var("vv_n")),
                   "tke_avm_surface": var("avm_k")[0], "avm_k": w("avm_k"),
                   "avt_k": w("avt_k"), "path": path}
            for leaf, n in L.TKE_RESTART.items():
                out[leaf] = w(n)[..., 1:nk]
        out["bt_hist"] = L.g1.nemo_histories(path.parent, kt + 1)   # entering kt + 1
        return out

    after = restart(a.kt)
    before = restart(a.kt - 1) if a.kind == "entry" or a.entry_source == "nemo" else None

    def substitute(state, fields, src):
        rep = {}
        for f in fields:
            if f == "bt_hist":
                rep[f] = (jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x)), state.bt_hist)
                          if src is None else src["bt_hist"])
                continue
            leaf = getattr(state, f)
            data = np.asarray(leaf.data) if src is None else src[f]
            rep[f] = leaf.replace(data=jnp.asarray(data, leaf.data.dtype).reshape(leaf.data.shape))
        return state._replace(**rep)

    orig, cap, sub = VM.compute_vertical_K_profiles, {}, {}

    def patched(*args, **kw):
        out = orig(*args, **kw)
        k_t, a_m = out[0], out[1]
        cap["K"], cap["A"] = tuple(k_t.shape), tuple(a_m.shape)
        if "K" in sub:
            k_t = jnp.asarray(sub["K"], k_t.dtype).reshape(k_t.shape)
        if "A" in sub:
            a_m = jnp.asarray(sub["A"], a_m.dtype).reshape(a_m.shape)
        return (k_t, a_m, *out[2:])

    if a.kind == "closure":
        VM.compute_vertical_K_profiles = patched

    with netCDF4.Dataset(after["path"]) as h:
        after_uv = {f: L.levels_last(np.asarray(h.variables[n][0], np.float64), nk)
                    for f, n in (("u", "un"), ("v", "vn"))}

    def score(st):
        lev, srf = L.card_levels(st), L.card_surface(st)
        o = {f: unequal(after[f] if f in TS else after_uv[f], lev[f]) for f in ("u", "v", "T", "S")}
        o["ssh"] = unequal(after["eta"], srf["ssh"])
        for leaf in TKE:
            o[leaf] = unequal(after[leaf], np.asarray(getattr(st, leaf).data))
        return o

    if a.entry_source == "nemo":
        entry = substitute(entry, ARM_FIELDS["nemo_all"], before)

    if a.kind == "trace":
        return trace(a, card, entry, after, fw, sf, git_sha())

    report = {"legoesm_git_sha": git_sha(), "kind": a.kind, "kt": a.kt,
              "entry_source": a.entry_source,
              "state_in": str(a.state_in), "restart": str(after["path"]),
              "arms": {}, "bit_identical_to_plain": {}}
    if a.kind == "entry":
        report["entry_minus_card"] = {}
        for f in ARM_FIELDS["nemo_all"]:
            if f == "bt_hist":
                pairs = zip(jax.tree_util.tree_leaves(before["bt_hist"]),
                            jax.tree_util.tree_leaves(entry.bt_hist), strict=True)
                report["entry_minus_card"][f] = [int(np.count_nonzero(np.asarray(x) != np.asarray(y)))
                                                 for x, y in pairs]
            else:
                c = np.asarray(getattr(entry, f).data)
                report["entry_minus_card"][f] = int(np.count_nonzero(
                    np.asarray(before[f]).reshape(c.shape) != c))
        print("entry_minus_card", report["entry_minus_card"], flush=True)
    base = None
    for arm in arms:
        sub.clear()
        cap.clear()
        state = entry
        if a.kind == "entry" and arm != "plain":
            state = substitute(entry, ARM_FIELDS[arm], None if arm == "self" else before)
        if a.kind == "closure" and arm != "plain":
            if arm == "self":     # the card's own published avm (with surface) on w 1..nk+1
                own = base[0]
                inner = np.asarray(own.tke_avm.data)
                zero = np.zeros_like(inner[..., :1])
                sub["A"] = w_layout(np.concatenate(
                    [np.asarray(own.tke_avm_surface.data)[..., None], inner, zero], -1),
                    base[1]["A"][-1], nk)
            if arm in ("nemo_avm", "nemo_both"):
                sub["A"] = w_layout(after["avm_k"], base[1]["A"][-1], nk)
            if arm in ("nemo_avt", "nemo_both"):
                sub["K"] = w_layout(after["avt_k"], base[1]["K"][-1], nk)
        model = LatLonCGridOceanModel(card.recipe.grid, z, cfg)
        st = jax.device_get(model.step(state, dt=card.dt_s, freshwater=fw, surface_forcing=sf))
        if arm == "plain":
            base = (st, dict(cap))
            if a.kind == "closure" and not cap:
                raise SystemExit("the closure patch never ran")
        else:
            report["bit_identical_to_plain"][arm] = all(
                np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(
                    jax.tree_util.tree_leaves(base[0]), jax.tree_util.tree_leaves(st), strict=True))
        report["arms"][arm] = score(st)
        print(arm, {f: r["unequal"] for f, r in report["arms"][arm].items()},
              "identical_to_plain" if report["bit_identical_to_plain"].get(arm) else "", flush=True)
    a.output.write_text(json.dumps(report, indent=1))
    return 0


def trace(a, card, entry, after, fw, sf, sha) -> int:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    grid, z, cfg = card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
    kw = dict(dt=card.dt_s, freshwater=fw, surface_forcing=sf)
    plain = jax.device_get(LatLonCGridOceanModel(grid, z, cfg).step(entry, **kw))
    tr = jax.device_get(LatLonCGridOceanModel(grid, z, cfg, _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
        expose_live_stage_operands=True)).step(entry, **kw))
    passive = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(
        jax.tree_util.tree_leaves(plain), jax.tree_util.tree_leaves(tr.state_after), strict=True))
    t = tr.tke_statement_trace
    arrays = {f"trace_{f}": np.asarray(getattr(t, f), np.float64) for f in t._fields
              if getattr(t, f) is not None and not isinstance(getattr(t, f), tuple)}
    for f in ("tke", "tke_avm", "tke_avt", "tke_dissl"):
        arrays[f"entry_{f}"] = np.asarray(getattr(entry, f).data, np.float64)
        arrays[f"card_{f}"] = np.asarray(getattr(plain, f).data, np.float64)
        arrays[f"nemo_{f}"] = np.asarray(after[f], np.float64)
    np.savez_compressed(a.output.with_suffix(".npz"), passive=passive, **arrays)
    a.output.write_text(json.dumps({"legoesm_git_sha": sha, "kind": "trace", "kt": a.kt,
                                    "restart": str(after["path"]), "passive": passive,
                                    "shapes": {k: list(v.shape) for k, v in arrays.items()}}, indent=1))
    print("passive", passive, flush=True)
    return 0 if passive else 3


if __name__ == "__main__":
    raise SystemExit(main())
