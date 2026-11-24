#!/usr/bin/env python
"""Elo debate tournament for round 2: standard schemes, round-1 survivors (J3, J17) and round-2 candidates, per panel.

pair   : write the next Swiss round's games (adjacent ratings, no rematch) and the judge prompt with held-out evidence.
update : read the two judges' verdict lines ("G<id>: A|B|D"), both agree = win, split or D = draw; Elo update.
Ratings start at START; K-factor K. Evidence per game: per-site held-out log-RMSE of both (same sites), median rank
among the 12 standard schemes, parameter count, formula. Details voted (coscientist/round2/votes.md).
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

START, K = 1200.0, 32.0
DUPLICATE_TOL = 1e-4  # per-site log-RMSE agreement counted as the same formulation
PANELS = ("m_stable", "m_unstable", "h_stable", "h_unstable")
STANDARD_FORMS = {
    "businger1971": "Kansas: phi_m 1+4.7z / (1-15z)^-1/4; phi_h 0.74+4.7z / 0.74(1-9z)^-1/2 (kappa 0.35)",
    "hogstrom1988": "phi_m 1+6z / (1-19.3z)^-1/4; phi_h 0.95+7.8z / 0.95(1-11.6z)^-1/2",
    "dyer1974": "1+5z / (1-16z)^-1/4 (m), (1-16z)^-1/2 (h)",
    "holtslag_debruin1988": "stable psi -[a z + b(z - c/d) e^-dz + bc/d], a=0.7 b=0.75 c=5 d=0.35; unstable Dyer",
    "beljaars_holtslag1991": "stable psi -[a z + b(z - c/d) e^-dz + bc/d] (a=1,b=2/3,c=5,d=0.35), heat (1+2az/3)^1.5 form",
    "grachev2007_sheba": "SHEBA: phi_m 1 + a z (1+z)^(1/3)/(1+bz), phi_h 1 + (a z + b z^2)/(1 + c z + z^2)",
    "gryanik2020": "phi_m 1 + 5z/(1+0.3z)^(2/3), phi_h 0.95 + 5z/(1+0.4z)",
    "coare30": "COARE3.0: Kansas/convective blend unstable, Beljaars-Holtslag stable",
    "wrf_revmm5": "WRF revised MM5: Cheng-Brutsaert stable, Kansas+free-convection blend unstable",
    "mynn_sfc": "MYNN surface layer psi_opt 0 (Cheng-Brutsaert stable, Fairall unstable)",
    "gfs": "GFS: log-ratio stable psi with saturating growth, Kansas unstable",
    "zeng1998_clm": "CLM: linear stable to 1 then log/power; Kansas unstable with -1/3 free convection beyond -1.574 (m) / -0.465 (h)",
    "J3_global": "round 1 J3: stable 1 + a_inf z + (a0-a_inf) z/(1+z/zc), heat Pr0 + that - 1; unstable Dyer",
    "J17_global": "round 1 J17: unstable (1-16z)^-1/4 (1-z/lambda)^-1/12; rest Dyer",
}


def _j3_j17(bench):
    g = bench["global_fit"]
    j3 = {k: round(g["J3"][k], 4) for k in ("a0", "a_inf", "zc", "pr0")}
    return {"J3_global": f"; fitted {j3}", "J17_global": f"; fitted lambda {g['J17']['lambda']:.4f}"}


def entrants(bench, cands, top):
    """{panel: {name: {"scores": {site: v}, "form": str, "n_params": int}}}; candidates: top-N screen-passing per panel."""
    out = {p: {} for p in PANELS}
    for s, v in bench["sites"].items():
        for panel, sc in v["offset_corrected"].items():
            for name in STANDARD_FORMS:
                val = sc.get(name, {}).get("gradient", {}).get("log_rmse", np.nan)
                if np.isfinite(val):
                    e = out[panel].setdefault(name, {"scores": {}, "form": STANDARD_FORMS[name] + _j3_j17(bench).get(name, ""),
                                                     "n_params": "fixed published" if name not in ("J3_global", "J17_global") else "fitted"})
                    e["scores"][s] = val
    for panel in PANELS:
        kind = panel[0]
        ok = [(c["rank"][panel]["median_rank"], n, c) for n, c in cands.items()
              if c.get("kind") == kind and c.get("screen", {}).get("pass") and panel in c.get("rank", {})]
        kept = []
        for rk, n, c in sorted(ok, key=lambda t: t[0]):  # collapse candidates whose panel scores coincide (algebraic duplicates)
            if any(set(c["scores"][panel]) == set(k["scores"][panel]) and
                   all(abs(c["scores"][panel][s] - k["scores"][panel][s]) < DUPLICATE_TOL for s in c["scores"][panel]) for _, _, k in kept):
                continue
            kept.append((rk, n, c))
        for _, n, c in kept[:top]:
            side = panel.split("_")[1]
            out[panel][n] = {"scores": c["scores"][panel], "form": f"stable {c['spec']['stable']}; unstable {c['spec']['unstable']}; "
                             f"fitted {json.dumps({k: round(v, 4) for k, v in c['params'].items()})}; local exponent d ln phi/d ln|zeta| at 10: "
                             f"stable {c['screen']['stable_exponent_at_10']:.2f}, unstable {c['screen']['unstable_exponent_at_10']:.2f}",
                             "n_params": c["screen"][f"{side}_n_params"]}
    for panel in PANELS:  # an entrant whose panel scores equal an earlier one is the same formula here: merged into it
        names = list(out[panel])
        for i, n in enumerate(names):
            for m in names[:i]:
                if m in out[panel] and n in out[panel] and _same(out[panel][n]["scores"], out[panel][m]["scores"]):
                    out[panel][m].setdefault("aliases", []).append(n)
                    del out[panel][n]
                    break
    return out


def _same(a, b):
    return set(a) == set(b) and all(abs(a[s] - b[s]) < DUPLICATE_TOL for s in a)


def median_rank(scores, panel, bench):
    std = bench["standard"]
    ranks = []
    for s, v in scores.items():
        sc = bench["sites"][s]["offset_corrected"].get(panel, {})
        ranks.append(1 + sum(sc[k]["gradient"]["log_rmse"] < v for k in std if np.isfinite(sc[k]["gradient"]["log_rmse"])))
    return float(np.median(ranks))


def pair(state, pool, bench):
    games = []
    for panel in PANELS:
        r = state["ratings"].setdefault(panel, {n: START for n in pool[panel]})
        for n in pool[panel]:
            r.setdefault(n, START)
        order = sorted(r, key=lambda n: -r[n])
        played = {tuple(sorted(g[1:3])) for g in state["games"] if g[0] == panel}
        free = list(order)
        while len(free) >= 2:
            a = free.pop(0)
            b = next((x for x in free if tuple(sorted((a, x))) not in played), None)
            if b is None:
                continue
            free.remove(b)
            games.append([panel, a, b])
    lines = []
    for i, (panel, a, b) in enumerate(games):
        sites = sorted(set(pool[panel][a]["scores"]) & set(pool[panel][b]["scores"]))
        rows = " ".join(f"{s}:{pool[panel][a]['scores'][s]:.4f}/{pool[panel][b]['scores'][s]:.4f}" for s in sites)
        al = lambda n: f" = {'/'.join(pool[panel][n]['aliases'])} in this panel" if pool[panel][n].get("aliases") else ""  # noqa: E731
        lines.append(f"G{len(state['games']) + i} [{panel}] A={a}{al(a)} ({pool[panel][a]['form']}; params {pool[panel][a]['n_params']}; "
                     f"median rank {median_rank(pool[panel][a]['scores'], panel, bench):.1f}/13) vs B={b}{al(b)} ({pool[panel][b]['form']}; "
                     f"params {pool[panel][b]['n_params']}; median rank {median_rank(pool[panel][b]['scores'], panel, bench):.1f}/13)\n"
                     f"   held-out per-site log-RMSE A/B (lower better): {rows}")
    return games, "\n".join(lines)


def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update(state, games, verdicts):
    """verdicts: list of {gid: 'A'|'B'|'D'} per judge."""
    base = len(state["games"])
    if len(verdicts) != 2:
        raise ValueError(f"need exactly two judges' verdict files, got {len(verdicts)}")
    for i, (panel, a, b) in enumerate(games):
        gid = f"G{base + i}"
        missing = [n for n, j in enumerate(verdicts) if gid not in j]
        if missing:
            raise ValueError(f"judge(s) {missing} gave no verdict for {gid}")
        v = [j[gid] for j in verdicts]
        score = 1.0 if all(x == "A" for x in v) else 0.0 if all(x == "B" for x in v) else 0.5
        r = state["ratings"][panel]
        ea = expected(r[a], r[b])
        r[a] += K * (score - ea)
        r[b] -= K * (score - ea)
        state["games"].append([panel, a, b, v, score])


def parse(path):
    return {m.group(1): m.group(2) for m in re.finditer(r"\b(G\d+)\s*[:=]\s*\**\s*([ABD])\b", open(path).read())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("pair", "update", "table"))
    ap.add_argument("--state", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--cands", nargs="*", default=[])
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--games")
    ap.add_argument("--prompt")
    ap.add_argument("--verdicts", nargs="*", default=[])
    a = ap.parse_args()
    bench = json.load(open(a.bench))
    cands = {}
    for f in a.cands:
        cands.update(json.load(open(f)))
    pool = entrants(bench, cands, a.top)
    try:
        state = json.load(open(a.state))
    except FileNotFoundError:
        state = {"ratings": {}, "games": []}
    if a.mode == "pair":
        games, text = pair(state, pool, bench)
        json.dump(state, open(a.state, "w"), indent=1)  # ratings initialised at pairing persist
        json.dump(games, open(a.games, "w"))
        open(a.prompt, "w").write(text + "\n")
        print(len(games), "games")
    elif a.mode == "update":
        update(state, json.load(open(a.games)), [parse(p) for p in a.verdicts])
        json.dump(state, open(a.state, "w"), indent=1)
    for panel in PANELS:
        r = state["ratings"].get(panel, {})
        n = {x: sum(x in g[1:3] for g in state["games"] if g[0] == panel) for x in r}
        print(panel, " ".join(f"{x}:{r[x]:.0f}({n[x]})" for x in sorted(r, key=lambda x: -r[x])))


if __name__ == "__main__":
    assert abs(expected(1200, 1200) - 0.5) < 1e-12 and expected(1400, 1200) > 0.75
    main()
