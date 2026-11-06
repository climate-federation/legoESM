#!/usr/bin/env python3
"""Proof (1): Every active-category forward bit unchanged vs 96bb44f0aeee (0 ULP, signed zeros and NaN bits included, eager AND jit)."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice import si3_o17 as o17_mod
from legoesm.ice import si3_thd_pnd as pnd_mod
from legoesm.ice.bitz_lipscomb import SI3DHFluxes
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import (
    DEFAULT_MASTER,
    _case,
)
from tests.ice.unit import test_si3_o9_zdf_dh_temp as o9_fixture
from tests.ice.unit import test_si3_o10_salinity as o10_fixture
from tests.ice.unit.test_si3_d194_traced_loops import _fixture as d194_fixture
from tests.ice.unit.test_si3_o12_pnd import (
    DECK_PND_NML,
    GRAVITY,
    R1_NLAY_I,
    R1_NLAY_S,
    R1_RDT,
    RDT,
    RHO_I,
    RHO_S,
    RHO_W,
)

REPO = Path(__file__).resolve().parents[2] / "claude_rounds/si3_nanfix/repo"
OUT_PATH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_nanfix_scratch/proof1_active_bits.json"
)


def _raw_bits(arr: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return a.view(np.uint64)


def _compare_bits(got: np.ndarray, want: np.ndarray, mask: np.ndarray | None = None) -> dict:
    g = np.asarray(got, dtype=np.float64)
    w = np.asarray(want, dtype=np.float64)
    assert g.shape == w.shape, f"shape mismatch: {g.shape} vs {w.shape}"
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        while m.ndim < g.ndim:
            m = m[..., None]
        m = np.broadcast_to(m, g.shape)
        g = g[m]
        w = w[m]
    else:
        g = g.ravel()
        w = w.ravel()
    gb = _raw_bits(g)
    wb = _raw_bits(w)
    eq = gb == wb
    diff_count = int(np.sum(~eq))
    max_ulp = 0
    if diff_count > 0:
        for gi, wi in zip(gb[~eq], wb[~eq]):
            max_ulp = max(max_ulp, int(abs(int(gi) - int(wi))))
    return {
        "compared": int(g.size),
        "equal": int(np.sum(eq)),
        "differing": diff_count,
        "max_ulp": max_ulp,
    }


def _load_parent_o12_stages():
    src_pnd = subprocess.check_output(
        ["git", "show", "96bb44f0aeee:packages/ice/legoesm/ice/si3_thd_pnd.py"],
        cwd=REPO,
        text=True,
    )
    mod_pnd = types.ModuleType("parent_si3_thd_pnd")
    exec(compile(src_pnd, "parent_si3_thd_pnd.py", "exec"), mod_pnd.__dict__)
    return mod_pnd.si3_ice_thd_o12_stages


def _o12_five_cat_fixture():
    namelist = {**o9_fixture._deck(), **o10_fixture.NML, **DECK_PND_NML,
                "ln_icedA": True, "rn_beta": 0.5, "rn_dmin": 8.0}
    categories = o9_fixture._categories(seed=19)
    rng = np.random.default_rng(20261119)
    ny, nx = 5, 6
    entries = []
    fluxes = []
    for jl in range(5):
        cells = [categories[jl][i % len(categories[jl])] for i in range(ny * nx)]
        e1, f1 = o9_fixture._entry(cells)
        entry = {k: v.reshape((ny, nx) + v.shape[1:]) for k, v in e1.items()}
        entry.update(
            at_i=jnp.full((ny, nx), 0.75),
            sst=jnp.full((ny, nx), 0.2),
            o_i=jnp.full((ny, nx), 10.0),
        )
        flux = SI3DHFluxes(**{f: getattr(f1, f).reshape(ny, nx) for f in f1._fields})
        entries.append(entry)
        fluxes.append(flux)
    a_ip = jnp.asarray(rng.uniform(0.02, 0.08, (ny, nx, 5)))
    h_ip = jnp.asarray(rng.uniform(0.05, 0.14, (ny, nx, 5)))
    h_il = jnp.asarray(rng.uniform(0.0, 0.02, (ny, nx, 5)))
    a_i = np.stack([np.asarray(e["a_i"]) for e in entries], axis=-1)
    zero = jnp.zeros((ny, nx))
    arrays = dict(
        category_entries=tuple(entries),
        category_fluxes=tuple(fluxes),
        sfx_bri=zero,
        sfx_lam=zero,
        hfx_thd=zero,
        wfx_lam=zero,
        a_ip=a_ip,
        v_ip=h_ip * a_ip,
        h_ip=h_ip,
        v_il=h_il * a_ip,
        h_il=h_il,
        a_ip_frac=jnp.asarray(np.asarray(a_ip) / np.maximum(a_i, 1.0e-10)),
        wfx_pnd=zero,
    )
    static = dict(
        namelist=namelist,
        rDt_ice=RDT,
        r1_Dt_ice=R1_RDT,
        r1_nlay_i=R1_NLAY_I,
        r1_nlay_s=R1_NLAY_S,
        rho_i=RHO_I,
        rho_s=RHO_S,
        rho_w=RHO_W,
        gravity=GRAVITY,
        nn_hls=1,
        constants=NEMO_SI3_CONSTANTS_CONFIG,
    )
    return arrays, static


class _StopAfterCapture(Exception):
    pass


def _probe_tile_o12_inputs():
    entry, forcing, card = _case(DEFAULT_MASTER)
    captured = {}
    orig_o12 = o17_mod.si3_ice_thd_o12_stages

    def _capture(**kwargs):
        captured.update(kwargs)
        raise _StopAfterCapture()

    o17_mod.si3_ice_thd_o12_stages = _capture
    try:
        with jax.disable_jit():
            try:
                o17_mod.si3_o17_step(entry, forcing, card)
            except _StopAfterCapture:
                pass
    finally:
        o17_mod.si3_ice_thd_o12_stages = orig_o12

    static_keys = {
        "namelist",
        "rDt_ice",
        "r1_Dt_ice",
        "r1_nlay_i",
        "r1_nlay_s",
        "rho_i",
        "rho_s",
        "rho_w",
        "gravity",
        "nn_hls",
        "constants",
    }
    arrays = {k: v for k, v in captured.items() if k not in static_keys}
    static = {k: v for k, v in captured.items() if k in static_keys}
    parent_entries = tuple(
        {
            k: v
            for k, v in e.items()
            if k
            not in {
                "active",
                "e_i_2d",
                "e_s_2d",
                "v_i",
                "v_s",
                "sv_i",
                "oa_i",
                "szv_i",
                "t_si",
            }
        }
        for e in arrays["category_entries"]
    )
    parent_arrays = {**arrays, "category_entries": parent_entries}
    return arrays, parent_arrays, static


def _score_active_categories(got_stages, want_stages, entries, epsi10: float = 1.0e-10) -> dict:
    total_compared = 0
    total_equal = 0
    total_differing = 0
    max_ulp = 0
    for jl in range(5):
        e = entries[jl]
        if "active" in e and e["active"] is not None:
            active = np.asarray(e["active"], dtype=bool)
        else:
            active = np.asarray(e["a_i"], dtype=np.float64) > epsi10
        got_leaves = jax.tree_util.tree_leaves(got_stages.categories[jl])
        want_leaves = jax.tree_util.tree_leaves(want_stages.categories[jl])
        for gl, wl in zip(got_leaves, want_leaves, strict=True):
            res = _compare_bits(np.asarray(gl), np.asarray(wl), active)
            total_compared += res["compared"]
            total_equal += res["equal"]
            total_differing += res["differing"]
            max_ulp = max(max_ulp, res["max_ulp"])
    return {
        "compared": total_compared,
        "equal": total_equal,
        "differing": total_differing,
        "max_ulp": max_ulp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("o12_5cat", "d194", "probe_tile"),
        required=True,
    )
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    parent_o12 = _load_parent_o12_stages()
    report = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    report.pop("status", None)

    t0 = time.perf_counter()
    if args.case == "o12_5cat":
        arr, stat = _o12_five_cat_fixture()
        parent_arr = arr
        prefix = "o12_5cat_fixture"
    elif args.case == "d194":
        arr, stat = d194_fixture()
        parent_arr = arr
        prefix = "d194_fixture"
    else:
        arr, parent_arr, stat = _probe_tile_o12_inputs()
        prefix = "probe_tile"

    with jax.disable_jit():
        got_eager = pnd_mod.si3_ice_thd_o12_stages(**arr, **stat)
        want_eager = parent_o12(**parent_arr, **stat)
    got_jit = jax.jit(lambda a: pnd_mod.si3_ice_thd_o12_stages(**a, **stat))(arr)
    # Since want_eager == want_jit at 96bb44f0aeee (verified by D194 digest),
    # comparing got_eager vs want_eager and got_jit vs got_eager proves both
    # got_eager == want_eager and got_jit == want_eager == want_jit at 0 ULP!
    report[f"{prefix}_eager_vs_parent"] = _score_active_categories(
        got_eager, want_eager, arr["category_entries"]
    )
    report[f"{prefix}_jit_vs_parent"] = _score_active_categories(
        got_jit, want_eager, arr["category_entries"]
    )
    report[f"{prefix}_eager_vs_jit"] = _score_active_categories(
        got_eager, got_jit, arr["category_entries"]
    )

    all_zero = all(
        v["differing"] == 0 and v["max_ulp"] == 0
        for k, v in report.items()
        if isinstance(v, dict)
    )
    report["status"] = "PASS_0_ULP_ACTIVE_CATEGORIES" if all_zero else "FAIL"
    OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[{args.case}] elapsed={time.perf_counter() - t0:.1f}s")
    print(json.dumps(report, indent=2))
    if not all_zero:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
