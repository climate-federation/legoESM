#!/usr/bin/env python
"""Score row-30 zbu inputs in NEMO source order using existing dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bn2_alpha_compare as loaders  # noqa: E402
import dump_lane  # noqa: E402
import ldf_slp_per_element as ldf  # noqa: E402
import zdf_chain_sweep as sweep  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402

BAR = 1.0e-15
SHAS = {
    "tke_dump_rn2b.bin": "fcd7ddee9e94b6158085f2ff5d174b2cb01e1b22a8524faa1b5be31a2b798961",
    "tke_dump_rn2.bin": "df573f8c07abab1cc0a2fc7fd2093dd8276dccf0319d3375220340d91b5a6004",
    "eiv_dump_prd_arg.bin": "12e67ec2d5cb951cc5efe02e41d56e4aa6395d3d1def7f653d2b88c91957ab1b",
    "eiv_dump_zgru_iik.bin": "d9d8d91277ae3766d9240642fae2abe5199552fa95e59c6254eb8a6b50543136",
    "eiv_dump_zbu_pre.bin": "df82610882fcc92c937ae0503b6ea45840731cebf16d9e2597400901e717f4ad",
    "seq_dump_rhd_nnn_kt00005761.bin": "81f60743004bbd52101227a66e6f358e2d85a98cbcf6d70d848e82f9f69d8232",
}
PARENT_SHA = "b2935557eb1ce6be74da08c7707e5323086091c1a4806fab6c454d1e6954c181"
BINARY_SHA = "a77fbaa9e302e699acb76b89a4c4d8e04d4ec0a77e6b0bbee8b8a45180a6d20f"
RESTART_SHA = "33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _kp1(a: np.ndarray) -> np.ndarray:
    return np.concatenate([a[..., 1:], np.zeros_like(a[..., :1])], axis=-1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mld-maps", type=Path, required=True)
    p.add_argument("--parent-artifact", type=Path, required=True)
    p.add_argument("--nemo-source", type=Path, required=True)
    p.add_argument("--nemo-binary", type=Path, required=True)
    p.add_argument("--expected-repo-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row30 zbu probe requires CPU fp64")
    if os.environ.get("DINO_1226_LANE") != "d180":
        raise SystemExit("row30 zbu probe requires DINO_1226_LANE=d180")
    if _git("rev-parse", "HEAD") != args.expected_repo_sha:
        raise SystemExit("repository HEAD differs from preregistered HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked worktree must be clean")
    run = args.run_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit("lane selector does not resolve to registered run")
    fixed = {
        args.parent_artifact: PARENT_SHA,
        args.nemo_binary: BINARY_SHA,
        run / "nemo": BINARY_SHA,
        run / "DINO_00005761_restart.nc": RESTART_SHA,
        **{run / name: digest for name, digest in SHAS.items()},
    }
    for path, expected in fixed.items():
        if sha256(path.resolve()) != expected:
            raise SystemExit(f"SHA receipt changed: {path}")
    if sha256(args.nemo_source.resolve()) != "8b4d8cffe35d66241eb77bdc508ef15d6dd90d8ff192fd60201ff83a7d523a29":
        raise SystemExit("active NEMO row30 source SHA changed")
    for name in ("tke_dump_rn2b.bin", "eiv_dump_prd_arg.bin", "eiv_dump_zbu_pre.bin"):
        if time_level_for_dump(name) != "before":
            raise SystemExit(f"time-level registry changed: {name}")
    source = args.nemo_source.read_text(errors="strict")
    quotes = (
        "* ( pn2(ji,jj,jk) + pn2(ji,jj,jk+1) ) * ( 1._wp - 0.5_wp * tmask(ji,jj,jk+1) )",
        "zbu = 0.5_wp * ( zdzr(ji,jj) + zdzr(ji+1,jj  ) )",
    )
    if any(q not in source for q in quotes):
        raise SystemExit("active NEMO zbu quote changed")

    st = ldf.build_state()
    _, loc = ldf.capture_locals(st["recall"], ldf.compute_nemo_native_slopes.__code__)
    captured_pn2 = np.asarray(loc["pn2"])
    captured_zbu = 0.5 * (np.asarray(loc["zdzr"]) + np.roll(np.asarray(loc["zdzr"]), -1, axis=1))
    model = SimpleNamespace(z_coord=st["z_coord"], config=st["model_config"])
    model._n2_nemo_before_tracers = MethodType(
        LatLonCGridOceanModel._n2_nemo_before_tracers, model)
    # The standalone bridge object does not install its separately loaded
    # NEMO tb/sb arrays into the leapfrog carry slots; the production twin's
    # --bridge-before path does. Seed those exact arrays before invoking the
    # production bundle, matching the real model-step precondition.
    bridge_state = st["bridge_state"]._replace(
        T_before=st["bridge_state"].T.replace(data=st["T"]),
        S_before=st["bridge_state"].S.replace(data=st["S"]))
    bundle = LatLonCGridOceanModel._tke_step_entry_n2_bundle(
        model, bridge_state)
    if bundle is None:
        raise SystemExit("production step-entry N2 bundle unexpectedly disabled")
    prod_pn2 = np.concatenate([
        np.zeros_like(np.asarray(bundle.rn2b)[..., :1]),
        np.asarray(bundle.rn2b)], axis=-1)

    ni, nj = st["active"].shape[1], st["active"].shape[0]
    jpi, jpj, _, hls = loaders._read_dims(str(run))
    nemo_pn2 = loaders._load_interior(str(run / "tke_dump_rn2b.bin"), ni, nj)
    nemo_prd = loaders._load_haloed(str(run / "eiv_dump_prd_arg.bin"), jpi, jpj, hls)
    nemo_zbu = loaders._load_haloed(str(run / "eiv_dump_zbu_pre.bin"), jpi, jpj, hls)
    nemo_zgru = loaders._load_haloed(str(run / "eiv_dump_zgru_iik.bin"), jpi, jpj, hls)
    nk = min(prod_pn2.shape[-1], nemo_pn2.shape[-1], nemo_prd.shape[-1], nemo_zbu.shape[-1])
    prod_pn2, nemo_pn2 = prod_pn2[..., :nk], nemo_pn2[..., :nk]
    prod_prd, nemo_prd = np.asarray(loc["prd"])[..., :nk], nemo_prd[..., :nk]
    act = np.asarray(st["active"])[..., :nk]
    act_kp1 = _kp1(act.astype(np.float64))
    # ldfslp's reverse jk loop is Fortran 2:jpkm1 only. Surface and dry
    # sentinel slots are initialized zeros, not evaluated operands.
    wet_t = np.zeros_like(act, dtype=bool)
    wet_w = np.zeros_like(act, dtype=bool)
    wet_u = np.zeros_like(act, dtype=bool)
    wet_t[..., ldf.KLO:min(ldf.KHI, nk)] = act[..., ldf.KLO:min(ldf.KHI, nk)]
    wet_w_full = ldf.wet_w_mask(np.asarray(st["active"]))[..., :nk]
    wet_u_full = ldf.wet_u_mask(
        np.asarray(st["active"]), st["u_mask"])[..., :nk]
    wet_w[..., ldf.KLO:min(ldf.KHI, nk)] = wet_w_full[..., ldf.KLO:min(ldf.KHI, nk)]
    wet_u[..., ldf.KLO:min(ldf.KHI, nk)] = wet_u_full[..., ldf.KLO:min(ldf.KHI, nk)]
    focus = sweep.focus_from_maps(args.mld_maps)

    # Oracle-derived intermediates in source order. Both sides intentionally
    # spell every binary64 boundary rather than algebraically refactor it.
    pp = prod_pn2
    npn = nemo_pn2
    prod_sum = pp + _kp1(pp)
    nemo_sum = npn + _kp1(npn)
    prod_prd1 = prod_prd + np.float64(1.0)
    nemo_prd1 = nemo_prd + np.float64(1.0)
    mask_factor = np.float64(1.0) - np.float64(0.5) * act_kp1
    prod_masked = prod_prd1 * prod_sum * mask_factor
    nemo_masked = nemo_prd1 * nemo_sum * mask_factor
    zm1_g = -np.float64(1.0) / np.float64(st["g"])
    # Preserve Fortran's left-associated source expression exactly:
    # (((zm1_g * (prd+1)) * (pn2+pn2_kp1)) * mask_factor).
    # Factoring zm1_g outside the completed product changes binary64 bits.
    prod_zdzr = ((zm1_g * prod_prd1) * prod_sum) * mask_factor
    nemo_zdzr = ((zm1_g * nemo_prd1) * nemo_sum) * mask_factor
    prod_pair = prod_zdzr + np.roll(prod_zdzr, -1, axis=1)
    nemo_pair = nemo_zdzr + np.roll(nemo_zdzr, -1, axis=1)
    prod_face = np.float64(0.5) * prod_pair
    nemo_face = np.float64(0.5) * nemo_pair

    ordered = (
        ("pn2_jk", pp, npn, wet_w),
        ("pn2_jkp1", _kp1(pp), _kp1(npn), wet_t),
        ("pn2_sum", prod_sum, nemo_sum, wet_t),
        ("prd_plus_one", prod_prd1, nemo_prd1, wet_t),
        ("mask_product", prod_masked, nemo_masked, wet_t),
        ("minus_one_over_g", prod_zdzr, nemo_zdzr, wet_t),
        ("east_face_pair_sum", prod_pair, nemo_pair, wet_u),
        ("half_face_average", prod_face, nemo_zbu[..., :nk], wet_u),
    )
    scores = {}
    first = None
    for name, actual, expected, wet in ordered:
        score = sweep.metrics(actual, expected, wet, focus, BAR)
        scores[name] = score
        if first is None and not score["pass"]:
            first = name
            break

    captured = {
        "pn2_vs_production_bundle_exact_unequal": int(np.count_nonzero(
            wet_w & (captured_pn2[..., :nk] != prod_pn2))),
        "captured_zbu": sweep.metrics(
            captured_zbu[..., :nk], nemo_zbu[..., :nk], wet_u, focus, BAR),
        "production_bundle_zbu": sweep.metrics(
            prod_face, nemo_zbu[..., :nk], wet_u, focus, BAR),
    }
    # Controls use the independently verified upstream zgru row, never an
    # output compared with itself and never the possibly-red stage under test.
    control_actual = np.asarray(loc["zgru"])[..., :nk]
    controls = sweep.planted_controls(
        control_actual, nemo_zgru[..., :nk], wet_u, BAR)
    swap = npn + np.concatenate([npn[..., :1], npn[..., :-1]], axis=-1)
    controls["jk_jkp1_swap_fails"] = not sweep.metrics(
        zm1_g * nemo_prd1 * swap * mask_factor,
        nemo_zdzr, wet_t, focus, BAR)["pass"]
    rn2 = loaders._load_interior(str(run / "tke_dump_rn2.bin"), ni, nj)[..., :nk]
    controls["rn2_for_rn2b_fails"] = not sweep.metrics(
        rn2, npn, wet_w, focus, BAR)["pass"]
    seq_now = loaders._load_haloed(
        str(run / "seq_dump_rhd_nnn_kt00005761.bin"), jpi, jpj, hls)[..., :nk]
    nk_now = min(seq_now.shape[-1], nemo_prd.shape[-1], wet_t.shape[-1])
    controls["current_prd_for_before_fails"] = not sweep.metrics(
        seq_now[..., :nk_now], nemo_prd[..., :nk_now],
        wet_t[..., :nk_now], focus, BAR)["pass"]
    exact_before = int(np.count_nonzero(
        wet_u & (control_actual != nemo_zgru[..., :nk])))
    ulp = control_actual.copy()
    idx = tuple(np.argwhere(
        wet_u & (control_actual == nemo_zgru[..., :nk]) & (control_actual != 0.0))[0])
    ulp[idx] = np.nextafter(ulp[idx], np.inf)
    controls["one_ulp_exact_identity_fired"] = int(np.count_nonzero(
        wet_u & (ulp != nemo_zgru[..., :nk]))) == exact_before + 1
    if not all(v for v in controls.values() if isinstance(v, bool)):
        raise SystemExit("row30 zbu planted control did not fire")

    harness_only = (
        first is None
        and captured["captured_zbu"]["n_diverged_columns"] > 0
        and captured["production_bundle_zbu"]["pass"])
    disposition = "DIVERGED-HARNESS" if harness_only else (
        "VERIFIED" if first is None else "DIVERGED")
    artifact = {
        "schema": "dino-zdf-row30-zbu-operands-v1",
        "disposition": disposition,
        "first_divergence": ("historical_probe_pn2" if harness_only else first),
        "bar": BAR,
        "lane": dump_lane.banner(),
        "focus_columns_ji": [list(x) for x in focus],
        "ordered_scores": scores,
        "harness_discrimination": captured,
        "controls": controls,
        "source_quotes": list(quotes),
        "repo_sha": args.expected_repo_sha,
        "provenance_sha256": {str(path.resolve()): sha256(path.resolve()) for path in (
            args.parent_artifact, args.nemo_source, args.nemo_binary, args.mld_maps,
            run / "mesh_mask.nc", run / "tke_dump_rn2b.bin",
            run / "tke_dump_rn2.bin", run / "eiv_dump_prd_arg.bin",
            run / "eiv_dump_zgru_iik.bin",
            run / "eiv_dump_zbu_pre.bin", run / "seq_dump_rhd_nnn_kt00005761.bin",
            run / "DINO_00005760_restart.nc", run / "DINO_00005761_restart.nc",
            run / "run.attempt1.log", Path(__file__), Path(ldf.__file__))},
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for name, score in scores.items():
        print(f"{name}: {score['n_diverged_columns']}/{score['n_wet_columns']} fail; focus_fail={sum(not x['pass'] for x in score['focus'])}")
        if name == first:
            break
    print(f"captured_zbu: {captured['captured_zbu']['n_diverged_columns']}/9758; production_bundle_zbu: {captured['production_bundle_zbu']['n_diverged_columns']}/9758")
    print(disposition)
    print(f"wrote {args.output}")
    return 0 if disposition in ("VERIFIED", "DIVERGED-HARNESS") else 30


if __name__ == "__main__":
    raise SystemExit(main())
