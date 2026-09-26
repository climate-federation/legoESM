#!/usr/bin/env python
"""Peel the row-30 post-bound U-slope denominator using existing receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

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
from legoesm.ocean.fidelity.nemo_io import (  # noqa: E402
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)

BAR = 1.0e-15
BINARY_SHA = "a77fbaa9e302e699acb76b89a4c4d8e04d4ec0a77e6b0bbee8b8a45180a6d20f"
RESTART_SHA = "33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c"
INPUT_RESTART_SHA = "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e"
PARENT_SHA = "6e0cbdc65e638c381cc1fcfa1a326c3e165079b2527d15e2dcf52dd4abea9549"
FIXED = {
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "eiv_dump_zau.bin": "78f5d2159baad245f6c2fbb29f4cfbc24bc2724cfbd8b4077953e5b3576398b1",
    "eiv_dump_zbu_pre.bin": "df82610882fcc92c937ae0503b6ea45840731cebf16d9e2597400901e717f4ad",
    "eiv_dump_zbu_post.bin": "ed2079f378739d502967d13d99f37eda3282811185260cae890985cfef123e9c",
}
SOURCE_SHAS = {
    "ldfslp": "8b4d8cffe35d66241eb77bdc508ef15d6dd90d8ff192fd60201ff83a7d523a29",
    "domqco": "e0232c99fdf5d1e7baa2e4a7518fba1d7899f006a89f3b5ea60849ecf9bd6d61",
    "domzgr": "fb18c1cb2d0baa3807a7b2d5be30f363349676cb79e23871467c25da09a03ba3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mld-maps", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--ldfslp-source", type=Path, required=True)
    parser.add_argument("--domqco-source", type=Path, required=True)
    parser.add_argument("--domzgr-source", type=Path, required=True)
    parser.add_argument("--nemo-binary", type=Path, required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row30 zbu limiter probe requires CPU fp64")
    if os.environ.get("DINO_1226_LANE") != "d180":
        raise SystemExit("row30 zbu limiter probe requires DINO_1226_LANE=d180")
    if git("rev-parse", "HEAD") != args.expected_repo_sha:
        raise SystemExit("repository HEAD differs from preregistered HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked worktree must be clean")
    run = args.run_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit("lane selector does not resolve to registered run")

    bound = {
        args.parent_artifact: PARENT_SHA,
        args.nemo_binary: BINARY_SHA,
        run / "nemo": BINARY_SHA,
        run / "DINO_00005760_restart.nc": INPUT_RESTART_SHA,
        run / "DINO_00005761_restart.nc": RESTART_SHA,
        args.mld_maps: "9fb7344d1e6f92232d211f6a52ff8636022f0d9b0b05acea2b0bae6e6afd9bf0",
        args.ldfslp_source: SOURCE_SHAS["ldfslp"],
        args.domqco_source: SOURCE_SHAS["domqco"],
        args.domzgr_source: SOURCE_SHAS["domzgr"],
        **{run / name: value for name, value in FIXED.items()},
    }
    for path, expected in bound.items():
        if sha256(path.resolve()) != expected:
            raise SystemExit(f"SHA receipt changed: {path}")

    ldf_source = args.ldfslp_source.read_text(errors="strict")
    domqco_source = args.domqco_source.read_text(errors="strict")
    domzgr_source = args.domzgr_source.read_text(errors="strict")
    source_quotes = (
        "zbu = MIN(  zbu, - z1_slpmax * ABS( zau ) , -7.e+3_wp/e3u(ji,jj,jk,Kmm)* ABS( zau )  )",
        "pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)",
        "# define  e3u(i,j,k,t)      (E3u_0(i,j,k) Tmsk(r3u,umask,i,j,k,t))",
    )
    if source_quotes[0] not in ldf_source:
        raise SystemExit("ldfslp.F90:247-248 quote changed")
    if source_quotes[1] not in domqco_source:
        raise SystemExit("domqco.F90:166 quote changed")
    if source_quotes[2] not in domzgr_source:
        raise SystemExit("domzgr e3u macro changed")

    state = ldf.build_state()
    _, loc = ldf.capture_locals(
        state["recall"], ldf.compute_nemo_native_slopes.__code__)
    ni, nj = state["active"].shape[1], state["active"].shape[0]
    jpi, jpj, _, hls = loaders._read_dims(str(run))
    nemo_zau = loaders._load_haloed(
        str(run / "eiv_dump_zau.bin"), jpi, jpj, hls)
    nemo_pre = loaders._load_haloed(
        str(run / "eiv_dump_zbu_pre.bin"), jpi, jpj, hls)
    nemo_post = loaders._load_haloed(
        str(run / "eiv_dump_zbu_post.bin"), jpi, jpj, hls)

    mesh = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(str(run / "DINO_00005760_restart.nc"), nn_hls=0)
    before = read_nemo_restart_before(
        str(run / "DINO_00005760_restart.nc"), nn_hls=0)
    e3u0 = np.asarray(mesh.e3u_0, dtype=np.float64)
    umask = np.asarray(mesh.umask, dtype=np.float64)
    e1e2t = np.asarray(mesh.e1t, dtype=np.float64) * np.asarray(
        mesh.e2t, dtype=np.float64)
    e1e2u = np.asarray(mesh.e1u, dtype=np.float64) * np.asarray(
        mesh.e2u, dtype=np.float64)
    hu0 = np.asarray(mesh.hu_0, dtype=np.float64)

    def qco_r3u(ssh: np.ndarray) -> np.ndarray:
        weighted = e1e2t * np.asarray(ssh, dtype=np.float64)
        pair = weighted + np.roll(weighted, -1, axis=1)
        r1_hu0 = np.zeros_like(hu0)
        r1_e1e2u = np.zeros_like(e1e2u)
        np.divide(np.float64(1.0), hu0, out=r1_hu0, where=hu0 > 0.0)
        np.divide(
            np.float64(1.0), e1e2u, out=r1_e1e2u, where=e1e2u > 0.0)
        return ((np.float64(0.5) * pair) * r1_hu0) * r1_e1e2u

    r3u_now = qco_r3u(np.asarray(now.ssh))
    r3u_before = qco_r3u(np.asarray(before.ssh))
    live_e3u = e3u0 * (np.float64(1.0) + r3u_now[..., None] * umask)
    before_e3u = e3u0 * (np.float64(1.0) + r3u_before[..., None] * umask)

    uv_code = next(
        item for item in ldf.compute_nemo_native_slopes.__code__.co_consts
        if isinstance(item, types.CodeType) and item.co_name == "_uv_slp")
    _, uv_calls = ldf.capture_return_locals(state["recall"], uv_code)
    if len(uv_calls) != 2:
        raise SystemExit(f"expected two _uv_slp calls, captured {len(uv_calls)}")
    u_loc, _ = uv_calls[0]
    production_pre = np.asarray(u_loc["zb_pair"])
    production_zau = np.asarray(u_loc["zau"])
    production_static_e3u = np.asarray(u_loc["e3_face"])
    nk = min(
        production_pre.shape[-1], nemo_pre.shape[-1], live_e3u.shape[-1])
    production_pre = production_pre[..., :nk]
    production_zau = production_zau[..., :nk]
    production_static_e3u = production_static_e3u[..., :nk]
    nemo_pre, nemo_zau, nemo_post = (
        nemo_pre[..., :nk], nemo_zau[..., :nk], nemo_post[..., :nk])
    e3u0, live_e3u, before_e3u = (
        e3u0[..., :nk], live_e3u[..., :nk], before_e3u[..., :nk])

    wet = np.zeros_like(state["active"][..., :nk], dtype=bool)
    wet_full = ldf.wet_u_mask(
        np.asarray(state["active"]), state["u_mask"])[..., :nk]
    wet[..., ldf.KLO:min(ldf.KHI, nk)] = wet_full[
        ..., ldf.KLO:min(ldf.KHI, nk)]
    focus = sweep.focus_from_maps(args.mld_maps)

    z1 = np.asarray(
        np.float64(1.0) / np.float64(state["gm_cfg"].S_max),
        dtype=np.float64)
    abs_zau = np.abs(production_zau)
    slope_cap = -z1 * abs_zau
    live_metric_cap = (-np.float64(7.0e3) / live_e3u) * abs_zau
    inner = np.minimum(slope_cap, live_metric_cap)
    literal_post = np.minimum(production_pre, inner)
    static_metric_cap = (-np.float64(7.0e3) / production_static_e3u) * abs_zau
    static_post = np.minimum(
        production_pre, np.minimum(slope_cap, static_metric_cap))

    ordered = (
        ("zbu_pre", production_pre, nemo_pre),
        ("zau", production_zau, nemo_zau),
        ("outer_min_live_e3u", literal_post, nemo_post),
    )
    scores: dict[str, dict] = {}
    first = None
    for name, actual, expected in ordered:
        score = sweep.metrics(actual, expected, wet, focus, BAR)
        scores[name] = score
        if first is None and not score["pass"]:
            first = name
            break

    discrimination = {
        "static_e3u_vs_live_e3u": sweep.metrics(
            production_static_e3u, live_e3u, wet, focus, BAR),
        "static_metric_cap_vs_nemo_post": sweep.metrics(
            static_post, nemo_post, wet, focus, BAR),
        "live_metric_cap_vs_nemo_post": sweep.metrics(
            literal_post, nemo_post, wet, focus, BAR),
        "raw_e3u0_vs_live_e3u": sweep.metrics(
            e3u0, live_e3u, wet, focus, BAR),
    }
    controls = sweep.planted_controls(production_zau, nemo_zau, wet, BAR)
    controls["static_e3u_fails"] = not discrimination[
        "static_metric_cap_vs_nemo_post"]["pass"]
    controls["before_ssh_fails"] = not sweep.metrics(
        np.minimum(
            production_pre,
            np.minimum(slope_cap, (-np.float64(7.0e3) / before_e3u) * abs_zau)),
        nemo_post, wet, focus, BAR)["pass"]
    controls["zonal_roll_fails"] = not sweep.metrics(
        np.roll(literal_post, 1, axis=1), nemo_post, wet, focus, BAR)["pass"]
    unequal = int(np.count_nonzero(wet & (literal_post != nemo_post)))
    ulp = literal_post.copy()
    candidates = np.argwhere(wet & (literal_post == nemo_post))
    if candidates.size == 0:
        raise SystemExit("no exact wet value available for one-ULP control")
    idx = tuple(candidates[0])
    ulp[idx] = np.nextafter(ulp[idx], np.inf)
    controls["one_ulp_exact_identity_fired"] = (
        int(np.count_nonzero(wet & (ulp != nemo_post))) == unequal + 1)
    if not all(v for v in controls.values() if isinstance(v, bool)):
        raise SystemExit("row30 zbu limiter planted control did not fire")

    owner = (
        first is None
        and not discrimination["static_metric_cap_vs_nemo_post"]["pass"]
        and discrimination["live_metric_cap_vs_nemo_post"]["pass"])
    disposition = "DIVERGED-LIVE-E3U" if owner else (
        "VERIFIED" if first is None else "DIVERGED")
    artifact = {
        "schema": "dino-zdf-row30-zbu-limiter-v1",
        "disposition": disposition,
        "owner": "e3u(Kmm) at ldfslp.F90:248" if owner else first,
        "bar": BAR,
        "lane": dump_lane.banner(),
        "focus_columns_ji": [list(x) for x in focus],
        "ordered_scores": scores,
        "literal_intermediates": {
            "z1_slpmax": float(z1),
            "r3u_now_min": float(np.min(r3u_now)),
            "r3u_now_max": float(np.max(r3u_now)),
        },
        "discrimination": discrimination,
        "controls": controls,
        "source_quotes": list(source_quotes),
        "repo_sha": args.expected_repo_sha,
        "provenance_sha256": {
            str(path.resolve()): sha256(path.resolve())
            for path in (*bound.keys(), Path(__file__), Path(ldf.__file__))
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for name, score in scores.items():
        focus_fail = sum(not item["pass"] for item in score["focus"])
        print(
            f"{name}: {score['n_diverged_columns']}/{score['n_wet_columns']} "
            f"fail bar {BAR:.1e}; focus_fail={focus_fail}")
    print(
        "static e3u substitution: "
        f"{discrimination['static_metric_cap_vs_nemo_post']['n_diverged_columns']}"
        f"/{discrimination['static_metric_cap_vs_nemo_post']['n_wet_columns']}; "
        "live e3u substitution: "
        f"{discrimination['live_metric_cap_vs_nemo_post']['n_diverged_columns']}"
        f"/{discrimination['live_metric_cap_vs_nemo_post']['n_wet_columns']}")
    print(f"DISPOSITION: {disposition}; owner={artifact['owner']}")
    return 30 if owner or first is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
