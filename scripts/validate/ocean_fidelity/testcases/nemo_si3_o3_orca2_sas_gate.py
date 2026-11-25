#!/usr/bin/env python3
"""SI3 Rung O3 (`si3_r9`) offline score: NEMO's initial HFN thickness-distribution fit.

Replays the ADMITTED O2 record (`si3_rounds/round7/oracle_o2_orca2_sas_ice`), no
new acquisition.  O2's own scalar state (already 0 ULP) feeds the O3 fit
`si3_o3_istate_fill_fields` (`iceistate.f90:356-434`, `icevar.f90:1182-1327`),
then `ice_var_glo2eqv(1)`/`ice_var_agg(1)` (`icestp.f90:336-337`), and every
fitted per-category array, layer field, equivalent and aggregate is compared
with NEMO's `entry` snapshot written at the end of `ice_init`.  Production
`jax.jit` with every input passed as an argument (no constant folding), CPU,
binary64, glibc `exp` (`transcendentals="libm"`), and eager parity.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
import legoesm.ice.si3_orca2_state as si3_state_module
from legoesm.ice.si3_orca2_state import (
    SI3ORCA2O2Forcing,
    si3_o1_step,
    si3_o2_scalar_initial_state,
    si3_o2_step,
    si3_update_snwice_mass,
    si3_o3_istate_fill_fields,
    si3_var_agg,
    si3_var_glo2eqv,
)
from scripts.validate.ocean_fidelity.testcases import nemo_si3_o2_orca2_sas_gate as o2_gate
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o1_orca2_sas_gate import (
    _compare_si3_states_wet_and_full,
    max_normalized_diff,
    max_ulp_diff,
    o1_snapshot_to_si3_state,
    read_o1_binary_snapshot,
    stitch_o1_binary_snapshots,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o2_orca2_sas_gate import (
    o2_istate_snapshot_to_config,
    read_o2_istate_snapshot,
    stitch_o2_istate_snapshots,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
O2_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round7/oracle_o2_orca2_sas_ice"
)
_INPUT_NAMES = (
    "sst_m", "sss_m", "tmask", "ssmask", "smask0_i", "e1e2t", "ff_t", "gphit",
    "ssh_before_kbb", "ssh_before_kmm",
)
_PROG = ("a_i", "v_i", "v_s", "sv_i", "t_su", "oa_i", "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i")
_AGG_O2 = ("vt_i", "vt_s", "at_i", "at_ip", "vt_ip", "vt_il", "snwice_mass", "snwice_mass_b")
PLANTS = ("1ulp-v_i", "skip-snow-transfer", "skip-amax", "native-exp")
# The fit outputs O2's own sums and mass read (`si3_o2_scalar_initial_state`, step 5).
_FIT_FEEDS_O2 = ("v_i", "v_s", "a_i", "a_ip", "v_ip", "v_il")
# Entry fields ice_istate's fit does not produce; the O3-entry step keeps NEMO's.
RECORDED_ENTRY_FIELDS = ("u_ice", "v_ice", "cnd_ice", "t_si", "snwice_fmass", "prather", "bounds")


def _fit_perturbed(fill_fn):
    """Wrap the fit so its O2-feeding outputs grow by 2**-10 (relative) in the first interior ice column.

    Not one ULP: a one-ULP change of one category can round away in the
    five-category sum, and a small change of one column can round away in the
    global SSH mean; either would make the dependence test vacuous.
    """
    def perturbed(*args, **kwargs):
        o3 = fill_fn(*args, **kwargs)
        h = int(kwargs["config"].nn_hls)
        col = jnp.any(o3.v_i > 0.0, axis=-1)
        col = col & jnp.zeros(col.shape, dtype=bool).at[h:-h, h:-h].set(True)
        hit = jnp.zeros(col.size, dtype=bool).at[jnp.argmax(col.reshape(-1))].set(True).reshape(col.shape)
        return o3._replace(**{
            k: jnp.where(hit[..., None], getattr(o3, k) * (1.0 + 2.0**-10), getattr(o3, k))
            for k in _FIT_FEEDS_O2
        })
    return perturbed


def _o3_entry(inputs: tuple[jax.Array, ...], o2_cfg, cfg, plant: str | None):
    kw = dict(zip(_INPUT_NAMES, inputs))
    o2 = si3_o2_scalar_initial_state(**kw, config=o2_cfg, state_config=cfg)
    if plant in ("skip-snow-transfer", "skip-amax"):
        o3 = si3_o3_istate_fill_fields(
            **{k: getattr(o2, k) for k in (
                "zht_i_ini", "zht_s_ini", "zat_i_ini", "ztm_i_ini", "ztm_s_ini", "zt_su_ini",
                "zsm_i_ini", "zapnd_ini", "zhpnd_ini", "zhlid_ini", "rn_amax_2d")},
            tmask=kw["tmask"], sss_m=kw["sss_m"], config=cfg,
            _plant_skip_snow_transfer=plant == "skip-snow-transfer",
            _plant_skip_amax=plant == "skip-amax",
        )
        o2 = o2._replace(o3=o3)
    o3 = o2.o3
    oa_i = jnp.zeros_like(o3.a_i)
    eqv = si3_var_glo2eqv(
        v_i=o3.v_i, v_s=o3.v_s, sv_i=o3.sv_i, a_i=o3.a_i, oa_i=oa_i, a_ip=o3.a_ip,
        v_ip=o3.v_ip, v_il=o3.v_il, e_s=o3.e_s, e_i=o3.e_i, szv_i=o3.szv_i, kn=1,
        hi_max_top=cfg.rn_himax, sss_m=kw["sss_m"], config=cfg,
    )
    t_si = jnp.zeros_like(o3.a_i)
    agg = si3_var_agg(
        v_i=o3.v_i, v_s=eqv.v_s, sv_i=o3.sv_i, a_i=eqv.a_i, oa_i=oa_i, a_ip=o3.a_ip,
        v_ip=o3.v_ip, v_il=o3.v_il, t_su=o3.t_su, t_si=t_si, e_s=o3.e_s, e_i=o3.e_i,
        szv_i=o3.szv_i, a_ip_eff=eqv.a_ip_eff, t_i=eqv.t_i, t_s=eqv.t_s, kn=1, config=cfg,
    )
    prog = {
        "a_i": eqv.a_i, "v_i": o3.v_i, "v_s": eqv.v_s, "sv_i": o3.sv_i, "t_su": o3.t_su,
        "oa_i": oa_i, "a_ip": o3.a_ip, "v_ip": o3.v_ip, "v_il": o3.v_il, "e_s": o3.e_s,
        "e_i": o3.e_i, "szv_i": o3.szv_i,
    }
    return prog, eqv, agg, {k: getattr(o2, k) for k in _AGG_O2}, o3.t1_ice


def _region_masks(tmask_np: np.ndarray, h: int) -> dict[str, np.ndarray]:
    wet = tmask_np != 0.0
    ii, jj = np.indices(wet.shape)
    seam = ((ii < h) | (ii >= wet.shape[0] - h)) & (jj >= h) & (jj < wet.shape[1] - h)
    fold = jj >= wet.shape[1] - h
    interior = (ii >= h) & (ii < wet.shape[0] - h) & (jj >= h) & (jj < wet.shape[1] - h)
    return {"wet": wet, "interior": wet & interior, "seam": wet & seam, "fold": wet & fold}


def _score(ref: np.ndarray, act: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, Any]:
    ref = np.asarray(ref, dtype=np.float64)
    act = np.asarray(act, dtype=np.float64)
    if ref.shape != act.shape:
        raise ValueError(f"STATUS PLANT-FIRED: shape {act.shape} != NEMO {ref.shape}")
    out: dict[str, Any] = {"full_ulp": max_ulp_diff(ref, act)}
    for name, m2 in regions.items():
        m = np.broadcast_to(m2.reshape(m2.shape + (1,) * (ref.ndim - 2)), ref.shape)
        out[f"{name}_ulp"] = max_ulp_diff(ref, act, mask=m.astype(np.float64))
    wet = np.broadcast_to(regions["wet"].reshape(regions["wet"].shape + (1,) * (ref.ndim - 2)), ref.shape)
    out["wet_norm"] = max_normalized_diff(ref, act, mask=wet.astype(np.float64))
    bad = np.argwhere((ref.view(np.int64) != act.view(np.int64)) & wet)
    out["first_wet_diff"] = bad[0].tolist() if len(bad) else None
    return out


def score_o3(evidence_dir: str | Path = O2_RECORD, plant: str | None = None) -> dict[str, Any]:
    """Score the O3 fit against the admitted O2 record; raise on any nonzero wet or full ULP."""
    if plant is not None and plant not in PLANTS:
        raise ValueError(f"unknown plant {plant!r}")
    set_policy(PrecisionPolicy.fp64(transcendentals="native" if plant == "native-exp" else "libm"))
    ev = Path(evidence_dir)
    adm = ev / "o2_admission.json"
    if not adm.is_file() or json.loads(adm.read_text())["status"] != "PASS":
        raise ValueError(f"STATUS PLANT-FIRED: {adm} is not an admitted O2 record")
    rec = ev / "rec_2step" / "o2_si3_records"
    ist = stitch_o2_istate_snapshots([read_o2_istate_snapshot(p) for p in sorted(rec.glob("o2_si3_istate_rank*.bin"))])
    ent = stitch_o1_binary_snapshots([read_o1_binary_snapshot(p) for p in sorted(rec.glob("o2_si3_state_entry_rank*.bin"))])
    if not ist["fields"] or not ent["fields"]:
        raise ValueError("STATUS PLANT-FIRED: missing O2 istate/entry snapshot")
    entry, cfg = o1_snapshot_to_si3_state(ent)
    o2_cfg = o2_istate_snapshot_to_config(ist)._replace(ln_pnd=cfg.ln_pnd, ln_pnd_lids=cfg.ln_pnd_lids)
    f = ist["fields"]
    inputs = tuple(jnp.asarray(f[n], dtype=jnp.float64) for n in _INPUT_NAMES)

    fn = lambda *a: _o3_entry(a, o2_cfg, cfg, plant)  # noqa: E731
    jit_out = jax.jit(fn)(*inputs)
    eager_out = fn(*inputs)
    leaves_j = jax.tree_util.tree_leaves(jit_out)
    leaves_e = jax.tree_util.tree_leaves(eager_out)
    jit_eager_ulp = max(max_ulp_diff(a, b) for a, b in zip(leaves_j, leaves_e))

    prog, eqv, agg, o2agg, t1_ice = jit_out
    if plant == "1ulp-v_i":
        v = np.asarray(prog["v_i"]).copy()
        idx = tuple(np.argwhere(v > 0.0)[0])
        v[idx] = np.nextafter(v[idx], np.inf)
        prog = {**prog, "v_i": jnp.asarray(v)}

    regions = _region_masks(np.asarray(f["tmask"]), cfg.nn_hls)
    rows: dict[str, Any] = {}
    for k in _PROG:
        rows[k] = _score(getattr(entry, k), prog[k], regions)
    for k in eqv._fields:
        rows[f"equivalent.{k}"] = _score(getattr(entry.equivalent, k), getattr(eqv, k), regions)
    for k in agg._fields:
        rows[f"aggregated.{k}"] = _score(getattr(entry.aggregated, k), getattr(agg, k), regions)
    for k in _AGG_O2:
        rows[f"o2.{k}"] = _score(f[k], o2agg[k], regions)
    # `iceistate.f90:411-414` defines t1_ice on A2D(0) only; its halos are not scored.
    t1 = _score(entry.t1_ice, t1_ice, {"wet": regions["interior"]})
    rows["t1_ice"] = {**t1, "full_ulp_unscored_halos": t1.pop("full_ulp"), "full_ulp": 0}

    worst = max((r["full_ulp"], r["wet_ulp"]) for r in rows.values())
    failing = [k for k, r in rows.items() if r["full_ulp"] or r["wet_ulp"]]
    if failing or jit_eager_ulp:
        raise ValueError(
            f"STATUS PLANT-FIRED: O3 non-bit fields {failing[:6]} (first {failing[:1] and rows[failing[0]]}), "
            f"jit_vs_eager_ULP={jit_eager_ulp}"
        )
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "status": "PASS",
        "evidence_dir": str(ev),
        "o2_admission_sha256": hashlib.sha256(adm.read_bytes()).hexdigest(),
        "scorer_head": head,
        "execution_backend": jax.default_backend(),
        "execution_x64": bool(jax.config.jax_enable_x64),
        "execution_jit": True,
        "transcendentals": "libm",
        "jit_vs_eager_max_ulp": jit_eager_ulp,
        "arrays_scored": len(rows),
        "max_ulp_wet_full": list(worst),
        "wet_t_cells": {k: int(v.sum()) for k, v in regions.items()},
        "rows": rows,
    }


def o2_admission_fit_dependence(evidence_dir: str | Path = O2_RECORD) -> dict[str, Any]:
    """Run the O2 admission twice, the second time with the fit's output perturbed.

    Every comparison's candidate is captured (the comparators are replaced by
    recorders, so neither run can refuse) and reported as fit-dependent iff a
    bit of it moved.
    """
    def run(bump: bool) -> list[tuple[str, dict[str, np.ndarray]]]:
        cap: list[tuple[str, dict[str, np.ndarray]]] = []
        names = iter(("kt1", "kt2", "rst_entry", "rst_step1", "rst_vs_kt2"))

        def rec_o2(ref_fields, ref_area, ref_zsshadj, actual, tmask, **_kw):
            scored = tuple(getattr(o2_gate, "_O2_SCALAR_STATE_FIELDS")) + ("area", "zsshadj")
            cap.append(("o2_istate", {k: np.asarray(getattr(actual, k)) for k in scored}))
            return {"max_ulp_wet": 0, "max_ulp_full": 0, "max_norm_wet": 0.0, "area": 0.0, "zsshadj": 0.0}

        def rec_state(expected, actual, *_a, **_kw):
            leaves = jax.tree_util.tree_flatten_with_path(actual)[0]
            cap.append((next(names), {jax.tree_util.keystr(p): np.asarray(v) for p, v in leaves}))
            return {"max_ulp_wet": 0, "max_ulp_full": 0, "max_norm_wet": 0.0}

        def rec_restart(candidate, _raw, mask=None):
            cap.append(("restart_variables", {str(len(cap)): np.asarray(candidate)}))
            return 0

        patches = [
            mock.patch.object(o2_gate, "_compare_o2_scalar_state_against_recipe_and_ref", rec_o2),
            mock.patch.object(o2_gate, "_compare_si3_states_wet_and_full", rec_state),
            mock.patch.object(o2_gate, "max_ulp_diff", rec_restart),
        ]
        if bump:
            patches.append(mock.patch.object(
                si3_state_module, "si3_o3_istate_fill_fields",
                _fit_perturbed(si3_state_module.si3_o3_istate_fill_fields),
            ))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            o2_gate.admit_evidence_directory(evidence_dir)
        return cap

    base, bumped = run(False), run(True)
    if [n for n, _ in base] != [n for n, _ in bumped]:
        raise ValueError("STATUS PLANT-FIRED: the two admission runs made different comparisons")
    out: dict[str, Any] = {}
    for (name, a), (_, b) in zip(base, bumped):
        moved = sorted(k for k in a if a[k].tobytes() != b[k].tobytes())
        row = out.setdefault(name, {"candidates": 0, "moved": []})
        row["candidates"] += len(a)
        row["moved"] += moved if name != "restart_variables" else (["any"] if moved else [])
    for row in out.values():
        row["depends_on_fit"] = bool(row["moved"])
    return out


def score_o3_entry_steps(evidence_dir: str | Path = O2_RECORD, *, bump_fit: bool = False) -> dict[str, Any]:
    """Step kt 1 and kt 2 from the entry state O3 builds, scored against NEMO's steps.

    The fit-built fields replace the recorded entry; RECORDED_ENTRY_FIELDS stay NEMO's.
    """
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    ev = Path(evidence_dir)
    rec = ev / "rec_2step" / "o2_si3_records"
    snaps = {
        tag: stitch_o1_binary_snapshots([read_o1_binary_snapshot(p) for p in sorted(rec.glob(f"o2_si3_state_{tag}_rank*.bin"))])
        for tag in ("entry", "step0001", "step0002")
    }
    ist = stitch_o2_istate_snapshots([read_o2_istate_snapshot(p) for p in sorted(rec.glob("o2_si3_istate_rank*.bin"))])
    entry, cfg = o1_snapshot_to_si3_state(snaps["entry"])
    o2_cfg = o2_istate_snapshot_to_config(ist)._replace(ln_pnd=cfg.ln_pnd, ln_pnd_lids=cfg.ln_pnd_lids)
    f = ist["fields"]
    inputs = tuple(jnp.asarray(f[n], dtype=jnp.float64) for n in _INPUT_NAMES)
    sss = tuple(jnp.asarray(snaps[t]["fields"]["sss_m"], dtype=jnp.float64) for t in ("entry", "step0002"))

    # Every array, recorded entry included, is a jit argument (no constant folding).
    def two_steps(rec_entry, sss_entry, sss_step2, *a):
        kw = dict(zip(_INPUT_NAMES, a))
        prog, eqv, agg, _o2agg, t1_ice = _o3_entry(a, o2_cfg, cfg, None)
        # ice_init re-sets the mass after ice_istate from the aggregates
        # (`iceupdate.f90:482-483`, start-from-rest arm of update_rst('READ')).
        mass, mass_b, _ = si3_update_snwice_mass(
            vt_i=agg.vt_i, vt_s=agg.vt_s, vt_ip=agg.vt_ip, vt_il=agg.vt_il,
            snwice_mass_prev=jnp.zeros_like(agg.vt_i), tmask=kw["tmask"], config=cfg, is_init=True,
        )
        o3_entry = rec_entry._replace(
            **prog, equivalent=eqv, aggregated=agg, t1_ice=t1_ice, snwice_mass=mass, snwice_mass_b=mass_b,
        )
        forcing = SI3ORCA2O2Forcing(
            tmask=kw["tmask"], sst_m=kw["sst_m"], sss_m=sss_entry, ff_t=kw["ff_t"], gphit=kw["gphit"],
            e1e2t=kw["e1e2t"], ssmask=kw["ssmask"], smask0_i=kw["smask0_i"],
            ssh_before_kbb=kw["ssh_before_kbb"], ssh_before_kmm=kw["ssh_before_kmm"], apply_halos=False,
        )
        kt1, _ = si3_o2_step(o3_entry, forcing, config=cfg, o2_init_config=o2_cfg)
        return mass, mass_b, kt1, si3_o1_step(kt1, tmask=kw["tmask"], sss_m=sss_step2, config=cfg)

    with contextlib.ExitStack() as stack:
        if bump_fit:
            stack.enter_context(mock.patch.object(
                si3_state_module, "si3_o3_istate_fill_fields",
                _fit_perturbed(si3_state_module.si3_o3_istate_fill_fields),
            ))
        mass, mass_b, kt1, kt2 = jax.jit(two_steps)(entry, *sss, *inputs)
    kw = dict(zip(_INPUT_NAMES, inputs))
    tm = kw["tmask"]
    um = jnp.asarray(f["umask"], dtype=jnp.float64)
    vm = jnp.asarray(f["vmask"], dtype=jnp.float64)
    out = {"label": "O3-FIT-ENTRY", "recorded_entry_fields": list(RECORDED_ENTRY_FIELDS), "entry_mass_ulp": 0}
    for tag, mine in (("kt1", kt1), ("kt2", kt2)):
        nemo, _ = o1_snapshot_to_si3_state(snaps[f"step000{tag[-1]}"], config=cfg)
        c = _compare_si3_states_wet_and_full(nemo, mine, tm, um, vm, check_fmass=True, check_prather=True)
        if c["max_ulp_wet"] or c["max_ulp_full"]:
            raise ValueError(f"STATUS PLANT-FIRED: O3-entry {tag} wet_ULP={c['max_ulp_wet']} full_ULP={c['max_ulp_full']}")
        out[f"max_ulp_{tag}"] = [c["max_ulp_wet"], c["max_ulp_full"]]
    regions = _region_masks(np.asarray(f["tmask"]), cfg.nn_hls)
    for k, mine in (("snwice_mass", mass), ("snwice_mass_b", mass_b)):
        r = _score(getattr(entry, k), mine, regions)
        if r["full_ulp"] or r["wet_ulp"]:
            raise ValueError(f"STATUS PLANT-FIRED: O3-entry {k} {r}")
    out["status"] = "PASS"
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SI3 Rung O3 offline score against the admitted O2 record.")
    p.add_argument("--evidence", type=Path, default=O2_RECORD)
    p.add_argument("--plant", choices=PLANTS, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--fit-dependence", action="store_true", help="which O2-admission comparisons read the fit")
    p.add_argument("--entry-steps", action="store_true", help="step kt 1-2 from the O3-built entry state")
    p.add_argument("--bump-fit", action="store_true", help="with --entry-steps: perturb the fit output (must refuse)")
    a = p.parse_args(argv)
    try:
        if a.fit_dependence:
            report = {"status": "MEASURED", "comparisons": o2_admission_fit_dependence(a.evidence)}
        elif a.entry_steps:
            report = score_o3_entry_steps(a.evidence, bump_fit=a.bump_fit)
        else:
            report = score_o3(a.evidence, plant=a.plant)
    except Exception as exc:
        print(f"REFUSE: GATE_FAIL: {exc}", file=sys.stderr)
        return 1
    if a.json is not None:
        a.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
