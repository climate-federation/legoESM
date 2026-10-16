"""Round-13 step 4: free run of the OFFLINE-pretrained network as the closure,
no online training, CPU/libm, from rest.

The driver's ``free-run`` command needs a training run (manifest, log, the
window's last checkpoint) and admits only a trained closure block, so this
probe steps the card through the same closure path by import:
``driver.free_run_day`` (swg_online_step at every RK3 stage, zero stage
noise, noise scale 0, the double-gyre driver's daily-mean stepper) from the
card's rest state seeded as ``driver.free_run`` seeds it, one compiled day
per model day, and writes every day's end-of-day snapshot and daily mean in
the same day-file format (same keys, dtypes and labels; the file-write block
is ``driver.free_run``'s).  The manifest is this probe's own schema (its
closure is not one the held-out scorer admits): the pretrained file, its
SHA-256, the network spec, the device block, the namelist, the numerical
health and every day file's digest.

Reported every REPORT_DAYS days: the basin KE of the end-of-day snapshot,
0.5*(mean u^2 over wet U faces + mean v^2 over wet V faces) [m2 s-2]
(truth_ke_reference's definition), the peak speed max(|u|, |v|) over wet
faces, and finiteness; beside it the uncorrected nu=10 control's KE at the
same days from its stored day files (the same definition) and the smoothed
truth's KE level from the round-11 KE time-series receipt.
Controls, asserted: the pretrained file's digest equals the one the
pretraining receipt recorded; the first day's u/v are finite.
Usage: SWGONL_PRETRAIN=<round-13 dir> free_run_pretrained.py [--days N]
Writes <SWGONL_PRETRAIN>/free_run_pretrained/day*.npz, manifest.json,
<SWGONL_PRETRAIN>/free_run_pretrained_ke.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from legoesm.ml.checkpoint_io import load_checkpoint_or_fail
from legoesm.training.swg_online import (
    DAILY_MEAN_FIELDS,
    DAILY_MEAN_LABEL,
    DUAL_SCHEMA,
    SNAPSHOT_FIELDS,
    SNAPSHOT_LABEL,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    build_closure_geometry,
    create_swg_optimizer,
)

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "config/swg_online/pretrain_template_ring_weighted.json"
CONTROL = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round9_pilot"
               "/controls/swgonl_control_nu10_gpu")
KE_RECEIPT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds"
                  "/round11_pilot_ring_weighted/ke_timeseries_truths_vs_trained.json")
SMOOTHED_TRUTH = "smoothed truth (50 km, the training target)"
DAYS = 3 * 365
REPORT_DAYS = 30
SCHEMA = "legoesm-swg-online-offline-pretrained-free-run-probe-v1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ke_and_peak(u, v, umask, vmask):
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    ke = 0.5 * (np.sum(u**2 * umask) / umask.sum() + np.sum(v**2 * vmask) / vmask.sum())
    peak = max(np.abs(u[umask]).max(), np.abs(v[vmask]).max())
    return float(ke), float(peak)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=DAYS)
    args = parser.parse_args(argv)
    out_root = Path(os.environ["SWGONL_PRETRAIN"])
    output = out_root / "free_run_pretrained"
    driver = _load("run_swg_online_training", REPO / "scripts/run/run_swg_online_training.py")
    from legoesm.ocean.fidelity.provenance import git_sha
    commit = git_sha(allow_dirty=True)
    device = driver.select_device("cpu", "libm", None)
    config = driver.load_config(CONFIG)
    receipt = json.loads((out_root / "offline_pretrained_ours.json").read_text())
    pretrained = Path(config["network"]["initial_checkpoint"])
    assert pretrained == Path(receipt["checkpoint"]["path"]), (pretrained, receipt["checkpoint"])
    digest = driver._sha256(pretrained)
    assert digest == receipt["checkpoint"]["sha256"], "pretrained file differs from its receipt"

    card = driver.build_deck_card(config["deck"])
    geometry = build_closure_geometry(card)
    spec, _ = driver.network_spec(config, card, geometry)
    skeleton = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(config["seeds"][0]))
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    network, _ = load_checkpoint_or_fail(
        pretrained, (skeleton, optimizer.init(eqx.filter(skeleton, eqx.is_inexact_array))),
        what="offline-pretrained closure")
    umask = np.asarray(geometry.umask, dtype=bool)
    vmask = np.asarray(geometry.vmask, dtype=bool)

    dg = driver._double_gyre_driver()
    per_day = driver.steps_per_day(card)
    namelist = driver.free_run_namelist(card, args.days)
    key = jax.random.PRNGKey(config["seeds"][0])   # inert: zero stage noise, scale 0
    forcing, one_day = driver.free_run_day(network, card, geometry, key=key)
    state = dg.seed_steady_forcing_carry(card.recipe.initial_state, forcing)
    output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA, "legoesm_git_sha": commit,
        **{k: device[k] for k in ("device", "backend", "precision_policy", "math_library",
                                  "xla_flags", "jax_version", "gpu_name")},
        "days": args.days, "steps_per_day": per_day, "start": "rest",
        "integration": "driver.free_run_day: one jitted scan per model day of swg_online_step "
                       "(certified SWE RK3 step, closure potential at every stage, zero stage "
                       "noise, noise scale 0) in the double-gyre driver's daily-mean stepper",
        "sampling": driver.FREE_RUN_SAMPLING,
        "namelist": namelist._asdict(),
        "closure": {"kind": "offline_pretrained", "online_training": "none",
                    "config": str(CONFIG), "config_sha256": driver._sha256(CONFIG),
                    "checkpoint_path": str(pretrained), "checkpoint_sha256": digest,
                    "pretraining_receipt": str(out_root / "offline_pretrained_ours.json"),
                    "network_spec": driver._jsonable(spec._asdict()),
                    "objective": config["objective"], "wall_ring": config["wall_ring"]},
        "numerical_health": None,
    }
    pending = output / "manifest.pending.json"
    hashes = {}
    series = []
    health = None
    started = time.perf_counter()
    for day in range(1, args.days + 1):
        state, means = one_day(state)
        jax.block_until_ready(state.eta.data)
        daily = dg.numerical_health(state, namelist)
        health = daily if health is None else {
            "all_finite": health["all_finite"] and daily["all_finite"],
            "minimum_total_layer_thickness_m": min(health["minimum_total_layer_thickness_m"],
                                                   daily["minimum_total_layer_thickness_m"]),
            "maximum_u_courant": max(health["maximum_u_courant"], daily["maximum_u_courant"]),
            "maximum_v_courant": max(health["maximum_v_courant"], daily["maximum_v_courant"])}
        manifest["numerical_health"] = health
        snapshot = (state.eta.data, state.u.data[:, 1:, 0], state.v.data[1:, :, 0])
        name = f"day{day:04d}.npz"   # driver.free_run's day file, verbatim
        temporary = output / f".{name}.tmp"
        with temporary.open("wb") as stream:
            np.savez(stream, schema=np.str_(DUAL_SCHEMA), legoesm_git_sha=np.str_(commit),
                     day=np.int64(day), absolute_day=np.int64(day), step=np.int64(day * per_day),
                     dt_s=np.float64(card.dt_s), snapshot_sampling=np.str_(SNAPSHOT_LABEL),
                     daily_mean_sampling=np.str_(DAILY_MEAN_LABEL),
                     daily_mean_sample_count=np.int64(per_day),
                     **{k: np.asarray(v, dtype=np.float64) for k, v
                        in zip(SNAPSHOT_FIELDS + DAILY_MEAN_FIELDS, snapshot + tuple(means), strict=True)})
        temporary.replace(output / name)
        hashes[name] = driver._sha256(output / name)
        ke, peak = ke_and_peak(snapshot[1], snapshot[2], umask, vmask)
        series.append({"day": day, "ke_m2_s2": ke, "peak_speed_m_s": peak,
                       "finite": bool(daily["all_finite"])})
        if day == 1:
            assert daily["all_finite"], "day 1 is not finite"
        if day % REPORT_DAYS == 0 or day == args.days:
            print(f"day {day:4d} KE {ke:.5f} peak {peak:.3f} m/s finite {daily['all_finite']} "
                  f"({time.perf_counter() - started:.0f} s)")
            sys.stdout.flush()
            dg._write_hashes_atomic(output / "daily.sha256", hashes)
            driver._write_json(pending, manifest | {"status": "pending", "completed_days": day,
                                                    "daily_sha256": hashes})
            _write_series(out_root, series, receipt, umask, vmask, args.days)
        if not daily["all_finite"]:
            print("NOT FINITE: stopping")
            break
    driver._write_json(pending, manifest | {
        "status": "complete" if health["all_finite"] and day == args.days else "blew_up",
        "completed_days": day, "daily_sha256": hashes,
        "execution_seconds": time.perf_counter() - started})
    pending.replace(output / "manifest.json")
    print("WROTE", output)


def _write_series(out_root, series, receipt, umask, vmask, days):
    control = []
    for day in range(REPORT_DAYS, days + 1, REPORT_DAYS):
        with np.load(CONTROL / f"day{day:04d}.npz") as f:
            ke, peak = ke_and_peak(f["u_m_s"], f["v_m_s"], umask, vmask)
        control.append({"day": day, "ke_m2_s2": ke, "peak_speed_m_s": peak})
    truth = json.loads(KE_RECEIPT.read_text())[SMOOTHED_TRUTH]
    out = {
        "definition": __doc__,
        "ke_definition": "0.5*(mean u^2 over wet U faces + mean v^2 over wet V faces) of the "
                         "end-of-day snapshot [m2 s-2]; peak = max |u|,|v| over wet faces",
        "pretrained_closure": {"checkpoint": receipt["checkpoint"], "daily": series,
                               "every_30_days": [row for row in series if row["day"] % REPORT_DAYS == 0]},
        "uncorrected_control_nu10": {"source": str(CONTROL), "every_30_days": control},
        "smoothed_truth": {"source": str(KE_RECEIPT), "label": SMOOTHED_TRUTH,
                           "years": truth["years"], "ke_m2_s2": truth["ke_m2_s2"],
                           "mean_ke_m2_s2": float(np.mean(truth["ke_m2_s2"]))},
    }
    (out_root / "free_run_pretrained_ke.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
