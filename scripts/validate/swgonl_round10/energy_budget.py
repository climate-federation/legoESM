"""Round-10 step 4: closure, wind, viscous and drag work on the flow per unit mass.

Every term is the model's own stage-3 program differenced against itself at
the step's entry state (swe_rk3_stage with entry == before_entry, as stage 1
is called): closure on minus off, wind on minus off (stress zeroed in the
entry), viscosity on minus off (A_h = 0 in the config), drag on minus off
(bottom_drag_r = 0).  Work per unit mass = mean over wet U faces of u*a_u
plus mean over wet V faces of v*a_v [m2 s-3], the KE-per-unit-mass
convention of the operator's probe (0.5*(mean u^2 + mean v^2)).

Two products: (a) instantaneous samples on stored end-of-day snapshots of
the named runs; (b) hourly means from CPU/libm reintegration of the named
spans (the free-run stepper, one hour = 9 steps per jitted call).
Usage: SWGONL_RUN=<run dir> SWGONL_DIAG=<output dir> energy_budget.py snapshots | spans
The run directory (seed directory holding run_manifest.json, checkpoints/,
dataset_contract.json, training_log.json) and the output directory are
required: there is no default run, so the probe refuses to start without
them.  The run's configuration and seed are read from its own run manifest.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ml.checkpoint_io import load_checkpoint_or_fail
from legoesm.ocean.dynamics.swe_rk3 import _entry_from_state, swe_rk3_stage
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    build_swg_card,
    swg_surface_wind,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.training.swg_online import (
    SWGDatasetContract,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    build_closure_geometry,
    coarse_state_from_native,
    create_swg_optimizer,
    swg_online_step,
)

REPO = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/claude-swgonl-r10")
# Round-9 root: the no-closure control runs the step-4 snapshot mode compares against.
R9 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round9_pilot")


def _required_dir(variable):
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(f"{variable} is not set: name the directory explicitly "
                         "(no default run; refusing to guess)")
    path = Path(value)
    if not path.is_dir():
        raise SystemExit(f"{variable}={value} is not a directory")
    return path


RUN = _required_dir("SWGONL_RUN")
DIAG = _required_dir("SWGONL_DIAG")
MANIFEST = json.loads((RUN / "run_manifest.json").read_text())
SEED = int(MANIFEST["seed"]["value"])
STEPS_PER_HOUR = 9


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup():
    assert jax.default_backend() == "cpu"
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    driver = _load("run_swg_online_training",
                   REPO / "scripts/run/run_swg_online_training.py")
    config = MANIFEST["config"]   # the run's own (driver-validated) configuration
    card = build_swg_card(SWG_DG50_NAMELIST)
    geometry = build_closure_geometry(card)
    spec, _ = driver.network_spec(config, card, geometry)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    key = jax.random.PRNGKey(SEED)
    nets = {}
    for w in (1, 2):
        skeleton = SWGScalarPotentialCNN(spec, key=key)
        nets[w], _ = load_checkpoint_or_fail(
            RUN / "checkpoints" / f"w{w:02d}_e05.eqx",
            (skeleton, optimizer.init(eqx.filter(skeleton, eqx.is_inexact_array))),
            what="pilot closure")
    tau_u, tau_v = swg_surface_wind(card.resolved_namelist)
    forcing = OceanSurfaceForcing(tau_i_native=jnp.asarray(tau_u, dtype=jnp.float64),
                                  tau_j_native=jnp.asarray(tau_v, dtype=jnp.float64))
    return driver, config, card, geometry, nets, forcing, key


def make_terms(card, geometry, network, forcing, key, rings=None):
    """(ssh, u, v) -> dict of work terms [m2 s-3] and KE [m2 s-2] at that state.

    rings: optional (ring_u, ring_v) boolean face masks (implied_subgrid_work.
    face_rings); when given, the closure work is also split into the wall ring
    and the interior, both normalised by the full wet-face counts so that
    ring + interior == total.
    """
    template = card.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    grid, z_coord, cfg = card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
    mask = template.land_mask.data
    tau = (forcing.tau_i_native, forcing.tau_j_native)
    zero_tau = (jnp.zeros_like(tau[0]), jnp.zeros_like(tau[1]))
    cfg_no_visc = cfg._replace(lateral_viscosity=cfg.lateral_viscosity._replace(A_h=0.0))
    cfg_no_drag = cfg._replace(bottom_drag=cfg.bottom_drag._replace(bottom_drag_r=0.0))
    umask = np.asarray(geometry.umask, dtype=bool)
    vmask = np.asarray(geometry.vmask, dtype=bool)
    n_u, n_v = int(umask.sum()), int(vmask.sum())
    dt = card.dt_s   # stage 3's stage_dt = dt / 1

    def stage3(entry, potential, config):
        out = swe_rk3_stage(entry, entry, dt, 3, grid, z_coord, mask, config,
                            potential_f=potential)
        return out.u_k, out.v_k

    def terms(ssh, u, v):
        state = coarse_state_from_native(template, ssh, u, v)
        entry = _entry_from_state(state, tau, tau, grid, z_coord, cfg)
        entry_no_wind = entry._replace(utau_before=zero_tau[0], vtau_before=zero_tau[1],
                                       utau_now=zero_tau[0], vtau_now=zero_tau[1])
        base_u, base_v = stage3(entry, None, cfg)
        if network is None:
            clo_u, clo_v = base_u, base_v
        else:
            potential = network(entry, geometry, key=jax.random.fold_in(key, 1),
                                noise=jnp.zeros_like(entry.ssh_k),
                                noise_scale=jnp.asarray(0.0))
            clo_u, clo_v = stage3(entry, potential, cfg)
        nw_u, nw_v = stage3(entry_no_wind, None, cfg)
        nv_u, nv_v = stage3(entry, None, cfg_no_visc)
        nd_u, nd_v = stage3(entry, None, cfg_no_drag)

        def work(au, av, sel_u=umask, sel_v=vmask):
            return (jnp.sum(u * au * sel_u) / n_u + jnp.sum(v * av * sel_v) / n_v)

        def accel(with_u, with_v, without_u, without_v):
            return (with_u - without_u) / dt, (with_v - without_v) / dt

        a_clo = accel(clo_u, clo_v, base_u, base_v)
        a_wind = accel(base_u, base_v, nw_u, nw_v)
        a_visc = accel(base_u, base_v, nv_u, nv_v)
        a_drag = accel(base_u, base_v, nd_u, nd_v)
        out = {
            "closure_work": work(*a_clo), "wind_work": work(*a_wind),
            "viscous_work": work(*a_visc), "drag_work": work(*a_drag),
            "closure_accel_max": jnp.maximum(jnp.max(jnp.abs(a_clo[0])), jnp.max(jnp.abs(a_clo[1]))),
            "closure_accel_rms": jnp.sqrt(0.5 * (jnp.sum(a_clo[0]**2 * umask) / n_u
                                                 + jnp.sum(a_clo[1]**2 * vmask) / n_v)),
            "ke": 0.5 * (jnp.sum(u**2 * umask) / n_u + jnp.sum(v**2 * vmask) / n_v),
            "max_speed": jnp.maximum(jnp.max(jnp.abs(u)), jnp.max(jnp.abs(v))),
        }
        if rings is not None:
            ring_u, ring_v = rings
            out["closure_work_ring"] = work(*a_clo, ring_u, ring_v)
            out["closure_work_interior"] = work(*a_clo, umask & ~ring_u, vmask & ~ring_v)
        return out

    return terms


def snapshots(driver, card, geometry, nets, forcing, key):
    """Instantaneous terms on stored end-of-day states, every `stride` days."""
    runs = {
        "pilot_w01_from_rest": (RUN / "free_runs/w01", nets[1], 10),
        "pilot_w02_from_rest": (RUN / "free_runs/w02", nets[2], 10),
        "control_nu10_from_rest": (R9 / "controls/swgonl_control_nu10_gpu", None, 10),
        "w01_from_day4265": (DIAG / "free_runs_day4265/w01", nets[1], 10),
        "w02_from_day4265": (DIAG / "free_runs_day4265/w02", nets[2], 10),
        "control_nu10_from_day4265": (DIAG / "free_runs_day4265/swgonl_control_nu10_gpu_day4265", None, 10),
    }
    out = {"units": {"work": "m2 s-3 (per unit mass, domain mean over wet faces)",
                     "ke": "m2 s-2", "closure_accel": "m s-2"},
           "definition": __doc__, "runs": {}}
    for name, (root, net, stride) in runs.items():
        if not (root / "manifest.json").is_file():
            print("skip (no manifest)", name)
            continue
        fn = jax.jit(make_terms(card, geometry, net, forcing, key))
        days = min(1095, json.loads((root / "manifest.json").read_text())["completed_days"])
        rows = []
        for day in list(range(1, days + 1, stride)) + ([days] if days % stride != 1 else []):
            with np.load(root / f"day{day:04d}.npz") as f:
                values = fn(jnp.asarray(f["ssh_m"]), jnp.asarray(f["u_m_s"]), jnp.asarray(f["v_m_s"]))
            rows.append({"day": day, **{k: float(v) for k, v in values.items()}})
        out["runs"][name] = rows
        print(name, "days", len(rows), "closure_work day1..", [round(r["closure_work"], 12) for r in rows[:3]])
        sys.stdout.flush()
    (DIAG / "step4_energy_budget_snapshots.json").write_text(json.dumps(out, indent=1))
    print("WROTE snapshots")


def spans(driver, config, card, geometry, nets, forcing, key):
    """Hourly-mean terms over reintegrated spans (CPU/libm, free-run stepper)."""
    stage_noise = jnp.zeros((3,) + tuple(geometry.tmask.shape), jnp.float64)
    template = card.recipe.initial_state

    def hour_program(network):
        terms = make_terms(card, geometry, network, forcing, key)

        def step(carry, _):
            state, acc = carry
            values = terms(state.eta.data, state.u.data[:, 1:, 0], state.v.data[1:, :, 0])
            state = swg_online_step(network, state, stage_noise, key, jnp.asarray(0.0),
                                    card=card, geometry=geometry, forcing=forcing)
            acc = {k: acc[k] + values[k] / STEPS_PER_HOUR for k in acc}
            return (state, acc), None

        @jax.jit
        def one_hour(state):
            zero = {k: jnp.asarray(0.0) for k in (
                "closure_work", "wind_work", "viscous_work", "drag_work",
                "closure_accel_max", "closure_accel_rms", "ke", "max_speed")}
            ke0 = 0.5 * (jnp.sum(state.u.data[:, 1:, 0]**2 * geometry.umask) / jnp.sum(geometry.umask)
                         + jnp.sum(state.v.data[1:, :, 0]**2 * geometry.vmask) / jnp.sum(geometry.vmask))
            (state, acc), _ = jax.lax.scan(step, (state, zero), None, length=STEPS_PER_HOUR)
            ke1 = 0.5 * (jnp.sum(state.u.data[:, 1:, 0]**2 * geometry.umask) / jnp.sum(geometry.umask)
                         + jnp.sum(state.v.data[1:, :, 0]**2 * geometry.vmask) / jnp.sum(geometry.vmask))
            acc["dke_dt"] = (ke1 - ke0) / 3600.0
            return state, acc

        return one_hour

    def seeded(ssh, u, v):
        return driver._double_gyre_driver().seed_steady_forcing_carry(
            coarse_state_from_native(template, ssh, u, v), forcing)

    def snapshot_state(root, day):
        with np.load(root / f"day{day:04d}.npz") as f:
            return seeded(f["ssh_m"], f["u_m_s"], f["v_m_s"])

    contract = SWGDatasetContract.from_json(RUN / "dataset_contract.json", card=card)
    batch = next(contract.batches(split="train", window_days=2, epoch=0, seed=SEED))
    train_start = seeded(*(np.asarray(value[0]) for value in batch.initial))
    day4265 = driver.load_config(REPO / "config/swg_online/pilot_template_ring_excluded_day4265.json")
    truth_state, truth_block = driver.resolve_start(day4265, card)
    truth_start = driver._double_gyre_driver().seed_steady_forcing_carry(truth_state, forcing)
    rest = driver._double_gyre_driver().seed_steady_forcing_carry(template, forcing)
    jobs = [
        ("w02_training_window_day734_first_2d", nets[2], train_start, 48,
         {"start": "first training batch window 0, rebalanced", "start_absolute_hour": int(batch.start_absolute_hour[0])}),
        ("w02_from_day4265_days_1_30", nets[2], truth_start, 720,
         {"start": "record state day 4265, rebalanced", "block": truth_block}),
        ("control_nu10_from_day4265_days_1_30", None, truth_start, 720,
         {"start": "record state day 4265, rebalanced"}),
        ("w02_from_rest_days_1_30", nets[2], rest, 720, {"start": "rest"}),
        ("w02_from_rest_days_400_500", nets[2], snapshot_state(RUN / "free_runs/w02", 399), 2400,
         {"start": "pilot w02 end-of-day snapshot day 399 (GPU/native run), continued on CPU/libm"}),
        ("w02_from_rest_days_550_650", nets[2], snapshot_state(RUN / "free_runs/w02", 549), 2400,
         {"start": "pilot w02 end-of-day snapshot day 549 (GPU/native run), continued on CPU/libm"}),
    ]
    out = {"units": {"work": "m2 s-3", "ke": "m2 s-2", "closure_accel": "m s-2",
                     "dke_dt": "m2 s-3 (KE change over the hour / 3600 s)"},
           "sampling": "hourly means of per-step values at each step's entry state; 9 steps of 400 s per hour",
           "definition": __doc__, "spans": {}}
    programs = {}
    for name, net, state, hours, meta in jobs:
        tag = "none" if net is None else id(net)
        if tag not in programs:
            programs[tag] = hour_program(net)
        one_hour = programs[tag]
        rows = []
        for hour in range(1, hours + 1):
            state, acc = one_hour(state)
            rows.append({"hour": hour, **{k: float(v) for k, v in acc.items()}})
            if not np.isfinite(rows[-1]["ke"]):
                print("NONFINITE at", name, hour)
                break
        out["spans"][name] = {"meta": meta, "hourly": rows}
        daily = lambda k, lo, hi: float(np.mean([r[k] for r in rows[lo:hi]]))
        print(name, f"hours {hours}: closure {daily('closure_work', 0, hours):.3e} wind {daily('wind_work', 0, hours):.3e} "
              f"visc {daily('viscous_work', 0, hours):.3e} drag {daily('drag_work', 0, hours):.3e} "
              f"dKE/dt {daily('dke_dt', 0, hours):.3e} KE_end {rows[-1]['ke']:.4f}")
        sys.stdout.flush()
        (DIAG / "step4_energy_budget_spans.json").write_text(json.dumps(out, indent=1))
    print("WROTE spans")


if __name__ == "__main__":
    driver, config, card, geometry, nets, forcing, key = setup()
    if sys.argv[1] == "snapshots":
        snapshots(driver, card, geometry, nets, forcing, key)
    else:
        spans(driver, config, card, geometry, nets, forcing, key)
