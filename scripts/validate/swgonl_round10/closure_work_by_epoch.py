"""Round-10 step 6: did training reward the closure's energy injection?

Closure work on the flow per unit mass [m2 s-3] (closure on minus off at the
same entry state, the model's own stage-3 program: energy_budget.make_terms,
reused unchanged) for the initial network and each of the ten epoch
checkpoints, on fifteen real window starts: the first window of each of the
first ten training batches of the w=1 stage (epoch 0 sampler, the one the
training loop drew for epoch 1) and the first five validation-window starts
(the fixed chronological lattice), all rebalanced by the contract's own
start_state.  Per network: mean/median closure work, fraction of starts with
positive work, closure acceleration rms, next to that epoch's logged train
and validation loss; the wind work alongside as the constant reference; the
Pearson correlation across the eleven networks between mean closure work and
validation loss.

The closure work is also split into the first interior wall ring and the
interior (implied_subgrid_work.face_rings: wet faces with either adjoining T
cell in the outermost wet T row/column; both parts normalised by the full
wet-face counts, so ring + interior == total).

Controls: a zero-potential closure must give exactly 0.0 work and the
initial network's acceleration ratio is re-measured with the driver's own
gate (initial_closure_acceleration_ratio) on the same ten training starts.
Usage: SWGONL_RUN=<run dir> SWGONL_DIAG=<output dir> closure_work_by_epoch.py
(both required, see energy_budget).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import energy_budget as eb  # noqa: E402
from implied_subgrid_work import face_rings  # noqa: E402
from legoesm.ml.checkpoint_io import load_checkpoint_or_fail  # noqa: E402
from legoesm.training.swg_online import (  # noqa: E402
    SWGDatasetContract,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    create_swg_optimizer,
    initial_closure_acceleration_ratio,
)

OUT = eb.DIAG / "step6_closure_work_by_epoch.json"
N_TRAIN_BATCHES = 10
N_VALIDATION_STARTS = 5


def networks(driver, config, card, geometry):
    """{'init': fresh net, 'w01_e01': ..., 'w02_e05': ...} in training order."""
    spec, _ = driver.network_spec(config, card, geometry)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    key = jax.random.PRNGKey(eb.SEED)
    nets = {"init": SWGScalarPotentialCNN(spec, key=key)}
    for w in (1, 2):
        for e in range(1, 6):
            skeleton = SWGScalarPotentialCNN(spec, key=key)
            nets[f"w{w:02d}_e{e:02d}"], _ = load_checkpoint_or_fail(
                eb.RUN / "checkpoints" / f"w{w:02d}_e{e:02d}.eqx",
                (skeleton, optimizer.init(eqx.filter(skeleton, eqx.is_inexact_array))),
                what="pilot closure")
    return nets


def starts(card):
    """[(label, ssh, u, v)]: ten training starts then five validation starts."""
    contract = SWGDatasetContract.from_json(eb.RUN / "dataset_contract.json", card=card)
    rows, first_batch = [], None
    train = contract.batches(split="train", window_days=1, epoch=0, seed=eb.SEED,
        target_interval_hours=1)
    for index in range(N_TRAIN_BATCHES):
        batch = next(train)
        if first_batch is None:
            first_batch = batch
        rows.append((f"train_batch{index:02d}_win0_h{int(batch.start_absolute_hour[0])}",
                     *(np.asarray(value[0]) for value in batch.initial)))
    batch = next(contract.batches(split="validation", window_days=1, target_interval_hours=1))
    for index in range(N_VALIDATION_STARTS):
        rows.append((f"validation_win{index}_h{int(batch.start_absolute_hour[index])}",
                     *(np.asarray(value[index]) for value in batch.initial)))
    return rows, first_batch


def main():
    driver, config, card, geometry, _, forcing, key = eb.setup()
    log = json.loads((eb.RUN / "training_log.json").read_text())
    losses = {"init": {"train_loss_mean": None, "validation_loss": None}}
    for entry in log["epochs"]:
        losses[f"w{entry['window_days']:02d}_e{entry['epoch']:02d}"] = {
            "train_loss_mean": entry["train_loss_mean"],
            "validation_loss": entry["validation_loss"]}
    nets = networks(driver, config, card, geometry)
    rows, first_batch = starts(card)
    labels = [row[0] for row in rows]
    print("starts:", labels)

    # Control 1: zero potential -> exactly 0.0 closure work and acceleration.
    probe = nets["init"]
    shape = None

    def zero_potential(entry, geometry, *, key, noise, noise_scale):
        return jnp.zeros(shape, dtype=jnp.float64)

    # The potential's shape: ask the network once on the first start.
    from legoesm.ocean.dynamics.swe_rk3 import _entry_from_state
    from legoesm.training.swg_online import coarse_state_from_native
    template = card.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    tau = (forcing.tau_i_native, forcing.tau_j_native)
    entry0 = _entry_from_state(coarse_state_from_native(template, *(jnp.asarray(a) for a in rows[0][1:])),
                               tau, tau, card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    shape = tuple(probe(entry0, geometry, key=key, noise=jnp.zeros_like(entry0.ssh_k),
                        noise_scale=jnp.asarray(0.0)).shape)
    zero_fn = jax.jit(eb.make_terms(card, geometry, zero_potential, forcing, key))
    zero_vals = [zero_fn(*(jnp.asarray(a) for a in row[1:])) for row in rows]
    zero_work = [float(v["closure_work"]) for v in zero_vals]
    zero_accel = [float(v["closure_accel_max"]) for v in zero_vals]
    assert all(w == 0.0 for w in zero_work) and all(a == 0.0 for a in zero_accel), (zero_work, zero_accel)
    print("control zero potential: closure_work", set(zero_work), "accel_max", set(zero_accel))

    # Control 2: the driver's own initial-acceleration gate on the ten training starts.
    gate = initial_closure_acceleration_ratio(
        nets["init"], driver.batch_start_states(card, first_batch), card, geometry, key=key)
    print(f"control init gate: max ratio {gate['maximum_ratio']:.4%} "
          f"(logged {log['gates']['initial_closure_acceleration']['maximum_ratio']:.4%})")

    ring_u, ring_v, _, _ = face_rings(geometry)
    per_network = {}
    for name, net in nets.items():
        fn = jax.jit(eb.make_terms(card, geometry, net, forcing, key, rings=(ring_u, ring_v)))
        samples = []
        for label, ssh, u, v in rows:
            values = fn(jnp.asarray(ssh), jnp.asarray(u), jnp.asarray(v))
            samples.append({"start": label, **{k: float(x) for k, x in values.items()}})
        clo = np.array([s["closure_work"] for s in samples])
        ring = np.array([s["closure_work_ring"] for s in samples])
        interior = np.array([s["closure_work_interior"] for s in samples])
        assert np.allclose(ring + interior, clo, rtol=1e-9, atol=1e-30), name
        wind = np.array([s["wind_work"] for s in samples])
        rms = np.array([s["closure_accel_rms"] for s in samples])
        train_mask = np.array([s["start"].startswith("train") for s in samples])
        summary = {
            **losses[name],
            "closure_work_mean": float(clo.mean()),
            "closure_work_median": float(np.median(clo)),
            "closure_work_mean_train_starts": float(clo[train_mask].mean()),
            "closure_work_mean_validation_starts": float(clo[~train_mask].mean()),
            "closure_work_positive_fraction": float((clo > 0).mean()),
            "closure_work_ring_mean": float(ring.mean()),
            "closure_work_interior_mean": float(interior.mean()),
            "closure_accel_rms_mean": float(rms.mean()),
            "closure_over_wind_work_mean": float(clo.mean() / wind.mean()),
            "wind_work_mean": float(wind.mean()),
            "wind_work_median": float(np.median(wind)),
            "wind_work_positive_fraction": float((wind > 0).mean()),
            "samples": samples,
        }
        per_network[name] = summary
        print(f"{name:8s} clo mean {summary['closure_work_mean']:+.3e} "
              f"(ring {summary['closure_work_ring_mean']:+.3e} int {summary['closure_work_interior_mean']:+.3e}) "
              f"med {summary['closure_work_median']:+.3e} "
              f"pos {summary['closure_work_positive_fraction']:.2f} a_rms {summary['closure_accel_rms_mean']:.3e} "
              f"clo/wind {summary['closure_over_wind_work_mean']:+.3f} "
              f"train {summary['train_loss_mean']} val {summary['validation_loss']}")
        sys.stdout.flush()

    trained = [n for n in per_network if n != "init"]
    means = np.array([per_network[n]["closure_work_mean"] for n in trained])
    vals = np.array([per_network[n]["validation_loss"] for n in trained])
    trains = np.array([per_network[n]["train_loss_mean"] for n in trained])
    # Eleven networks: the initial network has no logged loss, so its loss is
    # the pre-training value; the task asks for eleven, so also report the
    # ten-trained-only correlation, which needs no imputation.
    corr_10 = float(np.corrcoef(means, vals)[0, 1])
    corr_10_train = float(np.corrcoef(means, trains)[0, 1])
    # The w=1 and w=2 losses are different objectives (window length), so
    # the pooled correlation mixes a scale change with training; within-stage
    # correlations (five epochs each) are the comparison that is one variable.
    within = {}
    for w in ("w01", "w02"):
        pick = [i for i, n in enumerate(trained) if n.startswith(w)]
        within[w] = {"pearson_vs_validation_loss": float(np.corrcoef(means[pick], vals[pick])[0, 1]),
                     "pearson_vs_train_loss": float(np.corrcoef(means[pick], trains[pick])[0, 1])}
    out = {
        "units": {"work": "m2 s-3 (per unit mass, mean over wet faces)", "closure_accel": "m s-2"},
        "definition": __doc__,
        "machinery": "energy_budget.make_terms (stage-3 program differenced closure on/off), CPU/libm fp64",
        "run": str(eb.RUN),
        "ring_choice": "wet face with either adjoining T cell in the outermost wet T row/column "
                       "(implied_subgrid_work.face_rings); ring and interior work normalised by the "
                       "full wet-face counts, ring + interior == total",
        "starts": labels,
        "controls": {
            "zero_potential_closure_work": zero_work,
            "zero_potential_closure_accel_max": zero_accel,
            "initial_network_gate": gate,
            "initial_network_gate_logged_maximum_ratio":
                log["gates"]["initial_closure_acceleration"]["maximum_ratio"],
        },
        "per_network": per_network,
        "pearson_mean_closure_work_vs_validation_loss_10_trained": corr_10,
        "pearson_mean_closure_work_vs_train_loss_10_trained": corr_10_train,
        "pearson_within_stage": within,
        "note": "the initial network has no logged loss (not evaluated by the driver), so the "
                "correlation is over the ten checkpoints; eleven networks are tabulated",
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"pearson(mean closure work, validation loss) over 10 trained nets: {corr_10:+.3f}; vs train loss {corr_10_train:+.3f}")
    print("within-stage pearson(mean closure work, validation loss):",
          {w: round(v["pearson_vs_validation_loss"], 3) for w, v in within.items()})
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
