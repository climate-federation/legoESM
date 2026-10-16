"""Round-13 step 1: continue the offline fit of OUR closure architecture to
convergence and save it in the training driver's checkpoint format.

Picks up capacity_test's fit of architecture (a) (SWGScalarPotentialCNN, 16
channels, 4 hidden layers, 3x3, Eq. 6 scales; 7,696 weights) from its stored
weights (fit_ours_c16_h4_k3.eqx, which was still improving at the 3,000-step
cap) and keeps fitting it on the same capacity dataset (dataset.npz: 500
training and 100 validation matched starts, target = the representable
one-hour correction a_S) with capacity_test.fit: the same loss, optimiser
(Adam, fresh moments), rate, batch, evaluation cadence and validation early
stopping (best validation weights kept, PATIENCE steps without improvement),
only the step cap raised to MAX_STEPS.

Reported: the metrics of the warm start (asserted equal to capacity_test.json's
stored numbers: the continuation starts from the stored fit) and of the
converged network, per split and region (capacity_test.metrics).
Controls, each asserted: the warm-started network's production-path
acceleration equals the convolution-primitive path's (relative 1e-12) and the
saved checkpoint, loaded through the driver's loader into the driver's
skeleton, holds the fitted weights leaf for leaf.

Writes <SWGONL_DIAG>/offline_pretrained_ours.eqx ((network, fresh optimiser
state) as the driver's checkpoints), offline_pretrained_ours.json.
Usage: SWGONL_RUN=<pilot-2 seed dir> SWGONL_DIAG=<round-13 dir> pretrain_offline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity_test as ct  # noqa: E402
import energy_budget as eb  # noqa: E402
import implied_subgrid_work as isw  # noqa: E402
from legoesm.ml.training import save_checkpoint  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d  # noqa: E402
from legoesm.ocean.dynamics.swe_rk3 import swe_rotated_potential_forcing  # noqa: E402
from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands  # noqa: E402
from legoesm.training.swg_online import (  # noqa: E402
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    ShiftedConv2d,
    create_swg_optimizer,
    network_parameter_count,
)

REPO = Path(__file__).resolve().parents[3]   # this clone's driver, not energy_budget's
CAPACITY = eb.R9.parent / "round11_pilot_ring_weighted/capacity_test"
ARCH = "ours_c16_h4_k3"
MAX_STEPS = 30_000   # the only setting changed from capacity_test (its cap of 3,000 bound)
START_TOL = 1e-9     # warm-start metrics vs the stored capacity_test.json numbers


def main():
    eb.REPO = REPO
    driver, config, card, geometry, nets, forcing, key = eb.setup()
    with np.load(CAPACITY / "dataset.npz") as f:
        data = {k: f[k] for k in f.files}
    stored = json.loads((CAPACITY / "capacity_test.json").read_text())
    assert stored["dataset"]["path"] == str(CAPACITY / "dataset.npz")

    ring_u, ring_v, umask, vmask = isw.face_rings(geometry)
    n_u, n_v = int(umask.sum()), int(vmask.sum())
    vtx = np.asarray(geometry.vertex_interior, dtype=bool)
    grid, z_coord = card.recipe.grid, card.recipe.z_coord
    umask3, vmask3 = compute_face_masks_3d(z_coord.is_active, grid)
    ops = nemo_qco_resolved_mesh_operands(z_coord, grid, umask3, vmask3, jnp.float64, 1)
    operator_masks = (ops.e2u, ops.e1v, umask3[:, 1:, 0], vmask3[1:, :, 0], jnp.asarray(vtx))
    train = np.flatnonzero(data["split"] == "train")
    val = np.flatnonzero(data["split"] == "validation")
    assert train.size == ct.N_TRAIN and val.size == ct.N_VAL
    regions = {"all": np.ones(n_u + n_v, dtype=bool),
               "ring": np.concatenate([ring_u[umask], ring_v[vmask]]),
               "interior": np.concatenate([(umask & ~ring_u)[umask], (vmask & ~ring_v)[vmask]])}

    def pack(au, av):
        return np.concatenate([np.asarray(au)[:, umask], np.asarray(av)[:, vmask]], axis=1)

    @jax.jit
    def accel(phi, e3u, e3v):
        return jax.vmap(lambda p, a, b: swe_rotated_potential_forcing(p, *operator_masks[:2], a, b,
                                                                      *operator_masks[2:]))(phi, e3u, e3v)

    target = pack(data["rep_u"], data["rep_v"])
    a2 = float(np.mean(target[train] ** 2))
    jd = {k: jnp.asarray(data[k]) for k in ("ssh", "u", "v", "e3u", "e3v")}
    base_spec, _ = driver.network_spec(config, card, geometry)
    index = list(ct.ARCHS).index(ARCH)
    net = ct.make_network(ARCH, base_spec, jax.random.fold_in(key, 100 + index))
    assert net.spec == base_spec, "architecture (a) is the driver's configured network"
    net = eqx.tree_deserialise_leaves(CAPACITY / f"fit_{ARCH}.eqx", net)
    run = ct.make_runner(accel, geometry, key)

    def table(pred):
        return {split: {region: ct.metrics(pred[idx][:, sel], target[idx][:, sel])
                        for region, sel in regions.items()}
                for split, idx in (("train", train), ("validation", val))}

    # Warm start = the stored fit, production convolution path.
    production_call = ShiftedConv2d.__call__
    pred_production = pack(*ct.predict(net, jd, run))
    start = table(pred_production)
    worst = max(abs(start[s][r][m] - stored["metrics"][ARCH][s][r][m])
                for s in start for r in start[s] for m in start[s][r])
    print("warm start vs capacity_test.json max abs metric difference", worst)
    assert worst <= START_TOL, worst
    ShiftedConv2d.__call__ = ct.conv_primitive_call
    run = ct.make_runner(accel, geometry, key)
    pred_primitive = pack(*ct.predict(net, jd, run))
    controls = {"warm_start_metrics_vs_stored_max_abs": worst,
                "conv_primitive_vs_production_max_rel": float(np.max(
                    np.linalg.norm(pred_primitive - pred_production, axis=1)
                    / np.linalg.norm(pred_production, axis=1)))}
    assert controls["conv_primitive_vs_production_max_rel"] <= 1e-12, controls

    ct.MAX_STEPS = MAX_STEPS
    t0 = time.time()
    net_fit, curve = ct.fit(net, jd, target, train, val, a2, accel, run, pack, geometry, key, None)
    wall = time.time() - t0
    ShiftedConv2d.__call__ = production_call
    run = ct.make_runner(accel, geometry, key)
    final = table(pack(*ct.predict(net_fit, jd, run)))   # production path
    print(ct.render({"warm_start": start, "converged": final}))

    # The driver's checkpoint format: (network, optimiser state), a fresh
    # optimiser of the experiment's settings.
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    path = eb.DIAG / "offline_pretrained_ours.eqx"
    save_checkpoint((net_fit, optimizer.init(eqx.filter(net_fit, eqx.is_inexact_array))), path)
    skeleton = SWGScalarPotentialCNN(base_spec, key=jax.random.PRNGKey(eb.SEED))
    loaded, _ = driver.load_checkpoint_or_fail(
        path, (skeleton, optimizer.init(eqx.filter(skeleton, eqx.is_inexact_array))), what="pretrained")
    for mine, theirs in zip(jax.tree.leaves(eqx.filter(net_fit, eqx.is_inexact_array)),
                            jax.tree.leaves(eqx.filter(loaded, eqx.is_inexact_array)), strict=True):
        assert np.array_equal(np.asarray(mine), np.asarray(theirs))
    controls["saved_checkpoint_reloads_leaf_for_leaf"] = True

    out = {
        "definition": __doc__,
        "architecture": {**ct.ARCHS[ARCH], "parameters": network_parameter_count(net_fit),
                         "spec": driver._jsonable(base_spec._asdict())},
        "warm_start": {"weights": str(CAPACITY / f"fit_{ARCH}.eqx"),
                       "weights_sha256": driver._sha256(CAPACITY / f"fit_{ARCH}.eqx"),
                       "stored_fit": stored["fits"][ARCH] | {"curve": "see capacity_test.json"}},
        "dataset": {"path": str(CAPACITY / "dataset.npz"),
                    "sha256": driver._sha256(CAPACITY / "dataset.npz"),
                    "n_train": int(train.size), "n_validation": int(val.size)},
        "fit_settings": {"optimizer": "optax.adam (fresh moments at the warm start)",
                         "learning_rate": ct.LR, "batch": ct.BATCH, "max_steps": MAX_STEPS,
                         "eval_every": ct.EVAL_EVERY, "patience_steps": ct.PATIENCE,
                         "batch_order_seed": eb.SEED, "loss": stored["fit_settings"]["loss"],
                         "convolutions": "XLA primitive during the fit (capacity_test control e); "
                                         "production ShiftedConv2d for every reported metric"},
        "fit": curve | {"wall_s": wall, "converged": curve["steps_run"] < MAX_STEPS},
        "metrics": {"warm_start": start, "converged": final},
        "controls": controls,
        "checkpoint": {"path": str(path), "sha256": driver._sha256(path),
                       "format": "(SWGScalarPotentialCNN, optax state of create_swg_optimizer("
                                 f"{json.dumps(config['optimiser'])}) at step 0)"},
        "legoesm_git_sha": _git_sha(),
    }
    (eb.DIAG / "offline_pretrained_ours.json").write_text(json.dumps(out, indent=1))
    print("controls", json.dumps(controls))
    print(f"FIT steps_run={curve['steps_run']} best_step={curve['best_step']} "
          f"best_val_loss={curve['best_val_loss']:.4f} wall_s={wall:.0f}")
    print("WROTE", path, out["checkpoint"]["sha256"])


def _git_sha():
    from legoesm.ocean.fidelity.provenance import git_sha
    return git_sha(allow_dirty=True)


if __name__ == "__main__":
    main()
