"""Offline capacity test: how much of the representable data-implied forcing
can each closure architecture capture when fitted with no model in the loop?

Dataset.  Window starts of the pilot-2 dataset contract: N_TRAIN starts
evenly spaced over the contract's training split and N_VAL over its
validation split (the two round-11 snapshot starts, train h26955 and
validation h102360, are forced into their sets so the online network's
numbers can be checked against the stored snapshot summary).  Every start is
the stored state at its hour, the start's sea level rebalanced by the
contract's own start_state (the training's launch state, where the closure
and its operator are evaluated), while the target is step 8's a_S from the
MATCHED un-rebalanced start: the coarse model stepped one hour from the
stored state (no closure, swg_online_step's free-run stepper) against the
stored truth one hour later, a_S = (truth_1h - model_1h) / 3600 s, then
projected exactly onto the range of the closure operator at the rebalanced
launch state (residual_decomposition.RangeProjector; the potential Phi on
the interior F points and the face accelerations A Phi are both kept).

Fits.  Three architectures, inputs and output scaled exactly as the training
driver (Eq. 6 analogue scales from driver.network_spec):
  (a) ours: SWGScalarPotentialCNN, 16 channels, 4 hidden layers, 3x3;
  (b) Maddison-sized: seven 5x5 convolutions of 32 filters (ELU) with
      trainable-scalar residual skips from every earlier hidden state, then
      a 1x1 linear projection to the potential (ResidualPotentialCNN here);
  (c) ours with wider reach: SWGScalarPotentialCNN, 16 channels, 6 hidden
      layers, 5x5.
Each predicts the F potential; the loss is the mean squared face
acceleration error through the closure's own rotated-gradient operator at
the sample's live face cross-sections (swe_rotated_potential_forcing),
normalised by the training-set mean squared target acceleration.  Adam
(optax.adam, constant rate LR), batch BATCH, at most MAX_STEPS steps, the
validation loss checked every EVAL_EVERY steps, the best validation weights
kept, stopped after PATIENCE steps without improvement.

Reported per architecture, per split, split into interior / first wall ring
(implied_subgrid_work.face_rings) / all wet faces: explained variance
(pooled over the split's faces and starts, 1 - SSE / SS about the pooled
mean), pattern correlation (mean over starts of the per-start Pearson over
the region's stacked U and V faces, the round-11 snapshot statistic) and the
amplitude ratio (mean over starts of rms(pred)/rms(target)).  The same for
the ONLINE-trained w=8 epoch-5 network evaluated at the same starts and for
the persistence baseline (the training-set mean target field).

Controls, each asserted:
  (a) the sparse projector's potential, pushed through the JAX operator the
      fits use, reproduces the projected acceleration (relative 1e-12);
  (b) the zero potential gives exactly zero acceleration through that
      operator on every start;
  (c) the online w=8 network's correlation and amplitude ratio at the two
      snapshot starts equal the round-11 snapshot summary (relative 1e-6);
  (d) a fit to a synthetic target inside the network class (the online w=8
      network's own acceleration on the same inputs) reaches validation
      correlation > 0.99 for every architecture (it proves the pipeline);
  (e) the fits run the convolutions through XLA's primitive (3x faster on
      the CPU than the production slice-stack): the online w=8 network
      evaluated that way reproduces the dataset's production-path
      acceleration (relative 1e-12).
Writes <SWGONL_DIAG>/dataset.npz, fit_<arch>.eqx, synthetic_<arch>.eqx,
capacity_test.json, capacity_figure.png.
Usage: SWGONL_RUN=<pilot-2 seed dir> SWGONL_DIAG=<output dir> capacity_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import energy_budget as eb  # noqa: E402
import implied_subgrid_work as isw  # noqa: E402
import residual_decomposition as rd  # noqa: E402
import subgrid_snapshots as ss  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d  # noqa: E402
from legoesm.ocean.dynamics.swe_rk3 import (  # noqa: E402
    SWERK3StageEntry,
    _entry_from_state,
    swe_rotated_potential_forcing,
)
from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands  # noqa: E402
from legoesm.training.swg_online import (  # noqa: E402
    SWGDatasetContract,
    SWGNetworkSpec,
    SWGScalarPotentialCNN,
    ShiftedConv2d,
    balanced_fields_from_native,
    coarse_state_from_native,
    network_parameter_count,
    stored_state_at,
    swg_online_step,
)

N_TRAIN, N_VAL = 500, 100
SNAPSHOT_STARTS = {"train": 26955, "validation": 102360}   # round-11 snapshot starts
BATCH, LR, MAX_STEPS, EVAL_EVERY, PATIENCE = 16, 1e-3, 3000, 50, 600
SYNTHETIC_TARGET_CORR = 0.995   # a synthetic fit stops once validation correlation passes this
SYNTHETIC_PASS_CORR = 0.99      # control (d) bar
ARCHS = {
    "ours_c16_h4_k3": {"kind": "plain", "hidden_channels": 16, "hidden_layers": 4, "kernel_size": 3},
    "maddison_c32_conv7_k5": {"kind": "residual", "hidden_channels": 32, "convolutions": 7, "kernel_size": 5},
    "ours_c16_h6_k5": {"kind": "plain", "hidden_channels": 16, "hidden_layers": 6, "kernel_size": 5},
}
FIELDS = ("ssh", "u", "v", "e3u", "e3v", "phi", "rep_u", "rep_v", "a_s_u", "a_s_v", "w08_u", "w08_v")


class ResidualPotentialCNN(eqx.Module):
    """Maddison-sized residual CNN with the SWGScalarPotentialCNN call contract.

    ``convolutions`` kernels of ``spec.kernel_size`` and ``spec.hidden_channels``
    filters (the first from the four input channels), ELU after each; hidden
    state k (k >= 1) is ELU(conv_k(h_{k-1})) plus trainable scalars times every
    earlier hidden state h_0..h_{k-1} (initialised to the plain residual
    h_{k-1}); a 1x1 bias-free projection gives the potential, scaled and
    masked as the production network.
    """

    convs: tuple
    skips: jax.Array
    proj: ShiftedConv2d
    spec: SWGNetworkSpec = eqx.field(static=True)

    def __init__(self, spec: SWGNetworkSpec, *, convolutions: int, key):
        keys = jax.random.split(key, convolutions + 1)
        width, k, pad = spec.hidden_channels, spec.kernel_size, spec.kernel_size // 2
        convs = [ShiftedConv2d(4, width, k, padding=pad, key=keys[0])]
        for layer_key in keys[1:convolutions]:
            convs.append(ShiftedConv2d(width, width, k, padding=pad, key=layer_key))
        self.convs = tuple(convs)
        # Flat lower triangle: entry k(k-1)/2 + j weights h_j (j < k) into h_k;
        # initialised to the plain residual (weight 1 on h_{k-1}), no inert leaf.
        self.skips = jnp.asarray([1.0 if j == k - 1 else 0.0
                                  for k in range(1, convolutions) for j in range(k)], dtype=jnp.float64)
        self.proj = ShiftedConv2d(width, 1, 1, padding=0, use_bias=False, key=keys[-1])
        self.spec = spec

    def __call__(self, entry, geometry, *, key, noise, noise_scale):
        value = input_channels(entry, geometry, self.spec, key, noise, noise_scale)
        hidden = [jax.nn.elu(self.convs[0](value))]
        for k, conv in enumerate(self.convs[1:], start=1):
            out = jax.nn.elu(conv(hidden[-1]))
            for j in range(k):
                out = out + self.skips[k * (k - 1) // 2 + j] * hidden[j]
            hidden.append(out)
        potential = self.proj(hidden[-1])[0]
        return self.spec.potential_scale_m3_s2 * potential * geometry.vertex_interior


def conv_primitive_call(self, value):
    """ShiftedConv2d's map through XLA's convolution: the same cross-correlation
    with the same weights (identical to round-off; asserted on the online
    network in main), 3x faster per fit step on the CPU for the 5x5 nets.  The
    production lane avoids the primitive for cross-device reproducibility; an
    offline fit has no such need."""
    if value.ndim != 3 or value.shape[0] != self.in_channels:
        raise ValueError("shifted convolution input must have shape (channels,y,x)")
    out = jax.lax.conv_general_dilated(
        value[None], self.weight, window_strides=(1, 1),
        padding=((self.padding, self.padding), (self.padding, self.padding)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))[0]
    return out if self.bias is None else out + self.bias


def input_channels(entry, geometry, spec, key, noise, noise_scale):
    """The production network's four input channels (SWGScalarPotentialCNN.__call__)."""
    u, v = jnp.asarray(entry.u_k), jnp.asarray(entry.v_k)
    u_west = jnp.concatenate([jnp.zeros_like(u[:, :1]), u[:, :-1]], axis=1)
    v_south = jnp.concatenate([jnp.zeros_like(v[:1]), v[:-1]], axis=0)
    keyed_noise = noise + noise_scale * jax.random.normal(key, noise.shape, noise.dtype)
    return jnp.stack((jnp.asarray(entry.ssh_k) / spec.ssh_scale_m + keyed_noise,
                      0.5 * (u + u_west) / spec.velocity_scale_m_s,
                      0.5 * (v + v_south) / spec.velocity_scale_m_s,
                      geometry.tmask.astype(u.dtype)), axis=0)


def make_network(name, base_spec, key):
    arch = ARCHS[name]
    spec = base_spec._replace(hidden_channels=arch["hidden_channels"], kernel_size=arch["kernel_size"],
                              hidden_layers=arch.get("hidden_layers", base_spec.hidden_layers))
    if arch["kind"] == "plain":
        return SWGScalarPotentialCNN(spec, key=key)
    if arch["kind"] == "residual":
        return ResidualPotentialCNN(spec, convolutions=arch["convolutions"], key=key)
    raise ValueError(f"unknown architecture kind {arch['kind']!r}")


def split_hours(contract):
    """Evenly spaced start hours per split, the snapshot start forced in."""
    bounds = {name: (lo, hi) for name, lo, hi in contract.splits}
    out = {}
    for split, n in (("train", N_TRAIN), ("validation", N_VAL)):
        lo, hi = bounds[split]
        hours = np.unique(np.round(np.linspace(lo, hi - 1, n)).astype(np.int64))   # target at +1 h stays inside
        assert hours.size == n, (split, hours.size)
        forced = SNAPSHOT_STARTS[split]
        hours[np.argmin(np.abs(hours - forced))] = forced
        out[split] = np.unique(hours)
        assert out[split].size == n and forced in out[split]
    return out


def build_dataset(driver, card, geometry, nets, forcing, key, contract, hours):
    ring_u, ring_v, umask, vmask = isw.face_rings(geometry)
    vtx = np.asarray(geometry.vertex_interior, dtype=bool)
    template = card.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    grid, z_coord, cfg = card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
    tau = (forcing.tau_i_native, forcing.tau_j_native)
    dt = card.dt_s
    steps_per_hour = int(isw.HOUR_S / dt)
    assert steps_per_hour * dt == isw.HOUR_S
    umask3, vmask3 = compute_face_masks_3d(z_coord.is_active, grid)
    ops = nemo_qco_resolved_mesh_operands(z_coord, grid, umask3, vmask3, jnp.float64, 1)
    e2u, e1v = np.asarray(ops.e2u, dtype=np.float64), np.asarray(ops.e1v, dtype=np.float64)

    def step_a(state, _):   # step 7's stepper, verbatim
        return swg_online_step(None, state, None, None, None, card=card,
                               geometry=geometry, forcing=forcing), None

    @jax.jit
    def hour_a(state):
        state, _ = jax.lax.scan(step_a, state, None, length=steps_per_hour)
        return state

    @jax.jit
    def at_state(ssh, u, v):
        """Live face thicknesses and the online w=8 acceleration at a state."""
        state = coarse_state_from_native(template, ssh, u, v)
        entry = _entry_from_state(state, tau, tau, grid, z_coord, cfg)
        one = jnp.asarray(1.0, dtype=jnp.float64)
        e3u = (ops.e3u_0 * (one + entry.r3u_k[..., None] * ops.umask3))[..., 0]
        e3v = (ops.e3v_0 * (one + entry.r3v_k[..., None] * ops.vmask3))[..., 0]
        potential = nets[8](entry, geometry, key=jax.random.fold_in(key, 1),
                            noise=jnp.zeros_like(entry.ssh_k), noise_scale=jnp.asarray(0.0))
        return e3u, e3v, swe_rotated_potential_forcing(
            potential, ops.e2u, ops.e1v, e3u, e3v, umask3[:, 1:, 0], vmask3[1:, :, 0], vtx)

    def native(state):
        return (np.asarray(state.eta.data), np.asarray(state.u.data[:, 1:, 0]),
                np.asarray(state.v.data[1:, :, 0]))

    def seeded(ssh, u, v):
        return driver._double_gyre_driver().seed_steady_forcing_carry(
            coarse_state_from_native(card.recipe.initial_state, ssh, u, v), forcing)

    n_u = int(umask.sum())
    data = {name: [] for name in FIELDS}
    meta = {"hour": [], "split": [], "normal_equation_residual": [], "rebalance_iterations": []}
    t0 = time.time()
    for split, split_hours_ in hours.items():
        for hour in split_hours_:
            hour = int(hour)
            ssh_s, u, v = (np.asarray(a, dtype=np.float64)
                           for a in stored_state_at(contract.sources, contract.objective, hour)[0])
            truth = [np.asarray(a, dtype=np.float64)
                     for a in stored_state_at(contract.sources, contract.objective, hour + 1)[0]]
            (ssh_r, u_r, v_r), report = contract.start_state(hour)
            assert np.array_equal(u, u_r) and np.array_equal(v, v_r), hour
            model_s = native(hour_a(seeded(ssh_s, u, v)))
            a_s = ((truth[1] - model_s[1]) / isw.HOUR_S * umask, (truth[2] - model_s[2]) / isw.HOUR_S * vmask)
            e3u, e3v, w_u, w_v = (np.asarray(x) for x in jax.tree.leaves(at_state(ssh_r, u, v)))
            A, _ = rd.rotated_gradient_matrix(e2u * e3u, e1v * e3v, umask, vmask, vtx)
            scale = float(np.median((e2u * e3u)[umask]))
            rep_vec, _, phi_vec, residual = rd.RangeProjector(A * scale)(
                np.concatenate([a_s[0][umask], a_s[1][vmask]]))
            phi = np.zeros(vtx.shape)
            phi[vtx] = phi_vec * scale   # (sA)^T(sA) Phi' = (sA)^T a  =>  Phi = s Phi'
            rep_u, rep_v = np.zeros(umask.shape), np.zeros(vmask.shape)
            rep_u[umask], rep_v[vmask] = rep_vec[:n_u], rep_vec[n_u:]
            for name, value in zip(FIELDS, (ssh_r, u, v, e3u, e3v, phi, rep_u, rep_v, a_s[0], a_s[1], w_u, w_v)):
                data[name].append(value)
            meta["hour"].append(hour)
            meta["split"].append(split)
            meta["normal_equation_residual"].append(residual)
            meta["rebalance_iterations"].append(report["iterations"])
            done = len(meta["hour"])
            if done % 50 == 0 or done == 1:
                print(f"dataset {done} ({(time.time() - t0) / done:.2f} s/start)")
                sys.stdout.flush()
    elapsed = time.time() - t0
    n_starts = len(meta["hour"])
    arrays = {name: np.stack(values) for name, values in data.items()}
    arrays["hour"] = np.asarray(meta["hour"], dtype=np.int64)
    arrays["split"] = np.asarray(meta["split"])
    arrays["normal_equation_residual"] = np.asarray(meta["normal_equation_residual"])
    arrays["rebalance_iterations"] = np.asarray(meta["rebalance_iterations"], dtype=np.int64)
    arrays["cost_s_per_start"] = np.asarray(elapsed / n_starts)
    arrays["cost_s_total"] = np.asarray(elapsed)
    return arrays


def main():
    driver, config, card, geometry, nets, forcing, key = eb.setup()
    contract = SWGDatasetContract.from_json(eb.RUN / "dataset_contract.json", card=card)
    hours = split_hours(contract)
    dataset_path = eb.DIAG / "dataset.npz"
    wanted = np.concatenate([hours["train"], hours["validation"]])
    if dataset_path.is_file() and np.array_equal(np.load(dataset_path)["hour"], wanted):
        with np.load(dataset_path) as f:
            data = {k: f[k] for k in f.files}
        print("dataset loaded", dataset_path)
    else:
        data = build_dataset(driver, card, geometry, nets, forcing, key, contract, hours)
        np.savez(dataset_path, **data)
        print(f"WROTE {dataset_path}: {float(data['cost_s_per_start']):.2f} s/start, "
              f"{float(data['cost_s_total']):.0f} s total")
    sys.stdout.flush()

    ring_u, ring_v, umask, vmask = isw.face_rings(geometry)
    n_u, n_v = int(umask.sum()), int(vmask.sum())
    vtx = np.asarray(geometry.vertex_interior, dtype=bool)
    grid, z_coord = card.recipe.grid, card.recipe.z_coord
    umask3, vmask3 = compute_face_masks_3d(z_coord.is_active, grid)
    ops = nemo_qco_resolved_mesh_operands(z_coord, grid, umask3, vmask3, jnp.float64, 1)
    operator_masks = (ops.e2u, ops.e1v, umask3[:, 1:, 0], vmask3[1:, :, 0], jnp.asarray(vtx))
    train = np.flatnonzero(data["split"] == "train")
    val = np.flatnonzero(data["split"] == "validation")
    assert train.size == N_TRAIN and val.size == N_VAL
    regions = {"all": np.ones(n_u + n_v, dtype=bool),
               "ring": np.concatenate([ring_u[umask], ring_v[vmask]]),
               "interior": np.concatenate([(umask & ~ring_u)[umask], (vmask & ~ring_v)[vmask]])}

    def pack(au, av):   # (N, ny, nx) x 2 -> (N, faces)
        return np.concatenate([np.asarray(au)[:, umask], np.asarray(av)[:, vmask]], axis=1)

    target = pack(data["rep_u"], data["rep_v"])
    online = pack(data["w08_u"], data["w08_v"])
    controls = {}

    # Control (a)/(b): the JAX operator the fits use vs the sparse projector; zero potential.
    @jax.jit
    def accel(phi, e3u, e3v):
        return jax.vmap(lambda p, a, b: swe_rotated_potential_forcing(p, *operator_masks[:2], a, b,
                                                                      *operator_masks[2:]))(phi, e3u, e3v)

    rep_jax = pack(*accel(jnp.asarray(data["phi"]), jnp.asarray(data["e3u"]), jnp.asarray(data["e3v"])))
    controls["jax_operator_vs_sparse_projection_max_rel"] = float(
        np.max(np.linalg.norm(rep_jax - target, axis=1) / np.linalg.norm(target, axis=1)))
    assert controls["jax_operator_vs_sparse_projection_max_rel"] <= 1e-12
    zero = pack(*accel(jnp.zeros_like(jnp.asarray(data["phi"])), jnp.asarray(data["e3u"]), jnp.asarray(data["e3v"])))
    controls["zero_potential_max_abs_acceleration"] = float(np.abs(zero).max())
    assert controls["zero_potential_max_abs_acceleration"] == 0.0
    # Control (c): the online w=8 network's snapshot numbers.
    snapshot = {r["absolute_hour"]: r["networks"]["w08"]
                for r in json.loads((eb.R9.parent / "round11_pilot_ring_weighted/subgrid_snapshots/summary.json")
                                    .read_text())["per_start"]}
    controls["online_w08_vs_snapshot_summary_max_rel"] = 0.0
    for hour in SNAPSHOT_STARTS.values():
        i = int(np.flatnonzero(data["hour"] == hour)[0])
        ref = snapshot[hour]
        mine = {k: ss.pearson(online[i][s], target[i][s]) for k, s in regions.items()}
        amp = rms(online[i]) / rms(target[i])
        worst = max([abs(mine[k] / ref["corr_vs_representable"][k] - 1.0) for k in regions]
                    + [abs(amp / ref["amplitude_ratio_rms"] - 1.0)])
        controls["online_w08_vs_snapshot_summary_max_rel"] = max(
            controls["online_w08_vs_snapshot_summary_max_rel"], worst)
        assert worst <= 1e-6, (hour, mine, ref)

    # Fits.
    base_spec, _ = driver.network_spec(config, card, geometry)
    a2 = float(np.mean(target[train] ** 2))
    jd = {k: jnp.asarray(data[k]) for k in ("ssh", "u", "v", "e3u", "e3v")}
    results = {"architectures": {}, "fits": {}, "synthetic_control": {}}
    predictions = {"persistence": np.broadcast_to(target[train].mean(axis=0), target.shape),
                   "online_w08_e05": online}
    run = make_runner(accel, geometry, key)
    # The dataset's online w=8 acceleration went through the production
    # ShiftedConv2d; the fits use the convolution primitive.  Same map:
    ShiftedConv2d.__call__ = conv_primitive_call
    online_primitive = pack(*predict(nets[8], jd, run))
    controls["conv_primitive_vs_shifted_online_w08_max_rel"] = float(
        np.max(np.linalg.norm(online_primitive - online, axis=1) / np.linalg.norm(online, axis=1)))
    assert controls["conv_primitive_vs_shifted_online_w08_max_rel"] <= 1e-12
    for index, name in enumerate(ARCHS):
        net = make_network(name, base_spec, jax.random.fold_in(key, 100 + index))
        results["architectures"][name] = {**ARCHS[name], "parameters": network_parameter_count(net)}
        print(name, "parameters", results["architectures"][name]["parameters"])
        for tag, tgt, stop_corr in (("fit", target, None), ("synthetic", online, SYNTHETIC_TARGET_CORR)):
            path = eb.DIAG / f"{tag}_{name}.eqx"
            curve_path = eb.DIAG / f"{tag}_{name}_curve.json"
            if path.is_file() and curve_path.is_file():
                net_fit = eqx.tree_deserialise_leaves(path, net)
                curve = json.loads(curve_path.read_text())
                print("loaded", path)
            else:
                net_fit, curve = fit(net, jd, tgt, train, val, a2, accel, run, pack, geometry, key, stop_corr)
                eqx.tree_serialise_leaves(path, net_fit)
                curve_path.write_text(json.dumps(curve))
            pred = pack(*predict(net_fit, jd, run))
            if tag == "fit":
                predictions[name] = pred
                results["fits"][name] = curve
            else:
                corr = {split: float(np.mean([ss.pearson(pred[i], tgt[i]) for i in idx]))
                        for split, idx in (("train", train), ("validation", val))}
                results["synthetic_control"][name] = {"corr_all": corr, "steps": curve["steps_run"],
                                                      "best_step": curve["best_step"]}
                print(name, "synthetic control corr", corr)
                assert corr["validation"] > SYNTHETIC_PASS_CORR, (name, corr)
            sys.stdout.flush()

    # Metrics.
    table = {}
    for model, pred in predictions.items():
        table[model] = {}
        for split, idx in (("train", train), ("validation", val)):
            table[model][split] = {region: metrics(pred[idx][:, sel], target[idx][:, sel])
                                   for region, sel in regions.items()}
    results.update({
        "definition": __doc__, "controls": controls,
        "dataset": {"path": str(dataset_path), "n_train": N_TRAIN, "n_validation": N_VAL,
                    "train_hours": [int(hours["train"][0]), int(hours["train"][-1])],
                    "validation_hours": [int(hours["validation"][0]), int(hours["validation"][-1])],
                    "contract_splits": {name: [lo, hi] for name, lo, hi in contract.splits},
                    "cost_s_per_start": float(data["cost_s_per_start"]),
                    "cost_s_total": float(data["cost_s_total"]),
                    "normal_equation_residual_max": float(data["normal_equation_residual"].max()),
                    "target_rms_m_s2": {"train": rms(target[train]), "validation": rms(target[val])},
                    "representable_fraction_of_a_S_rms": {
                        split: rms(target[idx]) / rms(pack(data["a_s_u"], data["a_s_v"])[idx])
                        for split, idx in (("train", train), ("validation", val))}},
        "fit_settings": {"optimizer": "optax.adam", "learning_rate": LR, "batch": BATCH, "max_steps": MAX_STEPS,
                         "eval_every": EVAL_EVERY, "patience_steps": PATIENCE,
                         "loss": "mean over batch of the face-mean squared acceleration error "
                                 "(U and V wet faces stacked) / training-set mean squared target",
                         "synthetic_stop_corr": SYNTHETIC_TARGET_CORR},
        "metrics": table, "online_reference": str(eb.RUN / "checkpoints/w08_e05.eqx"),
        "regions": "ring = wet face with either adjoining T cell in the outermost wet T row/column"})
    (eb.DIAG / "capacity_test.json").write_text(json.dumps(results, indent=1))
    figure(data, predictions, target, val, umask, vmask, geometry, eb.DIAG)
    print(render(table))
    print("controls", json.dumps(controls))
    print("WROTE", eb.DIAG / "capacity_test.json")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def metrics(pred, tgt):
    """Pooled explained variance, mean per-start Pearson, mean per-start rms ratio."""
    return {"explained_variance": float(1.0 - np.sum((pred - tgt) ** 2) / np.sum((tgt - tgt.mean()) ** 2)),
            "corr": float(np.mean([ss.pearson(p, t) for p, t in zip(pred, tgt)])),
            "amplitude_ratio": float(np.mean([rms(p) / rms(t) for p, t in zip(pred, tgt)]))}


def make_runner(accel, geometry, key):
    """One compiled (per architecture) batch forward: inputs -> face accelerations."""
    @eqx.filter_jit
    def run(net, ssh, u, v, e3u, e3v):
        phi = jax.vmap(lambda s, a, b: net(
            SWERK3StageEntry(s, a, b, *(None,) * 10), geometry, key=jax.random.fold_in(key, 1),
            noise=jnp.zeros_like(s), noise_scale=jnp.asarray(0.0)))(ssh, u, v)
        return accel(phi, e3u, e3v)
    return run


def predict(net, jd, run, chunk=50):
    """Face accelerations of ``net`` on every sample, in chunks."""
    n = jd["ssh"].shape[0]
    parts = [run(net, *(jd[k][i:i + chunk] for k in ("ssh", "u", "v", "e3u", "e3v"))) for i in range(0, n, chunk)]
    return np.concatenate([np.asarray(p[0]) for p in parts]), np.concatenate([np.asarray(p[1]) for p in parts])


def fit(net, jd, target, train, val, a2, accel, run, pack, geometry, key, stop_corr):
    """Adam on the normalised face-acceleration MSE with validation early stopping."""
    umask = np.asarray(geometry.umask, dtype=bool)
    vmask = np.asarray(geometry.vmask, dtype=bool)
    n_u, n_faces = int(umask.sum()), int(umask.sum()) + int(vmask.sum())
    tgt_u, tgt_v = np.zeros(jd["u"].shape), np.zeros(jd["v"].shape)
    tgt_u[:, umask], tgt_v[:, vmask] = target[:, :n_u], target[:, n_u:]
    tgt_u, tgt_v = jnp.asarray(tgt_u), jnp.asarray(tgt_v)
    umask_j, vmask_j = jnp.asarray(umask), jnp.asarray(vmask)
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))

    def loss_fn(net, ssh, u, v, e3u, e3v, tu, tv):
        phi = jax.vmap(lambda s, a, b: net(
            SWERK3StageEntry(s, a, b, *(None,) * 10), geometry, key=jax.random.fold_in(key, 1),
            noise=jnp.zeros_like(s), noise_scale=jnp.asarray(0.0)))(ssh, u, v)
        a_u, a_v = accel(phi, e3u, e3v)
        sse = jnp.sum((a_u - tu) ** 2 * umask_j) + jnp.sum((a_v - tv) ** 2 * vmask_j)
        return sse / (ssh.shape[0] * n_faces * a2)

    @eqx.filter_jit
    def step(net, opt_state, *batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(net, *batch)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(net, eqx.is_inexact_array))
        return eqx.apply_updates(net, updates), opt_state, loss

    def val_metrics(net):
        pred = pack(*predict(net, {k: jd[k][val] for k in jd}, run))
        tgt = target[val]
        return (float(np.mean(np.sum((pred - tgt) ** 2, axis=1)) / (n_faces * a2)),
                float(np.mean([ss.pearson(p, t) for p, t in zip(pred, tgt)])))

    rng = np.random.default_rng(eb.SEED)
    order, cursor = rng.permutation(train), 0
    best, best_step, best_net = np.inf, 0, net
    curve, running, t0 = [], [], time.time()
    for s in range(1, MAX_STEPS + 1):
        if cursor + BATCH > order.size:
            order, cursor = rng.permutation(train), 0
        idx = order[cursor:cursor + BATCH]
        cursor += BATCH
        net, opt_state, loss = step(net, opt_state, *(jd[k][idx] for k in ("ssh", "u", "v", "e3u", "e3v")),
                                    tgt_u[idx], tgt_v[idx])
        running.append(float(loss))
        if s % EVAL_EVERY == 0:
            vloss, vcorr = val_metrics(net)
            curve.append({"step": s, "train_loss": float(np.mean(running)), "val_loss": vloss,
                          "val_corr_all": vcorr, "wall_s": time.time() - t0})
            running = []
            if vloss < best:
                best, best_step, best_net = vloss, s, net
            print(f"  step {s:5d} train {curve[-1]['train_loss']:.4f} val {vloss:.4f} "
                  f"corr {vcorr:.4f} best@{best_step} ({curve[-1]['wall_s']:.0f} s)")
            sys.stdout.flush()
            if (stop_corr is not None and vcorr > stop_corr) or s - best_step >= PATIENCE:
                break
    return best_net, {"curve": curve, "best_step": best_step, "best_val_loss": best, "steps_run": s}


def figure(data, predictions, target, val, umask, vmask, geometry, out_dir):
    """Curl of the acceleration at the first validation start, shared scale."""
    i = int(val[0])
    vtx = np.asarray(geometry.vertex_interior, dtype=bool)
    bounds = geometry.wet_bounds
    n_u = int(umask.sum())

    def curl(vec):
        au, av = np.zeros(umask.shape), np.zeros(vmask.shape)
        au[umask], av[vmask] = vec[:n_u], vec[n_u:]
        return np.asarray(balanced_fields_from_native(jnp.asarray(au), jnp.asarray(av), geometry,
                                                      objective="template").relative_vorticity_s1)

    panels = [("implied (representable a_S)", curl(target[i]))]
    panels += [(name, curl(pred[i])) for name, pred in predictions.items() if name != "persistence"]
    vmax = max(np.nanpercentile(np.abs(ss._crop(f, vtx, bounds)), ss.PERCENTILE) for _, f in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.2), constrained_layout=True)
    for ax, (name, f) in zip(axes, panels):
        im = ax.imshow(ss._crop(f, vtx, bounds), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        title = name if name == panels[0][0] else f"{name}\ncorr {ss.pearson(predictions[name][i], target[i]):+.2f}"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.6, location="bottom", pad=0.02, label="curl of acceleration (s$^{-2}$)")
    fig.suptitle(f"validation start h{int(data['hour'][i])}: implied vs offline fits vs online w=8", fontsize=11)
    fig.savefig(out_dir / "capacity_figure.png", dpi=110)
    plt.close(fig)


def render(table):
    lines = ["model                       split       region     EV      corr    amp"]
    for model, splits in table.items():
        for split, regions in splits.items():
            for region, m in regions.items():
                lines.append(f"{model:27s} {split:11s} {region:9s} {m['explained_variance']:+6.3f}  "
                             f"{m['corr']:+6.3f}  {m['amplitude_ratio']:5.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
