"""Loss floor: how much of the online training loss could ANY closure in the
operator class (boundary-zero F potential through the rotated gradient at the
live face cross-sections, swe_rotated_potential_forcing) remove?

Windows.  The pilot-2 contract's own training batches: the first N_BATCHES[w]
batches the contract issues for epoch EPOCH of window w (the same sampler,
seed, offset and shuffle the training used), so twenty 1-day and ten 2-day
windows.  The loss is exactly the pilot's: deterministic_trajectory_loss,
template objective, wall ring weighted (the run manifest's values), i.e.
Maddison Eq. 7's mean over windows and hourly right endpoints; the hourly
profile is the same sum restricted to one right endpoint, so the mean of the
profile over the window's hours is the window loss.

Cases, every one run through the same closure path (the potential applied at
every RK3 stage of every step):
  no_closure          the free model (network None);
  pilot2              the pilot-2 trained network of that window length
                      (checkpoints/w0{1,2}_e05.eqx of the pilot run);
  exp1_seed1          the converged full-experiment seed-1 network of that
                      window length (EXP1 checkpoints);
  offline_fit         the capacity test's offline fit of our architecture to
                      the representable one-hour a_S (fit_ours_c16_h4_k3.eqx);
  oracle_fixed_start  the representable a_S of the window's start (the
                      capacity test's recipe: model one hour from the stored
                      un-rebalanced start against the stored truth one hour
                      later, projected onto the operator's range at the
                      rebalanced launch state), held fixed in time through
                      the whole window.  Limitation: one potential for the
                      whole window, so it decays as the flow moves on;
  oracle_hourly       the greedy in-class oracle: at every hour h the free
                      model is stepped one hour from the CURRENT closed-loop
                      state M(h), a_S(h) = (truth(h+1) - free(h+1)) / 3600 s
                      (truth tendency minus model tendency, both from M(h)),
                      projected onto the operator's range at M(h), and that
                      potential is applied through the closure path over hour
                      h.  The best a closure of this class could do with one
                      hour of foresight; it costs one extra model hour per
                      model hour;
  oracle_truth_path   the implied forcing sequence along the truth: a_S(h)
                      from the stored truth state at hour h (free model one
                      hour from it against the truth at h+1, projected at the
                      truth state's thicknesses; hour 0 is oracle_fixed_start's
                      potential), applied over hour h of the closed-loop run.
                      A closure that knew the truth's own forcing but not the
                      model's drift.

Controls, each asserted:
  (a) the driver's loss (build_batch_functions' value function, windows_per
      _call as the run) of the previous epoch's checkpoint on batch 0 of the
      epoch reproduces the pilot's logged pre-update training loss for that
      batch (relative LOGGED_LOSS_RTOL; CPU/libm here, GPU/native there);
  (b) the per-hour profile's mean over hours equals the driver's loss value
      (relative 1e-12) for the free model and the pilot network;
  (c) the zero potential through the constant-potential path gives the free
      model's hourly losses (relative 1e-12);
  (d) the fixed-start potential, through the JAX operator at the launch
      state, reproduces the projected acceleration (relative 1e-12), and the
      truth-path oracle's hour-0 potential is that potential exactly.
Writes <SWGONL_DIAG>/loss_floor.json and loss_floor_profiles.png.
Usage: SWGONL_RUN=<pilot-2 seed dir> SWGONL_DIAG=<output dir> loss_floor.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import energy_budget as eb  # noqa: E402
import implied_subgrid_work as isw  # noqa: E402
import residual_decomposition as rd  # noqa: E402
from legoesm.ml.checkpoint_io import load_checkpoint_or_fail  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d  # noqa: E402
from legoesm.ocean.dynamics.swe_rk3 import (  # noqa: E402
    _entry_from_state,
    swe_rk3_step,
    swe_rotated_potential_forcing,
)
from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands  # noqa: E402
from legoesm.training.swg_online import (  # noqa: E402
    SWGDatasetContract,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    build_segment_fn,
    coarse_state_from_native,
    create_swg_optimizer,
    diagnose_native_trajectory,
    eq7_point_weights,
    maddison_half_width_m,
    balanced_fields_from_native,
    stored_state_at,
)

REPO = Path(__file__).resolve().parents[3]
EXP1 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round12_full_experiment"
            "/swgonl_exp1_template_ring_weighted/seed20261013/checkpoints")
CAPACITY = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round11_pilot_ring_weighted"
                "/capacity_test")
N_BATCHES = {1: 2, 2: 1}       # batches of ten windows per window length
EPOCH = 5                      # the pilot's last epoch of each window length
LOGGED_LOSS_RTOL = 1e-6        # control (a): CPU/libm reintegration vs the GPU/native log
HOURS_PER_DAY = 24
CASES = ("no_closure", "pilot2", "exp1_seed1", "offline_fit",
         "oracle_fixed_start", "oracle_hourly", "oracle_truth_path")


def git_sha(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def hourly_eq7_loss(predicted_native, target_native, geometry, *, objective, beta_m1_s1,
                    basin_length_m, wall_ring):
    """Eq. 7's weighted vorticity error per (window, hour): the loss resolved
    to its hourly right endpoints.  ``mean()`` over both axes is
    deterministic_trajectory_loss of the same arrays."""
    predicted = jax.vmap(lambda s, u, v: diagnose_native_trajectory(
        (s, u, v), geometry, objective=objective))(*predicted_native)
    target = jax.vmap(lambda s, u, v: diagnose_native_trajectory(
        (s, u, v), geometry, objective=objective))(*target_native)
    half_width = maddison_half_width_m(basin_length_m, geometry)
    weights = eq7_point_weights(geometry, wall_ring=wall_ring)
    error = predicted.relative_vorticity_s1 - target.relative_vorticity_s1
    scale = 1.0e4 / (beta_m1_s1**2 * half_width**2) / (4.0 * half_width**2)
    return scale * jnp.sum(error**2 * weights, axis=(-2, -1))


class Rollouts:
    """Compiled window and hour steppers sharing the production step and closure path."""

    def __init__(self, card, geometry, forcing, window_days, key):
        grid, z_coord, cfg = card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
        self.template = card.recipe.initial_state._replace(
            tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
        steps_per_hour = int(isw.HOUR_S / card.dt_s)
        assert steps_per_hour * card.dt_s == isw.HOUR_S
        self.n_hours = HOURS_PER_DAY * window_days
        umask3, vmask3 = compute_face_masks_3d(z_coord.is_active, grid)
        self.ops = nemo_qco_resolved_mesh_operands(z_coord, grid, umask3, vmask3, jnp.float64, 1)
        self.umask3, self.vmask3 = umask3, vmask3
        tau = (forcing.tau_i_native, forcing.tau_j_native)
        vtx = jnp.asarray(geometry.vertex_interior)
        template = self.template
        ops = self.ops
        raw, _ = build_segment_fn(card, geometry, window_days=window_days, checkpoint=False,
            target_interval_hours=1)
        n_steps = self.n_hours * steps_per_hour
        shape = tuple(geometry.tmask.shape)

        def native(state):
            return state.eta.data, state.u.data[:, 1:, 0], state.v.data[1:, :, 0]

        def hour(state, potential_fn):
            def step(s, _):
                return swe_rk3_step(s, card.dt_s, forcing, grid, z_coord, cfg,
                                    stage_potential_fn=potential_fn), None
            return jax.lax.scan(step, state, None, length=steps_per_hour)[0]

        @eqx.filter_jit
        def window_network(network, ssh, u, v):
            """The training segment (build_segment_fn), hourly natives, over windows."""
            noise = jnp.zeros((n_steps, 3) + shape, dtype=jnp.float64)

            def one(s, a, b):
                return raw(network, coarse_state_from_native(template, s, a, b), noise, key,
                           jnp.asarray(0.0))[1]
            return jax.vmap(one)(ssh, u, v)

        @jax.jit
        def window_constant(ssh, u, v, phi_hours):
            """Per-hour constant potentials (windows, hours, y, x) through the closure path."""
            def one(s, a, b, phis):
                def one_hour(state, phi):
                    state = hour(state, lambda entry, stage: phi)
                    return state, native(state)
                return jax.lax.scan(one_hour, coarse_state_from_native(template, s, a, b), phis)[1]
            return jax.vmap(one)(ssh, u, v, phi_hours)

        @jax.jit
        def hour_free(ssh, u, v):
            return jax.vmap(lambda s, a, b: native(hour(coarse_state_from_native(template, s, a, b), None)))(
                ssh, u, v)

        @jax.jit
        def hour_constant(ssh, u, v, phi):
            return jax.vmap(lambda s, a, b, p: native(hour(coarse_state_from_native(template, s, a, b),
                                                           lambda entry, stage: p)))(ssh, u, v, phi)

        @jax.jit
        def thickness(ssh, u, v):
            """Live face thicknesses (e3u, e3v) at a state, as the closure path sees them."""
            def one(s, a, b):
                entry = _entry_from_state(coarse_state_from_native(template, s, a, b), tau, tau,
                                          grid, z_coord, cfg)
                one_ = jnp.asarray(1.0, dtype=jnp.float64)
                return ((ops.e3u_0 * (one_ + entry.r3u_k[..., None] * ops.umask3))[..., 0],
                        (ops.e3v_0 * (one_ + entry.r3v_k[..., None] * ops.vmask3))[..., 0])
            return jax.vmap(one)(ssh, u, v)

        @jax.jit
        def operator(phi, e3u, e3v):
            return jax.vmap(lambda p, a, b: swe_rotated_potential_forcing(
                p, ops.e2u, ops.e1v, a, b, umask3[:, 1:, 0], vmask3[1:, :, 0], vtx))(phi, e3u, e3v)

        self.window_network, self.window_constant = window_network, window_constant
        self.hour_free, self.hour_constant = hour_free, hour_constant
        self.thickness, self.operator = thickness, operator


class Projector:
    """Representable part of an acceleration at a state: the capacity test's projection."""

    def __init__(self, rollouts, geometry):
        self.rollouts = rollouts
        _, _, self.umask, self.vmask = isw.face_rings(geometry)
        self.vtx = np.asarray(geometry.vertex_interior, dtype=bool)
        self.e2u = np.asarray(rollouts.ops.e2u, dtype=np.float64)
        self.e1v = np.asarray(rollouts.ops.e1v, dtype=np.float64)
        self.n_u = int(self.umask.sum())

    def pack(self, a_u, a_v):
        """(windows, y, x) x 2 -> (windows, wet faces), U then V."""
        return np.concatenate([np.asarray(a_u)[:, self.umask], np.asarray(a_v)[:, self.vmask]], axis=1)

    def __call__(self, a_u, a_v, ssh, u, v):
        """(phi, rep_u, rep_v, residual) per window; a_* and the state are (windows, y, x)."""
        e3u, e3v = (np.asarray(x) for x in self.rollouts.thickness(ssh, u, v))
        phi = np.zeros((a_u.shape[0],) + self.vtx.shape)
        rep_u, rep_v = np.zeros(a_u.shape), np.zeros(a_v.shape)
        residual = np.zeros(a_u.shape[0])
        for w in range(a_u.shape[0]):
            A, _ = rd.rotated_gradient_matrix(self.e2u * e3u[w], self.e1v * e3v[w],
                                              self.umask, self.vmask, self.vtx)
            scale = float(np.median((self.e2u * e3u[w])[self.umask]))
            rep_vec, _, phi_vec, residual[w] = rd.RangeProjector(A * scale)(
                np.concatenate([(a_u[w] * self.umask)[self.umask], (a_v[w] * self.vmask)[self.vmask]]))
            phi[w][self.vtx] = phi_vec * scale   # (sA)^T(sA) Phi' = (sA)^T a  =>  Phi = s Phi'
            rep_u[w][self.umask], rep_v[w][self.vmask] = rep_vec[:self.n_u], rep_vec[self.n_u:]
        return phi, rep_u, rep_v, residual


def implied_potential(project, rollouts, base, truth_next, at):
    """a_S = (truth_next - free model one hour from ``base``) / 3600 s, projected at ``at``."""
    free = [np.asarray(x) for x in rollouts.hour_free(*base)]
    a_u = (truth_next[1] - free[1]) / isw.HOUR_S
    a_v = (truth_next[2] - free[2]) / isw.HOUR_S
    return project(a_u, a_v, *at) + (a_u * project.umask, a_v * project.vmask)


def weighted_vorticity_rms(a_u, a_v, geometry, *, objective, wall_ring):
    """Eq. 7-weighted rms of the loss's own vorticity diagnostic of a face
    acceleration field (windows, y, x): what the loss can see of it."""
    weights = eq7_point_weights(geometry, wall_ring=wall_ring)
    zeta = jax.vmap(lambda u, v: balanced_fields_from_native(
        u, v, geometry, objective=objective).relative_vorticity_s1)(jnp.asarray(a_u), jnp.asarray(a_v))
    return float(jnp.sqrt(jnp.mean(jnp.sum(zeta**2 * weights, axis=(-2, -1)))))


def oracle_hourly(project, rollouts, batch):
    """Greedy in-class oracle from the model's own state; hourly natives and diagnostics."""
    state = tuple(np.asarray(x, dtype=np.float64) for x in batch.initial)
    natives, residuals, rep_fraction = [], [], []
    for h in range(rollouts.n_hours):
        truth_next = tuple(np.asarray(x[:, h], dtype=np.float64) for x in batch.target)
        free = [np.asarray(x) for x in rollouts.hour_free(*state)]
        a_u = (truth_next[1] - free[1]) / isw.HOUR_S
        a_v = (truth_next[2] - free[2]) / isw.HOUR_S
        phi, rep_u, rep_v, residual = project(a_u, a_v, *state)
        rep_fraction.append(rms(project.pack(rep_u, rep_v)) / rms(project.pack(a_u, a_v)))
        residuals.append(float(residual.max()))
        state = tuple(np.asarray(x) for x in rollouts.hour_constant(*state, jnp.asarray(phi)))
        natives.append(state)
    stacked = tuple(np.stack([n[i] for n in natives], axis=1) for i in range(3))
    return stacked, {"normal_equation_residual_max": max(residuals),
                     "representable_fraction_of_a_S_rms_by_hour": rep_fraction}


def oracle_truth_path(project, rollouts, batch, phi0):
    """The implied forcing sequence along the stored truth, applied to the closed-loop run."""
    phis = [phi0]
    for h in range(1, rollouts.n_hours):
        base = tuple(np.asarray(x[:, h - 1], dtype=np.float64) for x in batch.target)
        truth_next = tuple(np.asarray(x[:, h], dtype=np.float64) for x in batch.target)
        phis.append(implied_potential(project, rollouts, base, truth_next, base)[0])
    phi_hours = jnp.asarray(np.stack(phis, axis=1))
    return tuple(np.asarray(x) for x in rollouts.window_constant(*batch.initial, phi_hours))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def load_network(path, spec, optimizer, key, what):
    skeleton = SWGScalarPotentialCNN(spec, key=key)
    network, _ = load_checkpoint_or_fail(
        path, (skeleton, optimizer.init(eqx.filter(skeleton, eqx.is_inexact_array))), what=what)
    return network


def main():
    driver, config, card, geometry, nets, forcing, key = eb.setup()
    driver = eb._load("run_swg_online_training", REPO / "scripts/run/run_swg_online_training.py")
    contract = SWGDatasetContract.from_json(eb.RUN / "dataset_contract.json", card=card)
    log = json.loads((eb.RUN / "training_log.json").read_text())
    spec, _ = driver.network_spec(config, card, geometry)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    namelist = card.resolved_namelist
    loss_kw = {"objective": config["objective"], "beta_m1_s1": namelist.rn_beta_m_s,
               "basin_length_m": namelist.rn_domsiz_m, "wall_ring": config["wall_ring"]}
    windows_per_call = eb.MANIFEST["execution"]["windows_per_call"]
    capacity = json.loads((CAPACITY / "capacity_test.json").read_text())
    fit_name = next(name for name, arch in capacity["architectures"].items()
                    if arch == {"kind": "plain", "hidden_channels": spec.hidden_channels,
                                "hidden_layers": spec.hidden_layers, "kernel_size": spec.kernel_size,
                                "parameters": arch["parameters"]})
    offline_fit = eqx.tree_deserialise_leaves(CAPACITY / f"fit_{fit_name}.eqx",
                                              SWGScalarPotentialCNN(spec, key=key))
    hourly = jax.jit(lambda p, t: hourly_eq7_loss(p, t, geometry, **loss_kw))
    projector = None
    results = {"cases": {name: {} for name in CASES}, "controls": {}, "windows": {}, "oracle": {}}
    t0 = time.time()
    for window_days, n_batches in N_BATCHES.items():
        tag = f"w{window_days:02d}"
        issued = contract.batches(split="train", window_days=window_days, epoch=EPOCH - 1, seed=eb.SEED,
            target_interval_hours=1)
        batches = [next(issued) for _ in range(n_batches)]
        rollouts = Rollouts(card, geometry, forcing, window_days, key)
        projector = Projector(rollouts, geometry)
        _, loss_fn = driver.build_batch_functions(card, geometry, window_days=window_days,
                                                  objective=config["objective"], wall_ring=config["wall_ring"],
                                                  training_precision=config["training_precision"],
                                                      target_interval_hours=1,
                                                  checkpoint=True)
        networks = {
            "pilot2": nets[window_days],
            "exp1_seed1": load_network(EXP1 / f"{tag}_e05.eqx", spec, optimizer, key, "exp1 seed-1 closure"),
            "offline_fit": offline_fit}
        previous = load_network(eb.RUN / "checkpoints" / f"{tag}_e{EPOCH - 1:02d}.eqx", spec, optimizer, key,
                                "previous epoch's closure")
        logged = next(e for e in log["epochs"] if e["window_days"] == window_days and e["epoch"] == EPOCH)
        results["windows"][tag] = {
            "epoch": EPOCH, "batches": n_batches,
            "start_absolute_hours": [b.start_absolute_hour.tolist() for b in batches],
            "logged_train_loss_per_batch": logged["train_loss_per_batch"][:n_batches]}
        # Control (a): the previous checkpoint on batch 0 reproduces the logged pre-update loss.
        mine = driver.batch_loss_value(loss_fn, previous, batches[0], key, windows_per_call)
        results["controls"][f"{tag}_logged_loss_batch0"] = {
            "logged": logged["train_loss_per_batch"][0], "recomputed_cpu": mine,
            "rel": abs(mine / logged["train_loss_per_batch"][0] - 1.0),
            "network": f"checkpoints/{tag}_e{EPOCH - 1:02d}.eqx"}
        assert results["controls"][f"{tag}_logged_loss_batch0"]["rel"] <= LOGGED_LOSS_RTOL, \
            results["controls"][f"{tag}_logged_loss_batch0"]
        print(tag, "control (a)", results["controls"][f"{tag}_logged_loss_batch0"], f"({time.time() - t0:.0f} s)")
        sys.stdout.flush()

        per_hour = {name: [] for name in CASES}   # per batch: (windows, hours)
        oracle_diag = []
        for b, batch in enumerate(batches):
            initial = tuple(jnp.asarray(x) for x in batch.initial)
            target = tuple(jnp.asarray(x) for x in batch.target)

            def profile(predicted):
                return np.asarray(hourly(tuple(jnp.asarray(x) for x in predicted), target))

            free = rollouts.window_network(None, *initial)
            per_hour["no_closure"].append(profile(free))
            for name, network in networks.items():
                per_hour[name].append(profile(rollouts.window_network(network, *initial)))
            # Control (b): hourly profile vs the driver's loss value.
            for name, network in (("no_closure", None), ("pilot2", networks["pilot2"])):
                value = driver.batch_loss_value(loss_fn, network, batch, key, windows_per_call)
                rel = abs(float(per_hour[name][-1].mean()) / value - 1.0)
                results["controls"].setdefault(f"{tag}_profile_mean_vs_driver_loss_max_rel", 0.0)
                results["controls"][f"{tag}_profile_mean_vs_driver_loss_max_rel"] = max(
                    results["controls"][f"{tag}_profile_mean_vs_driver_loss_max_rel"], rel)
                assert rel <= 1e-12, (tag, b, name, value, per_hour[name][-1].mean())
            # Control (c): zero potential through the constant path == free model.
            zeros = jnp.zeros((batch.initial[0].shape[0], rollouts.n_hours) + geometry.vertex_interior.shape)
            zero_profile = profile(rollouts.window_constant(*initial, zeros))
            rel = float(np.max(np.abs(zero_profile / per_hour["no_closure"][-1] - 1.0)))
            results["controls"][f"{tag}_zero_potential_vs_free_max_rel"] = max(
                results["controls"].get(f"{tag}_zero_potential_vs_free_max_rel", 0.0), rel)
            assert rel <= 1e-12, (tag, b, rel)
            # Oracle potentials.  Fixed start: the capacity test's recipe (free hour from the
            # stored un-rebalanced start, projected at the rebalanced launch state).
            stored = tuple(np.stack(x) for x in zip(*[
                [np.asarray(a, dtype=np.float64)
                 for a in stored_state_at(contract.sources, contract.objective, int(h))[0]]
                for h in batch.start_absolute_hour]))
            assert all(np.array_equal(stored[i], batch.initial[i]) for i in (1, 2))
            launch = tuple(np.asarray(x, dtype=np.float64) for x in batch.initial)
            truth_1h = tuple(np.asarray(x[:, 0], dtype=np.float64) for x in batch.target)
            phi0, rep_u, rep_v, residual, a_u, a_v = implied_potential(
                projector, rollouts, stored, truth_1h, launch)
            # What the loss can see of the orthogonal remainder: A^T r = 0 is a zero
            # thickness-weighted curl, so its Eq. 7 vorticity is only the thickness variation.
            vorticity_fraction = {
                part: weighted_vorticity_rms(au, av, geometry, objective=config["objective"],
                                             wall_ring=config["wall_ring"])
                / weighted_vorticity_rms(a_u, a_v, geometry, objective=config["objective"],
                                         wall_ring=config["wall_ring"])
                for part, (au, av) in (("representable", (rep_u, rep_v)),
                                       ("remainder", (a_u - rep_u, a_v - rep_v)))}
            # Control (d): the potential through the JAX operator at the launch state == projection
            e3u, e3v = rollouts.thickness(*launch)
            op_u, op_v = (np.asarray(x) for x in rollouts.operator(jnp.asarray(phi0), e3u, e3v))
            rel = float(np.max(np.linalg.norm(projector.pack(op_u, op_v) - projector.pack(rep_u, rep_v), axis=1)
                               / np.linalg.norm(projector.pack(rep_u, rep_v), axis=1)))
            results["controls"][f"{tag}_fixed_potential_operator_vs_projection_max_rel"] = max(
                results["controls"].get(f"{tag}_fixed_potential_operator_vs_projection_max_rel", 0.0), rel)
            assert rel <= 1e-12, (tag, b, rel)
            fixed = jnp.asarray(np.repeat(phi0[:, None], rollouts.n_hours, axis=1))
            per_hour["oracle_fixed_start"].append(profile(rollouts.window_constant(*initial, fixed)))
            greedy, diag = oracle_hourly(projector, rollouts, batch)
            per_hour["oracle_hourly"].append(profile(greedy))
            diag["fixed_start_normal_equation_residual_max"] = float(residual.max())
            diag["fixed_start_representable_fraction_of_a_S_rms"] = (
                rms(projector.pack(rep_u, rep_v)) / rms(projector.pack(a_u, a_v)))
            diag["fixed_start_eq7_vorticity_fraction_of_a_S"] = vorticity_fraction
            oracle_diag.append(diag)
            per_hour["oracle_truth_path"].append(profile(oracle_truth_path(projector, rollouts, batch, phi0)))
            print(tag, "batch", b, {name: float(np.mean(per_hour[name][-1])) for name in CASES},
                  f"({time.time() - t0:.0f} s)")
            sys.stdout.flush()
        results["oracle"][tag] = oracle_diag
        reference = float(np.mean(np.concatenate(per_hour["no_closure"])))
        for name in CASES:
            rows = np.concatenate(per_hour[name])          # (windows, hours)
            results["cases"][name][tag] = {
                "loss": float(rows.mean()), "ratio_to_no_closure": float(rows.mean() / reference),
                "per_window": rows.mean(axis=1).tolist(),
                "hourly": rows.mean(axis=0).tolist(),
                "hourly_ratio_to_no_closure": (
                    rows.mean(axis=0) / np.concatenate(per_hour["no_closure"]).mean(axis=0)).tolist()}
    results.update({
        "definition": __doc__,
        "provenance": {"legoesm_git_sha": git_sha(REPO), "run": str(eb.RUN), "seed": eb.SEED,
                       "exp1_checkpoints": str(EXP1), "offline_fit": str(CAPACITY / f"fit_{fit_name}.eqx"),
                       "loss": loss_kw, "windows_per_call": windows_per_call,
                       "backend": jax.default_backend(), "transcendentals": "libm"},
        "hours": "hourly[k] is the loss at right endpoint k+1 of the window",
        "wall_s": time.time() - t0})
    (eb.DIAG / "loss_floor.json").write_text(json.dumps(results, indent=1))
    figure(results, eb.DIAG / "loss_floor_profiles.png")
    print(render(results))
    print("controls", json.dumps(results["controls"]))
    print("WROTE", eb.DIAG / "loss_floor.json")


def render(results):
    lines = ["case                 " + "".join(f"{tag:>14s}{'ratio':>8s}" for tag in results["windows"])]
    for name in CASES:
        row = f"{name:20s}"
        for tag in results["windows"]:
            entry = results["cases"][name][tag]
            row += f"{entry['loss']:14.5f}{entry['ratio_to_no_closure']:8.3f}"
        lines.append(row)
    return "\n".join(lines)


def figure(results, path):
    tags = list(results["windows"])
    fig, axes = plt.subplots(1, len(tags), figsize=(6.0 * len(tags), 4.2), constrained_layout=True)
    for ax, tag in zip(np.atleast_1d(axes), tags):
        for name in CASES:
            hourly = results["cases"][name][tag]["hourly"]
            ax.plot(np.arange(1, len(hourly) + 1), hourly, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("hour of window (right endpoint)")
        ax.set_ylabel("Eq. 7 loss at that hour")
        ax.set_title(f"{tag}: {results['windows'][tag]['batches'] * 10} training windows")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("hourly loss profile per closure (mean over windows)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
