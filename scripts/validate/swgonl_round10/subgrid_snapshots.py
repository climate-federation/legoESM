"""Round-11 snapshots: the implied sub-grid forcing next to the trained closures.

At four of the twenty stored window starts of step 7 (implied_subgrid_work.
starts: the first two w=1 training batches' first windows and the first and
last validation-lattice windows) this draws, as maps, the one-hour sub-grid
acceleration the data implies from the STORED start (step 8's a_S: truth
one-hour change minus the coarse model's from the matched stored state,
over 3600 s), reduced to its representable part (the exact projection onto
the range of the closure operator at the rebalanced launch state,
residual_decomposition.RangeProjector), and the acceleration each pilot-2
network (w = 1, 2, 4, 8 days, epoch 5) produces at that state.

Per start one figure: rows = implied (representable), w=1, 2, 4, 8; columns
= curl of the acceleration on interior F points (the vorticity tendency,
s-2; balanced_fields_from_native's native C-grid curl), the U-face and the
V-face components (m s-2); one symmetric colour scale per column shared by
the five rows (the 99.5th percentile of |field| over the five rows' wet
points, so a wall spike does not blank the interior).  Row titles carry the
pattern correlation with the implied representable field (Pearson over the
wet U and V faces stacked) and the ratio of rms amplitudes.  A sixth figure
shows the work density u*a_u + v*a_v at T points (half of each adjoining
wet face) implied vs the w=8 network.

Network accelerations are the exact operator swe_rotated_potential_forcing
applied to the network's potential at the state's live face cross-sections
(step 8 control (b): identical to the stage-differenced closure to ~4e-13).

Controls, each asserted:
  (a) the implied a_S and its representable part reproduce step 8's stored
      work (total, ring, interior) and rms for every start (relative 1e-9;
      the step-8 receipt of the first pilot, whose seed, dataset contract
      and starts the second pilot shares; the implied field does not depend
      on the network);
  (b) every network's acceleration reproduces energy_budget.make_terms'
      closure rms and work at the state (relative 1e-9);
  (c) a zero potential through the same operator gives an identically zero
      acceleration, curl and work density;
  (d) the T-point work density sums to the face-summed work exactly.
Writes <SWGONL_DIAG>/snapshot_<start>.png, work_density_w08.png, summary.json.
Usage: SWGONL_RUN=<pilot-2 seed dir> SWGONL_DIAG=<output dir> subgrid_snapshots.py
"""
from __future__ import annotations

import json
import sys
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
from legoesm.ocean.dynamics.swe_rk3 import _entry_from_state, swe_rotated_potential_forcing  # noqa: E402
from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands  # noqa: E402
from legoesm.training.swg_online import (  # noqa: E402
    SWGDatasetContract,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    balanced_fields_from_native,
    coarse_state_from_native,
    create_swg_optimizer,
    stored_state_at,
    swg_online_step,
)

STEP8_REF = eb.R9 / "round10_diagnosis/step8_residual_decomposition.json"
WINDOWS = (1, 2, 4, 8)
START_INDICES = (0, 1, 10, 19)   # of isw.starts: train batches 0, 1; validation windows 0, 9
PERCENTILE = 99.5
TOL = 1e-9


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


def work_density_t(u, v, au, av, umask, vmask):
    """u*a_u + v*a_v at T points: half of each adjoining wet face (native
    layout: U (j, i) is the east face of T (j, i), V (j, i) its north face).
    Summed over T it equals the face-summed work exactly."""
    wu = u * au * umask
    wv = v * av * vmask
    west = np.concatenate([np.zeros_like(wu[:, :1]), wu[:, :-1]], axis=1)
    south = np.concatenate([np.zeros_like(wv[:1]), wv[:-1]], axis=0)
    return 0.5 * (wu + west + wv + south)


def load_networks(driver, config, card, geometry, key, windows=WINDOWS):
    """Epoch-5 network of every window length, as energy_budget.setup loads w=1, 2."""
    spec, _ = driver.network_spec(config, card, geometry)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    nets = {}
    for w in windows:
        skeleton = SWGScalarPotentialCNN(spec, key=key)
        nets[w], _ = load_checkpoint_or_fail(
            eb.RUN / "checkpoints" / f"w{w:02d}_e05.eqx",
            (skeleton, optimizer.init(eqx.filter(skeleton, eqx.is_inexact_array))),
            what=f"pilot closure w={w}")
    return nets


def main():
    driver, config, card, geometry, nets12, forcing, key = eb.setup()
    nets = load_networks(driver, config, card, geometry, key)
    for w in (1, 2):   # the same checkpoints energy_budget.setup loaded
        assert all(bool(jnp.array_equal(a, b)) for a, b in zip(
            jax.tree_util.tree_leaves(eqx.filter(nets[w], eqx.is_array)),
            jax.tree_util.tree_leaves(eqx.filter(nets12[w], eqx.is_array))))
    ring_u, ring_v, umask, vmask = isw.face_rings(geometry)
    n_u, n_v = int(umask.sum()), int(vmask.sum())
    assert n_u == n_v
    vtx = np.asarray(geometry.vertex_interior, dtype=bool)
    wet = np.asarray(geometry.tmask, dtype=bool)
    j0, j1, i0, i1 = geometry.wet_bounds
    template = card.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    grid, z_coord, cfg = card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
    tau = (forcing.tau_i_native, forcing.tau_j_native)
    dt = card.dt_s
    steps_per_hour = int(isw.HOUR_S / dt)
    assert steps_per_hour * dt == isw.HOUR_S
    umask3, vmask3 = compute_face_masks_3d(z_coord.is_active, grid)
    ops = nemo_qco_resolved_mesh_operands(z_coord, grid, umask3, vmask3, jnp.float64, 1)
    e2u = np.asarray(ops.e2u, dtype=np.float64)
    e1v = np.asarray(ops.e1v, dtype=np.float64)

    def step_a(state, _):   # step 7's stepper, verbatim
        return swg_online_step(None, state, None, None, None, card=card,
                               geometry=geometry, forcing=forcing), None

    @jax.jit
    def hour_a(state):
        state, _ = jax.lax.scan(step_a, state, None, length=steps_per_hour)
        return state

    @jax.jit
    def predicted(ssh, u, v):
        """Live face thicknesses and every network's acceleration at a state,
        plus the zero-potential control through the same operator."""
        state = coarse_state_from_native(template, ssh, u, v)
        entry = _entry_from_state(state, tau, tau, grid, z_coord, cfg)
        one = jnp.asarray(1.0, dtype=jnp.float64)
        e3u = (ops.e3u_0 * (one + entry.r3u_k[..., None] * ops.umask3))[..., 0]
        e3v = (ops.e3v_0 * (one + entry.r3v_k[..., None] * ops.vmask3))[..., 0]

        def accel(potential):
            return swe_rotated_potential_forcing(
                potential, ops.e2u, ops.e1v, e3u, e3v, umask3[:, 1:, 0], vmask3[1:, :, 0], vtx)

        out = {}
        for w, net in nets.items():
            potential = net(entry, geometry, key=jax.random.fold_in(key, 1),
                            noise=jnp.zeros_like(entry.ssh_k), noise_scale=jnp.asarray(0.0))
            out[w] = accel(potential)
        return e3u, e3v, out, accel(jnp.zeros_like(entry.ssh_k))

    terms = {w: jax.jit(eb.make_terms(card, geometry, nets[w], forcing, key)) for w in WINDOWS}

    def native(state):
        return (np.asarray(state.eta.data), np.asarray(state.u.data[:, 1:, 0]),
                np.asarray(state.v.data[1:, :, 0]))

    def work(u, v, au, av, sel_u=umask, sel_v=vmask):
        return float(np.sum(u * au * sel_u) / n_u + np.sum(v * av * sel_v) / n_v)

    def work3(u, v, au, av):
        return {"total": work(u, v, au, av),
                "ring": work(u, v, au, av, ring_u, ring_v),
                "interior": work(u, v, au, av, umask & ~ring_u, vmask & ~ring_v)}

    def seeded(ssh, u, v):
        return driver._double_gyre_driver().seed_steady_forcing_carry(
            coarse_state_from_native(card.recipe.initial_state, ssh, u, v), forcing)

    def pack(au, av):
        return np.concatenate([np.asarray(au)[umask], np.asarray(av)[vmask]])

    def unpack(vec):
        au = np.zeros(umask.shape)
        av = np.zeros(vmask.shape)
        au[umask] = vec[:n_u]
        av[vmask] = vec[n_u:]
        return au, av

    def curl(au, av):
        return np.asarray(balanced_fields_from_native(
            jnp.asarray(au), jnp.asarray(av), geometry, objective="template").relative_vorticity_s1)

    def rel(a, b):
        return abs(a / b - 1.0) if b != 0 else abs(a)

    sel = {"all": np.ones(n_u + n_v, dtype=bool),
           "ring": np.concatenate([ring_u[umask], ring_v[vmask]]),
           "interior": np.concatenate([(umask & ~ring_u)[umask], (vmask & ~ring_v)[vmask]])}
    contract = SWGDatasetContract.from_json(eb.RUN / "dataset_contract.json", card=card)
    step8 = {r["start"]: r for r in json.loads(STEP8_REF.read_text())["per_start"]}
    rows = isw.starts(card)
    chosen = [rows[k] for k in START_INDICES]
    assert [r[0].split("_")[0] for r in chosen] == ["train", "train", "validation", "validation"]
    controls = {"step8_work_and_rms_max_rel": {}, "make_terms_rms_and_work_max_rel": {},
                "zero_potential_max_abs": {}, "work_density_sum_vs_face_sum_rel": {}}
    results, work_panels = [], []
    for label, hour, start, truth_1h, _ in chosen:
        ssh_r, u, v = (np.asarray(a, dtype=np.float64) for a in start)
        ssh_s, u_s, v_s = (np.asarray(a, dtype=np.float64)
                           for a in stored_state_at(contract.sources, contract.objective, hour)[0])
        assert np.array_equal(u, u_s) and np.array_equal(v, v_s), label
        model_s = native(hour_a(seeded(ssh_s, u, v)))
        aS = ((truth_1h[1] - model_s[1]) / isw.HOUR_S * umask,
              (truth_1h[2] - model_s[2]) / isw.HOUR_S * vmask)
        e3u, e3v, pred, zero = predicted(jnp.asarray(ssh_r), jnp.asarray(u), jnp.asarray(v))
        cross_u, cross_v = e2u * np.asarray(e3u), e1v * np.asarray(e3v)
        A, _ = rd.rotated_gradient_matrix(cross_u, cross_v, umask, vmask, vtx)
        proj = rd.RangeProjector(A * float(np.median(cross_u[umask])))
        rep_vec, _, _, _ = proj(pack(*aS))
        rep = unpack(rep_vec)
        # Control (a): step 8's stored numbers.
        ref = step8[label]
        mine = {"stored_launch": (work3(u, v, *aS), rd.isw_rms(aS, umask, vmask, n_u, n_v)),
                "representable_S": (work3(u, v, *rep), rd.isw_rms(rep, umask, vmask, n_u, n_v))}
        worst = max([rel(w3[k], ref["work"][part][k]) for part, (w3, _) in mine.items() for k in w3]
                    + [rel(r, ref["rms"][part]) for part, (_, r) in mine.items()])
        assert worst <= TOL, (label, worst)
        controls["step8_work_and_rms_max_rel"][label] = worst
        # Control (c): zero potential -> zero everywhere.
        z = max(float(np.abs(np.asarray(zero[0])).max()), float(np.abs(np.asarray(zero[1])).max()),
                float(np.abs(curl(*zero)).max()),
                float(np.abs(work_density_t(u, v, *(np.asarray(x) for x in zero), umask, vmask)).max()))
        assert z == 0.0, (label, z)
        controls["zero_potential_max_abs"][label] = z
        # Control (d): T-point work density sums to the face work.
        wd_rep = work_density_t(u, v, *rep, umask, vmask)
        face_sum = float(np.sum(u * rep[0] * umask) + np.sum(v * rep[1] * vmask))
        controls["work_density_sum_vs_face_sum_rel"][label] = rel(float(wd_rep.sum()), face_sum)
        assert controls["work_density_sum_vs_face_sum_rel"][label] <= 1e-12, label

        row = {"start": label, "absolute_hour": hour,
               "implied_representable_S": {"work": mine["representable_S"][0],
                                           "rms": mine["representable_S"][1]},
               "implied_stored_launch_S": {"work": mine["stored_launch"][0],
                                           "rms": mine["stored_launch"][1]},
               "networks": {}}
        fields = {"implied": rep}
        a_full = pack(*aS)
        for w in WINDOWS:
            au, av = (np.asarray(x) for x in pred[w])
            fields[w] = (au, av)
            vec = pack(au, av)
            t = {k: float(x) for k, x in terms[w](jnp.asarray(ssh_r), jnp.asarray(u), jnp.asarray(v)).items()}
            rms_w = rd.isw_rms((au, av), umask, vmask, n_u, n_v)
            worst = max(rel(rms_w, t["closure_accel_rms"]), rel(work(u, v, au, av), t["closure_work"]))
            assert worst <= TOL, (label, w, worst)
            controls["make_terms_rms_and_work_max_rel"][f"{label}/w{w:02d}"] = worst
            row["networks"][f"w{w:02d}"] = {
                "corr_vs_representable": {k: pearson(vec[s], rep_vec[s]) for k, s in sel.items()},
                "corr_vs_full_a_S": pearson(vec, a_full),
                "amplitude_ratio_rms": rms_w / mine["representable_S"][1],
                "work": work3(u, v, au, av), "rms": rms_w}
        results.append(row)
        work_panels.append((label, wd_rep, work_density_t(u, v, *fields[8], umask, vmask)))
        plot_snapshot(label, fields, row, curl, wet, vtx, (j0, j1, i0, i1), eb.DIAG)
        print(label, json.dumps({w: row["networks"][w]["corr_vs_representable"] for w in row["networks"]}))
        sys.stdout.flush()

    plot_work_density(work_panels, wet, (j0, j1, i0, i1), eb.DIAG)
    out = {"definition": __doc__, "units": {"accel": "m s-2", "curl": "s-2",
                                             "work": "m2 s-3 (per unit mass, mean over wet faces)"},
           "networks": {f"w{w:02d}": str(eb.RUN / "checkpoints" / f"w{w:02d}_e05.eqx") for w in WINDOWS},
           "step8_reference": str(STEP8_REF), "colour_scale": f"symmetric, {PERCENTILE}th percentile of "
           "|field| over the five rows' wet points, per column",
           "ring_choice": "wet face with either adjoining T cell in the outermost wet T row/column",
           "controls": controls, "per_start": results}
    (eb.DIAG / "summary.json").write_text(json.dumps(out, indent=1))
    print(table(results))
    print("controls (max):", json.dumps({k: max(v.values()) for k, v in controls.items()}))
    print("WROTE", eb.DIAG / "summary.json")


def table(results):
    lines = ["start                        net   corr_all  corr_int  corr_ring  amp_ratio"]
    for row in results:
        for w, r in row["networks"].items():
            c = r["corr_vs_representable"]
            lines.append(f"{row['start']:28s} {w}  {c['all']:+8.3f}  {c['interior']:+8.3f}  "
                         f"{c['ring']:+9.3f}  {r['amplitude_ratio_rms']:9.3f}")
    return "\n".join(lines)


def _crop(field, mask, bounds):
    j0, j1, i0, i1 = bounds
    return np.where(mask, field, np.nan)[j0:j1 + 1, i0:i1 + 1]


def plot_snapshot(label, fields, row, curl, wet, vtx, bounds, out_dir):
    names = ["implied"] + list(WINDOWS)
    umask_t, vmask_t = wet, wet   # faces drawn on their T cell (east/north face of T)
    panels = {n: (curl(*fields[n]), fields[n][0], fields[n][1]) for n in names}
    masks = (vtx, umask_t, vmask_t)
    vmax = [max(np.nanpercentile(np.abs(_crop(panels[n][c], masks[c], bounds)), PERCENTILE)
                for n in names) for c in range(3)]
    fig, axes = plt.subplots(5, 3, figsize=(12, 15), constrained_layout=True)
    cols = ["curl of acceleration (s$^{-2}$)", "a$_u$ (m s$^{-2}$)", "a$_v$ (m s$^{-2}$)"]
    for r, n in enumerate(names):
        for c in range(3):
            ax = axes[r, c]
            im = ax.imshow(_crop(panels[n][c], masks[c], bounds), origin="lower", cmap="RdBu_r",
                           vmin=-vmax[c], vmax=vmax[c], interpolation="nearest")
            if r == 0:
                ax.set_title(cols[c] if c else f"implied (representable)\n{cols[c]}", fontsize=10)
            else:
                net = row["networks"][f"w{n:02d}"]
                cc = net["corr_vs_representable"]
                ax.set_title(f"w={n}: corr {cc['all']:+.2f} (int {cc['interior']:+.2f}, "
                             f"ring {cc['ring']:+.2f}), amp {net['amplitude_ratio_rms']:.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 4:
                fig.colorbar(im, ax=axes[:, c], shrink=0.4, location="bottom", pad=0.01)
    fig.suptitle(f"{label}: implied sub-grid acceleration vs pilot-2 networks (epoch 5)", fontsize=12)
    fig.savefig(out_dir / f"snapshot_{label}.png", dpi=110)
    plt.close(fig)


def plot_work_density(panels, wet, bounds, out_dir):
    fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 7.5), constrained_layout=True)
    for c, (label, implied, pred) in enumerate(panels):
        vmax = max(np.nanpercentile(np.abs(_crop(f, wet, bounds)), PERCENTILE) for f in (implied, pred))
        for r, (name, f) in enumerate((("implied (representable)", implied), ("w=8 network", pred))):
            im = axes[r, c].imshow(_crop(f, wet, bounds), origin="lower", cmap="RdBu_r",
                                   vmin=-vmax, vmax=vmax, interpolation="nearest")
            axes[r, c].set_title(f"{label}\n{name}: mean {np.nanmean(_crop(f, wet, bounds)):+.2e}",
                                 fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        fig.colorbar(im, ax=axes[:, c], shrink=0.5, location="bottom", pad=0.01)
    fig.suptitle("work density u a$_u$ + v a$_v$ at T points (m$^2$ s$^{-3}$)", fontsize=12)
    fig.savefig(out_dir / "work_density_w08.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
