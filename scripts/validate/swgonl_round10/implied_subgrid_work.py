"""Round-10 step 7: the sub-grid forcing the data implies, a priori.

For twenty real window starts (the first window of each of the first ten
w=1 training batches, epoch-0 sampler, and the first ten validation-lattice
windows; all rebalanced by the contract's own start_state, exactly as the
training uses them) the coarse model is stepped WITHOUT closure from the
start through one hour (9 steps of 400 s, swg_online_step with no network,
the free-run stepper) and its one-hour velocity change is compared with the
filtered truth's: the batch target at lead 1 is the stored filtered state one
hour after the start, so the implied sub-grid acceleration on the native U
and V faces is

    a_sg = (truth(h+1) - model(h+1)) / 3600 s

(both tendencies start from the same stored velocities; only the start's
sea level was rebalanced).  Per start: rms and max of a_sg against the wind
acceleration (stage-3 program, wind on minus off at the start entry, as
energy_budget.make_terms) and against the trained closures' acceleration;
the WORK of a_sg per unit mass with the start velocity, mean over wet U
faces of u*a_u plus mean over wet V faces of v*a_v [m2 s-3], against the
wind work and the closures' work; that work split into the first interior
wall ring and the interior; the cosine between the trained w=2 (epoch 5)
closure acceleration and a_sg over wet faces.  Also the KE of the filtered
truth at the start and one day later (target lead 24) next to the coarse
model's KE after 24 h without closure.

Ring choice: a wet face is in the "first interior wall ring" when either of
its two adjoining T cells lies in the outermost wet T row or column (the T
cells whose corners are the F points the ring-excluded loss zeroes).

Controls: (a) the one-hour change by the training lane's stepper
(swg_online_step, no network, 9-step scan) equals the double-gyre driver's
own scanned stepper (LatLonCGridOceanModel.step) to 1e-12 m/s; (b) a_sg
with the truth replaced by the model's own one-hour state is exactly 0;
(c) the closure and wind acceleration fields rebuilt here reproduce
energy_budget.make_terms' rms/work on every start to 1e-12 relative.
Usage: implied_subgrid_work.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import energy_budget as eb  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.dynamics.swe_rk3 import _entry_from_state, swe_rk3_stage  # noqa: E402
from legoesm.training.swg_online import (  # noqa: E402
    SWGDatasetContract,
    coarse_state_from_native,
    swg_online_step,
)

OUT = eb.DIAG / "step7_implied_subgrid_work.json"
N_TRAIN_BATCHES = 10
N_VALIDATION_STARTS = 10
HOUR_S = 3600.0
HOURS_PER_DAY = 24


def starts(card):
    """[(label, absolute start hour, start (ssh,u,v), truth at +1 h, truth at +24 h)]."""
    contract = SWGDatasetContract.from_json(eb.RUN / "dataset_contract.json", card=card)
    rows = []
    train = contract.batches(split="train", window_days=1, epoch=0, seed=eb.SEED,
        target_interval_hours=1)
    for index in range(N_TRAIN_BATCHES):
        batch = next(train)
        rows.append((f"train_batch{index:02d}_win0_h{int(batch.start_absolute_hour[0])}",
                     int(batch.start_absolute_hour[0]),
                     tuple(np.asarray(a[0]) for a in batch.initial),
                     tuple(np.asarray(a[0, 0]) for a in batch.target),
                     tuple(np.asarray(a[0, HOURS_PER_DAY - 1]) for a in batch.target)))
    batch = next(contract.batches(split="validation", window_days=1, target_interval_hours=1))
    for index in range(N_VALIDATION_STARTS):
        rows.append((f"validation_win{index}_h{int(batch.start_absolute_hour[index])}",
                     int(batch.start_absolute_hour[index]),
                     tuple(np.asarray(a[index]) for a in batch.initial),
                     tuple(np.asarray(a[index, 0]) for a in batch.target),
                     tuple(np.asarray(a[index, HOURS_PER_DAY - 1]) for a in batch.target)))
    return rows


def face_rings(geometry):
    """(ring_u, ring_v, umask, vmask): first-interior-ring wet faces, native layout."""
    tmask = np.asarray(geometry.tmask, dtype=bool)
    umask = np.asarray(geometry.umask, dtype=bool)
    vmask = np.asarray(geometry.vmask, dtype=bool)
    j0, j1, i0, i1 = geometry.wet_bounds
    # Native U (j, i) is the east face of T (j, i); V (j, i) the north face.
    east = np.zeros_like(tmask)
    east[:, :-1] = tmask[:, 1:]
    north = np.zeros_like(tmask)
    north[:-1, :] = tmask[1:, :]
    assert np.array_equal(umask, tmask & east), "U face convention is not east-of-T"
    assert np.array_equal(vmask, tmask & north), "V face convention is not north-of-T"
    tring = np.zeros_like(tmask)
    tring[[j0, j1], i0:i1 + 1] = True
    tring[j0:j1 + 1, [i0, i1]] = True
    tring &= tmask
    ring_u = umask & (tring | np.roll(tring, -1, axis=1))
    ring_v = vmask & (tring | np.roll(tring, -1, axis=0))
    return ring_u, ring_v, umask, vmask


def main():
    driver, config, card, geometry, nets, forcing, key = eb.setup()
    ring_u, ring_v, umask, vmask = face_rings(geometry)
    n_u, n_v = int(umask.sum()), int(vmask.sum())
    template = card.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    grid, z_coord, cfg = card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
    mask = template.land_mask.data
    tau = (forcing.tau_i_native, forcing.tau_j_native)
    dt = card.dt_s
    steps_per_hour = int(HOUR_S / dt)
    assert steps_per_hour * dt == HOUR_S == 9 * 400.0

    # Route A: the training lane's free-run stepper (swg_online_step, no network).
    def step_a(state, _):
        return swg_online_step(None, state, None, None, None, card=card,
                               geometry=geometry, forcing=forcing), None

    @jax.jit
    def hour_a(state):
        state, _ = jax.lax.scan(step_a, state, None, length=steps_per_hour)
        return state

    # Route B: the double-gyre driver's own scanned stepper on the card's model.
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    hour_b = driver._double_gyre_driver().build_scanned_stepper(model, dt, forcing, steps_per_hour)

    net = nets[2]

    @jax.jit
    def fields(ssh, u, v):
        """Closure (w02_e05) and wind acceleration fields at a state, as make_terms."""
        state = coarse_state_from_native(template, ssh, u, v)
        entry = _entry_from_state(state, tau, tau, grid, z_coord, cfg)
        zero = (jnp.zeros_like(tau[0]), jnp.zeros_like(tau[1]))
        entry_nw = entry._replace(utau_before=zero[0], vtau_before=zero[1],
                                  utau_now=zero[0], vtau_now=zero[1])

        def stage3(e, potential):
            out = swe_rk3_stage(e, e, dt, 3, grid, z_coord, mask, cfg, potential_f=potential)
            return out.u_k, out.v_k

        base = stage3(entry, None)
        potential = net(entry, geometry, key=jax.random.fold_in(key, 1),
                        noise=jnp.zeros_like(entry.ssh_k), noise_scale=jnp.asarray(0.0))
        clo = stage3(entry, potential)
        nw = stage3(entry_nw, None)
        return ((clo[0] - base[0]) / dt, (clo[1] - base[1]) / dt,
                (base[0] - nw[0]) / dt, (base[1] - nw[1]) / dt)

    terms_fn = jax.jit(eb.make_terms(card, geometry, net, forcing, key))

    def native(state):
        return (np.asarray(state.eta.data), np.asarray(state.u.data[:, 1:, 0]),
                np.asarray(state.v.data[1:, :, 0]))

    def ke(u, v):
        return 0.5 * (np.sum(u**2 * umask) / n_u + np.sum(v**2 * vmask) / n_v)

    def rms(au, av):
        return float(np.sqrt(0.5 * (np.sum(au**2 * umask) / n_u + np.sum(av**2 * vmask) / n_v)))

    def amax(au, av):
        return float(max(np.abs(au * umask).max(), np.abs(av * vmask).max()))

    def work(u, v, au, av, sel_u=umask, sel_v=vmask):
        return float(np.sum(u * au * sel_u) / n_u + np.sum(v * av * sel_v) / n_v)

    def implied(truth_1h, model_1h):
        """a_sg on U and V faces: (truth - model) one-hour change / 3600 s."""
        return ((truth_1h[1] - model_1h[1]) / HOUR_S * umask,
                (truth_1h[2] - model_1h[2]) / HOUR_S * vmask)

    def seeded(ssh, u, v):
        return driver._double_gyre_driver().seed_steady_forcing_carry(
            coarse_state_from_native(card.recipe.initial_state, ssh, u, v), forcing)

    rows = starts(card)
    print("starts:", [r[0] for r in rows])
    out_rows = []
    controls = {"stepper_route_max_abs_diff_m_s": [], "self_truth_a_sg_max": [],
                "fields_vs_make_terms_rel": []}
    for label, _hour, start, truth_1h, truth_24h in rows:
        ssh, u, v = (np.asarray(a, dtype=np.float64) for a in start)
        state = seeded(ssh, u, v)
        # Control (a): both steppers give the same one-hour state.
        s_a = hour_a(state)
        s_b = hour_b(state)
        diff = max(float(np.abs(np.asarray(s_a.u.data) - np.asarray(s_b.u.data)).max()),
                   float(np.abs(np.asarray(s_a.v.data) - np.asarray(s_b.v.data)).max()),
                   float(np.abs(np.asarray(s_a.eta.data) - np.asarray(s_b.eta.data)).max()))
        assert diff <= 1e-12, (label, diff)
        controls["stepper_route_max_abs_diff_m_s"].append(diff)
        model_1h = native(s_a)
        a_u, a_v = implied(truth_1h, model_1h)
        # Control (b): truth replaced by the model's own trajectory -> exactly 0.
        z_u, z_v = implied(model_1h, model_1h)
        assert float(np.abs(z_u).max()) == 0.0 and float(np.abs(z_v).max()) == 0.0
        controls["self_truth_a_sg_max"].append(float(max(np.abs(z_u).max(), np.abs(z_v).max())))
        # Closure and wind fields; control (c) against make_terms.
        c_u, c_v, w_u, w_v = (np.asarray(x) for x in fields(jnp.asarray(ssh), jnp.asarray(u), jnp.asarray(v)))
        terms = {k: float(x) for k, x in terms_fn(jnp.asarray(ssh), jnp.asarray(u), jnp.asarray(v)).items()}
        rel = max(abs(rms(c_u, c_v) / terms["closure_accel_rms"] - 1.0),
                  abs(work(u, v, c_u, c_v) / terms["closure_work"] - 1.0),
                  abs(work(u, v, w_u, w_v) / terms["wind_work"] - 1.0))
        assert rel <= 1e-12, (label, rel)
        controls["fields_vs_make_terms_rel"].append(rel)
        # 24 h without closure.
        s24 = state
        for _ in range(HOURS_PER_DAY):
            s24 = hour_a(s24)
        m24 = native(s24)
        dot = np.sum(c_u * a_u * umask) + np.sum(c_v * a_v * vmask)
        cos = float(dot / np.sqrt((np.sum(c_u**2 * umask) + np.sum(c_v**2 * vmask))
                                  * (np.sum(a_u**2 * umask) + np.sum(a_v**2 * vmask))))
        # Least-squares scale of the closure along a_sg: a_sg ~ s * a_clo.
        scale = float(dot / (np.sum(c_u**2 * umask) + np.sum(c_v**2 * vmask)))
        row = {
            "start": label,
            "a_sg_rms": rms(a_u, a_v), "a_sg_max": amax(a_u, a_v),
            "wind_accel_rms": rms(w_u, w_v), "wind_accel_max": amax(w_u, w_v),
            "closure_w02_accel_rms": terms["closure_accel_rms"],
            "a_sg_work": work(u, v, a_u, a_v),
            "a_sg_work_ring": work(u, v, a_u, a_v, ring_u, ring_v),
            "a_sg_work_interior": work(u, v, a_u, a_v, umask & ~ring_u, vmask & ~ring_v),
            "wind_work": terms["wind_work"],
            "closure_w02_work": terms["closure_work"],
            "drag_work": terms["drag_work"], "viscous_work": terms["viscous_work"],
            "cos_closure_w02_vs_a_sg": cos,
            "closure_w02_lsq_scale_onto_a_sg": scale,
            "ke_truth_start": float(ke(u, v)),
            "ke_truth_1h": float(ke(truth_1h[1], truth_1h[2])),
            "ke_model_1h": float(ke(model_1h[1], model_1h[2])),
            "ke_truth_24h": float(ke(truth_24h[1], truth_24h[2])),
            "ke_model_24h": float(ke(m24[1], m24[2])),
        }
        row["model_minus_truth_ke_gain_per_day"] = (
            (row["ke_model_24h"] - row["ke_truth_start"]) - (row["ke_truth_24h"] - row["ke_truth_start"]))
        out_rows.append(row)
        print(f"{label:28s} a_sg rms {row['a_sg_rms']:.3e} max {row['a_sg_max']:.3e} "
              f"| wind rms {row['wind_accel_rms']:.3e} | work a_sg {row['a_sg_work']:+.3e} "
              f"(ring {row['a_sg_work_ring']:+.3e} int {row['a_sg_work_interior']:+.3e}) "
              f"wind {row['wind_work']:+.3e} clo {row['closure_w02_work']:+.3e} "
              f"| cos {cos:+.3f} | KE truth0 {row['ke_truth_start']:.4f} truth24 {row['ke_truth_24h']:.4f} "
              f"model24 {row['ke_model_24h']:.4f}")
        sys.stdout.flush()

    def stat(key):
        x = np.array([r[key] for r in out_rows])
        return {"mean": float(x.mean()), "median": float(np.median(x)),
                "min": float(x.min()), "max": float(x.max())}

    summary = {k: stat(k) for k in (
        "a_sg_rms", "a_sg_max", "wind_accel_rms", "a_sg_work", "a_sg_work_ring",
        "a_sg_work_interior", "wind_work", "closure_w02_work", "cos_closure_w02_vs_a_sg",
        "closure_w02_lsq_scale_onto_a_sg", "ke_truth_start", "ke_truth_24h", "ke_model_24h",
        "model_minus_truth_ke_gain_per_day")}
    summary["a_sg_work_over_wind_work_mean"] = summary["a_sg_work"]["mean"] / summary["wind_work"]["mean"]
    summary["a_sg_rms_over_wind_accel_rms_mean"] = summary["a_sg_rms"]["mean"] / summary["wind_accel_rms"]["mean"]
    summary["a_sg_work_sink_fraction_of_starts"] = float(np.mean([r["a_sg_work"] < 0 for r in out_rows]))
    out = {
        "definition": __doc__,
        "units": {"accel": "m s-2", "work": "m2 s-3 (per unit mass, mean over wet faces)",
                  "ke": "m2 s-2", "ke_gain": "m2 s-2 per day"},
        "machinery": "swg_online_step (no network) 9-step scan, CPU/libm fp64; "
                     "closure/wind fields = stage-3 on/off at the start entry (energy_budget.make_terms)",
        "reference_closure_accel_rms_step6": {"w01_e05": 4.434116523541503e-08, "w02_e05": 5.464487152048272e-08},
        "reference_closure_work_step6": {"w01_e05": 2.652447170004434e-09, "w02_e05": 1.4320505513246925e-09},
        "ring_choice": "wet face with either adjoining T cell in the outermost wet T row/column",
        "ring_face_counts": {"u": int(ring_u.sum()), "v": int(ring_v.sum()), "wet_u": n_u, "wet_v": n_v},
        "controls": controls,
        "per_start": out_rows,
        "summary": summary,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
