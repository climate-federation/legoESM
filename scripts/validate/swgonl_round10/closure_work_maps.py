"""Round-11: where the trained closures put and take kinetic energy (Maddison 2026 Fig. 10 analogue).

For every window length of the run (the checkpoints w{01,02,04,08}_e05) the
closure's acceleration at a stored state is the model's own stage-3 program
differenced closure-on minus closure-off (energy_budget.make_terms, reused),
and its LOCAL work per unit mass is the per-face product u*a_u on wet U faces
and v*a_v on wet V faces [m2 s-3].  A T-point map is the half-sum of the two
adjoining U-face products plus the half-sum of the two adjoining V-face
products (closed wall faces contribute zero), so the sum of the map over wet
T cells equals the sum over wet faces and the probe's basin-mean work (mean
over wet U faces plus mean over wet V faces, n_u == n_v) is the map's wet-T
sum divided by n_u.  Titles quote that face-mean convention (the one every
round-10 number uses); the plain wet-T-cell mean is the same number times
n_u / n_T and both are in the json.

Products
1. closure_work_maps_by_window.png: time-mean map per window length, from
   that window's own 20-year free run from rest, end-of-day snapshots every
   10 days over years 2-20 (days 366, 376, ..., 7296), one shared symmetric
   colour scale (99.5th percentile of |map| over the four panels).
2. data_implied_vs_w08_maps.png: the time-mean map of the data-implied
   one-hour correction on the twenty step-7 starts, matched-start version
   (coarse model launched from the STORED start, residual against the stored
   truth one hour later, projected onto the range of the closure's rotated-
   gradient operator at the launch state exactly as residual_decomposition's
   representable_S), beside the trained w=8 closure's work on the same
   twenty starts (closure evaluated at the rebalanced start, the training's
   state and the step-6 convention), same colour scale.
3. closure_work_spectra.png: isotropic wavenumber spectrum of the closure's
   energy input per window, and of the data-implied representable
   correction.  Maddison's section 3.5 states only the local quantity (his
   eq. 9, -psi*S up to a flux term) and that spectra come from the sine
   transform of the streamfunction; the spectral forcing itself is not
   written out, so ours is stated here: velocities and accelerations
   averaged to T points on the 80x80 wet rectangle, each field reflected
   evenly across both walls (the cosine-transform periodisation, which
   removes the wall discontinuity without a taper), 2-D DFT, cross-spectrum
   Re[conj(u_hat) a_u_hat + conj(v_hat) a_v_hat] normalised so that its sum
   over all wavenumbers equals the wet-T mean of u_T*a_uT + v_T*a_vT, binned
   into unit shells of |k| in cycles per basin width L = 80 * 50 km (the
   reflection doubles the period, so shells are half-integer in L-cycles and
   are reported at their centres).  Time mean of the per-sample spectra.
4. closure_work_maps.json: per-window basin means (face-mean and T-mean
   conventions), wall-ring / interior split (implied_subgrid_work.face_rings
   convention: a wet face with either adjoining T cell in the outermost wet
   T row/column; both parts normalised by the full counts so ring +
   interior == total), the step-6 reference on the fifteen training/
   validation starts and the difference, the spectra arrays, controls.
   closure_work_maps.npz holds the maps.

Controls, asserted: (a) on every sample the map's wet-T sum / n_u equals
make_terms' closure_work to 1e-10 relative; (b) a zero-potential closure's
map is exactly zero on the first sample of every window; (c) on the fifteen
starts the step-6 probe scored, the w=8 closure's per-start work reproduces
step 6's stored value to 1e-10 relative; (d) the projected data-implied
correction reproduces residual_decomposition's control that the closure's
own acceleration is in range (remainder <= 1e-12 relative) and its
wet-T-sum / n_u equals the face work of the projected field exactly;
(e) each spectrum's sum equals its sample-mean T-point work to 1e-10.
Usage: SWGONL_RUN=<run dir> SWGONL_DIAG=<output dir> closure_work_maps.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d  # noqa: E402
from legoesm.ocean.dynamics.swe_rk3 import _entry_from_state, swe_rk3_stage  # noqa: E402
from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands  # noqa: E402
from legoesm.training.swg_online import (  # noqa: E402
    SWGDatasetContract,
    coarse_state_from_native,
    stored_state_at,
    swg_online_step,
)

WINDOWS = (1, 2, 4, 8)
FIRST_DAY, LAST_DAY, STRIDE_DAYS = 366, 7300, 10
HOUR_S = isw.HOUR_S
# Step-6 receipts per window (the round root holds w1/w2; w4 and w8 were run separately).
STEP6 = {1: "step6_closure_work_by_epoch.json", 2: "step6_closure_work_by_epoch.json",
         4: "step6_w4/step6_closure_work_by_epoch.json", 8: "step6_w8/step6_closure_work_by_epoch.json"}


def to_t(p_u, p_v):
    """Half-sum of the two adjoining U-face values plus the two adjoining
    V-face values at each T point; U(j,i) is the east face of T(j,i), V(j,i)
    its north face, so the west/south neighbours are the i-1 / j-1 faces."""
    west = np.concatenate([np.zeros_like(p_u[:, :1]), p_u[:, :-1]], axis=1)
    south = np.concatenate([np.zeros_like(p_v[:1]), p_v[:-1]], axis=0)
    return 0.5 * (p_u + west) + 0.5 * (p_v + south)


def isotropic_cross_spectrum(u_t, v_t, au_t, av_t, bounds):
    """(k centres in cycles per basin width, shell spectrum [m2 s-3]) of
    Re[conj(u_hat) a_hat] on the wet rectangle, evenly reflected (see module
    docstring); the shell sum equals the wet-T mean of u_t*au_t + v_t*av_t."""
    j0, j1, i0, i1 = bounds
    fields = []
    for f in (u_t, v_t, au_t, av_t):
        r = f[j0:j1 + 1, i0:i1 + 1]
        r = np.concatenate([r, r[::-1]], axis=0)
        r = np.concatenate([r, r[:, ::-1]], axis=1)
        fields.append(np.fft.fft2(r))
    ny, nx = fields[0].shape
    cross = (np.conj(fields[0]) * fields[2] + np.conj(fields[1]) * fields[3]).real / (ny * nx) ** 2
    ky = np.fft.fftfreq(ny) * ny / 2.0   # cycles per basin width (period is 2L)
    kx = np.fft.fftfreq(nx) * nx / 2.0
    kk = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    edges = np.arange(0.0, np.floor(kk.max()) + 2.0, 1.0)
    which = np.digitize(kk.ravel(), edges) - 1
    spectrum = np.bincount(which, weights=cross.ravel(), minlength=len(edges))[:len(edges) - 1]
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, spectrum


def main():
    driver, config, card, geometry, nets, forcing, key = eb.setup()
    assert set(nets) >= set(WINDOWS), sorted(nets)
    ring_u, ring_v, umask, vmask = isw.face_rings(geometry)
    tmask = np.asarray(geometry.tmask, dtype=bool)
    n_u, n_v, n_t = int(umask.sum()), int(vmask.sum()), int(tmask.sum())
    assert n_u == n_v
    bounds = geometry.wet_bounds
    vtx = np.asarray(geometry.vertex_interior, dtype=bool)
    template = card.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    grid, z_coord, cfg = card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
    mask = template.land_mask.data
    tau = (forcing.tau_i_native, forcing.tau_j_native)
    dt = card.dt_s
    steps_per_hour = int(HOUR_S / dt)
    assert steps_per_hour * dt == HOUR_S
    umask3, vmask3 = compute_face_masks_3d(z_coord.is_active, grid)
    ops = nemo_qco_resolved_mesh_operands(z_coord, grid, umask3, vmask3, jnp.float64, 1)
    e2u = np.asarray(ops.e2u, dtype=np.float64)
    e1v = np.asarray(ops.e1v, dtype=np.float64)
    round_root = eb.RUN.parents[2]

    def accel_fn(network):
        """jit: (ssh, u, v) -> closure acceleration on U/V faces [m s-2] and
        the live face thicknesses e3u, e3v at that state."""
        @jax.jit
        def fn(ssh, u, v):
            state = coarse_state_from_native(template, ssh, u, v)
            entry = _entry_from_state(state, tau, tau, grid, z_coord, cfg)

            def stage3(potential):
                out = swe_rk3_stage(entry, entry, dt, 3, grid, z_coord, mask, cfg, potential_f=potential)
                return out.u_k, out.v_k

            base = stage3(None)
            potential = network(entry, geometry, key=jax.random.fold_in(key, 1),
                                noise=jnp.zeros_like(entry.ssh_k), noise_scale=jnp.asarray(0.0))
            clo = stage3(potential)
            one = jnp.asarray(1.0, dtype=jnp.float64)
            e3u = (ops.e3u_0 * (one + entry.r3u_k[..., None] * ops.umask3))[..., 0]
            e3v = (ops.e3v_0 * (one + entry.r3v_k[..., None] * ops.vmask3))[..., 0]
            return (clo[0] - base[0]) / dt, (clo[1] - base[1]) / dt, e3u, e3v
        return fn

    def face_work(u, v, au, av, sel_u=umask, sel_v=vmask):
        return float(np.sum(u * au * sel_u) / n_u + np.sum(v * av * sel_v) / n_v)

    def t_velocity(u, v):
        return to_t(u * umask, np.zeros_like(u)), to_t(np.zeros_like(v), v * vmask)

    class Accumulator:
        """Time means of the T-point work map, the ring/interior/total face
        work, and the isotropic cross-spectrum."""

        def __init__(self):
            self.n, self.map, self.total, self.ring, self.interior = 0, np.zeros(tmask.shape), 0.0, 0.0, 0.0
            self.spectrum, self.k, self.t_work = None, None, 0.0

        def add(self, u, v, au, av):
            p_u, p_v = u * au * umask, v * av * vmask
            m = to_t(p_u, p_v) * tmask
            total = face_work(u, v, au, av)
            assert abs(m.sum() / n_u - total) <= 1e-10 * max(abs(total), 1e-30), (m.sum() / n_u, total)
            u_t, v_t = t_velocity(u, v)
            au_t, av_t = t_velocity(au, av)
            k, s = isotropic_cross_spectrum(u_t, v_t, au_t, av_t, bounds)
            t_work = float(np.sum((u_t * au_t + v_t * av_t) * tmask) / n_t)
            assert abs(s.sum() - t_work) <= 1e-10 * max(abs(t_work), 1e-30), (s.sum(), t_work)
            self.n += 1
            self.map += m
            self.total += total
            self.ring += face_work(u, v, au, av, ring_u, ring_v)
            self.interior += face_work(u, v, au, av, umask & ~ring_u, vmask & ~ring_v)
            self.t_work += t_work
            self.k = k
            self.spectrum = s if self.spectrum is None else self.spectrum + s
            return total

        def result(self):
            n = self.n
            m = self.map / n
            return {"samples": n, "map": m,
                    "basin_mean_face_convention": float(m.sum() / n_u),
                    "basin_mean_wet_t_cells": float(m.sum() / n_t),
                    "ring": self.ring / n, "interior": self.interior / n,
                    "t_point_product_mean": self.t_work / n,
                    "k_cycles_per_basin_width": self.k.tolist(),
                    "spectrum_m2_s3": (self.spectrum / n).tolist()}

    # (1) Free-run maps per window.
    days = list(range(FIRST_DAY, LAST_DAY + 1, STRIDE_DAYS))
    out = {"definition": __doc__, "units": {"work": "m2 s-3 per unit mass", "accel": "m s-2",
                                             "spectrum": "m2 s-3 per unit shell of |k| (sum = T-point work)"},
           "sampling": {"free_run_days": [days[0], days[-1], STRIDE_DAYS, len(days)],
                        "basin_width_m": geometry.spacing_m * (bounds[3] - bounds[2] + 1)},
           "face_counts": {"u": n_u, "v": n_v, "wet_t": n_t, "ring_u": int(ring_u.sum()), "ring_v": int(ring_v.sum())},
           "controls": {"map_sum_vs_make_terms_rel_max": {}, "zero_potential_map_max_abs": {},
                        "w08_vs_step6_per_start_rel_max": None},
           "per_window": {}, "data_implied": {}}
    maps = {}
    shape = None
    for w in WINDOWS:
        net = nets[w]
        fn = accel_fn(net)
        terms = jax.jit(eb.make_terms(card, geometry, net, forcing, key))
        acc = Accumulator()
        rel_max = 0.0
        root = eb.RUN / "free_runs" / f"w{w:02d}"
        assert json.loads((root / "manifest.json").read_text())["completed_days"] >= LAST_DAY
        for n, day in enumerate(days):
            with np.load(root / f"day{day:04d}.npz") as f:
                ssh, u, v = (np.asarray(f[k], dtype=np.float64) for k in ("ssh_m", "u_m_s", "v_m_s"))
            au, av, _, _ = (np.asarray(x) for x in fn(jnp.asarray(ssh), jnp.asarray(u), jnp.asarray(v)))
            total = acc.add(u, v, au, av)
            ref = float(terms(jnp.asarray(ssh), jnp.asarray(u), jnp.asarray(v))["closure_work"])
            rel = abs(total - ref) / max(abs(ref), 1e-30)
            assert rel <= 1e-10, (w, day, total, ref)
            rel_max = max(rel_max, rel)
            if n == 0:
                # Control (b): zero potential -> exactly zero map.
                entry0 = _entry_from_state(coarse_state_from_native(template, jnp.asarray(ssh), jnp.asarray(u), jnp.asarray(v)),
                                           tau, tau, grid, z_coord, cfg)
                shape = tuple(net(entry0, geometry, key=key, noise=jnp.zeros_like(entry0.ssh_k),
                                  noise_scale=jnp.asarray(0.0)).shape)
                zero = accel_fn(lambda entry, geometry, *, key, noise, noise_scale: jnp.zeros(shape, jnp.float64))
                z_u, z_v, _, _ = (np.asarray(x) for x in zero(jnp.asarray(ssh), jnp.asarray(u), jnp.asarray(v)))
                z_map = to_t(u * z_u * umask, v * z_v * vmask)
                assert float(np.abs(z_map).max()) == 0.0
                out["controls"]["zero_potential_map_max_abs"][f"w{w:02d}"] = float(np.abs(z_map).max())
            if n % 100 == 0:
                print(f"w{w:02d} day {day} work {total:+.3e}")
                sys.stdout.flush()
        res = acc.result()
        maps[f"w{w:02d}"] = res.pop("map")
        step6 = json.loads((round_root / STEP6[w]).read_text())["per_network"][f"w{w:02d}_e05"]
        res["step6_reference"] = {
            "closure_work_mean": step6["closure_work_mean"],
            "closure_work_ring_mean": step6["closure_work_ring_mean"],
            "closure_work_interior_mean": step6["closure_work_interior_mean"],
            "states": "15 rebalanced training/validation starts of the filtered truth (step 6)"}
        res["free_run_minus_step6"] = res["basin_mean_face_convention"] - step6["closure_work_mean"]
        res["free_run_over_step6"] = res["basin_mean_face_convention"] / step6["closure_work_mean"]
        out["per_window"][f"w{w:02d}"] = res
        out["controls"]["map_sum_vs_make_terms_rel_max"][f"w{w:02d}"] = rel_max
        print(f"w{w:02d}: samples {res['samples']} basin mean {res['basin_mean_face_convention']:+.3e} "
              f"(ring {res['ring']:+.3e} interior {res['interior']:+.3e}) step6 {step6['closure_work_mean']:+.3e}")
        sys.stdout.flush()

    # (2) Data-implied correction on the twenty step-7 starts, matched start,
    # representable part; the w=8 closure on the same starts.
    def step_a(state, _):
        return swg_online_step(None, state, None, None, None, card=card,
                               geometry=geometry, forcing=forcing), None

    @jax.jit
    def hour_a(state):
        state, _ = jax.lax.scan(step_a, state, None, length=steps_per_hour)
        return state

    def seeded(ssh, u, v):
        return driver._double_gyre_driver().seed_steady_forcing_carry(
            coarse_state_from_native(card.recipe.initial_state, ssh, u, v), forcing)

    def native(state):
        return (np.asarray(state.eta.data), np.asarray(state.u.data[:, 1:, 0]),
                np.asarray(state.v.data[1:, :, 0]))

    def pack(au, av):
        return np.concatenate([au[umask], av[vmask]])

    def unpack(vec):
        au, av = np.zeros(umask.shape), np.zeros(vmask.shape)
        au[umask], av[vmask] = vec[:n_u], vec[n_u:]
        return au, av

    contract = SWGDatasetContract.from_json(eb.RUN / "dataset_contract.json", card=card)
    fn8 = accel_fn(nets[8])
    step6_8 = {s["start"]: s["closure_work"] for s in
               json.loads((round_root / STEP6[8]).read_text())["per_network"]["w08_e05"]["samples"]}
    acc_rep, acc_raw, acc_w8 = Accumulator(), Accumulator(), Accumulator()
    rel6, in_range, proj_rel = 0.0, 0.0, 0.0
    starts = isw.starts(card)
    for label, hour, start, truth_1h, _ in starts:
        ssh_r, u, v = (np.asarray(a, dtype=np.float64) for a in start)
        ssh_s, u_s, v_s = (np.asarray(a, dtype=np.float64)
                           for a in stored_state_at(contract.sources, contract.objective, hour)[0])
        assert np.array_equal(u, u_s) and np.array_equal(v, v_s), label
        model_s = native(hour_a(seeded(ssh_s, u, v)))
        a_u = (truth_1h[1] - model_s[1]) / HOUR_S * umask
        a_v = (truth_1h[2] - model_s[2]) / HOUR_S * vmask
        c_u, c_v, e3u, e3v = (np.asarray(x) for x in fn8(jnp.asarray(ssh_r), jnp.asarray(u), jnp.asarray(v)))
        A, _ = rd.rotated_gradient_matrix(e2u * e3u, e1v * e3v, umask, vmask, vtx)
        proj = rd.RangeProjector(A * float(np.median((e2u * e3u)[umask])))
        # Control (d): the closure's acceleration is in range.
        c_vec = pack(c_u, c_v)
        _, rem, _, _ = proj(c_vec)
        in_range = max(in_range, float(np.linalg.norm(rem) / np.linalg.norm(c_vec)))
        assert in_range <= 1e-12, (label, in_range)
        rep, _, _, res = proj(pack(a_u, a_v))
        proj_rel = max(proj_rel, res)
        r_u, r_v = unpack(rep)
        acc_rep.add(u, v, r_u, r_v)
        acc_raw.add(u, v, a_u, a_v)
        w8 = acc_w8.add(u, v, c_u, c_v)
        # Control (c): step 6's stored per-start value for the fifteen shared starts.
        if label in step6_8:
            rel6 = max(rel6, abs(w8 - step6_8[label]) / abs(step6_8[label]))
            assert rel6 <= 1e-10, (label, w8, step6_8[label])
        print(f"{label:28s} a_S rep work {acc_rep.total / acc_rep.n:+.3e} raw {acc_raw.total / acc_raw.n:+.3e} w08 {w8:+.3e}")
        sys.stdout.flush()
    assert len(step6_8) == 15 and sum(l in step6_8 for l, *_ in starts) == 15
    for name, acc in (("representable_S", acc_rep), ("raw_a_S", acc_raw), ("w08_closure_on_starts", acc_w8)):
        res = acc.result()
        maps[name] = res.pop("map")
        out["data_implied"][name] = res
    out["data_implied"]["definition"] = (
        "a_S = (stored truth at +1 h - coarse model launched from the STORED start at +1 h) / 3600 s on wet "
        "faces, twenty step-7 starts (10 training, 10 validation); representable_S = least-squares projection "
        "onto the range of the rotated-gradient operator at the rebalanced launch state (residual_decomposition); "
        "w08 closure evaluated at the rebalanced start")
    out["controls"]["w08_vs_step6_per_start_rel_max"] = rel6
    out["controls"]["closure_in_range_remainder_rel_max"] = in_range
    out["controls"]["projection_normal_residual_max"] = proj_rel
    out["controls"]["starts"] = [l for l, *_ in starts]

    # Figures.
    j0, j1, i0, i1 = bounds
    km = geometry.spacing_m / 1e3
    extent = [0, (i1 - i0 + 1) * km, 0, (j1 - j0 + 1) * km]

    def crop(m):
        return np.where(tmask, m, np.nan)[j0:j1 + 1, i0:i1 + 1]

    def panels(fig, axes, names, titles, vmax):
        for ax, name, title in zip(axes, names, titles):
            im = ax.imshow(crop(maps[name]), origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x [km]")
            ax.set_ylabel("y [km]")
        fig.colorbar(im, ax=axes, shrink=0.8, label="closure work on the flow [m$^2$ s$^{-3}$ per unit mass]")

    names = [f"w{w:02d}" for w in WINDOWS]
    vmax = float(np.nanpercentile(np.abs(np.stack([crop(maps[n]) for n in names])), 99.5))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    panels(fig, axes, names, [
        f"w = {w} d: basin mean {out['per_window'][f'w{w:02d}']['basin_mean_face_convention']:+.2e} m$^2$ s$^{{-3}}$"
        for w in WINDOWS], vmax)
    fig.suptitle(f"Time-mean local closure work, free runs from rest, days {days[0]}-{days[-1]} every {STRIDE_DAYS} d "
                 f"({len(days)} snapshots); colour scale +-{vmax:.1e}", fontsize=10)
    fig1 = eb.DIAG / "closure_work_maps_by_window.png"
    fig.savefig(fig1, dpi=130)
    plt.close(fig)

    vmax2 = float(np.nanpercentile(np.abs(np.stack([crop(maps["representable_S"]), crop(maps["w08_closure_on_starts"])])), 99.5))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    panels(fig, axes, ["representable_S", "w08_closure_on_starts"], [
        f"data-implied correction (representable, matched start): basin mean "
        f"{out['data_implied']['representable_S']['basin_mean_face_convention']:+.2e}",
        f"trained w = 8 d closure on the same 20 starts: basin mean "
        f"{out['data_implied']['w08_closure_on_starts']['basin_mean_face_convention']:+.2e}"], vmax2)
    fig.suptitle(f"Local work per unit mass over the twenty step-7 starts; colour scale +-{vmax2:.1e}", fontsize=10)
    fig2 = eb.DIAG / "data_implied_vs_w08_maps.png"
    fig.savefig(fig2, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for w in WINDOWS:
        r = out["per_window"][f"w{w:02d}"]
        ax.plot(r["k_cycles_per_basin_width"], r["spectrum_m2_s3"], label=f"closure, w = {w} d (free run)")
    r = out["data_implied"]["representable_S"]
    ax.plot(r["k_cycles_per_basin_width"], r["spectrum_m2_s3"], "k--", label="data-implied correction (representable, 20 starts)")
    r = out["data_implied"]["w08_closure_on_starts"]
    ax.plot(r["k_cycles_per_basin_width"], r["spectrum_m2_s3"], "k:", label="closure w = 8 d on the same 20 starts")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("isotropic wavenumber k [cycles per basin width L = %.0f km]" % (extent[1]))
    ax.set_ylabel("energy input per shell [m$^2$ s$^{-3}$]")
    ax.set_title("Isotropic spectrum of the closure's energy input, Re[conj(u_hat) a_hat], T points, even reflection", fontsize=9)
    sec = ax.secondary_xaxis("top", functions=(lambda k: extent[1] / np.maximum(k, 1e-9), lambda lam: extent[1] / np.maximum(lam, 1e-9)))
    sec.set_xlabel("wavelength [km]")
    ax.legend(fontsize=8)
    fig3 = eb.DIAG / "closure_work_spectra.png"
    fig.savefig(fig3, dpi=130)
    plt.close(fig)

    np.savez(eb.DIAG / "closure_work_maps.npz", tmask=tmask, **maps)
    (eb.DIAG / "closure_work_maps.json").write_text(json.dumps(out, indent=1))
    print("controls:", json.dumps({k: v for k, v in out["controls"].items() if k != "starts"}, indent=1))
    print("WROTE", fig1, fig2, fig3)


if __name__ == "__main__":
    main()
