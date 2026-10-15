"""Round-10 step 8: split the one-hour training residual into its parts.

Step 7 (implied_subgrid_work.py) measured the one-hour correction the
training contract demands: the coarse model launched from the REBALANCED
start (sea level rebalanced, velocities stored) against the stored truth one
hour later.  That residual mixes (i) the sea-level rebalancing, (ii) what the
closure's operator cannot represent, and (iii) the rest.  On the same twenty
starts this script separates them:

1. Launch the coarse model also from the STORED (un-rebalanced) start and
   form the same residual against the same truth endpoint,
   a_S = (truth_1h - model_stored_1h) / 3600 s.  The rebalanced-launch
   residual a_R is recomputed with step 7's own code path and must reproduce
   step 7 bit for bit.  a_R - a_S = (model_stored_1h - model_rebalanced_1h)
   / 3600 s is the rebalancing contribution (it does not involve the truth).
2. Project a_R and a_S onto the range of the closure's exact discrete
   operator at the launch state: boundary-zero F potential Phi -> clockwise
   rotated gradient over the live face cross-sections
   (swe_rotated_potential_forcing, the map the RK3 stage applies to the
   network's output).  Least squares in the wet-face L2 inner product (the
   work's inner product: mean over wet U faces plus mean over wet V faces;
   with equal U and V face counts this is the plain face L2).  The operator
   is assembled as a sparse matrix (two entries per face) and the normal
   equations A^T A Phi = A^T a are solved exactly by sparse LU
   (scipy splu); the reported residual is |A^T (a - A Phi)| / |A^T a|.

Work per unit mass of every part with the start velocity, split into the
first interior wall ring and the interior exactly as step 7.

Controls, each asserted:
  (a) the rebalanced-launch ring/interior/total work equals step 7's stored
      value for every start (==);
  (b) the sparse operator applied to the trained w=2 network's own potential
      reproduces the stage-differenced closure acceleration (relative
      max-abs difference quoted), and projecting that acceleration returns
      it with an orthogonal remainder of relative norm ~1e-15 (quoted);
  (c) a field of the form cross * grad(p), p any wet-T scalar, is exactly
      orthogonal to the range (A^T a = 0 analytically): its representable
      part has relative norm ~1e-15 (quoted);
  (d) the wind acceleration (stage-3 wind on minus off): stated beforehand,
      its zonal stress depends on y only and tau_y = 0, so its transport
      divergence is nonzero only in the westernmost and easternmost wet T
      columns; it is therefore NOT in the range (|A^T a_wind| > 0), the
      representable part has zero basin-summed U and V transport
      (telescoping), and the orthogonal remainder carries the full
      transport divergence of the wind: the remainder's divergence equals
      the wind's at every T cell and vanishes off the two wall columns.
Writes round10_diagnosis/step8_residual_decomposition.json.
Usage: residual_decomposition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent))
import energy_budget as eb  # noqa: E402
import implied_subgrid_work as isw  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d  # noqa: E402
from legoesm.ocean.dynamics.swe_rk3 import (  # noqa: E402
    _entry_from_state,
    swe_rk3_stage,
    swe_rotated_potential_forcing,
)
from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands  # noqa: E402
from legoesm.training.swg_online import (  # noqa: E402
    SWGDatasetContract,
    coarse_state_from_native,
    stored_state_at,
    swg_online_step,
)

OUT = eb.DIAG / "step8_residual_decomposition.json"
STEP7 = eb.DIAG / "step7_implied_subgrid_work.json"
HOUR_S = isw.HOUR_S


def rotated_gradient_matrix(cross_u, cross_v, umask, vmask, vertex_interior):
    """Sparse A: boundary-zero F potential (interior F points, row-major
    over ``vertex_interior``) -> stacked wet-face acceleration
    [a_u over umask, a_v over vmask], the map of swe_rotated_potential_forcing:
    a_u(j,i) = (Phi(j,i) - Phi(j-1,i)) / cross_u(j,i),
    a_v(j,i) = -(Phi(j,i) - Phi(j,i-1)) / cross_v(j,i).
    Returns (A, phi_index) with phi_index[j,i] the column of Phi(j,i) or -1."""
    umask = np.asarray(umask, dtype=bool)
    vmask = np.asarray(vmask, dtype=bool)
    vtx = np.asarray(vertex_interior, dtype=bool)
    cross_u = np.asarray(cross_u, dtype=np.float64)
    cross_v = np.asarray(cross_v, dtype=np.float64)
    phi_index = -np.ones(vtx.shape, dtype=np.int64)
    phi_index[vtx] = np.arange(int(vtx.sum()))
    n_u = int(umask.sum())
    rows, cols, vals = [], [], []
    ju, iu = np.nonzero(umask)
    for face, (j, i) in enumerate(zip(ju, iu)):
        if phi_index[j, i] >= 0:
            rows.append(face)
            cols.append(phi_index[j, i])
            vals.append(1.0 / cross_u[j, i])
        if j >= 1 and phi_index[j - 1, i] >= 0:
            rows.append(face)
            cols.append(phi_index[j - 1, i])
            vals.append(-1.0 / cross_u[j, i])
    jv, iv = np.nonzero(vmask)
    for face, (j, i) in enumerate(zip(jv, iv)):
        if phi_index[j, i] >= 0:
            rows.append(n_u + face)
            cols.append(phi_index[j, i])
            vals.append(-1.0 / cross_v[j, i])
        if i >= 1 and phi_index[j, i - 1] >= 0:
            rows.append(n_u + face)
            cols.append(phi_index[j, i - 1])
            vals.append(1.0 / cross_v[j, i])
    A = sp.csc_matrix((vals, (rows, cols)), shape=(n_u + int(vmask.sum()), int(vtx.sum())))
    return A, phi_index


class RangeProjector:
    """Exact least-squares projection onto range(A) in the plain L2 of the
    stacked wet faces: Phi = (A^T A)^-1 A^T a by sparse LU, once per A."""

    def __init__(self, A):
        self.A = A
        self.lu = spla.splu((A.T @ A).tocsc())

    def __call__(self, a):
        """(representable part, orthogonal remainder, Phi, normal-equation
        residual |A^T r| / |A^T a|)."""
        a = np.asarray(a, dtype=np.float64)
        rhs = self.A.T @ a
        phi = self.lu.solve(rhs)
        rep = self.A @ phi
        rem = a - rep
        norm_rhs = float(np.linalg.norm(rhs))
        res = float(np.linalg.norm(self.A.T @ rem)) / norm_rhs if norm_rhs > 0 else 0.0
        return rep, rem, phi, res


def main():
    driver, config, card, geometry, nets, forcing, key = eb.setup()
    ring_u, ring_v, umask, vmask = isw.face_rings(geometry)
    n_u, n_v = int(umask.sum()), int(vmask.sum())
    assert n_u == n_v, "unequal face counts: the work inner product is not the plain face L2"
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

    # Step 7's stepper, verbatim (swg_online_step, no network, 9-step scan).
    def step_a(state, _):
        return swg_online_step(None, state, None, None, None, card=card,
                               geometry=geometry, forcing=forcing), None

    @jax.jit
    def hour_a(state):
        state, _ = jax.lax.scan(step_a, state, None, length=steps_per_hour)
        return state

    net = nets[2]

    @jax.jit
    def fields(ssh, u, v):
        """At a state: closure (w02_e05) potential and acceleration, wind
        acceleration (stage-3 on/off as step 7), live face thicknesses."""
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
        one = jnp.asarray(1.0, dtype=jnp.float64)
        e3u = (ops.e3u_0 * (one + entry.r3u_k[..., None] * ops.umask3))[..., 0]
        e3v = (ops.e3v_0 * (one + entry.r3v_k[..., None] * ops.vmask3))[..., 0]
        exact = swe_rotated_potential_forcing(
            potential, ops.e2u, ops.e1v, e3u, e3v, umask3[:, 1:, 0], vmask3[1:, :, 0], vtx)
        return (potential, (clo[0] - base[0]) / dt, (clo[1] - base[1]) / dt,
                (base[0] - nw[0]) / dt, (base[1] - nw[1]) / dt, e3u, e3v, exact[0], exact[1])

    def native(state):
        return (np.asarray(state.eta.data), np.asarray(state.u.data[:, 1:, 0]),
                np.asarray(state.v.data[1:, :, 0]))

    def work(u, v, au, av, sel_u=umask, sel_v=vmask):
        return float(np.sum(u * au * sel_u) / n_u + np.sum(v * av * sel_v) / n_v)

    def work3(u, v, au, av):
        return {"total": work(u, v, au, av),
                "ring": work(u, v, au, av, ring_u, ring_v),
                "interior": work(u, v, au, av, umask & ~ring_u, vmask & ~ring_v)}

    def implied(truth_1h, model_1h):
        return ((truth_1h[1] - model_1h[1]) / HOUR_S * umask,
                (truth_1h[2] - model_1h[2]) / HOUR_S * vmask)

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

    def norm(vec):
        return float(np.linalg.norm(vec))

    def divergence(cross_u, cross_v, au, av):
        """Transport divergence at T cells: east+north faces of T(j,i) minus
        its west+south faces."""
        tu = cross_u * au * umask
        tv = cross_v * av * vmask
        west = np.concatenate([np.zeros_like(tu[:, :1]), tu[:, :-1]], axis=1)
        south = np.concatenate([np.zeros_like(tv[:1]), tv[:-1]], axis=0)
        return tu - west + tv - south

    contract = SWGDatasetContract.from_json(eb.RUN / "dataset_contract.json", card=card)
    step7 = {r["start"]: r for r in json.loads(STEP7.read_text())["per_start"]}
    rows = isw.starts(card)
    j0, j1, i0, i1 = geometry.wet_bounds
    wet = np.asarray(geometry.tmask, dtype=bool)
    rng = np.random.default_rng(eb.SEED)
    out_rows, controls = [], {
        "step7_bitwise": [], "stored_uv_equal_rebalanced_uv": [],
        "sparse_vs_swe_rotated_potential_forcing_rel": [],
        "sparse_vs_stage_differenced_closure_rel": [],
        "closure_in_range_remainder_rel": [], "closure_projection_normal_residual": [],
        "cross_grad_p_representable_rel": [],
        "wind_normal_rhs_over_norm": [], "wind_representable_transport_sum_u_v": [],
        "wind_divergence_off_wall_columns_rel": [], "wind_minus_remainder_divergence_rel": [],
        "normal_equation_residual_R": [], "normal_equation_residual_S": []}
    parts = ("rebalanced_launch", "stored_launch", "rebalancing_part",
             "representable_R", "orthogonal_R", "representable_S", "orthogonal_S")
    for label, hour, start, truth_1h, _truth_24h in rows:
        ssh_r, u, v = (np.asarray(a, dtype=np.float64) for a in start)
        ssh_s, u_s, v_s = (np.asarray(a, dtype=np.float64)
                           for a in stored_state_at(contract.sources, contract.objective, hour)[0])
        assert np.array_equal(u, u_s) and np.array_equal(v, v_s), label
        controls["stored_uv_equal_rebalanced_uv"].append(True)
        model_r = native(hour_a(seeded(ssh_r, u, v)))
        model_s = native(hour_a(seeded(ssh_s, u, v)))
        aR = implied(truth_1h, model_r)
        aS = implied(truth_1h, model_s)
        wR = work3(u, v, *aR)
        ref = step7[label]
        bitwise = (wR["ring"] == ref["a_sg_work_ring"] and wR["interior"] == ref["a_sg_work_interior"]
                   and wR["total"] == ref["a_sg_work"])
        assert bitwise, (label, wR, ref["a_sg_work_ring"], ref["a_sg_work_interior"])
        controls["step7_bitwise"].append(bitwise)

        # Operator at the rebalanced launch state (the state the training's
        # closure acts on); the same operator projects a_S.
        (potential, c_u, c_v, w_u, w_v, e3u, e3v, x_u, x_v) = (
            np.asarray(x) for x in fields(jnp.asarray(ssh_r), jnp.asarray(u), jnp.asarray(v)))
        cross_u, cross_v = e2u * e3u, e1v * e3v
        A, phi_index = rotated_gradient_matrix(cross_u, cross_v, umask, vmask, vtx)
        scale = float(np.median(cross_u[umask]))   # column scaling only; the projection is invariant
        proj = RangeProjector(A * scale)
        phi_vec = potential[vtx]
        exact_vec = pack(x_u, x_v)
        c_vec = pack(c_u, c_v)
        rel_fn = norm(A @ phi_vec - exact_vec) / norm(exact_vec)
        rel_stage = norm(A @ phi_vec - c_vec) / norm(c_vec)
        controls["sparse_vs_swe_rotated_potential_forcing_rel"].append(rel_fn)
        controls["sparse_vs_stage_differenced_closure_rel"].append(rel_stage)
        assert rel_fn <= 1e-13 and rel_stage <= 1e-10, (label, rel_fn, rel_stage)
        # Control (b): the closure's own acceleration is in range.
        rep, rem, _, res = proj(c_vec)
        controls["closure_in_range_remainder_rel"].append(norm(rem) / norm(c_vec))
        controls["closure_projection_normal_residual"].append(res)
        assert norm(rem) / norm(c_vec) <= 1e-12, label
        # Control (c): cross * grad p is orthogonal to the range.
        p = np.where(wet, rng.standard_normal(wet.shape), 0.0)
        g_u = cross_u * (np.roll(p, -1, axis=1) - p) * umask
        g_v = cross_v * (np.roll(p, -1, axis=0) - p) * vmask
        g_vec = pack(g_u, g_v)
        rep_g, _, _, _ = proj(g_vec)
        controls["cross_grad_p_representable_rel"].append(norm(rep_g) / norm(g_vec))
        assert norm(rep_g) / norm(g_vec) <= 1e-12, label
        # Control (d): the wind.
        wind_vec = pack(w_u, w_v)
        rep_w, rem_w, _, _ = proj(wind_vec)
        controls["wind_normal_rhs_over_norm"].append(norm(A.T @ wind_vec) * scale / norm(wind_vec))
        rw_u, rw_v = unpack(rep_w)
        controls["wind_representable_transport_sum_u_v"].append(
            [float(np.sum(cross_u * rw_u * umask)), float(np.sum(cross_v * rw_v * vmask))])
        div_w = divergence(cross_u, cross_v, w_u, w_v)
        div_rem = divergence(cross_u, cross_v, *unpack(rem_w))
        off = np.ones_like(wet)
        off[:, [i0, i1]] = False
        controls["wind_divergence_off_wall_columns_rel"].append(
            float(np.abs(div_w * off).max() / np.abs(div_w).max()))
        controls["wind_minus_remainder_divergence_rel"].append(
            float(np.abs(div_w - div_rem).max() / np.abs(div_w).max()))

        repR, remR, _, resR = proj(pack(*aR))
        repS, remS, _, resS = proj(pack(*aS))
        controls["normal_equation_residual_R"].append(resR)
        controls["normal_equation_residual_S"].append(resS)
        fields_by_part = {
            "rebalanced_launch": aR, "stored_launch": aS,
            "rebalancing_part": (aR[0] - aS[0], aR[1] - aS[1]),
            "representable_R": unpack(repR), "orthogonal_R": unpack(remR),
            "representable_S": unpack(repS), "orthogonal_S": unpack(remS)}
        row = {"start": label, "absolute_hour": hour,
               "work": {k: work3(u, v, *f) for k, f in fields_by_part.items()},
               "rms": {k: isw_rms(f, umask, vmask, n_u, n_v) for k, f in fields_by_part.items()},
               "cos_closure_w02_vs_representable_S": float(
                   np.dot(c_vec, repS) / (norm(c_vec) * norm(repS)))}
        out_rows.append(row)
        print(f"{label:28s} " + " ".join(
            f"{k[:6]} {row['work'][k]['ring']:+.2e}/{row['work'][k]['interior']:+.2e}" for k in parts))
        sys.stdout.flush()

    def mean(part, key):
        return float(np.mean([r["work"][part][key] for r in out_rows]))

    summary = {part: {key: mean(part, key) for key in ("total", "ring", "interior")} for part in parts}
    summary["sink_fraction_of_starts"] = {
        part: float(np.mean([r["work"][part]["total"] < 0 for r in out_rows])) for part in parts}
    summary["ring_sink_surviving_as_representable_S_fraction"] = (
        summary["representable_S"]["ring"] / summary["rebalanced_launch"]["ring"])
    out = {"definition": __doc__,
           "units": {"work": "m2 s-3 (per unit mass, mean over wet faces)", "accel": "m s-2"},
           "solver": "sparse LU (scipy.sparse.linalg.splu) on the normal equations A^T A Phi = A^T a; "
                     "A assembled from swe_rotated_potential_forcing's stencil with the live face "
                     "cross-sections e2u*e3u, e1v*e3v at the rebalanced launch state; "
                     "inner product = plain L2 over the stacked wet faces (n_u == n_v)",
           "unknowns": int(vtx.sum()), "faces": n_u + n_v,
           "ring_face_counts": {"u": int(ring_u.sum()), "v": int(ring_v.sum())},
           "controls": controls, "per_start": out_rows, "summary": summary}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    print("controls (max over starts):", json.dumps({
        k: (max(v) if not isinstance(v[0], (list, bool)) else v[0] if isinstance(v[0], bool) and all(v) else
            [max(abs(x[0]) for x in v), max(abs(x[1]) for x in v)]) for k, v in controls.items()}, indent=1))
    print("WROTE", OUT)


def isw_rms(field, umask, vmask, n_u, n_v):
    au, av = field
    return float(np.sqrt(0.5 * (np.sum(au**2 * umask) / n_u + np.sum(av**2 * vmask) / n_v)))


if __name__ == "__main__":
    main()
