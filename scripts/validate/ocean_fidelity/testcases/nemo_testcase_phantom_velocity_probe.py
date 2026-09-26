#!/usr/bin/env python3
"""OVERFLOW-zps below-seabed ("phantom") velocity: census + UP3 leverage scaling.

NEMO's rule, read from the source before any hypothesis (Rule 0):

* ``stprk3_stg.F90:367-368`` (``ln_dynadv_vec .OR. lk_linssh``) and
  ``:375-380`` (the compiled ``key_qco`` branch) multiply EVERY WS-RK3 stage
  velocity by ``umask(ji,jj,jk)`` -- the 3-D mask;
* ``stprk3_stg.F90:443-446`` adds the barotropic correction as
  ``uu(...,Kaa) = uu(...,Kaa) + zub(ji,jj)*umask(ji,jj,jk)``, again 3-D;
* so ``uu`` is EXACTLY ZERO below the seabed, and ``dynadv_up3.F90:142-143``
  (``zlu_uu``), ``:160`` (``zFu``) and ``:166-176`` (``zFu_t``) read those
  zeros when the k-slab stencil reaches a dry face.

legoESM's WS-RK3 stage program masks with the 2-D ``state.u_mask`` broadcast
over levels (``ocean_model_latlon_cgrid.py`` ``u_mask_3d`` at the module's
step, consumed by ``_replace_stage_mean``), so a face that is wet at ANY level
keeps a nonzero velocity at every level BELOW its seabed -- and
``_bc_horizontal_momentum_advection_flux_form`` feeds that value, unmasked,
into ``_up3_reconstruct`` as a stencil neighbour of the wet bottom level of
the deeper shelf-break faces.

Two commands, both fp64 (``PrecisionPolicy.fp64()`` set BEFORE the card is
built + ``JAX_ENABLE_X64=1``; dtypes printed), CPU:

``census``
    Where the below-seabed velocity is, how big, how it grows, whether it is
    depth-CONSTANT (the fingerprint of a depth-mean broadcast over a masked
    column), and which stage/hook arm carries it.

``scaling``
    The phantom's leverage on the horizontal UP3 momentum advection, measured
    by calling legoESM's OWN ``tendencies()`` twice on the SAME stage-2 state
    with ONE variable -- the below-seabed velocity -- and, separately, the
    size of NEMO's ``* umask`` factor on the UP3 curvature
    (``dynadv_up3.F90:142-143``), replayed on the already-masked state.
    Compared against the measured one-step injections.

The tool prints no verdict; interpretation belongs in the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

CASE = "OVERFLOW-zps"
LOCK = "LOCK_EXCHANGE-zco"
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10")
DEFAULT_OUT = Path("/data/abyssal/dbalwada/nemo-testcases-l1/phantom_velocity")
FRONT_FACES = tuple(range(16, 30))


class ProbeError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProbeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(name: str):
    script = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_fp64():
    """Rule 1c: the policy must be set BEFORE the card is built (constructors
    cast to ``get_policy().control``, which defaults to float32)."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")


def git_sha(allow_dirty: bool) -> str:
    from legoesm.ocean.fidelity.provenance import git_sha as _stamp
    return _stamp(allow_dirty=allow_dirty)


def build(case: str, hooks=None):
    """Card + model + the 3-D LIVE u-face mask and the 2-D broadcast."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    card = build_nemo_testcase_card(case)
    z_coord = card.recipe.z_coord
    live_u = np.asarray(
        compute_face_masks_3d(z_coord.is_active, card.recipe.grid)[0]).astype(float)
    flat_u = np.broadcast_to(
        np.asarray(card.recipe.initial_state.u_mask.data)[..., None], live_u.shape)
    model = LatLonCGridOceanModel(
        card.recipe.grid, z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks if hooks is not None else _NEMOWSRK3TestHooks())
    return card, model, live_u, np.asarray(flat_u, dtype=float)


def dtype_report(card, state) -> dict:
    z = card.recipe.z_coord
    report = {
        "T": str(np.asarray(state.T.data).dtype), "u": str(np.asarray(state.u.data).dtype),
        "eta": str(np.asarray(state.eta.data).dtype),
        "h_partial": str(np.asarray(z.h_partial).dtype),
        "dz_ref": str(np.asarray(z.dz_ref).dtype),
    }
    require(set(report.values()) == {"float64"}, f"non-fp64 arrays: {report}")
    return report


def dry_field(u, live_u):
    """The below-live-seabed part of a u field (zero on every wet face)."""
    return np.where(live_u > 0.0, 0.0, np.asarray(u))


def dry_summary(u, live_u) -> dict:
    dry = dry_field(u, live_u)
    a = np.abs(dry)
    idx = np.unravel_index(int(a.argmax()), a.shape)
    row, face = int(idx[0]), int(idx[1])
    col = dry[row, face]
    below = col[np.asarray(live_u)[row, face] == 0.0]
    return {
        "max_abs": float(a.max()),
        "argmax": [int(v) for v in idx],
        "n_nonzero": int((a > 0.0).sum()),
        "argmax_column_below_seabed_unique_values": int(np.unique(below).size),
        "argmax_column_below_seabed_ptp": float(np.ptp(below)) if below.size else 0.0,
        "argmax_column_below_seabed_value": float(below[0]) if below.size else 0.0,
        "wet_max_abs": float(np.abs(np.where(live_u > 0.0, np.asarray(u), 0.0)).max()),
    }


def face_profile(u, live_u, faces=FRONT_FACES) -> dict:
    dry = dry_field(u, live_u)
    out = {}
    for f in faces:
        col = dry[1, f]
        if not np.any(col):
            continue
        k = int(np.abs(col).argmax())
        out[int(f)] = {"k_of_max": k, "value": float(col[k]),
                       "n_below_seabed": int((np.asarray(live_u)[1, f] == 0.0).sum())}
    return out


def _transport_operand_below_seabed(card, live_u) -> dict:
    """Below-live-seabed ``max abs`` of the transport velocity each WS-RK3
    stage hands to the horizontal UP3 momentum advection.

    The end-of-step prognostic state is not this array: the stage program
    builds a separate barotropically-reconciled transport and passes it as
    ``momentum_flux_transport_velocity``.  NEMO masks the same correction
    there (``stprk3_stg.F90:273-274``), so it must be zero below the seabed
    too.  Measured by intercepting the model's own ``tendencies`` calls with
    tracing disabled, so the arrays are concrete; the model is otherwise
    untouched."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)

    live_j = jnp.asarray(live_u)
    original = LatLonCGridOceanModel.tendencies
    out = {}
    try:
        for label, legacy in (("faithful", False), ("legacy_2d_mask", True)):
            seen = []

            def spy(self, *args, _seen=seen, **kwargs):
                pair = kwargs.get("momentum_flux_transport_velocity")
                if pair is not None:
                    _seen.append(float(jnp.max(jnp.abs(
                        jnp.where(live_j == 0.0, pair[0], 0.0)))))
                return original(self, *args, **kwargs)

            LatLonCGridOceanModel.tendencies = spy
            model = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
                _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                    legacy_2d_stage_face_mask=legacy))
            with jax.disable_jit():
                model.step(card.recipe.initial_state, dt=card.dt_s)
            require(len(seen) > 0,
                    "no stage handed a transport velocity to tendencies(); the "
                    "probe is inspecting nothing")
            out[label] = seen
    finally:
        LatLonCGridOceanModel.tendencies = original
    require(max(out["legacy_2d_mask"]) > 1.0e-3,
            "the 2-D arm does not reproduce the pre-fix transport operand: "
            f"{out['legacy_2d_mask']}")
    return out


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------
def census(*, max_kt: int, out_dir: Path, allow_dirty: bool) -> dict:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    TRAJ = _load("nemo_testcase_phase3_trajectory_gate")
    BARO = _load("nemo_testcase_overflow_barotropic_gate")
    legoesm_git_sha = git_sha(allow_dirty)
    set_fp64()
    card, model, live_u, flat_u = build(CASE)
    dtypes = dtype_report(card, card.recipe.initial_state)
    masks = TRAJ.expected_masks(card)
    artifacts = {}

    report = {
        "format": "nemo-testcase-l1-phantom-census-v1", "case": CASE,
        "legoesm_git_sha": legoesm_git_sha, "backend": jax.default_backend(),
        "dtypes": dtypes, "dt_s": card.dt_s, "artifacts": artifacts,
        "mask_geometry": {
            "n_points_where_live_3d_differs_from_2d_broadcast":
                int((live_u != flat_u).sum()),
            "u_face_wet_level_counts_row1_faces_16_30":
                [int(v) for v in live_u[1, 16:30].sum(axis=-1)],
        },
        "free_run": {}, "exact_entry_one_step": {}, "arms": {},
    }

    # --- free run: the phantom the certified card actually carries ---
    state = card.recipe.initial_state
    report["free_run"][1] = {"dry": dry_summary(state.u.data, live_u),
                             "faces": face_profile(state.u.data, live_u)}
    require(report["free_run"][1]["dry"]["max_abs"] == 0.0,
            "the card's initial state already carries a below-seabed velocity")
    for kt in range(2, max_kt + 1):
        state = model.step(state, dt=card.dt_s)
        report["free_run"][kt] = {"dry": dry_summary(state.u.data, live_u),
                                  "faces": face_profile(state.u.data, live_u)}
        print(f"[free] kt={kt} |u below seabed| max "
              f"{report['free_run'][kt]['dry']['max_abs']:.6e} at face "
              f"{report['free_run'][kt]['dry']['argmax'][1]} k "
              f"{report['free_run'][kt]['dry']['argmax'][2]}; wet max "
              f"{report['free_run'][kt]['dry']['wet_max_abs']:.6e}", flush=True)

    # --- one step from NEMO's EXACT entry: the phantom is regenerated, not
    #     inherited (NEMO's own entries are umask'd, so the entry is clean) ---
    for kt in range(2, min(max_kt, 5) + 1):
        path = ROOT / f"oracle_step_entry_kt{kt:08d}.bin"
        entry = TRAJ.read_entry(path, CASE)
        artifacts[path.name] = sha256(path)
        st0 = BARO.state_from_oracle_entry(card.recipe.initial_state, entry, masks)
        clean = dry_summary(st0.u.data, live_u)
        require(clean["max_abs"] == 0.0,
                f"NEMO's kt={kt} entry is not zero below the seabed: {clean}")
        st1 = model.step(st0, dt=card.dt_s)
        report["exact_entry_one_step"][kt] = {
            "entry_dry_max_abs": clean["max_abs"],
            "after_one_step": dry_summary(st1.u.data, live_u),
        }
        print(f"[exact@{kt}] one step -> |u below seabed| max "
              f"{report['exact_entry_one_step'][kt]['after_one_step']['max_abs']:.6e}",
              flush=True)

    # --- origin arms, kt=1 (from rest: u0 == 0, so everything below the
    #     seabed at the end of the step was WRITTEN during the step) ---
    for label, hooks in (
        ("stage1_velocity", _NEMOWSRK3TestHooks(expose_momentum_stage=1)),
        ("stage2_velocity", _NEMOWSRK3TestHooks(expose_momentum_stage=2)),
        ("no_stage_barotropic_replacement",
         _NEMOWSRK3TestHooks(stage_barotropic_correction=False)),
    ):
        _, m, _, _ = build(CASE, hooks)
        s = m.step(card.recipe.initial_state, dt=card.dt_s)
        report["arms"][label] = {"kt1": dry_summary(s.u.data, live_u)}
        print(f"[arm {label}] kt=1 |u below seabed| max "
              f"{report['arms'][label]['kt1']['max_abs']:.6e}", flush=True)

    # --- the OPERAND, not just the state: the array actually handed to
    #     dyn_adv_up3 as its transport.  NEMO masks the barotropic correction
    #     inside it too (stprk3_stg.F90:273-274,
    #     ``zFu = e2u*e3u(Kmm)*( uu(Kmm) + zub*umask(ji,jj,jk) )``), and the
    #     end-of-step STATE can be clean while this array is not -- which is
    #     exactly what an independent review caught. ---
    report["transport_operand"] = _transport_operand_below_seabed(card, live_u)
    print(f"[transport operand] below-seabed |transport_u| per stage: "
          f"{report['transport_operand']['faithful']} (faithful) vs "
          f"{report['transport_operand']['legacy_2d_mask']} (2-D arm)", flush=True)
    require(max(report["transport_operand"]["faithful"]) == 0.0,
            "the transport velocity handed to dyn_adv_up3 is nonzero below "
            f"the seabed: {report['transport_operand']['faithful']}")

    # --- control: LOCK's 3-D live mask IS the 2-D broadcast, so no card
    #     without a staircase can carry a phantom at all ---
    lock_card, lock_model, lock_live, lock_flat = build(LOCK)
    lock_state = lock_model.step(lock_card.recipe.initial_state, dt=lock_card.dt_s)
    report["control_lock_exchange"] = {
        "live_3d_equals_2d_broadcast": bool(np.array_equal(lock_live, lock_flat)),
        "kt2_dry_max_abs": float(np.abs(dry_field(lock_state.u.data, lock_live)).max()),
    }
    require(report["control_lock_exchange"]["live_3d_equals_2d_broadcast"],
            "LOCK's 3-D live u mask is not the 2-D broadcast")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "census.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------
def scaling(*, kts, out_dir: Path, allow_dirty: bool) -> dict:
    """Leverage of the phantom (and of NEMO's curvature umask) on du/dt.

    Both arms call legoESM's OWN ``tendencies()`` on the SAME stage-2 state,
    with the SAME geometry, forcing and config -- one variable each:

      A  u = the stage-2 velocity legoESM actually carries (phantom present)
      B  u = the same with every below-seabed face zeroed  (NEMO's rule)
      C  B, plus ``* umask`` on the T-point UP3 curvature (dynadv_up3.F90:
         142-143), the ONE reconstruction whose NEMO form is a single
         centre mask.

    ``A - B`` is the phantom's leverage; ``C - B`` is the curvature factor's.
    Because the two calls share every other operand, the difference is a
    controlled one-variable measurement even where the replayed geometry is
    the step-entry (Kbb) bundle rather than the stage's own.
    """
    import jax
    import jax.numpy as jnp
    import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    TRAJ = _load("nemo_testcase_phase3_trajectory_gate")
    BARO = _load("nemo_testcase_overflow_barotropic_gate")
    legoesm_git_sha = git_sha(allow_dirty)
    set_fp64()
    card, model, live_u, _ = build(CASE)
    _, stage2_model, _, _ = build(CASE, _NEMOWSRK3TestHooks(expose_momentum_stage=2))
    dtypes = dtype_report(card, card.recipe.initial_state)
    masks = TRAJ.expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    act_u = np.asarray(masks["u"])
    artifacts = {}
    live_j = jnp.asarray(live_u)
    original_reconstruct = pe._up3_reconstruct

    def masked_curvature_reconstruct(far_pos, adv_pos, adv_neg, far_neg, selector):
        """dynadv_up3.F90:142-143 -- ``zlu_uu`` carries ``umask`` at ITS OWN
        u-point.  Algebraically the branch value is
        ``0.5*(adv_pos+adv_neg) - (1/6)*m*curvature``; with ``m = 1`` it is
        exactly the production formula.  Only the T-point (same-direction)
        u stencil has this shape in NEMO -- the F-point ones carry an
        fmask PAIR -- so every other stencil falls through unchanged."""
        m_pos = live_j[:, :-1, :]
        m_neg = live_j[:, 1:, :]
        if adv_pos.shape != m_pos.shape or adv_neg.shape != m_neg.shape:
            return original_reconstruct(far_pos, adv_pos, adv_neg, far_neg, selector)
        centre = 0.5 * (adv_pos + adv_neg)
        curv_pos = adv_neg + far_pos - 2.0 * adv_pos
        curv_neg = far_neg + adv_pos - 2.0 * adv_neg
        return jnp.where(
            selector > 0.0,
            centre - (1.0 / 6.0) * m_pos * curv_pos,
            centre - (1.0 / 6.0) * m_neg * curv_neg)

    def du_dt(state, patched: bool):
        kwargs = dict(dt=card.dt_s, momentum_only=True, up3_upwind_selector="velocity")
        if not patched:
            return np.asarray(model.tendencies(state, None, **kwargs).du_dt.data)
        pe._up3_reconstruct = masked_curvature_reconstruct
        try:
            return np.asarray(model.tendencies(state, None, **kwargs).du_dt.data)
        finally:
            pe._up3_reconstruct = original_reconstruct

    def wet_rows(delta) -> dict:
        wet = np.asarray(delta) * live_u
        a = np.abs(wet)
        idx = np.unravel_index(int(a.argmax()), a.shape)
        faces = {}
        for f in FRONT_FACES:
            col = wet[1, f]
            if not np.any(col):
                continue
            k = int(np.abs(col).argmax())
            faces[int(f)] = {"k_of_max": k, "d_du_dt": float(col[k]),
                             "dt_times_d": float(card.dt_s * col[k])}
        return {"max_abs_d_du_dt": float(a.max()),
                "dt_times_max_abs": float(card.dt_s * a.max()),
                "argmax": [int(v) for v in idx], "faces": faces}

    report = {
        "format": "nemo-testcase-l1-phantom-scaling-v1", "case": CASE,
        "legoesm_git_sha": legoesm_git_sha, "backend": jax.default_backend(),
        "dtypes": dtypes, "dt_s": card.dt_s, "artifacts": artifacts,
        "note": ("model u-face index f == trajectory-gate face index f-1; "
                 "the gate scores state.u[:, 1:, :nlev]"),
        "rows": {},
    }

    for kt in kts:
        if kt == 1:
            entry_state = card.recipe.initial_state
        else:
            path = ROOT / f"oracle_step_entry_kt{kt:08d}.bin"
            entry = TRAJ.read_entry(path, CASE)
            artifacts[path.name] = sha256(path)
            entry_state = BARO.state_from_oracle_entry(
                card.recipe.initial_state, entry, masks)
        stage2 = stage2_model.step(entry_state, dt=card.dt_s)
        u2 = np.asarray(stage2.u.data)
        v2 = np.asarray(stage2.v.data)
        phantom = dry_summary(u2, live_u)
        st_a = entry_state._replace(u=entry_state.u.replace(data=jnp.asarray(u2)),
                                    v=entry_state.v.replace(data=jnp.asarray(v2)))
        st_b = entry_state._replace(u=entry_state.u.replace(data=jnp.asarray(u2 * live_u)),
                                    v=entry_state.v.replace(data=jnp.asarray(v2)))
        # one variable: only u differs between the two states
        require(np.array_equal(np.asarray(st_a.v.data), np.asarray(st_b.v.data))
                and np.array_equal(np.asarray(st_a.T.data), np.asarray(st_b.T.data))
                and np.array_equal(np.asarray(st_a.S.data), np.asarray(st_b.S.data))
                and np.array_equal(np.asarray(st_a.eta.data), np.asarray(st_b.eta.data)),
                f"kt={kt}: the two arms differ in more than u")
        require(np.array_equal(np.asarray(st_a.u.data) * live_u,
                               np.asarray(st_b.u.data) * live_u),
                f"kt={kt}: zeroing the dry faces moved a WET value")
        a = du_dt(st_a, False)
        b = du_dt(st_b, False)
        c = du_dt(st_b, True)
        # instrument control: with no phantom present the arm must be EXACTLY
        # zero, and the patched reconstruction must reduce to production where
        # every stencil point is wet.
        control = du_dt(st_b, False) - b
        require(float(np.abs(control).max()) == 0.0,
                f"kt={kt}: the phantom arm is not reproducible ({np.abs(control).max()})")
        row = {
            "stage2_phantom": phantom,
            "phantom_leverage": wet_rows(a - b),
            "curvature_umask_leverage": wet_rows(c - b),
        }
        report["rows"][kt] = row
        print(f"[scaling] kt={kt}: phantom {phantom['max_abs']:.4e} m/s at face "
              f"{phantom['argmax'][1]} k {phantom['argmax'][2]}; "
              f"dt*|d(du/dt)| phantom {row['phantom_leverage']['dt_times_max_abs']:.4e} "
              f"at face {row['phantom_leverage']['argmax'][1]} k "
              f"{row['phantom_leverage']['argmax'][2]}; curvature-umask "
              f"{row['curvature_umask_leverage']['dt_times_max_abs']:.4e}", flush=True)

    # phantom-free control on NEMO's own entry velocity: leverage must vanish
    path = ROOT / "oracle_step_entry_kt00000003.bin"
    entry = TRAJ.read_entry(path, CASE)
    artifacts[path.name] = sha256(path)
    st = BARO.state_from_oracle_entry(card.recipe.initial_state, entry, masks)
    zero = du_dt(st, False) - du_dt(
        st._replace(u=st.u.replace(data=jnp.asarray(np.asarray(st.u.data) * live_u))),
        False)
    report["control_phantom_free_entry"] = {
        "max_abs_d_du_dt": float(np.abs(zero).max()),
        "n_active_u_points": int(act_u[..., :nlev].sum()),
    }
    require(report["control_phantom_free_entry"]["max_abs_d_du_dt"] == 0.0,
            "the phantom arm is nonzero on a phantom-free state")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaling.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("census", "scaling"))
    parser.add_argument("--max-kt", type=int, default=6)
    parser.add_argument("--kts", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="stamp '<sha>-dirty' instead of refusing a dirty tree")
    args = parser.parse_args(argv)
    if args.command == "census":
        report = census(max_kt=args.max_kt, out_dir=args.out_dir,
                        allow_dirty=args.allow_dirty)
    else:
        report = scaling(kts=tuple(args.kts), out_dir=args.out_dir,
                         allow_dirty=args.allow_dirty)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
