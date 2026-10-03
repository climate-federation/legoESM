#!/usr/bin/env python3
"""Split the VORTEX FLUX card's ``dyn_adv_up3`` into its two written halves.

Round 204 put the whole of the flux card's completed stage-1 error inside the
flux-form momentum advection NEMO calls at ``stprk3_stg.f90:316``:

    advtrend.u  depth-varying 4.540458839058695e-11   (4.585e-06 relative)
    advtrend.v  depth-varying 4.536336143204694e-11   (4.582e-06 relative)

``dyn_adv_up3`` writes that trend from TWO places, each with its own metric
division:

    dynadv_up3.f90:174-215   the horizontal UP3 flux divergence, k-slab
    dynadv_up3.f90:245-360   the vertical advection block
    dynadv_up3.f90:206-207,335-336,357   the shared divisor
                             ``r1_e1e2u / (e3t_1d(jk)*(1+r3u(Kmm)*umask))``

The record carries only the TOTAL (``adv`` minus ``base``), so neither NEMO
half can be read out of it.  What CAN be done without a new record, and is
what this walk does, is (a) publish legoESM's own two halves through a
WRITE-only seam and show their per-level support, (b) use the levels where
the TOTAL difference is exactly zero to EXONERATE whichever half is active
there, and (c) bound the shared metric divisor from round 200's measured
transport agreement.  Every number is read out of the production-jitted step.

Exactness here is bit equality, never AT-BAR.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nemo_testcase_l1_vortex"))

_spec = importlib.util.spec_from_file_location(
    "vortex_round200_flux_stage1",
    HERE / "nemo_testcase_l1_vortex_round200_flux_stage1.py")
_r200 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r200)

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _seed_from_record, _u_full, _v_full, read_bt_frame, read_stage,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    GateError, expected_masks, lego_fields, read_entry, require,
)

CASE = "VORTEX-zco"
DEFAULT_ROOT = _r200.DEFAULT_ROOT
PLANTS = ("hadv.u", "hadv.v", "vadv.u", "vadv.v",
          "zub.u", "zub.v")
# Round 200's measured relative agreement of the stage advective transports
# zFu/zFv (stprk3_stg.f90:276-277).  Those transports carry the SAME face
# thickness the trend divides by, against stage-entry velocities that are
# seeded bit-identical from NEMO's own record, so this number bounds the
# thickness's own relative disagreement.  Quoted, not re-measured.
ZF_RELATIVE_ROUND200 = 1.4230574434837798e-16


def _per_level(values: np.ndarray, active: np.ndarray) -> list[float]:
    """Largest magnitude on the active mask, level by level."""
    out = []
    for k in range(values.shape[-1]):
        sel = active[..., k]
        out.append(float(np.max(np.abs(values[..., k][sel])))
                   if sel.any() else 0.0)
    return out


def _unequal_per_level(delta, active) -> list[int]:
    bad = (delta != 0.0) & active
    return [int(bad[..., k].sum()) for k in range(delta.shape[-1])]


def run(root: Path, *, plant: str | None = None,
        allow_dirty: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "this walk must run on CPU")
    require(plant is None or plant in PLANTS,
            f"unknown plant {plant!r}; expected one of {PLANTS}")

    card = build_nemo_testcase_card(CASE)
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry1 = read_entry(root / "oracle_step_entry_kt00000001.bin", CASE,
                        expect_interior=interior)
    entry2 = read_entry(root / "oracle_step_entry_kt00000002.bin", CASE,
                        expect_interior=interior)
    read_stage(root / "oracle_stage_kt00000001_s1.bin",
               expect_step=1, expect_stage=1)
    groups = _r200.read_flux_stage_terms(root, 1)
    frame = read_bt_frame(root / "oracle_bt_frames_kt00000001.bin",
                          expect_step=1)
    external = (
        jnp.asarray(entry2["ssh"]),
        jnp.asarray(_u_full(frame["uu_b"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vv_b"][..., None])[..., 0]),
        jnp.asarray(_u_full(frame["un_adv"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vn_adv"][..., None])[..., 0]),
    )
    seed = _seed_from_record(card.recipe.initial_state, entry1, nlev)

    def model_step(hooks):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return model.step(seed, dt=card.dt_s)

    def _owned(values, face):
        values = np.asarray(values)
        return values[:, 1:, :] if face == "u" else values[1:, :, :]

    # THE SEAM CONTROL (round 200's): an exposure that went inert would hand
    # the walk the ORDINARY step output and every row would still score.
    plain = lego_fields(model_step(_NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external)))

    def _frame(arm, transport_operand=""):
        fields = lego_fields(model_step(_NEMOWSRK3TestHooks(
            stage_barotropic_output_override=external,
            momentum_transport_stage1_operand=transport_operand,
            expose_stage1_momentum_rhs_split=arm)))
        out = {}
        for face in ("u", "v"):
            values = np.asarray(fields[face])[..., :nlev]
            _r200.require_live(arm, face, values, plain[face])
            out[face] = values
        return out

    completed = _frame("completed")
    pre_adv = _frame("pre_advection")
    hadv = _frame("advection_horizontal")
    vadv = _frame("advection_vertical")
    zub = _frame("advection_zub_increment")
    # THE CAUSAL ARM, one variable: the stage-1 advective transport
    # subtracts NEMO's separately prognostic uu_b(Kmm)
    # (``stprk3_stg.f90:270``) instead of a depth mean re-reduced from the
    # three-dimensional velocity.  Nothing else changes.
    causal = {}
    for arm in ("prognostic_mean", "qco_depth"):
        causal[arm] = {
            face: (_frame("completed", arm)[face]
                   - _frame("pre_advection", arm)[face])
            for face in ("u", "v")}
    # The record's own stage-entry velocity, level by level: the claim that
    # NEMO's HORIZONTAL half vanishes below the vortex rests on it, because
    # every horizontal flux dynadv_up3.f90:174-196 builds is the transport
    # times the ADVECTED velocity ``zui``/``zl`` -- both of which are
    # ``uu(:,:,:,Kmm)`` alone, with no barotropic correction.
    entry_profile = {
        face: [float(np.max(np.abs(np.asarray(entry1[face])[..., k])))
               for k in range(nlev)]
        for face in ("u", "v")}

    rows = []
    analysis = {}
    for face in ("u", "v"):
        active = np.asarray(masks[face], dtype=bool)
        total_lego = completed[face] - pre_adv[face]
        total_nemo = (groups[f"adv_{face}"][..., :nlev]
                      - groups[f"base_{face}"][..., :nlev])
        H = hadv[face]
        V = vadv[face]
        Z = zub[face]
        if plant in (f"hadv.{face}", f"vadv.{face}", f"zub.{face}"):
            which = plant.split(".")[0]
            target = {"hadv": H, "vadv": V, "zub": Z}[which].copy()
            target[tuple(d // 2 for d in target.shape)] += 1.0
            H, V, Z = {"hadv": (target, V, Z), "vadv": (H, target, Z),
                       "zub": (H, V, target)}[which]
        delta = total_lego - total_nemo

        full = active.all(axis=-1)
        cols = delta[full]
        depth_varying = float(np.max(np.abs(
            cols - cols.mean(axis=-1, keepdims=True))))
        # P3: the two halves are the same partition round 204 scored.
        partition = float(np.max(np.abs((H + V) - total_lego)[active]))
        rows.append({
            "name": f"{CASE}.stage1.advsplit.{face}",
            "nemo_boundary": "dyn_adv_up3's trend (stprk3_stg.f90:316)",
            "execution_regime": "production_step_jit",
            "planted": plant in (f"hadv.{face}", f"vadv.{face}",
                                 f"zub.{face}"),
            "cells_unequal": int(np.count_nonzero(
                (total_lego != total_nemo)[active])),
            "max_abs": float(np.max(np.abs(delta[active]))),
            "depth_varying": depth_varying,
            "partition_max_abs": partition,
            "hadv_max_abs": float(np.max(np.abs(H[active]))),
            "vadv_max_abs": float(np.max(np.abs(V[active]))),
            "levels_unequal": _unequal_per_level(
                np.where(total_lego != total_nemo, 1.0, 0.0), active),
            "levels_max_hadv": _per_level(H, active),
            "levels_max_vadv": _per_level(V, active),
            "levels_max_delta": _per_level(delta, active),
            "zub_max_abs": float(np.max(np.abs(Z[active]))),
            "levels_max_zub": _per_level(Z, active),
            "levels_max_entry_velocity": entry_profile[face],
        })
        # ---- THE CAUSAL ARMS, one operand each ------------------------
        # Each replaces ONE operand of stprk3_stg.f90:270 with NEMO's and
        # nothing else.  ``moved_vs_baseline`` is the non-vacuity control:
        # an arm that changed nothing at all is INERT and its "no effect"
        # says nothing.
        for arm, totals in causal.items():
            tp = totals[face]
            dp = tp - total_nemo
            cols_p = dp[full]
            rows[-1][f"causal_{arm}"] = {
                "max_abs": float(np.max(np.abs(dp[active]))),
                "cells_unequal": int(np.count_nonzero(
                    (tp != total_nemo)[active])),
                "depth_varying": float(np.max(np.abs(
                    cols_p - cols_p.mean(axis=-1, keepdims=True)))),
                "moved_vs_baseline": float(np.max(np.abs(
                    (tp - total_lego)[active]))),
                "levels_unequal": _unequal_per_level(
                    np.where(tp != total_nemo, 1.0, 0.0), active),
            }
        # ---- P4: EXONERATION BY THE ZERO LEVELS ------------------------
        # On every level where the TOTAL difference is zero in every active
        # cell, both halves agree bit for bit there.  A half that is ACTIVE
        # on such a level is exonerated ON THAT LEVEL; a half that is ~0
        # there is not (its agreement is vacuous).
        zero_levels = [k for k in range(nlev)
                       if rows[-1]["levels_unequal"][k] == 0]
        analysis[face] = {
            "zero_difference_levels": zero_levels,
            "hadv_on_zero_levels": [rows[-1]["levels_max_hadv"][k]
                                    for k in zero_levels],
            "vadv_on_zero_levels": [rows[-1]["levels_max_vadv"][k]
                                    for k in zero_levels],
            "hadv_peak_level": int(np.argmax(rows[-1]["levels_max_hadv"])),
            "vadv_peak_level": int(np.argmax(rows[-1]["levels_max_vadv"])),
            "delta_peak_level": int(np.argmax(rows[-1]["levels_max_delta"])),
        }
        # ---- P5: is the difference a uniform rescaling of the trend? ----
        # A divisor error gives delta = T * (d_nemo/d_lego - 1) exactly, so
        # the ratio is a metric field, not a scatter.  Scored on the cells
        # within a decade of the trend's own peak, where the ratio is
        # meaningful.
        peak = float(np.max(np.abs(total_nemo[active])))
        big = active & (np.abs(total_nemo) > peak * 0.1) & (delta != 0.0)
        if big.any():
            ratio = delta[big] / total_nemo[big]
            q1, med, q3 = np.percentile(ratio, [25, 50, 75])
            analysis[face].update(
                ratio_cells=int(big.sum()),
                ratio_median=float(med),
                ratio_iqr=float(q3 - q1),
                ratio_iqr_over_median=float(abs(q3 - q1)
                                            / max(abs(med), 1.0e-300)),
                ratio_min=float(ratio.min()), ratio_max=float(ratio.max()))
        else:
            analysis[face]["ratio_cells"] = 0
        # ---- IS THE DIFFERENCE A RESCALING OF THE zub CROSS-TERM? ------
        # The horizontal half carries two pieces with different powers of
        # the velocity: the transport's own ``uu(Kmm)`` part, quadratic in
        # the vortex profile, and the DEPTH-UNIFORM barotropic correction
        # ``zub`` (stprk3_stg.f90:264-277), which makes a piece LINEAR in
        # it.  If ``delta`` is a fixed multiple of the ``zub`` increment,
        # cellwise and not merely level by level, the owner is that
        # cross-term and the multiple is its relative error.
        zbig = active & (np.abs(Z) > np.abs(Z[active]).max() * 0.1)
        if zbig.any():
            zratio = delta[zbig] / Z[zbig]
            zq1, zmed, zq3 = np.percentile(zratio, [25, 50, 75])
            analysis[face].update(
                zub_ratio_cells=int(zbig.sum()),
                zub_ratio_median=float(zmed),
                zub_ratio_iqr=float(zq3 - zq1),
                zub_ratio_iqr_over_median=float(abs(zq3 - zq1)
                                                / max(abs(zmed), 1e-300)),
                zub_ratio_min=float(zratio.min()),
                zub_ratio_max=float(zratio.max()),
                zub_residual_max_abs=float(np.max(np.abs(
                    delta[active] - zmed * Z[active]))))
        # The metric bound itself, stated as arithmetic on round 200's
        # measured transport agreement (see ZF_RELATIVE_ROUND200).
        analysis[face]["metric_divisor_bound"] = (
            peak * ZF_RELATIVE_ROUND200)

    # The verdict is computed, never asserted: a half is NAMED only if the
    # other half is active on a level where the total difference is zero.
    verdict = {}
    for face in ("u", "v"):
        a = analysis[face]
        row = next(r for r in rows if r["name"].endswith(f".{face}"))
        hz = max(a["hadv_on_zero_levels"], default=0.0)
        vz = max(a["vadv_on_zero_levels"], default=0.0)
        verdict[face] = {
            "horizontal_exonerated_on_zero_levels": bool(
                a["zero_difference_levels"]
                and hz > 1.0e-9),
            "vertical_exonerated_on_zero_levels": bool(
                a["zero_difference_levels"]
                and vz > 1.0e-9),
            "hadv_max_on_zero_levels": hz,
            "vadv_max_on_zero_levels": vz,
            "metric_bound_vs_depth_varying": (
                a["metric_divisor_bound"] / max(row["depth_varying"],
                                                1.0e-300)),
        }
    return {
        "case": CASE, "oracle_root": str(root), "legoesm_git_sha": sha,
        "plant": plant, "rows": rows, "analysis": analysis,
        "verdict": verdict, "status": "MEASURED",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    # SCOPE: a plant perturbs the scored ARRAY after it is read, so it
    # proves the row's reduction reacts -- NOT that the exposure is live.
    # Seam liveness is ``require_live``, which refuses any exposure that
    # handed back the ordinary step output.
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--clean-report", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_dir, plant=args.plant,
                     allow_dirty=args.allow_dirty)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)
                               + "\n")
    for row in report["rows"]:
        print(f"{row['name']}: cells={row['cells_unequal']} "
              f"max_abs={row['max_abs']:.6e} "
              f"depth_varying={row['depth_varying']:.6e}")
        print(f"   |H|max={row['hadv_max_abs']:.6e}  "
              f"|V|max={row['vadv_max_abs']:.6e}  "
              f"partition={row['partition_max_abs']:.6e}")
        print(f"   unequal/level {row['levels_unequal']}")
        print("   |H|/level  " + " ".join(f"{v:.3e}"
                                          for v in row["levels_max_hadv"]))
        print("   |V|/level  " + " ".join(f"{v:.3e}"
                                          for v in row["levels_max_vadv"]))
        print("   |zub|/lvl  " + " ".join(f"{v:.3e}"
                                          for v in row["levels_max_zub"]))
        for arm in ("prognostic_mean", "qco_depth"):
            c = row[f"causal_{arm}"]
            print(f"   CAUSAL {arm:16s} cells={c['cells_unequal']:6d} "
                  f"max_abs={c['max_abs']:.6e} "
                  f"depth_varying={c['depth_varying']:.6e} "
                  f"moved={c['moved_vs_baseline']:.3e}")
        print("   |vel|/lvl  " + " ".join(
            f"{v:.3e}" for v in row["levels_max_entry_velocity"]))
    print(json.dumps(report["verdict"], indent=2, sort_keys=True))
    print(json.dumps(report["analysis"], indent=2, sort_keys=True))
    if args.plant:
        planted = [r for r in report["rows"] if r["planted"]]
        if len(planted) != 1:
            print(f"REFUSE: {len(planted)} planted rows", file=sys.stderr)
            return 2
        clean = (json.loads(args.clean_report.read_text())
                 if args.clean_report
                 else run(args.oracle_dir, plant=None,
                          allow_dirty=args.allow_dirty))
        if clean.get("case") != report["case"] or clean.get("plant"):
            print("REFUSE: --clean-report is not an unplanted report",
                  file=sys.stderr)
            return 2
        key = args.plant.split(".")[0].replace("hadv", "hadv") + "_max_abs"
        name = planted[0]["name"]
        before = {r["name"]: r[key] for r in clean["rows"]}
        visible = planted[0][key] != before[name]
        print(f"PLANT {args.plant} "
              f"{'VISIBLE' if visible else 'NOT VISIBLE'}")
        return 1 if visible else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
