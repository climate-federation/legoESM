#!/usr/bin/env python3
"""Walk a VORTEX_SMT card's stage-1 TRACER chain over partial cells.

Decision 92 (operator note CD).  The seamount cards' first tracer row over
the ``1e-15`` bar is ``T`` at kt=2.  Round 218's acquisition
(``run.sh --variant smtvectra|smtflxtra``) records NEMO's own tracer
operands and outputs at every RK3 stage of kt=1, which no earlier seamount
record carried, and this walk scores legoESM's value of each one against
NEMO's in the compiled stage order:

    stprk3_stg.F90:276-277   zFu/zFv = e2u*(e3u_3d*(1+r3u(Kmm)*umask))*(uu+zub)
    stprk3_stg.F90:298-301   wzv -> ww, zFw = e1e2t*ww   (flux form)
    stprk3_stg.F90:463       tra_adv_trp updates/computes zFu,zFv,zFw
    stprk3_stg.F90:519       tra_adv -> tra_adv_fct      (ln_traadv_fct)
      traadv_fct.f90:170       fct_up1_2stp (upstream, two steps)
      traadv_fct.f90:502-508     1st-step upstream fluxes
      traadv_fct.f90:538         mid-step guess / (e3t_3d*(1+r3t(Kmm)))
      traadv_fct.f90:569-573     2nd-step AVERAGED upstream fluxes
      traadv_fct.f90:197-198     2nd-order centred anti-diffusive (nn_fct_h=2)
      traadv_fct.f90:265-266     2nd-order centred vertical       (nn_fct_v=2)
      traadv_fct.f90:316         nonosc flux limiter
      traadv_fct.f90:327         Krhs += ztra / (e3t_3d*(1+r3t(Kmm)))
    stprk3_stg.F90:521       tra_sbc_RK3 (this deck forces all fluxes to 0)
    stprk3_stg.F90:552-554   ts(Kaa) = ((1+r3t(Kbb))ts(Kbb)
                                        + rDt(1+r3t(Kmm))ts(Krhs)tmask)
                                       / (1+r3t(Kaa))

THE ONE-VARIABLE ARM.  The last row hands legoESM's tracer step NEMO's own
recorded ``zFu/zFv/zFw`` triplet and changes nothing else
(``stage1_tracer_transport_override``), so a tracer tendency that comes to
the bar says the FCT statements themselves are faithful over partial cells
and the tracer debt is INHERITED from the transports; a tendency that does
not says a tracer-path statement owns it.

Bit equality (``cells_unequal == 0``) is the standard here, not AT-BAR.
Note BD: no record size is predicted; every group is parsed from its own
declared rank and extents by the acquisition's own checker.
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

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _seed_from_record, _strip2, _strip3, _u_full, _v_full, read_bt_frame,
    read_stage,
)
from nemo_testcase_l1_vortex_round200_flux_stage1 import (  # noqa: E402
    require_live,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    GateError, expected_masks, lego_fields, read_entry, require, score,
)

CARDS = {"vec": "VORTEX_SMT_VEC-zps", "flux": "VORTEX_SMT-zps"}
DEFAULT_ROOTS = {
    "vec": Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/"
                "round7/VORTEX_SMT_R7_VEC_R8_OMIP_L1_P3/tracer"),
    "flux": Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/"
                 "round7/VORTEX_SMT_R7_OMIP_L1_P3/tracer"),
}
ORDER = ("zfu", "zfv", "zfw", "ww", "adv.T", "adv.S", "out.T", "out.S")
PLANTS = ORDER


def read_tracer_terms(root: Path, stage: int) -> dict[str, np.ndarray]:
    """Every named group of the round-218 tracer record, self-described."""
    checker_path = HERE / "nemo_testcase_l1_vortex" / "check_records.py"
    spec = importlib.util.spec_from_file_location(
        "vortex_record_checker_r18", checker_path)
    require(spec is not None and spec.loader is not None,
            "cannot load the record checker")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    path = root / f"oracle_tracer_terms_kt00000001_s{stage}.bin"
    require(path.is_file(), f"missing admitted record {path}")
    parsed = checker.parse_record(path)
    require(parsed["stage"] == stage, f"{path}: parsed the wrong stage")
    raw = path.read_bytes()
    offset = 16 + 4 * checker._FAMILIES["oracle_tracer_terms_kt"][1]
    out: dict[str, np.ndarray] = {}
    for name, meta in parsed["groups"].items():
        offset += 32
        count = meta["doubles"]
        values = np.frombuffer(raw[offset:offset + 8 * count],
                               dtype=np.float64).copy()
        offset += 8 * count
        shape = tuple(meta["shape"])
        if meta["rank"] == 3:
            out[name] = _strip3(values, *shape)
        else:
            # Reviewer finding 3: the shared stripper, not a local copy with
            # a hard-coded halo width.
            out[name] = _strip2(values, *shape)
    require(offset == len(raw), f"{path}: parser did not consume the record")
    return out


def run(root: Path, card_key: str, *, plant: str | None = None,
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
    require(card_key in CARDS, f"unknown card key {card_key!r}")

    case = CARDS[card_key]
    card = build_nemo_testcase_card(case)
    cfg = card.recipe.model_config
    # The deck this record came from, read back off the card: a walk that
    # scored a different scheme would be measuring another experiment.
    require(cfg.tracer_advection in ("fct2", "fct"),
            f"{case} no longer selects NEMO's FCT tracer advection "
            f"({cfg.tracer_advection!r})")
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry1 = read_entry(root / "oracle_step_entry_kt00000001.bin", case,
                        expect_interior=interior)
    entry2 = read_entry(root / "oracle_step_entry_kt00000002.bin", case,
                        expect_interior=interior)
    stage1 = read_stage(root / "oracle_stage_kt00000001_s1.bin",
                        expect_step=1, expect_stage=1)
    groups = read_tracer_terms(root, 1)
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

    # THE RECORD'S OWN CROSS-CHECK, before anything below is believed: the
    # group written at the end of stage 1 must BE the stage-1 tracer the
    # older state record already carries.  If they disagree the new writer
    # is not at the boundary it claims.
    for tracer, slot in (("T", "out_t"), ("S", "out_s")):
        mismatch = int(np.count_nonzero(
            groups[slot][..., :nlev] != np.asarray(stage1[tracer])[..., :nlev]))
        require(mismatch == 0,
                f"the tracer record's {slot} differs from the stage-1 state "
                f"record in {mismatch} cells: the writer is not at the "
                "boundary it claims")

    def model_step(hooks):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return model.step(seed, dt=card.dt_s)

    plain = lego_fields(model_step(_NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external)))

    rows: list[dict] = []

    def _row(label, reference, candidate, mask, nemo_boundary, planted):
        reference = np.asarray(reference)
        candidate = np.asarray(candidate)
        if planted:
            # Reviewer finding 4: perturb a cell that is actually SCORED.
            # The geometric centre can be masked, and a plant that lands on
            # a masked cell reports NOT VISIBLE for a live seam.
            active_idx = np.argwhere(np.asarray(mask, dtype=bool))
            require(active_idx.size > 0, f"{label}: empty scored support")
            candidate = candidate.copy()
            candidate[tuple(active_idx[len(active_idx) // 2])] += 1.0
        row = score(f"{case}.stage1.{label}", reference, candidate, mask)
        active = np.asarray(mask, dtype=bool)
        delta = candidate - reference
        row["cells_unequal"] = int(np.count_nonzero(
            (candidate != reference)[active]))
        row["max_abs"] = float(np.max(np.abs(delta[active])))
        peak = float(np.max(np.abs(reference[active])))
        row["relative_max_abs"] = row["max_abs"] / max(peak, 1.0e-300)
        row["bit_exact"] = row["cells_unequal"] == 0
        row["nemo_boundary"] = nemo_boundary
        row["execution_regime"] = "production_step_jit"
        row["planted"] = bool(planted)
        bad = (candidate != reference) & active
        idx = np.argwhere(bad)
        structure = {"n_unequal": int(bad.sum()), "n_active": int(active.sum())}
        if idx.size and reference.ndim == 3:
            structure["levels"] = np.bincount(
                idx[:, -1], minlength=reference.shape[-1]).tolist()
            structure["j_range"] = [int(idx[:, 0].min()), int(idx[:, 0].max())]
            structure["i_range"] = [int(idx[:, 1].min()), int(idx[:, 1].max())]
        row["structure"] = structure
        rows.append(row)
        return row

    # ---- 1. the stage-1 tracer transports, AS tra_adv receives them ------
    hooks = _NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external,
        expose_tracer_transport_stage=1)
    fields = lego_fields(model_step(hooks))
    for face in ("u", "v"):
        require_live("zf" + face, face, np.asarray(fields[face])[..., :nlev],
                     plain[face])
        _row(f"zf{face}", groups[f"zf{face}"][..., :nlev],
             np.asarray(fields[face])[..., :nlev], masks[face],
             "zFu/zFv as tra_adv receives them, with the partial-cell "
             "reference face thickness (stprk3_stg.f90:276-277, :463)",
             plant == f"zf{face}")
    require_live("zfw", "T", np.asarray(fields["T"])[..., :nlev], plain["T"])
    _row("zfw", groups["zfw"][..., :nlev],
         np.asarray(fields["T"])[..., :nlev], masks["T"],
         "zFw as tra_adv receives it (stprk3_stg.f90:301, :463)",
         plant == "zfw")

    # ---- 2. the stage-1 continuity solve without the metric --------------
    hooks = _NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external,
        expose_tracer_transport_stage=1, expose_tracer_transport_as_ww=True)
    fields = lego_fields(model_step(hooks))
    require_live("ww", "T", np.asarray(fields["T"])[..., :nlev], plain["T"])
    _row("ww", groups["ww"][..., :nlev],
         np.asarray(fields["T"])[..., :nlev], masks["T"],
         "ww after the stage continuity solve (stprk3_stg.f90:298)",
         plant == "ww")

    # ---- 3. the tracer right-hand side after FCT + the surface flux ------
    hooks = _NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external,
        expose_tracer_stage1_boundary="after_sbc")
    fields = lego_fields(model_step(hooks))
    for tracer, slot in (("T", "adv_t"), ("S", "adv_s")):
        require_live(f"adv.{tracer}", tracer,
                     np.asarray(fields[tracer])[..., :nlev], plain[tracer])
        _row(f"adv.{tracer}", groups[slot][..., :nlev],
             np.asarray(fields[tracer])[..., :nlev], masks["T"],
             "ts(Krhs) after tra_adv + tra_sbc_RK3 "
             "(stprk3_stg.f90:519,521; traadv_fct.f90:327)",
             plant == f"adv.{tracer}")

    # ---- 4. the stage-1 after-tracer -------------------------------------
    hooks = _NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external, expose_tracer_stage=1)
    fields = lego_fields(model_step(hooks))
    for tracer, slot in (("T", "out_t"), ("S", "out_s")):
        require_live(f"out.{tracer}", tracer,
                     np.asarray(fields[tracer])[..., :nlev], plain[tracer])
        _row(f"out.{tracer}", groups[slot][..., :nlev],
             np.asarray(fields[tracer])[..., :nlev], masks["T"],
             "ts(Kaa) after the qco thickness-weighted stage step "
             "(stprk3_stg.f90:552-554)", plant == f"out.{tracer}")

    # ---- 5. THE ONE-VARIABLE ARM: NEMO's transports, everything else ours
    carrier_rows: list[dict] = []
    if plant is None:
        # The hook takes the OWNED triplet on the T-column index -- it
        # restores the one ghost itself (``_tracer_transport_geometry_override``
        # concatenates before dividing by dy_u/dx_v/area_T) -- and the
        # vertical member carries the w half, nlev+1 deep.
        require(groups["zfu"].shape[:2] == groups["zfw"].shape[:2]
                == tuple(np.asarray(
                    card.recipe.initial_state.T.data).shape[:2]),
                "the record's transports are not on the card's T-column index")
        require(groups["zfw"].shape[-1] == nlev + 1,
                f"the record's zFw is {groups['zfw'].shape[-1]} deep, the "
                f"card needs {nlev + 1}")
        # REVIEWER FINDING 1, MEASURED RATHER THAN ARGUED.  The hook sends
        # the vertical member through ``zfw / area_T`` and the tracer helper
        # multiplies by ``area_T`` again, so a recorded zFw that happened to
        # be identically zero would make the w half of this arm VACUOUS --
        # bit equality for free, proving nothing about the vertical path.
        # It is not zero: this refuses unless the recorded field is live
        # over the scored support.
        _zfw_nonzero = int(np.count_nonzero(
            groups["zfw"][..., :nlev][np.asarray(masks["T"], dtype=bool)]))
        require(_zfw_nonzero > 0,
                "the recorded zFw is identically zero over the scored "
                "support; the vertical half of the transport arm would be "
                "vacuous")
        override = (
            jnp.asarray(groups["zfu"][..., :nlev]),
            jnp.asarray(groups["zfv"][..., :nlev]),
            jnp.asarray(groups["zfw"]),
        )
        for exposure, slots in (("after_sbc", ("adv_t", "adv_s")),
                                ("stage", ("out_t", "out_s"))):
            kwargs = {"stage_barotropic_output_override": external,
                      "stage1_tracer_transport_override": override}
            if exposure == "after_sbc":
                kwargs["expose_tracer_stage1_boundary"] = "after_sbc"
            else:
                kwargs["expose_tracer_stage"] = 1
            carried = lego_fields(model_step(_NEMOWSRK3TestHooks(**kwargs)))
            for tracer, slot in zip(("T", "S"), slots):
                # REVIEWER FINDING 2: the arm that carries this round is the
                # one exposure that was not seam-controlled.  An override
                # that silently went inert would hand back the plain step
                # output and still score.
                require_live(f"nemo_transport.{slot}", tracer,
                             np.asarray(carried[tracer])[..., :nlev],
                             plain[tracer])
                reference = groups[slot][..., :nlev]
                candidate = np.asarray(carried[tracer])[..., :nlev]
                active = np.asarray(masks["T"], dtype=bool)
                before = next(r for r in rows
                              if r["name"].endswith(
                                  f".{'adv' if exposure == 'after_sbc' else 'out'}"
                                  f".{tracer}"))
                carrier_rows.append({
                    "name": f"{case}.stage1.nemo_transport.{slot}",
                    "cells_unequal": int(np.count_nonzero(
                        (candidate != reference)[active])),
                    "max_abs": float(np.max(np.abs(
                        (candidate - reference)[active]))),
                    "max_abs_before": before["max_abs"],
                    "relative_max_abs": float(np.max(np.abs(
                        (candidate - reference)[active]))) / max(
                            float(np.max(np.abs(reference[active]))),
                            1.0e-300),
                    "execution_regime": "production_step_jit",
                    "nemo_boundary": (
                        "the same boundary, with NEMO's recorded zFu/zFv/zFw "
                        "as the ONLY substituted operand"),
                })

    first = next((r for r in rows if not r["bit_exact"]), None)
    report = {
        "case": case, "oracle_root": str(root), "legoesm_git_sha": sha,
        "plant": plant, "rows": rows, "carrier_rows": carrier_rows,
        "first_non_bit": first["name"] if first else None,
        "first_non_bit_max_abs": first["max_abs"] if first else 0.0,
        "status": "MEASURED" if (plant is not None or first is not None)
                  else "NO_OWNER",
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", choices=sorted(CARDS), default="vec")
    parser.add_argument("--oracle-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--clean-report", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    root = args.oracle_dir or DEFAULT_ROOTS[args.card]
    try:
        report = run(root, args.card, plant=args.plant,
                     allow_dirty=args.allow_dirty)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)
                               + "\n")
    for row in report["rows"]:
        print("{name:28s} bit={bit!s:5s} cells={cells:7d} "
              "max_abs={m:.6e} rel={r:.3e}".format(
                  name=row["name"].split(".", 1)[1], bit=row["bit_exact"],
                  cells=row["cells_unequal"], m=row["max_abs"],
                  r=row["relative_max_abs"]))
    for row in report["carrier_rows"]:
        print("{name:28s} cells={cells:7d} max_abs={m:.6e} "
              "(was {b:.6e})".format(
                  name=row["name"].split(".", 1)[1], cells=row["cells_unequal"],
                  m=row["max_abs"], b=row["max_abs_before"]))
    print("first non-bit:", report["first_non_bit"])
    print("status:", report["status"])
    if args.plant:
        planted = [r for r in report["rows"] if r["planted"]]
        if len(planted) != 1:
            print(f"REFUSE: {len(planted)} planted rows, expected 1",
                  file=sys.stderr)
            return 2
        clean = (json.loads(args.clean_report.read_text())
                 if args.clean_report
                 else run(root, args.card, plant=None,
                          allow_dirty=args.allow_dirty))
        if clean.get("case") != report["case"] or clean.get("plant"):
            print("REFUSE: --clean-report is not an unplanted report for "
                  f"{report['case']}", file=sys.stderr)
            return 2
        before = {r["name"]: r["max_abs"] for r in clean["rows"]}
        name = planted[0]["name"]
        visible = name in before and planted[0]["max_abs"] != before[name]
        print(f"PLANT {args.plant} {'VISIBLE' if visible else 'NOT VISIBLE'}")
        return 1 if visible else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
