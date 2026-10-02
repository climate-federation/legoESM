#!/usr/bin/env python3
"""Walk the VORTEX FLUX card's stage-1 momentum chain, statement by statement.

Round 200's stage-local walk puts 4.4e-08 of the flux card's kt=2 velocity
error inside STAGE 1 -- and stage 1 of the flux-form program runs exactly one
momentum statement beyond what ``stp_2D`` already completed:

    stprk3_stg.F90:298     CALL wzv( ..., zFu, zFv, ww, np_transport )
    stprk3_stg.F90:301     zFw = e1e2t * ww
    stprk3_stg.F90:315     IF( .NOT.ln_dynadv_vec ) CALL dyn_adv( ..., zFu, zFv, zFw )
    stprk3_stg.F90:371-378 Kaa = ( e3u(Kbb)*u(Kbb) + rDt*e3u(Kmm)*Krhs ) / e3u(Kaa)
    stprk3_stg.F90:437-448 barotropic replacement, then the stage output

Round 200's acquisition records NEMO's operands at every one of those
boundaries (``oracle_stage_flux_terms_kt00000001_s1.bin``).  This walk scores
legoESM's own value of each operand, taken from the PRODUCTION-jitted step
through the existing WRITE-only seams, against NEMO's, in that order, and
names the FIRST one that is not bit-identical.

Exactness here is bit equality (``cells_unequal == 0``), never AT-BAR: the
trajectory bar classifies a row, it does not certify a statement.

Note BD: no record size or header tuple is predicted; every group is parsed
from its own declared rank and extents by the acquisition's own checker.
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
    _seed_from_record, _strip3, _u_full, _v_full, read_bt_frame, read_stage,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    GateError, expected_masks, lego_fields, read_entry, require, score,
)

CASE = "VORTEX-zco"
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round200/"
    "oracle_stage123_flux_terms")
# The boundaries this walk can score with an existing WRITE-only seam, in
# NEMO's own stage-1 execution order.
PLANTS = ("base.u", "base.v", "zfu", "zfv", "zfw", "ww",
          "out.u", "out.v")


def read_flux_stage_terms(root: Path, stage: int) -> dict[str, np.ndarray]:
    """Every named group of the round-200 flux record, self-described."""
    checker_path = HERE / "nemo_testcase_l1_vortex" / "check_records.py"
    spec = importlib.util.spec_from_file_location(
        "vortex_record_checker", checker_path)
    require(spec is not None and spec.loader is not None,
            "cannot load the record checker")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    path = root / f"oracle_stage_flux_terms_kt00000001_s{stage}.bin"
    parsed = checker.parse_record(path)
    require(parsed["stage"] == stage, f"{path}: parsed the wrong stage")
    raw = path.read_bytes()
    offset = 16 + 4 * 15
    out = {}
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
            nx, ny = shape
            out[name] = values.reshape((nx, ny), order="F")[2:-2, 2:-2].T
    require(offset == len(raw), f"{path}: parser did not consume the record")
    return out


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
    stage1 = read_stage(root / "oracle_stage_kt00000001_s1.bin",
                        expect_step=1, expect_stage=1)
    groups = read_flux_stage_terms(root, 1)
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

    # The record's own cross-check: the group written after the barotropic
    # correction must BE the stage-1 output the older stage record carries.
    # If these two disagree the new record is instrumented at the wrong
    # boundary and nothing below is believable.
    for face in ("u", "v"):
        mismatch = int(np.count_nonzero(
            groups[f"out_{face}"][..., :nlev]
            != np.asarray(stage1[face])[..., :nlev]))
        require(mismatch == 0,
                f"the flux record's out_{face} differs from the stage-1 "
                f"output record in {mismatch} cells: the writer is not at "
                "the boundary it claims")

    def model_step(hooks):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return model.step(seed, dt=card.dt_s)

    def _owned(values, face):
        """NEMO's local interior stores one x record per T column."""
        values = np.asarray(values)
        return values[:, 1:, :] if face == "u" else values[1:, :, :]

    # THE SEAM CONTROL.  Every row below is read out of a state slot that the
    # ordinary step also fills, so an exposure hook that silently went inert
    # would hand the walk the PLAIN step output and every row would still
    # score.  Perturbing the candidate afterwards cannot catch that -- it
    # proves the scoring reacts, not that the seam is live.  So the plain
    # step is run once here and each exposed slot must DIFFER from it.
    plain = lego_fields(model_step(_NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external)))

    def _require_live(label, slot, values):
        same = int(np.count_nonzero(
            np.asarray(values) == np.asarray(plain[slot])[
                ..., :np.asarray(values).shape[-1]]))
        total = int(np.asarray(values).size)
        require(same != total,
                f"{label}: the exposed {slot} slot is identical to the "
                "ordinary step output in every cell; the seam is inert")

    observed: dict[str, np.ndarray] = {}

    def _observe(name):
        def _hook(values):
            observed[name] = np.asarray(values)
        return _hook

    rows = []

    def _row(label, reference, candidate, mask, nemo_boundary, planted,
             convention_sensitive=False):
        reference = np.asarray(reference)
        candidate = np.asarray(candidate)
        if planted:
            candidate = candidate.copy()
            candidate[tuple(d // 2 for d in candidate.shape)] += 1.0
        row = score(f"{CASE}.stage1.{label}", reference, candidate, mask)
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
        # A row whose two sides are the same NAME but not provably the same
        # QUANTITY cannot carry an attribution.  In flux form NEMO leaves the
        # advection OUT of the three-dimensional Krhs and puts its depth mean
        # straight into the two-dimensional Ue_rhs/Ve_rhs instead
        # (stp2d.F90:170 `dyn_adv_up3(..., pUe=Ue_rhs, pVe=Ve_rhs)`, labelled
        # "2D RHS only", then :183 cumulates), while legoESM's observed
        # pre-stage array is its own completed right-hand side.  The row is
        # reported, never used to name an owner.
        row["convention_sensitive"] = bool(convention_sensitive)
        rows.append(row)
        return row

    # ---- 1. the completed pre-stage momentum RHS (NEMO's stage-1 `base`) --
    # stp2d.F90:126-171 builds it; stprk3_stg carries it into stage 1 as Krhs.
    # REPORTED ONLY: see the convention note on _row.
    for face in ("u", "v"):
        hooks = _NEMOWSRK3TestHooks(
            stage_barotropic_output_override=external,
            slow_forcing_rhs_observer=_observe(f"base_{face}"),
            slow_forcing_rhs_observer_face=face)
        model_step(hooks)
        require(f"base_{face}" in observed,
                f"the pre-stage RHS observer never fired for face {face}")
        _row(f"base.{face}", groups[f"base_{face}"][..., :nlev],
             _owned(observed[f"base_{face}"], face)[..., :nlev], masks[face],
             "Krhs at the stage-1 boundary (stp2d.F90:126-171)",
             plant == f"base.{face}", convention_sensitive=True)

    # ---- 2. the stage-1 horizontal advective transports ------------------
    # stprk3_stg.F90:276-277:
    #   zFu = e2u*(e3t_1d*(1+r3u(Kmm)*umask)) * ( uu(Kmm) + zub*umask )
    # These are dyn_adv's horizontal operands, and the SAME arrays the tracer
    # transport consumes -- NEMO builds them once for both.
    hooks = _NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external,
        expose_tracer_transport_stage=1)
    fields = lego_fields(model_step(hooks))
    for face in ("u", "v"):
        _require_live("zf" + face, face,
                      np.asarray(fields[face])[..., :nlev])
        _row(f"zf{face}", groups[f"zf{face}"][..., :nlev],
             np.asarray(fields[face])[..., :nlev], masks[face],
             "zFu/zFv, the stage advective transports "
             "(stprk3_stg.F90:276-277)", plant == f"zf{face}")
    _require_live("zfw", "T", np.asarray(fields["T"])[..., :nlev])
    _row("zfw", groups["zfw"][..., :nlev],
         np.asarray(fields["T"])[..., :nlev], masks["T"],
         "zFw = e1e2t*ww (stprk3_stg.F90:301)", plant == "zfw")

    # ---- 3. the stage-1 continuity solve (NEMO's np_transport wzv) -------
    # stprk3_stg.F90:298.  The same exposure, asked for ww rather than the
    # area-weighted transport, so the solve is scored without the metric.
    hooks = _NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external,
        expose_tracer_transport_stage=1,
        expose_tracer_transport_as_ww=True)
    fields = lego_fields(model_step(hooks))
    _require_live("ww", "T", np.asarray(fields["T"])[..., :nlev])
    _row("ww", groups["ww"][..., :nlev],
         np.asarray(fields["T"])[..., :nlev], masks["T"],
         "ww after wzv(..., np_transport) (stprk3_stg.F90:298)",
         plant == "ww")

    # ---- 4. the stage-1 output, after the barotropic replacement ---------
    hooks = _NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external,
        expose_momentum_stage=1, expose_tracer_stage=1)
    fields = lego_fields(model_step(hooks))
    for face in ("u", "v"):
        _require_live("out." + face, face,
                      np.asarray(fields[face])[..., :nlev])
        _row(f"out.{face}", groups[f"out_{face}"][..., :nlev],
             np.asarray(fields[face])[..., :nlev], masks[face],
             "stage-1 output after the barotropic correction "
             "(stprk3_stg.F90:437-450)", plant == f"out.{face}")

    first = next((r for r in rows
                  if not r["bit_exact"] and not r["convention_sensitive"]),
                 None)
    report = {
        "case": CASE, "oracle_root": str(root), "legoesm_git_sha": sha,
        "plant": plant, "rows": rows,
        "first_non_bit": first["name"] if first else None,
        "first_non_bit_max_abs": first["max_abs"] if first else 0.0,
        "status": "MEASURED",
    }
    if plant is None and first is None:
        # Every seam this walk can reach is bit-exact while the stage-1
        # OUTPUT is known to be over the bar: that is a refusal, not a pass.
        report["status"] = "NO_OWNER"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS,
                        help="perturb ONE scored candidate; the run MUST "
                             "report that row non-bit, or the seam is inert")
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
        print("{name:34s} bit={bit!s:5s} cells={cells:7d} "
              "max_abs={m:.6e} rel={r:.3e}{tag}".format(
                  name=row["name"].split(".", 1)[1], bit=row["bit_exact"],
                  cells=row["cells_unequal"], m=row["max_abs"],
                  r=row["relative_max_abs"],
                  tag="  (convention-sensitive, not an owner)"
                      if row["convention_sensitive"] else ""))
    print("first non-bit:", report["first_non_bit"])
    print("status:", report["status"])
    if args.plant:
        planted = [r for r in report["rows"] if r["planted"]]
        if len(planted) != 1:
            print(f"REFUSE: {len(planted)} planted rows, expected 1",
                  file=sys.stderr)
            return 2
        visible = not planted[0]["bit_exact"]
        print(f"PLANT {args.plant} "
              f"{'VISIBLE' if visible else 'NOT VISIBLE'}")
        return 1 if visible else 0
    return 0 if report["status"] == "MEASURED" else 3


if __name__ == "__main__":
    sys.exit(main())
