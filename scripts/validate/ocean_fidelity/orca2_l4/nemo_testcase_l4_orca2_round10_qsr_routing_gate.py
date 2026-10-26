#!/usr/bin/env python3
"""ORCA2 round-10 gate: the PRODUCTION ROUTING of NEMO's ``qsr_RGBc`` deposit.

The RGB kernel itself is already gated cell by cell against this record by
``nemo_testcase_l4_orca2_rgb_gate.py``, which calls it directly with
oracle-supplied operands.  What that gate does NOT cover is the thing round 10
changed: whether the ORCA2 CARD's own configuration and the operand expressions
the production sites build reach that kernel unaltered.

So this gate takes nothing by hand.  The scheme, the infrared fraction and
length, the chlorophyll profile and the time step come from the card; the
reference depth ladder and the reference thickness come from the card's
vertical coordinate and are first required to equal NEMO's own ``gdepw_1d`` /
``e3t_1d`` / ``e3t_0``; the live operands are built with the SAME two
expressions the production sites use
(``packages/ocean/legoesm/ocean/physics/combined.py`` and
``packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py``):

    dz_live            = e3t_0 * (1 + r3t*tmask)
    gdepw_bottom_live  = -z_half_ref[1:] * (1 + r3t)

and the dispatcher the production sites call is
``apply_shortwave_penetration``, not the kernel underneath it.  The comparison
is against the record's own ``qsr`` right-hand-side increment, every owned wet
cell, bit for bit.

Controls (each must FIRE, i.e. leave cells unequal):
  --control two-band     the card's own two-band sibling instead of RGB
  --control static       the reference ladder instead of the live one
  --control surface-chl  the surface chlorophyll profile instead of Morel-Berthon
  --plant                one representable value moved on one wet cell
  --control pipeline-plant  the same, on the SHARED PIPELINE row
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nemo_testcase_l4_orca2_rgb_gate import (  # noqa: E402
    ACTIVE_Z,
    GateError,
    read_qsr,
    read_rgb,
    read_stage3_before_qsr,
    require,
    score,
    sha256,
)

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402

# Resolved ORCA2 settings, each read from the record's own ocean.output.  The
# card must agree with every one; a card that drifts from the record is the
# defect this table exists to catch.
RESOLVED = {
    "scheme": ("nemo_qsr_rgb", "ln_qsr_rgb = T / nn_chldta = 1, ocean.output:1207,1211"),
    "rgb_chl_profile": ("morel_berthon", "nn_chlprfl = 1, ocean.output:1212"),
    "rgb_ir_fraction": (0.58, "rn_abs, ocean.output:1213"),
    "rgb_ir_extinction_m": (0.35, "rn_si0, ocean.output:1214"),
    "nemo_time_step_s": (10800.0, "rn_Dt, ocean.output:217"),
}
RESOLVED_RHO0 = (1026.0, "rho0, ocean.output:172")
RESOLVED_RCP = (3991.8679571196299, "rcp, ocean.output:174")


def card_shortwave_operands(deck_root: Path):
    """The card's OWN configuration and reference vertical ladder."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card

    card = build_orca2_zps_card(deck_root)
    config = card.recipe.model_config
    sw_cfg = config.physics.shortwave_penetration
    for field, (expected, provenance) in RESOLVED.items():
        actual = getattr(sw_cfg, field)
        require(actual == expected,
                f"ORCA2 card {field} is {actual!r}, record resolves "
                f"{expected!r} ({provenance})")
    require(config.rho_0 == RESOLVED_RHO0[0],
            f"card rho_0 {config.rho_0} vs record {RESOLVED_RHO0}")
    require(config.physics.constants.c_sw == RESOLVED_RCP[0],
            f"card c_sw {config.physics.constants.c_sw} vs record {RESOLVED_RCP}")
    z_coord = card.recipe.z_coord
    return card, config, sw_cfg, z_coord


def production_rgb_tendency(sw_cfg, config, z_coord, *, qsr, chl, h_ref, r3t,
                            tmask, live: bool):
    """The two operand expressions the production sites build, verbatim.

    ``combined.py`` builds ``gdepw_bottom_live = -z_half_ref[1:]*J`` and
    ``dz_live = compute_layer_thickness(...)``; the RK3 stage-3 seam in
    ``ocean_model_latlon_cgrid.py`` builds the same pair on the stage ladder.
    Here the stretch is the RECORD's own ``1 + r3t`` so the comparison isolates
    the routing, not the free surface.
    """
    from legoesm.ocean.physics.shortwave_penetration import (
        apply_shortwave_penetration,
    )

    stretch = np.float64(1.0) + (r3t if live else np.zeros_like(r3t))
    # ``compute_layer_thickness`` IS ``h_partial * J`` and ``J`` IS NEMO's
    # ``1 + r3t`` (vertical.py) -- ONE stretch, not two.  ``h_partial`` is
    # ZERO below the seafloor, so ``dz_live > 0`` is the wet mask, which is
    # what NEMO's ``wmask`` factor does at traqsr.f90:388-391.  No separate
    # tmask multiply: NEMO's ``e3t_0`` is non-zero everywhere and legoESM's
    # reference thickness is not.
    del tmask
    dz_live = h_ref * stretch[..., None]
    gdepw_ref = -jnp.asarray(z_coord.z_half_ref, dtype=jnp.float64)
    gdepw_bottom_live = gdepw_ref[1:] * jnp.asarray(stretch)[..., jnp.newaxis]
    dz_live = jnp.asarray(dz_live)
    return apply_shortwave_penetration(
        sw_cfg,
        jnp.asarray(qsr),
        chl=jnp.asarray(chl),
        dz_live=dz_live,
        wet_cell=jnp.asarray(dz_live > 0.0, dtype=jnp.float64),
        gdepw_bottom_live=gdepw_bottom_live,
        gdepw_ref=gdepw_ref,
        e3t_ref=jnp.asarray(z_coord.dz_ref, dtype=jnp.float64),
        rho_0=config.rho_0,
        c_sw=config.physics.constants.c_sw,
    )


def pipeline_shortwave_deposit(card, config, state):
    """The deposit the SHARED PHYSICS PIPELINE itself applies.

    This calls ``make_ocean_physics`` -- the production factory -- with the
    card's OWN shortwave configuration and every other module switched off, so
    the only tendency it can return is the shortwave block of
    ``packages/ocean/legoesm/ocean/physics/combined.py``.  That block is the
    thing round 10 changed: before it, this call RAISES, which is what makes
    this row bind on the diff rather than merely re-run a kernel that was
    already there.
    """
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.combined import OceanPhysicsConfig, make_ocean_physics
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig

    only_shortwave = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=config.physics.shortwave_penetration,
        mle=None,
        constants=config.physics.constants,
    )
    physics_fn = make_ocean_physics(only_shortwave)
    return physics_fn


def validate(deck_root: Path, root: Path, *, plant: bool, control: str | None) -> dict:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "routing gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    rgb = read_rgb(root / "oracle_rgb_chl_kt00000001.bin")
    qsr = read_qsr(root / "oracle_qsr_stage3_kt00000001.bin")
    before_qsr = read_stage3_before_qsr(
        root / "oracle_rktracer_stage3_kt00000001.bin")
    with Dataset(root / "mesh_mask_0000.nc") as dataset:
        tmask = np.asarray(dataset["tmask"][0, :ACTIVE_Z].data, dtype=np.float64)
        e3t_0 = np.asarray(dataset["e3t_0"][0, :ACTIVE_Z].data, dtype=np.float64)
        e3t_1d = np.asarray(dataset["e3t_1d"][0, :ACTIVE_Z].data, dtype=np.float64)
    tmask = tmask.transpose(1, 2, 0)
    e3t_0 = e3t_0.transpose(1, 2, 0)
    wet = tmask > 0.0

    card, config, sw_cfg, z_coord = card_shortwave_operands(deck_root)

    # The card's OWN reference ladder must be NEMO's, or the operand
    # expressions above are being fed a different ocean.
    # ``+ 0.0`` maps the negated surface point's -0.0 back onto NEMO's +0.0;
    # they are the same number and only the bit pattern differs.
    gdepw_card = -np.asarray(z_coord.z_half_ref, dtype=np.float64) + 0.0
    gdepw_record = np.asarray(rgb["gdepw_1d"], dtype=np.float64)
    dz_card = np.asarray(z_coord.dz_ref, np.float64)
    gdepw_bad = np.nonzero(
        gdepw_card.view(np.uint64) != gdepw_record.view(np.uint64))[0]
    e3t1d_bad = np.nonzero(dz_card.view(np.uint64) != e3t_1d.view(np.uint64))[0]
    h_ref = np.asarray(z_coord.h_partial, dtype=np.float64)[:, :90]
    e3t0_unequal = int(
        (h_ref[wet[:, :90]].view(np.uint64)
         != e3t_0[:, :90][wet[:, :90]].view(np.uint64)).sum())
    ladder_rows = {
        # The operand that actually enters the deposit is the 3-D partial-cell
        # thickness; it must be NEMO's, bit for bit.
        "e3t_0_unequal": e3t0_unequal,
        # The 1-D reference ladder is an INDEPENDENT statement of its own.  It
        # is measured, named and reported here rather than assumed; the gate
        # refuses only if a disagreeing level could reach the deposit.
        "gdepw_1d_unequal_levels": gdepw_bad.tolist(),
        "gdepw_1d_max_abs": float(np.abs(
            gdepw_card - gdepw_record).max(initial=0.0)),
        "e3t_1d_unequal_levels": e3t1d_bad.tolist(),
        "e3t_1d_max_abs": float(np.abs(dz_card - e3t_1d).max(initial=0.0)),
    }
    require(e3t0_unequal == 0,
            f"the card's partial-cell thickness is not NEMO's e3t_0: "
            f"{e3t0_unequal} cells unequal")

    if control == "two-band":
        sw_cfg = sw_cfg._replace(scheme="nemo_qsr_2bd")
    elif control == "surface-chl":
        # The record resolves nn_chlprfl = 1; the kernel must REFUSE any other
        # chlorophyll profile rather than silently substituting one.
        sw_cfg = sw_cfg._replace(rgb_chl_profile="surface")
        try:
            production_rgb_tendency(
                sw_cfg, config, z_coord, qsr=rgb["qsr"], chl=rgb["chl"],
                h_ref=h_ref, r3t=np.asarray(rgb["r3t"], dtype=np.float64),
                tmask=tmask[:, :90], live=True)
        except ValueError as exc:
            raise GateError(
                f"control surface-chl refused at the kernel, as it must: {exc}")
        raise GateError(
            "control surface-chl did NOT fire: an unresolved chlorophyll "
            "profile was silently accepted")

    from legoesm.ocean.physics.shortwave_penetration import (
        apply_shortwave_penetration,
    )

    r3t = np.asarray(rgb["r3t"], dtype=np.float64)
    if control == "two-band":
        # Same dispatcher, the card's two-band sibling: it takes the reference
        # ladder plus a stretch, never the chlorophyll operands.
        from legoesm.ocean.physics.shortwave_penetration import (
            shortwave_penetration_tendency,
        )
        stretch = jnp.asarray(np.float64(1.0) + r3t)
        raw = shortwave_penetration_tendency(
            jnp.asarray(rgb["qsr"]), jnp.asarray(z_coord.dz_ref),
            jnp.asarray(z_coord.z_half_ref), stretch, sw_cfg,
            config.rho_0, config.physics.constants.c_sw,
            z_half_stretch=stretch,
        )
    else:
        raw = production_rgb_tendency(
            sw_cfg, config, z_coord,
            qsr=rgb["qsr"], chl=rgb["chl"], h_ref=h_ref,
            r3t=r3t, tmask=tmask[:, :90], live=(control != "static"),
        )
    # The writer records ``(RHS_before + increment) - RHS_before``; reproduce
    # that association so the comparison is not measuring a different rounding.
    candidate = np.asarray(nemo_source_round(
        nemo_source_round(jnp.asarray(before_qsr) + raw) - jnp.asarray(before_qsr)
    )).copy()

    if plant:
        candidate = np.asarray(qsr["increment"]).copy()
        index = tuple(np.argwhere(wet[:, :90])[0])
        candidate[index] = np.nextafter(candidate[index], np.inf)

    result = score(candidate, qsr["increment"], wet[:, :90])

    # --- the row that BINDS on round 10's diff -------------------------------
    # Everything above tests the kernel and the operand expressions.  This runs
    # the SHARED PHYSICS PIPELINE itself, through the production factory, with
    # the card's own shortwave configuration -- the code path round 10 changed,
    # and the one that REFUSED before it.  The pipeline derives its own stretch
    # from the sea surface, so the state is given the sea surface that carries
    # the record's own r3t; that round trip is exact (measured below, and the
    # gate refuses if it stops being exact).
    pipeline = None
    if not (plant or control) or control == "pipeline-plant":
        from legoesm.ocean.state import OceanSurfaceForcing
        from legoesm.ocean.vertical import compute_ocean_jacobian

        state = card.recipe.initial_state
        H_bathy = np.asarray(state.H_bathy.data, dtype=np.float64)
        r3t_full = np.zeros(H_bathy.shape, dtype=np.float64)
        r3t_full[:, :90] = r3t
        eta = r3t_full * H_bathy
        jac = np.asarray(compute_ocean_jacobian(
            jnp.asarray(eta), state.H_bathy.data, z_coord))
        surface_wet = np.asarray(z_coord.is_active)[..., 0] > 0.0
        stretch_unequal = int((
            jac[surface_wet].view(np.uint64)
            != (np.float64(1.0) + r3t_full)[surface_wet].view(np.uint64)).sum())
        require(stretch_unequal == 0,
                "the sea surface that carries the record's r3t no longer "
                f"reproduces it through the pipeline's own jacobian: "
                f"{stretch_unequal} columns unequal")
        chl_full = np.zeros(H_bathy.shape, dtype=np.float64)
        chl_full[:, :90] = np.asarray(rgb["chl"], dtype=np.float64)
        qsr_full = np.zeros(H_bathy.shape, dtype=np.float64)
        qsr_full[:, :90] = np.asarray(rgb["qsr"], dtype=np.float64)
        physics_fn = pipeline_shortwave_deposit(card, config, state)
        tendencies = physics_fn(
            state._replace(eta=state.eta.replace(data=jnp.asarray(eta))),
            card.recipe.grid, z_coord,
            OceanSurfaceForcing(sw_down=jnp.asarray(qsr_full),
                                chl=jnp.asarray(chl_full)),
        )
        deposit = np.asarray(tendencies.dT_dt.data)[:, :90, :ACTIVE_Z]
        pipeline_candidate = np.asarray(nemo_source_round(
            nemo_source_round(jnp.asarray(before_qsr) + jnp.asarray(deposit))
            - jnp.asarray(before_qsr))).copy()
        if control == "pipeline-plant":
            index = tuple(np.argwhere(wet[:, :90])[0])
            pipeline_candidate[index] = np.nextafter(
                pipeline_candidate[index], np.inf)
        pipeline = score(pipeline_candidate, qsr["increment"], wet[:, :90])
        if control == "pipeline-plant":
            require(pipeline["status"] != "AT_BAR",
                    "the pipeline-row plant did not fire; that row is vacuous")
            raise GateError(
                "control pipeline-plant rejected through the scorer "
                f"({pipeline['unequal']}/{pipeline['count']} unequal)")
        require(pipeline["status"] == "AT_BAR",
                "the SHARED PHYSICS PIPELINE's own shortwave deposit is not "
                f"NEMO's: {pipeline['unequal']}/{pipeline['count']} unequal, "
                f"max |delta| {pipeline['max_abs']:.6e}")

    if plant or control:
        require(result["unequal"] > 0,
                f"control {control or 'plant'!r} did not fire; it proves nothing")
        raise GateError(
            f"control {control or 'plant'} rejected through the scorer "
            f"({result['unequal']}/{result['count']} unequal, "
            f"max |delta| {result['max_abs']:.6e})")
    return {
        "status": "PASS_ROUTING_MEASUREMENT_COMPLETE",
        "boundary": "O2-RGB-ROUTING/tra_qsr",
        "owner": "ORCA2_OWNER",
        "label": "given NEMO's entry",
        "result": result,
        "shared_physics_pipeline_result": pipeline,
        "card_resolved_settings": {
            field: {"value": getattr(
                config.physics.shortwave_penetration, field),
                "record_provenance": provenance}
            for field, (_, provenance) in RESOLVED.items()
        },
        "card_reference_ladder_unequal": ladder_rows,
        "operand_expressions": {
            "dz_live": "h_partial*(1+r3t) == compute_layer_thickness  (traqsr.f90:388)",
            "gdepw_bottom_live": "-z_half_ref[1:]*(1+r3t)  (traqsr.f90:349)",
            "dispatcher": "apply_shortwave_penetration (the production entry)",
        },
        "record_sha256": {
            name: sha256(root / name) for name in (
                "oracle_rgb_chl_kt00000001.bin",
                "oracle_qsr_stage3_kt00000001.bin",
                "oracle_rktracer_stage3_kt00000001.bin",
            )
        },
        "execution": {
            "backend": jax.default_backend(),
            "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
            "comparison_domain": "rank0 owned wet T cells, 90x148x30",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--control",
                        choices=("two-band", "static", "surface-chl",
                                 "pipeline-plant"))
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.oracle_root,
                          plant=args.plant, control=args.control)
    except (GateError, ValueError, IndexError) as exc:
        print(f"FAIL: {exc}")
        return 2
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
