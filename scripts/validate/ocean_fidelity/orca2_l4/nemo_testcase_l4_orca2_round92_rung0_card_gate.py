#!/usr/bin/env python3
"""Build and score the explicit ORCA2 hierarchy rung-0 stage-1 card."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_orca2_initial_ts,
    build_orca2_zps_card,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round84_rung0_frame_gate as frames,
)


FIELDS = ("T", "S", "u", "v", "ssh")
PLANTS = ("none", "card-module", "mixing-coeff", "frame-layout", "entry-bit")


class GateError(RuntimeError):
    """The candidate is not the admitted Decision-80 rung-0 identity."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def build_rung0_card(deck_root: Path):
    """Private measurement card; no package selector lands before admission."""

    import jax.numpy as jnp

    shipped = build_orca2_zps_card(deck_root)
    model = shipped.recipe.model_config
    vertical = model.physics.vertical_mixing
    vertical = vertical._replace(
        scheme="constant",
        constant=vertical.constant._replace(
            A_v=1.2e-4, K_v=1.2e-5, lat_dependent=False),
        iwm=vertical.iwm._replace(enabled=False, require_forcing_maps=False),
        ddm=vertical.ddm._replace(enabled=False),
        vmix_background_mode="additive",
        # Decision 130: the shipped card's V1 TKE statements stay off on a
        # card that does not run TKE (stated, not inherited).
        tke=vertical.tke._replace(
            nemo_bn2_source_rounded=False,
            nemo_sh2_step_entry_ssh=False,
            nemo_tke_solve_source_rounded=False,
            nemo_sh2_fold_top_v=False,
            nemo_tke_matrix_rhs_source_rounded=False,
            nemo_sh2_r3_source_rounded=False,
            nemo_sh2_r3_stored_reciprocal=False,
        ),
    )
    physics = model.physics._replace(
        vertical_mixing=vertical,
        # Round 309: the step-entry EVD trigger needs the step-entry bn2
        # bundle, which a constant-mixing card does not build (refuses).
        convection=model.physics.convection._replace(
            enhanced_diffusion=model.physics.convection.enhanced_diffusion._replace(
                trigger_step_entry_bn2=False)),
        shortwave_penetration=None,
        mle=None,
    )
    model = model._replace(
        A_v=0.0,
        K_v=0.0,
        gm_redi=None,
        physics=physics,
        bbl_adv_option=0,
        bbl_gamma_s=0.0,
        bbl_diffusive_option=0,
        bbl_aht_m2_s=0.0,
        runoff_depth_spread_m=0.0,
        runoff_depth_spread_map=None,
        # Decision 133: the shipped card's ln_lin implicit bundle and G7's
        # bottom statements are inherited unchanged (zdfdrg.f90:276).
        # G4's stage-3 heat/Kmm unit is certified on the forced ORCA2 path.
        # Rung 0 has no heat/runoff/shortwave module, so B90 requires these
        # forcing-path selectors to stay off here and on OMT-1..4.
        rk3_stage3_heat_nemo_written=False,
        rk3_stage3_tracer_nemo_stored_thickness=False,
        # G5's stage-3 salt statement reads the recorded sfx (round 304).
        rk3_stage3_salt_nemo_written=False,
    )
    root = Path(deck_root)
    temperature, salinity = build_orca2_initial_ts(
        root / "data_1m_potential_temperature_nomask.nc",
        root / "data_1m_salinity_nomask.nc",
        np.asarray(shipped.recipe.z_coord.is_active, dtype=bool),
        # The hierarchy rung-0 deck resolves ln_tsd_dmp=.false.; compiled
        # dtatsd.f90:218-254 therefore skips the ORCA_R2 alteration block.
        apply_hand_alterations=False,
    )
    state = shipped.recipe.initial_state._replace(
        T=shipped.recipe.initial_state.T.replace(
            data=jnp.asarray(temperature, dtype=jnp.float64)),
        S=shipped.recipe.initial_state.S.replace(
            data=jnp.asarray(salinity, dtype=jnp.float64)),
        eta=shipped.recipe.initial_state.eta.replace(
            data=jnp.zeros_like(shipped.recipe.initial_state.eta.data)),
        tke=None,
        tke_avm=None,
        tke_avt=None,
        tke_avm_surface=None,
        tke_dissl=None,
        dtke=None,
    )
    recipe = shipped.recipe._replace(
        model_config=model,
        physics_config=physics,
        initial_state=state,
        iwm_forcing=None,
    )
    # This is deliberately gate-local until the internal record admits.  The
    # shipped card and its six SI3 selectors remain untouched.
    return shipped._replace(
        case="ORCA2-rung0-zps",
        recipe=recipe,
        n_steps=240,
        bbl_adv_option=0,
        bbl_diffusive_option=0,
        bbl_aht_m2_s=0.0,
        bbl_gamma_s=0.0,
        surface_boundary_condition="nemo_flx_zero",
        # This explicit value distinguishes rung 0's resolved
        # ln_tsd_dmp=.false. input path from the shipped card's live damping
        # arm; it is card metadata, not a production model selector.
        surface_input_operator="nemo_fld_read_no_tsd_dmp",
        # NEMO's linear implicit drag is selected and measured (D133, round
        # 305); without ice it is D122-measured, not ladder-certified.
        unmeasured_features=(),
        icebergs_enabled=False,
        iceberg_inputs=(),
    )


def validate_rung0_card(card, *, execution: bool = False) -> None:
    """Fail closed on every rung-0 hierarchy selector used by this probe."""

    if execution and card.unmeasured_features:
        raise ValueError(
            "ORCA2-rung0-zps is not execution-ready; unresolved selected "
            "arms: " + ", ".join(card.unmeasured_features))
    cfg = card.recipe.model_config
    vmix = cfg.physics.vertical_mixing
    selectors = (
        card.case == "ORCA2-rung0-zps",
        cfg.gm_redi is None,
        cfg.physics.mle is None,
        cfg.physics.shortwave_penetration is None,
        vmix.scheme == "constant",
        vmix.constant.A_v == 1.2e-4,
        vmix.constant.K_v == 1.2e-5,
        vmix.constant.lat_dependent is False,
        vmix.iwm.enabled is False,
        vmix.ddm.enabled is False,
        (cfg.A_v, cfg.K_v) == (0.0, 0.0),
        cfg.physics.convection.scheme == "enhanced_diffusion",
        (cfg.bbl_adv_option, cfg.bbl_diffusive_option) == (0, 0),
        cfg.runoff_depth_spread_m == 0.0,
        cfg.runoff_depth_spread_map is None,
        card.surface_boundary_condition == "nemo_flx_zero",
        card.surface_input_operator == "nemo_fld_read_no_tsd_dmp",
        card.unmeasured_features == (),
        cfg.bottom_drag.bottom_drag_scheme == "nemo_linear",
        (cfg.bottom_drag.bottom_drag_cd0, cfg.bottom_drag.bottom_drag_uc0)
        == (1.0e-3, 0.4),
        (cfg.bottom_drag.bottom_drag_r, cfg.bottom_drag.bottom_drag_bbl_thickness,
         cfg.bottom_drag.bottom_drag_bg_velocity) == (0.0, 0.0, 0.0),
        cfg.zdf_drag_in_matrix,
        cfg.barotropic_drag_substep,
        cfg.nemo_bottom_drag_written_statements,
        not cfg.implicit_ice_ocean_top_drag,
    )
    require(all(selectors), "rung-0 card module exclusion or selector moved")
    require(all(getattr(card.recipe.initial_state, name) is None
                for name in ("tke", "tke_avm", "tke_avt",
                             "tke_avm_surface", "tke_dissl", "dtke")),
            "rung-0 card carries inactive TKE state")
    require(not np.any(np.asarray(card.recipe.initial_state.eta.data)),
            "rung-0 independent no-ice SSH is not zero")


def _payload(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    record = frames.read_frame(path)
    raw = path.read_bytes()
    values = {
        name: np.frombuffer(
            raw, dtype="=f8", count=field["count"],
            offset=record["offsets"][name],
        ).copy()
        for name, field in record["fields"].items()
    }
    return record["header"], values


def assemble_frame(
    root: Path, kt: int, stage: int, *, plant_layout: bool = False,
) -> dict[str, np.ndarray]:
    """Assemble both rank-owned slabs using their self-described layouts."""

    assembled = {
        name: np.empty((148, 180) if name == "ssh" else (148, 180, 30),
                       dtype=np.float64)
        for name in FIELDS
    }
    coverage = np.zeros((148, 180), dtype=np.int8)
    for rank in (0, 1):
        path = root / (
            f"oracle_r84_frame_rank{rank:04d}_kt{kt:08d}_s{stage}.bin")
        header, payload = _payload(path)
        require(header["kt"] == kt and header["stage"] == stage
                and header["rank"] == rank, f"{path.name}: identity mismatch")
        nx, ny, nz = header["shape"]
        nimpp, njmpp = header["origin"]
        ntsi, ntsj, ntei, ntej = header["owned"]
        require((nx, ny, nz) == (94, 152, 31),
                f"{path.name}: unexpected local shape")
        require((ntsi, ntsj, ntei, ntej) == (3, 3, 92, 150),
                f"{path.name}: owned bounds moved")
        i0 = nimpp + ntsi - 4
        j0 = njmpp + ntsj - 4
        if plant_layout and rank == 1:
            i0 -= 1
        i1 = i0 + ntei - ntsi + 1
        j1 = j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: global placement {(j0, j1, i0, i1)}")
        coverage[j0:j1, i0:i1] += 1
        for name in FIELDS:
            if name == "ssh":
                local = payload[name].reshape((nx, ny), order="F")
                block = local[ntsi - 1:ntei, ntsj - 1:ntej].T
            else:
                local = payload[name].reshape((nx, ny, nz), order="F")
                block = local[ntsi - 1:ntei, ntsj - 1:ntej, :30].transpose(1, 0, 2)
            assembled[name][j0:j1, i0:i1, ...] = block
    require(bool(np.all(coverage == 1)),
            "rank-owned frame slabs do not cover the domain exactly once")
    return assembled


def candidate_fields(state) -> dict[str, np.ndarray]:
    return {
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        "u": np.asarray(state.u.data)[:, 1:],
        "v": np.asarray(state.v.data)[1:],
        "ssh": np.asarray(state.eta.data),
    }


def bridge_entry(card, entry: dict[str, np.ndarray], *, plant_bit: bool = False):
    """Replace exactly the five recorded prognostic fields."""

    import jax.numpy as jnp

    state = card.recipe.initial_state
    u = jnp.zeros_like(state.u.data).at[:, 1:, :].set(jnp.asarray(entry["u"]))
    v = jnp.zeros_like(state.v.data).at[1:, :, :].set(jnp.asarray(entry["v"]))
    temperature = jnp.asarray(entry["T"])
    if plant_bit:
        temperature = temperature.at[0, 0, 0].set(
            jnp.nextafter(temperature[0, 0, 0], jnp.asarray(np.inf)))
    state = state._replace(
        T=state.T.replace(data=temperature),
        S=state.S.replace(data=jnp.asarray(entry["S"])),
        u=state.u.replace(data=u),
        v=state.v.replace(data=v),
        eta=state.eta.replace(data=jnp.asarray(entry["ssh"])),
    )
    # Compiled istate.f90:147-162 builds uu_b/vv_b from the recorded zero
    # velocity and copies Kbb to Kmm; the exact result is numerical zero.
    require(bool(np.all(np.asarray(state.uu_b.data) == 0.0)
                 and np.all(np.asarray(state.vv_b.data) == 0.0)),
            "unrecorded barotropic entry carry is not source-proved zero")
    return state


def _card_census(card) -> dict[str, object]:
    cfg = card.recipe.model_config
    vmix = cfg.physics.vertical_mixing
    return {
        "case": card.case,
        "gm_redi": cfg.gm_redi is not None,
        "mle": cfg.physics.mle is not None,
        "shortwave": cfg.physics.shortwave_penetration is not None,
        "vertical_scheme": vmix.scheme,
        "constant_A_v": vmix.constant.A_v,
        "constant_K_v": vmix.constant.K_v,
        "iwm": vmix.iwm.enabled,
        "ddm": vmix.ddm.enabled,
        "enhanced_diffusion": cfg.physics.convection.scheme,
        "bbl_adv_option": cfg.bbl_adv_option,
        "bbl_diffusive_option": cfg.bbl_diffusive_option,
        "runoff_depth_spread_m": cfg.runoff_depth_spread_m,
        "surface_boundary_condition": card.surface_boundary_condition,
        "unmeasured_features": list(card.unmeasured_features),
    }


def run(deck_root: Path, record_root: Path, plant: str = "none") -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    require(plant in PLANTS, f"unknown plant {plant}")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "rung-0 replay requires production JIT on CPU")

    shipped = build_orca2_zps_card(deck_root)
    card = build_rung0_card(deck_root)
    if plant == "card-module":
        card = card._replace(recipe=card.recipe._replace(
            model_config=card.recipe.model_config._replace(
                gm_redi=shipped.recipe.model_config.gm_redi)))
    elif plant == "mixing-coeff":
        cfg = card.recipe.model_config
        vmix = cfg.physics.vertical_mixing
        physics = cfg.physics._replace(vertical_mixing=vmix._replace(
            constant=vmix.constant._replace(K_v=np.nextafter(1.2e-5, np.inf))))
        card = card._replace(recipe=card.recipe._replace(
            model_config=cfg._replace(physics=physics),
            physics_config=physics))
    validate_rung0_card(card)
    require(shipped.unmeasured_features[-1] == "si3_jpl5_layered_prather_state",
            "shipped ORCA2 sea-ice debt tuple moved")
    validate_rung0_card(card, execution=True)

    entry = assemble_frame(
        record_root, 1, 0, plant_layout=plant == "frame-layout")
    state = bridge_entry(card, entry, plant_bit=plant == "entry-bit")
    entry_rows = ladder.compare_fields(candidate_fields(state), entry)
    require(entry_rows["first_non_bit_field"] is None,
            f"stage-0 bridge differs in {entry_rows['first_non_bit_field']}")

    shape = entry["ssh"].shape
    zero = jnp.zeros(shape, dtype=jnp.float64)
    freshwater = FreshwaterForcing(zero, zero, zero, zero, zero)
    surface = OceanSurfaceForcing(
        sw_down=zero, q_net=zero, tau_x=zero, tau_y=zero,
        freshwater=zero, salt_flux=zero, taum=zero,
        tau_i_native=zero, tau_j_native=zero,
    )
    momentum_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_momentum_stage=1),
    )
    tracer_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_tracer_stage=1),
    )
    momentum_stage = jax.device_get(momentum_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    tracer_stage = jax.device_get(tracer_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    candidate_stage1 = {
        "T": np.asarray(tracer_stage.T.data),
        "S": np.asarray(tracer_stage.S.data),
        "u": np.asarray(momentum_stage.u.data)[:, 1:],
        "v": np.asarray(momentum_stage.v.data)[1:],
        "ssh": np.asarray(tracer_stage.eta.data),
    }
    oracle_stage1 = assemble_frame(record_root, 1, 1)
    stage1_rows = ladder.compare_fields(candidate_stage1, oracle_stage1)
    require(stage1_rows["first_non_bit_field"] is not None,
            "R92-P4 REFUTED: stage 1 stayed bit-exact; continue to stage 2")
    return {
        "status": "PASS_RUNG0_CARD_STAGE1",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "card_census": _card_census(card),
        "shipped_card_unmeasured_features": list(shipped.unmeasured_features),
        "stage0_entry": entry_rows,
        "stage1": stage1_rows,
        "first_non_bit_boundary": {
            "kt": 1,
            "boundary": "after-complete-stage-1",
            "field": stage1_rows["first_non_bit_field"],
            "statement": "UNMEASURED_STAGE1_INTERNAL_BOUNDARY",
            "reason": (
                "the admitted rank-complete record brackets the whole stage; "
                "its inherited internal streams are not rank-complete and "
                "were not admitted for the rung-0 hierarchy claim"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.deck_root, args.record_root, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, OSError, ValueError, IndexError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_RUNG0_CARD_STAGE1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
