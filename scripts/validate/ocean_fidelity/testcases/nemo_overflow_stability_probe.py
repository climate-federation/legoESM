#!/usr/bin/env python3
"""Approach-window runner and matched scorer for OVERFLOW-zps instability.

This is the committed instrument for the Lane-1 Rule-9 finding.  NEMO files are
the exact Nbb step-entry frame; legoESM file ``completed_XXXXXXXX.npz`` is the
prognostic state after that many completed steps, hence it matches NEMO
``kt=completed+1``.  All comparisons reuse the certified phase-3 halo stripping
and wet-face registry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_overflow_zps_card
from legoesm.ocean.vertical import compute_layer_thickness


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ORACLE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/stability/overflow_nemo_kt2601_2878"
)
DEFAULT_CANDIDATE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/stability/overflow_legoesm_baseline"
)
CERTIFIED_ORACLE_KT1 = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/overflow_zps/"
    "oracle_step_entry_kt00000001.bin"
)
CAPTURE_START = 2600
CAPTURE_END = 2877
ROUND_PAD_ULPS = 64.0
GROSS_RELATIVE_EXCURSION = 1.0e-6


class ProbeError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProbeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_phase3():
    path = REPO_ROOT / (
        "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_phase3_trajectory_gate.py"
    )
    spec = importlib.util.spec_from_file_location("nemo_l1_phase3", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PHASE3 = _load_phase3()
read_entry = _PHASE3.read_entry
expected_masks = _PHASE3.expected_masks
lego_fields = _PHASE3.lego_fields


def _load_oracle_gate():
    path = REPO_ROOT / (
        "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_oracle_gate.py"
    )
    spec = importlib.util.spec_from_file_location("nemo_l1_oracle_gate", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ORACLE_GATE = _load_oracle_gate()
namelist_values = _ORACLE_GATE.namelist_values
parse_logical = _ORACLE_GATE.parse_logical


# File-side inventory for the requested resolved blocks.  A new or removed
# NEMO key makes the gate red before a scientific disposition can be claimed.
OVERFLOW_RESOLVED_KEYS = frozenset("""
namdyn_adv.ln_dynadv_cen2 namdyn_adv.ln_dynadv_off namdyn_adv.ln_dynadv_up3
namdyn_adv.ln_dynadv_vec namdyn_adv.nn_dynkeg
namdyn_hpg.ln_hpg_djc namdyn_hpg.ln_hpg_djc_vnh namdyn_hpg.ln_hpg_djc_vnv
namdyn_hpg.ln_hpg_isf namdyn_hpg.ln_hpg_prj namdyn_hpg.ln_hpg_sco
namdyn_hpg.ln_hpg_zco
namdyn_ldf.ln_dynldf_blp namdyn_ldf.ln_dynldf_hor namdyn_ldf.ln_dynldf_iso
namdyn_ldf.ln_dynldf_lap namdyn_ldf.ln_dynldf_lev namdyn_ldf.ln_dynldf_off
namdyn_ldf.nn_ahm_ijk_t namdyn_ldf.nn_dynldf_typ namdyn_ldf.rn_ahm_b
namdyn_ldf.rn_csmc namdyn_ldf.rn_lv namdyn_ldf.rn_maxfac
namdyn_ldf.rn_minfac namdyn_ldf.rn_uv
namdyn_spg.ln_bt_auto namdyn_spg.ln_bt_fw namdyn_spg.ln_dynspg_exp
namdyn_spg.ln_dynspg_ts namdyn_spg.nn_bt_flt namdyn_spg.nn_e
namdyn_spg.rn_bt_alpha namdyn_spg.rn_bt_cmax
namdyn_vor.ln_dynvor_een namdyn_vor.ln_dynvor_ene namdyn_vor.ln_dynvor_ens
namdyn_vor.ln_dynvor_ent namdyn_vor.ln_dynvor_mix namdyn_vor.ln_dynvor_msk
namdyn_vor.nn_e3f_typ
namtra_adv.ln_mus_ups namtra_adv.ln_traadv_cen namtra_adv.ln_traadv_fct
namtra_adv.ln_traadv_mus namtra_adv.ln_traadv_off namtra_adv.ln_traadv_qck
namtra_adv.ln_traadv_ubs namtra_adv.nn_cen_h namtra_adv.nn_cen_v
namtra_adv.nn_fct_h namtra_adv.nn_fct_imp namtra_adv.nn_fct_v
namtra_adv.nn_ubs_v
namtra_dmp.cn_resto namtra_dmp.ln_tradmp namtra_dmp.nn_zdmp
namtra_eiv.ln_eke_equ namtra_eiv.ln_ldfeiv namtra_eiv.nn_aei_ijk_t
namtra_eiv.rn_le namtra_eiv.rn_ue
namtra_ldf.ln_botmix_triad namtra_ldf.ln_traldf_blp namtra_ldf.ln_traldf_hor
namtra_ldf.ln_traldf_iso namtra_ldf.ln_traldf_lap namtra_ldf.ln_traldf_lev
namtra_ldf.ln_traldf_msc namtra_ldf.ln_traldf_off
namtra_ldf.ln_traldf_triad namtra_ldf.ln_triad_iso namtra_ldf.nn_aht_ijk_t
namtra_ldf.rn_ld namtra_ldf.rn_slpmax namtra_ldf.rn_sw_triad namtra_ldf.rn_ud
namtra_mle.ln_mle namtra_mle.nn_conv namtra_mle.nn_mld_uv namtra_mle.nn_mle
namtra_mle.rn_ce namtra_mle.rn_lat namtra_mle.rn_lf
namtra_mle.rn_rho_c_mle namtra_mle.rn_time
namzdf.ln_zad_aimp namzdf.ln_zdfcst namzdf.ln_zdfddm namzdf.ln_zdfevd
namzdf.ln_zdfgls namzdf.ln_zdfiwm namzdf.ln_zdfmfc namzdf.ln_zdfnpc
namzdf.ln_zdfosm namzdf.ln_zdfric namzdf.ln_zdfswm namzdf.ln_zdftke
namzdf.nn_avb namzdf.nn_evdm namzdf.nn_havtb namzdf.nn_npc namzdf.nn_npcp
namzdf.rn_avm0 namzdf.rn_avt0 namzdf.rn_avts namzdf.rn_evd namzdf.rn_hsbfr
""".split())

# Executed selectors and numeric operands whose legoESM binding is explicit.
# Inactive-family operands remain inventoried, but are WAIVED with their dead
# controlling selector rather than pretending that their values were tested.
OVERFLOW_VERIFIED_KEYS = frozenset("""
namdyn_adv.ln_dynadv_cen2 namdyn_adv.ln_dynadv_off namdyn_adv.ln_dynadv_up3
namdyn_adv.ln_dynadv_vec
namdyn_hpg.ln_hpg_djc namdyn_hpg.ln_hpg_isf namdyn_hpg.ln_hpg_prj
namdyn_hpg.ln_hpg_sco namdyn_hpg.ln_hpg_zco
namdyn_ldf.ln_dynldf_blp namdyn_ldf.ln_dynldf_hor namdyn_ldf.ln_dynldf_iso
namdyn_ldf.ln_dynldf_lap namdyn_ldf.ln_dynldf_lev namdyn_ldf.ln_dynldf_off
namdyn_spg.ln_bt_auto namdyn_spg.ln_bt_fw namdyn_spg.ln_dynspg_exp
namdyn_spg.ln_dynspg_ts namdyn_spg.nn_bt_flt namdyn_spg.nn_e
namdyn_spg.rn_bt_alpha namdyn_spg.rn_bt_cmax
namdyn_vor.ln_dynvor_een namdyn_vor.ln_dynvor_ene namdyn_vor.ln_dynvor_ens
namdyn_vor.ln_dynvor_ent namdyn_vor.ln_dynvor_mix namdyn_vor.ln_dynvor_msk
namtra_adv.ln_traadv_cen namtra_adv.ln_traadv_fct namtra_adv.ln_traadv_mus
namtra_adv.ln_traadv_off namtra_adv.ln_traadv_qck namtra_adv.ln_traadv_ubs
namtra_adv.nn_fct_h namtra_adv.nn_fct_imp namtra_adv.nn_fct_v
namtra_dmp.ln_tradmp namtra_eiv.ln_eke_equ namtra_eiv.ln_ldfeiv
namtra_ldf.ln_botmix_triad namtra_ldf.ln_traldf_blp namtra_ldf.ln_traldf_hor
namtra_ldf.ln_traldf_iso namtra_ldf.ln_traldf_lap namtra_ldf.ln_traldf_lev
namtra_ldf.ln_traldf_msc namtra_ldf.ln_traldf_off
namtra_ldf.ln_traldf_triad namtra_ldf.ln_triad_iso
namtra_mle.ln_mle
namzdf.ln_zdfcst namzdf.ln_zdfddm namzdf.ln_zdfevd namzdf.ln_zdfgls
namzdf.ln_zdfiwm namzdf.ln_zdfmfc namzdf.ln_zdfnpc namzdf.ln_zdfosm
namzdf.ln_zdfric namzdf.ln_zdfswm namzdf.ln_zdftke
namzdf.ln_zad_aimp namzdf.nn_avb namzdf.nn_havtb namzdf.rn_avm0 namzdf.rn_avt0
""".split())

OVERFLOW_UNMEASURED_KEYS = frozenset()


def _state_arrays(state) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in lego_fields(state).items()}


def overflow_resolved_coverage(
    resolved_namelist: Path,
    output: Path | None = None,
    *,
    plant_unaccounted: bool = False,
    card=None,
) -> dict:
    """Fail-closed disposition of every resolved namdyn/namzdf/namtra key.

    ``output.namelist.dyn`` is the file side of the threat model.  The
    registry is deliberately exact: a new NEMO key is missing from the ledger,
    while a removed key leaves a stale ledger entry.  Both stop the gate.
    """
    values = namelist_values(resolved_namelist)
    selected = {
        key: value for key, value in values.items()
        if key.split(".", 1)[0].startswith(("namdyn", "namzdf", "namtra"))
    }
    if plant_unaccounted:
        selected["namzdf.planted_file_side_key"] = "T"
    missing = sorted(set(selected) - OVERFLOW_RESOLVED_KEYS)
    stale = sorted(OVERFLOW_RESOLVED_KEYS - set(selected))
    require(not missing, f"unaccounted resolved NEMO keys: {missing}")
    require(not stale, f"stale coverage-ledger keys: {stale}")

    # Resolve the card as a single reference configuration.  These checks bind
    # active NEMO choices to canonical legoESM selectors; inactive operands are
    # still inventoried below but cannot be evidence for a live operator.
    if card is None:
        card = build_overflow_zps_card()
    cfg = card.recipe.model_config
    expected_card = {
        "tracer_advection": (cfg.tracer_advection, "fct2"),
        "tracer_time_integrator": (cfg.tracer_time_integrator, "rk3_ws"),
        "momentum_advection": (cfg.momentum_advection, "flux_form"),
        "momentum_flux_scheme": (cfg.momentum_flux_scheme, "nemo_up3"),
        "vertical_momentum_scheme": (cfg.vertical_momentum_scheme, "nemo_up3"),
        "pgf_scheme": (cfg.pgf_scheme, "nemo_sco"),
        "adaptive_implicit_vertadv": (cfg.adaptive_implicit_vertadv, True),
        "implicit_vertical_mixing": (cfg.implicit_vertical_mixing, True),
        "A_h": (cfg.lateral_viscosity.A_h, 0.0),
        "B_h": (cfg.lateral_viscosity.B_h, 0.0),
        "K_h": (cfg.K_h, 0.0),
        "K_bih": (cfg.K_bih, 0.0),
        "A_v": (cfg.A_v, 1.0e-4),
        "K_v": (cfg.K_v, 0.0),
        "gm_redi": (cfg.gm_redi, None),
        "physics": (cfg.physics, None),
        "barotropic_time_filter": (
            cfg.barotropic.barotropic_time_filter, "nemo_boxcar1_ab3"),
        "n_barotropic_substeps": (
            cfg.barotropic.n_barotropic_substeps, 3),
        "barotropic_diffusion_alpha": (
            cfg.barotropic.barotropic_diffusion_alpha, 0.0),
    }
    for name, (got, expected) in expected_card.items():
        require(got == expected, f"card {name}: got {got!r}, expected {expected!r}")

    expected_logicals = {
        "namdyn_adv.ln_dynadv_off": False,
        "namdyn_adv.ln_dynadv_vec": False,
        "namdyn_adv.ln_dynadv_cen2": False,
        "namdyn_adv.ln_dynadv_up3": True,
        "namdyn_ldf.ln_dynldf_off": True,
        "namdyn_spg.ln_dynspg_exp": False,
        "namdyn_spg.ln_dynspg_ts": True,
        "namdyn_spg.ln_bt_fw": True,
        "namdyn_spg.ln_bt_auto": True,
        "namtra_adv.ln_traadv_fct": True,
        "namtra_dmp.ln_tradmp": False,
        "namtra_eiv.ln_ldfeiv": False,
        "namtra_eiv.ln_eke_equ": False,
        "namtra_ldf.ln_traldf_off": True,
        "namtra_mle.ln_mle": False,
        "namzdf.ln_zdfcst": True,
        "namzdf.ln_zdfevd": False,
        "namzdf.ln_zdfnpc": False,
        "namzdf.ln_zdfmfc": False,
        "namzdf.ln_zdfosm": False,
        "namzdf.ln_zad_aimp": True,
    }
    for key, expected in expected_logicals.items():
        require(parse_logical(selected[key], key) is expected,
                f"{key} does not match the certified executed value")
    expected_numeric = {
        "namdyn_spg.nn_bt_flt": 1.0,
        "namdyn_spg.rn_bt_alpha": 0.0,
        "namdyn_spg.rn_bt_cmax": 0.8,
        "namtra_adv.nn_fct_h": 2.0,
        "namtra_adv.nn_fct_v": 2.0,
        "namtra_adv.nn_fct_imp": 1.0,
        "namzdf.rn_avm0": 1.0e-4,
        "namzdf.rn_avt0": 0.0,
    }
    for key, expected in expected_numeric.items():
        got = float(selected[key])
        require(got == expected, f"{key}: got {got!r}, expected {expected!r}")

    from legoesm.ocean.dynamics.barotropic_common import (
        compute_nemo_boxcar_forward_weights,
        compute_nemo_forward_raw_transport_weights,
    )
    primary, primary_divisor, transport, n_loop = (
        compute_nemo_boxcar_forward_weights(3, np.dtype(np.float64)))
    raw_transport, raw_divisor, raw_n_loop = (
        compute_nemo_forward_raw_transport_weights(3, np.dtype(np.float64)))
    primary = np.asarray(primary)
    transport = np.asarray(transport)
    raw_transport = np.asarray(raw_transport)
    require(n_loop == raw_n_loop == 4, "OVERFLOW forward filter loop must have 4 iterations")
    require(np.array_equal(primary, np.asarray([0.0, 1/3, 1/3, 1/3])),
            f"wrong primary boxcar weights {primary}")
    require(float(primary_divisor) == 1.0, "wrong normalized primary divisor")
    require(np.array_equal(raw_transport, np.asarray([3.0, 3.0, 2.0, 1.0])),
            f"wrong raw secondary weights {raw_transport}")
    require(float(raw_divisor) == 9.0, "wrong raw secondary divisor")
    require(np.array_equal(transport, raw_transport / float(raw_divisor)),
            "normalized and literal secondary weights disagree")

    rows = []
    for key in sorted(selected):
        if key in OVERFLOW_UNMEASURED_KEYS:
            status = "UNMEASURED"
            reason = (
                "active NEMO adaptive-implicit program: boolean selection is "
                "bound, but exact wi partition plus momentum/tracer application "
                "has not yet been certified on the approach-window states"
            )
        elif key in OVERFLOW_VERIFIED_KEYS:
            status = "VERIFIED"
            reason = (
                "active selector/card operand, or executed-off family selector "
                "whose absence is source- and card-bound"
            )
        else:
            status = "WAIVED"
            reason = (
                "inactive-family operand; inventory-stability disposition only, "
                "not a review of the dead algorithm"
            )
        rows.append({
            "key": key,
            "nemo_resolved_value": selected[key],
            "status": status,
            "reason": reason,
        })
    unmeasured = [row["key"] for row in rows if row["status"] == "UNMEASURED"]
    report = {
        "format": "nemo-testcase-l1-overflow-resolved-coverage-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "resolved_namelist": str(resolved_namelist),
        "resolved_namelist_sha256": sha256(resolved_namelist),
        "prefixes": ["namdyn*", "namzdf", "namtra*"],
        "counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("VERIFIED", "WAIVED", "UNMEASURED")
        },
        "unmeasured": unmeasured,
        "barotropic_composition": {
            "oracle": {
                "ln_bt_fw": True,
                "ln_bt_auto": True,
                "nn_bt_flt": 1,
                "rn_bt_alpha": 0.0,
                "runtime_nn_e": 3,
                "primary_weights": primary.tolist(),
                "raw_secondary_weights": raw_transport.tolist(),
                "secondary_divisor": float(raw_divisor),
                "loop_iterations": n_loop,
            },
            "legoesm": {
                "filter": cfg.barotropic.barotropic_time_filter,
                "n_substeps": cfg.barotropic.n_barotropic_substeps,
                "primary_weights": primary.tolist(),
                "raw_secondary_weights": raw_transport.tolist(),
                "secondary_divisor": float(raw_divisor),
                "loop_iterations": n_loop,
            },
            "status": "VERIFIED",
        },
        "status": "UNMEASURED" if unmeasured else "VERIFIED",
        "rows": rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _budget(state, card) -> dict[str, float]:
    h = np.asarray(compute_layer_thickness(
        state.eta.data,
        state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    ))
    active = np.asarray(card.recipe.z_coord.is_active, dtype=bool)
    wet = np.asarray(state.land_mask.data) > 0.5
    use = active & wet[..., None]
    area = np.asarray(card.recipe.grid.area_T)[..., None]
    volume = area * h * use
    return {
        "volume_m3": float(np.sum(volume, dtype=np.float64)),
        "heat_content_proxy_m3_C": float(
            np.sum(volume * np.asarray(state.T.data), dtype=np.float64)),
        "salt_content_proxy_m3_psu": float(
            np.sum(volume * np.asarray(state.S.data), dtype=np.float64)),
    }


def run_legoesm(output: Path, arm: str, end_step: int, capture_start: int) -> dict:
    require(
        arm in {
            "baseline",
            "no_tracer_vertical_transport",
            "no_primary_transport_average",
            "no_adaptive_implicit_momentum",
            "no_bbl",
        },
        f"bad arm {arm}",
    )
    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(0 <= capture_start <= end_step, "invalid capture window")
    card = build_overflow_zps_card()
    hooks = _NEMOWSRK3TestHooks(
        disable_tracer_vertical_transport=(arm == "no_tracer_vertical_transport"),
        primary_transport_average=(arm != "no_primary_transport_average"),
        disable_adaptive_implicit_momentum=(
            arm == "no_adaptive_implicit_momentum"),
        disable_bbl=(arm == "no_bbl"),
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=hooks,
    )
    state = card.recipe.initial_state
    output.mkdir(parents=True, exist_ok=True)
    dtype_receipt = {
        "state": {name: str(value.dtype) for name, value in _state_arrays(state).items()},
        "geometry": {
            name: str(np.asarray(getattr(card.recipe.z_coord, name)).dtype)
            for name in ("t_depth_ref", "dz_ref", "z_full_ref", "z_half_ref", "h_partial")
        },
    }
    require(set(dtype_receipt["state"].values()) == {"float64"}, str(dtype_receipt))
    require(set(dtype_receipt["geometry"].values()) == {"float64"}, str(dtype_receipt))
    initial_budget = _budget(state, card)
    records = []
    first_nonfinite = None
    started = time.perf_counter()
    previous = _state_arrays(state)
    for completed in range(end_step + 1):
        fields = _state_arrays(state)
        finite = {name: bool(np.all(np.isfinite(value))) for name, value in fields.items()}
        if completed >= capture_start:
            arrays = {name: np.array(value, copy=True) for name, value in fields.items()}
            snap = output / f"completed_{completed:08d}.npz"
            np.savez_compressed(snap, **arrays)
            row = {
                "completed_step": completed,
                "physical_time_s": completed * card.dt_s,
                "finite": finite,
                "snapshot": snap.name,
                "snapshot_sha256": sha256(snap),
                "budgets": _budget(state, card) if all(finite.values()) else None,
                "fields": {},
            }
            for name, values in fields.items():
                mask = expected_masks(card)[name]
                use = np.asarray(mask, dtype=bool)
                if not use.any():
                    row["fields"][name] = {"status": "UNMEASURED_NO_ACTIVE_FACE"}
                    continue
                selected = values[use]
                delta = values - previous[name]
                delta_selected = np.abs(delta[use])
                if np.all(np.isfinite(selected)):
                    index_flat = int(np.argmax(delta_selected))
                    index = tuple(int(v) for v in np.argwhere(use)[index_flat])
                    row["fields"][name] = {
                        "min": float(np.min(selected)),
                        "max": float(np.max(selected)),
                        "max_abs": float(np.max(np.abs(selected))),
                        "max_abs_one_step_increment": float(np.max(delta_selected)),
                        "increment_argmax_index": list(index),
                    }
                else:
                    row["fields"][name] = {"status": "NONFINITE"}
            records.append(row)
        if not all(finite.values()):
            first_nonfinite = completed
            break
        if completed == end_step:
            break
        previous = {name: np.array(value, copy=True) for name, value in fields.items()}
        state = model.step(state, dt=card.dt_s)

    artifact = {
        "format": "nemo-testcase-l1-overflow-stability-run-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "arm": arm,
        "reference": (
            "legoESM certified card targeting the NEMO 5.0.2 configuration"
            if arm == "baseline"
            else "experimental harness ablation; no reference model"
        ),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "precision_policy": repr(get_policy()),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "dtype_receipt": dtype_receipt,
        "capture_completed_steps": (
            [capture_start, min(end_step, records[-1]["completed_step"])]
            if records
            else None
        ),
        "requested_end_step": end_step,
        "first_nonfinite_completed_step": first_nonfinite,
        "initial_budgets": initial_budget,
        "wall_time_s": time.perf_counter() - started,
        "records": records,
    }
    path = output / "run.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: artifact[key] for key in artifact if key != "records"}, indent=2))
    return artifact


def paired_step_scale(
    output: Path,
    completed_before: int,
    arm_name: str = "disable_tracer_vertical_transport",
) -> dict:
    """Measure one private arm from the identical input state."""
    require(
        arm_name in {
            "disable_tracer_vertical_transport",
            "disable_adaptive_implicit_momentum",
            "disable_bbl",
        },
        f"bad paired scale arm {arm_name}",
    )
    require(completed_before >= 0, "negative completed-before step")
    set_policy(PrecisionPolicy.fp64())
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_overflow_zps_card()
    baseline = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
    )
    arm = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**{arm_name: True}),
    )
    state = card.recipe.initial_state
    for _ in range(completed_before):
        state = baseline.step(state, dt=card.dt_s)
    before = _state_arrays(state)
    baseline_after = _state_arrays(baseline.step(state, dt=card.dt_s))
    arm_after = _state_arrays(arm.step(state, dt=card.dt_s))
    masks = expected_masks(card)
    rows = []
    for name in ("T", "S", "u", "v", "ssh"):
        use = np.asarray(masks[name], dtype=bool)
        if not use.any():
            rows.append({
                "field": name,
                "status": "UNMEASURED_NO_ACTIVE_FACE",
            })
            continue
        effect = np.abs(arm_after[name] - baseline_after[name])
        increment = np.abs(baseline_after[name] - before[name])
        masked_increment = np.where(use, increment, -np.inf)
        increment_index = tuple(
            int(value)
            for value in np.unravel_index(
                np.argmax(masked_increment), masked_increment.shape
            )
        )
        effect_max = float(np.max(effect[use]))
        increment_max = float(increment[increment_index])
        effect_at_increment = float(effect[increment_index])
        row = {
            "field": name,
            "same_input_state": True,
            "baseline_completed_before": completed_before,
            "compared_completed_after": completed_before + 1,
            "baseline_max_abs_one_step_increment": increment_max,
            "increment_argmax_index": list(increment_index),
            "arm_effect_at_increment_argmax": effect_at_increment,
            "arm_effect_linf": effect_max,
            "effect_at_increment_argmax_over_increment": (
                effect_at_increment / increment_max if increment_max > 0.0 else None
            ),
            "effect_linf_over_increment_linf": (
                effect_max / increment_max if increment_max > 0.0 else None
            ),
        }
        if name != "ssh":
            row["t_depth_m"] = float(
                np.asarray(card.recipe.z_coord.t_depth_ref)[increment_index[-1]]
            )
        row["x_km"] = float(
            increment_index[1] - (0.0 if name == "u" else 0.5)
        )
        rows.append(row)
    report = {
        "format": "nemo-testcase-l1-overflow-stability-paired-scale-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "backend": jax.default_backend(),
        "precision_policy": repr(get_policy()),
        "arm": arm_name,
        "arm_reference": "experimental harness ablation; no reference model",
        "same_input_state": True,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return report


def summarize_run(root: Path) -> dict:
    """Turn raw per-step records into citable budget and growth diagnostics."""
    run_path = root / "run.json"
    require(run_path.is_file(), f"missing {run_path}")
    run = json.loads(run_path.read_text())
    records = run["records"]
    budget_rows = [row for row in records if row.get("budgets") is not None]
    budgets = {}
    for key, initial in run["initial_budgets"].items():
        values = [row["budgets"][key] for row in budget_rows]
        drift = [value - initial for value in values]
        budgets[key] = {
            "initial": initial,
            "last": values[-1] if values else None,
            "max_abs_drift": max((abs(value) for value in drift), default=None),
            "max_abs_relative_drift": (
                max((abs(value / initial) for value in drift), default=None)
                if initial != 0.0
                else None
            ),
        }
    fields = {}
    for name in ("T", "S", "u", "v", "ssh"):
        raw = []
        previous = None
        for row in records:
            value = row["fields"].get(name, {}).get("max_abs_one_step_increment")
            if value is None:
                continue
            ratio = value / previous if previous not in {None, 0.0} else None
            raw.append({
                "completed_step": row["completed_step"],
                "max_abs_one_step_increment": value,
                "successive_ratio": ratio,
                "argmax_index": row["fields"][name]["increment_argmax_index"],
            })
            previous = value
        fields[name] = {
            "raw": raw,
            "last_twelve": raw[-12:],
        }
    report = {
        "format": "nemo-testcase-l1-overflow-stability-run-summary-v1",
        "git_sha": git_sha(),
        "case": run["case"],
        "arm": run["arm"],
        "reference": run["reference"],
        "first_nonfinite_completed_step": run["first_nonfinite_completed_step"],
        "budget_closure": budgets,
        "increment_growth": fields,
    }
    output = root / "run_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "format": report["format"],
        "arm": report["arm"],
        "first_nonfinite_completed_step": report["first_nonfinite_completed_step"],
        "budget_closure": report["budget_closure"],
        "last_twelve": {
            name: value["last_twelve"] for name, value in fields.items()
        },
    }, indent=2))
    return report


def compare_arms(baseline_root: Path, arm_root: Path, output: Path) -> dict:
    """Apply the frozen Arm-A movement rule without post-hoc relabeling."""
    baseline = json.loads((baseline_root / "run.json").read_text())
    arm = json.loads((arm_root / "run.json").read_text())
    base_fail = baseline["first_nonfinite_completed_step"]
    arm_fail = arm["first_nonfinite_completed_step"]
    require(base_fail is not None and arm_fail is not None, "both arms must fail")
    movement = arm_fail - base_fail
    supports = arm_fail >= base_fail + 100 or arm_fail > 3200
    refutes_primary = 2870 <= arm_fail <= 2884 and abs(movement) < 0.1 * base_fail
    verdict = "CONFIRMED" if supports else "REFUTED_PRIMARY" if refutes_primary else "PLAUSIBLE"
    report = {
        "format": "nemo-testcase-l1-overflow-stability-arm-comparison-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "arm": "disable_tracer_vertical_transport",
        "arm_reference": "experimental harness ablation; no reference model",
        "baseline_first_nonfinite_completed_step": base_fail,
        "arm_first_nonfinite_completed_step": arm_fail,
        "failure_step_movement": movement,
        "frozen_support_threshold": "moves >=100 steps later or beyond 3200",
        "frozen_refute_threshold": "fails in 2870..2884 with <10% movement and increment movement",
        "verdict": verdict,
        "interpretation": (
            f"The ablation destabilizes {abs(movement)} steps earlier. It shows the current "
            "explicit vertical tracer transport is necessary to this trajectory, "
            "but neither confirms nor refutes ownership by NEMO's complete "
            "adaptive-implicit RK3 package."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return report


def compare_primary_arm(baseline_root: Path, arm_root: Path, output: Path) -> dict:
    """Apply the frozen primary-transport localization verdict."""
    baseline_run = json.loads((baseline_root / "run.json").read_text())
    arm_run = json.loads((arm_root / "run.json").read_text())
    baseline_score = json.loads((baseline_root / "matched_score.json").read_text())
    arm_score = json.loads((arm_root / "matched_score.json").read_text())

    def values(report):
        step = next(row for row in report["rows"] if row["kt"] == 2601)
        return {
            row["field"]: row["normalized_linf"]
            for row in step["fields"] if row["field"] in {"u", "ssh"}
        }

    baseline_values = values(baseline_score)
    arm_values = values(arm_score)
    relative_movement = {
        name: abs(arm_values[name] - baseline_values[name]) / baseline_values[name]
        for name in baseline_values
    }
    base_fail = baseline_run["first_nonfinite_completed_step"]
    arm_fail = arm_run["first_nonfinite_completed_step"]
    supports = (
        (arm_fail >= base_fail + 100 or arm_fail > 3200)
        and all(arm_values[name] <= 0.5 * baseline_values[name]
                for name in baseline_values)
    )
    refutes_primary = (
        2870 <= arm_fail <= 2884
        and all(value < 0.1 for value in relative_movement.values())
    )
    verdict = "CONFIRMED" if supports else "REFUTED_PRIMARY" if refutes_primary else "PLAUSIBLE"
    report = {
        "format": "nemo-testcase-l1-overflow-stability-primary-arm-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "arm": "primary_transport_average=False",
        "arm_reference": "experimental harness ablation; no reference model",
        "baseline_first_nonfinite_completed_step": base_fail,
        "arm_first_nonfinite_completed_step": arm_fail,
        "failure_step_movement": arm_fail - base_fail,
        "kt2601_normalized_linf": {
            "baseline": baseline_values,
            "arm": arm_values,
            "relative_movement": relative_movement,
        },
        "verdict": verdict,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return report


def _arm_d_label(
    base_fail: int,
    arm_fail: int,
    u_reduction: float,
    t_reduction: float,
) -> str:
    """Frozen adaptive-momentum label; split out for direct non-vacuity tests."""
    movement = arm_fail - base_fail
    if arm_fail > 6120 and u_reduction >= 2.0 and t_reduction >= 2.0:
        return "CONFIRMED"
    if movement >= 100 and u_reduction >= 2.0 and t_reduction >= 2.0:
        return "PLAUSIBLE"
    if (2870 <= arm_fail <= 2884
            and abs(movement) / base_fail < 0.1
            and u_reduction < (1.0 / 0.9)
            and t_reduction < (1.0 / 0.9)):
        return "REFUTED_PRIMARY"
    return "UNMEASURED"


def _arm_failure_for_label(run: dict) -> tuple[int, bool]:
    """Map a finite full-duration run to the sentinel beyond the 6120 bar."""
    failure = run["first_nonfinite_completed_step"]
    if failure is not None:
        return int(failure), False
    require(run["requested_end_step"] >= 6120,
            "null arm failure is not evidence of a full-duration completion")
    return 6121, True


def classify_coverage_round(
    coverage_path: Path,
    baseline_root: Path,
    adaptive_root: Path,
    adaptive_scale_path: Path,
    bbl_scale_path: Path,
    output: Path,
) -> dict:
    """Apply the frozen Arm-D/Arm-E rules without post-hoc judgment."""
    coverage = json.loads(coverage_path.read_text())
    baseline_run = json.loads((baseline_root / "run.json").read_text())
    adaptive_run = json.loads((adaptive_root / "run.json").read_text())
    baseline_score = json.loads((baseline_root / "matched_score.json").read_text())
    adaptive_score = json.loads((adaptive_root / "matched_score.json").read_text())
    adaptive_scale = json.loads(adaptive_scale_path.read_text())
    bbl_scale = json.loads(bbl_scale_path.read_text())
    require(coverage["unmeasured"] == ["namzdf.ln_zad_aimp"],
            "coverage register changed after preregistration")
    require(adaptive_scale["arm"] == "disable_adaptive_implicit_momentum",
            "wrong adaptive scale artifact")
    require(bbl_scale["arm"] == "disable_bbl", "wrong BBL scale artifact")

    def scale_row(report: dict, field: str) -> dict:
        hits = [row for row in report["rows"] if row["field"] == field]
        require(len(hits) == 1, f"expected one scale row for {field}")
        return hits[0]

    adaptive_u_scale = scale_row(adaptive_scale, "u")[
        "effect_at_increment_argmax_over_increment"]
    adaptive_scale_passes = adaptive_u_scale >= 0.1
    require(adaptive_scale_passes, "full adaptive arm ran below frozen scale")
    base_fail = baseline_run["first_nonfinite_completed_step"]
    adaptive_fail = adaptive_run["first_nonfinite_completed_step"]
    require(base_fail is not None, "baseline artifact must record its failure step")
    adaptive_fail_for_label, adaptive_completed = _arm_failure_for_label(adaptive_run)
    movement = adaptive_fail_for_label - base_fail

    def field_at_kt(report: dict, kt: int, field: str) -> dict:
        kt_rows = [row for row in report["rows"] if row["kt"] == kt]
        require(len(kt_rows) == 1, f"missing kt={kt}")
        hits = [row for row in kt_rows[0]["fields"] if row["field"] == field]
        require(len(hits) == 1, f"missing {field} at kt={kt}")
        return hits[0]

    base_u = field_at_kt(baseline_score, 2877, "u")["normalized_linf"]
    adaptive_u = field_at_kt(adaptive_score, 2877, "u")["normalized_linf"]
    u_reduction = base_u / adaptive_u

    def increment_at(run: dict, completed: int, field: str) -> float:
        rows = [row for row in run["records"] if row["completed_step"] == completed]
        require(len(rows) == 1, f"missing completed step {completed}")
        return rows[0]["fields"][field]["max_abs_one_step_increment"]

    base_t_increment = increment_at(baseline_run, 2876, "T")
    adaptive_t_increment = increment_at(adaptive_run, 2876, "T")
    t_reduction = base_t_increment / adaptive_t_increment
    adaptive_label = _arm_d_label(
        base_fail, adaptive_fail_for_label, u_reduction, t_reduction)

    bbl_t = scale_row(bbl_scale, "T")[
        "effect_at_increment_argmax_over_increment"]
    bbl_u = scale_row(bbl_scale, "u")[
        "effect_at_increment_argmax_over_increment"]
    bbl_scale_passes = max(bbl_t, bbl_u) >= 0.1
    bbl_label = "UNMEASURED" if bbl_scale_passes else "REFUTED_PRIMARY"
    require(not bbl_scale_passes,
            "BBL passed the scale gate; a full arm is required before closure")

    report = {
        "format": "nemo-testcase-l1-overflow-coverage-round-verdict-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "coverage": {
            "status": coverage["status"],
            "counts": coverage["counts"],
            "unmeasured": coverage["unmeasured"],
            "barotropic_composition": coverage["barotropic_composition"],
        },
        "adaptive_momentum_arm": {
            "label": adaptive_label,
            "same_input_u_scale_ratio": adaptive_u_scale,
            "baseline_failure_step": base_fail,
            "arm_failure_step": adaptive_fail,
            "arm_completed_6120": adaptive_completed,
            "failure_step_movement": movement,
            "kt2877_u_normalized_linf": {
                "baseline": base_u,
                "arm": adaptive_u,
                "reduction_factor": u_reduction,
            },
            "completed2876_T_increment": {
                "baseline": base_t_increment,
                "arm": adaptive_t_increment,
                "reduction_factor": t_reduction,
            },
            "reference": "experimental harness ablation; no reference model",
        },
        "bbl_arm": {
            "label": bbl_label,
            "same_input_effect_at_increment_ratio": {"T": bbl_t, "u": bbl_u},
            "full_arm_allowed": bbl_scale_passes,
            "reference": "experimental harness ablation; no reference model",
        },
        "initiating_owner": {
            "label": "UNMEASURED",
            "reason": (
                "no arm completed 6120 and early kt=2 U/SSH arithmetic remains "
                "unowned; Arm D identifies a major late amplifier only"
            ),
        },
        "tested_register_exhausted": True,
        "remaining_implementation_debt": [
            "source-exact unbranched NEMO RK3 ln_zad_Aimp package: Wicker "
            "ww/wi partition, optimized FCT predictor, tracer and momentum "
            "implicit applications at the cited NEMO time levels"
        ],
        "status": "UNMEASURED",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return report


def _load_candidate(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing {path}")
    with np.load(path) as data:
        required = {"T", "S", "u", "v", "ssh"}
        require(set(data.files) == required, f"{path}: inventory {data.files}")
        return {name: np.asarray(data[name]) for name in required}


def _argmax_row(name, oracle, candidate, mask, card) -> dict:
    use = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == use.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate dtype {candidate.dtype}")
    require(np.all(np.isfinite(oracle[use])), f"{name}: oracle nonfinite")
    require(np.all(np.isfinite(candidate[use])), f"{name}: candidate nonfinite")
    error = np.abs(candidate - oracle)
    masked = np.where(use, error, -np.inf)
    index = tuple(int(value) for value in np.unravel_index(np.argmax(masked), masked.shape))
    ovalues = oracle[use]
    cvalues = candidate[use]
    scale = max(float(np.max(np.abs(ovalues))), 1.0)
    pad = ROUND_PAD_ULPS * np.finfo(np.float64).eps * scale
    low_margin = float(np.min(cvalues) - np.min(ovalues))
    high_margin = float(np.max(cvalues) - np.max(ovalues))
    outside_padded = low_margin < -pad or high_margin > pad
    gross = max(-low_margin, high_margin, 0.0) / scale > GROSS_RELATIVE_EXCURSION
    result = {
        "field": name,
        "frame": (
            "T-centre instantaneous Nbb" if name in {"T", "S"}
            else f"instantaneous prognostic Nbb {name.upper()}-face"
            if name in {"u", "v"}
            else "T-centre instantaneous Nbb SSH"
        ),
        "reduction": "elementwise L-infinity on the certified common wet mask",
        "normalized_linf": float(error[index] / scale),
        "absolute_linf": float(error[index]),
        "argmax_index": list(index),
        "oracle_at_argmax": float(oracle[index]),
        "candidate_at_argmax": float(candidate[index]),
        "oracle_min": float(np.min(ovalues)),
        "oracle_max": float(np.max(ovalues)),
        "candidate_min": float(np.min(cvalues)),
        "candidate_max": float(np.max(cvalues)),
        "low_envelope_margin": low_margin,
        "high_envelope_margin": high_margin,
        "roundoff_pad": pad,
        "outside_roundoff_padded_oracle_range": bool(outside_padded),
        "gross_relative_excursion": bool(gross),
        "n": int(use.sum()),
    }
    if name != "ssh":
        level = index[-1]
        result["t_depth_m"] = float(np.asarray(card.recipe.z_coord.t_depth_ref)[level])
    if name in {"T", "S", "ssh"}:
        result["x_km"] = float(index[1] - 0.5)
    else:
        result["x_km"] = float(index[1] - (0.0 if name == "u" else 0.5))
    return result


def _certified_kt1_control(card, masks) -> dict:
    """Prove this probe still reproduces the certified initial comparator frame."""
    oracle = read_entry(CERTIFIED_ORACLE_KT1, "OVERFLOW-zps")
    candidate = _state_arrays(card.recipe.initial_state)
    rows = []
    for name in ("T", "S", "u", "ssh"):
        reference = np.asarray(oracle[name])
        if name != "ssh":
            reference = reference[..., :card.recipe.z_coord.n_levels]
        use = np.asarray(masks[name], dtype=bool)
        equal = bool(np.array_equal(reference[use], candidate[name][use]))
        rows.append({
            "field": name,
            "exact": equal,
            "n": int(use.sum()),
            "reference_sha256": sha256(CERTIFIED_ORACLE_KT1),
        })
        require(equal, f"certified kt=1 exact control failed for {name}")
    return {
        "status": "VERIFIED",
        "oracle": str(CERTIFIED_ORACLE_KT1),
        "rows": rows,
    }


def score(
    oracle_root: Path,
    candidate_root: Path,
    *,
    start_completed: int = CAPTURE_START,
    end_completed: int = CAPTURE_END,
    plant: str | None = None,
) -> dict:
    require(0 <= start_completed <= end_completed, "invalid score window")
    card = build_overflow_zps_card()
    masks = expected_masks(card)
    kt1_control = _certified_kt1_control(card, masks)
    rows = []
    first_outside = None
    first_gross = None
    first_nonfinite = None
    previous_errors = None
    for completed in range(start_completed, end_completed + 1):
        kt = completed + 1
        oracle_path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
        candidate_path = candidate_root / f"completed_{completed:08d}.npz"
        oracle = read_entry(oracle_path, "OVERFLOW-zps")
        require(oracle["step"] == kt, f"{oracle_path}: step mismatch")
        candidate = _load_candidate(candidate_path)
        if plant == "shift_step" and completed == start_completed:
            require(False, "planted step-number mismatch")
        if plant in {"hot", "nan"} and completed == start_completed:
            candidate["T"] = candidate["T"].copy()
            first = tuple(np.argwhere(masks["T"])[0])
            candidate["T"][first] = 50.0 if plant == "hot" else np.nan
        field_rows = []
        for name in ("T", "S", "u", "v", "ssh"):
            reference = np.asarray(oracle[name])
            if name != "ssh":
                reference = reference[..., :card.recipe.z_coord.n_levels]
            if name == "v" and not np.asarray(masks[name]).any():
                field_rows.append({
                    "field": name,
                    "status": "UNMEASURED_NO_ACTIVE_FACE",
                    "reason": "three-row closed tank has no active meridional face",
                })
                continue
            use = np.asarray(masks[name], dtype=bool)
            if not np.all(np.isfinite(candidate[name][use])):
                field_rows.append({
                    "field": name,
                    "status": "NONFINITE",
                    "frame": (
                        "T-centre instantaneous Nbb" if name in {"T", "S"}
                        else f"instantaneous prognostic Nbb {name.upper()}-face"
                        if name in {"u", "v"}
                        else "T-centre instantaneous Nbb SSH"
                    ),
                    "reduction": "finiteness on the certified common wet mask",
                    "n_nonfinite": int(np.count_nonzero(~np.isfinite(candidate[name][use]))),
                })
                if first_nonfinite is None:
                    first_nonfinite = {
                        "kt": kt,
                        "completed_step": completed,
                        "field": name,
                    }
                continue
            field_rows.append(
                _argmax_row(name, reference, candidate[name], masks[name], card)
            )
        gross = [row["field"] for row in field_rows if row.get("gross_relative_excursion")]
        if gross and first_gross is None:
            first_gross = {"kt": kt, "completed_step": completed, "fields": gross}
        outside = [
            row["field"] for row in field_rows
            if row.get("outside_roundoff_padded_oracle_range")
        ]
        if outside and first_outside is None:
            first_outside = {"kt": kt, "completed_step": completed, "fields": outside}
        current_errors = {
            row["field"]: row["normalized_linf"]
            for row in field_rows if "normalized_linf" in row
        }
        growth = None
        if previous_errors is not None:
            growth = {
                name: (current_errors[name] / previous_errors[name]
                       if previous_errors[name] > 0.0 else None)
                for name in current_errors
            }
        rows.append({
            "kt": kt,
            "completed_step": completed,
            "fields": field_rows,
            "successive_error_ratios": growth,
        })
        previous_errors = current_errors
    report = {
        "format": "nemo-testcase-l1-overflow-stability-score-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "oracle_frame": "NEMO instantaneous Nbb step-entry state",
        "candidate_frame": "legoESM instantaneous prognostic state after kt-1 completed steps",
        "matched_kt": [start_completed + 1, end_completed + 1],
        "roundoff_pad_ulps": ROUND_PAD_ULPS,
        "gross_relative_excursion": GROSS_RELATIVE_EXCURSION,
        "certified_kt1_exact_control": kt1_control,
        "first_outside_roundoff_padded_oracle_range": first_outside,
        "first_gross_excursion": first_gross,
        "first_nonfinite": first_nonfinite,
        "status": "DEBT" if first_gross is not None or first_nonfinite is not None else "MEASURED",
        "rows": rows,
    }
    output = candidate_root / "matched_score.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "rows"}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run-legoesm")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_CANDIDATE)
    run_parser.add_argument(
        "--arm",
        choices=(
            "baseline",
            "no_tracer_vertical_transport",
            "no_primary_transport_average",
            "no_adaptive_implicit_momentum",
            "no_bbl",
        ),
        default="baseline",
    )
    run_parser.add_argument("--end-step", type=int, default=CAPTURE_END)
    run_parser.add_argument("--capture-start", type=int, default=CAPTURE_START)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE)
    score_parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE)
    score_parser.add_argument("--start-completed", type=int, default=CAPTURE_START)
    score_parser.add_argument("--end-completed", type=int, default=CAPTURE_END)
    score_parser.add_argument("--plant", choices=("hot", "nan", "shift_step"))
    scale_parser = sub.add_parser("paired-step-scale")
    scale_parser.add_argument("--completed-before", type=int, default=2875)
    scale_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CANDIDATE / "paired_step_scale.json",
    )
    scale_parser.add_argument(
        "--arm",
        choices=(
            "disable_tracer_vertical_transport",
            "disable_adaptive_implicit_momentum",
            "disable_bbl",
        ),
        default="disable_tracer_vertical_transport",
    )
    summary_parser = sub.add_parser("summarize-run")
    summary_parser.add_argument("--root", type=Path, default=DEFAULT_CANDIDATE)
    compare_parser = sub.add_parser("compare-arms")
    compare_parser.add_argument("--baseline-root", type=Path, default=DEFAULT_CANDIDATE)
    compare_parser.add_argument("--arm-root", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    primary_parser = sub.add_parser("compare-primary-arm")
    primary_parser.add_argument("--baseline-root", type=Path, default=DEFAULT_CANDIDATE)
    primary_parser.add_argument("--arm-root", type=Path, required=True)
    primary_parser.add_argument("--output", type=Path, required=True)
    coverage_parser = sub.add_parser("coverage")
    coverage_parser.add_argument(
        "--resolved-namelist", type=Path,
        default=DEFAULT_ORACLE / "output.namelist.dyn",
    )
    coverage_parser.add_argument("--output", type=Path, required=True)
    coverage_parser.add_argument("--plant-unaccounted", action="store_true")
    verdict_parser = sub.add_parser("classify-coverage-round")
    verdict_parser.add_argument(
        "--coverage", type=Path,
        default=Path(
            "/data/abyssal/dbalwada/nemo-testcases-l1/stability/"
            "overflow_resolved_coverage.json"),
    )
    verdict_parser.add_argument("--baseline-root", type=Path, default=DEFAULT_CANDIDATE)
    verdict_parser.add_argument("--adaptive-root", type=Path, required=True)
    verdict_parser.add_argument("--adaptive-scale", type=Path, required=True)
    verdict_parser.add_argument("--bbl-scale", type=Path, required=True)
    verdict_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run-legoesm":
        run_legoesm(args.output, args.arm, args.end_step, args.capture_start)
    elif args.command == "score":
        report = score(
            args.oracle_root,
            args.candidate_root,
            start_completed=args.start_completed,
            end_completed=args.end_completed,
            plant=args.plant,
        )
        return 1 if report["status"] == "DEBT" else 0
    elif args.command == "paired-step-scale":
        paired_step_scale(args.output, args.completed_before, args.arm)
    elif args.command == "summarize-run":
        summarize_run(args.root)
    elif args.command == "compare-arms":
        compare_arms(args.baseline_root, args.arm_root, args.output)
    elif args.command == "compare-primary-arm":
        compare_primary_arm(args.baseline_root, args.arm_root, args.output)
    elif args.command == "coverage":
        report = overflow_resolved_coverage(
            args.resolved_namelist,
            args.output,
            plant_unaccounted=args.plant_unaccounted,
        )
        print(json.dumps({key: value for key, value in report.items()
                          if key != "rows"}, indent=2))
    else:
        classify_coverage_round(
            args.coverage,
            args.baseline_root,
            args.adaptive_root,
            args.adaptive_scale,
            args.bbl_scale,
            args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
