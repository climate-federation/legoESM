#!/usr/bin/env python3
"""SWG kt=1..10 ladder: restart after-states now, stage entries when acquired."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nemo_testcase_phase3_trajectory_gate import BAR, require, score  # noqa: E402

DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "round0_probe/kt10")
DEFAULT_STAGE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "round1/oracle_swg_r1/run")
RESTART_FIELDS = (
    "sshn", "un", "vn", "uu_n", "vv_n", "utau_b", "vtau_b",
    "emp_b", "qns_b", "sfx_b",
)
# Compiled sbc_oce.f90:116-117 defines the before stresses at U/V, and
# sbcmod.f90:567-568 reads the restart records with cd_type='U'/'V'.
U_FIELDS = frozenset(("un", "uu_n", "utau_b"))
V_FIELDS = frozenset(("vn", "vv_n", "vtau_b"))
STAGE_U_FIELDS = frozenset((
    "u_k", "r3u_k", "u_before", "utau_before", "utau_now"))
STAGE_V_FIELDS = frozenset((
    "v_k", "r3v_k", "v_before", "vtau_before", "vtau_now"))
STAGE_F_FIELDS = frozenset(("r3f",))


def _stage_checker():
    path = HERE / "nemo_testcase_l1_swg/check_records.py"
    spec = importlib.util.spec_from_file_location("swg_check_records", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def read_restarts(root: Path) -> dict[int, dict[str, np.ndarray]]:
    """Read the ten restart AFTER states, refusing omissions or hidden casts."""
    import netCDF4
    files = sorted(root.glob("SWG_????????_restart.nc"))
    expected = [root / f"SWG_{kt:08d}_restart.nc" for kt in range(1, 11)]
    require(files == expected, f"restart sequence is {[p.name for p in files]}")
    out = {}
    for kt, path in enumerate(files, 1):
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            file_kt = int(np.asarray(ds["kt"][:]).item())
            require(file_kt == kt, f"{path.name}: kt={file_kt}")
            require(float(np.asarray(ds["rdt"][:]).item()) == 3600.0,
                    f"{path.name}: rdt is not 3600 s")
            row = {}
            for name in RESTART_FIELDS:
                require(name in ds.variables, f"{path.name}: missing {name}")
                var = ds[name]
                require(var.dtype == np.dtype("float64"),
                        f"{path.name}.{name}: dtype {var.dtype}")
                a = np.asarray(var[0], dtype=np.float64)
                if name in ("un", "vn"):
                    require(a.shape[0] == 2, f"{path.name}.{name}: no dummy record")
                    a = a[0]                 # executed level; crop dummy bottom
                require(a.shape == (84, 84) and np.all(np.isfinite(a)),
                        f"{path.name}.{name}: bad shape/non-finite")
                row[name] = a
            out[kt] = row
    return out


def read_stage_entries(root: Path) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Read all acquired stage entries through the admission parser."""
    cr = _stage_checker()
    out = {}
    for kt in range(1, 11):
        for stage in (1, 2, 3):
            path = root / f"oracle_swgstage_kt{kt:08d}_s{stage}.bin"
            require(path.is_file(), f"missing {path}")
            head, groups = cr.parse(path)
            require((head[1], head[2]) == (kt, stage),
                    f"{path.name}: header {(head[1], head[2])}")
            is0, js0, ie0, je0 = head[cr.BOUNDS_AT:cr.BOUNDS_AT + 4]
            out[kt, stage] = {
                name: np.asarray(a[js0 - 1:je0, is0 - 1:ie0])
                for name, a in groups.items()
            }
    return out


def _card_masks(card) -> dict[str, np.ndarray]:
    """Return native NEMO T/U/V/F masks from shared card operands."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks

    tmask = np.asarray(card.recipe.land_mask, dtype=bool)
    u_full, v_full = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    fmask = np.asarray(
        card.recipe.z_coord.nemo_een_barotropic.fe3mask[..., 0], dtype=bool)
    return {
        "T": tmask,
        "U": np.asarray(u_full[:, 1:], dtype=bool),
        "V": np.asarray(v_full[1:, :], dtype=bool),
        "F": fmask,
    }


def _field_mask(field: str, masks: dict[str, np.ndarray]) -> np.ndarray:
    return masks[
        "U" if field in U_FIELDS | STAGE_U_FIELDS
        else "V" if field in V_FIELDS | STAGE_V_FIELDS
        else "F" if field in STAGE_F_FIELDS else "T"]


def _row(name, ref, cand, mask):
    ref, cand = np.asarray(ref), np.asarray(cand)
    use = np.asarray(mask, dtype=bool)
    result = score(name, ref, cand, use)
    delta = cand - ref
    bad = np.argwhere((ref != cand) & use)
    result.update({
        "bit_identical": bool(bad.size == 0),
        "n_unequal": int(bad.shape[0]),
        "first_unequal_cell": bad[0].tolist() if bad.size else None,
        "max_abs": float(np.max(np.abs(delta[use]))),
    })
    # This campaign's requested endpoint is bit-for-bit.  Retain the TSUNAMI
    # 1e-15 numerical score, but never let a non-bit row claim AT-BAR.
    if bad.size and result["status"] == "AT-BAR":
        result["numerical_status"] = "AT-BAR"
        result["status"] = "DEBT"
        result["reason"] = "bit-for-bit SWG ladder: non-identical value"
    return result


def _gate_rows(out: dict) -> dict:
    rows = [row for step in out.get("per_kt", ()) for row in step["rows"]]
    controls = list(out.get("controls", ()))
    gated = rows + controls
    out["all_at_bar"] = bool(gated) and all(
        row["status"] == "AT-BAR" for row in gated)
    out["all_bit_identical"] = bool(gated) and all(
        row["bit_identical"] for row in gated)
    out["first_kt_over_bar"] = next(
        (step["kt"] for step in out.get("per_kt", ())
         if any(row["status"] == "DEBT" for row in step["rows"])), None)
    return out


def _exit_code(out: dict) -> int:
    return 0 if out.get("all_at_bar", True) else 1


def _stage3_input_audit(entries, masks) -> dict:
    """Audit the recorded kt=2 stage-3 wind and whole-step inputs."""
    stage3 = entries[2, 3]
    step_entry = entries[2, 1]

    def compare(left, right, mask):
        use = np.asarray(mask, dtype=bool)
        delta = np.asarray(left) - np.asarray(right)
        unequal = (np.asarray(left) != np.asarray(right)) & use
        return {
            "bit_identical": bool(not np.any(unequal)),
            "n_unequal": int(np.count_nonzero(unequal)),
            "max_abs": float(np.max(np.abs(delta[use]))),
        }

    return {
        "wind_before_vs_now": {
            "u": compare(stage3["utau_before"], stage3["utau_now"], masks["U"]),
            "v": compare(stage3["vtau_before"], stage3["vtau_now"], masks["V"]),
        },
        "whole_step_entry_vs_stage3_entry": {
            "u_k": compare(step_entry["u_k"], stage3["u_k"], masks["U"]),
            "v_k": compare(step_entry["v_k"], stage3["v_k"], masks["V"]),
            "r3f": compare(step_entry["r3f"], stage3["r3f"], masks["F"]),
        },
    }


def self_check(root: Path, *, plant: bool = False, allow_dirty: bool = False) -> dict:
    """Calibrate restart decoding only; this is not a fidelity measurement."""
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha
    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card
    masks = _card_masks(build_swg_card())
    records = read_restarts(root)
    per_kt = []
    for kt, fields in records.items():
        rows = []
        for name, ref in fields.items():
            cand = ref.copy()
            if plant and kt == 1 and name == "sshn":
                cand[42, 42] += 1.0e-3
            rows.append(_row(f"self_check.kt{kt}.{name}", ref, cand,
                             _field_mask(name, masks)))
        per_kt.append({"kt": kt, "rows": rows})
    return _gate_rows({
        "label": "RESTART-READER-SELF-CHECK (not fidelity)",
        "legoesm_git_sha": sha, "bar": BAR, "plant": plant,
        "per_kt": per_kt,
    })


def independent(root: Path, *, allow_dirty: bool = False) -> dict:
    """Run the independently initialized, production ``step()`` ladder."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_NAMELIST, build_swg_card, swg_horizontal_coordinates,
        validate_nemo_testcase_card_for_execution)
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha
    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "ladder must run on CPU")
    card = build_swg_card()
    masks = _card_masks(card)
    validate_nemo_testcase_card_for_execution(card)
    records = read_restarts(root)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config)
    src = swg_horizontal_coordinates()
    # Compiled usrdef_sbc.f90:99-106, with the shipped theta=0 deck.  This is
    # generated from the card, never read from an oracle stage/restart record.
    tau_u = (-np.float64(SWG_NAMELIST.rn_tau_pa)
             * np.cos(np.float64(np.pi) * src["gphit"]
                      / np.float64(SWG_NAMELIST.rn_domsiz_m)))
    tau_v = np.zeros_like(tau_u)
    forcing = OceanSurfaceForcing(
        tau_i_native=jax.numpy.asarray(tau_u, dtype=jax.numpy.float64),
        tau_j_native=jax.numpy.asarray(tau_v, dtype=jax.numpy.float64))
    state = card.recipe.initial_state
    per_kt = []
    for kt in range(1, 11):
        state = model.step(state, card.dt_s, surface_forcing=forcing)
        cand = {
            "sshn": np.asarray(state.eta.data),
            "un": np.asarray(state.u.data)[:, 1:, 0],
            "vn": np.asarray(state.v.data)[1:, :, 0],
            "uu_n": np.asarray(state.uu_b.data)[:, 1:],
            "vv_n": np.asarray(state.vv_b.data)[1:, :],
        }
        rows = [_row(f"independent.kt{kt}.{n}", records[kt][n], cand[n],
                     _field_mask(n, masks))
                for n in cand]
        per_kt.append({"kt": kt, "rows": rows})
    return _gate_rows({"label": "INDEPENDENT", "legoesm_git_sha": sha,
                       "bar": BAR,
                       "per_kt": per_kt})


def carried_state_audit(stage_root: Path, *, allow_dirty: bool = False) -> dict:
    """Compare the fifth production step and rebuilt sixth-step boundary."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.dynamics.swe_rk3 import _entry_from_state
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_NAMELIST, build_swg_card, swg_horizontal_coordinates,
        validate_nemo_testcase_card_for_execution)
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha
    from legoesm.ocean.state import OceanSurfaceForcing

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "carry audit must run on CPU")
    card = build_swg_card()
    validate_nemo_testcase_card_for_execution(card)
    masks = _card_masks(card)
    entries = read_stage_entries(stage_root)
    target = entries[6, 1]
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config)
    src = swg_horizontal_coordinates()
    tau_u = (-np.float64(SWG_NAMELIST.rn_tau_pa)
             * np.cos(np.float64(np.pi) * src["gphit"]
                      / np.float64(SWG_NAMELIST.rn_domsiz_m)))
    tau_v = np.zeros_like(tau_u)
    forcing = OceanSurfaceForcing(
        tau_i_native=jax.numpy.asarray(tau_u, dtype=jax.numpy.float64),
        tau_j_native=jax.numpy.asarray(tau_v, dtype=jax.numpy.float64))

    initial = card.recipe.initial_state
    state = initial
    state4 = None
    for kt in range(1, 6):
        if kt == 5:
            state4 = state
        state = model.step(state, card.dt_s, surface_forcing=forcing)
    require(state4 is not None, "failed to retain the fourth-step state")

    now = (forcing.tau_i_native, forcing.tau_j_native)
    before4 = (
        now[0] if state4.tau_x_prev is None else state4.tau_x_prev,
        now[1] if state4.tau_y_prev is None else state4.tau_y_prev,
    )
    boundary5 = _entry_from_state(
        state4, before4, now, card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config)
    s1 = model.replay_swe_rk3_stage(boundary5, card.dt_s, 1, boundary5)
    s2 = model.replay_swe_rk3_stage(s1, card.dt_s, 2, boundary5)
    s3 = model.replay_swe_rk3_stage(s2, card.dt_s, 3, boundary5)

    before6 = (state.tau_x_prev, state.tau_y_prev)
    rebuilt6 = _entry_from_state(
        state, before6, now, card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config)
    stage_rows = [
        _row(f"carry.stage3_to_kt6.{name}", target[name], value,
             _field_mask(name, masks))
        for name, value in s3._asdict().items()
    ]
    rebuilt_rows = [
        _row(f"carry.rebuilt_kt6.{name}", target[name], value,
             _field_mask(name, masks))
        for name, value in rebuilt6._asdict().items()
    ]
    # The carried boundary is now fully specified.  Execute the next ordinary
    # production step as one compiled whole-step graph and compare its public
    # result with the admitted next entry.  The separately compiled stage
    # chain remains the source-order control for the same transition.
    state6 = model.step(state, card.dt_s, surface_forcing=forcing)
    target7 = entries[7, 1]
    whole_step6_rows = [
        _row("carry.whole_step6.ssh", target7["ssh_k"], state6.eta.data,
             masks["T"]),
        _row("carry.whole_step6.u", target7["u_k"], state6.u.data[:, 1:, 0],
             masks["U"]),
        _row("carry.whole_step6.v", target7["v_k"], state6.v.data[1:, :, 0],
             masks["V"]),
    ]

    public_rows = [
        _row("carry.public.ssh", target["ssh_k"], state.eta.data, masks["T"]),
        _row("carry.public.u", target["u_k"], state.u.data[:, 1:, 0], masks["U"]),
        _row("carry.public.v", target["v_k"], state.v.data[1:, :, 0], masks["V"]),
        _row("carry.public.uu_b", np.zeros_like(target["u_k"]),
             state.uu_b.data[:, 1:], masks["U"]),
        _row("carry.public.vv_b", np.zeros_like(target["v_k"]),
             state.vv_b.data[1:, :], masks["V"]),
    ]
    unchanged_fields = ("T", "S", "H_bathy", "land_mask", "u_mask",
                        "v_mask", "w", "uu_b", "vv_b")
    unchanged = {}
    for name in unchanged_fields:
        after = np.asarray(getattr(getattr(state, name), "data"))
        start = np.asarray(getattr(getattr(initial, name), "data"))
        unchanged[name] = {
            "bit_identical": bool(np.array_equal(after, start)),
            "n_unequal": int(np.count_nonzero(after != start)),
        }
    require(all(row["bit_identical"] for row in public_rows),
            "step-five public state differs before the rebuilt boundary")
    require(all(item["bit_identical"] for item in unchanged.values()),
            "an inert/static SWG state leaf changed")
    optional_none = [
        name for name, value in state._asdict().items()
        if value is None
    ]
    accounted = set(unchanged_fields) | {
        "u", "v", "eta", "tau_x_prev", "tau_y_prev"
    } | set(optional_none)
    require(accounted == set(state._fields),
            f"unaccounted SWG state leaves: {sorted(set(state._fields)-accounted)}")
    redundant = {
        "u_west": {
            "shape": list(np.asarray(state.u.data[:, :1, :]).shape),
            "n_nonzero": int(np.count_nonzero(np.asarray(state.u.data[:, :1, :]))),
        },
        "v_south": {
            "shape": list(np.asarray(state.v.data[:1, :, :]).shape),
            "n_nonzero": int(np.count_nonzero(np.asarray(state.v.data[:1, :, :]))),
        },
    }
    require(redundant["u_west"]["n_nonzero"] == 0
            and redundant["v_south"]["n_nonzero"] == 0,
            "walled redundant face storage is nonzero")
    return _gate_rows({
        "label": "CARRIED-STATE-AUDIT", "legoesm_git_sha": sha,
        "bar": BAR,
        "state_disposition": {
            "scored_public": [row["name"] for row in public_rows],
            "unchanged": unchanged,
            "optional_none": optional_none,
            "redundant_faces": redundant,
        },
        "per_kt": [{"kt": 6,
                    "rows": stage_rows + rebuilt_rows + whole_step6_rows}],
    })


def swe_rk3_stages(stage_root: Path, *, mode: str = "given",
                   plant: str | None = None,
                   allow_dirty: bool = False) -> dict:
    """Drive the JIT production stages on non-degenerate given/chained rows."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.dynamics.swe_rk3 import SWERK3StageEntry

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "ladder must run on CPU")
    card = build_swg_card()
    vor = card.recipe.model_config.vorticity_scheme
    require(vor.scheme == "ens_total",
            "SWG card does not select ens_total")
    require(vor.ln_dynvor_msk is True,
            "SWG card does not explicitly select ln_dynvor_msk=true")
    require(mode in ("given", "chained"),
            f"unknown stage replay mode {mode!r}")
    entries = read_stage_entries(stage_root)
    masks = _card_masks(card)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    def item(fields):
        return SWERK3StageEntry(**{
            name: jax.numpy.asarray(fields[name], dtype=jax.numpy.float64)
            for name in SWERK3StageEntry._fields
        })

    def replay(fields, before_fields, stage):
        result = model.replay_swe_rk3_stage(
            item(fields), card.dt_s, stage, item(before_fields))
        return {name: np.asarray(value) for name, value in result._asdict().items()}

    identity = all(np.array_equal(entries[1, 1][name], entries[1, 2][name])
                   for name in SWERK3StageEntry._fields)
    require(identity, "kt=1 stage-1 boundary is no longer an identity")
    nondiscriminating = {
        "kt": 1, "stage": 1, "status": "NON-DISCRIMINATING",
        "reason": "oracle entry and target are identical on all 13 fields",
    }

    per_kt = []
    if mode == "given":
        for kt in range(2, 11):
            before = entries[kt, 1]
            for stage in (1, 2, 3):
                if stage == 3 and kt == 10:
                    continue
                target = entries[kt, stage + 1] if stage < 3 else entries[kt + 1, 1]
                candidate = replay(entries[kt, stage], before, stage)
                require(all(value.dtype == np.dtype("float64")
                            for value in candidate.values()),
                        "production stage output is not fp64")
                rows = [
                    _row(f"given.kt{kt}.s{stage}.{name}", target[name], value,
                         _field_mask(name, masks))
                    for name, value in candidate.items()
                ]
                per_kt.append({"kt": kt, "stage": stage, "rows": rows})
    else:
        current = {name: value.copy() for name, value in entries[2, 1].items()}
        for kt in range(2, 11):
            before = {name: value.copy() for name, value in current.items()}
            for stage in (1, 2, 3):
                if stage == 3 and kt == 10:
                    continue
                candidate = replay(current, before, stage)
                target = entries[kt, stage + 1] if stage < 3 else entries[kt + 1, 1]
                rows = [
                    _row(f"chained.kt{kt}.s{stage}.{name}", target[name], value,
                         _field_mask(name, masks))
                    for name, value in candidate.items()
                ]
                per_kt.append({"kt": kt, "stage": stage, "rows": rows})
                current = candidate
    controls = []
    if plant is not None:
        require(plant == "ulp",
                f"unknown stage-entry plant {plant!r}")
        # One genuine ULP of an admitted, nonzero, wet value.  The whole-step
        # before entry stays byte-identical, so exactly one current-stage input
        # differs before the JIT production stage consumes it.
        current = {name: value.copy() for name, value in entries[2, 1].items()}
        before = {name: value.copy() for name, value in entries[2, 1].items()}
        wet_nonzero = np.argwhere(masks["U"] & (current["u_k"] != 0.0))
        require(wet_nonzero.size > 0,
                "admitted kt=2 stage-1 U field has no nonzero wet value")
        at = tuple(wet_nonzero[np.argmin(
            np.abs(current["u_k"][tuple(wet_nonzero.T)]))])
        original = np.float64(current["u_k"][at])
        planted_value = np.nextafter(original, np.float64(np.inf))
        require(original != 0.0 and planted_value != original,
                "recorded-entry nextafter plant is not one binding ULP")
        baseline = replay(current, before, 1)
        current["u_k"][at] = planted_value
        moved = replay(current, before, 1)
        control = _row(
            "production_recorded_nonzero_u_one_ulp", baseline["ssh_k"],
            moved["ssh_k"], masks["T"])
        require(not control["bit_identical"],
                "recorded-entry one-ULP plant did not change production SSH")
        control.update({
            "input_field": "u_k",
            "input_cell": [int(x) for x in at],
            "input_before": float(original),
            "input_after": float(planted_value),
            "input_ulp_delta": 1,
        })
        controls.append(control)
    return _gate_rows({
        "label": f"SWE-RK3-STAGES-{mode.upper()}", "legoesm_git_sha": sha,
        "bar": BAR, "plant": plant, "production_stage_executed": True,
        "stage3_input_audit": _stage3_input_audit(entries, masks),
        "nondiscriminating": [nondiscriminating], "controls": controls,
        "per_kt": per_kt,
    })


def ens_stage1(stage_root: Path, *, plant: str | None = None,
               allow_dirty: bool = False) -> dict:
    """Compatibility name for the complete given-entry production stage arm."""
    return swe_rk3_stages(
        stage_root, mode="given", plant=plant, allow_dirty=allow_dirty)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=(
        "self-check", "independent", "stage-entry-read", "ens-stage1",
        "stages-given", "stages-chained", "carried-state"),
                   required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    p.add_argument("--plant", nargs="?", const="ulp", choices=("ulp",))
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    try:
        if a.arm == "self-check":
            out = self_check(a.root, plant=a.plant is not None,
                             allow_dirty=a.allow_dirty)
        elif a.arm == "independent":
            out = independent(a.root, allow_dirty=a.allow_dirty)
        elif a.arm == "carried-state":
            out = carried_state_audit(
                a.stage_root, allow_dirty=a.allow_dirty)
        elif a.arm == "stage-entry-read":
            entries = read_stage_entries(a.stage_root)
            out = {"label": "STAGE-ENTRY-READ", "records": len(entries),
                   "keys": [list(k) for k in entries]}
        elif a.arm == "ens-stage1":
            out = ens_stage1(a.stage_root, plant=a.plant,
                             allow_dirty=a.allow_dirty)
        else:
            out = swe_rk3_stages(
                a.stage_root,
                mode="given" if a.arm == "stages-given" else "chained",
                plant=a.plant, allow_dirty=a.allow_dirty)
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    if a.json:
        a.json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(out["label"])
    if "all_at_bar" in out:
        print(f"all_at_bar={out['all_at_bar']} bit_identical={out['all_bit_identical']}")
        return _exit_code(out)
    print(f"records={out.get('records', len(out.get('per_kt', [])))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
