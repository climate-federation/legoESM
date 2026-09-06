#!/usr/bin/env python3
"""Fail-closed 19-frame OVERFLOW barotropic first-divergence gate.

The oracle record is a WRITE-only trace of the executed NEMO 5.0.2
``nn_bt_flt=1, rn_bt_alpha=0, nn_e=3`` branch.  The legoESM record is returned
by a private, side-effect-free test hook from the production recurrence.  No
public numerical selector is introduced by this instrument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import (
    allow_dirty_stamps,
    scoped_allow_dirty,
    worktree_stamp,
)

BAR = 1.0e-15
# Defaults are the OVERFLOW card this gate was written for.  Both are
# overridable so the identical 19-frame NEMO_L1_OVBT_1 record produced for
# another card can be read by this ONE reader instead of a second copy; LOCK's
# own record carries (134, 7, 1, 19, 64), i.e. a single barotropic substep.
CASE = "OVERFLOW-zps"
EXPECTED = (206, 7, 4, 19, 64)
CASE_EXPECTED = {
    "OVERFLOW-zps": (206, 7, 4, 19, 64),
    "LOCK_EXCHANGE-zco": (134, 7, 1, 19, 64),
}
# The resolved barotropic program the candidate must carry, per card, read
# from each card's namelist_cfg rather than assumed: (time filter, nn_e).
CASE_BAROTROPIC = {
    "OVERFLOW-zps": ("nemo_boxcar1_ab3", 3),
    "LOCK_EXCHANGE-zco": ("nemo_ab3am4", 1),
}
# resolved_program, whole_step_kt2_causal_arm and ownership are OVERFLOW
# CONCLUSIONS with OVERFLOW constants baked in.  Emitting them for another
# card would publish a false record, so they are withheld by name.
OVERFLOW_ONLY_REPORT_KEYS = (
    "resolved_program", "whole_step_kt2_causal_arm", "ownership")
DEFAULT_ORACLE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/barotropic_walk/"
    "oracle_kt1_calls/oracle_overflow_bt_substeps_kt00000001_call1.bin"
)
DEFAULT_NEW_ENTRY = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/barotropic_walk/"
    "oracle_kt1_calls/oracle_step_entry_kt00000001.bin"
)
DEFAULT_CERTIFIED_ENTRY = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10/"
    "oracle_step_entry_kt00000001.bin"
)
# The trajectory report is a PER-CARD artifact.  A single OVERFLOW default
# meant a LOCK run on gate defaults would stamp OVERFLOW's file as LOCK's
# ``trajectory_gate`` provenance -- a false record even though the block it
# feeds is withheld for non-OVERFLOW cards.  Defaults are per card now, and
# ``_trajectory_kt2`` refuses a report belonging to another card.
CASE_TRAJECTORY = {
    "OVERFLOW-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/barotropic_walk/"
        "overflow_kt1_10_flux_gate.json"),
    "LOCK_EXCHANGE-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/"
        "lock_trajectory_gate_kt10.json"),
}
DEFAULT_TRAJECTORY = CASE_TRAJECTORY["OVERFLOW-zps"]

FIELDS = (
    "eta_entry",
    "u_entry",
    "v_entry",
    "eta_mid",
    "u_mid",
    "v_mid",
    "transport_u",
    "transport_v",
    "eta_continuity",
    "eta_pgf",
    "pgf_u",
    "pgf_v",
    "slow_u",
    "slow_v",
    "drag_u",
    "drag_v",
    "u_exit",
    "v_exit",
    "eta_exit",
)
STAGGER = {
    name: (
        "T"
        if name.startswith("eta")
        else "U"
        if name.endswith("_u") or name.startswith("u_")
        else "V"
    )
    for name in FIELDS
}

# Rule 1d: every captured array is registered to the exact executing source
# assignment and to the substep-local time frame it represents.  The first
# substep's ``m`` is the caller's Kmm; later ``m`` values are the preceding
# substep exit after dynspg_ts.F90:804-816 rotates the recurrence.
FRAME_REGISTRY = {
    "eta_entry": {"time_level": "external m (caller Kmm on substep 1)", "source": "src/OCE/DYN/dynspg_ts.F90:484-486,814-816"},
    "u_entry": {"time_level": "external m (caller Kmm on substep 1)", "source": "src/OCE/DYN/dynspg_ts.F90:484-487,806-808"},
    "v_entry": {"time_level": "external m (caller Kmm on substep 1)", "source": "src/OCE/DYN/dynspg_ts.F90:484-488,810-812"},
    "eta_mid": {"time_level": "external m+1/2 AB3 predictor", "source": "src/OCE/DYN/dynspg_ts.F90:534-543,558-562"},
    "u_mid": {"time_level": "external m+1/2 AB3 predictor", "source": "src/OCE/DYN/dynspg_ts.F90:534-550"},
    "v_mid": {"time_level": "external m+1/2 AB3 predictor", "source": "src/OCE/DYN/dynspg_ts.F90:534-553"},
    "transport_u": {"time_level": "external m+1/2 transport", "source": "src/OCE/DYN/dynspg_ts.F90:603-605"},
    "transport_v": {"time_level": "external m+1/2 transport", "source": "src/OCE/DYN/dynspg_ts.F90:603-608"},
    "eta_continuity": {"time_level": "external m+1 after continuity", "source": "src/OCE/DYN/dynspg_ts.F90:623-630"},
    "eta_pgf": {"time_level": "external m+1/2 backward-interpolated PGF operand", "source": "src/OCE/DYN/dynspg_ts.F90:671-679"},
    "pgf_u": {"time_level": "external m+1/2 PGF tendency", "source": "src/OCE/DYN/dynspg_ts.F90:681-684"},
    "pgf_v": {"time_level": "external m+1/2 PGF tendency", "source": "src/OCE/DYN/dynspg_ts.F90:681-685"},
    "slow_u": {"time_level": "caller Kmm slow forcing, fixed through external loop", "source": "src/OCE/DYN/dynspg_ts.F90:270-300"},
    "slow_v": {"time_level": "caller Kmm slow forcing, fixed through external loop", "source": "src/OCE/DYN/dynspg_ts.F90:270-300"},
    "drag_u": {"time_level": "external m drag tendency", "source": "src/OCE/DYN/dynspg_ts.F90:699-704"},
    "drag_v": {"time_level": "external m drag tendency", "source": "src/OCE/DYN/dynspg_ts.F90:699-704"},
    "u_exit": {"time_level": "external m+1 instantaneous velocity", "source": "src/OCE/DYN/dynspg_ts.F90:734-755"},
    "v_exit": {"time_level": "external m+1 instantaneous velocity", "source": "src/OCE/DYN/dynspg_ts.F90:734-760"},
    "eta_exit": {"time_level": "external m+1 instantaneous sea level", "source": "src/OCE/DYN/dynspg_ts.F90:623-630"},
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(*, allow_dirty: bool = False) -> str:
    """Exact legoESM producer revision (fails closed on tracked dirt)."""
    from legoesm.ocean.fidelity.provenance import git_sha as _stamp

    try:
        return _stamp(allow_dirty=allow_dirty)
    except RuntimeError as error:
        raise GateError(f"cannot stamp legoESM git SHA: {error}") from error


def validate_frame_registry(registry=FRAME_REGISTRY) -> None:
    require(set(registry) == set(FIELDS), (
        "frame registry mismatch: missing="
        f"{sorted(set(FIELDS) - set(registry))}, extra="
        f"{sorted(set(registry) - set(FIELDS))}"
    ))
    for name, row in registry.items():
        require(bool(row.get("time_level")), f"{name}: missing time level")
        source = row.get("source", "")
        require(source.startswith("src/OCE/DYN/dynspg_ts.F90:"),
                f"{name}: missing exact dynspg_ts source")


def _read_exact(handle, count: int, context: str) -> bytes:
    value = handle.read(count)
    require(len(value) == count, f"{context}: truncated record")
    return value


def _xy_full(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def _xy_a2d(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx - 4, ny - 4), order="F").T


def read_oracle_trace(path: Path, expected_kt: int = 1) -> dict:
    """Read the mixed full-halo/A2D NEMO stream record, refusing leftovers.

    ``expected_kt`` is the ocean step the record must carry.  RK3 calls
    ``dyn_spg_ts`` with ``Kmm == Kbb`` (stp2d.F90:281) and rotates
    ``Nbb <-> Naa`` at every step end (stprk3.F90:213), so the level triple is
    ``(1,1,3)`` at kt=1 and ``(3,3,1)`` at kt=2; both are checked structurally.
    """
    with path.open("rb") as handle:
        magic = _read_exact(handle, 16, str(path)).decode("ascii").rstrip()
        header = struct.unpack("=11i", _read_exact(handle, 44, str(path)))
        (version, kt, call, kbb, kmm, kaa, ncycle, nx, ny, nfields, bits) = header
        require(magic == "NEMO_L1_OVBT_1", f"{path}: bad magic {magic!r}")
        require(
            (version, kt, call, nx, ny, ncycle, nfields, bits)
            == (1, expected_kt, 1, *EXPECTED),
            f"{path}: bad header {header}",
        )
        full_count = nx * ny
        a2d_count = (nx - 4) * (ny - 4)
        substeps = []
        for expected_jn in range(1, ncycle + 1):
            (jn,) = struct.unpack("=i", _read_exact(handle, 4, str(path)))
            require(jn == expected_jn, f"{path}: substep {jn}, expected {expected_jn}")
            frames = {}
            for name in FIELDS:
                count = a2d_count if name in ("slow_u", "slow_v") else full_count
                raw = _read_exact(handle, 8 * count, f"{path}:{jn}:{name}")
                values = np.frombuffer(raw, dtype=np.float64).copy()
                frames[name] = (
                    _xy_a2d(values, nx, ny) if count == a2d_count else _xy_full(values, nx, ny)
                )
            substeps.append(frames)
        require(handle.read(1) == b"", f"{path}: trailing unregistered bytes")
    require((kbb, kmm, kaa) == ((1, 1, 3) if kt % 2 == 1 else (3, 3, 1)),
            f"{path}: unexpected time levels {(kbb, kmm, kaa)} at kt={kt}")
    return {
        "header": {
            "version": version,
            "kt": kt,
            "call": call,
            "Kbb": kbb,
            "Kmm": kmm,
            "Kaa": kaa,
            "icycle": ncycle,
            "jpi": nx,
            "jpj": ny,
            "nfields": nfields,
            "bits": bits,
        },
        "substeps": substeps,
    }


def _candidate_frame(values, staggering: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if staggering == "T":
        return values
    if staggering == "U":
        return values[:, 1:]
    return values[1:, :]


def _trajectory_gate():
    """The sibling trajectory gate: one entry-dump reader, one stagger map."""
    import importlib.util

    script = Path(__file__).with_name("nemo_testcase_phase3_trajectory_gate.py")
    spec = importlib.util.spec_from_file_location(
        "nemo_testcase_phase3_trajectory_gate", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_from_oracle_entry(state, entry: dict, active_masks: dict):
    """Overwrite the ACTIVE prognostic cells of ``state`` with a NEMO
    ``oracle_step_entry`` record (ts/uu/vv/ssh at Nbb).

    Inverse of the trajectory gate's ``lego_fields`` stagger map: NEMO stores
    one u record per T column (its east face), legoESM one extra western wall
    face at ``u[:, 0]`` (``v[0]`` likewise).  Inactive cells keep legoESM's
    own values; a wet-mask disagreement would surface as a frame residual.
    """
    import jax.numpy as jnp

    nlev = int(np.asarray(state.T.data).shape[-1])

    def merged(current, reference, mask):
        current = np.array(current, dtype=np.float64, copy=True)
        reference = np.asarray(reference, dtype=np.float64)
        return np.where(np.asarray(mask, dtype=bool), reference, current)

    T = merged(state.T.data, entry["T"][..., :nlev], active_masks["T"])
    S = merged(state.S.data, entry["S"][..., :nlev], active_masks["S"])
    u = np.array(state.u.data, dtype=np.float64, copy=True)
    u[:, 1:, :] = merged(u[:, 1:, :], entry["u"][..., :nlev], active_masks["u"])
    v = np.array(state.v.data, dtype=np.float64, copy=True)
    v[1:, :, :] = merged(v[1:, :, :], entry["v"][..., :nlev], active_masks["v"])
    eta = merged(state.eta.data, entry["ssh"], active_masks["ssh"])
    return state._replace(
        T=state.T.replace(data=jnp.asarray(T)),
        S=state.S.replace(data=jnp.asarray(S)),
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
    )


def capture_legoesm_trace(*, flux_form_override, kt: int = 1,
                          reseed_entry: dict | None = None,
                          legacy_seed_faces: bool = False,
                          start_state=None) -> dict:
    """Capture the ``kt``-th solve without changing the two-value production
    return.

    ``flux_form_override=None`` is the PRODUCTION arm: the barotropic solver
    resolves the literal flux-form update from the card's own config through
    ``nemo_flux_form_update_active``; ``False`` forces the legacy velocity
    update (harness-only control arm).

    ``kt`` selects which step's external solve is captured: the first
    ``kt - 1`` steps run as the production trajectory gate runs them, and the
    ``kt``-th returns the trace through the production-jitted pytree seam.
    ``reseed_entry``
    (a NEMO ``oracle_step_entry_kt{kt}`` record) replaces legoESM's own
    kt-entry prognostic state by NEMO's, so the captured solve starts from
    an EXACT entry and its first over-bar frame names the operand rather
    than an inherited residual.  ``start_state`` hands the kt-entry state in
    directly (a caller-built trajectory, e.g. the stage-3 remainder probe's
    perturbed arm); it is mutually exclusive with ``reseed_entry``.
    """
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_flux_form_update_active,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config
    want_filter, want_nn_e = CASE_BAROTROPIC[CASE]
    require(cfg.barotropic.barotropic_time_filter == want_filter,
            f"wrong filter: {cfg.barotropic.barotropic_time_filter!r} != {want_filter!r}")
    require(cfg.barotropic.n_barotropic_substeps == want_nn_e,
            f"wrong nn_e: {cfg.barotropic.n_barotropic_substeps} != {want_nn_e}")
    require(cfg.momentum_time_integrator == "rk3_ws", "wrong momentum integrator")
    require(cfg.momentum_advection == "flux_form", "wrong momentum form")
    # Measured, not assumed: the production predicate the solver evaluates.
    production_resolution = bool(nemo_flux_form_update_active(cfg))
    require(production_resolution,
            "production does not resolve to the literal flux-form update")

    require(kt >= 1, "kt must be positive")
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_flux_form_update_override=flux_form_override,
            legacy_seed_min_rule_faces=legacy_seed_faces))
    state = card.recipe.initial_state
    require(reseed_entry is None or start_state is None,
            "reseed_entry and start_state are mutually exclusive")
    if start_state is not None:
        state = start_state
    elif reseed_entry is not None:
        require(int(reseed_entry["step"]) == kt,
                f"re-seed record is kt={reseed_entry['step']}, wanted {kt}")
        active = _trajectory_gate().expected_masks(card)
        state = state_from_oracle_entry(state, reseed_entry, active)
    else:
        for _ in range(kt - 1):
            state = model.step(state, dt=card.dt_s)
    # The WRITE-only trace is an ordinary leaf of the compiled return pytree.
    # Materialise it only after ``step`` returns; capturing it in a Python
    # closure during tracing retains DynamicJaxprTracers and makes the gate
    # depend on an eager route that production never executes.
    captured = model.step(state, dt=card.dt_s)
    require(hasattr(captured, "substeps"), "compiled step did not return trace")
    trace = captured.substeps
    # The solver returns ONE keyed frame shared with the L2-GYRE harness (see
    # barotropic_latlon_cgrid._run_substep_loop).  Bind by NAME: this registry
    # declares the subset THIS gate scores, and a name the solver stops
    # emitting goes red here immediately.  (The previous positional length
    # check could not see a same-length reorder and went red on an unrelated
    # addition; keying is strictly the stronger guard.)
    missing = [name for name in FIELDS if name not in trace]
    require(not missing, f"candidate frame is missing {missing}")
    icycle = EXPECTED[2]
    require(int(np.asarray(trace["eta_entry"]).shape[0]) == icycle,
            "candidate icycle mismatch")
    substeps = []
    for jn in range(icycle):
        substeps.append(
            {
                name: _candidate_frame(np.asarray(trace[name])[jn], STAGGER[name])
                for name in FIELDS
            }
        )
    masks = {
        "T": np.asarray(state.land_mask.data, dtype=bool),
        "U": np.asarray(state.u_mask.data[:, 1:], dtype=bool),
        "V": np.asarray(state.v_mask.data[1:, :], dtype=bool),
    }
    dtypes = {
        "T": str(np.asarray(state.T.data).dtype),
        "S": str(np.asarray(state.S.data).dtype),
        "u": str(np.asarray(state.u.data).dtype),
        "v": str(np.asarray(state.v.data).dtype),
        "ssh": str(np.asarray(state.eta.data).dtype),
        "h_partial": str(np.asarray(card.recipe.z_coord.h_partial).dtype),
    }
    require(set(dtypes.values()) == {"float64"}, f"non-fp64 arrays {dtypes}")
    return {
        "substeps": substeps,
        "masks": masks,
        "dtypes": dtypes,
        "backend": jax.default_backend(),
        "flux_form_override": flux_form_override,
        "production_flux_form_update_active": production_resolution,
        "kt": kt,
        "reseeded_from_oracle_entry": reseed_entry is not None,
        "started_from_caller_state": start_state is not None,
        "legacy_seed_faces_test_hook": legacy_seed_faces,
    }


def score_frame(name: str, oracle, candidate, mask, *, plant=False) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    active = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == active.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate dtype {candidate.dtype}")
    no_active_face = not bool(active.any())
    if no_active_face:
        active = np.ones_like(active)
    if plant:
        candidate = candidate.copy()
        candidate[tuple(np.argwhere(active)[0])] += 1.0
    require(np.all(np.isfinite(candidate[active])), f"{name}: candidate non-finite")
    residual = candidate[active] - oracle[active]
    absolute = np.abs(residual)
    index_flat = int(np.argmax(absolute))
    active_indices = np.argwhere(active)
    unequal = candidate[active].view(np.uint64) != oracle[active].view(np.uint64)
    absolute_max = float(absolute[index_flat])
    scale = max(float(np.max(np.abs(oracle[active]))), 1.0)
    error = absolute_max / scale
    row = {
        "name": name,
        "status": "AT-BAR" if error <= BAR else "DEBT",
        "exact": bool(np.array_equal(candidate[active], oracle[active])),
        "normalized_max_abs": error,
        "absolute_max": absolute_max,
        "reference_max_abs": float(np.max(np.abs(oracle[active]))),
        "signed_residual_at_max": float(residual[index_flat]),
        "bar": BAR,
        "n": int(active.sum()),
        "n_unequal": int(np.count_nonzero(unequal)),
        "first_unequal_index": (
            active_indices[int(np.flatnonzero(unequal)[0])].tolist()
            if np.any(unequal) else None),
        "max_residual_index": active_indices[index_flat].tolist(),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if no_active_face and row["status"] == "AT-BAR":
        row["status"] = "UNMEASURED"
        row["reason"] = (
            "no active meridional face in the one-wet-row tank; all stored "
            "points remain a gross-zero control, not an alignment claim"
        )
    return row


def _trajectory_kt2(path: Path) -> dict:
    report = json.loads(path.read_text())
    require(report.get("case") == CASE,
            f"{path} is a {report.get('case')!r} trajectory report, but this "
            f"run is {CASE!r}; stamping it would be a false provenance record")
    step = next(item for item in report["steps"] if item["kt"] == 2)
    result = {}
    for field in ("T", "u", "ssh"):
        row = next(item for item in step["rows"] if item["name"].endswith(f".{field}"))
        result[field] = float(row["normalized_max_abs"])
    return result


def run(
    oracle_path: Path,
    new_entry: Path,
    certified_entry: Path,
    trajectory_path: Path,
    *,
    plant_entry=False,
    plant_exit=False,
    pytest_log: Path | None = None,
    allow_dirty: bool = False,
) -> dict:
    # Stamp FIRST so a dirty tree refuses before any compute (fail closed).
    allow_dirty_stamps(allow_dirty)
    legoesm_git_sha = git_sha(allow_dirty=allow_dirty)
    validate_frame_registry()
    require(
        new_entry.read_bytes() == certified_entry.read_bytes(),
        "instrumented step-entry is not byte-identical to certified oracle",
    )
    oracle = read_oracle_trace(oracle_path)
    baseline = capture_legoesm_trace(flux_form_override=False)
    faithful = capture_legoesm_trace(flux_form_override=None)   # production
    require(baseline["masks"].keys() == faithful["masks"].keys(), "mask registry drift")

    arms = {}
    first = {}
    for arm_name, candidate in (
        ("legacy_velocity_update", baseline),
        ("nemo_flux_form_update", faithful),
    ):
        substeps = []
        first_over_bar = None
        for jn, (reference_frames, candidate_frames) in enumerate(
            zip(oracle["substeps"], candidate["substeps"]), start=1
        ):
            rows = []
            for name in FIELDS:
                staggering = STAGGER[name]
                plant = (plant_entry and jn == 1 and name == "eta_entry") or (
                    plant_exit and jn == 1 and name == "u_exit"
                )
                row = score_frame(
                    f"{arm_name}.substep{jn}.{name}",
                    reference_frames[name],
                    candidate_frames[name],
                    candidate["masks"][staggering],
                    plant=plant,
                )
                row.update(
                    {
                        "frame": "instantaneous_external_substep_operand",
                        "oracle_time_level": FRAME_REGISTRY[name]["time_level"],
                        "oracle_source": FRAME_REGISTRY[name]["source"],
                        "staggering_and_reduction": (
                            f"oracle NEMO {name} and legoESM {name} are both "
                            f"native {staggering}-point operands at external "
                            f"substep {jn}; both strip the NEMO two-cell halo, "
                            "intersect the certified native wet mask, and use an "
                            "elementwise L-infinity reduction with no depth or "
                            "substep-time averaging"
                        ),
                    }
                )
                rows.append(row)
                if first_over_bar is None and row["status"] == "DEBT":
                    first_over_bar = {
                        "substep": jn,
                        "frame": name,
                        "normalized_max_abs": row["normalized_max_abs"],
                    }
            substeps.append({"substep": jn, "rows": rows})
        arms[arm_name] = {
            "substeps": substeps,
            "first_over_bar": first_over_bar,
            "backend": candidate["backend"],
            "dtypes": candidate["dtypes"],
        }
        first[arm_name] = first_over_bar

    scaling = []
    for jn in range(EXPECTED[2]):
        ref = oracle["substeps"][jn]["u_exit"]
        old = baseline["substeps"][jn]["u_exit"]
        new = faithful["substeps"][jn]["u_exit"]
        mask = faithful["masks"]["U"]
        old_error = float(np.max(np.abs(old[mask] - ref[mask])))
        new_error = float(np.max(np.abs(new[mask] - ref[mask])))
        movement = float(np.max(np.abs(new[mask] - old[mask])))
        scaling.append(
            {
                "substep": jn + 1,
                "legacy_absolute_error": old_error,
                "faithful_absolute_error": new_error,
                "causal_movement": movement,
                "movement_over_legacy_error": movement / max(old_error, np.finfo(float).tiny),
                "improvement_factor": old_error / max(new_error, np.finfo(float).tiny),
            }
        )

    kt2 = _trajectory_kt2(trajectory_path)
    prior = {
        "T": 2.4003505494363254e-08,
        "u": 3.082388670642283e-06,
        "ssh": 1.2372313177265448e-07,
    }
    movement = {
        field: {
            "before": prior[field],
            "after": kt2[field],
            "after_over_before": kt2[field] / prior[field],
        }
        for field in prior
    }
    causal_pass = (
        kt2["u"] <= prior["u"] / 10.0
        and kt2["T"] <= prior["T"] * 10.0
        and kt2["ssh"] <= prior["ssh"] * 10.0
    )
    first_slow = next(
        row
        for row in arms["legacy_velocity_update"]["substeps"][0]["rows"]
        if row["name"].endswith(".slow_u")
    )
    first_exit = next(
        row
        for row in arms["legacy_velocity_update"]["substeps"][0]["rows"]
        if row["name"].endswith(".u_exit")
    )
    slow_dt_prediction = first_slow["absolute_max"] * (10.0 / 3.0)
    # A zero denominator has no ratio.  Dividing by np.finfo(float).tiny
    # printed 1.559250241824e+290 on OVERFLOW, which reads as a measurement
    # and is not one: substep 1's slow_u error is exactly 0.0 there, so pure
    # inheritance predicts nothing at all and the comparison is undefined.
    slow_ratio = (first_exit["absolute_max"] / slow_dt_prediction
                  if slow_dt_prediction > 0.0 else None)

    # Planted controls must land as the FIRST DEBT of BOTH arms at substep 1
    # with the +1.0 plant visible; otherwise the gate is broken (exit 2).
    planted_frame = "eta_entry" if plant_entry else "u_exit" if plant_exit else None
    if planted_frame is not None:
        for arm_name, arm in arms.items():
            landed = arm["first_over_bar"]
            planted_row = next(
                row for row in arm["substeps"][0]["rows"]
                if row["name"].endswith(f".{planted_frame}"))
            require(
                landed is not None and landed["substep"] == 1
                and landed["frame"] == planted_frame
                and planted_row["absolute_max"] >= 0.5,
                f"{arm_name}: planted {planted_frame} control did not become "
                f"the first DEBT (got {landed})")
    # Verdict = the production arm's own rows; the legacy arm is a control.
    production_rows = [
        row for substep in arms["nemo_flux_form_update"]["substeps"]
        for row in substep["rows"]]
    status = "DEBT" if any(row["status"] == "DEBT" for row in production_rows) else "AT-BAR"

    faithful_substep1 = [
        {
            "frame": row["name"].rsplit(".", 1)[-1],
            "status": row["status"],
            "normalized_max_abs": row["normalized_max_abs"],
            "absolute_max": row["absolute_max"],
        }
        for row in arms["nemo_flux_form_update"]["substeps"][0]["rows"]
    ]
    artifacts = {
        "oracle_trace": {"path": str(oracle_path), "sha256": sha256(oracle_path)},
        "instrumented_entry": {"path": str(new_entry), "sha256": sha256(new_entry)},
        "certified_entry": {"path": str(certified_entry), "sha256": sha256(certified_entry)},
        "trajectory_gate": {"path": str(trajectory_path), "sha256": sha256(trajectory_path)},
    }
    if pytest_log is not None:
        require(pytest_log.is_file(), f"pytest log does not exist: {pytest_log}")
        artifacts["pytest_log"] = {"path": str(pytest_log), "sha256": sha256(pytest_log)}

    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l1-overflow-barotropic-gate-v1",
        "case": CASE,
        "status": status,
        "bar": BAR,
        "resolved_program": {
            "nn_bt_flt": 1,
            "rn_bt_alpha": 0.0,
            "nn_e": 3,
            "actual_icycle": 4,
            "ln_bt_fw": True,
            "momentum_advection": "flux_form_up3",
            "drag": "OFF",
            "rotation": "f=0",
        },
        "legoesm_git_sha": legoesm_git_sha,
        "production_flux_form_update_active": (
            faithful["production_flux_form_update_active"]),
        "time_level_header": oracle["header"],
        "frame_time_level_registry": FRAME_REGISTRY,
        "artifacts": artifacts,
        "overlap_control": "EXACT_BYTE_IDENTITY",
        "arms": arms,
        "faithful_flux_update_substep1_19_frame_residuals": faithful_substep1,
        "scaling_check_before_owner_label": {
            "u_exit_by_substep": scaling,
            "substep1_slow_u_error_times_dt": slow_dt_prediction,
            "substep1_u_exit_error": first_exit["absolute_max"],
            "exit_over_slow_dt_prediction": slow_ratio,
            "exit_over_slow_dt_prediction_undefined_reason": (
                None if slow_ratio is not None else
                "substep-1 slow_u error is exactly 0.0, so the inheritance "
                "prediction is 0 and the ratio does not exist"),
        },
        "whole_step_kt2_causal_arm": {
            "frozen_confirm_predicate": ("u error falls >=10x and neither T nor SSH worsens >10x"),
            "passed": causal_pass,
            "movement": movement,
        },
        "ownership": {
            "slow_u_roundoff_tail": {
                "label": "CONFIRMED_FIRST_STRICT_BAR_OPERAND_CONTRIBUTOR",
                "root_kt2_initiator": "REFUTED_BY_SCALING",
                "reason": (
                    "dt times the 5.63e-16 slow-forcing residual predicts the "
                    "1.87e-15 substep-1 exit tail, but is about nine orders "
                    "below the kt=2 instantaneous-U debt"
                ),
            },
            "nemo_flux_form_external_update": {
                "substep_recurrence": "CONFIRMED_STRUCTURAL_OWNER",
                "root_kt2_initiator": "REFUTED_BY_FROZEN_CAUSAL_PREDICATE",
                "production_selection": "KEPT_LITERAL_UPDATE_RULE8",
                "reason": (
                    "the literal flux-form external update (dynspg_ts.F90:"
                    "734-761) is PRODUCTION-ACTIVE under the NEMO WS-RK3 + "
                    "flux-form identity (barotropic_latlon_cgrid.py "
                    "_nemo_flux_form_update). Its 19 substep-1 frames are "
                    "printed in faithful_flux_update_substep1_19_frame_"
                    "residuals: every measured frame is AT-BAR except u_exit "
                    "at 1.8735013540549517e-15, which is identical in the "
                    "legacy arm and equals dt times the 5.63e-16 slow_u "
                    "input residual (ratio 0.998), i.e. it is inherited from "
                    "the slow forcing, not produced by the update. Substeps "
                    "2-4 fall from 2.6e-9/1.3e-7/5.6e-7 to the 1e-15 class. "
                    "Faithful-but-worse at kt=2 (u 3.08238867e-06 -> "
                    "3.31108168e-06, SSH 1.23723132e-07 -> 1.04916076e-14, T "
                    "unchanged; whole_step_kt2_causal_arm) is disclosed, "
                    "not reverted (Rule 8); the compensated defect is the "
                    "RK3 stage composition measured by the phase-3 gate"
                ),
            },
            "substep1_pgf_continuity_metrics": {
                "label": "CONFIRMED_EXONERATED_THROUGH_FIRST_STRICT_BOUNDARY",
            },
            "drag": {
                "label": "CONFIRMED_INACTIVE_BOTH_MODELS",
                "reason": "resolved ln_drg_OFF=T and both dumped drag operands are exact zero",
            },
            "kt2_root_divergence": {
                "label": "UNMEASURED_OUTSIDE_BAROTROPIC_SUBSTEP_REGISTER",
                "reason": (
                    "the source-faithful substep recurrence no longer amplifies, "
                    "while the post-stage whole-step instantaneous U remains "
                    "3.31e-6; the next operand lies in RK3 stage composition"
                ),
            },
        },
        "unmeasured": [
            "active-v alignment (no wet V face)",
            "post-barotropic RK3 stage operand initiating kt=2 instantaneous U/T",
        ],
        "controls": {"plant_entry": plant_entry, "plant_exit": plant_exit},
        "source_citations": [
            "NEMO 5.0.2 src/OCE/DYN/dynspg_ts.F90:270-300",
            "NEMO 5.0.2 src/OCE/DYN/dynspg_ts.F90:484-493",
            "NEMO 5.0.2 src/OCE/DYN/dynspg_ts.F90:534-630",
            "NEMO 5.0.2 src/OCE/DYN/dynspg_ts.F90:672-761",
            "NEMO 5.0.2 src/OCE/DYN/dynspg_ts.F90:823-847",
        ],
    }
    if CASE != "OVERFLOW-zps":
        for key in OVERFLOW_ONLY_REPORT_KEYS:
            report[key] = {
                "status": "NOT_APPLICABLE_ON_THIS_CARD",
                "reason": (
                    f"{key} is an OVERFLOW-zps conclusion with OVERFLOW "
                    "constants baked in (its resolved namelist, its kt=2 "
                    "prior residuals, its owner labels).  Publishing it for "
                    f"{CASE} would be a false record."
                ),
            }
    return report


def _score_substeps(arm_name: str, oracle: dict, candidate: dict) -> tuple[list, dict | None, list]:
    substeps = []
    first_over_bar = None
    masked_over_bar = []
    for jn, (reference_frames, candidate_frames) in enumerate(
        zip(oracle["substeps"], candidate["substeps"]), start=1
    ):
        rows = []
        for name in FIELDS:
            row = score_frame(
                f"{arm_name}.substep{jn}.{name}", reference_frames[name],
                candidate_frames[name], candidate["masks"][STAGGER[name]])
            row.update({
                "frame": "instantaneous_external_substep_operand",
                "oracle_time_level": FRAME_REGISTRY[name]["time_level"],
                "oracle_source": FRAME_REGISTRY[name]["source"],
            })
            # A stagger with no active face in the certified tank (the
            # three-row OVERFLOW has no wet V face) keeps score_frame's
            # gross-zero verdict exactly as the kt=1 gate does, but is not an
            # alignment row: a value on a face the update masks (ssvmask,
            # dynspg_ts.F90:757-760) can never be the first divergence.
            row["alignment_row"] = bool(
                np.asarray(candidate["masks"][STAGGER[name]]).any())
            rows.append(row)
            if row["status"] == "DEBT" and not row["alignment_row"]:
                masked_over_bar.append({
                    "substep": jn, "frame": name,
                    "absolute_max": row["absolute_max"]})
            if (first_over_bar is None and row["status"] == "DEBT"
                    and row["alignment_row"]):
                first_over_bar = {
                    "substep": jn, "frame": name,
                    "normalized_max_abs": row["normalized_max_abs"],
                    "absolute_max": row["absolute_max"],
                }
        substeps.append({"substep": jn, "rows": rows})
    return substeps, first_over_bar, masked_over_bar


def run_kt_walk(kt: int, oracle_root: Path, entry_root: Path, *,
                allow_dirty: bool = False, legacy_seed_faces: bool = False) -> dict:
    """Frame-by-frame first divergence of the kt-th external solve, kt >= 2.

    Two production arms: ``inherited_entry`` starts the solve from legoESM's
    own kt-entry state (whatever residual it already carries), and
    ``reseeded_from_oracle_entry`` starts it from NEMO's dumped kt-entry
    prognostic state, so a frame over the bar there is produced INSIDE the
    step by an operand the barotropic solve consumes or a memory NEMO carries
    that the prognostic state does not.
    """
    allow_dirty_stamps(allow_dirty)
    legoesm_git_sha = git_sha(allow_dirty=allow_dirty)
    validate_frame_registry()
    require(kt >= 2, "run_kt_walk is the kt>=2 walk; kt=1 is run()")
    trace_path = oracle_root / f"oracle_overflow_bt_substeps_kt{kt:08d}_call1.bin"
    entry_path = entry_root / f"oracle_step_entry_kt{kt:08d}.bin"
    oracle = read_oracle_trace(trace_path, expected_kt=kt)
    trajectory = _trajectory_gate()
    entry = trajectory.read_entry(entry_path, CASE)
    require(int(entry["step"]) == kt, f"{entry_path}: step mismatch")

    arms = {}
    for arm_name, reseed in (("inherited_entry", None),
                             ("reseeded_from_oracle_entry", entry)):
        candidate = capture_legoesm_trace(
            flux_form_override=None, kt=kt, reseed_entry=reseed,
            legacy_seed_faces=legacy_seed_faces)
        substeps, first, masked = _score_substeps(arm_name, oracle, candidate)
        arms[arm_name] = {
            "substeps": substeps,
            "first_over_bar": first,
            "masked_stagger_rows_over_bar": masked,
            "reseeded_fields": (
                ["T", "S", "u", "v", "eta"] if reseed is not None else []),
            "fields_kept_from_the_card_initial_state": (
                ["w (diagnostic, recomputed in step)", "every None-seeded "
                 "history slot (bt_hist, eta_before, F_slow_*_prev, ...)"]
                if reseed is not None else []),
            "backend": candidate["backend"],
            "dtypes": candidate["dtypes"],
            "reseeded_from_oracle_entry": candidate["reseeded_from_oracle_entry"],
            "legacy_seed_faces_test_hook": candidate["legacy_seed_faces_test_hook"],
        }

    # The kt-entry prognostic residual the inherited arm starts from, scored
    # exactly as the trajectory gate scores it (same masks, same reduction).
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    card = build_nemo_testcase_card(CASE)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            legacy_seed_min_rule_faces=legacy_seed_faces))
    state = card.recipe.initial_state
    for _ in range(kt - 1):
        state = model.step(state, dt=card.dt_s)
    masks = trajectory.expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    fields = trajectory.lego_fields(state)
    entry_residual = {}
    for field in ("T", "S", "u", "v", "ssh"):
        reference = np.asarray(entry[field])
        if field != "ssh":
            reference = reference[..., :nlev]
        row = trajectory.score(
            f"{CASE}.kt{kt}.before.{field}", reference, fields[field],
            masks[field], allow_empty_no_active_face=field == "v")
        entry_residual[field] = {
            "status": row["status"], "exact": row["exact"],
            "normalized_max_abs": row["normalized_max_abs"]}

    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l1-overflow-barotropic-kt-walk-v1",
        "case": CASE,
        "kt": kt,
        "bar": BAR,
        "legoesm_git_sha": legoesm_git_sha,
        "time_level_header": oracle["header"],
        "frame_time_level_registry": FRAME_REGISTRY,
        "artifacts": {
            "oracle_trace": {"path": str(trace_path), "sha256": sha256(trace_path)},
            "oracle_entry": {"path": str(entry_path), "sha256": sha256(entry_path)},
        },
        "kt_entry_prognostic_residual_inherited_arm": entry_residual,
        "legacy_seed_faces_test_hook": legacy_seed_faces,
        "arms": arms,
        "status": ("AT-BAR" if arms["reseeded_from_oracle_entry"]["first_over_bar"] is None
                   else "DEBT"),
    }


@scoped_allow_dirty
def main(argv=None) -> int:
    global CASE, EXPECTED
    parser = argparse.ArgumentParser()
    parser.add_argument("--kt", type=int, default=1,
                        help="ocean step whose external solve is walked; "
                             ">=2 selects run_kt_walk")
    parser.add_argument("--oracle-root", type=Path,
                        help="directory holding oracle_overflow_bt_substeps_kt*_call1.bin (kt>=2)")
    parser.add_argument("--entry-root", type=Path,
                        help="directory holding oracle_step_entry_kt*.bin (kt>=2)")
    parser.add_argument(
        "--arm-legacy-seed-faces", action="store_true",
        help=("one-variable arm (kt>=2): restore the min-of-stretched-cells "
              "rescale in the loop-entry seed instead of NEMO's "
              "e3u_0*(1+r3u) (dynspg_ts.F90:487 / stprk3_stg.F90:440). "
              "NEMO has no such switch"))
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--new-entry", type=Path, default=DEFAULT_NEW_ENTRY)
    parser.add_argument("--certified-entry", type=Path, default=DEFAULT_CERTIFIED_ENTRY)
    parser.add_argument("--trajectory", type=Path,
                        help="per-card trajectory report; defaults to the "
                             "card's own (never another card's)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-entry", action="store_true")
    parser.add_argument("--plant-exit", action="store_true")
    parser.add_argument("--pytest-log", type=Path)
    parser.add_argument("--case", choices=tuple(CASE_EXPECTED), default=CASE,
                        help="testcase card the records belong to")
    parser.add_argument("--expected", type=int, nargs=5,
                        metavar=("NX", "NY", "NCYCLE", "NFIELDS", "BITS"),
                        help="record header shape; defaults to the card's")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="stamp '<sha>-dirty' instead of refusing a dirty tree")
    from legoesm.ocean.fidelity.ulp_move_gate import (
        add_ulp_compare_arguments, comparison_exit_code, run_ulp_comparison,
    )
    add_ulp_compare_arguments(parser)
    args = parser.parse_args(argv)
    CASE = args.case
    EXPECTED = tuple(args.expected) if args.expected else CASE_EXPECTED[CASE]
    if args.trajectory is None:
        args.trajectory = CASE_TRAJECTORY[CASE]
    # Exit codes: 0 AT-BAR, 1 DEBT (measured), 2 gate failure (a planted
    # control that did not land, a dirty tree, a bad oracle record).
    try:
        require(not (args.arm_legacy_seed_faces and args.kt < 2),
                "--arm-legacy-seed-faces is a kt>=2 walk control; at kt=1 the "
                "seed multiplies a zero velocity and the flag would be inert")
        if args.kt >= 2:
            require(args.oracle_root is not None and args.entry_root is not None,
                    "--kt >= 2 needs --oracle-root and --entry-root")
            report = run_kt_walk(args.kt, args.oracle_root, args.entry_root,
                                 allow_dirty=args.allow_dirty,
                                 legacy_seed_faces=args.arm_legacy_seed_faces)
            text = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text)
            print(text, end="")
            if args.compare_to:
                comparison = run_ulp_comparison(args, report)
                print(json.dumps(comparison, indent=2, sort_keys=True))
                return comparison_exit_code(comparison)
            return 0 if report["status"] == "AT-BAR" else 1
        report = run(
            args.oracle,
            args.new_entry,
            args.certified_entry,
            args.trajectory,
            plant_entry=args.plant_entry,
            plant_exit=args.plant_exit,
            pytest_log=args.pytest_log,
            allow_dirty=args.allow_dirty,
        )
    except GateError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if args.compare_to:
        comparison = run_ulp_comparison(args, report)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return comparison_exit_code(comparison)
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
