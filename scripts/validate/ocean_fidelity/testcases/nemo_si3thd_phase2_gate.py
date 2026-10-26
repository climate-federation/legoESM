#!/usr/bin/env python3
"""Phase-2 legoESM column gate against the accepted C1D_OMIP_L3 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import jax
import numpy as np

BAR = 1.0e-15
STAGES = (
    (0, "ENTRY", "now: global pre-thermodynamics", "icethd.F90:112"),
    (1, "POST_ZDF", "now: selected-category 1D", "icethd.F90:151-152"),
    (2, "POST_DH", "now: selected-category 1D", "icethd.F90:154-155"),
    (3, "POST_TEMP1", "now: selected-category 1D", "icethd.F90:157-158"),
    (4, "POST_SAL", "now: selected-category 1D", "icethd.F90:160-161"),
    (5, "POST_TEMP2", "now: selected-category 1D", "icethd.F90:163-164"),
    (6, "POST_DO", "now: global post-open-water growth, pre-correction", "icethd.F90:189-190"),
    (7, "EXIT", "now: global post-correction/LBC", "icethd.F90:221-225"),
)
SELECTED_NAMES = ("a_i", "h_i", "h_s", "t_su", "e_i", "e_s", "sz_i")
GLOBAL_NAMES = (
    "a_i", "v_i", "v_s", "sv_i", "oa_i", "t_su", "a_ip", "v_ip",
    "v_il", "e_i", "e_s", "szv_i",
)
ZDF_INPUT_NAMES = (
    "qns_ice", "qsr_ice", "dqns_ice", "qtr_ice_top", "t_bottom", "sss",
    "evaporation", "snow_precipitation", "qprec_ice", "qcn_ice_bottom",
    "qsb_ice_bottom", "fhld", "qlead",
)
ZDF_STATE_OPERAND_NAMES = ("t_s",)
ZDF_STATE_OPERAND_REGISTRY = {
    "t_s": {
        "shape": "(npti,nlay_s)",
        "time_level": (
            "current ice step after ice_var_glo2eqv(2) and ice_thd_1d2d; "
            "immediately before ice_thd_zdf"
        ),
        "source": (
            "icestp.F90:182-206; icethd.F90:343-355,418-435; "
            "icethd_zdf_bl99.F90:189-199"
        ),
    },
}
ZDF_OPERAND_REGISTRY = {
    0: ("INIT", 25, "icethd_zdf_bl99.F90:159-230"),
    1: ("ITER_P07_KAPPA", 20, "icethd_zdf_bl99.F90:261-335"),
    2: ("ITER_CAP_FLUX", 8, "icethd_zdf_bl99.F90:338-379"),
    3: ("ITER_MATRIX", 28, "icethd_zdf_bl99.F90:393-514"),
    4: ("ITER_FORWARD", 14, "icethd_zdf_bl99.F90:516-529"),
    5: ("ITER_SOLUTION", 9, "icethd_zdf_bl99.F90:531-558"),
    6: ("ITER_CONVERGENCE", 10, "icethd_zdf_bl99.F90:563-589"),
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_thd_step(stream, wanted_step: int) -> list[dict[str, np.ndarray]]:
    frames = []
    for stage, _, _, _ in STAGES:
        require(stream.read(16) == b"NEMO_L3THD_001  ", "thermo magic")
        header = struct.unpack("=11i", stream.read(44))
        version, step, got_stage, payload, nx, ny, nc, ni, ns, npti, bits = header
        require((version, step, got_stage, nc, ni, ns, bits) ==
                (1, wanted_step, stage, 1, 3, 3, 64), f"thermo header {header}")
        n2 = nx * ny
        if payload == 0:
            nval = n2 * nc * 9 + n2 * nc * (ni + ns + ni)
        else:
            nval = npti * 4 + npti * (ni + ns + ni)
        values = np.fromfile(stream, np.float64, nval)
        require(values.size == nval and np.all(np.isfinite(values)), "bad thermo payload")
        fields: dict[str, np.ndarray] = {}
        off = 0
        names = GLOBAL_NAMES if payload == 0 else SELECTED_NAMES
        base = n2 * nc if payload == 0 else npti
        for name in names:
            mult = ni if name in ("e_i", "szv_i", "sz_i") else ns if name == "e_s" else 1
            arr = values[off:off + base * mult]
            if mult > 1:
                arr = arr.reshape((base, mult), order="F")
            fields[name] = arr
            off += base * mult
        fields["_payload"] = np.asarray(payload)
        fields["_shape"] = np.asarray((nx, ny))
        frames.append(fields)
    return frames


def _read_exchange_step(stream, wanted_step: int) -> dict[str, np.ndarray]:
    require(stream.read(16) == b"NEMO_L3XCHG_001 ", "exchange magic")
    version, step, nx, ny, nc, bits = struct.unpack("=6i", stream.read(24))
    require((version, step, nc, bits) == (1, wanted_step, 1, 64), "exchange header")
    n2 = nx * ny
    nr = max(1, nx - 4) * max(1, ny - 4)
    nr1 = max(1, nx - 2) * max(1, ny - 2)
    nval = 13 * nr + 13 * nr + nr1 + 9 * n2
    v = np.fromfile(stream, np.float64, nval)
    require(v.size == nval and np.all(np.isfinite(v)), "bad exchange payload")
    # C1D's reduced field is one scalar.  Full/halo fields are skipped with
    # their declared allocation sizes (sbc_ice.F90:124-146).
    require((nr, nr1, n2) == (1, 9, 25), "unexpected C1D allocation")
    out = {name: v[i:i + 1] for i, name in enumerate(
        ("qns_ice", "qsr_ice", "qla_ice", "dqla_ice", "dqns_ice",
         "tn_ice", "alb_ice", "qml_ice", "qcn_ice", "qtr_ice_top"))}
    out["utau_ice"] = v[10:35]
    out["vtau_ice"] = v[35:60]
    out["emp_ice"] = v[60:61]
    out["evap_ice"] = v[61:62]
    out["devap_ice"] = v[62:63]
    out["qns_oce"] = v[63:64]
    out["qsr_oce"] = v[64:65]
    out["qemp_oce"] = v[65:66]
    out["qemp_ice"] = v[66:67]
    out["qevap_ice"] = v[67:68]
    out["qprec_ice"] = v[68:69]
    out["emp_oce"] = v[69:70]
    out["wndm_ice"] = v[70:71]
    out["sstfrz"] = v[71:72]
    out["rCdU_ice"] = v[72:81]
    return out


def _read_zdf_input_step(stream, wanted_step: int) -> dict[str, np.ndarray]:
    magic = stream.read(16)
    require(magic in (b"NEMO_L3ZIN_001  ", b"NEMO_L3ZIN_002  "),
            "ZDF-input magic")
    version, step, category, npti, bits, nval = struct.unpack("=6i", stream.read(24))
    expected = len(ZDF_INPUT_NAMES) + (3 if version == 2 else 0)
    require((version, step, category, npti, bits, nval)
            == (version, wanted_step, 1, 1, 64, expected)
            and version in (1, 2)
            and magic == (b"NEMO_L3ZIN_002  " if version == 2
                          else b"NEMO_L3ZIN_001  "), "ZDF-input header")
    values = np.fromfile(stream, np.float64, nval)
    require(values.size == nval and np.all(np.isfinite(values)), "bad ZDF-input payload")
    result = {name: values[i:i + 1] for i, name in enumerate(ZDF_INPUT_NAMES)}
    if version == 2:
        result["t_s"] = values[len(ZDF_INPUT_NAMES):].reshape((npti, 3), order="F")
    result["_version"] = np.asarray(version)
    return result


def _read_zdf_operands(path: Path) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            require(magic == b"NEMO_L3ZDF_001  ", "ZDF-operand magic")
            header = struct.unpack("=9i", stream.read(36))
            version, step, frame, iteration, npti, ni, ns, bits, nval = header
            require((version, step, npti, ni, ns, bits) == (1, 1, 1, 3, 3, 64),
                    f"ZDF-operand header {header}")
            require(frame in ZDF_OPERAND_REGISTRY, f"unregistered ZDF frame {frame}")
            name, expected_nval, source = ZDF_OPERAND_REGISTRY[frame]
            require(nval == expected_nval, f"ZDF-operand payload count {header}")
            values = np.fromfile(stream, np.float64, nval)
            require(values.size == nval and np.all(np.isfinite(values)),
                    "bad ZDF-operand payload")
            frames.append({"frame": frame, "name": name, "iteration": iteration,
                           "source": source, "values": values})
    require(len(frames) >= 7 and (len(frames) - 1) % 6 == 0,
            "incomplete ZDF-operand iteration registry")
    niter = (len(frames) - 1) // 6
    expected = [(0, 0)] + [(frame, iteration) for iteration in range(1, niter + 1)
                           for frame in range(1, 7)]
    require([(f["frame"], f["iteration"]) for f in frames] == expected,
            "ZDF-operand frame order/iteration count")
    convergence = [np.asarray(f["values"]) for f in frames if f["frame"] == 6]
    require(all(values[8] == 0.0 for values in convergence[:-1])
            and convergence[-1][8] == 1.0,
            "ZDF-operand convergence flags")
    return frames


def _center_global(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    nx, ny = map(int, frame["_shape"])
    idx = nx // 2 + nx * (ny // 2)
    return {k: np.asarray(v[idx]) for k, v in frame.items() if k in GLOBAL_NAMES}


def _entry_arrays(frame):
    from legoesm.ice.bitz_lipscomb import SI3ColumnArrays

    g = _center_global(frame)
    a, vi, vs = g["a_i"], g["v_i"], g["v_s"]
    # Mirror ice_thd_1d2d's zero-volume convention exactly
    # (`icethd.F90:416-434`): zero volumetric enthalpy replaces division by a
    # physically meaningless <= epsi20 category volume.  This becomes active
    # when the C1D seasonal cycle loses all snow.
    e_i = np.zeros_like(g["e_i"])
    e_s = np.zeros_like(g["e_s"])
    np.divide(g["e_i"] * 3.0, vi, out=e_i, where=vi > 1.0e-20)
    np.divide(g["e_s"] * 3.0, vs, out=e_s, where=vs > 1.0e-20)
    return SI3ColumnArrays(
        np.atleast_1d(a), np.atleast_1d(vi / a), np.atleast_1d(vs / a),
        np.atleast_1d(g["t_su"]), np.atleast_2d(e_i),
        np.atleast_2d(e_s), np.atleast_1d(g["sv_i"] / vi),
        np.atleast_2d(g["szv_i"] * 3.0 / vi), np.atleast_1d(g["oa_i"]),
    )


def _forcing(exact):
    import jax.numpy as jnp
    from legoesm.ice.bitz_lipscomb import SI3SurfaceForcing

    def val(x):
        return jnp.asarray(x, dtype=jnp.float64)

    return SI3SurfaceForcing(*(val(exact[name]) for name in ZDF_INPUT_NAMES))


def _reconstructed_forcing(exchange, entry, post_zdf, exact):
    """Reviewed pre-fix bridge, retained only for the registered owner arm."""

    qns_entry = exchange["qns_ice"] - exchange["dqns_ice"] * (
        post_zdf["t_su"] - entry.T_surface
    )
    return _forcing(exact)._replace(qns_ice=qns_entry)


def _score(rows, name, oracle, lego, *, plant=False, exact=False):
    oracle, lego = np.asarray(oracle), np.asarray(lego)
    if plant:
        oracle = oracle.copy()
        oracle.flat[0] += 1.0
    require(oracle.shape == lego.shape, f"{name}: shape {oracle.shape}!={lego.shape}")
    require(np.all(np.isfinite(oracle)) and np.all(np.isfinite(lego)), f"{name}: nonfinite")
    require(lego.dtype == np.float64, f"{name}: legoESM dtype {lego.dtype}")
    scale = max(1.0, float(np.max(np.abs(oracle))))
    error_abs = 0.0 if np.array_equal(oracle, lego) else float(np.max(np.abs(lego - oracle)))
    error = 0.0 if exact and error_abs == 0.0 else (float("inf") if exact else error_abs / scale)
    rows.append({"name": name, "status": "AT-BAR" if error <= BAR else "DEBT",
                 "absolute_max": error_abs, "scale": scale,
                 "normalized_max_abs": error, "bar": 0.0 if exact else BAR,
                 "oracle_dtype": str(oracle.dtype), "legoesm_dtype": str(lego.dtype)})


def _selected_from_arrays(state) -> dict[str, np.ndarray]:
    return {"a_i": np.asarray(state.concentration), "h_i": np.asarray(state.h_ice),
            "h_s": np.asarray(state.h_snow), "t_su": np.asarray(state.T_surface),
            "e_i": np.asarray(state.e_ice), "e_s": np.asarray(state.e_snow),
            "sz_i": np.asarray(state.S_layers)}


def _global_from_arrays(state) -> dict[str, np.ndarray]:
    a = np.asarray(state.concentration)
    h = np.asarray(state.h_ice)
    hs = np.asarray(state.h_snow)
    # Mirror `ice_thd_1d2d`'s written multiplication order
    # (`icethd.F90:440-456`).  The former volume-first reassociation alone can
    # cross the 1e-15 oracle bar at a global boundary.
    vi, vs = h * a, hs * a
    inverse_ice_layers = np.float64(1.0 / 3.0)
    inverse_snow_layers = np.float64(1.0 / 3.0)
    z = np.zeros_like(a)
    return {"a_i": a, "v_i": vi, "v_s": vs, "sv_i": np.asarray(state.S_bulk) * vi,
            "oa_i": np.asarray(state.age_volume), "t_su": np.asarray(state.T_surface),
            "a_ip": z, "v_ip": z, "v_il": z,
            "e_i": (np.asarray(state.e_ice) * h[..., None]
                    * a[..., None] * inverse_ice_layers),
            "e_s": (np.asarray(state.e_snow) * hs[..., None]
                    * a[..., None] * inverse_snow_layers),
            "szv_i": (np.asarray(state.S_layers) * vi[..., None]
                       * inverse_ice_layers)}


def run(*, plant_geometry=False, plant_stage=False, plant_selector=False):
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ice.bitz_lipscomb import (
        _si3_zdf_bl99_step,
        si3_column_step_arrays,
    )
    from legoesm.ice.c1d_omip_l3 import (
        FORCING_SHA256,
        ORACLE_V1_EXCHANGE_STREAM_SHA256,
        ORACLE_V1_ROOT,
        ORACLE_V1_ZDF_INPUT_STREAM_SHA256,
        ORACLE_V1_ZDF_OPERAND_STREAM_SHA256,
        THERMO_STREAM_SHA256,
        build_c1d_omip_l3_card,
    )
    from legoesm.ice.config import SI3ThermoConfig, validate_si3_thermo_config

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy")
    require(jax.default_backend() == "cpu", f"backend {jax.default_backend()} is not CPU")
    # This is the retained Phase-2 boundary gate and its original compact ZDF
    # operand schema.  Pin it explicitly to V1 while the canonical card points
    # new certification at scalar-math V2.
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V1_ROOT)
    if plant_selector:
        bad = card.config._replace(si3=SI3ThermoConfig(n_ice_layers=2))
        validate_si3_thermo_config(bad)  # must raise
        raise GateError("selector plant did not fire")
    root = card.oracle_root
    thd_path = root / "oracle_si3_thd_frames.bin"
    xchg_path = root / "oracle_si3_exchange_frames.bin"
    zin_path = root / "oracle_si3_zdf_inputs.bin"
    zdf_path = root / "oracle_si3_zdf_operands.bin"
    require(sha256(card.forcing_path) == FORCING_SHA256, "forcing SHA256")
    require(sha256(thd_path) == THERMO_STREAM_SHA256, "thermo stream SHA256")
    require(sha256(xchg_path) == ORACLE_V1_EXCHANGE_STREAM_SHA256,
            "V1 exchange stream SHA256")
    require(sha256(zin_path) == ORACLE_V1_ZDF_INPUT_STREAM_SHA256,
            "V1 ZDF-input stream SHA256")
    require(sha256(zdf_path) == ORACLE_V1_ZDF_OPERAND_STREAM_SHA256,
            "V1 ZDF-operand stream SHA256")
    operand_frames = _read_zdf_operands(zdf_path)

    with (
        thd_path.open("rb") as thd,
        xchg_path.open("rb") as xchg,
        zin_path.open("rb") as zin,
    ):
        frames = _read_thd_step(thd, 1)
        exchange = _read_exchange_step(xchg, 1)
        zdf_input = _read_zdf_input_step(zin, 1)

    entry = _entry_arrays(frames[0])
    rows = []
    geometry = {
        "n_ice_layers": 3, "n_snow_layers": 3, "hti": 2.0, "hts": 0.2,
        "ati": 0.9, "tsu": 270.0, "smi": 6.3,
    }
    actual = {"n_ice_layers": entry.e_ice.shape[-1], "n_snow_layers": entry.e_snow.shape[-1],
              "hti": float(entry.h_ice[0]), "hts": float(entry.h_snow[0]),
              "ati": float(entry.concentration[0]), "tsu": float(entry.T_surface[0]),
              "smi": float(entry.S_bulk[0])}
    for name, wanted in geometry.items():
        got = np.asarray([actual[name]], dtype=np.float64)
        _score(rows, f"geometry_ic.{name}", np.asarray([wanted], np.float64), got,
               plant=plant_geometry and name == "hti", exact=True)
    require(all(r["status"] == "AT-BAR" for r in rows), "geometry/IC gate")

    def compare_step(kt, step_frames, step_input, *, plant):
        """Compare one isolated oracle-entry step in registry/write order."""

        step_entry = _entry_arrays(step_frames[0])
        row_start = len(rows)
        # Entry bridge is independently round-tripped to global write format.
        for name, oracle in _center_global(step_frames[0]).items():
            lego_entry = _global_from_arrays(step_entry)[name]
            oracle_entry = (
                np.atleast_2d(oracle)
                if lego_entry.ndim == 2
                else np.atleast_1d(oracle)
            )
            _score(rows, f"kt{kt}.ENTRY.{name}", oracle_entry, lego_entry)
            if plant and name == "t_su":
                rows.pop()
                _score(
                    rows,
                    f"kt{kt}.ENTRY.{name}",
                    oracle_entry,
                    lego_entry,
                    plant=True,
                )
        if plant:
            require(
                all(r["status"] == "AT-BAR" for r in rows[row_start:]),
                "ENTRY gate",
            )

        step_entry_jax = jax.tree.map(
            lambda x: jnp.asarray(x, dtype=jnp.float64), step_entry
        )
        step_forcing = _forcing(step_input)
        trace = si3_column_step_arrays(
            step_entry_jax,
            step_forcing,
            card.dt_seconds,
            card.config.ice_constants,
        )
        stage_states = (
            trace.post_zdf,
            trace.post_dh,
            trace.post_temp1,
            trace.post_sal,
            trace.post_temp2,
            trace.post_do,
            trace.exit,
        )
        for stage_index, state in enumerate(stage_states, start=1):
            stage_name = STAGES[stage_index][1]
            lego = (
                _selected_from_arrays(state)
                if stage_index <= 5
                else _global_from_arrays(state)
            )
            oracle = (
                step_frames[stage_index]
                if stage_index <= 5
                else _center_global(step_frames[stage_index])
            )
            names = SELECTED_NAMES if stage_index <= 5 else GLOBAL_NAMES
            for name in names:
                oracle_value = np.asarray(oracle[name])
                if np.asarray(lego[name]).ndim == 2 and oracle_value.ndim == 1:
                    oracle_value = np.atleast_2d(oracle_value)
                _score(
                    rows,
                    f"kt{kt}.{stage_name}.{name}",
                    np.atleast_1d(oracle_value),
                    np.atleast_1d(lego[name]),
                )
        step_rows = rows[row_start:]
        first_step_debt = next(
            (row for row in step_rows if row["status"] == "DEBT"), None
        )
        return first_step_debt, step_entry_jax, step_forcing

    first, entry_jax, forcing = compare_step(
        1, frames, zdf_input, plant=plant_stage
    )
    owner_entry_jax = entry_jax
    steps_examined = 1

    # Continue in time only when the preceding isolated step is entirely at
    # bar.  Reopening and consuming step 1 makes the binary cursor discipline
    # explicit while keeping the common first-step path cheap.
    if first is None:
        with (
            thd_path.open("rb") as thd,
            xchg_path.open("rb") as xchg,
            zin_path.open("rb") as zin,
        ):
            _read_thd_step(thd, 1)
            _read_exchange_step(xchg, 1)
            _read_zdf_input_step(zin, 1)
            for kt in range(2, card.nsteps + 1):
                step_frames = _read_thd_step(thd, kt)
                _read_exchange_step(xchg, kt)
                step_input = _read_zdf_input_step(zin, kt)
                candidate, candidate_entry, candidate_forcing = compare_step(
                    kt,
                    step_frames,
                    step_input,
                    plant=False,
                )
                steps_examined = kt
                if candidate is not None:
                    first = candidate
                    entry_jax = candidate_entry
                    forcing = candidate_forcing
                    break

    hypothesis = "SUPERSEDED_BY_OWNER_FIX"

    # The first registered operand mismatch in the reviewed bridge is the
    # entry-time qns scalar.  NEMO updates qns before its final Picard surface
    # temperature (`icethd_zdf_bl99.F90:372-379,553-558`), so reconstructing
    # qns from the final POST_ZDF temperature is a time-level error.
    legacy_forcing = _reconstructed_forcing(
        exchange, owner_entry_jax, frames[1], zdf_input
    )
    oracle_qns = np.asarray(operand_frames[0]["values"])[13:14]
    require(np.array_equal(oracle_qns, zdf_input["qns_ice"]),
            "INIT qns disagrees with exact ZDF-input stream")
    legacy_result = _si3_zdf_bl99_step(
        owner_entry_jax.e_ice, owner_entry_jax.e_snow, owner_entry_jax.S_layers,
        owner_entry_jax.h_ice, owner_entry_jax.h_snow, owner_entry_jax.T_surface,
        legacy_forcing, card.dt_seconds, card.config.ice_constants,
    )
    arm_forcing = legacy_forcing._replace(qns_ice=jnp.asarray(oracle_qns))
    arm_result = _si3_zdf_bl99_step(
        owner_entry_jax.e_ice, owner_entry_jax.e_snow, owner_entry_jax.S_layers,
        owner_entry_jax.h_ice, owner_entry_jax.h_snow, owner_entry_jax.T_surface,
        arm_forcing, card.dt_seconds, card.config.ice_constants,
    )
    oracle_tsu = np.asarray(frames[1]["t_su"])
    legacy_tsu_error = float(np.max(np.abs(np.asarray(legacy_result.T_surface) - oracle_tsu)))
    arm_tsu_error = float(np.max(np.abs(np.asarray(arm_result.T_surface) - oracle_tsu)))
    owner_factor = legacy_tsu_error / max(arm_tsu_error, np.finfo(np.float64).tiny)
    require(owner_factor >= 100.0, "qns entry-time owner arm did not confirm")

    oracle_iterations = max(int(frame["iteration"]) for frame in operand_frames)
    iteration_rows = []
    for iteration in range(1, oracle_iterations + 1):
        partial = _si3_zdf_bl99_step(
            owner_entry_jax.e_ice, owner_entry_jax.e_snow,
            owner_entry_jax.S_layers, owner_entry_jax.h_ice,
            owner_entry_jax.h_snow, owner_entry_jax.T_surface,
            arm_forcing, card.dt_seconds, card.config.ice_constants,
            _maximum_iterations=iteration,
        )
        oracle_solution = np.asarray(next(
            frame["values"] for frame in operand_frames
            if frame["frame"] == 5 and frame["iteration"] == iteration
        ))
        for name, oracle, lego in (
            ("t_su", oracle_solution[0:1], partial.T_surface),
            ("t_s", oracle_solution[1:4][None, :], partial.T_snow),
            ("t_i", oracle_solution[4:7][None, :], partial.T_ice),
            ("qns_ice", oracle_solution[8:9], partial.qns_ice),
        ):
            _score(iteration_rows, f"iteration{iteration}.{name}", oracle, lego)

    # Scale-first one-variable arms through the private stage hook.  Each arm
    # perturbs exactly one input by 1e-6 of its dimensional scale and reports
    # the resulting normalized T_surface response; it does not tune the model.
    base = _si3_zdf_bl99_step(
        entry_jax.e_ice,
        entry_jax.e_snow,
        entry_jax.S_layers,
        entry_jax.h_ice,
        entry_jax.h_snow,
        entry_jax.T_surface,
        forcing,
        card.dt_seconds,
        card.config.ice_constants,
    )
    arms = {}
    for name, changed in {
        "qns_ice": forcing._replace(
            qns_ice=forcing.qns_ice
            + 1e-6 * np.maximum(1.0, np.abs(forcing.qns_ice))
        ),
        "dqns_ice": forcing._replace(
            dqns_ice=forcing.dqns_ice
            + 1e-6 * np.maximum(1.0, np.abs(forcing.dqns_ice))
        ),
        "t_bottom": forcing._replace(
            t_bottom=forcing.t_bottom
            + 1e-6 * np.maximum(1.0, np.abs(forcing.t_bottom))
        ),
        "snow_layer_state": None,
        "ice_layer_state": None,
    }.items():
        ei, es, f = entry_jax.e_ice, entry_jax.e_snow, forcing
        if name == "snow_layer_state":
            es = es.at[..., 0].add(
                1e-6 * np.maximum(1.0, np.abs(es[..., 0]))
            )
        elif name == "ice_layer_state":
            ei = ei.at[..., 0].add(
                1e-6 * np.maximum(1.0, np.abs(ei[..., 0]))
            )
        else:
            f = changed
        arm = _si3_zdf_bl99_step(
            ei,
            es,
            entry_jax.S_layers,
            entry_jax.h_ice,
            entry_jax.h_snow,
            entry_jax.T_surface,
            f,
            card.dt_seconds,
            card.config.ice_constants,
        )
        response = float(np.max(np.abs(np.asarray(arm.T_surface - base.T_surface))))
        scale = max(1.0, float(np.max(np.abs(np.asarray(base.T_surface)))))
        arms[name] = response / scale

    return {
        "status": "DEBT" if first is not None else "AT-BAR",
        "precision_policy": str(get_policy()),
        "backend": jax.default_backend(),
        "dtypes": sorted({r["legoesm_dtype"] for r in rows}),
        "card": {"name": card.name, "dt_seconds": card.dt_seconds, "nsteps": card.nsteps,
                 "selectors": dict(card.selector_sources)},
        "hashes": {"forcing": sha256(card.forcing_path), "thermo_stream": sha256(thd_path),
                   "exchange_stream": sha256(xchg_path),
                   "zdf_input_stream": sha256(zin_path),
                   "zdf_operand_stream": sha256(zdf_path)},
        "frame_registry": {str(i): {"name": n, "time_level": t, "source": s}
                           for i, n, t, s in STAGES},
        "sweep_steps_examined": steps_examined,
        "first_divergent_step": (
            int(first["name"].split(".", 1)[0][2:]) if first is not None else None
        ),
        "first_divergence": first,
        "preregistered_hypothesis": hypothesis,
        "zdf_operand_registry": {
            str(frame): {"name": name, "time_level": "current selected-category 1D",
                         "source": source}
            for frame, (name, _, source) in ZDF_OPERAND_REGISTRY.items()
        },
        "zdf_iterations": {
            "oracle": oracle_iterations,
            "legoesm": int(np.asarray(arm_result.iterations)[0]),
            "maximum": 200,
            "tolerance_K": 1.0e-4,
            "all_registered_iterate_fields_at_bar": all(
                row["status"] == "AT-BAR" for row in iteration_rows
            ),
            "all_registered_iterate_fields_bit_exact": all(
                row["absolute_max"] == 0.0 for row in iteration_rows
            ),
            "rows": iteration_rows,
        },
        "owner": "qns_ice entry time level in the column bridge, before the BL99 solve",
        "owner_arm": {
            "variable": "qns_ice_entry",
            "oracle_scale_W_m-2": max(1.0, float(np.max(np.abs(oracle_qns)))),
            "input_absolute_error_W_m-2": float(np.max(np.abs(
                np.asarray(legacy_forcing.qns_ice) - oracle_qns))),
            "input_normalized_error": float(np.max(np.abs(
                np.asarray(legacy_forcing.qns_ice) - oracle_qns)))
                / max(1.0, float(np.max(np.abs(oracle_qns)))),
            "legacy_POST_ZDF_t_su_absolute_error_K": legacy_tsu_error,
            "oracle_scalar_arm_POST_ZDF_t_su_absolute_error_K": arm_tsu_error,
            "improvement_factor": owner_factor,
            "verdict": "CONFIRMED",
        },
        "one_variable_arms": arms,
        "rows": rows,
        "unmeasured_beyond_first_divergence": (
            "The sweep stops at the first post-fix over-bar registry row; no "
            "later-step trajectory ownership is claimed."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plant-geometry", action="store_true")
    p.add_argument("--plant-stage", action="store_true")
    p.add_argument("--plant-selector", action="store_true")
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    result = run(plant_geometry=args.plant_geometry, plant_stage=args.plant_stage,
                 plant_selector=args.plant_selector)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    if result["status"] != "AT-BAR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
