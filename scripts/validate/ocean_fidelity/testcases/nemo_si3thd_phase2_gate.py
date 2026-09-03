#!/usr/bin/env python3
"""Phase-2 legoESM column gate against the accepted C1D_OMIP_L3 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import jax
import netCDF4
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


def _center_global(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    nx, ny = map(int, frame["_shape"])
    idx = nx // 2 + nx * (ny // 2)
    return {k: np.asarray(v[idx]) for k, v in frame.items() if k in GLOBAL_NAMES}


def _entry_arrays(frame):
    from legoesm.ice.bitz_lipscomb import SI3ColumnArrays

    g = _center_global(frame)
    a, vi, vs = g["a_i"], g["v_i"], g["v_s"]
    return SI3ColumnArrays(
        np.atleast_1d(a), np.atleast_1d(vi / a), np.atleast_1d(vs / a),
        np.atleast_1d(g["t_su"]), np.atleast_2d(g["e_i"] * 3.0 / vi),
        np.atleast_2d(g["e_s"] * 3.0 / vs), np.atleast_1d(g["sv_i"] / vi),
        np.atleast_2d(g["szv_i"] * 3.0 / vi), np.atleast_1d(g["oa_i"]),
    )


def _teos10_freezing_temperature(sss):
    z = np.sqrt(np.abs(sss) / 35.16504)
    return (((((1.46873e-3 * z - 9.64972e-3) * z + 2.28348e-2) * z
               - 3.12775e-2) * z + 2.07679e-2) * z - 5.87701e-2) * sss + 273.15


def _forcing(exchange, entry, post_zdf, sf, sss):
    import jax.numpy as jnp
    from legoesm.ice.bitz_lipscomb import SI3SurfaceForcing

    # qns_ice is dumped after ZDF.  The Picard update telescopes exactly from
    # entry to POST_ZDF (`icethd_zdf_bl99.F90:189-194,372-377`).
    qns_entry = exchange["qns_ice"] - exchange["dqns_ice"] * (
        post_zdf["t_su"] - entry.T_surface
    )
    def val(x):
        return jnp.asarray(x, dtype=jnp.float64)

    zeros = np.zeros((1,), np.float64)
    return SI3SurfaceForcing(
        val(qns_entry), val(exchange["qsr_ice"]), val(exchange["dqns_ice"]),
        val(exchange["qtr_ice_top"]), val(_teos10_freezing_temperature(sss)),
        val(sss), val(exchange["evap_ice"]), val(sf),
        val(exchange["qprec_ice"]), val(zeros), val(zeros), val(zeros), val(zeros),
    )


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
    vi, vs = a * h, a * hs
    z = np.zeros_like(a)
    return {"a_i": a, "v_i": vi, "v_s": vs, "sv_i": np.asarray(state.S_bulk) * vi,
            "oa_i": np.asarray(state.age_volume), "t_su": np.asarray(state.T_surface),
            "a_ip": z, "v_ip": z, "v_il": z,
            "e_i": np.asarray(state.e_ice) * vi[..., None] / 3.0,
            "e_s": np.asarray(state.e_snow) * vs[..., None] / 3.0,
            "szv_i": np.asarray(state.S_layers) * vi[..., None] / 3.0}


def run(*, plant_geometry=False, plant_stage=False, plant_selector=False):
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ice.bitz_lipscomb import (
        _si3_zdf_bl99_step,
        si3_column_step_arrays,
    )
    from legoesm.ice.c1d_omip_l3 import (
        EXCHANGE_STREAM_SHA256,
        FORCING_SHA256,
        THERMO_STREAM_SHA256,
        build_c1d_omip_l3_card,
    )
    from legoesm.ice.config import SI3ThermoConfig, validate_si3_thermo_config

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy")
    require(jax.default_backend() == "cpu", f"backend {jax.default_backend()} is not CPU")
    card = build_c1d_omip_l3_card()
    if plant_selector:
        bad = card.config._replace(si3=SI3ThermoConfig(n_ice_layers=2))
        validate_si3_thermo_config(bad)  # must raise
        raise GateError("selector plant did not fire")
    root = card.oracle_root
    thd_path = root / "oracle_si3_thd_frames.bin"
    xchg_path = root / "oracle_si3_exchange_frames.bin"
    require(sha256(card.forcing_path) == FORCING_SHA256, "forcing SHA256")
    require(sha256(thd_path) == THERMO_STREAM_SHA256, "thermo stream SHA256")
    require(sha256(xchg_path) == EXCHANGE_STREAM_SHA256, "exchange stream SHA256")

    with (
        thd_path.open("rb") as thd,
        xchg_path.open("rb") as xchg,
        netCDF4.Dataset(card.forcing_path) as era5,
    ):
        frames = _read_thd_step(thd, 1)
        exchange = _read_exchange_step(xchg, 1)
        sf = np.asarray(era5["sf"][0]).reshape(1).astype(np.float64)
        sss = np.asarray(era5["sss"][0]).reshape(1).astype(np.float64)

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

    def compare_step(kt, step_frames, step_exchange, step_sf, step_sss, *, plant):
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
        step_forcing = _forcing(
            step_exchange,
            step_entry_jax,
            step_frames[1],
            step_sf,
            step_sss,
        )
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
        1, frames, exchange, sf, sss, plant=plant_stage
    )
    steps_examined = 1

    # Continue in time only when the preceding isolated step is entirely at
    # bar.  Reopening and consuming step 1 makes the binary cursor discipline
    # explicit while keeping the common first-step path cheap.
    if first is None:
        with (
            thd_path.open("rb") as thd,
            xchg_path.open("rb") as xchg,
            netCDF4.Dataset(card.forcing_path) as era5,
        ):
            _read_thd_step(thd, 1)
            _read_exchange_step(xchg, 1)
            for kt in range(2, card.nsteps + 1):
                step_frames = _read_thd_step(thd, kt)
                step_exchange = _read_exchange_step(xchg, kt)
                step_sf = np.asarray(era5["sf"][kt - 1]).reshape(1).astype(np.float64)
                step_sss = np.asarray(era5["sss"][kt - 1]).reshape(1).astype(np.float64)
                candidate, candidate_entry, candidate_forcing = compare_step(
                    kt,
                    step_frames,
                    step_exchange,
                    step_sf,
                    step_sss,
                    plant=False,
                )
                steps_examined = kt
                if candidate is not None:
                    first = candidate
                    entry_jax = candidate_entry
                    forcing = candidate_forcing
                    break

    hypothesis = (
        "CONFIRMED"
        if first is not None and first["name"].startswith("kt1.POST_ZDF.")
        else "REFUTED"
    )

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
                   "exchange_stream": sha256(xchg_path)},
        "frame_registry": {str(i): {"name": n, "time_level": t, "source": s}
                           for i, n, t, s in STAGES},
        "sweep_steps_examined": steps_examined,
        "first_divergent_step": (
            int(first["name"].split(".", 1)[0][2:]) if first is not None else None
        ),
        "first_divergence": first,
        "preregistered_hypothesis": hypothesis,
        "owner": (
            "BL99 snow/surface tridiagonal solve "
            "(icethd_zdf_bl99.F90:307-590); physical selectors and inputs "
            "are pinned, but the remaining operation-order discriminator is unresolved"
        ),
        "one_variable_arms": arms,
        "rows": rows,
        "unmeasured_beyond_first_divergence": (
            "No kt>1 trajectory ownership or claim is made after kt=1 "
            "POST_ZDF debt."
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
