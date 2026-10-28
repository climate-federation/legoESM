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


def _nemo_salprof(
    s_i: np.ndarray,
    sss_m: np.ndarray | float | None = None,
    *,
    zsi1: float = 4.5,
    zsi0: float = 3.5,
    rn_sinew: float = 0.75,
    rn_simin: float = 0.1,
    nlay_i: int = 3,
) -> np.ndarray:
    """Mirror `ice_var_salprof` option-2 profile (`icevar.F90:471-472,488-506`)."""

    require(sss_m is not None, "missing sea-surface salinity (sss_m)")
    s_i = np.atleast_1d(np.asarray(s_i, dtype=np.float64))
    sss_arr = np.atleast_1d(np.asarray(sss_m, dtype=np.float64))
    require(
        sss_arr.size > 0 and bool(np.all(np.isfinite(sss_arr))),
        "invalid or non-finite sea-surface salinity (sss_m)",
    )
    sss = np.broadcast_to(sss_arr, s_i.shape)
    sal_si1 = np.float64(zsi1)
    sal_si0 = np.float64(zsi0)
    sinew = np.float64(rn_sinew)
    simin = np.float64(rn_simin)
    nlay = np.float64(nlay_i)
    z1_ds = np.float64(1.0) / (sal_si1 - sal_si0)
    zalpha = np.where(
        s_i >= sal_si1,
        0.0,
        np.where(s_i <= sal_si0, 1.0, (sal_si1 - s_i) * z1_ds),
    )
    # Constant profile when SSS is too low (`icevar.F90:499`):
    zalpha = np.where(s_i >= (np.float64(0.5) * sinew * sss), 0.0, zalpha)
    # Minimum bulk salinity guard (`icevar.F90:500`):
    zalpha = np.where(
        s_i <= (nlay * np.minimum(simin, sinew * sss)),
        0.0,
        zalpha,
    )
    jk = np.arange(1, int(nlay_i) + 1, dtype=np.float64)
    r1_nlay_i = np.float64(1.0) / nlay
    return (
        zalpha[..., None]
        * s_i[..., None]
        * np.float64(2.0)
        * (jk - np.float64(0.5))
        * r1_nlay_i
        + (np.float64(1.0) - zalpha[..., None]) * s_i[..., None]
    )


def _entry_arrays(
    frame,
    *,
    sss_m: np.ndarray | float | None = None,
    _nemo_entry_conversion: bool = True,
):
    from legoesm.ice.bitz_lipscomb import SI3ColumnArrays

    require(sss_m is not None, "missing sea-surface salinity (sss_m)")
    g = _center_global(frame)
    a, vi, vs = g["a_i"], g["v_i"], g["v_s"]
    # Mirror ice_var_glo2eqv (`icevar.F90:368,374,390,412,488-506`) and
    # ice_thd_1d2d's zero-volume convention (`icethd.F90:416-434`): zero
    # volumetric enthalpy replaces division by a physically meaningless
    # <= epsi20 category volume, and non-zero volume divides before
    # multiplying by nlay_{i,s} (`e / v * 3`).
    e_i = np.zeros_like(g["e_i"])
    e_s = np.zeros_like(g["e_s"])
    s_i = np.atleast_1d(g["sv_i"] / vi)
    if _nemo_entry_conversion:
        np.divide(g["e_i"], vi, out=e_i, where=vi > 1.0e-20)
        e_i *= 3.0
        np.divide(g["e_s"], vs, out=e_s, where=vs > 1.0e-20)
        e_s *= 3.0
        sz_i = _nemo_salprof(s_i, sss_m=sss_m)
    else:
        np.divide(g["e_i"] * 3.0, vi, out=e_i, where=vi > 1.0e-20)
        np.divide(g["e_s"] * 3.0, vs, out=e_s, where=vs > 1.0e-20)
        sz_i = np.atleast_2d(g["szv_i"] * 3.0 / vi)
    return SI3ColumnArrays(
        np.atleast_1d(a), np.atleast_1d(vi / a), np.atleast_1d(vs / a),
        np.atleast_1d(g["t_su"]), np.atleast_2d(e_i),
        np.atleast_2d(e_s), s_i,
        np.atleast_2d(sz_i), np.atleast_1d(g["oa_i"]),
    )


def _forcing(exact):
    import jax.numpy as jnp
    from legoesm.ice.bitz_lipscomb import SI3SurfaceForcing

    require(
        "sss" in exact and exact["sss"] is not None,
        "missing sea-surface salinity (sss)",
    )

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


def _si3_zdf_bl99_step(
    e_ice,
    e_snow,
    S_layers,
    h_ice,
    h_snow,
    T_surface,
    forcing,
    dt: float,
    constants,
    *,
    _maximum_iterations: int = 200,
    _nemo_branch_ranges: bool = True,
    _nemo_snow_temperature_bounds: bool = True,
    _nemo_eos_order: bool = True,
    _nemo_zdf_layer_reciprocals: bool = True,
    _nemo_glo2eqv_snow_temp: bool = True,
    _nemo_source_round: bool = True,
):
    """D5 harness adapter: delegates to `bl.si3_ice_thd_zdf_bl99` unless a historical ablation arm is requested."""
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ice import bitz_lipscomb as bl

    e_ice = jnp.asarray(e_ice)
    dtype = e_ice.dtype
    h_ice = jnp.asarray(h_ice, dtype=dtype)
    h_snow = jnp.asarray(h_snow, dtype=dtype)
    S_layers = jnp.asarray(S_layers, dtype=dtype)
    e_snow = jnp.asarray(e_snow, dtype=dtype)
    T_i_old = bl.ice_temperature_from_enthalpy(
        e_ice, S_layers, constants,
        _nemo_order=_nemo_eos_order, _nemo_source_round=_nemo_source_round,
    )
    T_s_old = bl.snow_temperature_from_enthalpy(
        e_snow, constants,
        _nemo_order=_nemo_eos_order,
        _nemo_glo2eqv_snow_temp=_nemo_glo2eqv_snow_temp,
        _nemo_source_round=_nemo_source_round,
    )
    if not _nemo_snow_temperature_bounds:
        inverse_rho_snow = 1.0 / constants.rho_snow
        inverse_c_ice = 1.0 / constants.c_ice
        T_s_old = constants.T0 + (
            -e_snow * inverse_rho_snow * inverse_c_ice
            + constants.latent_fusion * inverse_c_ice
        )
    T_s_old = jnp.where((h_snow > 0.0)[..., None], T_s_old, constants.T0)
    if all((
        _nemo_branch_ranges,
        _nemo_snow_temperature_bounds,
        _nemo_eos_order,
        _nemo_zdf_layer_reciprocals,
        _nemo_glo2eqv_snow_temp,
        _nemo_source_round,
    )):
        zero = jnp.zeros_like(h_ice)
        out = bl.si3_ice_thd_zdf_bl99(
            h_i=h_ice, h_s=h_snow, t_su=jnp.asarray(T_surface, dtype=dtype),
            t_i=T_i_old, t_s=T_s_old, e_i=e_ice, e_s=e_snow, sz_i=S_layers,
            a_i=zero, t_bo=jnp.asarray(forcing.t_bottom, dtype=dtype),
            qns_ice=jnp.asarray(forcing.qns_ice, dtype=dtype),
            qsr_ice=jnp.asarray(forcing.qsr_ice, dtype=dtype),
            dqns_ice=jnp.asarray(forcing.dqns_ice, dtype=dtype),
            qtr_ice_top=jnp.asarray(forcing.qtr_ice_top, dtype=dtype),
            hfx_err_dif=zero, hfx_dif=zero, rDt_ice=dt,
            rn_cnd_s=float(constants.k_snow),
            rn_kappa_i=bl._ICE_EXTINCTION, rn_kappa_s=bl._SNOW_EXTINCTION,
            nn_qtrice=0, nn_snwfra=2, ln_cndi_P07=True, ln_cndflx=False,
            ln_virtual_itd=False, ln_zdf_chkcvg=False, constants=constants,
            _maximum_iterations=_maximum_iterations,
        )
        return bl.SI3ZDFResult(
            out.t_su, out.t_i, out.t_s, out.e_i, out.e_s, out.qns_ice,
            out.qtr_ice_bot, out.qcn_ice_top, out.qcn_ice_bot, out.n_picard,
        )
    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    T_bottom = jnp.asarray(forcing.t_bottom, dtype=dtype)
    core = bl._si3_zdf_bl99_core(
        h_i=h_ice, h_s=h_snow, t_su=jnp.asarray(T_surface, dtype=dtype),
        t_i_old=T_i_old, t_s_old=T_s_old, sz_i=S_layers, t_bo=T_bottom,
        qns_ice=jnp.asarray(forcing.qns_ice, dtype=dtype),
        qsr_ice=jnp.asarray(forcing.qsr_ice, dtype=dtype),
        dqns_ice=jnp.asarray(forcing.dqns_ice, dtype=dtype),
        qtr_ice_top=jnp.asarray(forcing.qtr_ice_top, dtype=dtype),
        dt=dt, constants=constants,
        rn_kappa_i=bl._ICE_EXTINCTION, rn_kappa_s=bl._SNOW_EXTINCTION,
        _maximum_iterations=_maximum_iterations,
        _nemo_branch_ranges=_nemo_branch_ranges,
        _nemo_zdf_layer_reciprocals=_nemo_zdf_layer_reciprocals,
        _nemo_source_round=_nemo_source_round,
    )
    Tsu, Ti, Ts = core.t_su, core.t_i, core.t_s
    e_i = bl.ice_enthalpy_from_temperature(
        Ti, S_layers, constants,
        _nemo_order=_nemo_eos_order, _nemo_source_round=_nemo_source_round,
    )
    e_s = bl.snow_enthalpy_from_temperature(
        Ts, constants, _nemo_source_round=_nemo_source_round
    )
    if _nemo_zdf_layer_reciprocals:
        qcn_bottom, qcn_top = bl._zdf_interface_fluxes(core, T_bottom, sr)
    else:
        nlay_i, nlay_s = Ti.shape[-1], Ts.shape[-1]
        dz_i = jnp.maximum(h_ice, bl._H_MIN) / nlay_i
        dz_s = jnp.maximum(h_snow, bl._H_MIN) / nlay_s
        snow_present = h_snow > 0.0
        k_top_i = bl.p07_conductivity(
            Ti[..., 0], S_layers[..., 0], constants,
            _nemo_source_round=_nemo_source_round,
        ) / dz_i
        k_base_i = bl.p07_conductivity(
            T_bottom, S_layers[..., -1], constants,
            _nemo_source_round=_nemo_source_round,
        ) / dz_i
        k_top = jnp.where(snow_present, constants.k_snow / dz_s, k_top_i)
        T_first = jnp.where(snow_present, Ts[..., 0], Ti[..., 0])
        qcn_top = 2.0 * k_top * (Tsu - T_first)
        qcn_bottom = 2.0 * k_base_i * (Ti[..., -1] - T_bottom)
    return bl.SI3ZDFResult(
        Tsu, Ti, Ts, e_i, e_s, core.qns_ice, core.qtr_ice_bot, qcn_top,
        qcn_bottom, core.iterations
    )


def _piecewise_remap(
    thickness,
    enthalpy,
    new_total,
    n_layers: int = 3,
):
    """Conservative constant-cell remap retained as the non-NEMO ablation arm."""
    import jax.numpy as jnp
    from legoesm.ice import bitz_lipscomb as bl

    old_lo = jnp.cumsum(thickness, axis=-1) - thickness
    old_hi = jnp.cumsum(thickness, axis=-1)
    dz_new = new_total / n_layers
    idx = jnp.arange(n_layers, dtype=new_total.dtype)
    new_lo = dz_new[..., None] * idx
    new_hi = dz_new[..., None] * (idx + 1.0)
    overlap = jnp.maximum(
        0.0,
        jnp.minimum(new_hi[..., :, None], old_hi[..., None, :])
        - jnp.maximum(new_lo[..., :, None], old_lo[..., None, :]),
    )
    content = jnp.sum(overlap * enthalpy[..., None, :], axis=-1)
    return jnp.where(dz_new[..., None] > bl._EPS10, content / dz_new[..., None], 0.0)


def _nemo_snow_enthalpy_remap(
    thickness,
    enthalpy,
    *,
    _nemo_source_round: bool = True,
):
    """D5 3-layer NEMO ``snw_ent`` cumulative remap delegating to `bl._cumulative_remap`."""
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ice import bitz_lipscomb as bl

    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    h_arr, e_arr = jnp.asarray(thickness), jnp.asarray(enthalpy)
    n_seg = h_arr.shape[-1]
    return bl._cumulative_remap(
        [h_arr[..., k] for k in range(n_seg)],
        [sr(e_arr[..., k] * h_arr[..., k]) for k in range(n_seg)],
        3,
        sr,
    )


def _nemo_ice_enthalpy_remap(
    thickness,
    enthalpy=None,
    *,
    content=None,
    _nemo_source_round: bool = True,
):
    """D5 3-layer NEMO ``ice_var_vremap`` delegating to `bl._cumulative_remap`."""
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ice import bitz_lipscomb as bl

    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    h_arr = jnp.asarray(thickness)
    pts_old = (
        jnp.asarray(content)
        if content is not None
        else sr(h_arr * jnp.asarray(enthalpy))
    )
    n_seg = h_arr.shape[-1]
    return bl._cumulative_remap(
        [h_arr[..., k] for k in range(n_seg)],
        [pts_old[..., k] for k in range(n_seg)],
        3,
        sr,
    )


def _call_prod_dh(state, zdf, forcing, dt, constants, *, _snow_ice_salinity: bool = True):
    """Delegate C1D DH step to production `bl.si3_ice_thd_dh`."""
    import jax.numpy as jnp
    from legoesm.ice import bitz_lipscomb as bl

    a = state.concentration
    dtype = zdf.e_ice.dtype
    zero = jnp.zeros_like(state.h_ice)
    out = bl.si3_ice_thd_dh(
        h_i=state.h_ice, h_s=state.h_snow, t_su=zdf.T_surface, a_i=a, at_i=a,
        t_i=zdf.T_ice, t_s=zdf.T_snow, e_i=zdf.e_ice, e_s=zdf.e_snow,
        sz_i=state.S_layers, s_i=state.S_bulk, t_bo=forcing.t_bottom,
        sss=forcing.sss, sst=forcing.t_bottom - constants.T0,
        qns_ice=zdf.qns_ice, qsr_ice=forcing.qsr_ice,
        qtr_ice_top=forcing.qtr_ice_top, qcn_ice_top=zdf.qcn_ice_top,
        qcn_ice_bot=zdf.qcn_ice_bottom, qtr_ice_bot=zdf.qtr_ice_bottom,
        qsb_ice_bot=forcing.qsb_ice_bottom, fhld=forcing.fhld,
        frq_m=jnp.asarray(forcing.frq_m, dtype=dtype),
        sprecip=forcing.snow_precipitation, qprec_ice=forcing.qprec_ice,
        evap_ice=forcing.evaporation,
        fluxes=bl.SI3DHFluxes(*([zero] * len(bl.SI3DHFluxes._fields))),
        rDt_ice=dt, rn_snwblow=bl._SNOW_BLOW_FRACTION,
        rn_sinew=bl._NEW_ICE_SALINITY_FRACTION, nn_icesal=2,
        ln_cndflx=False, ln_cndemulate=False, constants=constants,
    )
    s_i = out.s_i
    if not _snow_ice_salinity:
        s_i = jnp.where(out.dh.dh_snowice > 0.0, state.S_bulk, s_i)
    return bl.SI3ColumnArrays(out.a_i, out.h_i, out.h_s, out.t_su, out.e_i, out.e_s,
                              s_i, state.S_layers, state.age_volume)


def _dh_step(
    state,
    zdf,
    forcing,
    dt: float,
    constants,
    *,
    _snow_deposition: bool = True,
    _surface_melt: bool = True,
    _basal_melt: bool = True,
    _nemo_basal_layer_loop: bool = True,
    _snow_ice_salinity: bool = True,
    _nemo_snow_sublimation_order: bool = True,
    _nemo_snow_remap: bool = True,
    _nemo_ice_remap: bool = True,
    _nemo_dh_ice_density_reciprocal: bool = True,
    _nemo_dh_ice_content_accumulation: bool = True,
    _nemo_dh_snow_update: bool = True,
    _nemo_source_round: bool = True,
):
    """D5 harness thickness sequence: delegates to `bl.si3_ice_thd_dh` unless an ablation arm is requested."""
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ice import bitz_lipscomb as bl
    from legoesm.ice.snow import snow_ice_flooding

    base_flags = (
        _snow_deposition, _surface_melt, _basal_melt, _nemo_basal_layer_loop,
        _nemo_snow_sublimation_order, _nemo_snow_remap, _nemo_ice_remap,
        _nemo_dh_ice_density_reciprocal, _nemo_source_round,
    )
    if all(base_flags) and _nemo_dh_ice_content_accumulation and _nemo_dh_snow_update:
        return _call_prod_dh(
            state, zdf, forcing, dt, constants,
            _snow_ice_salinity=_snow_ice_salinity,
        )
    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    a = state.concentration
    h_i0, h_s0 = state.h_ice, state.h_snow
    snow_partition = 1.0 - jnp.power(
        jnp.maximum(1.0 - a, 0.0), bl._SNOW_BLOW_FRACTION
    )
    inverse_snow_density = 1.0 / constants.rho_snow
    if _nemo_snow_sublimation_order:
        h_precip = (
            snow_partition * forcing.snow_precipitation * dt
            * inverse_snow_density / jnp.maximum(a, bl._EPS10)
        )
        old_snow_layer = h_s0 * (1.0 / 3.0)
    else:
        h_precip = snow_partition * forcing.snow_precipitation * dt / (
            jnp.maximum(a, bl._EPS10) * constants.rho_snow
        )
        old_snow_layer = h_s0 / 3.0
    hseg_s = jnp.concatenate(
        [h_precip[..., None],
         jnp.broadcast_to(old_snow_layer[..., None], state.e_snow.shape)],
        axis=-1,
    )
    h_s_work = h_s0 + h_precip
    precip_enthalpy = jnp.where(
        forcing.snow_precipitation > 0.0,
        jnp.maximum(0.0, -forcing.qprec_ice),
        0.0,
    )
    deposited_enthalpy = constants.rho_snow * (
        constants.latent_fusion
        - constants.c_ice * (zdf.T_surface - constants.T0)
    )
    eseg_s = jnp.concatenate(
        [precip_enthalpy[..., None], zdf.e_snow], axis=-1
    )
    if _nemo_snow_sublimation_order:
        evaporation = jnp.where(
            _snow_deposition | (forcing.evaporation >= 0.0),
            forcing.evaporation,
            0.0,
        )
        snow_delta = jnp.maximum(
            -evaporation * inverse_snow_density * dt,
            -h_s_work,
        )
        remaining_mass = evaporation * dt + snow_delta * constants.rho_snow
        eseg_s = eseg_s.at[..., 0].set(jnp.where(
            (snow_delta > 0.0) & (eseg_s[..., 0] == 0.0),
            deposited_enthalpy,
            eseg_s[..., 0],
        ))
        kept = []
        for k in range(4):
            delta = jnp.maximum(-hseg_s[..., k], snow_delta)
            h_s_work = jnp.maximum(0.0, h_s_work + delta)
            kept.append(jnp.maximum(0.0, hseg_s[..., k] + delta))
            snow_delta = jnp.minimum(snow_delta - delta, 0.0)
        hseg_s = jnp.stack(kept, axis=-1)
    else:
        deposition = jnp.where(
            _snow_deposition,
            jnp.maximum(-forcing.evaporation * dt / constants.rho_snow, 0.0),
            0.0,
        )
        eseg_s = eseg_s.at[..., 0].set(jnp.where(
            (deposition > 0.0) & (eseg_s[..., 0] == 0.0),
            deposited_enthalpy,
            eseg_s[..., 0],
        ))
        hseg_s = hseg_s.at[..., 0].add(deposition)
        remaining_mass = jnp.maximum(forcing.evaporation * dt, 0.0)
        kept = []
        for k in range(4):
            remove = jnp.minimum(
                remaining_mass / constants.rho_snow, hseg_s[..., k]
            )
            kept.append(hseg_s[..., k] - remove)
            remaining_mass = jnp.maximum(
                remaining_mass - remove * constants.rho_snow, 0.0
            )
        hseg_s = jnp.stack(kept, axis=-1)
        h_s_work = jnp.sum(hseg_s, axis=-1)

    qml = jnp.where(
        zdf.T_surface >= constants.T0,
        zdf.qns_ice + forcing.qsr_ice - forcing.qtr_ice_top - zdf.qcn_ice_top,
        0.0,
    )
    surface_energy = jnp.where(_surface_melt, jnp.maximum(qml * dt, 0.0), 0.0)
    melted_snow = []
    for k in range(4):
        active = (hseg_s[..., k] > 0.0) & (surface_energy > 0.0)
        delta = jnp.where(
            active,
            jnp.maximum(
                -surface_energy / jnp.maximum(eseg_s[..., k], 1.0e-20),
                -hseg_s[..., k],
            ),
            0.0,
        )
        melted_snow.append(hseg_s[..., k] + delta)
        h_s_work = jnp.maximum(0.0, h_s_work + delta)
        surface_energy = jnp.maximum(
            surface_energy + sr(delta * eseg_s[..., k]), 0.0
        )
    hseg_s = jnp.stack(melted_snow, axis=-1)

    inverse_ice_density = 1.0 / constants.rho_ice
    ice_density_divisor = jax.lax.optimization_barrier(
        jnp.asarray(constants.rho_ice, dtype=zdf.e_ice.dtype)
    )
    old_dz = h_i0 / 3.0
    old_h = jnp.broadcast_to(old_dz[..., None], state.e_ice.shape)
    melt_temperature = (
        constants.T0 - constants.liquidus_slope * state.S_layers
    )
    for k in range(3):
        internal_melt = _surface_melt & (
            zdf.T_ice[..., k] >= melt_temperature[..., k]
        )
        ice_specific = (
            sr(-zdf.e_ice[..., k] * inverse_ice_density)
            if _nemo_dh_ice_density_reciprocal
            else sr(-zdf.e_ice[..., k] / ice_density_divisor)
        )
        water_specific = sr(
            constants.c_ocean
            * (-constants.liquidus_slope * state.S_layers[..., k])
        )
        energy_difference = ice_specific - water_specific
        surface_remove = jnp.where(
            internal_melt,
            old_h[..., k],
            jnp.where(
                _surface_melt,
                jnp.minimum(
                    jnp.maximum(
                        surface_energy
                        / jnp.maximum(
                            -energy_difference * constants.rho_ice, 1.0e-20
                        ),
                        0.0,
                    ),
                    old_h[..., k],
                ),
                0.0,
            ),
        )
        old_h = old_h.at[..., k].add(-surface_remove)
        surface_energy = jnp.where(
            internal_melt,
            surface_energy,
            jnp.maximum(
                surface_energy
                - surface_remove * (-energy_difference * constants.rho_ice),
                0.0,
            ),
        )
        sublimation = jnp.minimum(
            remaining_mass / constants.rho_ice, old_h[..., k]
        )
        old_h = old_h.at[..., k].add(-sublimation)
        remaining_mass = jnp.maximum(
            remaining_mass - sublimation * constants.rho_ice, 0.0
        )

    zf = (
        zdf.qcn_ice_bottom
        + forcing.qsb_ice_bottom
        + forcing.fhld
        + zdf.qtr_ice_bottom * getattr(forcing, "frq_m", 1.0)
    )
    S_new = bl._NEW_ICE_SALINITY_FRACTION * forcing.sss
    Tm = -constants.liquidus_slope * S_new
    Tnew_c = forcing.t_bottom - constants.T0
    e_i_specific = (
        constants.c_ice * (Tnew_c - Tm)
        - constants.latent_fusion
        * jnp.maximum(0.0, 1.0 - Tm / jnp.minimum(Tnew_c, -bl._EPS10))
        + constants.c_ocean * Tm
    )
    e_water = constants.c_ocean * (forcing.t_bottom - constants.T0)
    dE = e_i_specific - e_water
    dh_growth = dt * jnp.maximum(0.0, zf / (dE * constants.rho_ice))
    basal_energy = dt * jnp.maximum(zf, 0.0)
    if _nemo_basal_layer_loop:
        for k in (2, 1, 0):
            internal_melt = (zf > 0.0) & (
                zdf.T_ice[..., k] >= melt_temperature[..., k]
            )
            ice_specific = (
                sr(-zdf.e_ice[..., k] * inverse_ice_density)
                if _nemo_dh_ice_density_reciprocal
                else sr(-zdf.e_ice[..., k] / ice_density_divisor)
            )
            water_specific = sr(
                constants.c_ocean
                * (-constants.liquidus_slope * state.S_layers[..., k])
            )
            energy_difference = ice_specific - water_specific
            basal_remove = jnp.where(
                _basal_melt,
                jnp.where(
                    internal_melt,
                    old_h[..., k],
                    jnp.minimum(
                        jnp.maximum(
                            basal_energy
                            / jnp.maximum(
                                -energy_difference * constants.rho_ice,
                                1.0e-20,
                            ),
                            0.0,
                        ),
                        old_h[..., k],
                    ),
                ),
                0.0,
            )
            old_h = old_h.at[..., k].add(-basal_remove)
            basal_energy = jnp.where(
                internal_melt,
                basal_energy,
                jnp.maximum(
                    basal_energy
                    - basal_remove
                    * (-energy_difference * constants.rho_ice),
                    0.0,
                ),
            )
        h_i_pre_flood = jnp.sum(old_h, axis=-1) + dh_growth
    else:
        bottom_specific = (
            sr(-zdf.e_ice[..., -1] * inverse_ice_density)
            if _nemo_dh_ice_density_reciprocal
            else sr(-zdf.e_ice[..., -1] / ice_density_divisor)
        )
        bottom_water = sr(
            constants.c_ocean
            * (-constants.liquidus_slope * state.S_layers[..., -1])
        )
        bottom_difference = bottom_specific - bottom_water
        legacy_remove = jnp.where(
            _basal_melt,
            jnp.minimum(
                jnp.maximum(
                    -basal_energy / (bottom_difference * constants.rho_ice),
                    0.0,
                ),
                jnp.sum(old_h, axis=-1),
            ),
            0.0,
        )
        h_i_pre_flood = jnp.sum(old_h, axis=-1) + dh_growth - legacy_remove
        old_h = old_h.at[..., -1].set(
            jnp.maximum(old_h[..., -1] - legacy_remove, 0.0)
        )

    hseg_i = jnp.concatenate(
        [jnp.zeros_like(h_i0)[..., None], old_h,
         dh_growth[..., None]], axis=-1
    )
    e_new_ice = -e_i_specific * constants.rho_ice
    eseg_i = jnp.concatenate(
        [jnp.zeros_like(h_i0)[..., None], zdf.e_ice,
         e_new_ice[..., None]], axis=-1
    )

    h_i, h_s_final, flood = snow_ice_flooding(
        h_i_pre_flood,
        h_s_work,
        constants.rho_ice,
        constants.rho_snow,
        constants.rho_ocean,
    )
    hseg_i = hseg_i.at[..., 0].set(flood)
    flood_remaining = flood
    flood_content = (
        (constants.rho_snow - constants.rho_ice)
        * flood
        * constants.c_ocean
        * (forcing.t_bottom - constants.T0)
    )
    flooded_snow = list(jnp.moveaxis(hseg_s, -1, 0))
    for k in (3, 2, 1, 0):
        amount = jnp.minimum(flood_remaining, flooded_snow[k])
        flooded_snow[k] = jnp.maximum(flooded_snow[k] - amount, 0.0)
        flood_content = flood_content + amount * eseg_s[..., k]
        flood_remaining = jnp.maximum(flood_remaining - amount, 0.0)
    hseg_s = jnp.stack(flooded_snow, axis=-1)
    eseg_i = eseg_i.at[..., 0].set(jnp.where(
        flood > 0.0, flood_content / jnp.maximum(flood, 1.0e-20), 0.0
    ))
    e_i = (
        _nemo_ice_enthalpy_remap(
            hseg_i,
            eseg_i,
            _nemo_source_round=_nemo_source_round,
        )
        if _nemo_ice_remap
        else _piecewise_remap(hseg_i, eseg_i, h_i)
    )
    e_s = (
        _nemo_snow_enthalpy_remap(
            hseg_s, eseg_s, _nemo_source_round=_nemo_source_round
        )
        if _nemo_snow_remap
        else _piecewise_remap(hseg_s, eseg_s, h_s_final)
    )

    snow_ice_salinity = (
        forcing.sss
        * (constants.rho_ice - constants.rho_snow)
        / constants.rho_ice
    )
    snow_ice_salt_change = jnp.where(
        _snow_ice_salinity,
        (snow_ice_salinity - state.S_bulk) * flood,
        0.0,
    )
    S_bulk = state.S_bulk + (
        snow_ice_salt_change + (S_new - state.S_bulk) * dh_growth
    ) / jnp.maximum(h_i, bl._EPS10)
    legacy = bl.SI3ColumnArrays(a, h_i, h_s_final, zdf.T_surface, e_i, e_s,
                                S_bulk, state.S_layers, state.age_volume)
    if all(base_flags) and (_nemo_dh_ice_content_accumulation or _nemo_dh_snow_update):
        prod = _call_prod_dh(
            state, zdf, forcing, dt, constants,
            _snow_ice_salinity=_snow_ice_salinity,
        )
        return (
            prod._replace(h_snow=legacy.h_snow, e_snow=legacy.e_snow)
            if _nemo_dh_ice_content_accumulation
            else legacy._replace(h_snow=prod.h_snow, e_snow=prod.e_snow)
        )
    return legacy


def _si3_column_step_arrays_ablation(
    state,
    forcing,
    dt: float,
    constants,
    *,
    _snow_deposition: bool = True,
    _surface_melt: bool = True,
    _zdf_branch_ranges: bool = True,
    _basal_melt: bool = True,
    _nemo_snow_temperature_bounds: bool = True,
    _nemo_basal_layer_loop: bool = True,
    _snow_ice_salinity: bool = True,
    _nemo_eos_order: bool = True,
    _nemo_snow_sublimation_order: bool = True,
    _nemo_snow_remap: bool = True,
    _nemo_ice_remap: bool = True,
    _nemo_dh_ice_density_reciprocal: bool = True,
    _nemo_dh_ice_content_accumulation: bool = True,
    _nemo_dh_snow_update: bool = True,
    _nemo_salprof_factoring: bool = True,
    _nemo_zdf_layer_reciprocals: bool = True,
    _nemo_glo2eqv_snow_temp: bool = True,
    _nemo_source_round: bool = True,
    si3=None,
):
    """D5 harness step wrapper: delegates to production `si3_column_step_arrays` unless an ablation arm is requested."""
    import jax.numpy as jnp
    from legoesm import constants as canonical_constants
    from legoesm.ice import bitz_lipscomb as bl

    if si3 is None or getattr(si3, "salinity_scheme", None) is None:
        raise ValueError("_si3_column_step_arrays_ablation requires an explicit SI3ThermoConfig (si3); no default")
    if all((
        _snow_deposition,
        _surface_melt,
        _zdf_branch_ranges,
        _basal_melt,
        _nemo_snow_temperature_bounds,
        _nemo_basal_layer_loop,
        _snow_ice_salinity,
        _nemo_eos_order,
        _nemo_snow_sublimation_order,
        _nemo_snow_remap,
        _nemo_ice_remap,
        _nemo_dh_ice_density_reciprocal,
        _nemo_dh_ice_content_accumulation,
        _nemo_dh_snow_update,
        _nemo_salprof_factoring,
        _nemo_zdf_layer_reciprocals,
        _nemo_glo2eqv_snow_temp,
        _nemo_source_round,
    )):
        return bl.si3_column_step_arrays(state, forcing, dt, constants, si3=si3)

    zdf = _si3_zdf_bl99_step(
        state.e_ice, state.e_snow, state.S_layers, state.h_ice, state.h_snow,
        state.T_surface, forcing, dt, constants,
        _nemo_branch_ranges=_zdf_branch_ranges,
        _nemo_snow_temperature_bounds=_nemo_snow_temperature_bounds,
        _nemo_eos_order=_nemo_eos_order,
        _nemo_zdf_layer_reciprocals=_nemo_zdf_layer_reciprocals,
        _nemo_glo2eqv_snow_temp=_nemo_glo2eqv_snow_temp,
        _nemo_source_round=_nemo_source_round,
    )
    post_zdf = state._replace(T_surface=zdf.T_surface, e_ice=zdf.e_ice,
                              e_snow=zdf.e_snow)
    post_dh = _dh_step(
        post_zdf, zdf, forcing, dt, constants,
        _snow_deposition=_snow_deposition,
        _surface_melt=_surface_melt,
        _basal_melt=_basal_melt,
        _nemo_basal_layer_loop=_nemo_basal_layer_loop,
        _snow_ice_salinity=_snow_ice_salinity,
        _nemo_snow_sublimation_order=_nemo_snow_sublimation_order,
        _nemo_snow_remap=_nemo_snow_remap,
        _nemo_ice_remap=_nemo_ice_remap,
        _nemo_dh_ice_density_reciprocal=_nemo_dh_ice_density_reciprocal,
        _nemo_dh_ice_content_accumulation=_nemo_dh_ice_content_accumulation,
        _nemo_dh_snow_update=_nemo_dh_snow_update,
        _nemo_source_round=_nemo_source_round,
    )
    zero = jnp.zeros_like(post_dh.h_ice)
    sal_out = bl.si3_ice_thd_sal(
        h_i=post_dh.h_ice, t_su=post_dh.T_surface, t_i=zdf.T_ice,
        sz_i=post_dh.S_layers, s_i=post_dh.S_bulk, t_bo=forcing.t_bottom,
        sss=forcing.sss, a_i=post_dh.concentration, dh_i_sum=zero,
        dh_s_sum=zero, sfx_bri=zero, sfx_res=zero,
        namelist={
            "nn_icesal": int(si3.salinity_scheme),
            "rn_simin": bl._MINIMUM_ICE_SALINITY,
            "rn_sinew": bl._NEW_ICE_SALINITY_FRACTION,
            "rn_sal_gd": bl._DRAINAGE_TARGET,
            "rn_time_gd": bl._DRAINAGE_TIMESCALE,
            "rn_sal_fl": bl._FLUSHING_TARGET,
            "rn_time_fl": bl._FLUSHING_TIMESCALE,
        },
        rDt_ice=float(dt), r1_Dt_ice=1.0 / float(dt),
        r1_nlay_i=1.0 / post_dh.S_layers.shape[-1],
        rho_i=constants.rho_ice, rho_s=constants.rho_snow,
        rho_w=canonical_constants.rho_water, gravity=canonical_constants.g_nemo,
        constants=constants,
    )
    S_bulk = sal_out.smin_drain
    S_layers = (
        sal_out.sz_i
        if (_nemo_source_round and _nemo_salprof_factoring)
        else bl.option2_salinity_profile(
            S_bulk,
            forcing.sss,
            _nemo_source_round=_nemo_source_round,
            _nemo_salprof_factoring=_nemo_salprof_factoring,
        )
    )
    return bl.si3_column_step_arrays(
        state, forcing, dt, constants, si3=si3,
        _zdf_dh_override=(post_zdf, post_dh, S_bulk, S_layers),
    )


def run(*, plant_geometry=False, plant_stage=False, plant_selector=False):
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ice.bitz_lipscomb import si3_column_step_arrays
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

    entry = _entry_arrays(frames[0], sss_m=zdf_input.get("sss"))
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

        step_entry = _entry_arrays(step_frames[0], sss_m=step_input.get("sss"))
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
            si3=card.config.si3,
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
