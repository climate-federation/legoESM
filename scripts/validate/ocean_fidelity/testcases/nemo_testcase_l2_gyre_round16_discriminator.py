#!/usr/bin/env python3
"""Round-16 no-physics-change scalar-math discriminator for NEMO GYRE.

This probe transcribes the executed, preprocessed NEMO statements with an
explicit binary64 result after every source operation.  Scalar transcendental
calls use the same ``libm.so.6`` entry points as the scalar-math oracle.  It
does not call a second ocean implementation and does not mutate model code.

Oracle references:

* ``BLD/ppsrc/nemo/traqsr.f90:615-645`` (two-band RK3 path), and
* ``BLD/ppsrc/nemo/usrdef_sbc.f90:101-201`` (all seasonal SBC statements).
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from nemo_testcase_l2_gyre_phase3_gate import read_qsr_stage3, read_tracer_stage3
from nemo_testcase_l2_gyre_round15_eligibility import _candidate, _read_sbc
from legoesm.ocean.fidelity.provenance import worktree_stamp

F64 = np.float64
U64 = np.uint64
LIBM = ctypes.CDLL("libm.so.6")


def _libm(name: str, value) -> np.float64:
    function = getattr(LIBM, name)
    function.argtypes = (ctypes.c_double,)
    function.restype = ctypes.c_double
    return F64(function(float(F64(value))))


def _add(left, right):
    return F64(F64(left) + F64(right))


def _sub(left, right):
    return F64(F64(left) - F64(right))


def _mul(left, right):
    return F64(F64(left) * F64(right))


def _div(left, right):
    return F64(F64(left) / F64(right))


def _neg(value):
    return F64(-F64(value))


def _ordered_bits(values) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float64).view(np.uint64)
    sign = bits >> U64(63)
    return np.where(sign != 0, ~bits, bits | U64(1 << 63))


def _compare(name: str, oracle, candidate, mask=None) -> dict[str, object]:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if oracle.shape != candidate.shape:
        raise AssertionError(f"{name}: shape mismatch {oracle.shape} != {candidate.shape}")
    if mask is None:
        mask = np.ones(oracle.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    different = oracle.view(np.uint64) != candidate.view(np.uint64)
    ulps = np.abs(
        _ordered_bits(oracle).astype(np.int64)
        - _ordered_bits(candidate).astype(np.int64)
    )
    selected = different & mask
    first = np.argwhere(selected)
    return {
        "name": name,
        "status": "BIT-EXACT" if not np.any(selected) else "DIFFERENT",
        "differing": int(np.count_nonzero(selected)),
        "n": int(np.count_nonzero(mask)),
        "max_abs": float(np.max(np.abs(candidate - oracle)[mask], initial=0.0)),
        "max_ulp": int(np.max(ulps[mask], initial=0)),
        "first_index": None if not first.size else [int(v) for v in first[0]],
        "first_oracle_hex": None if not first.size else oracle[tuple(first[0])].hex(),
        "first_candidate_hex": None if not first.size else candidate[tuple(first[0])].hex(),
    }


def _ddpdd_sum(values: np.ndarray, mask: np.ndarray) -> np.float64:
    """NEMO DDPDD, lib_fortran_generic.h90:143-149/lib_mpp.F90:1258-1265."""
    high = F64(0.0)
    low = F64(0.0)
    # NEMO's DO_2D expands with ji as the inner loop.
    for jj in range(values.shape[0]):
        for ji in range(values.shape[1]):
            addend = _mul(values[jj, ji], mask[jj, ji])
            zt1 = _add(addend, high)
            zerr = _sub(zt1, addend)
            zt2 = _add(
                _add(_sub(high, zerr), _sub(addend, _sub(zt1, zerr))),
                low,
            )
            normalized = _add(zt1, zt2)
            low = _sub(zt2, _sub(normalized, zt1))
            high = normalized
    return high


def _literal_qsr(
    qsr: np.ndarray,
    r3t: np.ndarray,
    gdepw_1d: np.ndarray,
    e3t_0: np.ndarray,
    tmask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Literal executed qsr_2BD statements, including nk0=2/nkV=17."""
    rho0 = F64(1026.0)
    rcp = F64(3991.86795711963)
    rho0_rcp = _mul(rho0, rcp)
    r1_rho0_rcp = _div(F64(1.0), rho0_rcp)
    rn_abs = F64(0.58)
    rn_si0 = F64(0.35)
    rn_si1 = F64(23.0)
    r1_si0 = _div(F64(1.0), rn_si0)
    r1_si1 = _div(F64(1.0), rn_si1)
    zz0 = _mul(rn_abs, r1_rho0_rcp)
    zz1 = _mul(_sub(F64(1.0), rn_abs), r1_rho0_rcp)
    ny, nx = qsr.shape
    tendency = np.zeros((ny, nx, gdepw_1d.size), dtype=np.float64)
    args0 = np.zeros((ny, nx, 18), dtype=np.float64)
    args1 = np.zeros((ny, nx, 18), dtype=np.float64)
    exp0 = np.zeros_like(args0)
    exp1 = np.zeros_like(args1)

    for jj in range(ny):
        for ji in range(nx):
            stretch = _add(F64(1.0), r3t[jj, ji])

            def argument(level: int, reciprocal: np.float64) -> np.float64:
                depth = _mul(gdepw_1d[level], stretch)
                return _mul(_neg(depth), reciprocal)

            arg0 = argument(0, r1_si0)
            arg1 = argument(0, r1_si1)
            args0[jj, ji, 0], args1[jj, ji, 0] = arg0, arg1
            exp0[jj, ji, 0], exp1[jj, ji, 0] = _libm("exp", arg0), _libm("exp", arg1)
            zatt = _add(_mul(zz0, exp0[jj, ji, 0]), _mul(zz1, exp1[jj, ji, 0]))

            # Fortran jk=1..nkV maps to zero-based T levels 0..16 and
            # gdepw_1d(jk+1) maps to zero-based interface indices 1..17.
            for level in range(17):
                interface = level + 1
                arg1 = argument(interface, r1_si1)
                args1[jj, ji, interface] = arg1
                exp1[jj, ji, interface] = _libm("exp", arg1)
                wmask = F64(tmask[level, jj, ji] * tmask[interface, jj, ji])
                if level < 2:
                    arg0 = argument(interface, r1_si0)
                    args0[jj, ji, interface] = arg0
                    exp0[jj, ji, interface] = _libm("exp", arg0)
                    zzatt = _mul(
                        _add(
                            _mul(zz0, exp0[jj, ji, interface]),
                            _mul(zz1, exp1[jj, ji, interface]),
                        ),
                        wmask,
                    )
                else:
                    zzatt = _mul(_mul(zz1, exp1[jj, ji, interface]), wmask)
                ze3t = _mul(
                    e3t_0[level, jj, ji],
                    _add(F64(1.0), _mul(r3t[jj, ji], tmask[level, jj, ji])),
                )
                tendency[jj, ji, level] = _div(
                    _mul(qsr[jj, ji], _sub(zatt, zzatt)), ze3t
                )
                zatt = zzatt
    return tendency, {
        "arg_si0": args0,
        "arg_si1": args1,
        "exp_si0": exp0,
        "exp_si1": exp1,
    }


def _literal_sbc(
    lat: np.ndarray,
    wet: np.ndarray,
    surface_ct: np.ndarray,
    surface_pt: np.ndarray,
    kt: int = 1,
    nyear: int = 1,
    qsr_pi: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Literal usrdef_sbc_oce statements at step ``kt``, and nine SIN/COS sites.

    ``kt`` and ``nyear`` are the routine's OWN time operands
    (``BLD/ppsrc/nemo/usrdef_sbc.f90:107-108``:
    ``ztime = REAL(kt)*rn_Dt/(rmmss*rhhmm) - (nyear-1)*rjjhh*zyydd``).  They
    default to the kt=1 / year-1 values this file was written for, so round
    16's own numbers are byte-unchanged; the day-by-day owner round passes the
    day boundaries' kt and reads ``nyear`` from the run's own ``ndastp``.
    ``qsr_pi`` overrides the source's literal ``3.1415`` and exists ONLY for
    the owner round's plant, which must be able to make this gate red.
    """
    rpi = F64(3.141592653589793)
    zyydd = F64(360.0)
    ztime = _sub(
        _div(_mul(F64(kt), F64(14400.0)), _mul(F64(60.0), F64(60.0))),
        _mul(_mul(F64(nyear - 1), F64(24.0)), zyydd),
    )
    ztimemax1 = _mul(_add(_mul(F64(5.0), F64(30.0)), F64(21.0)), F64(24.0))
    ztimemin1 = _add(ztimemax1, _div(_mul(F64(24.0), zyydd), F64(2.0)))
    ztimemax2 = _mul(_add(_mul(F64(6.0), F64(30.0)), F64(21.0)), F64(24.0))
    ztimemin2 = _sub(ztimemax2, _div(_mul(F64(24.0), zyydd), F64(2.0)))
    seasonal1_arg = _mul(
        _div(_sub(ztime, ztimemax1), _sub(ztimemin1, ztimemax1)), rpi
    )
    seasonal2_arg = _mul(
        _div(_sub(ztime, ztimemax2), _sub(ztimemax2, ztimemin2)), rpi
    )
    zcos_sais1 = _libm("cos", seasonal1_arg)
    zcos_sais2 = _libm("cos", seasonal2_arg)
    ny, nx = lat.shape
    qsr = np.empty_like(lat)
    qns = np.empty_like(lat)
    emp_raw = np.empty_like(lat)
    t_star = np.empty_like(lat)
    utau = np.empty_like(lat)
    vtau = np.empty_like(lat)
    sites = {
        name: np.empty_like(lat)
        for name in (
            "tstar_cos_arg", "qsr_cos_arg", "emp_south_sin_arg",
            "emp_north_sin_arg", "wind_utau_sin_arg", "wind_vtau_sin_arg",
        )
    }
    sites["seasonal1_cos_arg"] = np.full_like(lat, seasonal1_arg)
    sites["seasonal2_cos_arg"] = np.full_like(lat, seasonal2_arg)
    ztimemax = ztimemax1
    ztimemin = _add(ztimemax, _div(_mul(F64(24.0), zyydd), F64(2.0)))
    wind_season_arg = _mul(
        _div(_sub(ztime, ztimemax), _sub(ztimemin, ztimemax)), rpi
    )
    sites["wind_season_cos_arg"] = np.full_like(lat, wind_season_arg)
    ztau = _div(F64(0.105), _libm("sqrt", F64(2.0)))
    ztaun = _sub(ztau, _mul(F64(0.015), _libm("cos", wind_season_arg)))

    for jj in range(ny):
        for ji in range(nx):
            phi = lat[jj, ji]
            tstar_scale = _add(
                F64(1.0), _mul(_div(F64(1.0), F64(50.0)), zcos_sais2)
            )
            tstar_den_inner = _add(
                F64(1.0),
                _mul(_div(F64(11.0), F64(53.5)), zcos_sais2),
            )
            tstar_den = _mul(_mul(F64(53.5), tstar_den_inner), F64(2.0))
            tstar_arg = _div(_mul(rpi, _sub(phi, F64(5.0))), tstar_den)
            sites["tstar_cos_arg"][jj, ji] = tstar_arg
            t_star[jj, ji] = _mul(
                _mul(F64(28.3), tstar_scale), _libm("cos", tstar_arg)
            )
            qsr_arg = _div(
                _mul(F64(3.1415) if qsr_pi is None else F64(qsr_pi),
                     _sub(phi, _mul(F64(23.5), zcos_sais1))),
                _mul(F64(0.9), F64(180.0)),
            )
            sites["qsr_cos_arg"][jj, ji] = qsr_arg
            qsr[jj, ji] = _mul(F64(230.0), _libm("cos", qsr_arg))
            qns[jj, ji] = _sub(
                _mul(F64(-40.0), _sub(surface_ct[jj, ji], t_star[jj, ji])),
                qsr[jj, ji],
            )

            south_arg = _div(
                _mul(_div(rpi, F64(2.0)), _sub(phi, F64(37.2))),
                _sub(F64(24.6), F64(37.2)),
            )
            north_arg = _div(
                _mul(_div(rpi, F64(2.0)), _sub(phi, F64(37.2))),
                _sub(F64(46.8), F64(37.2)),
            )
            sites["emp_south_sin_arg"][jj, ji] = south_arg
            sites["emp_north_sin_arg"][jj, ji] = north_arg
            if phi >= F64(14.845) and F64(37.2) >= phi:
                seasonal = _sub(
                    F64(1.0),
                    _mul(_div(F64(0.1), F64(0.7)), zcos_sais1),
                )
                emp_raw[jj, ji] = _mul(
                    _mul(_mul(F64(0.7), F64(3.16e-5)), _libm("sin", south_arg)),
                    seasonal,
                )
            else:
                seasonal = _sub(
                    F64(1.0),
                    _mul(_div(F64(0.1), F64(0.8)), zcos_sais1),
                )
                emp_raw[jj, ji] = _mul(
                    _mul(_mul(F64(-0.8), F64(3.16e-5)), _libm("sin", north_arg)),
                    seasonal,
                )

            wind_arg = _div(
                _mul(rpi, _sub(phi, F64(15.0))), _sub(F64(29.0), F64(15.0))
            )
            sites["wind_utau_sin_arg"][jj, ji] = wind_arg
            sites["wind_vtau_sin_arg"][jj, ji] = wind_arg
            wind_sin = _libm("sin", wind_arg)
            utau[jj, ji] = _mul(_neg(ztaun), wind_sin)
            vtau[jj, ji] = _mul(ztaun, wind_sin)

    zsumemp = _ddpdd_sum(emp_raw, wet.astype(np.float64))
    zsurf = _ddpdd_sum(wet.astype(np.float64), wet.astype(np.float64))
    mean = _div(zsumemp, zsurf)
    emp = np.array(emp_raw, copy=True)
    rcp = F64(3991.86795711963)
    for jj in range(ny):
        for ji in range(nx):
            emp[jj, ji] = _sub(emp[jj, ji], _mul(mean, wet[jj, ji]))
            qns[jj, ji] = _sub(
                qns[jj, ji],
                _mul(_mul(emp[jj, ji], surface_pt[jj, ji]), rcp),
            )
    return {"qsr": qsr, "qns": qns, "emp": emp, "utau": utau, "vtau": vtau}, sites


def _current_qsr_sites(card, r3t):
    from legoesm.core.transcendentals import exp

    z = jnp.asarray(card.recipe.z_coord.z_half_ref)
    stretch = jnp.asarray(1.0 + r3t)

    @jax.jit
    def evaluate(z_ref, factor):
        live = z_ref * factor[..., None]
        arg0 = live / 0.35
        arg1 = live / 23.0
        return arg0, arg1, exp(arg0), exp(arg1)

    return tuple(np.asarray(value) for value in evaluate(z, stretch))


class _PoisonFunction:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name
        self.argtypes = None
        self.restype = None

    def __call__(self, value):
        self.owner.counts[self.name] += 1
        result = _libm(self.name, value)
        return float(np.nextafter(result, np.inf))


class _PoisonLibm:
    def __init__(self):
        self.counts = {name: 0 for name in ("exp", "tanh", "sin", "cos")}
        self.functions = {name: _PoisonFunction(self, name) for name in self.counts}

    def __getattr__(self, name):
        return self.functions[name]


def _routing_control(qsr_record, tracer_record) -> dict[str, object]:
    import legoesm.core.transcendentals as transcendental_module

    clean = _candidate("libm", qsr_record, tracer_record)
    original = transcendental_module._LIBM
    poison = _PoisonLibm()
    try:
        transcendental_module._LIBM = poison
        jax.clear_caches()
        planted = _candidate("libm", qsr_record, tracer_record)
    finally:
        transcendental_module._LIBM = original
        jax.clear_caches()
    changed = {
        name: not np.array_equal(
            np.asarray(clean[name]).view(np.uint64),
            np.asarray(planted[name]).view(np.uint64),
        )
        for name in ("qsr_increment", "qsr", "qns", "emp", "utau", "vtau")
    }
    return {
        "poison": "one nextafter step toward +inf per scalar libm result",
        "callback_calls": poison.counts,
        "outputs_changed": changed,
        "exp_route_pass": poison.counts["exp"] > 0 and changed["qsr_increment"],
        "sin_cos_route_pass": (
            poison.counts["sin"] > 0
            and poison.counts["cos"] > 0
            and any(changed[name] for name in ("qsr", "qns", "emp", "utau", "vtau"))
        ),
    }


def run(oracle_root: Path, mesh_path: Path) -> dict[str, object]:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.eos import nemo_potential_temperature_from_conservative

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    qsr_record = read_qsr_stage3(oracle_root / "oracle_qsr_stage3_kt00000001.bin")
    tracer_record = read_tracer_stage3(
        oracle_root / "oracle_rktracer_stage3_kt00000001.bin"
    )
    sbc_record = _read_sbc(oracle_root / "oracle_sbc_kt00000001.bin")
    current = _candidate("libm", qsr_record, tracer_record)
    card = current["card"]
    with xr.open_dataset(mesh_path, decode_times=False) as dataset:
        gdepw_1d = np.asarray(dataset["gdepw_1d"]).squeeze().astype(np.float64)
        e3t_0 = np.asarray(dataset["e3t_0"]).squeeze().astype(np.float64)
        tmask = np.asarray(dataset["tmask"]).squeeze().astype(np.float64)
        lat = np.asarray(dataset["gphit"]).squeeze().astype(np.float64)
    wet = tmask[0] > 0.5
    r3t = np.asarray(tracer_record["r3t_Kmm"], dtype=np.float64)
    literal_qsr, literal_qsr_sites = _literal_qsr(
        qsr_record["qsr"], r3t, gdepw_1d, e3t_0, tmask
    )
    surface_ct = np.asarray(card.recipe.initial_state.T.data[..., 0], dtype=np.float64)
    surface_salinity = np.asarray(card.recipe.initial_state.S.data[..., 0], dtype=np.float64)
    surface_pt = np.asarray(
        jax.jit(nemo_potential_temperature_from_conservative)(
            jnp.asarray(surface_ct), jnp.asarray(surface_salinity)
        )
    )
    literal_sbc, literal_sbc_sites = _literal_sbc(lat, wet, surface_ct, surface_pt)

    current_arg0, current_arg1, current_exp0, current_exp1 = _current_qsr_sites(card, r3t)
    interfaces = slice(0, 18)
    ir_executed = np.zeros(literal_qsr_sites["arg_si0"].shape, dtype=bool)
    ir_executed[..., :3] = True
    site_rows = [
        _compare(
            "qsr.arg_si0",
            literal_qsr_sites["arg_si0"],
            current_arg0[..., interfaces],
            ir_executed,
        ),
        _compare("qsr.arg_si1", literal_qsr_sites["arg_si1"], current_arg1[..., interfaces]),
        _compare(
            "qsr.exp_si0",
            literal_qsr_sites["exp_si0"],
            current_exp0[..., interfaces],
            ir_executed,
        ),
        _compare("qsr.exp_si1", literal_qsr_sites["exp_si1"], current_exp1[..., interfaces]),
    ]
    active = np.transpose(tmask[:30] > 0.5, (1, 2, 0))
    before_qsr = np.asarray(tracer_record["after_sbc_T"])[..., :30]
    after_qsr = np.asarray(tracer_record["after_qsr_T"])[..., :30]
    literal_replay = np.asarray(
        before_qsr + literal_qsr[..., :30], dtype=np.float64
    )
    current_replay = np.asarray(
        before_qsr + np.asarray(current["qsr_increment"]), dtype=np.float64
    )
    qsr_rows = [
        _compare(
            "qsr.literal_accumulation_replay_vs_oracle_after",
            after_qsr,
            literal_replay,
            active,
        ),
        _compare(
            "qsr.current_accumulation_replay_vs_oracle_after",
            after_qsr,
            current_replay,
            active,
        ),
        _compare(
            "qsr.recovered_delta_dump_vs_literal_increment",
            qsr_record["dT_dt"][..., :30],
            literal_qsr[..., :30],
            active,
        ),
    ]
    qsr_rows[2].update({
        "scored": False,
        "status_detail": "NOT_BIT_EXACT_CANCELLATION_NOISE",
        "superseded_by": "qsr.literal_accumulation_replay_vs_oracle_after",
        "reason": (
            "The dump recovers Krhs_after-Krhs_before after both accumulators "
            "have rounded.  Subtractive cancellation makes this derived delta "
            "non-bit-exact even when the literal increment and accumulation "
            "boundary are bit-exact."
        ),
    })
    sbc_rows = []
    for name in ("qsr", "qns", "emp", "utau", "vtau"):
        sbc_rows.append(
            _compare(f"sbc.{name}.literal_vs_oracle", sbc_record[name], literal_sbc[name], wet)
        )
        sbc_rows.append(
            _compare(f"sbc.{name}.current_vs_oracle", sbc_record[name], current[name], wet)
        )

    routing = _routing_control(qsr_record, tracer_record)
    literal_qsr_exact = qsr_rows[0]["status"] == "BIT-EXACT"
    current_qsr_exact = qsr_rows[1]["status"] == "BIT-EXACT"
    literal_sbc_exact = all(row["status"] == "BIT-EXACT" for row in sbc_rows[::2])
    h1 = literal_qsr_exact and not current_qsr_exact
    h1_status = (
        "CONFIRMED" if h1 else
        "SUPERSEDED_BY_FIX" if literal_qsr_exact and current_qsr_exact else
        "NOT_CONFIRMED"
    )
    h2_exp = not routing["exp_route_pass"]
    h2_sin_cos = not routing["sin_cos_route_pass"]
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round16-discriminator-v1",
        "base_git_sha": "b9311e65de2b784b4f12c1ceffc4ad4e394b4e74",
        "regime": "production-jit/cpu/fp64/scalar-libm",
        "oracle_root": str(oracle_root),
        "mesh": str(mesh_path),
        "qsr_dump_semantics": (
            "oracle_qsr_stage3 stores Krhs_after-Krhs_before; direct tendency "
            "certification therefore uses the independently dumped after_sbc_T "
            "and after_qsr_T accumulator replay"
        ),
        "qsr_rows": qsr_rows,
        "qsr_statement_rows": site_rows,
        "sbc_rows": sbc_rows,
        "sbc_literal_site_arguments": {
            name: {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "first_hex": values.flat[0].hex(),
            }
            for name, values in literal_sbc_sites.items()
        },
        "routing_control": routing,
        "hypotheses": {
            "H1_literal_association": h1_status,
            "H1_pre_fix_evidence": {
                "artifact": "round16/discriminator.json",
                "sha256": "3c44294d3dd5051958931baa42a16d670bc6226723f7061522201b88bf163405",
            },
            "H2_exp_callback_route": "CONFIRMED" if h2_exp else "REFUTED",
            "H2_sin_cos_callback_route": "CONFIRMED" if h2_sin_cos else "REFUTED",
            "literal_sbc_all_fields": "BIT-EXACT" if literal_sbc_exact else "DEBT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.oracle_root, args.mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "ROUND16_DISCRIMINATOR "
        f"H1={result['hypotheses']['H1_literal_association']} "
        f"H2_exp={result['hypotheses']['H2_exp_callback_route']} "
        f"H2_sin_cos={result['hypotheses']['H2_sin_cos_callback_route']} "
        f"literal_sbc={result['hypotheses']['literal_sbc_all_fields']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
