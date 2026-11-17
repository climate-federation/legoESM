#!/usr/bin/env python3
"""Fail-closed C0/C1 scorer; C2--C4 remain explicitly unscorable."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from legoesm.core.nemo_namelist import parse_namelist
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.core.source_rounding import nemo_source_round as sr
from legoesm.ice.si3_finalize import (
    SI3StressInputs,
    SI3StressOutputs,
    si3_ice_update_flx,
    si3_ice_update_tau,
)
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig, SI3State
from legoesm.ice.si3_surface import SI3SurfaceConfig
from legoesm.ocean.forcing.nemo_surface import (
    NemoSurfaceMeans,
    nemo_sbc_ssm_step,
    nemo_t_stress_to_uv,
)

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
    _fluxes_from_record,
    _late_config,
    _state_from_record,
    _update_forcing,
    read_record_verified,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o16_master_gate import (
    record_name,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_composition_scorer import (
    read_record as read_o17_record,
)

MAGIC = b"CLAD0001"
UNSCORABLE = "NOT_YET_SCORABLE_NO_COMPOSED_LEGOESM_COUPLED_STEP"
_HEADER_BYTES = 8 + 4 + 11 * 4
_KIND1_FLAGS = (
    ("restart_means_present", "i4"),
    ("use_conservative_temperature", "i4"),
    ("remove_inverse_barometer", "i4"),
)
_KIND1_BEFORE_IB = (
    ("temperature_now", "F2"),
    ("salinity_now", "F2"),
    ("u_before", "F2"),
    ("v_before", "F2"),
    ("ssh_now", "F2"),
)
_KIND1_IB = (("ssh_inverse_barometer", "F2"),)
_KIND1_AFTER_IB = tuple(
    (name, "F2")
    for name in (
        "e3t_reference",
        "r3t_now",
        "surface_mask",
        "shortwave_fraction_top",
        "ssu_prior",
        "ssv_prior",
        "sst_prior",
        "sss_prior",
        "ssh_prior",
        "e3t_prior",
        "frq_prior",
    )
)
_FIXED_SPECS = {
    2: tuple((name, "F2") for name in ("ssu", "ssv", "sst", "sss", "ssh", "e3t", "frq")),
    3: tuple((name, "F2") for name in ("ssu", "ssv", "sst", "sss", "ssh", "e3t", "frq")),
    4: (
        ("utau", "F2"),
        ("vtau", "F2"),
        ("taum", "A2D0"),
        ("utauU", "F2"),
        ("vtauV", "F2"),
        ("qsr", "A2D0"),
        ("qns", "A2D0"),
        ("emp", "F2"),
        ("sfx", "A2D0"),
        ("fr_i", "F2"),
        ("fwfice", "A2D0"),
        ("tmask", "F2"),
        ("umask", "F2"),
        ("vmask", "F2"),
    ),
    5: (("ts", "F4"), ("uu", "F3"), ("vv", "F3"), ("ssh", "F2")),
    6: (("ts", "F4"), ("uu", "F3"), ("vv", "F3"), ("ssh", "F2")),
}


def _record_specs(kind: int, remove_inverse_barometer: int | None = None):
    if kind == 1:
        if remove_inverse_barometer not in (0, 1):
            raise ValueError(
                "invalid kind-1 remove_inverse_barometer flag: "
                f"expected 0 or 1, got {remove_inverse_barometer}"
            )
        optional = _KIND1_IB if remove_inverse_barometer == 1 else ()
        return _KIND1_FLAGS + _KIND1_BEFORE_IB + optional + _KIND1_AFTER_IB
    try:
        return _FIXED_SPECS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported C record kind: {kind}") from exc


def _field_shape(code: str, *, jpi: int, jpj: int, jpk: int, jpts: int) -> tuple[int, ...]:
    if code == "F2":
        return jpi, jpj
    if code == "A2D0":
        if jpi <= 4 or jpj <= 4:
            raise ValueError(f"invalid C record A2D(0) domain: jpi={jpi}, jpj={jpj}")
        return jpi - 4, jpj - 4
    if code == "F3":
        return jpi, jpj, jpk
    if code == "F4":
        return jpi, jpj, jpk, jpts
    raise ValueError(f"unsupported C record field shape: {code}")


def read_c_record(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < _HEADER_BYTES:
        raise ValueError(
            f"truncated C record header: expected {_HEADER_BYTES} bytes, found {len(raw)}"
        )
    if raw[:8] != MAGIC:
        raise ValueError(f"bad C record magic: expected {MAGIC!r}, found {raw[:8]!r}")
    version, *meta = struct.unpack_from("<12i", raw, 8)
    kind, kt, kbb, kmm, rank, jpi, jpj, jpk, jpts, bits, nfields = meta
    if version != 1:
        raise ValueError(f"unsupported C record version: expected 1, got {version}")
    if bits != 64:
        raise ValueError(f"unsupported C record real width: expected 64 bits, got {bits}")
    if min(jpi, jpj, jpk, jpts) <= 0:
        raise ValueError(
            f"invalid C record dimensions: jpi={jpi}, jpj={jpj}, jpk={jpk}, jpts={jpts}"
        )
    offset = _HEADER_BYTES
    fields = {}
    if kind == 1:
        for name, _ in _KIND1_FLAGS:
            stop = offset + 4
            if stop > len(raw):
                raise ValueError(f"truncated C record payload: kind=1 field={name}")
            fields[name] = struct.unpack_from("<i", raw, offset)[0]
            if fields[name] not in (0, 1):
                raise ValueError(
                    f"invalid kind-1 {name} flag: expected 0 or 1, got {fields[name]}"
                )
            offset += 4
        specs = _record_specs(kind, fields["remove_inverse_barometer"])
    else:
        specs = _record_specs(kind)
    if nfields != len(specs):
        if kind == 1:
            flag = fields["remove_inverse_barometer"]
            raise ValueError(
                "inconsistent kind-1 layout: "
                f"remove_inverse_barometer={flag} requires nfields={len(specs)}, got {nfields}"
            )
        raise ValueError(
            f"inconsistent kind-{kind} layout: expected nfields={len(specs)}, got {nfields}"
        )
    for name, code in specs[len(_KIND1_FLAGS) if kind == 1 else 0 :]:
        shape = _field_shape(code, jpi=jpi, jpj=jpj, jpk=jpk, jpts=jpts)
        count = int(np.prod(shape))
        stop = offset + count * 8
        if stop > len(raw):
            raise ValueError(
                f"truncated C record payload: kind={kind} field={name}; "
                f"need {count * 8} bytes, have {max(0, len(raw) - offset)}"
            )
        fields[name] = np.frombuffer(raw, "<f8", count=count, offset=offset).reshape(
            shape, order="F"
        )
        offset = stop
    if offset != len(raw):
        raise ValueError(
            f"trailing C record bytes: kind={kind}; parsed {offset}, file has {len(raw)}"
        )
    return {
        "kind": kind,
        "kt": kt,
        "kbb": kbb,
        "kmm": kmm,
        "rank": rank,
        "jpi": jpi,
        "jpj": jpj,
        "jpk": jpk,
        "jpts": jpts,
        "nfields": nfields,
        "fields": fields,
    }


def _write_stub_kind1_record(
    path: Path,
    *,
    remove_inverse_barometer: int,
    old_unconditional_layout: bool,
) -> None:
    """Write a flag-governed kind-1 record or the allocation-off old-layout plant."""
    jpi, jpj, jpk, jpts = 6, 7, 2, 2
    if old_unconditional_layout and remove_inverse_barometer != 0:
        raise ValueError("old unconditional kind-1 plant requires allocation-off flag")
    specs = _record_specs(1, remove_inverse_barometer)
    nfields = 20 if old_unconditional_layout else len(specs)
    header = MAGIC + struct.pack(
        "<12i",
        1,
        1,
        1,
        1,
        1,
        0,
        jpi,
        jpj,
        jpk,
        jpts,
        64,
        nfields,
    )
    payload = bytearray()
    flag_values = (0, 0, remove_inverse_barometer)
    for (_, code), value in zip(_KIND1_FLAGS, flag_values, strict=True):
        assert code == "i4"
        payload.extend(struct.pack("<i", value))
    for index, (_, code) in enumerate(specs[len(_KIND1_FLAGS) :], start=1):
        shape = _field_shape(code, jpi=jpi, jpj=jpj, jpk=jpk, jpts=jpts)
        payload.extend(np.full(shape, float(index), dtype="<f8").tobytes(order="F"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + payload)


def _equal(name: str, got, want) -> None:
    if not np.array_equal(np.asarray(got).view(np.uint64), np.asarray(want).view(np.uint64)):
        raise ValueError(f"STATUS PLANT-FIRED: {name} is not 0 ULP")


def _coupled_resolved_namelist(run: Path) -> dict:
    """Read SI3 parameters plus the coupled run's executed ice-drag arm."""
    ice_path = run / "output.namelist.ice"
    if not ice_path.is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing resolved ice namelist: {ice_path}")
    values = parse_namelist(ice_path, "ice")
    required = (
        "jpl",
        "nlay_i",
        "nlay_s",
        "rn_simin",
        "rn_sinew",
        "nn_icesal",
        "ln_pnd",
        "ln_pnd_lev",
        "ln_pnd_topo",
        "ln_pnd_lids",
        "ln_pnd_alb",
        "nn_snwfra",
    )
    parsed = {}
    for name in required:
        matches = [tokens for (_, _, key), tokens in values.items() if key == name]
        if len(matches) != 1 or len(matches[0]) != 1:
            raise ValueError(
                f"STATUS PLANT-FIRED: coupled resolved ice namelist lacks one scalar {name}"
            )
        parsed[name if name != "ln_pnd_lev" else "ln_pnd_LEV"] = matches[0][0]
    parsed["ln_pnd_TOPO"] = parsed.pop("ln_pnd_topo")
    ocean_output = run / "ocean.output"
    if not ocean_output.is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing coupled run log: {ocean_output}")
    drag_values = re.findall(
        r"implicit ice-ocean drag\s+ln_drgice_imp\s*=\s*([TF])\b",
        ocean_output.read_text(errors="replace"),
        flags=re.I,
    )
    resolved = {value.upper() == "T" for value in drag_values}
    if len(resolved) != 1:
        raise ValueError(
            "STATUS PLANT-FIRED: coupled ocean.output does not resolve one "
            "executed ln_drgice_imp value"
        )
    parsed["ln_drgice_imp"] = resolved.pop()
    return parsed


def _interior_master_fields(record: dict) -> dict[str, np.ndarray]:
    """Crop full local-domain fields to the MPI shard's recorded interior."""
    meta = record["meta"]
    sl = (
        slice(meta["ntsi"] - 1, meta["ntei"]),
        slice(meta["ntsj"] - 1, meta["ntej"]),
    )
    full_shape = (meta["jpi"], meta["jpj"])
    return {
        name: value[sl] if value.ndim >= 2 and value.shape[:2] == full_shape else value
        for name, value in record["fields"].items()
    }


def _nemo_cadenced_tau_step(
    inputs: SI3StressInputs,
    *,
    config: SI3ORCA2StateConfig,
    ln_drgice_imp: bool,
    update_now: bool,
) -> SI3StressOutputs:
    """Replay NEMO's outer ``MOD(kt-1,nn_fsbc)`` guard for C1 scoring only."""
    if update_now:
        return si3_ice_update_tau(
            inputs,
            config=config._replace(nn_fsbc=1),
            ln_drgice_imp=ln_drgice_imp,
        )

    h = int(config.nn_hls)
    sl = (slice(h, -h), slice(h, -h))
    left = (slice(h - 1, -h - 1), slice(h, -h))
    below = (slice(h, -h), slice(h - 1, -h - 1))
    at_i = inputs.at_i[sl]
    zflagi = jnp.asarray(0.0 if ln_drgice_imp else 1.0, dtype=inputs.u_ice.dtype)
    pu = sr(inputs.uu_kbb[sl] + inputs.uu_kbb[left])
    pv = sr(inputs.vv_kbb[sl] + inputs.vv_kbb[below])
    urel = sr(sr(inputs.u_ice[sl] + inputs.u_ice[left]) - sr(zflagi * pu))
    vrel = sr(sr(inputs.v_ice[sl] + inputs.v_ice[below]) - sr(zflagi * pv))
    half = jnp.asarray(0.5, dtype=inputs.u_ice.dtype)
    one = jnp.asarray(1.0, dtype=inputs.u_ice.dtype)
    zutau_ice = sr(sr(half * inputs.tmod_io[sl]) * urel)
    zvtau_ice = sr(sr(half * inputs.tmod_io[sl]) * vrel)
    utau_i = sr(sr(sr(one - at_i) * inputs.utau_oce) + sr(at_i * zutau_ice))
    vtau_i = sr(sr(sr(one - at_i) * inputs.vtau_oce) + sr(at_i * zvtau_ice))
    return SI3StressOutputs(
        utau=inputs.utau.at[sl].set(utau_i),
        vtau=inputs.vtau.at[sl].set(vtau_i),
        taum=inputs.taum,
        tmod_io=inputs.tmod_io,
        utau_oce=inputs.utau_oce,
        vtau_oce=inputs.vtau_oce,
    )


def score_c0(records: Path, *, plant: str | None = None) -> dict:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    entries = sorted(records.glob("c_kind1_*.bin"))
    if not entries:
        raise ValueError("STATUS PLANT-FIRED: no C0 entry records")
    for entry_path in entries:
        entry = read_c_record(entry_path)
        exit_path = records / entry_path.name.replace("c_kind1_", "c_kind2_")
        exit_record = read_c_record(exit_path)
        f = entry["fields"]
        prior = NemoSurfaceMeans(
            *(
                jnp.asarray(f[f"{name}_prior"])
                for name in ("ssu", "ssv", "sst", "sss", "ssh", "e3t", "frq")
            )
        )
        got = nemo_sbc_ssm_step(
            prior,
            kt=entry["kt"],
            nit000=1,
            forcing_stride=2,
            restart_means_present=bool(f["restart_means_present"]),
            use_conservative_temperature=bool(f["use_conservative_temperature"]),
            remove_inverse_barometer=bool(f["remove_inverse_barometer"]),
            u_before=f["u_before"],
            v_before=f["v_before"],
            temperature_now=f["temperature_now"],
            salinity_now=f["salinity_now"],
            ssh_now=f["ssh_now"],
            # The production helper does not use this argument when the recorded
            # flag is false; reuse an existing same-shape field without inventing
            # data that the allocation-off NEMO run did not emit.
            ssh_inverse_barometer=f.get("ssh_inverse_barometer", f["ssh_now"]),
            e3t_reference=f["e3t_reference"],
            r3t_now=f["r3t_now"],
            surface_mask=f["surface_mask"],
            shortwave_fraction_top=f["shortwave_fraction_top"],
        )
        expected = dict(exit_record["fields"])
        if plant == "numeric" and entry_path == entries[0]:
            expected["ssu"] = expected["ssu"].copy()
            expected["ssu"].flat[0] = np.nextafter(expected["ssu"].flat[0], np.inf)
        for name, value in zip(NemoSurfaceMeans._fields, got, strict=True):
            _equal(f"C0 step={entry['kt']} rank={entry['rank']} {name}", value, expected[name])
    return {"status": "PASS_C0_0_ULP", "records": len(entries)}


def score_c1(evidence: Path, *, plant: str | None = None) -> dict:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    run = evidence / "REC_C3"
    records = run / "o8_o16_master_records"
    namelist = _coupled_resolved_namelist(run)
    ice_dimensions = tuple(int(namelist[name]) for name in ("jpl", "nlay_i", "nlay_s"))
    first = read_record_verified(
        records / record_name(15, 1, 0, 0),
        expected_ice_dimensions=ice_dimensions,
    )
    cfg = _late_config(
        namelist,
        first["scalars"],
        ice_dimensions=ice_dimensions,
        nn_hls=2,
    )._replace(nn_fsbc=2)
    flux_cfg = cfg._replace(nn_hls=0)
    surface_config = SI3SurfaceConfig.from_resolved_namelists(
        run / "output.namelist.dyn", run / "output.namelist.ice"
    )
    steps = sorted(
        {int(path.name.split("_step")[1][:4]) for path in records.glob("o8o16_kind15_*.bin")}
    )
    for kt in steps:
        for rank in (0, 1):
            r15 = read_record_verified(
                records / record_name(15, kt, 0, rank),
                expected_ice_dimensions=ice_dimensions,
            )
            r16 = read_record_verified(
                records / record_name(16, kt, 0, rank),
                expected_ice_dimensions=ice_dimensions,
            )
            state_fields = dict(r15["fields"])
            sample = state_fields["a_i"]
            template = SI3State.zeros(
                sample.shape[0],
                sample.shape[1],
                config=flux_cfg,
            )
            state_fields["hi_max"] = np.asarray(template.bounds.hi_max)
            state_fields["hi_mean"] = np.asarray(template.bounds.hi_mean)
            state = _state_from_record(state_fields, flux_cfg, full_orca2=False)
            flux = _fluxes_from_record(r15["fields"])
            _, out = si3_ice_update_flx(
                state,
                flux,
                _update_forcing(_interior_master_fields(r15)),
                config=flux_cfg,
                surface_config=surface_config,
            )
            expected = _interior_master_fields(r16)
            for name in out._fields:
                _equal(f"C1 flux step={kt} rank={rank} {name}", getattr(out, name), expected[name])
        # Flux records occur only on SI3 calls; stress records occur every ocean step.
    tau_steps = sorted(
        {int(path.name.split("_step")[1][:4]) for path in records.glob("o8o16_kind19_*.bin")}
    )
    for kt in tau_steps:
        for rank in (0, 1):
            r19 = read_record_verified(
                records / record_name(19, kt, 0, rank),
                expected_ice_dimensions=ice_dimensions,
            )
            r20 = read_record_verified(
                records / record_name(20, kt, 0, rank),
                expected_ice_dimensions=ice_dimensions,
            )
            private_record = read_o17_record(
                evidence
                / "REC_C3/o17_master_records"
                / f"o17_kind34_tau_private_step{kt:04d}_rank{rank:04d}.bin"
            )
            private_dimensions = tuple(
                int(private_record["meta"][name]) for name in ("jpl", "nlay_i", "nlay_s")
            )
            if private_dimensions != ice_dimensions:
                raise ValueError(
                    "STATUS PLANT-FIRED: O17 record ice dimensions "
                    f"{private_dimensions} disagree with resolved run {ice_dimensions}"
                )
            stress = SI3StressInputs(
                **{name: jnp.asarray(r19["fields"][name]) for name in SI3StressInputs._fields}
            )
            cadence = (kt - 1) % 2 == 0
            if plant == "wrong-parity":
                cadence = not cadence
            got = _nemo_cadenced_tau_step(
                stress,
                config=cfg,
                ln_drgice_imp=bool(namelist["ln_drgice_imp"]),
                update_now=cadence,
            )
            for name in ("utau", "vtau", "taum"):
                _equal(
                    f"C1 tau step={kt} rank={rank} {name}", getattr(got, name), r20["fields"][name]
                )
            private = private_record["fields"]
            for name in ("tmod_io", "utau_oce", "vtau_oce"):
                _equal(
                    f"C1 tau private step={kt} rank={rank} {name}",
                    getattr(got, name),
                    private[name],
                )
            c4 = read_c_record(
                evidence / "REC_C3/c_ladder_records" / f"c_kind4_step{kt:04d}_rank{rank:04d}.bin"
            )["fields"]
            meta = r20["meta"]
            i0, i1 = meta["ntsi"] - 1, meta["ntei"]
            j0, j1 = meta["ntsj"] - 1, meta["ntej"]
            got_u, got_v = nemo_t_stress_to_uv(
                c4["utau"][i0:i1, j0:j1],
                c4["vtau"][i0:i1, j0:j1],
                c4["utau"][i0 + 1 : i1 + 1, j0:j1],
                c4["vtau"][i0:i1, j0 + 1 : j1 + 1],
                c4["tmask"][i0:i1, j0:j1],
                c4["tmask"][i0 + 1 : i1 + 1, j0:j1],
                c4["tmask"][i0:i1, j0 + 1 : j1 + 1],
                c4["umask"][i0:i1, j0:j1],
                c4["vmask"][i0:i1, j0:j1],
            )
            _equal(
                f"C1 final native U stress step={kt} rank={rank}", got_u, c4["utauU"][i0:i1, j0:j1]
            )
            _equal(
                f"C1 final native V stress step={kt} rank={rank}", got_v, c4["vtauV"][i0:i1, j0:j1]
            )
    return {"status": "PASS_C1_0_ULP", "flux_steps": len(steps), "tau_steps": len(tau_steps)}


def status_report() -> dict:
    return {"C2": UNSCORABLE, "C3": UNSCORABLE, "C4": UNSCORABLE}


def self_test(plant: str | None, scratch: Path) -> dict:
    if plant is None:
        valid = scratch / "c_kind1_step0001_rank0000.bin"
        legacy = scratch / "old_unconditional_kind1.bin"
        _write_stub_kind1_record(
            valid, remove_inverse_barometer=0, old_unconditional_layout=False
        )
        record = read_c_record(valid)
        if record["nfields"] != 19 or "ssh_inverse_barometer" in record["fields"]:
            raise ValueError("allocation-off kind-1 stub did not use the 19-field layout")
        _write_stub_kind1_record(
            legacy, remove_inverse_barometer=0, old_unconditional_layout=True
        )
        try:
            read_c_record(legacy)
        except ValueError as exc:
            refusal = str(exc)
            if "remove_inverse_barometer=0 requires nfields=19, got 20" not in refusal:
                raise
        else:
            raise ValueError("old unconditional kind-1 layout was accepted")
        return {
            "status": "PASS_C_SCORER_SELF_TEST",
            "kind1_ln_apr_dyn_off": "PASS_19_FIELDS_NO_SSH",
            "old_unconditional_kind1_plant": f"REFUSED: {refusal}",
            **status_report(),
        }
    if plant == "numeric":
        _equal("numeric plant", np.array([1.0]), np.array([np.nextafter(1.0, np.inf)]))
    if plant == "truncation":
        path = scratch / "truncated.bin"
        path.write_bytes(MAGIC)
        read_c_record(path)
    if plant == "wrong-parity":
        expected = tuple((kt - 1) % 2 == 0 for kt in range(1, 5))
        planted = tuple(not value for value in expected)
        if planted != expected:
            raise ValueError("STATUS PLANT-FIRED: SI3 cadence parity/time level")
    return {"status": "PASS_C_SCORER_SELF_TEST", **status_report()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--ppsrc", type=Path)
    parser.add_argument("--plant", choices=("numeric", "wrong-parity"))
    parser.add_argument(
        "--self-test", choices=("baseline", "numeric", "truncation", "wrong-parity")
    )
    parser.add_argument("--scratch", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.self_test:
        print(
            json.dumps(
                self_test(None if args.self_test == "baseline" else args.self_test, args.scratch),
                indent=2,
            )
        )
        return 0
    if args.evidence is None or args.ppsrc is None:
        parser.error("--evidence and --ppsrc are required outside --self-test")
    report = {
        "C0": score_c0(
            args.evidence / "REC_C3/c_ladder_records",
            plant=args.plant if args.plant == "numeric" else None,
        ),
        "C1": score_c1(args.evidence, plant=args.plant),
        **status_report(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
