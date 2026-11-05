#!/usr/bin/env python3
"""Fail-closed C0/C1 scorer; C2--C4 remain explicitly unscorable."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from legoesm.ice.si3_finalize import SI3StressInputs, si3_ice_update_flx, si3_ice_update_tau
from legoesm.ice.si3_orca2_state import SI3State
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
    parse_resolved_ice_namelist,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o16_master_gate import (
    read_master_record,
    record_name,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_composition_scorer import (
    read_record as read_o17_record,
)

MAGIC = b"CLAD0001"
UNSCORABLE = "NOT_YET_SCORABLE_NO_COMPOSED_LEGOESM_COUPLED_STEP"
NAMES = {
    1: (
        "restart_means_present",
        "use_conservative_temperature",
        "remove_inverse_barometer",
        "temperature_now",
        "salinity_now",
        "u_before",
        "v_before",
        "ssh_now",
        "ssh_inverse_barometer",
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
    ),
    2: ("ssu", "ssv", "sst", "sss", "ssh", "e3t", "frq"),
    3: ("ssu", "ssv", "sst", "sss", "ssh", "e3t", "frq"),
    4: (
        "utau",
        "vtau",
        "taum",
        "utauU",
        "vtauV",
        "qsr",
        "qns",
        "emp",
        "sfx",
        "fr_i",
        "fwfice",
        "tmask",
        "umask",
        "vmask",
    ),
}


def read_c_record(path: Path) -> dict:
    raw = path.read_bytes()
    header = 8 + 4 + 11 * 4
    if len(raw) < header or raw[:8] != MAGIC:
        raise ValueError("STATUS PLANT-FIRED: truncated/bad C record header")
    version, *meta = struct.unpack_from("<12i", raw, 8)
    kind, kt, kbb, kmm, rank, jpi, jpj, jpk, jpts, bits, nfields = meta
    if version != 1 or bits != 64 or kind not in NAMES or nfields != len(NAMES[kind]):
        raise ValueError("STATUS PLANT-FIRED: unsupported C record schema")
    offset = header
    fields = {}
    for index, name in enumerate(NAMES[kind]):
        if kind == 1 and index < 3:
            fields[name] = struct.unpack_from("<i", raw, offset)[0]
            offset += 4
        else:
            count = jpi * jpj
            stop = offset + count * 8
            if stop > len(raw):
                raise ValueError("STATUS PLANT-FIRED: truncated C record payload")
            fields[name] = np.frombuffer(raw[offset:stop], "<f8").reshape((jpi, jpj), order="F")
            offset = stop
    if offset != len(raw):
        raise ValueError("STATUS PLANT-FIRED: trailing C record bytes")
    return {"kind": kind, "kt": kt, "rank": rank, "fields": fields}


def _equal(name: str, got, want) -> None:
    if not np.array_equal(np.asarray(got).view(np.uint64), np.asarray(want).view(np.uint64)):
        raise ValueError(f"STATUS PLANT-FIRED: {name} is not 0 ULP")


def score_c0(records: Path, *, plant: str | None = None) -> dict:
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
            ssh_inverse_barometer=f["ssh_inverse_barometer"],
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


def score_c1(evidence: Path, ppsrc: Path, *, plant: str | None = None) -> dict:
    run = evidence / "REC_C3"
    records = run / "o8_o16_master_records"
    first = read_master_record(records / record_name(15, 1, 0, 0))
    namelist = parse_resolved_ice_namelist(
        run / "output.namelist.ice", run / "output.namelist.dyn", ppsrc
    )
    cfg = _late_config(namelist, first["scalars"], nn_hls=2)._replace(nn_fsbc=2)
    surface_config = SI3SurfaceConfig.from_resolved_namelists(
        run / "output.namelist.dyn", run / "output.namelist.ice"
    )
    steps = sorted(
        {int(path.name.split("_step")[1][:4]) for path in records.glob("o8o16_kind15_*.bin")}
    )
    for kt in steps:
        for rank in (0, 1):
            r15 = read_master_record(records / record_name(15, kt, 0, rank))
            r16 = read_master_record(records / record_name(16, kt, 0, rank))
            state_fields = dict(r15["fields"])
            sample = state_fields["a_i"]
            template = SI3State.zeros(
                sample.shape[0] + 2 * cfg.nn_hls,
                sample.shape[1] + 2 * cfg.nn_hls,
                config=cfg,
            )
            state_fields["hi_max"] = np.asarray(template.bounds.hi_max)
            state_fields["hi_mean"] = np.asarray(template.bounds.hi_mean)
            state = _state_from_record(state_fields, cfg, full_orca2=True)
            flux = _fluxes_from_record(r15["fields"])
            _, out = si3_ice_update_flx(
                state,
                flux,
                _update_forcing(r15["fields"]),
                config=cfg,
                surface_config=surface_config,
            )
            for name in out._fields:
                _equal(
                    f"C1 flux step={kt} rank={rank} {name}", getattr(out, name), r16["fields"][name]
                )
        # Flux records occur only on SI3 calls; stress records occur every ocean step.
    tau_steps = sorted(
        {int(path.name.split("_step")[1][:4]) for path in records.glob("o8o16_kind19_*.bin")}
    )
    for kt in tau_steps:
        for rank in (0, 1):
            r19 = read_master_record(records / record_name(19, kt, 0, rank))
            r20 = read_master_record(records / record_name(20, kt, 0, rank))
            stress = SI3StressInputs(
                **{name: jnp.asarray(r19["fields"][name]) for name in SI3StressInputs._fields}
            )
            cadence = (kt - 1) % 2 == 0
            if plant == "wrong-parity":
                cadence = not cadence
            got = si3_ice_update_tau(
                stress,
                config=cfg,
                ln_drgice_imp=bool(namelist["ln_drgice_imp"]),
                update_cadence=cadence,
            )
            for name in ("utau", "vtau", "taum"):
                _equal(
                    f"C1 tau step={kt} rank={rank} {name}", getattr(got, name), r20["fields"][name]
                )
            private = read_o17_record(
                evidence
                / "REC_C3/o17_master_records"
                / f"o17_kind34_tau_private_step{kt:04d}_rank{rank:04d}.bin"
            )["fields"]
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
            args.evidence / "record_C3/c_ladder_records",
            plant=args.plant if args.plant == "numeric" else None,
        ),
        "C1": score_c1(args.evidence, args.ppsrc, plant=args.plant),
        **status_report(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
