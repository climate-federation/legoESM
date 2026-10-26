#!/usr/bin/env python3
"""Real-flag, per-function FP passivity census for the O8--O16 observers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o9_observer_census import _flags, fp_counts

HERE = Path(__file__).resolve().parent / "nemo_si3_o8_o16_master_acquisition"
NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
BUILD = NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD"
TOOLCHAIN = Path("/home/dbalwada/miniconda3/envs/nemo-build")
UNITS = {
    "icestp": ("__icestp_MOD_ice_stp", "rDt_ice = rDt_ice + 1._wp"),
    "icethd": ("__icethd_MOD_ice_thd", "a_i(1,1,1) = a_i(1,1,1) + 1._wp"),
    "icethd_do": ("__icethd_do_MOD_ice_thd_frazil", "ht_i_new(1,1) = ht_i_new(1,1) + 1._wp"),
    "icethd_da": ("__icethd_da_MOD_ice_thd_da", "a_i_1d(1) = a_i_1d(1) + 1._wp"),
    "icethd_sal": ("__icethd_sal_MOD_ice_thd_sal", "sz_i_1d(1,1) = sz_i_1d(1,1) + 1._wp"),
    "icethd_pnd": ("__icethd_pnd_MOD_ice_thd_pnd", "a_ip(1,1,1) = a_ip(1,1,1) + 1._wp"),
    "icethd_zdf_bl99": ("__icethd_zdf_bl99_MOD_ice_thd_zdf_bl99", "t_i_1d(1,1) = t_i_1d(1,1) + 1._wp"),
    "icealb": ("__icealb_MOD_ice_alb", "palb_ice(1,1,1) = palb_ice(1,1,1) + 1._wp"),
    "iceupdate": ("__iceupdate_MOD_ice_update_tau", "taum(1,1) = taum(1,1) + 1._wp"),
}


def _patched(source: Path, patch: Path, destination: Path) -> None:
    destination.write_bytes(source.read_bytes())
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", str(destination), str(patch)],
        check=True,
        capture_output=True,
        text=True,
    )


def _plant(text: str, statement: str) -> str:
    marker = "! O8O16_OBSERVER_END"
    call = text.find("CALL o8o16")
    at = text.find(marker, call)
    if at < 0:
        raise ValueError("observer marker absent")
    end = text.find("\n", at)
    return text[: end + 1] + "      " + statement + "\n" + text[end + 1 :]


def _plant_salinity(text: str, statement: str) -> str:
    marker = "      SELECT CASE ( nn_icesal )"
    if marker not in text:
        raise ValueError("salinity routine executable marker absent")
    return text.replace(marker, "      " + statement + "\n" + marker, 1)


def _plant_pond(text: str, statement: str) -> str:
    marker = "      IF( ln_timing )   CALL timing_start('icethd_pnd')"
    if marker not in text:
        raise ValueError("pond routine executable marker absent")
    return text.replace(marker, "      " + statement + "\n" + marker, 1)


def _plant_icealb(text: str, statement: str) -> str:
    marker = "      IF( ln_timing )   CALL timing_start('icealb')"
    if marker not in text:
        raise ValueError("albedo routine executable marker absent")
    return text.replace(marker, "      " + statement + "\n" + marker, 1)


def _preprocess(source: Path, destination: Path, build: Path, env: dict[str, str]) -> None:
    with destination.open("w") as stream:
        subprocess.run(
            ["cpp", "-Dkey_nosignedzero", "-P", "-traditional", "-Dkey_si3",
             "-Dkey_linssh", "-Dkey_vco_1d3d", "-Dkey_RK3", f"-I{build.parent / 'WORK'}",
             str(source)], check=True, env=env, stdout=stream, text=True,
        )


def census(build: Path = BUILD, toolchain: Path = TOOLCHAIN) -> dict:
    flags = _flags(build)
    report = {"flags": flags, "units": {}}
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        env = dict(os.environ, PATH=f"{toolchain / 'bin'}:{os.environ['PATH']}")
        subprocess.run(
            ["mpif90", "-c", "-cpp", *flags, f"-I{build / 'inc'}", f"-J{tmp}",
             str(HERE / "o8_o16_master_writer.F90"), "-o", str(tmp / "writer.o")],
            check=True, env=env, capture_output=True, text=True,
        )
        for unit, (routine, plant_statement) in UNITS.items():
            original_raw = NEMO / "src/ICE" / f"{unit}.F90"
            patched_raw = tmp / f"{unit}_instrumented.F90"
            _patched(original_raw, HERE / f"{unit}_o8_o16.patch", patched_raw)
            original = tmp / f"{unit}_baseline.f90"
            patched = tmp / f"{unit}_instrumented.f90"
            _preprocess(original_raw, original, build, env)
            _preprocess(patched_raw, patched, build, env)
            planted = tmp / f"{unit}_plant.f90"
            if unit == "icethd_sal":
                plant_text = _plant_salinity(patched.read_text(), plant_statement)
            elif unit == "icethd_pnd":
                plant_text = _plant_pond(patched.read_text(), plant_statement)
            elif unit == "icealb":
                plant_text = _plant_icealb(patched.read_text(), plant_statement)
            else:
                plant_text = _plant(patched.read_text(), plant_statement)
            planted.write_text(plant_text)
            # Local compilation adds the writer .mod directory for USE resolution.
            def compile_local(src: Path) -> Path:
                obj = src.with_suffix(".o")
                subprocess.run(
                    ["mpif90", "-c", *flags, f"-I{tmp}", f"-I{build / 'inc'}",
                     f"-J{tmp}", str(src), "-o", str(obj)], check=True, env=env,
                    capture_output=True, text=True,
                )
                return obj
            plain = compile_local(original)
            observed = compile_local(patched)
            plant = compile_local(planted)
            a, b, c = fp_counts(plain), fp_counts(observed), fp_counts(plant)
            functions = sorted(set(a) | set(b))
            diffs = {
                fn: {key: [a.get(fn, {}).get(key, 0), b.get(fn, {}).get(key, 0)]
                     for key in sorted(set(a.get(fn, {})) | set(b.get(fn, {})))
                     if a.get(fn, {}).get(key, 0) != b.get(fn, {}).get(key, 0)}
                for fn in functions
            }
            diffs = {fn: value for fn, value in diffs.items() if value}
            total = sum(a.get(routine, {}).values())
            plant_fired = c.get(routine) != b.get(routine)
            report["units"][unit] = {
                "routine": routine,
                "plain_fp_total": total,
                "instrumented_fp_total": sum(b.get(routine, {}).values()),
                "differing_mnemonics": diffs,
                "plant_fired": plant_fired,
                "source_sha256": hashlib.sha256(original_raw.read_bytes()).hexdigest(),
                "passive": total > 0 and not diffs and plant_fired,
            }
    report["pass"] = all(row["passive"] for row in report["units"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument("--toolchain", type=Path, default=TOOLCHAIN)
    args = parser.parse_args()
    try:
        result = census(args.build, args.toolchain)
    except subprocess.CalledProcessError as exc:
        print(f"STATUS PLANT-FIRED: {exc}\n{exc.stderr or ''}")
        return 1
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"STATUS PLANT-FIRED: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
