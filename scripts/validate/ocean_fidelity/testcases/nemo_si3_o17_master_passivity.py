#!/usr/bin/env python3
"""Production-flag FP passivity census for the O17 master observers."""

from __future__ import annotations

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
    "icedyn_adv_pra": ("__icedyn_adv_pra_MOD_ice_dyn_adv_pra", "pa_i(1,1,1) = pa_i(1,1,1) + 1._wp"),
    "icedyn": ("__icedyn_MOD_ice_dyn", "u_ice(1,1) = u_ice(1,1) + 1._wp"),
    "icesbc": ("__icesbc_MOD_ice_sbc_flx", "qlead(1,1) = qlead(1,1) + 1._wp"),
    "sbcblk": ("__sbcblk_MOD_blk_ice_1", "wndm_ice(1,1) = wndm_ice(1,1) + 1._wp"),
}


def _source(unit: str) -> Path:
    return NEMO / ("src/OCE/SBC" if unit == "sbcblk" else "src/ICE") / f"{unit}.F90"


def _patch(source: Path, unit: str, destination: Path) -> None:
    destination.write_bytes(source.read_bytes())
    o17 = HERE / f"{unit}_o17.patch"
    if o17.is_file():
        subprocess.run(["patch", "--batch", "--fuzz=0", str(destination), str(o17)],
                       check=True, capture_output=True, text=True)
    combined = HERE / f"{unit}_o8_o16_o17.patch"
    o8_o16 = combined if combined.is_file() else HERE / f"{unit}_o8_o16.patch"
    if o8_o16.is_file():
        subprocess.run(["patch", "--batch", "--fuzz=0", str(destination), str(o8_o16)],
                       check=True, capture_output=True, text=True)


def _preprocess(source: Path, destination: Path, env: dict[str, str]) -> None:
    with destination.open("w") as stream:
        subprocess.run([
            "cpp", "-Dkey_nosignedzero", "-P", "-traditional", "-Dkey_si3",
            "-Dkey_linssh", "-Dkey_vco_1d3d", "-Dkey_RK3", f"-I{BUILD.parent/'WORK'}",
            str(source),
        ], check=True, env=env, stdout=stream, text=True)


def _plant(text: str, statement: str) -> str:
    call = text.find("CALL o17_")
    end_marker = text.find("! O8O16_OBSERVER_END", call)
    newline = text.find("\n", end_marker)
    if call < 0 or end_marker < 0 or newline < 0:
        raise ValueError("O17 observer marker absent")
    return text[:newline+1] + "      " + statement + "\n" + text[newline+1:]


def census() -> dict:
    flags = _flags(BUILD)
    report: dict = {"flags": flags, "units": {}}
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        env = dict(os.environ, PATH=f"{TOOLCHAIN/'bin'}:{os.environ['PATH']}")
        for writer in ("o8_o16_master_writer.F90", "o17_master_writer.F90"):
            subprocess.run(["mpif90", "-c", "-cpp", *flags, f"-I{tmp}", f"-I{BUILD/'inc'}",
                            f"-J{tmp}", str(HERE/writer), "-o", str(tmp/f"{writer}.o")],
                           check=True, env=env, capture_output=True, text=True)
        for unit, (routine, statement) in UNITS.items():
            source = _source(unit)
            patched_raw = tmp / f"{unit}_patched.F90"
            _patch(source, unit, patched_raw)
            original = tmp / f"{unit}_plain.f90"
            patched = tmp / f"{unit}_patched.f90"
            planted = tmp / f"{unit}_plant.f90"
            _preprocess(source, original, env)
            _preprocess(patched_raw, patched, env)
            planted.write_text(_plant(patched.read_text(), statement))

            def compile_one(path: Path) -> Path:
                obj = path.with_suffix(".o")
                subprocess.run(["mpif90", "-c", *flags, f"-I{tmp}", f"-I{BUILD/'inc'}",
                                f"-J{tmp}", str(path), "-o", str(obj)], check=True, env=env,
                               capture_output=True, text=True)
                return obj

            a, b, c = (fp_counts(compile_one(path)) for path in (original, patched, planted))
            all_functions = sorted(set(a) | set(b))
            differences = {
                fn: {op: [a.get(fn, {}).get(op, 0), b.get(fn, {}).get(op, 0)]
                     for op in set(a.get(fn, {})) | set(b.get(fn, {}))
                     if a.get(fn, {}).get(op, 0) != b.get(fn, {}).get(op, 0)}
                for fn in all_functions
            }
            differences = {fn: ops for fn, ops in differences.items() if ops}
            row = {
                "routine": routine,
                "plain_fp_total": sum(a.get(routine, {}).values()),
                "instrumented_fp_total": sum(b.get(routine, {}).values()),
                "differing_mnemonics": differences,
                "plant_fired": c.get(routine) != b.get(routine),
            }
            row["passive"] = row["plain_fp_total"] > 0 and not differences and row["plant_fired"]
            report["units"][unit] = row
    report["pass"] = all(row["passive"] for row in report["units"].values())
    return report


def main() -> int:
    try:
        result = census()
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
