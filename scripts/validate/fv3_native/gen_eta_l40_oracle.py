"""Read L40 from the pinned, unmodified FV3 set_eta, on a SLURM CPU node.

Zenodo 8327578 outer archive md5 25b2f54c0a807275e1c4edf4aa41a197.
This is a source oracle, not a reconstruction of the paper's L32 binary.
No physics CPP macros; explicit fp64; existing pinned FMS constants linkage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile


def generate(source: Path, archive: Path) -> dict:
    """Compile the whole source once and emit roundtrip coefficients/provenance."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SLURM only")
    source, archive = Path(source).resolve(), Path(archive).resolve()
    root = Path(__file__).resolve().parents[3]
    module = importlib.import_module("legoesm.grids.fv3_native_gridstruct")
    imported = Path(module.__file__).resolve()
    print("IMPORT", imported, flush=True)
    if not imported.is_relative_to(root):
        raise RuntimeError(f"import outside {root}: {imported}")
    digest = hashlib.md5()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    md5 = digest.hexdigest()
    if md5 != "25b2f54c0a807275e1c4edf4aa41a197":
        raise ValueError(f"archive md5 mismatch: {md5}")
    names = ("tools/fv_eta.F90", "tools/fv_eta.h", "model/fv_arrays.F90")
    with zipfile.ZipFile(archive) as outer:
        inner = outer.read("Code and simulations files/atmos_cubed_sphere-symmetryclean.zip")
    with zipfile.ZipFile(io.BytesIO(inner)) as upstream:
        for name in names:
            if upstream.read("atmos_cubed_sphere-symmetryclean/" + name) != (source / name).read_bytes():
                raise ValueError(f"source differs from pinned archive: {name}")
    flags = ["-cpp", "-fdefault-real-8", "-fdefault-double-8",
             "-ffree-line-length-none", "-O0", "-ffp-contract=off"]
    constants = {name: getattr(module, "FV3_" + name.upper())
                 for name in ("rdgas", "grav", "kappa", "cp_air")}
    stubs = "module constants_mod\n" + "\n".join(
        f"real, parameter :: {key} = {value:.17e}" for key, value in constants.items())
    stubs += """
end module
module fv_mp_mod
contains
logical function is_master()
is_master = .false.
end function
end module
module mpp_mod
integer, parameter :: FATAL=1
contains
subroutine mpp_error(level, message)
integer :: level
character(*) :: message
print *, message
stop 1
end subroutine
end module
"""
    driver = """
program main
use fv_eta_mod, only: set_eta
implicit none
real :: ak(41), bk(41), ptop
integer :: ks, k
character(24) :: npz_type = ''
call set_eta(40, ks, ptop, ak, bk, npz_type)
write(*,'(I4,1X,ES26.17E3)') ks, ptop
do k=1,41
write(*,'(2(ES26.17E3,1X))') ak(k), bk(k)
enddo
end program
"""
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        (work / "stubs.f90").write_text(stubs)
        (work / "driver.f90").write_text(driver)
        command = ["gfortran", *flags, "-I", str(source / "tools"),
                   "stubs.f90", str(source / "tools/fv_eta.F90"),
                   "driver.f90", "-o", "eta"]
        subprocess.run(command, cwd=work, check=True)
        output = subprocess.check_output([str(work / "eta")], text=True)
        preprocessed = subprocess.check_output(
            ["gfortran", *flags, "-E", "-I", str(source / "tools"),
             str(source / "tools/fv_eta.F90")], text=True)
    lines = output.splitlines()
    assert len(lines) == 42, output
    ks, ptop = lines[0].split()
    pairs = [list(map(float, line.split())) for line in lines[1:]]
    return dict(ak=[a for a, _ in pairs], bk=[b for _, b in pairs],
                ptop=float(ptop), ks=int(ks), source=str(source),
                source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                               for name in names},
                archive_md5=md5, inner_archive_md5=hashlib.md5(inner).hexdigest(),
                compiler=subprocess.check_output(
                    ["gfortran", "--version"], text=True).splitlines()[0],
                flags=flags, physics_cpp_macros=[], constants=constants,
                preprocessed_sha256=hashlib.sha256(preprocessed.encode()).hexdigest(),
                slurm_job_id=os.environ["SLURM_JOB_ID"],
                git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=root, text=True).strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(generate(args.source, args.archive), indent=2) + "\n")
    print("WROTE", args.out, flush=True)
