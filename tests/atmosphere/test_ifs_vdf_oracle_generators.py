"""The VDFMAIN oracle-harness generators (scripts/validate/oracle/ifs_vdf):
gen_vdfmain_driver.py (the 180-argument CALL scaffolding) and gen_surf_stubs.py
(surface-routine stubs from their interface headers), on minimal sources."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "oracle" / "ifs_vdf"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vdfmain_driver_scaffolding(tmp_path):
    src = tmp_path / "vdfmain.F90"
    src.write_text(
        "SUBROUTINE VDFMAIN(YDMODEL, KLON, KLEV, &\n"
        " !  OUTPUT TENDENCIES\n"
        " & PTE, LDLAND, CDCONF)\n"
        "TYPE(MODEL), INTENT(IN) :: YDMODEL\n"
        "INTEGER(KIND=JPIM), INTENT(IN) :: KLON\n"
        "INTEGER(KIND=JPIM), INTENT(IN) :: KLEV\n"
        "REAL(KIND=JPRB), INTENT(INOUT) :: PTE(KLON,KLEV)\n"
        "LOGICAL, INTENT(IN) :: LDLAND(KLON)\n"
        "CHARACTER(LEN=1), INTENT(IN) :: CDCONF\n"
        "REAL(KIND=JPRB) :: ZLOCAL\n")
    _load("gen_vdfmain_driver").main(str(src), str(tmp_path))
    call = (tmp_path / "vm_call.inc").read_text()
    assert call.split("(", 1)[1].replace("&", "").split() == [
        "YDMODEL,", "VM_KLON,", "VM_KLEV,", "VM_PTE,", "VM_LDLAND,", "CDCONF)"]
    assert "ALLOCATE(VM_PTE(VM_KLON,VM_KLEV))" in (tmp_path / "vm_alloc.inc").read_text()
    zero = (tmp_path / "vm_zero.inc").read_text()
    assert "VM_LDLAND = .FALSE." in zero and "VM_KLON = 0" in zero


def test_surface_stub_zeroes_outputs_and_reads_the_module(tmp_path):
    hdr = tmp_path / "surfexcdriver.h"
    hdr.write_text(
        "INTERFACE\nSUBROUTINE SURFEXCDRIVER(YDSURF, KLON, PCFMLEV, PZ0M, PFOO)\n"
        "USE, INTRINSIC :: ISO_C_BINDING\nUSE PARKIND1, ONLY : JPIM, JPRB\nIMPLICIT NONE\n"
        "TYPE(C_PTR), INTENT(IN) :: YDSURF\nINTEGER(KIND=JPIM), INTENT(IN) :: KLON\n"
        "REAL(KIND=JPRB), INTENT(OUT) :: PCFMLEV(:)\nREAL(KIND=JPRB), INTENT(OUT) :: PZ0M(:)\n"
        "REAL(KIND=JPRB), OPTIONAL, INTENT(OUT) :: PFOO(:)\n"
        "END SUBROUTINE SURFEXCDRIVER\nEND INTERFACE\n")
    _load("gen_surf_stubs").main(str(hdr), str(tmp_path / "i.h"), str(tmp_path / "s.F90"))
    stub = (tmp_path / "s.F90").read_text()
    assert "TYPE(TSURF_STUB)" in (tmp_path / "i.h").read_text()
    assert "PCFMLEV = SFC_CFM(1:SIZE(PCFMLEV,1))" in stub
    assert "PZ0M = 0.0_JPRB" in stub
    assert "IF (PRESENT(PFOO)) PFOO = 0.0_JPRB" in stub
