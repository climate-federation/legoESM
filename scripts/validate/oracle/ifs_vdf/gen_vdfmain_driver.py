"""Generate the VDFMAIN call scaffolding for driver_main.F90 from vdfmain.F90's
own argument declarations (180 arguments).

usage: python gen_vdfmain_driver.py <vdfmain.F90> <out_dir>

Writes vm_decl.inc (allocatable driver variables, VM_ prefix), vm_alloc.inc
(ALLOCATE with the Fortran's own bounds, KLON = 1), vm_zero.inc (every real 0,
integer 0, logical .FALSE.) and vm_call.inc (the positional CALL VDFMAIN).
The driver then sets the inputs it means and prints the outputs.
"""
import re
import sys
from pathlib import Path

SCALAR_SET_BY_DRIVER = {"YDMODEL", "CDCONF"}


def main(src, out):
    text = Path(src).read_text()
    head = text[text.index("SUBROUTINE VDFMAIN"):]
    args_txt = head[:head.index(")")]
    args_txt = re.sub(r"!.*", "", args_txt)
    args = [a.strip() for a in re.sub(r"[&\n]", " ", args_txt.split("(", 1)[1]).split(",") if a.strip()]
    decl_txt = head[:head.index("REAL(KIND=JPRB) ::")]
    decl = {}
    for ln in decl_txt.splitlines():
        if "INTENT" not in ln:
            continue
        typ = ln.split(",")[0].strip()
        name_part = ln.split("::")[1].strip()
        m = re.match(r"(\w+)\s*(\((.*)\))?", name_part)
        decl[m.group(1)] = (typ, m.group(3))
    missing = [a for a in args if a not in decl]
    if missing:
        raise SystemExit(f"undeclared args {missing}")
    dec, alloc, zero = [], [], []
    for a in args:
        typ, dims = decl[a]
        if a in SCALAR_SET_BY_DRIVER:
            continue
        v = "VM_" + a
        if dims is None:
            dec.append(f"{typ} :: {v}")
            init = ".FALSE." if typ.startswith("LOGICAL") else ("0" if typ.startswith("INTEGER") else "0.0_JPRB")
            zero.append(f"{v} = {init}")
            continue
        rank = len(re.split(r",(?![^(]*\))", dims))
        vdims = re.sub(r"(?<![%\w])([A-Z]\w*)\b",
                       lambda m: "VM_" + m.group(1) if m.group(1) in decl and m.group(1) not in SCALAR_SET_BY_DRIVER
                       else m.group(1), dims)
        dec.append(f"{typ}, ALLOCATABLE :: {v}({','.join([':'] * rank)})")
        alloc.append(f"ALLOCATE({v}({vdims}))")
        init = ".FALSE." if typ.startswith("LOGICAL") else ("0" if typ.startswith("INTEGER") else "0.0_JPRB")
        zero.append(f"{v} = {init}")
    call = "CALL VDFMAIN( &\n" + ", &\n".join(
        ("  " + a) if a in SCALAR_SET_BY_DRIVER else ("  VM_" + a) for a in args) + ")\n"
    o = Path(out)
    for n, body in (("vm_decl.inc", dec), ("vm_alloc.inc", alloc), ("vm_zero.inc", zero)):
        (o / n).write_text("\n".join(body) + "\n")
    (o / "vm_call.inc").write_text(call)
    print("VDFMAIN args", len(args))


if __name__ == "__main__":
    main(*sys.argv[1:3])
