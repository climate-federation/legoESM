"""Generate stub interfaces and bodies for OpenIFS surface routines from their
own interface headers (surf/interface/*.h), for the VDFMAIN oracle harness.

usage: python gen_surf_stubs.py <header.h> <out_interface.h> <out_stub.F90>

The surface handle TYPE(C_PTR) becomes the harness's TSURF_STUB.  The stub
body zeroes every INTENT(OUT) argument and then copies the values VDFMAIN's
prescribed-flux path reads from module YOMSFCSTUB (SURFEXCDRIVER only):
PCFMLEV (SURFEXCDRIVER's surface momentum coefficient), PKMFL (u*^2) and
PCFHTI (per-tile heat coefficient).  Snow evaporation PEVAPSNW stays 0.
"""
import re
import sys
from pathlib import Path

FROM_MODULE = {"PCFMLEV": "SFC_CFM", "PKMFL": "SFC_KMFL", "PCFHTI": "SFC_CFHTI",
               "PCFQTI": "SFC_CFHTI"}


def main(src, out_h, out_f):
    text = Path(src).read_text()
    body = text[text.index("SUBROUTINE"):text.index("END SUBROUTINE")]
    name = re.match(r"SUBROUTINE\s+(\w+)", body).group(1)
    body = body.replace("TYPE(C_PTR)", "TYPE(TSURF_STUB)")
    body = re.sub(r"USE,\s*INTRINSIC\s*::\s*ISO_C_BINDING[^\n]*\n|USE\s+ISO_C_BINDING[^\n]*\n",
                  "USE YOEPHY, ONLY : TSURF_STUB\n", body)
    decl = [ln for ln in body.splitlines() if "::" in ln and not ln.lstrip().startswith("!")]
    outs = []
    for ln in decl:
        if re.search(r"INTENT\(OUT\)", ln, re.I):
            outs += [re.match(r"\s*(\w+)", v).group(1) for v in ln.split("::")[1].split(",")
                     if re.match(r"\s*\w", v)]
    Path(out_h).write_text("INTERFACE\n" + body + f"END SUBROUTINE {name}\nEND INTERFACE\n")
    lines = []
    for o in outs:
        opt = any(o in ln and "OPTIONAL" in ln.upper() for ln in decl)
        typ = next(ln for ln in decl if re.search(rf"::\s*{o}\b", ln))
        zero = ".FALSE." if "LOGICAL" in typ.upper() else ("0" if "INTEGER" in typ.upper() else "0.0_JPRB")
        if "CHARACTER" in typ.upper():
            zero = "' '"
        val = FROM_MODULE.get(o, zero)
        if o in FROM_MODULE:
            val = f"{FROM_MODULE[o]}(1:SIZE({o},1){',:' if o in ('PCFHTI', 'PCFQTI') else ''})"
        s = f"{o} = {val}"
        lines.append(f"IF (PRESENT({o})) {s}" if opt else s)
    stub = (body.replace("IMPLICIT NONE", "USE YOMSFCSTUB\nIMPLICIT NONE", 1)
            + "\n".join(lines) + f"\nEND SUBROUTINE {name}\n")
    Path(out_f).write_text(stub)
    print(name, "outs", len(outs))


if __name__ == "__main__":
    main(*sys.argv[1:4])
