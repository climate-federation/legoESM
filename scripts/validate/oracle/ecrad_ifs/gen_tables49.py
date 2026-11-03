"""Generate tables49.F90: dump every RRTMG table the 49r1 ecRad gas optics read.

usage: python -I gen_tables49.py <build src dir> <out tables49.F90>

The program runs the same RRTMG setup ecRad runs (SURRTAB, SURRTPK, SURRTRF,
RRTM_INIT_140GP, SRTM_INIT; radiation_ifs_rrtm.F90 setup_gas_optics) and then
writes every non-PARAMETER REAL(JPRB)/INTEGER(JPIM) variable of the modules the
runtime gas-optics routines USE, as text: "name kind rank lb1 ub1 ...", then
the values in Fortran order (reals ES26.17E3). The JAX port reads these
numbers; it never re-derives a table.
"""
import re
import sys

MODULES = ([f"yoerrta{i}" for i in range(1, 17)] + ["yoerrtm", "yoerrtrf", "yoerrtwn", "yoerrtftr"]
           + [f"yoesrta{i}" for i in range(16, 30)] + ["yoesrtm", "yoesrtwn"])


def split_top(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def declared(path):
    text = open(path).read()
    text = re.sub(r"&\s*\n\s*&?", " ", text)            # join continuations
    body = re.split(r"^\s*CONTAINS\b", text, flags=re.M | re.I)[0]
    names = []
    for line in body.splitlines():
        line = line.split("!")[0]
        m = re.match(r"^\s*(REAL\(KIND=JPRB\)|INTEGER\(KIND=JPIM\))([^:]*)::(.*)$", line, re.I)
        if not m or "PARAMETER" in m.group(2).upper():
            continue
        kind = "r" if m.group(1).upper().startswith("REAL") else "i"
        for item in split_top(m.group(3)):
            n = re.match(r"\s*([A-Za-z]\w*)", item)
            if n:
                names.append((kind, n.group(1).upper(), "ALLOCATABLE" in m.group(2).upper()))
    return names


def main(src, out):
    lines = ["PROGRAM TABLES49", "USE PARKIND1, ONLY : JPRB, JPIM"]
    calls = []
    for mod in MODULES:
        decl = declared(f"{src}/{mod}.F90")
        if not decl:
            continue
        lines.append(f"USE {mod.upper()}, ONLY : " + ", ".join(
            f"{mod.upper()}_{n} => {n}" for _, n, _ in decl))
        for kind, n, alloc in decl:
            ref = f"{mod.upper()}_{n}"
            call = f"CALL W{kind.upper()}('{mod}.{n}', {ref})"
            calls.append(f"IF (ALLOCATED({ref})) {call}" if alloc else call)
    lines += ["IMPLICIT NONE", "CHARACTER(LEN=512) :: CLDIR", "INTEGER :: IU",
              "CALL GET_COMMAND_ARGUMENT(1, CLDIR)",
              "CALL SURRTAB", "CALL SURRTPK", "CALL SURRTRF",
              "CALL RRTM_INIT_140GP(TRIM(CLDIR))", "CALL SRTM_INIT(TRIM(CLDIR))",
              "OPEN(NEWUNIT=IU, FILE='tables49.txt', STATUS='REPLACE', ACTION='WRITE')"]
    lines += calls
    lines += ["CLOSE(IU)", "CONTAINS", """
SUBROUTINE HDR(CDN, CDK, KR, KL, KU)
  CHARACTER(LEN=*), INTENT(IN) :: CDN, CDK
  INTEGER, INTENT(IN) :: KR, KL(:), KU(:)
  INTEGER :: J
  WRITE(IU,'(A,1X,A,1X,I0,*(1X,I0))') CDN, CDK, KR, (KL(J), KU(J), J=1,KR)
END SUBROUTINE HDR
SUBROUTINE WR(CDN, P)
  CHARACTER(LEN=*), INTENT(IN) :: CDN
  REAL(KIND=JPRB), INTENT(IN) :: P(..)
  CALL HDR(CDN, 'r', RANK(P), LBOUND(P), UBOUND(P))
  SELECT RANK (P)
  RANK (0); WRITE(IU,'(ES26.17E3)') P
  RANK (1); WRITE(IU,'(ES26.17E3)') P
  RANK (2); WRITE(IU,'(ES26.17E3)') P
  RANK (3); WRITE(IU,'(ES26.17E3)') P
  RANK (4); WRITE(IU,'(ES26.17E3)') P
  RANK DEFAULT; STOP 'TABLES49: rank > 4'
  END SELECT
END SUBROUTINE WR
SUBROUTINE WI(CDN, K)
  CHARACTER(LEN=*), INTENT(IN) :: CDN
  INTEGER(KIND=JPIM), INTENT(IN) :: K(..)
  CALL HDR(CDN, 'i', RANK(K), LBOUND(K), UBOUND(K))
  SELECT RANK (K)
  RANK (0); WRITE(IU,'(I0)') K
  RANK (1); WRITE(IU,'(I0)') K
  RANK (2); WRITE(IU,'(I0)') K
  RANK (3); WRITE(IU,'(I0)') K
  RANK DEFAULT; STOP 'TABLES49: integer rank > 3'
  END SELECT
END SUBROUTINE WI
END PROGRAM TABLES49"""]
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{len(calls)} variables from {len(MODULES)} modules")


if __name__ == "__main__":
    main(*sys.argv[1:3])
