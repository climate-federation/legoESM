"""Write an explicit-interface block (<name>.intfb.h) for one 49r1 subroutine.

The 49r1 interface blocks are generated at IFS build time and are not in the
git tree. This writes the equivalent block from the definition: the
SUBROUTINE statement, its USE / IMPLICIT statements and the declarations of
the dummy arguments, nothing else. Any declaration that mixes dummy and
local names, or a dummy left undeclared, is an error (no guessing).
ifs49_build.sh then proves every block equals its definition with the same
procedure-pointer check build49.sh uses (gfortran rejects pointer
assignment unless the characteristics agree), plus a planted mismatch.

usage: python -I ifs49_intfb.py <source.F90> <out.intfb.h>
       python -I ifs49_intfb.py --verify <source.F90> <intfb.h>   (ordered dummies + declarations)
"""
import re
import sys

TYPE_START = re.compile(r"^(REAL|INTEGER|LOGICAL|CHARACTER|TYPE\s*\(|CLASS\s*\(|COMPLEX)", re.I)


def statements(text):
    """Join free-form continuation lines; drop comments (no '!' inside strings here)."""
    out, cur = [], ""
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].rstrip()
        s = line.strip()
        if not s:
            continue
        if s.startswith("&"):
            s = s[1:]
        if cur:
            cur += " " + s.strip()
        else:
            cur = s
        if cur.endswith("&"):
            cur = cur[:-1]
            continue
        out.append(cur)
        cur = ""
    return out


def split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def main(src, dst):
    st = statements(open(src).read())
    i = next(k for k, s in enumerate(st) if re.match(r"^SUBROUTINE\s", s, re.I))
    m = re.match(r"^SUBROUTINE\s+(\w+)\s*\((.*)\)\s*$", st[i], re.I)
    name, args = m.group(1), [a.strip() for a in m.group(2).split(",")]
    dummies = {a.upper() for a in args}
    keep, declared = [], set()
    for s in st[i + 1:]:
        if re.match(r"^(CONTAINS|IF\s*\(\s*LHOOK)", s, re.I):
            break
        if s.startswith("#"):
            continue
        if re.match(r"^(USE\s|IMPLICIT\s)", s, re.I):
            keep.append(s)
            continue
        if TYPE_START.match(s) and "::" in s:
            names = [re.split(r"[\s(=*]", e, 1)[0].upper() for e in split_top(s.split("::", 1)[1])]
            hit = [n for n in names if n in dummies]
            if hit and len(hit) != len(names):
                sys.exit(f"{src}: declaration mixes dummy and local names: {s}")
            if hit:
                keep.append(s)
                declared.update(hit)
    missing = dummies - declared
    if missing:
        sys.exit(f"{src}: dummies not declared with '::': {sorted(missing)}")
    with open(dst, "w") as f:
        f.write("INTERFACE\n")
        f.write(f"SUBROUTINE {name}({', '.join(args)})\n")
        for s in keep:
            f.write(s + "\n")
        f.write(f"END SUBROUTINE {name}\nEND INTERFACE\n")


def signature(path):
    """(ordered dummy names, {dummy: declaration statement}) of the first SUBROUTINE in path."""
    st = statements(open(path).read())
    i = next(k for k, s in enumerate(st) if re.match(r"^SUBROUTINE\s", s, re.I))
    m = re.match(r"^SUBROUTINE\s+(\w+)\s*\((.*)\)\s*$", st[i], re.I)
    args = [a.strip().upper() for a in m.group(2).split(",")]
    decl = {}
    for s in st[i + 1:]:
        if re.match(r"^(CONTAINS|IF\s*\(\s*LHOOK|END\s+SUBROUTINE)", s, re.I):
            break
        if TYPE_START.match(s) and "::" in s:
            for e in split_top(s.split("::", 1)[1]):
                n = re.split(r"[\s(=*]", e, 1)[0].upper()
                if n in args:
                    decl[n] = re.sub(r"\s+", "", (s.split("::", 1)[0] + "::" + e).upper())
    return m.group(1).upper(), args, decl


def verify(src, hdr):
    """Independent check: same name, same ordered dummies, same declaration of each."""
    a, b = signature(src), signature(hdr)
    if a != b:
        sys.exit(f"INTERFACE TEXT MISMATCH {hdr}: {[x for x in zip(a[1], b[1]) if x[0] != x[1]][:3]}")
    print(f"verified {a[0]} ({len(a[1])} dummies)")


if __name__ == "__main__":
    if sys.argv[1] == "--verify":
        verify(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
