"""Exhaustive MIN/MAX/SIGN/SQRT audit of the SI3 O9 routines' object code.

Recompiles each routine's ppsrc with the record build's flags plus ``-g``,
refuses unless the instruction stream equals the record object's, then lists
EVERY ``minsd``/``maxsd``/``sqrtsd`` instruction inside the O9 routines' source
ranges, mapped through the DWARF line table to its source statement, with the
kind of each operand resolved by reaching definitions over the function's
control-flow graph:

- ``ZERO`` (``pxor``/``xorpd`` of a register with itself), ``CONST:<hex>``
  (a ``.rodata`` load), ``VAR`` (anything else: loads, arithmetic, call
  clobbers), ``MIXED`` (different kinds reach on different paths).

``maxsd src, dst`` returns ``src`` on a tie or NaN, so for a site whose one
operand is ``ZERO`` the kind pair says exactly which operand survives a
signed-zero tie.  Two ``VAR`` operands are reported as such: the audit does not
name which Fortran variable each register holds.

It also censuses the source in both directions: every ``MIN``/``MAX``/``SIGN``
token in the ranges and every instruction must meet, or the report lists the
mismatch.  Prints a JSON report; ``--check`` compares it to a committed one.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
BUILD = NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD"
TOOLCHAIN = Path("/home/dbalwada/miniconda3/envs/nemo-build")

# Object -> the O9 routines it holds (ppsrc SUBROUTINE/FUNCTION names).
# icethd.o: only ice_thd_temp; the rest of ice_thd is outside O9.
ROUTINES = {
    "icethd_zdf": ("ice_thd_zdf",),
    "icethd_zdf_bl99": ("ice_thd_zdf_BL99",),
    "icethd_dh": ("ice_thd_dh", "snw_ent"),
    "icethd": ("ice_thd_temp",),
    "icevar": ("ice_var_enthalpy", "ice_var_vremap", "ice_var_snwblow_1d"),
}
OPS = ("minsd", "maxsd", "sqrtsd")
TOKEN = re.compile(r"\b(MIN|MAX|SIGN|SQRT)\s*\(", re.IGNORECASE)


def _flags(build: Path) -> list[str]:
    text = (build / "arch_nemo.fcm").read_text()
    m = re.search(r"^%FCFLAGS\s+(.*)$", text, re.MULTILINE)
    if not m:
        raise SystemExit("no %FCFLAGS in arch_nemo.fcm")
    return m.group(1).split()


def _stream(obj: Path) -> str:
    """Instruction stream with addresses and symbolic targets removed."""
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(obj)],
                         check=True, capture_output=True, text=True).stdout
    lines = []
    for line in out.splitlines():
        m = re.match(r"^\s+[0-9a-f]+:\t(.*)$", line)
        if m:
            lines.append(re.sub(r"[0-9a-f]+ <[^>]*>", "ADDR", m.group(1)).split("#")[0].rstrip())
    return "\n".join(lines)


def _rodata(obj: Path) -> dict[str, bytes]:
    out = subprocess.run(["objdump", "-s", str(obj)], check=True, capture_output=True,
                         text=True).stdout
    sections, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"Contents of section (\S+):", line)
        if m:
            cur = m.group(1) if m.group(1).startswith(".rodata") else None
            if cur:
                sections[cur] = bytearray()
            continue
        m = re.match(r"^ ([0-9a-f]{4,}) ((?:[0-9a-f]{2,8} ?){1,4})", line)
        if m and cur:
            sections[cur] += bytes.fromhex(m.group(2).replace(" ", ""))
    return {k: bytes(v) for k, v in sections.items()}


def _symbols(obj: Path) -> dict[str, tuple[str, int]]:
    """Local ``.LC*`` symbol -> (section, offset)."""
    out = subprocess.run(["objdump", "-t", str(obj)], check=True, capture_output=True,
                         text=True).stdout
    syms = {}
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]+)\s.*\s(\.rodata\S*)\s+[0-9a-f]+\s+(\S+)$", line)
        if m:
            syms[m.group(3)] = (m.group(2), int(m.group(1), 16))
    return syms


def _const(reloc: str | None, ro: dict[str, bytes], syms: dict) -> str | None:
    """Resolve a PC32 relocation (``.LC3-0x4`` or ``.rodata.cst8+0x10-0x4``) to the double's hex."""
    if not reloc:
        return None
    m = re.match(r"([.\w]+)((?:[+-]0x[0-9a-f]+)*)$", reloc)
    if not m:
        return None
    name, addend = m.group(1), sum(int(x, 16) for x in re.findall(r"[+-]0x[0-9a-f]+", m.group(2)))
    if name in syms:
        sec, base = syms[name]
    elif name in ro:
        sec, base = name, 0
    else:
        return None
    off = base + addend + 4
    data = ro[sec]
    if off < 0 or off + 8 > len(data):
        return None
    return float.hex(struct.unpack("<d", data[off:off + 8])[0])


def _split(args: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in args:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _parse(obj: Path, src_name: str):
    """Instructions per function: dicts with addr, op, args, reloc, line."""
    out = subprocess.run(["objdump", "-drl", "--no-show-raw-insn", str(obj)], check=True,
                         capture_output=True, text=True).stdout
    funcs, func, line = collections.OrderedDict(), None, None
    for raw in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", raw)
        if m:
            func = m.group(1)
            funcs[func] = []
            continue
        m = re.match(r"^/\S*/" + re.escape(src_name) + r":(\d+)", raw)
        if m:
            line = int(m.group(1))
            continue
        if re.match(r"^/\S*\.f90:\d+", raw):  # a line from another file (none expected)
            line = None
            continue
        m = re.match(r"^\s+[0-9a-f]+: (R_X86_64_\w+)\s+(\S+)", raw)
        if m and func and funcs[func]:
            funcs[func][-1]["reloc"] = m.group(2)
            continue
        m = re.match(r"^\s*([0-9a-f]+):\t(\S+)\s*(.*?)\s*$", raw)
        if m and func:
            funcs[func].append(dict(addr=int(m.group(1), 16), op=m.group(2),
                                    args=_split(m.group(3).split("#")[0].strip()),
                                    reloc=None, line=line))
    return funcs


_NOWRITE = ("ucomi", "comi", "cmp", "test", "j", "push", "ret", "nop", "endbr")


def _effect(ins, ro, syms):
    """(register written, kind or ('COPY', reg)) for an xmm write, else None; 'CALL' clobbers all."""
    op, a = ins["op"], ins["args"]
    if op.startswith("call"):
        return "CALL"
    if not a or not a[-1].startswith("%xmm") or op.startswith(_NOWRITE):
        return None
    dst = a[-1]
    if op in ("pxor", "xorpd", "xorps") and len(a) == 2 and a[0] == dst:
        return dst, "ZERO"
    if op.startswith(("movsd", "movq", "movapd", "movaps", "movupd")) and len(a) == 2:
        if "%rip" in a[0]:
            c = _const(ins["reloc"], ro, syms)
            return dst, f"CONST:{c}" if c else "VAR"
        if a[0].startswith("%xmm"):
            return dst, ("COPY", a[0])
        return dst, "VAR"
    return dst, "VAR"


def _cfg(insns):
    addrs = [i["addr"] for i in insns]
    index = {a: k for k, a in enumerate(addrs)}
    succ = []
    for k, ins in enumerate(insns):
        op, nxt = ins["op"], ([k + 1] if k + 1 < len(insns) else [])
        if op.startswith("j"):
            t = re.match(r"^([0-9a-f]+)", ins["args"][0]) if ins["args"] else None
            if not t or "*" in ins["args"][0]:
                raise SystemExit(f"indirect jump at {ins['addr']:#x}; CFG not exact")
            tgt = int(t.group(1), 16)
            targets = [index[tgt]] if tgt in index else []
            succ.append(targets if op in ("jmp", "jmpq") else targets + nxt)
        elif op.startswith("ret") or op.startswith("ud2"):
            succ.append([])
        else:
            succ.append(nxt)
    return succ


def _reaching(insns, ro, syms):
    """Per instruction: the definitions (instruction index or 'ENTRY') of each xmm register reaching it."""
    succ = _cfg(insns)
    regs = [f"%xmm{n}" for n in range(16)]
    effects = [_effect(i, ro, syms) for i in insns]
    inn = [None] * len(insns)
    inn[0] = {r: frozenset({"ENTRY"}) for r in regs}
    work = [0]
    while work:
        k = work.pop()
        state = dict(inn[k])
        e = effects[k]
        if e == "CALL":
            state = {r: frozenset({("CALL", k)}) for r in regs}
        elif e:
            state[e[0]] = frozenset({k})
        for s in succ[k]:
            if inn[s] is None:
                inn[s] = state
                work.append(s)
            else:
                merged = {r: inn[s][r] | state[r] for r in regs}
                if merged != inn[s]:
                    inn[s] = merged
                    work.append(s)
    return inn, effects


def _kinds(k, reg, inn, effects, seen=frozenset()):
    """Set of operand kinds reaching instruction ``k`` in ``reg``."""
    kinds = set()
    for d in inn[k][reg]:
        if d == "ENTRY" or isinstance(d, tuple):
            kinds.add("VAR")
            continue
        e = effects[d][1]
        if isinstance(e, tuple):  # register copy: kinds of the source at the copy
            if (d, e[1]) not in seen:
                kinds |= _kinds(d, e[1], inn, effects, seen | {(d, e[1])})
        else:
            kinds.add(e)
    return kinds


def _kind(k, reg, inn, effects):
    kinds = sorted(_kinds(k, reg, inn, effects))
    return kinds[0] if len(kinds) == 1 else "MIXED(" + "|".join(kinds) + ")"


def _statements(path: Path):
    """Physical line -> (statement first line, statement text without comments)."""
    lines = path.read_text().splitlines()
    owner, text, k = {}, {}, 0
    while k < len(lines):
        start, parts = k + 1, []
        while True:
            code = re.sub(r"!.*$", "", lines[k]).rstrip()
            owner[k + 1] = start
            cont = code.endswith("&")
            parts.append(code.rstrip("&").strip().lstrip("&").strip())
            k += 1
            if not cont or k >= len(lines):
                break
        text[start] = " ".join(p for p in parts if p)
    return owner, text


def _ranges(path: Path, names):
    src = path.read_text().splitlines()
    out = {}
    for name in names:
        beg = next(n for n, l in enumerate(src, 1)
                   if re.match(rf"^\s*(SUBROUTINE|FUNCTION)\s+{name}\b", l, re.IGNORECASE))
        end = next(n for n, l in enumerate(src, 1)
                   if re.match(rf"^\s*END\s+(SUBROUTINE|FUNCTION)\s+{name}\b", l, re.IGNORECASE))
        out[name] = (beg, end)
    return out


def audit(build: Path = BUILD, toolchain: Path = TOOLCHAIN) -> dict:
    flags = _flags(build)
    env = dict(os.environ, PATH=f"{toolchain / 'bin'}:{os.environ['PATH']}")
    report = {"build": str(build), "flags": flags, "objects": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for obj_name, names in ROUTINES.items():
            src = build / "ppsrc/nemo" / f"{obj_name}.f90"
            gobj = Path(tmp) / f"{obj_name}.o"
            subprocess.run(["mpif90", "-c", "-cpp", *flags, "-g", f"-I{build / 'inc'}",
                            f"-J{tmp}", f"-I{toolchain / 'include'}", str(src), "-o", str(gobj)],
                           check=True, env=env, capture_output=True)
            record = build / "obj" / f"{obj_name}.o"
            same = _stream(gobj) == _stream(record)
            if not same:
                raise SystemExit(f"{obj_name}: -g recompile differs from the record object")
            ro = _rodata(gobj)
            syms = _symbols(gobj)
            owner, text = _statements(src)
            ranges = _ranges(src, names)
            in_scope = lambda ln: ln is not None and any(b <= ln <= e for b, e in ranges.values())
            sites = collections.defaultdict(lambda: collections.Counter())
            for func, insns in _parse(gobj, src.name).items():
                if not any(i["op"] in OPS and in_scope(i["line"]) for i in insns):
                    continue
                inn, effects = _reaching(insns, ro, syms)
                for k, ins in enumerate(insns):
                    if ins["op"] not in OPS or not in_scope(ins["line"]):
                        continue
                    stmt = owner[ins["line"]]
                    if ins["op"] == "sqrtsd":
                        sites[stmt][("sqrtsd", "-", "-")] += 1
                        continue
                    s, d = ins["args"]
                    if "%rip" in s:
                        c = _const(ins["reloc"], ro, syms)
                        sk = f"CONST:{c}" if c else "VAR"
                    elif s.startswith("%xmm"):
                        sk = _kind(k, s, inn, effects)
                    else:
                        sk = "VAR"
                    sites[stmt][(ins["op"], sk, _kind(k, d, inn, effects))] += 1
            tokens = {}
            for name, (b, e) in ranges.items():
                for ln in range(b, e + 1):
                    if owner.get(ln) == ln:
                        found = [t.upper() for t in TOKEN.findall(text[ln])]
                        if found:
                            tokens[ln] = sorted(found)
            report["objects"][obj_name] = {
                "record_object_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
                "ppsrc_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "routines": {k: list(v) for k, v in ranges.items()},
                "sites": {str(ln): {"statement": text[ln],
                                    "tokens": tokens.get(ln, []),
                                    "instructions": [dict(op=o, src=s, dst=d, count=n)
                                                     for (o, s, d), n in sorted(c.items())]}
                          for ln, c in sorted(sites.items())},
                "token_statements_without_instruction": {
                    str(ln): {"statement": text[ln], "tokens": t}
                    for ln, t in sorted(tokens.items())
                    if ln not in sites and any(x in ("MIN", "MAX", "SQRT") for x in t)},
                "instructions_without_token": sorted(
                    str(ln) for ln, c in sites.items()
                    if not any(t in ("MIN", "MAX", "SQRT") for t in tokens.get(ln, []))),
                "sign_tokens": sorted(str(ln) for ln, t in tokens.items() if "SIGN" in t),
            }
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--build", type=Path, default=BUILD)
    p.add_argument("--toolchain", type=Path, default=TOOLCHAIN)
    p.add_argument("--check", type=Path, help="committed report; exit 1 on any difference")
    args = p.parse_args(argv)
    rep = audit(args.build, args.toolchain)
    text = json.dumps(rep, indent=1, sort_keys=True)
    if args.check:
        same = json.loads(args.check.read_text()) == json.loads(text)
        print("MINMAX_AUDIT", "MATCH" if same else "DIFFERS")
        return 0 if same else 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
