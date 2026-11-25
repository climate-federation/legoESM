#!/usr/bin/env python3
"""Which operand a compiled Fortran MIN/MAX returns on a tie or a NaN, per copy.

x86 ``minsd/maxsd SRC, DST`` (AT&T order) return SRC unless DST is strictly
smaller (larger), so SRC is what survives a +0/-0 tie and any NaN operand.
gfortran >= 9 lowers MIN/MAX to MIN_EXPR/MAX_EXPR with unspecified tie order,
and loop versioning/unrolling can give one source line copies with different
orders, so the order is read per instruction, never assumed.

The tool recompiles the record build's ppsrc with the build's own compiler and
``%FCFLAGS`` and refuses unless the object is instruction-identical to the
record's ``BLD/obj``.  For every min/max instruction at a requested line it
reads SRC's value name from ``-fverbose-asm``, finds its MIN/MAX_EXPR in the
optimized GIMPLE of the same function, and prints the surviving operand as an
expression over named arrays and constants (pre-ivopts ``cunroll`` GIMPLE).
``--expect`` checks every copy against a committed table of survivors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

NUM = r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
ASM_LINE = re.compile(r"^# .*/(\w+\.f90):(\d+):")
ASM_MINMAX = re.compile(r"^\s*(v?(?:min|max)sd)\s+(\S+),\s*(%xmm\d+)\s*#\s*([^,]*),\s*(\S+)\s*$")
GIMPLE_FUNC = re.compile(r"^;; Function \S+ \((\S+), funcdef_no")
GIMPLE_LOC = re.compile(r"\[/[^\]]*?/(\w+\.f90):(\d+):\d+[^\]]*\]")
GIMPLE_MINMAX = re.compile(r"^\s*\[(\w+\.f90):(\d+)\]\s*(\S+) = (MIN|MAX)_EXPR <([^,]+), ([^>]+)>;")
GIMPLE_DEF = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?([\w.]+) = (.*);\s*$")
ANON_BASES = (
    "ivtmp",
    "pretmp",
    "prephitmp",
    "M",
    "D",
    "stride",
    "offset",
    "size",
    "ubound",
    "lbound",
)


class AuditError(RuntimeError):
    """Fail-closed audit error."""


def fcflags(build: Path) -> list[str]:
    for line in (build / "arch_nemo.fcm").read_text().splitlines():
        if line.startswith("%FCFLAGS"):
            return line.split()[1:]
    raise AuditError("no %FCFLAGS in arch_nemo.fcm")


def instructions(obj: Path) -> str:
    out = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return "\n".join(out.splitlines()[2:])  # drop the file-name header


def compile_unit(build: Path, gfortran: str, unit: str, work: Path) -> dict[str, Path]:
    src = build / "ppsrc" / "nemo" / f"{unit}.f90"
    (work / "mod").mkdir(parents=True, exist_ok=True)
    asm, obj, record = work / f"{unit}.s", work / f"{unit}.o", build / "obj" / f"{unit}.o"
    subprocess.run(
        [
            gfortran,
            *fcflags(build),
            f"-I{build / 'inc'}",
            f"-J{work / 'mod'}",
            "-S",
            "-fverbose-asm",
            "-fdump-tree-optimized-lineno",
            "-fdump-tree-cunroll-lineno",
            "-o",
            str(asm),
            str(src),
        ],
        check=True,
        cwd=work,
    )
    subprocess.run([gfortran, "-c", str(asm), "-o", str(obj)], check=True, cwd=work)
    if instructions(obj) != instructions(record):
        raise AuditError(f"{unit}: recompiled instructions differ from the record object {record}")
    dumps = {p.name.rsplit(".", 1)[-1]: p for p in work.glob(f"{unit}.f90.*t.*")}
    return {
        "asm": asm,
        "optimized": dumps["optimized"],
        "cunroll": dumps["cunroll"],
        "record": record,
    }


def asm_minmax(asm: Path, lines: set[int]) -> list[dict]:
    out, line, func = [], None, None
    for raw in asm.read_text().splitlines():
        m = ASM_LINE.match(raw)
        if m:
            line = int(m.group(2))
        elif raw.startswith("__") and raw.rstrip().endswith(":"):
            func = raw.rstrip()[:-1]
        elif (m := ASM_MINMAX.match(raw)) and line in lines:
            mn, src, dst, src_name, dst_name = m.groups()
            out.append(
                {
                    "line": line,
                    "function": func,
                    "insn": f"{mn} {src},{dst}",
                    "src": src,
                    "src_name": src_name.strip(),
                    "dst_name": dst_name,
                }
            )
    return out


def minmax_lines(asm: Path, prefixes) -> dict[int, int]:
    """Source line -> min/max count inside the named functions (and their gcc clones)."""
    out: dict[int, int] = {}
    line, func = None, ""
    for raw in asm.read_text().splitlines():
        if m := ASM_LINE.match(raw):
            line = int(m.group(2))
        elif raw.startswith("__") and raw.rstrip().endswith(":"):
            func = raw.rstrip()[:-1].split("_MOD_")[-1]
        elif ASM_MINMAX.match(raw) and any(func == p or func.startswith(p + ".") for p in prefixes):
            out[line] = out.get(line, 0) + 1
    return out


def uncovered(report: dict, complete: dict, non_executing: dict) -> list[str]:
    """Min/max instructions in the rung's functions on a line neither audited nor declared idle."""
    errors = []
    for unit, prefixes in complete.items():
        u = report["units"].get(unit)
        if u is None:
            errors.append(f"{unit}: completeness unit not compiled")
            continue
        known = set(u["lines"]) | {
            int(k.split(":")[1]) for k in non_executing if k.startswith(unit + ":")
        }
        found = minmax_lines(Path(u["asm"]), prefixes)
        u["minmax_lines_in_rung_functions"] = found
        errors += [
            f"{unit}:{n} has {c} min/max instruction(s) not in the table"
            for n, c in sorted(found.items())
            if n not in known
        ]
    return errors


def gimple_functions(dump: Path) -> dict[str, list[str]]:
    funcs: dict[str, list[str]] = {}
    cur = None
    for raw in dump.read_text().splitlines():
        if m := GIMPLE_FUNC.match(raw):
            cur = m.group(1)
            funcs[cur] = []
        elif cur is not None:
            funcs[cur].append(GIMPLE_LOC.sub(lambda g: f"[{g.group(1)}:{g.group(2)}]", raw))
    return funcs


GIMPLE_STMT = re.compile(r"^\s*\[(\w+\.f90):(\d+)\]\s*([\w.]+) = (.*);\s*$")


def def_lines(lines: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in lines:
        if (m := GIMPLE_STMT.match(raw)) and m.group(3) not in out:
            out[m.group(3)] = int(m.group(2))
    return out


def gimple_defs(lines: list[str]) -> dict[str, str]:
    defs: dict[str, str] = {}
    for raw in lines:
        if (m := GIMPLE_DEF.match(raw)) and m.group(1) not in defs:
            defs[m.group(1)] = re.sub(r"\[\w+\.f90:\d+\]\s*", "", m.group(2)).strip()
    return defs


def _base(name: str) -> str | None:
    """Source variable behind a named SSA value (``phrmin.0_13`` -> ``phrmin``)."""
    m = re.fullmatch(r"([A-Za-z]\w*?)(?:\.\d+)?_\d+(?:\(D\))?", name)
    if m and not m.group(1).startswith(ANON_BASES) and not m.group(1).startswith("_"):
        return m.group(1)
    return None


def express(name: str, defs: dict[str, str], depth: int = 6) -> str:
    """The value as an expression over named variables, arrays and constants."""
    name = name.strip()
    if re.fullmatch(NUM, name):
        return repr(float(name))
    if (b := _base(name)) is not None:
        return b
    rhs = defs.get(name)
    if rhs is None or depth == 0:
        return "?"
    if m := re.search(r"\b(?:__\w+_MOD_)?([a-z]\w*)\.data\b", rhs):
        return f"{m.group(1)}[]"
    if "__builtin_malloc" in rhs:
        return "local[]"
    if re.fullmatch(r"[a-z]\w*", rhs):  # a module/global variable load
        return rhs
    if m := re.fullmatch(
        r"(?:MEM\s*(?:<[^>]*>)?\s*\[\(.*?\*\)\s*([\w.]+)[^\]]*\](?:\[.*\])?|\*([\w.]+)|\(\*([\w.]+)\)\[.*\])",
        rhs,
    ):
        return express(m.group(1) or m.group(2) or m.group(3), defs, depth - 1)
    if m := re.fullmatch(
        r"([\w.]+) (?:p)?\+ .*", rhs
    ):  # pointer arithmetic: the base names the array
        inner = express(m.group(1), defs, depth - 1)
        if inner != "?":
            return inner
    if m := re.fullmatch(r"(?:__builtin_|\.)(sqrt|exp)\s*\((.*)\)", rhs):
        return f"{m.group(1).upper()}({express(m.group(2), defs, depth - 1)})"
    if m := re.fullmatch(r"-(\S+)", rhs):
        return f"-({express(m.group(1), defs, depth - 1)})"
    if m := re.fullmatch(r"(\S+) ([-+*/]) (\S+)", rhs):
        left, right = express(m.group(1), defs, depth - 1), express(m.group(3), defs, depth - 1)
        return f"({left} {m.group(2)} {right})"
    if m := re.fullmatch(r"\((?:real|integer)\(kind=\d\)\) (\S+)", rhs):
        return express(m.group(1), defs, depth - 1)
    if m := re.fullmatch(r"(MIN|MAX)_EXPR <([^,]+), ([^>]+)>", rhs):
        left, right = express(m.group(2), defs, depth - 1), express(m.group(3), defs, depth - 1)
        return f"{m.group(1)}({left}, {right})"
    return "?"


def classify(insn: dict, opt: dict[str, list[str]], cun: dict[str, list[str]]) -> dict:
    func = insn["function"]
    if func not in opt:
        raise AuditError(f"{func} absent from the optimized dump")
    odefs = gimple_defs(opt[func])
    kind = "MIN" if "min" in insn["insn"] else "MAX"
    stmts = [
        (lhs, a.strip(), b.strip())
        for raw in opt[func]
        if (m := GIMPLE_MINMAX.match(raw))
        and int(m.group(2)) == insn["line"]
        and m.group(4) == kind
        for lhs, a, b in [(m.group(3), m.group(5), m.group(6))]
    ]
    src, src_name, dst = insn["src"], insn["src_name"], insn["dst_name"]

    def is_src(op: str) -> bool:
        if src.startswith(".LC") or src_name.startswith("tmp"):  # constant-pool / RTL constant
            return bool(re.fullmatch(NUM, op))
        if src_name.startswith("MEM"):
            return re.sub(r"\s", "", odefs.get(op, "")) == re.sub(r"\s", "", src_name)
        return op == src_name or _base(op) == src_name

    def hits_where(dst_ok):
        return [
            (lhs, (a, b), k)
            for lhs, a, b in stmts
            for k, op in enumerate((a, b))
            if is_src(op) and (a, b)[1 - k] != op and dst_ok(lhs, (a, b)[1 - k])
        ]

    # The asm destination names the result when it can.  Otherwise it must name the
    # statement's other operand, so both instruction operands are that statement's
    # operands, and that pair must occur in no other MIN/MAX of the function.
    hits, match = hits_where(lambda lhs, other: dst == lhs), "result"
    if not hits:
        hits, match = hits_where(lambda lhs, other: dst == other), "input"
        pairs = [
            frozenset(x.strip() for x in m.group(5, 6))
            for raw in opt[func]
            if (m := GIMPLE_MINMAX.match(raw))
        ]
        if len(hits) == 1 and pairs.count(frozenset(hits[0][1])) != 1:
            hits = []
    if len(hits) != 1:
        return {**insn, "status": "UNMATCHED", "candidates": stmts}
    lhs, ops, k = hits[0]
    defs = {**odefs, **gimple_defs(cun.get(func, []))}  # pre-ivopts names win
    lines = def_lines(opt[func])
    used = sorted(
        {
            int(m.group(2))
            for raw in opt[func]
            if (m := GIMPLE_STMT.match(raw))
            and re.search(rf"(?<![\w.]){re.escape(lhs)}(?![\w.])", m.group(4))
        }
    )
    return {
        **insn,
        "status": "MATCHED",
        "matched_by": match,
        "gimple": f"{lhs} = {kind}_EXPR <{ops[0]}, {ops[1]}>",
        "kind": kind,
        "keeps": express(ops[k], defs),
        "other": express(ops[1 - k], defs),
        "keeps_def_line": lines.get(ops[k]),
        "other_def_line": lines.get(ops[1 - k]),
        "consumer_lines": used,
    }


def audit(build: Path, gfortran: str, sites: dict[str, list[int]], work: Path) -> dict:
    report: dict = {
        "build": str(build),
        "gfortran": gfortran,
        "fcflags": fcflags(build),
        "units": {},
    }
    for unit, lines in sites.items():
        art = compile_unit(build, gfortran, unit, work)
        opt, cun = gimple_functions(art["optimized"]), gimple_functions(art["cunroll"])
        report["units"][unit] = {
            "record_object": str(art["record"]),
            "record_object_sha256": hashlib.sha256(art["record"].read_bytes()).hexdigest(),
            "recompiled_instructions_equal_record": True,
            "asm": str(art["asm"]),
            "lines": lines,
            "rows": [classify(i, opt, cun) for i in asm_minmax(art["asm"], set(lines))],
        }
    return report


RULE_KEYS = ("kind", "function", "keeps_def_line", "other_def_line")


def _rule_matches(rule: dict, row: dict) -> bool:
    if any(
        k in rule
        and rule[k] != (row["function"].split("_MOD_")[-1] if k == "function" else row.get(k))
        for k in RULE_KEYS
    ):
        return False
    if "consumer" in rule and rule["consumer"] not in row.get("consumer_lines", []):
        return False
    if "not_consumer" in rule and rule["not_consumer"] in row.get("consumer_lines", []):
        return False
    return True


def _keeps_ok(rule: dict, keeps: str) -> bool:
    if "keeps_any" in rule:
        return keeps in rule["keeps_any"]
    if "keeps_re" in rule:
        return re.fullmatch(rule["keeps_re"], keeps) is not None
    return keeps == rule["keeps"]


def check_expectations(report: dict, expect: dict) -> list[str]:
    """Every copy matches exactly one committed rule and keeps that rule's operand.

    ``keeps_any`` marks a copy-dependent site: every listed survivor must occur.
    """
    errors = []
    for site, spec in expect.items():
        unit, line = site.split(":")
        rows = [r for r in report["units"].get(unit, {}).get("rows", []) if r["line"] == int(line)]
        if len(rows) != spec["copies"]:
            errors.append(f"{site}: {len(rows)} copies, expected {spec['copies']}")
        seen: dict[int, set] = {}
        for r in rows:
            idx = [
                i
                for i, u in enumerate(spec["rules"])
                if r["status"] == "MATCHED" and _rule_matches(u, r)
            ]
            if len(idx) != 1:
                errors.append(f"{site} {r['insn']}: {len(idx)} rules match")
                continue
            rule = spec["rules"][idx[0]]
            seen.setdefault(idx[0], set()).add(r["keeps"])
            if not _keeps_ok(rule, r["keeps"]):
                errors.append(f"{site} {r['insn']}: keeps {r['keeps']!r}, table: {rule}")
        for i, rule in enumerate(spec["rules"]):
            if "keeps_any" in rule and seen.get(i, set()) != set(rule["keeps_any"]):
                errors.append(
                    f"{site}: copy-dependent survivors {seen.get(i)} != {rule['keeps_any']}"
                )
    return errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--gfortran", required=True)
    p.add_argument("--site", action="append", required=True, help="unit:line[,line...]")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument(
        "--expect", type=Path, help="JSON {unit:line: {copies, rules: [{kind, ..., keeps}]}}"
    )
    p.add_argument("--artifact", type=Path)
    a = p.parse_args()
    sites = {u: [int(x) for x in ls.split(",") if x] for u, ls in (s.split(":") for s in a.site)}
    report = audit(a.build, a.gfortran, sites, a.work)
    for unit, u in report["units"].items():
        for r in u["rows"]:
            print(
                f"{unit}:{r['line']}\t{r['function'].split('_MOD_')[-1]}\t{r.get('kind')}\t{r['status']}\t"
                f"keeps={r.get('keeps')}@{r.get('keeps_def_line')}\tother={r.get('other')}@{r.get('other_def_line')}"
                f"\tused={r.get('consumer_lines')}"
            )
    errors = [
        f"{u}:{r['line']} {r['insn']} UNMATCHED"
        for u, v in report["units"].items()
        for r in v["rows"]
        if r["status"] != "MATCHED"
    ]
    if a.expect:
        table = json.loads(a.expect.read_text())
        errors += check_expectations(report, table["sites"])
        errors += uncovered(report, table.get("complete", {}), table.get("non_executing", {}))
    report["errors"] = errors
    if a.artifact:
        a.artifact.write_text(json.dumps(report, indent=2) + "\n")
    for e in errors:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
