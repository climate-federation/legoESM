#!/usr/bin/env python3
"""Derive complete O8--O16 read-sets from NEMO's preprocessed Fortran.

This is deliberately a small, fail-closed Fortran data-flow scanner rather than
a hand-maintained inventory.  It follows locally resolvable physics callees,
removes declared locals/dummies, and reports external state read before its
first definition.  Arrays must be in a binary boundary; scalars must be in the
binary header, structural metadata, resolved namelists, or pinned source.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PPSRC = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
    "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo"
)
HERE = Path(__file__).resolve().parent
GATE_PATH = HERE / "nemo_si3_o8_o16_master_gate.py"
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
for package in ("core", "ocean", "ice", "atmosphere", "land", "coupler", "driver"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package))

SOURCE_UNITS = (
    "icethd", "icethd_do", "icethd_zdf", "icethd_zdf_bl99", "icethd_dh",
    "icethd_sal", "icethd_da", "icethd_pnd", "iceitd", "icecor", "icevar",
    "iceupdate", "icealb", "icedyn_adv_pra",
)

ROOTS: dict[str, tuple[str, ...]] = {
    "O8": ("ice_thd_frazil",),
    "O9": ("ice_thd_zdf", "ice_thd_dh", "ice_thd_temp"),
    "O10": ("ice_thd_sal", "ice_thd_temp"),
    "O11": ("ice_thd_da",),
    "O12": ("ice_thd_pnd",),
    "O13": ("ice_itd_rem",),
    "O14": ("ice_thd_do",),
    "O15": ("ice_cor",),
    "O16": ("ice_var_glo2eqv", "ice_var_agg", "ice_update_flx", "ice_update_tau"),
}

ENTRY_KINDS: dict[str, tuple[int, ...]] = {
    "O8": (1,), "O9": (21, 3), "O10": (21, 6), "O11": (21, 8),
    "O12": (10,), "O13": (11,), "O14": (12,), "O15": (13,),
    "O16": (18, 15, 19),
}

HEADER_REQUIRED_SCALARS = {"nn_snwfra"}

# These O9 values are outputs of the stage, not boundary inputs.  In
# ice_thd_zdf_bl99 each is assigned for the active category before any later
# consumption.  They must not be copied from the previous category at kind 3.
STAGE_LOCAL_FILLS: dict[str, dict[str, tuple[str, str]]] = {
    "O9": {
        "qtr_ice_bot": ("ice_thd_zdf_bl99", "qtr_ice_bot_1d"),
        "qcn_ice_bot": ("ice_thd_zdf_bl99", "qcn_ice_bot_1d"),
        "qcn_ice_top": ("ice_thd_zdf_bl99", "qcn_ice_top_1d"),
    },
    "O16": {
        "qt_atm_oi": ("ice_update_flx", "qt_atm_oi"),
        "qt_oce_ai": ("ice_update_flx", "qt_oce_ai"),
    },
}

# These arrays belong to diagnostics, bounds, or the observer's own sparse
# gather machinery.  They do not enter the replayed numerical stage.
NON_REPLAY_ARRAYS = {
    "diag_fs", "diag_ft", "diag_fv", "diag_s", "diag_t", "diag_v",
    "rdiag_fs", "rdiag_ft", "rdiag_fv", "rdiag_s", "rdiag_t", "rdiag_v",
    "nptidx", "tice_cvgerr", "tice_cvgstp",
}

META_SCALARS = {
    "jpi", "jpj", "jpl", "nlay_i", "nlay_s", "ntsi", "ntei", "ntsj", "ntej",
    "nis0", "nie0", "njs0", "nje0", "npti", "jpij", "nit000", "kmm", "kt", "nn_hls",
}

# Non-numerical I/O/diagnostic controls and variables derived from values that
# are already captured in the header/deck.  None is a replay input array.
SOURCE_OR_CONTROL_SCALARS = {
    "e", "iiceprt", "jiceprt", "ldstop", "l_prtctl", "ll_diag_pnd", "nice_zdf", "nitrst", "ln_icectl",
    "ln_icediachk", "ln_timing", "lrst_ice", "lwp", "lwm", "not", "numout",
    "sn_cfctl", "r1_dt_ice", "r1_rcpi", "r1_rho0", "r1_rhoi", "r1_rhos",
    "grav", "rcp", "rcpi", "rho0", "rhoi", "rhos", "rhow", "rlfus", "rpi",
    "rt0", "rtmlt",
}

KEYWORDS = set(
    "allocate allocatable and associate backspace block call case class close "
    "common contains continue cycle data deallocate default dimension do else "
    "elseif elsewhere end enddo endif endwhere entry equivalence error exit "
    "external format function if implicit in include inquire integer intent "
    "interface intrinsic logical module namelist none only open operator optional "
    "or out parameter pointer print private procedure program public pure read "
    "real recursive result return rewind save select stop subroutine target then "
    "stat type use volatile where while write".split()
)
INTRINSICS = set(
    "abs allocated all any ceiling cos count dim exp huge int nint lbound log max maxloc "
    "maxval merge min minloc minval mod pack present product real reshape shape "
    "iom_use iom_varid sign sin size sqrt sum tanh tiny transpose trim ubound unpack".split()
)


@dataclass(frozen=True)
class Routine:
    name: str
    source: str
    statements: tuple[str, ...]
    locals: frozenset[str]
    dummies: frozenset[str]


def _without_comment(line: str) -> str:
    quote = None
    out = []
    for char in line:
        if char in "'\"":
            quote = None if quote == char else (char if quote is None else quote)
        if char == "!" and quote is None:
            break
        out.append(char)
    return "".join(out)


def _logical_statements(text: str) -> list[str]:
    statements: list[str] = []
    current = ""
    for raw in text.splitlines():
        line = _without_comment(raw).strip()
        if not line:
            continue
        if line.startswith("&"):
            line = line[1:].lstrip()
        continued = line.endswith("&")
        if continued:
            line = line[:-1]
        current += " " + line
        if not continued:
            statements.extend(part.strip() for part in current.split(";") if part.strip())
            current = ""
    if current:
        raise ValueError("unterminated Fortran continuation")
    return statements


def _declared_names(statement: str) -> set[str]:
    low = statement.lower()
    if "::" not in low and not re.match(
        r"^(integer|real|logical|character|complex|type\s*\(|class\s*\()\b", low
    ):
        return set()
    tail = low.split("::", 1)[-1]
    names = set()
    depth = 0
    start = 0
    for pos, char in enumerate(tail + ","):
        depth += char == "("
        depth -= char == ")"
        if char == "," and depth == 0:
            item = tail[start:pos].strip()
            match = re.match(r"([a-z]\w*)", item)
            if match:
                names.add(match.group(1))
            start = pos + 1
    return names


def _index_routines(ppsrc: Path) -> tuple[dict[str, Routine], dict[str, tuple[str, ...]]]:
    found: dict[str, Routine] = {}
    interfaces: dict[str, tuple[str, ...]] = {}
    begin = re.compile(r"^\s*(?:[a-z][\w(),=* ]*\s+)?(subroutine|function)\s+([a-z]\w*)", re.I)
    end = re.compile(r"^\s*end\s+(subroutine|function)\b", re.I)
    for unit in SOURCE_UNITS:
        path = ppsrc / f"{unit}.f90"
        if not path.is_file():
            raise ValueError(f"missing preprocessed NEMO source: {path}")
        statements = _logical_statements(path.read_text())
        interface_name = ""
        interface_procedures: list[str] = []
        for statement in statements:
            low = statement.lower()
            match = re.match(r"^interface\s+([a-z]\w*)$", low)
            if match:
                interface_name = match.group(1)
                interface_procedures = []
                continue
            if interface_name and re.match(r"^end\s+interface\b", low):
                interfaces[interface_name] = tuple(interface_procedures)
                interface_name = ""
                interface_procedures = []
                continue
            if interface_name:
                match = re.match(r"^module\s+procedure\s+(.+)$", low)
                if match:
                    interface_procedures.extend(re.findall(r"\b[a-z]\w*\b", match.group(1)))
        pos = 0
        while pos < len(statements):
            match = begin.match(statements[pos])
            if not match:
                pos += 1
                continue
            name = match.group(2).lower()
            stop = pos + 1
            while stop < len(statements) and not end.match(statements[stop]):
                stop += 1
            if stop == len(statements):
                raise ValueError(f"unterminated routine {name} in {path}")
            body = tuple(statements[pos : stop + 1])
            args_match = re.search(rf"\b{name}\s*\(([^)]*)\)", body[0], re.I)
            dummies = frozenset(re.findall(r"\b[a-z]\w*\b", args_match.group(1).lower())) if args_match else frozenset()
            local = set(dummies)
            for statement in body[1:]:
                local.update(_declared_names(statement))
            found[name] = Routine(name, path.name, body, frozenset(local), dummies)
            pos = stop + 1
    return found, interfaces


def _identifiers(text: str) -> set[str]:
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "", text.lower())
    return set(re.findall(r"\b[a-z]\w*\b", text))


def _array_identifiers(text: str) -> set[str]:
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "", text.lower())
    return set(re.findall(r"\b([a-z]\w*)\s*\(", text))


def _assignment(statement: str) -> tuple[str, str] | None:
    depth = 0
    quote = ""
    for pos, char in enumerate(statement):
        if char in "'\"":
            quote = "" if quote == char else (char if not quote else quote)
            continue
        if quote:
            continue
        depth += char == "("
        depth -= char == ")"
        before = statement[pos - 1] if pos else ""
        after = statement[pos + 1] if pos + 1 < len(statement) else ""
        if char == "=" and depth == 0 and before not in "<>=/" and after != "=":
            return statement[:pos], statement[pos + 1 :]
    return None


def _lhs_parts(lhs: str) -> tuple[str, str, bool]:
    """Return assignment target, expressions read on the LHS, and whole-write status."""
    value = lhs.strip()
    prefix_reads: list[str] = []
    conditional = False
    for keyword in ("if", "where", "elsewhere"):
        if not re.match(rf"^{keyword}\b", value):
            continue
        conditional = True
        open_pos = value.find("(")
        if open_pos >= 0:
            depth = 0
            close_pos = -1
            for pos in range(open_pos, len(value)):
                depth += value[pos] == "("
                depth -= value[pos] == ")"
                if depth == 0:
                    close_pos = pos
                    break
            if close_pos >= 0:
                prefix_reads.append(value[open_pos + 1 : close_pos])
                value = value[close_pos + 1 :].strip()
        break
    match = re.match(r"^([a-z]\w*)\s*(.*)$", value)
    if not match:
        return "", " ".join(prefix_reads), False
    target, suffix = match.groups()
    whole_slice = bool(re.fullmatch(r"\(\s*:\s*(?:,\s*:\s*)*\)", suffix.strip()))
    indexed = "(" in suffix and not whole_slice
    if indexed:
        open_pos = suffix.find("(")
        close_pos = suffix.rfind(")")
        prefix_reads.append(suffix[open_pos + 1 : close_pos] if close_pos > open_pos else suffix)
    return target, " ".join(prefix_reads), not conditional and not indexed and (
        not suffix.strip() or whole_slice
    )


def _routine_reads(
    routine: Routine, callable_names: set[str]
) -> tuple[set[str], set[str], set[str]]:
    array_reads: set[str] = set()
    scalar_reads: set[str] = set()
    calls: set[str] = set()
    written_whole: set[str] = set()
    for statement in routine.statements[1:-1]:
        low = statement.lower()
        if _declared_names(low):
            continue
        statement_calls = set(re.findall(r"\bcall\s+([a-z]\w*)", low))
        statement_calls.update(_array_identifiers(low) & callable_names)
        calls.update(statement_calls)
        assignment = _assignment(low)
        read_text = low
        whole_write = ""
        if assignment:
            lhs, rhs = assignment
            lhs_base, lhs_reads, is_whole_write = _lhs_parts(lhs)
            read_text = rhs + " " + lhs_reads
            if lhs_base and is_whole_write:
                whole_write = lhs_base
        candidates = _array_identifiers(read_text)
        candidates -= routine.locals | KEYWORDS | INTRINSICS | callable_names | calls
        candidates -= written_whole
        array_reads.update(candidates)
        identifiers = _identifiers(read_text)
        identifiers -= routine.locals | KEYWORDS | INTRINSICS | callable_names | calls
        identifiers -= candidates | written_whole | {"wp", "true", "false"}
        scalar_reads.update(
            value for value in identifiers if not re.fullmatch(r"[ed]\d+", value)
        )
        if whole_write:
            written_whole.add(whole_write)
    return array_reads, scalar_reads, calls


def _normalise(name: str) -> str:
    value = re.sub(r"_(?:1d|2d|k1)$", "", name.lower())
    return {"a_ib": "a_i_b", "h_ib": "h_i_b", "sss": "sss_m"}.get(value, value)


def _load_gate():
    spec = importlib.util.spec_from_file_location("o8o16_readset_gate", GATE_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load O8-O16 schema")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_coverage(ppsrc: Path) -> tuple[set[str], set[str]]:
    """Mechanically collect namelist names and source-declared constants."""
    namelist: set[str] = set()
    parameters: set[str] = set()
    for path in ppsrc.glob("*.f90"):
        for statement in _logical_statements(path.read_text()):
            low = statement.lower()
            match = re.match(r"namelist\s*/[^/]+/\s*(.*)", low)
            if match:
                namelist.update(re.findall(r"\b[a-z]\w*\b", match.group(1)))
            if ("parameter" in low or "=" in low) and (
                "::" in low or re.match(r"^(integer|real|logical)", low)
            ):
                parameters.update(_declared_names(low))
    return namelist, parameters


def derive_and_gate(ppsrc: Path = DEFAULT_PPSRC, plant: str | None = None) -> dict[str, object]:
    routines, interfaces = _index_routines(ppsrc)
    routine_names = set(routines)
    callable_names = routine_names | set(interfaces)
    gate = _load_gate()
    namelist_names, source_parameters = _source_coverage(ppsrc)
    recorded_scalars = {name.lower() for name in gate.SCALAR_NAMES}
    if plant == "drop-nn_snwfra":
        recorded_scalars.discard("nn_snwfra")
    rows: dict[str, object] = {}
    for rung, roots in ROOTS.items():
        pending = list(roots)
        visited: set[str] = set()
        reads: set[str] = set()
        scalar_reads: set[str] = set()
        sources: set[str] = set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            if name not in routines:
                raise ValueError(f"missing registered NEMO routine {name} for {rung}")
            visited.add(name)
            routine = routines[name]
            sources.add(f"{routine.source}:{name}")
            direct, direct_scalars, calls = _routine_reads(routine, callable_names)
            reads.update(_normalise(value) for value in direct)
            scalar_reads.update(_normalise(value) for value in direct_scalars)
            resolved_calls: set[str] = set()
            for call in calls:
                if call in routines:
                    resolved_calls.add(call)
                resolved_calls.update(interfaces.get(call, ()))
            pending.extend(sorted(resolved_calls))
        reads = {
            name for name in reads
            if name not in NON_REPLAY_ARRAYS and not name.startswith(("diag_", "rdiag_"))
        }
        stage_local_fills: dict[str, str] = {}
        for name, (producer_name, source_name) in STAGE_LOCAL_FILLS.get(rung, {}).items():
            producer = routines[producer_name]
            assigned = any(
                re.search(rf"^\s*{re.escape(source_name)}\s*\([^=]*\)\s*=", statement, re.I)
                for statement in producer.statements
            )
            if not assigned:
                raise ValueError(
                    f"STATUS PLANT-FIRED: {rung} lost stage-local fill for {name}"
                )
            stage_local_fills[name] = f"{producer.source}:{producer_name}"
        reads.difference_update(stage_local_fills)
        covered: set[str] = set()
        for kind in ENTRY_KINDS[rung]:
            fields = gate.KINDS[kind].fields
            if plant == "drop-snwice_mass" and kind == 15:
                fields = tuple(field for field in fields if field.name != "snwice_mass")
            if plant == "drop-rmw-sfx_lam" and kind == 8:
                fields = tuple(field for field in fields if field.name != "sfx_lam")
            covered.update(_normalise(field.name) for field in fields)
        scalar_reads = {
            name for name in scalar_reads - reads - covered
            if not name.startswith(("diag_", "rdiag_"))
        }
        scalar_reads.difference_update(stage_local_fills)
        if plant == "drop-sst_m":
            covered.discard("sst_m")
        elif plant == "drop-qlead":
            covered.discard("qlead")
        missing = sorted(reads - covered)
        if missing:
            raise ValueError(f"STATUS PLANT-FIRED: {rung} source read-set is not recorded: {missing}")
        scalar_coverage: dict[str, str] = {}
        for name in sorted(scalar_reads):
            if name in HEADER_REQUIRED_SCALARS:
                if name in recorded_scalars:
                    scalar_coverage[name] = "record_header"
            elif name in recorded_scalars:
                scalar_coverage[name] = "record_header"
            elif name in META_SCALARS:
                scalar_coverage[name] = "record_metadata"
            elif name in namelist_names:
                scalar_coverage[name] = "resolved_namelist_in_manifest"
            elif name in source_parameters or name in SOURCE_OR_CONTROL_SCALARS:
                scalar_coverage[name] = "pinned_binary_or_source_in_manifest"
        if plant == "drop-rn_hinew" and rung == "O8":
            scalar_coverage.pop("rn_hinew", None)
        missing_scalars = sorted(scalar_reads - set(scalar_coverage))
        if missing_scalars:
            raise ValueError(
                f"STATUS PLANT-FIRED: {rung} scalar read-set is not covered: {missing_scalars}"
            )
        rows[rung] = {
            "roots": list(roots), "callees": sorted(visited - set(roots)),
            "sources": sorted(sources), "array_reads": sorted(reads),
            "scalar_reads": sorted(scalar_reads), "scalar_coverage": scalar_coverage,
            "entry_kinds": list(ENTRY_KINDS[rung]),
            "boundary_kinds": list(gate.PROJECTIONS[rung]), "coverage": sorted(covered),
            "stage_local_fills": stage_local_fills,
        }
    if "sst_m" not in rows["O15"]["array_reads"]:
        raise ValueError("STATUS PLANT-FIRED: O15 mechanical read-set lost sst_m")
    return {"status": "PASS_SOURCE_DERIVED_READ_SETS", "rungs": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ppsrc", type=Path, default=DEFAULT_PPSRC)
    parser.add_argument(
        "--plant", choices=("drop-sst_m", "drop-qlead", "drop-rn_hinew", "drop-snwice_mass",
                            "drop-nn_snwfra", "drop-rmw-sfx_lam")
    )
    args = parser.parse_args(argv)
    try:
        print(json.dumps(derive_and_gate(args.ppsrc, args.plant), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
