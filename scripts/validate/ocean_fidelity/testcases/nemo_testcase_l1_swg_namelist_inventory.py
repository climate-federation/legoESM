#!/usr/bin/env python3
"""Derive every namelist variable the compiled NEMO SWG_DG executable can read.

The inventory is derived from the configuration's preprocessed sources
(``tests/SWG_DG/BLD/ppsrc/nemo``); it is not a hand-written list:

1. parse every routine, generic interface and ``NAMELIST`` declaration;
2. walk the static call graph from ``nemo_gcm``, the only call in NEMO's main
   program (``src/OCE/nemo.f90``), following every ``CALL`` and every
   reference to a known function or generic interface WHATEVER ``IF`` or
   ``SELECT`` guards it.  The result therefore over-approximates what one run
   reads: a group behind a switch that is off is still listed;
3. keep every ``READ`` in a reachable routine whose internal-file unit is a
   NEMO namelist buffer (``numnam*``), resolve its group through the
   declaration in scope, and refuse a ``READ`` it cannot resolve.

A namelist ``READ`` from any other unit (``load_nml`` self-tests the compiler
on the literal string ``cltest`` with groups ``nl1`` and ``nl2``) reads no
namelist file; it is reported under ``excluded_reads``, never dropped silently.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
REFERENCE_INVENTORY = (
    REPO_ROOT / "docs/ocean/fidelity/testcases/"
    "swg_dg_compiled_namelist_reads.json")
SCHEMA = "swg-dg-compiled-namelist-reads-v1"
ENTRY_ROUTINE = "nemo_gcm"

# NEMO spelling -> legoESM card field, only where the names differ beyond
# letter case (the card adds unit suffixes; namsbc's ln_usr is ln_usr_sbc).
CARD_ALIASES = {
    "rn_dt": "rn_Dt_s",
    "rn_theta": "rn_theta_deg",
    "rn_domsiz": "rn_domsiz_m",
    "rn_dx": "rn_dx_m",
    "rn_tau": "rn_tau_pa",
    "rn_modified_grav": "rn_modified_grav_m_s2",
    "rn_rfr": "rn_rfr_s",
    "rn_beta": "rn_beta_m_s",
    "rn_f0": "rn_f0_s",
    "rn_fsp": "rn_fsp_s",
    "rn_time_dmp": "rn_time_dmp_days",
    "rn_time_dmp_out": "rn_time_dmp_out_days",
    "rn_rst_e3t": "rn_rst_e3t_days",
    "rn_lf_cutoff": "rn_lf_cutoff_days",
    "ln_usr": "ln_usr_sbc",
    "rn_uv": "rn_Uv_m_s",
    "rn_lv": "rn_Lv_m",
    "rn_ahm_b": "rn_ahm_b_m2_s",
}

_IDENT = r"[a-z_][a-z0-9_]*"
_UNIT_START = re.compile(
    r"^(?:(?:recursive|pure|elemental|impure|module)\s+)*"
    r"(?:(?:integer|real|logical|character|complex|double\s+precision"
    r"|type\s*\([^)]*\))(?:\s*\([^)]*\))?\s*(?:,\s*[^:]*)?\s*)?"
    rf"(subroutine|function)\s+({_IDENT})", re.I)
_END_UNIT = re.compile(r"^end\s*(subroutine|function)\b", re.I)
_MODULE_START = re.compile(rf"^module\s+(?!procedure\b)({_IDENT})\s*$", re.I)
_END_MODULE = re.compile(r"^end\s*module\b", re.I)
_INTERFACE_START = re.compile(
    rf"^(?:abstract\s+)?interface\b\s*({_IDENT})?", re.I)
_END_INTERFACE = re.compile(r"^end\s*interface\b", re.I)
_MODULE_PROCEDURE = re.compile(r"^(?:module\s+)?procedure\s+(.*)$", re.I)
_NAMELIST = re.compile(r"^namelist\s*(/.*)$", re.I)
_NAMELIST_GROUP = re.compile(rf"/\s*({_IDENT})\s*/([^/]*)", re.I)
_CALL = re.compile(rf"\bcall\s+({_IDENT})", re.I)
_REFERENCE = re.compile(rf"\b({_IDENT})\s*\(", re.I)
_READ = re.compile(r"\bread\s*\(", re.I)
_ECHO = re.compile(
    rf"\bwrite\s*\(\s*numond\s*,\s*(?:nml\s*=\s*)?({_IDENT})\s*\)", re.I)


def _strip_comment(line: str) -> str:
    """Remove a Fortran comment while keeping exclamation marks in strings."""
    quote = None
    for index, character in enumerate(line):
        if character in "'\"":
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        elif character == "!" and quote is None:
            return line[:index]
    return line


def _mask_strings(text: str) -> str:
    """Blank string-literal contents so text inside quotes is never code."""
    masked, quote = [], None
    for character in text:
        if character in "'\"":
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            masked.append(character)
        else:
            masked.append(" " if quote is not None else character)
    return "".join(masked)


def _split_statements(text: str) -> list[str]:
    """Split one logical line at semicolons outside strings."""
    parts, quote, current = [], None, ""
    for character in text:
        if character in "'\"":
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        if character == ";" and quote is None:
            parts.append(current)
            current = ""
        else:
            current += character
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


def logical_statements(text: str) -> list[tuple[int, str]]:
    """Return ``(first physical line, statement)`` with continuations joined."""
    statements: list[tuple[int, str]] = []
    buffer, first = "", 0
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if buffer:
            buffer += " " + (line[1:] if line.startswith("&") else line)
        else:
            buffer, first = line, number
        if buffer.endswith("&"):
            buffer = buffer[:-1]
            continue
        statements.extend((first, part) for part in _split_statements(buffer))
        buffer = ""
    if buffer:
        raise ValueError(f"unterminated continuation starting at line {first}")
    return statements


def _balanced(text: str, start: int) -> tuple[str, int]:
    """Return the text inside the parenthesis opening at ``start``."""
    depth, quote = 0, None
    for index in range(start, len(text)):
        character = text[index]
        if character in "'\"":
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        if quote is not None:
            continue
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index
    raise ValueError(f"unbalanced parenthesis in statement: {text}")


def _top_level_items(text: str) -> list[str]:
    items, depth, quote, current = [], 0, None, ""
    for character in text:
        if character in "'\"":
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        if quote is None and character == "(":
            depth += 1
        elif quote is None and character == ")":
            depth -= 1
        if character == "," and depth == 0 and quote is None:
            items.append(current.strip())
            current = ""
        else:
            current += character
    items.append(current.strip())
    return items


def parse_namelist_declaration(path: Path, group: str) -> tuple[str, ...]:
    """Return names from one preprocessed Fortran ``NAMELIST/group/`` line."""
    for _, statement in logical_statements(path.read_text(errors="replace")):
        match = _NAMELIST.match(statement)
        if not match:
            continue
        for found in _NAMELIST_GROUP.finditer(match.group(1)):
            if found.group(1).lower() == group:
                names = tuple(name.strip().lower()
                              for name in found.group(2).split(",")
                              if name.strip())
                if not names:
                    raise ValueError(f"empty {group} declaration in {path}")
                return names
    raise ValueError(f"missing NAMELIST/{group}/ in {path}")


def _parse_sources(source_root: Path) -> dict:
    """Parse routines, interfaces and namelist declarations of a ppsrc tree."""
    files = sorted(source_root.glob("*.f90"))
    if not files:
        raise ValueError(f"no preprocessed *.f90 sources under {source_root}")
    routines: dict[str, list[dict]] = {}
    module_namelists: dict[str, dict[str, tuple[str, ...]]] = {}
    generic: dict[str, set[str]] = {}
    functions: set[str] = set()
    for path in files:
        module = None
        stack: list[dict] = []
        interface = None
        for number, statement in logical_statements(
                path.read_text(errors="replace")):
            lower = statement.lower()
            if interface is not None:
                if _END_INTERFACE.match(lower):
                    interface = None
                elif interface and (found := _MODULE_PROCEDURE.match(lower)):
                    generic.setdefault(interface, set()).update(
                        name.strip() for name in found.group(1).split(",")
                        if name.strip())
                continue
            found = _INTERFACE_START.match(lower)
            if found and not re.match(
                    r"^interface\s+(operator|assignment)\b", lower):
                interface = (found.group(1) or "").lower()
                continue
            if not stack and (found := _MODULE_START.match(lower)):
                module = found.group(1)
                continue
            if _END_MODULE.match(lower):
                module = None
                continue
            found = _UNIT_START.match(lower)
            if found and not lower.startswith("end"):
                stack.append({
                    "name": found.group(2), "file": path.name,
                    "module": module, "body": [], "namelists": {},
                })
                if found.group(1) == "function":
                    functions.add(found.group(2))
                continue
            if _END_UNIT.match(lower):
                if not stack:
                    raise ValueError(f"unmatched END at {path.name}:{number}")
                record = stack.pop()
                routines.setdefault(record["name"], []).append(record)
                continue
            found = _NAMELIST.match(statement)
            if found:
                for group in _NAMELIST_GROUP.finditer(found.group(1)):
                    fields = tuple(
                        name.strip().lower()
                        for name in group.group(2).split(",") if name.strip())
                    target = (stack[-1]["namelists"] if stack else
                              module_namelists.setdefault(module or "", {}))
                    target[group.group(1).lower()] = fields
                continue
            if stack:
                stack[-1]["body"].append((number, _mask_strings(statement)))
        if stack:
            raise ValueError(
                f"unterminated routine {stack[-1]['name']} in {path.name}")
    return {"routines": routines, "module_namelists": module_namelists,
            "generic": generic, "functions": functions}


def _reachable(program: dict, entry: str) -> set[str]:
    routines, generic = program["routines"], program["generic"]
    callable_names = program["functions"] | set(generic)
    if entry not in routines:
        raise ValueError(f"entry routine {entry} is not defined in the sources")
    seen: set[str] = set()
    pending = [entry]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        targets = set(generic.get(name, ()))
        for record in routines.get(name, ()):
            for _, statement in record["body"]:
                targets.update(found.group(1).lower()
                               for found in _CALL.finditer(statement))
                targets.update(
                    found.group(1).lower()
                    for found in _REFERENCE.finditer(statement)
                    if found.group(1).lower() in callable_names)
        pending.extend(target for target in targets if target not in seen)
    return seen


def derive_read_inventory(source_root: Path,
                          entry: str = ENTRY_ROUTINE) -> dict:
    """Derive the namelist groups and fields reachable from ``entry``."""
    program = _parse_sources(Path(source_root))
    every_group = {
        group
        for records in program["routines"].values() for record in records
        for group in record["namelists"]
    } | {
        group for groups in program["module_namelists"].values()
        for group in groups
    }
    reachable = _reachable(program, entry)
    groups: dict[str, dict] = {}
    excluded: list[dict] = []
    echoed: set[str] = set()
    for name in sorted(reachable):
        for record in program["routines"].get(name, ()):
            scope = dict(program["module_namelists"].get(record["module"] or "", {}))
            scope.update(record["namelists"])
            for number, statement in record["body"]:
                echoed.update(found.group(1).lower()
                              for found in _ECHO.finditer(statement))
                read = _READ.search(statement)
                if read is None:
                    continue
                control, _ = _balanced(statement, read.end() - 1)
                items = _top_level_items(control)
                keyword = [re.fullmatch(rf"nml\s*=\s*({_IDENT})", item, re.I)
                           for item in items]
                group = next((found.group(1).lower()
                              for found in keyword if found), None)
                if group is None and len(items) > 1 and re.fullmatch(
                        _IDENT, items[1], re.I):
                    candidate = items[1].lower()
                    group = candidate if candidate in every_group else None
                unit = re.match(rf"\s*({_IDENT})", items[0], re.I)
                unit_name = unit.group(1).lower() if unit else items[0]
                from_buffer = unit_name.startswith("numnam")
                if group is None:
                    if from_buffer:
                        raise ValueError(
                            f"unresolved namelist READ at {record['file']}:"
                            f"{number}: {statement}")
                    continue
                if not from_buffer:
                    excluded.append({
                        "group": group, "file": record["file"],
                        "routine": name, "line": number, "unit": unit_name,
                    })
                    continue
                if group not in scope:
                    raise ValueError(
                        f"group {group} read at {record['file']}:{number} has "
                        "no NAMELIST declaration in the routine or its module")
                entry_record = groups.setdefault(
                    group, {"fields": [], "read_sites": []})
                for field in scope[group]:
                    if field not in entry_record["fields"]:
                        entry_record["fields"].append(field)
                site = [record["file"], name, number]
                if site not in entry_record["read_sites"]:
                    entry_record["read_sites"].append(site)
    for group, entry_record in groups.items():
        entry_record["echo_written"] = group in echoed
    excluded.sort(key=lambda item: (item["group"], item["file"], item["line"]))
    return {
        "schema": SCHEMA,
        "entry_routine": entry,
        "reachable_routines": len(reachable),
        "groups": dict(sorted(groups.items())),
        "excluded_reads": excluded,
        "group_count": len(groups),
        "field_count": sum(len(item["fields"]) for item in groups.values()),
    }


def load_reference_inventory(path: Path = REFERENCE_INVENTORY) -> dict:
    """Return the committed derivation from the reference SWG_DG build."""
    inventory = json.loads(Path(path).read_text())
    if inventory.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} inventory")
    return inventory


def inventory_differences(derived: dict, reference: dict) -> list[str]:
    """Name every difference in what the two builds read."""
    differences = []
    for group in sorted(set(derived["groups"]) | set(reference["groups"])):
        mine = derived["groups"].get(group)
        theirs = reference["groups"].get(group)
        if mine is None or theirs is None:
            differences.append(
                f"group {group} read only by the "
                f"{'reference' if mine is None else 'derived'} build")
            continue
        for key in ("fields", "read_sites", "echo_written"):
            if mine[key] != theirs[key]:
                differences.append(
                    f"{group}.{key}: derived={mine[key]} reference={theirs[key]}")
    if derived["excluded_reads"] != reference["excluded_reads"]:
        differences.append(
            f"excluded_reads: derived={derived['excluded_reads']} "
            f"reference={reference['excluded_reads']}")
    if derived["entry_routine"] != reference["entry_routine"]:
        differences.append("entry routine differs")
    return differences


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True,
        help="the build's preprocessed sources, e.g. "
             "$NEMO_ROOT/tests/SWG_DG/BLD/ppsrc/nemo")
    parser.add_argument("--reference", type=Path, default=REFERENCE_INVENTORY)
    parser.add_argument("--json", type=Path,
                        help="write the derived inventory (never overwrites)")
    args = parser.parse_args(argv)
    try:
        derived = derive_read_inventory(args.source_root)
        if args.json is not None:
            if args.json.exists():
                raise ValueError(f"refusing to overwrite {args.json}")
            args.json.write_text(json.dumps(derived, indent=2) + "\n")
        differences = inventory_differences(
            derived, load_reference_inventory(args.reference))
        if differences:
            raise ValueError(
                "this build's namelist reads differ from the committed "
                "reference inventory: " + "; ".join(differences))
    except Exception as exc:  # noqa: BLE001 - fail-closed validation CLI
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"SWG_DG_NAMELIST_INVENTORY PASS groups={derived['group_count']} "
          f"fields={derived['field_count']} "
          f"excluded_reads={len(derived['excluded_reads'])} "
          "matches_committed_reference=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
