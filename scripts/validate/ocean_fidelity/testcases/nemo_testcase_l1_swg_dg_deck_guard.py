#!/usr/bin/env python3
"""Guard that a NEMO SWG_DG run read the deck legoESM's card encodes.

Every namelist field the compiled executable can read (the committed
inventory derived from the build's preprocessed sources) is either COMPARED
with the legoESM card or, when the card has no counterpart, PINNED to the
value the reference run read, with a reason per field in the committed
registry.  The guard refuses any echoed or read field that is neither, any
field it cannot resolve, any echoed group or field outside the inventory or
missing from it, and any card field that is neither matched nor registered as
card-only.  Only two output controls may differ from the card.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
NEMO_ONLY_REGISTRY = (
    REPO_ROOT / "docs/ocean/fidelity/testcases/"
    "swg_dg_nemo_only_namelist_fields.json")
sys.path.insert(0, str(HERE))
from nemo_testcase_checkout import bootstrap_checkout  # noqa: E402

CHECKOUT_IMPORT = bootstrap_checkout(__file__, enforce_x64=__name__ == "__main__")
from nemo_testcase_l1_swg_namelist_inventory import (  # noqa: E402
    CARD_ALIASES,
    load_reference_inventory,
)
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

ALLOWLIST = {
    "sn_cfctl.l_runstat": (
        "NEMO writes run.stat for completion evidence; legoESM has no NEMO "
        "run-stat writer, so this output-control switch cannot change state"),
    "nn_stock": (
        "NEMO writes annual restart tiles while legoESM writes a daily NPZ; "
        "restart cadence is output control and cannot change state"),
}

_NOT_READ_BY_SWE = (
    "the {group} group is not read by the SWE executable (absent from the "
    "derived compiled-read inventory); the card states the shipped value only")
# Card fields with no namelist counterpart in the compiled-read inventory.
CARD_ONLY = {
    "key_xios": (
        "CPP key, not a namelist value: build_swg_dg.sh removes it with "
        "makenemo del_key, after refusing any shipped key set other than "
        "'key_xios key_qco key_vco_3d key_RK3'"),
    "key_qco": "CPP key bound by the same build check, not a namelist value",
    "key_vco_3d": "CPP key bound by the same build check, not a namelist value",
    "key_RK3": "CPP key bound by the same build check, not a namelist value",
    "ln_Iperio": (
        "periodicity is hard-coded by SWG usr_def_nam (closed domain, "
        "ldIperio = .FALSE.), not read from a namelist"),
    "ln_Jperio": (
        "periodicity is hard-coded by SWG usr_def_nam (closed domain, "
        "ldJperio = .FALSE.), not read from a namelist"),
    "ln_drg_OFF": _NOT_READ_BY_SWE.format(group="namdrg"),
    "ln_dynspg_exp": _NOT_READ_BY_SWE.format(group="namdyn_spg"),
    "nn_bt_flt": _NOT_READ_BY_SWE.format(group="namdyn_spg"),
    "rn_bt_alpha": _NOT_READ_BY_SWE.format(group="namdyn_spg"),
    "ln_zdftke": _NOT_READ_BY_SWE.format(group="namzdf"),
    "ln_zdfevd": _NOT_READ_BY_SWE.format(group="namzdf"),
    "nn_evdm": _NOT_READ_BY_SWE.format(group="namzdf"),
    "rn_evd_m2_s": _NOT_READ_BY_SWE.format(group="namzdf"),
    "rn_avm0_m2_s": _NOT_READ_BY_SWE.format(group="namzdf"),
    "rn_avt0_m2_s": _NOT_READ_BY_SWE.format(group="namzdf"),
    "nn_avb": _NOT_READ_BY_SWE.format(group="namzdf"),
    "nn_havtb": _NOT_READ_BY_SWE.format(group="namzdf"),
    "nn_etau": _NOT_READ_BY_SWE.format(group="namzdf_tke"),
}

_ASSIGNMENT = re.compile(
    r"([A-Za-z][A-Za-z0-9_%]*(?:\([^)]*\))?)\s*=\s*(.*?)\s*,?$")


def _without_fortran_comment(raw: str) -> str:
    """Remove a Fortran comment while retaining exclamation marks in strings."""
    quote = None
    for index, character in enumerate(raw):
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        elif character == "!" and quote is None:
            return raw[:index]
    return raw


def parse_namelist_file(path: Path) -> tuple[dict, dict, dict]:
    """Return values, unparsed lines and block counts per lower-case group."""
    require(path.is_file(), f"missing NEMO namelist source {path}")
    groups: dict[str, dict[str, object]] = {}
    unparsed: dict[str, list[str]] = {}
    blocks: dict[str, int] = {}
    group = None
    for raw in path.read_text().splitlines():
        line = _without_fortran_comment(raw).strip()
        if not line:
            continue
        if line.startswith("&"):
            group = line[1:].split()[0].lower()
            groups.setdefault(group, {})
            blocks[group] = blocks.get(group, 0) + 1
            continue
        if line == "/":
            group = None
            continue
        if group is None:
            continue
        match = _ASSIGNMENT.fullmatch(line)
        if match:
            groups[group][match.group(1).lower()] = _fortran_value(
                match.group(2))
        else:
            unparsed.setdefault(group, []).append(line)
    require(groups, f"no namelist groups parsed from {path}")
    return groups, unparsed, blocks


def parse_namelist_groups(path: Path) -> dict[str, dict[str, object]]:
    """Parse scalar assignments by group from a NEMO namelist or echo."""
    return parse_namelist_file(path)[0]


def parse_resolved_echo(path: Path) -> dict[str, object]:
    """Return the legacy flat view of NEMO's generated resolved echo."""
    return {
        name: value
        for fields in parse_namelist_groups(path).values()
        for name, value in fields.items()
    }


def _fortran_value(raw: str) -> object:
    value = raw.strip().rstrip(",").strip()
    lower = value.lower()
    if lower in {"t", ".true."}:
        return True
    if lower in {"f", ".false."}:
        return False
    quoted = re.fullmatch(r"'([^']*)'|\"([^\"]*)\"", value)
    if quoted:
        return (quoted.group(1) if quoted.group(1) is not None
                else quoted.group(2)).strip()
    repeated = re.fullmatch(r"(\d+)\s*\*\s*([+-]?[0-9]+)", value)
    if repeated:
        return tuple(
            int(repeated.group(2)) for _ in range(int(repeated.group(1))))
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: object) -> object:
    """The JSON form both sides of a pinned comparison are reduced to."""
    return json.loads(json.dumps(value))


def validate_run_input_hashes(run_dir: Path) -> dict[str, str]:
    """Require the two namelists to match the run's pre-execution manifest."""
    manifest = run_dir / "manifest.sha256"
    require(manifest.is_file(), f"missing run manifest {manifest}")
    recorded = {}
    for raw in manifest.read_text().splitlines():
        parts = raw.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            recorded[parts[1].lstrip("*")] = parts[0]
    result = {}
    for name in ("namelist_ref", "namelist_cfg"):
        path = run_dir / name
        require(name in recorded, f"run manifest does not hash {name}")
        actual = _sha256(path)
        require(actual == recorded[name],
                f"run input hash differs for {name}: "
                f"recorded={recorded[name]} actual={actual}")
        result[name] = actual
    return result


def load_nemo_only_registry(path: Path = NEMO_ONLY_REGISTRY) -> dict:
    """Return the committed per-field registry of NEMO-only fields."""
    registry = json.loads(Path(path).read_text())
    require(registry.get("schema") == "swg-dg-nemo-only-namelist-fields-v1",
            f"{path} is not a NEMO-only field registry")
    for key, entry in registry["fields"].items():
        require(isinstance(entry.get("reason"), str) and entry["reason"],
                f"registry field {key} has no reason")
        require("reference_value" in entry,
                f"registry field {key} has no reference value")
    return registry


def _entries(values: dict[str, object], field: str) -> dict[str, object]:
    """The keys one declared field occupies: itself, %members, (elements)."""
    return {key: value for key, value in values.items()
            if key == field or key.startswith((f"{field}%", f"{field}("))}


def _collapse(entries: dict[str, object], field: str) -> object:
    if set(entries) == {field}:
        return entries[field]
    return dict(sorted(entries.items()))


def resolve_inventory(
        inventory: dict, echo: dict[str, dict[str, object]],
        reference: dict[str, dict[str, object]],
        config: dict[str, dict[str, object]],
) -> dict[tuple[str, str], dict]:
    """Resolve every inventory field from the echo or NEMO's ref->cfg order.

    An echoed group yields NEMO's EFFECTIVE values (NEMO may adjust an input
    before echoing it, e.g. nn_hls = MAX(2, nn_hls)), so a raw input that NEMO
    normalises to the reference value passes; an unechoed group yields the raw
    values read from namelist_ref then namelist_cfg.
    """
    resolved = {}
    for group, item in inventory["groups"].items():
        fields = item["fields"]
        if group in echo:
            claimed = set()
            for field in fields:
                entries = _entries(echo[group], field)
                require(entries, f"echoed group {group} lacks compiled-read "
                        f"field {field}: a comparison row would disappear")
                claimed.update(entries)
                resolved[(group, field)] = {
                    "value": _collapse(entries, field),
                    "source": "output.namelist.dyn"}
            extra = sorted(set(echo[group]) - claimed)
            require(not extra, f"echoed group {group} has field(s) {extra} "
                    "outside the compiled-read inventory")
            continue
        for field in fields:
            from_ref = _entries(reference.get(group, {}), field)
            from_cfg = _entries(config.get(group, {}), field)
            require(from_ref or from_cfg,
                    f"unresolved compiled-read field {group}.{field}: absent "
                    "from output.namelist.dyn, namelist_cfg, and namelist_ref")
            merged = {**from_ref, **from_cfg}
            source = ("namelist_ref" if not from_cfg
                      else "namelist_cfg" if set(from_cfg) == set(merged)
                      else "namelist_ref with namelist_cfg override")
            resolved[(group, field)] = {
                "value": _collapse(merged, field), "source": source}
    return resolved


def classify_fields(inventory: dict, card_fields, registry: dict) -> dict:
    """Map each inventory field to one card field or one registry entry."""
    by_lower = {name.lower(): name for name in card_fields}
    card_of = {}
    for group, item in inventory["groups"].items():
        for field in item["fields"]:
            name = CARD_ALIASES.get(field, field).lower()
            if name in by_lower:
                card_of[(group, field)] = by_lower[name]
    matched = {}
    for key, card_name in card_of.items():
        require(card_name not in matched,
                f"card field {card_name} matches both "
                f"{'.'.join(matched.get(card_name, ()))} and {'.'.join(key)}")
        matched[card_name] = key
    registered = {tuple(key.split(".", 1)) for key in registry["fields"]}
    every = {(group, field) for group, item in inventory["groups"].items()
             for field in item["fields"]}
    both = sorted(".".join(key) for key in set(card_of) & registered)
    require(not both, f"field(s) both compared and registered: {both}")
    neither = sorted(".".join(key) for key in every - set(card_of) - registered)
    require(not neither, "compiled-read field(s) neither compared with the "
            f"card nor registered as NEMO-only: {neither}")
    stale = sorted(".".join(key) for key in registered - every)
    require(not stale, f"registry field(s) outside the inventory: {stale}")
    unmatched = sorted(set(card_fields) - set(matched) - set(CARD_ONLY))
    require(not unmatched, "card field(s) neither matched by a compiled read "
            f"nor registered as card-only: {unmatched}")
    stale_card_only = sorted(set(CARD_ONLY) & set(matched))
    require(not stale_card_only, "card-only field(s) are compiled reads: "
            f"{stale_card_only}")
    missing_card_only = sorted(set(CARD_ONLY) - set(card_fields))
    require(not missing_card_only,
            f"card-only registry names absent card field(s): {missing_card_only}")
    return card_of


def _row(*, group: str, field: str, kind: str, nemo_value: object,
         expected: object, source: str, reason: str | None = None) -> dict:
    equal = _json_value(nemo_value) == _json_value(expected)
    return {
        "group": group,
        "field": field,
        "kind": kind,
        "nemo": nemo_value,
        "expected": expected,
        "source": source,
        "equal": equal,
        "allowlisted": (kind == "card" and not equal and field in ALLOWLIST),
        "reason": (ALLOWLIST.get(field) if kind == "card" and not equal
                   else reason),
    }


def compare_resolved(
        echo: dict[str, dict[str, object]],
        reference: dict[str, dict[str, object]],
        config: dict[str, dict[str, object]], lego_namelist, *,
        inventory: dict | None = None, registry: dict | None = None,
) -> dict:
    """Compare card fields, pin NEMO-only fields, and refuse any gap."""
    inventory = load_reference_inventory() if inventory is None else inventory
    registry = load_nemo_only_registry() if registry is None else registry
    expected_echo = sorted(registry["reference_echo_groups"])
    require(sorted(echo) == expected_echo,
            f"echoed groups differ from the reference run: "
            f"observed={sorted(echo)} expected={expected_echo}")
    card = lego_namelist._asdict()
    card_of = classify_fields(inventory, card, registry)
    resolved = resolve_inventory(inventory, echo, reference, config)
    rows = []
    for (group, field), item in resolved.items():
        if (group, field) in card_of:
            card_name = card_of[(group, field)]
            if card_name == "sn_cfctl":
                members = dict(card[card_name])
                nemo = item["value"]
                require(isinstance(nemo, dict) and set(nemo) == {
                    f"sn_cfctl%{member.lower()}" for member in members},
                    "sn_cfctl members differ between NEMO and the card")
                rows.extend(_row(
                    group=group, field=f"sn_cfctl.{member}", kind="card",
                    nemo_value=nemo[f"sn_cfctl%{member.lower()}"],
                    expected=value, source=item["source"])
                    for member, value in members.items())
                continue
            rows.append(_row(
                group=group, field=card_name, kind="card",
                nemo_value=item["value"], expected=card[card_name],
                source=item["source"]))
            continue
        entry = registry["fields"][f"{group}.{field}"]
        rows.append(_row(
            group=group, field=f"{group}.{field}", kind="nemo-only-pinned",
            nemo_value=item["value"], expected=entry["reference_value"],
            source=item["source"], reason=entry["reason"]))
    card_rows = [row for row in rows if row["kind"] == "card"]
    pinned_rows = [row for row in rows if row["kind"] != "card"]
    expected_rows = (len(card_of) - 1 + len(dict(card["sn_cfctl"]))
                     + len(registry["fields"]))
    require(len(rows) == expected_rows,
            f"comparison rows {len(rows)} differ from the inventory's "
            f"{expected_rows}")
    differences = [row for row in rows if not row["equal"]]
    refused = [row for row in differences if not row["allowlisted"]]
    require(not refused, "non-allowlisted deck differences: "
            + "; ".join(f"{row['field']} nemo={row['nemo']!r} "
                        f"expected={row['expected']!r} ({row['source']})"
                        for row in refused))
    require({row["field"] for row in differences} == set(ALLOWLIST),
            "run-control allowlist did not bind exactly: "
            f"observed={[row['field'] for row in differences]}")
    return {
        "compiled_inventory_groups": len(inventory["groups"]),
        "compiled_inventory_fields": len(resolved),
        "card_compared_fields": len(card_of),
        "nemo_only_pinned_fields": len(pinned_rows),
        "card_only_fields": CARD_ONLY,
        "comparison_rows": len(rows),
        "card_rows": len(card_rows),
        "resolved_sources": {
            f"{group}.{field}": item["source"]
            for (group, field), item in resolved.items()},
        "rows": rows,
        "differences": differences,
        "allowlist": ALLOWLIST,
        "all_non_allowlisted_values_equal": not refused,
    }


def crosscheck_unechoed_groups(
        ocean_output: Path, comparison: dict, inventory: dict,
) -> dict[str, dict]:
    """Check groups NEMO reads without echoing when ocean.output prints them."""
    require(ocean_output.is_file(), f"missing NEMO output {ocean_output}")
    text = ocean_output.read_text(errors="replace")
    silent = {group for group, item in inventory["groups"].items()
              if not item["echo_written"]}
    by_group = {}
    for row in comparison["rows"]:
        if row["group"] not in silent:
            continue
        raw_name = row["field"].split(".")[-1].lower()
        group_result = by_group.setdefault(row["group"], {
            "status": "not_printed",
            "matched_fields": [],
            "note": (
                "NEMO's ocean.output contains no machine-readable value for "
                "this echo-omitted group; the content-hashed namelist_ref/"
                "namelist_cfg precedence is authoritative"),
        })
        nemo_name = next((nemo for nemo, card in CARD_ALIASES.items()
                          if card.lower() == raw_name), raw_name)
        found = re.findall(
            rf"\b{re.escape(nemo_name)}\s*=\s*([^\s,]+)", text, re.IGNORECASE)
        if not found:
            continue
        printed = _fortran_value(found[-1])
        require(printed == row["nemo"],
                f"ocean.output cross-check differs for {row['group']}."
                f"{nemo_name}: printed={printed!r} resolved={row['nemo']!r}")
        group_result["status"] = "matched"
        group_result["matched_fields"].append(nemo_name)
        group_result["note"] = "printed value(s) agree with resolved inputs"
    return by_group


def _load_driver():
    path = REPO_ROOT / "scripts/run/run_swg_double_gyre.py"
    spec = importlib.util.spec_from_file_location("swg_dg_deck_driver", path)
    require(spec is not None and spec.loader is not None,
            f"cannot load driver {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strict_inputs(run_dir: Path, inventory: dict):
    """Parse the three sources; refuse lines or blocks a reader would skip."""
    parsed = {}
    for name in ("output.namelist.dyn", "namelist_ref", "namelist_cfg"):
        values, unparsed, blocks = parse_namelist_file(run_dir / name)
        for group in inventory["groups"]:
            require(group not in unparsed,
                    f"{name} group {group} has line(s) the guard cannot "
                    f"parse: {unparsed.get(group)}")
            require(blocks.get(group, 0) <= 1,
                    f"{name} repeats the {group} group; NEMO reads only the "
                    "first block")
        parsed[name] = values
    return parsed


def run(nemo_run_dir: Path, days: int) -> dict:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card

    run_dir = nemo_run_dir.resolve()
    require(run_dir.is_dir(), f"missing NEMO run directory {run_dir}")
    input_hashes = validate_run_input_hashes(run_dir)
    inventory = load_reference_inventory()
    parsed = _strict_inputs(run_dir, inventory)
    driver = _load_driver()
    namelist = driver.resolved_daily_namelist(days)
    card = build_swg_card(namelist)
    comparison = compare_resolved(
        parsed["output.namelist.dyn"], parsed["namelist_ref"],
        parsed["namelist_cfg"], namelist, inventory=inventory)
    crosschecks = crosscheck_unechoed_groups(
        run_dir / "ocean.output", comparison, inventory)
    return {
        "schema": "swg-dg-cross-model-deck-guard-v3",
        "checkout_import": CHECKOUT_IMPORT,
        "jax_enable_x64": (
            "selected by the checkout bootstrap before importing JAX"),
        "nemo_run_dir": str(run_dir),
        "nemo_sources": {
            "echo": str((run_dir / "output.namelist.dyn").resolve()),
            "reference": str((run_dir / "namelist_ref").resolve()),
            "configuration": str((run_dir / "namelist_cfg").resolve()),
            "precedence": "namelist_ref then namelist_cfg override",
            "manifest_validated_input_sha256": input_hashes,
            "ocean_output_cross_checks": crosschecks,
        },
        "compiled_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        "nemo_only_registry_sha256": _sha256(NEMO_ONLY_REGISTRY),
        "legoesm_days": days,
        "legoesm_card_name": card.case,
        "legoesm_namelist": namelist._asdict(),
        **comparison,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-run-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.json.exists(), f"refusing to overwrite {args.json}")
        payload = run(args.nemo_run_dir, args.days)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 - fail-closed command boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        "SWG_DG_DECK_GUARD PASS "
        f"compiled_reads={payload['compiled_inventory_fields']} "
        f"groups={payload['compiled_inventory_groups']} "
        f"card_compared={payload['card_compared_fields']} "
        f"nemo_only_pinned={payload['nemo_only_pinned_fields']} "
        f"comparison_rows={payload['comparison_rows']} "
        f"allowed_differences={len(payload['differences'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
