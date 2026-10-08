#!/usr/bin/env python3
"""Render and verify the Decision-79 ORCA2 hierarchy rung-0 deck."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


SOURCE_SHA256 = "036f3cec148b2e89cede910d13189db4cc8e2e74a9b9ecdede7a87a5c14a98ec"
CPP_SHA256 = "2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67"
ZERO_FILE = "rung0_zero_flux"

# Existing assignments changed by the binding rung-0 definition.  Values are
# normalised only for comparison; rendering preserves the source comments.
CHANGES = {
    "namtsd.ln_tsd_dmp": ".false.",
    "namsbc.ln_blk": ".false.",
    "namsbc.nn_ice": "0",
    "namsbc.ln_traqsr": ".false.",
    "namsbc.ln_ssr": ".false.",
    "namsbc.ln_rnf": ".false.",
    "namsbc.nn_fwb": "0",
    "namtra_qsr.ln_qsr_rgb": ".false.",
    "namsbc_rnf.ln_rnf_mouth": ".false.",
    "namagrif.ln_spc_dyn": ".false.",
    "nambbc.ln_trabbc": ".false.",
    "nambbl.ln_trabbl": ".false.",
    "namtra_mle.ln_mle": ".false.",
    "namtra_eiv.ln_ldfeiv": ".false.",
    "namtra_dmp.ln_tradmp": ".false.",
    "namzdf.ln_zdftke": ".false.",
    "namzdf.ln_zdfddm": ".false.",
    "namzdf.ln_zdfiwm": ".false.",
    "namzdf.nn_havtb": "0",
    "namzdf_iwm.ln_tsdiff": ".false.",
}

ADDED = {
    "namsbc.ln_usr": ".false.",
    "namsbc.ln_flx": ".true.",
    "namsbc.ln_abl": ".false.",
    "namsbc.ln_cpl": ".false.",
    "namsbc.ln_mixcpl": ".false.",
    "namsbc.ln_dm2dc": ".false.",
    "namzdf.ln_zdfcst": ".true.",
    "namzdf.ln_zdfric": ".false.",
    "namzdf.ln_zdfgls": ".false.",
    "namzdf.ln_zdfosm": ".false.",
    "namzdf.ln_zdfnpc": ".false.",
    "namzdf.ln_zdfmfc": ".false.",
    "namzdf.ln_zdfswm": ".false.",
}

EXPECTED = {
    **CHANGES,
    **ADDED,
    "namzdf.ln_zdfevd": ".true.",
    "namzdf.rn_avm0": "1.2e-4",
    "namzdf.rn_avt0": "1.2e-5",
}

PLANTS = ("none", "extra-delta", "mixing-value", "live-module", "cpp")


class GateError(RuntimeError):
    """The rung-0 deck violated a frozen round-82 predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(value: str) -> str:
    return value.split("!", 1)[0].strip().lower()


def _group_spans(text: str) -> dict[str, tuple[int, int]]:
    spans = {}
    pattern = re.compile(
        r"(?ms)^[ \t]*&([A-Za-z_]\w*)\b.*?^[ \t]*/[ \t]*(?:![^\n]*)?$"
    )
    for match in pattern.finditer(text):
        group = match.group(1).lower()
        require(group not in spans, f"duplicate namelist group {group}")
        spans[group] = match.span()
    return spans


def _replace_in_group(text: str, qualified: str, value: str) -> str:
    group, key = qualified.split(".", 1)
    spans = _group_spans(text)
    require(group in spans, f"missing group {group}")
    start, end = spans[group]
    body = text[start:end]
    pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*)([^!\n]*)(.*)$", re.I)
    body, count = pattern.subn(rf"\g<1>{value:<12}\g<3>", body)
    require(count == 1, f"{qualified}: expected one assignment, found {count}")
    return text[:start] + body + text[end:]


def _add_to_group(text: str, qualified: str, value: str) -> str:
    group, key = qualified.split(".", 1)
    spans = _group_spans(text)
    require(group in spans, f"missing group {group}")
    start, end = spans[group]
    body = text[start:end]
    require(re.search(rf"(?mi)^\s*{re.escape(key)}\s*=", body) is None,
            f"{qualified}: assignment already exists")
    slash = body.rfind("/")
    require(slash >= 0, f"{group}: missing terminator")
    body = body[:slash] + f"   {key:<14} = {value}\n" + body[slash:]
    return text[:start] + body + text[end:]


def render_rung0(source: str) -> str:
    """Return the exact rung-0 namelist derived from the admitted ORCA2 deck."""
    text = source
    for qualified, value in CHANGES.items():
        text = _replace_in_group(text, qualified, value)
    for qualified, value in ADDED.items():
        text = _add_to_group(text, qualified, value)

    spans = _group_spans(text)
    start, _ = spans["namsbc"]
    insert = text.find("\n", spans["namsbc"][1]) + 1
    require(insert > start, "cannot place namsbc_flx after namsbc")
    flux = f"""!-----------------------------------------------------------------------
&namsbc_flx    ! rung-0 exact-zero flux formulation (ln_flx=T)
!-----------------------------------------------------------------------
   cn_dir      = './'
   sn_utau     = '{ZERO_FILE}', -12., 'utau', .false., .true., 'yearly', '', '', ''
   sn_vtau     = '{ZERO_FILE}', -12., 'vtau', .false., .true., 'yearly', '', '', ''
   sn_qtot     = '{ZERO_FILE}', -12., 'qtot', .false., .true., 'yearly', '', '', ''
   sn_qsr      = '{ZERO_FILE}', -12., 'qsr',  .false., .true., 'yearly', '', '', ''
   sn_emp      = '{ZERO_FILE}', -12., 'emp',  .false., .true., 'yearly', '', '', ''
/
"""
    return text[:insert] + flux + text[insert:]


def _changed_assignments(before: dict[str, str], after: dict[str, str]) -> tuple[list[str], list[str]]:
    changed = sorted(key for key in before if key in after and _normalise(before[key]) != _normalise(after[key]))
    added = sorted(set(after) - set(before))
    return changed, added


def validate(source: Path, candidate: Path, cpp: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    require(sha256(source) == SOURCE_SHA256, "source namelist SHA-256 changed")
    source_text = source.read_text()
    expected_text = render_rung0(source_text)
    actual_text = candidate.read_text()
    cpp_digest = sha256(cpp)
    if plant == "extra-delta":
        actual_text = actual_text.replace("nn_fsbc     = 2", "nn_fsbc     = 3", 1)
    elif plant == "mixing-value":
        actual_text = actual_text.replace("rn_avm0     =   1.2e-4", "rn_avm0     =   1.3e-4", 1)
    elif plant == "live-module":
        actual_text, count = re.subn(
            r"(?m)^(\s*ln_tradmp\s*=\s*)\.false\.", r"\g<1>.true.",
            actual_text, count=1,
        )
        require(count == 1, "live-module plant target is absent")
    elif plant == "cpp":
        cpp_digest = "0" * 64
    require(cpp_digest == CPP_SHA256, "CPP card SHA-256 changed")
    require(actual_text == expected_text, "candidate is not the exact rendered rung-0 deck")

    before = namelist_values(source)
    # Parse the possibly planted text without accepting a post-hoc temp file.
    after = namelist_values(candidate)
    changed, added = _changed_assignments(before, after)
    require(changed == sorted(CHANGES), f"changed assignment inventory differs: {changed}")
    expected_added = sorted(set(ADDED) | {
        "namsbc_flx.cn_dir", "namsbc_flx.sn_utau", "namsbc_flx.sn_vtau",
        "namsbc_flx.sn_qtot", "namsbc_flx.sn_qsr", "namsbc_flx.sn_emp",
    })
    require(added == expected_added, f"added assignment inventory differs: {added}")
    for key, wanted in EXPECTED.items():
        require(key in after and _normalise(after[key]) == wanted,
                f"{key}: resolved deck value {_normalise(after.get(key, 'MISSING'))!r} != {wanted!r}")
    for key in ("sn_utau", "sn_vtau", "sn_qtot", "sn_qsr", "sn_emp"):
        value = _normalise(after[f"namsbc_flx.{key}"])
        require(f"'{ZERO_FILE}'" in value and "-12." in value and ".true." in value,
                f"namsbc_flx.{key}: not the frozen climatological zero stream")
    return {
        "format": "nemo-testcase-l4-orca2-round82-rung0-deck-v1",
        "status": "PASS_RUNG0_DECK",
        "source_sha256": SOURCE_SHA256,
        "candidate_sha256": hashlib.sha256(expected_text.encode()).hexdigest(),
        "cpp_sha256": CPP_SHA256,
        "changed_assignments": changed,
        "added_assignments": added,
        "constant_vertical_mixing": {"rn_avm0": 1.2e-4, "rn_avt0": 1.2e-5},
        "surface_boundary": "exact-zero flux file",
        "ice": "off",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--cpp", type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        require(sha256(args.source) == SOURCE_SHA256, "source namelist SHA-256 changed")
        if args.render:
            require(args.plant == "none", "render mode does not accept plants")
            args.render.write_text(render_rung0(args.source.read_text()))
            print(f"STATUS RENDERED_RUNG0_DECK {args.render}")
            return 0
        require(args.candidate is not None and args.cpp is not None,
                "validation requires --candidate and --cpp")
        report = validate(args.source, args.candidate, args.cpp, args.plant)
    except (GateError, KeyError, OSError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_RUNG0_DECK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
