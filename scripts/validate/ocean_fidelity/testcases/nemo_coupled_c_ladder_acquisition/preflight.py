#!/usr/bin/env python3
"""Read-only source/deck contract for the coupled ocean--SI3 acquisition.

This is deliberately not a NEMO runner.  It records facts common to every
candidate C-ladder design and refuses if the live coupled ppsrc no longer has
the call and averaging order on which any future observer patches must sit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NEMO_DEFAULT = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
COUPLED = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY"
STANDALONE = "ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y"
CANDIDATES = {
    "ORCA2_ICE_PISCES": "EXPREF",
    "ORCA2_SAS_ICE": "EXPREF",
    COUPLED: "EXP00",
    STANDALONE: "EXP00",
}
ICE_CADENCE = r"IF\s*\(\s*MOD\s*\(\s*kt\s*-\s*1\s*,\s*nn_fsbc\s*\)\s*==\s*0\s*\)\s*THEN"
WRITER = Path(__file__).with_name("c_ladder_writer.F90")


class ContractError(ValueError):
    """A required live-source or candidate-deck fact is absent."""


def _last_int(text: str, name: str) -> int:
    values = re.findall(rf"^\s*{re.escape(name)}\s*=\s*(-?\d+)", text, re.I | re.M)
    if not values:
        raise ContractError(f"{name} is not set")
    return int(values[-1])


def _keys(path: Path) -> list[str]:
    text = path.read_text()
    match = re.search(r"bld::tool::fppkeys\s+(.*)", text)
    if not match:
        raise ContractError(f"no fppkeys in {path}")
    return match.group(1).split()


def _ordered(text: str, labels: tuple[tuple[str, str], ...]) -> list[str]:
    cursor = 0
    found: list[str] = []
    for label, pattern in labels:
        match = re.search(pattern, text[cursor:], re.I | re.S)
        if not match:
            raise ContractError(f"missing/out-of-order {label}")
        cursor += match.end()
        found.append(label)
    return found


def _writer_layout_contract(plant: str | None) -> dict:
    text = WRITER.read_text()
    if plant == "old-unconditional-layout":
        text = text.replace("19+MERGE(1,0,have_ssh_ib)", "20", 1)
        text = text.replace("IF(have_ssh_ib) WRITE(u) ssh_ib", "WRITE(u) ssh_ib", 1)
    required = {
        "allocation": "have_ssh_ib=ALLOCATED(ssh_ib)",
        "flag-allocation-consistency": "IF(aprdyn.NEQV.have_ssh_ib) ERROR STOP",
        "dynamic-field-count": "19+MERGE(1,0,have_ssh_ib)",
        "conditional-payload": "IF(have_ssh_ib) WRITE(u) ssh_ib",
    }
    missing = [label for label, needle in required.items() if needle not in text]
    if missing:
        raise ContractError(f"C writer kind-1 layout lacks {', '.join(missing)}")
    return {"status": "PASS_C_WRITER_ALLOCATION_LAYOUT", "checks": list(required)}


def _candidate_report(nemo: Path, name: str, exp: str) -> dict:
    root = nemo / "cfgs" / name
    cfg = (root / exp / "namelist_cfg").read_text()
    keys = _keys(root / f"cpp_{name}.fcm")
    return {
        "cpp_keys": keys,
        "driver": "stp_RK3" if "key_RK3" in keys else "stpmlf",
        "key_si3": "key_si3" in keys,
        "key_top": "key_top" in keys,
        "nn_fsbc": _last_int(cfg, "nn_fsbc"),
        "nn_ice": _last_int(cfg, "nn_ice"),
    }


def source_contract(nemo: Path, plant: str | None = None) -> dict:
    writer_layout = _writer_layout_contract(plant)
    reports = {
        name: _candidate_report(nemo, name, exp)
        for name, exp in CANDIDATES.items()
    }
    coupled = reports[COUPLED]
    if not coupled["key_si3"] or coupled["key_top"]:
        raise ContractError("coupled lane deck must have SI3 and no TOP")
    if coupled["driver"] != "stp_RK3" or coupled["nn_fsbc"] != 2 or coupled["nn_ice"] != 2:
        raise ContractError("coupled lane deck is not RK3/SI3 with nn_fsbc=2")
    # The round-342 receipt describes a build-time del_key/namelist overlay;
    # the mutable cfg directory is not the immutable receipt and currently
    # carries its parent profile.  Report that fact, but do not rewrite history
    # by treating the present directory as the certified executed deck.
    standalone = reports[STANDALONE]

    ppsrc = nemo / "cfgs" / COUPLED / "BLD" / "ppsrc" / "nemo"
    texts = {
        name: (ppsrc / f"{name}.f90").read_text()
        for name in ("stprk3", "sbcmod", "sbcssm", "icestp")
    }
    if plant == "drop-sbc-ssm":
        texts["sbcmod"] = re.sub(
            r"CALL\s+sbc_ssm\s*\([^\n]+", "", texts["sbcmod"], count=1, flags=re.I
        )
    elif plant == "swap-flux-update":
        texts["icestp"] = texts["icestp"].replace(
            "CALL ice_update_flx( kt )", "CALL ice_update_tau( kt )", 1
        )
    elif plant == "pretend-mlf":
        reports[COUPLED]["driver"] = "stpmlf"
    if reports[COUPLED]["driver"] != "stp_RK3":
        raise ContractError("plant fired: active coupled driver is not stp_RK3")

    order = {
        "stp_RK3": _ordered(texts["stprk3"], (
            ("sbc(kstp,Nbb,Nbb)", r"CALL\s+sbc\s*\(\s*kstp\s*,\s*Nbb\s*,\s*Nbb\s*\)"),
            ("stp_2D", r"CALL\s+stp_2D\s*\("),
            ("RK3-stage-1", r"CALL\s+stp_RK3_stg\s*\(\s*1\s*,"),
            ("RK3-stage-2", r"CALL\s+stp_RK3_stg\s*\(\s*2\s*,"),
            ("RK3-stage-3", r"CALL\s+stp_RK3_stg\s*\(\s*3\s*,"),
        )),
        "sbc": _ordered(texts["sbcmod"], (
            ("sbc_ssm", r"CALL\s+sbc_ssm\s*\("),
            ("ice_stp", r"CALL\s+ice_stp\s*\("),
            ("runoff", r"CALL\s+sbc_rnf\s*\("),
            ("final-flux-halo", r"CALL\s+lbc_lnk\s*\(\s*'sbcmod'\s*,\s*utau"),
            ("T-to-UV-stress", r"utauU\s*\([^=]+="),
        )),
        "ice_stp": _ordered(texts["icestp"], (
            ("ice-cadence", ICE_CADENCE),
            ("mean-ocean-copy", r"u_oce\s*\([^=]+?=\s*ssu_m"),
            ("ice_sbc_tau", r"CALL\s+ice_sbc_tau\s*\("),
            ("ice_dyn", r"CALL\s+ice_dyn\s*\("),
            ("ice_sbc_flx", r"CALL\s+ice_sbc_flx\s*\("),
            ("ice_thd", r"CALL\s+ice_thd\s*\("),
            ("ice_update_flx", r"CALL\s+ice_update_flx\s*\("),
            ("ice-cadence-end", r"ENDIF\s*!\s*End sea-ice time step only"),
            ("ice_update_tau-every-ocean-step", r"CALL\s+ice_update_tau\s*\("),
        )),
        "sbc_ssm": _ordered(texts["sbcssm"], (
            ("reset", r"sst_m\s*\(:,:\)\s*=\s*0\._wp"),
            ("accumulate", r"sst_m\s*\(:,:\)\s*=\s*sst_m\s*\(:,:\)\s*\+"),
            ("mean-cadence", ICE_CADENCE),
            ("divide-to-mean", r"sst_m\s*\(:,:\)\s*=\s*sst_m\s*\(:,:\)\s*\*\s*zcoef"),
        )),
    }
    if re.search(r"\bsbc_ice_si3\b", "\n".join(texts.values()), re.I):
        raise ContractError("unexpected sbc_ice_si3 symbol; audit the NEMO version")
    return {
        "status": "PASS_COUPLED_C_SOURCE_CONTRACT",
        "nemo_run": False,
        "candidates": reports,
        "live_ppsrc": str(ppsrc),
        "call_order": order,
        "writer_layout": writer_layout,
        "facts": {
            "sbc_ice_si3_exists": False,
            "ice_is_called_every_ocean_steps": coupled["nn_fsbc"],
            "standalone_cfg_currently_has_parent_si3_profile": (
                standalone["key_si3"] and standalone["nn_ice"] == 2
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, default=NEMO_DEFAULT)
    parser.add_argument(
        "--plant",
        choices=(
            "drop-sbc-ssm",
            "swap-flux-update",
            "pretend-mlf",
            "old-unconditional-layout",
        ),
    )
    args = parser.parse_args()
    try:
        report = source_contract(args.nemo_root.resolve(), args.plant)
    except (ContractError, FileNotFoundError) as exc:
        print(f"STATUS PLANT-FIRED: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
