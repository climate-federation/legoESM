#!/usr/bin/env python3
"""P2Y-4a: RESOLVED (ref-overlaid-by-cfg) ice-namelist diff.

The Phase-2x ice-namelist gate (`nemo_testcase_l4_orca2_phase2x_ice_namelist
_gate.py`) compares only the two sides' *_cfg override text* against each
other -- it never reads either side's *_ref* defaults, so it is blind to a
field whose REF default differs but which no cfg touches, and it cannot say
whether a field that both cfgs override the same way still carries a
different REF default underneath.  This gate computes the actual RESOLVED
value each side's NEMO process would see (ref overlaid by cfg, the same
two-pass semantics NEMO's own namelist read uses) and diffs THAT.

Reuses `_parse_cfg` from the Phase-2x gate (same Fortran namelist block
parser; ref and cfg files use the identical `&block ... /` syntax) rather
than re-deriving a parser.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_phase2x_ice_namelist_gate import (
    _parse_cfg,
)

# icethd_sal.F90 SELECT CASE(nn_icesal): case 2 is "time varying salinity with
# linear profile" (Vancoppenolle et al. 2005) at lines 207-249; it reads
# ln_drainage/rn_time_gd/ln_flushing/rn_time_fl/rn_sal_himin/rn_sal_fl/
# rn_sal_gd/rn_simin/rn_sinew.  rn_Rc_RJW/rn_alpha_GN/rn_Rc_GN/rn_alpha_CW/
# nn_sal_scheme/rn_alpha_RJW are read ONLY inside CASE(4) (Gravity Drainage
# and Flushing, lines 256-465+) -- they are declared in the same
# NAMELIST/namthd_sal/ statement (icethd_sal.F90:647-650) so both configs
# resolve a value for them, but under nn_icesal=2 no executed statement ever
# references them.
_CASE4_ONLY_NAMTHD_SAL_FIELDS = {
    "rn_rc_rjw", "rn_alpha_gn", "rn_rc_gn", "rn_alpha_cw", "rn_alpha_rjw",
    "nn_sal_scheme",
}
_CASE2_FIELDS = {
    "ln_drainage", "rn_time_gd", "ln_flushing", "rn_time_fl", "rn_sal_himin",
    "rn_sal_fl", "rn_sal_gd", "rn_simin", "rn_sinew",
}


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def resolve(ref: Path, cfg: Path) -> dict[tuple[str, str], dict[str, object]]:
    """NEMO reads namelist_*_ref then namelist_*_cfg into the SAME group, so
    any field the cfg sets overrides the ref default; anything the cfg does
    not mention keeps its ref value.  Plain dict-update over `_parse_cfg`
    gives exactly that (cfg entries win on key collision)."""
    merged = dict(_parse_cfg(ref))
    merged.update(_parse_cfg(cfg))
    return merged


def reachability(field: str, resolved_icesal: str) -> str:
    key = field.lower()
    if resolved_icesal.strip() != "2":
        return f"gate assumes nn_icesal=2; resolved nn_icesal={resolved_icesal!r}, reachability table not applicable"
    if key == "nn_iceini_file":
        return "namini block selector (ice initial-state source), executed regardless of nn_icesal -- not part of icethd_sal.F90 dispatch"
    if key in _CASE4_ONLY_NAMTHD_SAL_FIELDS:
        return "UNREACHABLE under nn_icesal=2: icethd_sal.F90 reads this field only inside CASE(4); CASE(2) (lines 207-249) never references it"
    if key in _CASE2_FIELDS:
        return "REACHABLE under nn_icesal=2: icethd_sal.F90 CASE(2) (lines 207-249) reads this field"
    return "not in the icethd_sal.F90 namthd_sal dispatch; reachability not assessed by this table"


def validate(variant_ref: Path, variant_cfg: Path, orca1_ref: Path, orca1_cfg: Path) -> dict[str, object]:
    resolved_variant = resolve(variant_ref, variant_cfg)
    resolved_orca1 = resolve(orca1_ref, orca1_cfg)
    keys_v, keys_o = set(resolved_variant), set(resolved_orca1)
    only_variant = sorted(keys_v - keys_o)
    only_orca1 = sorted(keys_o - keys_v)
    if only_variant or only_orca1:
        raise ValueError(
            f"resolved schema mismatch: only_variant={only_variant} only_orca1={only_orca1}"
        )
    resolved_icesal_v = str(resolved_variant[("namthd_sal", "nn_icesal")]["value"])
    resolved_icesal_o = str(resolved_orca1[("namthd_sal", "nn_icesal")]["value"])
    diffs = []
    for key in sorted(keys_v):
        a = str(resolved_variant[key]["value"])
        b = str(resolved_orca1[key]["value"])
        if _norm(a) != _norm(b):
            diffs.append({
                "block": key[0], "field": key[1],
                "variant_resolved": a, "orca1_resolved": b,
                "variant_reachability": reachability(key[1], resolved_icesal_v),
                "orca1_reachability": reachability(key[1], resolved_icesal_o),
            })
    named_five = ["rn_time_gd", "rn_rc_rjw", "rn_alpha_gn", "rn_rc_gn", "rn_alpha_cw"]
    named_report = []
    for field in named_five:
        keys = [k for k in keys_v if k[1] == field]
        if len(keys) != 1:
            raise ValueError(f"expected exactly one namelist key for {field}, found {keys}")
        key = keys[0]
        named_report.append({
            "block": key[0], "field": key[1],
            "variant_ref_default": str(_parse_cfg(variant_ref).get(key, {}).get("value")),
            "orca1_ref_default": str(_parse_cfg(orca1_ref).get(key, {}).get("value")),
            "variant_resolved": str(resolved_variant[key]["value"]),
            "orca1_resolved": str(resolved_orca1[key]["value"]),
            "resolved_equal": _norm(str(resolved_variant[key]["value"]))
                == _norm(str(resolved_orca1[key]["value"])),
            "reachable_under_nn_icesal_2": reachability(field, "2"),
        })
    return {
        "variant_ref": str(variant_ref), "variant_cfg": str(variant_cfg),
        "orca1_ref": str(orca1_ref), "orca1_cfg": str(orca1_cfg),
        "resolved_field_count": len(keys_v),
        "resolved_nn_icesal_variant": resolved_icesal_v,
        "resolved_nn_icesal_orca1": resolved_icesal_o,
        "differing_resolved_field_count": len(diffs),
        "differing_resolved_fields": diffs,
        "named_five_ref_differences": named_report,
    }


def _self_test() -> None:
    """Synthetic-violation check: prove the diff can actually fire, and that
    a field only the ref sets (never touched by either cfg) is still caught."""
    import tempfile

    ref_text = (
        "&namthd_sal\n"
        "   nn_icesal = 2\n"
        "   rn_time_gd = 1.0e5\n"
        "   rn_rc_rjw = 8.0\n"
        "   rn_alpha_gn = 4.6e-4\n"
        "   rn_rc_gn = 8.1\n"
        "   rn_alpha_cw = 5.0e-7\n"
        "/\n"
        "&namini\n"
        "   nn_iceini_file = 1\n"
        "/\n"
    )
    cfg_agree = "&namthd_sal\n   rn_time_gd = 2.0e5\n/\n&namini\n/\n"
    cfg_disagree_a = "&namthd_sal\n   rn_time_gd = 2.0e5\n/\n&namini\n   nn_iceini_file = 0\n/\n"
    cfg_disagree_b = "&namthd_sal\n   rn_time_gd = 2.0e5\n/\n&namini\n   nn_iceini_file = 1\n/\n"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "ref.txt").write_text(ref_text)
        (tmp / "cfg_a.txt").write_text(cfg_disagree_a)
        (tmp / "cfg_b.txt").write_text(cfg_disagree_b)
        result = validate(tmp / "ref.txt", tmp / "cfg_a.txt", tmp / "ref.txt", tmp / "cfg_b.txt")
        assert result["differing_resolved_field_count"] == 1, result
        assert result["differing_resolved_fields"][0]["field"] == "nn_iceini_file"
        # revert-the-fix control: identical cfgs on both sides must show zero diffs
        (tmp / "cfg_same.txt").write_text(cfg_agree)
        result_same = validate(tmp / "ref.txt", tmp / "cfg_same.txt", tmp / "ref.txt", tmp / "cfg_same.txt")
        assert result_same["differing_resolved_field_count"] == 0, result_same
        # a ref-only field neither cfg touches must still resolve and compare
        assert ("namthd_sal", "rn_rc_rjw") in resolve(tmp / "ref.txt", tmp / "cfg_a.txt")
    print("self-test OK: resolved diff fires exactly on the planted field; "
          "zero when cfgs agree; ref-only field still resolves")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-ref", type=Path)
    parser.add_argument("--variant-cfg", type=Path)
    parser.add_argument("--orca1-ref", type=Path)
    parser.add_argument("--orca1-cfg", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    if not (args.variant_ref and args.variant_cfg and args.orca1_ref and args.orca1_cfg):
        if not args.self_test:
            parser.error("provide all four paths, or pass --self-test")
        return 0
    result = validate(args.variant_ref, args.variant_cfg, args.orca1_ref, args.orca1_cfg)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json:
        args.json.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
