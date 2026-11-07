#!/usr/bin/env python3
"""Substitute NEMO's recorded vertical-mixing operands, one arm at a time.

Round 78 left the question: NEMO's mixing-length floor is 1.0e-3 m, legoESM had
been running an effective 1.0 m, and correcting it made the ladder worse, so the
1 m floor was covering something.  This reads the round-79b record and scores
each arm of NEMO's vertical-physics chain against the diffusivity that retired
floor produced, plus the background profiles the internal-wave initialisation
resets, so the owner is named from measurement rather than from reading.

Everything here is computed from the admitted record and from the ORCA2 card's
own resolved configuration; no model step is run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_GATE_PATH = (Path(__file__).resolve().parent
              / "nemo_testcase_l4_orca2_round79b_vertical_mixing_gate.py")


def _load_record_gate():
    spec = importlib.util.spec_from_file_location(
        "nemo_testcase_l4_orca2_round79b_vertical_mixing_gate", _GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RECORD = _load_record_gate()


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


# The mixing length legoESM ran before the fold-in corrected it, in metres.
RETIRED_FLOOR_M = 1.0


def _stats(values: np.ndarray, wet: np.ndarray) -> dict:
    sel = values[wet]
    if sel.size == 0:
        return {"median": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
    return {
        "median": float(np.median(sel)),
        "mean": float(sel.mean()),
        "max": float(sel.max()),
        "n": int(sel.size),
    }


def arms_for_record(fields: dict[str, np.ndarray]) -> dict:
    """Each arm's contribution to the tracer diffusivity, on wet w-points."""
    wet = fields["wmask"] > 0.0
    ediff = float(fields["closure_scalars"][0, 0, 0])
    sqrt_en = np.sqrt(np.maximum(fields["en"], 0.0))
    # The retired floor raised the mixing length to 1 m wherever NEMO's own
    # length is below it; the closure's diffusivity is ediff * length * sqrt(e),
    # so the excess it produced is ediff * sqrt(e) * (1 m - length)+.
    floor_excess = ediff * sqrt_en * np.maximum(
        0.0, RETIRED_FLOOR_M - fields["mxlm"])
    return {
        "closure_avt": _stats(fields["avt_after_tke"], wet),
        "closure_avm": _stats(fields["avm_after_tke"], wet),
        "retired_1m_floor_excess_avt": _stats(floor_excess, wet),
        "floor_bound_fraction": float((fields["mxlm"][wet]
                                       < RETIRED_FLOOR_M).mean()),
        "river_mouth_avt": _stats(
            fields["avt_after_rnf"] - fields["avt_after_tke"], wet),
        "convection_avt": _stats(
            fields["avt_after_evd"] - fields["avt_after_rnf"], wet),
        "salt_heat_split_avm": _stats(
            fields["avm_after_ddm"] - fields["avm_after_evd"], wet),
        "salt_heat_split_avs_minus_avt": _stats(
            np.abs(fields["avs_after_ddm"] - fields["avt_after_ddm"]), wet),
        "internal_wave_avt": _stats(
            fields["avt_after_iwm"] - fields["avt_after_ddm"], wet),
        "internal_wave_avm": _stats(
            fields["avm_after_iwm"] - fields["avm_after_ddm"], wet),
        "final_avt": _stats(fields["avt_after_iwm"], wet),
        "background_avtb": float(fields["avtb"][0, 0, 1]),
        "background_avmb": float(fields["avmb"][0, 0, 1]),
        "background_avtb_2d_wet_values": sorted(
            float(v) for v in np.unique(fields["avtb_2d"][..., 0][
                fields["tmask"][..., 0] > 0.0]))[:4],
        "mixing_length_floor": float(fields["closure_scalars"][0, 0, 1]),
    }


def card_backgrounds(deck_root: Path) -> dict:
    """What the ORCA2 card's turbulence closure floors its coefficients at."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card

    card = build_orca2_zps_card(deck_root)
    tke = card.recipe.model_config.physics.vertical_mixing.tke
    iwm = card.recipe.model_config.physics.vertical_mixing.iwm
    ddm = card.recipe.model_config.physics.vertical_mixing.ddm
    return {
        "card_avtb": float(tke.kappaH_min),
        "card_avmb": float(tke.kappaM_min),
        "card_mixing_length_floor": float(tke.mxl_min),
        "card_iwm_enabled": bool(iwm.enabled),
        "card_iwm_tsdiff": bool(iwm.tsdiff),
        "card_iwm_mevar": bool(iwm.mevar),
        "card_ddm_enabled": bool(ddm.enabled),
        "card_unmeasured_features": list(card.unmeasured_features),
    }


def run_gate(root: Path, deck_root: Path, *, plant: str | None = None) -> dict:
    admitted = RECORD.run_gate(root)
    require(admitted["status"] == "PASS", "record is not admitted")
    per_step: dict[str, dict] = {}
    for rank in RECORD.RANKS:
        for step in range(1, RECORD.N_STEPS + 1):
            name = f"oracle_zdf_vmix_kt{step:08d}_r{rank:04d}.bin"
            record = RECORD.read_record(root / name)
            per_step[name] = arms_for_record(record["fields"])
    card = card_backgrounds(deck_root)
    if plant == "background":
        card["card_avtb"] = per_step[
            f"oracle_zdf_vmix_kt00000001_r{RECORD.RANKS[0]:04d}.bin"
        ]["background_avtb"]
    if plant == "arm":
        for row in per_step.values():
            row["internal_wave_avt"] = {"median": 0.0, "mean": 0.0,
                                        "max": 0.0, "n": 0}

    first = per_step[f"oracle_zdf_vmix_kt00000001_r{RECORD.RANKS[0]:04d}.bin"]
    # Findings the table must support, each refused if it stops being true.
    require(card["card_avtb"] != first["background_avtb"],
            "the card's tracer background already equals NEMO's, so the "
            "background reset is not a finding")
    require(card["card_avmb"] != first["background_avmb"],
            "the card's momentum background already equals NEMO's")
    require(first["internal_wave_avt"]["mean"] > 0.0,
            "the record carries no internal-wave increment at all")
    require(first["floor_bound_fraction"] > 0.5,
            "the retired 1 m floor did not bind on most wet points, so it "
            "cannot be what was compensating")
    return {
        "status": "PASS",
        "retired_floor_m": RETIRED_FLOOR_M,
        "card": card,
        "records": per_step,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plant", default=None,
                        choices=["background", "arm"])
    args = parser.parse_args(argv)
    try:
        result = run_gate(args.root, args.deck_root, plant=args.plant)
    except (GateError, RECORD.GateError) as error:
        print(f"REFUSE: {error}")
        print("STATUS PLANT-FIRED" if args.plant else "STATUS REFUSE")
        return 1
    if args.plant:
        print("STATUS PLANT-SURVIVED")
        return 0
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    first = result["records"]["oracle_zdf_vmix_kt00000001_r0000.bin"]
    print("STATUS PASS")
    print(f"  NEMO backgrounds      avtb={first['background_avtb']:.3e} "
          f"avmb={first['background_avmb']:.3e} "
          f"avtb_2d={first['background_avtb_2d_wet_values']}")
    print(f"  card  backgrounds     avtb={result['card']['card_avtb']:.3e} "
          f"avmb={result['card']['card_avmb']:.3e}")
    print(f"  retired 1 m floor     mean {first['retired_1m_floor_excess_avt']['mean']:.3e}"
          f"  median {first['retired_1m_floor_excess_avt']['median']:.3e}"
          f"  bound on {first['floor_bound_fraction']:.1%} of wet points")
    print(f"  internal-wave arm     mean {first['internal_wave_avt']['mean']:.3e}"
          f"  median {first['internal_wave_avt']['median']:.3e}")
    print(f"  river-mouth arm       mean {first['river_mouth_avt']['mean']:.3e}")
    print(f"  convection arm        mean {first['convection_avt']['mean']:.3e}")
    print(f"  salt/heat split (avm) mean {first['salt_heat_split_avm']['mean']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
