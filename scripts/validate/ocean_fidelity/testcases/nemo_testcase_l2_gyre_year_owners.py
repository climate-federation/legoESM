#!/usr/bin/env python3
"""WHO OWNS THE GYRE DAY-30 GAP -- the day-by-day walk, the decomposition,
the surface-forcing STATEMENT gate, and the threshold-switch trace.

Preregistration: ``docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_year_owners.md``.
Read it first.  Every constant, expectation and falsifier below is fixed there.

WHY THIS EXISTS.  The year receipt scored GYRE DISTINGUISHABLE at every day and
left the OWNER unnamed: the gap is `1.4241e-02` K at day 30 against a floor of
`6.8065e-10` K, and `2.768e-03` K of it is already present after TEN steps.
Nothing in the campaign resolves the interval between.

MODES
  --step-gap N       legoESM stepped N steps, scored against NEMO's per-step
                     ENTRY dumps.  The certified card's own writer covers
                     kstp = nit000..nit000+59, so days 0..10 are already on
                     disk at STEP resolution -- no new NEMO run is needed for
                     them.  (cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/MY_SRC/stprk3.F90:90)
  --decompose DAY    the gap at DAY by FIELD, by DEPTH and by REGION, plus the
                     fraction-of-NEMO's-own-from-rest-signal table that is the
                     only one of the three that can name a LEADING FIELD.
  --forcing-gate     legoESM's CURRENT surface forcing against the LITERAL
                     usrdef_sbc transcription, BIT-EXACT, evaluated on NEMO's
                     OWN state at every day boundary the record holds.  This is
                     the statement test the kt=1 dump cannot perform, because
                     kt=1 samples the seasonal clock at ONE of its 2160 values.
  --switch-trace     the enhanced-vertical-diffusion trigger mask, step by
                     step, on two members; reports the FIRST step at which the
                     two masks differ.
  --self-check       the arithmetic and every plant.

PLANTS (each exits NON-ZERO; each is exercised by the committed unit test)
  forcing-phase    evaluates legoESM's forcing one step late
  forcing-qsr-pi   swaps usrdef_sbc's literal 3.1415 for rpi in the literal arm
  forcing-nyear    restores the (nyear-1) subtraction as if the run were year 2
  day-offset       the day-by-day walk reads NEMO one day late
  switch-blind     freezes the EVD trigger mask so no crossing can be found
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

CASE = "GYRE-zco"
DT_S = 14400.0
STEPS_PER_DAY = 6
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners")
YEAR_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest")
YEAR_HEAD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest_head")
DEFAULT_MESH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10/mesh_mask.nc")
# The per-step ENTRY dumps.  The year run's own pristine member wrote 60 of
# them; the ten-step ladder's run wrote 10.  Both are the same instrument.
DEFAULT_ENTRY_ROOT = YEAR_ROOT / "nemo_pristine"
# usrdef_sbc.f90:140 -- the emp branch latitude, and :192-193 the wind's half
# period.  These are READ OFF THE SOURCE, not chosen.
EMP_SPLIT_LAT_DEG = 37.2
WIND_BAND_LAT_DEG = (15.0, 29.0)
FIELDS = ("T", "S", "u", "v", "ssh")
SBC_FIELDS = ("qsr", "qns", "emp", "utau", "vtau")


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _year():
    return _load("_gyre_year", "nemo_testcase_l2_gyre_year_fromrest.py")


def _gate():
    return _load("nemo_testcase_l2_gyre_phase3_gate",
                 "nemo_testcase_l2_gyre_phase3_gate.py")


def _round16():
    _load("nemo_testcase_l2_gyre_round15_eligibility",
          "nemo_testcase_l2_gyre_round15_eligibility.py")
    return _load("nemo_testcase_l2_gyre_round16_discriminator",
                 "nemo_testcase_l2_gyre_round16_discriminator.py")


def _policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    import jax
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")


def _rms(values, mask) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values[mask] ** 2)))


# ------------------------------------------------------- the per-step walk ---
def step_gap(steps: int, out: Path, *, entry_root: Path = DEFAULT_ENTRY_ROOT,
             plant: str | None = None) -> dict:
    """legoESM step by step against NEMO's per-step ENTRY dumps.

    NEMO's ``oracle_step_entry_kt{n}.bin`` holds ``ts/uu/vv/ssh(...,Nbb)`` at
    the ENTRY of step ``n``, i.e. the state after ``n-1`` completed steps.  So
    the dump at ``kt = n`` is compared against legoESM after ``n-1`` calls to
    ``model.step``.  That is the same convention the certified ladder uses.

    ponytail: this walks the card with the certified gate's OWN
    ``_surface_forcings`` + ``LatLonCGridOceanModel.step`` -- the same two
    calls ``run_member`` and ``--census`` make -- rather than adding a third
    copy of the physics.  What is new here is only the comparison cadence.
    """
    _policy()
    import jax.numpy as jnp  # noqa: F401  (policy must be set before jax use)
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    gate = _gate()
    card = build_nemo_testcase_card(CASE)
    masks = gate.expected_masks(card)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config)
    state = card.recipe.initial_state
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    rest = gate.read_entry(
        Path(entry_root) / "oracle_step_entry_kt00000001.bin")
    rows = []
    started = time.time()
    for completed in range(steps + 1):
        kt = completed + 1                      # the dump this state matches
        path = Path(entry_root) / f"oracle_step_entry_kt{kt:08d}.bin"
        if path.is_file():
            read_kt = kt + 1 if plant == "day-offset" else kt
            read = Path(entry_root) / f"oracle_step_entry_kt{read_kt:08d}.bin"
            require(read.is_file(),
                    f"plant day-offset: no dump at kt={read_kt}")
            oracle = gate.read_entry(read)
            fields = gate.lego_fields(state)
            row = {"kt": kt, "steps_completed": completed,
                   "day": completed / STEPS_PER_DAY,
                   "oracle": str(read), "oracle_sha256": gate.sha256(read)}
            for name in FIELDS:
                mask = masks[name]
                left = np.asarray(fields[name], dtype=np.float64)
                right = np.asarray(oracle[name], dtype=np.float64)
                require(left.shape == right.shape == mask.shape,
                        f"kt={kt} {name}: shapes {left.shape} {right.shape} "
                        f"{mask.shape}")
                row[f"rms_{name}"] = _rms(left - right, mask)
                row[f"max_{name}"] = float(np.max(np.abs((left - right)[mask])))
                row[f"unequal_{name}"] = int(np.count_nonzero(
                    left[mask].view(np.uint64) != right[mask].view(np.uint64)))
                # The dimensionless version: the gap over the signal NEMO
                # ITSELF has developed from rest by this step.  Five fields in
                # five units cannot be ranked any other way.
                signal = _rms(
                    right - np.asarray(rest[name], dtype=np.float64), mask)
                row[f"frac_{name}"] = (row[f"rms_{name}"] / signal
                                       if signal > 0.0 else float("nan"))
            rows.append(row)
        if completed == steps:
            break
        freshwater, surface = gate._surface_forcings(card, state, completed + 1)
        state = model.step(state, dt=card.dt_s, freshwater=freshwater,
                           surface_forcing=surface)
        require(all(bool(np.all(np.isfinite(np.asarray(value))))
                    for value in gate.lego_fields(state).values()),
                f"non-finite legoESM state after step {completed + 1}")

    report = {"format": "gyre-year-owners-step-gap-v1", "case": CASE,
              "steps": steps, "entry_root": str(entry_root), "plant": plant,
              "rows": rows, "wall_seconds": time.time() - started,
              "worktree": worktree_stamp()}
    (out / "step_gap.json").write_text(json.dumps(report, indent=2))
    print(f"{'kt':>5s}{'day':>7s}" + "".join(f"{f'rms {n}':>14s}" for n in FIELDS)
          + "".join(f"{f'frac {n}':>12s}" for n in FIELDS))
    for row in rows:
        print(f"{row['kt']:>5d}{row['day']:>7.3f}"
              + "".join(f"{row[f'rms_{n}']:>14.4e}" for n in FIELDS)
              + "".join(f"{row[f'frac_{n}']:>12.4e}" for n in FIELDS))
    return report
