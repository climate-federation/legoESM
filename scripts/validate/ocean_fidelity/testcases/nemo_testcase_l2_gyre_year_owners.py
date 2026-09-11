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
                # The oracle's binary record carries NEMO's jpk = 31 levels;
                # the card and its masks carry the 30 the model integrates.
                # Crop the ORACLE, never pad the candidate.
                right = np.asarray(oracle[name], dtype=np.float64)
                if right.ndim == 3:
                    right = right[..., :left.shape[-1]]
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
                base = np.asarray(rest[name], dtype=np.float64)
                if base.ndim == 3:
                    base = base[..., :left.shape[-1]]
                signal = _rms(right - base, mask)
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


# ------------------------------------------------------- the decomposition ---
def _regions(lat, wet2):
    """The REGION cut.  Two of the three boundaries are read off usrdef_sbc."""
    columns = np.flatnonzero(np.any(wet2, axis=0))
    require(columns.size > 0, "no wet column")
    lo, hi = int(columns.min()), int(columns.max())
    third = (hi - lo + 1) / 3.0
    index = np.arange(wet2.shape[1])[None, :] * np.ones_like(wet2, dtype=int)
    west = wet2 & (index < lo + third)
    east = wet2 & (index >= lo + 2 * third)
    middle = wet2 & ~west & ~east
    return {
        "west_third": west, "interior_third": middle, "east_third": east,
        # usrdef_sbc.f90:140 -- the emp branch splits at 37.2 N.
        f"emp_south_le_{EMP_SPLIT_LAT_DEG}N": wet2 & (lat <= EMP_SPLIT_LAT_DEG),
        f"emp_north_gt_{EMP_SPLIT_LAT_DEG}N": wet2 & (lat > EMP_SPLIT_LAT_DEG),
        # usrdef_sbc.f90:192-193 -- the wind's half period runs 15 N to 29 N.
        "wind_band_15_29N": wet2 & (lat >= WIND_BAND_LAT_DEG[0])
                                 & (lat <= WIND_BAND_LAT_DEG[1]),
    }


def decompose(day: int, *, lego_root: Path = YEAR_HEAD,
              nemo_root: Path = YEAR_ROOT, seed: int = 0,
              mesh_path: Path = DEFAULT_MESH) -> dict:
    """The gap at ``day`` by FIELD, by DEPTH and by REGION.

    Each cut's blind spot is in the preregistration's section 3 and is
    reported next to its table rather than left to the reader.
    """
    _policy()
    year = _year()
    gate = _gate()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, dz, dy, area, bands = year._geometry(card, mesh_path)
    gate_masks = gate.expected_masks(card)
    lat = np.asarray(mesh["gphit"], dtype=np.float64)
    depth = np.asarray(card.recipe.z_coord.nemo_gdept_0,
                       dtype=np.float64)[..., :wet3.shape[-1]]
    nlev = wet3.shape[-1]

    lego = year._load_lego(lego_root, seed, day)
    nemo = year._load_nemo(nemo_root, seed, day, nlev)
    # The REST state, so every field's gap can be expressed as a fraction of
    # the signal NEMO itself has developed.  Read from the card, which the
    # year harness's A1 leg proves is bit-identical to NEMO's step-1 BEFORE
    # level on every field.
    rest = gate.lego_fields(card.recipe.initial_state)

    masks = {"T": wet3, "S": wet3, "ssh": wet2,
             "u": gate_masks["u"], "v": gate_masks["v"]}
    field_rows = {}
    for name in FIELDS:
        mask = masks[name]
        left = np.asarray(lego[name], dtype=np.float64)
        right = np.asarray(nemo[name], dtype=np.float64)
        base = np.asarray(rest[name], dtype=np.float64)
        require(left.shape == right.shape == mask.shape == base.shape,
                f"{name}: shapes {left.shape} {right.shape} {mask.shape}")
        signal = _rms(right - base, mask)
        field_rows[name] = {
            "gap_rms": _rms(left - right, mask),
            "nemo_from_rest_rms": signal,
            "fraction_of_nemo_signal": (_rms(left - right, mask) / signal
                                        if signal > 0.0 else float("nan")),
            "nemo_std": float(np.std(right[mask])),
            "wet_cells": int(mask.sum()),
        }
    leading = max(
        (name for name in FIELDS
         if np.isfinite(field_rows[name]["fraction_of_nemo_signal"])),
        key=lambda name: field_rows[name]["fraction_of_nemo_signal"])

    difference = np.asarray(lego["T"], dtype=np.float64) - np.asarray(
        nemo["T"], dtype=np.float64)
    total = float(np.sum(difference[wet3] ** 2))
    depth_rows = {}
    for name, mask in list(bands.items()) + [("top_cell", wet3 & (
            np.arange(nlev)[None, None, :] == 0))]:
        depth_rows[name] = {
            "gap_rms_K": _rms(difference, mask),
            "share_of_sum_dT2": float(np.sum(difference[mask] ** 2)) / total,
            "cells": int(mask.sum()),
            "depth_range_m": [float(depth[mask].min()),
                              float(depth[mask].max())],
        }
    region_rows = {}
    for name, mask2 in _regions(lat, wet2).items():
        mask = wet3 & mask2[..., None]
        region_rows[name] = {
            "gap_rms_K": _rms(difference, mask),
            "share_of_sum_dT2": float(np.sum(difference[mask] ** 2)) / total,
            "cells": int(mask.sum()),
            "share_of_cells": float(mask.sum()) / float(wet3.sum()),
        }
    peak = np.unravel_index(int(np.argmax(np.abs(np.where(wet3, difference,
                                                          0.0)))),
                            difference.shape)
    report = {
        "format": "gyre-year-owners-decompose-v1", "case": CASE, "day": day,
        "seed": seed, "lego_root": str(lego_root), "nemo_root": str(nemo_root),
        "nemo_restart": nemo["path"], "nemo_sha256": nemo["sha256"],
        "mesh_sha256": mesh["mesh_sha256"],
        "fields": field_rows, "leading_field": leading,
        "depth": depth_rows, "region": region_rows,
        "peak": {"j": int(peak[0]), "i": int(peak[1]), "k": int(peak[2]),
                 "dT_K": float(difference[peak]),
                 "depth_m": float(depth[peak]),
                 "lat_deg": float(lat[peak[0], peak[1]])},
        "worktree": worktree_stamp(),
    }
    print(f"\nDAY {day}: FIELD -- gap, NEMO's own from-rest signal, and the ratio")
    print(f"  {'field':>6s}{'gap rms':>14s}{'NEMO signal':>14s}"
          f"{'gap/signal':>14s}{'gap/std':>12s}{'cells':>8s}")
    for name in FIELDS:
        row = field_rows[name]
        ratio = (row["gap_rms"] / row["nemo_std"] if row["nemo_std"] > 0
                 else float("nan"))
        print(f"  {name:>6s}{row['gap_rms']:>14.4e}"
              f"{row['nemo_from_rest_rms']:>14.4e}"
              f"{row['fraction_of_nemo_signal']:>14.4e}{ratio:>12.4e}"
              f"{row['wet_cells']:>8d}")
    print(f"  LEADING FIELD (dimensionless): {leading}")
    print(f"\nDAY {day}: DEPTH -- temperature only")
    print(f"  {'band':>12s}{'gap rms [K]':>14s}{'share dT^2':>12s}{'cells':>8s}")
    for name, row in depth_rows.items():
        print(f"  {name:>12s}{row['gap_rms_K']:>14.4e}"
              f"{row['share_of_sum_dT2']:>12.4f}{row['cells']:>8d}")
    print(f"\nDAY {day}: REGION -- temperature only")
    print(f"  {'region':>24s}{'gap rms [K]':>14s}{'share dT^2':>12s}"
          f"{'share cells':>13s}")
    for name, row in region_rows.items():
        print(f"  {name:>24s}{row['gap_rms_K']:>14.4e}"
              f"{row['share_of_sum_dT2']:>12.4f}{row['share_of_cells']:>13.4f}")
    print(f"\n  peak |dT| {report['peak']['dT_K']:+.4e} K at "
          f"j={report['peak']['j']} i={report['peak']['i']} "
          f"k={report['peak']['k']} ({report['peak']['depth_m']:.1f} m, "
          f"{report['peak']['lat_deg']:.2f} N)")
    return report


# ------------------------------------------- the surface-forcing STATEMENT ---
def _nyear_from_restart(path: Path) -> tuple[int, int, float, int]:
    """``ndastp``/``kt``/``adatrj`` read off NEMO's OWN restart.

    ``nyear`` is an OPERAND of the forcing (``usrdef_sbc.f90:108``) whose value
    cannot be read off the source.  Rule 0 says read it; this reads it from the
    run rather than assuming the run is in year 1.
    """
    import netCDF4

    with netCDF4.Dataset(path) as handle:
        ndastp = int(np.asarray(handle.variables["ndastp"][...]))
        kt = int(np.asarray(handle.variables["kt"][...]))
        adatrj = float(np.asarray(handle.variables["adatrj"][...]))
    return ndastp // 10000, kt, adatrj, ndastp


def forcing_gate(*, nemo_root: Path = YEAR_ROOT, seed: int = 0,
                 mesh_path: Path = DEFAULT_MESH,
                 entry_root: Path = DEFAULT_ENTRY_ROOT,
                 days: tuple[int, ...] = tuple(range(30, 361, 30)),
                 plant: str | None = None) -> dict:
    """legoESM's CURRENT forcing vs the LITERAL usrdef_sbc, on NEMO'S OWN state.

    WHAT THIS ANSWERS.  The kt=1 oracle dump certifies all five forcing fields
    BIT-EXACT -- at ONE value of a clock with 2160 distinct values in the year.
    Every seasonal statement in ``usrdef_sbc`` is a function of that scalar, so
    a wrong phase, denominator or calendar operand is invisible there and
    finite on day 30.  Evaluating BOTH transcriptions on the SAME state
    separates the STATEMENT from the state: any difference here is a
    transcription defect, because the inputs are identical by construction.

    WHAT IT CANNOT SEE, stated rather than discovered later: a statement
    misread the SAME way by both arms.  The mitigation is that the literal arm
    is certified against NEMO's own kt=1 dump (round 8 / round 16) and that the
    receipt re-reads the compiled routine line by line.
    """
    _policy()
    import jax.numpy as jnp
    year = _year()
    gate = _gate()
    r16 = _round16()
    from legoesm.ocean.eos import (
        nemo_potential_temperature_from_conservative)
    from legoesm.ocean.fidelity.nemo_recipe import nemo_gyre_qns
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card, gyre_surface_boundary_condition)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, _dz, _dy, _area, _bands = year._geometry(card, mesh_path)
    lat = np.asarray(mesh["gphit"], dtype=np.float64)
    nlev = wet3.shape[-1]

    def current(surface_ct, surface_sa, kt):
        """legoESM's production forcing path, on a supplied surface state.

        This is the certified gate's own composition
        (``nemo_testcase_l2_gyre_phase3_gate._surface_forcings``), with the
        state passed in instead of read off a live model, so that the LITERAL
        and the CURRENT arm see identical inputs.
        """
        t_seconds = kt * card.dt_s
        sbc = gyre_surface_boundary_condition(card, t_seconds)
        ct = jnp.asarray(surface_ct, dtype=jnp.float64)
        sa = jnp.asarray(surface_sa, dtype=jnp.float64)
        pt = nemo_potential_temperature_from_conservative(ct, sa)
        qns = nemo_gyre_qns(ct, pt, sbc.t_star_c, sbc.qsr_w_m2,
                            sbc.emp_kg_m2_s)
        return {"qsr": np.asarray(sbc.qsr_w_m2, dtype=np.float64),
                "qns": np.asarray(qns, dtype=np.float64),
                "emp": np.asarray(sbc.emp_kg_m2_s, dtype=np.float64),
                "utau": np.asarray(sbc.utau_pa, dtype=np.float64),
                "vtau": np.asarray(sbc.vtau_pa, dtype=np.float64)}

    def literal(surface_ct, surface_sa, kt, nyear):
        ct = np.asarray(surface_ct, dtype=np.float64)
        pt = np.asarray(nemo_potential_temperature_from_conservative(
            jnp.asarray(ct), jnp.asarray(surface_sa)), dtype=np.float64)
        fields, _sites = r16._literal_sbc(
            lat, wet2, ct, pt, kt=kt, nyear=nyear,
            qsr_pi=(3.141592653589793 if plant == "forcing-qsr-pi" else None))
        return fields

    rows = []
    # kt = 1, the ONE point the oracle dump certifies, carried here as the
    # gate's own anchor so a regression in either arm shows up next to the
    # day rows rather than in another artifact.
    anchor = gate.read_entry(
        Path(entry_root) / "oracle_step_entry_kt00000001.bin")
    cases = [("kt1", 1, np.asarray(anchor["T"])[..., 0],
              np.asarray(anchor["S"])[..., 0], 1, str(entry_root))]
    for day in days:
        step = day * STEPS_PER_DAY
        restart = year._load_nemo(nemo_root, seed, day, nlev)
        nyear, recorded_kt, adatrj, ndastp = _nyear_from_restart(
            Path(restart["path"]))
        require(recorded_kt == step,
                f"day {day}: restart kt={recorded_kt}, expected {step}")
        # The forcing that CONSUMES this state is step kt = n+1's, and its
        # nyear is the one `day(n+1)` sets.  Within a year the two agree; the
        # gate asserts that rather than assuming it, and records ndastp.
        # The forcing that CONSUMES the day-d restart is step d*6+1's.  At
        # d = 360 that step lies PAST nn_itend and the run never takes it;
        # the row is kept as a THIRTEENTH SAMPLE OF THE SEASONAL CLOCK (and
        # is labelled so), not as a step of the scored trajectory.  Its nyear
        # is the day-360 restart's, which is why the label says "clock".
        rows_kt = step + 1
        label = "clock" if step >= 2160 else "entering"
        cases.append((f"day{day:03d}_{label}_kt{rows_kt}", rows_kt,
                      np.asarray(restart["T"])[..., 0],
                      np.asarray(restart["S"])[..., 0], nyear,
                      restart["path"]))

    for name, kt, ct, sa, nyear, source in cases:
        # The plants perturb ONE arm.  legoESM's path has no nyear operand at
        # all -- that is the point of the nyear plant -- so the nyear plant
        # has to enter through the LITERAL arm, and the phase plant through
        # the current one.  A first version computed use_nyear and then passed
        # nyear, so the nyear plant was a dead variable and proved nothing.
        use_kt = kt + 1 if plant == "forcing-phase" else kt
        use_nyear = 2 if plant == "forcing-nyear" else nyear
        left = current(ct, sa, use_kt)
        right = literal(ct, sa, kt, use_nyear)
        row = {"case": name, "kt": kt, "nyear": nyear, "source": source,
               "ztime_hours": 4.0 * kt - (nyear - 1) * 24.0 * 360.0}
        for field in SBC_FIELDS:
            a = np.asarray(left[field], dtype=np.float64)
            b = np.asarray(right[field], dtype=np.float64)
            require(a.shape == b.shape == wet2.shape,
                    f"{name} {field}: shapes {a.shape} {b.shape}")
            unequal = int(np.count_nonzero(
                a[wet2].view(np.uint64) != b[wet2].view(np.uint64)))
            row[field] = {
                "wet_cells_unequal": unequal,
                "max_abs": float(np.max(np.abs((a - b)[wet2]))),
                "exact": unequal == 0,
                "field_max_abs": float(np.max(np.abs(b[wet2]))),
            }
        row["exact"] = all(row[f]["exact"] for f in SBC_FIELDS)
        rows.append(row)

    exact = all(row["exact"] for row in rows)
    report = {"format": "gyre-year-owners-forcing-gate-v1", "case": CASE,
              "plant": plant, "seed": seed, "nemo_root": str(nemo_root),
              "mesh_sha256": mesh["mesh_sha256"], "rows": rows,
              "all_bit_exact": exact, "worktree": worktree_stamp()}
    print(f"\nSURFACE-FORCING STATEMENT GATE -- legoESM's CURRENT path vs the "
          f"LITERAL usrdef_sbc, on NEMO'S OWN state.  BIT-EXACT bar.")
    print(f"  {'case':>26s}{'kt':>6s}{'ztime h':>10s}"
          + "".join(f"{f:>12s}" for f in SBC_FIELDS) + "   verdict")
    for row in rows:
        print(f"  {row['case']:>26s}{row['kt']:>6d}{row['ztime_hours']:>10.1f}"
              + "".join(f"{row[f]['wet_cells_unequal']:>12d}"
                        for f in SBC_FIELDS)
              + ("   BIT-EXACT" if row["exact"] else "   DEBT"))
    print(f"  ALL BIT-EXACT: {exact}")
    if not exact:
        worst = max((row[f]["max_abs"], row["case"], f)
                    for row in rows for f in SBC_FIELDS)
        print(f"  worst |difference| {worst[0]:.6e} in {worst[2]} at "
              f"{worst[1]}")
    return report


# ------------------------------------------------------- the day-by-day gap --
def day_gap(*, lego_root: Path = DEFAULT_ROOT, lego_tag: str = "daily",
            nemo_root: Path | None = None, seed: int = 0,
            mesh_path: Path = DEFAULT_MESH, days: tuple[int, ...] = (),
            plant: str | None = None) -> dict:
    """The DAY-BY-DAY gap: one row per day, one metric, both sides identical.

    ``nemo_root`` must hold a restart at every requested day.  Until the
    early-days acquisition runs, the only days NEMO has are the year record's
    multiples of 30, so this refuses loudly rather than silently scoring a
    shorter list than it was asked for (the year round's own lesson).
    """
    _policy()
    year = _year()
    gate = _gate()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, _dz, _dy, _area, bands = year._geometry(card, mesh_path)
    gate_masks = gate.expected_masks(card)
    masks = {"T": wet3, "S": wet3, "ssh": wet2,
             "u": gate_masks["u"], "v": gate_masks["v"]}
    rest = gate.lego_fields(card.recipe.initial_state)
    nlev = wet3.shape[-1]
    rows = []
    for day in days:
        read_day = day + 1 if plant == "day-offset" else day
        lego = year._load_lego(Path(lego_root), seed, day) if not lego_tag else (
            year._load_lego(Path(lego_root), f"{seed}_{lego_tag}", day))
        nemo = year._load_nemo(Path(nemo_root), seed, read_day, nlev)
        row = {"day": day, "nemo_day_read": read_day,
               "nemo_restart": nemo["path"], "nemo_sha256": nemo["sha256"]}
        for name in FIELDS:
            mask = masks[name]
            difference = (np.asarray(lego[name], dtype=np.float64)
                          - np.asarray(nemo[name], dtype=np.float64))
            signal = _rms(np.asarray(nemo[name], dtype=np.float64)
                          - np.asarray(rest[name], dtype=np.float64), mask)
            row[f"rms_{name}"] = _rms(difference, mask)
            row[f"frac_{name}"] = (row[f"rms_{name}"] / signal
                                   if signal > 0.0 else float("nan"))
        for name, mask in bands.items():
            row[f"rms_T3D_{name}"] = _rms(
                np.asarray(lego["T"], dtype=np.float64)
                - np.asarray(nemo["T"], dtype=np.float64), mask)
        rows.append(row)
    ratios = [rows[i]["rms_T"] / rows[i - 1]["rms_T"]
              for i in range(1, len(rows)) if rows[i - 1]["rms_T"] > 0]
    report = {"format": "gyre-year-owners-day-gap-v1", "case": CASE,
              "seed": seed, "plant": plant, "rows": rows,
              "max_day_to_day_ratio": max(ratios) if ratios else None,
              "worktree": worktree_stamp()}
    print(f"\nDAY-BY-DAY GAP -- legoESM vs NEMO, wet rms, fp64")
    print(f"  {'day':>5s}" + "".join(f"{f'rms {n}':>14s}" for n in FIELDS)
          + f"{'ratio':>9s}   leads")
    previous = None
    for row in rows:
        ratio = (row["rms_T"] / previous if previous else float("nan"))
        previous = row["rms_T"]
        finite = [n for n in FIELDS if np.isfinite(row[f"frac_{n}"])]
        leads = max(finite, key=lambda n: row[f"frac_{n}"]) if finite else "-"
        print(f"  {row['day']:>5d}"
              + "".join(f"{row[f'rms_{n}']:>14.4e}" for n in FIELDS)
              + f"{ratio:>9.3f}   {leads}")
    if report["max_day_to_day_ratio"] is not None:
        print(f"  max day-to-day T ratio {report['max_day_to_day_ratio']:.3f}")
    return report


# ------------------------------------------------------- the switch trace ---
def switch_trace(steps: int, out: Path, *, seeds: tuple[int, int] = (0, 1),
                 mesh_path: Path = DEFAULT_MESH, every: int = 1,
                 start: int = 1, plant: str | None = None) -> dict:
    """The ENHANCED-VERTICAL-DIFFUSION trigger mask, step by step, two members.

    The year receipt measures one member moving five orders of magnitude
    between day 180 and day 210 while three others did not, and names NEMO's
    ``ln_zdfevd`` hard branch (``IF( MIN(rn2,rn2b) <= -1.e-12 ) p_avt =
    rn_evd``) as the PLAUSIBLE mechanism with no cell-level trace run.  This
    is that trace.

    The mask is taken from the MODEL'S OWN convective coefficient field, not
    from a re-derivation of its trigger: ``_enhanced_diffusion_K`` is the
    function the step calls, and a cell has fired iff its tracer diffusivity
    reaches ``K_conv``.  ponytail: importing the model's own private helper in
    a probe beats re-deriving the trigger and then arguing they agree.
    """
    _policy()
    import jax.numpy as jnp
    year = _year()
    gate = _gate()
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        _enhanced_diffusion_K)

    card = build_nemo_testcase_card(CASE)
    conv = card.recipe.model_config.physics.convection
    k_conv = float(conv.enhanced_diffusion.K_conv)
    require(k_conv > 0.0, "the card's EVD K_conv is zero; nothing can fire")
    mesh = year.nemo_operands(mesh_path)
    active = (np.asarray(card.recipe.z_coord.is_active)
              & (np.asarray(card.recipe.land_mask) > 0.5)[..., None])
    depth = np.asarray(card.recipe.z_coord.nemo_gdept_0, dtype=np.float64)
    lat = np.asarray(mesh["gphit"], dtype=np.float64)

    def fired(state):
        # NEMO's trigger is MIN(rn2, rn2b) (zdfevd.f90:108).  On THIS card
        # stprk3.F90:173-174 sets `rn2 = rn2b` and then calls
        # `zdf_phy(kstp, Nbb, Nbb, Nrhs)`, so both arms ARE the whole-step
        # entry tracer and the MIN is over two identical fields.  The before
        # arm is passed explicitly anyway, so the two-level branch the model
        # runs is the branch measured here rather than an argued equivalent.
        K, _A = _enhanced_diffusion_K(
            state, card.recipe.z_coord, conv,
            before_tracers=(state.T.data, state.S.data),
            cc=card.recipe.model_config.physics.constants)
        mask = np.asarray(K, dtype=np.float64) >= k_conv
        if plant == "switch-blind":
            mask = np.zeros_like(mask)
        return mask

    models, states = {}, {}
    for seed in seeds:
        pert = year.nemo_istate_perturbation(
            np.broadcast_to(depth, depth.shape), np.broadcast_to(
                np.asarray(card.recipe.grid.native_lat_T_deg,
                           dtype=np.float64)[..., None], depth.shape),
            active.astype(np.float64), seed)
        state = card.recipe.initial_state
        state = state._replace(T=state.T.replace(
            data=jnp.asarray(np.asarray(state.T.data) + pert,
                             dtype=jnp.float64)))
        states[seed] = state
        models[seed] = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord,
            card.recipe.model_config)

    left, right = seeds
    rows = []
    first_difference = None
    started = time.time()
    for completed in range(steps):
        kt = completed + 1
        if kt >= start and (kt - start) % every == 0:
            masks = {seed: fired(states[seed]) for seed in seeds}
            differing = np.asarray(masks[left] != masks[right])
            count = int(np.count_nonzero(differing))
            row = {"kt": kt, "day": (kt - 1) / STEPS_PER_DAY,
                   "fired_left": int(masks[left].sum()),
                   "fired_right": int(masks[right].sum()),
                   "cells_differing": count,
                   "T_rms": _rms(np.asarray(states[left].T.data)
                                 - np.asarray(states[right].T.data), active)}
            if count and first_difference is None:
                j, i, k = (int(x) for x in np.argwhere(differing)[0])
                first_difference = {
                    "kt": kt, "day": (kt - 1) / STEPS_PER_DAY,
                    "j": j, "i": i, "k": k,
                    "depth_m": float(depth[j, i, k]),
                    "lat_deg": float(lat[j, i]),
                    "cells_differing": count,
                    "T_rms_at_crossing": row["T_rms"],
                    "fired_left": row["fired_left"],
                    "fired_right": row["fired_right"]}
                row["first_difference"] = first_difference
            rows.append(row)
            print(f"  kt {kt:>5d} day {row['day']:>7.2f}  fired "
                  f"{row['fired_left']:>5d}/{row['fired_right']:>5d}  "
                  f"differing {count:>4d}  T rms {row['T_rms']:.4e}",
                  flush=True)
        for seed in seeds:
            freshwater, surface = gate._surface_forcings(
                card, states[seed], kt)
            states[seed] = models[seed].step(
                states[seed], dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface)
    require(any(row["fired_left"] or row["fired_right"] for row in rows),
            "the EVD trigger NEVER fired on either member over the traced "
            "window; the instrument sees nothing and no 'no crossing' "
            "statement may be made from it")
    report = {"format": "gyre-year-owners-switch-trace-v1", "case": CASE,
              "seeds": list(seeds), "steps": steps, "every": every,
              "start": start, "plant": plant, "K_conv": k_conv,
              "n2_threshold": float(conv.enhanced_diffusion.n2_threshold),
              "evd_n2_time_level": conv.enhanced_diffusion.evd_n2_time_level,
              "rows": rows, "first_difference": first_difference,
              "wall_seconds": time.time() - started,
              "worktree": worktree_stamp()}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"switch_trace_seeds{left}{right}.json").write_text(
        json.dumps(report, indent=2))
    print(f"  FIRST DIFFERING TRIGGER MASK: {first_difference}")
    return report


# ------------------------------------------------------------- self-check ---
def self_check() -> int:
    """The arithmetic, and every plant SHOWN to fail."""
    failures = []

    def expect_raises(label, fn):
        try:
            fn()
        except BaseException as error:               # noqa: BLE001
            print(f"  PLANT {label}: raised {type(error).__name__} -- OK")
            return
        failures.append(label)
        print(f"  PLANT {label}: DID NOT FAIL")

    # 1. the region cut partitions the wet surface exactly once
    lat = np.tile(np.linspace(10.0, 50.0, 6)[:, None], (1, 6))
    wet2 = np.zeros((6, 6), dtype=bool)
    wet2[1:-1, 1:-1] = True
    regions = _regions(lat, wet2)
    thirds = (regions["west_third"].astype(int)
              + regions["interior_third"].astype(int)
              + regions["east_third"].astype(int))
    if not np.array_equal(thirds, wet2.astype(int)):
        failures.append("thirds do not partition the wet surface")
    else:
        print("  thirds partition the wet surface exactly once -- OK")
    emp = (regions[f"emp_south_le_{EMP_SPLIT_LAT_DEG}N"].astype(int)
           + regions[f"emp_north_gt_{EMP_SPLIT_LAT_DEG}N"].astype(int))
    if not np.array_equal(emp, wet2.astype(int)):
        failures.append("the emp split does not partition the wet surface")
    else:
        print("  the emp split partitions the wet surface exactly once -- OK")
    for name in ("west_third", "east_third", "wind_band_15_29N"):
        if not regions[name].any():
            failures.append(f"region {name} is empty -- the cut is vacuous")
    # 2. the literal SBC's kt default is byte-unchanged by the new argument
    r16 = _round16()
    small_lat = np.array([[20.0, 40.0]])
    small_wet = np.array([[True, True]])
    ct = np.array([[20.0, 10.0]])
    pt = np.array([[19.9, 9.9]])
    base, _ = r16._literal_sbc(small_lat, small_wet, ct, pt)
    again, _ = r16._literal_sbc(small_lat, small_wet, ct, pt, kt=1, nyear=1)
    for field in SBC_FIELDS:
        if not np.array_equal(np.asarray(base[field]).view(np.uint64),
                              np.asarray(again[field]).view(np.uint64)):
            failures.append(f"_literal_sbc default changed {field}")
    else:
        print("  _literal_sbc's kt=1/nyear=1 default is byte-unchanged -- OK")
    # 3. the literal SBC MOVES with kt, so a phase plant CAN be detected
    later, _ = r16._literal_sbc(small_lat, small_wet, ct, pt, kt=181)
    moved = [f for f in SBC_FIELDS
             if not np.array_equal(np.asarray(base[f]).view(np.uint64),
                                   np.asarray(later[f]).view(np.uint64))]
    if set(moved) != set(SBC_FIELDS):
        failures.append(f"only {moved} move between kt=1 and kt=181; a phase "
                        "plant could not be seen on the others")
    else:
        print("  every SBC field moves between kt=1 and kt=181 -- the phase "
              "plant is non-vacuous -- OK")
    # 4. the nyear term is a NO-OP inside year 1 and NOT a no-op at year 2
    y2, _ = r16._literal_sbc(small_lat, small_wet, ct, pt, kt=181, nyear=2)
    if np.array_equal(np.asarray(later["qsr"]).view(np.uint64),
                      np.asarray(y2["qsr"]).view(np.uint64)):
        failures.append("the nyear plant changes nothing; it is vacuous")
    else:
        print("  the nyear=2 plant moves qsr -- non-vacuous -- OK")
    # 5. the qsr_pi plant moves qsr (usrdef_sbc's literal is 3.1415, not rpi)
    swapped, _ = r16._literal_sbc(small_lat, small_wet, ct, pt,
                                  qsr_pi=3.141592653589793)
    if np.array_equal(np.asarray(base["qsr"]).view(np.uint64),
                      np.asarray(swapped["qsr"]).view(np.uint64)):
        failures.append("3.1415 -> rpi changes nothing; the plant is vacuous")
    else:
        print(f"  3.1415 -> rpi moves qsr by "
              f"{float(np.max(np.abs(base['qsr'] - swapped['qsr']))):.4e} "
              "W/m2 -- non-vacuous -- OK")
    # 6. the registry now reads the sixty dumps the card writes, and only those
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    for kt in (1, 10, 11, 60):
        if time_level_for_dump(f"oracle_step_entry_kt{kt:08d}.bin") != "before":
            failures.append(f"entry dump kt={kt} is not registered 'before'")
    expect_raises("registry-beyond-60",
                  lambda: time_level_for_dump(
                      "oracle_step_entry_kt00000061.bin"))
    if failures:
        for item in failures:
            print(f"  FAILED: {item}")
        return 1
    print("  self-check: all checks passed")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-gap", type=int, default=None,
                        help="walk N legoESM steps against NEMO's per-step "
                             "entry dumps")
    parser.add_argument("--decompose", type=int, default=None,
                        help="decompose the gap at this day")
    parser.add_argument("--forcing-gate", action="store_true")
    parser.add_argument("--day-gap", action="store_true")
    parser.add_argument("--switch-trace", type=int, default=None,
                        help="trace the EVD trigger mask over N steps")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--lego-root", type=Path, default=None)
    parser.add_argument("--lego-tag", default="daily")
    parser.add_argument("--nemo-root", type=Path, default=YEAR_ROOT)
    parser.add_argument("--entry-root", type=Path, default=DEFAULT_ENTRY_ROOT)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--days", default=None,
                        help="comma-separated day list")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plant", default=None)
    args = parser.parse_args(argv)

    report = None
    if args.self_check:
        return self_check()
    if args.step_gap is not None:
        report = step_gap(args.step_gap, args.root,
                          entry_root=args.entry_root, plant=args.plant)
    elif args.decompose is not None:
        report = decompose(args.decompose,
                           lego_root=(args.lego_root or YEAR_HEAD),
                           nemo_root=args.nemo_root, seed=args.seed,
                           mesh_path=args.mesh)
    elif args.forcing_gate:
        days = (tuple(int(x) for x in args.days.split(","))
                if args.days else tuple(range(30, 361, 30)))
        report = forcing_gate(nemo_root=args.nemo_root, seed=args.seed,
                              mesh_path=args.mesh,
                              entry_root=args.entry_root, days=days,
                              plant=args.plant)
        if not report["all_bit_exact"]:
            print("FORCING STATEMENT GATE: DEBT")
            if args.json:
                Path(args.json).write_text(json.dumps(report, indent=2))
            return 1
    elif args.day_gap:
        require(args.days is not None, "--day-gap needs --days")
        report = day_gap(lego_root=(args.lego_root or args.root),
                         lego_tag=args.lego_tag, nemo_root=args.nemo_root,
                         seed=args.seed, mesh_path=args.mesh,
                         days=tuple(int(x) for x in args.days.split(",")),
                         plant=args.plant)
    elif args.switch_trace is not None:
        left, right = (int(x) for x in args.seeds.split(","))
        report = switch_trace(args.switch_trace, args.root,
                              seeds=(left, right), mesh_path=args.mesh,
                              every=args.every, start=args.start,
                              plant=args.plant)
    else:
        parser.error("choose a mode")
    if args.json and report is not None:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
