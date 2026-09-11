#!/usr/bin/env python3
"""Reconcile the GYRE ladder's configuration with the year harness's.

Two receipts on this branch contradict each other on the SAME card:

  * the kt=1..10 ladder (round-8 receipt) scores step 2 at ``7.6e-16`` (T) and
    ``9.6e-16`` (S), i.e. AT-BAR-NOT-EXACT;
  * the year-owners receipt runs step 2 from NEMO's own entry state and finds
    ``4.1543946279e-04`` K on 17999 of 18000 wet cells.

Eleven orders apart on one card cannot be one configuration.  This gate
resolves BOTH programs -- the object, never the deck -- and reports:

  ``--config-diff``  one row per model-config field on which the two harnesses
                     DISAGREE, read off the instantiated NamedTuples, plus an
                     AST reading of each harness's own model-construction call
                     so the table is attributed to a STATEMENT and not to a
                     transcription.
  ``--two-path``     the measurement.  kt=1..2 from rest on both programs,
                     bit-compared field by field at every step boundary; then
                     step 2 from NEMO's OWN entry state on both programs,
                     scored against NEMO's kt=3 entry.  The ``4.15e-04`` K must
                     appear on one program and not the other, or the config
                     table is not the explanation.

After unification the two programs are the same object and every row of
``--config-diff`` is empty; the gate then FAILS if they ever diverge again,
which is the whole point of committing it.

Plants (each must exit non-zero):
  ``--plant config-drift``   perturbs one resolved field of the year program
  ``--plant vacuous-reseed`` runs the equal-input arm without reseeding
  ``--plant same-program``   asserts the diff table is empty while it is not
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

CASE = "GYRE-zco"
FIELDS = ("T", "S", "u", "v", "ssh")
HERE = Path(__file__).resolve().parent
LADDER_GATE = HERE / "nemo_testcase_l2_gyre_phase3_gate.py"
YEAR_HARNESS = HERE / "nemo_testcase_l2_gyre_year_fromrest.py"
OWNERS_HARNESS = HERE / "nemo_testcase_l2_gyre_year_owners.py"
# The ladder's oracle root and the year's are DIFFERENT NEMO RUNS of the same
# namelist (they differ only in nn_itend/nn_stock/nn_write).  They are not
# bit-identical to each other past kt=1, so the entry root is an explicit
# argument at every call site and its identity is stamped in the report.
LADDER_ENTRY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10")
YEAR_ENTRY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/"
    "nemo_pristine")


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(name: str, path: Path):
    require(path.is_file(), f"missing {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _policy():
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")


# ------------------------------------------------------- the STATEMENT read --
def model_construction_kwargs(path: Path, function: str) -> dict:
    """Read the ``_replace`` a harness applies before it builds its model.

    Rule 10 applied to the SOURCE: the resolved-object diff below says the two
    programs differ; this says WHICH STATEMENT makes them differ, so the row
    can be attributed rather than described.  Returns ``{}`` when the harness
    hands the card's own config to ``LatLonCGridOceanModel`` untouched -- which
    is itself the finding, not a missing measurement.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            target = node
            break
    require(target is not None, f"{path.name}: no function {function}")
    # Every `<name> = card.recipe.model_config._replace(...)` in the function,
    # plus every inline `_replace` inside a LatLonCGridOceanModel(...) call.
    found: dict[str, object] = {}
    sites = 0
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == "_replace"
                and ast.unparse(func.value) == "card.recipe.model_config"):
            sites += 1
            for keyword in node.keywords:
                require(keyword.arg is not None,
                        f"{path.name}: **kwargs in a model-config _replace")
                try:
                    found[keyword.arg] = ast.literal_eval(keyword.value)
                except ValueError:
                    found[keyword.arg] = ast.unparse(keyword.value)
    require(sites <= 1,
            f"{path.name}:{function}: {sites} distinct model-config _replace "
            "sites; this reader reports one program per function")
    return found


def model_config_argument(path: Path, function: str) -> str:
    """The third positional argument of the harness's own model construction."""
    tree = ast.parse(path.read_text(), filename=str(path))
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            target = node
            break
    require(target is not None, f"{path.name}: no function {function}")
    for node in ast.walk(target):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "LatLonCGridOceanModel"):
            require(len(node.args) >= 3,
                    f"{path.name}:{function}: model built without a config arg")
            return ast.unparse(node.args[2])
    raise GateError(f"{path.name}:{function}: no LatLonCGridOceanModel call")


# ------------------------------------------------------- the RESOLVED object --
def flatten(config, prefix: str = "") -> dict:
    """Every leaf of a nested config NamedTuple, keyed by dotted path."""
    out: dict[str, object] = {}
    fields = getattr(config, "_fields", None)
    if fields is None:
        out[prefix.rstrip(".")] = config
        return out
    for name in fields:
        value = getattr(config, name)
        key = f"{prefix}{name}"
        if getattr(value, "_fields", None) is not None:
            out.update(flatten(value, prefix=f"{key}."))
        else:
            out[key] = value
    return out


def _render(value) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}:{hashlib.sha256(value.tobytes()).hexdigest()[:12]}"
    try:
        if isinstance(value, float):
            return repr(value)
        return repr(value)
    except Exception:                                        # noqa: BLE001
        return f"<{type(value).__name__}>"


def config_rows(ladder, year) -> list[dict]:
    left, right = flatten(ladder), flatten(year)
    require(set(left) == set(right),
            f"the two configs are not the same TYPE: "
            f"{sorted(set(left) ^ set(right))[:8]}")
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            same = np.array_equal(np.asarray(a), np.asarray(b))
        else:
            same = bool(a == b) and type(a) is type(b)
        if not same:
            rows.append({"field": key, "ladder": _render(a), "year": _render(b)})
    return rows


def _legacy_year_config(card):
    """The year harness's program as it stood before unification.

    Not a card and not a choice: the two values are read from the FIELD
    DEFAULTS of ``LatLonCGridOceanConfig`` and from the GYRE card as it was,
    so this arm reconstructs the historical year program for the controlled
    comparison and nothing else.  ``--two-path`` needs it to show that the
    ``4.15e-04`` K belongs to this program and not to the ladder's.
    """
    return card.recipe.model_config._replace(
        freshwater_closure="virtual_salt_flux", fix_eta_drift=False)


def resolve_programs(*, plant: str | None = None):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    card = build_nemo_testcase_card(CASE)
    ladder_kwargs = model_construction_kwargs(LADDER_GATE, "run")
    ladder = card.recipe.model_config._replace(**ladder_kwargs) if ladder_kwargs \
        else card.recipe.model_config
    year_kwargs = model_construction_kwargs(YEAR_HARNESS, "run_member")
    year = card.recipe.model_config._replace(**year_kwargs) if year_kwargs \
        else card.recipe.model_config
    if plant == "config-drift":
        year = year._replace(A_h=float(year.A_h) * 2.0)
    return card, ladder, year, ladder_kwargs, year_kwargs


def config_diff(*, plant: str | None = None) -> dict:
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    _policy()
    card, ladder, year, ladder_kwargs, year_kwargs = resolve_programs(plant=plant)
    rows = config_rows(ladder, year)
    statements = {
        "ladder": {
            "file": LADDER_GATE.name,
            "function": "run",
            "model_config_argument": model_config_argument(LADDER_GATE, "run"),
            "replace_kwargs": ladder_kwargs,
        },
        "year": {
            "file": YEAR_HARNESS.name,
            "function": "run_member",
            "model_config_argument": model_config_argument(
                YEAR_HARNESS, "run_member"),
            "replace_kwargs": year_kwargs,
        },
    }
    report = {
        "format": "gyre-card-reconciliation-config-diff-v1",
        "case": CASE,
        "plant": plant,
        "statements": statements,
        "differing_fields": rows,
        "n_differing": len(rows),
        "ladder_gate_sha256": sha256(LADDER_GATE),
        "year_harness_sha256": sha256(YEAR_HARNESS),
        "worktree": worktree_stamp(),
    }
    print("\nTHE SIDE-BY-SIDE TABLE: one row per model-config field on which "
          "the two harnesses' RESOLVED programs differ")
    print(f"  ladder  {statements['ladder']['file']}:run  "
          f"config = {statements['ladder']['model_config_argument']}  "
          f"_replace{ladder_kwargs or ' (none)'}")
    print(f"  year    {statements['year']['file']}:run_member  "
          f"config = {statements['year']['model_config_argument']}  "
          f"_replace{year_kwargs or ' (none)'}")
    if not rows:
        print("  (empty -- the two programs are the SAME RESOLVED OBJECT)")
    for row in rows:
        print(f"  {row['field']:<52s} ladder={row['ladder']:<24s} "
              f"year={row['year']}")
    if plant == "same-program":
        require(not rows, "plant: the diff table is not empty")
    return report


# -------------------------------------------------------------- the MEASUREMENT
def _model(card, cfg):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)

    return LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)


def _bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def two_path(*, steps: int = 2, entry_root: Path = LADDER_ENTRY_ROOT,
             plant: str | None = None) -> dict:
    """kt=1..``steps`` from rest on BOTH programs, bit-compared per boundary,
    then step 2 from NEMO's OWN entry state on both, scored against NEMO's
    kt=3 entry.
    """
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    _policy()
    gate = _load("gyre_phase3_gate", LADDER_GATE)
    baro = _load("nemo_testcase_overflow_barotropic_gate",
                 HERE / "nemo_testcase_overflow_barotropic_gate.py")
    card, ladder, year, _, _ = resolve_programs()
    if not config_rows(ladder, year):
        # After unification the year harness resolves the certified card, so
        # the historical program is reconstructed explicitly as the CONTROL.
        # Without it this mode would compare a program against itself and
        # could never fail -- the anti-pattern the plants exist to prevent.
        year = _legacy_year_config(card)
        legacy_control = True
    else:
        legacy_control = False
    masks = gate.expected_masks(card)
    # THE TWO ROWS ARE ONE SELECTION, NOT TWO FREE FIELDS.  The model REFUSES
    # freshwater_closure="real_freshwater" with fix_eta_drift=False
    # (ocean_model_latlon_cgrid.py:3006-3018), so only three of the four
    # combinations are constructible and the fourth arm cannot be built.  The
    # third program below is the one that IS constructible and it isolates
    # fix_eta_drift inside the year's own closure; ladder-vs-isolate is then
    # the freshwater closure alone.
    isolate = year._replace(fix_eta_drift=not bool(year.fix_eta_drift))
    programs = {"ladder": _model(card, ladder), "year": _model(card, year),
                "isolate_fix_eta_drift": _model(card, isolate)}

    # A. from rest, both programs, bit-compared at every boundary.
    states = {name: card.recipe.initial_state for name in programs}
    boundaries = []
    for kt in range(1, steps + 1):
        for name, model in programs.items():
            freshwater, surface = gate._surface_forcings(card, states[name], kt)
            states[name] = model.step(states[name], dt=card.dt_s,
                                      freshwater=freshwater,
                                      surface_forcing=surface)
        left = gate.lego_fields(states["ladder"])
        right = gate.lego_fields(states["year"])
        row = {"entering_kt": kt + 1, "fields": {}}
        for name in FIELDS:
            mask = np.asarray(masks[name], dtype=bool)
            a = np.asarray(left[name], dtype=np.float64)
            b = np.asarray(right[name], dtype=np.float64)
            row["fields"][name] = {
                "max_abs": float(np.max(np.abs((a - b)[mask]))),
                "cells_unequal": int(np.count_nonzero(
                    _bits(a)[mask] != _bits(b)[mask])),
                "cells": int(mask.sum()),
            }
        boundaries.append(row)

    # B. step 2 from NEMO's OWN entry state, both programs, against kt=3.
    entry = {n: gate.read_entry(
        Path(entry_root) / f"oracle_step_entry_kt{n:08d}.bin") for n in (2, 3)}
    equal_input = {}
    for name, model in programs.items():
        start = card.recipe.initial_state
        freshwater, surface = gate._surface_forcings(card, start, 1)
        start = model.step(start, dt=card.dt_s, freshwater=freshwater,
                           surface_forcing=surface)
        seeded = baro.state_from_oracle_entry(start, entry[2], masks)
        if plant == "vacuous-reseed":
            seeded = start
        moved = max(float(np.max(np.abs(
            np.asarray(gate.lego_fields(seeded)[f], dtype=np.float64)
            - np.asarray(gate.lego_fields(start)[f], dtype=np.float64))))
            for f in FIELDS)
        require(moved > 0.0,
                f"{name}: the reseed changed NOTHING, so the equal-input arm "
                "is the free run under another name and measures nothing")
        freshwater, surface = gate._surface_forcings(card, seeded, 2)
        out = gate.lego_fields(model.step(seeded, dt=card.dt_s,
                                          freshwater=freshwater,
                                          surface_forcing=surface))
        row = {"reseed_moved_input_by": moved, "fields": {}}
        for field in FIELDS:
            mask = np.asarray(masks[field], dtype=bool)
            a = np.asarray(out[field], dtype=np.float64)
            b = np.asarray(entry[3][field], dtype=np.float64)
            if b.ndim == 3:
                b = b[..., :a.shape[-1]]
            diff = (a - b)[mask]
            row["fields"][field] = {
                "rms": float(np.sqrt(np.mean(diff ** 2))),
                "max_abs": float(np.max(np.abs(diff))),
                "cells_unequal": int(np.count_nonzero(
                    _bits(a)[mask] != _bits(b)[mask])),
                "cells": int(mask.sum()),
            }
        equal_input[name] = row

    report = {
        "format": "gyre-card-reconciliation-two-path-v1",
        "case": CASE,
        "plant": plant,
        "steps": steps,
        "entry_root": str(entry_root),
        "legacy_control": legacy_control,
        "from_rest_boundaries": boundaries,
        "equal_input_kt2": equal_input,
        "ladder_gate_sha256": sha256(LADDER_GATE),
        "worktree": worktree_stamp(),
    }
    print(f"\nTWO PATHS FROM REST (entry root {entry_root.name}"
          + (", year arm = reconstructed legacy control" if legacy_control
             else "") + ")")
    for row in boundaries:
        cells = "  ".join(
            f"{f} {row['fields'][f]['max_abs']:.4e}"
            f"/{row['fields'][f]['cells_unequal']}"
            for f in FIELDS)
        print(f"  entering kt={row['entering_kt']}  max|ladder-year| / cells "
              f"unequal:  {cells}")
    print("\nSTEP 2 FROM NEMO'S OWN ENTRY STATE, scored against NEMO's kt=3 "
          "entry")
    print(f"  {'program':>10s}" + "".join(f"{'rms ' + f:>16s}" for f in FIELDS))
    for name, row in equal_input.items():
        print(f"  {name:>10s}"
              + "".join(f"{row['fields'][f]['rms']:>16.6e}" for f in FIELDS))
    return report


def oracle_floor(*, steps: int = 3, roots=(LADDER_ENTRY_ROOT, YEAR_ENTRY_ROOT)) -> dict:
    """How far apart are the ORACLE'S OWN two records of this card?

    The ladder scores against ``gyre_kt1_10`` and the year against
    ``year_fromrest/nemo_pristine``.  Their namelists differ only in run length
    and output cadence, so they are the same configuration -- but they are not
    the same executable, and NEMO is not bit-reproducible across builds.  A
    residual smaller than THIS is a residual the campaign cannot attribute to
    legoESM at all, so the number belongs next to every ladder row and is
    measured here in THE LADDER'S OWN UNITS (``score``'s
    ``normalized_max_abs``: max abs difference over max(max|oracle|, 1)).
    """
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    _policy()
    gate = _load("gyre_phase3_gate", LADDER_GATE)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    card = build_nemo_testcase_card(CASE)
    masks = gate.expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    left_root, right_root = (Path(r) for r in roots)
    rows = []
    for kt in range(1, steps + 1):
        a = gate.read_entry(left_root / f"oracle_step_entry_kt{kt:08d}.bin")
        b = gate.read_entry(right_root / f"oracle_step_entry_kt{kt:08d}.bin")
        for field in FIELDS:
            mask = np.asarray(masks[field], dtype=bool)
            x = np.asarray(a[field], dtype=np.float64)
            y = np.asarray(b[field], dtype=np.float64)
            if x.ndim == 3:
                x, y = x[..., :nlev], y[..., :nlev]
            absolute = float(np.max(np.abs((x - y)[mask])))
            reference = float(np.max(np.abs(x[mask])))
            rows.append({
                "kt": kt, "field": field,
                "absolute_max": absolute,
                "reference_max_abs": reference,
                "normalized_max_abs": absolute / max(reference, 1.0),
                "cells_unequal": int(np.count_nonzero(
                    _bits(x)[mask] != _bits(y)[mask])),
                "cells": int(mask.sum()),
                "bar": 1.0e-15,
            })
    report = {
        "format": "gyre-card-reconciliation-oracle-floor-v1",
        "case": CASE, "roots": [str(left_root), str(right_root)],
        "rows": rows, "worktree": worktree_stamp(),
    }
    print(f"\nTHE ORACLE'S OWN FLOOR: {left_root.name} against "
          f"{right_root.name}, in the ladder's units")
    print(f"  {'kt':>3s} {'field':>6s} {'abs':>14s} {'normalized':>14s} "
          f"{'cells':>14s}  vs bar 1e-15")
    for row in rows:
        verdict = ("ABOVE THE BAR" if row["normalized_max_abs"] > row["bar"]
                   else "under")
        print(f"  {row['kt']:>3d} {row['field']:>6s} "
              f"{row['absolute_max']:>14.6e} "
              f"{row['normalized_max_abs']:>14.6e} "
              f"{row['cells_unequal']:>7d}/{row['cells']:<6d}  {verdict}")
    require(any(r["absolute_max"] > 0.0 for r in rows),
            "the two oracle records are bit-identical at every scored row; "
            "this mode measured nothing")
    return report


def score_both_roots(*, steps: int = 3) -> dict:
    """Score the LADDER'S OWN program against BOTH oracle records.

    The ladder reports one residual per row against ``gyre_kt1_10``.  If the
    oracle's two records of this card differ by as much as that residual, the
    row is not a measurement of legoESM -- so every row is taken twice here,
    once against each record, and the oracle-vs-oracle floor is printed beside
    it.  A row whose two residuals differ by about the floor is a row whose
    number belongs to the ORACLE'S build, not to the model.
    """
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    _policy()
    gate = _load("gyre_phase3_gate", LADDER_GATE)
    card, ladder, _, _, _ = resolve_programs()
    model = _model(card, ladder)
    masks = gate.expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    roots = {"ladder_root": LADDER_ENTRY_ROOT, "year_root": YEAR_ENTRY_ROOT}
    state = card.recipe.initial_state
    rows = []
    for kt in range(1, steps + 1):
        fields = gate.lego_fields(state)
        entries = {name: gate.read_entry(
            root / f"oracle_step_entry_kt{kt:08d}.bin")
            for name, root in roots.items()}
        for field in FIELDS:
            mask = np.asarray(masks[field], dtype=bool)
            mine = np.asarray(fields[field], dtype=np.float64)
            row = {"kt": kt, "field": field, "bar": 1.0e-15}
            for name, entry in entries.items():
                theirs = np.asarray(entry[field], dtype=np.float64)
                if theirs.ndim == 3:
                    theirs = theirs[..., :nlev]
                absolute = float(np.max(np.abs((mine - theirs)[mask])))
                reference = float(np.max(np.abs(theirs[mask])))
                row[name] = {
                    "absolute_max": absolute,
                    "normalized_max_abs": absolute / max(reference, 1.0),
                    "cells_unequal": int(np.count_nonzero(
                        _bits(mine)[mask] != _bits(theirs)[mask])),
                    "exact": bool(np.array_equal(mine[mask], theirs[mask])),
                }
            rows.append(row)
        if kt < steps:
            freshwater, surface = gate._surface_forcings(card, state, kt)
            state = model.step(state, dt=card.dt_s, freshwater=freshwater,
                               surface_forcing=surface)
    report = {"format": "gyre-card-reconciliation-both-roots-v1", "case": CASE,
              "roots": {k: str(v) for k, v in roots.items()}, "rows": rows,
              "worktree": worktree_stamp()}
    print("\nTHE LADDER'S PROGRAM SCORED AGAINST BOTH ORACLE RECORDS")
    print(f"  {'kt':>3s} {'field':>6s} {'vs ladder root':>16s} {'exact':>6s}"
          f" {'vs year root':>16s} {'exact':>6s}")
    for row in rows:
        print(f"  {row['kt']:>3d} {row['field']:>6s} "
              f"{row['ladder_root']['normalized_max_abs']:>16.6e} "
              f"{str(row['ladder_root']['exact']):>6s} "
              f"{row['year_root']['normalized_max_abs']:>16.6e} "
              f"{str(row['year_root']['exact']):>6s}")
    return report


def self_check() -> int:
    """Every plant must be refused, and the readers must be non-vacuous."""
    failures = []

    def expect_raises(label, fn):
        try:
            fn()
        except GateError:
            print(f"  PLANT REFUSED  {label}")
            return
        failures.append(label)
        print(f"  PLANT ACCEPTED {label}  <-- the gate cannot see it")

    print("SELF-CHECK")
    # The AST readers must actually find the two statements.
    ladder_arg = model_config_argument(LADDER_GATE, "run")
    year_arg = model_config_argument(YEAR_HARNESS, "run_member")
    print(f"  ladder model config argument: {ladder_arg}")
    print(f"  year   model config argument: {year_arg}")
    if "model_config" not in ladder_arg and "cfg" not in ladder_arg:
        failures.append("ladder model-config argument reader")
    expect_raises("config-drift", lambda: config_diff(plant="config-drift")
                  and require(not config_diff(plant="config-drift")[
                      "differing_fields"], "config-drift left no row"))
    expect_raises("same-program-while-different", _plant_same_program)
    expect_raises("missing-function", lambda: model_construction_kwargs(
        LADDER_GATE, "no_such_function"))
    for name in failures:
        print(f"  FAILED: {name}")
    return 1 if failures else 0


def _plant_same_program():
    """Assert an empty table while a field is deliberately perturbed."""
    _policy()
    card, ladder, _, _, _ = resolve_programs()
    drifted = ladder._replace(A_h=float(ladder.A_h) * 2.0)
    rows = config_rows(ladder, drifted)
    require(not rows, "plant: the diff table is not empty")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-diff", action="store_true")
    parser.add_argument("--two-path", action="store_true")
    parser.add_argument("--oracle-floor", action="store_true")
    parser.add_argument("--score-both-roots", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--entry-root", type=Path, default=LADDER_ENTRY_ROOT)
    parser.add_argument("--plant", choices=("config-drift", "vacuous-reseed",
                                            "same-program"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-unified", action="store_true",
                        help="exit non-zero unless the two programs resolve to "
                             "the same object (the unification gate)")
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    require(args.config_diff or args.two_path or args.oracle_floor
            or args.score_both_roots,
            "choose --config-diff, --two-path, --oracle-floor, "
            "--score-both-roots or --self-check")
    report = {}
    if args.config_diff:
        report["config_diff"] = config_diff(plant=args.plant)
    if args.oracle_floor:
        report["oracle_floor"] = oracle_floor()
    if args.score_both_roots:
        report["score_both_roots"] = score_both_roots(steps=args.steps)
    if args.two_path:
        report["two_path"] = two_path(steps=args.steps,
                                      entry_root=args.entry_root,
                                      plant=args.plant)
    if args.require_unified:
        rows = report.get("config_diff", {}).get("differing_fields")
        require(rows is not None,
                "--require-unified needs --config-diff")
        require(not rows,
                f"the ladder and the year resolve DIFFERENT programs: {rows}")
        print("\nUNIFIED: the ladder and the year resolve the same object.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                          default=str) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
