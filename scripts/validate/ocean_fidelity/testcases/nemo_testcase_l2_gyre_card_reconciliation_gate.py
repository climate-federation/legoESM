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
# TWO ORACLE RECORDS OF ONE CARD, and the campaign already knows why: v1 is
# built without `-fno-tree-vectorize`, so gfortran vectorises transcendental
# calls into glibc's libmvec, whose low bits differ from scalar libm; v2 is
# built with the flag (arch/arch-conda-scalarmath.fcm).  Same CPP keys, same
# physics, different low bits from kt=2 on.  The certified receipts pass
# `--oracle-root .../round19_oracle_v2_external`, which is BIT-IDENTICAL to
# the year's own root; the phase-3 gate module's OWN default is still v1.  The
# root is therefore an explicit argument at every call site here and its
# identity is stamped in every report -- this gate does not choose one.
ORACLE_V1_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10")
ORACLE_V2_ROOT = Path(
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


# ------------------------------------------------ the program, as EXECUTED --
# An adversarial review KILLED the first version of this reader.  It matched a
# `_replace` whose receiver unparsed to exactly `card.recipe.model_config`, so
# rebinding through a local --
#     cfg = card.recipe.model_config
#     cfg = cfg._replace(freshwater_closure="virtual_salt_flux", ...)
# -- reintroduced the exact defect this gate exists to catch and the gate
# reported "the SAME RESOLVED OBJECT", exit 0.  Reading source text for one
# statement shape and calling the result "the program" was the bug.  So the
# harness's OWN construction is EXECUTED and the object it hands to
# LatLonCGridOceanModel is CAPTURED (Rule 10: instantiate and print, never
# trust a declaration).  No spelling of the assignment can hide from this,
# because every spelling ends at the constructor.
_NO_MODEL_SOURCE: Path | None = None


class _Captured(Exception):
    """Raised by the spy the instant a model is constructed."""


def capture_model_config(module_name: str, path: Path, call, *,
                         card_factory=None):
    """Run ``call(module)`` and return the config it hands to the model.

    ``card_factory`` replaces ``build_nemo_testcase_card`` for the duration,
    which is how the ``program-drift`` plant makes a harness hand over a
    DIFFERENT program without touching the construction statement at all.
    """
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as impl
    import legoesm.ocean.fidelity.nemo_testcase_recipe as recipe

    module = _load(module_name, path)
    real_model = impl.LatLonCGridOceanModel
    real_card = recipe.build_nemo_testcase_card
    captured: list = []

    def spy(grid, z_coord, cfg, *args, **kwargs):
        captured.append(cfg)
        raise _Captured

    impl.LatLonCGridOceanModel = spy
    if card_factory is not None:
        recipe.build_nemo_testcase_card = card_factory
    try:
        call(module)
    except _Captured:
        pass
    finally:
        impl.LatLonCGridOceanModel = real_model
        recipe.build_nemo_testcase_card = real_card
    require(bool(captured),
            f"{path.name}: the call constructed no ocean model, so this gate "
            "captured no program and would have compared nothing")
    return captured[0]


def model_config_argument(path: Path, function: str) -> str:
    """The source text at the construction site.

    DISPLAY ONLY.  It names the first ``LatLonCGridOceanModel(`` call in the
    function so the table can point at a line; nothing is ASSERTED from it,
    because the review above proved that source text is not the program.
    """
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


def _legacy_card_factory(real_factory):
    """A card whose config is the PRE-UNIFICATION one, for the plant."""

    def factory(case):
        card = real_factory(case)
        recipe = card.recipe._replace(
            model_config=_legacy_year_config(card))
        return card._replace(recipe=recipe)

    return factory


def resolve_programs(*, plant: str | None = None):
    import tempfile

    import legoesm.ocean.fidelity.nemo_testcase_recipe as recipe

    card = recipe.build_nemo_testcase_card(CASE)
    ladder = capture_model_config(
        "gyre_phase3_gate", LADDER_GATE,
        lambda m: m.run(m.ROOT, trajectory_only=True, max_step=2))
    with tempfile.TemporaryDirectory() as scratch:
        year = capture_model_config(
            "gyre_year_fromrest", YEAR_HARNESS,
            lambda m: m.run_member(0, scratch, days=1),
            card_factory=(_legacy_card_factory(recipe.build_nemo_testcase_card)
                          if plant == "program-drift" else None))
    owners = capture_model_config(
        "gyre_year_owners", OWNERS_HARNESS,
        lambda m: m.equal_input_step(2))
    if plant == "config-drift":
        # A NESTED field, so the walk is exercised rather than the top level.
        year = year._replace(lateral_viscosity=year.lateral_viscosity._replace(
            A_h=float(year.lateral_viscosity.A_h) * 2.0))
    return card, ladder, year, owners


def initial_state_rows(card, *, plant: str | None = None) -> list[dict]:
    """The other half of the call-chain diff: does the year START where the
    ladder starts?

    The ladder steps ``card.recipe.initial_state`` directly.  The year harness
    round-trips it through numpy and ADDS a perturbation field documented to be
    exactly zero for the control member -- documented, which is not the same as
    measured.  Every prognostic field is compared BIT for BIT.
    """
    import jax.numpy as jnp

    year = _load("gyre_year_fromrest", YEAR_HARNESS)
    depth3 = np.asarray(card.recipe.z_coord.nemo_gdept_0, dtype=np.float64)
    lat3 = np.broadcast_to(
        np.asarray(card.recipe.grid.native_lat_T_deg,
                   dtype=np.float64)[..., None], depth3.shape)
    active = (np.asarray(card.recipe.z_coord.is_active)
              & (np.asarray(card.recipe.land_mask) > 0.5)[..., None])
    pert = year.nemo_istate_perturbation(depth3, lat3,
                                         active.astype(np.float64), 0)
    if plant == "state-drift":
        pert = pert + 1.0e-18
    start = card.recipe.initial_state
    seeded = start._replace(T=start.T.replace(
        data=jnp.asarray(np.asarray(start.T.data) + pert, dtype=jnp.float64)))
    rows = []
    for name in ("T", "S", "u", "v", "eta"):
        a = np.asarray(getattr(start, name).data, dtype=np.float64)
        b = np.asarray(getattr(seeded, name).data, dtype=np.float64)
        rows.append({
            "field": name,
            "max_abs": float(np.max(np.abs(a - b))),
            "cells_unequal": int(np.count_nonzero(_bits(a) != _bits(b))),
            "cells": int(a.size),
        })
    return rows


def config_diff(*, plant: str | None = None) -> dict:
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    _policy()
    card, ladder, year, owners = resolve_programs(plant=plant)
    rows = config_rows(ladder, year)
    owner_rows = config_rows(ladder, owners)
    statements = {
        "ladder": {"file": LADDER_GATE.name, "function": "run",
                   "construction_site_source": model_config_argument(
                       LADDER_GATE, "run")},
        "year": {"file": YEAR_HARNESS.name, "function": "run_member",
                 "construction_site_source": model_config_argument(
                     YEAR_HARNESS, "run_member")},
        "year_owners": {"file": OWNERS_HARNESS.name,
                        "function": "equal_input_step",
                        "construction_site_source": model_config_argument(
                            OWNERS_HARNESS, "equal_input_step")},
    }
    state_rows = initial_state_rows(card, plant=plant)
    report = {
        "format": "gyre-card-reconciliation-config-diff-v3",
        "case": CASE,
        "plant": plant,
        "statements": statements,
        "differing_fields": rows,
        "n_differing": len(rows),
        "differing_fields_year_owners": owner_rows,
        "initial_state_rows": state_rows,
        "ladder_gate_sha256": sha256(LADDER_GATE),
        "year_harness_sha256": sha256(YEAR_HARNESS),
        "year_owners_sha256": sha256(OWNERS_HARNESS),
        "worktree": worktree_stamp(),
    }
    print("\nTHE SIDE-BY-SIDE TABLE: one row per model-config field on which "
          "the harnesses' EXECUTED programs differ")
    print("  (each program is the object its own harness hands to "
          "LatLonCGridOceanModel, captured by running it -- not read off the "
          "source)")
    for name, row in statements.items():
        print(f"  {name:<12s} {row['file']}:{row['function']}  "
              f"construction site reads `{row['construction_site_source']}`")
    if not rows and not owner_rows:
        print("  (empty -- all three programs are the SAME RESOLVED OBJECT)")
    for row in rows:
        print(f"  ladder vs year        {row['field']:<40s} "
              f"ladder={row['ladder']:<24s} year={row['year']}")
    for row in owner_rows:
        print(f"  ladder vs year-owners {row['field']:<40s} "
              f"ladder={row['ladder']:<24s} year={row['year']}")
    print("  the year's own INITIAL STATE against the card's, bit for bit:  "
          + "  ".join(f"{r['field']} {r['cells_unequal']}/{r['cells']}"
                      for r in state_rows))
    require(all(row["cells_unequal"] == 0 for row in state_rows),
            "the year harness does not START from the card's own state: "
            f"{[r for r in state_rows if r['cells_unequal']]}")
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


def two_path(*, steps: int = 2, entry_root: Path = ORACLE_V1_ROOT,
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
    card, ladder, year, _owners = resolve_programs()
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


def oracle_floor(*, steps: int = 3, roots=(ORACLE_V1_ROOT, ORACLE_V2_ROOT)) -> dict:
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
    card, ladder, _year, _owners = resolve_programs()
    model = _model(card, ladder)
    masks = gate.expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    roots = {"oracle_v1": ORACLE_V1_ROOT, "oracle_v2": ORACLE_V2_ROOT}
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
    print(f"  {'kt':>3s} {'field':>6s} {'vs oracle v1':>16s} {'exact':>6s}"
          f" {'vs oracle v2':>16s} {'exact':>6s}")
    for row in rows:
        print(f"  {row['kt']:>3d} {row['field']:>6s} "
              f"{row['oracle_v1']['normalized_max_abs']:>16.6e} "
              f"{str(row['oracle_v1']['exact']):>6s} "
              f"{row['oracle_v2']['normalized_max_abs']:>16.6e} "
              f"{str(row['oracle_v2']['exact']):>6s}")
    return report


def closure_ablation(*, steps: int = 2) -> dict:
    """IS THE FRESHWATER CHANNEL ALIVE ON THIS LANE AT ALL?

    Both independent reviews of this round raised the same arithmetic: GYRE's
    ``emp`` reaches ``+-3.7e-05`` kg/m2/s, so over one ``14400`` s step in a
    ``10`` m top cell a VIRTUAL SALT FLUX should move surface salinity by about
    ``1.8e-03`` g/kg and a REAL FRESHWATER source should move ``ssh`` by about
    ``5.2e-04`` m.  The measured difference between the two closures after two
    steps is ``1.4e-14`` g/kg and ``4.3e-19`` m.  Eleven and fifteen orders
    below.  That is not a small effect, it is an ABSENT one -- and a knob that
    selects between two absent channels is cosmetic, so the card change this
    round lands would be hygiene rather than physics.

    So it is measured instead of argued: hold ``fix_eta_drift`` fixed and step
    the card under all three closures, bit-comparing the trajectory.  If the
    three are bit-identical the channel is inert on this lane -- which would
    mean NEMO's ``emp`` reaches the state by another statement (the card's own
    NEMO-literal ssh/wzv update, ``sshwzv.f90:137``) and the closure selector
    never fires.  The PREDICTED sizes are printed next to the measured ones so
    the reader can see the ratio rather than take a verdict.
    """
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    _policy()
    gate = _load("gyre_phase3_gate", LADDER_GATE)
    card, ladder, _year, _owners = resolve_programs()
    masks = gate.expected_masks(card)
    arms = {}
    for closure in ("real_freshwater", "virtual_salt_flux", "none"):
        cfg = ladder._replace(freshwater_closure=closure, fix_eta_drift=True)
        model = _model(card, cfg)
        state = card.recipe.initial_state
        for kt in range(1, steps + 1):
            freshwater, surface = gate._surface_forcings(card, state, kt)
            state = model.step(state, dt=card.dt_s, freshwater=freshwater,
                               surface_forcing=surface)
        arms[closure] = gate.lego_fields(state)
    # The forcing this is all about, printed rather than assumed.
    freshwater, _ = gate._surface_forcings(card, card.recipe.initial_state, 1)
    evap = np.asarray(getattr(freshwater, "evap", freshwater), dtype=np.float64)
    emp_max = float(np.max(np.abs(evap)))
    dz0 = float(np.asarray(card.recipe.z_coord.dz_ref)[0])
    rho0 = float(ladder.rho_0)
    predicted = {
        "emp_max_kg_m2_s": emp_max,
        "virtual_salt_step_g_kg": emp_max * 35.0 * card.dt_s / (rho0 * dz0),
        "real_freshwater_step_m": emp_max * card.dt_s / rho0,
    }
    rows = []
    reference = "real_freshwater"
    for closure in ("virtual_salt_flux", "none"):
        for field in FIELDS:
            mask = np.asarray(masks[field], dtype=bool)
            a = np.asarray(arms[reference][field], dtype=np.float64)
            b = np.asarray(arms[closure][field], dtype=np.float64)
            rows.append({
                "against": closure, "field": field,
                "max_abs": float(np.max(np.abs((a - b)[mask]))),
                "cells_unequal": int(np.count_nonzero(
                    _bits(a)[mask] != _bits(b)[mask])),
                "cells": int(mask.sum()),
            })
    report = {"format": "gyre-card-reconciliation-closure-ablation-v1",
              "case": CASE, "steps": steps, "predicted_if_live": predicted,
              "rows": rows, "worktree": worktree_stamp()}
    print(f"\nIS THE FRESHWATER CHANNEL ALIVE?  {steps} steps, fix_eta_drift "
          f"held True, real_freshwater as the reference arm")
    print(f"  GYRE's own max |emp| this step: "
          f"{predicted['emp_max_kg_m2_s']:.4e} kg/m2/s")
    print(f"  IF the virtual-salt channel were live, S would move "
          f"{predicted['virtual_salt_step_g_kg']:.4e} g/kg per step")
    print(f"  IF the real-freshwater channel were live, ssh would move "
          f"{predicted['real_freshwater_step_m']:.4e} m per step")
    for row in rows:
        print(f"  real_freshwater vs {row['against']:<18s} {row['field']:>4s}  "
              f"max abs {row['max_abs']:.6e}   cells "
              f"{row['cells_unequal']}/{row['cells']}")
    return report


# Which NEMO BUILD wrote each record, and whether that build's transcendentals
# were vectorised.  The campaign's own arch comment says libmvec's low bits
# differ from scalar libm's; this checks it against the binaries rather than
# repeating the sentence.
ORACLE_BUILDS = {
    "vectorized_v1": Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
                          "cfgs/GYRE_OMIP_L2_P3/BLD/bin/nemo.exe"),
    "scalarmath_R41ADVSP": Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
        "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/bin/nemo.exe"),
    "scalarmath_YRPERT": Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
        "GYRE_OMIP_L2_P3_SM_YRPERT/BLD/bin/nemo.exe"),
}


def oracle_provenance(*, roots=None) -> dict:
    """WHICH BUILD wrote each record, and was its math vectorised?

    Two NEMO records of one card cannot both be the oracle, and the answer is
    not in a receipt -- it is in the binaries.  For each build: the count of
    libmvec (``_ZGV*``) call sites in its executable, which is 0 exactly when
    ``-fno-tree-vectorize`` was applied.  For each record: the executable
    beside it, and whether that matches the ``binary.sha256`` stamped next to
    it -- a stamp that names a different build than the file beside it is a
    provenance defect, and one of these has that.
    """
    import subprocess

    from legoesm.ocean.fidelity.provenance import worktree_stamp

    roots = dict(roots or {
        "oracle_v1 (ladder gate default)": ORACLE_V1_ROOT,
        "oracle_v2 (the year, and the certified receipts)": ORACLE_V2_ROOT,
    })
    builds = {}
    for name, exe in ORACLE_BUILDS.items():
        row = {"path": str(exe), "exists": exe.is_file()}
        if row["exists"]:
            row["sha256"] = sha256(exe)
            dump = subprocess.run(["objdump", "-d", str(exe)],
                                  capture_output=True, text=True)
            row["libmvec_call_sites"] = sum(
                1 for line in dump.stdout.splitlines() if "_ZGV" in line)
            row["vectorized_transcendentals"] = row["libmvec_call_sites"] > 0
        builds[name] = row
    by_sha = {row.get("sha256"): name for name, row in builds.items()
              if row.get("sha256")}
    records = {}
    for name, root in roots.items():
        root = Path(root)
        beside = root / "nemo"
        stamp = root / "binary.sha256"
        row = {"root": str(root)}
        row["executable_sha256"] = sha256(beside) if beside.is_file() else None
        row["build"] = by_sha.get(row["executable_sha256"])
        if stamp.is_file():
            first = stamp.read_text().split()
            row["stamped_sha256"] = first[0] if first else None
            row["stamped_path"] = first[1] if len(first) > 1 else None
            row["stamp_agrees_with_file_beside_it"] = (
                row["executable_sha256"] is None
                or row["stamped_sha256"] == row["executable_sha256"])
        records[name] = row
    report = {"format": "gyre-card-reconciliation-oracle-provenance-v1",
              "builds": builds, "records": records,
              "worktree": worktree_stamp()}
    print("\nWHICH BUILD WROTE WHICH RECORD")
    for name, row in builds.items():
        if not row["exists"]:
            print(f"  {name:>22s}  MISSING {row['path']}")
            continue
        print(f"  {name:>22s}  sha {row['sha256'][:16]}  libmvec call sites "
              f"{row['libmvec_call_sites']:>5d}  vectorised "
              f"{row['vectorized_transcendentals']}")
    for name, row in records.items():
        exe = (row["executable_sha256"] or "none")[:16]
        print(f"  {name}: executable beside it {exe} -> build {row['build']}")
        if "stamped_sha256" in row and not row["stamp_agrees_with_file_beside_it"]:
            print(f"      STAMP DISAGREES: binary.sha256 names "
                  f"{row['stamped_sha256'][:16]} at {row['stamped_path']}")
    require(any(row.get("libmvec_call_sites") for row in builds.values()),
            "no build shows a libmvec call site; the objdump reader found "
            "nothing and this mode would report every build as scalar")
    return report


def _write_no_model_source() -> Path:
    """A harness that builds no model, for the capture's own plant."""
    import tempfile

    path = Path(tempfile.mkdtemp()) / "harness_without_a_model.py"
    path.write_text("def run():\n    return 1\n")
    return path


def self_check() -> int:
    """Every plant must be refused, and the readers must be non-vacuous."""
    global _NO_MODEL_SOURCE
    _NO_MODEL_SOURCE = _write_no_model_source()
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
    # Each plant must be REFUSED.  A plant that merely changes a number the
    # gate prints is not a plant; these all have to reach an exception.
    def _drift():
        rows = config_diff(plant="config-drift")["differing_fields"]
        require(not rows, f"config-drift is visible: {rows}")

    def _program_drift():
        rows = config_diff(plant="program-drift")["differing_fields"]
        require(not rows, f"program-drift is visible: {rows}")

    expect_raises("config-drift", _drift)
    expect_raises("program-drift", _program_drift)
    expect_raises("same-program-while-different", _plant_same_program)
    expect_raises("state-drift", lambda: config_diff(plant="state-drift"))
    expect_raises("missing-function", lambda: model_config_argument(
        LADDER_GATE, "no_such_function"))
    expect_raises("capture-with-no-model", lambda: capture_model_config(
        "self_check_no_model", _NO_MODEL_SOURCE, lambda module: module.run()))
    expect_raises("oracle-floor-against-itself", lambda: oracle_floor(
        steps=2, roots=(ORACLE_V1_ROOT, ORACLE_V1_ROOT)))
    for name in failures:
        print(f"  FAILED: {name}")
    return 1 if failures else 0


def _plant_same_program():
    """Assert an empty table while a field is deliberately perturbed."""
    _policy()
    card, ladder, _year, _owners = resolve_programs()
    drifted = ladder._replace(
        lateral_viscosity=ladder.lateral_viscosity._replace(
            A_h=float(ladder.lateral_viscosity.A_h) * 2.0))
    rows = config_rows(ladder, drifted)
    require(not rows, "plant: the diff table is not empty")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-diff", action="store_true")
    parser.add_argument("--two-path", action="store_true")
    parser.add_argument("--oracle-floor", action="store_true")
    parser.add_argument("--score-both-roots", action="store_true")
    parser.add_argument("--closure-ablation", action="store_true")
    parser.add_argument("--oracle-provenance", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--steps", type=int, default=2,
                        help="how many step boundaries to walk; "
                             "--oracle-floor and "
                             "--score-both-roots read it too")
    parser.add_argument("--entry-root", type=Path, default=ORACLE_V2_ROOT,
                        help="the oracle record to score against; v2 is what "
                             "the certified receipts pass")
    parser.add_argument("--plant", choices=("config-drift", "program-drift",
                                            "vacuous-reseed", "same-program",
                                            "state-drift"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-unified", action="store_true",
                        help="exit non-zero unless the two programs resolve to "
                             "the same object (the unification gate)")
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    require(args.config_diff or args.two_path or args.oracle_floor
            or args.score_both_roots
            or args.closure_ablation
            or args.oracle_provenance,
            "choose --config-diff, --two-path, --oracle-floor, "
            "--score-both-roots, --closure-ablation, "
            "--oracle-provenance or --self-check")
    report = {}
    if args.config_diff:
        report["config_diff"] = config_diff(plant=args.plant)
    if args.oracle_floor:
        report["oracle_floor"] = oracle_floor(steps=args.steps)
    if args.score_both_roots:
        report["score_both_roots"] = score_both_roots(steps=args.steps)
    if args.closure_ablation:
        report["closure_ablation"] = closure_ablation(steps=args.steps)
    if args.oracle_provenance:
        report["oracle_provenance"] = oracle_provenance()
    if args.two_path:
        report["two_path"] = two_path(steps=args.steps,
                                      entry_root=args.entry_root,
                                      plant=args.plant)
    if args.require_unified:
        diff = report.get("config_diff")
        require(diff is not None, "--require-unified needs --config-diff")
        rows = diff["differing_fields"] + diff["differing_fields_year_owners"]
        require(not rows,
                f"the harnesses resolve DIFFERENT programs: {rows}")
        print("\nUNIFIED: the ladder, the year and the year-owners harness "
              "all hand the model the same object.")
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
