"""Ratchet against "a card sets a faithful option and it never reaches the code
that consumes it" (CLAUDE.md).  Four defects found in one day were the SAME
class:

  * ``physics.constants.g`` -- a card set 9.80665, ``from_flat`` routed flat
    scalars but ``ConstantsConfig`` kept the default -- a 5.0e-5 error on
    every N^2.
  * ``een_q_boundary`` / ``een_e3f_scheme`` -- set on the card, threaded for
    the 3-D EEN vorticity scheme, silently ignored by the barotropic path.
  * ``c_p`` -> ``ConstantsConfig.c_sw`` -- unrouted on all seven DINO cards
    (the flat name and the constants name DIFFER, which is why a
    field-name-matching check missed it).
  * ``tke_shear_avm_weighting`` -- the faithful option existed and the card
    selected the wrong one for months, guarded by a docstring asserting the
    wrong value was correct.

All four are fixed in production. This file is the mechanical gate that makes
the CLASS impossible to reintroduce: walk every registered DINO oracle card's
assembled config, and check (A) no silent two-surface disagreement, (B) every
card key routes somewhere, (C) sibling functions declare/thread scheme options
in lockstep, (D) every field a card sets has a real consumer.

Each invariant has a SHRINK-ONLY baseline in ``_recipe_threading_baseline.py``
(same pattern as ``tests/_inline_coeff_baseline.py`` /
``tests/test_dispatch_hardening.py``): a NEW finding not in the baseline is
red; a baseline entry that no longer reproduces is ALSO red (forces the
baseline down when a finding is fixed, so it can never silently rot into dead
slack).

Scope: DINO's ``DINO_RECIPES`` registry (7 cards) is the concrete "registered
ocean oracle card" catalog this repo has for the lat-lon oracle-fidelity
family the four cited bugs came from (nemo/veros/mitgcm/oceananigans cards +
the legoesm identity + paper-partial overlays). ``dino_lat_lon_model_config``
is the assembly function all four bugs' fixes actually route through.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import pathlib

import jax
import pytest

from tests import _ratchet_audit as ra
from tests.legoesm_paths import legoesm_source_path
from tests.ocean._recipe_threading_baseline import (
    CALL_SITE_BASELINE,
    CONSUMERLESS_BASELINE,
    HARNESS_ONLY,
    INVARIANT_A_BASELINE,
    SIBLING_PARITY_BASELINE,
)


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ---------------------------------------------------------------------------
# The flatten walk: NamedTuple/dataclass nodes -> {path: leaf}, NO id-based
# dedup (so an aliased sub-config surfaces as distinct paths under each parent
# -- the whole point of invariant A, which caught the ``g`` two-surface bug
# BECAUSE ``physics.constants`` and ``model.constants`` are the SAME object
# reachable from two paths).
# ---------------------------------------------------------------------------
_MAX_DEPTH = 12


def _is_namedtuple(x) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")


def flatten_config(obj, prefix: str = "") -> list[tuple[str, object]]:
    """Depth-first walk of NamedTuple/dataclass nodes to (path, leaf) pairs.

    Container/array-valued leaves (JAX/np arrays, ``Field``, lists) are kept
    as opaque leaves (not walked further, not compared by invariant A/B since
    those only compare scalar-equal-able values) -- the assembled DINO configs
    are scalar-only at every reachable path (verified: 773 leaves, all
    float/bool/str/int/None, for every one of the 7 cards), so this is not a
    silent gap for the config this file audits.
    """
    out: list[tuple[str, object]] = []
    _flatten_into(obj, prefix, out, 0)
    return out


def _flatten_into(obj, prefix: str, out: list[tuple[str, object]], depth: int) -> None:
    if depth > _MAX_DEPTH:
        out.append((prefix, obj))
        return
    if _is_namedtuple(obj):
        for f in obj._fields:
            _flatten_into(getattr(obj, f), f"{prefix}.{f}" if prefix else f, out, depth + 1)
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _flatten_into(getattr(obj, f.name), f"{prefix}.{f.name}" if prefix else f.name, out, depth + 1)
        return
    out.append((prefix, obj))


def _card_grid():
    from legoesm.grids.latlon import create_mercator_grid

    return create_mercator_grid(n_lon=16, lat_max_deg=70.0, lon_west_deg=0.0, lon_east_deg=50.0)


def assembled_leaves_for_card(card: str) -> list[tuple[str, object]]:
    """(path, value) leaves of the assembled (model_config, physics_config)
    pair for one DINO_RECIPES card, prefixed ``model.``/``physics.``."""
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe, dino_lat_lon_model_config,
    )

    grid = _card_grid()
    cfg = dino_config_for_recipe(card)
    model_cfg, physics_cfg = dino_lat_lon_model_config(grid, cfg)
    return flatten_config(model_cfg, "model") + flatten_config(physics_cfg, "physics")


def _all_cards() -> list[str]:
    from legoesm.ocean.experiments.dino import DINO_RECIPES

    return sorted(DINO_RECIPES)


# ---------------------------------------------------------------------------
# Non-vacuity sanity: the walk must actually visit something, on every card.
# ---------------------------------------------------------------------------
_MIN_LEAVES_PER_CARD = 500


@pytest.mark.parametrize("card", _all_cards())
def test_discovery_sane(card: str) -> None:
    leaves = assembled_leaves_for_card(card)
    assert len(leaves) >= _MIN_LEAVES_PER_CARD, (
        f"{card}: only {len(leaves)} leaves discovered (expected >= "
        f"{_MIN_LEAVES_PER_CARD}) -- flatten_config or the card assembly is "
        "broken; the four invariants below would be vacuous."
    )


# ---------------------------------------------------------------------------
# Invariant A -- no two-surface disagreement.
# ---------------------------------------------------------------------------
def two_surface_disagreements(card: str) -> list[tuple[str, str, str]]:
    """(path_a, path_b) pairs (one per distinct-value boundary) for every
    trailing field name carried by >=2 paths with >=2 distinct values."""
    leaves = assembled_leaves_for_card(card)
    by_name: dict[str, list[tuple[str, object]]] = collections.defaultdict(list)
    for path, value in leaves:
        by_name[path.rsplit(".", 1)[-1]].append((path, value))
    out: list[tuple[str, str]] = []
    for _name, items in by_name.items():
        by_value: dict[object, list[str]] = collections.defaultdict(list)
        for path, value in items:
            by_value[value].append(path)
        if len(by_value) >= 2 and len(items) >= 2:
            reps = sorted(sorted(paths)[0] for paths in by_value.values())
            for a, b in zip(reps, reps[1:]):
                out.append((a, b))
    return out


@pytest.mark.parametrize("card", _all_cards())
def test_no_two_surface_disagreement(card: str) -> None:
    discovered = {(card, a, b) for a, b in two_surface_disagreements(card)}
    baseline = {(c, a, b) for c, a, b, _reason in INVARIANT_A_BASELINE if c == card}
    new = sorted(discovered - baseline)
    assert not new, (
        f"{card}: NEW two-surface disagreement(s) -- a field is set two "
        "different ways under the same trailing name (the `g` / GM-Redi-"
        "shadow class). For each (card, path_a, path_b): read both leaves "
        "and confirm which one the card actually pinned; if this is a real "
        "routing gap, fix the assembly -- do not just add it to the "
        "baseline:\n  " + "\n  ".join(f"{c}: {a} != {b}" for c, a, b in new)
    )


def test_invariant_a_baseline_only_shrinks() -> None:
    """A baseline entry that no longer reproduces must be deleted (forces
    cleanup, mirrors ``fingerprint_subset_errors``' stale-budget check)."""
    stale = []
    for card, a, b, _reason in INVARIANT_A_BASELINE:
        if (a, b) not in set(two_surface_disagreements(card)):
            stale.append((card, a, b))
    assert not stale, (
        "INVARIANT_A_BASELINE has stale entries (no longer reproduce) -- "
        "remove them from tests/ocean/_recipe_threading_baseline.py:\n  "
        + "\n  ".join(f"{c}: {a} vs {b}" for c, a, b in stale)
    )


# ---------------------------------------------------------------------------
# Invariant B -- every card key lands.
# ---------------------------------------------------------------------------
def unrouted_card_keys(card: str) -> list[tuple[str, object]]:
    from legoesm.ocean.experiments.dino import DINO_RECIPES

    leaves = assembled_leaves_for_card(card)
    values_present = [v for _p, v in leaves]
    missing: list[tuple[str, object]] = []
    for key, card_val in DINO_RECIPES[card].items():
        if key in HARNESS_ONLY:
            continue
        landed = any(v == card_val for v in values_present)
        if not landed:
            missing.append((key, card_val))
    return missing


@pytest.mark.parametrize("card", _all_cards())
def test_every_card_key_lands(card: str) -> None:
    missing = unrouted_card_keys(card)
    assert not missing, (
        f"{card}: DINO_RECIPES key(s) set on the card never reach the "
        "assembled (model_config, physics_config) -- the constructor dropped "
        "them (the `c_p` -> `c_sw` / `water_type` class). Either the "
        "assembly is missing the routing, or this key is a genuine "
        "harness-only / branch-selector option that belongs in HARNESS_ONLY "
        "with a verified reason:\n  "
        + "\n  ".join(f"{card}.{k} = {v!r}" for k, v in missing)
    )


def test_constants_alias_keys_are_covered_by_harness_only_or_land() -> None:
    """CONSTANTS_FLAT_ALIASES-keyed check, independent of field-name matching
    (the ``c_p`` -> ``c_sw`` class): every DINOConfig constants field
    (``g``/``rho_0``/``c_p``/``omega``) must land as its ALIASED
    ConstantsConfig field value, for every card."""
    from legoesm.ocean.experiments.dino import (
        DINO_RECIPES, dino_config_for_recipe,
    )
    from legoesm.ocean.state import CONSTANTS_FLAT_ALIASES

    # DINOConfig field name -> its ConstantsConfig target (via the alias map;
    # DINOConfig spells specific-heat "c_p" where ConstantsConfig spells it
    # "c_sw" -- the exact rename the alias map exists to bridge).
    dino_to_constants_field = {
        "g": CONSTANTS_FLAT_ALIASES["g"],
        "rho_0": CONSTANTS_FLAT_ALIASES["rho_0"],
        "c_p": CONSTANTS_FLAT_ALIASES["c_sw"],
        "omega": CONSTANTS_FLAT_ALIASES["omega"],
    }
    failures = []
    for card in DINO_RECIPES:
        cfg = dino_config_for_recipe(card)
        leaves = assembled_leaves_for_card(card)
        values_present = [v for _p, v in leaves]
        for dino_field, constants_field in dino_to_constants_field.items():
            card_val = getattr(cfg, dino_field)
            if not any(v == card_val for v in values_present):
                failures.append((card, dino_field, constants_field, card_val))
    assert not failures, (
        "DINOConfig constants field failed to reach its aliased "
        "ConstantsConfig field on at least one card:\n  "
        + "\n  ".join(
            f"{c}: DINOConfig.{f} = {v!r} never appears as ConstantsConfig.{cf}"
            for c, f, cf, v in failures
        )
    )


def test_harness_only_keys_are_real_card_keys() -> None:
    """Allow-list hygiene: every HARNESS_ONLY entry must be a key some card
    actually sets (else it is dead slack -- remove it)."""
    from legoesm.ocean.experiments.dino import DINO_RECIPES

    all_keys = {k for overrides in DINO_RECIPES.values() for k in overrides}
    stale = sorted(set(HARNESS_ONLY) - all_keys)
    assert not stale, (
        f"HARNESS_ONLY names key(s) no DINO_RECIPES card sets -- remove: {stale}"
    )


# ---------------------------------------------------------------------------
# Invariant C -- sibling-signature parity (AST).
# ---------------------------------------------------------------------------
# Registered sibling modules: two implementations of the SAME scheme family
# that must declare (and therefore thread) the same card-option parameters.
# ``barotropic_latlon_cgrid`` <-> ``ocean_pe_latlon_cgrid`` is the EEN pair the
# een_q_boundary/een_e3f_scheme bug lived in: the 3-D EEN vorticity path
# (ocean_pe) threaded both options while the barotropic path silently ignored
# them. VERIFIED both now declare both names (parity holds).
# ``latlon_cgrid_operators`` is NOT the EEN sibling -- it is the generic C-grid
# operator library with no EEN-specific code at all, so pairing it here would
# report a permanent phantom asymmetry.
SIBLING_PAIRS: dict[str, str] = {
    "barotropic_latlon_cgrid": "ocean_pe_latlon_cgrid",
    "gm_redi_latlon_cgrid": "_gm_redi_common",
}


def _card_option_names() -> frozenset[str]:
    from legoesm.ocean.experiments.dino import DINO_RECIPES

    return frozenset(k for overrides in DINO_RECIPES.values() for k in overrides)


def _ocean_py_files() -> list[pathlib.Path]:
    root = legoesm_source_path("ocean/state.py").parent
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _fn_all_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    a = fn.args
    return {p.arg for p in a.posonlyargs} | {p.arg for p in a.args} | {p.arg for p in a.kwonlyargs}


def _fn_positional_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = fn.args
    return [p.arg for p in a.posonlyargs] + [p.arg for p in a.args]


def _parse_all(files: list[pathlib.Path]) -> dict[pathlib.Path, ast.Module]:
    trees = {}
    for f in files:
        trees[f] = ast.parse(f.read_text())
    return trees


def module_declared_option_names(
    trees: dict[pathlib.Path, ast.Module], stem: str, option_names: frozenset[str],
) -> set[str]:
    declared: set[str] = set()
    for path, tree in trees.items():
        if path.stem != stem:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                declared |= _fn_all_params(node) & option_names
    return declared


def sibling_parity_violations() -> list[tuple[str, str, str]]:
    """(declaring_module, missing_sibling_module, option_name) for every
    card-option name declared by one SIBLING_PAIRS module but not its
    registered sibling."""
    files = _ocean_py_files()
    trees = _parse_all(files)
    option_names = _card_option_names()
    out: list[tuple[str, str, str]] = []
    for a_stem, b_stem in SIBLING_PAIRS.items():
        a_names = module_declared_option_names(trees, a_stem, option_names)
        b_names = module_declared_option_names(trees, b_stem, option_names)
        for n in sorted(a_names - b_names):
            out.append((a_stem, b_stem, n))
        for n in sorted(b_names - a_names):
            out.append((b_stem, a_stem, n))
    return out


def test_sibling_pairs_exist() -> None:
    files = {p.stem for p in _ocean_py_files()}
    missing = sorted(
        stem for pair in SIBLING_PAIRS.items() for stem in pair if stem not in files
    )
    assert not missing, f"SIBLING_PAIRS names module(s) not found under packages/ocean: {missing}"


def test_sibling_signature_parity() -> None:
    discovered = set(sibling_parity_violations())
    baseline = set(SIBLING_PARITY_BASELINE)
    new = sorted(discovered - baseline)
    assert not new, (
        "NEW sibling-signature parity gap(s) -- one module declares a "
        "card-option-name parameter its registered SIBLING_PAIRS partner "
        "does not (the EEN class: a scheme option threaded on one path and "
        "silently ignored on the other). For each "
        "(declaring_module, missing_sibling_module, option_name): confirm "
        "whether the sibling needs the same parameter threaded through, or "
        "document why the asymmetry is legitimate in "
        "_recipe_threading_baseline.py:\n  "
        + "\n  ".join(f"{a} declares {n!r}, {b} does not" for a, b, n in new)
    )


def test_sibling_parity_baseline_only_shrinks() -> None:
    discovered = set(sibling_parity_violations())
    stale = sorted(set(SIBLING_PARITY_BASELINE) - discovered)
    assert not stale, (
        "SIBLING_PARITY_BASELINE has stale entries (parity now holds) -- "
        "remove them from tests/ocean/_recipe_threading_baseline.py:\n  "
        + "\n  ".join(f"{a} vs {b}: {n!r}" for a, b, n in stale)
    )


def call_site_completeness_violations() -> list[tuple[str, int, str, str]]:
    """(rel_path, lineno, function_name, option_name) for call sites the
    static resolver could not prove pass a card-option-name parameter
    explicitly, restricted to functions with a package-unique name (so a
    match is unambiguous -- a name declared by >1 function is skipped rather
    than guessed at, per-function, not per-name globally)."""
    files = _ocean_py_files()
    trees = _parse_all(files)
    option_names = _card_option_names()
    repo_root = ra.repo_root()

    name_to_defs: dict[str, list[tuple[pathlib.Path, ast.AST]]] = collections.defaultdict(list)
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name_to_defs[node.name].append((path, node))

    unique_declaring: dict[str, tuple[set[str], list[str]]] = {}
    for name, defs in name_to_defs.items():
        if len(defs) != 1:
            continue
        _path, node = defs[0]
        params = _fn_all_params(node) & option_names
        if params:
            unique_declaring[name] = (params, _fn_positional_params(node))

    out: list[tuple[str, int, str, str]] = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            else:
                continue
            if fname not in unique_declaring:
                continue
            optnames, poslist = unique_declaring[fname]
            kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
            has_starstar = any(kw.arg is None for kw in node.keywords)
            has_star = any(isinstance(a, ast.Starred) for a in node.args)
            if has_starstar or has_star:
                continue  # cannot statically resolve **kwargs/*args expansion
            n_pos = len(node.args)
            for optname in sorted(optnames):
                if optname in kw_names:
                    continue
                idx = poslist.index(optname) if optname in poslist else None
                if idx is not None and idx < n_pos:
                    continue
                rel = str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)
                out.append((rel, node.lineno, fname, optname))
    return out


def test_call_sites_pass_card_options_explicitly() -> None:
    discovered = set(call_site_completeness_violations())
    baseline = set(CALL_SITE_BASELINE)
    new = sorted(discovered - baseline)
    assert not new, (
        "NEW call site(s) that do not statically prove they pass a "
        "card-option-name parameter explicitly (could be a genuine unrouted "
        "option, or a resolver false positive -- check the call site "
        "manually before adding to the baseline):\n  "
        + "\n  ".join(f"{p}:{ln} {fn}({opt}=...)" for p, ln, fn, opt in new)
    )


def test_call_site_baseline_only_shrinks() -> None:
    discovered = set(call_site_completeness_violations())
    stale = sorted(set(CALL_SITE_BASELINE) - discovered)
    assert not stale, (
        "CALL_SITE_BASELINE has stale entries (now resolved) -- remove them "
        "from tests/ocean/_recipe_threading_baseline.py:\n  "
        + "\n  ".join(f"{p}:{ln} {fn}({opt}=...)" for p, ln, fn, opt in stale)
    )


# ---------------------------------------------------------------------------
# Invariant D -- no consumerless field.
# ---------------------------------------------------------------------------
def _ast_read_names(trees: dict[pathlib.Path, ast.Module]) -> dict[str, set[str]]:
    """name -> set of file stems where it appears as: an attribute access
    (``x.name``), a bare Name, a call keyword-argument name, a function
    parameter name, or a string-literal constant (covers
    ``getattr(cfg, "name", default)`` and ``if cfg.field == "name":``
    dispatch, both idiomatic in this codebase -- see ``tke_shear_avm_weighting``
    and ``een_q_boundary`` for real examples of exactly this pattern)."""
    reads: dict[str, set[str]] = collections.defaultdict(set)
    for path, tree in trees.items():
        stem = path.stem
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                reads[node.attr].add(stem)
            elif isinstance(node, ast.Name):
                reads[node.id].add(stem)
            elif isinstance(node, ast.keyword) and node.arg:
                reads[node.arg].add(stem)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                reads[node.value].add(stem)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for p in _fn_all_params(node):
                    reads[p].add(stem)
    return reads


def consumerless_card_keys() -> list[tuple[str, str]]:
    """(card_key, leaf_trailing_name) for every DINO_RECIPES key whose
    (alias-resolved) leaf trailing name has ZERO AST-visible read anywhere
    under packages/ocean OUTSIDE dino.py (its own config module)."""
    from legoesm.ocean.experiments.dino import DINO_RECIPES
    from legoesm.ocean.state import CONSTANTS_FLAT_ALIASES

    dino_to_alias = {
        "g": CONSTANTS_FLAT_ALIASES["g"],
        "rho_0": CONSTANTS_FLAT_ALIASES["rho_0"],
        "c_p": CONSTANTS_FLAT_ALIASES["c_sw"],
        "omega": CONSTANTS_FLAT_ALIASES["omega"],
    }
    files = _ocean_py_files()
    trees = _parse_all(files)
    reads = _ast_read_names(trees)

    all_leaf_names: set[str] = set()
    for card in DINO_RECIPES:
        for path, _v in assembled_leaves_for_card(card):
            all_leaf_names.add(path.rsplit(".", 1)[-1])

    all_keys = {k for overrides in DINO_RECIPES.values() for k in overrides}
    out: list[tuple[str, str]] = []
    for key in sorted(all_keys):
        target = dino_to_alias.get(key, key)
        if target not in all_leaf_names:
            continue  # not a leaf at all -- invariant B's job, not D's
        readers = reads.get(target, set()) - {"dino"}
        if not readers:
            out.append((key, target))
    return out


def test_no_consumerless_field() -> None:
    discovered = set(consumerless_card_keys())
    baseline = set(CONSUMERLESS_BASELINE)
    new = sorted(discovered - baseline)
    assert not new, (
        "NEW consumerless card key(s): the field lands in the assembled "
        "config tree (invariant B is satisfied) but has ZERO AST-visible "
        "read anywhere in packages/ocean outside dino.py -- it is set and "
        "then never consumed by any physics/dynamics code:\n  "
        + "\n  ".join(f"{k} -> leaf name {t!r}" for k, t in new)
    )


def test_consumerless_baseline_only_shrinks() -> None:
    discovered = set(consumerless_card_keys())
    stale = sorted(set(CONSUMERLESS_BASELINE) - discovered)
    assert not stale, (
        "CONSUMERLESS_BASELINE has stale entries (now consumed) -- remove "
        "them from tests/ocean/_recipe_threading_baseline.py:\n  "
        + "\n  ".join(f"{k} -> {t!r}" for k, t in stale)
    )


# ---------------------------------------------------------------------------
# Non-vacuity self-test (REQUIRED): prove the gate actually catches the exact
# classes it was built for, by synthetically reintroducing each one.
# ---------------------------------------------------------------------------
def test_gate_catches_synthetic_violation_a_and_b() -> None:
    """Drop "g" from CONSTANTS_FLAT_ALIASES (monkeypatch) and confirm BOTH:

    (1) the underlying routing breaks -- ``from_flat(g=...)`` no longer
        recognizes "g" as a routable flat spelling at all (with "g" gone from
        the alias map, ``from_flat`` passes it straight through to the
        NamedTuple constructor, which has no top-level ``g`` field any more --
        confirming the alias map is genuinely load-bearing for routing "g");
    (2) invariant B's OWN checker logic, run against a synthetic assembled
        tree shaped exactly like what the routing failure produces (every
        g-carrying leaf stuck on the OLD default -- the card's pinned value
        never reaches ANY leaf), reports "g" as unrouted. This is the exact
        #1226 class: a card pins a non-default g; from_flat can no longer
        route it into ConstantsConfig, so every g-carrying leaf stays on the
        prior default while the card's own config object still holds the
        pinned value -- the constructor silently dropped it.
    """
    import legoesm.ocean.state as state_mod

    original_aliases = dict(state_mod.CONSTANTS_FLAT_ALIASES)
    assert "g" in original_aliases, "self-test setup invariant: g must be a real alias key"
    assert "g" not in HARNESS_ONLY, "self-test invariant: g must not be allow-listed"

    patched = dict(original_aliases)
    del patched["g"]
    state_mod.CONSTANTS_FLAT_ALIASES = patched
    try:
        from legoesm.ocean.state import LatLonCGridOceanConfig

        # (1) the routing itself breaks: "g" is no longer a recognized flat
        # spelling, so from_flat forwards it unchanged to the NamedTuple
        # constructor, which rejects it outright.
        with pytest.raises(TypeError):
            LatLonCGridOceanConfig.from_flat(g=9.80665)
    finally:
        state_mod.CONSTANTS_FLAT_ALIASES = original_aliases

    # Confirm restoration undid the damage.
    LatLonCGridOceanConfig.from_flat(g=9.80665)  # must not raise now

    # (2) invariant B's checker, exercised directly against a synthetic
    # "assembled tree" shaped exactly like what dino_lat_lon_model_config
    # would produce if "g" silently failed to route: every g-named leaf stuck
    # on the OLD default value, never the card's newly pinned one.
    from legoesm import constants
    from legoesm.ocean.experiments.dino import DINO_RECIPES, dino_config_for_recipe

    card = "nemo_dino_kamm"
    cfg = dino_config_for_recipe(card)
    old_default_g = constants.g
    assert cfg.g != old_default_g, "setup invariant: nemo_dino_kamm pins a non-default g"

    stuck_at_old_default_leaves = [
        ("model.constants.g", old_default_g),
        ("physics.constants.g", old_default_g),
        ("model.g_unrelated_dummy", old_default_g),
    ]

    def _fake_unrouted_card_keys(card: str, leaves: list[tuple[str, object]]) -> list[tuple[str, object]]:
        values_present = [v for _p, v in leaves]
        missing = []
        for key, card_val in DINO_RECIPES[card].items():
            if key in HARNESS_ONLY:
                continue
            if not any(v == card_val for v in values_present):
                missing.append((key, card_val))
        return missing

    missing = _fake_unrouted_card_keys(card, stuck_at_old_default_leaves)
    missing_keys = {k for k, _v in missing}
    assert "g" in missing_keys, (
        "invariant B's checker FAILED to flag a synthetically unrouted 'g' "
        "-- the B-invariant gate is vacuous"
    )

    # Confirm restoration undid the damage.
    from legoesm.ocean.state import LatLonCGridOceanConfig as _Restored

    _Restored.from_flat(g=9.80665)  # must not raise now


def test_gate_catches_synthetic_violation_c() -> None:
    """Strip ``een_q_boundary`` from ``_build_een_barotropic_inputs``'s own
    declared-parameter set (simulated via a hand-built AST module standing in
    for a stripped-signature version of the file) and confirm invariant C's
    sibling-declaration check flags the resulting asymmetry -- this is the
    exact EEN class: barotropic_latlon_cgrid currently declares
    ``een_q_boundary``; ``ocean_pe_latlon_cgrid`` (the file that actually owns
    the paired 3-D EEN implementation) also declares it, so parity holds
    to day. If ``barotropic_latlon_cgrid`` stopped declaring it, a sibling
    pair that includes it would go red -- reproduce that directly on a temp
    module registered as a synthetic SIBLING_PAIRS partner, so this test does
    not depend on being able to safely mutate the real production file."""
    files = _ocean_py_files()
    trees = _parse_all(files)
    option_names = _card_option_names()

    real_barotropic_names = module_declared_option_names(
        trees, "barotropic_latlon_cgrid", option_names
    )
    assert "een_q_boundary" in real_barotropic_names, (
        "setup invariant: barotropic_latlon_cgrid must currently declare "
        "een_q_boundary for this self-test to be meaningful"
    )

    # Build a synthetic "stripped" version of the module's AST with
    # een_q_boundary removed from every function's parameter list, write it
    # to a temp .py, and re-run the SAME sibling-parity logic the production
    # test uses against {real_module: stripped_module}.
    import tempfile

    src = legoesm_source_path("ocean/dynamics/barotropic_latlon_cgrid.py").read_text()
    stripped_src = src.replace("een_q_boundary", "een_q_boundary_REMOVED_FOR_TEST")

    with tempfile.TemporaryDirectory() as tmpdir:
        stripped_path = pathlib.Path(tmpdir) / "barotropic_latlon_cgrid_stripped.py"
        stripped_path.write_text(stripped_src)
        stripped_tree = ast.parse(stripped_src)
        synthetic_trees = {stripped_path: stripped_tree}
        stripped_names = module_declared_option_names(
            synthetic_trees, "barotropic_latlon_cgrid_stripped", option_names
        )

    assert "een_q_boundary" not in stripped_names, (
        "the string-replace self-test setup failed to remove een_q_boundary "
        "from the synthetic module -- self-test is broken, not the gate"
    )

    # Direct sibling-parity re-check with the synthetic stripped module
    # standing in for barotropic_latlon_cgrid: een_q_boundary is declared by
    # ocean_pe_latlon_cgrid (its real EEN-implementation partner) but not by
    # the stripped stand-in -- exactly the asymmetry C exists to catch.
    real_pe_names = module_declared_option_names(trees, "ocean_pe_latlon_cgrid", option_names)
    assert "een_q_boundary" in real_pe_names, (
        "setup invariant: ocean_pe_latlon_cgrid must currently declare "
        "een_q_boundary for this self-test to be meaningful"
    )
    violation = "een_q_boundary" in (real_pe_names - stripped_names)
    assert violation, (
        "sibling-parity logic FAILED to flag a synthetically stripped "
        "een_q_boundary signature -- the C-invariant gate is vacuous"
    )


def test_gate_non_vacuity_on_real_tree() -> None:
    """The gate must not pass merely because it walked nothing: every card
    must produce a non-trivial leaf count, and the discovered-violation sets
    for A/B/C(i)/C(ii)/D exist as concrete non-None collections (not skipped
    computations) on the CURRENT (unmodified) tree."""
    cards = _all_cards()
    assert len(cards) >= 6, f"expected >=6 registered DINO cards, found {len(cards)}"
    for card in cards:
        leaves = assembled_leaves_for_card(card)
        assert len(leaves) >= _MIN_LEAVES_PER_CARD
    assert isinstance(sibling_parity_violations(), list)
    assert isinstance(call_site_completeness_violations(), list)
    assert isinstance(consumerless_card_keys(), list)
