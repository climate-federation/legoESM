"""Repo-wide ratchet: no hardcoded physical constants outside ``constants.py``.

CLAUDE.md / CONTRIBUTING.md mandate that every physical constant comes from
``legoesm.constants`` — no bare ``273.15`` / ``9.80616`` / ``2.501e6`` literals in
production source, tests, scripts, or plotters. The motivation is an *invisible*
failure mode: an AI (or a human) re-types ``273.15`` for a C↔K conversion that
should track ``constants.T_freeze``; if the canonical value ever shifts, the
stray literal silently diverges and the bug surfaces only as a slow physical
bias. The ocean already enforces a scoped version of this rule
(``tests/ocean/unit/test_constants_audit.py``); this generalises it repo-wide.

**This is a tripwire, not a proof.** It catches a curated, high-value set of
textual regressions; it cannot catch a wrong equation, a wrong unit, misuse of a
correctly-named constant, or a value pulled from a config/JSON layer. Pair it
with the conservation/equivariance/analytic gates for actual physics correctness.

Detection (see ``tests/_ratchet_audit.py``):
  * AST + value comparison, not regex — comment/docstring text is never a
    ``Constant`` and a ``constants.T_freeze`` attribute read is not a literal.
  * Alternate spellings (``273.150``, ``6.371e6``) collapse to one float, and
    simple obfuscations (``6371 * 1000``, ``270 + 3.15``) are folded — counted at
    the *outermost* banned level so a nested expression is not double-counted.
  * The ratchet keys on the **normalised source line** of each hit, not just the
    value/count — so deleting a sanctioned ``283.0 - 273.15`` and adding a new
    bad ``T - 273.15`` in the same file produces a different fingerprint with no
    allowance and goes red. Removals shrink the observed set and need no edit;
    for the exact (``_RATCHET_DOWN``) entries a leftover allowance is flagged.
  * A genuine one-off non-physical literal can be annotated inline with a real
    ``# const-ok: <reason>`` comment (tokenised, not substring-matched).

Self-tests exercise folding, exemption, the fingerprint ratchet, and a synthetic
violation; ``test_discovery_sane`` proves the scan actually found the tree.
"""

from __future__ import annotations

import ast
import collections

import pytest

from tests import _ratchet_audit as ra

# Canonical physical constants (packages/core/legoesm/constants.py) — the
# high-value subset enumerated in CLAUDE.md, plus the exact canonical spellings
# (R_d=287.05, R_earth=6.371229e6) so both the rounded literal and the precise
# value are caught. NOT exhaustive of every constant (see module docstring).
BANNED: dict[float, str] = {
    273.15: "T_freeze",
    287.0: "R_d",
    287.05: "R_d",
    1004.64: "c_pd",
    2.501e6: "L_v",
    461.51: "R_v",
    0.622: "epsilon",
    9.80616: "g",
    6.371e6: "R_earth",
    6.371229e6: "R_earth",
    7.292e-5: "Omega",
}
_BANNED_VALUES = frozenset(BANNED)
_EXEMPT_TAG = "const"

# Files exempt entirely: the definition site, and this guard file (its BANNED
# table necessarily names the values).
_WHITELIST = frozenset(
    {
        "packages/core/legoesm/constants.py",
        "tests/test_no_hardcoded_constants.py",
    }
)

# Per-file budget keyed by normalised hit-line fingerprint -> count, seeded from
# the measured baseline (iter 2026-06-09). Files in ``_RATCHET_DOWN`` are exact
# (a removed site must drop its allowance); the rest are PERMANENT test fixtures
# carrying assert/oracle reference values (new/extra sites still caught).
LITERAL_BUDGET: dict[str, dict[str, int]] = {
    # --- production / scripts: EXACT (ratchet-down) ------------------------
    # Veros oracle-verbatim ``theta0_C = 283.0 - 273.15`` (VerosNonlin{2,3}).
    "packages/ocean/legoesm/ocean/eos.py": {
        "theta0_C: float = 283.0 - 273.15 # 9.85 °C": 2,
    },
    "scripts/run/run_les_plane.py": {
        "return jnp.where(z > 600.0, 273.15 + 0.01 * (z - 600.0), 273.15)": 2,
    },
    # ``discover_py_files`` used to match its ``scripts/tmp/`` exclusion against
    # the *absolute* checkout path, so every checkout living under a directory
    # literally named ``tmp`` (every worktree this week) silently excluded ALL
    # of scripts/ from every ratchet built on it (fixed 2026-09-03, see
    # ``tests/_ratchet_audit.py::discover_py_files``). These four sites were
    # already present in scripts/validate/ — pre-existing debt the fix
    # surfaced, not introduced by it. Not fixed here (out of scope for the
    # discovery-bug fix); ratchet-down so paying them off shrinks the budget.
    "scripts/validate/land_beta_soil_bake_discriminator.py": {
        "0.716, 0.622, 0.723, 0.457, 0.507, 0.504, 0.488, 0.477)": 1,  # pre-existing, surfaced by discovery fix 2026-09-03
    },
    # Zero allowance, kept EXPLICIT rather than absent (GLM review): an
    # absent file is default-deny too -- proved by planting a 6371229.0 and
    # watching the gate report "new/extra site (x1, budget 0)" -- but absent
    # also drops the file out of _RATCHET_DOWN and therefore out of EXACT
    # mode.  A file in ceiling mode whose allowance is later re-added can go
    # stale forever without the gate noticing, which is the failure this pair
    # of entries just cost a session to clean up.  Empty + ratchet-down keeps
    # them self-cleaning.
    "scripts/validate/ocean_fidelity/dino_1226/southern_wall_balance.py": {},
    "scripts/validate/ocean_fidelity/dino_1226/vertex_area_pair_analysis.py": {},
    # --- test fixtures: PERMANENT -----------------------------------------
    "tests/atmosphere/dycore/regression/test_pad_halo_4d_monotone_clip_iter490.py": {
        "field = jnp.full((6, n, n, nlev), 273.15)": 1,
        "np.asarray(padded), 273.15,": 1,
    },
    "tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py": {
        "_Rd = 287.0 # J/(kg*K) — DCMIP uses 287.0, not 287.05": 1,
    },
    "tests/atmosphere/hydrostatic/unit/test_gwd_e3sm_beres.py": {
        "ORACLE_G = 9.80616": 1,
    },
    "tests/atmosphere/hydrostatic/unit/test_gwd_e3sm_cam.py": {
        "ORACLE_CPAIR = 1004.64": 1,
        "ORACLE_G = 9.80616": 1,
    },
    "tests/atmosphere/hydrostatic/unit/test_gwd_e3sm_ediff.py": {
        "ORACLE_CPAIR = 1004.64": 1,
        "ORACLE_G = 9.80616": 1,
    },
    "tests/atmosphere/hydrostatic/unit/test_semi_implicit_cdgrid.py": {
        "dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)": 8,
        "dx_min = 6.371e6 * (np.pi / 2) / 8 / np.sqrt(3)": 1,
    },
    "tests/distributed/test_coupler_mpi.py": {
        "ocean_sst = _pattern(shape, base=287.0, face_scale=0.35, x_scale=0.08, y_scale=-0.04)": 1,
    },
    "tests/ocean/unit/test_eos_veros_nonlin2.py": {"_THETA0 = 283.0 - 273.15": 1},
    "tests/ocean/unit/test_eos_veros_nonlin3.py": {"_VEROS_THETA0 = 283.0 - 273.15": 1},
    "tests/unit/test_kuo_oracle_faithful.py": {"Tc = T - 273.15": 1},
    "tests/unit/test_m6_ice_nucleation.py": {
        "target = min(5.0 * math.exp(0.304 * (273.15 - T)), 5.0e5) / rho": 1,
        "assert 5.0 * math.exp(0.304 * (273.15 - T)) > 5.0e5": 1,
    },
    "tests/unit/test_m_bigg_freezing.py": {
        "assert _mnuccr(273.15) == (0.0, 0.0) # exactly 0 °C ⇒ ΔT=0 ⇒ 0": 1,
    },
    "tests/unit/test_physics_units.py": {
        "assert abs(constants.L_v - 2.501e6) / 2.501e6 < 0.01": 2,
        "assert abs(constants.R_earth - 6.371e6) / 6.371e6 < 0.01": 2,
        "assert abs(constants.epsilon - 0.622) < 0.005": 1,
        "assert abs(constants.T_freeze - 273.15) < 0.01": 1,
    },
    "tests/unit/test_rce_script.py": {"assert constants.T_freeze == 273.15": 1},
    "tests/unit/test_sea_ice_dynamics.py": {"T_max=273.15)": 1},
    "tests/unit/test_sea_ice_new_physics.py": {
        "refreeze_threshold=273.15,": 3,
        "refreeze_threshold=273.15, pond_to_ice_max_area=0.6,": 1,
    },
    "tests/unit/test_zm_dilute_parcel.py": {
        "Tc = T - 273.15": 1,
        "qsat = 0.622 * es / jnp.maximum(p_full - es, 1.0)": 1,
    },
}

# Entries held to an exact budget (a removed site must drop its allowance).
_RATCHET_DOWN = frozenset(
    {
        "packages/ocean/legoesm/ocean/eos.py",
        "scripts/run/run_les_plane.py",
        "scripts/validate/land_beta_soil_bake_discriminator.py",
        "scripts/validate/ocean_fidelity/dino_1226/southern_wall_balance.py",
        "scripts/validate/ocean_fidelity/dino_1226/vertex_area_pair_analysis.py",
    }
)


def banned_hits(src: str) -> list[tuple[int, float, str]]:
    """(lineno, value, normalized_line) for every maximal banned-constant
    expression, minus lines carrying a ``# const-ok`` comment. Raises
    ``SyntaxError`` to the caller on an unparseable file."""
    tree = ast.parse(src)
    src_lines = src.splitlines()
    exempt = ra.comment_tagged_lines(src, _EXEMPT_TAG)
    return [
        (node.lineno, value, ra.normalized_line(src_lines, node.lineno))
        for node, value in ra.iter_banned_numeric(tree, _BANNED_VALUES)
        if node.lineno not in exempt
    ]


_FILES = ra.discover_py_files()


def test_discovery_sane() -> None:
    ra.assert_discovery_sane(_FILES)


def test_discovery_excludes_only_scripts_tmp_regardless_of_checkout_path(
    tmp_path, monkeypatch
) -> None:
    """Regression (2026-09-03): ``discover_py_files`` used to exclude a file by
    testing ``"tmp" in rp.parts and "scripts" in rp.parts`` against the
    *absolute* resolved path. Any checkout living under a directory literally
    named ``tmp`` (every worktree used the week this was found did, e.g.
    ``/tmp/wt-...``) puts the exact part ``"tmp"`` in every file's parts tuple,
    so that condition was true for *every* file under ``scripts/`` — not just
    the intended ``scripts/tmp/`` throwaway probes — silently making the
    ratchet's verdict on all of ``scripts/`` vacuous in such a checkout. Fixed
    by matching the path relative to the repo root instead.

    Builds a fake checkout under a directory literally named ``tmp`` and
    checks both ends: a real ``scripts/validate/`` file is discovered *and* its
    banned literal is flagged (not just "discovery returns the path"), while a
    ``scripts/tmp/`` probe stays excluded."""
    checkout = (tmp_path / "tmp" / "fake_checkout").resolve()
    assert "tmp" in checkout.parts  # the exact trigger condition for the old bug
    validate_dir = checkout / "scripts" / "validate"
    validate_dir.mkdir(parents=True)
    probe_dir = checkout / "scripts" / "tmp"
    probe_dir.mkdir(parents=True)
    leaky = validate_dir / "leaky_probe.py"
    leaky.write_text("g = 9.80616\n")
    (probe_dir / "throwaway.py").write_text("g = 9.80616\n")

    monkeypatch.setattr(ra, "repo_root", lambda: checkout)
    files = {ra.rel(f) for f in ra.discover_py_files()}

    assert "scripts/validate/leaky_probe.py" in files
    assert "scripts/tmp/throwaway.py" not in files
    assert [v for _ln, v, _fp in banned_hits(leaky.read_text())] == [9.80616]


def test_no_stale_budget_entries() -> None:
    missing = [rel for rel in LITERAL_BUDGET if not (ra.repo_root() / rel).is_file()]
    assert not missing, f"LITERAL_BUDGET names non-existent files: {missing}"
    bad = [r for r in _RATCHET_DOWN if r not in LITERAL_BUDGET]
    assert not bad, f"_RATCHET_DOWN names files absent from LITERAL_BUDGET: {bad}"


@pytest.mark.parametrize("path", _FILES, ids=ra.rel)
def test_no_hardcoded_physical_constants(path) -> None:
    rel = ra.rel(path)
    if rel in _WHITELIST:
        pytest.skip("definition site / guard file")
    observed: collections.Counter[str] = collections.Counter()
    for _ln, _val, fp in banned_hits(path.read_text()):
        observed[fp] += 1
    allowed = LITERAL_BUDGET.get(rel, {})
    errors = ra.fingerprint_subset_errors(
        dict(observed), allowed, permanent=rel not in _RATCHET_DOWN
    )
    assert not errors, (
        f"{rel}: hardcoded physical-constant ratchet failed. Replace the literal "
        f"with the canonical ``legoesm.constants`` reference, annotate a genuine "
        f"non-physical one-off with ``# const-ok: <reason>``, or (if intentional) "
        f"adjust LITERAL_BUDGET:\n  " + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# Non-vacuity / behaviour self-tests
# ---------------------------------------------------------------------------
def test_banned_table_is_the_documented_set() -> None:
    assert _BANNED_VALUES == frozenset(
        {273.15, 287.0, 287.05, 1004.64, 2.501e6, 461.51, 0.622, 9.80616, 6.371e6,
         6.371229e6, 7.292e-5}
    )


def test_detector_flags_synthetic_violation() -> None:
    flagged = sorted(BANNED[v] for _ln, v, _fp in banned_hits("x = T + 273.15\ny=0.622*p\n"))
    assert flagged == ["T_freeze", "epsilon"], flagged


def test_detector_flags_literal_in_expression() -> None:
    assert [v for _ln, v, _fp in banned_hits("a = 283.0 - 273.15\n")] == [273.15]


def test_detector_folds_obfuscated_literal() -> None:
    assert [v for _ln, v, _fp in banned_hits("R = 6371 * 1000\n")] == [6.371e6]


def test_detector_does_not_double_count_nested_fold() -> None:
    """``(270 + 3.15) + 0`` folds to 273.15 at the outer level only — one hit."""
    assert [v for _ln, v, _fp in banned_hits("R = (270 + 3.15) + 0\n")] == [273.15]


def test_detector_allows_constant_reference() -> None:
    src = (
        "from legoesm import constants\n"
        "x = T + constants.T_freeze\n"
        "y = constants.epsilon * p\n"
    )
    assert banned_hits(src) == []


def test_inline_exemption_requires_real_comment() -> None:
    # Real comment exempts; the same text inside a string does NOT.
    assert banned_hits("ratio = 0.622  # const-ok: aspect ratio, not epsilon\n") == []
    assert len(banned_hits('label = '"'"'0.622 # const-ok'"'"'\nx = 0.622\n')) == 1


def test_detector_survives_pathologically_deep_expression() -> None:
    """A ~3000-deep BinOp must not crash the audit: ``fold_numeric`` degrades to
    ``None`` (RecursionError-guarded) and the *iterative* descent still finds the
    banned leaf. Built as an AST directly to exercise the fold/descent guards
    without tripping CPython's separate parser recursion limit."""
    node: ast.AST = ast.Constant(value=273.15)
    for _ in range(3000):
        node = ast.BinOp(left=node, op=ast.Add(), right=ast.Constant(value=1.0))
    assert ra.fold_numeric(node) is None  # too deep to fold -> graceful
    found = [v for _n, v in ra.iter_banned_numeric(node, _BANNED_VALUES)]
    assert 273.15 in found


def test_fingerprint_ratchet_catches_same_value_relocation() -> None:
    """Delete a sanctioned site, add a new bad one of the same value elsewhere:
    different normalised line => no allowance => red (round-2 HIGH)."""
    errs = ra.fingerprint_subset_errors(
        {"T_c = T - 273.15": 1}, {"theta0 = 283.0 - 273.15": 1}, permanent=True
    )
    assert errs
