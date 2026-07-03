"""G-C5 — audit guard for the ocean constants de-mirroring (Phase G, Q4).

G-C2/G-C4 removed the module-level constant *mirrors* (e.g. ``_RHO_0 =
constants.rho_ocean``) and the inline ``constants.g`` / ``constants.rho_ocean``
/ ``constants.c_sw`` / ``constants.Omega`` / ``constants.R_earth`` tendency
reads from the ocean vertical-mixing + lateral-mixing modules, replacing them
with values threaded from ``config.constants`` (``ConstantsConfig``). A recipe
can therefore pin those constants to a reference model (Veros) purely through
the public config API — that is what let the ``override_constants`` monkey-patch
be deleted.

This guard FAILS if any de-mirrored module reintroduces a module-level mirror
or an inline read of a ConstantsConfig-scoped constant. The ONLY allowed
appearance of a scoped constant in these modules is as a *function-parameter
default* (``def f(..., g: float = constants.g)``) — the doctrine-approved
fallback that callers override with ``config.constants.g`` (CLAUDE.md: "No
hardcoded physical constants in function signatures or bodies ... use
``g: float = constants.g``").

The detection logic is itself tested against a synthetic source string
(``test_audit_detects_synthetic_violations``) so the gate is provably not
vacuous.
"""

from __future__ import annotations

import ast
import pathlib

import legoesm

# Attribute names on the ``legoesm.constants`` module that ``ConstantsConfig``
# owns (g, rho_0->rho_ocean, c_sw, Omega, R_earth). A read of any of these in a
# de-mirrored module — other than as a function-parameter default — is a
# regression of the de-mirroring.
SCOPED = frozenset({"g", "rho_ocean", "c_sw", "Omega", "R_earth"})

# Modules de-mirrored in G-C2/G-C4 (see autonomous_progress.md iters 10, 11, 18).
# Each now reads the scoped constants ONLY via a value threaded from
# config.constants, with a ``= constants.X`` function-parameter default as the
# fallback. Relative to ``src/legoesm``.
DEMIRRORED = (
    "ocean/physics/vertical_mixing/integration.py",
    "ocean/physics/vertical_mixing/k_profiles.py",
    "ocean/physics/vertical_mixing/tke.py",
    "ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py",
)


def _resolve_module(rel: str) -> pathlib.Path | None:
    """Resolve a ``src/legoesm``-relative module path against the ``legoesm``
    namespace package's ``__path__``.

    ``legoesm`` is a federation NAMESPACE package: it has no ``__file__`` (it is
    ``None``), and its source is split across several ``packages/<pkg>/legoesm``
    roots exposed via ``__path__``. The de-mirrored ocean modules live under
    ``packages/ocean/legoesm``, so we probe every ``__path__`` entry and return
    the first that actually contains the module (or ``None`` if missing)."""
    for root in legoesm.__path__:
        path = pathlib.Path(root).resolve() / rel
        if path.exists():
            return path
    return None


def _scoped_constant_attrs(node: ast.AST) -> list[ast.Attribute]:
    """Every ``constants.<SCOPED>`` Attribute node reachable from ``node``."""
    out: list[ast.Attribute] = []
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Attribute)
            and n.attr in SCOPED
            and isinstance(n.value, ast.Name)
            and n.value.id == "constants"
        ):
            out.append(n)
    return out


def _param_default_attr_ids(tree: ast.AST) -> set[int]:
    """ids of scoped-constant Attribute nodes that are function-parameter
    defaults (positional ``defaults`` or keyword-only ``kw_defaults``)."""
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]
            for d in defaults:
                for a in _scoped_constant_attrs(d):
                    allowed.add(id(a))
    return allowed


def _violations_in_source(src: str, name: str) -> list[tuple[str, int, str]]:
    """Return (name, lineno, attr) for every scoped-constant read that is NOT a
    function-parameter default — i.e. a module-level mirror or an inline body
    read. This single rule catches both regression classes at once."""
    tree = ast.parse(src)
    allowed = _param_default_attr_ids(tree)
    return [
        (name, n.lineno, n.attr)
        for n in _scoped_constant_attrs(tree)
        if id(n) not in allowed
    ]


def test_demirrored_modules_have_no_scoped_constant_reads():
    """The de-mirrored ocean modules must read ConstantsConfig-scoped constants
    ONLY as function-parameter defaults. Any module-level mirror or inline body
    read is a regression of the G-C2/G-C4 de-mirroring."""
    violations: list[tuple[str, int, str]] = []
    for rel in DEMIRRORED:
        path = _resolve_module(rel)
        assert path is not None, f"de-mirrored module missing: {rel}"
        violations += _violations_in_source(path.read_text(), rel)

    assert not violations, (
        "Reintroduced ConstantsConfig-scoped constant read(s) in de-mirrored "
        "ocean module(s). Thread the value from config.constants instead; the "
        "only allowed form is a function-parameter default ``= constants.X``:\n"
        + "\n".join(
            f"  {rel}:{lineno}  constants.{attr}"
            for rel, lineno, attr in violations
        )
    )


def test_audit_detects_synthetic_violations():
    """The detector must FLAG a module-level mirror and an inline body read,
    but ALLOW a function-parameter default — proving the gate is not vacuous
    (spec: confirm the guard goes red on a deliberate violation)."""
    src = (
        "from legoesm import constants\n"
        "_RHO_0 = constants.rho_ocean\n"           # module-level mirror -> FLAG
        "def f(g: float = constants.g):\n"         # param default       -> ALLOW
        "    return g * constants.c_sw\n"           # inline body read    -> FLAG
    )
    flagged = sorted(attr for _name, _ln, attr in _violations_in_source(src, "<syn>"))
    # rho_ocean (mirror) and c_sw (body) flagged; g (param default) NOT.
    assert flagged == ["c_sw", "rho_ocean"], flagged


def test_audit_passes_on_pure_default_source():
    """A module that reads scoped constants ONLY as parameter defaults must be
    clean — the canonical de-mirrored shape."""
    src = (
        "from legoesm import constants\n"
        "def f(g: float = constants.g, rho_0: float = constants.rho_ocean,\n"
        "      *, c_sw: float = constants.c_sw):\n"
        "    return g + rho_0 + c_sw\n"
    )
    assert _violations_in_source(src, "<syn>") == []
