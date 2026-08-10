"""Coverage gate for NEMO's SHARED MUTABLE STATE (oracle-fidelity Rule 1,
applied to module-scope `SAVE` data rather than to the mesh/restart/namelist).

WHY THIS EXISTS
---------------
NEMO is a shared-memory Fortran program.  ``sbc_oce.F90:107`` declares::

    REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   utau

``PUBLIC`` + ``SAVE`` is a global mutable array any routine that does
``USE sbc_oce`` can read or write, persisting across calls AND across
timesteps.  Producer and consumer can sit hundreds of lines apart:
``CALL sbc( kstp, Nbb, Nnn )`` (``stpmlf.F90:170``) deposits ``utau``/``qns``/
``sbc_tsc`` into module memory and returns NOTHING; ``tra_sbc``
(``stpmlf.F90:387``) picks them up 217 lines later and averages
``0.5*(sbc_tsc_b + sbc_tsc)`` at ``trasbc.F90:152``.

legoESM is functional: ``step(state: LatLonCGridOceanState, ...) -> state``,
a pure pytree in and out, because ``jax.jit``/``jax.grad`` cannot trace a
``SAVE`` array.  So **every implicit coupling in NEMO must become an explicit
argument in ours, and any dropped link silently takes a default with no
error.**  That is the structural signature of this translation and the class
of four separate defects found in one week (``physics.constants.g``,
``een_q_boundary``, ``c_p``->``c_sw``, ``n2_tracers_before``).

``test_recipe_option_threading.py`` covers CONFIG threading; the geometry gate
(``scripts/validate/ocean_fidelity/dino_1226/nemo_geometry_gate.py``) covers
the MESH.  Nothing covered SHARED STATE.  This does.

WHAT IS ENUMERATED (the scope; see ``EXCLUDED_MODULES`` for what is not)
-----------------------------------------------------------------------
1. The LIVE ``stp_MLF`` call list is taken from
   ``scripts/validate/ocean_fidelity/dino_1226/gen_step_wiring.py`` -- its
   ``parse_stpmlf`` already resolves cpp branches against ``cpp_DINO.fcm`` and
   ``IF( ln_* )`` guards against the RESOLVED namelist in
   ``RUN_GDB/ocean.output``.  It is REUSED, not rebuilt.
2. Each LIVE call is resolved to the file that defines it (DINO ``MY_SRC``
   wins over ``src/OCE``, which wins over ``src/SAS``|``src/SWE``|``src/OFF``),
   plus the concrete dispatch targets ``stpmlf_call_coverage.py`` already
   resolved one level (``zdf_phy -> zdf_tke``, ...).  That set of files is the
   LIVE CHAIN.
3. For every live-chain file: the module-scope, non-``PARAMETER``,
   array-or-``SAVE`` declarations of (a) every module it ``USE``s -- ``PUBLIC``
   symbols only, and a bare ``USE mod`` with no ``ONLY`` imports ALL of them,
   which is how ``utau`` reaches ``sbcmod`` -- and (b) its OWN module,
   including module-PRIVATE ``SAVE`` state (this is how ``zdftke::dissl``, a
   private ``SAVE`` array that carries between timesteps, is caught at all).

FOUR DISPOSITIONS, NO FIFTH
---------------------------
``IN_STATE`` / ``THREADED`` / ``INERT`` / ``WAIVED``.  Anything unaccounted is
a hard failure.  ``INERT`` is not a free-text claim: each carries machine
EVIDENCE (a namelist switch value from ``ocean.output``, a cpp key from
``cpp_DINO.fcm``, or an exact printed line) which this module re-checks and
FAILS CLOSED on if the evidence cannot be found.
"""
from __future__ import annotations

import collections
import importlib.util
import os
import re
import sys

import pytest

from tests.ocean._nemo_shared_state_baseline import (
    DISPOSITIONS,
    EXCLUDED_MODULES,
    IN_STATE,
    INERT,
    MODULE_DISPOSITION,
    SYMBOL_DISPOSITION,
    THREADED,
    UNREFERENCED_REASON,
    WAIVED,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GEN_DIR = os.path.join(REPO, "scripts/validate/ocean_fidelity/dino_1226")
GEN_PY = os.path.join(GEN_DIR, "gen_step_wiring.py")


# ---------------------------------------------------------------------------
# Oracle access.  Root is env-overridable and every oracle-reading test SKIPs
# cleanly without the checkout (same policy as test_step_wiring_generated.py).
# ---------------------------------------------------------------------------
def _load_gen():
    spec = importlib.util.spec_from_file_location("gen_step_wiring_sss", GEN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    if not os.path.exists(GEN_PY):
        pytest.skip(f"generator missing: {GEN_PY}")
    return _load_gen()


@pytest.fixture(scope="module")
def oracle(gen):
    for path in (gen.STPMLF, gen.CPP_FCM, gen.OCEAN_OUTPUT):
        if not os.path.exists(path):
            pytest.skip(f"NEMO oracle checkout not present: {path}")
    if not os.path.isdir(nemo_root(gen)):
        pytest.skip(f"NEMO source tree not present: {nemo_root(gen)}")
    return gen


def nemo_root(gen) -> str:
    """``<nemo>`` -- ``gen.ORACLE`` is ``<nemo>/cfgs/DINO``."""
    return os.path.abspath(os.path.join(gen.ORACLE, "..", ".."))


# ---------------------------------------------------------------------------
# Fortran parsing.  Deliberately small and fail-loud: every helper below is
# exercised by an oracle-INDEPENDENT unit test at the bottom of this file, so
# a machine without the checkout still covers the logic that decides what the
# enumeration contains.
# ---------------------------------------------------------------------------
_DEF_RE = re.compile(
    r"^\s*(?:RECURSIVE\s+)?(?:SUBROUTINE|(?:[\w()]+\s+)?FUNCTION|INTERFACE)\s+(\w+)",
    re.I)
_MODULE_RE = re.compile(r"^\s*MODULE\s+(\w+)\s*$", re.I)
_USE_RE = re.compile(r"^\s*USE\s+(\w+)\s*(.*)$", re.I)
_ONLY_RE = re.compile(r"^\s*,?\s*ONLY\s*:", re.I)
_TYPE_RE = re.compile(
    r"^\s*(REAL|INTEGER|LOGICAL|CHARACTER|COMPLEX|DOUBLE PRECISION|TYPE)\b", re.I)
# "state" = something that can hold a value between calls: an array, or an
# explicitly SAVEd scalar.  A bare `LOGICAL, PUBLIC :: ln_x` is CONFIG, covered
# by test_recipe_option_threading.py, and is excluded (see module docstring).
_STATE_ATTR_RE = re.compile(r"\b(SAVE|ALLOCATABLE|DIMENSION)\b")

_ALL_IMPORT = "*"       # sentinel for `USE mod` with no ONLY clause


def strip_comment(line: str) -> str:
    """Drop a trailing Fortran ``!`` comment, respecting quoted strings."""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` at bracket depth 0 (``DIMENSION(:,:)`` stays whole)."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def find_double_colon(text: str) -> int:
    """Index of the ``::`` that separates a declaration's attributes from its
    names, at bracket depth 0.  A naive ``[^:]*`` split fails on
    ``DIMENSION(:,:)``, which is exactly how the first draft of this parser
    silently enumerated ZERO of ``sbc_oce``'s arrays -- i.e. it would have
    passed while walking nothing."""
    depth = 0
    for i in range(len(text) - 1):
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == ":" and text[i + 1] == ":" and depth == 0:
            return i
    return -1


def parse_module_state(text: str, *, public_only: bool) -> dict[str, str]:
    """``{name: attribute-string}`` for module-scope state declarations.

    Only the specification part (everything before the module's ``CONTAINS``)
    is scanned, so routine locals can never be mistaken for shared state.
    ``PARAMETER`` is excluded: a compile-time constant cannot carry state.
    """
    out: dict[str, str] = {}
    buf = ""
    for raw in text.split("\n"):
        s = strip_comment(raw).strip()
        if not s or s.startswith("#"):
            continue
        if re.fullmatch(r"CONTAINS", s, re.I):
            break
        # A leading `&` continues the PREVIOUS logical line; leaving it in
        # makes the first name of the continuation unparseable (`& split_b`),
        # which silently DROPS declarations rather than erroring.
        s = s.lstrip("&").strip()
        if s.endswith("&"):
            buf += s[:-1]
            continue
        line, buf = buf + s, ""
        if not _TYPE_RE.match(line):
            continue
        idx = find_double_colon(line)
        if idx < 0:
            continue
        attrs, names = line[:idx].upper(), line[idx + 2:]
        if re.search(r"\bPARAMETER\b", attrs):
            continue
        if public_only and "PUBLIC" not in attrs:
            continue
        if not _STATE_ATTR_RE.search(attrs):
            continue
        for chunk in split_top_level(names):
            name = re.split(r"[=(]", chunk.strip())[0].strip()
            if re.fullmatch(r"\w+", name or ""):
                out[name] = " ".join(line[:idx].split())
    return out


def parse_uses(text: str) -> dict[str, set[str]]:
    """``{module: {symbol, ...}}``; ``{"*"}`` for a bare ``USE mod`` (which
    imports every PUBLIC symbol -- how ``utau`` reaches ``sbcmod.F90``)."""
    out: dict[str, set[str]] = collections.defaultdict(set)
    pending = None
    for raw in text.split("\n"):
        s = strip_comment(raw).strip()
        if not s:
            continue
        if pending is not None:
            cont = s.endswith("&")
            for n in s.strip("&").split(","):
                n = n.strip()
                if re.fullmatch(r"\w+", n):
                    out[pending].add(n)
            if not cont:
                pending = None
            continue
        m = _USE_RE.match(s)
        if not m:
            continue
        mod, body = m.group(1).lower(), m.group(2)
        if not _ONLY_RE.match(body):
            out[mod].add(_ALL_IMPORT)
            continue
        body = _ONLY_RE.sub("", body).rstrip()
        cont = body.endswith("&")
        for n in body.rstrip("&").split(","):
            n = n.strip()
            if re.fullmatch(r"\w+", n):
                out[mod].add(n)
        if cont:
            pending = mod
    return out


# ---------------------------------------------------------------------------
# Enumeration.
# ---------------------------------------------------------------------------
def _source_rank(path: str) -> int:
    """DINO ``MY_SRC`` overrides ``src/OCE``, which is the OCE build; the
    ``SAS``/``SWE``/``OFF`` sibling programs are not what DINO compiles."""
    if "MY_SRC" in path:
        return 0
    if "/src/OCE/" in path:
        return 1
    return 2


def _fortran_files(gen) -> list[str]:
    nemo = nemo_root(gen)
    out = []
    for root, _dirs, files in os.walk(os.path.join(nemo, "src")):
        out += [os.path.join(root, f) for f in files if f.endswith(".F90")]
    my_src = os.path.join(gen.ORACLE, "MY_SRC")
    out += [os.path.join(my_src, f) for f in sorted(os.listdir(my_src))
            if f.endswith(".F90")]
    return sorted(out)


class Enumeration:
    """The joined result: shared symbols, the live-chain files, and which of
    those files reference each symbol by name."""

    def __init__(self, symbols, chain_files, referenced, routines):
        self.symbols = symbols          # {(module, name): attrs}
        self.chain_files = chain_files  # {basename, ...}
        self.referenced = referenced    # {(module, name): [basename, ...]}
        self.routines = routines        # {routine: file}


def build_enumeration(gen) -> Enumeration:
    files = _fortran_files(gen)
    texts = {p: open(p, errors="replace").read() for p in files}

    defs: dict[str, list[str]] = collections.defaultdict(list)
    module_of: dict[str, str] = {}
    for path, text in texts.items():
        for line in text.split("\n"):
            m = _DEF_RE.match(line)
            if m:
                defs[m.group(1).lower()].append(path)
            mm = _MODULE_RE.match(strip_comment(line))
            if mm and path not in module_of:
                module_of[path] = mm.group(1).lower()

    # PUBLIC state index, per module.
    public_index: dict[str, dict[str, str]] = {}
    for path, text in texts.items():
        mod = module_of.get(path)
        if not mod:
            continue
        decls = parse_module_state(text, public_only=True)
        if decls:
            public_index.setdefault(mod, {}).update(decls)

    # 1. LIVE stp_MLF calls (REUSED parse) + 2. one-level dispatch descent.
    calls = gen.parse_stpmlf(gen.STPMLF)
    live_names = {c.name for c in calls if c.live == "LIVE"}
    live_lines = {c.line for c in calls if c.live == "LIVE"}
    sys.path.insert(0, GEN_DIR)
    import stpmlf_call_coverage as cov  # noqa: E402  (oracle-side join source)

    for entry in cov.CALLS:
        if entry.line in live_lines:
            tail = entry.routine.split("->")[-1]
            live_names |= set(re.findall(r"[A-Za-z_]\w*", tail))

    routines: dict[str, str] = {}
    for name in sorted(live_names):
        candidates = sorted(set(defs.get(name.lower(), [])), key=_source_rank)
        if candidates:
            routines[name] = candidates[0]

    chain_paths = sorted(set(routines.values()))

    # 3. Shared symbols reachable from each live-chain file.
    symbols: dict[tuple[str, str], str] = {}
    for path in chain_paths:
        own = module_of.get(path, "")
        for mod, syms in parse_uses(texts[path]).items():
            table = public_index.get(mod, {})
            names = set(table) if _ALL_IMPORT in syms else (syms & set(table))
            for name in names:
                symbols[(mod, name)] = table[name]
        # The file's OWN module, including PRIVATE SAVE state: this is the
        # only way `zdftke::dissl` (module-private, carries between steps) is
        # visible at all.
        if own:
            for name, attrs in parse_module_state(texts[path],
                                                  public_only=False).items():
                symbols.setdefault((own, name), attrs)

    symbols = {k: v for k, v in symbols.items() if k[0] not in EXCLUDED_MODULES}

    # Reference join: which live-chain files name the symbol in CODE.  A hit
    # is not proof of a runtime read; a MISS is proof there is no reference,
    # which is the only direction a waiver is allowed to lean on.
    codes = {os.path.basename(p):
             "\n".join(strip_comment(x) for x in texts[p].split("\n"))
             for p in chain_paths}
    referenced: dict[tuple[str, str], list[str]] = {}
    for key in symbols:
        pat = re.compile(rf"\b{re.escape(key[1])}\b")
        hits = sorted(b for b, c in codes.items() if pat.search(c))
        if hits:
            referenced[key] = hits

    return Enumeration(symbols, set(codes), referenced, routines)


@pytest.fixture(scope="module")
def enum(oracle):
    return build_enumeration(oracle)


# ---------------------------------------------------------------------------
# Evidence resolution for INERT.
# ---------------------------------------------------------------------------
def _read_switches(gen) -> dict[str, object]:
    """Resolved namelist values NEMO PRINTED for the run (``ln_*``/``nn_*``/
    ``rn_*``).  The namelist FILES do not show what unset entries defaulted
    to, which is why this reads the output."""
    out: dict[str, object] = {}
    for line in open(gen.OCEAN_OUTPUT, errors="replace"):
        for name, val in re.findall(r"\b(ln_\w+)\s*=\s*([TF])\b", line):
            out[name] = (val == "T")
        for name, val in re.findall(
                r"\b((?:nn|rn)_\w+)\s*=\s*([-+0-9.eEdD]+)", line):
            try:
                out[name] = float(val.replace("D", "E").replace("d", "e"))
            except ValueError:
                pass
    return out


def resolve_evidence(gen, evidence) -> tuple[bool, str]:
    """``(holds, explanation)`` for one INERT evidence tuple.  Unknown kinds
    and unfindable switches return ``False`` -- the gate fails CLOSED rather
    than accepting an unverifiable INERT claim."""
    kind = evidence[0]
    if kind == "switch":
        _k, name, want = evidence
        got = _read_switches(gen).get(name)
        if got is None:
            return False, f"{name} is not printed in ocean.output"
        return bool(got) == bool(want), f"{name}={got} (ocean.output)"
    if kind == "value":
        _k, name, want = evidence
        got = _read_switches(gen).get(name)
        if got is None:
            return False, f"{name} is not printed in ocean.output"
        return float(got) == float(want), f"{name}={got} (ocean.output)"
    if kind == "cpp":
        _k, key, want = evidence
        present = key in gen.read_cpp_keys(gen.CPP_FCM)
        return present == bool(want), f"{key} {'in' if present else 'absent from'} cpp_DINO.fcm"
    if kind == "text":
        _k, needle = evidence
        found = needle in open(gen.OCEAN_OUTPUT, errors="replace").read()
        return found, f"{needle!r} {'found' if found else 'NOT found'} in ocean.output"
    return False, f"unknown evidence kind {kind!r}"


def dispose(key: tuple[str, str], enum: Enumeration):
    """``(disposition, reason, evidence|None)`` for one symbol, or ``None`` if
    nothing accounts for it (which is the hard failure)."""
    if key in SYMBOL_DISPOSITION:
        entry = SYMBOL_DISPOSITION[key]
        return entry[0], entry[1], (entry[2] if len(entry) > 2 else None)
    if key not in enum.referenced:
        return WAIVED, UNREFERENCED_REASON, None
    if key[0] in MODULE_DISPOSITION:
        entry = MODULE_DISPOSITION[key[0]]
        return entry[0], entry[1], (entry[2] if len(entry) > 2 else None)
    return None


# ---------------------------------------------------------------------------
# Anti-vacuity: the enumeration must actually walk something.  Every threshold
# below is a floor a broken parser falls through, not a pinned exact count.
# ---------------------------------------------------------------------------
_MIN_SYMBOLS = 300
_MIN_CHAIN_FILES = 20
_MIN_REFERENCED = 150


def test_enumeration_is_non_trivial(enum):
    assert len(enum.symbols) >= _MIN_SYMBOLS, (
        f"only {len(enum.symbols)} shared symbols enumerated (expected >= "
        f"{_MIN_SYMBOLS}) -- the Fortran parse is broken and every check "
        "below would be vacuous")
    assert len(enum.chain_files) >= _MIN_CHAIN_FILES, len(enum.chain_files)
    assert len(enum.referenced) >= _MIN_REFERENCED, len(enum.referenced)
    # The motivating declaration itself must be in the enumeration, reached
    # through the bare `USE sbc_oce` in sbcmod.F90 (no ONLY clause).
    assert ("sbc_oce", "utau") in enum.symbols
    assert "sbcmod.F90" in enum.referenced[("sbc_oce", "utau")]
    # ...and the module-PRIVATE SAVE case, which a PUBLIC-only scan misses.
    assert ("zdftke", "dissl") in enum.symbols
    # Config scalars are NOT state and must not be enumerated.
    assert ("sbc_oce", "ln_blk") not in enum.symbols


def test_every_live_call_resolves_to_a_file(oracle, enum):
    calls = oracle.parse_stpmlf(oracle.STPMLF)
    live = sorted({c.name for c in calls if c.live == "LIVE"})
    missing = [r for r in live if r not in enum.routines]
    assert not missing, (
        "LIVE stp_MLF CALL(s) whose defining file could not be located -- the "
        f"enumeration is missing their shared state entirely: {missing}")


# ---------------------------------------------------------------------------
# THE GATE.
# ---------------------------------------------------------------------------
def test_every_shared_symbol_has_a_disposition(enum):
    unaccounted = sorted(k for k in enum.symbols if dispose(k, enum) is None)
    assert not unaccounted, (
        f"{len(unaccounted)} NEMO shared-state symbol(s) reachable from a LIVE "
        "stp_MLF call have NO disposition.  Each must be IN_STATE (name the "
        "LatLonCGridOceanState field), THREADED (name the call site), INERT "
        "(name the switch and its resolved value) or WAIVED (written reason) "
        "in tests/ocean/_nemo_shared_state_baseline.py.  Anything else is a "
        "hard failure -- do NOT baseline a real gap into silence:\n  "
        + "\n  ".join(f"{m}::{s}" for m, s in unaccounted))


def test_dispositions_use_only_the_four_kinds(enum):
    bad = sorted((k, dispose(k, enum)[0]) for k in enum.symbols
                 if dispose(k, enum)[0] not in DISPOSITIONS)
    assert not bad, f"disposition outside {sorted(DISPOSITIONS)}: {bad}"


def test_every_disposition_has_a_reason(enum):
    empty = sorted(k for k in enum.symbols
                   if not (dispose(k, enum)[1] or "").strip())
    assert not empty, f"disposition with an empty reason: {empty}"


def test_inert_evidence_holds(oracle, enum):
    """``INERT`` is a claim about the ORACLE's resolved configuration, so it is
    re-checked against ``ocean.output``/``cpp_DINO.fcm`` rather than trusted.
    Fails CLOSED: an evidence tuple that cannot be resolved is a failure, not
    a pass."""
    failures = []
    for key in sorted(enum.symbols):
        kind, reason, evidence = dispose(key, enum)
        if kind != INERT:
            continue
        if evidence is None:
            failures.append(f"{key[0]}::{key[1]}: INERT with no evidence tuple")
            continue
        holds, why = resolve_evidence(oracle, evidence)
        if not holds:
            failures.append(f"{key[0]}::{key[1]}: {reason} -- but {why}")
    assert not failures, (
        "INERT claim(s) the oracle's own resolved configuration does not "
        "support:\n  " + "\n  ".join(failures))


def test_baseline_only_shrinks(enum):
    """A baseline entry that no longer matches an enumerated symbol must be
    deleted, so the table can never rot into dead slack that hides a new gap
    (same rule as ``_recipe_threading_baseline``)."""
    stale_sym = sorted(k for k in SYMBOL_DISPOSITION if k not in enum.symbols)
    modules = {m for m, _s in enum.symbols}
    stale_mod = sorted(set(MODULE_DISPOSITION) - modules)
    assert not stale_sym and not stale_mod, (
        "stale entries in tests/ocean/_nemo_shared_state_baseline.py -- remove "
        f"them.\n  symbols: {stale_sym}\n  modules: {stale_mod}")


def test_excluded_modules_are_real_and_declared(enum, oracle):
    """A scope exclusion must name a module the enumeration WOULD otherwise
    have produced -- otherwise the carve-out is silently hiding nothing (or,
    worse, was a typo that hid something else)."""
    # Rebuild without the exclusion filter to prove each excluded module is
    # genuinely reachable.
    saved = dict(EXCLUDED_MODULES)
    try:
        EXCLUDED_MODULES.clear()
        full = build_enumeration(oracle)
    finally:
        EXCLUDED_MODULES.update(saved)
    modules = {m for m, _s in full.symbols}
    missing = sorted(set(EXCLUDED_MODULES) - modules)
    assert not missing, (
        f"EXCLUDED_MODULES names module(s) the enumeration never produces: "
        f"{missing} -- a scope carve-out that excludes nothing is dead slack")


def test_seeded_carries_are_all_explicitly_dispositioned(enum):
    """Every symbol on the seed list of "known-missing carries" must be in the
    enumeration AND carry an EXPLICIT per-symbol disposition -- never the
    automatic unreferenced waiver, and never a module-wide rule.  Four of them
    turn out NOT to be gaps (``qns_b``/``sfx_b`` are written and never read;
    ``emp_b`` is INERT because DINO's emp is identically zero; ``fraqsr_1lev``
    has no live consumer without key_top), and that verdict has to be written
    down per symbol, not inherited."""
    seeds = [("sbc_oce", n) for n in
             ("utau_b", "qns_b", "sbc_tsc_b", "qsr_hc_b", "emp_b", "sfx_b")]
    seeds += [("zdf_oce", "avm_k"), ("zdf_oce", "avt_k"),
              ("zdftke", "dissl"), ("oce", "fraqsr_1lev")]
    for key in seeds:
        assert key in enum.symbols, f"{key} vanished from the enumeration"
        assert key in SYMBOL_DISPOSITION, (
            f"{key[0]}::{key[1]} has no EXPLICIT disposition -- a seeded "
            "known-missing carry may not fall through to a module rule or to "
            "the unreferenced waiver")


def test_the_known_missing_carries_read_as_findings(enum):
    """Seed check (oracle-fidelity Rule 1: do not baseline a real gap into
    silence).  Every NEMO cross-step carry legoESM is known NOT to reproduce
    must be classified IN_STATE (present) or explicitly not-INERT/not-WAIVED
    with a reason that says so -- never quietly waived.

    Cited, not re-derived: injecting NEMO's own ``avt_k``/``avm_k`` leaves
    98.9% of the per-step divergence, and the ``sbc`` carry is bounded offline
    at 2.9e-5 K/step against 0.152 -- so these are REAL GAPS that are NOT the
    current driver.  That does not make them waivable.
    """
    must_be_findings = [
        ("sbc_oce", "sbc_tsc_b"),
        ("sbc_oce", "qsr_hc_b"),
        ("zdf_oce", "avm_k"),
        ("zdf_oce", "avt_k"),
        ("zdftke", "dissl"),
    ]
    for key in must_be_findings:
        assert key in enum.symbols, f"{key} vanished from the enumeration"
        kind, reason, _ev = dispose(key, enum)
        assert kind == IN_STATE, (
            f"{key[0]}::{key[1]} is disposed {kind}, not {IN_STATE} -- a known "
            "missing cross-step carry must stay visible as a state gap, not "
            f"be waived away.  reason: {reason}")
        assert "MISSING" in reason, (
            f"{key[0]}::{key[1]} is IN_STATE but its reason does not mark the "
            f"gap: {reason}")


def test_tally(enum, capsys):
    """Reports the split; asserts only that the enumeration is not degenerate.
    The NUMBERS are the deliverable, not a pass/fail verdict."""
    counts = collections.Counter(dispose(k, enum)[0] for k in enum.symbols)
    with capsys.disabled():
        print(f"\n  NEMO shared-state coverage: {len(enum.symbols)} symbols "
              f"over {len(enum.chain_files)} live-chain files")
        for kind in DISPOSITIONS:
            print(f"    {kind:9s} {counts.get(kind, 0):4d}")
        print(f"    (referenced by a live-chain file: {len(enum.referenced)}; "
              f"excluded modules: {sorted(EXCLUDED_MODULES)})")
    assert sum(counts.values()) == len(enum.symbols)
    for kind in (IN_STATE, THREADED, INERT, WAIVED):
        assert counts.get(kind, 0) > 0, f"no symbol disposed {kind}"


# ---------------------------------------------------------------------------
# NON-VACUITY SELF-TESTS.  A gate that can pass by walking nothing is worse
# than none; six tests written this week passed with their feature removed.
# ---------------------------------------------------------------------------
def test_gate_catches_synthetic_violation(enum):
    """Remove a symbol's account and the gate must go RED.

    Two independent removals, because the gate has two accounting paths:
      (a) an explicitly-dispositioned symbol loses its SYMBOL_DISPOSITION
          entry AND is referenced, so nothing accounts for it;
      (b) an INERT module rule loses its evidence.
    """
    victim = ("sbc_oce", "utau")
    assert victim in enum.symbols
    assert dispose(victim, enum) is not None, "setup: victim must be accounted"

    saved = SYMBOL_DISPOSITION.pop(victim)
    try:
        assert victim in enum.referenced, (
            "setup invariant: the victim must be REFERENCED, or the "
            "unreferenced-waiver path would account for it and this self-test "
            "would prove nothing")
        assert dispose(victim, enum) is None, (
            "removing the disposition left the symbol accounted -- the gate is "
            "VACUOUS")
        with pytest.raises(AssertionError, match="NO disposition"):
            test_every_shared_symbol_has_a_disposition(enum)
    finally:
        SYMBOL_DISPOSITION[victim] = saved
    # restored
    test_every_shared_symbol_has_a_disposition(enum)


def test_gate_catches_a_symbol_dropped_from_the_enumeration(enum):
    """The other direction: if the ENUMERATION stops producing a symbol, the
    shrink-only check must flag its now-stale baseline entry.  Without this,
    a parser regression that quietly enumerates fewer symbols would make the
    whole gate weaker while still passing."""
    shrunk = Enumeration(
        {k: v for k, v in enum.symbols.items() if k != ("zdf_oce", "avt_k")},
        enum.chain_files, enum.referenced, enum.routines)
    with pytest.raises(AssertionError, match="stale entries"):
        test_baseline_only_shrinks(shrunk)
    test_baseline_only_shrinks(enum)   # restored


def test_gate_catches_a_false_inert_claim(oracle, enum):
    """An INERT claim whose switch does NOT resolve as stated must fail."""
    holds, why = resolve_evidence(oracle, ("switch", "ln_zdftke", False))
    assert not holds, f"setup: ln_zdftke is True for DINO, got {why}"
    holds, _why = resolve_evidence(oracle, ("switch", "ln_zdftke", True))
    assert holds
    # An unresolvable switch fails CLOSED rather than passing.
    holds, why = resolve_evidence(oracle, ("switch", "ln_not_a_switch", False))
    assert not holds and "not printed" in why


def test_unreferenced_waiver_is_evidence_backed(enum):
    """The bulk waiver leans on ABSENCE of any textual reference, which is the
    only sound direction.  Prove the reference join is real by checking a
    symbol that IS referenced is not swept into it."""
    swept = [k for k in enum.symbols
             if k in enum.referenced and dispose(k, enum)[1] == UNREFERENCED_REASON]
    assert not swept, (
        f"referenced symbol(s) waived as unreferenced: {swept[:5]}")
    unref = [k for k in enum.symbols if k not in enum.referenced]
    assert len(unref) > 20, (
        f"only {len(unref)} unreferenced symbols -- the reference join looks "
        "broken (it should account for the whole-module `USE` tail)")


# ---------------------------------------------------------------------------
# Oracle-INDEPENDENT unit tests for the parser (run everywhere).
# ---------------------------------------------------------------------------
def test_find_double_colon_survives_dimension_colons():
    line = "   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   utau"
    idx = find_double_colon(line)
    assert idx > 0 and line[idx:idx + 2] == "::"
    assert line[idx + 2:].strip() == "utau"


def test_parse_module_state_shapes():
    src = "\n".join([
        "MODULE fake",
        "   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   utau",
        "   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   qns, qns_b  !: two",
        "   REAL(wp),         ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: dissl  ! private",
        "   LOGICAL , PUBLIC ::   ln_blk       !: config scalar, NOT state",
        "   INTEGER , PUBLIC, PARAMETER ::   jp_usr = 1",
        "   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   split_a, &",
        "      &   split_b",
        "CONTAINS",
        "   SUBROUTINE fake_sub",
        "      REAL(wp), DIMENSION(10) :: local_not_shared",
        "   END SUBROUTINE fake_sub",
        "END MODULE fake",
    ])
    pub = parse_module_state(src, public_only=True)
    assert set(pub) == {"utau", "qns", "qns_b", "split_a", "split_b"}, pub
    allst = parse_module_state(src, public_only=False)
    assert "dissl" in allst and "local_not_shared" not in allst
    assert "ln_blk" not in allst and "jp_usr" not in allst


def test_parse_uses_whole_module_and_only():
    src = "\n".join([
        "   USE sbc_oce        ! surface boundary condition: ocean",
        "   USE zdf_oce , ONLY : avm, avt, &",
        "      &                 en",
        "   USE dom_oce, ONLY: tmask",
    ])
    uses = parse_uses(src)
    assert uses["sbc_oce"] == {_ALL_IMPORT}
    assert uses["zdf_oce"] == {"avm", "avt", "en"}
    assert uses["dom_oce"] == {"tmask"}


def test_source_rank_prefers_my_src_then_oce():
    assert _source_rank("/x/cfgs/DINO/MY_SRC/dynzdf.F90") == 0
    assert _source_rank("/x/src/OCE/DYN/dynzdf.F90") == 1
    assert _source_rank("/x/src/SWE/stpmlf.F90") == 2


def test_split_top_level_keeps_bracketed_commas():
    assert split_top_level("a, b(1,2), c") == ["a", " b(1,2)", " c"]


def test_baseline_tables_are_well_formed():
    """Runs without the oracle: every table entry must be shaped correctly and
    every INERT entry must carry an evidence tuple (its VALUE is checked
    against the oracle by ``test_inert_evidence_holds``)."""
    for table, label in ((SYMBOL_DISPOSITION, "SYMBOL_DISPOSITION"),
                         (MODULE_DISPOSITION, "MODULE_DISPOSITION")):
        for key, entry in table.items():
            assert entry[0] in DISPOSITIONS, f"{label}[{key}]: {entry[0]}"
            assert entry[1].strip(), f"{label}[{key}]: empty reason"
            if entry[0] == INERT:
                assert len(entry) == 3 and isinstance(entry[2], tuple), (
                    f"{label}[{key}]: INERT needs an evidence tuple")
                assert entry[2][0] in ("switch", "value", "cpp", "text"), entry[2]
    assert DISPOSITIONS == frozenset({IN_STATE, THREADED, INERT, WAIVED})
