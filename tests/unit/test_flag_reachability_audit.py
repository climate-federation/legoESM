"""Resolver-family flag reachability — the boolean/scalar sibling of #691.

``test_params_reachability_audit`` proves a ``__param_spec__`` FLOAT is
settable from a calibration file.  It says nothing about whether the value,
once set, still reaches the scheme config the kernel is built from.  That
second gap has now bitten this campaign FOUR times, each time silently:

1. ``convection_config_for`` (2026-07-23) — the MPAS/spectral lanes built
   ``ConvectionConfig(scheme=...)`` from the scheme STRING, so every tuned
   ``bechtold_*`` / ``sbm_*`` scalar was dropped on the floor.
2. ``gwd_config_for`` (2026-07-24) — same shape: every production AMIP lane
   rebuilt ``GravityWaveDragConfig(scheme=...)`` bare, so ``mcfarlane_k_wave``
   / ``hines_total_rms_wind`` never reached any kernel outside AIMIP training.
3. ``homogeneous_ice_nucleation`` (2026-07-25) — the read existed in
   ``_resolve_microphysics`` but was NESTED inside a hard-saturation-override
   guard whose two scalars both default to ``None``, so the flag was inert on
   the FV lane while every unit test on the applier passed.
4. ``convective_buoyancy_death_memory`` (2026-07-25, found by the adversarial
   review OF this audit) — the leaf ``BechtoldConfig.buoyancy_death_memory``
   exists and is consumed by ``bechtold.py``, but no pipeline code maps the
   ExperimentConfig scalar onto it.  Classified in ``DANGLING_FIELDS`` below.

Case 3 is the one a static "is it wired?" check cannot catch: the field WAS
read, from the right function, with the right name.  So this audit is
behavioural.  For every ``ExperimentConfig`` field a resolver reads (derived
from its AST, not a hand list — a hand list rots), perturbing that field must
CHANGE the resolver's returned config in at least one production-plausible
host config.  A read that cannot change the output is either a validation-only
read or dead wiring, and must be named in ``INERT_READS`` with a reason.

The baselines are EXACT-match, so both directions go red: a newly dead flag,
and a baselined flag that came alive without the baseline shrinking.  (The
revival direction only has teeth once a baseline is non-empty; ``INERT_READS``
is empty today, which makes the dead-flag direction maximally strict — ANY
inert read is a failure.)

SCOPE AND LIMITS (stated so the gate is not over-read):

* It proves the resolver's RETURNED config differs, not that the kernel
  consumes the changed leaf.  A kernel that ignores its own config field is
  out of scope (that is the per-scheme tendency tests' job).
* Only the four resolvers in ``_RESOLVERS`` are audited.  Radiation and other
  pipeline builders are deliberately out of scope; add them here when they
  grow a tuned-leaf overlay.
* Host semantics are EXISTENTIAL ("live under at least one host"), matching
  the ``--params`` audit.  A field live on one lane and dropped on another is
  not detected.
* Every probe config — base AND perturbed — must pass ``validate_strict``, so
  the audit can never certify a field "live" through a configuration no
  production run could legally use.
"""
from __future__ import annotations

import ast
import functools
import inspect
import pathlib
import typing

import numpy as np
import pytest
from jax import tree_util
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.convection.config import BechtoldConfig, SBMConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
from legoesm.atmosphere.physics.microphysics.morrison import MorrisonConfig
from legoesm.driver import physics_pipeline as pp
from legoesm.driver.config import ExperimentConfig

from tests import _ratchet_audit

# Resolvers whose job is to hand a tuned leaf config to a kernel builder.
# Each entry: name -> (callable, [host ExperimentConfig kwargs to try]).
# A field counts as REACHABLE if it changes the output under ANY host — the
# same "reachable in at least one production configuration" semantics the
# --params audit uses.  Every host must itself pass validate_strict.
_RESOLVERS = {
    "turbulence_config_for": (
        lambda c: pp.turbulence_config_for(c),
        [{"turbulence": "louis"}, {"turbulence": "tke"},
         {"turbulence": "holtslag_boville"}],
    ),
    "convection_config_for": (
        lambda c: pp.convection_config_for(c),
        [{"convection": "bechtold"}, {"convection": "sbm"},
         {"convection": "tiedtke"}, {"convection": "zhang_mcfarlane"}],
    ),
    "gwd_config_for": (
        lambda c: pp.gwd_config_for(c),
        [{"gravity_wave_drag": "mcfarlane+hines"},
         {"gravity_wave_drag": "mcfarlane"}, {"gravity_wave_drag": "hines"}],
    ),
    "_resolve_microphysics": (
        lambda c: pp._resolve_microphysics(c)[1],   # leaf only; fn is a closure
        # The hard-sat overrides are GATED: validate_strict refuses a threshold
        # without ``hard_saturation_adjustment=True`` (it would be silently
        # inert), so the gate-on host is what makes those two scalars legally
        # probeable at all.
        [{"microphysics": "morrison"},
         {"microphysics": "morrison", "hard_saturation_adjustment": True},
         # Same gate shape for the hom-nucleation number: validate_strict
         # refuses morrison_hom_ice_nuc_N without the hom flag (silently-
         # inert guard), so the flag-on host makes it legally probeable.
         {"microphysics": "morrison", "homogeneous_ice_nucleation": True},
         {"microphysics": "thompson"},
         {"microphysics": "kessler"}, {"microphysics": "p3"}],
    ),
}

# Probe outcomes.  Kept distinct so a CRASH can never masquerade as "inert":
# an inert verdict is only ever awarded to a field that was ACTUALLY evaluated,
# on a legal config, and left the resolver output unchanged.
_LIVE = "live"                  # perturbation changed the returned config
_INERT = "inert"                # evaluated on a legal config, output unchanged
_UNPROBEABLE = "unprobeable"    # no legal perturbation exists — nothing proven

# Reads that were evaluated and provably cannot change the resolver output.
# SHRINK-ONLY: adding an entry means consciously accepting dead wiring in
# review.  A field listed here that becomes live goes red until it is removed.
INERT_READS: dict[tuple[str, str], str] = {}

# Reads for which NO legal perturbation exists, so the audit proves NOTHING
# about them.  Deliberately a SEPARATE baseline from INERT_READS: conflating
# "proved dead" with "never tested" is how a vacuous gate gets built.
# SHRINK-ONLY.
UNPROBEABLE_READS = {
    # The scheme selector itself is the host-config axis, not a tunable: it
    # picks WHICH leaf is returned, and the audit holds it fixed per host.
    ("turbulence_config_for", "turbulence"): "scheme selector (host axis)",
    ("convection_config_for", "convection"): "scheme selector (host axis)",
    ("gwd_config_for", "gravity_wave_drag"): "scheme selector (host axis)",
    ("_resolve_microphysics", "microphysics"): "scheme selector (host axis)",
    # Full-config injection slots: typed ``Any``, so there is no scalar to
    # perturb — exercising them means substituting a whole foreign config,
    # which the dedicated override tests cover (validate_strict enforces
    # scheme agreement).
    ("turbulence_config_for", "turbulence_override"): "full-config override slot",
    ("gwd_config_for", "gravity_wave_drag_override"): "full-config override slot",
}

# Alternate values for tunable STRING knobs.  A string field is NOT
# automatically a scheme selector: ``surface_bulk_scheme`` /
# ``convective_precip_split`` / ``bechtold_subsidence_solve`` are ordinary
# knobs, and a dropped string knob is exactly the #870 defect (the MPAS lane
# silently ran the default constant surface layer).  A string with no entry
# here has no legal perturbation and lands in _UNPROBEABLE, where it must be
# classified consciously — it never silently becomes a "host axis".
_STRING_ALTS = {
    "surface_bulk_scheme": "coare3",
    "surface_thermo_convention": "aerobulk",
    "surface_stability_scheme": "grachev2007_sheba",
    "convective_precip_split": "autoconversion",
    "bechtold_subsidence_solve": "advective",
    "morrison_flavor": "sam",
}

_HINTS = typing.get_type_hints(ExperimentConfig)
_SCALARS = (bool, float, int)


def _reads(fn, _seen: set[str] | None = None) -> set[str]:
    """ExperimentConfig field names ``fn`` reads, FOLLOWING pipeline callees.

    Matches both access forms the codebase uses interchangeably:
    ``config.X`` / ``cfg.X`` attribute reads, and
    ``getattr(config, "X", <default>)`` -- the defensive form that made the
    gwd gap invisible to grep.

    TRANSITIVE, because the resolvers DELEGATE.  Body-only parsing found
    exactly ONE read for ``convection_config_for`` (the scheme selector): its
    whole tuned-leaf overlay lives in ``_resolve_convection``, and
    ``turbulence_config_for``'s surface-flux overlay in
    ``apply_surface_flux_config``.  A body-only audit was therefore blind to
    case 1 — the defect it is named after.  Recursion follows only statically
    resolvable module-level ``physics_pipeline`` callees that are handed the
    config object itself; dynamic dispatch and helpers imported from other
    modules stay out of reach (a known limit, not a completeness claim).
    """
    seen = set() if _seen is None else _seen
    key = getattr(fn, "__qualname__", repr(fn))
    if key in seen:
        return set()
    seen.add(key)

    fields = set(ExperimentConfig._fields)
    out: set[str] = set()
    for node in _walk_own_scope(ast.parse(inspect.getsource(fn).lstrip())):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.value.id in _CFG_NAMES
                and node.attr in fields):
            out.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if (node.func.id == "getattr" and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in _CFG_NAMES
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in fields):
                out.add(node.args[1].value)
                continue
            callee = getattr(pp, node.func.id, None)
            if (inspect.isfunction(callee) and callee.__module__ == pp.__name__
                    and _binds_config(node, callee)):
                out |= _reads(callee, seen)
    return out


_CFG_NAMES = ("config", "cfg")


def _shadows_config(node) -> bool:
    """Does this nested def/lambda REBIND ``config``/``cfg`` to its own thing?

    ``physics_pipeline`` really contains one (``_resolve_microphysics``'s
    nested ``micro_fn(..., config)``, where ``config`` is a MICROPHYSICS
    config), so a blind ``ast.walk`` would attribute its attribute loads to the
    experiment config.  No field name collides today; the collision would
    manifest as a phantom read -> a phantom "inert" verdict -> a baseline entry
    that hides a real defect, so it is excluded structurally.
    """
    a = node.args
    names = [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    names += [x.arg for x in (a.vararg, a.kwarg) if x is not None]
    return any(n in _CFG_NAMES for n in names)


def _walk_own_scope(tree):
    """``ast.walk`` restricted to the outermost function's own scope."""
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                and _shadows_config(node)
                and node is not tree.body[0]):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _binds_config(call, callee) -> bool:
    """Is the config object passed into a parameter the callee CALLS config?

    ``f(tc, config)`` only delegates config reads when ``f``'s matching
    parameter is itself the experiment config; passing it into some other slot
    (a grid, a state) must not pull the callee's unrelated ``config`` reads in.
    """
    try:
        params = list(inspect.signature(callee).parameters)
    except (TypeError, ValueError):                     # pragma: no cover
        return False
    for i, arg in enumerate(call.args):
        if (isinstance(arg, ast.Name) and arg.id in _CFG_NAMES
                and i < len(params) and params[i] in _CFG_NAMES):
            return True
    return any(isinstance(k.value, ast.Name) and k.value.id in _CFG_NAMES
               and k.arg in _CFG_NAMES for k in call.keywords)


def _candidates(field, value) -> list:
    """Legal alternate values to try for ``field``, best first (may be empty).

    Numeric candidates are two-sided AND two-scale.  Two-sided so a default
    sitting at the EDGE of its validated range still has an admissible
    neighbour.  Two-scale because a 5% nudge cannot cross a rounding or
    threshold boundary: a resolver doing ``int(config.hines_Fmax)`` maps both
    0.105 and 0.095 to 0 and would be certified INERT, while the equally legal
    1.2 changes the result (codex round 2).  ``_INERT`` means "no legal value
    tried changed the output", so the sampled set has to span scales.

    Admissibility is NOT decided here: ``validate_strict`` (in :func:`_probe`)
    is the judge, so a candidate outside a validated range is simply discarded.
    That is what keeps the range fix honest — ``value * 1.5 + 1.0`` took
    ``bechtold_downdraft_entrain_rate`` from 3e-4 to 1.00045, outside its
    validated ``[1e-4, 2e-3]``, and the first version of this audit accepted
    that illegal value as proof of reachability.

    ``None``-default fields are OPT-IN SENTINELS, not absent values — the two
    that formed the dead guard in case 3 (``hard_sat_adjust_threshold`` /
    ``hard_sat_max_heating_K``) both default to ``None``.  Treating them as
    unperturbable would make the audit blind to exactly the defect it exists
    to catch, so the concrete type comes from the annotation.
    """
    if isinstance(value, bool):          # BEFORE int: bool subclasses int
        return [not value]
    if isinstance(value, float):
        if value == 0.0:
            return [1.0e-3, 1.0]
        return [value * 1.05, value * 0.95, value * 2.0 + 1.0, value * 0.5]
    if isinstance(value, int):
        return [value + 1, value - 1, value * 2 + 10]
    if isinstance(value, str):
        alt = _STRING_ALTS.get(field)
        return [alt] if alt is not None else []
    if value is None:
        # Only a plain ``T | None`` carrying exactly one permitted scalar T
        # yields a value.  Substring matching on the annotation string would
        # read "float" out of ``dict[str, float] | None`` and invent 1.5.
        args = typing.get_args(_HINTS.get(field))
        scalars = [a for a in args if a in _SCALARS]
        other = [a for a in args if a not in _SCALARS and a is not type(None)]
        if len(scalars) != 1 or other:
            return []
        return {bool: [True], float: [0.5, 1.5], int: [2]}[scalars[0]]
    return []


def _valid(cfg) -> bool:
    """Would production accept this config?  ``validate_strict`` owns the range
    checks and gate-combination rules (a hard-sat threshold without its boolean
    gate is REFUSED precisely because it would be silently inert), so
    delegating here means the audit can never prove reachability through a
    config no run could legally use."""
    try:
        cfg.validate_strict()
    except Exception:
        return False
    return True


def _differs(a, b) -> bool:
    """Config inequality that survives array-valued leaves.

    ``a != b`` on a NamedTuple holding a JAX array returns an ARRAY, whose
    truth value is ambiguous, and a ``repr`` fallback is LOSSY: NumPy elides
    the middle of a large array, so two 100k-element arrays differing only in
    the elided region share a repr and would compare EQUAL (codex round 2).
    Compare structurally instead — treedef first, then leaf by leaf.
    Per-column turbulence overrides make array leaves a live possibility here,
    not a hypothetical.
    """
    leaves_a, tree_a = tree_util.tree_flatten(a)
    leaves_b, tree_b = tree_util.tree_flatten(b)
    if tree_a != tree_b:
        return True
    return any(not np.array_equal(x, y) for x, y in zip(leaves_a, leaves_b))


def _probe(name, resolver, hosts, field) -> str:
    """Classify ``field`` for ``resolver``: _LIVE / _INERT / _UNPROBEABLE.

    A resolver exception is NEVER swallowed.  Both configs handed to it have
    already passed ``validate_strict``, so a raise is a genuine defect (or a
    stale host table) and propagates — the original ``except Exception:
    continue`` turned a totally broken resolver into a silent "inert" verdict,
    which a baseline entry would then have made permanently green.
    """
    evaluated = False
    for host in hosts:
        base = ExperimentConfig(**host)
        assert _valid(base), (
            f"{name}: host {host} is not a legal ExperimentConfig — fix the "
            "host table; the audit must only probe production-plausible configs")
        for cand in _candidates(field, getattr(base, field)):
            trial = base._replace(**{field: cand})
            if not _valid(trial):
                continue          # inadmissible perturbation: not evidence
            evaluated = True
            if _differs(resolver(base), resolver(trial)):
                return _LIVE
    return _INERT if evaluated else _UNPROBEABLE


def _classify() -> dict[tuple[str, str], str]:
    """{(resolver, field): status} for every audited read."""
    return {
        (name, field): _probe(name, resolver, hosts, field)
        for name, (resolver, hosts) in _RESOLVERS.items()
        for field in sorted(_reads(getattr(pp, name)))
    }


def test_every_resolver_read_reaches_the_leaf():
    """EXACT match against the baselines — dead wiring and silent revivals both
    go red.  A new red entry means a flag a user can set that never reaches the
    scheme config; that is the defect class this campaign hit four times."""
    status = _classify()
    inert = {k for k, v in status.items() if v == _INERT}
    unprobeable = {k for k, v in status.items() if v == _UNPROBEABLE}

    assert not (inert - set(INERT_READS)), (
        "resolver reads an ExperimentConfig field that CANNOT change its "
        f"output — silently inert wiring: {sorted(inert - set(INERT_READS))}. "
        "Fix the resolver, or add it to INERT_READS with a reason if the read "
        "is genuinely validation-only.")
    assert not (unprobeable - set(UNPROBEABLE_READS)), (
        "no legal perturbation exists for these reads, so the audit proves "
        f"NOTHING about them: {sorted(unprobeable - set(UNPROBEABLE_READS))}. "
        "Give the field an entry in _STRING_ALTS / a host that unlocks it, or "
        "classify it in UNPROBEABLE_READS with a reason.")

    revived = {k for k in INERT_READS if status.get(k) != _INERT}
    assert not revived, (
        f"INERT_READS entries are no longer inert — shrink the baseline: "
        f"{sorted(revived)}")
    moved = {k for k in UNPROBEABLE_READS if status.get(k) != _UNPROBEABLE}
    assert not moved, (
        f"UNPROBEABLE_READS entries are now probeable — shrink the baseline: "
        f"{sorted(moved)}")


# ---------------------------------------------------------------------------
# Half B: dangling knobs — a field NO production module reads at all.
#
# Half A only sees fields a resolver reads.  The gwd gap (case 2) was invisible
# to it: the pre-fix resolver did not read the scalars, so there was no read to
# check.  This half closes that by scanning production source for any read of
# each ExperimentConfig field.
#
# KNOWN FALSE NEGATIVES — this is a coarse syntactic scan, not dataflow
# analysis.  A field counts as "read" when its name appears as ANY attribute
# load or ``getattr(<anything>, "name")`` in production source, so it also
# credits dead code, validation/serialization/logging-only reads, a same-named
# attribute on an unrelated object, and — the important one —
# CONFIG-TO-CONFIG PLUMBING: ``mcfarlane_N_ref`` is documented INERT in
# config.py (there is no ``McFarlaneConfig.N_ref`` leaf) yet passes this half,
# because two ExperimentConfig->AMIP-config copies mention it.  That case is
# pinned mechanically in KNOWN_INERT_DESPITE_READS below.
# KNOWN FALSE POSITIVES: a dynamic ``getattr(config, name)`` or ``_asdict()``
# access is missed entirely, so a genuinely-consumed field could look dangling.
# Narrowing the scan to exclude config.py was measured and REJECTED: it moves
# 27 further fields into "dangling", ~26 of which (louis_*, cloud_*,
# sundqvist_*) are genuinely consumed through exactly that plumbing.
# ---------------------------------------------------------------------------

# Fields no production module reads.  Each entry states WHY, because "dangling"
# has several very different causes needing different fixes, and carries a
# MECHANICAL leaf claim so the prose cannot rot:
#     (ConfigClass, "leaf_field", True)   -> the OWNING config HAS that field
#     (ConfigClass, "leaf_field", False)  -> the OWNING config LACKS it
#     None                                -> no leaf claim made
# Absence is scoped to the OWNING class, never global: ``N_ref`` exists on an
# ocean vertical-mixing config and ``precip_efficiency`` on four other
# convection configs, so a repo-wide name search would refute a claim that is
# true (there is no McFarlaneConfig.N_ref, no SBMConfig.precip_efficiency).
# SHRINK-ONLY.
DANGLING_FIELDS = {
    # (2026-07-26: the five morrison_* scalars were WIRED — an overlay in
    # ``_resolve_microphysics`` maps them onto the MorrisonConfig leaves with
    # a fail-loud guard on non-Morrison schemes — and ``morrison_dep_coeff``'s
    # declaration was corrected 1e-8 -> 1e-3 to match the leaf BEFORE wiring,
    # so the overlay is a no-op at defaults.  Entries removed per exact-match.)
    # (2026-09-26: convective_buoyancy_death_memory WIRED — _resolve_convection
    # maps it onto TiedtkeConfig.buoyancy_death_memory; entry removed.)
    # --- Cause 3: the leaf field does not exist ANYWHERE.  The config comment
    # documents physics that was never implemented; these are fiction and the
    # honest fix is deletion (or implementing the described behaviour).
                # The original baseline claimed an SBMConfig.precip_efficiency leaf;
    # adversarial review (2026-07-25) refuted it — SBMConfig ends at
    # cloud_mask_sharpness.  Cause 3, not cause 1.
    "sbm_precip_efficiency": (
        "NO SUCH LEAF: SBMConfig has no precip_efficiency "
        "(convective_precip_efficiency is the wired knob)",
        (SBMConfig, "precip_efficiency", False)),
    # --- Cause 4: advertised knob with no consumer at all.
    # run_bomex_les.py reads its own argparse ``args.micro_substeps``, NOT this
    # ExperimentConfig field, so the field itself is inert everywhere.  (The
    # original baseline credited the LES driver with consuming it.)
    "micro_substeps": (
        "inert: the LES driver reads args.micro_substeps, not this field",
        None),
    # Documented as "log warnings when array dtypes mismatch policy"; nothing
    # reads it, so the advertised debug behaviour never happens.  Being a debug
    # knob rather than a physics knob does not make an inert knob correct.
    "debug_precision": (
        "inert: advertised dtype-mismatch logging has no reader",
        None),
}

# Members of the target defect class that Half B structurally CANNOT see,
# pinned so the known blind spot is mechanical rather than prose: each names a
# leaf field asserted ABSENT FROM ITS OWNING CONFIG (the same owning-class
# scope DANGLING_FIELDS uses — ``N_ref`` does exist on an unrelated ocean
# mixing config), so wiring the knob turns this red and forces the entry out.
# SHRINK-ONLY.
KNOWN_INERT_DESPITE_READS = {
    "mcfarlane_N_ref": (
        "config->config plumbing only; no McFarlaneConfig.N_ref leaf",
        (McFarlaneConfig, "N_ref", False)),
}


@functools.lru_cache(maxsize=1)
def _production_scan() -> frozenset:
    """ExperimentConfig field names read anywhere in production source.

    Roots come from the SHARED ratchet helper (``tests/_ratchet_audit.py``)
    rather than a private literal list, so this audit inherits its repo-root
    anchoring (immune to pytest's CWD), its namespace resolution — which also
    picks up ``packages/ml`` and ``packages/tools`` — and its anti-vacuity
    sentinel check.  A parse failure is NOT skipped silently: an unparseable
    production file would hollow out the scan.
    """
    fields = set(ExperimentConfig._fields)
    files: list[pathlib.Path] = []
    for root in _ratchet_audit.production_roots():
        files.extend(p for p in sorted(root.rglob("*.py"))
                     if "__pycache__" not in p.parts)
    _ratchet_audit.assert_discovery_sane(files)

    read: set[str] = set()
    for path in files:
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError as exc:                      # pragma: no cover
            raise AssertionError(
                f"production file {path} does not parse ({exc}); the "
                "reachability scan would be silently partial") from exc
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Load)
                    and node.attr in fields):
                read.add(node.attr)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "getattr" and len(node.args) >= 2
                  and isinstance(node.args[1], ast.Constant)
                  and node.args[1].value in fields):
                read.add(node.args[1].value)
    return frozenset(read)


def test_no_new_dangling_experiment_config_field():
    """EXACT match: a new user-settable field that no production module reads
    is a knob that does nothing, and a knob that does nothing is worse than no
    knob — it is documented, tunable, and a lie."""
    dangling = set(ExperimentConfig._fields) - _production_scan()
    new = dangling - set(DANGLING_FIELDS)
    assert not new, (
        f"ExperimentConfig fields no production module reads: {sorted(new)}. "
        "Wire them through the owning resolver, or classify them in "
        "DANGLING_FIELDS with the reason.")
    fixed = set(DANGLING_FIELDS) - dangling
    assert not fixed, (
        f"DANGLING_FIELDS entries are now read (good) — shrink the baseline: "
        f"{sorted(fixed)}")


@pytest.mark.parametrize(
    "field,claim",
    sorted(((f, c) for f, (_r, c) in
            {**DANGLING_FIELDS, **KNOWN_INERT_DESPITE_READS}.items() if c),
           key=lambda fc: fc[0]),
)
def test_baseline_leaf_claims_are_true(field, claim):
    """Every leaf claim a baseline entry makes is MACHINE-checked.

    The first version of this baseline shipped four false claims (an SBM leaf
    that does not exist, a SegmentCarry that does not carry the flag, an LES
    driver that reads argparse instead of the field, ...).  Review caught them;
    CI could not, because free-text reasons are never executed.  These are.
    """
    owner, leaf, present = claim
    if present:
        assert leaf in owner._fields, (
            f"{field}: the baseline claims {owner.__name__}.{leaf} exists, and "
            "it does not — the reason is wrong (exactly how the SBM claim "
            "shipped false).")
    else:
        assert leaf not in owner._fields, (
            f"{field}: the baseline claims {owner.__name__} has NO {leaf!r} "
            "leaf, but it does — the knob is now wirable; wire it and shrink "
            "the baseline.")


def test_baseline_has_no_stale_entries():
    """A baseline key naming a read/field that no longer exists is silent rot:
    it would mask a genuinely dead field if that name ever came back."""
    all_reads = {(name, f) for name in _RESOLVERS
                 for f in _reads(getattr(pp, name))}
    stale = (set(INERT_READS) | set(UNPROBEABLE_READS)) - all_reads
    assert not stale, (
        f"baseline names reads that do not exist — delete them: {sorted(stale)}")
    overlap = set(INERT_READS) & set(UNPROBEABLE_READS)
    assert not overlap, (
        f"a read cannot be both proven-inert and never-tested: {sorted(overlap)}")
    fields = set(ExperimentConfig._fields)
    gone = (set(DANGLING_FIELDS) | set(KNOWN_INERT_DESPITE_READS)) - fields
    assert not gone, (
        f"baseline names ExperimentConfig fields that no longer exist — "
        f"delete them: {sorted(gone)}")


@pytest.mark.parametrize("name", sorted(_RESOLVERS))
def test_resolver_exposes_a_checkable_read(name):
    """Guards the AST extractor itself.  Requires a read that is NOT baselined:
    ``convection_config_for``'s only own-body read is the scheme selector,
    which UNPROBEABLE_READS lists — so a bare "reads something" assertion was
    satisfied by the one read that proves nothing, and the audit passed
    vacuously for the very resolver it is named after."""
    baselined = set(INERT_READS) | set(UNPROBEABLE_READS)
    checkable = {f for f in _reads(getattr(pp, name))
                 if (name, f) not in baselined}
    assert checkable, (
        f"{name}: no checkable ExperimentConfig read found — the AST matcher "
        "is broken, or the resolver stopped consuming tuned scalars")


def test_delegated_reads_are_discovered():
    """Pins the transitive extractor against the two known delegations: a
    body-only matcher regresses this audit to blind for convection."""
    conv = _reads(pp.convection_config_for)
    assert {"bechtold_cape_threshold", "sbm_tau_c",
            "convective_precip_efficiency"} <= conv, (
        "convection_config_for delegates its tuned-leaf overlay to "
        f"_resolve_convection; those reads must be followed. Got {sorted(conv)}")
    turb = _reads(pp.turbulence_config_for)
    assert {"surface_bulk_scheme", "surface_gustiness_zi"} <= turb, (
        "turbulence_config_for delegates the surface-flux overlay to "
        f"apply_surface_flux_config. Got {sorted(turb)}")


class TestSelfCheck:
    """Synthetic-violation self-test: the audit must FAIL on wiring that is
    provably dead, or it is a vacuous gate (guardrail doctrine)."""

    _HOST = [{"microphysics": "morrison", "hard_saturation_adjustment": True}]

    def test_detects_a_dropped_read(self):
        """A resolver that reads a field and ignores it must be caught."""
        def _dead_resolver(config):
            _ = config.hard_sat_adjust_threshold      # read, then discarded
            return "constant"

        assert _reads(_dead_resolver) == {"hard_sat_adjust_threshold"}
        assert _probe("dead", _dead_resolver, self._HOST,
                      "hard_sat_adjust_threshold") == _INERT

    def test_detects_a_live_read(self):
        def _live_resolver(config):
            return config.hard_sat_adjust_threshold

        assert _probe("live", _live_resolver, self._HOST,
                      "hard_sat_adjust_threshold") == _LIVE

    def test_getattr_form_is_matched(self):
        """The gwd gap hid behind ``getattr(config, "x", default)``."""
        def _r(config):
            return getattr(config, "mcfarlane_k_wave", 0.0)

        assert _reads(_r) == {"mcfarlane_k_wave"}

    def test_a_crashing_resolver_is_never_called_inert(self):
        """THE vacuity hole: ``except Exception: continue`` classified a
        totally broken resolver as "inert", which a baseline entry would then
        have made permanently green."""
        def _crasher(config):
            raise RuntimeError("resolver is broken")

        with pytest.raises(RuntimeError, match="resolver is broken"):
            _probe("crasher", _crasher, self._HOST, "hard_sat_adjust_threshold")

    def test_unprobeable_is_distinct_from_inert(self):
        """A field with no legal perturbation must NOT be reported as proven
        dead — an untested field and a dead field need different fixes."""
        def _r(config):
            return config.turbulence_override

        assert _probe("opaque", _r, [{"turbulence": "louis"}],
                      "turbulence_override") == _UNPROBEABLE

    def test_illegal_perturbation_is_not_evidence(self):
        """A perturbation ``validate_strict`` refuses cannot certify liveness:
        the hard-sat threshold is illegal without its boolean gate, and the
        first version of this audit called the field "live" using exactly that
        illegal config."""
        def _live_resolver(config):
            return config.hard_sat_adjust_threshold

        assert _probe("gate-off", _live_resolver, [{"microphysics": "morrison"}],
                      "hard_sat_adjust_threshold") == _UNPROBEABLE

    def test_out_of_range_candidate_is_never_the_evidence(self):
        """``v*1.5+1.0`` took bechtold_downdraft_entrain_rate to 1.00045,
        outside its validated [1e-4, 2e-3].  The candidate list deliberately
        SPANS that range now (rounding boundaries need it), so the guarantee
        lives in ``_probe``: an in-family candidate is offered first, an
        out-of-range one exists and is DISCARDED, and the field still resolves
        LIVE — i.e. the verdict rests on a legal value."""
        base = ExperimentConfig(convection="bechtold")
        cands = _candidates("bechtold_downdraft_entrain_rate",
                            base.bechtold_downdraft_entrain_rate)
        assert _valid(base._replace(bechtold_downdraft_entrain_rate=cands[0]))
        assert any(not _valid(base._replace(bechtold_downdraft_entrain_rate=c))
                   for c in cands), "expected a range-crossing candidate"
        assert _probe("bech", pp.convection_config_for,
                      [{"convection": "bechtold"}],
                      "bechtold_downdraft_entrain_rate") == _LIVE

    def test_rounding_resolver_is_not_called_inert(self):
        """A 5%-only candidate list cannot cross an ``int()`` boundary: 0.105
        and 0.095 both round to 0, so a LIVE field was certified inert
        (codex round 2).  The legal 1.2 must be reachable as evidence."""
        def _rounding(config):
            return int(config.hines_Fmax)

        assert _probe("round", _rounding, [{"gravity_wave_drag": "hines"}],
                      "hines_Fmax") == _LIVE

    def test_elided_array_difference_is_detected(self):
        """``repr`` elides the middle of a large array, so a repr fallback
        called two genuinely different configs equal.  Structural comparison
        must not."""
        a = np.zeros(100_000)
        b = a.copy()
        b[50_000] = 1.0
        assert repr(a) == repr(b), "precondition: reprs are elided-equal"
        assert _differs((a,), (b,))
        assert not _differs((a,), (a.copy(),))

    def test_nested_scope_config_is_not_credited(self):
        """physics_pipeline's nested ``micro_fn(..., config)`` rebinds
        ``config`` to a MICROPHYSICS config; crediting its attribute loads to
        the experiment config would invent phantom reads."""
        def _outer(config):
            def _inner(config):          # a DIFFERENT config object
                return config.sbm_tau_c
            return _inner

        assert _reads(_outer) == set()

    def test_recursion_requires_the_callee_to_bind_config(self, monkeypatch):
        """Passing the config into a NON-config parameter must not pull the
        callee's unrelated ``config`` reads in."""
        def _unbound(other, config=None):
            return other.sbm_tau_c

        def _bound(config):
            return config.sbm_tau_c

        # recursion only follows physics_pipeline-owned callees
        _unbound.__module__ = _bound.__module__ = pp.__name__

        def _caller_unbound(config):
            return _probe_callee(config)     # noqa: F821 - parsed, not run

        def _caller_bound(config):
            return _probe_callee(config)     # noqa: F821 - parsed, not run

        monkeypatch.setattr(pp, "_probe_callee", _unbound, raising=False)
        assert _reads(_caller_unbound) == set()
        monkeypatch.setattr(pp, "_probe_callee", _bound, raising=False)
        assert _reads(_caller_bound) == {"sbm_tau_c"}

    def test_container_annotation_yields_no_nonsense_scalar(self):
        """Substring matching on the annotation string read "float" out of
        ``dict[str, float] | None``; get_args-based selection must not."""
        saved = dict(_HINTS)
        try:
            _HINTS["_probe_container"] = dict[str, float] | None
            _HINTS["_probe_literal"] = typing.Literal["a", "b"] | None
            _HINTS["_probe_float"] = float | None
            assert _candidates("_probe_container", None) == []
            assert _candidates("_probe_literal", None) == []
            assert _candidates("_probe_float", None)
        finally:
            _HINTS.clear()
            _HINTS.update(saved)

    def test_nested_guard_regression(self):
        """THE case-3 shape: the read is real, from the right function, with
        the right name -- but sits under a guard that is False by default, so
        it is inert in production.  A static check passes; this must not."""
        def _guarded(config):
            leaf = {"nuc": False}
            if getattr(config, "hard_sat_adjust_threshold", None) is not None:
                # unreachable at defaults -> the flag never lands
                leaf["nuc"] = config.homogeneous_ice_nucleation
            return tuple(sorted(leaf.items()))

        assert "homogeneous_ice_nucleation" in _reads(_guarded)
        assert _probe("guarded", _guarded, [{"microphysics": "morrison"}],
                      "homogeneous_ice_nucleation") == _INERT, (
            "the nested-guard defect must be detected as inert")
