"""LES-tuned turbulence parameter defaults for the single-column model.

The turbulence closures were calibrated to eight LES benchmark cases (a 10-seed
uncertainty campaign; the committed values are the median seed per scheme). This
module loads those tuned coefficients from the tracked YAML and splices them into
a :class:`TurbulenceConfig` for the ACTIVE scheme.

Split of responsibility (deliberate):
  * SCM -- applies these by DEFAULT via :func:`scm_turbulence_config`
    (``les_tuned=True``); the single-column model is what they were fit on.
  * AMIP / production -- OPT-IN only, via ``run_amip.py --params <file>``
    (the same registry-qualified format).  A global run is not a column, so the
    LES-column fit is offered, not imposed.  NOTE: the tracked YAML carries the
    tuned params for ALL eight closures at once; the generic ``--params`` router
    (correctly) ABORTS on any param whose scheme is not the one selected for the
    run.  So for AMIP pass only the ACTIVE scheme's slice --
    :func:`write_active_scheme_params` writes that slice for a given scheme.

Production ``*Config`` NamedTuple defaults are NEVER mutated (repo rule): the
overrides are spliced with :func:`legoesm.core.param_overrides.apply_param_overrides`
at construction, exactly as the trainer injects tuned leaves. ``les_tuned=False``
returns the library defaults unchanged.
"""
from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

from legoesm.core.param_overrides import apply_param_overrides

from . import config as _turb_config
from .config import TurbulenceConfig

_log = logging.getLogger(__name__)

#: The tuned coefficients ship as package data (next to this module), so a
#: normal ``pip install`` wheel carries them -- unlike the old repo-root
#: ``configs/`` path, which only existed in an editable checkout.
_TUNED_YAML_NAME = "turbulence_les_tuned.yaml"


def _read_tuned_yaml_text(path: str | None) -> tuple[str, str]:
    """Return ``(text, source_label)`` for the tuned YAML.

    ``path`` overrides the packaged default (used by tests / hand-edited files).
    The default is read via :mod:`importlib.resources`, so it resolves in both
    an editable checkout and an installed wheel.
    """
    if path is not None:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(
                f"LES-tuned turbulence YAML not found at {src}. Regenerate with "
                "scripts/data/build_tuned_turbulence_yaml.py.")
        return src.read_text(encoding='utf-8'), str(src)
    res = resources.files(__package__) / _TUNED_YAML_NAME
    if not res.is_file():
        raise FileNotFoundError(
            f"packaged {_TUNED_YAML_NAME} missing from {__package__}. Regenerate "
            "with scripts/data/build_tuned_turbulence_yaml.py.")
    return res.read_text(encoding='utf-8'), f"{__package__}/{_TUNED_YAML_NAME}"


# Registry-qualified names are ``atm.turb.<ClassName>.<field>``.
_QUAL_PREFIX = "atm.turb."


def load_les_tuned_overrides(
    path: str | None = None,
) -> dict[str, dict[str, float]]:
    """Parse the tuned YAML into ``{ClassName: {field: value}}``, FULLY validated.

    Not cached: the read is cheap and rare (once per SCM config build), and a
    cache would return a stale dict if the file is rewritten mid-process (a real
    test hazard with a reused temp path). Validation raises on:
      * a key not shaped ``atm.turb.<Class>.<field>``;
      * an unknown config class (a class-name TYPO would otherwise silently ship
        the library default -- indistinguishable from an intentionally-absent
        scheme like CLUBB; the whole point is tuned-by-default, so this must be
        loud) -- checked against ``config.__param_spec__``;
      * an unknown field on that class, or a non-numeric value;
      * a value outside the field's ``__param_spec__`` bounds (a hand-edit guard;
        ``apply_param_overrides`` itself does not range-check).
    """
    import yaml

    from . import clubb as _clubb          # CLUBB's coefficients (CLUBBParams)
    # live in a SEPARATE spec + a NESTED sub-config (CLUBBConfig.params), not in
    # the flat turbulence config spec. Merge the two so CLUBBParams / CLUBBConfig
    # keys validate; the algebraic schemes are unaffected.
    text, src = _read_tuned_yaml_text(path)
    _dup = set(_turb_config.__param_spec__) & set(_clubb.__param_spec__)
    if _dup:
        raise ValueError(
            f"turbulence config and clubb __param_spec__ share class name(s) "
            f"{sorted(_dup)}; the merge would silently shadow one. Disambiguate "
            "before trusting bounds validation.")
    spec = {**_turb_config.__param_spec__, **_clubb.__param_spec__}
    raw = yaml.safe_load(text) or {}
    grouped: dict[str, dict[str, float]] = {}
    for qual, value in raw.items():
        if not qual.startswith(_QUAL_PREFIX):
            raise ValueError(
                f"{src}: key {qual!r} does not start with {_QUAL_PREFIX!r}; the "
                "file must hold registry-qualified turbulence params only.")
        cls, field = qual[len(_QUAL_PREFIX):].rsplit(".", 1)
        if cls not in spec:
            raise ValueError(
                f"{src}: unknown turbulence config class {cls!r} (key {qual!r}); "
                f"known: {sorted(spec)}. A class-name typo would silently leave "
                "the scheme untuned.")
        params = spec[cls].get("params", {})
        if field not in params:
            raise ValueError(
                f"{src}: {cls} has no spec'd param {field!r} (key {qual!r}); "
                f"known: {sorted(params)}.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{src}: key {qual!r} value {value!r} is not a number.")
        lo, hi = params[field]["bounds"]
        if not (lo <= float(value) <= hi):
            raise ValueError(
                f"{src}: {qual}={value} outside spec bounds [{lo}, {hi}].")
        grouped.setdefault(cls, {})[field] = float(value)
    return grouped


def _subconfig_field_for_scheme(turb: TurbulenceConfig, scheme: str) -> str:
    """The TurbulenceConfig attribute holding ``scheme``'s sub-config.

    The attribute name equals the scheme name for every closure
    (``turb.louis``, ``turb.ysu``, ...); this asserts it rather than assuming,
    so a renamed field fails loudly instead of silently skipping the overrides.
    """
    if not hasattr(turb, scheme):
        raise ValueError(
            f"TurbulenceConfig has no sub-config attribute {scheme!r}; known "
            f"schemes: {[f for f in turb._fields if f != 'scheme']}")
    return scheme


def apply_les_tuned_turbulence(
    turb: TurbulenceConfig, *, path: str | None = None,
) -> TurbulenceConfig:
    """Splice the LES-tuned coefficients for ``turb.scheme`` into ``turb``.

    Only the ACTIVE scheme's sub-config is touched; the others keep their library
    defaults (they are inert when that scheme is not selected). A scheme with no
    tuned entry in the YAML (e.g. CLUBB, whose campaign is unfinished) is returned
    unchanged, with no error -- the fit simply does not exist yet. ``scheme="none"``
    (no turbulence, no sub-config -- a valid selection) is likewise a no-op.
    """
    if turb.scheme == "none":
        return turb
    overrides = load_les_tuned_overrides(path)
    field = _subconfig_field_for_scheme(turb, turb.scheme)
    sub = getattr(turb, field)

    # CLUBB is the ONE scheme whose tuned coefficients are NESTED
    # (``CLUBBConfig.params`` : ``CLUBBParams``, keyed ``CLUBBParams`` in the
    # YAML) rather than flat on the sub-config. Handled explicitly -- an
    # allowlist, not a duck-typed ``.params`` probe, so a future sub-config that
    # happens to grow a ``.params`` field cannot be spliced by accident.
    if turb.scheme == "clubb":
        return _apply_clubb(turb, field, sub, overrides)

    cls_name = type(sub).__name__
    scheme_overrides = overrides.get(cls_name)
    if not scheme_overrides:
        # No tuned entry for the ACTIVE scheme. Returning library defaults is
        # correct, but "les_tuned" then silently runs untuned physics -- warn so
        # the label can't mislead (P1 #3).
        _log.warning(
            "les_tuned: scheme %r (%s) has no tuned entry in the LES YAML; "
            "running LIBRARY DEFAULTS untuned. Tuned schemes: %s.",
            turb.scheme, cls_name, sorted(overrides))
        return turb
    return turb._replace(**{field: apply_param_overrides(sub, scheme_overrides)})


def _apply_clubb(turb, field, sub, overrides):
    """Splice CLUBB's nested tuned coefficients (``CLUBBParams``).

    CLUBB's coefficients were fit on the PROGNOSTIC path and live in
    ``CLUBBConfig.params``. Correctness rules that avoid a silent physics
    mismatch (GLM review):
      * a ``CLUBBParams`` entry MUST be present -- CLUBB's tuning is the nested
        group, so a missing one is not "library defaults" but a broken YAML, and
        raises (never a prognostic-CLUBB-with-default-coefficients fall-through);
      * ``sub is None`` (the opt-in default) becomes ``CLUBBConfig(prognostic=True)``;
      * an explicit DIAGNOSTIC ``CLUBBConfig`` (``prognostic=False``) is REFUSED --
        applying prognostic-fit coefficients to the diagnostic path is an
        out-of-sample regime mismatch. Pass ``prognostic=True`` or ``les_tuned=False``.
    """
    from .clubb import CLUBBConfig

    nested_over = overrides.get("CLUBBParams")
    if not nested_over:
        raise ValueError(
            "les_tuned: CLUBB selected but the LES YAML has no CLUBBParams entry. "
            f"Tuned groups: {sorted(overrides)}. Regenerate the YAML or pass "
            "les_tuned=False.")
    if sub is None:
        sub = CLUBBConfig(prognostic=True)
    elif not sub.prognostic:
        raise ValueError(
            "les_tuned: CLUBB tuned coefficients were fit on the PROGNOSTIC path, "
            "but the supplied CLUBBConfig has prognostic=False. Pass "
            "prognostic=True or les_tuned=False.")
    # A CLUBBConfig-level group (rare/none today) also splices, for symmetry.
    direct = overrides.get("CLUBBConfig")
    if direct:
        sub = apply_param_overrides(sub, direct)
    sub = sub._replace(params=apply_param_overrides(sub.params, nested_over))
    return turb._replace(**{field: sub})


def write_active_scheme_params(scheme: str, out_path: str,
                               *, path: str | None = None) -> int:
    """Write the tuned params for ONE ``scheme`` as a ``run_amip --params`` slice.

    The tracked YAML holds all closures; the generic ``--params`` router aborts
    on any param whose scheme is not selected for the run, so an AMIP opt-in must
    pass only the active scheme's slice.  This writes that slice
    (``atm.turb.<ActiveClass>.<field>: value``) and returns the count.  Raises if
    ``scheme`` has no tuned entry (nothing to opt into).

    CLUBB is REFUSED: its coefficients were fit on the prognostic path, but
    ``--params`` sets only scalar param values and cannot flip
    ``CLUBBConfig.prognostic`` (a structural config field). A ``--params`` slice
    alone would therefore apply the fitted coefficients to a DIAGNOSTIC CLUBB --
    silently the wrong physics (codex review). Opt CLUBB in via a two-step:
    ``--config`` to select prognostic CLUBB, then ``--params`` for the
    coefficients; this function will not emit a slice that is unsafe by itself.
    """
    import yaml

    if scheme == "clubb":
        raise ValueError(
            "CLUBB cannot be opted into via a --params slice alone: its tuned "
            "coefficients are prognostic-fit and --params cannot set "
            "CLUBBConfig.prognostic. Enable prognostic CLUBB via --config, then "
            "pass the coefficients. (The SCM path scm_turbulence_config handles "
            "this automatically.)")
    probe = TurbulenceConfig(scheme=scheme)
    field = _subconfig_field_for_scheme(probe, scheme)
    cls_name = type(getattr(probe, field)).__name__   # flat schemes only here
    all_over = load_les_tuned_overrides(path)
    overrides = all_over.get(cls_name)
    if not overrides:
        raise ValueError(
            f"scheme {scheme!r} ({cls_name}) has no tuned entry in the LES YAML; "
            f"tuned schemes: {sorted(all_over)}.")
    slice_doc = {f"{_QUAL_PREFIX}{cls_name}.{f}": v
                 for f, v in sorted(overrides.items())}
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(slice_doc, sort_keys=True), encoding='utf-8')
    return len(slice_doc)


def scm_turbulence_config(
    scheme: str = "smagorinsky", *, les_tuned: bool = True,
    path: str | None = None, **overrides,
) -> TurbulenceConfig:
    """TurbulenceConfig for the SCM, LES-tuned by DEFAULT.

    ``les_tuned=True`` (default) splices the median-seed LES-tuned coefficients
    for ``scheme``; ``les_tuned=False`` returns the library defaults.

    Extra keyword ``overrides`` go to ``TurbulenceConfig`` and are applied FIRST,
    so for the ACTIVE scheme the tuned splice OVERWRITES any overlapping field a
    caller set through ``overrides`` -- the tuned value wins, not the caller's.
    To keep full control of the active scheme's sub-config, pass
    ``les_tuned=False`` (or splice your own values afterwards). ``overrides`` for
    a NON-active scheme survive, since tuning only touches the active one.
    """
    turb = TurbulenceConfig(scheme=scheme, **overrides)
    if not les_tuned:
        return turb
    return apply_les_tuned_turbulence(turb, path=path)
