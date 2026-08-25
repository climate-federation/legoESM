"""LES-tuned turbulence parameter defaults for the single-column model.

The turbulence closures were calibrated to eight LES benchmark cases (a 10-seed
uncertainty campaign; the committed values are the median seed per scheme). This
module loads those tuned coefficients from the tracked YAML and splices them into
a :class:`TurbulenceConfig` for the ACTIVE scheme.

Split of responsibility (deliberate):
  * SCM -- applies these by DEFAULT via :func:`scm_turbulence_config`
    (``les_tuned=True``); the single-column model is what they were fit on.
  * AMIP / production -- OPT-IN only, via ``run_amip.py --params
    configs/tuned/turbulence_les.yaml`` (the same registry-qualified format).
    A global run is not a column, so the LES-column fit is offered, not imposed.

Production ``*Config`` NamedTuple defaults are NEVER mutated (repo rule): the
overrides are spliced with :func:`legoesm.core.param_overrides.apply_param_overrides`
at construction, exactly as the trainer injects tuned leaves. ``les_tuned=False``
returns the library defaults unchanged.
"""
from __future__ import annotations

from pathlib import Path

from legoesm.core.param_overrides import apply_param_overrides

from . import config as _turb_config
from .config import TurbulenceConfig


def _find_repo_file(rel: str) -> Path:
    """Locate a repo-root file by walking up from this module.

    The double-name federation layout puts ``configs/`` at the repo root, not in
    any package, so relative-to-``__file__`` with a hardcoded parent count is
    brittle. Walk up until a directory containing both ``pyproject.toml`` and the
    target is found; raise with the search root if neither turns up.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / rel
        if cand.exists() and (parent / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError(
        f"{rel} not found walking up from {here}. Regenerate with "
        "scripts/data/build_tuned_turbulence_yaml.py, and confirm it runs from "
        "the repo (editable install).")


LES_TUNED_YAML = _find_repo_file("configs/tuned/turbulence_les.yaml")

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

    src = Path(path) if path is not None else LES_TUNED_YAML
    if not src.exists():
        raise FileNotFoundError(
            f"LES-tuned turbulence YAML not found at {src}. Regenerate with "
            "scripts/data/build_tuned_turbulence_yaml.py.")
    spec = _turb_config.__param_spec__
    raw = yaml.safe_load(src.read_text()) or {}
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
    cls_name = type(sub).__name__
    scheme_overrides = overrides.get(cls_name)
    if not scheme_overrides:
        return turb
    tuned_sub = apply_param_overrides(sub, scheme_overrides)
    return turb._replace(**{field: tuned_sub})


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
