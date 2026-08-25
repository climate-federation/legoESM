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

import functools
from pathlib import Path

from legoesm.core.param_overrides import apply_param_overrides

from .config import TurbulenceConfig

# configs/tuned/turbulence_les.yaml at the repo root. Resolved relative to this
# file so it works from any CWD and under an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[6]
LES_TUNED_YAML = _REPO_ROOT / "configs" / "tuned" / "turbulence_les.yaml"

# Registry-qualified names are ``atm.turb.<ClassName>.<field>``; this is the
# prefix stripped to recover ``<ClassName>.<field>``.
_QUAL_PREFIX = "atm.turb."


@functools.lru_cache(maxsize=1)
def load_les_tuned_overrides(
    path: str | None = None,
) -> dict[str, dict[str, float]]:
    """Parse the tuned YAML into ``{ClassName: {field: value}}``.

    Cached: the file is small and constant for a run. Pass ``path`` (bypasses
    the cache key only if different) to load an alternate file in tests.
    """
    import yaml

    src = Path(path) if path is not None else LES_TUNED_YAML
    if not src.exists():
        raise FileNotFoundError(
            f"LES-tuned turbulence YAML not found at {src}. Regenerate with "
            "scripts/data/build_tuned_turbulence_yaml.py.")
    raw = yaml.safe_load(src.read_text()) or {}
    grouped: dict[str, dict[str, float]] = {}
    for qual, value in raw.items():
        if not qual.startswith(_QUAL_PREFIX):
            raise ValueError(
                f"{src}: key {qual!r} does not start with {_QUAL_PREFIX!r}; the "
                "file must hold registry-qualified turbulence params only.")
        cls_field = qual[len(_QUAL_PREFIX):]          # <ClassName>.<field>
        cls, field = cls_field.rsplit(".", 1)
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
    unchanged, with no error -- the fit simply does not exist yet.
    """
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
    for ``scheme``; ``les_tuned=False`` returns the library defaults. Extra
    keyword ``overrides`` go to ``TurbulenceConfig`` and are applied BEFORE the
    tuned splice, so an explicit sub-config still wins where the caller sets one.
    """
    turb = TurbulenceConfig(scheme=scheme, **overrides)
    if not les_tuned:
        return turb
    return apply_les_tuned_turbulence(turb, path=path)
