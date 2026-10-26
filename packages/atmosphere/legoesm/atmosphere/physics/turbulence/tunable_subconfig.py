"""Descend into a turbulence scheme's tunable leaf config (nested-CLUBB aware).

Most turbulence ``*Config`` NamedTuples carry their spec'd tunable coefficients as
DIRECT fields, so the config itself is the ``apply_param_overrides`` target. Full
CLUBB is the exception: its coefficients live in a nested ``CLUBBParams`` under
``CLUBBConfig.params`` (that nested tuple owns the ``__param_spec__`` + scheme_key).

These two helpers are the SINGLE shared descend/re-wrap used by every SCM tuner (the
RCE campaign, the RCE AD trainer, and the LES-suite tuner) so the CLUBB-nesting rule
is declared once, not re-derived per driver. Extracted from
``scripts/run/run_scm_rce_campaign.py`` (which held the only copy, privately) so the
LES-suite tuner can wire full CLUBB without importing a private symbol or duplicating
the isinstance check.

The ``CLUBBConfig`` import is function-scoped on purpose: this module must not force a
clubb import at load (avoids an import cycle) and the flat-config common path never
needs it.
"""
from __future__ import annotations


def tunable_subconfig(subcfg):
    """The object whose DIRECT NamedTuple fields carry the spec'd tunable params.

    Flat for every scheme except full CLUBB, which nests its closure coefficients in
    ``.params`` (a ``CLUBBParams``); that nested tuple — not the ``CLUBBConfig``
    wrapper — is the ``apply_param_overrides`` target. ``None`` passes through.
    """
    if subcfg is None:
        return None
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    if isinstance(subcfg, CLUBBConfig):
        return subcfg.params
    return subcfg


def rewrap_tunable_subconfig(subcfg, tuned):
    """Inverse of :func:`tunable_subconfig`: fold a tuned tunable-object back into the
    scheme sub-config the driver writes onto the active ``TurbulenceConfig`` field.
    """
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    if isinstance(subcfg, CLUBBConfig):
        return subcfg._replace(params=tuned)
    return tuned
