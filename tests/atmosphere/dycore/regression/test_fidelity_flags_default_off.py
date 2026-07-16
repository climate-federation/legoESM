"""Fidelity-flags-default-OFF sentinel (behavioral replacement).

Restores the invariant the archived source-grep sentinels
(`iter873`/`iter928`/`iter369`) used to guard: every FV3-fidelity opt-in flag
(`use_fv3_*`) on the cubed-sphere PE and NH dycore configs must default to
``False``, so the bare/default config is the conservative non-faithful baseline
and fidelity is *opt-in* (via ``make_fv3_faithful_{pe,nh}_config`` — pinned by
``test_fv3_faithful_config_factories_iter392.py``).

Why behavioral, not source-grep: the archived sentinels grepped
``scripts/matrix/run_atmosphere_test_matrix.py`` source text for config-construction
strings and broke when the federation restructure moved/renamed the runner.
This version constructs the actual config objects and reads their defaults, so
it is robust to layout changes and to *new* fidelity flags — it auto-discovers
every ``use_fv3_*`` field rather than hard-coding the list, so a newly added
flag that accidentally defaults ON fails here.

Codex adversarial-review HIGH-2 follow-up.
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    make_fv3_component_fidelity_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    make_fv3_component_fidelity_pe_config,
)

pytestmark = pytest.mark.tier1


def _fidelity_flags(config) -> dict[str, bool]:
    """All ``use_fv3_*`` boolean fields on a NamedTuple config."""
    return {
        name: getattr(config, name)
        for name in config._fields
        if name.startswith("use_fv3_")
    }


@pytest.mark.parametrize(
    "cfg_cls",
    [CDGridPrimitiveEquationConfig, CDGridCompressibleEulerConfig],
    ids=["pe", "nh"],
)
def test_default_config_has_all_fidelity_flags_off(cfg_cls):
    flags = _fidelity_flags(cfg_cls())
    assert flags, f"{cfg_cls.__name__} exposes no use_fv3_* flags (API moved?)"
    on = {k: v for k, v in flags.items() if v is not False}
    assert not on, (
        f"{cfg_cls.__name__} default must keep every FV3-fidelity flag OFF "
        f"(opt-in only); these defaulted non-False: {on}"
    )


@pytest.mark.parametrize(
    "factory",
    [make_fv3_component_fidelity_pe_config, make_fv3_component_fidelity_nh_config],
    ids=["pe", "nh"],
)
def test_faithful_factory_turns_some_flags_on(factory):
    # Complement: the faithful factory must enable at least one fidelity flag the
    # default leaves OFF (it is the opt-in path).  The *specific* production set
    # is pinned by test_fv3_faithful_config_factories_iter392.py; here we only
    # assert the factory is not a no-op vs the default.
    faithful = _fidelity_flags(factory())
    assert any(v is True for v in faithful.values()), (
        f"{factory.__name__} enabled no fidelity flags — the faithful path is a "
        f"no-op vs the default-OFF config."
    )
