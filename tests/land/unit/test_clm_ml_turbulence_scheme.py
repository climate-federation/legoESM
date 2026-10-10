"""Selectable CLM-ML canopy-airspace turbulence scheme — config + dispatch.

``CLMMLCanopyConfig.turbulence_scheme`` selects whether CLM-ML's roughness-
sublayer correction ψ̂ is active: ``"rsl_bonan"`` (Harman & Finnigan roughness
sublayer, CLM-ML's native scheme, the default) or ``"most"`` (ψ̂ ≡ 0, leaving
Monin-Obukhov ψ).  ``"most"`` is NOT a reproduction of the two-leaf/big-leaf
surface layer — β, the displacement height, the u(hc)=u*/β canopy-top anchor and
the within-canopy mixing length all stay Harman & Finnigan.

This file covers the legoESM-side contract that needs NO ``clm-ml-jax``: the
default, the fail-early validator, and the fact that the scheme dispatch
REJECTS an unknown value instead of silently running the default RSL physics.
``tests/land/integration/test_clm_ml_turbulence_real.py`` drives the real
installed model and proves the selected scheme actually changes the physics.
"""

from __future__ import annotations

import ast
import inspect
import importlib
import sys
import types

import numpy as np
import pytest

from legoesm.land.canopy import clm_ml_interface as iface
from legoesm.land.canopy.config import (
    VALID_CLM_ML_TURBULENCE_SCHEMES,
    CLMMLCanopyConfig,
)
from legoesm.land.canopy.clm_ml_interface import _apply_turbulence_scheme


def test_default_is_native_rsl():
    """Stand-alone default stays CLM-ML's own Bonan RSL formulation."""
    assert CLMMLCanopyConfig().turbulence_scheme == "rsl_bonan"


def test_valid_schemes_are_the_two_implemented_ones():
    assert VALID_CLM_ML_TURBULENCE_SCHEMES == ("rsl_bonan", "most")


@pytest.mark.parametrize("scheme", VALID_CLM_ML_TURBULENCE_SCHEMES)
def test_validate_accepts_every_valid_scheme(scheme):
    cfg = CLMMLCanopyConfig(turbulence_scheme=scheme)
    assert cfg.validate() is cfg


def test_validate_rejects_unknown_scheme():
    """A typo must abort at setup, not silently run RSL."""
    cfg = CLMMLCanopyConfig(turbulence_scheme="monin_obukhov")
    with pytest.raises(ValueError, match="unknown turbulence_scheme"):
        cfg.validate()


def test_apply_turbulence_scheme_rejects_unknown_before_touching_clm():
    """The applier is the backstop guard, independent of the config validator.

    The rejection happens before the ``multilayer_canopy`` import, so this holds
    whether or not the optional dependency is installed.
    """
    with pytest.raises(ValueError, match="unknown CLM-ML turbulence_scheme"):
        _apply_turbulence_scheme("rsl")


def test_scheme_is_a_static_leaf_not_a_traced_one():
    """Scheme selection is host-side Python state, so it must be a plain str."""
    assert isinstance(CLMMLCanopyConfig(turbulence_scheme="most").turbulence_scheme, str)


def test_production_entry_applies_the_scheme():
    """Tripwire: deleting the dispatch call must go red.

    The behavioural tests below and in the integration file drive
    ``_apply_turbulence_scheme`` directly, so they would all still pass if the
    call were dropped from the public flux entry point and the config field
    became decorative.  This asserts the wiring itself.
    """
    tree = ast.parse(inspect.getsource(iface.compute_clm_ml_canopy_fluxes))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_apply_turbulence_scheme" in called, (
        "compute_clm_ml_canopy_fluxes no longer applies the turbulence scheme; "
        "CLMMLCanopyConfig.turbulence_scheme would be silently ignored")


# ---------------------------------------------------------------------------
# Table-swap mechanics against a FAKE MLCanopyTurbulenceMod.
#
# The real-dependency tests importorskip, so without this a CI box lacking
# clm-ml-jax would accept an implementation that never swaps anything.  The
# fake mirrors the upstream contract that matters: four module-global psihat
# tables, read by four lookup entry points.
# ---------------------------------------------------------------------------

_BACKEND = "legoesm.land.canopy.clm_ml_backend.multilayer_canopy"

_TABLE_ATTRS = ("psigridM", "psigridH", "_psigridM_jax", "_psigridH_jax")


def _fake_turbulence_module():
    mod = types.ModuleType("multilayer_canopy.MLCanopyTurbulenceMod")
    for i, attr in enumerate(_TABLE_ATTRS):
        # Distinct non-zero tables so a swap that mixes them up is visible.
        setattr(mod, attr, np.full((3, 4), float(i + 1)))
    # Lookups read the module globals, exactly as upstream does.
    mod._LookupPsihatM = lambda z, d: float(mod.psigridM.mean())
    mod._LookupPsihatH = lambda z, d: float(mod.psigridH.mean())
    mod._LookupPsihatM_scalar = lambda z, d: float(mod._psigridM_jax.mean())
    mod._LookupPsihatH_scalar = lambda z, d: float(mod._psigridH_jax.mean())
    return mod


@pytest.fixture
def fake_clm(monkeypatch):
    """Install a fake upstream module and isolate the interface's caches.

    Restoring the caches matters for real: they are module globals, so leaking
    fake tables would make the real-dependency tests in the same pytest process
    silently validate against a stub.
    """
    pkg = importlib.import_module(_BACKEND)
    mod = _fake_turbulence_module()
    monkeypatch.setattr(pkg, "MLCanopyTurbulenceMod", mod, raising=False)
    monkeypatch.setitem(sys.modules, f"{_BACKEND}.MLCanopyTurbulenceMod", mod)
    monkeypatch.setattr(iface, "_ensure_clm_initialized", lambda: None)
    monkeypatch.setattr(iface, "_PSIHAT_RSL", None)
    monkeypatch.setattr(iface, "_DIFF_TURBULENCE_SCHEME", None)
    return mod


def test_most_zeroes_all_four_tables(fake_clm):
    _apply_turbulence_scheme("most")
    for attr in _TABLE_ATTRS:
        assert not getattr(fake_clm, attr).any(), attr


def test_rsl_restores_all_four_tables(fake_clm):
    before = {a: getattr(fake_clm, a).copy() for a in _TABLE_ATTRS}
    _apply_turbulence_scheme("most")
    _apply_turbulence_scheme("rsl_bonan")
    for attr, ref in before.items():
        assert np.array_equal(getattr(fake_clm, attr), ref), attr


def test_clm_never_holds_the_pristine_snapshot_object(fake_clm):
    """Upstream writes psigrid* in place; aliasing would corrupt the snapshot."""
    _apply_turbulence_scheme("rsl_bonan")
    snapshot = iface._PSIHAT_RSL
    for attr in _TABLE_ATTRS:
        assert getattr(fake_clm, attr) is not snapshot[attr], attr
    # Simulate an upstream in-place re-initialisation writing into the table
    # CLM currently holds, then confirm the RSL reference survived it.
    fake_clm.psigridM[:, :] = -99.0
    _apply_turbulence_scheme("most")
    _apply_turbulence_scheme("rsl_bonan")
    assert np.array_equal(getattr(fake_clm, "psigridM"), snapshot["psigridM"])
    assert not np.any(getattr(fake_clm, "psigridM") == -99.0)


def test_all_zero_tables_are_rejected_as_the_rsl_reference(fake_clm):
    """Snapshotting a pre-init module would pin BOTH schemes to MOST.

    ``MLCanopyTurbulenceMod`` allocates psigrid* as zeros at import, so this is
    the real failure mode if the tables are read before LookupPsihatINI.
    """
    for attr in _TABLE_ATTRS:
        setattr(fake_clm, attr, np.zeros((3, 4)))
    with pytest.raises(RuntimeError, match="all zeros"):
        _apply_turbulence_scheme("rsl_bonan")


def test_ineffective_swap_is_detected(fake_clm):
    """If ψ̂ stops coming from those tables, 'most' must fail, not lie."""
    fake_clm._LookupPsihatM = lambda z, d: 1.0  # upstream moved ψ̂ elsewhere
    with pytest.raises(RuntimeError, match="did not remove the roughness-"):
        _apply_turbulence_scheme("most")


def test_diff_mode_refuses_a_mid_process_scheme_switch(fake_clm):
    """A traced step bakes the table in; a later switch cannot reach it."""
    iface._commit_diff_turbulence_scheme("rsl_bonan")
    with pytest.raises(RuntimeError, match="cannot switch turbulence_scheme"):
        _apply_turbulence_scheme("most", differentiable=True)


def test_applying_alone_does_not_lock_the_process(fake_clm):
    """The lock belongs to the TRACE, not to the table swap.

    ``compute_clm_ml_canopy_fluxes`` runs several diff-mode preflight checks
    (capability probe, structural-int extraction) AFTER applying the scheme and
    BEFORE tracing.  If applying committed the lock, a run that raised in
    preflight would lock the process to a scheme it never compiled and then
    wrongly reject a later valid run under the other scheme.
    """
    _apply_turbulence_scheme("rsl_bonan", differentiable=True)
    assert iface._DIFF_TURBULENCE_SCHEME is None
    # ... so the other scheme is still available until something is traced.
    _apply_turbulence_scheme("most", differentiable=True)
    assert not fake_clm.psigridM.any()


def test_the_lock_is_committed_only_after_a_successful_trace():
    """Tripwire on the ORDER, not merely the presence, of the commit.

    The lock must be applied AFTER ``MLCanopyFluxes`` returns.  Committing
    earlier — anywhere between the table swap and a successful trace — would
    lock the process to a scheme that a failed call never actually compiled,
    and then wrongly reject a later valid run under the other scheme.  A
    presence-only check would not catch the commit drifting back upwards.
    """
    src = inspect.getsource(iface.compute_clm_ml_canopy_fluxes)
    tree = ast.parse(src)
    lines = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            lines.setdefault(node.func.id, []).append(node.lineno)
    for name in ("_apply_turbulence_scheme", "_commit_diff_turbulence_scheme",
                 "MLCanopyFluxes"):
        assert name in lines, f"{name} is no longer called; the turbulence " \
                              "scheme would be ignored or never locked"
    assert min(lines["_apply_turbulence_scheme"]) < min(lines["MLCanopyFluxes"]), \
        "the turbulence scheme must be applied before the canopy step is traced"
    assert min(lines["_commit_diff_turbulence_scheme"]) > max(lines["MLCanopyFluxes"]), \
        ("the differentiable-mode lock is committed before MLCanopyFluxes "
         "returns, so a failed trace would still lock the process")


def test_forward_mode_may_switch_freely(fake_clm):
    """Forward CLM-ML is eager, so each step re-reads the globals."""
    _apply_turbulence_scheme("most")
    _apply_turbulence_scheme("rsl_bonan")
    _apply_turbulence_scheme("most")
    assert not fake_clm.psigridM.any()
