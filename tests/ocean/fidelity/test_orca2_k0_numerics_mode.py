"""Run-wide numerics mode on the standalone ORCA2 K0 path.

The card names the mode it was built for; the step entry and both compiled
K0 programs refuse to run under any other active mode.  That the card's
``numerics`` alone keys both programs is pinned next to their trace-count test
in ``tests/ocean/unit/test_ncar_bulk.py``.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from legoesm.core.numerics_mode import numerics_scope
from legoesm.ocean import bulk_flux_omip as bulk


def _k0():
    return bulk.NemoStandaloneK0Config(1, False, True, 0, False, "nemo_k0_istate")


def _stub_card(numerics, ocean=None):
    return bulk.NemoStandaloneK0OceanCard(
        ocean=ocean, k0=_k0(), albedo=0.066, precip_factor=1.0,
        evap_factor=1.0, nb_iter=5, numerics=numerics)


def _fake_rung10_ocean():
    """Only the selectors ``_validate_k0_ocean_card`` reads, at their K0 values."""
    tke = SimpleNamespace(tke_mxl_choice=3, nemo_mxl0_surface_tmask=True,
                          nemo_mxl0_rmxl_min_overwrite=True, nemo_mxlice=0)
    config = SimpleNamespace(
        implicit_ice_ocean_top_drag=False,
        physics=SimpleNamespace(vertical_mixing=SimpleNamespace(tke=tke)))
    return SimpleNamespace(
        recipe=SimpleNamespace(model_config=config, initial_state=None),
        icebergs_enabled=False)


def _scope(mode):
    return numerics_scope(mode) if mode is not None else nullcontext()


def test_card_numerics_defaults_to_none_and_the_validator_refuses_other_values():
    assert bulk.NemoStandaloneK0OceanCard._field_defaults == {"numerics": None}
    for bad in ("fast", "exact", "NEMO_EXACT", "", 0, False):
        with pytest.raises(ValueError, match="numerics must be None or one of"):
            bulk._validate_k0_ocean_card(_stub_card(bad))


def test_each_k0_program_asserts_its_static_numerics_at_trace_time():
    """Both programs refuse a card/argument naming a mode that is not active.

    The assertion is the first statement of each body, so ``None`` operands
    never reach the arithmetic.
    """
    surface_kwargs = dict.fromkeys(
        ("s0f", "ice_if", "s0", "s2", "chl", "geothermal_heat_flux_mwm2",
         "prior_surface"))

    def surface(numerics):
        return bulk._nemo_orca2_k0_surface_compiled(
            _stub_card(numerics), None, None, *(None,) * 7,
            forcing_update=True, **surface_kwargs)

    def fraqsr(numerics):
        return bulk._nemo_orca2_k0_next_fraqsr_compiled(
            None, None, numerics, *(None,) * 7)

    for program in (surface, fraqsr):
        for static, active in (("nemo_exact", None), ("nemo_exact", "nemo_fast"),
                               ("nemo_fast", "nemo_exact"), (None, "nemo_exact")):
            with _scope(active):
                with pytest.raises(RuntimeError, match="built for numerics mode"):
                    program(static)


def test_k0_step_entry_refuses_a_card_built_for_another_mode():
    """``nemo_orca2_k0_ocean_step`` refuses before touching its operands, and
    lets a card through to its operand checks only under the card's own mode."""
    card = _stub_card(None, ocean=_fake_rung10_ocean())
    operands = dict(s0f=None, ice_if=None, s0=None, s2=None, chl=None,
                    geothermal_heat_flux_mwm2=None)
    for numerics, active in (("nemo_exact", "nemo_fast"), ("nemo_exact", None),
                             ("nemo_fast", None), (None, "nemo_exact")):
        with _scope(active):
            with pytest.raises(RuntimeError, match="built for numerics mode"):
                bulk.nemo_orca2_k0_ocean_step(
                    card._replace(numerics=numerics), None, nemo_kt=1, **operands)
    for numerics in (None, "nemo_exact", "nemo_fast"):
        with _scope(numerics):
            with pytest.raises(ValueError, match="requires nemo_kt"):
                bulk.nemo_orca2_k0_ocean_step(
                    card._replace(numerics=numerics), None, nemo_kt=None,
                    **operands)
    with pytest.raises(ValueError, match="numerics must be None or one of"):
        bulk.nemo_orca2_k0_ocean_step(
            card._replace(numerics="fast"), None, nemo_kt=1, **operands)
