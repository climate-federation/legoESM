"""Conformance for the Stage-A2 four-brick coupler Interface contract."""

from __future__ import annotations

import pytest

from legoesm.components import (
    CouplingBrick,
    FormBrick,
    Interface,
    RegridBrick,
    TransferBrick,
    validate_interface,
)


# --- toy bricks that satisfy each Protocol -----------------------------------
class _FluxForm:
    conservative = True

    def apply(self, exchanged, state_a, state_b):
        return (-exchanged, +exchanged)  # removed from a, added to b


class _StateForm:
    conservative = False

    def apply(self, exchanged, state_a, state_b):
        return (exchanged, exchanged)


class _BulkTransfer:
    conservative = True  # physics/bulk conserves the exchanged budget

    def __call__(self, state_a, state_b, surface_props):
        return state_a - state_b  # toy flux


class _MLTransferUnconstrained:
    conservative = False  # a learned transfer that did NOT opt into conservation

    def __call__(self, state_a, state_b, surface_props):
        return 0.123  # arbitrary learned flux


class _ConservativeRemap:
    conservative = True

    def __call__(self, field, src_grid, dst_grid):
        return field


class _BilinearRemap:
    conservative = False

    def __call__(self, field, src_grid, dst_grid):
        return field


class _Explicit:
    def advance(self, interface, state_a, state_b, dt):
        return (state_a, state_b)


def _make(form, regrid, transfer=None) -> Interface:
    return Interface(
        side_a="atmosphere",
        side_b="ocean",
        form=form,
        transfer=transfer or _BulkTransfer(),
        regrid=regrid,
        coupling=_Explicit(),
    )


def test_bricks_satisfy_protocols() -> None:
    assert isinstance(_FluxForm(), FormBrick)
    assert isinstance(_BulkTransfer(), TransferBrick)
    assert isinstance(_ConservativeRemap(), RegridBrick)
    assert isinstance(_Explicit(), CouplingBrick)


def test_missing_method_fails_protocol() -> None:
    class NoApply:
        conservative = True

    assert not isinstance(NoApply(), FormBrick)  # lacks apply()
    assert not isinstance(object(), TransferBrick)


def test_interface_composes_four_bricks() -> None:
    iface = _make(_FluxForm(), _ConservativeRemap())
    assert iface.side_a == "atmosphere" and iface.side_b == "ocean"
    # The transfer + form compose: flux is removed from a, added to b.
    flux = iface.transfer(10.0, 4.0, None)
    up_a, up_b = iface.form.apply(flux, None, None)
    assert up_a == -flux and up_b == flux


def test_conserves_requires_conservative_form_transfer_and_regrid() -> None:
    assert _make(_FluxForm(), _ConservativeRemap()).conserves()          # all three
    assert not _make(_StateForm(), _ConservativeRemap()).conserves()     # state form
    assert not _make(_FluxForm(), _BilinearRemap()).conserves()          # lossy regrid
    # An unconstrained learned transfer breaks conservation even with a
    # conservative form + regrid (D4 — the finding this guards).
    assert not _make(
        _FluxForm(), _ConservativeRemap(), transfer=_MLTransferUnconstrained()
    ).conserves()


def test_validate_interface_accepts_well_formed() -> None:
    validate_interface(_make(_FluxForm(), _ConservativeRemap()))  # no raise


def test_validate_interface_rejects_wrong_arity_transfer() -> None:
    class BadTransfer:
        conservative = True

        def __call__(self, state_a):  # missing (state_b, surface_props)
            return state_a

    with pytest.raises(TypeError, match="transfer"):
        validate_interface(_make(_FluxForm(), _ConservativeRemap(), transfer=BadTransfer()))


def test_validate_interface_rejects_missing_conservative_flag() -> None:
    class NoFlagRegrid:
        def __call__(self, field, src_grid, dst_grid):
            return field

    with pytest.raises(TypeError, match="conservative"):
        validate_interface(_make(_FluxForm(), NoFlagRegrid()))
