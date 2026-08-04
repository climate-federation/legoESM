"""#1360: an unvalidated sub-face tile factor must be REFUSED, not replicated.

Before this gate, requesting kt=4 (96 devices) fell through the activation
branch, left the step replicating the global state on every device, and died
with an XLA argument-size error that reads as an OOM rather than "this tiling
is not supported".
"""

from __future__ import annotations

import os

import pytest

from legoesm.parallel.sharded_dynamics import (
    VALIDATED_TILE_FACTORS,
    make_sharded_step,
)
from legoesm.parallel.mesh import DeviceConfig


class _Mesh:
    axis_names = ("face",)


def _config(kt: int) -> DeviceConfig:
    return DeviceConfig(
        mesh=_Mesh(), face_sharding=None, replicated_sharding=None,
        n_devices=6 * kt * kt, backend="GPU", is_distributed=False,
        tiling=(kt, kt), grid_type="cubed_sphere",
    )


def test_validated_factors_are_exactly_the_evidence_backed_ones():
    """Grow-only guard: kt=2/3 are the pair with bit-identity evidence."""
    assert VALIDATED_TILE_FACTORS == frozenset({2, 3})


@pytest.mark.parametrize("kt", [4, 5, 8])
def test_unvalidated_kt_raises_instead_of_replicating(kt, monkeypatch):
    monkeypatch.setenv("LEGOESM_TILED_SPMD", "1")
    with pytest.raises(ValueError, match="bit-identity-validated only at kt"):
        make_sharded_step(model=object(), config=_config(kt))


@pytest.mark.parametrize("kt", sorted(VALIDATED_TILE_FACTORS))
def test_validated_kt_is_not_refused(kt, monkeypatch):
    """The gate must not reject the counts that DO shard."""
    monkeypatch.setenv("LEGOESM_TILED_SPMD", "1")
    try:
        make_sharded_step(model=object(), config=_config(kt))
    except ValueError as e:  # pragma: no cover - would be the regression
        assert "bit-identity-validated only at kt" not in str(e), e
    except Exception:
        pass  # any other failure is downstream of the gate, not the gate


def test_gate_is_inert_when_tiled_spmd_is_off(monkeypatch):
    """kt only matters when the tiled backend is actually requested."""
    monkeypatch.delenv("LEGOESM_TILED_SPMD", raising=False)
    try:
        make_sharded_step(model=object(), config=_config(4))
    except ValueError as e:
        assert "bit-identity-validated only at kt" not in str(e), e
    except Exception:
        pass
