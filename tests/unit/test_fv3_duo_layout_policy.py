"""fv3_duo AUTO-ADAPT layout policy (user 2026-08-28: "adjust
automatically to the number of devices").

``resolve_fv3_duo_layout`` is the pure decision behind
``create_atmosphere_dycore``'s fv3_duo branch: single-process runs pick
their layout from the visible device count with no flag, while
multi-process stays explicit behind --distributed.  These pin the rules
with plain ints -- no jax devices, no model construction.
"""

from __future__ import annotations

import pytest

from legoesm.driver.component_factory import resolve_fv3_duo_layout


# ---- single-process auto-adapt (distributed=False) ----

@pytest.mark.parametrize("n", [2, 3, 6])
def test_auto_shard_on_divisor_device_counts(n):
    """2/3/6 LOCAL devices, no flag -> face-shard."""
    assert resolve_fv3_duo_layout(
        world=1, n_local=n, n_global=n, distributed=False) == "shard"


def test_auto_single_on_one_device():
    assert resolve_fv3_duo_layout(
        world=1, n_local=1, n_global=1, distributed=False) == "single"


@pytest.mark.parametrize("n", [4, 5, 7, 8])
def test_auto_single_fallback_on_nondivisor(n):
    """A device count that does not divide 6 falls back to single-device
    (the caller logs it loudly) rather than erroring on the auto path."""
    assert resolve_fv3_duo_layout(
        world=1, n_local=n, n_global=n, distributed=False) == "single"


# ---- multi-process must be explicit ----

def test_multiprocess_without_distributed_refused():
    """world>1 with no --distributed is the silent-mis-shard hazard: each
    process would auto-shard over its local devices and clobber the
    others. Refuse."""
    with pytest.raises(ValueError, match="MULTI-PROCESS"):
        resolve_fv3_duo_layout(
            world=6, n_local=2, n_global=6, distributed=False)


def test_multiprocess_launcher_world_only_still_refused():
    """Even when jax.process_count()==1 (jax.distributed never inited),
    a launcher world>1 (folded in by the caller's max()) must refuse."""
    with pytest.raises(ValueError, match="MULTI-PROCESS"):
        resolve_fv3_duo_layout(
            world=6, n_local=1, n_global=1, distributed=False)


# ---- explicit --distributed ----

@pytest.mark.parametrize("n", [2, 3, 6])
def test_distributed_shards_on_global_count(n):
    assert resolve_fv3_duo_layout(
        world=n, n_local=1, n_global=n, distributed=True) == "shard"


def test_distributed_with_one_device_errors():
    """--distributed but only 1 device: a 1-device 'distributed' run
    certifies nothing -> hard error, not a silent single-device run."""
    with pytest.raises(ValueError, match="cannot face-shard"):
        resolve_fv3_duo_layout(
            world=1, n_local=1, n_global=1, distributed=True)


@pytest.mark.parametrize("n", [4, 5, 7])
def test_distributed_with_nondivisor_errors(n):
    """--distributed with a count that cannot evenly hold the 6 faces is
    a hard error (the user asked to distribute an unshardable count)."""
    with pytest.raises(ValueError, match="cannot face-shard"):
        resolve_fv3_duo_layout(
            world=n, n_local=1, n_global=n, distributed=True)


def test_distributed_uses_global_not_local():
    """Under --distributed the GLOBAL count decides (n_global=6 shards)
    even though each process has n_local=1 -- the opposite of the auto
    path, which uses n_local."""
    assert resolve_fv3_duo_layout(
        world=6, n_local=1, n_global=6, distributed=True) == "shard"
    # ... and the auto path with the same 1 local device is single.
    assert resolve_fv3_duo_layout(
        world=1, n_local=1, n_global=6, distributed=False) == "single"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
