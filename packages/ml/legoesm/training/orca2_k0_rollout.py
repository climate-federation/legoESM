"""Differentiable rollout of the composed standalone ORCA2 K0 step (surface + ocean).

The K0 step recomputes surface forcing every second step (``nn_fsbc = 2``) and
branches on that phase in Python, so the loop advances one PAIR of steps per
iteration: an odd step that updates the forcing, then an even step that reuses
it.  The step counter is traced; each half of the pair passes its phase as a
static ``forcing_update``.

Per-step controls (the dicts the standalone drivers build: ``s0f``, ``ice_if``,
``s0_base``, ``s2_base``, ``chl``, ``geothermal_heat_flux_mwm2``, ``sponge``) are
split leaf by leaf into: Python values equal on every step (kept static, so
``int(s0.forcing_stride)`` and friends stay Python), arrays equal on every step
(passed once), and everything else (stacked along a step axis and indexed by
the traced counter).  Damping-target stamps are checked on the host first.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.training.checkpoint_schedule import checkpointed_loop

_STATIC, _SHARED, _STACKED = "static", "shared", "stacked"
S2_CARRY = ("prior_runoff_mass", "prior_runoff_temperature_content",
            "prior_runoff_salinity_content", "prior_restoring_volume_flux",
            "prior_restoring_heat_flux", "prior_restoring_coefficient")


class K0Carry(NamedTuple):
    """What one K0 step hands the next besides its controls."""
    state: Any
    s0_prior: Any
    s0_shortwave_fraction_top: Any
    s2_prior: tuple


def carry_from_result(result) -> K0Carry:
    """The carry after an eager ``nemo_orca2_k0_ocean_step`` result."""
    ocean = result.ocean
    state = ocean.state_after if hasattr(ocean, "state_after") else ocean
    return K0Carry(state, result.next_s0.prior, result.next_s0.shortwave_fraction_top,
                   tuple(getattr(result.next_s2, f) for f in S2_CARRY))


def _is_array(x) -> bool:
    return isinstance(x, (np.ndarray, jax.Array))


def split_controls(controls, kt0: int):
    """``(treedef, kinds, values)`` for per-step controls of steps kt0, kt0+1, ...

    ``values[j]`` is the Python value (static), the single array (shared) or
    the stacked array (stacked) for leaf ``j``.  Raises on a stale damping
    target or on controls whose structure changes between steps.
    """
    clean = []
    for i, ctrl in enumerate(controls):
        stamp = ctrl["sponge"].nemo_kt
        if stamp != kt0 + i:
            raise ValueError(f"damping target for step {kt0 + i} has kt={stamp!r}")
        clean.append(dict(ctrl, sponge=ctrl["sponge"]._replace(nemo_kt=None)))
    flat = [jax.tree_util.tree_flatten(c, is_leaf=lambda x: x is None) for c in clean]
    treedef = flat[0][1]
    if any(td != treedef for _, td in flat):
        raise ValueError("per-step controls change structure between steps")
    kinds, values = [], []
    for j in range(len(flat[0][0])):
        leaves = [leaves[j] for leaves, _ in flat]
        first = leaves[0]
        if not _is_array(first) and all(not _is_array(v) and v == first for v in leaves):
            kinds.append(_STATIC)
            values.append(first)
        elif _is_array(first) and all(
                v is first or (_is_array(v) and v.shape == first.shape
                               and np.array_equal(np.asarray(v), np.asarray(first)))
                for v in leaves[1:]):
            kinds.append(_SHARED)
            values.append(jnp.asarray(first))
        else:
            kinds.append(_STACKED)
            values.append(jnp.stack([jnp.asarray(v) for v in leaves]))
    return treedef, tuple(kinds), values


def _controls_at(treedef, kinds, values, i):
    leaves = [v[i] if k == _STACKED else v for k, v in zip(kinds, values)]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def make_k0_rollout(card, model, n_pairs: int, kt0: int, treedef, kinds, *,
                    schedule: str = "uniform", checkpoints: int | None = None):
    """``(carry, values) -> carry`` after ``2 * n_pairs`` K0 steps from odd ``kt0``.

    ``values`` are the array operands from :func:`split_controls` (static
    entries are baked in from ``kinds``/``treedef`` at construction).
    """
    from legoesm.ocean import bulk_flux_omip as bulk

    if kt0 < 1 or kt0 % 2 != 1:
        raise ValueError(f"a pair starts on a forcing-update (odd) step, got kt0={kt0}")

    def one_step(carry: K0Carry, ctrl, kt, forcing_update, prior_surface):
        s0 = ctrl["s0_base"]._replace(prior=carry.s0_prior,
                                      shortwave_fraction_top=carry.s0_shortwave_fraction_top)
        s2 = ctrl["s2_base"]._replace(**dict(zip(S2_CARRY, carry.s2_prior)))
        res = bulk.nemo_orca2_k0_ocean_step(
            card, carry.state, model=model, nemo_kt=kt, forcing_update=forcing_update,
            s0f=ctrl["s0f"], ice_if=ctrl["ice_if"], s0=s0, s2=s2, chl=ctrl["chl"],
            geothermal_heat_flux_mwm2=ctrl["geothermal_heat_flux_mwm2"],
            prior_surface=prior_surface, sponge=ctrl["sponge"])
        return carry_from_result(res), res.surface

    def rollout(carry: K0Carry, values):
        def body(c):
            p, carry = c
            i = 2 * p
            kt = kt0 + i
            carry, surface = one_step(carry, _controls_at(treedef, kinds, values, i),
                                      kt, True, None)
            carry, _ = one_step(carry, _controls_at(treedef, kinds, values, i + 1),
                                kt + 1, False, surface)
            return p + 1, carry

        _, final = checkpointed_loop(body, (jnp.asarray(0, jnp.int32), carry), n_pairs,
                                     schedule=schedule, checkpoints=checkpoints)
        return final

    return rollout
