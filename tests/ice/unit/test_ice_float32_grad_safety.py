"""float32 reverse-mode gradient safety for ice-free / empty-category states.

Regression for the floored-denominator anti-pattern ``num / jnp.maximum(den,
TINY)``.

Why the obvious guards do NOT work (this is the whole point of the fix):

* JAX's ``div_p`` JVP w.r.t. the denominator is
  ``mul(mul(neg(g), num), integer_pow(den, -2))`` -- it forms ``den**-2``
  DIRECTLY rather than ``1/(den*den)``.  ``integer_pow(1e-20, -2) == 1e40``
  and ``integer_pow(1e-30, -2) == 1e60``; both overflow float32 to ``inf``.
  (Note ``1e-20**2 == 1e-40`` IS a representable float32 subnormal, so a naive
  ``-num/(den*den)`` reading of the rule wrongly suggests there is no bug.)
  With an ice-free numerator (exactly ``0.0``) the staged residual is
  ``0 * inf == NaN``.
* The outer ``jnp.where`` does not help: its transpose is a ``select`` that
  feeds cotangent ``0.0`` into the divide, and ``0.0 * NaN == NaN``.
* ``jnp.maximum`` does not help either: its JVP is ``mul(g, balanced_eq(...))``,
  a MULTIPLY by a 0/1 mask rather than a select, so ``NaN * 0 == NaN``.

The fix is the two-sided idiom already used by ``lipscomb_2001_remap``
(``itd.py`` L581/598)::

    den = jnp.where(cond, x, 1.0)
    out = jnp.where(cond, num / den, fallback)

float32 is the model default (``core/precision.py``) while the science suite
runs ``JAX_ENABLE_X64=1``, which is STRUCTURALLY BLIND to this
(``1e-30**-2 == 1e60`` is finite in float64).  This module therefore forces
float32 explicitly, asserts the gradient dtype so it can never go vacuous, and
restores the process-entry setting so the selection never leaks into another
test module.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ice.itd import aggregate_state, linear_remap
from legoesm.ice.ridging import apply_ridging

# Attribute form rather than ``jax.config.read(...)``: ``read`` is not stable
# across JAX releases, the attribute accessor is.
_ENTRY_X64 = bool(jax.config.jax_enable_x64)

_N_CAT = 5


@pytest.fixture(autouse=True)
def _force_float32():
    """Force float32 -- the model default -- for every test in this module."""
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", _ENTRY_X64)


def _assert_all_finite(grads, label):
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves, f"{label}: no gradient leaves produced"
    for i, g in enumerate(leaves):
        assert g.dtype == jnp.float32, (
            f"{label}[{i}]: expected float32 (x64 leaked in), got {g.dtype} -- "
            "this test is VACUOUS under JAX_ENABLE_X64=1"
        )
        assert bool(jnp.all(jnp.isfinite(g))), (
            f"{label}[{i}]: non-finite float32 reverse-mode gradient over an "
            f"ice-free state (floored-denominator NaN): {g}"
        )


def test_float32_reverse_mode_grad_finite_on_ice_free_state():
    """itd.aggregate_state / itd.linear_remap / ridging.apply_ridging."""
    zeros = jnp.zeros((_N_CAT,), dtype=jnp.float32)
    T_frz = jnp.full((_N_CAT,), constants.T_freeze_ocean, dtype=jnp.float32)

    # --- itd.aggregate_state (itd.py:180): conc_total == 0 -> floored divide.
    def loss_agg(conc):
        h_agg, T_agg, conc_agg = aggregate_state(zeros, T_frz, conc)
        return jnp.sum(h_agg) + jnp.sum(T_agg) + jnp.sum(conc_agg)

    _assert_all_finite(jax.grad(loss_agg)(zeros), "aggregate_state d/d_conc")

    # --- itd.linear_remap (itd.py:347, 381, 409): a_remap / h_clamped /
    #     vol_remap are all exactly zero for an empty ITD.
    def loss_remap(a_new, h_new):
        h_r, a_r, T_r = linear_remap(
            zeros, zeros, h_new, a_new, _N_CAT, T_new=T_frz,
        )
        return jnp.sum(h_r) + jnp.sum(a_r) + jnp.sum(T_r)

    _assert_all_finite(
        jax.grad(loss_remap, argnums=(0, 1))(zeros, zeros),
        "linear_remap d/d_(a_new, h_new)",
    )

    # --- ridging.apply_ridging (ridging.py:169): participation_weights returns
    #     EXACTLY zero for an ice-free column, so sum(weights) == 0.
    def loss_ridge(a_cat, h_cat):
        out = apply_ridging(
            a_cat, h_cat, zeros, zeros,
            jnp.asarray(0.0, dtype=jnp.float32),
            _N_CAT, 3600.0,
        )
        return sum(jnp.sum(v) for v in out.values())

    _assert_all_finite(
        jax.grad(loss_ridge, argnums=(0, 1))(zeros, zeros),
        "apply_ridging d/d_(a_cat, h_cat)",
    )


# ----------------------------------------------------------------------
# The five ``sea_ice.py`` sites, reachable only through ``step_sea_ice``.
#
# THREE separate cases are required -- a single ice-free step covers NONE of
# them, and an earlier revision of this module wrongly believed it did:
#
#   * ``brine.enabled=True`` routes dispatch to ``_step_dynamic_v2``, so the
#     legacy ``_step_dynamic`` lhflx_exch site is never entered at all.
#   * "cold + dark" is not a no-op: it is the LEAD-FREEZE source term.  With
#     ``sst == T_freeze_ocean`` the ``<=`` supercooling gate (sea_ice.py:1924)
#     passes, the column grows ~2.8e-3 concentration on step 1, and none of the
#     latent/SH bases ever reach their floor.
#   * ``ocean_heat_scale`` needs ice PRESENT and freezing; over an ice-free
#     cell the NaN it generates is severed by ``F_cond = jnp.where(ice_mask,
#     ...)`` (sea_ice.py:1798) before it can reach an output.
#
# Each case therefore asserts the denominator REALLY sat on its floor, so the
# test fails loudly rather than going quietly vacuous if upstream physics moves.
# ----------------------------------------------------------------------

_F32 = jnp.float32


def _cold_dark_forcing(shape, T_low=260.0):
    from legoesm.core.coupling_fields import AtmToSurface
    return AtmToSurface(
        sw_down=jnp.zeros(shape, _F32), lw_down=jnp.full(shape, 250.0, _F32),
        precip_total=jnp.zeros(shape, _F32),
        precip_snow=jnp.zeros(shape, _F32),
        T_lowest=jnp.full(shape, T_low, _F32),
        q_lowest=jnp.full(shape, 1.0e-3, _F32),
        u_lowest=jnp.full(shape, 3.0, _F32),
        v_lowest=jnp.full(shape, 1.0, _F32),
        p_lowest=jnp.full(shape, 1.0e5, _F32),
        p_surface=jnp.full(shape, 1.0e5, _F32),
        rho_lowest=jnp.full(shape, 1.3, _F32),
        cos_zenith=jnp.zeros(shape, _F32),
        co2_ppmv=jnp.asarray(400.0, _F32),
        has_radiation=jnp.asarray(1.0, _F32),
        has_precipitation=jnp.asarray(1.0, _F32),
    )


def test_float32_grad_finite_latent_and_stress_bases_ice_free():
    """sea_ice.py latent/SH bases: multicat (L2585/L2624), single-cat (L2674)
    and the per-ice-tile SH/stress divides (L2805).

    SST is held ABOVE the supercooling gate so lead freeze cannot fire and the
    concentration bases stay exactly on their floor.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ice import SeaIceConfig, init_dynamic_ice_state, step_sea_ice
    from legoesm.ice.config import BrineConfig
    from legoesm.ice.state import distribute_dynamic_state_to_categories

    shape = (4, 6)
    grid = create_latlon_grid(*shape)
    # +3 K keeps ocean_sst STRICTLY above config.T_freeze_ocean, so
    # dh_dt_open_raw is gated to 0 (sea_ice.py:1924) and the cell stays ice-free.
    sst = jnp.full(shape, float(constants.T_freeze_ocean) + 3.0, _F32)
    zero2d = jnp.zeros(shape, _F32)

    def _run(n_cat):
        cfg = SeaIceConfig(
            dynamics="free_drift", transport="none", n_categories=n_cat,
            itd_remap=("lipscomb2001" if n_cat > 1 else "simple"),
            brine=BrineConfig(enabled=True),   # -> _uses_new_physics -> v2
        )
        st0 = init_dynamic_ice_state(shape, S_ice_init=0.0)
        if n_cat > 1:
            st0 = distribute_dynamic_state_to_categories(st0, n_cat)

        def step(conc, h, T_low):
            st = st0._replace(
                concentration=st0.concentration.replace(data=conc),
                h_ice=st0.h_ice.replace(data=h),
            )
            return step_sea_ice(
                st, _cold_dark_forcing(shape, T_low), sst, zero2d, zero2d,
                cfg, U_min=0.0, dt=3600.0, grid=grid,
            )

        def loss(conc, h, T_low):
            new_st, resp = step(conc, h, T_low)
            # tau_x/tau_y included so all THREE divides at L2805 are covered,
            # not just the shflx one.
            return (
                jnp.sum(resp.lhflx) + jnp.sum(resp.shflx)
                + jnp.sum(resp.tau_x) + jnp.sum(resp.tau_y)
            )

        conc0 = jnp.zeros_like(st0.concentration.data)
        h0 = jnp.zeros_like(st0.h_ice.data)
        T_low0 = jnp.full(shape, 260.0, _F32)

        # NON-VACUITY GUARD: if lead freeze ever fires again, the bases leave
        # their floor and this test silently stops testing anything.
        new_st, _ = step(conc0, h0, T_low0)
        assert float(jnp.max(new_st.concentration.data)) == 0.0, (
            f"[n_cat={n_cat}] concentration grew to "
            f"{float(jnp.max(new_st.concentration.data)):.3e}: the latent/SH "
            "bases are NOT on their floor, so this case no longer exercises "
            "the floored-denominator divides (lead-freeze gate changed?)"
        )
        return jax.grad(loss, argnums=(0, 1, 2))(conc0, h0, T_low0)

    _assert_all_finite(_run(1), "step_sea_ice latent basis [n_cat=1, L2674]")
    _assert_all_finite(_run(_N_CAT),
                       f"step_sea_ice latent+stress [n_cat={_N_CAT}, L2585/L2805]")


def test_float32_grad_finite_legacy_multicat_lhflx_exch():
    """sea_ice.py:889 ``lhflx_exch`` -- the LEGACY ``_step_dynamic`` multicat
    branch, reached only when ``_uses_new_physics`` is False on EVERY clause.

    The basis here is the PRE-thermo ``conc_old`` snapshot, so lead freeze
    cannot lift it off the floor and the SST is irrelevant.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ice import SeaIceConfig, init_dynamic_ice_state, step_sea_ice
    from legoesm.ice.state import distribute_dynamic_state_to_categories

    shape = (4, 6)
    grid = create_latlon_grid(*shape)
    zero2d = jnp.zeros(shape, _F32)
    sst = jnp.full(shape, float(constants.T_freeze_ocean), _F32)

    # Every new-physics gate at its OFF default -> legacy _step_dynamic.
    cfg = SeaIceConfig(dynamics="none", transport="none",
                       n_categories=_N_CAT, itd_remap="simple")
    st0 = distribute_dynamic_state_to_categories(
        init_dynamic_ice_state(shape, S_ice_init=0.0), _N_CAT)

    def loss(conc):
        st = st0._replace(concentration=st0.concentration.replace(data=conc))
        _, resp = step_sea_ice(
            st, _cold_dark_forcing(shape), sst, zero2d, zero2d,
            cfg, U_min=0.0, dt=3600.0, grid=grid,
        )
        return jnp.sum(resp.lhflx)

    conc0 = jnp.zeros_like(st0.concentration.data)
    _assert_all_finite(jax.grad(loss)(conc0),
                       "legacy _step_dynamic lhflx_exch [L889]")


def test_float32_grad_finite_ocean_heat_scale_freezing_ice():
    """sea_ice.py:1851 ``ocean_heat_scale``.

    This one canNOT be folded into an ice-free case: with ``h == 0`` the NaN is
    severed by ``F_cond = jnp.where(ice_mask, ...)``.  It needs ice PRESENT and
    NOT basally melting (``F_cond >= F_ocean``) -- the dominant winter regime
    for the pack, not an edge case.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ice import SeaIceConfig, init_dynamic_ice_state, step_sea_ice
    from legoesm.ice.config import BrineConfig

    shape = (4, 6)
    grid = create_latlon_grid(*shape)
    zero2d = jnp.zeros(shape, _F32)
    # Just above the basal freezing point: F_ocean is small but NONZERO, so the
    # incoming cotangent is live rather than trivially zero.
    sst = jnp.full(shape, 271.5, _F32)
    cfg = SeaIceConfig(dynamics="none", transport="none", n_categories=1,
                       brine=BrineConfig(enabled=True))
    st0 = init_dynamic_ice_state(shape, S_ice_init=0.0)

    def loss(h, T_ice):
        st = st0._replace(
            h_ice=st0.h_ice.replace(data=h),
            T_ice=st0.T_ice.replace(data=T_ice),
            concentration=st0.concentration.replace(
                data=jnp.full(shape, 0.8, _F32)),
        )
        _, resp = step_sea_ice(
            # T_lowest well below T_melt_surface so no surface melt term.
            st, _cold_dark_forcing(shape, T_low=240.0), sst, zero2d, zero2d,
            cfg, U_min=0.0, dt=3600.0, grid=grid,
        )
        return jnp.sum(resp.ocean_heat_extraction)

    h0 = jnp.full(shape, 0.5, _F32)        # ice PRESENT -> ice_mask true
    T0 = jnp.full(shape, 250.0, _F32)      # cold -> F_cond >> F_ocean
    _assert_all_finite(jax.grad(loss, argnums=(0, 1))(h0, T0),
                       "thermo_v2 ocean_heat_scale [L1851]")
