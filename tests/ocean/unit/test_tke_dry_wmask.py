"""NEMO dry-w-point TKE (``TKEConfig.tke_dry_wmask``).

NEMO closes ``tke_tke`` with

    en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)

(DINO ``cfgs/DINO/MY_SRC/zdftke.F90:565`` = upstream
``src/OCE/ZDF/zdftke.F90:469``), so ``en`` is EXACTLY 0 below the seafloor.
legoESM transcribed the ``MAX`` and DROPPED the ``* wmask``, so its post-solve
``en`` at the dry sub-seafloor w-rows is ``tke_background`` (> 0).

That is felt one routine later, in ``tke_avn``: the buoyancy-length line
(``:759`` / upstream ``:651``) ``zmxlm = MAX(rmxl_min, SQRT(2*en/zrn2))``
carries NO wmask, so with ``en == 0`` NEMO's dry rows are EXACTLY ``rmxl_min``
while legoESM's are O(10^3 m) (``sqrt(2e)/N`` with ``N -> 0``). The ``nn_mxl=3``
ldown sweep (``:799-812`` / upstream ``:691-704``) runs THROUGH those rows, so
in NEMO the deepest wet interface gets

    ldn(mbkt) = MIN( rmxl_min + e3t(mbkt+1,Kmm), l_int(mbkt) )

whereas in legoESM every intervening dry row re-widens the ldown carry by one
``e3t`` and the bottom limitation never arrives.

The fix therefore lives at the ONE mis-transcribed line — the post-solve floor
in ``_solve_tke_backward_euler`` — not at the downstream mixing length: with
``e == 0`` the length falls out at ``rmxl_min`` on its own, exactly as in NEMO.

NOTE (corrects the earlier prose): the CARRIED TKE was never floored at
``tke_background`` below the seafloor — ``k_profiles`` and the model step both
multiply it by the wet-interface mask, so it is exactly 0 on entry. Only the
INTRA-CALL post-solve field was re-inflated, which is why the first
``compute_mixing_lengths`` call already saw NEMO's values and only the SECOND
(post-solve) one did not.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.k_profiles import (
    compute_vertical_K_profiles,
)
from legoesm.ocean.physics.vertical_mixing.tke import (
    _mixing_length_floor,
    _solve_tke_backward_euler,
    compute_mixing_lengths,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
)

RMXL_MIN = np.float64(1.0e-6) / (
    np.float64(0.1) * np.sqrt(np.float64(1.0e-6)))
DZ = 50.0
NLEV = 12
BOT = 6             # deepest ACTIVE T-cell index (0-based) -> NEMO mbkt


@pytest.fixture(autouse=True)
def _fp64():
    """Rule 1c: oracle-fidelity numerics run fp64 -- and JAX_ENABLE_X64 alone
    is NOT enough (state constructors cast to the PRECISION POLICY's control
    dtype, which defaults to float32)."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    orig_x64 = jax.config.jax_enable_x64
    orig_pol = get_policy()
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(orig_pol)
    jax.config.update("jax_enable_x64", orig_x64)


def _cfg(choice: int = 3) -> TKEConfig:
    return TKEConfig(tke_mxl_choice=choice, mxl_min=RMXL_MIN,
                     tke_background=1.0e-6)


# ---------------------------------------------------------------------------
# 1. The transcription itself: the `* wmask` on the post-solve `en`.
# ---------------------------------------------------------------------------


def _solve_args():
    """A benign 1-column backward-Euler solve with a seafloor at ``BOT``."""
    shape = (1, NLEV - 1)
    return dict(
        e_old=jnp.full(shape, 1.0e-3),
        K_M_old=jnp.full(shape, 1.0e-3),
        K_H_old=jnp.full(shape, 1.0e-4),
        P_s=jnp.full(shape, 1.0e-7),
        N2=jnp.full(shape, 1.0e-6),
        l_eps=jnp.full(shape, 10.0),
        dz_half=jnp.full(shape, DZ),
        surface_flux=jnp.zeros((1,)),
        dt=100.0,
        cfg=_cfg(),
    )


def _wmask():
    """NEMO wmask on the interior w-axis: interface k wet iff T-cell k+1 wet."""
    return jnp.asarray(np.arange(NLEV - 1) + 1 <= BOT, dtype=jnp.float64)[None]


class TestSolveMask:
    def test_dry_rows_are_exactly_zero(self):
        """zdftke.F90:565 -- `MAX(en,rn_emin) * wmask` => en == 0 below mbkt."""
        e = np.asarray(_solve_tke_backward_euler(
            **_solve_args(), w_active=_wmask()))
        assert np.all(e[0, BOT:] == 0.0), e[0, BOT:]
        # ...and the wet rows keep the rn_emin floor (not zeroed).
        assert np.all(e[0, :BOT] >= _cfg().tke_background)

    def test_legacy_reinflates_the_same_rows(self):
        """Negative control: without the mask the dry rows are NOT zero."""
        e = np.asarray(_solve_tke_backward_euler(**_solve_args()))
        assert np.all(e[0, BOT:] > 0.0)

    def test_default_is_bit_identical(self):
        a = _solve_tke_backward_euler(**_solve_args())
        b = _solve_tke_backward_euler(**_solve_args(), w_active=None)
        assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_all_wet_mask_is_bit_identical(self):
        """A mask that is 1 everywhere must change nothing, bit-for-bit."""
        a = _solve_tke_backward_euler(**_solve_args())
        b = _solve_tke_backward_euler(
            **_solve_args(), w_active=jnp.ones((1, NLEV - 1)))
        assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_bottom_dirichlet_row_is_zeroed_like_nemo(self):
        """Interaction with tke_bottom_bc (both ON in the kamm card).

        The T15 bottom BC pins interior interface ``bottom_level`` (NEMO's
        ``en(mbkt+1)``), which is the SAME row ``w_active`` first goes dry at.
        The mask is applied last, so the pin ends at 0 -- and that IS NEMO:
        ``en(mbkt+1)`` is set at MY_SRC/zdftke.F90:383, then the `:565` loop
        (jk = 2..jpkm1, which contains mbkt+1) multiplies it by
        ``wmask(mbkt+1) = tmask(mbkt+1)*tmask(mbkt) = 0*1 = 0``
        (dommsk.F90:180). A zeroed bottom-BC row here is faithful, not a bug.
        """
        args = _solve_args()
        args["cfg"] = args["cfg"]._replace(bottom_tke_bc=True)
        bd = jnp.full((1,), 5.0e-3)                 # a large, visible pin
        lvl = jnp.full((1,), BOT, dtype=jnp.int32)  # NEMO mbkt+1 -> row BOT
        pinned = np.asarray(_solve_tke_backward_euler(
            **args, bottom_dirichlet=bd, bottom_level=lvl))
        masked = np.asarray(_solve_tke_backward_euler(
            **args, bottom_dirichlet=bd, bottom_level=lvl,
            w_active=_wmask()))
        # non-vacuity: without the mask the pin really is held at its value
        assert pinned[0, BOT] == pytest.approx(5.0e-3, rel=1e-12)
        assert masked[0, BOT] == 0.0
        # and the deepest WET row is untouched by the masking of the pin
        assert masked[0, BOT - 1] == pytest.approx(pinned[0, BOT - 1],
                                                   rel=1e-12)

    def test_veros_positivity_raises(self):
        """Dispatch hardening: the Veros branch returns before the floor."""
        args = _solve_args()
        args["cfg"] = args["cfg"]._replace(
            positivity="veros_surface_correction")
        with pytest.raises(ValueError, match="tke_dry_wmask"):
            _solve_tke_backward_euler(**args, w_active=_wmask())


# ---------------------------------------------------------------------------
# 2. The consequence in tke_avn: with e == 0 the length falls out at rmxl_min
#    and NEMO's ldown bottom limitation reaches the seafloor.
# ---------------------------------------------------------------------------


def _column(dry_e: float):
    """One weakly stratified column with a seafloor at T-cell ``BOT``.

    ``dry_e`` is the sub-seafloor TKE: 0.0 = NEMO (post-``* wmask``),
    ``tke_background`` = legoESM's legacy re-inflated value.
    """
    e = jnp.full((1, NLEV - 1), 1.0e-3)
    # weak stratification: sqrt(2e)/sqrt(N2) ~ 1414 m >> the 50 m slope bound
    n2 = jnp.full((1, NLEV - 1), 1.0e-9)
    e = e.at[:, BOT:].set(dry_e)
    n2 = n2.at[:, BOT:].set(0.0)       # neutral rock fill -> N_safe floor
    dz_half = jnp.full((1, NLEV - 1), DZ)
    dz_cell = jnp.full((1, NLEV), DZ)
    return e, n2, dz_half, dz_cell


def _lengths(dry_e: float, choice: int = 3):
    e, n2, dz_half, dz_cell = _column(dry_e)
    return compute_mixing_lengths(
        e, n2, dz_half, _cfg(choice), dz_cell=dz_cell,
        l_surface_anchor=jnp.array([RMXL_MIN]))


class TestMixingLengthFallsOut:
    def test_dry_rows_are_exactly_rmxl_min(self):
        """zdftke.F90:759 with en == 0: MAX(rmxl_min, 0) = rmxl_min."""
        l_k, l_eps = _lengths(0.0)
        assert np.allclose(np.asarray(l_k)[0, BOT:], RMXL_MIN, rtol=0, atol=0)
        assert np.allclose(np.asarray(l_eps)[0, BOT:], RMXL_MIN,
                           rtol=0, atol=0)

    def test_ldown_reproduces_nemos_bottom_limit(self):
        """ldn(mbkt) = MIN(rmxl_min + e3t(mbkt+1), l_int(mbkt)).

        Closed form transcribed from zdftke.F90:799-812 with the :565/:759
        dry-row values, computed here INDEPENDENTLY of the implementation.
        """
        l_k, _ = _lengths(0.0)
        k = BOT - 1        # deepest wet interior interface
        e, n2, _, _ = _column(0.0)
        l_int = max(RMXL_MIN,
                    float(np.sqrt(2.0 * e[0, k])
                          / np.sqrt(max(n2[0, k], 1e-12))))
        expected = min(RMXL_MIN + DZ, l_int)      # e3t(mbkt+1) = DZ
        got = float(np.asarray(l_k)[0, k])
        assert got == pytest.approx(expected, rel=1e-12), (got, expected)
        # non-vacuity: the unlimited buoyancy length is ~1414 m, so this
        # assertion is only satisfiable if the bottom limit actually bound.
        assert l_int > 10.0 * expected

    def test_reinflated_dry_rows_destroy_the_bottom_limit(self):
        """Negative control: e = tke_background there is measurably wrong.

        Every one of the ``n_dry`` sub-seafloor rows re-widens the ldown carry
        by one ``e3t``, so the limit arrives ``(n_dry+1)*e3t`` too loose -- the
        error GROWS with the amount of rock below the column.
        """
        k = BOT - 1
        n_dry = (NLEV - 1) - BOT
        got_nemo = float(np.asarray(_lengths(0.0)[0])[0, k])
        got_legacy = float(np.asarray(_lengths(1.0e-6)[0])[0, k])
        assert got_nemo == pytest.approx(RMXL_MIN + DZ, rel=1e-12)
        assert got_legacy == pytest.approx(RMXL_MIN + DZ * (n_dry + 1),
                                           rel=1e-12), got_legacy

    def test_choice4_also_gets_it(self):
        """nn_mxl=2 (choice 4) shares the same sweeps -> same dry rows."""
        l_k, l_eps = _lengths(0.0, choice=4)
        assert np.allclose(np.asarray(l_k)[0, BOT:], RMXL_MIN, rtol=0, atol=0)
        assert np.array_equal(np.asarray(l_k), np.asarray(l_eps))


# ---------------------------------------------------------------------------
# 3. The MASK CONSTRUCTION, through the real caller (D1).
#
#    `k_profiles._wet_interface_mask` is ``is_active[..., 1:]``.  The whole
#    option is wrong by one row if that becomes ``[..., :-1]``, and no
#    assertion that hand-writes the same convention in its own fixture can
#    catch it -- so this goes through ``compute_vertical_K_profiles`` on a
#    STAIRCASE bathymetry and pins the ABSOLUTE row index.
# ---------------------------------------------------------------------------


def _staircase():
    """Lat-lon rest state on a partial-cell coord with 3 distinct depths."""
    grid = create_latlon_grid(n_lat=4, n_lon=6)
    z = create_ocean_z_star(n_levels=8, H_max=4000.0)
    depths = np.abs(np.asarray(z.z_half_ref))          # (nlev+1,)
    # Per-column bathymetry landing strictly INSIDE reference cells 2, 4, 6
    # so bottom_level is a genuine staircase (2 / 4 / 6) across the domain.
    h_bathy = np.empty((4, 6))
    for j, lvl in enumerate((2, 4, 6)):
        h_bathy[:, 2 * j:2 * j + 2] = 0.5 * (depths[lvl] + depths[lvl + 1])
    zc = create_partial_cell_coordinate(z, jnp.asarray(h_bathy))
    # NEUTRAL column (T_water == T_deep): the ldown bottom limitation only
    # BINDS where the buoyancy length sqrt(2e)/N exceeds it, i.e. in weakly
    # stratified water -- exactly the DINO abyss this option is about.  A
    # strongly stratified rest state makes l_int ~ 0.1 m and the whole option
    # a no-op, which would make the assertions below vacuous.
    state = rest_state_latlon_cgrid_ocean(
        grid, zc, T_water_init_C=4.0, T_deep=4.0, S_uniform=35.0)
    return zc, state


def _phys(dry_wmask: bool, **tke_over):
    base = OceanPhysicsConfig()
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="tke",
            # n2_mode="adiabatic": with the default "insitu" N2 a uniform
            # T/S column still reads N2 ~ 5e-5 (pure compression), l_int ~ 0.2 m
            # and the ldown limit never binds -- the option would be a silent
            # no-op and every assertion below vacuous.
            # kappaM_min/kappaH_min pushed below the closure output so the
            # floors cannot mask the effect being measured.
            tke=TKEConfig(prognostic=True, tke_mxl_choice=3,
                          mxl_min=RMXL_MIN, n2_mode="adiabatic",
                          kappaM_min=1e-12,
                          kappaH_min=1e-12, kappaM_max=1e12,
                          tke_dry_wmask=dry_wmask, **tke_over),
        ),
        lateral_mixing=type(base.lateral_mixing)(scheme="none"),
        surface_forcing=type(base.surface_forcing)(scheme="none"),
        shortwave_penetration=None,
        convection=OceanConvectionConfig(scheme="none"),
    )


def _run(zc, state, dry_wmask: bool):
    return compute_vertical_K_profiles(
        state, zc, None, _phys(dry_wmask),
        tke_old=None, dt_tke=900.0, return_tke=True)


class TestMaskConstruction:
    def test_pinned_row_is_at_bottom_level_not_one_shallower(self):
        """The zeroed w-row is interface ``bottom_level``, per column.

        Interior interface ``k`` sits between T-cells ``k`` and ``k+1``, so the
        FIRST dry interface is ``k = bottom_level``: the deepest WET one is
        ``bottom_level - 1`` and MUST survive. Mutating the construction to
        ``is_active[..., :-1]`` shifts the first zero to ``bottom_level + 1``
        and turns this red.
        """
        zc, state = _staircase()
        _kv, _av, tke_new = _run(zc, state, True)
        bl = np.asarray(zc.bottom_level)
        tke = np.asarray(tke_new)
        assert set(np.unique(bl)) == {2, 4, 6}, np.unique(bl)   # real staircase
        for lvl in (2, 4, 6):
            sel = bl == lvl
            assert tke[sel][:, lvl:].max() == 0.0, (lvl, tke[sel][:, lvl:])
            assert tke[sel][:, lvl - 1].min() > 0.0, (lvl, tke[sel][:, lvl - 1])

    def test_flag_changes_the_deepest_wet_interface(self):
        """Non-vacuity of the OPTION (not just of the mask): K moves at mbkt.

        The bottom limitation now reaches the seafloor, so the mixing length --
        and hence K -- at the deepest wet interface SHRINKS. It can only bind
        where the SURFACE-anchored lup sweep does not already bind first, which
        on this grid excludes the shallowest column (bottom_level=2, where
        lup ~ rn_mxl0 + e3t(0) ~ 48 m is the smaller of the two) -- so that
        column is asserted no-op, not "changed".

        This is ALSO red under the ``is_active[..., :-1]`` mutation: that
        leaves interface ``bottom_level`` unpinned, which restores exactly the
        legacy ldown carry and makes ``on == off`` here.
        """
        zc, state = _staircase()
        _k_off, a_off, _ = _run(zc, state, False)
        _k_on, a_on, _ = _run(zc, state, True)
        bl = np.asarray(zc.bottom_level)
        assert np.all(np.asarray(a_on) <= np.asarray(a_off))  # never widens
        for lvl in (4, 6):
            sel = bl == lvl
            k = lvl - 1
            off = np.asarray(a_off)[sel][:, k]
            on = np.asarray(a_on)[sel][:, k]
            assert np.all(on < off), (lvl, on, off)

    def test_shallower_interfaces_are_untouched_at_the_top(self):
        """Sanity: the surface interface is not moved by a seafloor option."""
        zc, state = _staircase()
        k_off, _a_off, _ = _run(zc, state, False)
        k_on, _a_on, _ = _run(zc, state, True)
        assert np.allclose(np.asarray(k_off)[..., 0], np.asarray(k_on)[..., 0])


# ---------------------------------------------------------------------------
# 4. Config wiring / dispatch hardening.
# ---------------------------------------------------------------------------


class TestConfigWiring:
    def test_default_is_off(self):
        assert TKEConfig().tke_dry_wmask is False

    def test_no_is_active_raises(self):
        """A pure z-star coord has no sub-seafloor row -> fail loudly."""
        grid = create_latlon_grid(n_lat=4, n_lon=6)
        z = create_ocean_z_star(n_levels=8, H_max=4000.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z, T_water_init_C=15.0, T_deep=2.0, S_uniform=35.0)
        with pytest.raises(ValueError, match="tke_dry_wmask"):
            compute_vertical_K_profiles(
                state, z, None, _phys(True),
                tke_old=None, dt_tke=900.0, return_tke=True)

    def test_post_mixing_veros_raises(self):
        """Dispatch hardening: that path's solve is the VEROS integrate_tke,
        which has no `MAX(en,rn_emin)` for the mask to ride on -- so the flag
        would be a SILENT no-op there. It must raise instead."""
        zc, state = _staircase()
        phys = _phys(True, buoyancy_timing="post_mixing_veros",
                     veros_dz_slots=True,
                     positivity="veros_surface_correction")
        with pytest.raises(ValueError, match="tke_dry_wmask"):
            compute_vertical_K_profiles(
                state, zc, None, phys,
                tke_old=None, dt_tke=900.0, return_tke=True)

    def test_dino_kamm_card_enables_it(self):
        from legoesm.ocean.experiments.dino import dino_config_for_recipe
        for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
            cfg = dino_config_for_recipe(recipe)
            assert cfg.tke_dry_wmask is True, recipe
            assert cfg.tke_mxl_choice == 3, recipe

    def test_kamm_tke_config_carries_the_flag(self):
        """The DINOConfig field must reach the TKEConfig the closure reads.

        Built through the PUBLIC model-config route, so this covers the actual
        wiring the twin runs, not a private helper.
        """
        from legoesm.ocean.experiments.dino import (
            dino_config_for_recipe,
            dino_lat_lon_grid,
            dino_lat_lon_model_config,
        )
        cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
        grid = dino_lat_lon_grid(cfg, n_lon=8)
        _model_cfg, physics = dino_lat_lon_model_config(grid, cfg)
        tke = physics.vertical_mixing.tke
        assert tke.tke_dry_wmask is True
        assert np.float64(_mixing_length_floor(tke)).view(np.uint64) == \
            RMXL_MIN.view(np.uint64)
        assert tke.tke_mxl_choice == 3
