"""NEMO ``mlf_baro_corr`` as a selectable option (``BarotropicConfig.
barotropic_after_reconcile``).

NEMO reconciles the 3-D momentum depth mean TWICE per step; legoESM reconciles
it once, inside the barotropic solve, at the NOW-level thickness.  The second
site (``cfgs/DINO/MY_SRC/stpmlf.F90:754-765``, called at ``:578`` after
``dyn_zdf`` at ``:396``) executes the live Kaa QCO thickness reduction and
reciprocal post-factor and, in doing so, discards whatever column mean the
implicit vertical solve deposited.  This file gates the option that builds it.

NON-VACUITY, stated precisely and RE-COUNTED against a measured revert.
Reverting the model-side insertion makes SIX of these fail: the two
``test_option_changes_the_after_state_on_both_step_paths``, the two
``test_unknown_scheme_raises_from_inside_each_step_path``, and the two
``test_the_change_is_a_column_mean_replacement_and_nothing_else`` (which gained
a "something moved" guard for exactly that reason).  Two earlier versions of
this paragraph were wrong in BOTH directions and both are retracted: the first
claimed "every test here fails when the insertion is removed" (false -- the
dispatch and kernel tests never touch the model); the second said "exactly
FOUR" (written before the guard above was added, and never re-counted).  The
count is now measured, not reasoned.

RETRACTED by registered round 49: the prior gate required the call site to
cancel the live QCO factor and hand the kernel a reference-only ladder.  That
is algebraically valid but not execution-equivalent at the last bit.  The new
gates require the raw pre-projection Kaa state, the live thickness recurrence,
and the independently materialized reciprocal; cancelled and stale-Kaa arms
are planted violations.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.barotropic_common import (
    AFTER_RECONCILE_SCHEMES,
    after_level_column_mean_reconcile,
    nemo_literal_after_level_reconcile,
    validate_after_reconcile,
)

_DT = 1800.0


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


# --------------------------------------------------------------- dispatch --
def test_validate_after_reconcile_accepts_only_the_declared_schemes():
    for ok in AFTER_RECONCILE_SCHEMES:
        assert validate_after_reconcile(ok) == ok


@pytest.mark.parametrize("bad", ["", "on", "nemo_mlf", "NEMO_MLF_BARO_CORR",
                                 "mlf_baro_corr", "velocity_avg"])
def test_validate_after_reconcile_raises_on_unknown(bad):
    with pytest.raises(ValueError, match="barotropic_after_reconcile"):
        validate_after_reconcile(bad)


def test_default_is_off_on_both_config_surfaces():
    """The option must stay opt-in on the model config AND on the DINO card
    config, or "default = current behaviour" is not true where it is read.

    The ONE exception is nemo_dino_kamm_mlf, which since #1455 R6 ships the
    NEMO-faithful pair deliberately (see
    ``test_kamm_mlf_ships_the_faithful_pair``); every other card stays off."""
    from legoesm.ocean.state import BarotropicConfig
    from legoesm.ocean.experiments.dino import DINOConfig, dino_config_for_recipe
    assert BarotropicConfig().barotropic_after_reconcile == "off"
    assert DINOConfig().barotropic_after_reconcile == "off"
    # and no OTHER shipped DINO recipe silently turns it on
    for recipe in ("nemo_dino_kamm", "nemo_paper", "legoesm_default",
                   "veros", "mitgcm", "oceananigans"):
        assert dino_config_for_recipe(recipe).barotropic_after_reconcile == "off"


def test_kamm_mlf_ships_the_faithful_pair():
    """#1455 R6 arm D: the kamm_mlf card runs NEMO's SECOND reconciliation
    (``stpmlf.F90`` ``mlf_baro_corr``) onto the PRIMARY velocity-weighted
    boxcar mean -- the average NEMO actually commits, because
    ``ln_dynadv_vec=.TRUE.`` makes ``dynspg_ts.F90`` accumulate velocities.

    The two knobs are ONE choice; either alone is NEMO at neither site. It is
    the arm the 90-day acceptance gate passed 5/5 on.

    NON-VACUITY. ``"velocity_avg"`` is ALSO the ``DINOConfig`` default, so the
    RESOLVED-value assertion on that half would still pass with the card line
    deleted; the assertion that earns it is the membership check on the card
    dict, which is why it is repeated here rather than left only to
    ``test_partial_cells_phase7.py::TestBarotropicReconcileTargetCard``.
    Deleting either card line turns THREE tests red: this one and that class's
    two."""
    from legoesm.ocean.experiments.dino import (
        DINO_RECIPES, dino_config_for_recipe)
    card = DINO_RECIPES["nemo_dino_kamm_mlf"]
    # membership first -- this is the half that cannot pass by default
    assert card.get("barotropic_after_reconcile") == "nemo_mlf_baro_corr"
    assert card.get("barotropic_reconcile_target") == "velocity_avg", (
        "the card must PIN the reconcile target explicitly; it equals the "
        "config default, so dropping the line would otherwise be invisible "
        "here and would take the after_reconcile line with it")
    c = dino_config_for_recipe("nemo_dino_kamm_mlf")
    assert c.barotropic_after_reconcile == "nemo_mlf_baro_corr"
    assert c.barotropic_reconcile_target == "velocity_avg"


# ----------------------------------------------------------------- kernel --
def _synthetic(seed=0, ny=5, nx=7, nz=6):
    rng = np.random.default_rng(seed)
    field = jnp.asarray(rng.normal(size=(ny, nx, nz)))
    h = jnp.asarray(rng.uniform(20.0, 400.0, size=(ny, nx, nz)))
    mask = np.ones((ny, nx, nz))
    mask[0, :, :] = 0.0            # a dry row
    mask[:, :, -2:] = 0.0          # two dry bottom levels everywhere
    mask[2, 3, :] = 0.0            # a dry column
    target = jnp.asarray(rng.normal(size=(ny, nx, 1)))
    return field, h, jnp.asarray(mask), target


def _column_mean(field, h, mask):
    hw = jnp.where(mask > 0, h, 0.0)
    return (jnp.sum(hw * field, axis=-1)
            / jnp.maximum(jnp.sum(hw, axis=-1), 1.0e-10))


def test_kernel_installs_exactly_the_target_column_mean():
    field, h, mask, target = _synthetic()
    out = after_level_column_mean_reconcile(field, h, target, mask, 1.0e-10)
    wet = np.asarray(jnp.sum(mask, axis=-1)) > 0
    got = np.asarray(_column_mean(out, h, mask))[wet]
    want = np.asarray(target[..., 0])[wet]
    assert np.max(np.abs(got - want)) < 1e-13


def test_kernel_zeroes_masked_cells_and_is_idempotent():
    field, h, mask, target = _synthetic(seed=3)
    out = after_level_column_mean_reconcile(field, h, target, mask, 1.0e-10)
    assert np.max(np.abs(np.asarray(out)[np.asarray(mask) == 0])) == 0.0
    twice = after_level_column_mean_reconcile(out, h, target, mask, 1.0e-10)
    assert np.max(np.abs(np.asarray(twice - out))) < 1e-13


def test_kernel_is_the_identity_when_the_mean_is_already_the_target():
    field, h, mask, _ = _synthetic(seed=7)
    already = _column_mean(field, h, mask)[..., None]
    out = after_level_column_mean_reconcile(field, h, already, mask, 1.0e-10)
    assert np.max(np.abs(np.asarray((out - field * mask)))) < 1e-12


def test_kernel_uses_the_THICKNESS_it_is_handed():
    """A kernel that ignored ``h_face_ref`` would pass every other test here;
    this is the only one that would catch it.  Note the perturbation is
    DEPTH-VARYING on purpose -- a column-uniform rescale leaves a weighted mean
    invariant, which is exactly why NEMO's free-surface factor cancels."""
    field, h_ref, mask, target = _synthetic(seed=11)
    h_other = h_ref * jnp.asarray(
        np.random.default_rng(12).uniform(1.05, 1.4, size=h_ref.shape))
    a = after_level_column_mean_reconcile(field, h_ref, target, mask, 1.0e-10)
    b = after_level_column_mean_reconcile(field, h_other, target, mask, 1.0e-10)
    assert np.max(np.abs(np.asarray(a - b))) > 1e-3


def test_kernel_is_invariant_to_a_column_uniform_thickness_rescale():
    """NEMO's own reconciliation is TIME-LEVEL INDEPENDENT: under ``key_qco``
    the ``(1+r3u)`` free-surface factor multiplies ``e3u`` and divides
    ``r1_hu``, so it cancels (``domzgr_substitute.h90:127,137``).  The kernel
    must inherit that: scaling a whole column's thicknesses by any factor may
    not change the answer.  This is what makes passing the REFERENCE ladder
    the faithful choice rather than a live one."""
    field, h_ref, mask, target = _synthetic(seed=17)
    fac = jnp.asarray(
        np.random.default_rng(18).uniform(1.02, 1.6,
                                          size=h_ref.shape[:-1]))[..., None]
    a = after_level_column_mean_reconcile(field, h_ref, target, mask, 1.0e-10)
    b = after_level_column_mean_reconcile(field, h_ref * fac, target, mask,
                                          1.0e-10)
    assert np.max(np.abs(np.asarray(a - b))) < 1e-13


def test_kernel_ignores_a_poisoned_masked_cell():
    """NEMO masks the VELOCITY inside its sum (``* umask(ji,jj,jk)``).  Without
    that, ``0 * NaN`` is NaN and one dry cell would poison the whole column."""
    field, h_ref, mask, target = _synthetic(seed=19)
    clean = after_level_column_mean_reconcile(field, h_ref, target, mask,
                                              1.0e-10)
    poisoned = jnp.where(mask > 0, field, jnp.nan)
    out = after_level_column_mean_reconcile(poisoned, h_ref, target, mask,
                                            1.0e-10)
    wet = np.asarray(mask) > 0
    assert np.all(np.isfinite(np.asarray(out)[wet]))
    assert np.max(np.abs(np.asarray(out - clean)[wet])) < 1e-13


def test_kernel_difference_is_depth_uniform_on_wet_columns():
    """Replacing a column mean shifts the whole column by one number.  This is
    the identity the step-level tests below lean on."""
    field, h, mask, target = _synthetic(seed=13)
    out = after_level_column_mean_reconcile(field, h, target, mask, 1.0e-10)
    d = np.asarray(out - field * mask)
    m = np.asarray(mask) > 0
    for j in range(d.shape[0]):
        for i in range(d.shape[1]):
            col = d[j, i][m[j, i]]
            if col.size > 1:
                assert np.max(np.abs(col - col[0])) < 1e-13


# ------------------------------------------------------------ step paths --
def _channel(outer="leapfrog", after="off", partial=True, dino_drag=False,
             ny=8, nx=16, nz=5, H=3000.0):
    """Channel matching the shape of the card this option was built for.

    ``partial=True`` puts the model on an ``OceanPartialCellCoordinate`` (every
    DINO kamm card does, and ``barotropic_drag_substep`` requires one);
    ``dino_drag=True`` adds the card's implicit-drag composition.  The fixture
    follows ``test_leapfrog_integrator._leapfrog_partial_cell_channel`` rather
    than re-deriving one.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.vertical import (
        create_ocean_z_star, create_partial_cell_coordinate,
    )
    grid = create_latlon_grid(n_lat=ny, n_lon=nx)
    z0c = create_ocean_z_star(n_levels=nz, H_max=H)
    H_bathy = jnp.asarray(
        np.full((ny, nx), H * 0.62)
        - 300.0 * np.sin(2 * np.pi * np.arange(nx) / nx)[None, :] ** 2)
    z = create_partial_cell_coordinate(z0c, H_bathy) if partial else z0c
    state = rest_state_latlon_cgrid_ocean(
        grid, z0c, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_bathy_override=H_bathy)
    # a sheared zonal current, so the column mean and the column structure are
    # distinguishable at all
    u = np.array(np.asarray(state.u.data))
    u[:] = 0.3 * np.cos(np.linspace(0, np.pi, u.shape[2]))[None, None, :]
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)))
    kw = dict(
        outer_integrator=outer, coriolis_scheme="explicit_ab2",
        vorticity_scheme="een_total", implicit_vertical_mixing=True,
        A_h=2.0e4, A_v=1.0e-3, K_v=1.0e-4, n_barotropic_substeps=8,
        enable_runtime_checks=False,
        barotropic_time_filter="nemo_boxcar_centred",
        barotropic_after_reconcile=after)
    if outer == "nemo_mlf":
        # construction requirement: nemo_mlf is a literal transcription, so it
        # hard-requires the NEMO implicit-ZDF identity (which carries e3w(Kmm))
        kw["zdf_implicit_solver_evaluation"] = "nemo_literal"
    if dino_drag:
        kw.update(bottom_drag_scheme="nemo_quadratic", zdf_drag_in_matrix=True,
                  zdf_baroclinic_only=True, barotropic_drag_substep=True)
    return state, LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(**kw))


def _second_step(method, **kw):
    """One production step to populate Nbb (the Euler start), then the real
    leap-frog step -- so the measurement lands on the LEAP-FROG site."""
    state, model = _channel(**kw)
    s1 = model._leapfrog_step(state, _DT)
    return getattr(model, method)(s1, _DT)


def _second_step_one_variable(method, outer, after, **kw):
    """Step 2 with BOTH arms sharing a BIT-IDENTICAL step 1.

    Since #1729 the option's site also runs on the Euler start, so stepping
    each arm twice under its own setting compares TWO applications plus a
    step of divergence -- a confound, not a result (oracle-fidelity Rule 7).
    Take step 1 with the option OFF in both arms, then step 2 under the arm's
    own setting, and the sole difference is one leap-frog-site
    reconciliation.
    """
    _, model_off = _channel(outer=outer, after="off", **kw)
    state, model = _channel(outer=outer, after=after, **kw)
    s1 = model_off._leapfrog_step(state, _DT)
    return getattr(model, method)(s1, _DT)


_PATHS = [("_leapfrog_step", "leapfrog"), ("_nemo_mlf_step", "nemo_mlf")]


@pytest.mark.parametrize("method,outer", _PATHS)
def test_option_changes_the_after_state_on_both_step_paths(method, outer):
    off = _second_step_one_variable(method, outer, "off", dino_drag=True)
    on = _second_step_one_variable(method, outer, "nemo_mlf_baro_corr",
                                   dino_drag=True)
    du = np.asarray(on.u.data - off.u.data)
    dv = np.asarray(on.v.data - off.v.data)
    assert np.max(np.abs(du)) > 1e-6, "the option is inert -- its site never ran"
    assert np.max(np.abs(dv)) > 1e-6


@pytest.mark.parametrize("method,outer", _PATHS)
def test_the_change_is_a_column_mean_replacement_and_nothing_else(method, outer):
    """On every wet column the option shifts EVERY level by the SAME number.
    Anything that touched the vertical structure would fail here."""
    off = _second_step_one_variable(method, outer, "off", dino_drag=True)
    on = _second_step_one_variable(method, outer, "nemo_mlf_baro_corr",
                                   dino_drag=True)
    du = np.asarray(on.u.data - off.u.data)
    # WITHOUT this line the test passes on du == 0 -- i.e. it would survive the
    # option being removed entirely, proving nothing. ``checked > 10`` below
    # counts multi-level wet COLUMNS, which exist either way; it guards against
    # an empty loop, not against a no-op.
    assert np.max(np.abs(du)) > 1e-6, "nothing moved -- this test is vacuous"
    wet = (np.abs(np.asarray(off.u.data)) + np.abs(np.asarray(on.u.data))) > 0
    checked = 0
    for j in range(du.shape[0]):
        for i in range(du.shape[1]):
            col = du[j, i][wet[j, i]]
            if col.size > 1:
                assert np.max(np.abs(col - col[0])) < 1e-11
                checked += 1
    assert checked > 10, "no multi-level wet column was actually checked"


@pytest.mark.parametrize("method,outer", _PATHS)
def test_off_matches_the_unset_default_bit_for_bit(method, outer):
    """"off" is the default, so selecting it explicitly may not move one bit --
    otherwise the option is not opt-in."""
    explicit = _second_step(method, outer=outer, after="off", dino_drag=True)
    state, model = _channel(outer=outer, dino_drag=True)
    s1 = model._leapfrog_step(state, _DT)
    default = getattr(model, method)(s1, _DT)
    for f in ("u", "v", "T", "S", "eta"):
        a = np.asarray(getattr(explicit, f).data)
        b = np.asarray(getattr(default, f).data)
        assert np.array_equal(a, b), f


@pytest.mark.parametrize("method,outer", _PATHS)
def test_unknown_scheme_raises_from_inside_each_step_path(method, outer):
    """The raise fires from the option's OWN site, so this doubles as the
    reachability proof for both call sites: a path that never reached the
    dispatch would return a state instead."""
    state, model = _channel(outer=outer, after="not_a_scheme", dino_drag=True)
    # #1729: the Euler start reaches the site too, so the raise fires on the
    # FIRST step. Both sites are still proved reachable -- the parametrisation
    # runs each method, and each method's own first step is what raises here.
    with pytest.raises(ValueError, match="barotropic_after_reconcile"):
        getattr(model, method)(state, _DT)

def test_zstar_makes_the_after_thickness_half_a_no_op():
    """A MEASURED BOUND on what this option can own -- asserted DIRECTLY.

    Under a pure z-star coordinate every layer in a column rescales by the same
    ``(H+eta)/H``, and the min-rule face depth inherits that, so a
    thickness-weighted column mean is INVARIANT between the now and after
    levels.  The after-thickness half of the reconciliation is therefore a
    no-op there, and the option can only bite through partial cells or through
    a column mean the implicit vertical solve deposited.

    An EARLIER version of this test asserted a whole-STEP null (option on vs
    off, no partial cells) and is RETRACTED: it differed in TWO variables at
    once, and -- fatally -- it passed just as well if the option never ran at
    all.  This asserts the geometric identity itself, on the two thicknesses,
    with no step involved: it fails if the z-star rescale ever stops being
    column-uniform, and it cannot be satisfied by the feature being absent.
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
    from legoesm.ocean.vertical import (
        create_ocean_z_star, compute_layer_thickness,
    )
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean

    ny, nx, nz, H = 8, 16, 5, 3000.0
    grid = create_latlon_grid(n_lat=ny, n_lon=nx)
    z0c = create_ocean_z_star(n_levels=nz, H_max=H)
    # Non-flat bathymetry is carried here for realism only.  MEASURED, so the
    # comment does not overstate it: under pure z-star the min-rule winner does
    # NOT vary with level on this fixture (dz_ref is column-independent, so
    # min(dz_ref*J_i, dz_ref*J_j) = dz_ref*min(J_i,J_j) and the same column wins
    # at every level), and every cell is wet.  The bathymetry is therefore inert
    # to this identity -- which is itself the point being asserted.
    H_bathy = jnp.asarray(
        np.full((ny, nx), H * 0.62)
        - 300.0 * np.sin(2 * np.pi * np.arange(nx) / nx)[None, :] ** 2)
    state = rest_state_latlon_cgrid_ocean(
        grid, z0c, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_bathy_override=H_bathy)
    rng = np.random.default_rng(5)
    u = jnp.asarray(rng.normal(size=np.asarray(state.u.data).shape))
    h0 = min_cell_to_uface(compute_layer_thickness(
        jnp.zeros((ny, nx)), H_bathy, z0c, min_water_column_m=1.0))
    mask = jnp.asarray(np.asarray(h0) > 0)

    eta_now = jnp.asarray(rng.uniform(-0.5, 0.5, size=(ny, nx)))
    eta_aft = eta_now + jnp.asarray(rng.uniform(0.05, 0.4, size=(ny, nx)))
    means = []
    for eta in (eta_now, eta_aft):
        h_u = min_cell_to_uface(compute_layer_thickness(
            eta, H_bathy, z0c, min_water_column_m=1.0))
        hw = jnp.where(mask, h_u, 0.0)
        means.append(np.asarray(
            jnp.sum(hw * u, axis=-1)
            / jnp.maximum(jnp.sum(hw, axis=-1), 1.0e-10)))
    spread = float(np.max(np.abs(means[0] - means[1])))
    assert spread < 1e-13, (
        "the z-star column rescale is no longer column-uniform "
        f"(|mean_now - mean_after| = {spread:.3e}); the after-thickness half "
        "of barotropic_after_reconcile is then NOT a no-op under pure z-star, "
        "and every bound quoted against that fact must be re-measured")


# ----------------------- reference-weighting algebra (generic kernel only) --
# These retain the hand-worked face interpolation check for callers of the
# generic kernel.  Production DINO mlf_baro_corr now takes the literal live-QCO
# path gated below; it no longer cites algebraic cancellation as execution.

# Reviewer's counterexample, verified by hand and reproduced verbatim here.
# Two levels; the two adjacent cells carry reference thicknesses [10, 1] and
# [10, 9]; the face velocity is [0, 2]; the target column mean is 0.
_CX_H_WEST = (10.0, 1.0)
_CX_H_EAST = (10.0, 9.0)
_CX_U = (0.0, 2.0)
# NEMO's rule, e3u_0 = 0.5*(e3t_0(i) + e3t_0(i+1))  -- zgr_lib.F90:231
_CX_H_FACE_NEMO = (10.0, 5.0)          # 0.5*(10+10), 0.5*(1+9)
# own mean = (10*0 + 5*2)/15 = 2/3  ->  u - 2/3
_CX_EXPECT_NEMO = (-2.0 / 3.0, 4.0 / 3.0)
# the MIN rule (MOM6/MITgcm hFacW) instead gives face [10, 1]:
_CX_H_FACE_MIN = (10.0, 1.0)
# own mean = (10*0 + 1*2)/11 = 2/11  ->  u - 2/11
_CX_EXPECT_MIN = (-2.0 / 11.0, 20.0 / 11.0)
# and the defect this exposes: re-weighted by the REFERENCE face thickness the
# min-rule answer does NOT have the target mean.
# (10*(-2/11) + 5*(20/11))/15 = (80/11)/15 = 16/33
_CX_RESIDUAL_OF_MIN_RULE = 16.0 / 33.0


def test_the_face_thickness_operator_follows_NEMO_and_not_the_min_rule():
    """The SHARED ``interp_cell_to_*`` must be the arithmetic mean NEMO's
    DINO/usrdef builder makes ``e3u_0`` with, and must be DISTINGUISHABLE from
    the min rule on this input -- both halves asserted against hand-computed
    literals, so neither side of the comparison is produced by the code under
    test.

    No new operator was added for this fix: ``interp_cell_to_uface`` /
    ``interp_cell_to_vface`` already existed and are halo-correct across a 2-D
    partition cut and the north fold, which a rank-local reimplementation
    would not have been."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        interp_cell_to_uface, interp_cell_to_vface, min_cell_to_uface,
        min_cell_to_vface,
    )
    # one lat row, two lon cells, two levels; u-face 1 sits between them
    cells = jnp.asarray(np.array([[list(_CX_H_WEST), list(_CX_H_EAST)]]))
    got_mean = np.asarray(interp_cell_to_uface(cells))[0, 1]
    got_min = np.asarray(min_cell_to_uface(cells))[0, 1]
    assert np.allclose(got_mean, np.array(_CX_H_FACE_NEMO), atol=0, rtol=1e-14)
    assert np.allclose(got_min, np.array(_CX_H_FACE_MIN), atol=0, rtol=1e-14)
    # v-face sibling: two lat rows, one lon cell; v-face 1 sits between them
    cells_v = jnp.asarray(np.array([[list(_CX_H_WEST)], [list(_CX_H_EAST)]]))
    assert np.allclose(np.asarray(interp_cell_to_vface(cells_v))[1, 0],
                       np.array(_CX_H_FACE_NEMO), atol=0, rtol=1e-14)
    assert np.allclose(np.asarray(min_cell_to_vface(cells_v))[1, 0],
                       np.array(_CX_H_FACE_MIN), atol=0, rtol=1e-14)


def test_the_min_rule_leaves_a_NONZERO_reference_weighted_column_mean():
    """WHY the rule matters, as arithmetic rather than as assertion.

    The reconciliation exists to install a target column mean *under NEMO's
    own weighting*.  Weighted by NEMO's reference face thickness, the min-rule
    answer misses the target by ``16/33`` on the reviewer's case -- i.e. the
    min rule fails the one identity the kernel is there to enforce, while the
    arithmetic mean satisfies it exactly.  Every number below is hand-computed
    in the block above; nothing here is read back out of the model."""
    field = jnp.asarray(np.array(_CX_U)).reshape(1, 1, 2)
    mask = jnp.ones((1, 1, 2))
    target = jnp.zeros((1, 1, 1))
    h_nemo = jnp.asarray(np.array(_CX_H_FACE_NEMO)).reshape(1, 1, 2)
    h_min = jnp.asarray(np.array(_CX_H_FACE_MIN)).reshape(1, 1, 2)

    out_nemo = np.asarray(after_level_column_mean_reconcile(
        field, h_nemo, target, mask, 1.0e-10))[0, 0]
    out_min = np.asarray(after_level_column_mean_reconcile(
        field, h_min, target, mask, 1.0e-10))[0, 0]
    assert np.allclose(out_nemo, np.array(_CX_EXPECT_NEMO), atol=1e-14)
    assert np.allclose(out_min, np.array(_CX_EXPECT_MIN), atol=1e-14)

    # re-weight BOTH answers by NEMO's reference face thickness
    w = np.array(_CX_H_FACE_NEMO)
    resid_nemo = float(np.sum(w * out_nemo) / np.sum(w))
    resid_min = float(np.sum(w * out_min) / np.sum(w))
    assert abs(resid_nemo) < 1e-14, (
        f"the arithmetic-mean weighting must install the target exactly, "
        f"got {resid_nemo:.3e}")
    assert abs(resid_min - _CX_RESIDUAL_OF_MIN_RULE) < 1e-14, (
        f"the min rule's reference-weighted residual is a hand-computed "
        f"16/33 = {_CX_RESIDUAL_OF_MIN_RULE:.6f}, got {resid_min:.6f}")


def _literal_fixture(scale):
    # Seed 49's first deterministic wide-dynamic-range column.  A uniform
    # scale still cancels algebraically, but executing it changes the result
    # by 1.07e-14, so the cancelled-arm control is provably non-vacuous.
    field = jnp.asarray([[[
        10.066808489312294, 0.00017757670918921292,
        38.79867555221713, -14050.712749141732,
    ]]], dtype=jnp.float64)
    h0 = jnp.asarray([[[
        1.2925695746815475, 2.1876066786437558e-05,
        51488.5701886061, 32.577880395543055,
    ]]], dtype=jnp.float64)
    live = h0 * jnp.asarray(scale, dtype=jnp.float64)
    reciprocal = 1.0 / jnp.sum(live, axis=-1)
    target = jnp.asarray([[[1.0948981886035902]]], dtype=jnp.float64)
    mask = jnp.ones_like(field)
    return field, h0, live, reciprocal, target, mask


def test_literal_kernel_matches_source_left_reference_eager_and_jit():
    field, _, live, reciprocal, target, mask = _literal_fixture(1.0000000003)
    expected = np.asarray(field)
    transport = np.asarray(live)[..., 0] * expected[..., 0]
    for jk in range(1, expected.shape[-1]):
        transport = transport + np.asarray(live)[..., jk] * expected[..., jk]
    expected = (expected - (transport * np.asarray(reciprocal))[..., None]
                + np.asarray(target)) * np.asarray(mask)
    eager = nemo_literal_after_level_reconcile(
        field, live, reciprocal, target, mask)
    compiled = jax.jit(nemo_literal_after_level_reconcile)(
        field, live, reciprocal, target, mask)
    assert np.array_equal(np.asarray(eager), expected)
    assert np.array_equal(np.asarray(compiled), expected)


def test_cancelled_association_is_a_planted_violation():
    field, h0, live, reciprocal, target, mask = _literal_fixture(1.0000000003)
    literal = np.asarray(nemo_literal_after_level_reconcile(
        field, live, reciprocal, target, mask))
    cancelled = np.asarray(after_level_column_mean_reconcile(
        field, h0, target, mask, 1.0e-10))
    assert not np.array_equal(literal, cancelled), (
        "planted cancelled-association arm no longer fires")


def test_stale_kaa_scale_is_a_planted_violation():
    field, _, live, reciprocal, target, mask = _literal_fixture(1.0000000003)
    faithful = np.asarray(nemo_literal_after_level_reconcile(
        field, live, reciprocal, target, mask))
    _, _, stale_live, stale_reciprocal, _, _ = _literal_fixture(0.9999999997)
    stale = np.asarray(nemo_literal_after_level_reconcile(
        field, stale_live, stale_reciprocal, target, mask))
    assert not np.array_equal(faithful, stale), (
        "planted stale-Kaa arm no longer fires")


@pytest.mark.parametrize("method,outer", _PATHS)
def test_call_site_carries_raw_kaa_and_executes_literal_kernel(method, outer):
    import unittest.mock as mock
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as omlc

    seen = {"eta": [], "kernel": 0}
    real_geometry = omlc.nemo_qco_live_face_geometry_from_operands
    real_kernel = omlc.nemo_literal_after_level_reconcile

    def capture_geometry(eta, *args, **kwargs):
        seen["eta"].append(np.asarray(eta))
        return real_geometry(eta, *args, **kwargs)

    def capture_kernel(*args, **kwargs):
        seen["kernel"] += 1
        return real_kernel(*args, **kwargs)

    state, model = _channel(outer=outer, after="nemo_mlf_baro_corr",
                            dino_drag=True)
    s1 = model._leapfrog_step(state, _DT)
    with mock.patch.object(
            omlc, "nemo_qco_live_face_geometry_from_operands",
            capture_geometry), mock.patch.object(
                omlc, "nemo_literal_after_level_reconcile", capture_kernel):
        naa = getattr(model, method)(s1, _DT)
    assert len(seen["eta"]) == 1
    assert seen["eta"][0].shape == np.asarray(naa.eta.data).shape
    assert seen["kernel"] == 2
    assert np.all(np.isfinite(np.asarray(naa.u.data)))
    assert np.all(np.isfinite(np.asarray(naa.v.data)))


# ------------------------------------------- the Euler start reconciles too --
def _first_step(method, **kw):
    """The FROM-REST first step -- NEMO's ``l_1st_euler``.

    The companion ``_second_step`` deliberately steps once before measuring,
    so nothing in this file used to score step one.  That is how #1729's gap
    survived: the Euler start returned before the reconciliation site and
    every test here stayed green.
    """
    state, model = _channel(**kw)
    assert state.u_before is None, (
        "fixture must start from rest, or this measures a leap-frog step")
    return getattr(model, method)(state, _DT)


@pytest.mark.parametrize("method,outer", _PATHS)
def test_the_euler_start_runs_the_reconciliation_too(method, outer):
    """#1729.  ``mlf_baro_corr`` is guarded on ``ln_dynspg_ts`` ALONE
    (stpmlf.f90:534), so NEMO runs it on its ``l_1st_euler`` step like any
    other.  legoESM used to return before the site and merely WARN about it.

    This is the gate on the fix: on the very first step from rest, selecting
    the option must move the velocity.  Restoring the early return makes it
    red on both paths.
    """
    off = _first_step(method, outer=outer, after="off", dino_drag=True)
    on = _first_step(method, outer=outer, after="nemo_mlf_baro_corr",
                     dino_drag=True)
    du = np.asarray(on.u.data - off.u.data)
    dv = np.asarray(on.v.data - off.v.data)
    assert np.max(np.abs(du)) > 1e-6, (
        "the Euler start never reached the reconciliation site")
    assert np.max(np.abs(dv)) > 1e-6


@pytest.mark.parametrize("method,outer", _PATHS)
def test_the_euler_start_change_is_a_column_mean_replacement(method, outer):
    """Same claim the leap-frog step is held to: every level of a wet column
    moves by the SAME number.  If the Euler start had grown its own kernel
    instead of reaching the shared one, this is where it would show."""
    off = _first_step(method, outer=outer, after="off", dino_drag=True)
    on = _first_step(method, outer=outer, after="nemo_mlf_baro_corr",
                     dino_drag=True)
    du = np.asarray(on.u.data - off.u.data)
    assert np.max(np.abs(du)) > 1e-6, "nothing moved -- this test is vacuous"
    wet = (np.abs(np.asarray(off.u.data)) + np.abs(np.asarray(on.u.data))) > 0
    checked = 0
    for j in range(du.shape[0]):
        for i in range(du.shape[1]):
            col = du[j, i][wet[j, i]]
            if col.size > 1:
                assert np.max(np.abs(col - col[0])) < 1e-11
                checked += 1
    assert checked > 10, "no multi-level wet column was actually checked"


@pytest.mark.parametrize("method,outer", _PATHS)
def test_the_euler_start_reconciliation_moves_ONLY_velocity(method, outer):
    """The claim the campaign actually relies on, asserted rather than argued.

    ``mlf_baro_corr`` is called after ``tra_zdf`` (stpmlf.f90:534 vs :507) and
    writes only puu/pvv, so on this step it can move NO tracer and NO sea
    level. That is what lets the temperature row be attributed elsewhere; if
    it were false, the whole #1729 attribution would be.

    (This replaces a test that compared the option OFF against the unset
    DEFAULT -- which IS off, so it compared a config with itself and could
    not fail. Review caught it; the version here is the claim that test was
    reaching for.)
    """
    off = _first_step(method, outer=outer, after="off", dino_drag=True)
    on = _first_step(method, outer=outer, after="nemo_mlf_baro_corr",
                     dino_drag=True)
    for name in ("T", "S", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(on, name).data),
            np.asarray(getattr(off, name).data),
            err_msg=f"the reconciliation moved {name}, which it cannot touch")
    # anti-vacuity: it DID run, it just did not reach the tracers
    assert np.max(np.abs(np.asarray(on.u.data - off.u.data))) > 1e-6
    assert np.max(np.abs(np.asarray(on.v.data - off.v.data))) > 1e-6


@pytest.mark.parametrize("method,outer", _PATHS)
def test_the_euler_start_no_longer_warns_because_it_no_longer_skips(
        method, outer):
    """The RuntimeWarning that used to announce this gap is DELETED, and this
    test refuses to let that deletion be the whole change.

    A test that only asserted silence would pass if someone removed the
    warning and left the gap.  So it asserts silence AND, on the same step,
    that the reconciliation moved the velocity.
    """
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        on = _first_step(method, outer=outer, after="nemo_mlf_baro_corr",
                         dino_drag=True)
    assert not [x for x in w if "forward-Euler start" in str(x.message)], (
        "the Euler start still claims to skip the reconciliation")
    off = _first_step(method, outer=outer, after="off", dino_drag=True)
    assert np.max(np.abs(np.asarray(on.u.data - off.u.data))) > 1e-6, (
        "silent AND inert: the warning went away but the gap did not")


def test_the_weighting_fix_is_bit_identical_on_the_SHIPPED_DINO_geometry():
    """WHY NO A/B WAS RUN FOR THE MIN->MEAN CORRECTION, as a measurement.

    The shipped ``nemo_dino_kamm_mlf`` card runs ``masked_zco``, which makes
    every cell thickness ``h_partial in {0, dz_ref[k]}`` exactly.  So at any
    face the both-cells-wet mask leaves OPEN, both cells carry ``dz_ref[k]``
    and ``min == mean`` bit-for-bit; at a CLOSED face the kernel zeroes the
    value under either rule.  The correction therefore cannot move a single
    number on this card, and a 90-day A/B would have been a null measurement
    of a provable identity.

    Measured on the card's OWN Mercator geometry and the card's OWN water-
    column floor -- both taken from the shipped constructors rather than
    hand-picked, after review caught an earlier fixture that built a uniform
    lat-lon grid and a 0.01 m floor and still called itself "the real DINO
    geometry".  The exact open-face counts are geometry-dependent and are NOT
    asserted; what is asserted is the identity and its non-vacuity.

    The closed faces DO differ (by hundreds of metres), which is what makes
    this non-vacuous -- the two rules are genuinely different operators, and
    the identity is a property of the shipped geometry, not of the code."""
    from legoesm.ocean.experiments.dino import (
        DINOConfig, create_dino_z_star, dino_masked_zco_coordinate,
        dino_lat_lon_bowl, dino_lat_lon_grid)
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d, interp_cell_to_uface, interp_cell_to_vface,
        min_cell_to_uface, min_cell_to_vface)
    from legoesm.ocean.vertical import compute_layer_thickness

    cfg = DINOConfig()
    g = dino_lat_lon_grid(cfg)           # the card's Mercator grid, not uniform
    coord, H_snap = dino_masked_zco_coordinate(
        create_dino_z_star(cfg), dino_lat_lon_bowl(g, cfg))
    # the floor the CALL SITE reads, not a hand-picked one
    mwc = LatLonCGridOceanConfig.from_flat().min_water_column_m
    h_cell = compute_layer_thickness(jnp.zeros_like(H_snap), H_snap, coord,
                                     min_water_column_m=mwc)
    au, av = compute_face_masks_3d(coord.is_active, g)
    au, av = np.asarray(au) > 0.5, np.asarray(av) > 0.5

    du = np.abs(np.asarray(min_cell_to_uface(h_cell))
                - np.asarray(interp_cell_to_uface(h_cell)))
    dv = np.abs(np.asarray(min_cell_to_vface(h_cell, g))
                - np.asarray(interp_cell_to_vface(h_cell, g)))
    assert au.sum() > 10_000 and av.sum() > 10_000, (
        f"geometry looks wrong: {au.sum()} open u-faces, {av.sum()} v-faces")
    assert du[au].max() == 0.0, (
        f"the min and mean rules must agree EXACTLY on every open u-face of "
        f"the shipped card, got {du[au].max():.3e} -- the fix is then NOT "
        "inert there and owes an A/B")
    assert dv[av].max() == 0.0, (
        f"same for v-faces, got {dv[av].max():.3e}")
    # non-vacuity: the two rules must be genuinely different operators, or the
    # assertions above would hold for any input and prove nothing
    assert du[~au].max() > 1.0 and dv[~av].max() > 1.0, (
        "min and mean agree even on CLOSED faces here, so this fixture cannot "
        "tell the two rules apart and the identity above is vacuous")
