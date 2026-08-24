"""NEMO ``mlf_baro_corr`` as a selectable option (``BarotropicConfig.
barotropic_after_reconcile``).

NEMO reconciles the 3-D momentum depth mean TWICE per step; legoESM reconciles
it once, inside the barotropic solve, at the NOW-level thickness.  The second
site (``cfgs/DINO/MY_SRC/stpmlf.F90:754-765``, called at ``:578`` after
``dyn_zdf`` at ``:396``) enforces the column mean at the AFTER-level thickness
and, in doing so, discards whatever column mean the implicit vertical solve
deposited.  This file gates the option that builds it.

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

SEPARATELY GATED, because a review put the defect back and every test stayed
green: WHICH vertical ladder the CALL SITE builds.  The kernel tests pin the
kernel's use of the thickness it is handed, but nothing pinned the caller
handing it a REFERENCE ladder rather than a live one -- and that distinction is
the entire content of the correction in ``d27dc0909``.  See
``test_call_site_hands_the_kernel_a_REFERENCE_ladder``.
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
        kw["implicit_vmix_e3t_now_divisor"] = True   # construction requirement
    if dino_drag:
        kw.update(bottom_drag_scheme="nemo_quadratic", zdf_drag_in_matrix=True,
                  zdf_baroclinic_only=True, barotropic_drag_substep=True)
    return state, LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(**kw))


def _second_step(method, **kw):
    """One production step to populate Nbb (the forward-Euler start), then the
    real leap-frog step -- the option's site is in the leap-frog branch, and a
    call on a fresh from-rest state would only exercise the Euler start."""
    state, model = _channel(**kw)
    s1 = model._leapfrog_step(state, _DT)
    return getattr(model, method)(s1, _DT)


_PATHS = [("_leapfrog_step", "leapfrog"), ("_nemo_mlf_step", "nemo_mlf")]


@pytest.mark.parametrize("method,outer", _PATHS)
def test_option_changes_the_after_state_on_both_step_paths(method, outer):
    off = _second_step(method, outer=outer, after="off", dino_drag=True)
    on = _second_step(method, outer=outer, after="nemo_mlf_baro_corr",
                      dino_drag=True)
    du = np.asarray(on.u.data - off.u.data)
    dv = np.asarray(on.v.data - off.v.data)
    assert np.max(np.abs(du)) > 1e-6, "the option is inert -- its site never ran"
    assert np.max(np.abs(dv)) > 1e-6


@pytest.mark.parametrize("method,outer", _PATHS)
def test_the_change_is_a_column_mean_replacement_and_nothing_else(method, outer):
    """On every wet column the option shifts EVERY level by the SAME number.
    Anything that touched the vertical structure would fail here."""
    off = _second_step(method, outer=outer, after="off", dino_drag=True)
    on = _second_step(method, outer=outer, after="nemo_mlf_baro_corr",
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
    s1 = model._leapfrog_step(state, _DT)   # Euler start: site not reached yet
    with pytest.raises(ValueError, match="barotropic_after_reconcile"):
        getattr(model, method)(s1, _DT)

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


@pytest.mark.parametrize("method,outer", _PATHS)
def test_call_site_hands_the_kernel_a_REFERENCE_ladder(method, outer):
    """THE GATE ON THE CORRECTION ITSELF, and it exists because it was missing.

    NEMO's ``mlf_baro_corr`` weights by the fixed reference ladder: under
    ``key_qco`` the free-surface factor ``(1+r3u)`` multiplies ``e3u`` and
    divides ``r1_hu``, so it cancels exactly and the reconciliation is
    TIME-LEVEL INDEPENDENT (``WORK/domzgr_substitute.h90:127,137,46,51``).  The
    first version of this option weighted by the LIVE after-level thickness
    instead.  That defect was caught by review, corrected -- and an adversarial
    re-review then put it BACK and watched all 24 tests stay green.  Nothing
    constrained which ladder the CALLER builds; the kernel tests only constrain
    what the kernel does with the one it is given.

    So this captures the argument at the call site and pins it: the thickness
    handed over must be the eta=0 reference ladder, and must NOT be the live
    one.  Both are computed here, and the test asserts they are DISTINGUISHABLE
    before asserting which one was used -- otherwise it would pass vacuously on
    a card where the two coincide.
    """
    import unittest.mock as mock
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as omlc
    from legoesm.ocean.dynamics.latlon_cgrid_operators import interp_cell_to_uface
    from legoesm.ocean.vertical import compute_layer_thickness

    seen = {}
    real = omlc.after_level_column_mean_reconcile

    def _capture(field, h_face_ref, target_mean, face_mask3, min_water_col):
        seen.setdefault("h", h_face_ref)
        return real(field, h_face_ref, target_mean, face_mask3, min_water_col)

    state, model = _channel(outer=outer, after="nemo_mlf_baro_corr",
                            dino_drag=True)
    s1 = model._leapfrog_step(state, _DT)
    with mock.patch.object(omlc, "after_level_column_mean_reconcile", _capture):
        naa = getattr(model, method)(s1, _DT)
    assert "h" in seen, "the option's site never ran -- nothing to gate"

    # WHAT THIS TEST DOES AND DOES NOT GATE, corrected after review.  The
    # cell->face rule here is ``interp_cell_to_uface`` because that is what the
    # call site uses, so assertion (b) below DOES constrain the weighting axis
    # too -- an earlier version of this note said it "gates the LADDER axis
    # alone", which was false and is RETRACTED.  What makes it a LADDER gate
    # specifically is assertion (c): the captured thickness must differ from
    # the LIVE (eta-carrying) ladder.  It rebuilds its baseline from the same
    # operator the code calls, so on its own it cannot tell a changed weighting
    # rule from a correct one; that axis is gated against HAND-COMPUTED values
    # in the two tests below.
    mwc = model.config.min_water_column_m
    ref = interp_cell_to_uface(compute_layer_thickness(
        jnp.zeros_like(naa.eta.data), state.H_bathy.data, model.z_coord,
        min_water_column_m=mwc))
    live = interp_cell_to_uface(compute_layer_thickness(
        naa.eta.data, state.H_bathy.data, model.z_coord,
        min_water_column_m=mwc))

    # (a) the two candidates must actually differ, or this test proves nothing
    spread = float(np.max(np.abs(np.asarray(ref - live))))
    assert spread > 1e-6, (
        f"reference and live ladders differ by only {spread:.3e} on this "
        "fixture, so this test cannot tell them apart -- it would pass "
        "vacuously and must be re-fixtured before it is trusted")
    # (b) and the call site must have used the REFERENCE one
    got = np.asarray(seen["h"])
    assert np.max(np.abs(got - np.asarray(ref))) < 1e-12, (
        "the call site handed the kernel a ladder that is not the eta=0 "
        "reference ladder NEMO's mlf_baro_corr weights by")
    assert np.max(np.abs(got - np.asarray(live))) > 1e-6, (
        "the call site handed the kernel the LIVE after-level thickness -- "
        "this is the exact defect corrected in d27dc0909 (the key_qco "
        "free-surface factor cancels in NEMO, so the faithful weight carries "
        "no eta at all)")


# ------------------------------------- the WEIGHTING RULE (not the ladder) --
# The ladder gate above (`..._hands_the_kernel_a_REFERENCE_ladder`) pins eta=0
# vs live.  It could not pin MIN vs ARITHMETIC MEAN, because it rebuilt the min
# rule as its own expected value -- it asserted the code against itself on this
# axis.  These two tests are hand-computed and use the reviewer's worked case.

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


@pytest.mark.parametrize("method,outer", _PATHS)
def test_call_site_weights_by_the_NEMO_ARITHMETIC_reference_face_thickness(
        method, outer):
    """THE CALL-SITE GATE ON THE WEIGHTING RULE.

    Companion to ``test_call_site_hands_the_kernel_a_REFERENCE_ladder``, which
    cannot gate this axis on its own because it rebuilds its baseline from the
    same operator the code calls.  MEASURED on a revert of the call site to
    ``min_cell_to_uface``/``min_cell_to_vface``: FOUR tests go red -- this one
    on both parametrizations and the ladder test on both.  (An earlier version
    of this docstring claimed the revert "leaves that one green"; that was
    written from intent, not measured, and is RETRACTED.)

    Both candidate face thicknesses are rebuilt here from the same cell ladder
    and the test asserts they are DISTINGUISHABLE on this fixture before
    asserting which one was used -- otherwise it would pass vacuously on a
    horizontally uniform full-step card (where the two coincide bit-for-bit,
    which is exactly why the shipped DINO card is unaffected by the fix)."""
    import unittest.mock as mock
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as omlc
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        interp_cell_to_uface, min_cell_to_uface,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    seen = {}
    real = omlc.after_level_column_mean_reconcile

    def _capture(field, h_face_ref, target_mean, face_mask3, min_water_col):
        seen.setdefault("h", h_face_ref)
        return real(field, h_face_ref, target_mean, face_mask3, min_water_col)

    state, model = _channel(outer=outer, after="nemo_mlf_baro_corr",
                            dino_drag=True)
    s1 = model._leapfrog_step(state, _DT)
    with mock.patch.object(omlc, "after_level_column_mean_reconcile", _capture):
        naa = getattr(model, method)(s1, _DT)
    assert "h" in seen, "the option's site never ran -- nothing to gate"

    h_cell = compute_layer_thickness(
        jnp.zeros_like(naa.eta.data), state.H_bathy.data, model.z_coord,
        min_water_column_m=model.config.min_water_column_m)
    nemo_rule = np.asarray(interp_cell_to_uface(h_cell))
    min_rule = np.asarray(min_cell_to_uface(h_cell))

    # (a) the two rules must actually differ here, or this proves nothing
    spread = float(np.max(np.abs(nemo_rule - min_rule)))
    assert spread > 1e-6, (
        f"the arithmetic-mean and min face thicknesses differ by only "
        f"{spread:.3e} on this fixture, so this test cannot tell them apart "
        "-- it must be re-fixtured onto a partial-cell bathymetry before it "
        "is trusted")
    # (b) and the call site must have used NEMO's arithmetic mean
    got = np.asarray(seen["h"])
    assert np.max(np.abs(got - nemo_rule)) < 1e-12, (
        "the call site handed the kernel a face thickness that is not NEMO's "
        "arithmetic reference-face rule (e3u_0 = 0.5*(e3t_0(i)+e3t_0(i+1)), "
        "zgr_lib.F90:231)")
    assert np.max(np.abs(got - min_rule)) > 1e-6, (
        "the call site handed the kernel the MIN-rule face thickness -- that "
        "is the MOM6/MITgcm hFacW convention, a different quantity, and it "
        "leaves a non-zero reference-weighted column mean (see "
        "test_the_min_rule_leaves_a_NONZERO_reference_weighted_column_mean)")


# ---------------------------------- the Euler-start gap is no longer SILENT --
@pytest.mark.parametrize("method,outer", _PATHS)
def test_euler_start_warns_that_it_skips_the_reconciliation(method, outer):
    """#1640 finding 3.  The forward-Euler start returns before the
    reconciliation site, so on that one step legoESM COMMITS a depth-mean
    deposit NEMO removes (NEMO runs mlf_baro_corr on l_1st_euler too).  That
    was silent while the card claimed the reference's second-site behaviour on
    every step.

    PARAMETRIZED OVER BOTH OUTER STEPS deliberately.  There are two separate
    early-return branches, and an earlier version keyed the once-only flag on a
    single process-wide bool -- so whichever path ran first consumed the
    warning and the OTHER site was never observed to warn at all.  Review
    caught it; this is the gate that keeps it caught.

    Not a raise: a genuine FROM-REST run of a card that ships this option has
    no before level to bridge, and the DINO twin's ``--legacy-euler-start``
    exists to reproduce artifacts recorded before 2026-08-24.  (An earlier
    version of this docstring justified that by the twin's default being
    ``bridge_before=False`` -- RETRACTED, the default is the bridged start
    since #1455; see the companion test below.)"""
    import warnings
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)

    LatLonCGridOceanModel._WARNED_EULER_SKIP = set()
    state, model = _channel(outer=outer, after="nemo_mlf_baro_corr",
                            dino_drag=True)
    assert state.u_before is None, "fixture must start on the Euler path"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        getattr(model, method)(state, _DT)
    msgs = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
    assert any("forward-Euler start" in m and method in m for m in msgs), (
        f"the Euler-start skip must announce itself from {method}, got {msgs}")

    # once per site, not once per step
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        getattr(model, method)(state, _DT)
    assert not [x for x in w2 if "forward-Euler start" in str(x.message)], (
        "the warning must be emitted once per site, not on every Euler step")


def test_no_euler_warning_once_the_before_level_is_populated():
    """#1455 (2026-08-24): under the twin's NEW default the warning must NOT
    fire -- and that is a property of the model, not of the harness.

    The bridged start hands ``model.step`` a populated ``u_before``, so the
    early-return branch the warning lives on is never taken.  The fixture gets
    there the same way the model does: step 1 is the Euler start (and warns),
    step 2 runs with the before level populated and must be silent.  Without
    this, "the default no longer warns" would be an assertion about a flag
    default rather than about the code path it selects.
    """
    import warnings
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)

    LatLonCGridOceanModel._WARNED_EULER_SKIP = set()
    state, model = _channel(after="nemo_mlf_baro_corr", dino_drag=True)
    s1 = model._leapfrog_step(state, _DT)          # the Euler start itself
    assert s1.u_before is not None, (
        "fixture must reach a populated before level, or this test passes "
        "vacuously by staying on the Euler path")
    # non-vacuity: the warning DID fire on the step that took the Euler branch
    assert "_leapfrog_step" in LatLonCGridOceanModel._WARNED_EULER_SKIP

    # ...and must not fire again now that the before level exists. Cleared, so
    # a silent result cannot be the once-per-site latch instead of the branch.
    LatLonCGridOceanModel._WARNED_EULER_SKIP = set()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model._leapfrog_step(s1, _DT)
    assert not [x for x in w if "forward-Euler start" in str(x.message)], (
        "a leap-frog step with a populated before level must never claim to "
        "be the Euler start")


def test_no_euler_warning_when_the_option_is_off():
    """Non-vacuity for the test above: a warning that fires unconditionally
    would pass it while telling the operator nothing."""
    import warnings
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)

    LatLonCGridOceanModel._WARNED_EULER_SKIP = set()
    state, model = _channel(after="off", dino_drag=True)
    assert state.u_before is None, (
        "fixture must start on the Euler path, or this test passes vacuously "
        "by never reaching the branch it is about")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model._leapfrog_step(state, _DT)
    assert not [x for x in w if "forward-Euler start" in str(x.message)], (
        "the default (option off) must stay silent")


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
