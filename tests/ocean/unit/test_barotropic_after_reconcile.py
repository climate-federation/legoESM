"""NEMO ``mlf_baro_corr`` as a selectable option (``BarotropicConfig.
barotropic_after_reconcile``).

NEMO reconciles the 3-D momentum depth mean TWICE per step; legoESM reconciles
it once, inside the barotropic solve, at the NOW-level thickness.  The second
site (``cfgs/DINO/MY_SRC/stpmlf.F90:754-765``, called at ``:578`` after
``dyn_zdf`` at ``:396``) enforces the column mean at the AFTER-level thickness
and, in doing so, discards whatever column mean the implicit vertical solve
deposited.  This file gates the option that builds it.

Every test here fails when the option's insertion is removed from
``ocean_model_latlon_cgrid`` or when the kernel stops using the thickness it is
handed -- checked by reverting, not asserted.
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
    """The option must be opt-in on the model config AND on the DINO card
    config, or "default = current behaviour" is not true where it is read."""
    from legoesm.ocean.state import BarotropicConfig
    from legoesm.ocean.experiments.dino import DINOConfig, dino_config_for_recipe
    assert BarotropicConfig().barotropic_after_reconcile == "off"
    assert DINOConfig().barotropic_after_reconcile == "off"
    # and no shipped DINO recipe silently turns it on
    for recipe in ("nemo_dino_kamm_mlf", "nemo_dino_kamm", "nemo_paper",
                   "legoesm_default"):
        assert dino_config_for_recipe(recipe).barotropic_after_reconcile == "off"


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
    """The whole point of the option is WHICH time level's thickness weights
    the mean.  A kernel that ignored ``h_face_after`` would pass every test
    above and this one is the only one that would catch it."""
    field, h_now, mask, target = _synthetic(seed=11)
    h_after = h_now * jnp.asarray(
        np.random.default_rng(12).uniform(1.05, 1.4, size=h_now.shape))
    a = after_level_column_mean_reconcile(field, h_now, target, mask, 1.0e-10)
    b = after_level_column_mean_reconcile(field, h_after, target, mask, 1.0e-10)
    assert np.max(np.abs(np.asarray(a - b))) > 1e-3


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


@pytest.mark.parametrize("method,outer", _PATHS)
def test_inert_on_pure_zstar_without_partial_cells(method, outer):
    """A MEASURED BOUND on what this option can own, not a convenience.

    Under a pure z-star coordinate every layer in a column rescales by the SAME
    factor ``(H+eta)/H``, and the min-rule face depth inherits that, so a
    thickness-weighted column mean is INVARIANT between the now and after
    levels.  The after-thickness half of the reconciliation is therefore
    exactly a no-op there, and the option can only bite through partial cells
    or through a column mean the implicit vertical solve deposited.  If this
    ever starts failing, the geometry argument above has changed and any
    ownership claim resting on it must be re-checked.
    """
    off = _second_step(method, outer=outer, after="off", partial=False)
    on = _second_step(method, outer=outer, after="nemo_mlf_baro_corr",
                      partial=False)
    assert np.max(np.abs(np.asarray(on.u.data - off.u.data))) < 1e-14
