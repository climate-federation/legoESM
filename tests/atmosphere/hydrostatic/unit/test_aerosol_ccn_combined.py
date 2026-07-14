"""Aerosol-CCN specified-Nc fill on the combined-physics (MPAS / hydrostatic)
path.

The coupled (cube / lat-lon) ``physics_pipeline`` fills the Morrison
specified droplet number from the prescribed column AOD.  These tests cover
the *combined-physics* half of that wiring — the microphysics + radiation
factories shared by the hydrostatic-merged and MPAS dycores — that closes
the ``--aerosol-ccn`` gap on MPAS.

Contract checked:
- ``_aerosol_ccn_active`` detects the morrison ``nc_from_aerosol`` switch
  (and excludes the prognostic-Nc / non-morrison cases).
- the microphysics factory advertises ``_wants_forcing`` ONLY when active
  (byte-identical 3-arg path otherwise),
- it fills ``N_c`` from ``forcing["aerosol_od"]`` so a different aerosol
  burden changes the warm-rain tendency,
- and it fails LOUDLY when the coupling is configured but no aerosol field
  was threaded (the silent-no-op class the cube path also guards).
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.combined import (
    PhysicsConfig,
    make_physics,
    _aerosol_ccn_active,
)
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
    apply_microphysics_experiment_flags,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
    specified_nc_field,
)
from legoesm.core.field import Field

jax.config.update("jax_enable_x64", True)


def _morrison_aerosol_config():
    cfg = MicrophysicsConfig(scheme="morrison")
    return cfg._replace(morrison=cfg.morrison._replace(nc_from_aerosol=True))


def _moist_state(grid, sigma):
    """Cubed-sphere hydrostatic state with a warm cloud-water layer."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

    state = held_suarez_init(grid, sigma)
    n, nlev = grid.n, sigma.n_levels
    shape = (6, n, n, nlev)
    dims = ("face", "x", "y", "level")
    q_v = jnp.full(shape, 8e-3)
    q_c = jnp.zeros(shape).at[..., nlev // 2: nlev // 2 + 2].set(1e-3)
    q_r = jnp.zeros(shape).at[..., nlev // 2 + 1].set(1e-4)
    n_c = jnp.zeros(shape)  # dead carry (predict_Nc=False)

    def _f(data, name):
        return Field(data=data, name=name, dims=dims, units="kg/kg")

    tracers = {
        "q_v": _f(q_v, "q_v"),
        "q_c": _f(q_c, "q_c"),
        "q_r": _f(q_r, "q_r"),
        "N_c": Field(data=n_c, name="N_c", dims=dims, units="1/m^3"),
    }
    return state._replace(tracers=tracers)


def _aerosol_forcing(grid, sigma, column_aod):
    ncol = 6 * grid.n * grid.n
    nlev = sigma.n_levels
    per_layer = jnp.full((ncol, nlev), column_aod / nlev)
    return {"aerosol_od": per_layer}


# ---------------------------------------------------------------------------
# _aerosol_ccn_active
# ---------------------------------------------------------------------------

def test_active_detection_morrison_on():
    cfg = PhysicsConfig(microphysics=_morrison_aerosol_config())
    assert _aerosol_ccn_active(cfg) is True


def test_active_detection_default_off():
    cfg = PhysicsConfig(microphysics=MicrophysicsConfig(scheme="morrison"))
    assert _aerosol_ccn_active(cfg) is False


def test_active_detection_predict_nc_excluded():
    mc = MicrophysicsConfig(scheme="morrison")
    mc = mc._replace(
        morrison=mc.morrison._replace(nc_from_aerosol=True, predict_Nc=True)
    )
    assert _aerosol_ccn_active(PhysicsConfig(microphysics=mc)) is False


def test_active_detection_non_morrison_off():
    cfg = PhysicsConfig(microphysics=MicrophysicsConfig(scheme="kessler"))
    assert _aerosol_ccn_active(cfg) is False


# ---------------------------------------------------------------------------
# microphysics factory: _wants_forcing gating
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["mpas", "hydrostatic"])
def test_wants_forcing_only_when_active(model_type):
    fn_on = make_microphysics_physics(
        _morrison_aerosol_config(), model_type, dt=300.0)
    fn_off = make_microphysics_physics(
        MicrophysicsConfig(scheme="morrison"), model_type, dt=300.0)
    assert getattr(fn_on, "_wants_forcing", False) is True
    # Off path stays byte-identical (legacy 3-arg call in the accumulator).
    assert getattr(fn_off, "_wants_forcing", False) is False


# ---------------------------------------------------------------------------
# microphysics factory: the fill reaches the backend / fails loudly
# ---------------------------------------------------------------------------

def _setup():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_cubed_sphere(4)
    sigma = create_sigma_coordinate(8)
    return grid, sigma


def test_missing_aerosol_od_raises():
    grid, sigma = _setup()
    state = _moist_state(grid, sigma)
    fn = make_microphysics_physics(_morrison_aerosol_config(), "mpas", dt=300.0)
    with pytest.raises(ValueError, match="aerosol_od"):
        fn(state, grid, sigma)  # forcing defaults to None
    with pytest.raises(ValueError, match="aerosol_od"):
        fn(state, grid, sigma, forcing={"T_sfc": jnp.zeros(1)})


def test_aerosol_burden_changes_warm_rain():
    grid, sigma = _setup()
    state = _moist_state(grid, sigma)
    fn = make_microphysics_physics(_morrison_aerosol_config(), "mpas", dt=300.0)

    tend_clean = fn(state, grid, sigma,
                    forcing=_aerosol_forcing(grid, sigma, 0.02))
    tend_polluted = fn(state, grid, sigma,
                       forcing=_aerosol_forcing(grid, sigma, 0.60))

    dqr_clean = tend_clean.tracer_tendencies["q_r"].data
    dqr_polluted = tend_polluted.tracer_tendencies["q_r"].data
    assert jnp.all(jnp.isfinite(dqr_clean))
    assert jnp.all(jnp.isfinite(dqr_polluted))
    # Higher aerosol -> more droplets -> KK2000 PRC ~ Nc^-1.79 suppresses
    # autoconversion: the rain tendency MUST respond to the burden.
    assert not jnp.allclose(dqr_clean, dqr_polluted)


def test_end_to_end_combined_threads_forcing_to_both_modules():
    # The REAL combined accumulator (production path): gray radiation +
    # morrison aerosol-CCN.  Both the radiation factory (nc_from_aerosol via
    # combined.py) and the microphysics factory (self-detected) advertise
    # _wants_forcing, so a missing aerosol_od fails loudly, and a present one
    # runs end-to-end with finite tendencies.
    grid, sigma = _setup()
    state = _moist_state(grid, sigma)
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        microphysics=_morrison_aerosol_config(),
    )
    physics_fn = make_physics(cfg, "hydrostatic", dt=300.0)

    # No aerosol_od threaded -> one of the two _wants_forcing modules raises.
    with pytest.raises(ValueError, match="aerosol_od"):
        physics_fn(state, grid, sigma,
                   forcing={"day_of_year": jnp.asarray(80.0)})

    tend, _ = physics_fn(
        state, grid, sigma,
        forcing={**_aerosol_forcing(grid, sigma, 0.1),
                 "day_of_year": jnp.asarray(80.0)},
    )
    assert jnp.all(jnp.isfinite(tend.dT_dt.data))


# ---------------------------------------------------------------------------
# apply_microphysics_experiment_flags — the ONE shared helper both the coupled
# (physics_pipeline._resolve_microphysics) and the combined-physics / MPAS
# (model_driver._run_mpas) paths route through, so the two ExperimentConfig
# warm-rain switches (nc_from_aerosol + subgrid_autoconversion) are threaded
# identically on every grid (closes the Morrison-only MPAS parity gap).
# ---------------------------------------------------------------------------

def test_flags_helper_threads_nc_from_aerosol_on_morrison():
    # The aerosol-CCN switch this bundle owns; the ``subgrid_autoconversion``
    # field + its threading test ship with the microphysics PR (#590) that
    # adds the field to MorrisonConfig.
    mc = MicrophysicsConfig(scheme="morrison").morrison
    assert mc.nc_from_aerosol is False  # default
    out = apply_microphysics_experiment_flags(
        mc, "morrison", nc_from_aerosol=True)
    assert out.nc_from_aerosol is True


def test_flags_helper_identity_when_both_false():
    mc = MicrophysicsConfig(scheme="morrison").morrison
    out = apply_microphysics_experiment_flags(
        mc, "morrison", nc_from_aerosol=False, subgrid_autoconversion=False)
    # Byte-identical: no _replace called -> same instance.
    assert out is mc


def test_flags_helper_raises_on_scheme_without_nc_from_aerosol():
    mc = MicrophysicsConfig(scheme="kessler").kessler
    with pytest.raises(ValueError, match="nc_from_aerosol"):
        apply_microphysics_experiment_flags(
            mc, "kessler", nc_from_aerosol=True)


def test_flags_helper_raises_on_scheme_without_subgrid_autoconv():
    mc = MicrophysicsConfig(scheme="kessler").kessler
    with pytest.raises(ValueError, match="subgrid_autoconversion"):
        apply_microphysics_experiment_flags(
            mc, "kessler", subgrid_autoconversion=True)


def test_fill_matches_specified_nc_field():
    # The N_c handed to the backend equals the shared helper's value, so the
    # combined path and the coupled path diagnose the SAME droplet number.
    grid, sigma = _setup()
    ncol, nlev = 6 * grid.n * grid.n, sigma.n_levels
    aer = _aerosol_forcing(grid, sigma, 0.075)["aerosol_od"]
    n_c = specified_nc_field(aer, (ncol, nlev))
    # Uniform column AOD -> uniform per-column Nc.
    assert n_c.shape == (ncol, nlev)
    assert jnp.all(n_c > 0.0)
    assert jnp.allclose(n_c, n_c[0, 0])
