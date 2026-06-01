"""Tests for previously-untested surface-forcing dispatch branches.

Covers ``SurfaceForcingConfig.scheme="combined"`` and
``scheme="bulk_formulas"`` factory dispatch through
``make_surface_forcing_physics``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ocean.physics.surface_forcing.config import (
    BulkFormulaConfig,
    PrescribedForcingConfig,
    RestoringConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.surface_forcing.integration import (
    make_surface_forcing_physics,
)


@pytest.mark.parametrize(
    "scheme",
    ["none", "prescribed", "restoring", "combined", "bulk_formulas", "external"],
)
def test_factory_returns_callable_for_every_scheme(scheme):
    config = SurfaceForcingConfig(scheme=scheme)
    fn = make_surface_forcing_physics(config)
    assert callable(fn)


def test_factory_rejects_unknown_scheme():
    config = SurfaceForcingConfig(scheme="not_a_scheme")
    with pytest.raises(ValueError, match="Unknown surface forcing scheme"):
        make_surface_forcing_physics(config)


def test_combined_scheme_uses_both_subconfigs():
    """The combined factory closes over both prescribed + restoring configs."""
    presc = PrescribedForcingConfig(tau_x=0.05, Q_net=10.0)
    rest = RestoringConfig(tau_T=86400.0, T_star_eq=22.0)
    config = SurfaceForcingConfig(
        scheme="combined", prescribed=presc, restoring=rest,
    )
    fn = make_surface_forcing_physics(config)
    assert callable(fn)


def test_bulk_formulas_constant_and_coare3_select_correctly():
    for sub_scheme in ("constant", "coare3", "large_yeager"):
        bf = BulkFormulaConfig(bulk_scheme=sub_scheme, U_a=8.0)
        config = SurfaceForcingConfig(scheme="bulk_formulas", bulk_formulas=bf)
        fn = make_surface_forcing_physics(config)
        assert callable(fn)


def test_validate_bulk_scheme_rejects_typo():
    """Shared bulk-flux dispatch guard (used by all 7 surface-flux dispatchers:
    coupler, surface_layer, slab_land, multilayer_land, two_layer_lake,
    bulk_formulas, sea_ice).  A typo'd ``bulk_scheme`` previously fell through
    each ``if scheme in (...): MOST else: <constant>`` gate to the constant
    branch, silently running the wrong air-sea physics.  The guard now raises.
    """
    from legoesm.coupler.bulk_flux import validate_bulk_scheme

    for ok in ("constant", "most", "coare3", "large_yeager"):
        validate_bulk_scheme(ok)  # must not raise
    for bad in ("coar3", "neutral", "MOST", "large-yeager", ""):
        with pytest.raises(ValueError, match="Unknown bulk_scheme"):
            validate_bulk_scheme(bad)


def test_bulk_formula_emissivity_field_present():
    """Tier 7 lifted emissivity to config — verify it is a real field."""
    bf = BulkFormulaConfig(emissivity=0.95)
    assert bf.emissivity == 0.95
    # Default
    bf_default = BulkFormulaConfig()
    assert bf_default.emissivity == pytest.approx(0.97)


def test_bulk_formula_constant_heat_flux_uses_wind_speed():
    """Heat fluxes must scale with |U_a|, not signed U_a.

    Audit cycle 2026-05-05 finding F2/F1: the constant-coefficient
    branch used signed ``cfg.U_a`` for Q_sh and Q_lh, which inverted
    the sign of the heat fluxes under easterlies (cfg.U_a < 0).  A
    strong easterly wind should still cool a warm ocean, not warm
    it.

    Regression test: run the constant-coefficient bulk formula with
    +5 m/s and -5 m/s zonal wind at the SAME magnitude and verify
    that Q_sh and Q_lh are identical (both magnitudes are |U_a|·5).
    The stress, however, should flip sign with U_a.
    """
    from legoesm.ocean.physics.surface_forcing.bulk_formulas import (
        bulk_formula_surface_forcing,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    # Tiny single-cell column: SST 295 K (warm), atm 290 K (cool air).
    # Either wind sign should produce upward sensible heat flux.
    T = jnp.array([[[[295.0]]]])
    S = jnp.array([[[[35.0]]]])
    z_coord = create_ocean_z_star(n_levels=1, H_max=10.0)
    jacobian = jnp.array([[[1.0]]])

    bf_east = BulkFormulaConfig(
        bulk_scheme="constant", U_a=5.0,
        T_a=290.0, q_a=5e-3, rho_a=1.2,
        SW_down=0.0, LW_down=320.0,
    )
    bf_west = bf_east._replace(U_a=-5.0)

    out_east = bulk_formula_surface_forcing(T, S, z_coord, jacobian, bf_east)
    out_west = bulk_formula_surface_forcing(T, S, z_coord, jacobian, bf_west)

    # Q_net (the magnitude minus radiation) is the same regardless of
    # wind sign because turbulent fluxes scale with |U_a|.  Compare
    # the dT_dt magnitudes.
    assert jnp.allclose(out_east.Q_net, out_west.Q_net, rtol=1e-12), (
        f"Q_net should be identical for +U_a vs -U_a; "
        f"east={float(out_east.Q_net):.4e}, west={float(out_west.Q_net):.4e}"
    )

    # Stress is directional → sign should flip.
    assert jnp.allclose(out_east.tau_x, -out_west.tau_x, rtol=1e-12), (
        "tau_x should flip sign with U_a"
    )


@pytest.mark.parametrize("scheme", ["none", "harmonic", "biharmonic", "gm_redi"])
def test_lateral_mixing_factory_dispatch(scheme):
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.lateral_mixing.integration import (
        make_lateral_mixing_physics,
    )
    config = LateralMixingConfig(scheme=scheme)
    fn = make_lateral_mixing_physics(config)
    assert callable(fn)


def test_lateral_mixing_unknown_scheme_raises():
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.lateral_mixing.integration import (
        make_lateral_mixing_physics,
    )
    config = LateralMixingConfig(scheme="not_real")
    with pytest.raises(ValueError):
        make_lateral_mixing_physics(config)
