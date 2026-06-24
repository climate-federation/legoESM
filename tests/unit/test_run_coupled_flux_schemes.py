"""CLI + dispatch tests for the per-tile surface-flux schemes wired into
run_coupled: land=MOST, slab ocean=MOST, ocean air-sea=COARE.

Covers (1) the run_coupled argument parser flags + defaults and (2) the slab
ocean "most" dispatch added to SimpleOcean._ocean_turbulent_fluxes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_run_coupled_mod", REPO / "scripts" / "run" / "run_coupled.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_flux_scheme_flag_defaults():
    """Defaults encode the user policy: land=MOST, slab=MOST, ocean air-sea
    scheme defaults 'constant' (set to coare3 at run time)."""
    args = mod.build_parser().parse_args([])
    assert args.land_bulk_scheme == "most"
    assert args.slab_bulk_scheme == "most"
    assert args.surface_bulk_scheme == "constant"


def test_flux_scheme_flags_roundtrip():
    """Each per-tile flag round-trips through the parser."""
    args = mod.build_parser().parse_args([
        "--land-bulk-scheme", "most",
        "--slab-bulk-scheme", "most",
        "--surface-bulk-scheme", "coare3",
    ])
    assert args.land_bulk_scheme == "most"
    assert args.slab_bulk_scheme == "most"
    assert args.surface_bulk_scheme == "coare3"


def test_surface_bulk_scheme_accepts_most():
    """'most' is now a valid choice for the air-sea/atm scheme too."""
    args = mod.build_parser().parse_args(["--surface-bulk-scheme", "most"])
    assert args.surface_bulk_scheme == "most"


@pytest.mark.parametrize("bad_flag", [
    "--land-bulk-scheme", "--slab-bulk-scheme", "--surface-bulk-scheme",
])
def test_flux_scheme_rejects_unknown(bad_flag):
    """argparse choices reject a typo'd scheme (dispatch hardening at the CLI)."""
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args([bad_flag, "garbage_scheme"])


def test_slab_ocean_most_dispatch_finite():
    """SimpleOcean._ocean_turbulent_fluxes accepts scheme='most' and returns
    finite SH/LH (the slab=MOST path)."""
    import jax.numpy as jnp
    from legoesm.ocean.simple_ocean import (
        SimpleOceanConfig, _ocean_turbulent_fluxes,
    )
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm.core.coupling_fields import AtmToSurface

    shape = (8, 16)
    z = jnp.zeros(shape)
    fc = AtmToSurface(
        sw_down=jnp.full(shape, 200.0), lw_down=jnp.full(shape, 350.0),
        precip_total=z, precip_snow=z,
        T_lowest=jnp.full(shape, 288.0), q_lowest=jnp.full(shape, 8e-3),
        u_lowest=jnp.full(shape, 4.0), v_lowest=z,
        p_lowest=jnp.full(shape, 1.0e5), p_surface=jnp.full(shape, 1.0e5),
        rho_lowest=jnp.full(shape, 1.15), cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.array(400.0), has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(0.0),
    )
    T_sfc = jnp.full(shape, 295.0)
    q_sfc = saturation_mixing_ratio(T_sfc, fc.p_surface)
    sh, lh = _ocean_turbulent_fluxes(
        T_sfc, q_sfc, fc, SimpleOceanConfig(bulk_scheme="most"))
    assert jnp.all(jnp.isfinite(sh)) and jnp.all(jnp.isfinite(lh))


def test_slab_ocean_unknown_scheme_raises():
    """Dispatch hardening: an unknown slab scheme still raises (not silent)."""
    import jax.numpy as jnp
    from legoesm.ocean.simple_ocean import (
        SimpleOceanConfig, _ocean_turbulent_fluxes,
    )
    from legoesm.core.coupling_fields import AtmToSurface

    shape = (4, 4)
    z = jnp.zeros(shape)
    fc = AtmToSurface(
        sw_down=z, lw_down=z, precip_total=z, precip_snow=z,
        T_lowest=jnp.full(shape, 288.0), q_lowest=jnp.full(shape, 8e-3),
        u_lowest=jnp.full(shape, 4.0), v_lowest=z,
        p_lowest=jnp.full(shape, 1.0e5), p_surface=jnp.full(shape, 1.0e5),
        rho_lowest=jnp.full(shape, 1.15), cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.array(400.0), has_radiation=jnp.array(0.0),
        has_precipitation=jnp.array(0.0),
    )
    with pytest.raises(ValueError, match="Unknown SimpleOceanConfig.bulk_scheme"):
        _ocean_turbulent_fluxes(
            jnp.full(shape, 295.0), jnp.full(shape, 1e-2), fc,
            SimpleOceanConfig(bulk_scheme="garbage"))
