"""#939: the global regular lat-lon BATHY path force-enables the Fourier polar
filter by default.

A global lat-lon ocean has converging meridians (dx = R*dlon*cos(lat) -> 0 at
the N-pole); without the filter the WOA cold-start blows the pole CFL regardless
of the integrator.  ``_create_setup("latlon", ..., use_bathymetry=True)`` must
therefore ship ``use_polar_filter=True`` at the documented 60 deg cutoff, and it
must NOT blanket-flip the default onto the plain (non-bathy) latlon config.

Tripole isolation (the default never reaches a ``dlon==0`` grid) is asserted by
construction: the tripole is built via the SEPARATE ``_create_setup("tripole")``
branch, which needs the NEMO eORCA1 ``mesh_mask`` file and so is not cheaply
reachable in a unit test -- covered instead by the non-bathy assertion below
(the bathy default is scoped, not global) plus the existing tripole tests.
"""

from __future__ import annotations

from scripts.run.run_omip import _create_setup

_RES = "18x36"  # coarse: the polar-filter default is resolution-independent
_NLEV = 4
_H_MAX = 5000.0


def _latlon_config(use_bathymetry: bool):
    _, _, config, _, kind = _create_setup(
        "latlon", _RES, _NLEV, _H_MAX,
        physics_preset="full", water_type="II",
        use_bathymetry=use_bathymetry,
    )
    assert kind == "latlon"
    return config


def test_latlon_bathy_forces_polar_filter_on_at_60deg():
    pf = _latlon_config(use_bathymetry=True).polar_filter
    assert pf.use_polar_filter is True
    assert pf.polar_filter_cutoff_lat_deg == 60.0


def test_plain_latlon_config_leaves_polar_filter_at_default_off():
    # The #939 force-on is scoped to the bathy branch; it must not leak into the
    # plain latlon config (guards against flipping the PolarFilterConfig default).
    assert _latlon_config(use_bathymetry=False).polar_filter.use_polar_filter is False


def test_latlon_bathy_polar_filter_stays_overridable_off():
    """#939 A/B / emergency disable: the forced-ON bathy default must remain
    overridable to OFF.  run_omip_core2 now exposes ``--no-polar-filter``
    (argparse.BooleanOptionalAction, tri-state), which forwards
    ``use_polar_filter=False`` into build_latlon_bathy's ``_ovr`` ->
    ``config.replace_flat``.  A one-way default would make the controlled A/B
    (change ONLY the filter, per the issue's own validation plan) impossible."""
    cfg = _latlon_config(use_bathymetry=True).replace_flat(use_polar_filter=False)
    assert cfg.polar_filter.use_polar_filter is False
