"""The Eady channel must not re-acquire the collocated-grid free-surface filter.

``barotropic_diffusion_alpha`` is a Laplacian on sea surface height. It exists
to damp the CHECKERBOARD mode of the collocated barotropic solver, and the
value 0.05 is the cubed-sphere default, raised there for a face-boundary
feedback. The Voronoi C-grid cases inherited that number rather than deriving
it, and on a C-grid it is aimed at a mode that does not exist: the grid-scale
mode there is the rotational one, which carries near-zero surface-height
gradient, so the filter cannot see it.

It was not free. Measured on this case at 70 km:

  * implied diffusivity ``alpha*areaCell/dt_ref`` = 3.5e6 m^2/s, against
    1e2-1e3 m^2/s for ocean lateral diffusivity;
  * the filter moved 42.7x more water than the actual flow (9.1x at the model
    default 0.01);
  * it was the DOMINANT destabiliser, monotone over four points --
    alpha 0.10 -> day 61.5, 0.05 -> day 67.4, 0.025 -> day 82.3, 0.0 -> day 122.9;
  * with it on, the time-averaged transport that advects thickness and tracers
    missed ~97% of the mass movement the free surface actually saw, because
    the filter's flux is not accumulated into that transport.

These tests pin the setting and, more importantly, pin the REASONING -- the
resolution-invariance below is the fact that decides against re-specifying the
knob in physical units, and it is the one I originally got backwards.
"""

from __future__ import annotations

import math

import pytest

from legoesm.ocean.experiments.eady_uniform import EadyUniformConfig
from legoesm.ocean.mpas_config import MPASOceanConfig


def test_free_surface_filter_is_off():
    """The one-line change. If this goes red, read the module docstring first."""
    assert EadyUniformConfig().barotropic_diffusion_alpha == 0.0


def test_divergence_damping_is_kept():
    """The other knob is NOT collateral damage.

    Divergence damping targets the divergent grid mode directly, and
    measurement shows it doing real work on this case: at 0.05 it fails at day
    67, and at BOTH 0.025 and 0.0 it goes non-finite sooner. Less is strictly
    worse, which is the signature of an instrument that is load-bearing."""
    assert EadyUniformConfig().barotropic_div_damp == 0.05


def test_the_filter_would_have_been_a_huge_diffusivity():
    """Non-vacuity: the setting matters because the number behind it is large.

    Guards the reasoning, not just the value -- if the coefficient's meaning
    ever changes so that 0.05 is no longer an enormous diffusivity, this test
    goes red and the docstring above needs rewriting rather than the config."""
    dt_ref = MPASOceanConfig().barotropic_diffusion_dt_ref
    dx = 70e3                       # the resolution this case runs
    area = math.sqrt(3) / 2 * dx ** 2   # hexagonal cell
    kappa_if_on = 0.05 * area / dt_ref
    assert kappa_if_on > 1e6, (
        f"the old setting implied {kappa_if_on:.2e} m^2/s; if that is no "
        f"longer true the justification for switching it off has changed")


def test_the_damping_timescale_is_resolution_invariant():
    """THE fact that decides the fix, and the one I first got backwards.

    The coefficient carries ``areaCell``, so the implied diffusivity grows as
    dx^2 -- which LOOKS like a mis-calibration that gets worse at coarse
    resolution. It is not. The grid-scale damping RATE is ``kappa*(pi/dx)^2``,
    and the dx^2 cancels the dx^-2 exactly: the two-cell mode is damped on the
    same ~140 s e-folding at every resolution.

    This is why the knob must not be re-specified as an explicit diffusivity in
    m^2/s: a constant kappa would be the genuinely mis-scaled form -- far too
    weak at coarse resolution and ruinous at fine. Switching the filter OFF is
    the right move; relabelling it is not."""
    dt_ref = MPASOceanConfig().barotropic_diffusion_dt_ref
    taus = []
    for dx_km in (4.0, 15.0, 70.0, 240.0):
        dx = dx_km * 1e3
        kappa = 0.05 * (math.sqrt(3) / 2 * dx ** 2) / dt_ref
        taus.append(1.0 / (kappa * (math.pi / dx) ** 2))
    assert max(taus) - min(taus) < 1e-9 * max(taus), (
        f"damping timescale is no longer resolution-invariant: {taus}")
    assert taus[0] == pytest.approx(140.4, rel=1e-3)


def test_sibling_channel_cases_are_flagged_not_silently_changed():
    """acc_channel and held_larichev carry the same inherited 0.05.

    They are deliberately NOT changed here -- each needs its own controlled
    run before its dissipation is altered. This test exists so the inherited
    value is visible rather than forgotten, and it must be updated (with
    evidence) if either is retuned."""
    from legoesm.ocean.experiments.acc_channel import ACCChannelConfig
    from legoesm.ocean.experiments.held_larichev import HeldLarichevConfig
    inherited = {
        "acc_channel": ACCChannelConfig().barotropic_diffusion_alpha,
        "held_larichev": HeldLarichevConfig().barotropic_diffusion_alpha,
    }
    assert inherited == {"acc_channel": 0.05, "held_larichev": 0.05}, (
        f"a sibling channel case changed its free-surface filter: {inherited}. "
        f"That is fine, but it needs its own measured justification -- update "
        f"this test with the evidence.")
