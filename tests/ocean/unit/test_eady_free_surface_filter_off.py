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
from pathlib import Path

import pytest

from legoesm.ocean.experiments.eady_uniform import EadyUniformConfig
from legoesm.ocean.mpas_config import MPASOceanConfig


def test_free_surface_filter_is_off():
    """The one-line change. If this goes red, read the module docstring first."""
    assert EadyUniformConfig().barotropic_diffusion_alpha == 0.0


def test_velocity_viscosity_is_on_and_is_what_cures_the_case():
    """The setting that makes the case run its full 200 days.

    It damps the ROTATIONAL grid mode, which the free-surface Laplacian cannot
    see (that mode carries near-zero surface-height gradient). Measured with
    the filter already off: 0 fails day 122.9; 1e3 and 1e4 both PASS the full
    200 days; 1e5 fails again at day 191.7.

    A WINDOW, not "more damping is better" -- the too-strong arm failing is the
    signature of a targeted instrument. 1e3 is the least dissipation that
    works and keeps 71% of the reference eddy speed (1.894 against 2.682 m/s)
    where 1e4 keeps 33%; this case exists to resolve eddies."""
    assert EadyUniformConfig().barotropic_u_viscosity == 1.0e3


def test_the_viscosity_actually_reaches_the_model():
    """A config field the matrix does not carry is a dangling knob.

    This bit twice in this investigation: the free-surface filter reaches the
    unstructured grid but is silently dropped on the lat-lon path, and three
    separate probe knobs turned out to be wired to nothing. So assert the
    scrape list carries it, rather than trusting that setting the field is
    enough."""
    matrix = (Path(__file__).resolve().parents[3]
              / "scripts" / "matrix" / "run_ocean_test_matrix.py").read_text()
    block = matrix[matrix.index('for attr in ("tracer_advection"'):]
    block = block[:block.index(")")]
    assert "barotropic_u_viscosity" in block, (
        "the matrix does not carry barotropic_u_viscosity, so setting it on "
        "the case config reaches nothing")


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


def test_sibling_channel_cases_are_off_too():
    """acc_channel and held_larichev carried the same inherited 0.05.

    Each was run on BOTH its grids with the filter off before being changed --
    both pass, and both also pass with the derived velocity viscosity added,
    so neither needs a replacement (unlike eady_uniform, which did).

    held_larichev's peak current on the unstructured grid moved 1.099 -> 0.726
    m/s. That is a science number, not just a stability one; the old value was
    produced with the spurious diffusivity active. Its acceptance band admits
    both, so this test is where the change is recorded."""
    from legoesm.ocean.experiments.acc_channel import ACCChannelConfig
    from legoesm.ocean.experiments.held_larichev import HeldLarichevConfig
    assert ACCChannelConfig().barotropic_diffusion_alpha == 0.0
    assert HeldLarichevConfig().barotropic_diffusion_alpha == 0.0
